import numpy as np
import torch
from grid2op.Environment import Environment
from grid2op.Action import ActionSpace
from sknetwork.clustering import Louvain
from scipy.sparse import csr_matrix


class ClusterUtils:
    """Clusters substations using Louvain community detection on the grid topology."""

    @staticmethod
    def create_connectivity_matrix(env: Environment) -> np.ndarray:
        connectivity = np.zeros((env.n_sub, env.n_sub))
        for line_id in range(env.n_line):
            orig = env.line_or_to_subid[line_id]
            ext = env.line_ex_to_subid[line_id]
            connectivity[orig, ext] = 1
            connectivity[ext, orig] = 1
        return connectivity + np.eye(env.n_sub)

    @staticmethod
    def cluster_substations(env: Environment) -> dict:
        """
        Returns a dict mapping agent_id -> list of substation IDs,
        produced by Louvain community detection on the grid topology.
        """
        matrix = ClusterUtils.create_connectivity_matrix(env)
        labels = Louvain().fit_predict(csr_matrix(matrix))
        clusters: dict = {}
        for node, label in enumerate(labels):
            clusters.setdefault(label, []).append(node)
        return {i: nodes for i, nodes in enumerate(clusters.values())}

    @staticmethod
    def cluster_action_count(substations: list, action_space: ActionSpace) -> int:
        """Returns the total number of topology actions for the given substations."""
        return sum(
            len(action_space.get_all_unitary_topologies_set(action_space, sub))
            for sub in substations
        )


class ActionConverter:
    """
    Converts between integer action indices and grid2op actions.

    Action index layout:
        0               -> do-nothing
        1 .. sub_pos[0] -> topology actions for subs[0]
        sub_pos[0]+1 .. sub_pos[1] -> topology actions for subs[1]
        ...

    Supports cluster-based action masking: given a subset of substations,
    produces a boolean mask that keeps only their actions (plus do-nothing),
    which an agent applies to its logits before sampling.
    """

    def __init__(self, env: Environment) -> None:
        self.action_space = env.action_space
        self.env = env
        self.sub_mask: list = []
        self._init_sub_topo()
        self._init_actions()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_sub_topo(self):
        self.subs = np.flatnonzero(self.action_space.sub_info)
        self.sub_to_topo_begin: list = []
        self.sub_to_topo_end: list = []
        idx = 0
        for num_topo in self.action_space.sub_info:
            self.sub_to_topo_begin.append(idx)
            idx += num_topo
            self.sub_to_topo_end.append(idx)

    def _init_actions(self):
        self.actions = [self.env.action_space({})]  # index 0: do-nothing
        self.n_sub_actions = np.zeros(len(self.subs), dtype=int)

        for i, sub in enumerate(self.subs):
            topo_actions = self.action_space.get_all_unitary_topologies_set(
                self.action_space, sub
            )
            self.actions += topo_actions
            self.n_sub_actions[i] = len(topo_actions)
            self.sub_mask.extend(
                range(self.sub_to_topo_begin[sub], self.sub_to_topo_end[sub])
            )

        # sub_pos[i] = 1-based end position of subs[i]'s actions within self.actions
        self.sub_pos = self.n_sub_actions.cumsum()
        self.n = int(self.sub_pos[-1]) if len(self.sub_pos) else 0

    # ------------------------------------------------------------------
    # Action lookup
    # ------------------------------------------------------------------

    def act(self, action_idx: int):
        return self.actions[action_idx]

    def action_idx(self, action) -> int:
        return self.actions.index(action)

    # ------------------------------------------------------------------
    # Cluster-based action masking
    # ------------------------------------------------------------------

    def get_cluster_mask(self, cluster_subs) -> np.ndarray:
        """
        Returns a boolean mask of shape (len(self.actions),).

        True at index 0 (do-nothing) and at every action index whose
        substation belongs to cluster_subs.

        Args:
            cluster_subs: iterable of substation IDs in the active cluster.

        Returns:
            np.ndarray[bool] of length len(self.actions).
        """
        cluster_set = set(cluster_subs)
        mask = np.zeros(len(self.actions), dtype=bool)
        mask[0] = True  # do-nothing is always valid

        for i, sub in enumerate(self.subs):
            if sub not in cluster_set:
                continue
            start = int(self.sub_pos[i - 1]) + 1 if i > 0 else 1
            end = int(self.sub_pos[i]) + 1
            mask[start:end] = True

        return mask

    def masked_action_indices(self, cluster_subs) -> np.ndarray:
        """Returns the valid action indices for the given cluster."""
        return np.flatnonzero(self.get_cluster_mask(cluster_subs))

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def apply_action_mask(
        logits: torch.Tensor, mask: np.ndarray, fill: float = -1e9
    ) -> torch.Tensor:
        """
        Fills logits for disallowed actions with `fill` (effectively -inf)
        so they get zero probability after softmax.

        Args:
            logits: (batch, n_actions) or (n_actions,) tensor.
            mask:   boolean array where True = action is allowed.
            fill:   value written to masked-out positions.

        Returns:
            logits tensor with disallowed positions filled.
        """
        mask_t = torch.tensor(mask, dtype=torch.bool, device=logits.device)
        return logits.masked_fill(~mask_t, fill)

    @staticmethod
    def one_hot_encode(tensor: torch.Tensor, num_classes: int) -> torch.Tensor:
        """One-hot encodes a 1-D tensor of class indices."""
        tensor = tensor.long()
        one_hot = torch.zeros(tensor.size(0), num_classes, device=tensor.device)
        one_hot.scatter_(1, tensor.unsqueeze(1), 1)
        return one_hot
