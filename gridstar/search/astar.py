import heapq
import itertools
from dataclasses import dataclass
from typing import Optional

from .node import SearchNode
from .goal import is_goal


@dataclass
class AStarResult:
    """
    Everything produced by a single A* run.

    `found`       — True if a safe state was reached
    `path`        — SearchNode list from root → goal (empty if not found)
    `all_nodes`   — every SearchNode created (root first; used for visualisation)
    `edges`       — (parent_node_id, child_node_id, action_idx) for every arc added
    `goal_node`   — terminal SearchNode if found, else None
    `n_expanded`  — nodes popped from the open heap
    `n_generated` — nodes pushed to the open heap
    """

    found: bool
    path: list
    all_nodes: list
    edges: list
    goal_node: Optional[SearchNode]
    n_expanded: int
    n_generated: int


class AStarSearch:
    """
    Neural-guided A* search over grid2op observations.

    At each expansion the policy supplies the top-K candidate action indices.
    grid2op's simulate() evaluates each one and the edge cost is derived from
    the resulting observation:

        edge_cost = max(ρ_max(child) − threshold, 0)   ← congestion severity
                  + 0.5 × n_offline_lines(child)        ← disconnected line penalty
                  + 0.01 × (action ≠ do-nothing)        ← action complexity

    States are keyed by topo_vect bytes to avoid revisiting the same topology.
    """

    def __init__(
        self,
        env,
        policy,
        top_k: int = 10,
        max_expansions: int = 200,
        max_depth: int = 5,
        thermal_limit: float = 0.98,
    ):
        self.env = env
        self.policy = policy
        self.top_k = top_k
        self.max_expansions = max_expansions
        self.max_depth = max_depth
        self.thermal_limit = thermal_limit

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def search(self, root_obs) -> AStarResult:
        counter = itertools.count()
        root = SearchNode(
            obs=root_obs,
            g=0.0,
            h=self.policy.heuristic(root_obs),
            parent=None,
            action_idx=-1,
            depth=0,
            node_id=next(counter),
        )

        if is_goal(root_obs, self.thermal_limit):
            return AStarResult(
                found=True,
                path=[root],
                all_nodes=[root],
                edges=[],
                goal_node=root,
                n_expanded=0,
                n_generated=1,
            )

        heap: list = [(root.f, root)]
        all_nodes: list = [root]
        edges: list = []
        visited: set = {root_obs.topo_vect.tobytes()}
        n_expanded = 0
        n_generated = 1

        while heap and n_expanded < self.max_expansions:
            _, node = heapq.heappop(heap)
            n_expanded += 1

            if node.depth >= self.max_depth:
                continue

            for action_idx in self.policy.top_k_actions(node.obs, self.top_k):
                child_obs, _, done, info = self.env.simulate(node.obs, action_idx)

                if done or info.get("is_illegal", False) or info.get("is_ambiguous", False):
                    continue

                state_key = child_obs.topo_vect.tobytes()
                if state_key in visited:
                    continue
                visited.add(state_key)

                g_child = node.g + self._edge_cost(child_obs, action_idx)
                child = SearchNode(
                    obs=child_obs,
                    g=g_child,
                    h=self.policy.heuristic(child_obs),
                    parent=node,
                    action_idx=action_idx,
                    depth=node.depth + 1,
                    node_id=next(counter),
                )
                all_nodes.append(child)
                edges.append((node.node_id, child.node_id, action_idx))
                n_generated += 1

                if is_goal(child_obs, self.thermal_limit):
                    return AStarResult(
                        found=True,
                        path=self._extract_path(child),
                        all_nodes=all_nodes,
                        edges=edges,
                        goal_node=child,
                        n_expanded=n_expanded,
                        n_generated=n_generated,
                    )

                heapq.heappush(heap, (child.f, child))

        return AStarResult(
            found=False,
            path=[],
            all_nodes=all_nodes,
            edges=edges,
            goal_node=None,
            n_expanded=n_expanded,
            n_generated=n_generated,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _edge_cost(self, child_obs, action_idx: int) -> float:
        rho_max = float(child_obs.rho.max())
        n_offline = int((~child_obs.line_status).sum())
        penalty = 0.01 if action_idx != self.env.do_nothing_idx else 0.0
        return max(rho_max - self.thermal_limit, 0.0) + 0.5 * n_offline + penalty

    @staticmethod
    def _extract_path(node: SearchNode) -> list:
        path: list = []
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))
