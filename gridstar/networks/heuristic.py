"""
h(s) heuristic network — predicts cost-to-recovery from a grid observation.
"""

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


# ── Individual backbone networks ──────────────────────────────────────────────


class VanillaNet(nn.Module):
    """
    Standard 3-hidden-layer MLP.

    Architecture:  obs_dim → H → H → H/2 → 1
    Normalisation: BatchNorm1d after each hidden layer.
    Activation:    ReLU throughout; Softplus on the output.

    Good default when training data is large enough to fill the capacity.

    Args:
        obs_dim:    length of the flat observation vector.
        hidden_dim: width H of the first two hidden layers (default 256).
        dropout:    dropout rate applied after each hidden activation (default 0).
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256, dropout: float = 0.0, **_):
        super().__init__()
        H = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, H),
            nn.BatchNorm1d(H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, H),
            nn.BatchNorm1d(H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, H // 2),
            nn.ReLU(),
            nn.Linear(H // 2, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EfficientNet(nn.Module):
    """
    Lightweight 2-hidden-layer MLP optimised for fast inference.

    Architecture:  obs_dim → H → H/2 → 1
    Normalisation: none (avoids BatchNorm overhead).
    Activation:    ReLU; Softplus on output.

    Designed for A* search where h(s) is called on *every* generated node.
    Fewer parameters → lower latency per call at the cost of some accuracy.

    Args:
        obs_dim:    length of the flat observation vector.
        hidden_dim: width H of the first hidden layer (default 128).
        dropout:    dropout rate (default 0).
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 128, dropout: float = 0.0, **_):
        super().__init__()
        H = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, H // 2),
            nn.ReLU(),
            nn.Linear(H // 2, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResidualBlock(nn.Module):
    """One residual block: Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm + skip → ReLU."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class DeepNet(nn.Module):
    """
    Deep MLP with residual connections for stable gradient flow.

    Architecture:
        Input projection:  obs_dim → H  (Linear + ReLU)
        Residual blocks:   n_blocks × (H → H → H with LayerNorm skip)
        Output head:       H → H/4 → 1

    Residual connections prevent vanishing gradients in deep stacks and let
    early layers pass useful features directly to the output head.

    Args:
        obs_dim:    length of the flat observation vector.
        hidden_dim: shared hidden width H (default 256).
        n_blocks:   number of residual blocks (default 4).
        dropout:    dropout inside each residual block (default 0).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.0,
        **_,
    ):
        super().__init__()
        H = hidden_dim
        self.input_proj = nn.Sequential(nn.Linear(obs_dim, H), nn.ReLU())
        self.blocks = nn.Sequential(
            *[_ResidualBlock(H, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(H, H // 4),
            nn.ReLU(),
            nn.Linear(H // 4, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.input_proj(x)))


class WideNet(nn.Module):
    """
    Wide 2-hidden-layer MLP with LayerNorm and LeakyReLU.

    Architecture:  obs_dim → H → H → 1
    Normalisation: LayerNorm (robust to small batch sizes unlike BatchNorm).
    Activation:    LeakyReLU(0.1) — avoids dead neurons in wide layers.

    Wide layers capture rich feature interactions in fewer depth steps.
    Preferred when training time is limited but obs_dim is small.

    Args:
        obs_dim:    length of the flat observation vector.
        hidden_dim: width H (default 512).
        dropout:    dropout rate (default 0).
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 512, dropout: float = 0.0, **_):
        super().__init__()
        H = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, H),
            nn.LayerNorm(H),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(H, H),
            nn.LayerNorm(H),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(H, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingNet(nn.Module):
    """
    Two-head architecture: shared trunk → overflow head + recovery head.

    Inspired by Dueling DQN but adapted for heuristic regression:

        shared trunk  — obs_dim → H → H/2  (common feature extraction)
        overflow head — H/2 → H/4 → 1      (immediate congestion cost)
        recovery head — H/2 → H/4 → 1      (steps needed to reach safety)
        output        — Softplus(overflow + recovery)

    Rationale
    ─────────
    The total remaining cost has two conceptually distinct parts:
      1. How bad is the current grid right now?  (captured by overflow head)
      2. How many more actions will be needed?    (captured by recovery head)

    Splitting these into separate heads lets each specialise. During training
    the gradients from the two heads update different sub-networks, which
    reduces interference and can speed convergence.

    Args:
        obs_dim:    length of the flat observation vector.
        hidden_dim: trunk width H (default 256).
        dropout:    dropout in the shared trunk (default 0).
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256, dropout: float = 0.0, **_):
        super().__init__()
        H = hidden_dim
        mid = H // 2
        quarter = mid // 2

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, H),
            nn.LayerNorm(H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, mid),
            nn.ReLU(),
        )
        self.overflow_head = nn.Sequential(
            nn.Linear(mid, quarter),
            nn.ReLU(),
            nn.Linear(quarter, 1),
        )
        self.recovery_head = nn.Sequential(
            nn.Linear(mid, quarter),
            nn.ReLU(),
            nn.Linear(quarter, 1),
        )
        self.out_act = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared = self.trunk(x)
        return self.out_act(self.overflow_head(shared) + self.recovery_head(shared))


# ── Registry ──────────────────────────────────────────────────────────────────

_NETWORKS: dict = {
    "vanilla":   VanillaNet,
    "efficient": EfficientNet,
    "deep":      DeepNet,
    "wide":      WideNet,
    "dueling":   DuelingNet,
}

AVAILABLE_NETS = list(_NETWORKS.keys())


# ── HeuristicModel ────────────────────────────────────────────────────────────


class HeuristicModel(nn.Module):
    """
    h(s) heuristic network for A* search — predicts cost-to-recovery.

    ── Role in A* ──────────────────────────────────────────────────────────────

    A* evaluates each node with  f(s) = g(s) + h(s)  where:
      • g(s) — accumulated edge cost from the root (computed exactly by search)
      • h(s) — estimated remaining cost to a safe state  ← THIS NETWORK

    The network lets A* prioritise nodes that are *close to safety* rather than
    just nodes reached cheaply. A well-trained h(s) reduces the number of nodes
    expanded before finding the optimal path.

    ── Admissibility ───────────────────────────────────────────────────────────

    For A* to return an optimal path, h(s) must be *admissible*:

        h(s)  ≤  true_cost_to_goal(s)   for all s

    All backbones here end with nn.Softplus() which ensures h(s) ≥ 0 (a
    necessary but not sufficient condition for admissibility). Admissibility
    itself is enforced through the training objective: regress h(s) to the
    *actual* cost-to-goal from A* search episodes, using a loss that penalises
    overestimates more heavily than underestimates (e.g. asymmetric Huber loss).

    ── Training Signal ─────────────────────────────────────────────────────────

    After each A* search episode the collected data is:

        (obs_vector, g_star)   ← obs at node s, optimal g from s to goal

    The heuristic is trained to minimise:

        L(θ) = E[ loss( h_θ(s),  g_star(s) ) ]

    where  g_star(s) = g_goal − g(s)  (remaining path cost from s).

    ── Available Networks ───────────────────────────────────────────────────────

        net         params (obs=132)   best for
        ─────────   ─────────────────  ─────────────────────────────────────
        vanilla     ~132 K             general purpose; large datasets
        efficient   ~18 K              fast inference during A* search
        deep        ~530 K             high accuracy; deep feature extraction
        wide        ~530 K             broad interactions; small datasets
        dueling     ~100 K             interpretable; separate congestion +
                                       recovery estimation

    ── Usage ───────────────────────────────────────────────────────────────────

        # Create
        h_net = HeuristicModel(obs_dim=132, net='efficient')

        # Forward pass (training)
        x = torch.randn(32, 132)       # batch of obs vectors
        pred = h_net(x)                # (32, 1) predicted costs

        # Inference from numpy vector
        cost_estimate = h_net.predict(obs_vector)   # float

        # Plug into A* via a policy wrapper
        heuristic_fn = h_net.as_heuristic(env.obs_to_vector)
        # then assign: policy.heuristic = heuristic_fn

    Args:
        obs_dim:  length of the flat observation vector (e.g. 132 for
                  l2rpn_case14_sandbox with the default obs_to_vector).
        net:      which backbone to use. One of 'vanilla', 'efficient',
                  'deep', 'wide', 'dueling'. Default 'vanilla'.
        **kwargs: forwarded to the chosen backbone (hidden_dim, dropout,
                  n_blocks for 'deep', etc.).

    Raises:
        ValueError: if `net` is not one of the registered keys.
    """

    def __init__(self, obs_dim: int, net: str = "vanilla", **kwargs):
        super().__init__()

        if net not in _NETWORKS:
            raise ValueError(
                f"Unknown network '{net}'. "
                f"Available: {AVAILABLE_NETS}"
            )

        self.obs_dim = obs_dim
        self.net_name = net
        self.backbone = _NETWORKS[net](obs_dim, **kwargs)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, obs_dim) float tensor.

        Returns:
            (batch_size, 1) non-negative cost-to-goal estimates.
        """
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def predict(
        self,
        obs_vector: np.ndarray,
        device: Optional[str] = None,
    ) -> float:
        """
        Predict h(s) from a single numpy observation vector.

        Args:
            obs_vector: 1-D float32 array of length obs_dim.
            device:     torch device string; defaults to the model's current device.

        Returns:
            Non-negative float: estimated cost-to-recovery.
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(obs_vector).float().unsqueeze(0).to(device)
            return float(self(x).squeeze().item())

    def as_heuristic(self, obs_to_vector_fn: Callable) -> Callable:
        """
        Returns a callable  heuristic(obs) -> float  compatible with
        BasePolicy.heuristic() and AStarSearch.

        Example
        -------
            policy.heuristic = h_net.as_heuristic(env.obs_to_vector)

        Args:
            obs_to_vector_fn: function that converts a grid2op observation
                              to a 1-D numpy array of length obs_dim.

        Returns:
            Callable that accepts a grid2op obs and returns a float h(s).
        """
        def heuristic(obs) -> float:
            vec = obs_to_vector_fn(obs).astype(np.float32)
            return self.predict(vec)

        return heuristic

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"HeuristicModel(net='{self.net_name}', obs_dim={self.obs_dim}, "
            f"params={n_params:,})"
        )
