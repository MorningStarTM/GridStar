"""
Safety predictor — binary classifier for sustained grid safety.
"""

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


# ── Shared residual block ─────────────────────────────────────────────────────


class _ResBlock(nn.Module):
    """Linear → LayerNorm → ReLU → Dropout → Linear → LayerNorm + skip → ReLU."""

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


# ── Backbone classifiers — all return a raw logit (batch, 1) ─────────────────
# No final activation here; Sigmoid is applied in SafetyPredictor.predict_proba()
# so that training can use the numerically stable BCEWithLogitsLoss.


class _VanillaClassifier(nn.Module):
    """
    3-hidden-layer MLP with BatchNorm.

    Architecture: obs_dim → H → H/2 → H/4 → 1
    Norm:         BatchNorm1d after each hidden linear.
    Activation:   ReLU; raw logit on output.

    Args:
        obs_dim:    length of the flat observation vector.
        hidden_dim: width H of the first hidden layer (default 256).
        dropout:    dropout rate after each ReLU (default 0).
    """

    def __init__(self, obs_dim: int, hidden_dim: int = 256, dropout: float = 0.0, **_):
        super().__init__()
        H = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, H),
            nn.BatchNorm1d(H),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H, H // 2),
            nn.BatchNorm1d(H // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(H // 2, H // 4),
            nn.ReLU(),
            nn.Linear(H // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _EfficientClassifier(nn.Module):
    """
    2-hidden-layer MLP without normalisation — lowest inference latency.

    Architecture: obs_dim → H → H/2 → 1
    Norm:         none.
    Activation:   ReLU; raw logit on output.

    Preferred default because the predictor is called on every A* candidate
    node and must not become the search bottleneck.

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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DeepClassifier(nn.Module):
    """
    Residual MLP for deep feature extraction.

    Architecture:
        Input projection: obs_dim → H  (Linear + ReLU)
        Residual blocks:  n_blocks × (H → H → H with LayerNorm skip)
        Output head:      H → H/4 → 1

    Args:
        obs_dim:    length of the flat observation vector.
        hidden_dim: shared hidden width H (default 256).
        n_blocks:   number of residual blocks (default 3).
        dropout:    dropout inside each residual block (default 0).
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        n_blocks: int = 3,
        dropout: float = 0.0,
        **_,
    ):
        super().__init__()
        H = hidden_dim
        self.proj = nn.Sequential(nn.Linear(obs_dim, H), nn.ReLU())
        self.blocks = nn.Sequential(*[_ResBlock(H, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(H, H // 4),
            nn.ReLU(),
            nn.Linear(H // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.proj(x)))


class _WideClassifier(nn.Module):
    """
    Wide 2-hidden-layer MLP with LayerNorm and LeakyReLU.

    Architecture: obs_dim → H → H/2 → 1
    Norm:         LayerNorm (robust to small batches).
    Activation:   LeakyReLU(0.1) — avoids dead neurons in wide layers.

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
            nn.Linear(H, H // 2),
            nn.LayerNorm(H // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(H // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _EnsembleClassifier(nn.Module):
    """
    N independent EfficientClassifiers whose logits are averaged.

    Averaging logits is equivalent to averaging log-odds, which approximates
    averaging probabilities when predictions are near 0.5 and is numerically
    more stable (no sigmoid before averaging).

    The per-member predictions are also exposed via member_probs() for
    uncertainty estimation: high variance → low confidence → keep searching.

    Args:
        obs_dim:    length of the flat observation vector.
        n_members:  number of ensemble members (default 5).
        hidden_dim: hidden width of each member (default 128).
        dropout:    dropout within each member (default 0).
    """

    def __init__(
        self,
        obs_dim: int,
        n_members: int = 5,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        **_,
    ):
        super().__init__()
        self.members = nn.ModuleList(
            [_EfficientClassifier(obs_dim, hidden_dim, dropout) for _ in range(n_members)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack([m(x) for m in self.members], dim=0)  # (M, B, 1)
        return stacked.mean(dim=0)                                   # (B, 1)

    def member_probs(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, n_members) — per-member probability; use for uncertainty estimates."""
        return torch.cat([torch.sigmoid(m(x)) for m in self.members], dim=1)


# ── Registry ──────────────────────────────────────────────────────────────────

_CLASSIFIERS: dict = {
    "vanilla":  _VanillaClassifier,
    "efficient": _EfficientClassifier,
    "deep":     _DeepClassifier,
    "wide":     _WideClassifier,
    "ensemble": _EnsembleClassifier,
}

AVAILABLE_NETS = list(_CLASSIFIERS.keys())


# ── SafetyPredictor ───────────────────────────────────────────────────────────


class SafetyPredictor(nn.Module):
    """
    Binary classifier: P(grid stays safe for K do-nothing steps).

    ── Problem ─────────────────────────────────────────────────────────────────

    A* search needs a cheap goal test at every candidate node. The naive test —
    simulate K consecutive do-nothing steps and check ρ_max < threshold each
    time — costs K simulate() calls per node, which is too slow when the search
    generates thousands of nodes.

    The SafetyPredictor replaces this with a single neural forward pass:

        P_safe = SafetyPredictor(obs_vector)   ← one forward pass
        if P_safe > 0.9:
            verify with real simulate() calls   ← only for the likely winner

    This 2-stage check (fast neural filter → expensive exact verification) keeps
    the search fast while avoiding false positives.

    ── What "safe for K steps" means ───────────────────────────────────────────

    Starting from observation s, apply the do-nothing action K consecutive times
    (following the load trajectory that grid2op generates). The grid is considered
    sustainably safe if:

        ρ_max(s_t) < thermal_limit    for all  t ∈ {1, 2, …, K}

    A state that is safe right now but would overflow in 3 steps (due to rising
    load) is NOT a valid goal — the A* solution must produce states that hold
    without further intervention.

    ── Training ────────────────────────────────────────────────────────────────

    1. Collect labels with SafetyPredictor.collect_label():
          label = 1   if do-nothing for K steps keeps ρ_max < threshold
          label = 0   otherwise

    2. Train with binary cross-entropy:
          loss = BCEWithLogitsLoss(model(x), labels)

    3. Calibrate the threshold (default 0.9) on a held-out set to balance
       precision (avoid false goals) vs recall (don't miss real goals).

    ── Two-Stage Goal Test in A* ────────────────────────────────────────────────

    The workflow in AStarSearch is:

        # Stage 1 — fast neural filter (one forward pass)
        obs_vec = env.obs_to_vector(obs)
        if safety_predictor.is_safe(obs_vec, threshold=0.9):

            # Stage 2 — exact verification (K simulate calls)
            do_nothing = env.actions[env.do_nothing_idx]
            verified = SafetyPredictor.collect_label(
                obs, do_nothing, k_steps=K, thermal_limit=0.98
            )
            if verified:
                return solution   # confirmed sustained safety

    ── Available Networks ───────────────────────────────────────────────────────

        net         params (obs=132)   best for
        ─────────   ─────────────────  ─────────────────────────────────────
        vanilla     ~87 K              general purpose; large datasets
        efficient   ~18 K              fast A* inference (default)
        deep        ~420 K             high accuracy; complex grid patterns
        wide        ~270 K             broad feature interactions
        ensemble    ~90 K (5×18K)      uncertainty-aware; best calibration

    ── Usage ───────────────────────────────────────────────────────────────────

        # Create
        sp = SafetyPredictor(obs_dim=132, net='efficient')
        sp = SafetyPredictor(obs_dim=132, net='ensemble', n_members=7)

        # Training
        logits = sp(x_batch)                             # (B, 1) raw logits
        loss   = nn.BCEWithLogitsLoss()(logits, labels)  # labels: (B, 1) float

        # Inference
        p    = sp.predict_proba(obs_vector)    # float in [0, 1]
        safe = sp.is_safe(obs_vector, 0.9)     # bool

        # Plug into A* goal test
        goal_fn = sp.as_goal_test(env.obs_to_vector, threshold=0.9)

        # Collect training label from one obs
        label = SafetyPredictor.collect_label(
            obs, do_nothing_action, k_steps=10, thermal_limit=0.98
        )

    Args:
        obs_dim:  length of the flat observation vector.
        net:      backbone architecture. One of 'vanilla', 'efficient',
                  'deep', 'wide', 'ensemble'. Default 'efficient'.
        **kwargs: forwarded to the backbone (hidden_dim, dropout,
                  n_blocks for 'deep', n_members for 'ensemble').

    Raises:
        ValueError: if `net` is not a registered key.
    """

    def __init__(self, obs_dim: int, net: str = "efficient", **kwargs):
        super().__init__()

        if net not in _CLASSIFIERS:
            raise ValueError(
                f"Unknown network '{net}'. Available: {AVAILABLE_NETS}"
            )

        self.obs_dim = obs_dim
        self.net_name = net
        self.backbone = _CLASSIFIERS[net](obs_dim, **kwargs)

    # ------------------------------------------------------------------
    # Core forward — returns raw logit for BCEWithLogitsLoss
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, obs_dim) float tensor.

        Returns:
            (batch_size, 1) raw logits (NOT probabilities).
            Pass directly to nn.BCEWithLogitsLoss during training.
        """
        return self.backbone(x)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        obs_vector: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> float:
        """
        Predict P(safe for K steps) from a single numpy observation vector.

        Args:
            obs_vector: 1-D float32 array of length obs_dim.
            device:     torch device; defaults to the model's current device.

        Returns:
            Float in [0, 1]: estimated probability of sustained safety.
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(obs_vector).float().unsqueeze(0).to(device)
            logit = self(x).squeeze()
            return float(torch.sigmoid(logit).item())

    def is_safe(
        self,
        obs_vector: np.ndarray,
        threshold: float = 0.9,
        device: Optional[torch.device] = None,
    ) -> bool:
        """
        Boolean safety prediction from a single numpy observation vector.

        A conservative threshold (default 0.9) reduces false positives —
        declaring an unsafe state as safe is more harmful than missing a
        goal and continuing the search.

        Args:
            obs_vector: 1-D float32 array of length obs_dim.
            threshold:  minimum probability to declare the state safe.
            device:     torch device; defaults to the model's current device.

        Returns:
            True if P(safe) ≥ threshold.
        """
        return self.predict_proba(obs_vector, device) >= threshold

    def as_goal_test(
        self,
        obs_to_vector_fn: Callable,
        threshold: float = 0.9,
    ) -> Callable:
        """
        Returns a callable  goal_test(obs) -> bool  for the fast Stage-1
        filter in A* search.

        The returned function converts a raw grid2op observation to a vector
        and runs the neural classifier in one forward pass. It does NOT perform
        the K-step exact verification — use collect_label() for Stage 2.

        Args:
            obs_to_vector_fn: converts a grid2op obs to a 1-D numpy array.
            threshold:        confidence threshold for a positive (safe) prediction.

        Returns:
            Callable obs → bool.

        Example
        -------
            goal_test = sp.as_goal_test(env.obs_to_vector, threshold=0.9)
            if goal_test(node.obs):
                # run exact K-step verification before accepting
        """
        def goal_test(obs) -> bool:
            vec = obs_to_vector_fn(obs).astype(np.float32)
            return self.is_safe(vec, threshold)

        return goal_test

    def member_uncertainty(
        self,
        obs_vector: np.ndarray,
        device: Optional[torch.device] = None,
    ) -> Optional[float]:
        """
        Std deviation of per-member probabilities for the ensemble backbone.

        Returns None for non-ensemble backbones. High uncertainty (std > 0.15)
        means the ensemble members disagree — prefer not to accept this state
        as a goal without exact verification.

        Args:
            obs_vector: 1-D float32 array of length obs_dim.
            device:     torch device; defaults to the model's current device.

        Returns:
            Float std deviation, or None if backbone is not 'ensemble'.
        """
        if not isinstance(self.backbone, _EnsembleClassifier):
            return None
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(obs_vector).float().unsqueeze(0).to(device)
            probs = self.backbone.member_probs(x).squeeze(0)   # (n_members,)
            return float(probs.std().item())

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    @staticmethod
    def collect_label(
        obs,
        do_nothing_action,
        k_steps: int = 10,
        thermal_limit: float = 0.98,
    ) -> bool:
        """
        Simulate K consecutive do-nothing steps from obs via chained simulate()
        calls. Returns True (label = 1) if ρ_max stays below thermal_limit for
        ALL K steps, False (label = 0) otherwise.

        Uses the same chained obs.simulate() mechanism as A* multi-step
        lookahead, so no environment state is mutated.

        Args:
            obs:               grid2op observation to start from.
            do_nothing_action: the do-nothing grid2op Action object
                               (e.g. env.actions[env.do_nothing_idx]).
            k_steps:           number of steps to simulate ahead (default 10,
                               i.e. 50 minutes at 5-min resolution).
            thermal_limit:     ρ threshold above which a line is overloaded.

        Returns:
            True  → state is sustainably safe for K steps.
            False → state overflows or ends within K steps.

        Example
        -------
            do_nothing = env.actions[env.do_nothing_idx]
            label = SafetyPredictor.collect_label(obs, do_nothing, k_steps=10)
            # use (obs_vector, float(label)) as a training pair
        """
        current = obs
        for _ in range(k_steps):
            sim_obs, _, done, info = current.simulate(do_nothing_action)
            if done or float(sim_obs.rho.max()) >= thermal_limit:
                return False
            current = sim_obs
        return True

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_params = sum(p.numel() for p in self.parameters())
        return (
            f"SafetyPredictor(net='{self.net_name}', obs_dim={self.obs_dim}, "
            f"params={n_params:,})"
        )
