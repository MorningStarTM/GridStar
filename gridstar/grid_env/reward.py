import numpy as np


class RewardFunction:
    """
    Reward and cost functions for GridStar.

    ── Design Philosophy ───────────────────────────────────────────────────────

    GridStar uses two coupled objectives that must stay consistent:

      1. A* edge cost  g(s,a,s')   — a non-negative cost that A* *minimises*
                                     along a search path.

      2. RL reward     r(s,a,s')   — a scalar that the policy network *maximises*
                                     during training.

    The two are related by:

        r(s, a, s')  ≈  – edge_cost(s, a, s')  +  terminal_signal

    so that the policy gradient and the A* planner optimize the same underlying
    objective. All weights (overflow_coeff, offline_coeff, …) are shared, which
    means tuning one formula tunes both.

    ── Reward Components ───────────────────────────────────────────────────────

    The reward after transitioning to state s' via action a has five terms:

    1. Overflow penalty  (dense, per-line, quadratic)
       ─────────────────────────────────────────────
       For every line i with loading ρ_i above the thermal limit θ:

           overflow_penalty = overflow_coeff × Σ_i  max(ρ_i − θ, 0)²

       Quadratic (squared excess) is used because:
         - It is zero and smooth below threshold (no gradient cliff at θ).
         - It grows faster than linear for severe overloads, teaching the agent
           to care more about very hot lines than mildly warm ones.
         - It gives a dense per-line signal rather than only the worst line.

    2. Safety margin reward  (dense, per-line)
       ────────────────────────────────────────
       For lines that are within their limits:

           margin_reward = margin_coeff × mean_i  max(θ − ρ_i, 0)

       Rewarding headroom (not just penalising excess) encourages the agent to
       keep the grid *comfortably* safe, not just barely below the threshold.

    3. Offline line penalty  (step penalty)
       ─────────────────────────────────────
       Each disconnected line incurs a fixed cost:

           offline_penalty = offline_coeff × n_offline_lines

       Disconnected lines reduce redundancy; a healthy grid keeps all lines on.

    4. Action complexity penalty  (regularisation)
       ──────────────────────────────────────────────
       Any topology action other than do-nothing incurs a tiny penalty:

           action_penalty = action_coeff  (if action ≠ do-nothing, else 0)

       This discourages the agent from making unnecessary topology changes that
       could destabilise the grid or waste its substation cooldown budget.

    5. Terminal signals  (sparse, episode-level)
       ────────────────────────────────────────────
         +goal_bonus      when ρ_max < θ  (congestion resolved — episode goal)
         +blackout_penalty when done=True  (cascading failure / game over)

    ── A* Edge Cost ────────────────────────────────────────────────────────────

    edge_cost() is a simplified, non-negative version used by the A* planner:

        cost(s', a) = max(ρ_max − θ, 0)        ← worst-line severity only
                    + offline_coeff × n_offline
                    + action_coeff × (a ≠ do-nothing)

    It uses the *maximum* ρ rather than the per-line sum to keep g values small
    and comparable across search paths of different lengths.

    ── Relationship Summary ─────────────────────────────────────────────────────

        Component           Edge cost (A*)          RL reward
        ─────────────────   ─────────────────────   ──────────────────────────
        Overflow            max(ρ_max − θ, 0)       −Σ max(ρ_i − θ, 0)²  (all lines)
        Safety margin       (not used)              +mean max(θ − ρ_i, 0) (all lines)
        Offline lines       offline_coeff × n        −offline_coeff × n
        Action penalty      action_coeff × (a≠dn)   −action_coeff × (a≠dn)
        Goal bonus          (not used; goal = g=0)   +goal_bonus
        Blackout penalty    (pruned via done flag)   +blackout_penalty  (negative)
    """

    def __init__(
        self,
        thermal_limit: float = 0.98,
        overflow_coeff: float = 1.0,
        margin_coeff: float = 0.1,
        offline_coeff: float = 0.5,
        action_coeff: float = 0.01,
        goal_bonus: float = 1.0,
        blackout_penalty: float = -10.0,
    ):
        """
        Args:
            thermal_limit:     ρ threshold above which a line is considered overloaded.
            overflow_coeff:    weight on the per-line quadratic overflow penalty.
            margin_coeff:      weight on the per-line safety-margin reward.
            offline_coeff:     penalty per disconnected line (shared with edge_cost).
            action_coeff:      flat penalty for any non-do-nothing action (shared).
            goal_bonus:        reward added when the grid is fully safe (ρ_max < θ).
            blackout_penalty:  reward added when the episode ends in a blackout.
        """
        self.thermal_limit = thermal_limit
        self.overflow_coeff = overflow_coeff
        self.margin_coeff = margin_coeff
        self.offline_coeff = offline_coeff
        self.action_coeff = action_coeff
        self.goal_bonus = goal_bonus
        self.blackout_penalty = blackout_penalty

    # ------------------------------------------------------------------
    # RL reward  (to maximise during training)
    # ------------------------------------------------------------------

    def __call__(
        self, obs, action_idx: int, done: bool, do_nothing_idx: int = 0
    ) -> float:
        """
        Compute the RL reward for arriving at `obs` via `action_idx`.

        Args:
            obs:            grid2op observation of the *resulting* state s'.
            action_idx:     integer index of the action taken.
            done:           True if the episode ended (blackout or scenario end).
            do_nothing_idx: index of the do-nothing action (default 0).

        Returns:
            Scalar reward in approximately [blackout_penalty, goal_bonus].
        """
        if done:
            return float(self.blackout_penalty)

        rho = obs.rho                                     # shape (n_lines,)
        n_lines = max(len(rho), 1)
        n_offline = int((~obs.line_status).sum())

        # 1. Overflow penalty — quadratic, per line
        excess = np.maximum(rho - self.thermal_limit, 0.0)
        overflow_penalty = self.overflow_coeff * float(np.sum(excess ** 2))

        # 2. Safety margin reward — linear, per line
        margin = np.maximum(self.thermal_limit - rho, 0.0)
        margin_reward = self.margin_coeff * float(np.mean(margin))

        # 3. Offline line penalty
        offline_penalty = self.offline_coeff * n_offline

        # 4. Action complexity penalty
        action_penalty = self.action_coeff if action_idx != do_nothing_idx else 0.0

        # 5. Goal bonus
        goal = self.goal_bonus if float(rho.max()) < self.thermal_limit else 0.0

        return goal + margin_reward - overflow_penalty - offline_penalty - action_penalty

    # ------------------------------------------------------------------
    # A* edge cost  (to minimise during search)
    # ------------------------------------------------------------------

    def edge_cost(
        self, child_obs, action_idx: int, do_nothing_idx: int = 0
    ) -> float:
        """
        Non-negative transition cost used by AStarSearch to accumulate g(s).

        Uses the worst-line ρ (not the per-line sum) to keep path costs small
        and comparable regardless of the grid size.

        Args:
            child_obs:      grid2op observation of state s' (after simulate).
            action_idx:     integer index of the action taken.
            do_nothing_idx: index of the do-nothing action (default 0).

        Returns:
            Non-negative float cost for this edge.
        """
        rho_max = float(child_obs.rho.max())
        n_offline = int((~child_obs.line_status).sum())
        action_penalty = self.action_coeff if action_idx != do_nothing_idx else 0.0
        return (
            max(rho_max - self.thermal_limit, 0.0)
            + self.offline_coeff * n_offline
            + action_penalty
        )
