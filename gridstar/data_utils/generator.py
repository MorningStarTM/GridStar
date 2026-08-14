"""
Safety predictor data generator.

Three collection strategies, each producing (obs_vector, label) pairs where
label = 1 means the grid stays safe for K consecutive do-nothing steps.

Strategies
──────────
  from_random_policy   — step with random topology actions across full episodes.
                         Produces broad coverage of the state space including
                         both safe and mildly congested observations.

  from_trained_policy  — label every node in the A* search tree produced by a
                         trained (or random-baseline) policy.  Focuses data
                         collection on states the planner actually visits.

  from_line_attacks    — systematically disconnect the most critical powerlines
                         to create diverse congestion patterns. Each attack
                         generates observations ranging from severe overload
                         (label=0) through gradual recovery (label=1).

Saved files
───────────
  data/safety/random/episode_{id}.npz
  data/safety/trained/episode_{id}.npz
  data/safety/attack/line_{lid}_ep_{eid}.npz

Each .npz contains:
  obs_vectors : float32  (N, obs_dim)  — flat observation vectors
  labels      : float32  (N,)          — 0.0 (unsafe) or 1.0 (safe for K steps)
  rho_max     : float32  (N,)          — max line loading at each observation
  steps       : int32    (N,)          — timestep within the episode
"""

import os
import random
import warnings
from collections import defaultdict
from typing import Callable, List, Optional, Tuple

import numpy as np
from grid2op.Exceptions import Grid2OpException, NoForecastAvailable
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction

from gridstar.grid_env.grid_env import GridStarEnv
from gridstar.networks.safety import SafetyPredictor

warnings.filterwarnings("ignore")

# ── Optional LightSim backend ─────────────────────────────────────────────────
try:
    from lightsim2grid import LightSimBackend
    _LIGHTSIM_AVAILABLE = True
except ImportError:
    _LIGHTSIM_AVAILABLE = False


class SafetyDataGenerator:
    """
    Generates (obs_vector, label) training pairs for SafetyPredictor.

    ── Label Definition ────────────────────────────────────────────────────────

    For each collected observation `obs`:

        label = 1  if  ρ_max(simulate(do-nothing, t)) < thermal_limit
                       for ALL t in {1, 2, …, k_steps}

        label = 0  otherwise (overflows or blackout within K steps)

    Labels are generated using SafetyPredictor.collect_label(), which chains
    obs.simulate(do-nothing) K times — no environment state is mutated.

    ── Strategy 1 — Random Policy ──────────────────────────────────────────────

    Step the environment with random topology actions for full episodes.
    Every step produces an observation that is labelled and stored.
    Handles `done=True` by resetting and fast-forwarding to the same timestep,
    matching the reference DataGeneration pattern exactly.

    Produces broad, diverse coverage — many safe states (label=1) with
    occasional congestion, especially at high-load timesteps.

    ── Strategy 2 — Trained Policy (A* search nodes) ───────────────────────────

    When the grid becomes congested, run AStarSearch. Label every node in the
    resulting search tree (result.all_nodes). This focuses data on states the
    planner actually visits — exactly the distribution seen at inference.

    Nodes near the solution path tend to be mildly congested or safe (label=1);
    nodes on dead-end branches tend to be severely congested (label=0).

    ── Strategy 3 — Line Attacks ───────────────────────────────────────────────

    Systematically disconnect the K most critical powerlines (those connected to
    the most highly connected substations) to create diverse congestion patterns.

    For each (line, scenario, timestep) triple:
      1. Fast-forward to a random timestep within the scenario.
      2. Disconnect the target line via set_line_status action.
      3. Collect the post-attack observation (likely label=0).
      4. Step with do-nothing for several more steps, collecting each obs.
         (Some of these will naturally recover → label=1; others stay hot → label=0.)

    This creates a challenging, adversarial training set that includes states
    far from the normal operating point.

    ── Dataset Balance ─────────────────────────────────────────────────────────

    Line attacks heavily skew toward label=0; random policy skews toward label=1.
    Combine both strategies and use class-weighted BCEWithLogitsLoss:

        pos_weight = n_negative / n_positive   # weight for positive (safe) class

    ── Usage ───────────────────────────────────────────────────────────────────

        from gridstar.data_utils.generator import SafetyDataGenerator

        gen = SafetyDataGenerator(
            env_name="l2rpn_case14_sandbox",
            k_steps=10,
            thermal_limit=0.98,
            save_dir="data/safety",
        )

        # Collect ~2 000 (obs, label) pairs per episode via random policy
        gen.from_random_policy(n_episodes=5, start_episode=0)

        # Label A* search-tree nodes from a trained policy
        from gridstar.search.astar import AStarSearch
        from gridstar.networks.policy import RandomPolicy
        policy  = RandomPolicy(n_actions=gen.env.action_size)
        searcher = AStarSearch(gen.env, policy, top_k=5, max_expansions=200)
        gen.from_trained_policy(searcher, n_episodes=5)

        # Create adversarial congestion via line attacks
        gen.from_line_attacks(n_episodes=20, top_n_substations=5)

        # Load everything for training
        obs_vecs, labels = gen.load()
        print(obs_vecs.shape, labels.mean())   # label rate ≈ fraction safe

    Args:
        env_name:      grid2op environment name (default l2rpn_case14_sandbox).
        k_steps:       number of do-nothing steps to simulate for label generation.
        thermal_limit: ρ threshold; line loading above this is considered unsafe.
        save_dir:      root directory for saved .npz files.
        use_lightsim:  use LightSimBackend if available (faster power flow).
        seed:          random seed for reproducible episode ordering.
    """

    def __init__(
        self,
        env_name: str = "l2rpn_case14_sandbox",
        k_steps: int = 10,
        thermal_limit: float = 0.98,
        save_dir: str = "data/safety",
        use_lightsim: bool = True,
        seed: int = 42,
    ):
        self.env_name = env_name
        self.k_steps = k_steps
        self.thermal_limit = thermal_limit
        self.save_dir = save_dir
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        backend = None
        if use_lightsim and _LIGHTSIM_AVAILABLE:
            backend = LightSimBackend()
            print("Using LightSimBackend.")
        else:
            print("Using default backend (install lightsim2grid for faster power flow).")

        self.env = GridStarEnv(env_name=env_name, thermal_limit=thermal_limit, backend=backend)
        self._do_nothing = self.env.actions[self.env.do_nothing_idx]

        os.makedirs(os.path.join(save_dir, "random"),  exist_ok=True)
        os.makedirs(os.path.join(save_dir, "trained"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "attack"),  exist_ok=True)

    # ── Strategy 1: Random Policy ─────────────────────────────────────────────

    def from_random_policy(
        self,
        n_episodes: Optional[int] = None,
        start_episode: int = 0,
        max_steps_per_episode: Optional[int] = None,
    ) -> None:
        """
        Collect safety labels by stepping with uniformly random topology actions.

        Each environment step produces one (obs_vector, label) pair. Episodes
        where `done=True` are handled by resetting and fast-forwarding to the
        same timestep — the subsequent observation is still collected, matching
        the reference DataGeneration pattern.

        Args:
            n_episodes:             number of chronics to run; defaults to all.
            start_episode:          first chronic ID to start from.
            max_steps_per_episode:  cap on steps per episode; defaults to full
                                    episode length.
        """
        n_eps = n_episodes or self.env.chronic_count
        end = min(start_episode + n_eps, self.env.chronic_count)

        for ep_id in range(start_episode, end):
            print(f"[random] episode {ep_id}/{end - 1}")
            obs = self.env.reset(chronic_id=ep_id)
            max_steps = max_steps_per_episode or self.env.env.max_episode_duration()

            obs_vecs, labels, rho_vals, steps = [], [], [], []

            for step in range(max_steps):
                try:
                    action_idx = random.randint(0, self.env.action_size - 1)
                    obs_next, _, done, _ = self.env.step(action_idx)

                    obs_vecs.append(self.env.obs_to_vector(obs))
                    labels.append(float(self._label(obs)))
                    rho_vals.append(float(obs.rho.max()))
                    steps.append(step)

                    obs = obs_next

                    if done:
                        # Reset, fast-forward to the same step, collect one more obs
                        self.env.env.set_id(ep_id)
                        obs = self.env.reset()
                        self.env.env.fast_forward_chronics(max(step - 1, 0))
                        obs_next, _, done, _ = self.env.step(action_idx)

                        obs_vecs.append(self.env.obs_to_vector(obs))
                        labels.append(float(self._label(obs)))
                        rho_vals.append(float(obs.rho.max()))
                        steps.append(step)

                        obs = obs_next

                except NoForecastAvailable:
                    self.env.env.set_id(ep_id)
                    obs = self.env.reset()
                    self.env.env.fast_forward_chronics(max(step - 1, 0))
                    continue
                except Grid2OpException as e:
                    print(f"  Grid2OpException at step {step}: {e}")
                    self.env.env.set_id(ep_id)
                    obs = self.env.reset()
                    self.env.env.fast_forward_chronics(max(step - 1, 0))
                    continue

            fname = os.path.join(self.save_dir, "random", f"episode_{ep_id}.npz")
            self._save(obs_vecs, labels, rho_vals, steps, fname)
            n1 = sum(labels)
            print(f"  saved {len(labels)} samples  (safe={int(n1)}, unsafe={int(len(labels)-n1)})")

    # ── Strategy 2: Trained Policy (A* nodes) ────────────────────────────────

    def from_trained_policy(
        self,
        searcher,
        n_episodes: Optional[int] = None,
        start_episode: int = 0,
        max_steps_to_congestion: int = 2000,
    ) -> None:
        """
        Label every node in the A* search tree produced by `searcher`.

        Advances each episode to a congested state, runs the A* search, then
        labels each node in result.all_nodes using K-step do-nothing simulation.
        This yields observations from states the planner actually visits —
        exactly the distribution seen during inference.

        Args:
            searcher:                 AStarSearch instance (with any policy).
            n_episodes:               number of chronics to run.
            start_episode:            first chronic ID.
            max_steps_to_congestion:  steps to advance before giving up on
                                      finding congestion.
        """
        n_eps = n_episodes or self.env.chronic_count
        end = min(start_episode + n_eps, self.env.chronic_count)

        for ep_id in range(start_episode, end):
            print(f"[trained] episode {ep_id}/{end - 1}")
            obs = self.env.reset(chronic_id=ep_id)
            obs = self.env.advance_to_congestion(obs, max_steps=max_steps_to_congestion)

            if not self.env.is_congested(obs):
                print("  no congestion found — skipping.")
                continue

            print(f"  ρ_max={self.env.get_rho_max(obs):.4f}. Running A*...")
            result = searcher.search(obs)
            print(f"  found={result.found}  nodes={len(result.all_nodes)}  expanded={result.n_expanded}")

            obs_vecs, labels, rho_vals, steps = [], [], [], []
            for node in result.all_nodes:
                obs_vecs.append(self.env.obs_to_vector(node.obs))
                labels.append(float(self._label(node.obs)))
                rho_vals.append(float(node.obs.rho.max()))
                steps.append(node.depth)

            fname = os.path.join(self.save_dir, "trained", f"episode_{ep_id}.npz")
            self._save(obs_vecs, labels, rho_vals, steps, fname)
            n1 = sum(labels)
            print(f"  saved {len(labels)} samples  (safe={int(n1)}, unsafe={int(len(labels)-n1)})")

    # ── Strategy 3: Line Attacks ─────────────────────────────────────────────

    def from_line_attacks(
        self,
        n_episodes: int = 20,
        top_n_substations: int = 5,
        steps_after_attack: int = 10,
        horizon_per_episode: int = 72,
    ) -> None:
        """
        Disconnect the most critical powerlines to create adversarial congestion.

        For each (episode, target_line) pair the procedure is:
          1. Fast-forward to a random timestep within the episode.
          2. Apply a set_line_status action to disconnect target_line.
          3. Collect the post-attack observation (usually label=0).
          4. Step with do-nothing for `steps_after_attack` more steps and
             collect each observation. As the grid (sometimes) self-recovers or
             worsens, this yields a mix of label=0 and label=1 samples.

        The lines targeted are those connected to the most highly connected
        substations — attacking these creates the largest grid disruption.

        Args:
            n_episodes:          number of random scenario + timestep draws.
            top_n_substations:   number of high-connectivity substations to
                                 focus attacks on.
            steps_after_attack:  do-nothing steps collected after each attack.
            horizon_per_episode: timestep is drawn uniformly from
                                 [0, horizon_per_episode) within the scenario.
        """
        attack_lines = self._get_attack_lines(top_n=top_n_substations)
        print(f"[attack] targeting {len(attack_lines)} lines: {attack_lines}")

        for ep_id in range(n_episodes):
            for line_id in attack_lines:
                print(f"[attack] episode={ep_id}  line={line_id}")

                obs_vecs, labels, rho_vals, steps = [], [], [], []

                try:
                    # Reset to a random scenario
                    chronic_id = random.randint(0, self.env.chronic_count - 1)
                    obs = self.env.reset(chronic_id=chronic_id)

                    # Fast-forward to a random timestep
                    dst_step = random.randint(1, horizon_per_episode)
                    self.env.env.fast_forward_chronics(dst_step - 1)
                    obs, _, done, _ = self.env.env.step(self._do_nothing)
                    if done:
                        continue

                    # --- Disconnect the target line ---
                    disconnect = np.zeros(obs.rho.shape, dtype=np.int32)
                    disconnect[line_id] = -1
                    attack_action = self.env.env.action_space(
                        {"set_line_status": disconnect}
                    )
                    obs, _, done, _ = self.env.env.step(attack_action)
                    if done:
                        continue

                    # Collect post-attack observation
                    obs_vecs.append(self.env.obs_to_vector(obs))
                    labels.append(float(self._label(obs)))
                    rho_vals.append(float(obs.rho.max()))
                    steps.append(dst_step)

                    # Collect follow-up do-nothing steps
                    for k in range(1, steps_after_attack + 1):
                        try:
                            obs_next, _, done, _ = self.env.env.step(self._do_nothing)
                            obs_vecs.append(self.env.obs_to_vector(obs))
                            labels.append(float(self._label(obs)))
                            rho_vals.append(float(obs.rho.max()))
                            steps.append(dst_step + k)
                            obs = obs_next
                            if done:
                                break
                        except Grid2OpException:
                            break

                except Grid2OpException as e:
                    print(f"  Grid2OpException: {e}")
                    continue

                if obs_vecs:
                    tag = f"line_{line_id}_ep_{ep_id}"
                    fname = os.path.join(self.save_dir, "attack", f"{tag}.npz")
                    self._save(obs_vecs, labels, rho_vals, steps, fname)
                    n1 = sum(labels)
                    print(f"  saved {len(labels)} samples  (safe={int(n1)}, unsafe={int(len(labels)-n1)})")

    # ── Load ─────────────────────────────────────────────────────────────────

    def load(
        self,
        strategy: Optional[str] = None,
        max_files: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load saved data from disk and concatenate into arrays.

        Args:
            strategy:   one of 'random', 'trained', 'attack', or None to load
                        all three strategies together.
            max_files:  cap on files loaded per strategy (useful for quick tests).

        Returns:
            obs_vectors: float32 array of shape (N, obs_dim).
            labels:      float32 array of shape (N,).
        """
        strategies = ["random", "trained", "attack"] if strategy is None else [strategy]
        all_obs, all_labels = [], []

        for strat in strategies:
            folder = os.path.join(self.save_dir, strat)
            if not os.path.isdir(folder):
                continue
            files = sorted(
                [f for f in os.listdir(folder) if f.endswith(".npz")]
            )
            if max_files is not None:
                files = files[:max_files]
            for fname in files:
                data = np.load(os.path.join(folder, fname))
                all_obs.append(data["obs_vectors"])
                all_labels.append(data["labels"])

        if not all_obs:
            raise FileNotFoundError(
                f"No data found in {self.save_dir}. Run a collection strategy first."
            )

        obs_vectors = np.concatenate(all_obs, axis=0).astype(np.float32)
        labels      = np.concatenate(all_labels, axis=0).astype(np.float32)

        n1 = int(labels.sum())
        print(f"Loaded {len(labels)} samples  "
              f"(safe={n1}, unsafe={len(labels)-n1}, "
              f"balance={n1/max(len(labels),1):.2%})")
        return obs_vectors, labels

    # ── Private helpers ───────────────────────────────────────────────────────

    def _label(self, obs) -> int:
        """
        Run K do-nothing simulate steps from obs.
        Returns 1 if grid stays safe throughout, 0 otherwise.
        No environment state is mutated (uses chained obs.simulate()).
        """
        return int(
            SafetyPredictor.collect_label(
                obs,
                self._do_nothing,
                k_steps=self.k_steps,
                thermal_limit=self.thermal_limit,
            )
        )

    def _save(
        self,
        obs_vecs: list,
        labels: list,
        rho_vals: list,
        steps: list,
        filepath: str,
    ) -> None:
        """Save one episode's data to a compressed .npz file."""
        if not obs_vecs:
            return
        np.savez_compressed(
            filepath,
            obs_vectors=np.array(obs_vecs, dtype=np.float32),
            labels=np.array(labels, dtype=np.float32),
            rho_max=np.array(rho_vals, dtype=np.float32),
            steps=np.array(steps, dtype=np.int32),
        )
        print(f"  → {filepath}")

    # ── Line attack utilities ─────────────────────────────────────────────────

    def _get_attack_lines(self, top_n: int = 5) -> List[int]:
        """
        Return IDs of powerlines connected to the top-N most connected substations.
        These lines cause the largest disruption when disconnected.
        """
        connections = self._substation_connections()
        sorted_subs = sorted(connections.items(), key=lambda x: x[1], reverse=True)
        target_subs = [sub for sub, _ in sorted_subs[:top_n]]
        lines_map = self._lines_for_substations(target_subs)

        attack_lines: List[int] = []
        seen = set()
        for sub in target_subs:
            for line_id in lines_map[sub]:
                if line_id not in seen:
                    attack_lines.append(line_id)
                    seen.add(line_id)
        return attack_lines

    def _substation_connections(self) -> dict:
        """
        Returns {sub_id: n_connected_lines} for all substations.
        Uses the raw grid2op env for topology information.
        """
        counts: dict = defaultdict(int)
        env = self.env.env
        for line_id in range(env.n_line):
            counts[env.line_or_to_subid[line_id]] += 1
            counts[env.line_ex_to_subid[line_id]] += 1
        return dict(counts)

    def _lines_for_substations(self, target_subs: list) -> dict:
        """
        Returns {sub_id: [line_id, …]} for lines touching each target substation.
        """
        result: dict = {sub: [] for sub in target_subs}
        env = self.env.env
        for line_id in range(env.n_line):
            for sub in (env.line_or_to_subid[line_id], env.line_ex_to_subid[line_id]):
                if sub in result:
                    result[sub].append(line_id)
        return result
