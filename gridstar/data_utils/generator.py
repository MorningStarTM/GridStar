"""
Safety predictor data generator.

Three collection strategies, each producing (obs_vector, label) pairs where
label = 1 means the grid stays safe for K consecutive do-nothing steps.

Strategies
──────────
  from_random_policy   — step with random topology actions across FULL episodes
                         (~8 064 steps each). Saves one episode_{id}.npz per
                         chronic. Handles done by reloading + fast-forwarding
                         to the same timestep, then continuing without any gap.

  from_trained_policy  — label every node in the A* search tree produced by a
                         trained (or random-baseline) policy. Focuses data
                         collection on states the planner actually visits.

  from_line_attacks    — systematically disconnect the most critical powerlines.
                         Creates a fresh env with shuffled chronics for each
                         (episode, line) pair; samples dst_step = ep * horizon
                         + rand to cover different times of day across episodes.

Saved files
───────────
  <save_dir>/random/episode_{id}.npz
  <save_dir>/trained/episode_{id}.npz
  <save_dir>/attack/line_{lid}_ep_{eid}.npz

Each .npz contains:
  obs_vectors : float32  (N, obs_dim)  — flat observation vectors
  labels      : float32  (N,)          — 0.0 (unsafe) or 1.0 (safe for K steps)
  rho_max     : float32  (N,)          — max line loading at each observation
  steps       : int32    (N,)          — timestep within the episode
  actions     : int32    (N,)          — action index into env.actions (-1 = line-disconnect)
"""

import os
import random
import warnings
from collections import defaultdict
from datetime import datetime
from typing import List, Optional, Tuple

import grid2op
import numpy as np
from grid2op.Exceptions import Grid2OpException, NoForecastAvailable

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

    ── Strategy 1 — Random Policy (full episodes) ──────────────────────────────

    Each chronic runs for its FULL duration (~8 064 steps at 5-min resolution).
    Every step collects one (obs_vector, label) pair. If the episode terminates
    early (done=True), the env is reloaded and fast-forwarded to the same
    timestep so the full step range is always covered.

    One file per chronic: random/episode_{id}.npz  (~8 000 rows each).

    ── Strategy 2 — Trained Policy (A* search nodes) ───────────────────────────

    When the grid becomes congested, run AStarSearch. Label every node in the
    resulting search tree (result.all_nodes). This focuses data on states the
    planner actually visits — exactly the distribution seen at inference.

    ── Strategy 3 — Line Attacks ───────────────────────────────────────────────

    For each (episode, line) pair a fresh env is created with shuffled chronics
    so every run sees scenarios in a different order. The target timestep is:

        dst_step = ep_id * horizon_per_episode + random.randint(0, horizon_per_episode)

    This samples episode 0 around the first 6 hours of each scenario,
    episode 1 around the next 6 hours, etc., building temporal diversity.

    After disconnecting the line, post-attack observations and follow-up
    do-nothing steps are collected, yielding a mix of label=0 (overload) and
    label=1 (self-recovery) samples.

    ── Usage ───────────────────────────────────────────────────────────────────

        from gridstar.data_utils.generator import SafetyDataGenerator

        gen = SafetyDataGenerator(
            env_name="l2rpn_case14_sandbox",
            k_steps=10,
            thermal_limit=0.98,
            save_dir="data/safety",
        )

        gen.from_random_policy(n_episodes=5, start_episode=0)

        from gridstar.search.astar import AStarSearch
        from gridstar.networks.policy import RandomPolicy
        policy   = RandomPolicy(n_actions=gen.env.action_size)
        searcher = AStarSearch(gen.env, policy, top_k=5, max_expansions=200)
        gen.from_trained_policy(searcher, n_episodes=5)

        gen.from_line_attacks(n_episodes=20, top_n_substations=5)

        obs_vecs, labels = gen.load()
        print(obs_vecs.shape, labels.mean())

    Args:
        env_name:      grid2op environment name (default l2rpn_case14_sandbox).
        k_steps:       number of do-nothing steps to simulate for label generation.
        thermal_limit: ρ threshold; line loading above this is considered unsafe.
        save_dir:      root directory for saved .npz files.
        use_lightsim:  use LightSimBackend if available (faster power flow).
        seed:          random seed for reproducibility.
    """

    def __init__(
        self,
        env_name: str = "l2rpn_case14_sandbox",
        k_steps: int = 10,
        thermal_limit: float = 0.98,
        save_dir: str = "data/safety",
        use_lightsim: bool = True,
        seed: int = 42,
        kaggle_dataset: Optional[str] = None,
        hf_dataset: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        self.env_name = env_name
        self.k_steps = k_steps
        self.thermal_limit = thermal_limit
        self.save_dir = save_dir
        self.seed = seed
        self.kaggle_dataset = kaggle_dataset
        self.hf_dataset = hf_dataset
        self.hf_token = hf_token
        random.seed(seed)
        np.random.seed(seed)

        self._backend_kwargs = {}
        if use_lightsim and _LIGHTSIM_AVAILABLE:
            self._backend_kwargs["backend"] = LightSimBackend()
            print("Using LightSimBackend.")
        else:
            print("Using default backend (install lightsim2grid for faster power flow).")

        self.env = GridStarEnv(
            env_name=env_name,
            thermal_limit=thermal_limit,
            backend=self._backend_kwargs.get("backend"),
        )
        self._do_nothing = self.env.actions[self.env.do_nothing_idx]

        os.makedirs(os.path.join(save_dir, "random"),  exist_ok=True)
        os.makedirs(os.path.join(save_dir, "trained"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "attack"),  exist_ok=True)

        if kaggle_dataset:
            self._init_kaggle_metadata(kaggle_dataset)
            print(f"Kaggle push enabled → {kaggle_dataset}")
        if hf_dataset:
            print(f"HuggingFace push enabled → {hf_dataset}")

    # ── Strategy 1: Random Policy — full episodes ─────────────────────────────

    def from_random_policy(
        self,
        n_episodes: Optional[int] = None,
        start_episode: int = 0,
        end_episode: Optional[int] = None,
    ) -> None:
        """
        Collect safety labels across FULL episodes (~8 064 steps per chronic).

        Steps through every timestep in max_episode_duration() using random
        topology actions. If the episode ends early (done=True at step i), the
        env is reloaded, fast-forwarded to step i-1, and one more action is
        taken so the collection resumes from the same point without a gap.

        One .npz file per chronic is written after the episode completes.

        Args:
            n_episodes:    number of chronics to run; defaults to all available.
                           Ignored when end_episode is set.
            start_episode: first chronic ID (inclusive).
            end_episode:   last chronic ID (exclusive). When set, overrides
                           n_episodes — runs exactly [start_episode, end_episode).
        """
        if end_episode is not None:
            end = min(end_episode, self.env.chronic_count)
        else:
            n_eps = n_episodes or self.env.chronic_count
            end   = min(start_episode + n_eps, self.env.chronic_count)
        raw   = self.env.env   # raw grid2op env

        for ep_id in range(start_episode, end):
            print(f"[random] episode {ep_id}/{end - 1}")

            raw.set_id(ep_id)
            obs = raw.reset()

            # Lists are local to this episode — naturally reset each iteration
            obs_vecs:    list = []
            labels:      list = []
            rho_vals:    list = []
            steps:       list = []
            action_idxs: list = []

            for i in range(raw.max_episode_duration()):
                try:
                    action_idx = random.randint(0, self.env.action_size - 1)
                    action     = self.env.actions[action_idx]
                    obs_, _reward, done, _info = raw.step(action)
                    #print(f"action {action_idx}  ρ_max={obs.rho.max():.4f}  done={done}")

                    # Compute everything before touching the lists — if _label
                    # throws, no partial append is left behind.
                    obs_vec = self.env.obs_to_vector(obs)
                    label   = float(self._label(obs))
                    rho     = float(obs.rho.max())

                    obs_vecs.append(obs_vec)
                    labels.append(label)
                    rho_vals.append(rho)
                    steps.append(i)
                    action_idxs.append(action_idx)

                    obs = obs_

                    if done:
                        # Reload at this episode and fast-forward to resume
                        raw.set_id(ep_id)
                        obs = raw.reset()
                        raw.fast_forward_chronics(max(i - 1, 0))

                        action_idx = random.randint(0, self.env.action_size - 1)
                        action     = self.env.actions[action_idx]
                        obs_, _reward, done, _info = raw.step(action)

                        obs_vec = self.env.obs_to_vector(obs)
                        label   = float(self._label(obs))
                        rho     = float(obs.rho.max())

                        obs_vecs.append(obs_vec)
                        labels.append(label)
                        rho_vals.append(rho)
                        steps.append(i)
                        action_idxs.append(action_idx)

                        obs = obs_

                except NoForecastAvailable as e:
                    print(f"  NoForecastAvailable at step {i}: {e}")
                    raw.set_id(ep_id)
                    obs = raw.reset()
                    raw.fast_forward_chronics(max(i - 1, 0))
                    continue

                except Grid2OpException as e:
                    print(f"  Grid2OpException at step {i}: {e}")
                    raw.set_id(ep_id)
                    obs = raw.reset()
                    raw.fast_forward_chronics(max(i - 1, 0))
                    continue

            # Save the full episode at once
            fname = os.path.join(self.save_dir, "random", f"episode_{ep_id}.npz")
            self._save(obs_vecs, labels, rho_vals, steps, fname, action_idxs)
            n1 = int(sum(labels))
            print(f"  {len(labels)} steps  safe={n1}  unsafe={len(labels) - n1}")
            self._push_to_kaggle(f"random episode_{ep_id}")
            self._push_to_hf(fname)

    # ── Strategy 2: Trained Policy (A* nodes) ────────────────────────────────

    def from_trained_policy(
        self,
        searcher,
        n_episodes: Optional[int] = None,
        start_episode: int = 0,
        end_episode: Optional[int] = None,
        max_steps_to_congestion: int = 2000,
    ) -> None:
        """
        Label every node in the A* search tree produced by `searcher`.

        Advances each episode to a congested state, runs the A* search, then
        labels each node in result.all_nodes using K-step do-nothing simulation.
        Nodes near the solution path tend to be label=1; dead-end branches
        tend to be label=0.

        Args:
            searcher:                AStarSearch instance (with any policy).
            n_episodes:              number of chronics to run. Ignored when
                                     end_episode is set.
            start_episode:           first chronic ID (inclusive).
            end_episode:             last chronic ID (exclusive). When set,
                                     overrides n_episodes.
            max_steps_to_congestion: steps to advance before giving up on
                                     finding congestion.
        """
        if end_episode is not None:
            end = min(end_episode, self.env.chronic_count)
        else:
            n_eps = n_episodes or self.env.chronic_count
            end   = min(start_episode + n_eps, self.env.chronic_count)

        for ep_id in range(start_episode, end):
            print(f"[trained] episode {ep_id}/{end - 1}")
            obs = self.env.reset(chronic_id=ep_id)
            obs = self.env.advance_to_congestion(obs, max_steps=max_steps_to_congestion)

            if not self.env.is_congested(obs):
                print("  no congestion found — skipping.")
                continue

            print(f"  ρ_max={self.env.get_rho_max(obs):.4f}  running A*...")
            result = searcher.search(obs)
            print(
                f"  found={result.found}  "
                f"nodes={len(result.all_nodes)}  "
                f"expanded={result.n_expanded}"
            )

            obs_vecs:    list = []
            labels:      list = []
            rho_vals:    list = []
            steps:       list = []
            action_idxs: list = []

            for node in result.all_nodes:
                obs_vec = self.env.obs_to_vector(node.obs)
                label   = float(self._label(node.obs))
                rho     = float(node.obs.rho.max())
                obs_vecs.append(obs_vec)
                labels.append(label)
                rho_vals.append(rho)
                steps.append(node.depth)
                action_idxs.append(getattr(node, "action_idx", self.env.do_nothing_idx))

            fname = os.path.join(self.save_dir, "trained", f"episode_{ep_id}.npz")
            self._save(obs_vecs, labels, rho_vals, steps, fname, action_idxs)
            n1 = int(sum(labels))
            print(f"  {len(labels)} nodes  safe={n1}  unsafe={len(labels) - n1}")
            self._push_to_kaggle(f"trained episode_{ep_id}")
            self._push_to_hf(fname)

    # ── Strategy 3: Line Attacks ──────────────────────────────────────────────

    def from_line_attacks(
        self,
        n_episodes: int = 20,
        start_episode: int = 0,
        end_episode: Optional[int] = None,
        top_n_substations: int = 5,
        steps_after_attack: int = 10,
        horizon_per_episode: int = 72,
    ) -> None:
        """
        Disconnect the most critical powerlines to create adversarial congestion.

        For each (episode, line) pair a FRESH env is created with shuffled
        chronics so every run sees scenarios in a different order. The target
        timestep advances with episode index:

            dst_step = ep_id * horizon_per_episode + random.randint(0, horizon_per_episode)

        This samples episode 0 from the first 6 hours of each scenario,
        episode 1 from the next 6 hours, and so on, ensuring temporal
        diversity across episodes.

        After disconnecting the line, the post-attack observation is collected
        (usually label=0). Then `steps_after_attack` do-nothing steps follow,
        yielding recovery observations (mix of label=0 and label=1).

        Args:
            n_episodes:          number of episode passes (each covers all lines
                                 across all scenarios). Ignored when end_episode
                                 is set.
            start_episode:       first episode-pass index (inclusive). Lets you
                                 split the ep_id range across parallel notebooks,
                                 e.g. notebook A: 0-9, notebook B: 10-19.
            end_episode:         last episode-pass index (exclusive). When set,
                                 overrides n_episodes — runs exactly
                                 [start_episode, end_episode).
            top_n_substations:   target lines connected to the top-N most
                                 connected substations.
            steps_after_attack:  do-nothing steps collected after each attack.
            horizon_per_episode: timestep window per episode (default 72 = 6 h).
        """
        attack_lines = self._get_attack_lines(top_n=top_n_substations)
        print(f"[attack] targeting {len(attack_lines)} lines: {attack_lines}")

        data_path     = self.env.env.get_path_env()
        scenario_path = self.env.env.chronics_handler.path
        n_scenarios   = len(os.listdir(scenario_path))

        end = end_episode if end_episode is not None else start_episode + n_episodes

        for ep_id in range(start_episode, end):
            for line_id in attack_lines:
                print(f"[attack] ep={ep_id}  line={line_id}")

                obs_vecs:    list = []
                labels:      list = []
                rho_vals:    list = []
                steps:       list = []
                action_idxs: list = []

                # Fresh env + shuffled chronics for diversity
                try:
                    backend = LightSimBackend() if _LIGHTSIM_AVAILABLE else None
                    mk_kwargs = {"dataset": data_path, "chronics_path": scenario_path}
                    if backend is not None:
                        mk_kwargs["backend"] = backend
                    ep_env = grid2op.make(**mk_kwargs)
                    ep_env.chronics_handler.shuffle(
                        shuffler=lambda x: x[
                            np.random.choice(len(x), size=len(x), replace=False)
                        ]
                    )
                except Exception as e:
                    print(f"  env creation failed: {e}")
                    continue

                for _chronic in range(n_scenarios):
                    try:
                        ep_env.reset()

                        # Timestep advances with episode so all hours are covered
                        dst_step = (
                            ep_id * horizon_per_episode
                            + random.randint(0, horizon_per_episode)
                        )
                        ep_env.fast_forward_chronics(max(dst_step - 1, 0))
                        obs, _r, done, _i = ep_env.step(ep_env.action_space({}))
                        if done:
                            continue

                        # Disconnect the target line
                        disconnect = np.zeros(obs.rho.shape, dtype=np.int32)
                        disconnect[line_id] = -1
                        attack_action = ep_env.action_space({"set_line_status": disconnect})
                        obs, _r, done, _i = ep_env.step(attack_action)
                        if done:
                            continue

                        # Post-attack observation (-1 = line-disconnect, not in action space)
                        obs_vecs.append(self.env.obs_to_vector(obs))
                        labels.append(float(self._label(obs)))
                        rho_vals.append(float(obs.rho.max()))
                        steps.append(dst_step)
                        action_idxs.append(-1)

                        # Follow-up do-nothing steps (recovery window)
                        for k in range(1, steps_after_attack + 1):
                            try:
                                obs_next, _r, done, _i = ep_env.step(self._do_nothing)
                                obs_vecs.append(self.env.obs_to_vector(obs))
                                labels.append(float(self._label(obs)))
                                rho_vals.append(float(obs.rho.max()))
                                steps.append(dst_step + k)
                                action_idxs.append(self.env.do_nothing_idx)
                                obs = obs_next
                                if done:
                                    break
                            except Grid2OpException:
                                break

                    except Exception as e:
                        print(f"  scenario error: {e}")
                        continue

                if obs_vecs:
                    fname = os.path.join(
                        self.save_dir, "attack", f"line_{line_id}_ep_{ep_id}.npz"
                    )
                    self._save(obs_vecs, labels, rho_vals, steps, fname, action_idxs)
                    n1 = int(sum(labels))
                    print(f"  {len(labels)} samples  safe={n1}  unsafe={len(labels) - n1}")
                    self._push_to_kaggle(f"attack line_{line_id}_ep_{ep_id}")
                    self._push_to_hf(fname)

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(
        self,
        strategy: Optional[str] = None,
        max_files: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load saved data from disk and concatenate into arrays.

        Args:
            strategy:  'random', 'trained', 'attack', or None to load all three.
            max_files: cap on files loaded per strategy (useful for quick tests).

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
            files = sorted(f for f in os.listdir(folder) if f.endswith(".npz"))
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
        print(
            f"Loaded {len(labels)} samples  "
            f"safe={n1}  unsafe={len(labels) - n1}  "
            f"balance={n1 / max(len(labels), 1):.2%}"
        )
        return obs_vectors, labels

    # ── Private helpers ───────────────────────────────────────────────────────

    def _label(self, obs) -> int:
        """K-step do-nothing simulation; returns 1 if safe throughout, 0 otherwise.
        Falls back to current-rho check when forecast data is unavailable."""
        try:
            return int(
                SafetyPredictor.collect_label(
                    obs,
                    self._do_nothing,
                    k_steps=self.k_steps,
                    thermal_limit=self.thermal_limit,
                )
            )
        except (NoForecastAvailable, Grid2OpException):
            return int(float(obs.rho.max()) < self.thermal_limit)

    def _save(
        self,
        obs_vecs: list,
        labels: list,
        rho_vals: list,
        steps: list,
        filepath: str,
        action_idxs: Optional[list] = None,
    ) -> None:
        """Save one episode's data to a compressed .npz file."""
        if not obs_vecs:
            return
        arrays = dict(
            obs_vectors=np.array(obs_vecs,   dtype=np.float32),
            labels=np.array(labels,          dtype=np.float32),
            rho_max=np.array(rho_vals,       dtype=np.float32),
            steps=np.array(steps,            dtype=np.int32),
        )
        if action_idxs is not None:
            arrays["actions"] = np.array(action_idxs, dtype=np.int32)
        np.savez_compressed(filepath, **arrays)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [{ts}] → {filepath}  ({len(obs_vecs)} steps)")

    # ── Kaggle push ───────────────────────────────────────────────────────────

    def _init_kaggle_metadata(self, dataset: str) -> None:
        """Write dataset-metadata.json required by the Kaggle API (once)."""
        import json
        meta_path = os.path.join(self.save_dir, "dataset-metadata.json")
        if os.path.exists(meta_path):
            return
        _, slug = dataset.split("/", 1)
        meta = {
            "title": slug.replace("-", " ").title(),
            "id": dataset,
            "licenses": [{"name": "CC0-1.0"}],
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  created {meta_path}")

    def _push_to_kaggle(self, version_note: str) -> None:
        """Push save_dir to Kaggle dataset as a new version (no-op if disabled).

        Downloads the current dataset first so files from other notebooks
        (different episode ranges) are not overwritten.
        """
        if not self.kaggle_dataset:
            return
        try:
            import kaggle
            kaggle.api.authenticate()

            # Pull existing files from the dataset into save_dir so that files
            # generated by other notebooks (different episode ranges) are preserved.
            try:
                kaggle.api.dataset_download_files(
                    self.kaggle_dataset,
                    path=self.save_dir,
                    unzip=True,
                    quiet=True,
                    force=False,   # skip files already present locally
                )
            except Exception:
                pass  # dataset may not exist yet on first push

            kaggle.api.dataset_create_version(
                self.save_dir,
                version_notes=version_note,
                quiet=True,
                convert_to_csv=False,
                delete_old_versions=False,
                dir_mode="zip",
            )
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"  [{ts}] pushed → kaggle:{self.kaggle_dataset}  ({version_note})")
        except Exception as e:
            print(f"  [kaggle push failed] {e}")

    def _push_to_hf(self, filepath: str) -> None:
        """Upload a single npz file to HuggingFace Hub dataset (no-op if disabled).

        Uses per-file upload so parallel notebooks never overwrite each other —
        each notebook only touches its own episode files.
        """
        if not self.hf_dataset:
            return
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=self.hf_token)
            # path_in_repo mirrors the folder structure under save_dir
            path_in_repo = os.path.relpath(filepath, self.save_dir).replace("\\", "/")
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=path_in_repo,
                repo_id=self.hf_dataset,
                repo_type="dataset",
                commit_message=f"add {path_in_repo}",
            )
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"  [{ts}] pushed → hf:{self.hf_dataset}/{path_in_repo}")
        except Exception as e:
            print(f"  [hf push failed] {e}")

    # ── Line attack utilities ─────────────────────────────────────────────────

    def _get_attack_lines(self, top_n: int = 5) -> List[int]:
        """Lines connected to the top-N most-connected substations."""
        connections = self._substation_connections()
        sorted_subs = sorted(connections.items(), key=lambda x: x[1], reverse=True)
        target_subs = [sub for sub, _ in sorted_subs[:top_n]]
        lines_map   = self._lines_for_substations(target_subs)

        seen: set = set()
        attack_lines: List[int] = []
        for sub in target_subs:
            for lid in lines_map[sub]:
                if lid not in seen:
                    attack_lines.append(lid)
                    seen.add(lid)
        return attack_lines

    def _substation_connections(self) -> dict:
        counts: dict = defaultdict(int)
        env = self.env.env
        for lid in range(env.n_line):
            counts[env.line_or_to_subid[lid]] += 1
            counts[env.line_ex_to_subid[lid]]  += 1
        return dict(counts)

    def _lines_for_substations(self, target_subs: list) -> dict:
        result: dict = {sub: [] for sub in target_subs}
        env = self.env.env
        for lid in range(env.n_line):
            for sub in (env.line_or_to_subid[lid], env.line_ex_to_subid[lid]):
                if sub in result:
                    result[sub].append(lid)
        return result
