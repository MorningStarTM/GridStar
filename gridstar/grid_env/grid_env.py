import numpy as np
import grid2op
from grid2op.Action import PlayableAction
from grid2op.Parameters import Parameters


class GridStarEnv:
    """
    Grid2Op environment wrapper for GridStar A* search.

    Handles environment creation, action set construction,
    observation processing, and simulate() calls.
    """

    def __init__(
        self,
        env_name="l2rpn_case14_sandbox",
        thermal_limit=0.98,
        backend=None,
    ):
        params = Parameters()
        params.MAX_SUB_CHANGED = 1

        kwargs = {
            "action_class": PlayableAction,
            "param": params,
        }
        if backend is not None:
            kwargs["backend"] = backend

        self.env = grid2op.make(env_name, **kwargs)
        self.thermal_limit = thermal_limit
        self.obs_space = self.env.observation_space
        self.act_space = self.env.action_space

        self._build_action_set()
        self._obs_dim = None

    def _build_action_set(self):
        do_nothing = self.act_space({})
        all_topos = self.act_space.get_all_unitary_topologies_set(self.act_space)
        self.actions = [do_nothing] + all_topos
        self.action_size = len(self.actions)
        self.do_nothing_idx = 0

    def reset(self, chronic_id=None):
        if chronic_id is not None:
            self.env.set_id(chronic_id)
        obs = self.env.reset()
        return obs

    def step(self, action_idx):
        action = self.actions[action_idx]
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def simulate(self, obs, action_idx):
        action = self.actions[action_idx]
        sim_obs, sim_reward, sim_done, sim_info = obs.simulate(action)
        return sim_obs, sim_reward, sim_done, sim_info

    def is_congested(self, obs):
        return obs.rho.max() >= self.thermal_limit

    def is_safe(self, obs):
        return not obs.done__ if hasattr(obs, "done__") else obs.rho.max() < self.thermal_limit

    def get_rho_max(self, obs):
        return float(obs.rho.max())

    def obs_to_vector(self, obs):
        rho = obs.rho.astype(np.float32)
        load_p = obs.load_p.astype(np.float32)
        load_q = obs.load_q.astype(np.float32)
        gen_p = obs.gen_p.astype(np.float32)
        gen_q = obs.gen_q.astype(np.float32)
        line_status = obs.line_status.astype(np.float32)
        topo_vect = obs.topo_vect.astype(np.float32)
        timestep = np.array([obs.current_step / 8064.0], dtype=np.float32)

        vec = np.concatenate([
            rho, load_p, load_q, gen_p, gen_q,
            line_status, topo_vect, timestep,
        ])
        return vec

    @property
    def obs_dim(self):
        if self._obs_dim is None:
            obs = self.env.reset()
            self._obs_dim = len(self.obs_to_vector(obs))
        return self._obs_dim

    def get_legal_actions(self, obs):
        legal = [self.do_nothing_idx]
        for i in range(1, self.action_size):
            action = self.actions[i]
            sub_id = int(action.as_dict().get("set_bus_vect", {}).get("modif_subs_id", [-1])[0])
            if sub_id >= 0 and obs.time_before_cooldown_sub[sub_id] > 0:
                continue
            legal.append(i)
        return legal

    def advance_to_congestion(self, obs=None, max_steps=8064):
        if obs is None:
            obs = self.reset()

        do_nothing = self.actions[self.do_nothing_idx]
        for _ in range(max_steps):
            if self.is_congested(obs):
                return obs
            obs, _, done, _ = self.env.step(do_nothing)
            if done:
                obs = self.reset()

        return obs

    @property
    def n_sub(self):
        return self.env.n_sub

    @property
    def n_line(self):
        return self.env.n_line

    @property
    def n_load(self):
        return self.env.n_load

    @property
    def n_gen(self):
        return self.env.n_gen

    @property
    def chronic_count(self):
        return len(self.env.chronics_handler.subpaths)
