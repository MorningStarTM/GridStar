import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from gridstar.grid_env.grid_env import GridStarEnv


def test_basic():
    env = GridStarEnv(env_name="l2rpn_case14_sandbox", max_actions=30)

    print(f"Environment: l2rpn_case14_sandbox")
    print(f"Action size: {env.action_size} (1 do-nothing + {env.action_size - 1} topology)")
    print(f"Substations: {env.n_sub}")
    print(f"Lines: {env.n_line}")
    print(f"Loads: {env.n_load}")
    print(f"Generators: {env.n_gen}")
    print(f"Chronics: {env.chronic_count}")
    print()

    obs = env.reset()
    rho_max = env.get_rho_max(obs)
    print(f"After reset:")
    print(f"  rho_max = {rho_max:.4f}")
    print(f"  congested = {env.is_congested(obs)}")
    print(f"  safe = {env.is_safe(obs)}")
    print()

    vec = env.obs_to_vector(obs)
    print(f"Obs vector: dim = {len(vec)}")
    print(f"  rho:         {obs.rho.shape}")
    print(f"  load_p:      {obs.load_p.shape}")
    print(f"  load_q:      {obs.load_q.shape}")
    print(f"  gen_p:       {obs.gen_p.shape}")
    print(f"  gen_q:       {obs.gen_q.shape}")
    print(f"  line_status: {obs.line_status.shape}")
    print(f"  topo_vect:   {obs.topo_vect.shape}")
    print(f"  + 1 timestep")
    expected_dim = (obs.rho.shape[0] + obs.load_p.shape[0] + obs.load_q.shape[0]
                    + obs.gen_p.shape[0] + obs.gen_q.shape[0]
                    + obs.line_status.shape[0] + obs.topo_vect.shape[0] + 1)
    assert len(vec) == expected_dim, f"Expected {expected_dim}, got {len(vec)}"
    print(f"  total = {expected_dim} (verified)")
    print()

    print(f"obs_dim property = {env.obs_dim}")
    print()

    sim_obs, sim_reward, sim_done, sim_info = env.simulate(obs, 0)
    print(f"simulate(do_nothing):")
    print(f"  rho_max = {env.get_rho_max(sim_obs):.4f}")
    print(f"  reward  = {sim_reward:.4f}")
    print(f"  done    = {sim_done}")
    print()

    if env.action_size > 1:
        sim_obs2, sim_reward2, sim_done2, sim_info2 = env.simulate(obs, 1)
        print(f"simulate(action_1):")
        print(f"  rho_max  = {env.get_rho_max(sim_obs2):.4f}")
        print(f"  reward   = {sim_reward2:.4f}")
        print(f"  done     = {sim_done2}")
        print(f"  illegal  = {sim_info2.get('is_illegal', False)}")
        print()

    legal = env.get_legal_actions(obs)
    print(f"Legal actions: {len(legal)} / {env.action_size}")
    print()

    obs2, reward2, done2, info2 = env.step(0)
    print(f"step(do_nothing):")
    print(f"  rho_max = {env.get_rho_max(obs2):.4f}")
    print(f"  reward  = {reward2:.4f}")
    print(f"  done    = {done2}")
    print()

    print("Advancing to congestion...")
    obs = env.reset()
    cong_obs = env.advance_to_congestion(obs, max_steps=500)
    rho = env.get_rho_max(cong_obs)
    print(f"  rho_max = {rho:.4f}")
    if rho >= 0.98:
        print(f"  congestion found at step {cong_obs.current_step}")
    else:
        print(f"  no congestion within 500 steps (normal for some chronics)")

    print()
    print("All tests passed.")


if __name__ == "__main__":
    test_basic()
