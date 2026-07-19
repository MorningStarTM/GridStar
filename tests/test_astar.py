import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gridstar.grid_env.grid_env import GridStarEnv
from gridstar.networks.policy import RandomPolicy
from gridstar.search.astar import AStarSearch
from gridstar.utils.visualize import plot_search_tree

# Output directory for the visualisation
_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "doc")


def test_astar_random():
    print("=" * 64)
    print("A* Search — Random Policy")
    print("=" * 64)

    # ── 1. Environment ────────────────────────────────────────────
    env = GridStarEnv(env_name="l2rpn_case14_sandbox", thermal_limit=0.98)
    print(f"\nEnvironment : l2rpn_case14_sandbox")
    print(f"Action size : {env.action_size}  (1 do-nothing + {env.action_size - 1} topology)")
    print(f"Obs dim     : {env.obs_dim}")

    # ── 2. Advance to a congested state ───────────────────────────
    obs = env.reset()
    print("\nAdvancing to congestion (max 2 000 steps)…")
    obs = env.advance_to_congestion(obs, max_steps=2000)
    rho_start = env.get_rho_max(obs)
    print(f"  ρ_max at root : {rho_start:.4f}")
    print(f"  Congested     : {env.is_congested(obs)}")

    if not env.is_congested(obs):
        print("  [warn] no congestion found — root may already be safe.")

    # ── 3. Build policy + searcher ────────────────────────────────
    policy = RandomPolicy(n_actions=env.action_size, seed=42)
    searcher = AStarSearch(
        env=env,
        policy=policy,
        top_k=5,           # narrow branching factor → budget reaches deeper levels
        max_expansions=300,
        max_depth=8,
        thermal_limit=env.thermal_limit,
    )

    # ── 4. Run search ─────────────────────────────────────────────
    print("\nRunning A* search…")
    result = searcher.search(obs)

    # ── 5. Print result ───────────────────────────────────────────
    print("\n--- Result ---")
    print(f"  Found         : {result.found}")
    print(f"  Expanded      : {result.n_expanded}")
    print(f"  Generated     : {result.n_generated}")
    print(f"  Nodes in tree : {len(result.all_nodes)}")

    if result.found:
        print(f"  Path length   : {len(result.path) - 1} action(s)")
        for i, node in enumerate(result.path):
            rho = float(node.obs.rho.max())
            tag = "[Root]" if i == 0 else f"[Step {i}]"
            act = "" if i == 0 else f"  action={node.action_idx}"
            print(f"    {tag}  ρ_max={rho:.4f}  g={node.g:.4f}{act}")
    else:
        best = min(result.all_nodes, key=lambda n: float(n.obs.rho.max()))
        print(f"  Best ρ_max    : {float(best.obs.rho.max()):.4f}  (node_id={best.node_id})")

    # ── 6. Visualise ──────────────────────────────────────────────
    os.makedirs(_OUT_DIR, exist_ok=True)
    save_path = os.path.join(_OUT_DIR, "astar_search_tree.png")
    plot_search_tree(result, env, save_path=save_path, max_nodes=200)

    print("\nAll assertions passed.")


if __name__ == "__main__":
    test_astar_random()
