import os, sys
from pathlib import Path

IS_KAGGLE = os.path.exists('/kaggle')


def _find_repo_root():
    p = Path.cwd()
    for candidate in [p, p.parent, p.parent.parent]:
        if (candidate / 'gridstar').is_dir():
            return str(candidate)
    return str(p)


REPO_ROOT = '/kaggle/working/GridStar'    if IS_KAGGLE else _find_repo_root()
SAVE_DIR  = '/kaggle/working/data/safety' if IS_KAGGLE else str(Path(REPO_ROOT) / 'data' / 'safety')

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

print(f'REPO_ROOT : {REPO_ROOT}')
print(f'SAVE_DIR  : {SAVE_DIR}')
print(f'Kaggle    : {IS_KAGGLE}')

from gridstar.data_utils.generator import SafetyDataGenerator

gen = SafetyDataGenerator(
    env_name="l2rpn_case14_sandbox",
    save_dir=SAVE_DIR,
    hf_dataset="ernestbeckham/gridstar-safety-data",
    hf_token=os.environ.get("HF_TOKEN"),   # set via: export HF_TOKEN=hf_xxx (or Kaggle Secrets)
    delete_after_push=True,                # keeps local disk usage flat — 30GB is plenty
)

# ── Strategy 1: Random Policy ─────────────────────────────────────────────────
# Notebook/instance A: episodes 0–49   |  B: episodes 50–99  | etc.
#gen.from_random_policy(start_episode=0, end_episode=50)

# ── Strategy 2: Line Attacks ──────────────────────────────────────────────────
# Notebook/instance A: episode-passes 0–9   |  B: 10–19  | etc.
gen.from_line_attacks(
    start_episode=800,
    end_episode=1000,
    top_n_substations=10,
    steps_after_attack=10,
    horizon_per_episode=72,
)

# ── Strategy 3: Trained Policy (A* nodes) ────────────────────────────────────
# from gridstar.search.astar import AStarSearch
# from gridstar.networks.policy import RandomPolicy
# policy   = RandomPolicy(n_actions=gen.env.action_size)
# searcher = AStarSearch(gen.env, policy, top_k=5, max_expansions=200)
# gen.from_trained_policy(searcher, start_episode=0, end_episode=20)
