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
from gridstar.networks.safety import SafetyPredictor
from gridstar.networks.policy import RandomPolicy
from gridstar.search.astar import AStarSearch

# ── Config (env vars — set via infra/__main__.py for EC2, or by hand for Kaggle/local) ──
HF_DATASET        = "ernestbeckham/gridstar-safety-data"
HF_MODEL_REPO      = os.environ.get("HF_MODEL_REPO", "ErnestBeckham/gridstar-safety-model")
SAFETY_CHECKPOINT  = os.environ.get("SAFETY_CHECKPOINT", "safety_predictor_slice1.safetensors")
HF_TOKEN           = os.environ.get("HF_TOKEN")

START_EPISODE      = int(os.environ.get("START_EPISODE", 0))
END_EPISODE        = int(os.environ.get("END_EPISODE", 50))
SEED                = int(os.environ.get("SEED", 42))

TOP_K              = int(os.environ.get("TOP_K", 30))     # was 10 — more candidates survive per node
MAX_EXPANSIONS     = int(os.environ.get("MAX_EXPANSIONS", 200))
MAX_DEPTH          = int(os.environ.get("MAX_DEPTH", 10))  # was 5 — longer paths = more g_star samples per success
SAFETY_THRESHOLD   = float(os.environ.get("SAFETY_THRESHOLD", 0.9))

MAX_STEPS_TO_CONGESTION = int(os.environ.get("MAX_STEPS_TO_CONGESTION", 2000))
MAX_ATTEMPTS            = int(os.environ.get("MAX_ATTEMPTS", 5))
# None lets from_astar_heuristic default this to searcher.max_depth (MAX_DEPTH above)
MIN_FORECAST_BUFFER     = (
    int(os.environ["MIN_FORECAST_BUFFER"]) if "MIN_FORECAST_BUFFER" in os.environ else None
)
# Extra do-nothing steps after congestion is first detected, to let it deepen
# before A* searches from it — without this, every success was a trivial
# 1-step fix (empirically confirmed: 20/20 sampled files had exactly 2 nodes).
CONGESTION_RIDE_OUT_STEPS = int(os.environ.get("CONGESTION_RIDE_OUT_STEPS", 4))

print(f'Episode range: {START_EPISODE} -> {END_EPISODE}  seed={SEED}')

# ── Pull the trained SafetyPredictor checkpoint from HF ──────────────────────
import json

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file

ckpt_path = hf_hub_download(
    repo_id=HF_MODEL_REPO,
    filename=SAFETY_CHECKPOINT,
    token=HF_TOKEN,
)
print(f"SafetyPredictor checkpoint: {ckpt_path}")

with safe_open(ckpt_path, framework="pt", device="cpu") as f:
    meta = f.metadata() or {}

obs_dim  = int(meta.get("obs_dim", 132))
net_name = meta.get("net_name", "efficient")
threshold = float(meta.get("threshold", SAFETY_THRESHOLD))

# hidden_dim/net_kwargs were only added to checkpoint metadata recently —
# older checkpoints (saved before that fix) won't have them, so fall back to
# an env var override. Set SAFETY_HIDDEN_DIM explicitly if loading an older
# checkpoint trained with a non-default hidden_dim.
hidden_dim = int(meta.get("hidden_dim", os.environ.get("SAFETY_HIDDEN_DIM", 128)))
net_kwargs = json.loads(meta["net_kwargs"]) if "net_kwargs" in meta else {}
print(f"  obs_dim={obs_dim}  net={net_name}  hidden_dim={hidden_dim}  net_kwargs={net_kwargs}  threshold={threshold}")

safety = SafetyPredictor(obs_dim=obs_dim, net=net_name, hidden_dim=hidden_dim, **net_kwargs)
safety.load_state_dict(load_file(ckpt_path, device="cpu"))
safety.eval()

# ── Build the generator + A* searcher ─────────────────────────────────────────
gen = SafetyDataGenerator(
    env_name="l2rpn_case14_sandbox",
    save_dir=SAVE_DIR,
    hf_dataset=HF_DATASET,
    hf_token=HF_TOKEN,
    delete_after_push=True,   # keeps local disk usage flat
    seed=SEED,
)

# RandomPolicy is the standard bootstrap baseline — h=0, random action
# sampling. safety_fn is what makes the search actually tractable: without a
# trained SafetyPredictor as the Stage-1 goal filter, A* falls back to the
# exact-only goal check and explores far more nodes per search.
policy = RandomPolicy(n_actions=gen.env.action_size, seed=SEED)
searcher = AStarSearch(
    gen.env,
    policy,
    top_k=TOP_K,
    max_expansions=MAX_EXPANSIONS,
    max_depth=MAX_DEPTH,
    safety_fn=safety.as_goal_test(gen.env.obs_to_vector, threshold=threshold),
)

# ── Generate ───────────────────────────────────────────────────────────────────
gen.from_astar_heuristic(
    searcher,
    start_episode=START_EPISODE,
    end_episode=END_EPISODE,
    max_steps_to_congestion=MAX_STEPS_TO_CONGESTION,
    max_attempts=MAX_ATTEMPTS,
    min_forecast_buffer=MIN_FORECAST_BUFFER,
    congestion_ride_out_steps=CONGESTION_RIDE_OUT_STEPS,
)
