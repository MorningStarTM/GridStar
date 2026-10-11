import os, sys
sys.path.insert(0, '/kaggle/working/GridStar')

from main import download_hf_dataset, load_npz_dataset
from gridstar.training.safety_trainer import SafetyPredictorTrainer

# ── Load data from HuggingFace ────────────────────────────────────────────────
# Pull one non-overlapping SLICE per session — e.g. first 12h run trains
# slice 0, next 12h run trains slice 1, etc. Deterministic: same num_slices
# always carves the file list the same way, so no file is ever trained on
# twice or skipped once you've gone through all slices.
HF_DATASET    = "ernestbeckham/gridstar-safety-data"
HF_TOKEN      = os.environ.get("HF_TOKEN")   # set in Kaggle Secrets → HF_TOKEN
STRATEGY      = "attack"   # "random" | "trained" | "attack" | None (= everything)
START_EPISODE = None        # e.g. 600 — leave None to not filter by range
END_EPISODE   = None        # e.g. 900

NUM_SLICES    = 5           # total number of sessions you plan to split training across
SLICE_INDEX   = 0           # CHANGE THIS each session: 0, then 1, then 2, ... up to NUM_SLICES-1

# Alternative: random (possibly-overlapping) subsample instead of a slice —
# set FRACTION and leave NUM_SLICES/SLICE_INDEX as None to use this instead.
FRACTION      = None        # e.g. 0.2 for ~20% of matching files, randomly sampled (seeded)
SEED          = 42          # seeds the fraction sampling, for reproducibility

data_dir = download_hf_dataset(
    HF_DATASET,
    token=HF_TOKEN,
    strategy=STRATEGY,
    start_episode=START_EPISODE,
    end_episode=END_EPISODE,
    fraction=FRACTION,
    num_slices=NUM_SLICES,
    slice_index=SLICE_INDEX,
    seed=SEED,
)
print(f"cached at {data_dir}")

obs_vectors, labels = load_npz_dataset(data_dir, strategy=STRATEGY, max_files=None)
print(f"Dataset: {obs_vectors.shape[0]} samples  obs_dim={obs_vectors.shape[1]}  pos_rate={labels.mean():.2%}")

# ── Held-out test split ───────────────────────────────────────────────────────
# fit() does its own internal train/val split (val_split below) for early
# stopping — but evaluating on the same obs_vectors/labels passed to fit() is
# data leakage (the model trained on most of it). Carve out a separate test
# set FIRST, never shown to fit() at all, for a trustworthy final metric.
import numpy as np

TEST_SPLIT = 0.15
rng = np.random.default_rng(42)
idx = rng.permutation(len(labels))
n_test = int(len(labels) * TEST_SPLIT)
test_idx, train_idx = idx[:n_test], idx[n_test:]

train_obs, train_labels = obs_vectors[train_idx], labels[train_idx]
test_obs,  test_labels  = obs_vectors[test_idx],  labels[test_idx]
print(f"Train: {len(train_labels)}  Test (held out): {len(test_labels)}")

# ── Train ────────────────────────────────────────────────────────────────────
trainer = SafetyPredictorTrainer(
    obs_dim=obs_vectors.shape[1],
    net="efficient",
    hidden_dim=128,
    dropout=0.1,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=256,
    epochs=50,
    val_split=0.15,       # internal split of train_obs/train_labels, for early stopping only
    patience=10,
    threshold=0.9,
    save_dir="/kaggle/working/checkpoints/safety",
    checkpoint_name="safety_predictor.safetensors",
    device="auto",
    seed=42,
)

history = trainer.fit(train_obs, train_labels)

# ── Final evaluation — on the held-out test split, never seen during fit() ──
metrics = trainer.evaluate(test_obs, test_labels)
print("── Final evaluation (held-out test set) ──")
print(f"  loss      = {metrics['loss']:.4f}")
print(f"  accuracy  = {metrics['acc']:.4f}")
print(f"  precision = {metrics['precision']:.4f}")
print(f"  recall    = {metrics['recall']:.4f}")
print(f"  f1        = {metrics['f1']:.4f}")
print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
print(f"\nCheckpoint: {trainer.checkpoint_path}")
