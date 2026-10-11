"""
GridStar command-line entry point.

Usage
─────
  python main.py <command> [options]

Available commands
──────────────────
  safety_trainer    Train SafetyPredictor on collected grid2op data.

Examples
────────
  # Train with defaults (efficient net, 50 epochs, data in data/safety)
  python main.py safety_trainer

  # Train with a wider net and more epochs, loading only attack data
  python main.py safety_trainer --net wide --epochs 100 --strategy attack

  # Deep net with residual blocks, custom learning rate, GPU
  python main.py safety_trainer --net deep --n-blocks 4 --lr 5e-4 --device cuda

  # Quick smoke-test: 2 files per strategy, 5 epochs
  python main.py safety_trainer --max-files 2 --epochs 5

  # Train directly from a HuggingFace dataset (downloads + caches automatically)
  python main.py safety_trainer --hf-dataset ernestbeckham/gridstar-safety-data

  # Pull only a range of random-policy episodes — much lighter on GPU/disk/network
  python main.py safety_trainer --hf-dataset owner/gridstar-safety-data \\
      --strategy random --start-episode 0 --end-episode 50

  # Pull only ~20% of the attack-strategy files, randomly — quick smaller pull
  python main.py safety_trainer --hf-dataset owner/gridstar-safety-data \\
      --strategy attack --fraction 0.2

  # Train across multiple 12h Kaggle sessions, one non-overlapping slice each:
  #   session 1:
  python main.py safety_trainer --hf-dataset owner/gridstar-safety-data \\
      --strategy attack --num-slices 5 --slice-index 0
  #   session 2 (different slice, same total slice count — no overlap with session 1):
  python main.py safety_trainer --hf-dataset owner/gridstar-safety-data \\
      --strategy attack --num-slices 5 --slice-index 1
"""

import argparse
import glob
import os
import sys

import numpy as np


# ── Sub-command argument definitions ─────────────────────────────────────────


def _add_safety_trainer_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Data")
    g.add_argument(
        "--data-dir", default="data/safety",
        help="Root directory of saved .npz files (sub-dirs: random/, trained/, attack/).",
    )
    g.add_argument(
        "--strategy", default=None, choices=["random", "trained", "attack"],
        help="Load only one strategy; default loads all three together.",
    )
    g.add_argument(
        "--max-files", type=int, default=None,
        help="Cap on files loaded per strategy (useful for quick tests).",
    )
    g.add_argument(
        "--hf-dataset", default=None,
        help="HuggingFace dataset repo id (e.g. owner/gridstar-safety-data). "
             "When set, downloads/caches the dataset and trains from it instead of --data-dir.",
    )
    g.add_argument(
        "--hf-token", default=None,
        help="HuggingFace token. Falls back to $HF_TOKEN or `huggingface-cli login` cache.",
    )
    g.add_argument(
        "--start-episode", type=int, default=None,
        help="First episode id to pull from HF (inclusive). Requires --strategy "
             "and --hf-dataset. Downloads only this range instead of the whole dataset.",
    )
    g.add_argument(
        "--end-episode", type=int, default=None,
        help="Last episode id to pull from HF (exclusive). Requires --strategy and --hf-dataset.",
    )
    g.add_argument(
        "--fraction", type=float, default=None,
        help="Pull only this fraction (0-1] of matching files from HF, randomly sampled "
             "(seeded by --seed). Useful for a quick smaller pull, but doesn't guarantee "
             "coverage across repeated calls — prefer --num-slices/--slice-index for "
             "training across multiple sessions without overlap. Requires --hf-dataset.",
    )
    g.add_argument(
        "--num-slices", type=int, default=None,
        help="Split matching HF files into this many non-overlapping, deterministic "
             "chunks. Use with --slice-index to train on one chunk per session (e.g. "
             "one 12h Kaggle run per slice) with guaranteed no duplicate/skipped files "
             "across the full set of slices. Requires --hf-dataset.",
    )
    g.add_argument(
        "--slice-index", type=int, default=None,
        help="Which chunk (0-based) to pull when --num-slices is set. "
             "E.g. --num-slices 5 --slice-index 0 pulls the first 1/5; "
             "--slice-index 1 the next 1/5 on your next session; etc.",
    )

    g = p.add_argument_group("Model")
    g.add_argument(
        "--net", default="efficient",
        choices=["vanilla", "efficient", "deep", "wide", "ensemble"],
        help="Backbone architecture.",
    )
    g.add_argument(
        "--hidden-dim", type=int, default=128,
        help="Width of first hidden layer.",
    )
    g.add_argument(
        "--dropout", type=float, default=0.0,
        help="Dropout rate (0 = disabled).",
    )
    g.add_argument(
        "--n-blocks", type=int, default=3,
        help="Number of residual blocks (--net deep only).",
    )
    g.add_argument(
        "--n-members", type=int, default=5,
        help="Ensemble members (--net ensemble only).",
    )

    g = p.add_argument_group("Training")
    g.add_argument("--lr",           type=float, default=1e-3,  help="Adam learning rate.")
    g.add_argument("--weight-decay", type=float, default=1e-4,  help="Adam L2 weight decay.")
    g.add_argument("--batch-size",   type=int,   default=256,   help="Mini-batch size.")
    g.add_argument("--epochs",       type=int,   default=50,    help="Maximum training epochs.")
    g.add_argument("--val-split",    type=float, default=0.15,  help="Validation fraction.")
    g.add_argument(
        "--patience", type=int, default=10,
        help="Early-stopping patience on validation loss.",
    )
    g.add_argument(
        "--threshold", type=float, default=0.9,
        help="Decision threshold for is_safe() and precision/recall metrics.",
    )
    g.add_argument(
        "--device", default="auto",
        help="'auto' (GPU if available), 'cpu', 'cuda', or 'cuda:N'.",
    )
    g.add_argument("--seed", type=int, default=42, help="Random seed.")

    g = p.add_argument_group("Output")
    g.add_argument(
        "--save-dir", default="checkpoints/safety",
        help="Directory for the best-model checkpoint.",
    )
    g.add_argument(
        "--checkpoint-name", default="safety_predictor.safetensors",
        help="Filename for the saved checkpoint.",
    )


# ── Data loading helpers ──────────────────────────────────────────────────────


def _build_hf_allow_patterns(strategy: str = None, start_episode: int = None, end_episode: int = None):
    """
    Build huggingface_hub `allow_patterns` so snapshot_download only pulls the
    files actually needed — full strategy folder, or a specific episode range.

    File naming on the hub (see SafetyDataGenerator._save):
      random/episode_{id}.npz
      trained/episode_{id}.npz
      attack/line_{lid}_ep_{eid}.npz
    """
    if start_episode is not None or end_episode is not None:
        if strategy is None:
            raise ValueError(
                "--start-episode/--end-episode require --strategy "
                "(episode-range filtering needs to know the file naming pattern)."
            )
        if start_episode is None or end_episode is None:
            raise ValueError("Both --start-episode and --end-episode must be set together.")

        if strategy in ("random", "trained"):
            return [f"{strategy}/episode_{i}.npz" for i in range(start_episode, end_episode)]
        else:  # attack — line id is unknown ahead of time, match any line for each ep
            return [f"attack/line_*_ep_{i}.npz" for i in range(start_episode, end_episode)]

    if strategy is not None:
        return [f"{strategy}/*.npz"]

    return None  # no filter — full dataset


def list_hf_npz_files(repo_id: str, token: str = None, strategy: str = None):
    """
    List .npz files in the HF dataset repo WITHOUT downloading them — cheap
    metadata-only API call, used to compute fraction/slice subsets before
    pulling any actual data.
    """
    from huggingface_hub import HfApi

    token = token or os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    files = [f for f in files if f.endswith(".npz")]
    if strategy:
        files = [f for f in files if f.startswith(f"{strategy}/")]
    return files


def _hf_candidates(repo_id, token, strategy, start_episode, end_episode):
    """List + episode-range-filter HF files; shared by fraction and slice modes."""
    candidates = list_hf_npz_files(repo_id, token=token, strategy=strategy)
    if start_episode is not None and end_episode is not None:
        episode_ids = set(range(start_episode, end_episode))
        candidates = [
            f for f in candidates
            if any(f"_{i}.npz" in f for i in episode_ids)
        ]
    if not candidates:
        raise FileNotFoundError(
            f"No .npz files found for strategy={strategy!r} in {repo_id} "
            f"(episode range {start_episode}-{end_episode})."
        )
    return sorted(candidates)  # deterministic order — required for slicing to be stable


def _slice_files(files: list, num_slices: int, slice_index: int) -> list:
    """
    Split `files` (already sorted) into `num_slices` contiguous, non-overlapping,
    near-equal chunks and return chunk `slice_index` (0-based).

    Deterministic — same inputs always produce the same chunk, so repeated
    calls across sessions (e.g. one Kaggle 12h session per slice) never
    overlap and together cover every file exactly once.
    """
    if num_slices < 1:
        raise ValueError(f"--num-slices must be >= 1, got {num_slices}")
    if not (0 <= slice_index < num_slices):
        raise ValueError(f"--slice-index must be in [0, {num_slices}), got {slice_index}")

    n = len(files)
    base, remainder = divmod(n, num_slices)
    # First `remainder` slices get one extra file so all slices differ by at most 1 file.
    start = slice_index * base + min(slice_index, remainder)
    size  = base + (1 if slice_index < remainder else 0)
    return files[start:start + size]


def download_hf_dataset(
    repo_id: str,
    token: str = None,
    strategy: str = None,
    start_episode: int = None,
    end_episode: int = None,
    fraction: float = None,
    num_slices: int = None,
    slice_index: int = None,
    seed: int = 42,
) -> str:
    """
    Download (or reuse cached) HuggingFace dataset repo; returns local dir.

    When `strategy` (and optionally an episode range) is given, only the
    matching files are downloaded via `allow_patterns` — much lighter on
    network/disk/GPU-idle-time than pulling the entire dataset.

    Two mutually-exclusive ways to pull less than everything:

    - `num_slices`/`slice_index`: deterministic, non-overlapping partition.
      E.g. num_slices=5, slice_index=0 pulls exactly the first 1/5 of the
      matching files (sorted, stable order); slice_index=1 the next 1/5, etc.
      Use this to train across multiple sessions (one slice per Kaggle 12h
      run) with zero duplicate or skipped files once you've gone through all
      slices. PREFERRED for iterative training.

    - `fraction`: random subsample (0 < fraction <= 1), seeded by `seed`.
      Different calls with the same seed return the same files, but there's
      no guarantee of coverage/no-overlap across different fraction calls —
      use slicing instead if you plan to train on multiple non-overlapping
      chunks over time.
    """
    from huggingface_hub import snapshot_download

    token = token or os.environ.get("HF_TOKEN")

    if num_slices is not None or slice_index is not None:
        if num_slices is None or slice_index is None:
            raise ValueError("--num-slices and --slice-index must be set together.")
        candidates = _hf_candidates(repo_id, token, strategy, start_episode, end_episode)
        allow_patterns = _slice_files(candidates, num_slices, slice_index)
        print(
            f"  slice {slice_index + 1}/{num_slices} of {len(candidates)} candidate files "
            f"-> pulling {len(allow_patterns)} file(s)"
        )

    elif fraction is not None:
        if not (0 < fraction <= 1):
            raise ValueError(f"--fraction must be in (0, 1], got {fraction}")
        import random as _random

        candidates = _hf_candidates(repo_id, token, strategy, start_episode, end_episode)
        n_select = max(1, round(len(candidates) * fraction))
        _random.seed(seed)
        allow_patterns = sorted(_random.sample(candidates, n_select))
        print(
            f"  fraction={fraction:.0%} of {len(candidates)} candidate files "
            f"-> pulling {len(allow_patterns)} file(s) (seed={seed})"
        )

    else:
        allow_patterns = _build_hf_allow_patterns(strategy, start_episode, end_episode)
        if allow_patterns:
            print(f"  pulling {len(allow_patterns)} pattern(s): {allow_patterns[:3]}{' ...' if len(allow_patterns) > 3 else ''}")

    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        allow_patterns=allow_patterns,
    )
    return local_dir


def load_npz_dataset(data_dir: str, strategy: str = None, max_files: int = None):
    """
    Load (obs_vectors, labels) from a directory of episode_*.npz / line_*.npz
    files laid out as data_dir/{random,trained,attack}/*.npz. Pure numpy —
    no grid2op environment needed just to read saved arrays.
    """
    strategies = ["random", "trained", "attack"] if strategy is None else [strategy]
    all_obs, all_labels = [], []

    for strat in strategies:
        folder = os.path.join(data_dir, strat)
        if not os.path.isdir(folder):
            continue
        files = sorted(glob.glob(os.path.join(folder, "*.npz")))
        if max_files:
            files = files[:max_files]
        for fp in files:
            d = np.load(fp)
            if len(d["labels"]) == 0:
                continue
            all_obs.append(d["obs_vectors"])
            all_labels.append(d["labels"])

    if not all_obs:
        raise FileNotFoundError(f"No data found in {data_dir}")

    obs_vectors = np.concatenate(all_obs, axis=0).astype(np.float32)
    labels      = np.concatenate(all_labels, axis=0).astype(np.float32)
    return obs_vectors, labels


# ── Sub-command runners ───────────────────────────────────────────────────────


def run_safety_trainer(args: argparse.Namespace) -> None:
    from gridstar.training.safety_trainer import SafetyPredictorTrainer

    # ── Load data ────────────────────────────────────────────────────────────
    if args.hf_dataset:
        print(f"Downloading HuggingFace dataset: {args.hf_dataset}")
        data_dir = download_hf_dataset(
            args.hf_dataset,
            token=args.hf_token,
            strategy=args.strategy,
            start_episode=args.start_episode,
            end_episode=args.end_episode,
            fraction=args.fraction,
            num_slices=args.num_slices,
            slice_index=args.slice_index,
            seed=args.seed,
        )
        print(f"  cached at {data_dir}")
    else:
        data_dir = args.data_dir

    print(f"Loading data from: {data_dir}")
    obs_vectors, labels = load_npz_dataset(data_dir, strategy=args.strategy, max_files=args.max_files)
    print(
        f"Dataset: {obs_vectors.shape[0]} samples  "
        f"obs_dim={obs_vectors.shape[1]}  pos_rate={labels.mean():.2%}"
    )

    obs_dim = obs_vectors.shape[1]

    # ── Net-specific kwargs ───────────────────────────────────────────────────
    net_kwargs = {}
    if args.net == "deep":
        net_kwargs["n_blocks"] = args.n_blocks
    if args.net == "ensemble":
        net_kwargs["n_members"] = args.n_members

    # ── Build trainer ─────────────────────────────────────────────────────────
    trainer = SafetyPredictorTrainer(
        obs_dim=obs_dim,
        net=args.net,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_split=args.val_split,
        patience=args.patience,
        threshold=args.threshold,
        save_dir=args.save_dir,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
        seed=args.seed,
        **net_kwargs,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer.fit(obs_vectors, labels)

    # ── Final evaluation on full dataset ──────────────────────────────────────
    print("── Final evaluation (full dataset) ──")
    metrics = trainer.evaluate(obs_vectors, labels)
    print(f"  loss      = {metrics['loss']:.4f}")
    print(f"  accuracy  = {metrics['acc']:.4f}")
    print(f"  precision = {metrics['precision']:.4f}")
    print(f"  recall    = {metrics['recall']:.4f}")
    print(f"  f1        = {metrics['f1']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
    print(f"\nCheckpoint: {trainer.checkpoint_path}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gridstar",
        description="GridStar — neural-guided A* for power grid topology.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subs = parser.add_subparsers(dest="command", metavar="command")

    sp = subs.add_parser(
        "safety_trainer",
        help="Train SafetyPredictor on collected grid2op data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_safety_trainer_args(sp)

    args = parser.parse_args()

    if args.command == "safety_trainer":
        run_safety_trainer(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
