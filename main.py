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
"""

import argparse
import sys


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

    g = p.add_argument_group("Environment")
    g.add_argument(
        "--env-name", default="l2rpn_case14_sandbox",
        help="grid2op environment name (used to obtain obs_dim).",
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
        "--checkpoint-name", default="safety_predictor.pt",
        help="Filename for the saved checkpoint.",
    )


# ── Sub-command runners ───────────────────────────────────────────────────────


def run_safety_trainer(args: argparse.Namespace) -> None:
    from gridstar.data_utils.generator import SafetyDataGenerator
    from gridstar.training.safety_trainer import SafetyPredictorTrainer

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading data from: {args.data_dir}")
    gen = SafetyDataGenerator(env_name=args.env_name, save_dir=args.data_dir)
    obs_vectors, labels = gen.load(strategy=args.strategy, max_files=args.max_files)
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
