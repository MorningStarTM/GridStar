"""
SafetyPredictorTrainer — supervised training for the binary safety classifier.

Training signal: (obs_vector, label) pairs from SafetyDataGenerator.
Loss:            BCEWithLogitsLoss with pos_weight for class imbalance.
Optimiser:       Adam + ReduceLROnPlateau scheduler.
Early stopping:  patience on validation loss.
Checkpointing:   best model (lowest val_loss) saved automatically.

Quick start
───────────
    from gridstar.training.safety_trainer import SafetyPredictorTrainer
    from gridstar.data_utils.generator import SafetyDataGenerator

    gen = SafetyDataGenerator(env_name="l2rpn_case14_sandbox")
    obs_vecs, labels = gen.load()

    trainer = SafetyPredictorTrainer(obs_dim=obs_vecs.shape[1], net="efficient")
    history = trainer.fit(obs_vecs, labels)

    metrics = trainer.evaluate(obs_vecs, labels)
    print(metrics)    # loss, acc, precision, recall, f1, tp/fp/fn/tn

Or via CLI:
    python main.py safety_trainer --data-dir data/safety --net efficient --epochs 50
"""

import os
import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from gridstar.networks.safety import SafetyPredictor, AVAILABLE_NETS


class SafetyPredictorTrainer:
    """
    Trains SafetyPredictor on (obs_vector, label) pairs.

    ── Loss ────────────────────────────────────────────────────────────────────

    BCEWithLogitsLoss with automatic pos_weight:

        pos_weight = n_negative / n_positive

    This compensates for the natural imbalance — the random-policy strategy
    produces many label=1 samples, while line attacks produce many label=0.
    Passing all three data sources together and letting pos_weight balance them
    is the recommended workflow.

    ── Training loop ────────────────────────────────────────────────────────────

    1. Split data into train / val (val_split fraction).
    2. Adam with ReduceLROnPlateau (factor=0.5, patience=5).
    3. After each epoch, evaluate on val set; if val_loss improved, save
       checkpoint. If val_loss has not improved for `patience` epochs, stop.
    4. On exit, the best checkpoint is reloaded automatically.

    ── Metrics ─────────────────────────────────────────────────────────────────

    At the decision threshold (default 0.9):
        - accuracy    — (TP + TN) / N
        - precision   — TP / (TP + FP)     ← how often predicted-safe is really safe
        - recall      — TP / (TP + FN)     ← how often a safe state is detected
        - f1          — harmonic mean of precision and recall

    A high threshold (0.9) biases toward precision — few false positives
    (wrongly declaring a state as goal) at the cost of lower recall.

    ── Args ────────────────────────────────────────────────────────────────────

    obs_dim:          flat observation vector length (from env.obs_dim).
    net:              backbone architecture — one of 'vanilla', 'efficient',
                      'deep', 'wide', 'ensemble'. Default 'efficient'.
    hidden_dim:       hidden layer width forwarded to the backbone (default 128).
    dropout:          dropout rate forwarded to the backbone (default 0).
    lr:               Adam learning rate (default 1e-3).
    weight_decay:     Adam L2 regularisation (default 1e-4).
    batch_size:       mini-batch size (default 256).
    epochs:           maximum training epochs (default 50).
    val_split:        fraction of data held out for validation (default 0.15).
    patience:         early-stopping patience on val_loss (default 10).
    threshold:        decision threshold for metrics and is_safe() (default 0.9).
    save_dir:         directory for checkpoint files (default checkpoints/safety).
    checkpoint_name:  filename for the best-model checkpoint (default safety_predictor.pt).
    device:           'auto', 'cpu', 'cuda', or 'cuda:N' (default 'auto').
    seed:             random seed for reproducible train/val splits (default 42).
    **net_kwargs:     forwarded to the backbone — e.g. n_blocks=4 for 'deep',
                      n_members=7 for 'ensemble'.
    """

    def __init__(
        self,
        obs_dim: int,
        net: str = "efficient",
        hidden_dim: int = 128,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 50,
        val_split: float = 0.15,
        patience: int = 10,
        threshold: float = 0.9,
        save_dir: str = "checkpoints/safety",
        checkpoint_name: str = "safety_predictor.pt",
        device: str = "auto",
        seed: int = 42,
        **net_kwargs,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._model = SafetyPredictor(
            obs_dim, net=net, hidden_dim=hidden_dim, dropout=dropout, **net_kwargs
        ).to(self.device)

        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_split = val_split
        self.patience = patience
        self.threshold = threshold
        self.save_dir = save_dir
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = os.path.join(save_dir, checkpoint_name)

        os.makedirs(save_dir, exist_ok=True)

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def model(self) -> SafetyPredictor:
        """The underlying SafetyPredictor module."""
        return self._model

    def fit(self, obs_vectors: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Train on (obs_vectors, labels).

        Args:
            obs_vectors: float32 array of shape (N, obs_dim).
            labels:      float32 array of shape (N,) — 0.0 (unsafe) or 1.0 (safe).

        Returns:
            History dict with keys 'train_loss', 'val_loss', 'val_acc', 'val_f1'
            (one value per epoch).
        """
        x = torch.from_numpy(obs_vectors).float()
        y = torch.from_numpy(labels).float().unsqueeze(1)
        dataset = TensorDataset(x, y)

        n_val   = max(1, int(len(dataset) * self.val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size * 2, shuffle=False)

        pos_rate  = float(labels.mean())
        pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-6)]).to(self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        history: Dict = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"\n{self._model}")
        print(
            f"Train: {n_train}  Val: {n_val}  "
            f"pos_rate={pos_rate:.2%}  pos_weight={pos_weight.item():.3f}  "
            f"device={self.device}\n"
        )
        _sep = "-" * 90
        print(_sep)
        print(
            f"{'Epoch':>6}  {'train_loss':>10}  {'val_loss':>9}  "
            f"{'val_acc':>8}  {'val_f1':>7}  {'lr':>9}  {'time':>6}"
        )
        print(_sep)

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            # ── Training pass ────────────────────────────────────────────────
            self._model.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self._model(xb), yb)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(xb)
            train_loss = running_loss / n_train

            # ── Validation ───────────────────────────────────────────────────
            vm = self._eval_loader(val_loader, criterion)
            scheduler.step(vm["loss"])

            history["train_loss"].append(train_loss)
            history["val_loss"].append(vm["loss"])
            history["val_acc"].append(vm["acc"])
            history["val_f1"].append(vm["f1"])

            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]["lr"]
            print(
                f"{epoch:6d}  {train_loss:10.4f}  {vm['loss']:9.4f}  "
                f"{vm['acc']:8.4f}  {vm['f1']:7.4f}  {lr_now:9.2e}  {elapsed:5.1f}s",
                end="",
            )

            if vm["loss"] < best_val_loss:
                best_val_loss = vm["loss"]
                patience_counter = 0
                self.save(self.checkpoint_path)
                print("  [saved]")
            else:
                patience_counter += 1
                print()
                if patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch} (patience={self.patience}).")
                    break

        print(_sep)
        self.load(self.checkpoint_path)
        print(f"Best val_loss={best_val_loss:.4f}  checkpoint: {self.checkpoint_path}\n")
        return history

    def evaluate(self, obs_vectors: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Compute loss, accuracy, precision, recall, F1 on a held-out dataset.

        Args:
            obs_vectors: float32 array of shape (N, obs_dim).
            labels:      float32 array of shape (N,).

        Returns:
            Dict with keys: loss, acc, precision, recall, f1, tp, fp, fn, tn.
        """
        x = torch.from_numpy(obs_vectors).float()
        y = torch.from_numpy(labels).float().unsqueeze(1)
        loader = DataLoader(TensorDataset(x, y), batch_size=self.batch_size * 2, shuffle=False)

        pos_rate   = float(labels.mean())
        pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-6)]).to(self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        return self._eval_loader(loader, criterion)

    def save(self, path: Optional[str] = None) -> None:
        """
        Save model weights + metadata to disk.

        The checkpoint is loadable with load() and compatible with
        SafetyPredictor state_dict format.
        """
        path = path or self.checkpoint_path
        torch.save(
            {
                "model_state": self._model.state_dict(),
                "obs_dim":     self._model.obs_dim,
                "net_name":    self._model.net_name,
                "threshold":   self.threshold,
            },
            path,
        )

    def load(self, path: Optional[str] = None) -> None:
        """
        Load model weights from a checkpoint produced by save().

        The trainer's threshold is also restored from the checkpoint so that
        evaluate() and the model's is_safe() use the same value that was set
        at training time.
        """
        path = path or self.checkpoint_path
        ckpt = torch.load(path, map_location=self.device)
        self._model.load_state_dict(ckpt["model_state"])
        self.threshold = ckpt.get("threshold", self.threshold)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _eval_loader(self, loader: DataLoader, criterion: nn.Module) -> Dict:
        """Run one evaluation pass; return metrics dict."""
        self._model.eval()
        total_loss = 0.0
        total = 0
        all_probs, all_labels = [], []

        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self._model(xb)
                total_loss += criterion(logits, yb).item() * len(xb)
                total      += len(xb)
                all_probs.append(torch.sigmoid(logits).cpu())
                all_labels.append(yb.cpu())

        probs  = torch.cat(all_probs).numpy().ravel()
        labels = torch.cat(all_labels).numpy().ravel()
        preds  = (probs >= self.threshold).astype(float)

        tp = float(((preds == 1) & (labels == 1)).sum())
        fp = float(((preds == 1) & (labels == 0)).sum())
        fn = float(((preds == 0) & (labels == 1)).sum())
        tn = float(((preds == 0) & (labels == 0)).sum())

        acc       = (tp + tn) / max(len(labels), 1)
        precision = tp / max(tp + fp, 1e-8)
        recall    = tp / max(tp + fn, 1e-8)
        f1        = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "loss":      total_loss / max(total, 1),
            "acc":       acc,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }
