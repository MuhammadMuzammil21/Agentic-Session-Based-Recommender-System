"""scripts/train.py — CLI training script for the ASBRS session encoder.

Loads processed data from data/processed/, trains SessionEncoder with
NextItemTrainer, applies early stopping on val Recall@10, and saves the
best checkpoint.

Usage:
    python scripts/train.py
    python scripts/train.py --config config/config.yaml --device cuda
    python scripts/train.py --resume checkpoints/epoch_003_recall0.1234.pt
"""

from __future__ import annotations

import argparse
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Config
from data.interfaces import EncodedSession
from data.preprocessor import SessionPreprocessor
from data.vocab import Vocabulary
from models.encoder import NextItemTrainer, SessionEncoder

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
# Silence the per-step / per-epoch INFO logs from the model so we get a clean
# table of one summary line per epoch (printed manually in the loop below).
for noisy in ("models.encoder", "data.preprocessor", "data.vocab"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _set_seeds(seed: int) -> None:
    """Set all global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seeds set to %d", seed)


def _load_pickle(path: Path) -> object:
    """Load a pickle file, raise FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data file not found: {path}\n"
            "Run `python scripts/download_data.py` first."
        )
    with path.open("rb") as fh:
        return pickle.load(fh)


def _print_epoch_summary(
    epoch: int,
    num_epochs: int,
    train_loss: float,
    val_metrics: dict,
    is_best: bool,
) -> None:
    """Print one fixed-width row per epoch.

    Layout:
       Epoch  |   loss  | R@5    R@10   R@20   MRR@10  best
       001/030  9.3850   0.0261  0.0366  0.0545  0.0152   *
    """
    r5  = val_metrics.get("Recall@5",  float("nan"))
    r10 = val_metrics.get("Recall@10", float("nan"))
    r20 = val_metrics.get("Recall@20", float("nan"))
    mrr = val_metrics.get("MRR@10",    float("nan"))
    marker = "*" if is_best else " "
    print(
        f"  {epoch+1:03d}/{num_epochs:03d}  "
        f"{train_loss:7.4f}  "
        f"{r5:6.4f}  {r10:6.4f}  {r20:6.4f}  "
        f"{mrr:7.4f}   {marker}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main(cfg: Config, device: torch.device, resume: str | None) -> None:
    """Run the full training loop.

    Args:
        cfg:    Validated project Config.
        device: Torch device (cpu / cuda).
        resume: Optional path to a checkpoint to resume from.
    """
    _set_seeds(cfg.project.seed)
    processed_dir = Path(cfg.data.processed_dir)

    # ── Load processed data ────────────────────────────────────────────────
    print("\nLoading processed data …")
    vocab: Vocabulary = Vocabulary.load(processed_dir / "vocab.json")
    encoded_train: List[EncodedSession] = _load_pickle(
        processed_dir / "encoded_train.pkl"
    )
    encoded_val: List[EncodedSession] = _load_pickle(
        processed_dir / "encoded_val.pkl"
    )
    print(
        f"  train={len(encoded_train):,}  val={len(encoded_val):,}"
        f"  vocab={len(vocab):,}"
    )

    # ── DataLoaders ────────────────────────────────────────────────────────
    prep = SessionPreprocessor()
    train_loader = prep.to_dataloader(
        encoded_train,
        batch_size=cfg.training.batch_size,
        shuffle=True,
    )
    val_loader = prep.to_dataloader(
        encoded_val,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    encoder = SessionEncoder(
        vocab_size=len(vocab),
        embed_dim=cfg.model.embedding_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_attention_heads,
        dropout=cfg.model.dropout,
    ).to(device)

    trainer = NextItemTrainer(encoder, vocab_size=len(vocab), cfg=cfg)

    if resume:
        trainer.load_checkpoint(resume)
        print(f"  Resumed from: {resume}")

    optimizer = torch.optim.Adam(
        encoder.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    # ── Training loop ──────────────────────────────────────────────────────
    best_recall = -1.0
    patience_counter = 0
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    k_values: List[int] = cfg.evaluation.k_values

    print(f"\nTraining for up to {cfg.training.num_epochs} epochs "
          f"(patience={cfg.training.patience}) on {device}")
    print("  (a * in the last column marks a new best validation Recall@10)\n")
    print(f"  {'Epoch':>7}  {'loss':>7}  {'R@5':>6}  {'R@10':>6}  "
          f"{'R@20':>6}  {'MRR@10':>7}  best")
    print("  " + "-" * 58)

    for epoch in range(cfg.training.num_epochs):
        train_loss = trainer.train_epoch(train_loader, optimizer, device)
        val_metrics = trainer.evaluate(val_loader, device, k_values=k_values)
        val_recall10 = val_metrics.get("Recall@10", 0.0)
        scheduler.step(val_recall10)

        is_best = val_recall10 > best_recall
        if is_best:
            best_recall = val_recall10
            patience_counter = 0
            ckpt_name = f"epoch_{epoch+1:03d}_recall{val_recall10:.4f}.pt"
            trainer.save_checkpoint(
                epoch=epoch,
                val_recall=val_recall10,
                path=str(ckpt_dir / ckpt_name),
            )
        else:
            patience_counter += 1

        _print_epoch_summary(epoch, cfg.training.num_epochs, train_loss, val_metrics, is_best)

        if patience_counter >= cfg.training.patience:
            print(f"\n  Early stopping at epoch {epoch+1} "
                  f"(no improvement for {cfg.training.patience} epochs).")
            break

    print(f"\n  Best Recall@10 = {best_recall:.4f}")
    print("  Training complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASBRS session encoder")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="Path to config YAML",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Compute device",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    cfg.validate()
    device = torch.device(args.device)

    main(cfg, device, resume=args.resume)
