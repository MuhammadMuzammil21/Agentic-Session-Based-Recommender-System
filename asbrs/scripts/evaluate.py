"""scripts/evaluate.py — Evaluate the trained GRU + Attention model.

Runs the trained SessionEncoder on the test set, computes Recall@K,
MRR@K, and HitRate@K for the configured K values, prints a single-row
results table, and saves it as CSV / Markdown.

Usage
-----
    python scripts/evaluate.py [--checkpoint checkpoints/best.pt]
                               [--config config/config.yaml]
                               [--output-dir evaluation]
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

# Ensure the package root (asbrs/) is importable when called as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from config.settings import Config
from data.vocab import PAD_IDX, UNK_IDX, Vocabulary
from evaluation.metrics import evaluate_model
from models.encoder import SessionEncoder

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_config(config_path: str) -> Config:
    cfg = Config.load(config_path)
    cfg.validate()
    logger.info("Config loaded from %s", config_path)
    return cfg


def _load_vocab(processed_dir: Path) -> Vocabulary:
    vocab_path = processed_dir / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")
    vocab = Vocabulary.load(vocab_path)
    logger.info("Vocabulary loaded: %d items", len(vocab))
    return vocab


def _load_test_sessions(processed_dir: Path) -> list:
    path = processed_dir / "test_sessions.pkl"
    if not path.exists():
        raise FileNotFoundError(f"test_sessions.pkl not found in {processed_dir}")
    with path.open("rb") as fh:
        sessions = pickle.load(fh)
    logger.info("Loaded %d test sessions", len(sessions))
    return sessions


def _find_best_checkpoint(checkpoint_dir: Path) -> Path | None:
    pts = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not pts:
        return None
    # Filenames look like epoch_NNN_recallX.XXXX.pt — alphabetical max picks
    # the highest recall.
    return max(pts, key=lambda p: p.name)


def _sessions_to_asin_lists(sessions: list) -> list[list[str]]:
    result: list[list[str]] = []
    for s in sessions:
        if hasattr(s, "item_ids"):
            result.append(list(s.item_ids))
        elif isinstance(s, (list, tuple)):
            result.append([str(x) for x in s])
    return result


def _build_encoder(
    cfg: Config, vocab_size: int, ckpt_path: Path | None
) -> SessionEncoder:
    encoder = SessionEncoder(
        vocab_size=vocab_size,
        embed_dim=cfg.model.embedding_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_attention_heads,
        dropout=cfg.model.dropout,
        padding_idx=0,
    )
    if ckpt_path and ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        encoder.load_state_dict(payload["model_state_dict"])
        logger.info("Loaded weights from %s", ckpt_path)
    else:
        logger.warning("No checkpoint loaded — running with random weights.")
    encoder.eval()
    return encoder


def _score_session(
    encoder: SessionEncoder,
    seed_int: list[int],
    max_len: int,
    top_k: int,
) -> list[int]:
    """Encode one session and return top-K item ids."""
    if len(seed_int) >= max_len:
        padded = seed_int[-max_len:]
        true_len = max_len
    else:
        true_len = len(seed_int)
        padded = [PAD_IDX] * (max_len - true_len) + seed_int

    input_t = torch.tensor([padded], dtype=torch.long)
    lengths_t = torch.tensor([true_len], dtype=torch.long)

    with torch.no_grad():
        session_repr, _attn, _hiddens = encoder(input_t, lengths_t)
        scores = encoder.predict_scores(
            session_repr, encoder.item_embedding.embedding.weight
        )[0]
        scores[PAD_IDX] = float("-inf")
        scores[UNK_IDX] = float("-inf")
        top_ids = torch.topk(scores, k=top_k).indices
    return top_ids.tolist()


# ── Main ──────────────────────────────────────────────────────────────────────


def main(
    checkpoint: str | None = None,
    config_path: str = "config/config.yaml",
    output_dir: str = "evaluation",
) -> None:
    """Evaluate the trained GRU + Attention model on the test set."""
    cfg = _load_config(config_path)
    processed_dir = Path(cfg.data.processed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vocab = _load_vocab(processed_dir)
    test_sessions_raw = _load_test_sessions(processed_dir)
    test_sessions = _sessions_to_asin_lists(test_sessions_raw)

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_path = Path(checkpoint) if checkpoint else _find_best_checkpoint(ckpt_dir)
    if ckpt_path and ckpt_path.exists():
        print(f"[evaluate] Loading checkpoint: {ckpt_path}")
    else:
        print("[evaluate] No checkpoint found — running with random weights.")
        ckpt_path = None

    encoder = _build_encoder(cfg, vocab_size=len(vocab), ckpt_path=ckpt_path)
    max_len = cfg.model.max_seq_len
    k_values = cfg.evaluation.k_values
    max_k = max(k_values)

    print("\n[evaluate] Scoring test sessions with GRU + Attention …")
    predictions: list[tuple[list[int], int]] = []
    for session in test_sessions:
        seed_strs = session[:-1]
        target_asin = session[-1]
        seed_int = [vocab.encode(a) for a in seed_strs]
        gt_idx = vocab.encode(target_asin)

        top_ids = _score_session(encoder, seed_int, max_len, max_k)
        predictions.append((top_ids, gt_idx))

    metrics = evaluate_model(predictions, k_values)

    row = {"Model": "GRU + Attention"}
    for k in k_values:
        row[f"Recall@{k}"] = metrics.get(f"Recall@{k}", float("nan"))
        row[f"MRR@{k}"] = metrics.get(f"MRR@{k}", float("nan"))
        row[f"HitRate@{k}"] = metrics.get(f"HitRate@{k}", float("nan"))
    results_df = pd.DataFrame([row])

    print("\n" + "=" * 72)
    print("  GRU + ATTENTION  ·  TEST-SET METRICS")
    print("=" * 72)
    print(results_df.to_string(index=False))
    print("=" * 72 + "\n")

    csv_path = output_path / "results.csv"
    md_path = output_path / "results.md"
    results_df.to_csv(csv_path, index=False)
    md_path.write_text(
        "# GRU + Attention — Test-set Metrics\n\n"
        + results_df.to_markdown(index=False, floatfmt=".4f")
        + "\n",
        encoding="utf-8",
    )
    print(f"[evaluate] Results saved to {csv_path}")
    print(f"[evaluate] Markdown saved to {md_path}")
    print(
        f"\nDone. Recall@10 = {metrics.get('Recall@10', float('nan')):.4f}"
        f"  ·  MRR@10 = {metrics.get('MRR@10', float('nan')):.4f}"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the trained GRU + Attention model on the test set."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific .pt checkpoint. Auto-detects best if omitted.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        dest="config_path",
        help="Path to config.yaml (default: config/config.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation",
        dest="output_dir",
        help="Directory for output files (default: evaluation).",
    )
    args = parser.parse_args()
    main(
        checkpoint=args.checkpoint,
        config_path=args.config_path,
        output_dir=args.output_dir,
    )
