"""scripts/evaluate.py — Evaluate the trained GRU + Attention model.

Runs the trained SessionEncoder on the test set, computes Recall@K, MRR@K
and HitRate@K for the configured K values, prints a single-row results
table, saves it as CSV / Markdown, and writes a human-evaluation HTML sheet
populated with real top-5 predictions from the model.

Usage
-----
    python scripts/evaluate.py [--checkpoint checkpoints/best.pt]
                               [--config config/config.yaml]
                               [--n-human 10]
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

from agent.interfaces import IntentResult, RecommendationOutput
from config.settings import Config
from data.vocab import PAD_IDX, UNK_IDX, Vocabulary
from evaluation.human_eval import HumanEvalExporter
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


def _load_item_metadata(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "item_metadata.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"item_metadata.pkl not found in {processed_dir}. "
            "Run scripts/download_data.py first."
        )
    with path.open("rb") as fh:
        df = pickle.load(fh)
    logger.info("Item metadata loaded: %d rows", len(df))
    return df


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
) -> tuple[list[int], list[float]]:
    """Encode one session and return top-K (item_ids, raw_scores)."""
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
        top_scores, top_ids = torch.topk(scores, k=top_k)
    return top_ids.tolist(), top_scores.tolist()


# ── Main ──────────────────────────────────────────────────────────────────────


def main(
    checkpoint: str | None = None,
    config_path: str = "config/config.yaml",
    n_human: int = 10,
    output_dir: str = "evaluation",
) -> None:
    """Evaluate the trained GRU + Attention model on the test set."""
    # 1. Load artefacts.
    cfg = _load_config(config_path)
    processed_dir = Path(cfg.data.processed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vocab = _load_vocab(processed_dir)
    test_sessions_raw = _load_test_sessions(processed_dir)
    test_sessions = _sessions_to_asin_lists(test_sessions_raw)
    item_metadata = _load_item_metadata(processed_dir)

    asin_to_title = dict(
        zip(
            item_metadata["item_id"].astype(str),
            item_metadata["title"].fillna("").astype(str),
        )
    )

    # 2. Resolve checkpoint.
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_path = Path(checkpoint) if checkpoint else _find_best_checkpoint(ckpt_dir)
    if ckpt_path and ckpt_path.exists():
        print(f"[evaluate] Loading checkpoint: {ckpt_path}")
    else:
        print("[evaluate] No checkpoint found — running with random weights.")
        ckpt_path = None

    # 3. Build encoder.
    encoder = _build_encoder(cfg, vocab_size=len(vocab), ckpt_path=ckpt_path)
    max_len = cfg.model.max_seq_len
    k_values = cfg.evaluation.k_values
    max_k = max(k_values)

    # 4. Score every test session and collect (top_ids, ground_truth).
    print("\n[evaluate] Scoring test sessions with GRU + Attention …")
    predictions: list[tuple[list[int], int]] = []
    real_recs: list[list[RecommendationOutput]] = []
    real_intents: list[IntentResult] = []
    display_sessions: list[list[str]] = []

    for session in test_sessions:
        seed_strs = session[:-1]
        target_asin = session[-1]
        seed_int = [vocab.encode(a) for a in seed_strs]
        gt_idx = vocab.encode(target_asin)

        top_ids, top_scores = _score_session(encoder, seed_int, max_len, max_k)
        predictions.append((top_ids, gt_idx))

        # For the human-eval HTML — keep only the first 5 with real titles.
        recs: list[RecommendationOutput] = []
        for rank_idx, (item_id, score) in enumerate(
            zip(top_ids[:5], top_scores[:5])
        ):
            asin = vocab.decode(item_id)
            title = asin_to_title.get(asin, asin) or asin
            recs.append(
                RecommendationOutput(
                    rank=rank_idx + 1,
                    item_id=item_id,
                    item_title=title,
                    final_score=float(score),
                    explanation=(
                        "Top-ranked by GRU+Attention score for your session."
                    ),
                )
            )
        real_recs.append(recs)

        session_titles = [asin_to_title.get(a, a) or a for a in seed_strs]
        display_sessions.append(session_titles)
        real_intents.append(
            IntentResult(
                intent_summary=" ".join(session_titles[:3]),
                top_items=session_titles[:3],
                confidence=1.0,
                keywords=[],
            )
        )

    # 5. Compute metrics.
    metrics = evaluate_model(predictions, k_values)

    # 6. Print and save the metrics row.
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

    # 7. Generate the human-eval HTML sheet.
    print("[evaluate] Generating human evaluation sheet …")
    html_path = output_path / "human_eval_sheet.html"
    exporter = HumanEvalExporter(output_path=html_path, seed=cfg.project.seed)
    exporter.generate_eval_sheet(
        sessions=display_sessions,
        recommendations=real_recs,
        intents=real_intents,
        n_sessions=n_human,
    )
    print(f"[evaluate] Human eval sheet written to {html_path}")

    # 8. Final summary line.
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
        "--n-human",
        type=int,
        default=10,
        dest="n_human",
        help="Number of sessions in the human eval sheet (default: 10).",
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
        n_human=args.n_human,
        output_dir=args.output_dir,
    )
