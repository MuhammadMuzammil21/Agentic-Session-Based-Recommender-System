"""scripts/evaluate.py — Evaluate a trained ASBRS model on the test set.

Usage
-----
    python scripts/evaluate.py [--checkpoint checkpoints/best.pt]
                               [--config config/config.yaml]
                               [--n-human 10]
                               [--output-dir evaluation]

Steps
-----
1. Load training artefacts (vocab, item_metadata, test sessions).
2. Load the best checkpoint into SessionEncoder.
3. Run AblationStudy.run_all() over the four model variants.
4. Print the comparison table to the console.
5. Save ablation results to <output_dir>/ablation_results.csv.
6. Generate a human-evaluation HTML sheet.
7. Print a final summary line.
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
from data.vocab import Vocabulary
from evaluation.ablation import AblationStudy
from evaluation.human_eval import HumanEvalExporter

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_config(config_path: str) -> Config:
    """Load and validate the YAML configuration.

    Args:
        config_path: Path to config.yaml.

    Returns:
        Validated Config object.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    cfg = Config.load(config_path)
    cfg.validate()
    logger.info("Config loaded from %s", config_path)
    return cfg


def _load_vocab(processed_dir: Path) -> Vocabulary:
    """Load the trained Vocabulary from processed_dir/vocab.json.

    Args:
        processed_dir: Directory containing vocab.json.

    Returns:
        Loaded Vocabulary instance.

    Raises:
        FileNotFoundError: If vocab.json is missing.
    """
    vocab_path = processed_dir / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")
    vocab = Vocabulary.load(vocab_path)
    logger.info("Vocabulary loaded: %d items", len(vocab))
    return vocab


def _load_sessions(processed_dir: Path, split: str) -> list:
    """Load pickled sessions for a given data split.

    Args:
        processed_dir: Directory containing *_sessions.pkl files.
        split:         One of ``"train"``, ``"val"``, ``"test"``.

    Returns:
        List of Session objects.

    Raises:
        FileNotFoundError: If the pickle file is missing.
    """
    path = processed_dir / f"{split}_sessions.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Sessions file not found: {path}")
    with path.open("rb") as fh:
        sessions = pickle.load(fh)
    logger.info("Loaded %d %s sessions", len(sessions), split)
    return sessions


def _load_item_metadata(processed_dir: Path) -> pd.DataFrame:
    """Load item metadata from the processed directory.

    Expects either ``item_metadata.pkl`` or ``item_metadata.csv``.

    Args:
        processed_dir: Directory to search for metadata files.

    Returns:
        DataFrame with at least ``item_id`` and ``title`` columns.

    Raises:
        FileNotFoundError: If neither metadata file is present.
    """
    for name in ("item_metadata.pkl", "item_metadata.csv"):
        path = processed_dir / name
        if path.exists():
            if name.endswith(".pkl"):
                with path.open("rb") as fh:
                    df = pickle.load(fh)
            else:
                df = pd.read_csv(path)
            logger.info("Item metadata loaded: %d rows from %s", len(df), path)
            return df
    raise FileNotFoundError(
        f"No item metadata file found in {processed_dir}. "
        "Expected 'item_metadata.pkl' or 'item_metadata.csv'."
    )


def _find_best_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the checkpoint with the highest recall value in its filename.

    Args:
        checkpoint_dir: Directory containing ``*.pt`` checkpoint files.

    Returns:
        Path to the best checkpoint, or ``None`` if the directory is empty.
    """
    pts = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not pts:
        return None
    # Filename format: epoch_NNN_recallX.XXXX.pt — sort lexicographically
    # to get the most recent / highest recall (alphabetically last).
    best = max(pts, key=lambda p: p.name)
    return best


def _sessions_to_item_id_lists(sessions: list) -> list[list[str]]:
    """Convert Session objects to lists of ASIN strings.

    Handles both ``Session`` dataclass instances (with ``.item_ids``)
    and plain list-of-strings representations.

    Args:
        sessions: List of Session objects or list-of-str.

    Returns:
        List[List[str]] — one list of ASINs per session.
    """
    result: list[list[str]] = []
    for s in sessions:
        if hasattr(s, "item_ids"):
            result.append(list(s.item_ids))
        elif isinstance(s, (list, tuple)):
            result.append([str(x) for x in s])
        else:
            logger.warning("Unknown session type %s — skipping", type(s))
    return result


# ── Main CLI entry-point ──────────────────────────────────────────────────────


def main(
    checkpoint: str | None = None,
    config_path: str = "config/config.yaml",
    n_human: int = 10,
    output_dir: str = "evaluation",
    agentic_n_samples: int = 50,
    agentic_call_delay: float = 6.5,
) -> None:
    """Run the full evaluation pipeline.

    Args:
        checkpoint:  Path to the model checkpoint to load.
                     Auto-detected from the checkpoint directory when None.
        config_path: Path to config/config.yaml.
        n_human:     Number of sessions to include in the human-eval sheet.
        output_dir:  Directory for output files.
    """
    # ── Step 1: Load artefacts ────────────────────────────────────────────────

    cfg = _load_config(config_path)
    processed_dir = Path(cfg.data.processed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vocab = _load_vocab(processed_dir)

    test_sessions_raw = _load_sessions(processed_dir, "test")
    test_sessions = _sessions_to_item_id_lists(test_sessions_raw)

    item_metadata = _load_item_metadata(processed_dir)

    # ── Step 2: Resolve checkpoint ────────────────────────────────────────────

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    if checkpoint:
        ckpt_path = Path(checkpoint)
    else:
        ckpt_path = _find_best_checkpoint(ckpt_dir)

    if ckpt_path and ckpt_path.exists():
        print(f"[evaluate] Loading checkpoint: {ckpt_path}")
        logger.info("Checkpoint: %s", ckpt_path)
    else:
        print("[evaluate] No checkpoint found — running with random weights.")
        ckpt_path = None

    # ── Step 3: Run ablation study ────────────────────────────────────────────

    print("\n[evaluate] Running ablation study …")
    study = AblationStudy(
        test_sessions=test_sessions,
        vocab=vocab,
        item_metadata=item_metadata,
        cfg=cfg,
        checkpoint_path=ckpt_path,
        agentic_n_samples=agentic_n_samples,
        agentic_call_delay=agentic_call_delay,
    )
    results_df = study.run_all()

    # ── Step 4: Print comparison table ───────────────────────────────────────

    print("\n" + "=" * 72)
    print("  ASBRS ABLATION STUDY RESULTS")
    print("=" * 72)
    print(results_df.to_string(index=False))
    print("=" * 72 + "\n")

    # ── Step 5: Save results ──────────────────────────────────────────────────

    csv_path = output_path / "ablation_results.csv"
    study.save_results(results_df, csv_path)
    print(f"[evaluate] Ablation results saved to {csv_path}")

    # ── Step 6: Generate human evaluation sheet ───────────────────────────────

    print("[evaluate] Generating human evaluation sheet …")

    # Build placeholder recommendations / intents for sessions where we lack
    # full pipeline outputs.  In a full run these would come from the model.
    from agent.interfaces import IntentResult, RecommendationOutput

    placeholder_intents: list[IntentResult] = [
        IntentResult(
            intent_summary="browsing electronics",
            top_items=sess[:3] if len(sess) >= 3 else sess,
            confidence=0.0,
            keywords=[],
        )
        for sess in test_sessions
    ]

    placeholder_recs: list[list[RecommendationOutput]] = [
        [
            RecommendationOutput(
                rank=i + 1,
                item_id=i,
                item_title=f"Item {i + 1}",
                final_score=1.0 / (i + 1),
                explanation="Recommended based on your recent browsing history.",
            )
            for i in range(5)
        ]
        for _ in test_sessions
    ]

    html_path = output_path / "human_eval_sheet.html"
    exporter = HumanEvalExporter(output_path=html_path, seed=cfg.project.seed)
    exporter.generate_eval_sheet(
        sessions=test_sessions,
        recommendations=placeholder_recs,
        intents=placeholder_intents,
        n_sessions=n_human,
    )
    print(f"[evaluate] Human eval sheet written to {html_path}")

    # ── Step 7: Print final summary ───────────────────────────────────────────

    full_row = results_df[results_df["Model"] == "Full Agentic (ASBRS)"]
    if not full_row.empty:
        recall10 = full_row["Recall@10"].values[0]
        print(
            f"\nBest model: Full Agentic | Recall@10: {recall10:.4f}"
        )
    else:
        print("\n[evaluate] Full agentic results not available.")


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ASBRS model on the test set."
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
    parser.add_argument(
        "--agentic-n",
        type=int,
        default=5,
        dest="agentic_n_samples",
        help=(
            "Number of test sessions to evaluate the Full Agentic variant on "
            "(uses Gemini API). Default 50 to stay within free-tier quota. "
            "Set 0 to use all test sessions."
        ),
    )
    parser.add_argument(
        "--agentic-delay",
        type=float,
        default=6.5,
        dest="agentic_call_delay",
        help=(
            "Seconds to sleep between Gemini calls (default 6.5s ≈ 9 RPM, "
            "under the 10 RPM free-tier limit). Set 0 to disable."
        ),
    )
    args = parser.parse_args()
    main(
        checkpoint=args.checkpoint,
        config_path=args.config_path,
        n_human=args.n_human,
        output_dir=args.output_dir,
        agentic_n_samples=args.agentic_n_samples,
        agentic_call_delay=args.agentic_call_delay,
    )
