"""scripts/download_data.py — End-to-end data pipeline CLI script.

Runs: stream → filter → build sessions → split → build vocab → preprocess → save.
Outputs processed splits and vocabulary to data/processed/.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --config config/config.yaml
    python scripts/download_data.py --max-records 50000
"""

from __future__ import annotations

import argparse
import logging
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Config
from data.interfaces import EncodedSession, Session
from data.loader import AmazonDataLoader
from data.preprocessor import SessionPreprocessor
from data.session_builder import SessionBuilder
from data.vocab import Vocabulary

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    """Set global random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info("Seeds set to %d", seed)


def _save_pickle(obj: object, path: Path) -> None:
    """Save an object to a pickle file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Saved %s", path)


def _print_summary(
    train: list,
    val: list,
    test: list,
    vocab: Vocabulary,
    encoded_train: list,
    encoded_val: list,
    encoded_test: list,
) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 60)
    print("  DATA PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  {'Split':<12} {'Sessions':>12} {'Encoded':>12}")
    print(f"  {'-'*36}")
    print(f"  {'Train':<12} {len(train):>12,} {len(encoded_train):>12,}")
    print(f"  {'Val':<12} {len(val):>12,} {len(encoded_val):>12,}")
    print(f"  {'Test':<12} {len(test):>12,} {len(encoded_test):>12,}")
    print(f"  {'-'*36}")
    total_s = len(train) + len(val) + len(test)
    total_e = len(encoded_train) + len(encoded_val) + len(encoded_test)
    print(f"  {'Total':<12} {total_s:>12,} {total_e:>12,}")
    print(f"\n  Vocabulary size : {len(vocab):,}")
    print("=" * 60 + "\n")


def main(cfg: Config) -> None:
    """Run the full data pipeline.

    Args:
        cfg: Loaded and validated Config object.
    """
    _set_seeds(cfg.project.seed)

    processed_dir = Path(cfg.data.processed_dir)

    # ── 1. Load ────────────────────────────────────────────────────────────────
    loader = AmazonDataLoader()
    df = loader.stream_reviews(
        category=cfg.data.hf_category_reviews,
        max_records=cfg.data.max_streaming_records,
        cfg=cfg,
    )

    # ── 2. Filter interactions ─────────────────────────────────────────────────
    df = loader.filter_interactions(df, min_item_freq=cfg.data.min_item_freq)

    # ── 2b. Fetch real product metadata from raw_meta_Electronics ─────────────
    # The reviews dataset is keyed by `asin` (child variant: e.g. shirt-size-M).
    # The metadata file is keyed by `parent_asin` (the umbrella product).
    # We join via parent_asin and fan results back out to every child asin
    # in our vocabulary.
    asin_to_parent = (
        df.drop_duplicates("item_id")
          .set_index("item_id")["parent_asin"]
          .astype(str)
          .to_dict()
    )
    parent_asins = set(asin_to_parent.values())
    logger.info(
        "Metadata join: %d unique asins → %d unique parent_asins",
        len(asin_to_parent),
        len(parent_asins),
    )

    parent_meta = loader.stream_item_metadata(
        category=cfg.data.hf_category_meta,
        target_asins=parent_asins,
    )
    parent_meta_indexed = parent_meta.set_index("item_id")

    # Build per-asin metadata: each child asin gets its parent's row.
    records = []
    matched = 0
    for asin, parent in asin_to_parent.items():
        if parent in parent_meta_indexed.index:
            row = parent_meta_indexed.loc[parent]
            records.append({
                "item_id":     asin,
                "title":       str(row["title"]),
                "description": str(row["description"]),
                "category":    str(row["category"]),
                "price":       row["price"],
            })
            matched += 1
        else:
            records.append({
                "item_id":     asin,
                "title":       asin,
                "description": "",
                "category":    "",
                "price":       None,
            })

    item_metadata = pd.DataFrame(records)
    logger.info(
        "Item metadata: %d/%d asins resolved to real titles (%d fell back to ASIN)",
        matched,
        len(asin_to_parent),
        len(asin_to_parent) - matched,
    )
    _save_pickle(item_metadata, processed_dir / "item_metadata.pkl")

    # ── 3. Build sessions ──────────────────────────────────────────────────────
    sb = SessionBuilder()
    sessions = sb.build_sessions(df, window_hours=cfg.data.session_window_hours)
    sessions = sb.filter_sessions(
        sessions,
        min_len=cfg.data.min_session_len,
        max_len=cfg.data.max_session_len,
    )

    # ── 4. Split ───────────────────────────────────────────────────────────────
    train_sessions, val_sessions, test_sessions = sb.split_sessions(
        sessions,
        train=cfg.data.train_split,
        val=cfg.data.val_split,
        test=cfg.data.test_split,
        seed=cfg.project.seed,
    )

    # ── 5. Build vocabulary (training items only) ──────────────────────────────
    vocab = Vocabulary()
    all_train_items = [iid for s in train_sessions for iid in s.item_ids]
    vocab.build(all_train_items)
    vocab.save(processed_dir / "vocab.json")

    # ── 6. Encode sessions ─────────────────────────────────────────────────────
    prep = SessionPreprocessor()
    encoded_train = prep.prepare(train_sessions, vocab, max_len=cfg.model.max_seq_len)
    encoded_val = prep.prepare(val_sessions, vocab, max_len=cfg.model.max_seq_len)
    encoded_test = prep.prepare(test_sessions, vocab, max_len=cfg.model.max_seq_len)

    # ── 7. Persist ─────────────────────────────────────────────────────────────
    _save_pickle(train_sessions, processed_dir / "train_sessions.pkl")
    _save_pickle(val_sessions, processed_dir / "val_sessions.pkl")
    _save_pickle(test_sessions, processed_dir / "test_sessions.pkl")
    _save_pickle(encoded_train, processed_dir / "encoded_train.pkl")
    _save_pickle(encoded_val, processed_dir / "encoded_val.pkl")
    _save_pickle(encoded_test, processed_dir / "encoded_test.pkl")

    # ── 8. Summary ─────────────────────────────────────────────────────────────
    _print_summary(
        train_sessions, val_sessions, test_sessions,
        vocab,
        encoded_train, encoded_val, encoded_test,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ASBRS data pipeline — stream, preprocess, and save."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Override cfg.data.max_streaming_records",
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.max_records is not None:
        cfg.data.max_streaming_records = args.max_records
    cfg.validate()

    main(cfg)
