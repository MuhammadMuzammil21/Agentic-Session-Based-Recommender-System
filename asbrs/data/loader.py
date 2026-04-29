"""data/loader.py — HuggingFace streaming loader for Amazon Reviews 2023.

All raw data ingestion goes through AmazonDataLoader.stream_reviews().
The full dataset is NEVER written to disk; only a subset is kept in RAM.

HF dataset layout (as of 2025):
  Reviews: hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/review_categories/{Category}.jsonl
  Metadata: hf://datasets/McAuley-Lab/Amazon-Reviews-2023/{config_name}/full-*.parquet

The old loading-script approach (trust_remote_code=True) is no longer
supported in datasets ≥3.0.  We load the raw JSONL files directly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from datasets import load_dataset  # module-level so patch("data.loader.load_dataset") works
from tqdm import tqdm

if TYPE_CHECKING:
    from config.settings import Config

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

REVIEW_COLUMNS = [
    "user_id",
    "item_id",
    "rating",
    "timestamp",
    "title",
    "description",
    "price",
    "category",
]

MIN_FREQ_DEFAULT: int = 5

# The raw JSONL URL template.  {category_name} is e.g. "Electronics".
_HF_REVIEW_URL = (
    "hf://datasets/McAuley-Lab/Amazon-Reviews-2023"
    "/raw/review_categories/{category_name}.jsonl"
)

# Map HF config name → plain category name in the JSONL path.
# e.g. "raw_review_Electronics" → "Electronics"
def _config_to_category(config_name: str) -> str:
    """Strip the 'raw_review_' prefix to get the JSONL filename stem."""
    return config_name.removeprefix("raw_review_").removeprefix("raw_meta_")


class AmazonDataLoader:
    """Load Amazon Reviews 2023 via HuggingFace streaming.

    No data is downloaded to disk. Records are accumulated in memory up
    to `max_records` rows per call.

    Example::

        loader = AmazonDataLoader()
        df = loader.stream_reviews(
            category="raw_review_Electronics",
            max_records=10_000,
            cfg=cfg,
        )
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def stream_reviews(
        self,
        category: str,
        max_records: int,
        cfg: Config,  # noqa: ARG002  (kept for API compatibility)
    ) -> pd.DataFrame:
        """Stream review records from HuggingFace without a full download.

        Loads the raw JSONL file directly via the ``hf://`` protocol so that
        no dataset loading script is required (compatible with datasets ≥3.0).

        The ``timestamp`` field in the source file is in **milliseconds**;
        it is converted to seconds automatically.

        Args:
            category:    HF dataset config name, e.g. ``"raw_review_Electronics"``.
            max_records: Maximum number of rows to load into memory.
            cfg:         Project config (kept for API compatibility).

        Returns:
            DataFrame with columns: user_id, item_id, rating, timestamp,
            title, description, price, category.

        Raises:
            RuntimeError: If the HF streaming connection fails.
        """
        category_name = _config_to_category(category)
        url = _HF_REVIEW_URL.format(category_name=category_name)

        logger.info(
            "Streaming up to %d records from '%s'",
            max_records,
            url,
        )

        try:
            ds = load_dataset(
                "json",
                data_files=url,
                streaming=True,
                split="train",
            )
        except Exception as exc:
            logger.error("Failed to connect to HuggingFace: %s", exc)
            raise RuntimeError(f"HuggingFace streaming failed: {exc}") from exc

        records = []
        progress = tqdm(
            ds.take(max_records),
            total=max_records,
            desc="Streaming reviews",
            unit=" rows",
            mininterval=0.5,
        )
        for item in progress:
            # timestamp is stored as a string of milliseconds in this dataset
            try:
                ts_ms = int(item.get("timestamp", 0))
                ts_sec = ts_ms // 1000 if ts_ms > 2_000_000_000 else ts_ms
            except (TypeError, ValueError):
                ts_sec = 0

            records.append(
                {
                    "user_id":     item.get("user_id", ""),
                    "item_id":     item.get("asin", ""),
                    "rating":      float(item.get("rating", 0.0)),
                    "timestamp":   ts_sec,
                    "title":       item.get("title", ""),
                    "description": item.get("text", ""),
                    "price":       item.get("price", None),
                    "category":    item.get("main_category", category_name),
                }
            )

        df = pd.DataFrame(records, columns=REVIEW_COLUMNS)
        logger.info(
            "Streaming complete — shape=%s, dtypes:\n%s",
            df.shape,
            df.dtypes.to_string(),
        )
        return df

    def filter_interactions(
        self,
        df: pd.DataFrame,
        min_item_freq: int = MIN_FREQ_DEFAULT,
    ) -> pd.DataFrame:
        """Remove items that appear fewer than min_item_freq times.

        Args:
            df:            DataFrame returned by :meth:`stream_reviews`.
            min_item_freq: Minimum number of interactions an item must have.

        Returns:
            Filtered DataFrame with low-frequency items removed.

        Raises:
            ValueError: If the DataFrame is missing required columns.
        """
        if "item_id" not in df.columns:
            raise ValueError("DataFrame must contain an 'item_id' column")

        counts = df["item_id"].value_counts()
        frequent_items = counts[counts >= min_item_freq].index
        before = len(df)
        df_filtered = df[df["item_id"].isin(frequent_items)].reset_index(drop=True)
        removed = before - len(df_filtered)
        n_items_removed = len(counts) - len(frequent_items)
        logger.info(
            "filter_interactions: removed %d rows (%d unique items below freq=%d); "
            "%d rows remaining",
            removed,
            n_items_removed,
            min_item_freq,
            len(df_filtered),
        )
        return df_filtered
