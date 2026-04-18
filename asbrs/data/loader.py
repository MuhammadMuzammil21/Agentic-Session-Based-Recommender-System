"""data/loader.py — HuggingFace streaming loader for Amazon Reviews 2023.

All raw data ingestion goes through AmazonDataLoader.stream_reviews().
The full dataset is NEVER written to disk; only a subset is kept in RAM.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
from datasets import load_dataset  # module-level so patch("data.loader.load_dataset") works

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


class AmazonDataLoader:
    """Load Amazon Reviews 2023 via HuggingFace streaming.

    No data is downloaded to disk. Records are accumulated in memory up
    to `max_records` rows per call.

    Example:
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
        cfg: Config,  # noqa: ARG002  (used for dataset name)
    ) -> pd.DataFrame:
        """Stream review records from HuggingFace without a full download.

        Args:
            category:    HF dataset config name, e.g. ``"raw_review_Electronics"``.
            max_records: Maximum number of rows to load into memory.
            cfg:         Project config (used for HF dataset name).

        Returns:
            DataFrame with columns: user_id, item_id, rating, timestamp,
            title, description, price, category.

        Raises:
            RuntimeError: If the HF streaming connection fails.
        """
        logger.info(
            "Streaming up to %d records from HF dataset '%s' / '%s'",
            max_records,
            cfg.data.hf_dataset_name,
            category,
        )

        try:
            ds = load_dataset(
                cfg.data.hf_dataset_name,
                category,
                streaming=True,
                trust_remote_code=True,
            )
        except Exception as exc:
            logger.error("Failed to connect to HuggingFace: %s", exc)
            raise RuntimeError(f"HuggingFace streaming failed: {exc}") from exc

        records = []
        for item in ds["full"].take(max_records):
            records.append(
                {
                    "user_id": item["user_id"],
                    "item_id": item["asin"],
                    "rating": float(item["rating"]),
                    "timestamp": int(item["timestamp"]),
                    "title": item.get("title", ""),
                    "description": item.get("text", ""),
                    "price": item.get("price", None),
                    "category": item.get("main_category", ""),
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
