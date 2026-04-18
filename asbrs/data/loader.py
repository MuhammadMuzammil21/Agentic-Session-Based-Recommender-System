"""
loader.py — Data loading utilities for the Amazon Electronics dataset.

Supports streaming from HuggingFace to avoid downloading the full archive.
Implemented in Module 01.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

# Placeholder — full implementation in Module 01


class AmazonDataLoader:
    """Load Amazon Reviews 2023 Electronics data via HuggingFace streaming."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.raw_dir = Path(cfg["raw_dir"])
        self.processed_dir = Path(cfg["processed_dir"])

    # ── Public API ────────────────────────────────────────────────────────────

    def stream_reviews(self) -> Iterator[dict]:
        """Yield raw review dicts from the HF streaming dataset."""
        raise NotImplementedError("Implemented in Module 01")

    def stream_metadata(self) -> Iterator[dict]:
        """Yield raw item metadata dicts from the HF streaming dataset."""
        raise NotImplementedError("Implemented in Module 01")

    def load_processed(self) -> tuple:
        """Load pre-processed data splits from disk."""
        raise NotImplementedError("Implemented in Module 01")
