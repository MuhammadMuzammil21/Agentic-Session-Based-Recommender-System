"""
preprocessor.py — Feature engineering and data cleaning for ASBRS.

Transforms raw sessions into model-ready tensors / DataFrames.
Implemented in Module 01.
"""

from __future__ import annotations

from typing import List, Tuple

# Placeholder — full implementation in Module 01


class Preprocessor:
    """Clean, filter, and featurize raw session data."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.min_item_freq = cfg["min_item_freq"]

    def fit_transform(
        self, sessions: List[dict]
    ) -> Tuple[List[dict], List[dict], List[dict]]:
        """
        Fit vocabulary stats and return (train, val, test) session splits.
        Uses leave-one-out splitting: last item → test, second-to-last → val.
        """
        raise NotImplementedError("Implemented in Module 01")

    def transform(self, sessions: List[dict]) -> List[dict]:
        """Apply fitted preprocessing to new sessions (inference time)."""
        raise NotImplementedError("Implemented in Module 01")
