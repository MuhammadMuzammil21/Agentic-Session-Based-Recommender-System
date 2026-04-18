"""
vocab.py — Item vocabulary (item_id ↔ integer index mapping).

Built from training data; handles <PAD> and <UNK> tokens.
Implemented in Module 01.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class Vocab:
    """Bidirectional item ↔ index mapping."""

    PAD_IDX: int = 0
    UNK_IDX: int = 1

    def __init__(self):
        self._item2idx: Dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self._idx2item: Dict[int, str] = {0: PAD_TOKEN, 1: UNK_TOKEN}

    # ── Construction ──────────────────────────────────────────────────────────

    def build(self, items: List[str], min_freq: int = 1) -> "Vocab":
        """Build vocab from a list of item IDs (with optional min frequency)."""
        raise NotImplementedError("Implemented in Module 01")

    # ── Lookup ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._item2idx)

    def item2idx(self, item: str) -> int:
        return self._item2idx.get(item, self.UNK_IDX)

    def idx2item(self, idx: int) -> str:
        return self._idx2item.get(idx, UNK_TOKEN)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        raise NotImplementedError("Implemented in Module 01")

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        raise NotImplementedError("Implemented in Module 01")
