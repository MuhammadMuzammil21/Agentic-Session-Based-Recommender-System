"""data/vocab.py — Item vocabulary (ASIN ↔ integer index mapping).

Built from training item IDs only. Handles PAD and UNK special tokens.
Persisted to disk as a JSON file for reproducibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PAD_TOKEN: str = "<PAD>"
UNK_TOKEN: str = "<UNK>"
PAD_IDX: int = 0
UNK_IDX: int = 1


class Vocabulary:
    """Bidirectional item ASIN ↔ integer index mapping.

    Special tokens:
        PAD (index 0): padding token for sequence alignment.
        UNK (index 1): unknown item fallback.

    Example:
        vocab = Vocabulary()
        vocab.build(["B001", "B002", "B003"])
        idx = vocab.encode("B001")   # → 2
        asin = vocab.decode(2)       # → "B001"
    """

    def __init__(self) -> None:
        self._item2idx: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
        self._idx2item: Dict[int, str] = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN}

    # ── Construction ──────────────────────────────────────────────────────────

    def build(self, item_ids: List[str]) -> None:
        """Populate vocabulary from a list of item IDs.

        Inserts each unique ID not already present. Special tokens are
        preserved at indices 0 and 1.

        Args:
            item_ids: List of raw item ASINs (duplicates are ignored).
        """
        before = len(self._item2idx)
        for item_id in item_ids:
            if item_id not in self._item2idx:
                idx = len(self._item2idx)
                self._item2idx[item_id] = idx
                self._idx2item[idx] = item_id
        added = len(self._item2idx) - before
        logger.info("Vocabulary built: %d items added, %d total", added, len(self))

    # ── Lookup ────────────────────────────────────────────────────────────────

    def encode(self, item_id: str) -> int:
        """Map an item ASIN to its integer index.

        Args:
            item_id: Item ASIN string.

        Returns:
            Integer index, or UNK_IDX if the item is not in vocabulary.
        """
        return self._item2idx.get(item_id, UNK_IDX)

    def decode(self, idx: int) -> str:
        """Map an integer index back to an item ASIN.

        Args:
            idx: Integer index.

        Returns:
            Item ASIN string, or UNK_TOKEN if the index is out of range.
        """
        return self._idx2item.get(idx, UNK_TOKEN)

    def __len__(self) -> int:
        """Return the total number of tokens (including special tokens)."""
        return len(self._item2idx)

    def __contains__(self, item_id: str) -> bool:
        """Return True if item_id is in the vocabulary."""
        return item_id in self._item2idx

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Serialize vocabulary to a JSON file.

        Args:
            path: Destination file path (parent directories created if needed).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"item2idx": self._item2idx}
        with path.open("w") as fh:
            json.dump(payload, fh)
        logger.info("Vocabulary saved to %s (%d tokens)", path, len(self))

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        """Deserialize a vocabulary from a JSON file.

        Args:
            path: Path to a previously saved vocabulary JSON.

        Returns:
            Reconstructed Vocabulary instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")
        with path.open("r") as fh:
            payload: dict = json.load(fh)
        vocab = cls()
        vocab._item2idx = {k: int(v) for k, v in payload["item2idx"].items()}
        vocab._idx2item = {int(v): k for k, v in payload["item2idx"].items()}
        logger.info("Vocabulary loaded from %s (%d tokens)", path, len(vocab))
        return vocab
