"""data/preprocessor.py — Encode sessions to tensors and build DataLoaders.

Converts Session objects (string item IDs) into EncodedSession objects
(integer indices) and wraps them in a PyTorch DataLoader.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from data.interfaces import EncodedSession, Session
from data.vocab import PAD_IDX, Vocabulary

logger = logging.getLogger(__name__)


# ── Internal Dataset ──────────────────────────────────────────────────────────


class _SessionDataset(Dataset):
    """PyTorch Dataset wrapping a list of EncodedSession objects."""

    def __init__(self, encoded: List[EncodedSession]) -> None:
        self._data = encoded

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        es = self._data[idx]
        return {
            "input_ids": torch.tensor(es.input_ids, dtype=torch.long),
            "lengths": torch.tensor(es.session_len, dtype=torch.long),
            "target": torch.tensor(es.target_id, dtype=torch.long),
        }


# ── Preprocessor ──────────────────────────────────────────────────────────────


class SessionPreprocessor:
    """Encode and pad sessions for model consumption.

    Example:
        prep = SessionPreprocessor()
        encoded = prep.prepare(sessions, vocab, max_len=50)
        loader = prep.to_dataloader(encoded, batch_size=256, shuffle=True)
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def prepare(
        self,
        sessions: List[Session],
        vocab: Vocabulary,
        max_len: int,
    ) -> List[EncodedSession]:
        """Encode item IDs and produce left-padded input/target pairs.

        For each session:
        - input_ids  = all items except the last, truncated and left-padded.
        - target_id  = the last item in the session.
        - session_len = min(len(session) - 1, max_len).

        Args:
            sessions: List of Session objects (must have length >= 2).
            vocab:    Fitted Vocabulary used for integer encoding.
            max_len:  Maximum input sequence length after truncation.

        Returns:
            List of EncodedSession objects, one per valid session.

        Raises:
            ValueError: If max_len < 1.
        """
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")

        encoded: List[EncodedSession] = []
        skipped = 0

        for session in sessions:
            if session.length < 2:
                skipped += 1
                continue

            target_id = vocab.encode(session.item_ids[-1])
            input_items = session.item_ids[:-1]

            # Truncate to most-recent max_len items
            if len(input_items) > max_len:
                input_items = input_items[-max_len:]

            true_len = len(input_items)
            encoded_items = [vocab.encode(iid) for iid in input_items]

            # Left-pad with PAD_IDX
            pad_len = max_len - true_len
            input_ids = [PAD_IDX] * pad_len + encoded_items

            encoded.append(
                EncodedSession(
                    input_ids=input_ids,
                    target_id=target_id,
                    session_len=true_len,
                )
            )

        logger.info(
            "prepare: %d sessions encoded, %d skipped (length < 2)",
            len(encoded),
            skipped,
        )
        return encoded

    def to_dataloader(
        self,
        encoded: List[EncodedSession],
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """Wrap encoded sessions in a PyTorch DataLoader.

        Each batch is a dict with keys:
            - ``"input_ids"``: LongTensor of shape (B, L)
            - ``"lengths"``:   LongTensor of shape (B,)
            - ``"target"``:    LongTensor of shape (B,)

        Args:
            encoded:    List of EncodedSession objects.
            batch_size: Number of samples per batch.
            shuffle:    Whether to shuffle at the start of each epoch.

        Returns:
            Configured PyTorch DataLoader.

        Raises:
            ValueError: If encoded is empty or batch_size < 1.
        """
        if not encoded:
            raise ValueError("encoded must be a non-empty list")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        dataset = _SessionDataset(encoded)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=torch.cuda.is_available(),
        )
        logger.info(
            "DataLoader created: %d samples, batch_size=%d, shuffle=%s",
            len(dataset),
            batch_size,
            shuffle,
        )
        return loader
