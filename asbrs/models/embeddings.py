"""models/embeddings.py — Item embedding layer with Xavier init and dropout.

Single responsibility: map integer item indices to dense vectors.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

XAVIER_GAIN: float = 1.0


class ItemEmbedding(nn.Module):
    """Learnable item embedding layer.

    Maps integer item indices to dense embedding vectors.
    Padding index (0) is excluded from gradient updates.
    Weight is initialised with Xavier uniform for faster convergence.

    Args:
        vocab_size:  Total number of items (including PAD and UNK tokens).
        embed_dim:   Dimensionality of each embedding vector.
        dropout:     Dropout probability applied after embedding lookup.
        padding_idx: Index of the PAD token (excluded from gradients).

    Example:
        emb = ItemEmbedding(vocab_size=500, embed_dim=64, dropout=0.2)
        x   = torch.randint(0, 500, (4, 10))  # [B=4, L=10]
        out = emb(x)                           # [4, 10, 64]
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        dropout: float = 0.2,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.embed_dim = embed_dim
        self.padding_idx = padding_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()
        logger.debug(
            "ItemEmbedding: vocab_size=%d, embed_dim=%d, dropout=%.2f",
            vocab_size,
            embed_dim,
            dropout,
        )

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Apply Xavier uniform initialisation, then zero the PAD row."""
        nn.init.xavier_uniform_(self.embedding.weight, gain=XAVIER_GAIN)
        with torch.no_grad():
            self.embedding.weight[self.padding_idx].fill_(0.0)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        """Embed a batch of item-index sequences.

        Args:
            x: LongTensor of shape [B, L] containing item indices.

        Returns:
            FloatTensor of shape [B, L, embed_dim].
        """
        return self.dropout(self.embedding(x))

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_all_embeddings(self) -> Tensor:
        """Return the full embedding weight matrix.

        Returns the live parameter tensor so gradients can flow during
        training (cross-entropy needs to update target/non-target rows).
        Eval call sites are responsible for using ``torch.no_grad()``.

        Returns:
            FloatTensor of shape [vocab_size, embed_dim].
        """
        return self.embedding.weight
