"""
embeddings.py — Item and positional embedding layers.

Implemented in Module 02.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ItemEmbedding(nn.Module):
    """Learnable item embedding with optional dropout."""

    def __init__(self, num_items: int, embedding_dim: int, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_ids: (batch, seq_len) int tensor
        Returns:
            (batch, seq_len, embedding_dim) float tensor
        """
        return self.dropout(self.embedding(item_ids))


class PositionalEmbedding(nn.Module):
    """Learnable positional embeddings (absolute position)."""

    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, embedding_dim)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        return self.embedding(positions)  # (1, seq_len, dim)
