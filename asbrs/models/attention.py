"""
attention.py — Multi-head self-attention and session-level attention pooling.

Implemented in Module 02.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head self-attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) bool — True for padding positions
        Returns:
            (batch, seq_len, hidden_dim)
        """
        raise NotImplementedError("Implemented in Module 02")


class AttentionPooling(nn.Module):
    """Attention-weighted pooling to compress a sequence into one vector."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) bool — True for padding
        Returns:
            (batch, hidden_dim)  — session-level embedding
        """
        raise NotImplementedError("Implemented in Module 02")
