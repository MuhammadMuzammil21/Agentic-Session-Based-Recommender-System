"""models/attention.py — Attention-pooling layer for session representations.

Single responsibility: pool a sequence of GRU hidden states into one
context vector, returning both the pooled representation and the
per-token attention weights (used for explainability in Module 04).
"""

from __future__ import annotations

import logging
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MASK_FILL_VALUE: float = -1e9


class SelfAttentionLayer(nn.Module):
    """Attention-pooling over GRU hidden states.

    Computes a query from the mean of non-padded hidden states, then
    performs scaled dot-product attention over key/value projections to
    produce a single context vector per sample plus per-token weights.

    Args:
        hidden_dim: Dimensionality of incoming hidden states (H).
        num_heads:  Number of parallel attention heads.
                    Each head uses hidden_dim // num_heads dimensions.
        dropout:    Dropout probability applied to attention weights.

    Raises:
        ValueError: If hidden_dim is not divisible by num_heads.

    Example:
        attn = SelfAttentionLayer(hidden_dim=128, num_heads=4, dropout=0.1)
        h    = torch.randn(4, 10, 128)          # [B, L, H]
        mask = torch.ones(4, 10)                # [B, L]
        ctx, weights = attn(h, mask)            # [B, H], [B, L]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by "
                f"num_heads ({num_heads})"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(p=dropout)

        self._init_weights()
        logger.debug(
            "SelfAttentionLayer: hidden_dim=%d, num_heads=%d, dropout=%.2f",
            hidden_dim,
            num_heads,
            dropout,
        )

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Xavier uniform init for all projection layers."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Pool hidden states into a single context vector via attention.

        Args:
            hidden_states: FloatTensor [B, L, H] — GRU output.
            mask:          FloatTensor [B, L] — 1 for real tokens, 0 for PAD.

        Returns:
            context:       FloatTensor [B, H] — attended session representation.
            attn_weights:  FloatTensor [B, L] — per-token attention weights
                           (summing to 1.0 over real tokens per sample).
        """
        B, L, H = hidden_states.shape

        # ── Query: mean of non-padded hidden states ──────────────────────────
        # mask: [B, L] → [B, L, 1] for broadcasting
        mask_3d = mask.unsqueeze(-1).float()                       # [B, L, 1]
        n_real = mask_3d.sum(dim=1).clamp(min=1.0)                # [B, 1]
        query = (hidden_states * mask_3d).sum(dim=1) / n_real     # [B, H]

        # ── Projections ──────────────────────────────────────────────────────
        Q = self.q_proj(query)                                     # [B, H]
        K = self.k_proj(hidden_states)                             # [B, L, H]
        V = self.v_proj(hidden_states)                             # [B, L, H]

        # Reshape to multi-head: [B, num_heads, *, head_dim]
        Q = Q.view(B, self.num_heads, self.head_dim)               # [B, nh, hd]
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, nh, L, hd]
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, nh, L, hd]

        # Q: [B, nh, hd] → [B, nh, 1, hd] for batched matmul
        Q = Q.unsqueeze(2)                                         # [B, nh, 1, hd]

        # ── Scaled dot-product scores ─────────────────────────────────────────
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, nh, 1, L]
        scores = scores.squeeze(2)                                  # [B, nh, L]

        # ── Apply padding mask (fill PAD positions with large negative) ───────
        # mask: [B, L] → [B, 1, L] → broadcast over heads
        pad_mask = (mask == 0).unsqueeze(1)                        # [B, 1, L]
        scores = scores.masked_fill(pad_mask, MASK_FILL_VALUE)     # [B, nh, L]

        # ── Softmax → dropout ─────────────────────────────────────────────────
        attn = F.softmax(scores, dim=-1)                           # [B, nh, L]
        attn = self.attn_dropout(attn)

        # ── Weighted sum over values ──────────────────────────────────────────
        # attn: [B, nh, L] → [B, nh, 1, L]
        context_heads = torch.matmul(
            attn.unsqueeze(2), V
        ).squeeze(2)                                               # [B, nh, hd]
        context_heads = context_heads.contiguous().view(B, H)     # [B, H]
        context = self.out_proj(context_heads)                     # [B, H]

        # ── Collapse attention weights across heads (mean) ────────────────────
        attn_weights = attn.mean(dim=1)                            # [B, L]

        return context, attn_weights
