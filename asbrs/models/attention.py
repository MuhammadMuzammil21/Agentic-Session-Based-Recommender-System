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
        """Pool a sequence of 128-num GRU memories into ONE 128-num summary.

        Intuition
        ---------
        The GRU gave us 50 memory snapshots per session — one for every
        item position. To score next-item candidates we need *one* vector
        summarising the whole session, not 50.

        Attention solves this as a *learned weighted average*:
            summary = w₁ · h₁ + w₂ · h₂ + ... + w₅₀ · h₅₀
        where the weights wᵢ sum to 1.0 and are CHOSEN BY THE MODEL based
        on how relevant each position is. Positions the model finds
        informative get higher weight; PAD positions get weight 0.

        Where do the weights come from? We:
          1. Build a "query" vector — the average of the real (non-PAD)
             hidden states. It represents the session's overall content.
          2. Compute a similarity score between the query and each
             individual hidden state ("keys").
          3. Softmax those scores → weights summing to 1.
          4. Use the weights to average the "values" (= hidden states).

        The "multi-head" trick repeats this in parallel with different
        learned projections so the model can attend to multiple types of
        patterns at once (e.g. recency, category, brand). We then merge.

        Args:
            hidden_states: [B, L, H] — the GRU memories from the encoder.
                           Each [b, t, :] is the GRU's 128-num memory
                           for session b after reading item t.
            mask:          [B, L]    — 1.0 for real items, 0.0 for PAD.

        Returns:
            context:       [B, H]    — the ONE session-summary vector
                                       per batch element.
            attn_weights:  [B, L]    — how much each position contributed
                                       (sums to 1.0 over real positions).
                                       Shown as the demo's bar chart.
        """
        # B = batch size, L = seq len (50), H = hidden dim (128).
        B, L, H = hidden_states.shape

        # ── STEP A: build the QUERY = mean of non-PAD hidden states ──────────
        # We need a "what is this session about?" vector to compare each
        # position against. The simplest choice: take the average of all
        # real (non-PAD) hidden states. That gives one 128-num query per
        # session, representing a coarse summary of the session's content.
        mask_3d = mask.unsqueeze(-1).float()                       # [B, L, 1]
        n_real = mask_3d.sum(dim=1).clamp(min=1.0)                 # [B, 1] — how many real items
        query = (hidden_states * mask_3d).sum(dim=1) / n_real     # [B, H] — average over real items

        # ── STEP B: apply LEARNED projections to query / keys / values ───────
        # In raw attention we'd compare `query` directly against `hidden_states`.
        # But we want the model to LEARN what "similar" means, so we pass each
        # through a trainable linear layer that can reshape the vectors into
        # whatever space makes attention work best for our task.
        #   Q (query)  = self.q_proj(query)         — what we're looking FOR
        #   K (keys)   = self.k_proj(hidden_states) — what to match against
        #   V (values) = self.v_proj(hidden_states) — what to weight & sum
        # The Q-K-V naming comes from the original Transformer paper.
        Q = self.q_proj(query)                                     # [B, H]
        K = self.k_proj(hidden_states)                             # [B, L, H]
        V = self.v_proj(hidden_states)                             # [B, L, H]

        # ── STEP C: split into multiple ATTENTION HEADS (4 heads × 32 dims) ──
        # Instead of computing one big similarity score over 128 dimensions,
        # we split each vector into 4 chunks of 32 dimensions and let each
        # chunk compute its own attention pattern in parallel. Then we
        # concatenate the results. This lets the model attend to different
        # *types* of patterns (e.g. one head learns recency, another learns
        # category similarity). It's strictly more expressive than one head.
        Q = Q.view(B, self.num_heads, self.head_dim)               # [B, nh, hd] = [B, 4, 32]
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, nh, L, hd]
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, nh, L, hd]

        # Q is a single query per session, K/V have one vector per position.
        # Add a length-1 dim to Q so we can batch-matmul against K's L positions.
        Q = Q.unsqueeze(2)                                         # [B, nh, 1, hd]

        # ── STEP D: SCALED DOT-PRODUCT SCORES ────────────────────────────────
        # For each head, compute the dot product of the query with every key.
        # High dot product = "this position's vector points in a similar
        # direction to the query" = "this position is relevant."
        # We divide by √head_dim to keep the numbers in a sane range before
        # softmax (otherwise large dimensions make softmax saturate).
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, nh, 1, L]
        scores = scores.squeeze(2)                                  # [B, nh, L]

        # ── STEP E: MASK OUT PAD positions ──────────────────────────────────
        # Set the score at PAD positions to a huge negative number so that
        # softmax later assigns them ~0 weight. We don't want the model to
        # accidentally "attend to" filler tokens.
        pad_mask = (mask == 0).unsqueeze(1)                        # [B, 1, L]
        scores = scores.masked_fill(pad_mask, MASK_FILL_VALUE)     # [B, nh, L]

        # ── STEP F: SOFTMAX scores → attention weights ───────────────────────
        # Softmax turns raw scores into a probability distribution: every
        # weight is between 0 and 1, and they sum to 1 across positions.
        # PAD positions get ~0 because their score was -1e9.
        attn = F.softmax(scores, dim=-1)                           # [B, nh, L]
        attn = self.attn_dropout(attn)                             # regularisation

        # ── STEP G: WEIGHTED AVERAGE of the value vectors ────────────────────
        # Now use those weights to combine the value vectors.
        # context_heads[b, head, :] = Σₜ attn[b, head, t] · V[b, head, t, :]
        # This is the "weighted average" we wanted — one 32-num vector per head.
        context_heads = torch.matmul(
            attn.unsqueeze(2), V
        ).squeeze(2)                                               # [B, nh, hd]

        # Glue the 4 head outputs (each 32-num) back into one 128-num vector,
        # then apply a final learned linear transform to mix them.
        context_heads = context_heads.contiguous().view(B, H)      # [B, H]
        context = self.out_proj(context_heads)                     # [B, H]

        # ── STEP H: average the per-head attention weights for visualisation ─
        # The 4 heads might disagree about which positions matter. For the
        # demo's bar chart we just take the mean across heads — one weight
        # per input position, summing to 1.
        attn_weights = attn.mean(dim=1)                            # [B, L]

        return context, attn_weights
