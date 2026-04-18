"""
encoder.py — Session encoder: GRU + Multi-Head Attention → session embedding.

Architecture:
    items → ItemEmbedding → GRU → MultiHeadSelfAttention → AttentionPooling
                                                          → session_embedding

Implemented in Module 02.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.embeddings import ItemEmbedding, PositionalEmbedding
from models.attention import MultiHeadSelfAttention, AttentionPooling


class SessionEncoder(nn.Module):
    """Encode a variable-length item sequence into a fixed-size embedding."""

    def __init__(self, cfg: dict, num_items: int):
        super().__init__()
        emb_dim = cfg["embedding_dim"]
        hid_dim = cfg["hidden_dim"]
        n_heads = cfg["num_attention_heads"]
        dropout = cfg["dropout"]
        max_len = cfg["max_seq_len"]

        self.item_emb = ItemEmbedding(num_items, emb_dim, dropout)
        self.pos_emb = PositionalEmbedding(max_len, emb_dim)

        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0,
        )
        self.attention = MultiHeadSelfAttention(hid_dim, n_heads, dropout)
        self.pooling = AttentionPooling(hid_dim)
        self.layer_norm = nn.LayerNorm(hid_dim)

    def forward(
        self,
        item_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            item_ids:     (batch, seq_len)
            padding_mask: (batch, seq_len) bool — True for padding positions
        Returns:
            session_emb:  (batch, hidden_dim)
        """
        raise NotImplementedError("Implemented in Module 02")

    def get_item_embeddings(self) -> torch.Tensor:
        """Return the full item embedding matrix for retrieval (num_items, emb_dim)."""
        return self.item_emb.embedding.weight.detach()
