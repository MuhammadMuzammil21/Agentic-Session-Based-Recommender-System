"""
test_models.py — Unit tests for neural model components.
"""

import pytest
import torch


class TestItemEmbedding:
    def test_output_shape(self):
        from models.embeddings import ItemEmbedding
        emb = ItemEmbedding(num_items=100, embedding_dim=32)
        ids = torch.randint(0, 100, (4, 10))  # (batch=4, seq=10)
        out = emb(ids)
        assert out.shape == (4, 10, 32)

    def test_padding_idx_zero_grad(self):
        from models.embeddings import ItemEmbedding
        emb = ItemEmbedding(num_items=10, embedding_dim=8)
        assert emb.embedding.padding_idx == 0


class TestPositionalEmbedding:
    def test_output_shape(self):
        from models.embeddings import PositionalEmbedding
        pos = PositionalEmbedding(max_seq_len=50, embedding_dim=32)
        out = pos(seq_len=10, device=torch.device("cpu"))
        assert out.shape == (1, 10, 32)


class TestSessionEncoder:
    def test_instantiation(self, cfg):
        from models.encoder import SessionEncoder
        num_items = 500
        enc = SessionEncoder(cfg["model"], num_items=num_items)
        assert enc is not None

    def test_item_embeddings_shape(self, cfg):
        from models.encoder import SessionEncoder
        num_items = 200
        enc = SessionEncoder(cfg["model"], num_items=num_items)
        weights = enc.get_item_embeddings()
        assert weights.shape[0] == num_items
        assert weights.shape[1] == cfg["model"]["embedding_dim"]
