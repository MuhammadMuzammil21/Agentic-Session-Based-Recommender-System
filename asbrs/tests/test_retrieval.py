"""tests/test_retrieval.py — Unit tests for retrieval components.

Tests stub interfaces only; full logic is implemented in Module 03.
"""

from __future__ import annotations

import pytest


class TestCollaborativeRetriever:
    def test_instantiation(self, cfg) -> None:
        from retrieval.collaborative import CollaborativeRetriever
        cr = CollaborativeRetriever(cfg.retrieval)
        assert cr.top_k == cfg.retrieval.cf_top_k

    def test_retrieve_raises_before_fit(self, cfg) -> None:
        from retrieval.collaborative import CollaborativeRetriever
        cr = CollaborativeRetriever(cfg.retrieval)
        with pytest.raises(NotImplementedError):
            cr.retrieve([0, 1, 2])


class TestContentBasedRetriever:
    def test_instantiation(self, cfg) -> None:
        from retrieval.content_based import ContentBasedRetriever
        cb = ContentBasedRetriever(cfg.retrieval)
        assert cb.top_k == cfg.retrieval.content_top_k


class TestHybridRetriever:
    def test_instantiation(self, cfg) -> None:
        from retrieval.collaborative import CollaborativeRetriever
        from retrieval.content_based import ContentBasedRetriever
        from retrieval.hybrid import HybridRetriever

        cf = CollaborativeRetriever(cfg.retrieval)
        cb = ContentBasedRetriever(cfg.retrieval)
        hybrid = HybridRetriever(cf, cb, cfg.retrieval)
        assert hybrid.final_top_k == cfg.retrieval.final_top_k
