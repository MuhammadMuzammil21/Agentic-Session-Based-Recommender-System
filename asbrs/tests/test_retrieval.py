"""tests/test_retrieval.py — Unit tests for Module 03: Hybrid Retrieval.

Covers:
  - ItemBasedCF: fit, get_candidates (counts, no dupes, no inputs), save/load
  - ContentBasedFilter: fit, get_candidates (counts, TF-IDF weighting), save/load
  - HybridRetriever: score merging, retrieve, retrieve_batch
  - Legacy stub backward-compatibility tests
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

# ── Helpers ───────────────────────────────────────────────────────────────────

NUM_USERS = 50
NUM_ITEMS = 100


def _make_interaction_matrix(
    num_users: int = NUM_USERS, num_items: int = NUM_ITEMS, density: float = 0.1
) -> csr_matrix:
    """Return a random sparse implicit-feedback matrix."""
    rng = np.random.default_rng(42)
    data = rng.random((num_users, num_items))
    mask = data < density
    data[~mask] = 0.0
    return csr_matrix(data)


def _make_vocab_and_metadata(num_items: int = NUM_ITEMS):
    """Return (Vocabulary, DataFrame) of dummy item metadata."""
    from data.vocab import Vocabulary

    categories = ["Electronics", "Accessories", "Cables", "Batteries", "Audio"]
    records = []
    item_asins = [f"B{i:08d}" for i in range(num_items)]
    for i, asin in enumerate(item_asins):
        records.append(
            {
                "item_id": asin,
                "title": f"Product title {i}",
                "description": f"This is a great product number {i} for all your needs",
                "category": categories[i % len(categories)],
                "price": float(i * 9.99),
            }
        )
    df = pd.DataFrame(records)

    vocab = Vocabulary()
    vocab.build(item_asins)
    return vocab, df


# ── ItemBasedCF ───────────────────────────────────────────────────────────────


class TestItemBasedCF:
    """Tests for retrieval.collaborative.ItemBasedCF."""

    def test_fit_runs_without_error(self) -> None:
        from retrieval.collaborative import ItemBasedCF

        cf = ItemBasedCF()
        matrix = _make_interaction_matrix()
        cf.fit(matrix)
        assert cf._sim_matrix is not None
        assert cf._num_items == NUM_ITEMS

    def test_fit_empty_matrix_raises(self) -> None:
        from retrieval.collaborative import ItemBasedCF

        cf = ItemBasedCF()
        with pytest.raises(ValueError):
            cf.fit(csr_matrix((0, 0)))

    def test_get_candidates_before_fit_raises(self) -> None:
        from retrieval.collaborative import ItemBasedCF

        cf = ItemBasedCF()
        with pytest.raises(RuntimeError):
            cf.get_candidates([0, 1], top_k=10)

    def test_get_candidates_returns_correct_count(self) -> None:
        from retrieval.collaborative import ItemBasedCF

        cf = ItemBasedCF()
        cf.fit(_make_interaction_matrix())
        top_k = 20
        results = cf.get_candidates([0, 1, 2], top_k=top_k)
        assert len(results) <= top_k

    def test_get_candidates_no_duplicates(self) -> None:
        from retrieval.collaborative import ItemBasedCF

        cf = ItemBasedCF()
        cf.fit(_make_interaction_matrix())
        results = cf.get_candidates([0, 1, 2], top_k=30)
        indices = [r[0] for r in results]
        assert len(indices) == len(set(indices)), "Duplicate candidates found"

    def test_get_candidates_excludes_input_items(self) -> None:
        from retrieval.collaborative import ItemBasedCF

        seed_ids = [0, 1, 2]
        cf = ItemBasedCF()
        cf.fit(_make_interaction_matrix())
        results = cf.get_candidates(seed_ids, top_k=50)
        returned_ids = {r[0] for r in results}
        for s in seed_ids:
            assert s not in returned_ids, f"Seed item {s} appeared in candidates"

    def test_get_candidates_sorted_descending(self) -> None:
        from retrieval.collaborative import ItemBasedCF

        cf = ItemBasedCF()
        cf.fit(_make_interaction_matrix())
        results = cf.get_candidates([0, 1], top_k=20)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_get_candidates_empty_seed(self) -> None:
        from retrieval.collaborative import ItemBasedCF

        cf = ItemBasedCF()
        cf.fit(_make_interaction_matrix())
        results = cf.get_candidates([], top_k=10)
        assert results == []

    def test_save_load_round_trip(self) -> None:
        from retrieval.collaborative import ItemBasedCF

        cf = ItemBasedCF()
        cf.fit(_make_interaction_matrix())
        original = cf.get_candidates([0, 1, 2], top_k=10)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "cf.pkl"
            cf.save(ckpt)
            cf2 = ItemBasedCF.load(ckpt)

        reloaded = cf2.get_candidates([0, 1, 2], top_k=10)
        assert original == reloaded, "save/load round-trip mismatch"

    def test_save_unfitted_raises(self) -> None:
        from retrieval.collaborative import ItemBasedCF

        cf = ItemBasedCF()
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(RuntimeError):
                cf.save(Path(tmp) / "cf.pkl")

    def test_load_missing_file_raises(self) -> None:
        from retrieval.collaborative import ItemBasedCF

        with pytest.raises(FileNotFoundError):
            ItemBasedCF.load(Path("nonexistent_cf.pkl"))


# ── ContentBasedFilter ────────────────────────────────────────────────────────


class TestContentBasedFilter:
    """Tests for retrieval.content_based.ContentBasedFilter."""

    def test_fit_runs_without_error(self) -> None:
        from retrieval.content_based import ContentBasedFilter

        _, df = _make_vocab_and_metadata()
        cb = ContentBasedFilter()
        cb.fit(df)
        assert cb._tfidf_matrix is not None
        assert cb._tfidf_matrix.shape[0] == NUM_ITEMS

    def test_fit_missing_column_raises(self) -> None:
        from retrieval.content_based import ContentBasedFilter

        df = pd.DataFrame({"item_id": ["A"], "title": ["T"]})
        cb = ContentBasedFilter()
        with pytest.raises(ValueError):
            cb.fit(df)

    def test_get_candidates_before_fit_raises(self) -> None:
        from retrieval.content_based import ContentBasedFilter

        vocab, _ = _make_vocab_and_metadata()
        cb = ContentBasedFilter()
        with pytest.raises(RuntimeError):
            cb.get_candidates([2, 3], top_k=10, vocab=vocab)

    def test_get_candidates_returns_correct_count(self) -> None:
        from retrieval.content_based import ContentBasedFilter

        vocab, df = _make_vocab_and_metadata()
        cb = ContentBasedFilter()
        cb.fit(df)
        top_k = 15
        # Use item idx 2 and 3 (which correspond to B00000002 and B00000003).
        seed_ids = [vocab.encode("B00000002"), vocab.encode("B00000003")]
        results = cb.get_candidates(seed_ids, top_k=top_k, vocab=vocab)
        assert len(results) <= top_k

    def test_get_candidates_no_duplicates(self) -> None:
        from retrieval.content_based import ContentBasedFilter

        vocab, df = _make_vocab_and_metadata()
        cb = ContentBasedFilter()
        cb.fit(df)
        seed_ids = [vocab.encode("B00000002")]
        results = cb.get_candidates(seed_ids, top_k=20, vocab=vocab)
        indices = [r[0] for r in results]
        assert len(indices) == len(set(indices))

    def test_get_candidates_excludes_input_items(self) -> None:
        from retrieval.content_based import ContentBasedFilter

        vocab, df = _make_vocab_and_metadata()
        cb = ContentBasedFilter()
        cb.fit(df)
        seed_asins = ["B00000002", "B00000005"]
        seed_ids = [vocab.encode(a) for a in seed_asins]
        results = cb.get_candidates(seed_ids, top_k=50, vocab=vocab)
        returned_ids = {r[0] for r in results}
        for s in seed_ids:
            assert s not in returned_ids, f"Seed item {s} appeared in candidates"

    def test_get_candidates_sorted_descending(self) -> None:
        from retrieval.content_based import ContentBasedFilter

        vocab, df = _make_vocab_and_metadata()
        cb = ContentBasedFilter()
        cb.fit(df)
        seed_ids = [vocab.encode("B00000002")]
        results = cb.get_candidates(seed_ids, top_k=20, vocab=vocab)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_get_candidates_empty_seed_returns_empty(self) -> None:
        from retrieval.content_based import ContentBasedFilter

        vocab, df = _make_vocab_and_metadata()
        cb = ContentBasedFilter()
        cb.fit(df)
        # Decode an out-of-vocab index → UNK which is not in metadata.
        results = cb.get_candidates([], top_k=10, vocab=vocab)
        assert results == []

    def test_tfidf_non_zero(self) -> None:
        """Verify TF-IDF matrix is actually non-zero (weights are active)."""
        from retrieval.content_based import ContentBasedFilter

        _, df = _make_vocab_and_metadata()
        cb = ContentBasedFilter()
        cb.fit(df)
        assert cb._tfidf_matrix.nnz > 0

    def test_save_load_round_trip(self) -> None:
        from retrieval.content_based import ContentBasedFilter

        vocab, df = _make_vocab_and_metadata()
        cb = ContentBasedFilter()
        cb.fit(df)
        seed_ids = [vocab.encode("B00000002"), vocab.encode("B00000003")]
        original = cb.get_candidates(seed_ids, top_k=10, vocab=vocab)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "cb.pkl"
            cb.save(ckpt)
            cb2 = ContentBasedFilter.load(ckpt)

        reloaded = cb2.get_candidates(seed_ids, top_k=10, vocab=vocab)
        # Compare item indices and scores.
        assert [r[0] for r in original] == [r[0] for r in reloaded]
        np.testing.assert_allclose(
            [r[1] for r in original], [r[1] for r in reloaded], rtol=1e-5
        )

    def test_save_unfitted_raises(self) -> None:
        from retrieval.content_based import ContentBasedFilter

        cb = ContentBasedFilter()
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(RuntimeError):
                cb.save(Path(tmp) / "cb.pkl")

    def test_load_missing_file_raises(self) -> None:
        from retrieval.content_based import ContentBasedFilter

        with pytest.raises(FileNotFoundError):
            ContentBasedFilter.load(Path("nonexistent_cb.pkl"))


# ── HybridRetriever ───────────────────────────────────────────────────────────


class TestHybridRetriever:
    """Tests for retrieval.hybrid.HybridRetriever."""

    @pytest.fixture()
    def fitted_retriever(self):
        """Return a HybridRetriever with CF and CB fitted on synthetic data."""
        from retrieval.collaborative import ItemBasedCF
        from retrieval.content_based import ContentBasedFilter
        from retrieval.hybrid import HybridRetriever

        vocab, df = _make_vocab_and_metadata()
        matrix = _make_interaction_matrix()

        cf = ItemBasedCF()
        cf.fit(matrix)

        cb = ContentBasedFilter()
        cb.fit(df)

        hybrid = HybridRetriever(cf, cb, cf_weight=0.5, cb_weight=0.5)
        return hybrid, vocab

    def test_instantiation(self) -> None:
        from retrieval.collaborative import ItemBasedCF
        from retrieval.content_based import ContentBasedFilter
        from retrieval.hybrid import HybridRetriever

        cf = ItemBasedCF()
        cb = ContentBasedFilter()
        hr = HybridRetriever(cf, cb, cf_weight=0.6, cb_weight=0.4)
        assert hr.cf_weight == 0.6
        assert hr.cb_weight == 0.4

    def test_negative_weights_raise(self) -> None:
        from retrieval.collaborative import ItemBasedCF
        from retrieval.content_based import ContentBasedFilter
        from retrieval.hybrid import HybridRetriever

        cf = ItemBasedCF()
        cb = ContentBasedFilter()
        with pytest.raises(ValueError):
            HybridRetriever(cf, cb, cf_weight=-0.1, cb_weight=0.5)

    def test_retrieve_returns_correct_count(self, fitted_retriever) -> None:
        hybrid, vocab = fitted_retriever
        seed = [vocab.encode("B00000002"), vocab.encode("B00000003")]
        top_k = 10
        results = hybrid.retrieve(seed, top_k=top_k, vocab=vocab)
        assert len(results) <= top_k

    def test_retrieve_excludes_seed_items(self, fitted_retriever) -> None:
        hybrid, vocab = fitted_retriever
        seed_asins = ["B00000002", "B00000010"]
        seed_ids = [vocab.encode(a) for a in seed_asins]
        results = hybrid.retrieve(seed_ids, top_k=20, vocab=vocab)
        returned = {r[0] for r in results}
        for s in seed_ids:
            assert s not in returned, f"Seed item {s} found in results"

    def test_retrieve_no_duplicates(self, fitted_retriever) -> None:
        hybrid, vocab = fitted_retriever
        seed = [vocab.encode("B00000002")]
        results = hybrid.retrieve(seed, top_k=30, vocab=vocab)
        indices = [r[0] for r in results]
        assert len(indices) == len(set(indices))

    def test_retrieve_sorted_descending(self, fitted_retriever) -> None:
        hybrid, vocab = fitted_retriever
        seed = [vocab.encode("B00000002"), vocab.encode("B00000003")]
        results = hybrid.retrieve(seed, top_k=20, vocab=vocab)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_fusion_is_weighted_sum(self, fitted_retriever) -> None:
        """Verify merged score equals cf_weight*cf_score + cb_weight*cb_score."""
        from retrieval.collaborative import ItemBasedCF
        from retrieval.content_based import ContentBasedFilter
        from retrieval.hybrid import HybridRetriever

        vocab, df = _make_vocab_and_metadata()
        matrix = _make_interaction_matrix()

        cf = ItemBasedCF()
        cf.fit(matrix)
        cb = ContentBasedFilter()
        cb.fit(df)

        cf_w, cb_w = 0.7, 0.3
        hybrid = HybridRetriever(cf, cb, cf_weight=cf_w, cb_weight=cb_w)

        seed = [vocab.encode("B00000002"), vocab.encode("B00000003")]
        top_k = 5

        cf_cands = dict(cf.get_candidates(seed, top_k))
        cb_cands = dict(cb.get_candidates(seed, top_k, vocab))
        merged = hybrid.retrieve(seed, top_k=top_k, vocab=vocab)

        for item_idx, score in merged:
            expected = cf_w * cf_cands.get(item_idx, 0.0) + cb_w * cb_cands.get(
                item_idx, 0.0
            )
            assert abs(score - expected) < 1e-6, (
                f"Item {item_idx}: expected {expected:.6f}, got {score:.6f}"
            )

    def test_retrieve_batch_returns_one_list_per_session(
        self, fitted_retriever
    ) -> None:
        hybrid, vocab = fitted_retriever
        seeds = [
            [vocab.encode("B00000002"), vocab.encode("B00000003")],
            [vocab.encode("B00000010")],
            [vocab.encode("B00000020"), vocab.encode("B00000021")],
        ]
        results = hybrid.retrieve_batch(seeds, top_k=10, vocab=vocab)
        assert len(results) == len(seeds)
        for r in results:
            assert isinstance(r, list)
            assert len(r) <= 10


# ── Legacy stub backward-compatibility tests ──────────────────────────────────


class TestCollaborativeRetriever:
    """Backward-compat tests for the original CollaborativeRetriever stub."""

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
    """Backward-compat tests for the original ContentBasedRetriever stub."""

    def test_instantiation(self, cfg) -> None:
        from retrieval.content_based import ContentBasedRetriever

        cb = ContentBasedRetriever(cfg.retrieval)
        assert cb.top_k == cfg.retrieval.content_top_k


class TestLegacyHybridRetriever:
    """Backward-compat tests for the HybridRetriever stub via cfg."""

    def test_instantiation(self, cfg) -> None:
        from retrieval.collaborative import CollaborativeRetriever
        from retrieval.content_based import ContentBasedRetriever
        from retrieval.hybrid import _LegacyHybridRetriever

        cf = CollaborativeRetriever(cfg.retrieval)
        cb = ContentBasedRetriever(cfg.retrieval)
        hybrid = _LegacyHybridRetriever(cf, cb, cfg.retrieval)
        assert hybrid.final_top_k == cfg.retrieval.final_top_k
