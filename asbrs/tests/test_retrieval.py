"""tests/test_retrieval.py — Unit tests for retrieval/collaborative.py.

Covers ItemBasedCF: fit, get_candidates, save/load.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
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
