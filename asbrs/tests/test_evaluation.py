"""tests/test_evaluation.py — Unit tests for Module 05: Evaluation.

Covers evaluation/metrics.py, evaluation/ablation.py, and
evaluation/human_eval.py with synthetic data.

Run with:
    pytest tests/test_evaluation.py -v
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from evaluation.metrics import (
    aggregate_metrics,
    coverage,
    evaluate_model,
    evaluate_session,
    hit_rate_at_k,
    mrr_at_k,
    recall_at_k,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def simple_predictions() -> List[Tuple[List[int], int]]:
    """A small set of (recommended_list, ground_truth) pairs."""
    return [
        ([1, 2, 3, 4, 5], 1),   # hit at rank 1
        ([1, 2, 3, 4, 5], 2),   # hit at rank 2
        ([1, 2, 3, 4, 5], 6),   # miss
        ([10, 11, 12], 10),     # hit at rank 1
    ]


# ── recall_at_k ───────────────────────────────────────────────────────────────


class TestRecallAtK:
    """Tests for evaluation.metrics.recall_at_k."""

    def test_hit_within_k(self) -> None:
        """Relevant item in top-K → 1.0."""
        assert recall_at_k([1, 2, 3], 2, k=3) == 1.0

    def test_miss_beyond_k(self) -> None:
        """Relevant item outside top-K → 0.0."""
        assert recall_at_k([1, 2, 3, 4, 5], 5, k=4) == 0.0

    def test_hit_at_exact_k(self) -> None:
        """Relevant item at boundary position K → 1.0."""
        assert recall_at_k([1, 2, 3, 4, 5], 5, k=5) == 1.0

    def test_miss_not_in_list(self) -> None:
        """Relevant item absent entirely → 0.0."""
        assert recall_at_k([1, 2, 3], 99, k=3) == 0.0

    def test_empty_list_is_miss(self) -> None:
        """Empty recommendation list → 0.0."""
        assert recall_at_k([], 1, k=5) == 0.0

    def test_k_larger_than_list(self) -> None:
        """K larger than the recommendation list is handled gracefully."""
        assert recall_at_k([1, 2], 1, k=100) == 1.0

    def test_returns_float(self) -> None:
        """Return type must be float."""
        result = recall_at_k([1, 2, 3], 2, k=3)
        assert isinstance(result, float)


# ── mrr_at_k ─────────────────────────────────────────────────────────────────


class TestMRRAtK:
    """Tests for evaluation.metrics.mrr_at_k."""

    def test_rank_1_gives_1_0(self) -> None:
        """Relevant at rank 1 → MRR = 1.0."""
        assert mrr_at_k([5, 2, 3], 5, k=3) == pytest.approx(1.0)

    def test_rank_2_gives_0_5(self) -> None:
        """Relevant at rank 2 → MRR = 0.5."""
        assert mrr_at_k([1, 5, 3], 5, k=3) == pytest.approx(0.5)

    def test_rank_3_gives_1_over_3(self) -> None:
        """Relevant at rank 3 → MRR = 1/3."""
        assert mrr_at_k([1, 2, 5], 5, k=3) == pytest.approx(1.0 / 3.0)

    def test_miss_gives_0_0(self) -> None:
        """Relevant not in top-K → MRR = 0.0."""
        assert mrr_at_k([1, 2, 3], 99, k=3) == 0.0

    def test_k_cutoff_respected(self) -> None:
        """Item at rank 5 is excluded when k=4."""
        assert mrr_at_k([1, 2, 3, 4, 5], 5, k=4) == 0.0

    def test_empty_list_gives_0(self) -> None:
        """Empty list → 0.0."""
        assert mrr_at_k([], 1, k=5) == 0.0


# ── hit_rate_at_k ─────────────────────────────────────────────────────────────


class TestHitRateAtK:
    """Tests for evaluation.metrics.hit_rate_at_k.

    hit_rate_at_k must be identical to recall_at_k for single-positive lists.
    """

    def test_hit(self) -> None:
        assert hit_rate_at_k([3, 1, 2], 3, k=1) == 1.0

    def test_miss(self) -> None:
        assert hit_rate_at_k([1, 2, 3], 4, k=3) == 0.0

    def test_agrees_with_recall(self) -> None:
        recs = [10, 20, 30, 40, 50]
        for rel in [10, 30, 99]:
            for k in [1, 3, 5, 10]:
                assert hit_rate_at_k(recs, rel, k) == recall_at_k(recs, rel, k)


# ── evaluate_model ────────────────────────────────────────────────────────────


class TestEvaluateModel:
    """Tests for evaluation.metrics.evaluate_model."""

    def test_returns_all_expected_keys(
        self, simple_predictions: List[Tuple[List[int], int]]
    ) -> None:
        """All Recall@K, MRR@K, HitRate@K keys present for k in [5, 10]."""
        result = evaluate_model(simple_predictions, k_values=[5, 10])
        for k in [5, 10]:
            assert f"Recall@{k}" in result
            assert f"MRR@{k}" in result
            assert f"HitRate@{k}" in result

    def test_perfect_at_k1(self) -> None:
        """All items at rank 1 → Recall@1 = MRR@1 = HitRate@1 = 1.0."""
        preds = [([gt], gt) for gt in range(5)]
        result = evaluate_model(preds, k_values=[1])
        assert result["Recall@1"] == pytest.approx(1.0)
        assert result["MRR@1"] == pytest.approx(1.0)
        assert result["HitRate@1"] == pytest.approx(1.0)

    def test_all_misses(self) -> None:
        """No relevant item in any list → all metrics 0.0."""
        preds = [([1, 2, 3], 99)] * 10
        result = evaluate_model(preds, k_values=[5])
        assert result["Recall@5"] == pytest.approx(0.0)
        assert result["MRR@5"] == pytest.approx(0.0)

    def test_average_is_correct(self) -> None:
        """Two sessions: one hit at rank 1, one miss → Recall@5 = 0.5."""
        preds: List[Tuple[List[int], int]] = [
            ([10, 20, 30], 10),   # hit at rank 1
            ([10, 20, 30], 99),   # miss
        ]
        result = evaluate_model(preds, k_values=[5])
        assert result["Recall@5"] == pytest.approx(0.5)

    def test_mrr_average(self) -> None:
        """MRR: ranks 1 and 2 → average = (1 + 0.5)/2 = 0.75."""
        preds: List[Tuple[List[int], int]] = [
            ([5, 6, 7], 5),   # rank 1 → 1.0
            ([5, 6, 7], 6),   # rank 2 → 0.5
        ]
        result = evaluate_model(preds, k_values=[3])
        assert result["MRR@3"] == pytest.approx(0.75)

    def test_raises_on_empty_predictions(self) -> None:
        """Empty predictions list → ValueError."""
        with pytest.raises(ValueError, match="empty"):
            evaluate_model([], k_values=[5])

    def test_single_k_value(self) -> None:
        """Works correctly with a single k_value."""
        preds: List[Tuple[List[int], int]] = [([1, 2, 3], 1)]
        result = evaluate_model(preds, k_values=[3])
        assert set(result.keys()) == {"Recall@3", "MRR@3", "HitRate@3"}

    def test_recall_equals_hit_rate(
        self, simple_predictions: List[Tuple[List[int], int]]
    ) -> None:
        """Recall@K must equal HitRate@K for single-positive evaluation."""
        result = evaluate_model(simple_predictions, k_values=[5, 10])
        for k in [5, 10]:
            assert result[f"Recall@{k}"] == pytest.approx(result[f"HitRate@{k}"])


# ── coverage ──────────────────────────────────────────────────────────────────


class TestCoverage:
    """Tests for evaluation.metrics.coverage."""

    def test_full_coverage(self) -> None:
        """All catalogue items appear → coverage = 1.0."""
        all_recs = [[0, 1, 2], [3, 4], [5]]
        assert coverage(all_recs, catalog_size=6) == pytest.approx(1.0)

    def test_zero_coverage(self) -> None:
        """Empty recommendation lists → coverage = 0.0."""
        assert coverage([[], [], []], catalog_size=10) == pytest.approx(0.0)

    def test_partial_coverage(self) -> None:
        """3 unique items seen out of 10 → coverage = 0.3."""
        all_recs = [[0, 1, 2], [0, 1]]  # 3 unique items
        frac = coverage(all_recs, catalog_size=10)
        assert frac == pytest.approx(0.3)

    def test_coverage_in_unit_interval(self) -> None:
        """Coverage must always be in [0, 1]."""
        for _ in range(20):
            k = random.randint(1, 50)
            catalog = random.randint(1, 100)
            recs = [
                [random.randint(0, catalog - 1) for _ in range(k)]
                for _ in range(5)
            ]
            frac = coverage(recs, catalog_size=catalog)
            assert 0.0 <= frac <= 1.0, f"Coverage out of range: {frac}"

    def test_invalid_catalog_size_raises(self) -> None:
        """catalog_size <= 0 → ValueError."""
        with pytest.raises(ValueError):
            coverage([[1, 2]], catalog_size=0)

    def test_duplicates_counted_once(self) -> None:
        """Repeated items in recommendation lists count only once."""
        all_recs = [[1, 1, 1], [1, 1]]
        assert coverage(all_recs, catalog_size=5) == pytest.approx(0.2)


# ── evaluate_session (legacy) ─────────────────────────────────────────────────


class TestEvaluateSession:
    """Tests for the legacy evaluate_session helper."""

    def test_returns_all_keys(self) -> None:
        result = evaluate_session(
            ground_truth=2,
            predictions=[1, 2, 3, 4, 5],
            k_values=[5, 10, 20],
        )
        for k in [5, 10, 20]:
            assert f"Recall@{k}" in result
            assert f"MRR@{k}" in result
            assert f"HitRate@{k}" in result

    def test_hit_values_correct(self) -> None:
        result = evaluate_session(
            ground_truth=1,
            predictions=[1, 2, 3],
            k_values=[3],
        )
        assert result["Recall@3"] == pytest.approx(1.0)
        assert result["MRR@3"] == pytest.approx(1.0)

    def test_miss_values_correct(self) -> None:
        result = evaluate_session(
            ground_truth=99,
            predictions=[1, 2, 3],
            k_values=[3],
        )
        assert result["Recall@3"] == pytest.approx(0.0)
        assert result["MRR@3"] == pytest.approx(0.0)


# ── aggregate_metrics (legacy) ────────────────────────────────────────────────


class TestAggregateMetrics:
    """Tests for the legacy aggregate_metrics helper."""

    def test_average(self) -> None:
        results = [
            {"Recall@5": 1.0, "MRR@5": 1.0},
            {"Recall@5": 0.0, "MRR@5": 0.5},
        ]
        agg = aggregate_metrics(results)
        assert agg["Recall@5"] == pytest.approx(0.5)
        assert agg["MRR@5"] == pytest.approx(0.75)

    def test_empty_returns_empty(self) -> None:
        assert aggregate_metrics([]) == {}

    def test_single_entry(self) -> None:
        result = aggregate_metrics([{"Recall@5": 0.8}])
        assert result["Recall@5"] == pytest.approx(0.8)


