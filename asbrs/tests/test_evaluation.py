"""
test_evaluation.py — Unit tests for evaluation metrics.
"""

import pytest
from evaluation.metrics import recall_at_k, mrr_at_k, hit_rate_at_k, evaluate_session, aggregate_metrics


class TestRecallAtK:
    def test_hit(self):
        assert recall_at_k(ground_truth=3, predictions=[1, 2, 3, 4, 5], k=5) == 1.0

    def test_miss(self):
        assert recall_at_k(ground_truth=6, predictions=[1, 2, 3, 4, 5], k=5) == 0.0

    def test_exactly_at_k(self):
        assert recall_at_k(ground_truth=5, predictions=[1, 2, 3, 4, 5], k=5) == 1.0

    def test_beyond_k(self):
        assert recall_at_k(ground_truth=5, predictions=[1, 2, 3, 4, 5], k=4) == 0.0


class TestMRRAtK:
    def test_first_position(self):
        assert mrr_at_k(ground_truth=1, predictions=[1, 2, 3], k=3) == pytest.approx(1.0)

    def test_second_position(self):
        assert mrr_at_k(ground_truth=2, predictions=[1, 2, 3], k=3) == pytest.approx(0.5)

    def test_third_position(self):
        assert mrr_at_k(ground_truth=3, predictions=[1, 2, 3], k=3) == pytest.approx(1 / 3)

    def test_miss(self):
        assert mrr_at_k(ground_truth=99, predictions=[1, 2, 3], k=3) == 0.0


class TestEvaluateSession:
    def test_returns_all_keys(self):
        result = evaluate_session(
            ground_truth=2,
            predictions=[1, 2, 3, 4, 5],
            k_values=[5, 10, 20],
        )
        for k in [5, 10, 20]:
            assert f"Recall@{k}" in result
            assert f"MRR@{k}" in result
            assert f"HitRate@{k}" in result


class TestAggregateMetrics:
    def test_average(self):
        results = [
            {"Recall@5": 1.0, "MRR@5": 1.0},
            {"Recall@5": 0.0, "MRR@5": 0.5},
        ]
        agg = aggregate_metrics(results)
        assert agg["Recall@5"] == pytest.approx(0.5)
        assert agg["MRR@5"] == pytest.approx(0.75)

    def test_empty_returns_empty(self):
        assert aggregate_metrics([]) == {}
