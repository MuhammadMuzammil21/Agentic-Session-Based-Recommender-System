"""
metrics.py — Ranking metrics for session-based recommendation evaluation.

Metrics:
  - Recall@K
  - MRR@K  (Mean Reciprocal Rank)
  - HitRate@K

All functions expect 0-indexed item IDs.
"""

from __future__ import annotations

from typing import List


def recall_at_k(
    ground_truth: int,
    predictions: List[int],
    k: int,
) -> float:
    """Return 1.0 if ground_truth is in predictions[:k], else 0.0."""
    return float(ground_truth in predictions[:k])


def mrr_at_k(
    ground_truth: int,
    predictions: List[int],
    k: int,
) -> float:
    """Return 1/rank if ground_truth is in top-k, else 0.0."""
    preds_k = predictions[:k]
    if ground_truth in preds_k:
        rank = preds_k.index(ground_truth) + 1
        return 1.0 / rank
    return 0.0


def hit_rate_at_k(
    ground_truth: int,
    predictions: List[int],
    k: int,
) -> float:
    """Alias for recall_at_k for single positive — included for clarity."""
    return recall_at_k(ground_truth, predictions, k)


def evaluate_session(
    ground_truth: int,
    predictions: List[int],
    k_values: List[int],
) -> dict:
    """
    Compute all metrics for a single session at multiple K values.

    Returns:
        {
          'Recall@5': float, 'MRR@5': float, 'HitRate@5': float,
          'Recall@10': ..., ...
        }
    """
    results = {}
    for k in k_values:
        results[f"Recall@{k}"] = recall_at_k(ground_truth, predictions, k)
        results[f"MRR@{k}"] = mrr_at_k(ground_truth, predictions, k)
        results[f"HitRate@{k}"] = hit_rate_at_k(ground_truth, predictions, k)
    return results


def aggregate_metrics(session_results: List[dict]) -> dict:
    """Average per-session metrics over the test set."""
    if not session_results:
        return {}
    keys = session_results[0].keys()
    return {k: sum(r[k] for r in session_results) / len(session_results) for k in keys}
