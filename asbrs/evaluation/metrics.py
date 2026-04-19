"""evaluation/metrics.py — Ranking metrics for session-based recommendation.

Pure-function module: no classes, no side-effects.

Metrics
-------
- Recall@K      : 1.0 if relevant item is in top-K predictions.
- MRR@K         : 1/rank if relevant item is in top-K, else 0.0.
- HitRate@K     : Alias for Recall@K for a single-positive scenario.
- evaluate_model : Aggregate all metrics over a list of prediction pairs.
- coverage      : Fraction of the item catalogue seen in any recommendation.

All item IDs are 0-indexed integers throughout.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


# ── Per-session helpers ───────────────────────────────────────────────────────


def recall_at_k(recommended: List[int], relevant: int, k: int) -> float:
    """Return 1.0 if *relevant* appears in *recommended[:k]*, else 0.0.

    Args:
        recommended: Ordered list of recommended item IDs (highest score first).
        relevant:    The single ground-truth item ID.
        k:           Cut-off rank.

    Returns:
        1.0 on hit, 0.0 on miss.
    """
    return float(relevant in recommended[:k])


def mrr_at_k(recommended: List[int], relevant: int, k: int) -> float:
    """Return the reciprocal rank if *relevant* is in *recommended[:k]*.

    Args:
        recommended: Ordered list of recommended item IDs (highest score first).
        relevant:    The single ground-truth item ID.
        k:           Cut-off rank.

    Returns:
        1/rank (1-based) on hit, 0.0 on miss.
    """
    top_k = recommended[:k]
    if relevant in top_k:
        rank = top_k.index(relevant) + 1  # 1-based rank
        return 1.0 / rank
    return 0.0


def hit_rate_at_k(recommended: List[int], relevant: int, k: int) -> float:
    """Alias for recall_at_k for a single positive item.

    Args:
        recommended: Ordered list of recommended item IDs (highest score first).
        relevant:    The single ground-truth item ID.
        k:           Cut-off rank.

    Returns:
        1.0 on hit, 0.0 on miss.
    """
    return recall_at_k(recommended, relevant, k)


# ── Batch evaluation ──────────────────────────────────────────────────────────


def evaluate_model(
    predictions: List[Tuple[List[int], int]],
    k_values: List[int],
) -> Dict[str, float]:
    """Compute averaged ranking metrics over a list of test predictions.

    Args:
        predictions: Each element is a tuple of
            (recommended_item_ids, ground_truth_item_id).
            ``recommended_item_ids`` must be sorted by descending score.
        k_values:    Cut-off ranks to evaluate, e.g. ``[5, 10, 20]``.

    Returns:
        Dictionary with keys ``"Recall@K"``, ``"MRR@K"``, ``"HitRate@K"``
        for every K in *k_values*, containing the macro-average over all
        prediction pairs.

    Raises:
        ValueError: If *predictions* is empty.
    """
    if not predictions:
        raise ValueError("predictions list must not be empty.")

    # Accumulators: key → running sum
    totals: Dict[str, float] = {}
    for k in k_values:
        totals[f"Recall@{k}"] = 0.0
        totals[f"MRR@{k}"] = 0.0
        totals[f"HitRate@{k}"] = 0.0

    n = len(predictions)
    for recommended, relevant in predictions:
        for k in k_values:
            totals[f"Recall@{k}"] += recall_at_k(recommended, relevant, k)
            totals[f"MRR@{k}"] += mrr_at_k(recommended, relevant, k)
            totals[f"HitRate@{k}"] += hit_rate_at_k(recommended, relevant, k)

    averaged: Dict[str, float] = {key: val / n for key, val in totals.items()}

    logger.info(
        "evaluate_model: %d samples, k_values=%s → %s",
        n,
        k_values,
        {k: f"{v:.4f}" for k, v in averaged.items()},
    )
    return averaged


def coverage(
    all_recommendations: List[List[int]],
    catalog_size: int,
) -> float:
    """Compute catalogue coverage of the recommendation lists.

    Args:
        all_recommendations: One list of recommended item IDs per user/session.
        catalog_size:        Total number of unique items in the catalogue.

    Returns:
        Fraction of catalogue items that appear in at least one list.
        Always in ``[0.0, 1.0]``.

    Raises:
        ValueError: If *catalog_size* is not a positive integer.
    """
    if catalog_size <= 0:
        raise ValueError(f"catalog_size must be > 0, got {catalog_size}.")

    seen: set[int] = set()
    for recs in all_recommendations:
        seen.update(recs)

    frac = len(seen) / catalog_size
    logger.info(
        "coverage: %d unique items seen / %d catalogue = %.4f",
        len(seen),
        catalog_size,
        frac,
    )
    return frac


# ── Legacy helpers (kept for backwards compatibility with existing tests) ──────


def recall_at_k_legacy(
    ground_truth: int,
    predictions: List[int],
    k: int,
) -> float:
    """Legacy signature — delegates to recall_at_k."""
    return recall_at_k(predictions, ground_truth, k)


def mrr_at_k_legacy(
    ground_truth: int,
    predictions: List[int],
    k: int,
) -> float:
    """Legacy signature — delegates to mrr_at_k."""
    return mrr_at_k(predictions, ground_truth, k)


def hit_rate_at_k_legacy(
    ground_truth: int,
    predictions: List[int],
    k: int,
) -> float:
    """Legacy signature — delegates to hit_rate_at_k."""
    return hit_rate_at_k(predictions, ground_truth, k)


def evaluate_session(
    ground_truth: int,
    predictions: List[int],
    k_values: List[int],
) -> Dict[str, float]:
    """Compute all metrics for a single session at multiple K values.

    Legacy helper kept for backwards compatibility with existing tests.

    Args:
        ground_truth: Single relevant item ID.
        predictions:  Ordered recommended item IDs.
        k_values:     List of cut-off ranks.

    Returns:
        Dictionary with Recall@K, MRR@K, HitRate@K for each K.
    """
    results: Dict[str, float] = {}
    for k in k_values:
        results[f"Recall@{k}"] = recall_at_k(predictions, ground_truth, k)
        results[f"MRR@{k}"] = mrr_at_k(predictions, ground_truth, k)
        results[f"HitRate@{k}"] = hit_rate_at_k(predictions, ground_truth, k)
    return results


def aggregate_metrics(session_results: List[Dict[str, float]]) -> Dict[str, float]:
    """Average per-session metric dicts over the test set.

    Args:
        session_results: List of per-session metric dictionaries.

    Returns:
        Macro-average of each metric key, or ``{}`` if the list is empty.
    """
    if not session_results:
        return {}
    keys = session_results[0].keys()
    return {
        k: sum(r[k] for r in session_results) / len(session_results)
        for k in keys
    }
