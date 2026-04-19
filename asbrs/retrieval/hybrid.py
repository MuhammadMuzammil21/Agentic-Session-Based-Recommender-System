"""retrieval/hybrid.py — Hybrid CF + content-based retrieval with score fusion.

Combines ItemBasedCF and ContentBasedFilter candidates using a weighted
linear combination and returns the top-K items by merged score.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from data.vocab import Vocabulary
from retrieval.collaborative import ItemBasedCF
from retrieval.content_based import ContentBasedFilter

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Merge collaborative and content-based candidates with score fusion.

    Scores are combined as:
        score = cf_weight * cf_score + cb_weight * cb_score

    Items appearing in only one source receive a zero contribution from
    the other source.

    Attributes:
        cf:         Fitted ItemBasedCF instance.
        cb:         Fitted ContentBasedFilter instance.
        cf_weight:  Weight for collaborative-filter scores.
        cb_weight:  Weight for content-based scores.
    """

    def __init__(
        self,
        cf: ItemBasedCF,
        cb: ContentBasedFilter,
        cf_weight: float = 0.5,
        cb_weight: float = 0.5,
    ) -> None:
        """Initialise the hybrid retriever.

        Args:
            cf:        Fitted ItemBasedCF.
            cb:        Fitted ContentBasedFilter.
            cf_weight: Weight applied to CF scores (default 0.5).
            cb_weight: Weight applied to CB scores (default 0.5).

        Raises:
            ValueError: If weights are negative.
        """
        if cf_weight < 0 or cb_weight < 0:
            raise ValueError("cf_weight and cb_weight must be non-negative.")
        self.cf = cf
        self.cb = cb
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight

    # ── Single-session retrieval ───────────────────────────────────────────────

    def retrieve(
        self,
        item_ids: List[int],
        top_k: int,
        vocab: Vocabulary,
    ) -> List[Tuple[int, float]]:
        """Retrieve and merge top-K candidates from CF and CB.

        Each source is queried for *top_k* candidates (to ensure sufficient
        diversity), scores are fused, and the final top-K is returned.

        Args:
            item_ids: Integer item indices from the current session.
            top_k:    Number of final candidates to return.
            vocab:    Vocabulary for CB decoding.

        Returns:
            List of (item_idx, merged_score) sorted descending by merged score.
        """
        cf_candidates = self.cf.get_candidates(item_ids, top_k)
        cb_candidates = self.cb.get_candidates(item_ids, top_k, vocab)

        # Accumulate weighted scores in a dict.
        merged: Dict[int, float] = {}

        for item_idx, score in cf_candidates:
            merged[item_idx] = merged.get(item_idx, 0.0) + self.cf_weight * score

        for item_idx, score in cb_candidates:
            merged[item_idx] = merged.get(item_idx, 0.0) + self.cb_weight * score

        # Exclude seed items.
        seed_set = set(item_ids)
        pool = {k: v for k, v in merged.items() if k not in seed_set}

        logger.debug(
            "HybridRetriever: CF=%d, CB=%d, pool=%d, top_k=%d",
            len(cf_candidates),
            len(cb_candidates),
            len(pool),
            top_k,
        )

        sorted_items = sorted(pool.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]

    # ── Batch retrieval ───────────────────────────────────────────────────────

    def retrieve_batch(
        self,
        sessions: List[List[int]],
        top_k: int,
        vocab: Vocabulary,
    ) -> List[List[Tuple[int, float]]]:
        """Retrieve candidates for multiple sessions in sequence.

        Args:
            sessions: List of sessions; each session is a list of item indices.
            top_k:    Number of candidates per session.
            vocab:    Vocabulary for CB decoding.

        Returns:
            List of candidate lists, one per input session.
        """
        results: List[List[Tuple[int, float]]] = []
        for session in tqdm(sessions, desc="HybridRetriever.retrieve_batch"):
            results.append(self.retrieve(session, top_k, vocab))
        return results


# ── Legacy stub (kept for backward compatibility) ─────────────────────────────

import numpy as np  # noqa: E402
from retrieval.collaborative import CollaborativeRetriever  # noqa: E402
from retrieval.content_based import ContentBasedRetriever  # noqa: E402


class _LegacyHybridRetriever:
    """Combine collaborative and content-based retrievers. (stub)"""

    def __init__(
        self,
        cf: CollaborativeRetriever,
        cb: ContentBasedRetriever,
        cfg: object,
        alpha: float = 0.5,
    ) -> None:
        self.cf = cf
        self.cb = cb
        self.final_top_k = cfg.final_top_k
        self.alpha = alpha

    def retrieve(
        self,
        session_items: List[int],
        session_embedding: np.ndarray,
        top_k: int | None = None,
    ) -> List[Tuple[int, float]]:
        """Merge CF and CB candidates with linear score fusion."""
        raise NotImplementedError("Implemented in Module 03")

    def retrieve_with_metadata(
        self,
        session_items: List[int],
        session_embedding: np.ndarray,
        item_meta: Dict[int, Any],
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Same as retrieve(), but also attach item metadata to results."""
        raise NotImplementedError("Implemented in Module 03")
