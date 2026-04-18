"""
hybrid.py — Hybrid retrieval: merge CF + content-based candidate lists and
score with a linear combination.

Implemented in Module 03.
"""

from __future__ import annotations

from typing import List, Tuple, Dict, Any
import numpy as np

from retrieval.collaborative import CollaborativeRetriever
from retrieval.content_based import ContentBasedRetriever


class HybridRetriever:
    """Combine collaborative and content-based retrievers."""

    def __init__(
        self,
        cf: CollaborativeRetriever,
        cb: ContentBasedRetriever,
        cfg: dict,
        alpha: float = 0.5,
    ):
        self.cf = cf
        self.cb = cb
        self.final_top_k = cfg["final_top_k"]
        self.alpha = alpha  # weight for CF vs CB scores

    def retrieve(
        self,
        session_items: List[int],
        session_embedding: np.ndarray,
        top_k: int | None = None,
    ) -> List[Tuple[int, float]]:
        """
        Merge CF and CB candidates with linear score fusion.

        Returns:
            List of (item_idx, score) sorted descending, len <= top_k.
        """
        raise NotImplementedError("Implemented in Module 03")

    def retrieve_with_metadata(
        self,
        session_items: List[int],
        session_embedding: np.ndarray,
        item_meta: Dict[int, Any],
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Same as retrieve(), but also attach item metadata to results.

        Returns:
            List of dicts {item_idx, score, title, category, ...}
        """
        raise NotImplementedError("Implemented in Module 03")
