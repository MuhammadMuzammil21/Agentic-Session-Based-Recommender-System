"""
collaborative.py — Collaborative filtering retrieval using item co-occurrence
or matrix factorisation similarities.

Implemented in Module 03.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np


class CollaborativeRetriever:
    """Retrieve candidates via user–item collaborative signals."""

    def __init__(self, cfg: dict):
        self.top_k = cfg.cf_top_k
        self._similarity_matrix: np.ndarray | None = None

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, sessions: List[dict]) -> "CollaborativeRetriever":
        """Build co-occurrence / similarity matrix from training sessions."""
        raise NotImplementedError("Implemented in Module 03")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self, session_items: List[int], top_k: int | None = None
    ) -> List[Tuple[int, float]]:
        """
        Return (item_idx, score) pairs for a given session.

        Args:
            session_items: ordered list of item indices in the current session
            top_k:         override default top_k
        Returns:
            List of (item_idx, score) sorted descending by score
        """
        raise NotImplementedError("Implemented in Module 03")
