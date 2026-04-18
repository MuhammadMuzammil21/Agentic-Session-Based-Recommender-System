"""
content_based.py — Content-based retrieval using item metadata features
(title, category, price, description) and session-encoder embeddings.

Implemented in Module 03.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np


class ContentBasedRetriever:
    """Retrieve candidates via item content similarity of session embedding."""

    def __init__(self, cfg: dict):
        self.top_k = cfg["content_top_k"]
        self._item_matrix: np.ndarray | None = None  # (num_items, dim)

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, item_embeddings: np.ndarray) -> "ContentBasedRetriever":
        """Index the item embedding matrix for fast similarity search."""
        raise NotImplementedError("Implemented in Module 03")

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(
        self, session_embedding: np.ndarray, top_k: int | None = None
    ) -> List[Tuple[int, float]]:
        """
        Return (item_idx, cosine_score) pairs for a session embedding.

        Args:
            session_embedding: 1-D float array of shape (hidden_dim,)
            top_k:             override default top_k
        Returns:
            List of (item_idx, score) sorted descending by score
        """
        raise NotImplementedError("Implemented in Module 03")
