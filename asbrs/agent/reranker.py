"""
reranker.py — Re-rank candidate items using planner intent + session signals.

Implemented in Module 04.
"""

from __future__ import annotations

from typing import Any, Dict, List


class Reranker:
    """
    Re-rank retrieval candidates using:
      - Planner intent / alpha weighting
      - Diversity penalty (MMR-style)
      - Optional score normalisation
    """

    def __init__(self, cfg: dict):
        self.final_top_k = cfg["final_top_k"]

    def rerank(
        self,
        candidates: List[Dict[str, Any]],
        plan: Dict[str, Any],
        session_embedding: "np.ndarray",  # noqa: F821
        top_k: int | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Args:
            candidates: list of {item_idx, cf_score, cb_score, ...}
            plan:       output from AgentPlanner.plan()
            session_embedding: (hidden_dim,) float array
            top_k:      number of final items to return
        Returns:
            Sorted list of candidate dicts with 'final_score' added.
        """
        raise NotImplementedError("Implemented in Module 04")
