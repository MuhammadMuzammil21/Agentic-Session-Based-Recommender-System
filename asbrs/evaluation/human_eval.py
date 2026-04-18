"""
human_eval.py — Human evaluation utilities.

Generates evaluation sheets and collects judgments on:
  - Relevance (1–5)
  - Diversity (1–5)
  - Explanation quality (1–5)

Implemented in Module 05.
"""

from __future__ import annotations

from typing import Any, Dict, List


class HumanEvalBuilder:
    """Build human evaluation tasks from model outputs."""

    def __init__(self, num_sessions: int = 50, num_items_per_session: int = 5):
        self.num_sessions = num_sessions
        self.num_items_per_session = num_items_per_session

    def build_eval_sheet(
        self,
        sessions: List[dict],
        recommendations: List[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Sample sessions and produce an evaluation sheet.

        Returns a list of task dicts ready for manual scoring or
        export to a spreadsheet / annotation platform.
        """
        raise NotImplementedError("Implemented in Module 05")

    def aggregate_judgments(
        self, judgments: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute mean scores across all judged sessions."""
        raise NotImplementedError("Implemented in Module 05")
