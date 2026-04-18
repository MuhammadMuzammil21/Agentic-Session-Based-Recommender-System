"""
explainer.py — Generate natural-language explanations for recommendations.

Uses Claude LLM to produce user-facing rationale for why each item
was recommended.

Implemented in Module 04.
"""

from __future__ import annotations

from typing import Any, Dict, List


class Explainer:
    """Produce human-readable recommendation explanations via LLM."""

    def __init__(self, cfg: dict):
        self.model = cfg.llm_model
        self.max_tokens = cfg.llm_max_tokens
        self._client = None  # Anthropic client, initialised lazily

    def explain(
        self,
        session_items: List[str],         # recent item titles seen by user
        recommendations: List[Dict[str, Any]],  # reranked candidates
        plan: Dict[str, Any],             # planner output
    ) -> List[Dict[str, Any]]:
        """
        Append an 'explanation' field to each recommendation dict.

        Returns:
            Same list with 'explanation': str added to each item dict.
        """
        raise NotImplementedError("Implemented in Module 04")

    def _build_prompt(
        self,
        session_items: List[str],
        recommendations: List[Dict[str, Any]],
        plan: Dict[str, Any],
    ) -> str:
        raise NotImplementedError("Implemented in Module 04")
