"""
planner.py — Agentic planner: uses Claude LLM to classify user intent and
decide retrieval strategy.

Implemented in Module 04.
"""

from __future__ import annotations

from typing import Any, Dict, List


# Intent taxonomy
INTENTS = [
    "explore",        # user is browsing broadly
    "refine",         # user is narrowing toward a specific need
    "complete",       # user is about to purchase / close on an item
    "repeat",         # user may want a replacement / reorder
]


class AgentPlanner:
    """
    LLM-powered planning module.

    Given a session context, the planner:
      1. Classifies the user intent.
      2. Selects a retrieval strategy (CF-heavy, CB-heavy, or balanced).
      3. Returns a structured plan for the reranker.
    """

    def __init__(self, cfg: dict):
        self.model = cfg["llm_model"]
        self.max_tokens = cfg["llm_max_tokens"]
        self.intent_top_items = cfg["intent_top_items"]
        self._client = None  # Anthropic client, initialised lazily

    # ── Core planning ─────────────────────────────────────────────────────────

    def plan(
        self,
        session_items: List[str],       # human-readable item titles
        candidates: List[Dict[str, Any]],  # hybrid retrieval output
    ) -> Dict[str, Any]:
        """
        Return a plan dict:
        {
          'intent': str,
          'strategy': str,          # 'cf', 'cb', 'balanced'
          'alpha': float,           # retrieval blending weight
          'rationale': str,         # short LLM rationale
        }
        """
        raise NotImplementedError("Implemented in Module 04")

    def _build_prompt(
        self, session_items: List[str], candidates: List[Dict[str, Any]]
    ) -> str:
        raise NotImplementedError("Implemented in Module 04")

    def _call_llm(self, prompt: str) -> str:
        raise NotImplementedError("Implemented in Module 04")
