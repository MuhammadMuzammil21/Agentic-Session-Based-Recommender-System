"""agent/explainer.py — Template-based recommendation explanation generator.

Produces human-readable explanations for each recommended item WITHOUT
calling the LLM, using a structured template and the session attention weights.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from agent.interfaces import IntentResult, RankedItem, RecommendationOutput

logger = logging.getLogger(__name__)

# Number of top session items to include in the attention heatmap.
_HEATMAP_TOPN = 5


class RecommendationExplainer:
    """Generate natural-language explanations for ranked recommendations.

    Explanations are fully template-driven (no LLM calls) to keep latency
    low during inference. The attention heatmap is also included for
    visualisation in the demo UI.
    """

    # ── Single-item explanation ───────────────────────────────────────────────

    def generate_explanation(
        self,
        ranked_item: RankedItem,
        session_items: List[str],
        attn_weights: List[float],
        intent: IntentResult,
    ) -> str:
        """Build a template explanation string for one recommendation.

        Template:
            "Recommended because your session shows strong interest in
             <top_item> (<top_weight:.0%>) and <second_item> (<second_weight:.0%>),
             suggesting you are <intent_summary>."

        If fewer than two session items are present, the template degrades
        gracefully.

        Args:
            ranked_item:    The candidate being explained.
            session_items:  Decoded item titles from the user's session
                            (most-attended first).
            attn_weights:   Attention weights corresponding to session_items.
            intent:         IntentResult from IntentPlanner.

        Returns:
            A non-empty explanation string.
        """
        if not session_items:
            return (
                f"Recommended based on your browsing activity, "
                f"suggesting you are {intent.intent_summary}."
            )

        top_item = session_items[0]
        top_weight = attn_weights[0] if attn_weights else 0.0

        if len(session_items) >= 2:
            second_item = session_items[1]
            second_weight = attn_weights[1] if len(attn_weights) > 1 else 0.0
            explanation = (
                f"Recommended because your session shows strong interest in "
                f"{top_item} ({top_weight:.0%}) and "
                f"{second_item} ({second_weight:.0%}), "
                f"suggesting you are {intent.intent_summary}."
            )
        else:
            explanation = (
                f"Recommended because your session shows strong interest in "
                f"{top_item} ({top_weight:.0%}), "
                f"suggesting you are {intent.intent_summary}."
            )

        logger.debug(
            "Explanation generated for item_id=%d: %s",
            ranked_item.item_id,
            explanation[:80],
        )
        return explanation

    # ── Batch formatting ──────────────────────────────────────────────────────

    def format_recommendations(
        self,
        ranked_items: List[RankedItem],
        session_items: List[str],
        attn_weights: List[float],
        intent: IntentResult,
    ) -> List[RecommendationOutput]:
        """Produce final RecommendationOutput objects for a ranked list.

        Each output includes:
        - Rank (1-based)
        - Item id and title
        - Final score
        - Natural-language explanation
        - Attention heatmap (top-5 session items → weight)

        Args:
            ranked_items:  Sorted list of RankedItem from IntentReranker.
            session_items: Session item titles (most-attended first).
            attn_weights:  Corresponding attention weights.
            intent:        IntentResult used for explanation text.

        Returns:
            List of RecommendationOutput, one per ranked_item.
        """
        heatmap = self._build_heatmap(session_items, attn_weights)
        outputs: List[RecommendationOutput] = []

        for rank, item in enumerate(ranked_items, start=1):
            explanation = self.generate_explanation(
                item, session_items, attn_weights, intent
            )
            outputs.append(
                RecommendationOutput(
                    rank=rank,
                    item_id=item.item_id,
                    item_title=item.item_title,
                    final_score=item.final_score,
                    explanation=explanation,
                    attention_heatmap=dict(heatmap),
                )
            )

        logger.info(
            "format_recommendations: produced %d outputs", len(outputs)
        )
        return outputs

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _build_heatmap(
        session_items: List[str],
        attn_weights: List[float],
    ) -> Dict[str, float]:
        """Build an attention heatmap dict for the top-N session items.

        Args:
            session_items: Item title strings.
            attn_weights:  Corresponding weights.

        Returns:
            Dict mapping item_title → weight (up to _HEATMAP_TOPN entries).
        """
        pairs = list(zip(session_items, attn_weights))
        top_pairs = pairs[:_HEATMAP_TOPN]
        return {title: float(weight) for title, weight in top_pairs}


# ── Legacy stub (kept for backward compatibility with existing tests) ──────────


class Explainer:
    """Produce human-readable recommendation explanations via LLM. (legacy stub)"""

    def __init__(self, cfg: object) -> None:
        self.model = cfg.llm_model
        self.max_tokens = cfg.llm_max_tokens
        self._client = None

    def explain(
        self,
        session_items: List[str],
        recommendations: List[dict],
        plan: dict,
    ) -> List[dict]:
        """Append an 'explanation' field to each recommendation dict."""
        raise NotImplementedError("Implemented in Module 04")

    def _build_prompt(
        self,
        session_items: List[str],
        recommendations: List[dict],
        plan: dict,
    ) -> str:
        raise NotImplementedError("Implemented in Module 04")
