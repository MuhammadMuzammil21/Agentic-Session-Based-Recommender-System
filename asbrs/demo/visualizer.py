"""demo/visualizer.py — Utilities to format model outputs for the demo frontend.

Converts model outputs into JSON-serialisable chart payloads used by
the index.html template.
"""

from __future__ import annotations

from typing import Any, Dict, List

from agent.interfaces import RecommendationOutput


class AttentionVisualizer:
    """Prepares attention data and recommendations for JSON serialization."""

    @staticmethod
    def heatmap_data(session_items: List[str], attn_weights: List[float]) -> Dict[str, Any]:
        """Convert raw attention weights into a normalized heatmap payload.

        Args:
            session_items: Ordered list of item titles in the session.
            attn_weights: List of raw attention weights for each item.

        Returns:
            Dictionary with 'labels' and 'values' (normalized to sum to 1.0).
        """
        # Ensure lengths match up to available weights
        n = min(len(session_items), len(attn_weights))
        items = session_items[-n:] if n > 0 else []
        weights = attn_weights[-n:] if n > 0 else []

        total = sum(weights)
        if total > 0:
            norm_weights = [w / total for w in weights]
        else:
            norm_weights = [0.0] * n

        return {
            "labels": items,
            "values": norm_weights
        }

    @staticmethod
    def recommendation_cards(outputs: List[RecommendationOutput]) -> List[Dict[str, Any]]:
        """Format recommendation outputs for the frontend template.

        Args:
            outputs: List of RecommendationOutput objects.

        Returns:
            List of dictionaries for rendering in the DOM.
        """
        cards = []
        for out in outputs:
            title = out.item_title
            if len(title) > 60:
                title = title[:57] + "..."
            
            cards.append({
                "rank": out.rank,
                "title": title,
                # Probability within top-K: render as percentage.
                "score": f"{out.final_score * 100:.1f}%",
                "explanation": out.explanation,
                "intent_badge": "intent_match"
            })
        return cards
