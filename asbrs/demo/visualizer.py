"""
visualizer.py — Utilities to build chart data for the demo frontend.

Converts model outputs into JSON-serialisable chart payloads used by
the index.html template.

Implemented in Module 06.
"""

from __future__ import annotations

from typing import Any, Dict, List


def build_score_chart(recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a Chart.js-compatible bar chart payload for recommendation scores.

    Args:
        recommendations: list of {title, final_score, cf_score, cb_score}
    Returns:
        {labels: [...], datasets: [...]}
    """
    raise NotImplementedError("Implemented in Module 06")


def build_session_timeline(session_items: List[str]) -> List[Dict[str, str]]:
    """
    Build a simple timeline payload for the session history panel.

    Args:
        session_items: ordered list of item titles
    Returns:
        List of {step, title} dicts
    """
    return [{"step": i + 1, "title": title} for i, title in enumerate(session_items)]
