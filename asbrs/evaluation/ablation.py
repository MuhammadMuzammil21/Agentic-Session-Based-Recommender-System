"""
ablation.py — Ablation study runner.

Compares model variants (e.g., no-attention, no-CF, no-agent) against
the full ASBRS system.

Implemented in Module 05.
"""

from __future__ import annotations

from typing import Any, Dict, List


ABLATION_VARIANTS = [
    "full",              # complete ASBRS
    "no_attention",      # GRU only, no self-attention
    "no_cf",             # content-based only retrieval
    "no_cb",             # collaborative only retrieval
    "no_agent",          # no LLM planner (fixed alpha=0.5)
    "no_rerank",         # skip reranker, use raw retrieval order
]


class AblationRunner:
    """Run ablation experiments and collect comparative metric tables."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.k_values: List[int] = cfg["k_values"]

    def run(
        self,
        test_sessions: List[dict],
        variant: str = "full",
    ) -> Dict[str, float]:
        """Run a single ablation variant and return aggregated metrics."""
        raise NotImplementedError("Implemented in Module 05")

    def run_all(self, test_sessions: List[dict]) -> Dict[str, Dict[str, float]]:
        """Run all variants and return {variant_name: metrics_dict}."""
        raise NotImplementedError("Implemented in Module 05")
