"""tests/test_agent.py — Unit tests for the agentic planning components.

These test the stub interfaces only; full logic is implemented in Module 04.
"""

from __future__ import annotations

import pytest


class TestAgentPlanner:
    def test_instantiation(self, cfg) -> None:
        from agent.planner import AgentPlanner, INTENTS
        planner = AgentPlanner(cfg.agent)
        assert planner.model == cfg.agent.llm_model
        assert len(INTENTS) == 4

    def test_plan_raises_not_implemented(self, cfg) -> None:
        from agent.planner import AgentPlanner
        planner = AgentPlanner(cfg.agent)
        with pytest.raises(NotImplementedError):
            planner.plan(session_items=["Item A"], candidates=[])


class TestReranker:
    def test_instantiation(self, cfg) -> None:
        from agent.reranker import Reranker
        r = Reranker(cfg.retrieval)
        assert r.final_top_k == cfg.retrieval.final_top_k


class TestExplainer:
    def test_instantiation(self, cfg) -> None:
        from agent.explainer import Explainer
        e = Explainer(cfg.agent)
        assert e.model == cfg.agent.llm_model
        assert e.max_tokens == cfg.agent.llm_max_tokens
