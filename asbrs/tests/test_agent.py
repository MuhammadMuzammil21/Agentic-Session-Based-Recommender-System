"""tests/test_agent.py — Unit tests for Module 04: Agentic Planner.

Covers:
  - IntentPlanner: valid JSON response, malformed response, cache deduplication
  - IntentReranker: final_score formula, fit/rerank correctness
  - RecommendationExplainer: explanation content, format_recommendations output
  - Legacy stub backward-compatibility tests (AgentPlanner, Reranker, Explainer)
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── Fixtures & helpers ────────────────────────────────────────────────────────


def _make_vocab_and_metadata(n: int = 10):
    """Return (Vocabulary, DataFrame) with n synthetic items."""
    from data.vocab import Vocabulary

    asins = [f"B{i:08d}" for i in range(n)]
    titles = [f"Electronics Product {i}" for i in range(n)]
    df = pd.DataFrame(
        {
            "item_id": asins,
            "title": titles,
            "description": [f"Great product {i}" for i in range(n)],
            "category": ["Electronics"] * n,
            "price": [float(i * 10) for i in range(n)],
        }
    )
    vocab = Vocabulary()
    vocab.build(asins)
    return vocab, df


def _make_intent_result(
    intent: str = "looking for audio accessories",
    confidence: float = 0.85,
) -> "IntentResult":
    from agent.interfaces import IntentResult

    return IntentResult(
        intent_summary=intent,
        top_items=["Wireless Headphones", "USB-C Cable"],
        confidence=confidence,
        keywords=["audio", "wireless", "headphones"],
    )


def _make_api_response(payload: dict) -> MagicMock:
    """Return a MagicMock whose .text is the JSON-encoded payload."""
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(payload)
    return mock_resp


@contextmanager
def _patch_gemini(response_text: str = "{}"):
    """Context manager: patch genai.Client so models.generate_content returns
    a MagicMock with .text = response_text.  Yields the generate_content mock.
    """
    with patch("agent.planner.genai.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance
        mock_gc = MagicMock()
        mock_gc.text = response_text
        mock_instance.models.generate_content.return_value = mock_gc
        yield mock_instance.models.generate_content


# ── IntentPlanner ─────────────────────────────────────────────────────────────


class TestIntentPlanner:
    """Tests for agent.planner.IntentPlanner."""

    _VALID_PAYLOAD = {
        "intent": "looking for wireless audio accessories",
        "keywords": ["wireless", "audio", "headphones"],
        "confidence": 0.9,
    }

    def test_valid_json_response_returns_intent_result(self) -> None:
        """IntentPlanner returns correct IntentResult on well-formed JSON."""
        from agent.planner import IntentPlanner

        with _patch_gemini(json.dumps(self._VALID_PAYLOAD)):
            planner = IntentPlanner(llm_model="gemini-2.5-flash", max_tokens=200)
            result = planner.infer_intent(
                session_items=["Wireless Headphones", "USB-C Cable", "Phone Stand"],
                attention_weights=[0.6, 0.3, 0.1],
            )

        assert result.intent_summary == "looking for wireless audio accessories"
        assert result.keywords == ["wireless", "audio", "headphones"]
        assert abs(result.confidence - 0.9) < 1e-6
        assert isinstance(result.top_items, list)

    def test_malformed_response_returns_default_no_crash(self) -> None:
        """IntentPlanner does NOT raise on malformed JSON; returns default."""
        from agent.planner import IntentPlanner

        with _patch_gemini("This is not JSON at all!!!"):
            planner = IntentPlanner(llm_model="gemini-2.5-flash", max_tokens=200)
            result = planner.infer_intent(
                session_items=["Product A"], attention_weights=[1.0]
            )

        assert isinstance(result.intent_summary, str)
        assert len(result.intent_summary) > 0
        assert result.confidence == 0.0  # default

    def test_api_exception_returns_default_no_crash(self) -> None:
        """IntentPlanner catches API exceptions and returns the default result."""
        from agent.planner import IntentPlanner

        with patch("agent.planner.genai.Client") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.models.generate_content.side_effect = RuntimeError("network error")

            planner = IntentPlanner(llm_model="gemini-2.5-flash", max_tokens=200)
            result = planner.infer_intent(
                session_items=["Product A"], attention_weights=[1.0]
            )

        assert isinstance(result.intent_summary, str)
        assert result.confidence == 0.0

    def test_cache_prevents_duplicate_api_calls(self) -> None:
        """Identical session_items hit the cache — only one API call made."""
        from agent.planner import IntentPlanner

        with _patch_gemini(json.dumps(self._VALID_PAYLOAD)) as mock_gc:
            planner = IntentPlanner(llm_model="gemini-2.5-flash", max_tokens=200)

            items = ["Wireless Headphones", "USB-C Cable"]
            weights = [0.7, 0.3]

            planner.infer_intent(items, weights)
            planner.infer_intent(items, weights)
            planner.infer_intent(items, weights)

            assert mock_gc.call_count == 1, (
                f"Expected 1 API call (cache hit on 2nd/3rd), got {mock_gc.call_count}"
            )

    def test_different_sessions_get_separate_cache_entries(self) -> None:
        """Different session items each trigger an API call."""
        from agent.planner import IntentPlanner

        with _patch_gemini(json.dumps(self._VALID_PAYLOAD)) as mock_gc:
            planner = IntentPlanner(llm_model="gemini-2.5-flash", max_tokens=200)

            planner.infer_intent(["Product A", "Product B"], [0.6, 0.4])
            planner.infer_intent(["Product C", "Product D"], [0.5, 0.5])

            assert mock_gc.call_count == 2

    def test_confidence_clamped_to_unit_interval(self) -> None:
        """Confidence values outside [0,1] from the API are clamped."""
        from agent.planner import IntentPlanner

        payload = {"intent": "test", "keywords": [], "confidence": 1.5}
        with _patch_gemini(json.dumps(payload)):
            planner = IntentPlanner(llm_model="gemini-2.5-flash", max_tokens=200)
            result = planner.infer_intent(["Item X"], [1.0])

        assert 0.0 <= result.confidence <= 1.0

    @patch("time.sleep", return_value=None)
    def test_batch_infer_returns_one_result_per_session(self, _) -> None:
        """batch_infer returns exactly len(sessions) results."""
        from agent.planner import IntentPlanner

        with _patch_gemini(json.dumps(self._VALID_PAYLOAD)):
            planner = IntentPlanner(llm_model="gemini-2.5-flash", max_tokens=200)

            sessions = [
                ["Product A", "Product B"],
                ["Product C"],
                ["Product D", "Product E", "Product F"],
            ]
            weights = [[0.6, 0.4], [1.0], [0.5, 0.3, 0.2]]

            results = planner.batch_infer(sessions, weights)

        assert len(results) == 3

    @patch("time.sleep", return_value=None)
    def test_batch_infer_uses_cache_for_duplicates(self, _) -> None:
        """batch_infer cache hits don't trigger extra API calls."""
        from agent.planner import IntentPlanner

        with _patch_gemini(json.dumps(self._VALID_PAYLOAD)) as mock_gc:
            planner = IntentPlanner(llm_model="gemini-2.5-flash", max_tokens=200)

            sessions = [["A", "B"], ["C", "D"], ["A", "B"]]  # 3rd duplicates 1st
            weights = [[0.6, 0.4], [0.7, 0.3], [0.6, 0.4]]

            planner.batch_infer(sessions, weights)
            assert mock_gc.call_count == 2  # "A,B" cached after first call

    def test_markdown_fenced_json_is_parsed(self) -> None:
        """IntentPlanner handles ```json ... ``` wrapped responses."""
        from agent.planner import IntentPlanner

        fenced = (
            '```json\n'
            '{"intent": "testing", "keywords": ["a"], "confidence": 0.5}\n'
            '```'
        )
        with _patch_gemini(fenced):
            planner = IntentPlanner(llm_model="gemini-2.5-flash", max_tokens=200)
            result = planner.infer_intent(["Product A"], [1.0])

        assert result.intent_summary == "testing"
        assert abs(result.confidence - 0.5) < 1e-6


# ── IntentReranker ────────────────────────────────────────────────────────────


class TestIntentReranker:
    """Tests for agent.reranker.IntentReranker."""

    def test_fit_runs_without_error(self) -> None:
        from agent.reranker import IntentReranker

        _, df = _make_vocab_and_metadata()
        rr = IntentReranker()
        rr.fit(df)
        assert rr._vectorizer is not None

    def test_fit_missing_title_column_raises(self) -> None:
        from agent.reranker import IntentReranker

        rr = IntentReranker()
        with pytest.raises(ValueError):
            rr.fit(pd.DataFrame({"item_id": ["A"]}))

    def test_rerank_before_fit_raises(self) -> None:
        from agent.reranker import IntentReranker

        vocab, df = _make_vocab_and_metadata()
        rr = IntentReranker()
        intent = _make_intent_result()
        with pytest.raises(RuntimeError):
            rr.rerank([(2, 0.8)], intent, vocab, df, top_k=5)

    def test_rerank_returns_correct_count(self) -> None:
        from agent.reranker import IntentReranker

        vocab, df = _make_vocab_and_metadata(n=10)
        rr = IntentReranker()
        rr.fit(df)
        intent = _make_intent_result()
        candidates = [(vocab.encode(f"B{i:08d}"), float(i) / 10) for i in range(2, 8)]
        results = rr.rerank(candidates, intent, vocab, df, top_k=3)
        assert len(results) == 3

    def test_rerank_final_score_formula(self) -> None:
        """final_score == 0.6 * retrieval_score + 0.4 * intent_score."""
        from agent.reranker import IntentReranker

        vocab, df = _make_vocab_and_metadata(n=5)
        rr = IntentReranker()
        rr.fit(df)
        intent = _make_intent_result()

        candidates = [(vocab.encode("B00000002"), 0.75)]
        results = rr.rerank(candidates, intent, vocab, df, top_k=1)
        assert len(results) == 1
        item = results[0]

        expected = 0.6 * item.retrieval_score + 0.4 * item.intent_score
        assert abs(item.final_score - expected) < 1e-6, (
            f"final_score={item.final_score:.6f} expected={expected:.6f}"
        )

    def test_rerank_sorted_descending(self) -> None:
        from agent.reranker import IntentReranker

        vocab, df = _make_vocab_and_metadata(n=10)
        rr = IntentReranker()
        rr.fit(df)
        intent = _make_intent_result()
        candidates = [(vocab.encode(f"B{i:08d}"), float(i) / 10) for i in range(1, 9)]
        results = rr.rerank(candidates, intent, vocab, df, top_k=8)
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_item_fields_populated(self) -> None:
        from agent.reranker import IntentReranker
        from agent.interfaces import RankedItem

        vocab, df = _make_vocab_and_metadata(n=5)
        rr = IntentReranker()
        rr.fit(df)
        intent = _make_intent_result()
        candidates = [(vocab.encode("B00000002"), 0.8)]
        results = rr.rerank(candidates, intent, vocab, df, top_k=1)

        item = results[0]
        assert isinstance(item, RankedItem)
        assert isinstance(item.item_title, str)
        assert item.retrieval_score == 0.8
        assert 0.0 <= item.intent_score <= 1.0


# ── RecommendationExplainer ───────────────────────────────────────────────────


class TestRecommendationExplainer:
    """Tests for agent.explainer.RecommendationExplainer."""

    def _make_ranked_item(self, item_id: int = 5) -> "RankedItem":
        from agent.interfaces import RankedItem

        return RankedItem(
            item_id=item_id,
            item_title=f"Product {item_id}",
            retrieval_score=0.8,
            intent_score=0.7,
            final_score=0.76,
        )

    def test_generate_explanation_non_empty(self) -> None:
        from agent.explainer import RecommendationExplainer

        explainer = RecommendationExplainer()
        intent = _make_intent_result()
        explanation = explainer.generate_explanation(
            ranked_item=self._make_ranked_item(),
            session_items=["Wireless Headphones", "USB-C Cable"],
            attn_weights=[0.7, 0.3],
            intent=intent,
        )
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_generate_explanation_contains_top_item(self) -> None:
        """Explanation must reference the highest-attended session item."""
        from agent.explainer import RecommendationExplainer

        explainer = RecommendationExplainer()
        top_item = "Wireless Headphones"
        intent = _make_intent_result()
        explanation = explainer.generate_explanation(
            ranked_item=self._make_ranked_item(),
            session_items=[top_item, "USB-C Cable"],
            attn_weights=[0.8, 0.2],
            intent=intent,
        )
        assert top_item in explanation, (
            f"Expected top item '{top_item}' in explanation: {explanation!r}"
        )

    def test_generate_explanation_contains_intent(self) -> None:
        from agent.explainer import RecommendationExplainer

        explainer = RecommendationExplainer()
        intent = _make_intent_result(intent="looking for audio accessories")
        explanation = explainer.generate_explanation(
            ranked_item=self._make_ranked_item(),
            session_items=["Headphones"],
            attn_weights=[1.0],
            intent=intent,
        )
        assert "looking for audio accessories" in explanation

    def test_generate_explanation_empty_session_no_crash(self) -> None:
        from agent.explainer import RecommendationExplainer

        explainer = RecommendationExplainer()
        intent = _make_intent_result()
        explanation = explainer.generate_explanation(
            ranked_item=self._make_ranked_item(),
            session_items=[],
            attn_weights=[],
            intent=intent,
        )
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_format_recommendations_count(self) -> None:
        from agent.explainer import RecommendationExplainer
        from agent.interfaces import RankedItem

        explainer = RecommendationExplainer()
        intent = _make_intent_result()
        ranked = [
            RankedItem(i, f"Product {i}", 0.8, 0.7, 0.76) for i in range(5)
        ]
        outputs = explainer.format_recommendations(
            ranked_items=ranked,
            session_items=["Headphones", "Cable"],
            attn_weights=[0.7, 0.3],
            intent=intent,
        )
        assert len(outputs) == 5

    def test_format_recommendations_rank_is_1_based(self) -> None:
        from agent.explainer import RecommendationExplainer
        from agent.interfaces import RankedItem

        explainer = RecommendationExplainer()
        intent = _make_intent_result()
        ranked = [RankedItem(i, f"P{i}", 0.8, 0.7, 0.76) for i in range(3)]
        outputs = explainer.format_recommendations(ranked, ["H"], [1.0], intent)
        assert [o.rank for o in outputs] == [1, 2, 3]

    def test_format_recommendations_attention_heatmap(self) -> None:
        from agent.explainer import RecommendationExplainer
        from agent.interfaces import RankedItem

        explainer = RecommendationExplainer()
        intent = _make_intent_result()
        session = ["Item A", "Item B", "Item C", "Item D", "Item E", "Item F"]
        weights = [0.4, 0.25, 0.15, 0.1, 0.07, 0.03]
        ranked = [RankedItem(1, "Product 1", 0.8, 0.7, 0.76)]
        outputs = explainer.format_recommendations(ranked, session, weights, intent)

        heatmap = outputs[0].attention_heatmap
        assert len(heatmap) == 5
        assert "Item A" in heatmap
        assert "Item F" not in heatmap  # 6th item excluded

    def test_format_recommendations_output_type(self) -> None:
        from agent.explainer import RecommendationExplainer
        from agent.interfaces import RankedItem, RecommendationOutput

        explainer = RecommendationExplainer()
        intent = _make_intent_result()
        ranked = [RankedItem(0, "Product 0", 0.8, 0.7, 0.76)]
        outputs = explainer.format_recommendations(ranked, ["H"], [1.0], intent)
        assert all(isinstance(o, RecommendationOutput) for o in outputs)


# ── Legacy stub backward-compatibility tests ──────────────────────────────────


class TestAgentPlanner:
    """Backward-compat tests for the original AgentPlanner stub."""

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
    """Backward-compat test for the original Reranker stub."""

    def test_instantiation(self, cfg) -> None:
        from agent.reranker import Reranker

        r = Reranker(cfg.retrieval)
        assert r.final_top_k == cfg.retrieval.final_top_k


class TestExplainer:
    """Backward-compat test for the original Explainer stub."""

    def test_instantiation(self, cfg) -> None:
        from agent.explainer import Explainer

        e = Explainer(cfg.agent)
        assert e.model == cfg.agent.llm_model
        assert e.max_tokens == cfg.agent.llm_max_tokens
