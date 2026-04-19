"""agent/planner.py — Intent planning via Gemini LLM.

Uses the google-genai SDK to infer a user's purchase intent from the
top-attended items in their current session. Results are cached to
avoid redundant API calls.

The GEMINI_API_KEY is read from a ``.env`` file (via python-dotenv) or
from the real environment — whichever is present."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, FrozenSet, List

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

# Load .env so GEMINI_API_KEY is available even without exporting it manually.
load_dotenv()

from agent.interfaces import IntentResult

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_INTENT = "browsing electronics"
_DEFAULT_KEYWORDS: List[str] = []
_DEFAULT_CONFIDENCE = 0.0

# Seconds to sleep between API calls in batch_infer.
_BATCH_SLEEP_SECS = 0.5

# Intent taxonomy kept for reference by downstream components.
INTENTS = [
    "explore",   # user is browsing broadly
    "refine",    # user is narrowing toward a specific need
    "complete",  # user is about to purchase / close on an item
    "repeat",    # user may want a replacement / reorder
]


class IntentPlanner:
    """Classify purchase intent from session attention using Gemini LLM.

    Results are cached by frozenset of session item titles to avoid
    duplicate API calls during evaluation / batch inference.

    Attributes:
        model:       google.generativeai.GenerativeModel instance.
        max_tokens:  Maximum output tokens allowed per API call.
        _cache:      Dict mapping frozenset(session_items) → IntentResult.
    """

    def __init__(self, llm_model: str, max_tokens: int) -> None:
        """Initialise the Gemini client.

        Args:
            llm_model:  Gemini model ID, e.g. ``"gemini-2.5-flash"``.
            max_tokens: Maximum tokens in the completion response.

        Raises:
            KeyError: If ``GEMINI_API_KEY`` is not set in the environment.
        """
        api_key = os.environ.get("GEMINI_API_KEY", "")
        self._client = genai.Client(api_key=api_key)
        self._model_id = llm_model
        self.max_tokens = max_tokens
        self._cache: Dict[FrozenSet[str], IntentResult] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def infer_intent(
        self,
        session_items: List[str],
        attention_weights: List[float],
    ) -> IntentResult:
        """Infer purchase intent from top-attended session items.

        Sends a short prompt to the Gemini API requesting a JSON response
        with fields ``intent``, ``keywords``, and ``confidence``.
        Caches results by frozenset of item titles.

        Args:
            session_items:     Decoded item titles (highest-attended first).
            attention_weights: Corresponding attention weights.

        Returns:
            IntentResult with intent_summary, top_items, confidence, keywords.
        """
        cache_key: FrozenSet[str] = frozenset(session_items)
        if cache_key in self._cache:
            logger.debug("Cache hit for session_items=%s", session_items)
            return self._cache[cache_key]

        prompt = self._build_prompt(session_items, attention_weights)
        try:
            response = self._client.models.generate_content(
                model=self._model_id,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=0.2,
                ),
            )
            text = response.text
            result = self._parse_response(text, session_items)
        except Exception as exc:  # noqa: BLE001  (catch-all for API errors)
            logger.warning(
                "Gemini API call failed (%s); returning default IntentResult.", exc
            )
            result = self._default_result(session_items)

        self._cache[cache_key] = result
        return result

    def batch_infer(
        self,
        sessions: List[List[str]],
        weights: List[List[float]],
    ) -> List[IntentResult]:
        """Infer intent for multiple sessions with rate-limit sleep.

        Args:
            sessions: List of session item-title lists.
            weights:  Corresponding attention weight lists.

        Returns:
            List of IntentResult, one per session.
        """
        results: List[IntentResult] = []
        api_calls = 0
        cache_hits = 0

        for items, attn in zip(sessions, weights):
            key = frozenset(items)
            if key in self._cache:
                cache_hits += 1
                results.append(self._cache[key])
            else:
                api_calls += 1
                results.append(self.infer_intent(items, attn))
                time.sleep(_BATCH_SLEEP_SECS)

        total = len(sessions)
        hit_rate = cache_hits / total if total > 0 else 0.0
        logger.info(
            "batch_infer: %d sessions, %d API calls, %d cache hits (%.1f%%)",
            total,
            api_calls,
            cache_hits,
            hit_rate * 100,
        )
        return results

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_prompt(
        self,
        session_items: List[str],
        attention_weights: List[float],
    ) -> str:
        """Construct a compact prompt for the Gemini model.

        Args:
            session_items:     Item titles (top-attended).
            attention_weights: Corresponding attention weights.

        Returns:
            Prompt string.
        """
        top_n = 3
        pairs = list(zip(session_items[:top_n], attention_weights[:top_n]))
        items_text = "\n".join(
            f"  - {title} (attention={weight:.2f})"
            for title, weight in pairs
        )
        return (
            "You are a shopping intent classifier.\n"
            "A user has been browsing these items (most attended first):\n"
            f"{items_text}\n\n"
            "Respond ONLY with a JSON object (no markdown, no extra text):\n"
            '{"intent": "<1-sentence purchase intent>", '
            '"keywords": ["kw1", "kw2", "kw3"], '
            '"confidence": 0.0}\n'
            "confidence must be a float between 0 and 1."
        )

    def _parse_response(
        self, text: str, session_items: List[str]
    ) -> IntentResult:
        """Parse Gemini JSON response into an IntentResult.

        Strips markdown code fences if present, then JSON-parses.
        Falls back to the default result on any parse error.

        Args:
            text:         Raw text from the API response.
            session_items: Session items (for top_items field).

        Returns:
            Parsed IntentResult.
        """
        cleaned = text.strip()
        # Strip optional ```json ... ``` fences.
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            payload = json.loads(cleaned)
            intent_summary = str(payload.get("intent", _DEFAULT_INTENT))
            keywords = [str(k) for k in payload.get("keywords", [])]
            confidence = float(payload.get("confidence", _DEFAULT_CONFIDENCE))
            confidence = max(0.0, min(1.0, confidence))
            return IntentResult(
                intent_summary=intent_summary,
                top_items=list(session_items[:3]),
                confidence=confidence,
                keywords=keywords,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(
                "Failed to parse Gemini response (%s); raw text: %r", exc, text
            )
            return self._default_result(session_items)

    @staticmethod
    def _default_result(session_items: List[str]) -> IntentResult:
        """Return a safe default IntentResult when the API call fails.

        Args:
            session_items: Session items used to populate top_items.

        Returns:
            Default IntentResult.
        """
        return IntentResult(
            intent_summary=_DEFAULT_INTENT,
            top_items=list(session_items[:3]),
            confidence=_DEFAULT_CONFIDENCE,
            keywords=list(_DEFAULT_KEYWORDS),
        )


# ── Legacy stub (kept for backward compatibility with existing tests) ──────────


class AgentPlanner:
    """LLM-powered planning module. (legacy stub)"""

    def __init__(self, cfg: object) -> None:
        self.model = cfg.llm_model
        self.max_tokens = cfg.llm_max_tokens
        self.intent_top_items = cfg.intent_top_items
        self._client = None

    def plan(
        self,
        session_items: List[str],
        candidates: List[object],
    ) -> object:
        """Return a structured plan dict."""
        raise NotImplementedError("Implemented in Module 04")

    def _build_prompt(
        self, session_items: List[str], candidates: List[object]
    ) -> str:
        raise NotImplementedError("Implemented in Module 04")

    def _call_llm(self, prompt: str) -> str:
        raise NotImplementedError("Implemented in Module 04")
