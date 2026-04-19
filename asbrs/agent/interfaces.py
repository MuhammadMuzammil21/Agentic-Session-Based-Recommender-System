"""agent/interfaces.py — Shared dataclasses (contracts) for the agent layer.

All agent components communicate through these types only.
No logic lives here — interfaces only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class IntentResult:
    """Output of IntentPlanner.infer_intent().

    Attributes:
        intent_summary: One-sentence description of inferred purchase intent.
        top_items:      The top attended item titles used to infer intent.
        confidence:     Model confidence in the intent classification (0.0–1.0).
        keywords:       Keywords extracted by the LLM (may be empty on failure).
    """

    intent_summary: str
    top_items: List[str]
    confidence: float
    keywords: List[str] = field(default_factory=list)


@dataclass
class RankedItem:
    """Output of IntentReranker.rerank() — a single re-ranked candidate.

    Attributes:
        item_id:          Integer item index (from Vocabulary).
        item_title:       Human-readable item title from metadata.
        retrieval_score:  Score from HybridRetriever (CF+CB fusion).
        intent_score:     Cosine similarity between item TF-IDF and intent text.
        final_score:      Weighted combination: 0.6·retrieval + 0.4·intent.
    """

    item_id: int
    item_title: str
    retrieval_score: float
    intent_score: float
    final_score: float


@dataclass
class RecommendationOutput:
    """Final output delivered to the demo / API layer.

    Attributes:
        rank:              1-based ranking position.
        item_id:           Integer item index (from Vocabulary).
        item_title:        Human-readable item title.
        final_score:       Merged ranking score.
        explanation:       Natural-language explanation string.
        attention_heatmap: Maps item_title → attention_weight for top-5 session
                           items (used by demo visualiser).
    """

    rank: int
    item_id: int
    item_title: str
    final_score: float
    explanation: str
    attention_heatmap: Dict[str, float] = field(default_factory=dict)
