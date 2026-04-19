"""agent/reranker.py — Intent-aware re-ranking of candidate items.

Uses a TF-IDF model fitted on item titles to compute semantic similarity
between the inferred intent text and each candidate, then combines that
with the retrieval score into a final ranking score.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from agent.interfaces import IntentResult, RankedItem
from data.vocab import Vocabulary

logger = logging.getLogger(__name__)

# Fusion weights for final_score = w_ret * retrieval_score + w_int * intent_score
_RETRIEVAL_WEIGHT = 0.6
_INTENT_WEIGHT = 0.4


class IntentReranker:
    """Re-rank retrieval candidates using TF-IDF intent similarity.

    Attributes:
        _vectorizer:  Fitted TfidfVectorizer on item titles.
        _item_titles: Ordered list matching the fitted matrix rows.
        _title_to_idx: Mapping from title string to matrix row.
    """

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._item_titles: List[str] = []
        self._title_to_idx: dict[str, int] = {}

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, item_metadata: pd.DataFrame) -> None:
        """Fit the TF-IDF vectorizer on item titles.

        Args:
            item_metadata: DataFrame with at least columns [item_id, title].

        Raises:
            ValueError: If the ``title`` column is missing.
        """
        if "title" not in item_metadata.columns:
            raise ValueError("item_metadata must contain a 'title' column.")

        self._item_titles = (
            item_metadata["title"].fillna("").astype(str).tolist()
        )
        self._title_to_idx = {t: i for i, t in enumerate(self._item_titles)}

        self._vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=1)
        self._vectorizer.fit(self._item_titles)
        logger.info(
            "IntentReranker fitted on %d item titles, vocab=%d",
            len(self._item_titles),
            len(self._vectorizer.vocabulary_),
        )

    # ── Re-ranking ────────────────────────────────────────────────────────────

    def rerank(
        self,
        candidates: List[Tuple[int, float]],
        intent: IntentResult,
        vocab: Vocabulary,
        item_metadata: pd.DataFrame,
        top_k: int,
    ) -> List[RankedItem]:
        """Re-rank candidate items by combining retrieval and intent scores.

        final_score = 0.6 * retrieval_score + 0.4 * intent_score

        The intent score is the cosine similarity between the vectorised
        intent_summary text and each candidate's title vector.

        Args:
            candidates:    List of (item_idx, retrieval_score) from HybridRetriever.
            intent:        IntentResult from IntentPlanner.infer_intent().
            vocab:         Vocabulary for decoding item indices to ASINs.
            item_metadata: DataFrame with [item_id, title] at minimum.
            top_k:         Maximum number of RankedItems to return.

        Returns:
            List of RankedItem sorted descending by final_score.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._vectorizer is None:
            raise RuntimeError("Call fit() before rerank().")

        # Build a title lookup: asin → title
        if "item_id" not in item_metadata.columns or "title" not in item_metadata.columns:
            raise ValueError("item_metadata must contain 'item_id' and 'title' columns.")

        asin_to_title: dict[str, str] = dict(
            zip(
                item_metadata["item_id"].astype(str),
                item_metadata["title"].fillna("").astype(str),
            )
        )

        # Vectorise the intent summary for similarity computation.
        intent_vec = self._vectorizer.transform([intent.intent_summary])

        ranked: List[RankedItem] = []
        for item_idx, retrieval_score in candidates:
            asin = vocab.decode(item_idx)
            title = asin_to_title.get(asin, asin)

            # Compute intent similarity by vectorising the item title.
            item_vec = self._vectorizer.transform([title])
            intent_score_arr = cosine_similarity(
                np.asarray(intent_vec.todense()),
                np.asarray(item_vec.todense()),
            )
            intent_score = float(intent_score_arr[0, 0])

            # Normalise retrieval score to [0, 1] using sigmoid-like clipping.
            norm_retrieval = max(0.0, min(1.0, retrieval_score))

            final_score = (
                _RETRIEVAL_WEIGHT * norm_retrieval
                + _INTENT_WEIGHT * intent_score
            )

            ranked.append(
                RankedItem(
                    item_id=item_idx,
                    item_title=title,
                    retrieval_score=retrieval_score,
                    intent_score=intent_score,
                    final_score=final_score,
                )
            )

        ranked.sort(key=lambda r: r.final_score, reverse=True)
        logger.debug(
            "IntentReranker: %d candidates → top %d returned", len(ranked), top_k
        )
        return ranked[:top_k]


# ── Legacy stub (kept for backward compatibility with existing tests) ──────────


class Reranker:
    """Re-rank retrieval candidates using planner intent. (legacy stub)"""

    def __init__(self, cfg: object) -> None:
        self.final_top_k = cfg.final_top_k

    def rerank(
        self,
        candidates: list,
        plan: dict,
        session_embedding: "np.ndarray",  # noqa: F821
        top_k: int | None = None,
    ) -> list:
        """Re-rank candidates. Stub — raises NotImplementedError."""
        raise NotImplementedError("Implemented in Module 04")
