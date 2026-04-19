"""retrieval/content_based.py — TF-IDF content-based filtering.

Builds a TF-IDF representation of item metadata (title + description +
category) and retrieves the top-K most similar items for a set of seed items.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from data.vocab import Vocabulary

logger = logging.getLogger(__name__)

# Required columns in the item metadata DataFrame.
_REQUIRED_COLS: List[str] = ["item_id", "title", "description", "category", "price"]


class ContentBasedFilter:
    """Content-based filter using TF-IDF vectors over item metadata.

    Attributes:
        _vectorizer:  Fitted TfidfVectorizer.
        _tfidf_matrix: Sparse (num_items × vocab) TF-IDF matrix.
        _item_ids:    Ordered list of item-id strings corresponding to rows.
    """

    def __init__(self) -> None:
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix: csr_matrix | None = None
        self._item_ids: List[str] = []

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, item_metadata: pd.DataFrame) -> None:
        """Build TF-IDF matrix from item metadata.

        Text is formed by concatenating title, description, and category
        for each item. Missing values are replaced with empty strings.

        Args:
            item_metadata: DataFrame with columns
                [item_id, title, description, category, price].

        Raises:
            ValueError: If required columns are missing.
        """
        missing = [c for c in _REQUIRED_COLS if c not in item_metadata.columns]
        if missing:
            raise ValueError(f"item_metadata missing columns: {missing}")

        df = item_metadata.copy()
        for col in ("title", "description", "category"):
            df[col] = df[col].fillna("").astype(str)

        corpus = (
            df["title"] + " " + df["description"] + " " + df["category"]
        ).tolist()
        self._item_ids = df["item_id"].astype(str).tolist()

        self._vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=1)
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)

        vocab_size = len(self._vectorizer.vocabulary_)
        logger.info(
            "ContentBasedFilter fitted: %d items, vocab_size=%d, matrix=%s",
            len(self._item_ids),
            vocab_size,
            self._tfidf_matrix.shape,
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_candidates(
        self,
        item_ids: List[int],
        top_k: int,
        vocab: Vocabulary,
    ) -> List[Tuple[int, float]]:
        """Return top-K content-similar items for a list of seed item indices.

        Seed integer indices are decoded to ASIN strings using *vocab*,
        looked up in the fitted metadata index, and used to aggregate
        cosine-similarity scores.

        Args:
            item_ids: Integer item indices (as stored in Vocabulary).
            top_k:    Maximum number of candidates to return.
            vocab:    Vocabulary instance for index-to-ASIN decoding.

        Returns:
            List of (item_idx, score) tuples sorted descending by score.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
        """
        if self._tfidf_matrix is None or self._vectorizer is None:
            raise RuntimeError("Call fit() before get_candidates().")

        # Build a mapping: asin → row index in _tfidf_matrix.
        asin_to_row = {asin: i for i, asin in enumerate(self._item_ids)}

        seed_rows: List[int] = []
        seed_asins: List[str] = []
        for idx in item_ids:
            asin = vocab.decode(idx)
            if asin in asin_to_row:
                seed_rows.append(asin_to_row[asin])
                seed_asins.append(asin)

        if not seed_rows:
            return []

        # Aggregate TF-IDF rows of seeds, then compute cosine similarity.
        seed_matrix = self._tfidf_matrix[seed_rows]  # (len(seed_rows), vocab)
        # seed_matrix.mean(axis=0) returns np.matrix on older scipy; convert
        # to a plain ndarray so sklearn's cosine_similarity accepts it.
        agg_vector = np.asarray(seed_matrix.mean(axis=0)).reshape(1, -1)  # (1, vocab)

        scores = cosine_similarity(agg_vector, self._tfidf_matrix).flatten()

        # Exclude seed items from results.
        seed_row_set = set(seed_rows)
        for r in seed_row_set:
            scores[r] = -np.inf

        k = min(top_k, len(self._item_ids) - len(seed_row_set))
        if k <= 0:
            return []

        top_rows = np.argpartition(scores, -k)[-k:]
        top_rows = top_rows[np.argsort(scores[top_rows])[::-1]]

        results: List[Tuple[int, float]] = []
        for row in top_rows:
            score = float(scores[row])
            if score <= -np.inf:
                continue
            asin = self._item_ids[row]
            item_idx = vocab.encode(asin)
            results.append((item_idx, score))

        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Serialise the fitted filter to disk using pickle.

        Args:
            path: Destination file path (parent directories created if needed).

        Raises:
            RuntimeError: If the filter has not been fitted.
        """
        if self._tfidf_matrix is None:
            raise RuntimeError("Cannot save an unfitted ContentBasedFilter.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "vectorizer": self._vectorizer,
            "tfidf_matrix": self._tfidf_matrix,
            "item_ids": self._item_ids,
        }
        with path.open("wb") as fh:
            pickle.dump(state, fh)
        logger.info("ContentBasedFilter saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> ContentBasedFilter:
        """Deserialise a fitted ContentBasedFilter from disk.

        Args:
            path: Path to a previously saved pickle file.

        Returns:
            Reconstructed ContentBasedFilter instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ContentBasedFilter checkpoint not found: {path}")
        with path.open("rb") as fh:
            state = pickle.load(fh)
        obj = cls()
        obj._vectorizer = state["vectorizer"]
        obj._tfidf_matrix = state["tfidf_matrix"]
        obj._item_ids = state["item_ids"]
        logger.info(
            "ContentBasedFilter loaded from %s (%d items)", path, len(obj._item_ids)
        )
        return obj


# ── Legacy stub (kept for backward compatibility) ─────────────────────────────


class ContentBasedRetriever:
    """Retrieve candidates via item content similarity of session embedding. (stub)"""

    def __init__(self, cfg: object) -> None:
        self.top_k = cfg.content_top_k
        self._item_matrix: np.ndarray | None = None

    def fit(self, item_embeddings: np.ndarray) -> "ContentBasedRetriever":
        """Index the item embedding matrix for fast similarity search."""
        raise NotImplementedError("Implemented in Module 03")

    def retrieve(
        self, session_embedding: np.ndarray, top_k: int | None = None
    ) -> List[Tuple[int, float]]:
        """Return (item_idx, cosine_score) pairs for a session embedding."""
        raise NotImplementedError("Implemented in Module 03")
