"""retrieval/collaborative.py — Item-based collaborative filtering retrieval.

Computes item–item cosine similarity from an implicit-feedback interaction
matrix and retrieves top-K candidate items for a given set of seed items.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class ItemBasedCF:
    """Item-based collaborative filter using cosine similarity.

    Attributes:
        _sim_matrix: Sparse item × item cosine-similarity matrix (csr_matrix).
        _num_items:  Number of items in the fitted matrix.
    """

    def __init__(self) -> None:
        self._sim_matrix: csr_matrix | None = None
        self._num_items: int = 0

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, interaction_matrix: csr_matrix) -> None:
        """Build item–item cosine similarity matrix from implicit feedback.

        Args:
            interaction_matrix: Sparse (users × items) matrix of implicit
                feedback values (e.g. rating counts or binary).

        Raises:
            ValueError: If the interaction matrix is empty.
        """
        if interaction_matrix.shape[0] == 0 or interaction_matrix.shape[1] == 0:
            raise ValueError("interaction_matrix must not be empty.")

        start = time.time()
        # Transpose so rows = items; cosine_similarity works row-wise.
        item_matrix = interaction_matrix.T  # (items, users)
        sim = cosine_similarity(item_matrix, dense_output=False)
        self._sim_matrix = sim.tocsr()
        self._num_items = sim.shape[0]
        elapsed = time.time() - start

        nnz = self._sim_matrix.nnz
        total = self._num_items ** 2
        density = nnz / total if total > 0 else 0.0
        logger.info(
            "ItemBasedCF fitted: %d items, density=%.6f, elapsed=%.2fs",
            self._num_items,
            density,
            elapsed,
        )

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_candidates(
        self, item_ids: List[int], top_k: int
    ) -> List[Tuple[int, float]]:
        """Return top-K candidate items by aggregated cosine similarity.

        Aggregation: sums the similarity columns of all seed items, then
        ranks all items not in *item_ids* by descending total score.

        Args:
            item_ids: Integer item indices already interacted with.
            top_k:    Maximum number of candidates to return.

        Returns:
            List of (item_idx, score) tuples, sorted descending by score.

        Raises:
            RuntimeError: If ``fit`` has not been called yet.
            ValueError:   If any item_id is out of range.
        """
        if self._sim_matrix is None:
            raise RuntimeError("Call fit() before get_candidates().")

        valid_ids = [
            i for i in item_ids if 0 <= i < self._num_items
        ]
        if not valid_ids:
            return []

        # Sum similarity scores across all seed columns.
        agg_scores = np.asarray(
            self._sim_matrix[:, valid_ids].sum(axis=1)
        ).flatten()

        # Mask out seed items.
        seed_set = set(item_ids)
        agg_scores[list(seed_set)] = -np.inf

        # Pick top-K by partitioning (faster than full sort for large N).
        k = min(top_k, self._num_items - len(seed_set))
        if k <= 0:
            return []

        top_indices = np.argpartition(agg_scores, -k)[-k:]
        top_indices = top_indices[np.argsort(agg_scores[top_indices])[::-1]]

        return [
            (int(idx), float(agg_scores[idx]))
            for idx in top_indices
            if agg_scores[idx] > -np.inf
        ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Serialise the fitted model to disk using pickle.

        Args:
            path: Destination file path (parent directories created if needed).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self._sim_matrix is None:
            raise RuntimeError("Cannot save an unfitted ItemBasedCF.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"sim_matrix": self._sim_matrix, "num_items": self._num_items}, fh)
        logger.info("ItemBasedCF saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> ItemBasedCF:
        """Deserialise a fitted ItemBasedCF from disk.

        Args:
            path: Path to a previously saved pickle file.

        Returns:
            Reconstructed ItemBasedCF instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ItemBasedCF checkpoint not found: {path}")
        with path.open("rb") as fh:
            state = pickle.load(fh)
        obj = cls()
        obj._sim_matrix = state["sim_matrix"]
        obj._num_items = state["num_items"]
        logger.info("ItemBasedCF loaded from %s (%d items)", path, obj._num_items)
        return obj


# ── Legacy stub (kept for backward compatibility with existing tests) ──────────


class CollaborativeRetriever:
    """Retrieve candidates via user–item collaborative signals. (stub)"""

    def __init__(self, cfg: object) -> None:
        self.top_k = cfg.cf_top_k
        self._similarity_matrix: np.ndarray | None = None

    def fit(self, sessions: List[dict]) -> "CollaborativeRetriever":
        """Build co-occurrence / similarity matrix from training sessions."""
        raise NotImplementedError("Implemented in Module 03")

    def retrieve(
        self, session_items: List[int], top_k: int | None = None
    ) -> List[Tuple[int, float]]:
        """Return (item_idx, score) pairs for a given session."""
        raise NotImplementedError("Implemented in Module 03")
