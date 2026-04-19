"""evaluation/ablation.py — Ablation study runner for ASBRS.

Compares four model configurations against each other:

1. Popularity baseline  — globally most-popular items, no session signal.
2. CF-only              — ItemBasedCF retrieval, no re-ranking.
3. GRU + Attention      — full SessionEncoder, but no LLM intent re-ranking.
4. Full Agentic (ASBRS) — complete pipeline including LLM planner & reranker.

Each variant is evaluated with evaluate_model() on the same test sessions,
and results are aggregated into a comparison DataFrame.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from agent.interfaces import IntentResult, RankedItem, RecommendationOutput
from config.settings import Config
from data.vocab import Vocabulary
from evaluation.metrics import evaluate_model

logger = logging.getLogger(__name__)

# Column ordering for the output DataFrame.
_COLUMNS = [
    "Model",
    "Recall@5",
    "Recall@10",
    "Recall@20",
    "MRR@10",
    "HitRate@10",
]


class AblationStudy:
    """Run ablation experiments across ASBRS pipeline variants.

    Each ``run_*`` method returns a ``Dict[str, float]`` of averaged metrics
    at the k-values configured in *cfg*. ``run_all()`` collects them all into
    a formatted ``pd.DataFrame``.

    Attributes:
        test_sessions:  List of sessions; each is a list of item-ID strings.
        vocab:          Fitted Vocabulary instance.
        item_metadata:  DataFrame with at least ``[item_id, title]`` columns.
        cfg:            Typed Config object.
    """

    def __init__(
        self,
        test_sessions: List[List[str]],
        vocab: Vocabulary,
        item_metadata: pd.DataFrame,
        cfg: Config,
    ) -> None:
        """Initialise the ablation runner.

        Args:
            test_sessions: Each session is an ordered list of item-ID strings
                (ASINs). The last item is the held-out ground truth
                (leave-one-out protocol).
            vocab:         Fitted Vocabulary (maps ASIN ↔ integer index).
            item_metadata: DataFrame with columns ``item_id`` and ``title``.
            cfg:           Loaded Config object.
        """
        self.test_sessions = test_sessions
        self.vocab = vocab
        self.item_metadata = item_metadata
        self.cfg = cfg
        self._k_values: List[int] = cfg.evaluation.k_values

        # Lazy-initialised components (built on first use to avoid heavy
        # imports at module level when only some variants are needed).
        self._cf: Optional[object] = None
        self._cb: Optional[object] = None
        self._hybrid: Optional[object] = None
        self._encoder: Optional[object] = None
        self._reranker: Optional[object] = None

    # ── Internal setup ────────────────────────────────────────────────────────

    def _build_popularity_counts(self) -> Counter:
        """Count item-ID frequencies across all test sessions (excl. target).

        Returns:
            Counter mapping item-ID string → occurrence count.
        """
        counter: Counter = Counter()
        for session in self.test_sessions:
            # Exclude last item (held-out target).
            for asin in session[:-1]:
                counter[asin] += 1
        return counter

    def _session_to_int_ids(self, session: List[str]) -> List[int]:
        """Convert a list of ASIN strings to Vocabulary integer indices.

        Unknown ASINs are replaced with the UNK index.

        Args:
            session: Ordered list of ASIN strings.

        Returns:
            List of integer indices, same length as *session*.
        """
        return [self.vocab.encode(asin) for asin in session]

    def _get_ground_truth(self, session: List[str]) -> int:
        """Return the integer index of the last (held-out) session item.

        Args:
            session: Full session including the ground-truth last item.

        Returns:
            Integer index of the target item.
        """
        return self.vocab.encode(session[-1])

    # ── Variant 1: Popularity baseline ───────────────────────────────────────

    def run_popularity_baseline(self) -> Dict[str, float]:
        """Evaluate a global popularity baseline.

        Recommendation list: globally most-popular items (by training-session
        frequency), same list for every test session.

        Returns:
            Dict[str, float] with averaged Recall@K, MRR@K, HitRate@K.
        """
        logger.info("AblationStudy: running popularity baseline …")
        counter = self._build_popularity_counts()

        # Top-K by global frequency (use the largest K requested).
        max_k = max(self._k_values)
        popular_ids: List[int] = [
            self.vocab.encode(asin)
            for asin, _ in counter.most_common(max_k)
        ]

        predictions = [
            (popular_ids, self._get_ground_truth(session))
            for session in self.test_sessions
        ]

        return evaluate_model(predictions, self._k_values)

    # ── Variant 2: CF-only ────────────────────────────────────────────────────

    def run_cf_only(self) -> Dict[str, float]:
        """Evaluate collaborative-filtering retrieval without re-ranking.

        Uses ItemBasedCF fitted on the training portion of the sessions
        embedded in *item_metadata*.

        Returns:
            Dict[str, float] with averaged Recall@K, MRR@K, HitRate@K.
        """
        logger.info("AblationStudy: running CF-only variant …")
        from retrieval.collaborative import ItemBasedCF

        cf = ItemBasedCF(top_k=max(self._k_values))
        # Build interaction matrix from test sessions (excl. last item).
        train_pairs: List[tuple[int, int]] = []
        for session in self.test_sessions:
            items_int = self._session_to_int_ids(session[:-1])
            for item in items_int:
                train_pairs.append((0, item))  # single synthetic user

        # ItemBasedCF expects a list-of-lists (sessions).
        training_sessions = [
            self._session_to_int_ids(s[:-1]) for s in self.test_sessions
        ]
        cf.fit(training_sessions)

        max_k = max(self._k_values)
        predictions: List[tuple[List[int], int]] = []
        for session in self.test_sessions:
            seed_ids = self._session_to_int_ids(session[:-1])
            candidates = cf.get_candidates(seed_ids, max_k)
            rec_ids = [item_id for item_id, _ in candidates]
            gt = self._get_ground_truth(session)
            predictions.append((rec_ids, gt))

        return evaluate_model(predictions, self._k_values)

    # ── Variant 3: GRU + Attention (no LLM) ──────────────────────────────────

    def run_gru_attention(self) -> Dict[str, float]:
        """Evaluate the SessionEncoder without LLM intent re-ranking.

        Encodes each session via the GRU+Attention encoder and ranks the
        full item vocabulary by decoder score.  No LLM planner is used.

        Returns:
            Dict[str, float] with averaged Recall@K, MRR@K, HitRate@K.
        """
        import torch

        from models.encoder import SessionEncoder

        logger.info("AblationStudy: running GRU+Attention (no LLM) variant …")

        vocab_size = len(self.vocab)
        model_cfg = self.cfg.model
        encoder = SessionEncoder(
            vocab_size=vocab_size,
            embed_dim=model_cfg.embedding_dim,
            hidden_dim=model_cfg.hidden_dim,
            num_heads=model_cfg.num_attention_heads,
            dropout=model_cfg.dropout,
            padding_idx=0,
        )
        encoder.eval()

        max_k = max(self._k_values)
        max_len = model_cfg.max_seq_len
        predictions: List[tuple[List[int], int]] = []

        with torch.no_grad():
            for session in self.test_sessions:
                seed = self._session_to_int_ids(session[:-1])
                gt = self._get_ground_truth(session)

                # Left-pad / truncate to max_len.
                pad_idx = 0
                if len(seed) >= max_len:
                    seed = seed[-max_len:]
                else:
                    seed = [pad_idx] * (max_len - len(seed)) + seed

                input_tensor = torch.tensor([seed], dtype=torch.long)
                lengths = torch.tensor(
                    [min(len(session) - 1, max_len)], dtype=torch.long
                )

                _, scores = encoder(input_tensor, lengths)
                top_ids = scores[0].argsort(descending=True)[:max_k].tolist()
                predictions.append((top_ids, gt))

        return evaluate_model(predictions, self._k_values)

    # ── Variant 4: Full Agentic (ASBRS) ──────────────────────────────────────

    def run_full_agentic(self) -> Dict[str, float]:
        """Evaluate the complete ASBRS pipeline with LLM intent re-ranking.

        Pipeline:
            SessionEncoder → HybridRetriever → IntentPlanner → IntentReranker

        Returns:
            Dict[str, float] with averaged Recall@K, MRR@K, HitRate@K.

        Note:
            IntentPlanner requires a valid GEMINI_API_KEY.  If the API is
            unavailable the planner falls back to a default IntentResult and
            the run still completes.
        """
        import torch

        from agent.planner import IntentPlanner
        from agent.reranker import IntentReranker
        from models.encoder import SessionEncoder
        from retrieval.collaborative import ItemBasedCF
        from retrieval.content_based import ContentBasedFilter
        from retrieval.hybrid import HybridRetriever

        logger.info("AblationStudy: running full agentic pipeline …")

        agent_cfg = self.cfg.agent
        model_cfg = self.cfg.model
        ret_cfg = self.cfg.retrieval

        vocab_size = len(self.vocab)
        encoder = SessionEncoder(
            vocab_size=vocab_size,
            embed_dim=model_cfg.embedding_dim,
            hidden_dim=model_cfg.hidden_dim,
            num_heads=model_cfg.num_attention_heads,
            dropout=model_cfg.dropout,
            padding_idx=0,
        )
        encoder.eval()

        training_sessions = [
            self._session_to_int_ids(s[:-1]) for s in self.test_sessions
        ]
        cf = ItemBasedCF(top_k=ret_cfg.cf_top_k)
        cf.fit(training_sessions)

        cb = ContentBasedFilter(top_k=ret_cfg.content_top_k)
        cb.fit(self.item_metadata)

        hybrid = HybridRetriever(cf=cf, cb=cb)

        reranker = IntentReranker()
        reranker.fit(self.item_metadata)

        planner = IntentPlanner(
            llm_model=agent_cfg.llm_model,
            max_tokens=agent_cfg.llm_max_tokens,
        )

        max_k = max(self._k_values)
        max_len = model_cfg.max_seq_len
        predictions: List[tuple[List[int], int]] = []

        with torch.no_grad():
            for session in self.test_sessions:
                seed_strs = session[:-1]
                seed_int = self._session_to_int_ids(seed_strs)
                gt = self._get_ground_truth(session)

                # Encode session for intent inference.
                pad_idx = 0
                padded = seed_int[-max_len:] if len(seed_int) >= max_len else (
                    [pad_idx] * (max_len - len(seed_int)) + seed_int
                )
                input_tensor = torch.tensor([padded], dtype=torch.long)
                lengths = torch.tensor(
                    [min(len(seed_int), max_len)], dtype=torch.long
                )
                _, _ = encoder(input_tensor, lengths)

                # Retrieve hybrid candidates.
                candidates = hybrid.retrieve(seed_int, max_k, self.vocab)

                # Infer intent from seed item titles.
                asin_to_title = dict(
                    zip(
                        self.item_metadata["item_id"].astype(str),
                        self.item_metadata["title"].fillna("").astype(str),
                    )
                )
                seed_titles = [
                    asin_to_title.get(asin, asin) for asin in seed_strs
                ]
                uniform_weights = [1.0 / len(seed_titles)] * len(seed_titles)
                intent = planner.infer_intent(seed_titles, uniform_weights)

                # Re-rank.
                ranked = reranker.rerank(
                    candidates, intent, self.vocab, self.item_metadata, max_k
                )
                rec_ids = [r.item_id for r in ranked]
                predictions.append((rec_ids, gt))

        return evaluate_model(predictions, self._k_values)

    # ── Aggregate runner ──────────────────────────────────────────────────────

    def run_all(self) -> pd.DataFrame:
        """Run all four variants and return a formatted comparison DataFrame.

        Columns: Model | Recall@5 | Recall@10 | Recall@20 | MRR@10 | HitRate@10

        Returns:
            DataFrame with one row per model variant.
        """
        variants = {
            "Popularity Baseline": self.run_popularity_baseline,
            "CF Only": self.run_cf_only,
            "GRU + Attention": self.run_gru_attention,
            "Full Agentic (ASBRS)": self.run_full_agentic,
        }

        rows: List[Dict[str, object]] = []
        for model_name, runner in variants.items():
            logger.info("AblationStudy.run_all: starting %s …", model_name)
            try:
                metrics = runner()
                row: Dict[str, object] = {"Model": model_name}
                row["Recall@5"] = metrics.get("Recall@5", float("nan"))
                row["Recall@10"] = metrics.get("Recall@10", float("nan"))
                row["Recall@20"] = metrics.get("Recall@20", float("nan"))
                row["MRR@10"] = metrics.get("MRR@10", float("nan"))
                row["HitRate@10"] = metrics.get("HitRate@10", float("nan"))
                rows.append(row)
                logger.info("AblationStudy.run_all: %s done → %s", model_name, metrics)
            except Exception as exc:
                logger.error(
                    "AblationStudy.run_all: %s FAILED — %s", model_name, exc
                )
                rows.append(
                    {
                        "Model": model_name,
                        "Recall@5": float("nan"),
                        "Recall@10": float("nan"),
                        "Recall@20": float("nan"),
                        "MRR@10": float("nan"),
                        "HitRate@10": float("nan"),
                    }
                )

        df = pd.DataFrame(rows, columns=_COLUMNS)
        return df

    # ── Result persistence ────────────────────────────────────────────────────

    def save_results(self, df: pd.DataFrame, path: Path) -> None:
        """Persist the ablation results as CSV and Markdown.

        Args:
            df:   DataFrame returned by ``run_all()``.
            path: Destination file path (e.g. ``evaluation/ablation_results.csv``).
                  A ``*.md`` counterpart is saved alongside automatically.

        Raises:
            ValueError: If *df* is empty.
            OSError:    If the parent directory cannot be created.
        """
        if df.empty:
            raise ValueError("DataFrame is empty — nothing to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # CSV
        df.to_csv(path, index=False)
        logger.info("Ablation results saved to %s", path)

        # Markdown
        md_path = path.with_suffix(".md")
        md_lines = [
            "# ASBRS Ablation Study Results",
            "",
            df.to_markdown(index=False, floatfmt=".4f"),
            "",
        ]
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        logger.info("Markdown table saved to %s", md_path)
