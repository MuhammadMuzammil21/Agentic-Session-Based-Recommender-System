"""tests/test_data.py — Comprehensive unit tests for Module 01: Data Pipeline.

All tests run offline. HuggingFace API calls are mocked via unittest.mock.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch

# ── Ensure project root is on path ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Config
from data.interfaces import EncodedSession, Session
from data.loader import AmazonDataLoader
from data.preprocessor import SessionPreprocessor
from data.session_builder import SessionBuilder
from data.vocab import PAD_IDX, UNK_IDX, Vocabulary


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="session")
def cfg() -> Config:
    """Load real project config."""
    return Config.load(PROJECT_ROOT / "config" / "config.yaml")


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small synthetic interactions DataFrame."""
    return pd.DataFrame(
        {
            "user_id": ["U1", "U1", "U1", "U2", "U2", "U3", "U3", "U3", "U3"],
            "item_id": ["A", "B", "C", "A", "D", "B", "C", "D", "E"],
            "rating": [5.0, 4.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 2.0],
            "timestamp": [
                1_000_000,
                1_003_600,
                1_007_200,  # U1 — 1 h gap
                2_000_000,
                2_090_000,  # U2 — 25 h gap → 2 sessions
                3_000_000,
                3_003_600,
                3_007_200,
                3_010_800,  # U3 — 3 h gap
            ],
            "title": [""] * 9,
            "description": [""] * 9,
            "price": [None] * 9,
            "category": ["Electronics"] * 9,
        }
    )


@pytest.fixture
def sample_sessions() -> List[Session]:
    """Small synthetic list of Session objects."""
    return [
        Session("U1", ["A", "B", "C", "D"], [100, 200, 300, 400]),
        Session("U2", ["B", "C", "D", "E", "F"], [110, 210, 310, 410, 510]),
        Session("U3", ["A", "C"], [120, 220]),
        Session("U4", ["D", "E", "F", "G", "H", "I"], [130, 230, 330, 430, 530, 630]),
    ]


@pytest.fixture
def vocab_with_items() -> Vocabulary:
    """Vocabulary pre-built with items A–I."""
    v = Vocabulary()
    v.build(["A", "B", "C", "D", "E", "F", "G", "H", "I"])
    return v


# ══════════════════════════════════════════════════════════════════════════════
# Config Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestConfig:
    def test_load_returns_config(self, cfg: Config) -> None:
        assert isinstance(cfg, Config)

    def test_nested_data_config(self, cfg: Config) -> None:
        assert cfg.data.min_session_len >= 2
        assert cfg.data.max_session_len > cfg.data.min_session_len

    def test_nested_model_config(self, cfg: Config) -> None:
        assert cfg.model.embedding_dim > 0
        assert cfg.model.hidden_dim % cfg.model.num_attention_heads == 0

    def test_validate_passes_on_good_config(self, cfg: Config) -> None:
        cfg.validate()  # should not raise

    def test_validate_raises_on_bad_splits(self, cfg: Config) -> None:
        original = cfg.data.train_split
        cfg.data.train_split = 0.9  # now sums > 1.0
        with pytest.raises(ValueError, match="splits must sum"):
            cfg.validate()
        cfg.data.train_split = original  # restore

    def test_validate_raises_on_bad_min_session_len(self, cfg: Config) -> None:
        original = cfg.data.min_session_len
        cfg.data.min_session_len = 1
        with pytest.raises(ValueError, match="min_session_len must be >= 2"):
            cfg.validate()
        cfg.data.min_session_len = original

    def test_validate_raises_on_bad_attention_heads(self, cfg: Config) -> None:
        original = cfg.model.hidden_dim
        cfg.model.hidden_dim = 7  # not divisible by num_heads
        with pytest.raises(ValueError, match="divisible"):
            cfg.validate()
        cfg.model.hidden_dim = original

    def test_load_raises_on_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            Config.load("/nonexistent/path/config.yaml")


# ══════════════════════════════════════════════════════════════════════════════
# Vocabulary Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestVocabulary:
    def test_initial_special_tokens(self) -> None:
        v = Vocabulary()
        assert v.encode("<PAD>") == PAD_IDX
        assert v.encode("<UNK>") == UNK_IDX
        assert v.decode(PAD_IDX) == "<PAD>"
        assert v.decode(UNK_IDX) == "<UNK>"

    def test_initial_length(self) -> None:
        v = Vocabulary()
        assert len(v) == 2

    def test_build_adds_items(self) -> None:
        v = Vocabulary()
        v.build(["X", "Y", "Z"])
        assert len(v) == 5  # PAD + UNK + 3

    def test_build_is_idempotent_for_duplicates(self) -> None:
        v = Vocabulary()
        v.build(["A", "A", "B", "A"])
        assert len(v) == 4  # PAD + UNK + A + B

    def test_encode_known_item(self, vocab_with_items: Vocabulary) -> None:
        idx = vocab_with_items.encode("A")
        assert idx >= 2  # not special token

    def test_encode_unknown_returns_unk(self, vocab_with_items: Vocabulary) -> None:
        assert vocab_with_items.encode("UNKNOWN_ITEM") == UNK_IDX

    def test_decode_roundtrip(self, vocab_with_items: Vocabulary) -> None:
        for item in ["A", "B", "C"]:
            idx = vocab_with_items.encode(item)
            assert vocab_with_items.decode(idx) == item

    def test_decode_unknown_idx_returns_unk(self, vocab_with_items: Vocabulary) -> None:
        assert vocab_with_items.decode(99_999) == "<UNK>"

    def test_contains(self, vocab_with_items: Vocabulary) -> None:
        assert "A" in vocab_with_items
        assert "MISSING" not in vocab_with_items

    def test_save_and_load(
        self, vocab_with_items: Vocabulary, tmp_path: Path
    ) -> None:
        save_path = tmp_path / "vocab.json"
        vocab_with_items.save(save_path)
        loaded = Vocabulary.load(save_path)
        for item in ["A", "B", "C"]:
            assert vocab_with_items.encode(item) == loaded.encode(item)
        assert len(vocab_with_items) == len(loaded)

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        v = Vocabulary()
        v.build(["A"])
        nested = tmp_path / "deep" / "path" / "vocab.json"
        v.save(nested)
        assert nested.exists()

    def test_load_raises_on_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            Vocabulary.load(Path("/nonexistent/vocab.json"))


# ══════════════════════════════════════════════════════════════════════════════
# Session Interface Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestSessionInterface:
    def test_session_length_property(self) -> None:
        s = Session("U1", ["A", "B", "C"], [1, 2, 3])
        assert s.length == 3

    def test_session_raises_on_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            Session("U1", ["A", "B"], [1])


# ══════════════════════════════════════════════════════════════════════════════
# AmazonDataLoader Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestAmazonDataLoader:
    def _make_fake_ds(self, n: int = 5) -> MagicMock:
        """Build a fake HF streaming dataset mock."""
        records = [
            {
                "user_id": f"U{i}",
                "asin": f"B{i:04d}",
                "rating": 4.0,
                "timestamp": 1_000_000 + i * 3600,
                "title": f"Product {i}",
                "text": "desc",
                "price": None,
                "main_category": "Electronics",
            }
            for i in range(n)
        ]
        fake_split = MagicMock()
        fake_split.take.return_value = iter(records)
        fake_ds = {"full": fake_split}
        return fake_ds

    def test_stream_reviews_returns_dataframe(self, cfg: Config) -> None:
        loader = AmazonDataLoader()
        with patch("data.loader.load_dataset", return_value=self._make_fake_ds(5)):
            df = loader.stream_reviews(
                category=cfg.data.hf_category_reviews,
                max_records=5,
                cfg=cfg,
            )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_stream_reviews_has_required_columns(self, cfg: Config) -> None:
        loader = AmazonDataLoader()
        with patch("data.loader.load_dataset", return_value=self._make_fake_ds(3)):
            df = loader.stream_reviews(
                category=cfg.data.hf_category_reviews,
                max_records=3,
                cfg=cfg,
            )
        expected = {"user_id", "item_id", "rating", "timestamp"}
        assert expected.issubset(set(df.columns))

    def test_stream_reviews_raises_on_hf_failure(self, cfg: Config) -> None:
        loader = AmazonDataLoader()
        with patch("data.loader.load_dataset", side_effect=Exception("network error")):
            with pytest.raises(RuntimeError, match="HuggingFace streaming failed"):
                loader.stream_reviews(
                    category="bad_category",
                    max_records=10,
                    cfg=cfg,
                )

    def test_filter_interactions_removes_rare_items(
        self, sample_df: pd.DataFrame
    ) -> None:
        loader = AmazonDataLoader()
        # item "E" only appears once
        filtered = loader.filter_interactions(sample_df, min_item_freq=2)
        assert "E" not in filtered["item_id"].values

    def test_filter_interactions_keeps_frequent_items(
        self, sample_df: pd.DataFrame
    ) -> None:
        loader = AmazonDataLoader()
        filtered = loader.filter_interactions(sample_df, min_item_freq=2)
        # Items A, B, C, D appear >= 2 times
        for item in ["A", "B", "C", "D"]:
            assert item in filtered["item_id"].values

    def test_filter_interactions_raises_on_missing_column(self) -> None:
        loader = AmazonDataLoader()
        bad_df = pd.DataFrame({"user_id": ["U1"], "rating": [5.0]})
        with pytest.raises(ValueError, match="item_id"):
            loader.filter_interactions(bad_df)

    def test_filter_min_freq_1_keeps_all(self, sample_df: pd.DataFrame) -> None:
        loader = AmazonDataLoader()
        filtered = loader.filter_interactions(sample_df, min_item_freq=1)
        assert len(filtered) == len(sample_df)


# ══════════════════════════════════════════════════════════════════════════════
# SessionBuilder Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestSessionBuilder:
    def test_build_sessions_returns_sessions(self, sample_df: pd.DataFrame) -> None:
        sb = SessionBuilder()
        sessions = sb.build_sessions(sample_df, window_hours=24)
        assert all(isinstance(s, Session) for s in sessions)

    def test_build_sessions_splits_on_gap(self, sample_df: pd.DataFrame) -> None:
        sb = SessionBuilder()
        # U2 has a 25-hour gap → should produce 2 sessions
        sessions = sb.build_sessions(sample_df, window_hours=24)
        u2_sessions = [s for s in sessions if s.user_id == "U2"]
        assert len(u2_sessions) == 2

    def test_build_sessions_no_gap_keeps_together(self) -> None:
        sb = SessionBuilder()
        df = pd.DataFrame(
            {
                "user_id": ["U1", "U1", "U1"],
                "item_id": ["A", "B", "C"],
                "timestamp": [0, 3600, 7200],  # 1-h gap — within 24 h window
            }
        )
        sessions = sb.build_sessions(df, window_hours=24)
        assert len(sessions) == 1
        assert sessions[0].length == 3

    def test_build_sessions_raises_on_missing_columns(self) -> None:
        sb = SessionBuilder()
        bad_df = pd.DataFrame({"user_id": ["U1"], "item_id": ["A"]})
        with pytest.raises(ValueError, match="missing required columns"):
            sb.build_sessions(bad_df, window_hours=24)

    def test_build_sessions_preserves_order(self) -> None:
        sb = SessionBuilder()
        df = pd.DataFrame(
            {
                "user_id": ["U1", "U1", "U1"],
                "item_id": ["C", "A", "B"],
                "timestamp": [300, 100, 200],  # out of order
            }
        )
        sessions = sb.build_sessions(df, window_hours=24)
        assert sessions[0].item_ids == ["A", "B", "C"]

    def test_filter_sessions_keeps_within_bounds(
        self, sample_sessions: List[Session]
    ) -> None:
        sb = SessionBuilder()
        filtered = sb.filter_sessions(sample_sessions, min_len=3, max_len=5)
        assert all(3 <= s.length <= 5 for s in filtered)

    def test_filter_sessions_removes_short(
        self, sample_sessions: List[Session]
    ) -> None:
        sb = SessionBuilder()
        filtered = sb.filter_sessions(sample_sessions, min_len=3, max_len=10)
        # U3 has length 2 → should be removed
        user_ids = [s.user_id for s in filtered]
        assert "U3" not in user_ids

    def test_filter_sessions_raises_on_invalid_bounds(
        self, sample_sessions: List[Session]
    ) -> None:
        sb = SessionBuilder()
        with pytest.raises(ValueError, match="min_len"):
            sb.filter_sessions(sample_sessions, min_len=5, max_len=3)

    def test_split_sessions_sizes(self, sample_sessions: List[Session]) -> None:
        sb = SessionBuilder()
        train, val, test = sb.split_sessions(
            sample_sessions, train=0.5, val=0.25, test=0.25, seed=0
        )
        assert len(train) + len(val) + len(test) == len(sample_sessions)

    def test_split_sessions_no_overlap(self, sample_sessions: List[Session]) -> None:
        sb = SessionBuilder()
        train, val, test = sb.split_sessions(sample_sessions, seed=0)
        all_sets = [id(s) for s in train + val + test]
        assert len(all_sets) == len(set(all_sets))

    def test_split_sessions_reproducible(
        self, sample_sessions: List[Session]
    ) -> None:
        sb = SessionBuilder()
        t1, v1, e1 = sb.split_sessions(sample_sessions, seed=42)
        t2, v2, e2 = sb.split_sessions(sample_sessions, seed=42)
        assert [s.user_id for s in t1] == [s.user_id for s in t2]

    def test_split_sessions_raises_on_bad_fractions(
        self, sample_sessions: List[Session]
    ) -> None:
        sb = SessionBuilder()
        with pytest.raises(ValueError, match="must equal 1.0"):
            sb.split_sessions(sample_sessions, train=0.9, val=0.1, test=0.5)


# ══════════════════════════════════════════════════════════════════════════════
# SessionPreprocessor Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestSessionPreprocessor:
    def test_prepare_returns_encoded_sessions(
        self, sample_sessions: List[Session], vocab_with_items: Vocabulary
    ) -> None:
        prep = SessionPreprocessor()
        encoded = prep.prepare(sample_sessions, vocab_with_items, max_len=10)
        assert all(isinstance(e, EncodedSession) for e in encoded)

    def test_prepare_skips_length_one_sessions(
        self, vocab_with_items: Vocabulary
    ) -> None:
        prep = SessionPreprocessor()
        short = [Session("U9", ["A"], [100])]
        encoded = prep.prepare(short, vocab_with_items, max_len=10)
        assert len(encoded) == 0

    def test_prepare_input_ids_length_equals_max_len(
        self, sample_sessions: List[Session], vocab_with_items: Vocabulary
    ) -> None:
        prep = SessionPreprocessor()
        max_len = 8
        encoded = prep.prepare(sample_sessions, vocab_with_items, max_len=max_len)
        for e in encoded:
            assert len(e.input_ids) == max_len

    def test_prepare_left_padding(
        self, vocab_with_items: Vocabulary
    ) -> None:
        prep = SessionPreprocessor()
        # Session with 3 items: input = 2 items, padded to max_len=5
        session = [Session("U1", ["A", "B", "C"], [1, 2, 3])]
        encoded = prep.prepare(session, vocab_with_items, max_len=5)
        e = encoded[0]
        assert e.session_len == 2
        # First 3 positions are PAD
        assert e.input_ids[:3] == [PAD_IDX, PAD_IDX, PAD_IDX]

    def test_prepare_truncation_keeps_most_recent(
        self, vocab_with_items: Vocabulary
    ) -> None:
        prep = SessionPreprocessor()
        # 6-item session input → truncated to max_len=3 (last 3 of first 5)
        session = [Session("U1", ["A", "B", "C", "D", "E", "F"], [1, 2, 3, 4, 5, 6])]
        encoded = prep.prepare(session, vocab_with_items, max_len=3)
        e = encoded[0]
        # target = F, input = last 3 of [A,B,C,D,E] = [C,D,E]
        assert e.target_id == vocab_with_items.encode("F")
        tail = [vocab_with_items.decode(i) for i in e.input_ids]
        assert tail == ["C", "D", "E"]

    def test_prepare_target_is_last_item(
        self, vocab_with_items: Vocabulary
    ) -> None:
        prep = SessionPreprocessor()
        session = [Session("U1", ["A", "B", "C"], [1, 2, 3])]
        encoded = prep.prepare(session, vocab_with_items, max_len=10)
        assert encoded[0].target_id == vocab_with_items.encode("C")

    def test_prepare_raises_on_invalid_max_len(
        self, sample_sessions: List[Session], vocab_with_items: Vocabulary
    ) -> None:
        prep = SessionPreprocessor()
        with pytest.raises(ValueError, match="max_len"):
            prep.prepare(sample_sessions, vocab_with_items, max_len=0)

    def test_to_dataloader_returns_dataloader(
        self, sample_sessions: List[Session], vocab_with_items: Vocabulary
    ) -> None:
        from torch.utils.data import DataLoader

        prep = SessionPreprocessor()
        encoded = prep.prepare(sample_sessions, vocab_with_items, max_len=10)
        loader = prep.to_dataloader(encoded, batch_size=2, shuffle=False)
        assert isinstance(loader, DataLoader)

    def test_to_dataloader_batch_shapes(
        self, sample_sessions: List[Session], vocab_with_items: Vocabulary
    ) -> None:
        prep = SessionPreprocessor()
        max_len = 10
        encoded = prep.prepare(sample_sessions, vocab_with_items, max_len=max_len)
        loader = prep.to_dataloader(encoded, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        assert batch["input_ids"].shape == (2, max_len)
        assert batch["lengths"].shape == (2,)
        assert batch["target"].shape == (2,)

    def test_to_dataloader_batch_dtypes(
        self, sample_sessions: List[Session], vocab_with_items: Vocabulary
    ) -> None:
        prep = SessionPreprocessor()
        encoded = prep.prepare(sample_sessions, vocab_with_items, max_len=10)
        loader = prep.to_dataloader(encoded, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        assert batch["input_ids"].dtype == torch.long
        assert batch["target"].dtype == torch.long

    def test_to_dataloader_raises_on_empty_encoded(
        self, vocab_with_items: Vocabulary
    ) -> None:
        prep = SessionPreprocessor()
        with pytest.raises(ValueError, match="non-empty"):
            prep.to_dataloader([], batch_size=4, shuffle=False)

    def test_to_dataloader_raises_on_invalid_batch_size(
        self, sample_sessions: List[Session], vocab_with_items: Vocabulary
    ) -> None:
        prep = SessionPreprocessor()
        encoded = prep.prepare(sample_sessions, vocab_with_items, max_len=10)
        with pytest.raises(ValueError, match="batch_size"):
            prep.to_dataloader(encoded, batch_size=0, shuffle=False)
