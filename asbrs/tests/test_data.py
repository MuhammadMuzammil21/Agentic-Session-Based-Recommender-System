"""
test_data.py — Unit tests for data pipeline components.
"""

import pytest


class TestVocab:
    def test_special_tokens(self):
        from data.vocab import Vocab, PAD_TOKEN, UNK_TOKEN
        v = Vocab()
        assert v.item2idx(PAD_TOKEN) == 0
        assert v.item2idx(UNK_TOKEN) == 1
        assert v.idx2item(0) == PAD_TOKEN
        assert v.idx2item(1) == UNK_TOKEN

    def test_unknown_item_returns_unk(self):
        from data.vocab import Vocab
        v = Vocab()
        assert v.item2idx("NOT_IN_VOCAB") == Vocab.UNK_IDX

    def test_initial_length(self):
        from data.vocab import Vocab
        v = Vocab()
        assert len(v) == 2  # PAD + UNK


class TestSessionBuilder:
    def test_instantiation(self, cfg):
        from data.session_builder import SessionBuilder
        sb = SessionBuilder(cfg["data"])
        assert sb.min_len == cfg["data"]["min_session_len"]
        assert sb.max_len == cfg["data"]["max_session_len"]

    def test_build_raises_not_implemented(self, cfg):
        from data.session_builder import SessionBuilder
        sb = SessionBuilder(cfg["data"])
        with pytest.raises(NotImplementedError):
            sb.build([])


class TestPreprocessor:
    def test_instantiation(self, cfg):
        from data.preprocessor import Preprocessor
        p = Preprocessor(cfg["data"])
        assert p.min_item_freq == cfg["data"]["min_item_freq"]
