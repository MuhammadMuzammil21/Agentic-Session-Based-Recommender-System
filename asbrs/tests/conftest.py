"""
conftest.py — Shared pytest fixtures for ASBRS test suite.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def cfg():
    """Return the full project config dict."""
    from config.settings import CFG
    return CFG


@pytest.fixture
def tiny_sessions():
    """Return a small synthetic session list for fast unit tests."""
    return [
        {
            "user_id": "U001",
            "session_id": "S001",
            "items": ["A", "B", "C", "D", "E"],
            "timestamps": [1700000000 + i * 3600 for i in range(5)],
            "ratings": [4.0, 3.5, 5.0, 4.0, 3.0],
        },
        {
            "user_id": "U002",
            "session_id": "S002",
            "items": ["B", "C", "F", "G"],
            "timestamps": [1700010000 + i * 3600 for i in range(4)],
            "ratings": [3.0, 4.0, 4.5, 5.0],
        },
    ]
