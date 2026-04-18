"""
session_builder.py — Build user sessions from raw review interactions.

A session is a temporally contiguous sequence of item interactions by a
single user, bounded by `session_window_hours`.
Implemented in Module 01.
"""

from __future__ import annotations

from typing import List

# Placeholder — full implementation in Module 01


class SessionBuilder:
    """Convert raw user–item interactions into session sequences."""

    def __init__(self, cfg: dict):
        self.min_len = cfg["min_session_len"]
        self.max_len = cfg["max_session_len"]
        self.window_hours = cfg["session_window_hours"]

    def build(self, interactions: List[dict]) -> List[dict]:
        """
        Group interactions into sessions.

        Returns a list of session dicts:
            {
              'user_id': str,
              'session_id': str,
              'items': List[str],        # ordered item IDs
              'timestamps': List[int],   # unix timestamps
              'ratings': List[float],
            }
        """
        raise NotImplementedError("Implemented in Module 01")
