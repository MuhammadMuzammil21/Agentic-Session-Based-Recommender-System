"""data/session_builder.py — Group user interactions into timed sessions.

A session is a contiguous sequence of interactions by one user where the
gap between consecutive events is <= session_window_hours.
"""

from __future__ import annotations

import logging
import random
from typing import List, Tuple

import pandas as pd

from data.interfaces import Session

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SECONDS_PER_HOUR: int = 3_600


class SessionBuilder:
    """Convert a flat interactions DataFrame into Session objects.

    Example:
        sb = SessionBuilder()
        sessions = sb.build_sessions(df, window_hours=24)
        sessions = sb.filter_sessions(sessions, min_len=3, max_len=50)
        train, val, test = sb.split_sessions(sessions)
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def build_sessions(
        self,
        df: pd.DataFrame,
        window_hours: int,
    ) -> List[Session]:
        """Group each user's interactions into temporally-bounded sessions.

        Interactions are sorted by timestamp. A new session is started
        whenever the gap between consecutive events exceeds `window_hours`.

        Args:
            df:           DataFrame with columns [user_id, item_id, timestamp].
            window_hours: Maximum intra-session gap in hours.

        Returns:
            List of Session objects, ordered by user then session start time.

        Raises:
            ValueError: If required columns are missing.
        """
        required = {"user_id", "item_id", "timestamp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        gap_seconds = window_hours * SECONDS_PER_HOUR
        sessions: List[Session] = []

        for user_id, group in df.groupby("user_id", sort=False):
            group = group.sort_values("timestamp")
            items = group["item_id"].tolist()
            times = group["timestamp"].tolist()

            # Greedy session splitting
            session_items: List[str] = [items[0]]
            session_times: List[int] = [times[0]]

            for i in range(1, len(items)):
                if (times[i] - times[i - 1]) > gap_seconds:
                    sessions.append(
                        Session(
                            user_id=str(user_id),
                            item_ids=session_items,
                            timestamps=session_times,
                        )
                    )
                    session_items = []
                    session_times = []
                session_items.append(items[i])
                session_times.append(times[i])

            # Flush last session
            sessions.append(
                Session(
                    user_id=str(user_id),
                    item_ids=session_items,
                    timestamps=session_times,
                )
            )

        logger.info(
            "build_sessions: %d sessions from %d users (window=%dh)",
            len(sessions),
            df["user_id"].nunique(),
            window_hours,
        )
        return sessions

    def filter_sessions(
        self,
        sessions: List[Session],
        min_len: int,
        max_len: int,
    ) -> List[Session]:
        """Keep only sessions whose length is within [min_len, max_len].

        Args:
            sessions: List of Session objects to filter.
            min_len:  Minimum number of interactions (inclusive).
            max_len:  Maximum number of interactions (inclusive).

        Returns:
            Filtered list of Session objects.

        Raises:
            ValueError: If min_len > max_len.
        """
        if min_len > max_len:
            raise ValueError(
                f"min_len ({min_len}) must be <= max_len ({max_len})"
            )
        kept = [s for s in sessions if min_len <= s.length <= max_len]
        removed = len(sessions) - len(kept)
        logger.info(
            "filter_sessions: kept=%d, removed=%d (bounds=[%d, %d])",
            len(kept),
            removed,
            min_len,
            max_len,
        )
        return kept

    def split_sessions(
        self,
        sessions: List[Session],
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[Session], List[Session], List[Session]]:
        """Randomly split sessions into train / val / test sets.

        Args:
            sessions: Full list of filtered sessions.
            train:    Fraction assigned to training set.
            val:      Fraction assigned to validation set.
            test:     Fraction assigned to test set.
            seed:     Random seed for reproducibility.

        Returns:
            Tuple of (train_sessions, val_sessions, test_sessions).

        Raises:
            ValueError: If splits do not sum to 1.0 (within 1e-6).
        """
        if abs(train + val + test - 1.0) > 1e-6:
            raise ValueError(
                f"train+val+test must equal 1.0, got {train+val+test:.4f}"
            )

        rng = random.Random(seed)
        shuffled = list(sessions)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * train)
        n_val = int(n * val)

        train_set = shuffled[:n_train]
        val_set = shuffled[n_train : n_train + n_val]
        test_set = shuffled[n_train + n_val :]

        logger.info(
            "split_sessions: train=%d, val=%d, test=%d (total=%d)",
            len(train_set),
            len(val_set),
            len(test_set),
            n,
        )
        return train_set, val_set, test_set
