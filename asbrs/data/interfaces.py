"""data/interfaces.py — Shared dataclasses (contracts) for the data pipeline.

All modules in the data package communicate through these types only.
No logic lives here — interfaces only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Session:
    """A temporally-bounded sequence of item interactions for one user.

    Attributes:
        user_id:    Unique user identifier.
        item_ids:   Ordered list of item ASINs within the session.
        timestamps: Unix timestamps (seconds) corresponding to each item.
    """

    user_id: str
    item_ids: List[str]
    timestamps: List[int]

    def __post_init__(self) -> None:
        if len(self.item_ids) != len(self.timestamps):
            raise ValueError(
                "item_ids and timestamps must have the same length, "
                f"got {len(self.item_ids)} vs {len(self.timestamps)}"
            )

    @property
    def length(self) -> int:
        """Number of interactions in the session."""
        return len(self.item_ids)


@dataclass
class EncodedSession:
    """A pre-processed, integer-encoded session ready for model input.

    Attributes:
        input_ids:   Integer-encoded item sequence (all but last item).
                     Left-padded with PAD token to max_len.
        target_id:   Integer index of the target item (last in session).
        session_len: True length of input_ids before padding.
    """

    input_ids: List[int]
    target_id: int
    session_len: int
