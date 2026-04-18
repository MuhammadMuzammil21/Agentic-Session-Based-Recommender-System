"""
download_data.py — Download and cache Amazon Electronics dataset via HF streaming.

Usage:
    python scripts/download_data.py [--max-records N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CFG


def main(max_records: int | None = None) -> None:
    """Stream and locally cache processed Amazon Electronics data."""
    cfg = CFG["data"]
    if max_records is not None:
        cfg = dict(cfg)
        cfg["max_streaming_records"] = max_records

    print(f"[download_data] Streaming up to {cfg['max_streaming_records']:,} reviews…")
    # Full implementation in Module 01
    raise NotImplementedError(
        "Run Module 01 to implement the streaming data pipeline."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Amazon Electronics data")
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    main(max_records=args.max_records)
