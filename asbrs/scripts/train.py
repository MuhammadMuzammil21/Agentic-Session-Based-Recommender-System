"""
train.py — Train the ASBRS session encoder.

Usage:
    python scripts/train.py [--config path/to/config.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CFG


def main(cfg: dict | None = None) -> None:
    cfg = cfg or CFG
    print(f"[train] Starting training — seed={cfg['project']['seed']}")
    # Full implementation in Module 02
    raise NotImplementedError("Run Module 02 to implement the session encoder training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASBRS session encoder")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.config:
        from config.settings import load_config
        cfg = load_config(args.config)
    else:
        cfg = CFG

    main(cfg)
