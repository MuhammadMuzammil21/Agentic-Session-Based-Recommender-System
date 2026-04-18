"""
evaluate.py — Evaluate a trained ASBRS checkpoint on the test set.

Usage:
    python scripts/evaluate.py [--checkpoint checkpoints/best.pt]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import CFG


def main(checkpoint: str | None = None) -> None:
    cfg = CFG
    ckpt = checkpoint or (Path(cfg["training"]["checkpoint_dir"]) / "best.pt")
    print(f"[evaluate] Loading checkpoint: {ckpt}")
    # Full implementation in Module 05
    raise NotImplementedError("Run Module 05 to implement evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASBRS model")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    main(checkpoint=args.checkpoint)
