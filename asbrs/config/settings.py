"""
settings.py — Load and expose typed config from config.yaml.
"""

import os
import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config(path: str | Path = _CONFIG_PATH) -> dict:
    """Load YAML config and return as a nested dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Singleton config object loaded at import time
CFG = load_config()

# ── Convenience accessors ────────────────────────────────────────────────────

PROJECT = CFG["project"]
DATA = CFG["data"]
MODEL = CFG["model"]
TRAINING = CFG["training"]
RETRIEVAL = CFG["retrieval"]
AGENT = CFG["agent"]
EVALUATION = CFG["evaluation"]
DEMO = CFG["demo"]

SEED: int = PROJECT["seed"]
