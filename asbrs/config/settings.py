"""config/settings.py — Typed configuration dataclasses for ASBRS.

Loads config/config.yaml and exposes nested, validated Config objects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


# ── Sub-configs ───────────────────────────────────────────────────────────────


@dataclass
class ProjectConfig:
    """Top-level project metadata."""

    name: str
    seed: int
    version: str


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    raw_dir: str
    processed_dir: str
    dataset: str
    download_url: str
    hf_dataset_name: str
    hf_category_reviews: str
    hf_category_meta: str
    max_streaming_records: int
    min_session_len: int
    max_session_len: int
    session_window_hours: int
    min_item_freq: int
    train_split: float
    val_split: float
    test_split: float


@dataclass
class ModelConfig:
    """Neural model architecture hyper-parameters."""

    embedding_dim: int
    hidden_dim: int
    num_attention_heads: int
    dropout: float
    max_seq_len: int


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    batch_size: int
    lr: float
    weight_decay: float
    num_epochs: int
    patience: int
    checkpoint_dir: str


@dataclass
class RetrievalConfig:
    """Retrieval module settings."""

    cf_top_k: int
    content_top_k: int
    final_top_k: int


@dataclass
class AgentConfig:
    """Agentic planner settings."""

    llm_model: str
    llm_max_tokens: int
    intent_top_items: int


@dataclass
class EvaluationConfig:
    """Evaluation protocol settings."""

    k_values: List[int]
    num_negatives: int


@dataclass
class DemoConfig:
    """Flask demo server settings."""

    host: str
    port: int
    debug: bool


# ── Root Config ───────────────────────────────────────────────────────────────


@dataclass
class Config:
    """Root configuration object for ASBRS.

    Example:
        cfg = Config.load("config/config.yaml")
        cfg.validate()
    """

    project: ProjectConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    retrieval: RetrievalConfig
    agent: AgentConfig
    evaluation: EvaluationConfig
    demo: DemoConfig

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
        """Load and parse config.yaml into a typed Config instance.

        Args:
            path: Path to the YAML config file.

        Returns:
            Fully populated Config object.

        Raises:
            FileNotFoundError: If the config file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with path.open("r") as fh:
            raw: dict = yaml.safe_load(fh)

        # --- Defensive type coercion -------------------------------------------
        # YAML can parse numeric literals like 1e-5 as strings in some
        # implementations.  Explicitly cast all float fields before handing
        # them to the dataclasses so a bad YAML value raises a clear error here
        # rather than a cryptic TypeError deep inside PyTorch.
        for key in ("lr", "weight_decay"):
            raw["training"][key] = float(raw["training"][key])
        for key in ("train_split", "val_split", "test_split"):
            raw["data"][key] = float(raw["data"][key])
        raw["model"]["dropout"] = float(raw["model"]["dropout"])
        # -----------------------------------------------------------------------

        return cls(
            project=ProjectConfig(**raw["project"]),
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw["model"]),
            training=TrainingConfig(**raw["training"]),
            retrieval=RetrievalConfig(**raw["retrieval"]),
            agent=AgentConfig(**raw["agent"]),
            evaluation=EvaluationConfig(**raw["evaluation"]),
            demo=DemoConfig(**raw["demo"]),
        )

    # ── Validation ────────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If any value is out of an acceptable range.
        """
        d = self.data
        splits_sum = d.train_split + d.val_split + d.test_split
        if abs(splits_sum - 1.0) > 1e-6:
            raise ValueError(
                f"data splits must sum to 1.0, got {splits_sum:.4f}"
            )
        if d.min_session_len < 2:
            raise ValueError(
                f"min_session_len must be >= 2, got {d.min_session_len}"
            )
        if d.min_session_len > d.max_session_len:
            raise ValueError(
                "min_session_len must be <= max_session_len, "
                f"got {d.min_session_len} > {d.max_session_len}"
            )
        if d.min_item_freq < 1:
            raise ValueError(
                f"min_item_freq must be >= 1, got {d.min_item_freq}"
            )

        m = self.model
        if m.hidden_dim % m.num_attention_heads != 0:
            raise ValueError(
                f"hidden_dim ({m.hidden_dim}) must be divisible by "
                f"num_attention_heads ({m.num_attention_heads})"
            )
        if not 0.0 <= m.dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {m.dropout}"
            )

        if not self.evaluation.k_values:
            raise ValueError("evaluation.k_values must be non-empty")

        logger.debug("Config validation passed")
