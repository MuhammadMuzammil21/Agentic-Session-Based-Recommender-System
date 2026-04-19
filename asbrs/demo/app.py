"""
app.py — Flask web demo for ASBRS.

Endpoints:
  GET  /                  → serve index.html
  POST /recommend         → run full recommendation pipeline
  GET  /health            → health check
"""

from __future__ import annotations

import logging
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

# Ensure package root is importable when running `python demo/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.explainer import RecommendationExplainer
from agent.planner import IntentPlanner
from agent.reranker import IntentReranker
from config.settings import Config
from data.vocab import Vocabulary
from demo.visualizer import AttentionVisualizer
from models.encoder import SessionEncoder
from retrieval.collaborative import ItemBasedCF
from retrieval.content_based import ContentBasedFilter
from retrieval.hybrid import HybridRetriever

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Global resource cache ─────────────────────────────────────────────────────

_cache: Dict[str, Any] = {
    "cfg": None,
    "vocab": None,
    "item_metadata": None,
    "title_to_asin": None,
    "asin_to_title": None,
    "encoder": None,
    "hybrid": None,
    "planner": None,
    "reranker": None,
    "explainer": None,
    "example_sessions": None,
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_config() -> Config:
    """Load and validate config/config.yaml."""
    cfg = Config.load()
    cfg.validate()
    return cfg


def _load_vocab(processed_dir: Path) -> Vocabulary:
    """Load Vocabulary from vocab.json."""
    vocab_path = processed_dir / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")
    return Vocabulary.load(vocab_path)


def _load_item_metadata(processed_dir: Path) -> pd.DataFrame:
    """Load item metadata DataFrame, padding any missing optional columns."""
    for name in ("item_metadata.pkl", "item_metadata.csv"):
        path = processed_dir / name
        if not path.exists():
            continue
        if name.endswith(".pkl"):
            with path.open("rb") as fh:
                df = pickle.load(fh)
        else:
            df = pd.read_csv(path)
        # ContentBasedFilter.fit() requires these 5 columns.
        for col in ("title", "description", "category", "price"):
            if col not in df.columns:
                df[col] = ""
        return df
    raise FileNotFoundError(
        f"No item metadata found in {processed_dir} "
        "(expected item_metadata.pkl or item_metadata.csv)"
    )


def _build_cf_interaction_matrix(
    train_sessions: List[Any],
    vocab: Vocabulary,
) -> csr_matrix:
    """Build a (1 × num_items) sparse interaction matrix from training sessions.

    All session items mapped to a single synthetic user so that CF can compute
    item–item co-occurrence similarity.

    Args:
        train_sessions: Session objects or plain list-of-ASIN-strings.
        vocab: Fitted Vocabulary.

    Returns:
        csr_matrix of shape (1, vocab_size).
    """
    vocab_size = len(vocab)
    row_indices: List[int] = []
    col_indices: List[int] = []

    for s in train_sessions:
        item_strs = s.item_ids if hasattr(s, "item_ids") else list(s)
        # Exclude the last item (leave-one-out held-out target).
        for asin in item_strs[:-1]:
            idx = vocab.encode(asin)
            row_indices.append(0)
            col_indices.append(idx)

    if not row_indices:
        # Fallback: empty 1-row matrix.
        return csr_matrix((1, vocab_size), dtype=np.float32)

    data = np.ones(len(row_indices), dtype=np.float32)
    matrix = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(1, vocab_size),
        dtype=np.float32,
    )
    return matrix


def _load_encoder(
    best_ckpt: Path,
    vocab_size: int,
    cfg: Config,
) -> SessionEncoder:
    """Load SessionEncoder from checkpoint, inferring dims from state_dict.

    Reads embed_dim and hidden_dim directly from the saved weight shapes to
    avoid RuntimeError when the checkpoint was created with different hyper-
    parameters than those currently in config.yaml.

    Args:
        best_ckpt: Path to the .pt checkpoint file.
        vocab_size: Total vocabulary size.
        cfg: Config object (for num_attention_heads fallback).

    Returns:
        SessionEncoder in eval mode with loaded weights.
    """
    payload = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    sd = payload["model_state_dict"]

    embed_dim: int = int(sd["item_embedding.embedding.weight"].shape[1])
    hidden_dim: int = int(sd["gru.weight_hh_l0"].shape[1])

    num_heads = cfg.model.num_attention_heads
    if hidden_dim % num_heads != 0:
        num_heads = 1

    encoder = SessionEncoder(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=0.0,   # inference — disable dropout
        padding_idx=0,
    )
    encoder.load_state_dict(sd)
    encoder.eval()
    logger.info(
        "SessionEncoder loaded: embed=%d hidden=%d heads=%d",
        embed_dim, hidden_dim, num_heads,
    )
    return encoder


def _sample_example_sessions(
    test_sessions: List[Any],
    asin_to_title: Dict[str, str],
    n: int = 5,
    seed: int = 42,
) -> List[List[str]]:
    """Sample n sessions from test set and convert ASINs to titles.

    Args:
        test_sessions: Session objects or plain lists.
        asin_to_title: ASIN → title mapping.
        n: Number of sessions to sample.
        seed: Random seed for reproducibility.

    Returns:
        List of n session-title lists (each ≥3 items excluding target).
    """
    random.seed(seed)
    valid: List[List[str]] = []
    for s in test_sessions:
        item_strs = s.item_ids if hasattr(s, "item_ids") else list(s)
        seed_asins = item_strs[:-1]  # exclude held-out target
        if len(seed_asins) >= 3:
            titles = [asin_to_title.get(a, a) for a in seed_asins]
            valid.append(titles)

    return random.sample(valid, min(n, len(valid)))


# ── Component loader ──────────────────────────────────────────────────────────


def load_components() -> None:
    """Load all ASBRS components into the global cache.

    Idempotent — returns immediately if already loaded.
    """
    if _cache["cfg"] is not None:
        return

    logger.info("Loading demo components …")
    t0 = time.time()

    cfg = _load_config()
    _cache["cfg"] = cfg

    processed_dir = Path(cfg.data.processed_dir)
    ckpt_dir = Path(cfg.training.checkpoint_dir)

    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab = _load_vocab(processed_dir)
    _cache["vocab"] = vocab

    # ── Item metadata ─────────────────────────────────────────────────────────
    df = _load_item_metadata(processed_dir)
    _cache["item_metadata"] = df
    _cache["title_to_asin"] = dict(
        zip(df["title"].astype(str), df["item_id"].astype(str))
    )
    _cache["asin_to_title"] = dict(
        zip(df["item_id"].astype(str), df["title"].astype(str))
    )

    # ── Encoder ───────────────────────────────────────────────────────────────
    pts = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not pts:
        raise FileNotFoundError(f"No *.pt checkpoints found in {ckpt_dir}")
    best_ckpt = max(pts, key=lambda p: p.name)
    _cache["encoder"] = _load_encoder(best_ckpt, len(vocab), cfg)

    # ── Retrievers ────────────────────────────────────────────────────────────
    train_path = processed_dir / "train_sessions.pkl"
    if not train_path.exists():
        raise FileNotFoundError(f"train_sessions.pkl not found in {processed_dir}")
    with train_path.open("rb") as fh:
        train_sessions = pickle.load(fh)

    interaction_matrix = _build_cf_interaction_matrix(train_sessions, vocab)
    cf = ItemBasedCF()
    cf.fit(interaction_matrix)

    cb = ContentBasedFilter()
    cb.fit(df)

    _cache["hybrid"] = HybridRetriever(cf=cf, cb=cb)

    # ── Agent modules ─────────────────────────────────────────────────────────
    reranker = IntentReranker()
    reranker.fit(df)
    _cache["reranker"] = reranker

    _cache["planner"] = IntentPlanner(
        llm_model=cfg.agent.llm_model,
        max_tokens=cfg.agent.llm_max_tokens,
    )
    _cache["explainer"] = RecommendationExplainer()

    # ── Example sessions ──────────────────────────────────────────────────────
    test_path = processed_dir / "test_sessions.pkl"
    if test_path.exists():
        with test_path.open("rb") as fh:
            test_sessions = pickle.load(fh)
        _cache["example_sessions"] = _sample_example_sessions(
            test_sessions,
            _cache["asin_to_title"],
            n=5,
            seed=cfg.project.seed,
        )
    else:
        _cache["example_sessions"] = []

    logger.info("Demo components loaded in %.2f s", time.time() - t0)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.route("/")
def index() -> str:
    """Render the main demo page.

    Loads components on first call (cached thereafter).
    Passes 5 random example sessions to the template.
    """
    if _cache["cfg"] is None:
        load_components()
    return render_template("index.html", example_sessions=_cache["example_sessions"])


@app.route("/health")
def health():
    """Return server health status.

    Returns:
        JSON with ``status`` and ``model_loaded`` flag.
    """
    return jsonify({"status": "ok", "model_loaded": _cache["cfg"] is not None})


@app.route("/recommend", methods=["POST"])
def recommend():
    """Run the full ASBRS recommendation pipeline for a given session.

    Accepts JSON:
        {"session_items": ["Wireless Mouse", "Mechanical Keyboard", ...]}

    Returns JSON:
        {
          "recommendations": [...card dicts...],
          "attention_heatmap": {"labels": [...], "values": [...]},
          "intent": "exploring gaming peripherals"
        }

    On error returns {"error": "..."} with HTTP 400.
    """
    t_start = time.time()

    try:
        if _cache["cfg"] is None:
            load_components()

        data = request.get_json(force=True)
        session_titles: List[str] = data.get("session_items", [])
        if not isinstance(session_titles, list) or not session_titles:
            return jsonify({"error": "session_items must be a non-empty list"}), 400

        cfg: Config = _cache["cfg"]
        vocab: Vocabulary = _cache["vocab"]
        encoder: SessionEncoder = _cache["encoder"]
        hybrid: HybridRetriever = _cache["hybrid"]
        planner: IntentPlanner = _cache["planner"]
        reranker: IntentReranker = _cache["reranker"]
        explainer: RecommendationExplainer = _cache["explainer"]
        item_metadata: pd.DataFrame = _cache["item_metadata"]
        title_to_asin: Dict[str, str] = _cache["title_to_asin"]
        asin_to_title: Dict[str, str] = _cache["asin_to_title"]

        logger.info("/recommend: session_len=%d", len(session_titles))

        # Step 1 — encode titles → integer IDs via vocab
        session_asins = [title_to_asin.get(t, t) for t in session_titles]
        seed_int = [vocab.encode(a) for a in session_asins]

        # Step 2 — run SessionEncoder.forward()
        max_len = cfg.model.max_seq_len
        PAD_IDX = 0
        if len(seed_int) >= max_len:
            padded = seed_int[-max_len:]
            true_len = max_len
        else:
            true_len = len(seed_int)
            padded = [PAD_IDX] * (max_len - true_len) + seed_int

        input_tensor = torch.tensor([padded], dtype=torch.long)
        lengths_tensor = torch.tensor([true_len], dtype=torch.long)

        with torch.no_grad():
            _, attn_weights_batch, _ = encoder(input_tensor, lengths_tensor)

        attn_weights_all: List[float] = attn_weights_batch[0].tolist()
        # Attention covers the full padded window; take only real-token part
        valid_attn = attn_weights_all[-true_len:]
        valid_titles = session_titles[-true_len:]

        # Step 3 — run HybridRetriever.retrieve()
        max_k = max(cfg.evaluation.k_values)
        candidates = hybrid.retrieve(seed_int, max_k, vocab)

        # Step 4 — run IntentPlanner.infer_intent()
        uniform_weights = [1.0 / len(session_titles)] * len(session_titles)
        intent = planner.infer_intent(session_titles, uniform_weights)

        # Step 5 — run IntentReranker.rerank()
        ranked = reranker.rerank(candidates, intent, vocab, item_metadata, max_k)

        # Step 6 — run RecommendationExplainer.format_recommendations()
        # Sort valid_titles/attn by weight descending (explainer expects this order)
        paired = sorted(zip(valid_attn, valid_titles), reverse=True)
        sorted_attn  = [w for w, _ in paired]
        sorted_titles = [t for _, t in paired]

        outputs = explainer.format_recommendations(
            ranked_items=ranked,
            session_items=sorted_titles,
            attn_weights=sorted_attn,
            intent=intent,
        )

        # Step 7 — serialise and return
        cards = AttentionVisualizer.recommendation_cards(outputs)
        heatmap = AttentionVisualizer.heatmap_data(valid_titles, valid_attn)

        elapsed = time.time() - t_start
        logger.info("/recommend: done in %.3f s — %d recs", elapsed, len(cards))

        return jsonify({
            "recommendations": cards,
            "attention_heatmap": heatmap,
            "intent": intent.intent_summary,
        })

    except ValueError as exc:
        logger.error("/recommend ValueError: %s", exc)
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("/recommend unexpected error: %s", exc)
        return jsonify({"error": str(exc)}), 400


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_components()
    cfg = _cache["cfg"]
    print(f"Starting ASBRS demo on http://{cfg.demo.host}:{cfg.demo.port}")
    app.run(host=cfg.demo.host, port=cfg.demo.port, debug=cfg.demo.debug)
