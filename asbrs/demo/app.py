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
import torch

# Ensure package root is importable when running `python demo/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.interfaces import RecommendationOutput
from config.settings import Config
from data.vocab import PAD_IDX, UNK_IDX, Vocabulary
from demo.visualizer import AttentionVisualizer
from models.encoder import SessionEncoder

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


def _load_encoder(
    best_ckpt: Path,
    vocab_size: int,
    cfg: Config,
) -> SessionEncoder:
    """Load SessionEncoder from checkpoint, inferring all dims from state_dict.

    Reads vocab_size, embed_dim, and hidden_dim directly from the saved weight
    shapes so the loader is robust to changes in config.yaml and to checkpoints
    built with different vocab sizes (e.g. synthetic smoke-test checkpoints).

    Args:
        best_ckpt:  Path to the .pt checkpoint file.
        vocab_size: Current vocabulary size (used only as fallback assertion).
        cfg:        Config object (for num_attention_heads fallback).

    Returns:
        SessionEncoder in eval mode with loaded weights.

    Raises:
        RuntimeError: If the checkpoint's vocab size does not match *vocab_size*.
    """
    payload = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    sd = payload["model_state_dict"]

    # Infer all architecture dimensions from the saved tensors.
    ckpt_vocab_size: int = int(sd["item_embedding.embedding.weight"].shape[0])
    embed_dim: int = int(sd["item_embedding.embedding.weight"].shape[1])
    hidden_dim: int = int(sd["gru.weight_hh_l0"].shape[1])

    if ckpt_vocab_size != vocab_size:
        raise RuntimeError(
            f"Checkpoint vocab size ({ckpt_vocab_size}) does not match "
            f"current vocabulary ({vocab_size}). "
            "Re-train or re-download the correct checkpoint."
        )

    num_heads = cfg.model.num_attention_heads
    if hidden_dim % num_heads != 0:
        num_heads = 1

    encoder = SessionEncoder(
        vocab_size=ckpt_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=0.0,   # inference — disable dropout
        padding_idx=0,
    )
    encoder.load_state_dict(sd)
    encoder.eval()
    logger.info(
        "SessionEncoder loaded: vocab=%d embed=%d hidden=%d heads=%d",
        ckpt_vocab_size, embed_dim, hidden_dim, num_heads,
    )
    return encoder


def _sample_example_sessions(
    test_sessions: List[Any],
    asin_to_title: Dict[str, str],
    n: int = 5,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Sample n sessions from test set with held-out targets for verification.

    Each example carries both the *input* (items[:-1]) and the *target*
    (the leave-one-out held-out item) so the demo can show "actual next item"
    next to predictions.

    Args:
        test_sessions: Session objects or plain lists.
        asin_to_title: ASIN → title mapping.
        n: Number of sessions to sample.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts: {"input": [titles...], "target": title_str}.
    """
    random.seed(seed)
    valid: List[Dict[str, Any]] = []
    for s in test_sessions:
        item_strs = s.item_ids if hasattr(s, "item_ids") else list(s)
        if len(item_strs) < 4:  # need ≥3 inputs + 1 target
            continue
        seed_asins = item_strs[:-1]
        target_asin = item_strs[-1]
        valid.append({
            "input": [asin_to_title.get(a, a) for a in seed_asins],
            "target": asin_to_title.get(target_asin, target_asin),
        })

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
        title_to_asin: Dict[str, str] = _cache["title_to_asin"]
        asin_to_title: Dict[str, str] = _cache["asin_to_title"]

        logger.info("/recommend: session_len=%d", len(session_titles))

        # Step 1 — encode titles → integer IDs via vocab
        session_asins = [title_to_asin.get(t, t) for t in session_titles]
        seed_int = [vocab.encode(a) for a in session_asins]

        # Step 2 — run SessionEncoder.forward()
        max_len = cfg.model.max_seq_len
        if len(seed_int) >= max_len:
            padded = seed_int[-max_len:]
            true_len = max_len
        else:
            true_len = len(seed_int)
            padded = [PAD_IDX] * (max_len - true_len) + seed_int

        input_tensor = torch.tensor([padded], dtype=torch.long)
        lengths_tensor = torch.tensor([true_len], dtype=torch.long)

        with torch.no_grad():
            session_repr, attn_weights_batch, _ = encoder(
                input_tensor, lengths_tensor
            )

            # Step 3 — score every item with the trained encoder, take top-K.
            scores = encoder.predict_scores(
                session_repr, encoder.item_embedding.embedding.weight
            )[0]
            scores[PAD_IDX] = float("-inf")
            scores[UNK_IDX] = float("-inf")
            # Mask items already in the user's session — recommending an item
            # they just clicked is uninteresting in a demo.
            for sid in seed_int:
                if 0 <= sid < scores.shape[0]:
                    scores[sid] = float("-inf")

            top_k = max(cfg.evaluation.k_values)
            top_scores, top_ids = torch.topk(scores, k=top_k)
            # Convert raw dot-product scores into a softmax probability over
            # just the top-K, so the UI can show interpretable percentages
            # like "28% confidence" instead of unbounded numbers like "2.99".
            top_probs = torch.softmax(top_scores, dim=0)

        attn_weights_all: List[float] = attn_weights_batch[0].tolist()
        valid_attn = attn_weights_all[-true_len:]
        valid_titles = session_titles[-true_len:]

        # Step 4 — build RecommendationOutput objects from top-K.
        outputs: List[RecommendationOutput] = []
        for rank_idx, (item_id, score, prob) in enumerate(
            zip(top_ids.tolist(), top_scores.tolist(), top_probs.tolist())
        ):
            asin = vocab.decode(item_id)
            title = asin_to_title.get(asin, asin) or asin
            # final_score = softmax probability within top-K (sums to 1.0).
            outputs.append(
                RecommendationOutput(
                    rank=rank_idx + 1,
                    item_id=item_id,
                    item_title=title,
                    final_score=float(prob),
                    explanation=(
                        f"Raw GRU+Attention score: {score:+.2f}. "
                        f"This is the model's relative preference for this item "
                        f"versus the other top candidates."
                    ),
                )
            )

        # Step 5 — serialise and return
        cards = AttentionVisualizer.recommendation_cards(outputs)
        heatmap = AttentionVisualizer.heatmap_data(valid_titles, valid_attn)

        elapsed = time.time() - t_start
        logger.info("/recommend: done in %.3f s — %d recs", elapsed, len(cards))

        return jsonify({
            "recommendations": cards,
            "attention_heatmap": heatmap,
            "intent": "GRU + Attention scoring (no intent layer)",
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
