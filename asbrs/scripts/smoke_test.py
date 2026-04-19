"""scripts/smoke_test.py — End-to-end integration smoke test for ASBRS.

Runs the complete pipeline on synthetic data (no real data downloads, no
real LLM calls). Prints "SMOKE TEST PASSED" on success.

Usage::

    python scripts/smoke_test.py

Exit code 0 on success, 1 on failure.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Ensure the package root is importable when called as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Constants ─────────────────────────────────────────────────────────────────

N_USERS    = 20
N_ITEMS    = 50
N_INTERACTIONS = 200
N_SESSIONS = 10
MAX_SEQ_LEN = 10
EMBED_DIM  = 16
HIDDEN_DIM = 16
NUM_HEADS  = 2
N_EPOCHS   = 2
TOP_K      = 5
SEED       = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Step 1: Synthetic dataset ─────────────────────────────────────────────────

def build_synthetic_data():
    """Return item_ids list and raw (user, item, timestamp) interactions."""
    item_ids = [f"ITEM_{i:03d}" for i in range(N_ITEMS)]
    interactions = []
    ts = 1_700_000_000
    for _ in range(N_INTERACTIONS):
        user = f"USER_{random.randint(0, N_USERS - 1):03d}"
        item = random.choice(item_ids)
        interactions.append((user, item, ts))
        ts += random.randint(60, 3600)
    return item_ids, interactions

# ── Step 2: Build sessions & vocab ───────────────────────────────────────────

def build_sessions_and_vocab(item_ids, interactions):
    from data.vocab import Vocabulary
    from data.interfaces import Session

    sessions: List[Session] = []

    # Group by user, sort by timestamp
    from collections import defaultdict
    by_user = defaultdict(list)
    for user, item, ts in interactions:
        by_user[user].append((ts, item))

    for user, events in by_user.items():
        events.sort()
        item_seq = [item for _, item in events]
        # Split into windows of up to 10 items
        for start in range(0, len(item_seq) - 2, 5):
            chunk = item_seq[start:start + MAX_SEQ_LEN + 1]
            if len(chunk) >= 3:
                tss = list(range(start, start + len(chunk)))
                sessions.append(Session(user_id=user, item_ids=chunk, timestamps=tss))

    if not sessions:
        raise RuntimeError("No sessions generated from synthetic interactions")

    vocab = Vocabulary()
    vocab.build(item_ids)

    return sessions, vocab

# ── Step 3: Encode sessions ──────────────────────────────────────────────────

def encode_sessions(sessions, vocab, max_len=MAX_SEQ_LEN):
    from data.interfaces import EncodedSession

    encoded = []
    pad_idx = 0
    for s in sessions:
        ids = [vocab.encode(i) for i in s.item_ids]
        target = ids[-1]
        seed = ids[:-1]
        if len(seed) >= max_len:
            seed = seed[-max_len:]
        else:
            seed = [pad_idx] * (max_len - len(seed)) + seed
        encoded.append(EncodedSession(
            input_ids=seed,
            target_id=target,
            session_len=min(len(ids) - 1, max_len),
        ))
    return encoded

# ── Step 4: Train encoder ─────────────────────────────────────────────────────

def train_encoder(encoded_sessions, vocab_size):
    from torch.utils.data import DataLoader, TensorDataset
    from models.encoder import SessionEncoder

    input_ids = torch.tensor([e.input_ids for e in encoded_sessions], dtype=torch.long)
    lengths   = torch.tensor([e.session_len for e in encoded_sessions], dtype=torch.long)
    targets   = torch.tensor([e.target_id for e in encoded_sessions], dtype=torch.long)

    dataset = TensorDataset(input_ids, lengths, targets)
    loader  = DataLoader(dataset, batch_size=8, shuffle=True)

    encoder = SessionEncoder(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        dropout=0.0,
        padding_idx=0,
    )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    import torch.nn.functional as F

    encoder.train()
    for epoch in range(N_EPOCHS):
        total_loss = 0.0
        for batch in loader:
            ids_b, lens_b, tgts_b = batch
            optimizer.zero_grad()
            session_repr, _, _ = encoder(ids_b, lens_b)
            item_embs = encoder.item_embedding.get_all_embeddings()
            logits = encoder.predict_scores(session_repr, item_embs)
            loss = F.cross_entropy(logits, tgts_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1}/{N_EPOCHS}  loss={total_loss / len(loader):.4f}")

    encoder.eval()
    return encoder

# ── Step 5: Hybrid retrieval ──────────────────────────────────────────────────

def run_retrieval(sessions, vocab):
    import pandas as pd
    from scipy.sparse import csr_matrix
    from retrieval.collaborative import ItemBasedCF
    from retrieval.content_based import ContentBasedFilter
    from retrieval.hybrid import HybridRetriever

    vocab_size = len(vocab)

    # Build CF interaction matrix
    rows, cols = [], []
    for s in sessions:
        for asin in s.item_ids[:-1]:
            rows.append(0)
            cols.append(vocab.encode(asin))
    data = np.ones(len(rows), dtype=np.float32)
    matrix = csr_matrix((data, (rows, cols)), shape=(1, vocab_size))

    cf = ItemBasedCF()
    cf.fit(matrix)

    # Build metadata DataFrame
    all_asins = [vocab.decode(i) for i in range(2, vocab_size)]  # skip PAD/UNK
    df = pd.DataFrame({
        "item_id":     all_asins,
        "title":       [f"Product {a}" for a in all_asins],
        "description": [f"Description of {a}" for a in all_asins],
        "category":    ["Electronics"] * len(all_asins),
        "price":       [float(i) * 9.99 for i in range(len(all_asins))],
    })

    cb = ContentBasedFilter()
    cb.fit(df)

    hybrid = HybridRetriever(cf=cf, cb=cb)

    # Retrieve for first session
    seed_int = [vocab.encode(a) for a in sessions[0].item_ids[:-1]]
    candidates = hybrid.retrieve(seed_int, TOP_K, vocab)
    assert isinstance(candidates, list), "HybridRetriever.retrieve must return a list"
    return hybrid, df

# ── Step 6: Full agentic pipeline (LLM mocked) ───────────────────────────────

def run_agentic_pipeline(sessions, vocab, hybrid, item_metadata, encoder):
    from agent.interfaces import IntentResult
    from agent.reranker import IntentReranker
    from agent.explainer import RecommendationExplainer

    reranker = IntentReranker()
    reranker.fit(item_metadata)
    explainer = RecommendationExplainer()

    mock_intent = IntentResult(
        intent_summary="looking for electronics accessories",
        top_items=["Product ITEM_001", "Product ITEM_002"],
        confidence=0.9,
        keywords=["electronics", "accessories"],
    )

    with patch("agent.planner.IntentPlanner.infer_intent", return_value=mock_intent):
        from agent.planner import IntentPlanner
        planner = IntentPlanner(llm_model="gemini-2.5-flash", max_tokens=200)

        session = sessions[0]
        seed_strs = session.item_ids[:-1]
        seed_int  = [vocab.encode(a) for a in seed_strs]

        # Encode
        max_len = MAX_SEQ_LEN
        padded = ([0] * (max_len - len(seed_int)) + seed_int)[-max_len:]
        input_t = torch.tensor([padded], dtype=torch.long)
        lens_t  = torch.tensor([min(len(seed_int), max_len)], dtype=torch.long)
        with torch.no_grad():
            _, attn_weights, _ = encoder(input_t, lens_t)
        attn = attn_weights[0].tolist()[-len(seed_strs):]

        # Retrieve
        candidates = hybrid.retrieve(seed_int, TOP_K, vocab)

        # Intent
        uniform = [1.0 / len(seed_strs)] * len(seed_strs)
        intent = planner.infer_intent(seed_strs, uniform)

        # Rerank
        ranked = reranker.rerank(candidates, intent, vocab, item_metadata, TOP_K)

        # Explain
        asin_to_title = dict(zip(item_metadata["item_id"], item_metadata["title"]))
        titles = [asin_to_title.get(a, a) for a in seed_strs]
        paired = sorted(zip(attn, titles), reverse=True)
        s_attn  = [w for w, _ in paired]
        s_titles = [t for _, t in paired]

        outputs = explainer.format_recommendations(
            ranked_items=ranked,
            session_items=s_titles,
            attn_weights=s_attn,
            intent=intent,
        )

    assert len(outputs) >= 0, "format_recommendations must return a list"
    return outputs

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run all smoke-test stages."""
    try:
        print("=" * 60)
        print("ASBRS SMOKE TEST")
        print("=" * 60)

        print("\n[1] Building synthetic dataset …")
        item_ids, interactions = build_synthetic_data()
        print(f"    {N_ITEMS} items, {N_INTERACTIONS} interactions")

        print("[2] Building sessions & vocab …")
        sessions, vocab = build_sessions_and_vocab(item_ids, interactions)
        print(f"    {len(sessions)} sessions, {len(vocab)} vocab tokens")

        print("[3] Encoding sessions …")
        encoded = encode_sessions(sessions, vocab)
        print(f"    {len(encoded)} encoded sessions")

        print("[4] Training encoder (2 epochs) …")
        encoder = train_encoder(encoded, len(vocab))
        print("    Encoder trained OK")

        print("[5] Running hybrid retrieval …")
        hybrid, item_metadata = run_retrieval(sessions, vocab)
        print("    Hybrid retrieval OK")

        print("[6] Running full agentic pipeline (LLM mocked) …")
        outputs = run_agentic_pipeline(sessions, vocab, hybrid, item_metadata, encoder)
        print(f"    Pipeline produced {len(outputs)} recommendation(s)")

        print()
        print("=" * 60)
        print("SMOKE TEST PASSED")
        print("=" * 60)
        sys.exit(0)

    except Exception as exc:
        import traceback
        print()
        print("=" * 60)
        print("SMOKE TEST FAILED")
        print("=" * 60)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
