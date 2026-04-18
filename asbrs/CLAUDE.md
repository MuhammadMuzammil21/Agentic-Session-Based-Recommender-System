# CLAUDE.md — Project State File
## Agentic Session-Based Recommender System (ASBRS)

### Project Overview
Session-based recommender system for Amazon Electronics dataset.
4-module agentic architecture: Memory (GRU+Attention) → Planning (LLM) → Action (Hybrid Retrieval) → Explanation.

### Module Status
| Module | Status | Notes |
|--------|--------|-------|
| Bootstrap | ✓ | |
| 01 Data Pipeline | ✓ | 53/53 tests passing |
| 02 Session Encoder | ✓ | 27/27 tests passing (99 total) |
| 03 Retrieval | ☐ | |
| 04 Agentic Planner | ☐ | |
| 05 Evaluation | ☐ | |
| 06 Demo UI | ☐ | |
| Integration | ☐ | |

### Key Decisions Log
- Dataset: Amazon Reviews 2023, Electronics subset
- Framework: PyTorch for neural components
- LLM API: Anthropic Claude (claude-haiku-3-5-20251001 for cost efficiency)
- Evaluation: leave-one-out, Recall@K + MRR@K + HitRate@K for K in [5,10,20]
- Config: Typed nested dataclasses (Config.load / Config.validate) in config/settings.py
- Vocab: JSON-backed Vocabulary (PAD=0, UNK=1); built from training items only
- Streaming: load_dataset() at module level (enables test mocking via patch)
- Sessions: greedy gap-splitting (window_hours threshold); left-one-out split for train/val/test
- Encoding: left-pad with PAD token; truncate to most-recent max_len items
- Attention: mean-pooled query, multi-head projection, masked softmax (PAD=-1e9)
- Training: cross-entropy over full vocab (no negative sampling); gradient clipping at 1.0
- Checkpoint naming: epoch_{n:03d}_recall{val:.4f}.pt; early stopping on Recall@10

### Data Schema

#### Raw DataFrame (from AmazonDataLoader.stream_reviews)
| Column      | Type    | Description                          |
|-------------|---------|--------------------------------------|
| user_id     | str     | Reviewer identifier                  |
| item_id     | str     | Item ASIN                            |
| rating      | float   | Star rating (1.0–5.0)                |
| timestamp   | int     | Unix epoch (seconds)                 |
| title       | str     | Review title                         |
| description | str     | Review text body                     |
| price       | float?  | Item price (may be None)             |
| category    | str     | Item main category                   |

#### Session (data/interfaces.py)
```python
@dataclass
class Session:
    user_id:    str
    item_ids:   List[str]   # ordered ASINs within the session
    timestamps: List[int]   # corresponding unix timestamps
```

#### EncodedSession (data/interfaces.py)
```python
@dataclass
class EncodedSession:
    input_ids:   List[int]  # left-padded, length == max_seq_len
    target_id:   int        # last item index (leave-one-out target)
    session_len: int        # true length before padding
```

#### DataLoader batch keys
- `"input_ids"` — LongTensor (B, max_seq_len)
- `"lengths"`   — LongTensor (B,)
- `"target"`    — LongTensor (B,)

#### Processed files (data/processed/)
| File                | Contents                      |
|---------------------|-------------------------------|
| vocab.json          | item2idx mapping (JSON)       |
| train_sessions.pkl  | List[Session]                 |
| val_sessions.pkl    | List[Session]                 |
| test_sessions.pkl   | List[Session]                 |
| encoded_train.pkl   | List[EncodedSession]          |
| encoded_val.pkl     | List[EncodedSession]          |
| encoded_test.pkl    | List[EncodedSession]          |

### Model Architecture

#### SessionEncoder Pipeline
```
input_ids [B, L]
    → ItemEmbedding (Xavier init, PAD zeroed, dropout)        → [B, L, D]
    → PackedGRU (batch_first=True)                            → [B, L, H]
    → SelfAttentionLayer (mean-query, masked softmax)         → session_repr [B, H]
                                                              → attn_weights  [B, L]

predict_scores: session_repr [B, H] × item_embs [V, D]ᵀ      → scores [B, V]
```

#### Key Dimensions
| Symbol | Meaning              | Config key              |
|--------|----------------------|-------------------------|
| B      | Batch size           | training.batch_size     |
| L      | Max sequence length  | model.max_seq_len       |
| D      | Embedding dim        | model.embedding_dim     |
| H      | Hidden (GRU) dim     | model.hidden_dim        |
| V      | Vocabulary size      | (from vocab.json)       |

#### Training
- **Loss**: Cross-entropy over full vocabulary
- **Optimiser**: Adam (lr, weight_decay from config)
- **Grad clip**: max_norm=1.0
- **Early stopping**: patience on val Recall@10
- **Checkpoint**: `checkpoints/epoch_{n:03d}_recall{val:.4f}.pt`

#### Evaluation
- Recall@K and MRR@K for K in cfg.evaluation.k_values

### File Paths & Config
- Raw data: data/raw/
- Processed: data/processed/
- Checkpoints: checkpoints/
- Config: config/config.yaml
- Vocab: data/processed/vocab.json
