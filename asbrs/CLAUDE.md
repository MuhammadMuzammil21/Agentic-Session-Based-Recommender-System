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
| 03 Retrieval | ✓ | 35/35 tests passing |
| 04 Agentic Planner | ✓ | 28/28 tests passing |
| 05 Evaluation | ✓ | 46/46 tests passing |
| 06 Demo UI | ✓ | Flask app with Vanilla JS frontend implemented |
| Integration | ✓ | smoke_test.py passes; 189/189 tests passing |

### Key Decisions Log
- Dataset: Amazon Reviews 2023, Electronics subset
- Framework: PyTorch for neural components
- LLM API: Gemini 2.5 Flash via google-genai SDK (aistudio.google.com — free, no credit card)
- Evaluation: leave-one-out, Recall@K + MRR@K + HitRate@K for K in [5,10,20]
- Config: Typed nested dataclasses (Config.load / Config.validate) in config/settings.py
- Vocab: JSON-backed Vocabulary (PAD=0, UNK=1); built from training items only
- Streaming: load_dataset() at module level (enables test mocking via patch)
- Sessions: greedy gap-splitting (window_hours threshold); left-one-out split for train/val/test
- Encoding: left-pad with PAD token; truncate to most-recent max_len items
- Attention: mean-pooled query, multi-head projection, masked softmax (PAD=-1e9)
- Training: cross-entropy over full vocab (no negative sampling); gradient clipping at 1.0
- Checkpoint naming: epoch_{n:03d}_recall{val:.4f}.pt; early stopping on Recall@10
- CF: Item-based cosine similarity via sklearn (csr_matrix); aggregated column-sum scoring
- CB: TF-IDF on (title + description + category); np.matrix → np.asarray fix for sklearn 1.6+
- Hybrid: linear score fusion (cf_weight · cf_score + cb_weight · cb_score); items in only one source get 0 from the other; legacy stubs retained as CollaborativeRetriever/ContentBasedRetriever/_LegacyHybridRetriever
- LLM: google-genai SDK (google.genai.Client); deprecated google-generativeai replaced
- IntentPlanner: Gemini 2.5 Flash; JSON prompt → {intent, keywords, confidence}; frozenset cache; markdown-fence strip; default on any parse/API error
- IntentReranker: TF-IDF on titles; final_score = 0.6·retrieval + 0.4·intent_similarity
- RecommendationExplainer: template-only (no LLM); attention_heatmap top-5; RecommendationOutput rank is 1-based
- agent/interfaces.py: IntentResult, RankedItem, RecommendationOutput dataclasses (contracts only)
- evaluation/metrics.py: Pure-function module; new signature recall_at_k(recommended, relevant, k); legacy evaluate_session/aggregate_metrics retained for backwards compat; added evaluate_model(predictions, k_values) and coverage(all_recommendations, catalog_size)
- evaluation/ablation.py: AblationStudy replaces AblationRunner stub; 4 variants: popularity baseline, CF-only, GRU+attention (no LLM), full agentic; run_all() returns DataFrame with columns [Model|Recall@5|Recall@10|Recall@20|MRR@10|HitRate@10]; save_results() writes CSV + Markdown via tabulate
- evaluation/human_eval.py: HumanEvalExporter replaces HumanEvalBuilder stub; generate_eval_sheet() samples n_sessions, validates aligned input lengths, writes self-contained HTML with inline CSS, CSS-only 5-star rating fields, no external JS/font dependencies
- scripts/evaluate.py: Full CLI with argparse; auto-detects best checkpoint; runs ablation; saves CSV/Markdown; generates HTML human-eval sheet; prints final summary line
- tabulate>=0.9.0 added to requirements.txt (needed for df.to_markdown())
- demo/visualizer.py: AttentionVisualizer.heatmap_data() normalises weights to sum 1.0; recommendation_cards() truncates title at 60 chars with ellipsis
- demo/app.py: global _cache dict; load_components() is idempotent; encoder dims inferred from state_dict shapes (not config) to survive checkpoint/config mismatch; ItemBasedCF.fit() receives csr_matrix built from train sessions; ContentBasedFilter.fit() pads missing columns; RecommendationExplainer.format_recommendations() takes (ranked, session_items, attn_weights, intent); request time logged; /health returns model_loaded flag
- demo/templates/index.html: 3-panel layout (session input | recommendations | attention + intent); vanilla JS fetch(); CSS-only spinner; inline SVG-style bar chart via DIV flex rows; navy #1F3864 + light-blue #93c5fd colour scheme; no external CSS frameworks


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

### Final Notes (Spring 2026)

#### Known Limitations
- `scripts/smoke_test.py` uses random weights; metrics are meaningless without a real trained checkpoint.
- CF similarity matrix is empty when only 1 synthetic user is provided — retrieval degrades to CB-only in that case.
- IntentPlanner occasionally returns a truncated JSON string from Gemini when `max_tokens` is too low; the planner falls back to a default IntentResult silently.
- `data/processed/` files must be generated by running `download_data.py` before `train.py` or `evaluate.py` can be used with real data.

#### How to Resume
1. Read this CLAUDE.md to recover state.
2. Run `pytest tests/ -q` to confirm all 189 tests still pass.
3. Run `python scripts/smoke_test.py` to confirm the pipeline is wired.
4. If continuing integration work: check the `/Integration` row is ✓.
5. Next natural step: `python scripts/download_data.py` then `python scripts/train.py` to produce a real trained checkpoint, then `python scripts/evaluate.py` to fill in the results table in README.md.
