# Claude Code Master Prompt
## Agentic Session-Based Recommender System (ASBRS)
### CS-4053 Recommender Systems — NUCES-FAST, Spring 2026

---

## HOW TO USE THIS PROMPT

This file is your single source of truth for the entire project. It is structured into
**phases**. At the start of every Claude Code session, paste **only the sections you need**
for that session — do not paste the entire file at once. This keeps token usage low and
context focused.

**Recommended session order:**
1. Paste `BOOTSTRAP` → sets up the repo skeleton and CLAUDE.md state file
2. Paste `MODULE 01` → data pipeline
3. Paste `MODULE 02` → session encoder
4. Paste `MODULE 03` → retrieval layer
5. Paste `MODULE 04` → agentic planner
6. Paste `MODULE 05` → evaluation
7. Paste `MODULE 06` → demo UI
8. Paste `INTEGRATION` → wire everything together + final checks

Each module section is **self-contained**. Claude Code reads CLAUDE.md at the start of
each session to recover full project context without you re-explaining anything.

---

## GLOBAL RULES (paste these at the top of EVERY session)

```
GLOBAL RULES — read before writing any code:

1. ALWAYS read `CLAUDE.md` at the start of this session to recover project state.
   Update it at the end of every session with what was completed and any decisions made.

2. MODULARITY: One responsibility per file. No file exceeds 300 lines.
   If a function grows beyond 40 lines, split it.

3. INTERFACES FIRST: Define dataclasses and abstract base classes before
   implementing any logic. Modules talk to each other through these contracts only.

4. TYPE HINTS: Every function signature must have full type hints.
   Use `from __future__ import annotations` at the top of every file.

5. DOCSTRINGS: Every public class and function gets a Google-style docstring.
   One-liners are fine for simple helpers.

6. LOGGING: Use `logging.getLogger(__name__)` — never use print() in library code.
   print() is allowed only in scripts (train.py, evaluate.py, demo/app.py).

7. CONFIGURATION: All hyperparameters, paths, and API keys go in `config/config.yaml`.
   Load via the `Config` dataclass in `config/settings.py`. Never hardcode values.

8. REPRODUCIBILITY: Set seeds at entry points:
   random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

9. TESTING: Write a pytest test for every public function.
   Tests go in `tests/test_<module_name>.py`. Run `pytest -x` after each module.

10. CHECKPOINTING: Save model state and vocabulary after every training epoch.
    Name checkpoints: `checkpoints/epoch_{n:03d}_recall{val:.4f}.pt`

11. ERROR HANDLING: Use specific exceptions (ValueError, FileNotFoundError, etc.).
    Never use bare `except:`. Always log the error before re-raising.

12. IMPORTS: Standard library → third-party → local, separated by blank lines.
    No wildcard imports (`from x import *`).

13. CONSTANTS: All uppercase, defined at module top or in constants.py.

14. COMMIT CHECKPOINTS: After each module is tested and working, print
    "MODULE X COMPLETE — updating CLAUDE.md" and update the state file.
```

---

## BOOTSTRAP (Run this once at the start of the project)

```
TASK: Bootstrap the full project repository structure.

Create the following directory tree and files:

asbrs/
├── CLAUDE.md                    ← project state file (see template below)
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py
│   ├── config.yaml
│   └── settings.py
├── data/
│   ├── __init__.py
│   ├── loader.py
│   ├── session_builder.py
│   ├── preprocessor.py
│   └── vocab.py
├── models/
│   ├── __init__.py
│   ├── embeddings.py
│   ├── attention.py
│   └── encoder.py
├── retrieval/
│   ├── __init__.py
│   ├── collaborative.py
│   ├── content_based.py
│   └── hybrid.py
├── agent/
│   ├── __init__.py
│   ├── planner.py
│   ├── reranker.py
│   └── explainer.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   ├── ablation.py
│   └── human_eval.py
├── demo/
│   ├── __init__.py
│   ├── app.py
│   ├── visualizer.py
│   └── templates/
│       └── index.html
├── scripts/
│   ├── download_data.py
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_retrieval.py
│   ├── test_agent.py
│   └── test_evaluation.py
└── checkpoints/
    └── .gitkeep

─────────────────────────────────────────────
CLAUDE.md TEMPLATE (create this file exactly):
─────────────────────────────────────────────

# CLAUDE.md — Project State File
## Agentic Session-Based Recommender System (ASBRS)

### Project Overview
Session-based recommender system for Amazon Electronics dataset.
4-module agentic architecture: Memory (GRU+Attention) → Planning (LLM) → Action (Hybrid Retrieval) → Explanation.

### Module Status
| Module | Status | Notes |
|--------|--------|-------|
| Bootstrap | ☐ | |
| 01 Data Pipeline | ☐ | |
| 02 Session Encoder | ☐ | |
| 03 Retrieval | ☐ | |
| 04 Agentic Planner | ☐ | |
| 05 Evaluation | ☐ | |
| 06 Demo UI | ☐ | |
| Integration | ☐ | |

### Key Decisions Log
(Append here as decisions are made)
- Dataset: Amazon Reviews 2023, Electronics subset
- Framework: PyTorch for neural components
- LLM API: Anthropic Claude (claude-haiku-3-5-20251001 for cost efficiency)
- Evaluation: leave-one-out, Recall@K + MRR@K + HitRate@K for K in [5,10,20]

### Data Schema
(Fill in after Module 01)

### Model Architecture
(Fill in after Module 02)

### File Paths & Config
- Raw data: data/raw/
- Processed: data/processed/
- Checkpoints: checkpoints/
- Config: config/config.yaml

─────────────────────────────────────────────
requirements.txt CONTENTS:
─────────────────────────────────────────────

torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0
anthropic>=0.25.0
flask>=3.0.0
tqdm>=4.66.0
pyyaml>=6.0.1
pytest>=7.4.0
pytest-cov>=4.1.0
datasets>=2.18.0

─────────────────────────────────────────────
config/config.yaml CONTENTS:
─────────────────────────────────────────────

project:
  name: "ASBRS"
  seed: 42
  version: "1.0.0"

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  dataset: "amazon_electronics_2023"
  download_url: "https://amazon-reviews-2023.github.io"
  hf_dataset_name: "McAuley-Lab/Amazon-Reviews-2023"
  hf_category_reviews: "raw_review_Electronics"
  hf_category_meta: "raw_meta_Electronics"
  max_streaming_records: 500000   # adjust based on RAM
  min_session_len: 3
  max_session_len: 50
  session_window_hours: 24
  min_item_freq: 5
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1

model:
  embedding_dim: 64
  hidden_dim: 128
  num_attention_heads: 4
  dropout: 0.2
  max_seq_len: 50

training:
  batch_size: 256
  lr: 0.001
  weight_decay: 1e-5
  num_epochs: 20
  patience: 5
  checkpoint_dir: "checkpoints"

retrieval:
  cf_top_k: 100
  content_top_k: 100
  final_top_k: 20

agent:
  llm_model: "claude-haiku-3-5-20251001"
  llm_max_tokens: 200
  intent_top_items: 3

evaluation:
  k_values: [5, 10, 20]
  num_negatives: 99

demo:
  host: "0.0.0.0"
  port: 5000
  debug: false

After creating all files, run:
  cd asbrs && pip install -r requirements.txt && pytest tests/ -v

Update CLAUDE.md: mark Bootstrap as ✓
```

---

## MODULE 01 — Data Pipeline

```
CONTEXT: Read CLAUDE.md first. We are implementing Module 01: Data Pipeline.
The goal is a clean, testable pipeline from raw Amazon Reviews JSON to
PyTorch-ready session tensors.

─────────────────────────────────────────────
File: config/settings.py
─────────────────────────────────────────────
Implement a `Config` dataclass that loads config/config.yaml using PyYAML.
Use nested dataclasses for each top-level key (DataConfig, ModelConfig, etc.).
Provide a `Config.load(path: str) -> Config` classmethod.
Include a `Config.validate()` method that raises ValueError for invalid values.

─────────────────────────────────────────────
File: data/loader.py
─────────────────────────────────────────────
Implement `AmazonDataLoader` with these public methods:

  stream_reviews(category: str, max_records: int, cfg: Config) -> pd.DataFrame
    Uses HuggingFace `datasets` with streaming=True so the full dataset is
    NEVER downloaded to disk. Loads only `max_records` rows from the specified
    category (default: cfg.data.hf_category_reviews).
    Returns df with columns:
    [user_id, item_id, rating, timestamp, title, description, price, category]
    Logs shape and dtypes after streaming completes.

    Reference implementation:

    ```python
    from datasets import load_dataset

    def stream_reviews(self, category: str, max_records: int) -> pd.DataFrame:
        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            category,
            streaming=True,
            trust_remote_code=True
        )
        records = []
        for item in ds['full'].take(max_records):
            records.append({
                'user_id':   item['user_id'],
                'item_id':   item['asin'],
                'rating':    item['rating'],
                'timestamp': item['timestamp'],
                'title':     item.get('title', ''),
                'description': item.get('description', ''),
                'price':     item.get('price', None),
                'category':  item.get('main_category', ''),
            })
        return pd.DataFrame(records)
    ```

    NOTE: `download()` and `load_reviews(filepath)` are REMOVED. All raw
    data ingestion goes through `stream_reviews()` exclusively.

  filter_interactions(df: pd.DataFrame, min_item_freq: int) -> pd.DataFrame
    Removes items appearing fewer than min_item_freq times.
    Logs items removed.

─────────────────────────────────────────────
File: data/session_builder.py
─────────────────────────────────────────────
Implement `SessionBuilder` with:

  build_sessions(df: pd.DataFrame, window_hours: int) -> List[Session]
    Groups each user's interactions chronologically.
    Splits into sessions using a window_hours gap threshold.
    A `Session` is a dataclass: {user_id, item_ids: List[str], timestamps: List[int]}

  filter_sessions(sessions: List[Session], min_len: int, max_len: int) -> List[Session]
    Keeps only sessions within length bounds. Logs kept/removed count.

  split_sessions(sessions, train=0.8, val=0.1, test=0.1, seed=42)
      -> Tuple[List[Session], List[Session], List[Session]]
    Stratified split. Returns (train, val, test). Logs split sizes.

─────────────────────────────────────────────
File: data/vocab.py
─────────────────────────────────────────────
Implement `Vocabulary` with:
  - Special tokens: PAD=0, UNK=1
  - build(item_ids: List[str]) -> None
  - encode(item_id: str) -> int
  - decode(idx: int) -> str
  - save(path: Path) -> None
  - load(path: Path) -> Vocabulary  (classmethod)
  - __len__() -> int

─────────────────────────────────────────────
File: data/preprocessor.py
─────────────────────────────────────────────
Implement `SessionPreprocessor` with:

  prepare(sessions: List[Session], vocab: Vocabulary, max_len: int)
      -> List[EncodedSession]
    Encodes item_ids to integers. Truncates to max_len (keep most recent).
    Pads shorter sessions with PAD token (left-pad).
    An `EncodedSession` is a dataclass:
      {input_ids: List[int], target_id: int, session_len: int}
    (input_ids = all but last item, target_id = last item)

  to_dataloader(encoded: List[EncodedSession], batch_size: int, shuffle: bool)
      -> DataLoader
    Returns a PyTorch DataLoader. Each batch is a dict:
      {"input_ids": LongTensor[B, L], "lengths": LongTensor[B], "target": LongTensor[B]}

─────────────────────────────────────────────
File: scripts/download_data.py
─────────────────────────────────────────────
CLI script. Accepts --config path argument.
Runs the full data pipeline end to end:
  load → filter → build sessions → split → build vocab → preprocess → save
Saves processed data to data/processed/ as pickle files.
Prints a summary table at the end.

─────────────────────────────────────────────
File: tests/test_data.py
─────────────────────────────────────────────
Write pytest tests for every public method above.
Use synthetic data (small DataFrames, no real downloads).
Tests must run offline — mock any HTTP calls with unittest.mock.patch.

AFTER COMPLETING: run `pytest tests/test_data.py -v`
Fix any failures before marking done.
Update CLAUDE.md: mark Module 01 as ✓, fill in Data Schema section.
```

---

## MODULE 02 — Session Encoder

```
CONTEXT: Read CLAUDE.md first. Module 01 is complete.
We are implementing Module 02: the GRU + Self-Attention session encoder.

─────────────────────────────────────────────
File: models/embeddings.py
─────────────────────────────────────────────
Implement `ItemEmbedding(nn.Module)`:
  __init__(vocab_size: int, embed_dim: int, padding_idx: int = 0)
  forward(x: Tensor) -> Tensor  # [B, L] -> [B, L, D]
  Includes dropout. Uses xavier_uniform_ init on weight.

─────────────────────────────────────────────
File: models/attention.py
─────────────────────────────────────────────
Implement `SelfAttentionLayer(nn.Module)`:

  __init__(hidden_dim: int, num_heads: int, dropout: float)

  forward(hidden_states: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]
    hidden_states: [B, L, H]
    mask: [B, L] — 1 for real tokens, 0 for padding
    Returns:
      context: [B, H]        ← weighted sum (the session representation)
      weights: [B, L]        ← attention weights (used for explainability)

    Implementation:
      - Compute query as mean of non-padded hidden states
      - Compute key/value projections
      - Scaled dot-product attention with mask applied before softmax
        (set padding positions to -1e9 before softmax)
      - Output is weighted sum of values

─────────────────────────────────────────────
File: models/encoder.py
─────────────────────────────────────────────
Implement `SessionEncoder(nn.Module)`:

  __init__(vocab_size: int, embed_dim: int, hidden_dim: int,
           num_heads: int, dropout: float, padding_idx: int = 0)

  forward(input_ids: Tensor, lengths: Tensor)
      -> Tuple[Tensor, Tensor, Tensor]
    input_ids: [B, L]
    lengths:   [B]
    Returns:
      session_repr: [B, H]   ← final session representation
      attn_weights: [B, L]   ← attention weights per item (for explainability)
      all_hiddens:  [B, L, H] ← all GRU hidden states

    Implementation:
      1. Embed input_ids → [B, L, D]
      2. Pack padded sequence → run GRU
      3. Unpack → all_hiddens: [B, L, H]
      4. Run SelfAttentionLayer(all_hiddens, mask) → session_repr, attn_weights
      5. Return all three

  predict_scores(session_repr: Tensor, item_embeddings: Tensor) -> Tensor
    session_repr:    [B, H]
    item_embeddings: [V, D]  ← all item embeddings from embedding layer
    Returns scores:  [B, V]  ← dot product similarity to all items

Implement `NextItemTrainer`:

  __init__(encoder: SessionEncoder, vocab_size: int, cfg: Config)

  train_epoch(dataloader: DataLoader, optimizer, device) -> float
    Returns mean loss for the epoch. Uses cross-entropy on next-item prediction.
    Logs batch loss every 100 steps.

  evaluate(dataloader: DataLoader, device, k_values: List[int]) -> Dict[str, float]
    Returns dict of Recall@K and MRR@K for each K.

  save_checkpoint(epoch: int, val_recall: float, path: str) -> None
  load_checkpoint(path: str) -> None

─────────────────────────────────────────────
File: scripts/train.py
─────────────────────────────────────────────
CLI training script. Arguments: --config, --device (cpu/cuda), --resume.
Loads data from data/processed/.
Runs train loop for cfg.training.num_epochs with early stopping.
Saves best checkpoint based on val Recall@10.
Prints epoch summary table each epoch.

─────────────────────────────────────────────
File: tests/test_models.py
─────────────────────────────────────────────
Test every module with small synthetic tensors (B=4, L=8, V=50).
Verify:
  - output shapes are correct
  - attention weights sum to 1.0 per sample (within tolerance)
  - padding positions have near-zero attention weight
  - model is trainable (loss decreases after one backward pass)

AFTER COMPLETING: run `pytest tests/test_models.py -v`
Update CLAUDE.md: mark Module 02 as ✓, fill in Model Architecture section.
```

---

## MODULE 03 — Retrieval Layer

```
CONTEXT: Read CLAUDE.md first. Modules 01–02 are complete.
We are implementing Module 03: Hybrid Retrieval (CF + Content-Based).

─────────────────────────────────────────────
File: retrieval/collaborative.py
─────────────────────────────────────────────
Implement `ItemBasedCF`:

  fit(interaction_matrix: csr_matrix) -> None
    interaction_matrix: users × items sparse matrix (implicit feedback).
    Computes item–item cosine similarity matrix. Stores as csr_matrix.
    Logs time taken and matrix density.

  get_candidates(item_ids: List[int], top_k: int) -> List[Tuple[int, float]]
    Given a list of recently interacted item IDs, returns top_k candidate
    items sorted by aggregated similarity score (sum of column similarities).
    Excludes items already in item_ids.

  save(path: Path) -> None
  load(path: Path) -> ItemBasedCF  (classmethod)

─────────────────────────────────────────────
File: retrieval/content_based.py
─────────────────────────────────────────────
Implement `ContentBasedFilter`:

  fit(item_metadata: pd.DataFrame) -> None
    item_metadata has columns: [item_id, title, description, category, price].
    Builds TF-IDF matrix on (title + " " + description + " " + category).
    Logs vocab size and matrix shape.

  get_candidates(item_ids: List[int], top_k: int,
                 vocab: Vocabulary) -> List[Tuple[int, float]]
    Decodes item_ids to strings, looks up TF-IDF rows, computes cosine
    similarity to all items, returns top_k (excluding input items).

  save(path: Path) -> None
  load(path: Path) -> ContentBasedFilter  (classmethod)

─────────────────────────────────────────────
File: retrieval/hybrid.py
─────────────────────────────────────────────
Implement `HybridRetriever`:

  __init__(cf: ItemBasedCF, cb: ContentBasedFilter,
           cf_weight: float = 0.5, cb_weight: float = 0.5)

  retrieve(item_ids: List[int], top_k: int,
           vocab: Vocabulary) -> List[Tuple[int, float]]
    Fetches cf_top_k from CF and cb_top_k from CB.
    Merges scores: score = cf_weight * cf_score + cb_weight * cb_score
    Returns top_k by merged score. Logs candidate pool size.

  retrieve_batch(sessions: List[List[int]], top_k: int,
                 vocab: Vocabulary) -> List[List[Tuple[int, float]]]
    Batch version with tqdm progress bar.

─────────────────────────────────────────────
File: tests/test_retrieval.py
─────────────────────────────────────────────
Use synthetic sparse matrix (50 users × 100 items) and dummy metadata DataFrame.
Test:
  - CF returns correct number of candidates, no duplicates, no input items
  - CB returns correct number, TF-IDF properly weighted
  - Hybrid score is correctly merged
  - save/load round-trips produce identical results

AFTER COMPLETING: run `pytest tests/test_retrieval.py -v`
Update CLAUDE.md: mark Module 03 as ✓.
```

---

## MODULE 04 — Agentic Planner

```
CONTEXT: Read CLAUDE.md first. Modules 01–03 are complete.
We are implementing Module 04: the Agentic Planning, Re-ranking, and Explanation layer.
This is the novel core of the project — handle it carefully.

─────────────────────────────────────────────
File: agent/planner.py
─────────────────────────────────────────────
Implement `IntentPlanner`:

  __init__(llm_model: str, max_tokens: int)
    Initialises the Anthropic client. Reads ANTHROPIC_API_KEY from env.

  infer_intent(session_items: List[str],
               attention_weights: List[float]) -> IntentResult
    session_items: decoded item titles of the top attended items
    attention_weights: corresponding attention weights

    Returns `IntentResult` dataclass:
      {intent_summary: str, top_items: List[str], confidence: float}

    Prompt strategy (keep it short to save tokens):
      - Provide the top 3 attended items with their attention weights
      - Ask for: 1-sentence purchase intent + 3 keywords + confidence 0–1
      - Request response as JSON: {"intent": "...", "keywords": [...], "confidence": 0.0}
      - Parse JSON response; on parse failure log warning and return a default
      - Cache results in a dict keyed by frozenset(session_items) to avoid
        redundant API calls during evaluation

  batch_infer(sessions: List[List[str]],
              weights: List[List[float]]) -> List[IntentResult]
    Calls infer_intent with a small sleep between calls to avoid rate limits.
    Logs API call count and cache hit rate at the end.

─────────────────────────────────────────────
File: agent/reranker.py
─────────────────────────────────────────────
Implement `IntentReranker`:

  __init__(encoder_model: sentence_transformers or simple TF-IDF)
    Use scikit-learn TF-IDF (no extra dependency) for semantic similarity.
    Fit on item titles in fit() method.

  fit(item_metadata: pd.DataFrame) -> None

  rerank(candidates: List[Tuple[int, float]],
         intent: IntentResult,
         vocab: Vocabulary,
         item_metadata: pd.DataFrame,
         top_k: int) -> List[RankedItem]

    Returns List[RankedItem] dataclass:
      {item_id: int, item_title: str, retrieval_score: float,
       intent_score: float, final_score: float}

    final_score = 0.6 * retrieval_score + 0.4 * intent_score
    Sorted descending by final_score. Returns top_k.

─────────────────────────────────────────────
File: agent/explainer.py
─────────────────────────────────────────────
Implement `RecommendationExplainer`:

  generate_explanation(ranked_item: RankedItem,
                       session_items: List[str],
                       attn_weights: List[float],
                       intent: IntentResult) -> str
    Builds a human-readable explanation string WITHOUT calling the LLM:
    "Recommended because your session shows strong interest in {top_item}
     ({top_weight:.0%}) and {second_item} ({second_weight:.0%}),
     suggesting you are {intent.intent_summary}."

  format_recommendations(ranked_items: List[RankedItem],
                         session_items: List[str],
                         attn_weights: List[float],
                         intent: IntentResult) -> List[RecommendationOutput]

    Returns List[RecommendationOutput] dataclass:
      {rank: int, item_id: int, item_title: str, final_score: float,
       explanation: str, attention_heatmap: Dict[str, float]}

    attention_heatmap maps item_title -> attention_weight for the top 5
    session items (used by the demo visualizer).

─────────────────────────────────────────────
File: tests/test_agent.py
─────────────────────────────────────────────
Mock all Anthropic API calls using unittest.mock.patch.
Test:
  - IntentPlanner returns IntentResult with correct fields on valid JSON response
  - IntentPlanner returns default IntentResult on malformed API response (no crash)
  - Cache prevents duplicate API calls for identical sessions
  - Reranker final_score formula is correctly applied
  - Explainer produces a non-empty explanation string containing the top item name

AFTER COMPLETING: run `pytest tests/test_agent.py -v`
Update CLAUDE.md: mark Module 04 as ✓.
```

---

## MODULE 05 — Evaluation

```
CONTEXT: Read CLAUDE.md first. Modules 01–04 are complete.
We are implementing Module 05: Evaluation (metrics, ablation, human eval export).

─────────────────────────────────────────────
File: evaluation/metrics.py
─────────────────────────────────────────────
Implement these as pure functions (no class needed):

  recall_at_k(recommended: List[int], relevant: int, k: int) -> float
    Returns 1.0 if relevant is in recommended[:k], else 0.0.

  mrr_at_k(recommended: List[int], relevant: int, k: int) -> float
    Returns 1/rank if relevant is in recommended[:k], else 0.0.

  hit_rate_at_k(recommended: List[int], relevant: int, k: int) -> float
    Same as recall_at_k for a single relevant item.

  evaluate_model(
      predictions: List[Tuple[List[int], int]],
      k_values: List[int]
  ) -> Dict[str, float]
    predictions: list of (recommended_item_ids, ground_truth_item_id)
    Returns {"Recall@5": ..., "MRR@5": ..., "HitRate@5": ...,
             "Recall@10": ..., ...} for all k_values.

  coverage(all_recommendations: List[List[int]], catalog_size: int) -> float
    Returns fraction of catalog that appears in at least one recommendation list.

─────────────────────────────────────────────
File: evaluation/ablation.py
─────────────────────────────────────────────
Implement `AblationStudy`:

  __init__(test_sessions, vocab, item_metadata, cfg)

  run_popularity_baseline() -> Dict[str, float]
    Recommends the globally most popular items. Evaluates with evaluate_model.

  run_cf_only() -> Dict[str, float]
    Uses only ItemBasedCF retrieval, no re-ranking.

  run_gru_attention() -> Dict[str, float]
    Uses full SessionEncoder but no LLM planner (skips intent re-ranking).

  run_full_agentic() -> Dict[str, float]
    Uses complete ASBRS pipeline.

  run_all() -> pd.DataFrame
    Runs all four and returns a formatted comparison DataFrame.
    Columns: Model | Recall@5 | Recall@10 | Recall@20 | MRR@10 | HitRate@10
    Logs each model as it runs.

  save_results(df: pd.DataFrame, path: Path) -> None
    Saves as CSV. Also saves a markdown table to path.with_suffix('.md').

─────────────────────────────────────────────
File: evaluation/human_eval.py
─────────────────────────────────────────────
Implement `HumanEvalExporter`:

  generate_eval_sheet(
      sessions: List[List[str]],
      recommendations: List[List[RecommendationOutput]],
      intents: List[IntentResult],
      n_sessions: int = 10
  ) -> None
    Picks n_sessions random sessions from the test set.
    Produces a human-readable HTML file at evaluation/human_eval_sheet.html.

    Format per session:
      - Session items (numbered list)
      - Inferred intent summary
      - Top 5 recommendations with explanations
      - Rating fields: Relevance (1-5), Explanation Quality (1-5)
    Opens cleanly in a browser. CSS is inline for portability.

─────────────────────────────────────────────
File: scripts/evaluate.py
─────────────────────────────────────────────
CLI script that:
  1. Loads best checkpoint
  2. Runs AblationStudy.run_all()
  3. Prints comparison table to console
  4. Saves results to evaluation/ablation_results.csv
  5. Generates human_eval_sheet.html
  6. Prints final summary: "Best model: Full Agentic | Recall@10: X.XXXX"

─────────────────────────────────────────────
File: tests/test_evaluation.py
─────────────────────────────────────────────
Test with synthetic prediction lists.
Verify:
  - Recall@K = 1.0 when relevant item is in top-K
  - Recall@K = 0.0 when relevant item is not in top-K
  - MRR@K = 1.0 when relevant is rank 1, 0.5 when rank 2
  - Coverage fraction is in [0, 1]
  - evaluate_model returns all expected keys

AFTER COMPLETING: run `pytest tests/test_evaluation.py -v`
Update CLAUDE.md: mark Module 05 as ✓.
```

---

## MODULE 06 — Demo UI

```
CONTEXT: Read CLAUDE.md first. Modules 01–05 are complete.
We are implementing Module 06: the interactive Flask demo.

─────────────────────────────────────────────
File: demo/visualizer.py
─────────────────────────────────────────────
Implement `AttentionVisualizer`:

  heatmap_data(session_items: List[str],
               attn_weights: List[float]) -> Dict
    Returns a dict ready for JSON serialisation:
      {"labels": [...item titles...], "values": [...weights...]}
    Normalises weights to sum to 1.0.

  recommendation_cards(outputs: List[RecommendationOutput]) -> List[Dict]
    Returns a list of dicts for the front-end template:
      [{"rank": 1, "title": "...", "score": "0.87",
        "explanation": "...", "intent_badge": "..."}]
    Truncates title to 60 chars with ellipsis if longer.

─────────────────────────────────────────────
File: demo/app.py
─────────────────────────────────────────────
Flask app with these routes:

  GET /
    Renders index.html. Loads 5 random example sessions from test set
    on first request (cached in memory). Returns them as session suggestions.

  POST /recommend
    Accepts JSON: {"session_items": ["item_title_1", "item_title_2", ...]}
    Pipeline:
      1. Encode item titles → item IDs via vocab
      2. Run SessionEncoder.forward()
      3. Run HybridRetriever.retrieve()
      4. Run IntentPlanner.infer_intent()
      5. Run IntentReranker.rerank()
      6. Run RecommendationExplainer.format_recommendations()
      7. Return JSON with recommendations + attention heatmap + intent

    On error: return {"error": "message"} with HTTP 400.
    Log request time.

  GET /health
    Returns {"status": "ok", "model_loaded": true/false}

─────────────────────────────────────────────
File: demo/templates/index.html
─────────────────────────────────────────────
Clean, single-page HTML/CSS/JS demo. No external CSS frameworks.
Layout:
  - Header: "ASBRS — Session-Based Recommender" + team names
  - Left panel: "Your Session" — input field to add items + example session buttons
  - Centre panel: "Recommendations" — card list with rank, title, score, explanation badge
  - Right panel: "Session Attention" — simple bar chart drawn with inline SVG
                 "Purchase Intent" — inferred intent shown as a highlighted pill

All API calls via fetch(). Show a loading spinner during /recommend.
Use only vanilla JS — no jQuery, no React.
Colour scheme: white + navy (#1F3864) + light blue accents.

AFTER COMPLETING:
  Run `python demo/app.py` and verify:
  - / loads without error
  - POST /recommend returns valid JSON for a 3-item session
  - Attention heatmap renders in the browser
Update CLAUDE.md: mark Module 06 as ✓.
```

---

## INTEGRATION (Final session)

```
CONTEXT: Read CLAUDE.md first. All 6 modules should be ✓.
This session wires everything together, fixes integration issues,
and prepares the final submission.

─────────────────────────────────────────────
TASK 1: Integration smoke test
─────────────────────────────────────────────
Write scripts/smoke_test.py that runs the complete pipeline end-to-end
on 10 synthetic sessions without any real data download.
It should:
  1. Create a tiny fake dataset (20 users, 50 items, 200 interactions)
  2. Build sessions and vocab
  3. Train the encoder for 2 epochs
  4. Run hybrid retrieval
  5. Mock the LLM API (use unittest.mock)
  6. Run the full agentic pipeline
  7. Print "SMOKE TEST PASSED" if no exceptions raised

─────────────────────────────────────────────
TASK 2: Full test suite
─────────────────────────────────────────────
Run: pytest tests/ -v --cov=. --cov-report=term-missing
Fix any failing tests. Target: all tests pass, coverage > 70%.

─────────────────────────────────────────────
TASK 3: README.md
─────────────────────────────────────────────
Write a clear README.md with these sections:
  - Project description (2 sentences)
  - Architecture diagram (ASCII art of the 4 modules)
  - Installation (pip install -r requirements.txt)
  - Quick start: download data → train → evaluate → demo
  - Project structure (directory tree)
  - Evaluation results table (placeholder — fill after real run)
  - Team: Daaim Ali Shiekh (22k-4363), Muhammad Muzammil (22k-4267)
  - Course: CS-4053 Recommender Systems, NUCES-FAST, Spring 2026

─────────────────────────────────────────────
TASK 4: Final CLAUDE.md update
─────────────────────────────────────────────
Mark Integration as ✓.
Add final notes: any known limitations, how to resume if needed.

─────────────────────────────────────────────
TASK 5: Submission checklist
─────────────────────────────────────────────
Verify the following and print a ✓ or ✗ for each:
  [ ] All 6 modules implemented
  [ ] pytest tests/ passes with 0 failures
  [ ] scripts/smoke_test.py passes
  [ ] scripts/train.py runs without error on small synthetic data
  [ ] scripts/evaluate.py produces ablation_results.csv
  [ ] demo/app.py serves the UI on localhost:5000
  [ ] No hardcoded API keys anywhere in the codebase
  [ ] requirements.txt is accurate and complete
  [ ] README.md is complete
  [ ] CLAUDE.md is up to date
```

---

## QUICK REFERENCE — Session Starters

Copy-paste one of these at the start of each new Claude Code session:

**Starting a fresh module:**
```
Read CLAUDE.md. We are starting [MODULE NAME].
Apply the GLOBAL RULES throughout.
Do not proceed to implementation until you have confirmed:
1. Which modules are already complete (from CLAUDE.md)
2. The key data schemas and interfaces from completed modules
Then implement exactly what the MODULE prompt specifies.
```

**Resuming after an interruption:**
```
Read CLAUDE.md. We were in the middle of [MODULE NAME].
The last completed step was: [describe].
Continue from where we left off. Apply GLOBAL RULES.
Do not re-implement already-completed files.
```

**Debugging a failing test:**
```
Read CLAUDE.md. tests/test_[name].py is failing.
Error: [paste error here]
Do not change the test — fix the implementation.
After fixing, run pytest tests/test_[name].py -v and confirm it passes.
```

---

## TOKEN BUDGET GUIDE

| Session | Estimated tokens consumed | Notes |
|---------|--------------------------|-------|
| Bootstrap | ~3,000 | Structure only, no logic |
| Module 01 | ~6,000 | Most boilerplate-heavy |
| Module 02 | ~7,000 | Core neural code |
| Module 03 | ~5,000 | Mostly sklearn ops |
| Module 04 | ~6,000 | LLM integration |
| Module 05 | ~5,000 | Pure functions + ablation |
| Module 06 | ~6,000 | Flask + HTML |
| Integration | ~4,000 | Wiring + tests |
| **Total** | **~42,000** | Well within Claude Code limits |

Tip: If a session is getting long, finish the current file, update CLAUDE.md,
end the session, and start a fresh one for the next file.
