# Agentic Session-Based Recommender System (ASBRS)

An end-to-end session-based recommender system for the Amazon Electronics dataset, combining a GRU + multi-head attention session encoder with a Gemini-powered agentic planner that infers purchase intent and re-ranks candidates for personalised, explainable recommendations.

## Architecture

```
User Session  (ordered item interactions)
      │
      ▼
┌─────────────────────────────────────────┐
│  MODULE 01 · Data Pipeline              │
│  AmazonDataLoader → SessionBuilder      │
│  → Vocabulary → EncodedSession          │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  MODULE 02 · Session Encoder (Memory)   │
│  ItemEmbedding → PackedGRU              │
│  → SelfAttentionLayer → session_repr    │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  MODULE 03 · Hybrid Retrieval (Action)  │
│  ItemBasedCF + ContentBasedFilter       │
│  → HybridRetriever (score fusion)       │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  MODULE 04 · Agentic Planner            │
│  IntentPlanner (Gemini 2.5 Flash)       │
│  → IntentReranker → RecommendationExp.  │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  MODULE 05 · Evaluation                 │
│  Recall@K · MRR@K · HitRate@K          │
│  AblationStudy · HumanEvalExporter      │
└─────────────────────────────────────────┘

Flask Demo (Module 06) ties all modules together
```

## Installation

```bash
# Clone / unzip the project
git clone <repo_url>
cd asbrs

# Create a virtual environment
python -m venv .venv

# Activate — Windows
.venv\Scripts\Activate.ps1
# Activate — macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file (or export in your shell) with your Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

Get a free key at <https://aistudio.google.com>.

## Quick Start

```bash
# 1. Download & preprocess the Amazon Electronics dataset
python scripts/download_data.py

# 2. Train the session encoder
python scripts/train.py

# 3. Evaluate all four model variants (ablation study)
python scripts/evaluate.py

# 4. Launch the interactive demo
python demo/app.py
#    → open http://localhost:5000 in your browser

# (Optional) Run the end-to-end smoke test without real data
python scripts/smoke_test.py
```

## Project Structure

```
asbrs/
├── config/
│   ├── config.yaml          # All hyperparameters and paths
│   └── settings.py          # Typed Config dataclasses
├── data/
│   ├── interfaces.py        # Session, EncodedSession dataclasses
│   ├── loader.py            # AmazonDataLoader (HuggingFace streaming)
│   ├── preprocessor.py      # SessionBuilder, leave-one-out splitter
│   └── vocab.py             # Vocabulary (ASIN ↔ int, PAD/UNK)
├── models/
│   ├── embeddings.py        # ItemEmbedding with Xavier init
│   ├── attention.py         # SelfAttentionLayer (multi-head)
│   └── encoder.py           # SessionEncoder + NextItemTrainer
├── retrieval/
│   ├── collaborative.py     # ItemBasedCF (cosine similarity / csr_matrix)
│   ├── content_based.py     # ContentBasedFilter (TF-IDF)
│   └── hybrid.py            # HybridRetriever (linear score fusion)
├── agent/
│   ├── interfaces.py        # IntentResult, RankedItem, RecommendationOutput
│   ├── planner.py           # IntentPlanner (Gemini 2.5 Flash)
│   ├── reranker.py          # IntentReranker (TF-IDF + intent cosine)
│   └── explainer.py         # RecommendationExplainer (template-based)
├── evaluation/
│   ├── metrics.py           # recall_at_k, mrr_at_k, hit_rate_at_k, coverage
│   ├── ablation.py          # AblationStudy (4 variants)
│   └── human_eval.py        # HumanEvalExporter → portable HTML sheet
├── demo/
│   ├── app.py               # Flask web application
│   ├── visualizer.py        # AttentionVisualizer (JSON payloads)
│   └── templates/
│       └── index.html       # Single-page vanilla JS UI
├── scripts/
│   ├── download_data.py     # Data download & preprocessing CLI
│   ├── train.py             # Training CLI
│   ├── evaluate.py          # Evaluation + ablation CLI
│   └── smoke_test.py        # End-to-end integration smoke test
├── tests/                   # pytest test suite (189 tests)
├── checkpoints/             # Saved model weights
├── requirements.txt
└── README.md
```

## Evaluation Results

Results below are obtained via `scripts/evaluate.py` on the Amazon Electronics 2023 test set (leave-one-out protocol).

| Model | Recall@5 | Recall@10 | Recall@20 | MRR@10 | HitRate@10 |
|---|---|---|---|---|---|
| Popularity Baseline | — | — | — | — | — |
| CF Only | — | — | — | — | — |
| GRU + Attention | — | — | — | — | — |
| **Full Agentic (ASBRS)** | — | — | — | — | — |

> **Note:** Run `python scripts/evaluate.py` after training on the full dataset to populate this table.

## Team

| Name | Student ID |
|---|---|
| Daaim Ali Shiekh | 22k-4363 |
| Muhammad Muzammil | 22k-4267 |

**Course:** CS-4053 Recommender Systems  
**Institution:** NUCES-FAST  
**Semester:** Spring 2026

## License

MIT
