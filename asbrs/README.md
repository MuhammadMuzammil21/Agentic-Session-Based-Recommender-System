# Agentic Session-Based Recommender System (ASBRS)

A session-based recommender system for the Amazon Electronics dataset, built with a 4-module agentic architecture.

## Architecture

```
User Session
    │
    ▼
┌─────────────────────────┐
│  Memory Module           │  GRU + Multi-Head Attention
│  (Session Encoder)       │  → session embedding
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Planning Module         │  Anthropic Claude LLM
│  (Agentic Planner)       │  → intent classification + strategy
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Action Module           │  Collaborative + Content-Based
│  (Hybrid Retrieval)      │  → candidate generation + reranking
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Explanation Module      │  Natural language rationale
│  (Explainer)             │  → user-facing justification
└─────────────────────────┘
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Download and preprocess data
python scripts/download_data.py

# Train the session encoder
python scripts/train.py

# Evaluate
python scripts/evaluate.py

# Launch demo
python demo/app.py
```

## Project Structure

```
asbrs/
├── config/          # Configuration files
├── data/            # Data pipeline (loader, preprocessor, vocab)
├── models/          # Neural model components (embeddings, attention, encoder)
├── retrieval/       # Hybrid retrieval (CF + content-based)
├── agent/           # Agentic planner, reranker, explainer
├── evaluation/      # Metrics, ablation, human eval
├── demo/            # Flask web demo
├── scripts/         # Training and evaluation scripts
├── tests/           # Unit and integration tests
└── checkpoints/     # Saved model weights
```

## Evaluation

- **Protocol**: Leave-one-out
- **Metrics**: Recall@K, MRR@K, HitRate@K for K ∈ {5, 10, 20}
- **Dataset**: Amazon Reviews 2023, Electronics subset

## License

MIT
