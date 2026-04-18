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

### 1. Create a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
```

**macOS / Linux**
```bash
python3 -m venv .venv
```

### 2. Activate the virtual environment

**Windows (PowerShell)**
```powershell
.venv\Scripts\Activate.ps1
```

> If you get an execution-policy error, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Windows (Command Prompt)**
```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Your prompt will show `(.venv)` when the environment is active.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the installation

```bash
pytest tests/ -v
```

All tests should pass with no errors.

### Deactivate

```bash
deactivate
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
