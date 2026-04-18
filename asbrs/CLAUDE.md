# CLAUDE.md — Project State File
## Agentic Session-Based Recommender System (ASBRS)

### Project Overview
Session-based recommender system for Amazon Electronics dataset.
4-module agentic architecture: Memory (GRU+Attention) → Planning (LLM) → Action (Hybrid Retrieval) → Explanation.

### Module Status
| Module | Status | Notes |
|--------|--------|-------|
| Bootstrap | ✓ | |
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
