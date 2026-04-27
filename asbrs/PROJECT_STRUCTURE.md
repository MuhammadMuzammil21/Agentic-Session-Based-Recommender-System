# ASBRS — Project Structure & Flow

A plain-English guide to what's in this project, where it lives, and how the
pieces fit together.

---

## 1. What this project is, in two sentences

We have a small neural network (called a **session encoder**) that learns to
predict what Amazon Electronics product a shopper will click next, based on
the items they've already clicked in the same shopping visit. We train it on
real Amazon reviews, evaluate it against simpler baselines, and serve it
through a small web demo.

---

## 2. Quick glossary

You'll see these words a lot. Here's what they mean in this project:

| Word | Plain meaning |
|---|---|
| **ASIN** | Amazon's product ID — a 10-character string like `B08N5WRWNW`. Every product has one. |
| **Session** | One user's run of clicks with no gap longer than 24 hours. Like one "shopping visit." |
| **Item / Item ID** | Same as ASIN. We sometimes also use an **integer** id (e.g. `42`) — the model only understands integers, so we map every ASIN to a number. |
| **Vocabulary (vocab)** | The dictionary that maps each ASIN to its integer id and back. Stored in `vocab.json`. |
| **Embedding** | A list of 64 numbers that represents one item. Items that get clicked together will end up with similar embeddings after training. |
| **PAD / UNK** | Special tokens. `PAD` (id 0) is empty padding. `UNK` (id 1) means "an item the model has never seen." |
| **GRU** | A type of recurrent neural network. It reads a sequence one step at a time and remembers context. |
| **Attention** | A layer that scores how important each past item is for predicting the next one. The score is what the demo's bar chart shows. |
| **Tensor** | PyTorch's name for a multi-dimensional array. `[256, 50]` means 256 rows of 50 numbers. |
| **Batch** | A group of N samples processed together for speed. Default 256. |
| **Recall@K, MRR@K** | Metrics. Recall@10 = "did we put the correct item in the top 10?" MRR = same but rewards being near the top. |
| **Checkpoint** | A saved snapshot of the trained model's weights. Files like `epoch_007_recall0.0470.pt`. |
| **HuggingFace** | A website that hosts datasets and ML models. We stream the Amazon reviews from there — no manual download. |

---

## 3. Folder-by-folder map

```
asbrs/
├── config/          configuration & typed loader
├── data/            data pipeline (stream → sessions → encode)
├── models/          neural network parts
├── retrieval/       classic (non-neural) recommenders
├── evaluation/      metrics, ablation runner, HTML eval sheet
├── demo/            Flask web app
├── scripts/         CLI entry points (download, train, evaluate, smoke)
├── tests/           pytest unit tests
├── checkpoints/     saved model weights
├── data/processed/  preprocessed pickles + vocab.json + item_metadata.pkl
├── evaluation/      output: ablation_results.csv/md, human_eval_sheet.html
└── requirements.txt
```

### `config/` — all the knobs in one place

| File | What it does |
|---|---|
| `config.yaml` | Every hyperparameter (learning rate, batch size, embedding dim, max session length, etc.). Edit this when you want to change behaviour. |
| `settings.py` | Loads the YAML into typed Python dataclasses, validates them. So you can write `cfg.training.lr` instead of digging through dicts. |

### `data/` — the data pipeline

This is where raw Amazon reviews become tensors the model can train on.

| File | What it does |
|---|---|
| `interfaces.py` | Two small dataclasses: `Session` (a user + their list of items) and `EncodedSession` (the same thing but as integer ids, ready for the model). |
| `loader.py` | `AmazonDataLoader` — connects to HuggingFace and streams the raw review JSONL file row by row. Also has a method to fetch real product titles from the metadata file. |
| `session_builder.py` | `SessionBuilder` — groups raw rows by user, splits them into sessions whenever the gap between clicks is longer than 24 hours, then filters short/long sessions and splits into train/val/test. |
| `preprocessor.py` | `SessionPreprocessor` — takes Session objects, looks up integer ids via the vocabulary, pads them to a fixed length, and wraps them in a PyTorch DataLoader (which feeds the model in batches). |
| `vocab.py` | `Vocabulary` — the ASIN ↔ integer mapping. Built only from training items, saved as `vocab.json`. |

After running `download_data.py` you'll see new files inside `data/processed/`:

| File | What it contains |
|---|---|
| `vocab.json` | The ASIN → integer mapping. |
| `train_sessions.pkl` | List of `Session` objects for training. |
| `val_sessions.pkl` | Same, for validation (used during training to decide when to stop). |
| `test_sessions.pkl` | Same, held out for final evaluation. |
| `encoded_train.pkl` | Training sessions already converted to integer ids and padded. |
| `encoded_val.pkl` / `encoded_test.pkl` | Same for val/test. |
| `item_metadata.pkl` | DataFrame with real product titles, descriptions, categories — used by the demo and human-eval HTML for human-readable display. |

### `models/` — the neural network

| File | What it does |
|---|---|
| `embeddings.py` | `ItemEmbedding` — a lookup table of `[vocab_size × 64]`. Given an integer id, returns its 64-number vector. Initialised randomly with the Xavier algorithm, then trained. |
| `attention.py` | `SelfAttentionLayer` — multi-head attention. Takes the GRU's hidden states for every step in the sequence and produces (a) one summary vector for the whole session, and (b) per-item attention weights (the bar-chart numbers). |
| `encoder.py` | Two classes: `SessionEncoder` (the full network: embedding → GRU → attention → score) and `NextItemTrainer` (the training loop with Adam optimiser, gradient clipping, and validation evaluation). |

### `retrieval/` — non-neural baseline

Classic algorithm used as a comparison point in the ablation study.

| File | What it does |
|---|---|
| `collaborative.py` | `ItemBasedCF` — "users who clicked X also clicked Y." Builds a sparse item × item cosine-similarity matrix from training sessions. Used by the **CF Only** ablation variant. |

### `evaluation/` — measuring how well it works

| File | What it does |
|---|---|
| `metrics.py` | Pure functions: `recall_at_k`, `mrr_at_k`, `hit_rate_at_k`, plus a batch helper `evaluate_model` that averages them across many sessions. |
| `ablation.py` | `AblationStudy` — runs three model variants (Popularity, CF Only, GRU + Attention) on the test set and produces a comparison DataFrame. |
| `human_eval.py` | `HumanEvalExporter` — writes a self-contained HTML page where humans can rate the model's recommendations on a 5-star scale. |

After running `evaluate.py` you'll see:

| File | What it contains |
|---|---|
| `evaluation/ablation_results.csv` | The 3-row metrics table, machine-readable. |
| `evaluation/ablation_results.md` | Same table as Markdown — paste into `README.md`. |
| `evaluation/human_eval_sheet.html` | Standalone HTML for human raters; shows N test sessions with the model's top-5 picks and a star widget. |

### `demo/` — the Flask web app

| File | What it does |
|---|---|
| `app.py` | The Flask server. On startup it loads the trained encoder + vocab + item metadata. Has two routes: `GET /` serves the HTML page, `POST /recommend` runs the model and returns top-K JSON. |
| `visualizer.py` | Small helpers that turn `RecommendationOutput` objects and attention weights into JSON the frontend can render (recommendation cards + bar-chart data). |
| `templates/index.html` | Single-page UI with three panels: session input (left), recommendations (centre), attention chart (right). Plain HTML + vanilla JavaScript — no React, no build step. |

### `scripts/` — what you actually run from the command line

| Command | What it does |
|---|---|
| `python scripts/download_data.py` | Downloads & preprocesses everything. Writes the files in `data/processed/`. |
| `python scripts/train.py` | Trains the model. Writes checkpoints to `checkpoints/`. |
| `python scripts/evaluate.py` | Runs the 3-variant ablation and produces the CSV/MD/HTML outputs. |
| `python scripts/smoke_test.py` | A quick end-to-end sanity check using random weights and synthetic data. Confirms the wiring works without needing real data. |
| `python demo/app.py` | Starts the Flask demo on `http://localhost:5000`. |

### `tests/`

`pytest` unit tests for every module. Run all of them with:
```bash
pytest tests/ -q
```
Should report **161 passed**.

---

## 4. The full flow, end-to-end

Here is what happens, step by step, when you run the project from scratch.

### Step 1 — `python scripts/download_data.py`

This script does seven things in sequence (see `download_data.py:main()`):

1. **Stream reviews from HuggingFace.** `AmazonDataLoader.stream_reviews()` opens
   a streaming connection to the Amazon Reviews 2023 dataset and pulls up to
   `max_streaming_records` rows (default 500 000) into a pandas DataFrame.
   No file is saved to disk during this — it's a live HTTP stream.
   Each row has fields like `user_id`, `asin`, `parent_asin`, `rating`,
   `timestamp`, plus the *review's* title and body.

2. **Drop rare items.** `filter_interactions()` removes any item that appears
   fewer than `min_item_freq` times (default 15). Rare items are noise — the
   model can't learn meaningful embeddings for items it sees once or twice.

3. **Fetch product metadata.** A second HuggingFace stream pulls product
   *titles* and *descriptions* from the `meta_Electronics.jsonl` file. The
   reviews dataset is keyed by `asin` (child variant) but metadata is keyed by
   `parent_asin` (umbrella product), so we look up each child's parent and
   fan the metadata back out. Result is saved as `data/processed/item_metadata.pkl`.

4. **Build sessions.** `SessionBuilder.build_sessions()` sorts each user's
   rows by timestamp, then walks through them and starts a new session
   whenever the gap is longer than 24 hours. Each session is one `Session`
   object: `{user_id, item_ids, timestamps}`.

5. **Filter sessions.** Drop sessions that are too short (< 3 items, no
   prediction signal) or too long (> 50 items, probably bots).

6. **Split into train / val / test.** Random shuffle with seed=42, then
   80% / 10% / 10%. Saved as three pickle files.

7. **Build the vocabulary** (only from training items, no leakage), then
   **encode** each session using leave-one-out:
   - `input_ids` = all items except the last, padded on the LEFT to length 50
   - `target_id` = the last item (held out for prediction)
   - `session_len` = the true length before padding

   Result: six pickle files — three split-level lists of `Session`s and three
   matching lists of `EncodedSession`s.

After this script you should have a populated `data/processed/` folder and a
console summary table showing how many sessions per split and the vocabulary size.

### Step 2 — `python scripts/train.py`

The training script (`train.py:main()`):

1. Loads vocab and the encoded train/val splits.
2. Wraps them in PyTorch `DataLoader`s with `batch_size=256`.
3. Builds a fresh `SessionEncoder` (random weights).
4. Creates an Adam optimiser with `lr=0.001` and a `ReduceLROnPlateau`
   scheduler that halves the learning rate after 2 stagnant epochs.
5. Loops up to 30 epochs:
   - For each batch: forward pass → compute cross-entropy loss against the
     target item → backpropagate → clip gradients to max-norm 1.0 → step the
     optimiser.
   - At the end of each epoch, evaluate on the validation set. Save the
     weights to `checkpoints/epoch_NNN_recallX.XXXX.pt` if Recall@10 improved.
   - Stop early if validation Recall@10 hasn't improved for 10 epochs.

The forward pass inside the encoder (`models/encoder.py:SessionEncoder.forward`):
```
input_ids [B, 50]
   → ItemEmbedding   →  [B, 50, 64]
   → packed-GRU      →  [B, 50, 128]    (hidden state per step)
   → SelfAttention   →  session_repr [B, 128]
                        attn_weights [B, 50]
```
Then `predict_scores(session_repr, all_item_embeddings)` returns one score per
item in the vocabulary. The cross-entropy loss compares those scores against
the integer id of the true target.

When the script ends, you'll see:
```
Best Recall@10 = 0.0470
Training complete.
```

### Step 3 — `python scripts/evaluate.py`

This runs the **ablation study** — three different recommenders compared on
the same test set.

Inside `AblationStudy.run_all()`, three variants run in sequence:

1. **Popularity Baseline** — count the most-clicked items across test
   sessions and recommend the top-K to *every* user. No personalisation, no
   training. The floor.

2. **CF Only** — fit `ItemBasedCF` on the test sessions, compute item × item
   cosine similarity, and for each session sum the columns of the items the
   user clicked → take the top-K most similar items. Pure 1990s technique.

3. **GRU + Attention** — load your trained checkpoint, encode each test
   session, multiply `session_repr` against all item embeddings → take the
   top-K highest-scoring items. PAD and UNK are masked out.

Each variant produces a `Dict[str, float]` of metric values which are
collected into a DataFrame and saved as CSV + Markdown.

Then the script generates a **human-eval HTML sheet** by encoding 10 random
test sessions through the trained model and writing the top-5 predictions
into a self-contained HTML file with star-rating widgets.

Final console output:
```
========================================================================
              Model  Recall@5  Recall@10  Recall@20   MRR@10  HitRate@10
Popularity Baseline    0.0221     0.0478     0.0653   0.0117      0.0478
            CF Only    0.0110     0.0212     0.0331   0.0065      0.0212
    GRU + Attention    0.0294     0.0423     0.0718   0.0206      0.0423
========================================================================

Best model: <whichever wins Recall@10>
```

### Step 4 — `python demo/app.py`

The Flask demo wires everything for an interactive browser experience.

On startup (`load_components()`):
1. Loads vocab, item metadata, and the best checkpoint.
2. Samples 5 random test sessions to use as one-click examples (each
   includes the held-out target item so you can verify predictions).

When a user submits a session via the UI (`POST /recommend`):
1. Convert each typed item title into an ASIN, then to an integer id.
2. Pad to length 50 and wrap as a tensor.
3. Run the encoder forward pass → `session_repr` + attention weights.
4. Score every item with `predict_scores()`. Mask out PAD, UNK, and items
   already in the user's session.
5. Take the top-20. Apply softmax over those 20 scores so they become
   readable probabilities (summing to 1.0).
6. Look up each top-20 item's integer id in the vocab → ASIN → product title.
7. Build a JSON response with three pieces:
   - `recommendations`: list of cards (rank, title, percent score, explanation)
   - `attention_heatmap`: per-input-item attention weights (the bar chart)
   - `intent`: a short label string identifying the scoring method

The frontend (`index.html`) renders the JSON into three panels and, if the
session came from an example button, highlights any recommendation that
matches the held-out item with a green ✓ MATCH badge.

### Step 5 (optional) — `python scripts/smoke_test.py`

A self-contained sanity test. It builds synthetic sessions and a synthetic
encoder, runs the entire pipeline (encode → retrieve → recommend) without
needing any real data or checkpoint, and asserts everything wires together.
Recommendations from this are meaningless (random weights) but the test
confirms no module is broken.

---

## 5. The headline files to read first

If you only have time to read a few files to understand the project, read
these in order:

1. **`config/config.yaml`** — see all the knobs.
2. **`scripts/download_data.py`** — the data pipeline as a linear story.
3. **`models/encoder.py`** — the entire neural network in ~200 lines.
4. **`scripts/train.py`** — the training loop.
5. **`evaluation/ablation.py`** — how the three variants are compared.
6. **`demo/app.py`** — the inference path used by the demo.

The rest of the codebase is helpers and tests.

---

## 6. Common questions

**Q: Does the model use product titles when training?**
No. The model only sees integer ids. Titles are purely for displaying things
to humans (the demo and the human-eval HTML).

**Q: Why do scores in the demo look like "10.0%" instead of probabilities like 0.05?**
We softmax-normalise the top-20 raw scores so they sum to 1.0, then display
as a percentage. So 10.0% means *"of the model's preference among the top
20 candidates, 10% goes to this one."* It's not the probability over the
full 6 988 vocabulary — that would be tiny and meaningless.

**Q: Why does the attention chart sometimes show equal weights?**
Because the attention layer hasn't found a strong differential signal in
your session — either the session is very short, or the trained model
genuinely treats those items roughly equally. With longer sessions and more
training data you'll see clearer 60/30/10-style splits.

**Q: Do I need a GPU?**
No. CPU is fine for everything except very long training. PyTorch will use
a GPU automatically if one is available.

**Q: Does the project make any external API calls?**
No. Everything runs locally. The only network activity is the initial
HuggingFace stream when you run `download_data.py` to pull the Amazon
dataset.
