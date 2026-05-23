# ASBRS — Viva Guide

A practical guide for the viva: what to run, what comes out, and how to
read it. Plus a section on the GRU since that's the part most likely to
get asked about.

---

## Part 1 — Project overview in one paragraph

We built a **session-based recommender system**. A user's recent
interactions in one shopping session (clicks, reviews) form a sequence of
item IDs. We feed that sequence through a neural network — an **embedding
layer**, then a **GRU** (a type of recurrent neural network), then a
**self-attention** pooling layer — and the network outputs a score for
every item in the catalogue. The top-scoring items become the
recommendations. We train the network to maximise the score of the
**actual next item** the user clicked. Dataset: Amazon Reviews 2023
Electronics. Roughly 0.5–1M raw reviews → ~10K shopping sessions → 7K
unique items.

---

## Part 2 — End-to-end runbook

Four commands. You've already done the first one, but explain it anyway
since the teacher will likely ask.

### A. Download & preprocess (already done)

```bash
python scripts/download_data.py
```

**What it does:** streams raw reviews from HuggingFace, groups them into
sessions, splits train/val/test, builds the vocabulary, and saves
everything to disk.

**Output files** — all land in `data/processed/`:

| File | Purpose |
|---|---|
| `vocab.json` | The ASIN → integer-id dictionary. The model only understands integers, so we map each Amazon product ID to a small number. |
| `train_sessions.pkl` | 80% of sessions, used to learn the model's weights. |
| `val_sessions.pkl` | 10% of sessions, used during training to check progress and decide when to stop. |
| `test_sessions.pkl` | 10% of sessions, held out completely for the final score in `evaluate.py`. |
| `encoded_train.pkl` / `_val.pkl` / `_test.pkl` | The same sessions but as already-encoded integer ID lists, left-padded to length 50. Saves time on every training run. |

**Console output to point out:**

```
========================================================================
  DATA PIPELINE SUMMARY
========================================================================
  Split            Sessions      Encoded
  ------------------------------------
  Train               5,971        5,971
  Val                   746          746
  Test                  747          747
  ------------------------------------
  Total               7,464        7,464

  Vocabulary size : 6,988
========================================================================
```

- **Sessions** = number of shopping visits per split.
- **Encoded** = same sessions after integer-encoding + left-padding.
  (Equal to *Sessions* unless a session was too short to encode.)
- **Vocabulary size** = number of unique items the model knows about
  (plus 2 special tokens — see "PAD" and "UNK" below).

> If asked "why are vocab and session counts different?" → vocab counts
> *unique items*, sessions count *shopping visits*.

### B. Train the model

```bash
python scripts/train.py
```

**What it does:** loads the encoded training sessions, builds the neural
network with random weights, and iterates over the data 20–30 times
(epochs). After every pass it checks performance on the validation set;
if validation metric stops improving for 10 epochs it stops early.

**Output files** — all land in `checkpoints/`:

| File | Purpose |
|---|---|
| `epoch_001_recall0.0470.pt` | A snapshot of the model's weights after epoch 1. The number `0.0470` is the validation Recall@10. |
| `epoch_004_recall0.0512.pt` | Saved only when validation Recall@10 *improved*. So the latest `.pt` file is always the best one so far. |

**Console output during training:**

```
       Epoch |     loss | Recall@10  MRR@10
  --------------------------------------------------
  Epoch 001/030 | loss=8.8322 | Recall@5=0.0239 ... Recall@10=0.0470 ... [*]
  Epoch 002/030 | loss=8.3160 | Recall@5=0.0285 ... Recall@10=0.0451 ...
  ...
  Early stopping at epoch N (no improvement for 10 epochs).
  Best Recall@10 = 0.0470
```

**How to read it:**

- **loss** — measure of "how wrong the model is" averaged over all
  training batches. Should drop epoch over epoch in the first few epochs,
  then plateau. The starting value should be close to `ln(vocab_size)`,
  which is the loss of a uniform-random guess.
- **Recall@K** — fraction of validation sessions where the correct next
  item was in the model's top-K predictions.
- **MRR@K** — Mean Reciprocal Rank: did the correct item appear near the
  *top* of the top-K, or buried near the bottom?
- **`[*]`** at the end of a row = "new best checkpoint saved."

> If asked "did the model learn?" → point to the loss dropping (e.g.
> 8.83 → 7.79) AND a validation metric improving. Either alone isn't
> conclusive.

### C. Evaluate on the test set

```bash
python scripts/evaluate.py
```

**What it does:** auto-loads the best checkpoint from `checkpoints/`,
runs the trained model on every session in `test_sessions.pkl`, and
computes the final metrics.

**Output files** — both in `evaluation/`:

| File | Purpose |
|---|---|
| `results.csv` | One-row CSV with every metric, machine-readable. |
| `results.md` | The same metrics formatted as a Markdown table — for pasting into `README.md` or your report. |

**Console output:**

```
========================================================================
  GRU + ATTENTION  ·  TEST-SET METRICS
========================================================================
           Model  Recall@5  MRR@5  HitRate@5  Recall@10  MRR@10  ...
 GRU + Attention    0.0294 0.0189     0.0294     0.0423  0.0206
========================================================================

Done. Recall@10 = 0.0423  ·  MRR@10 = 0.0206
```

**How to read these metrics:**

| Metric | Meaning | Range |
|---|---|---|
| **Recall@K** | Across all test sessions, what fraction had the *true* next item somewhere in the model's top-K predictions? | 0.0–1.0 (higher is better) |
| **MRR@K** | Same idea, but rewards correct items near the **top**. Top-1 hit = 1.0. Top-5 hit = 0.2. Miss = 0. | 0.0–1.0 (higher is better) |
| **HitRate@K** | Identical to Recall@K when each session has exactly one correct answer (true here). Kept for completeness. | 0.0–1.0 |

**Random-baseline reference:** with a vocabulary of N items, blindly
guessing K items gives Recall@K ≈ K/N. So for vocab=6988 and K=10,
random ≈ 0.0014. Our 0.0423 is **~30× random**, which is a real result.

> If asked "why are these numbers so small (4%)?" → "Recall is computed
> over thousands of possible items. 4% is ~30× random. Production
> session-based recommenders on million-session datasets hit 10–25%, but
> ours has only ~10K sessions so this is in line with what's achievable."

### D. Launch the demo

```bash
python demo/app.py
```

Then open `http://localhost:5000` in a browser.

**What it does:** Flask server loads the best checkpoint into memory and
serves an interactive web UI. The user types ASINs into a textarea; the
server runs the same encode → score → top-K pipeline as `evaluate.py`
and returns JSON; the browser renders three panels.

**Three panels you'll show in the viva:**

1. **Left — Your Session.** Textarea where you type ASINs (one per
   line). Five "Example Session" buttons pre-populate it with real test
   sessions whose actual next item is shown as `Actual next: B0XXXXXXXX`.

2. **Centre — Recommendations.** After clicking "Get Recommendations":
   - A coloured banner (green = ✓ Hit at rank #N / yellow = ✗ Not in
     top-K) showing whether the held-out target ASIN appears anywhere in
     the recommendation list.
   - 20 ranked cards: rank, ASIN, percent score, raw-score explanation.
   - Matching card is highlighted green with a ✓ MATCH badge.

3. **Right — Analysis.** Two boxes:
   - **Recommendation Method** — just a label saying "GRU + Attention scoring".
   - **Session Attention** — a bar chart showing how much weight the
     model assigned to each item in your input session. Sums to 100%.
     Bigger bar = "the model paid more attention to this item when
     deciding what to recommend next."

> If asked "what's the percent score on each card?" → "Softmax probability
> over the top-K. So 10% means 'of the model's preference among the top
> 20 items, this one gets 10% of the weight'. It's not the probability
> over the full 7,000-item catalogue (that would be tiny)."

> If asked "what does the attention chart tell you?" → "How much the
> model focused on each input item. Equal bars mean the model couldn't
> tell which inputs were more informative."

---

## Part 3 — How is a GRU different from a "normal" neural network?

This is the question most likely to come up. Here's the short version
with code references.

### Normal feedforward neural network (recap)

What you've seen in coursework so far probably looks like this:

```
input vector  →  Linear layer  →  ReLU  →  Linear layer  →  output
                              ↑                          ↑
                         trained weights            trained weights
```

- The whole input arrives at once as a **fixed-size vector**.
- Each layer applies a matrix multiply + a nonlinearity (e.g. ReLU).
- Training is **gradient descent + backpropagation**: compute loss,
  compute gradients of loss w.r.t. each weight, take a small step
  against the gradient. Adam or SGD or similar.

This is great for fixed-shape inputs like a 28×28 image. **It does not
handle sequences naturally.** If your input is "user clicked item A,
then B, then C", a feedforward network has no built-in concept of
"first, then, then." You'd have to concatenate all items into one giant
vector, and the network would have no way to know the order matters.

### What changes with a GRU

A **GRU** (Gated Recurrent Unit) is a type of **recurrent neural network
(RNN)**. It is still trained with the **same** gradient descent +
backpropagation. The training mechanics are unchanged. Only the
**architecture** is different — specifically, the layer keeps a
**hidden state** that carries information forward as the sequence is
read one step at a time.

The diagram looks like this for a session of 3 items:

```
   item₁ →  GRU  →  hidden₁
                       ↓ (carries memory)
   item₂ →  GRU  →  hidden₂
                       ↓
   item₃ →  GRU  →  hidden₃   ← summary of the whole session so far
```

The same GRU layer is applied at every step, but each step also
receives the previous hidden state. So `hidden₃` knows about item₁ and
item₂ as well as item₃.

**The "gated" part:** inside the GRU, learned mini-networks called
*gates* decide:
- **Update gate:** "how much of the previous hidden state should I keep
  vs replace with new info?"
- **Reset gate:** "how much of the previous state should I forget when
  forming new info?"

These gates let the GRU **remember important things across many steps**
and **forget irrelevant ones**, which a plain RNN can't do well (plain
RNNs suffer from "vanishing gradients" over long sequences). You don't
have to code the gates yourself — PyTorch has them in `nn.GRU`.

### What the training loop looks like (it's actually the same!)

Look at [models/encoder.py:235-250](asbrs/models/encoder.py#L235-L250):

```python
for step, batch in enumerate(dataloader):
    input_ids = batch["input_ids"].to(device)
    lengths   = batch["lengths"].to(device)
    targets   = batch["target"].to(device)

    optimizer.zero_grad()                                          # clear old gradients
    session_repr, _, _ = self.encoder(input_ids, lengths)          # forward pass through GRU
    item_embs = self.encoder.item_embedding.get_all_embeddings()
    logits = self.encoder.predict_scores(session_repr, item_embs)  # score every item
    loss = F.cross_entropy(logits, targets)                        # how wrong are we?
    loss.backward()                                                # backprop gradients
    nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
    optimizer.step()                                               # take a step against gradient
```

**Every line of this is identical to what you'd write for a feedforward
network.** `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`
are the standard PyTorch training trio. The optimiser
([scripts/train.py:142](asbrs/scripts/train.py#L142)) is just `Adam`:

```python
optimizer = torch.optim.Adam(
    encoder.parameters(),
    lr=cfg.training.lr,
    weight_decay=cfg.training.weight_decay,
)
```

So when the teacher asks "what's special about training a GRU?": **the
training loop is identical to a feedforward net.** What's different is
*the architecture of the forward pass* — the network internally
maintains a hidden state across time.

### Where the GRU actually lives in our code

Three short snippets, [models/encoder.py](asbrs/models/encoder.py):

**1. Building the GRU layer** ([encoder.py:80](asbrs/models/encoder.py#L80)):

```python
self.gru = nn.GRU(
    input_size=embed_dim,    # 64 — size of each item's embedding
    hidden_size=hidden_dim,  # 128 — size of the hidden state
    batch_first=True,
)
```

That's the entire GRU declaration. PyTorch's `nn.GRU` handles all the
gate maths internally. Three trainable weight matrices live inside
(update gate, reset gate, candidate state) — Adam updates them
automatically during `optimizer.step()`.

**2. Running the GRU forward** ([encoder.py:133-145](asbrs/models/encoder.py#L133-L145)):

```python
packed = nn.utils.rnn.pack_padded_sequence(   # ignore PAD positions
    embedded, lengths_cpu,
    batch_first=True, enforce_sorted=False,
)
packed_out, _ = self.gru(packed)              # ← the GRU walks the sequence
all_hiddens, _ = nn.utils.rnn.pad_packed_sequence(
    packed_out, batch_first=True, total_length=L,
)
```

- `pack_padded_sequence` is a PyTorch helper: it tells the GRU "for
  these padded inputs, only process the real items — skip the PAD
  positions." That's faster and avoids polluting the hidden state with
  PAD tokens.
- `self.gru(packed)` is *the actual GRU computation*. It walks the
  sequence one step at a time and returns the hidden state at every
  step.
- `all_hiddens` shape is `[batch, seq_len, hidden_dim]` — for each
  batch row, one 128-dim hidden state per time step.

**3. Attention pools the hidden states**
([encoder.py:153](asbrs/models/encoder.py#L153)):

```python
session_repr, attn_weights = self.attention(all_hiddens, mask)
```

The GRU gives us one hidden state per time step. We need to collapse
those into **one** vector representing the whole session. Attention
does that as a learned weighted average — it decides how important each
step was. The weights are what the demo's right-panel bar chart shows.

### One-paragraph viva summary you can memorise

> "A GRU is a recurrent neural network — it processes a sequence one
> item at a time and keeps a hidden state that summarises everything
> seen so far. It has built-in 'gates' that decide what to remember and
> what to forget across time steps. Crucially, **training a GRU uses
> the exact same gradient descent + backpropagation as a feedforward
> network** — `loss.backward()` followed by `optimizer.step()`. What's
> different is the architecture, not the optimisation. In our code,
> `nn.GRU(embed_dim, hidden_dim)` in `models/encoder.py` is the entire
> GRU layer. PyTorch handles the gate maths and gradient flow."

### A couple of follow-up answers prepared

**Q: Why not just average the item embeddings instead of using a GRU?**
A: An average loses order. Clicking *charger → cable → phone-case* is a
different shopping intent from *phone-case → cable → charger*. The GRU
preserves order because each step's hidden state depends on the
previous one.

**Q: Why GRU and not LSTM?**
A: GRU and LSTM are both gated RNNs. GRU is simpler (2 gates vs 3) and
typically trains faster with similar accuracy on short sequences.
Our sessions are short (≤50 items), so GRU is the sensible choice.

**Q: Why attention on top of GRU?**
A: The last GRU hidden state biases toward the last item. Attention
gives every position a learned weight, so the session summary considers
all items proportional to their importance, not just the latest one.

**Q: What's PAD and UNK?**
A: Two special vocabulary entries. PAD (id 0) is filler used to make
all sessions the same length so they fit into a tensor. UNK (id 1) is
the fallback for items that didn't appear in training — instead of
crashing, the model treats them as a generic "unknown item". At
inference we mask both out so they're never recommended.

---

## Part 4 — Likely viva questions & one-liner answers

| Question | Short answer |
|---|---|
| Why are review titles and product names not used? | The model trains on integer IDs only — text isn't needed for next-item prediction. We removed the metadata fetch to keep the pipeline lean. |
| Why 80/10/10 train/val/test split? | Industry standard. Train is what the model learns from; val tells us when to stop; test gives an unbiased final score on data the model has never seen. |
| Why is the loss the same magnitude at the start (~ln(vocab))? | A randomly-initialised classifier predicts every item with roughly equal probability, so cross-entropy = ln(vocab_size). Watching this drop = watching the model actually learn. |
| What is "leave-one-out" evaluation? | For each session, the **last** item is hidden; the model is shown the first N-1 and must predict item N. Standard protocol for sequential recommenders. |
| Why softmax probability instead of raw scores in the demo? | Raw dot-product scores can be any number; users find percentages easier to interpret. We softmax over the top-K so the 20 numbers sum to 100%. |
| How big is the model? | Encoder has ~vocab_size × 64 (embedding) + ~50k (GRU + attention) = roughly 0.5M parameters. Tiny by deep-learning standards — runs on CPU easily. |
| Could you replace the GRU with a Transformer? | Yes, conceptually that's what SASRec does. Same training loop, swap `nn.GRU` for `nn.TransformerEncoder`. We chose GRU for simplicity and because sessions are short. |
