# ASBRS — How the Neural Network Reaches Its Output

A line-by-line walkthrough of what happens after the embeddings are
created. Aimed at someone who has **never** built a neural network
before. We'll trace **one session through the model** with tiny example
numbers so you can see exactly what each step computes.

> Already comfortable with embeddings? Good — that's where we start.
> Read [models/encoder.py:forward](models/encoder.py) and
> [models/attention.py:forward](models/attention.py) alongside this doc;
> they now have heavy inline comments matching every step here.

---

## Setup — the running example

Imagine a user with this shopping session (3 items clicked):

```
Session = ["B001",  "B005",  "B042"]
```

The session goes through these conversions before reaching the network:

```
ASIN string        Vocabulary.encode()       Integer ID
"B001"      ───────────────────────────►    17
"B005"      ───────────────────────────►    42
"B042"      ───────────────────────────►    88
```

These integers are wrapped in a tensor of shape `[1, 50]` — one row of 50
positions (we always pad to 50). Older positions are filled with PAD (id
0) on the **left**, real items pushed to the **right**:

```
input_ids = [[0, 0, 0, ..., 0, 17, 42, 88]]    shape: [1, 50]
                        ↑   ↑   ↑   ↑
                  47 PAD slots │   │
                               │   │
                           item 1  item 2  item 3 (most recent)
```

We'll use real numbers but simplified dimensions (embed_dim=3 instead of
64, hidden_dim=4 instead of 128) so the matrices are small enough to
follow with your eyes.

---

## Stage 1 — Embedding lookup

[models/embeddings.py:83-92](models/embeddings.py#L83) — `ItemEmbedding.forward`

The integer IDs go in; rows of a learnable lookup table come out.

The lookup table (`embedding.weight`) is a matrix of shape `[vocab_size × embed_dim]`:

```
            col0    col1    col2
row 0 PAD:  0.00    0.00    0.00     ← always zero, never trained
row 1 UNK:  0.04   -0.21    0.13
row 2 B...: 0.27   -0.05   -0.14
...
row 17 B001: 0.21  -0.05    0.11    ← we'll use this
...
row 42 B005: 0.45   0.02   -0.30    ← and this
...
row 88 B042: 0.10   0.40    0.05    ← and this
...
```

`ItemEmbedding(input_ids)` looks up the row for each integer in
`input_ids` and stacks them. For our session:

```
position  →   0     1     2    ...   47    48    49
              ↓     ↓     ↓           ↓     ↓     ↓
embed = [[ 0,0,0 0,0,0 0,0,0 ... 0.21,-0.05,0.11  ← row 17 (B001)
                                  0.45, 0.02,-0.30  ← row 42 (B005)
                                  0.10, 0.40, 0.05]] ← row 88 (B042)

shape: [batch=1, seq_len=50, embed_dim=3]
```

So `embedded` now has 50 vectors per session, but only the last 3 are
real — the first 47 are zeros (because PAD's row is all zeros).

---

## Stage 2 — Tell the GRU to skip PAD

[models/encoder.py:131-138](models/encoder.py#L131)

We could feed all 50 positions to the GRU, including the 47 PADs. That
would work but be wasteful — the GRU would update its memory 47 times
on zero-input. To avoid this, PyTorch has a helper called
**`pack_padded_sequence`** that says to the GRU:

> "This session has only 3 real items at the end — skip the leading PADs."

```python
packed = nn.utils.rnn.pack_padded_sequence(
    embedded,
    lengths_cpu,        # tensor saying "real length = 3"
    batch_first=True,
    enforce_sorted=False,
)
```

You don't need to look inside `packed` — it's an internal format. Just
trust that the GRU now ignores PADs.

---

## Stage 3 — The GRU walks the sequence

[models/encoder.py:141](models/encoder.py#L141) — the single line that does ALL the work:

```python
packed_out, _ = self.gru(packed)
```

What this single line does, step by step:

### What the GRU is

A small mini-network (about 200 weights in our 4-dim toy example) that
takes **two** inputs at every step:

1. The **current item's embedding** (3 numbers in toy, 64 in real life).
2. The **previous memory state** (4 numbers in toy, 128 in real life).

And outputs **one** thing:

- The **new memory state** (4 numbers — same size as input memory).

### The same cell runs 3 times for our session

```
                    ┌────────────────────────┐
                    │                        │
   embed(B001) ───► │                        │ ───► memory₁
       ↓            │       GRU CELL         │
   memory₀=zero ──► │       (gates inside)   │
                    │                        │
                    └────────────────────────┘
```

Step 1 — read item B001:

```
Inputs:
  embed(B001)   = [0.21, -0.05,  0.11]                (3 numbers)
  prev memory   = [0.00,  0.00,  0.00,  0.00]         (4 numbers, starts at zero)

The GRU's gates do their thing internally (update gate + reset gate +
candidate state — explained shortly). Out pops a new memory:

  memory₁       = [0.05, -0.02,  0.08,  0.03]         (4 numbers)
```

Step 2 — read item B005:

```
Inputs:
  embed(B005)   = [0.45,  0.02, -0.30]
  prev memory   = [0.05, -0.02,  0.08,  0.03]         (memory₁ from step 1)

  memory₂       = [0.12,  0.04, -0.06,  0.09]
```

Step 3 — read item B042:

```
Inputs:
  embed(B042)   = [0.10,  0.40,  0.05]
  prev memory   = [0.12,  0.04, -0.06,  0.09]         (memory₂ from step 2)

  memory₃       = [0.18,  0.21, -0.02,  0.14]         ← knows about ALL 3 items
```

After unpacking, the variable `all_hiddens` in the code contains all 50
memory vectors (the first 47 are zeros, the last 3 are `memory₁`,
`memory₂`, `memory₃`):

```
all_hiddens shape: [1, 50, 4]    (in real model: [1, 50, 128])
```

### What's "inside" the GRU cell — the gates

You don't need to know this to use a GRU, but here it is for understanding.

At every step the cell computes three things using learned weight matrices:

1. **Update gate** `z` — for each of the 4 hidden dimensions, a number
   between 0 and 1 saying "how much of the OLD memory should I keep
   vs replace with new info?"

2. **Reset gate** `r` — again 0-1 per dimension. "When forming new info,
   how much of the OLD memory should I let influence it?"

3. **Candidate memory** `n` — a fresh proposal for what the new memory
   could be, computed from the input + `r·(old memory)`.

Then the new memory is computed as:

```
new_memory = (1 - z) · candidate + z · old_memory
```

If `z` is close to 1 → keep mostly the old memory (good for remembering
things across many steps). If `z` is close to 0 → replace with new info
(good for forgetting irrelevant past). The model **learns** when to do
which from training data.

Plain RNNs don't have these gates and so they "forget" early items over
long sequences. GRU's gates solve that.

### Recap of stage 3

After the GRU runs, we have one memory vector per position. The model
now "knows" the session's content at every point in time. The last
memory vector is the most informed — it's seen everything. But it's
biased toward the most recent item. To get a balanced summary we use
**attention**, the next stage.

---

## Stage 4 — Attention picks which memories matter most

[models/attention.py:forward](models/attention.py) (the whole file is one method)

The GRU left us with 50 memory vectors. We need to collapse them into
**one** vector that summarises the session for scoring.

### Why not just take the last memory?

The last memory `memory₃` knows about all 3 items, but it's most heavily
influenced by item 3 (the most recent). What if item 1 was actually the
most decisive click? Attention solves this by computing a **learned
weighted average**:

```
session_summary = w₁ · memory₁ + w₂ · memory₂ + w₃ · memory₃
```

with the weights `wᵢ` summing to 1 — and the **model learns those
weights** from data.

### Step A — Build a "query" vector

[attention.py:113-114](models/attention.py#L113)

The query represents "what is this session about overall?" Simplest
choice: just average all the real memories.

```
query = (memory₁ + memory₂ + memory₃) / 3
      = [0.117, 0.077, 0.000, 0.087]    (4 numbers)
```

### Step B — Pass query, keys, values through learned linear layers

[attention.py:116-119](models/attention.py#L116)

We don't compare the raw query against raw memories — we let the model
learn what "compare" should mean. So we transform each through a
trainable linear layer:

```
Q (query)  = q_proj(query)        — what we're looking FOR
K (keys)   = k_proj(memories)     — what we MATCH against
V (values) = v_proj(memories)     — what we WEIGHT & SUM
```

These three projections (`q_proj`, `k_proj`, `v_proj`) are just matrices
of trainable weights — `nn.Linear` layers. The Q/K/V naming comes from
the original Transformer paper.

```
Q shape: [1, 4]         — one 4-num query per session
K shape: [1, 50, 4]     — 50 keys per session (one per memory)
V shape: [1, 50, 4]     — 50 values per session
```

### Step C — Multi-head split (4 heads × 32 dims each in real model)

[attention.py:121-124](models/attention.py#L121)

In the real model with 128-dim hidden states, we split each Q/K/V into
**4 chunks of 32 dimensions** and run attention 4 times in parallel.
Each chunk can specialise (e.g. one head learns recency, another learns
category). Then we glue the results back together.

In our 4-dim toy we'd split into… well, let's pretend `num_heads=1` and
the head_dim is 4, just to keep numbers small.

### Step D — Compute similarity scores

[attention.py:129-131](models/attention.py#L129)

For each position, compute the **dot product** of the query with that
position's key:

```
scores[t] = Q · K[t]      (a single number per position)
```

Dot product is the standard way to measure "do two vectors point in
similar directions?" — big positive number = similar, near-zero = unrelated.

We then divide by √head_dim to keep the numbers manageable before softmax:

```
position →   0    1    2   ...  47   48   49
score   →   ~0   ~0   ~0   ...  0.13 0.27 0.18
(PAD positions get scored too — we'll mask them in the next step.)
```

### Step E — Mask PAD positions

[attention.py:135-136](models/attention.py#L135)

We don't want the model to attend to PAD filler. So we **overwrite** the
PAD scores with a huge negative number (`-1e9`):

```
scores after mask: [-1e9, -1e9, ..., -1e9, 0.13, 0.27, 0.18]
                    ↑ 47 PAD positions ↑   ↑ 3 real positions
```

### Step F — Softmax → attention weights

[attention.py:139](models/attention.py#L139)

Softmax converts arbitrary scores into a probability distribution: every
weight is between 0 and 1, and they sum to 1 across positions.

```
softmax([-1e9, ..., -1e9, 0.13, 0.27, 0.18])
   ↓
       [0,    ...,  0,    0.30, 0.40, 0.30]    ← these are the attention weights
```

PAD positions get 0 (because `softmax(-1e9) ≈ 0`). Real positions get
non-zero weights summing to 1.

**This is what the demo's bar chart visualises.** If you see equal bars
(33/33/33), it means the scores were similar, so softmax gave roughly
equal weights.

### Step G — Weighted sum of value vectors

[attention.py:142-148](models/attention.py#L142)

Now use those weights to combine the values:

```
context = Σ (weight[t] × V[t])
        = 0   × V[0]   ← PAD, contributes nothing
        + 0   × V[1]   ← PAD
        + ...
        + 0.30 × V[47]  ← item 1 (B001)
        + 0.40 × V[48]  ← item 2 (B005)   ← model decided this matters most
        + 0.30 × V[49]  ← item 3 (B042)

context (one 4-num vector) = [0.16, 0.18, -0.02, 0.13]
```

Then one final linear layer mixes the result:

```
session_repr = out_proj(context)    shape: [1, 4]    (real: [1, 128])
```

### Stage 4 output

We now have ONE 128-num vector (`session_repr`) summarising the whole
session, **and** a per-position weight vector (`attn_weights`) showing
which positions the model focused on.

---

## Stage 5 — Score every item in the catalogue

[models/encoder.py:160-177](models/encoder.py#L160) — `predict_scores`

To recommend, we need a score for every item in the vocabulary (7,000
of them). The simplest, well-tested approach is a **dot product**:

```
score[i] = session_repr · item_embedding[i]    (for every item i)
```

Items whose embeddings point in the same direction as the session
summary get high scores; items pointing differently get low scores.

There's a small wrinkle: `session_repr` is 128-dim (hidden_dim) but item
embeddings are 64-dim (embed_dim). So we first **project** the session
summary down to 64-dim with a learned linear layer
([encoder.py:94-97](models/encoder.py#L94)):

```python
proj_repr = self.projection(session_repr)        # [1, 128] → [1, 64]
scores    = proj_repr @ item_embeddings.T        # [1, 64] × [64, V] = [1, V]
```

Result: a tensor of shape `[batch=1, vocab_size]` — one score per item.

### Mask out junk

PAD (id 0) and UNK (id 1) are not real items — we never want to recommend
them. So we set their scores to `-inf`:

```python
scores[PAD_IDX] = float("-inf")
scores[UNK_IDX] = float("-inf")
```

This is done in [demo/app.py](demo/app.py) and [scripts/evaluate.py](scripts/evaluate.py).

### Take the top-K

```python
top_scores, top_ids = torch.topk(scores, k=20)
```

Returns the 20 item IDs with the highest scores. Decode those back to
ASIN strings with `vocab.decode(id)` and you have your recommendations.

---

## End-to-end summary in 7 lines of pseudocode

```
1. ids        = vocab.encode(session)              # ASIN → integer
2. embeddings = item_embedding(ids)                # integer → 64-num vector
3. memories   = gru(embeddings)                    # → 50 memory snapshots
4. mask       = (ids != PAD)                       # boolean: which positions are real
5. summary    = attention(memories, mask)          # → ONE session vector
6. scores     = summary @ item_embedding_table.T   # → score per item
7. top_K      = topk(scores, K)                    # → top K item IDs
```

Then `vocab.decode(top_K)` to get the predicted ASINs.

---

## Where training fits in

All of the steps above happen in **inference** (running the model). To
get good outputs, every layer's weights had to be **trained**. Training
is straightforward:

1. Forward pass (the 7 lines above) → produce scores.
2. Compare scores to the actual next item using cross-entropy loss.
3. `loss.backward()` — PyTorch computes how every weight should change.
4. `optimizer.step()` — Adam nudges every weight by `lr × gradient`.

Repeat for every batch, for many epochs. Over time the embedding rows
shift, the GRU's gate weights shift, the attention projections shift —
all moving in the direction that makes the model's top-K more often
contain the actual next item.

The training loop is in
[models/encoder.py:235-250](models/encoder.py#L235); the call to it is
in [scripts/train.py](scripts/train.py).

---

## TL;DR

Reading from one ASIN session to the final top-K recommendations, the
pipeline is:

| Stage | What | Where |
|---|---|---|
| 1 | ASIN → integer | Vocabulary dictionary lookup |
| 2 | Integer → 64-num vector | `ItemEmbedding` (learnable table) |
| 3 | 50 vectors → 50 memory snapshots | `nn.GRU` walks the sequence |
| 4 | 50 memories → 1 summary | `SelfAttentionLayer` weighted average |
| 5 | 1 summary → score per item | Dot product with item-embedding matrix |
| 6 | Scores → top-K item IDs | `torch.topk` |
| 7 | IDs → ASINs | Vocabulary reverse lookup |

The "neural" part is stages 2–5 — those are where trainable weights
live. Stages 1, 6, 7 are deterministic plumbing.
