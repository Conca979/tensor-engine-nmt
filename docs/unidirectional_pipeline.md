# Unidirectional Seq2Seq LSTM — Complete Pipeline

**Architecture:** Stacked unidirectional LSTM encoder + stacked unidirectional LSTM decoder + Luong
"general" attention + shared BPE vocabulary  
**Implementation target:** NumPy / CuPy from scratch (no autograd framework)  
**Dtype:** `float32` throughout (all weights, activations, embeddings). Integer tensors (`X`, `Yin`, `Yout`, `Xlen`, `Ylen`) are `int32`.  
**Teacher forcing:** Inverse sigmoid decay schedule (Bengio et al., 2015)  
**Companion files:** `notation_convention.md`, `dimension_trace.md`

---

## Table of Contents

1. [Notation Legend](#notation-legend)
2. [Global Hyperparameters](#global-hyperparameters)
3. [Per-batch Variables](#per-batch-variables)
4. [Stage 0 — Raw Batch](#stage-0--raw-batch)
5. [Stage 1 — Embedding Lookup](#stage-1--embedding-lookup)
6. [Stage 2 — Encoder Loop](#stage-2--encoder-loop)
7. [Stage 3 — Encoder→Decoder Handoff](#stage-3--encoderdecoder-handoff)
8. [Teacher Forcing Schedule](#teacher-forcing-schedule)
9. [Stage 4 — Decoder Loop](#stage-4--decoder-loop)
10. [Stage 5 — Loss](#stage-5--loss)
11. [End-to-End Shape Checklist](#end-to-end-shape-checklist)
12. [Worked Example Loop](#worked-example-loop)

---

## Notation Legend

Before reading anything else, internalize this table. Every symbol in the
equations below is defined here exactly once.

### Superscripts

| Superscript | Where you see it | What it means |
|---|---|---|
| `enc` | `h_t^enc[l]`, `W_ih^enc[l]` | Belongs to the **encoder** LSTM (not the decoder) |
| `dec` | `h_t^dec[l]`, `W_ih^dec[l]` | Belongs to the **decoder** LSTM |
| `[l]` | `h_t^enc[l]`, `C_t^dec[l]` | **Layer index** — `l=1` is the bottom layer (receives embeddings), `l=2` is the top layer (feeds attention/output). Square brackets distinguish it from a power. |
| `(i)` | `X^(i)`, `Tx^(i)` | **Batch index** — only used when explicitly comparing two different batches. Dropped in all other cases. |

### Subscripts

| Subscript | Where you see it | What it means |
|---|---|---|
| `t` | `x_t`, `h_t^enc[l]`, `s_t` | **Timestep** — the Python loop variable. Encoder: `t = 1…Tx`. Decoder: `t = 1…Ty`. |
| `t-1` | `h_(t-1)^enc[l]` | Previous timestep's hidden state — the "memory" carried forward |
| `ih` | `W_ih^enc[l]` | **Input-to-Hidden** weight matrix — multiplied by the current input |
| `hh` | `W_hh^enc[l]` | **Hidden-to-Hidden** weight matrix — multiplied by the previous hidden state |
| `b` | `X[b, t]` | **Example index inside the batch** — row index. `b = 0…B-1` |
| `j` | `H[b, j, :]`, `alpha_t[b, j]` | **Encoder position** index — used inside attention to refer to a specific encoder timestep |

### Special symbols

| Symbol | Meaning |
|---|---|
| `@` | Matrix multiply (NumPy `@` operator). Shapes: `(B, in) @ (in, out) → (B, out)` |
| `;` inside `[a ; b]` | Concatenation along the **feature axis** (last axis) |
| `σ(x)` | Sigmoid: `1 / (1 + exp(-x))` |
| `tanh(x)` | Hyperbolic tangent |
| `δX` | Gradient `∂L/∂X` — same shape as `X`. Used in the backward pass. |
| `:=` | "is defined as" / assignment notation |
| `∈ R^shape` | "is a real-valued tensor of this shape" |
| `PAD` | Token id `0` — the padding token |
| `UNK` | Token id `1` — unknown token |
| `START` | Token id `2` — decoder input at `t=1` |
| `END` | Token id `3` — marks the true end of a sequence |

### Row-vector convention (used everywhere)

```
y = x @ W + b
    ↑   ↑   ↑
    (B, in) @ (in, out) + (out,)  →  (B, out)
```

The batch dimension `B` is **always the first axis**. All weight matrices `W`
below are already written in this transposed / row-vector form.  
If you see a paper write `W · x` (column-vector), its `W` has the axes flipped
relative to what you see here.

### Dtype rules (implementation-level)

| Tensor category | dtype | Notes |
|---|---|---|
| All weights (`W_ih`, `W_hh`, `b`, `W_a`, `W_c`, `Wy`, `by`) | `float32` | Initialized with Xavier/Glorot or Orthogonal init |
| Embeddings (`Ex`, `Ey`) and their outputs (`x_t`, `y_t`) | `float32` | Embedding rows are float32; the gather produces float32 |
| All LSTM states (`h`, `C`), attention tensors (`e_t`, `alpha_t`, `z_t`, `s_tilde_t`) | `float32` | All intermediate activations |
| Logits and probabilities (`logits_t`, `p_t`) | `float32` | |
| Token ids (`X`, `Yin`, `Yout`) | `int32` | Used only for indexing — never in arithmetic |
| Length arrays (`Xlen`, `Ylen`) | `int32` | Used only for index math (`Xlen - 1`) |
| Boolean masks (`M`, `mask`) | `bool` (NumPy) | Cast to `float32` immediately before any arithmetic, e.g. `M.astype(np.float32)` |
| Loss scalar `L` | `float32` | |

**Critical rule:** never let NumPy silently upcast to `float64`. After every
`np.zeros`, `np.ones`, or `np.random.*` call, pass `dtype=np.float32`. When
mixing integer indices with float arrays always be explicit — `astype(np.float32)`
before multiplying, not after.

---

## Global Hyperparameters

These are **fixed for the whole training run**. They never change per batch.

| Symbol | Meaning | Example value |
|---|---|---|
| `N` | Total sentence pairs in the dataset | 4,000,000 |
| `V` | Shared BPE vocabulary size | **32,000** |
| `e` | Embedding dimension | 512 |
| `d` | LSTM hidden size (one layer, one direction) | 1024 |
| `B` | Batch size | Dynamic based on `max_tokens=4000` (e.g. ~64) |
| `L` | Number of stacked LSTM layers — same for encoder and decoder | 3 |
| `k` | Inverse sigmoid decay hyperparameter (teacher forcing) | 17000.0 |
| `min_tf` | Minimum teacher-forcing ratio floor (anchor against exposure bias trap) | 0.70 |

**Why "hidden size = `d`" means `4d` gates:**  
An LSTM has four gates — forget `f`, input `i`, gate `g`, output `o` — each
`d`-wide. They are computed together in one big `4d`-wide matrix multiply and
then split. So every weight matrix that produces gate pre-activations ends in
`4d` on its output axis.

---

## Per-batch Variables

These are **re-computed fresh for every batch**. The weights never change shape;
only the number of loop iterations changes.

| Symbol | Meaning |
|---|---|
| `Tx` | Encoder length for **this batch** = length of the longest EN sentence in it, after padding |
| `Ty` | Decoder length for **this batch** = length of longest VI sentence in it (+1 for START/END), after padding |

Example: batch 7 might have `Tx=12, Ty=9`; batch 8 might have `Tx=40, Ty=35`.

---

## Stage 0 — Raw Batch

**What enters here:** integer token ids straight from the data loader.  
**What leaves here:** the five tensors below — nothing is computed, just named.

| Symbol | Shape | Example shape | Meaning |
|---|---|---|---|
| `X` | `(B, Tx)` | `(64, 20)` | Encoder token ids (padded EN). `X[b, t]` = token id of example `b` at encoder timestep `t` |
| `Xlen` | `(B,)` | `(64,)` | True (pre-pad) length of each EN sentence. `Xlen[b]` tells you where padding starts for example `b` |
| `Yin` | `(B, Ty)` | `(64, 18)` | Decoder **input** ids: `[START, v1, v2, ..., PAD...]` — what the decoder reads |
| `Yout` | `(B, Ty)` | `(64, 18)` | Decoder **target** ids: `[v1, v2, ..., END, PAD...]` — what the decoder should predict |
| `Ylen` | `(B,)` | `(64,)` | True (pre-pad) length of each VI sentence (including END token) |

**Yin vs Yout — why two separate tensors?**  
Teacher forcing: at decoder step `t`, the decoder reads `Yin[:, t]` (the ground-truth
previous token) and must predict `Yout[:, t]` (the ground-truth next token).  
`Yin` is shifted one position to the right of `Yout` — `Yin` starts with `START`,
`Yout` ends with `END`.

---

## Stage 1 — Embedding Lookup

**What enters here:** integer token ids `X` and `Yin`.  
**What leaves here:** continuous floating-point vectors `x_t` and `y_t`.

### Embedding weight tables (allocated once, outside any loop)

| Symbol | Shape | Example shape | Meaning |
|---|---|---|---|
| `Ex` | `(V, e)` | `(32000, 512)` | Encoder embedding table. Row `v` = the learned `e`-dim vector for token `v`. dtype: `float32` |
| `Ey` | `(V, e)` | `(32000, 512)` | Decoder embedding table. Separate weights from `Ex` — shared vocabulary ≠ shared weights. dtype: `float32` |

### Inside the loop — one timestep at a time

```
x_t  = Ex[X[:, t]]      →  shape (B, e) = (64, 512)    # encoder step t
y_t  = Ey[Yin[:, t]]    →  shape (B, e) = (64, 512)    # decoder step t
```

`X[:, t]` is a `(B,)` integer array — it **indexes** into `Ex`, selecting one row per
example. This is a gather operation, not a matmul. Result is `(B, e)`.  
`x_t` and `y_t` are recomputed fresh inside each loop iteration — you never
hold the full `(B, Tx, e)` embedding matrix in memory simultaneously (though
you could, it's a design choice).

**Intuition:** the embedding layer is a learned lookup table that converts a
discrete token id (an integer with no mathematical meaning) into a continuous
`e`-dim vector where similar-meaning tokens end up close together.

---

## Stage 2 — Encoder Loop

**What enters here:** `x_t` for each `t`, initialized states.  
**What leaves here:** `H ∈ (B, Tx, d)` — the full encoder memory — and the
final hidden/cell states per layer for the handoff.

This is **unidirectional**: a single left-to-right pass, `t = 1 → Tx`.

### 2a. Weight matrices (allocated once, outside the loop)

One full set of weights per layer. Layer 1 receives embeddings (`e`-wide input);
every layer above receives the previous layer's hidden state (`d`-wide input).

| Symbol | Shape | Example (layer 1) | Example (layer 2) | Meaning |
|---|---|---|---|---|
| `W_ih^enc[1]` | `(e, 4d)` | `(512, 4096)` | — | Input-to-Hidden weights, layer 1 |
| `W_ih^enc[l]` for `l >= 2` | `(d, 4d)` | — | `(1024, 4096)` | Input-to-Hidden weights, deeper layers |
| `W_hh^enc[l]` | `(d, 4d)` | `(1024, 4096)` | `(1024, 4096)` | Hidden-to-Hidden weights, all layers |
| `b^enc[l]` | `(4d,)` | `(4096,)` | `(4096,)` | Bias for all gates combined |

**Why `4d` on the output axis?**  
One LSTM cell computes four gate pre-activations — forget (`f`), input (`i`),
gate (`g`), output (`o`) — each `d`-wide. Fusing them into one big matmul
(`→ 4d`) and then splitting is faster than four separate matmuls.

### 2b. Initialization (before the loop)

```
h_0^enc[l] = zeros(B, d)   for l = 1, 2, 3    →  (64, 1024) each
C_0^enc[l] = zeros(B, d)   for l = 1, 2, 3    →  (64, 1024) each
```

### 2c. Inside the loop — encoder step `t`, layer `l`

**Input routing:**

```
input_t^enc[1]  = x_t                  (B, e) = (64, 512)   ← embeddings feed layer 1
input_t^enc[l]  = h_t^enc[l-1]         (B, d) = (64, 1024)   ← layer l-1's output feeds layer l
```

**Gate pre-activations (the expensive matmul):**

```
gates^enc[l] = input_t^enc[l] @ W_ih^enc[l]
             + h_(t-1)^enc[l] @ W_hh^enc[l]
             + b^enc[l]
```

Shape breakdown:

```
input_t^enc[1] @ W_ih^enc[1]  :  (64, 512) @ (512, 4096)  →  (64, 4096)
h_(t-1)^enc[1] @ W_hh^enc[1] :  (64, 1024) @ (1024, 4096)  →  (64, 4096)
b^enc[1]                      :  broadcast (4096,)         →  (64, 4096)
gates^enc[1]                  :  (64, 4096)
```

For layer 2 and 3, the first term uses `(64, 1024) @ (1024, 4096)` instead — both
shapes produce the same `(64, 4096)` output.

**Split and activate:**

```
f_t, i_t, g_t, o_t = split(gates^enc[l], 4, axis=1)   # each (B, d) = (64, 1024)
f_t = σ(f_t)    # forget gate  — how much of the old cell state to keep
i_t = σ(i_t)    # input gate   — how much of the new candidate to write
g_t = tanh(g_t) # gate         — the new candidate values
o_t = σ(o_t)    # output gate  — how much of the cell to expose as hidden state
```

**Update cell and hidden states:**

```
C_t^enc[l] = f_t * C_(t-1)^enc[l] + i_t * g_t    →  (B, d) = (64, 1024)
h_t^enc[l] = o_t * tanh(C_t^enc[l])               →  (B, d) = (64, 1024)
```

`*` here is element-wise (Hadamard) multiplication.

**Intuition for each gate:**
- **Forget `f_t`:** "Of what I knew before, what fraction is still relevant?" Values
  near 1 keep the cell state intact; near 0 erase it.
- **Input `i_t`:** "How much of this new candidate do I write?" Gates the update.
- **Gate `g_t`:** "What new information do I want to write?" The actual candidate.
- **Output `o_t`:** "Of what's in the cell, what do I expose?" Filters the cell
  before producing the hidden state.

### 2d. Collecting encoder outputs

After the full loop over `t = 1…Tx` and all `L=3` layers, collect the top
layer's `(B, d)` hidden state at every timestep:

```
H = stack([h_1^enc[L], h_2^enc[L], ..., h_Tx^enc[L]], axis=1)
  → shape (B, Tx, d) = (64, 20, 1024)
```

`H` is the **encoder memory** — the full sequence of contextualized representations.
The decoder's attention mechanism will query every row of `H` at each decode step.

**Total encoder work per batch:**  
`Tx` iterations × `L` layers × 1 direction = `20 × 3 × 1 = 60` LSTM-cell
computations.

---

## Stage 3 — Encoder→Decoder Handoff

**What enters here:** the final hidden/cell states of the encoder.  
**What leaves here:** `h_0^dec[l]` and `C_0^dec[l]` — the decoder's starting states.

The decoder needs a starting hidden and cell state for each of its `L` layers.
We seed them from the encoder's **last meaningful** hidden/cell state — "last
meaningful" because of padding: example `b`'s encoder ran meaningful tokens
only for positions `0…Xlen[b]-1`. Positions after that are padding.

```
h_final^enc[l][b] = h_(Xlen[b])^enc[l][b]    →  shape (B, d) = (64, 1024)
C_final^enc[l][b] = C_(Xlen[b])^enc[l][b]    →  shape (B, d) = (64, 1024)
```

In NumPy, the gather looks like:

```python
# h_enc_all: (B, Tx, d) — all hidden states stored during the encoder loop
idx = Xlen - 1                                    # (B,) — 0-indexed last real position
h_final = h_enc_all[np.arange(B), idx, :]        # (B, d)
```

**Handoff (unidirectional — no bridge projection needed):**

Because the encoder is unidirectional, its final states are already `d`-wide —
exactly the size the decoder expects. No projection required.

```
h_0^dec[l] = h_final^enc[l]    →  (B, d) = (64, 1024)
C_0^dec[l] = C_final^enc[l]    →  (B, d) = (64, 1024)
```

This is done once per batch, per layer, before the decoder loop starts.  
**Total handoff cost:** 2 gathers (one for `h`, one for `C`) × `L = 3` layers = 6 gathers.
Zero matmuls — this is the simplicity advantage of unidirectional.

---

## Teacher Forcing Schedule

**Inverse sigmoid decay** (Bengio et al., 2015 — "Scheduled Sampling").

Instead of always feeding the ground-truth previous token to the decoder (pure
teacher forcing), the probability of doing so decays as training progresses.
The decoder learns to handle its own imperfect predictions rather than always
being corrected.

### The schedule formula

```
ε_i = max(min_tf, k / (k + exp(i / k)))
```

| Symbol | Meaning |
|---|---|
| `i` | **Global training step** — total number of batches processed since the start of training, counting across all epochs. Starts at `0`, incremented by `1` after every `optimizer.step()`. |
| `k` | Decay hyperparameter. Larger `k` = slower decay (more teacher forcing for longer). Chosen before training (default `17000.0`). |
| `min_tf` | Minimum teacher forcing floor (default `0.70`). Guarantees at least 70% ground truth to prevent catastrophic exposure bias collapse. |
| `ε_i` | Probability of using the **ground-truth** previous token at step `i`. Bounded in `[min_tf, 1.0)`. |
| `1 - ε_i` | Probability of using the **model's own argmax prediction** from the previous decoder step. |

### Behavior across training

```
i = 0    →  ε ≈ 1.0   (pure teacher forcing at the start, model hasn't learned anything)
i = k    →  ε ≈ k/(k + e) ≈ 0.73
i >> k   →  ε = min_tf (default 0.70, anchored to maintain stable target representations)
```

### At t=1 — always teacher forced

At the very first decoder step (`t=1`), the input is always `START` regardless
of `ε_i`. There is no previous model prediction to use — `y_hat_0` doesn't
exist. The schedule only applies for `t >= 2`.

### At inference — always free-running

At inference time, `ε_i = 0` effectively — the decoder always uses its own
previous argmax output. There is no ground-truth `Yin` available.

### Implementation note

```python
import numpy as np

# Called once per batch, before the decoder loop
def teacher_forcing_prob(i: int, k: float, min_tf: float = 0.0) -> float:
    return max(min_tf, float(k / (k + np.exp(i / k))))

epsilon = teacher_forcing_prob(global_step, hp.k, hp.min_tf)   # scalar float

# Inside the decoder loop, for t >= 2:
use_ground_truth = (np.random.rand() < epsilon)   # one coin flip per batch step

if use_ground_truth:
    input_token_ids = Yin[:, t]                   # int32, shape (B,)
else:
    input_token_ids = y_hat_prev                  # int32, shape (B,) — argmax from t-1

y_t = Ey[input_token_ids]                         # float32, shape (B, e)
```

**One flip per decoder step, not per example:** the same coin flip applies to
all `B` examples in the batch at timestep `t`. You could flip per-example, but
the per-step version is simpler and standard.

---

## Stage 4 — Decoder Loop

**What enters here:** `y_t` (schedule-selected input), `h_0^dec[l]`, `C_0^dec[l]`, `H` (encoder memory), current `global_step` `i`.  
**What leaves here:** `logits_t` for each `t`, which are stacked into the loss.

### 4a. Weight matrices (allocated once, outside the loop)

| Symbol | Shape | Example (layer 1) | Example (layer 2) |
|---|---|---|---|
| `W_ih^dec[1]` | `(e, 4d)` | `(512, 4096)` | — |
| `W_ih^dec[l]` for `l >= 2` | `(d, 4d)` | — | `(1024, 4096)` |
| `W_hh^dec[l]` | `(d, 4d)` | `(1024, 4096)` | `(1024, 4096)` |
| `b^dec[l]` | `(4d,)` | `(4096,)` | `(4096,)` |
| `W_a` | `(d, d)` | `(1024, 1024)` | — |
| `W_c` | `(2d, d)` | `(2048, 1024)` | — |
| `b_c` | `(d,)` | `(1024,)` | — |
| `Wy` | `(d, V)` | `(1024, 32000)` | — |
| `by` | `(V,)` | `(32000,)` | — |

### 4b. Before the loop — compute ε for this batch

```python
epsilon = k / (k + exp(i / k))   # scalar float — probability of teacher forcing this batch
```

`i` is the global step counter (incremented once per batch). `epsilon` is
computed once before the loop — it does not change per decoder timestep `t`.

### 4c. Inside the loop — decoder step `t`

#### Step 1: Select and embed the decoder input

```
# t = 1: always use START (no previous prediction exists)
if t == 1:
    input_ids = Yin[:, 0]                    # int32, (B,) — all START tokens

# t >= 2: schedule decides
else:
    use_gt = (random.rand() < epsilon)       # one coin flip for the whole batch
    if use_gt:
        input_ids = Yin[:, t-1]              # int32, (B,) — ground truth
    else:
        input_ids = y_hat_(t-1)              # int32, (B,) — argmax from previous step

y_t = Ey[input_ids]                         # float32, (B, e) = (64, 512)
```

`y_hat_(t-1)` is `argmax(p_(t-1), axis=1)` — computed at the end of the previous
iteration and carried forward. At `t=2`, `y_hat_1` was produced at the end of
step 1's output projection. Shape is always `(B,)` int32.

#### Step 2: LSTM layers (identical math to encoder, same gate computation)

**Layer 1** receives `y_t` as input:

```
gates^dec[1] = y_t @ W_ih^dec[1]
             + h_(t-1)^dec[1] @ W_hh^dec[1]
             + b^dec[1]
             → (B, 4d) = (64, 4096)

f,i,g,o = split(gates, 4, axis=1)    each (64, 1024)

C_t^dec[1] = σ(f) * C_(t-1)^dec[1] + σ(i) * tanh(g)    →  (64, 1024)
h_t^dec[1] = σ(o) * tanh(C_t^dec[1])                    →  (64, 1024)
```

**Layer l (e.g. 2, 3)** receives `h_t^dec[l-1]` as input (output of previous layer, **same timestep `t`**):

```
gates^dec[l] = h_t^dec[l-1] @ W_ih^dec[l]
             + h_(t-1)^dec[l] @ W_hh^dec[l]
             + b^dec[l]
             → (B, 4d) = (64, 4096)

C_t^dec[l] = σ(f) * C_(t-1)^dec[l] + σ(i) * tanh(g)    →  (64, 1024)
h_t^dec[l] = σ(o) * tanh(C_t^dec[l])                    →  (64, 1024)
```

**Top-layer alias:**

```
s_t := h_t^dec[L] = h_t^dec[3]    →  (B, d) = (64, 1024)
```

`s_t` is written separately because it is the **only** decoder state that feeds
attention. Lower layers never touch the attention computation.

#### Step 3: Attention (Luong "general" score)

**Purpose:** let the decoder look at all encoder positions and decide which ones
are most relevant for predicting the next token right now.

**Score computation:**

```
temp = s_t @ W_a                        →  (B, d) = (64, 1024)
e_t  = einsum('bd,btd->bt', temp, H)   →  (B, Tx) = (64, 20)
```

Breaking down `einsum('bd,btd->bt', temp, H)`:
- `temp`: `(B, d)` — one query vector per example
- `H`: `(B, Tx, d)` — all encoder hidden states
- For each example `b` and encoder position `j`: `e_t[b,j] = dot(temp[b], H[b,j,:])`
- Result: one scalar score per (example, encoder position) pair → `(B, Tx)`

**Mask padded positions:**

```
M    = (arange(Tx) < Xlen[:, None])      →  (B, Tx)  bool
e_t  = e_t + (M.astype(float) - 1)*1e9  →  (B, Tx)  # padded positions → ≈ -inf
```

`Xlen[:, None]` broadcasts `(B,)` to `(B, 1)` so the comparison works.
Padding positions become `≈ -1e9` before softmax, forcing their attention weight
to essentially zero.

**Softmax → attention weights:**

```
alpha_t = softmax(e_t, axis=1)    →  (B, Tx) = (64, 20)
```

Each row of `alpha_t` sums to 1. `alpha_t[b, j]` = how much example `b` attends
to encoder position `j` at this decoder step.

**Weighted sum → context vector:**

```
z_t = einsum('bt,btd->bd', alpha_t, H)    →  (B, d) = (64, 1024)
```

`z_t[b, :] = sum_j alpha_t[b,j] * H[b,j,:]` — a weighted average of encoder hidden
states, weighted by attention. This compresses the entire encoder sequence into
one `d`-dim summary relevant to the current decode step.

> **Symbol clash warning:** many papers call both the LSTM cell state and the
> attention context vector `c_t`. Here: `C_t` = cell state, `z_t` = context
> vector. Keep this sharp — it's the easiest place to silently mix up tensors.

#### Step 4: Fusion (combine decoder hidden state with context)

```
concat = [z_t ; s_t]           →  (B, 2d) = (64, 2048)   # concat along feature axis
s_tilde_t = tanh(concat @ W_c + b_c)
          = tanh((64, 2048) @ (2048, 1024) + (1024,))
          →  (B, d) = (64, 1024)
```

`s_tilde_t` is the **attentional hidden state** — it blends what the decoder
has processed so far (`s_t`) with what the encoder says is relevant right now
(`z_t`).

#### Step 5: Output projection

```
logits_t = s_tilde_t @ Wy + by
         = (64, 1024) @ (1024, 32000) + (32000,)
         →  (B, V) = (64, 32000)            # float32

p_t      = softmax(logits_t, axis=1)   →  (B, V) = (64, 32000)   # float32
y_hat_t  = argmax(p_t, axis=1)         →  (B,)   = (64,)          # int32
```

`logits_t[b, v]` = unnormalized log-probability that the next token is `v`,
for example `b`.  
`y_hat_t` is stored as `int32` — it will be used as an index into `Ey` on the
next iteration if the schedule selects free-running.  
`(64, 512) @ (512, 32000)` is the single most expensive matmul in the whole
per-timestep loop — worth profiling first.

---

## Stage 5 — Loss

**What enters here:** `logits_t` for every `t`, and `Yout` (ground-truth targets).  
**What leaves here:** a scalar loss `L`.

### Collect logits across all decoder timesteps

```
logits = stack([logits_1, logits_2, ..., logits_Ty], axis=1)
       →  (B, Ty, V) = (64, 18, 32000)    # float32
```

### Padding mask

```
mask = (Yout != PAD)                       →  (B, Ty)  bool
mask_f = mask.astype(np.float32)           →  (B, Ty)  float32  # for arithmetic
```

### Masked cross-entropy

For each example `b` and timestep `t` where `mask[b,t] = True`:

```
loss[b,t] = -log(p_t[b, Yout[b,t]])   # float32 — negative log-likelihood of the correct token
```

Final scalar:

```
L = sum(loss * mask_f) / sum(mask_f)   # float32 scalar
```

Dividing by `sum(mask_f)` (total non-padding tokens in the batch, not `B*Ty`)
means the loss is not inflated by heavily padded batches.

---

## End-to-End Shape Checklist

Copy these assertions into your code. If any array violates its expected shape,
you have a bug — check this table before checking your math.

```python
assert X.shape        == (B, Tx)
assert Xlen.shape     == (B,)
assert Yin.shape      == (B, Ty)
assert Yout.shape     == (B, Ty)

# Encoder
assert x_t.shape              == (B, e)          # every encoder t
assert h_t_enc[l].shape       == (B, d)          # every t, every l
assert C_t_enc[l].shape       == (B, d)          # every t, every l
assert H.shape                == (B, Tx, d)      # after full encoder loop

# Handoff
assert h_0_dec[l].shape       == (B, d)          # every l
assert C_0_dec[l].shape       == (B, d)          # every l

# Decoder
assert y_t.shape              == (B, e)          # every decoder t
assert s_t.shape              == (B, d)          # every decoder t  (= h_t_dec[L])
assert e_t.shape              == (B, Tx)         # every decoder t  (raw scores)
assert alpha_t.shape          == (B, Tx)         # every decoder t  (rows sum to 1)
assert z_t.shape              == (B, d)          # every decoder t
assert s_tilde_t.shape        == (B, d)          # every decoder t
assert logits_t.shape         == (B, V)          # every decoder t

# Loss
assert logits.shape           == (B, Ty, V)
assert mask.shape             == (B, Ty)
```

---

## Worked Example Loop

**Setup:**

```
B=2, e=4, d=6, V=8, L=2, Tx=3, Ty=2
Xlen = [3, 2]    # example 0 has 3 real tokens, example 1 has 2 (position 2 is padding)
```

We trace **one encoder step** at `t=1` through layer 1 only, then the full
decoder step at `t=1` including attention, to show every intermediate shape.

All numbers below are illustrative — they are not actual network outputs.

---

### Encoder step t=1, layer l=1

**Input `x_1`** (embedding of the first encoder token for each example):

```
x_1 = Ex[X[:, 0]]   →  shape (2, 4)

x_1 = [[0.1, -0.2,  0.5,  0.3],     # example 0, token at position 0
        [0.4,  0.1, -0.1,  0.2]]     # example 1, token at position 0
```

**Previous states** (initialized to zero):

```
h_0^enc[1] = [[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]]   →  (2, 6)

C_0^enc[1] = [[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]]   →  (2, 6)
```

**Gate pre-activations:**

```
gates^enc[1] = x_1 @ W_ih^enc[1] + h_0^enc[1] @ W_hh^enc[1] + b^enc[1]

x_1 @ W_ih^enc[1]:         (2, 4) @ (4, 24) → (2, 24)
h_0^enc[1] @ W_hh^enc[1]:  (2, 6) @ (6, 24) → (2, 24)   ← all zeros this first step
b^enc[1]:                   broadcast (24,)  → (2, 24)

gates^enc[1]:  (2, 24)       ← 4d=24 because d=6
```

**Split** (each chunk is `d=6` wide):

```
gates[:, 0:6]   →  f_pre   (2, 6)   forget gate logits
gates[:, 6:12]  →  i_pre   (2, 6)   input gate logits
gates[:, 12:18] →  g_pre   (2, 6)   gate (candidate) logits
gates[:, 18:24] →  o_pre   (2, 6)   output gate logits
```

**Activate:**

```
f_1 = σ(f_pre)     →  (2, 6)   values in (0, 1)
i_1 = σ(i_pre)     →  (2, 6)   values in (0, 1)
g_1 = tanh(g_pre)  →  (2, 6)   values in (-1, 1)
o_1 = σ(o_pre)     →  (2, 6)   values in (0, 1)
```

**Update:**

```
C_1^enc[1] = f_1 * C_0^enc[1] + i_1 * g_1   →  (2, 6)
           = f_1 * 0           + i_1 * g_1       # C_0 = 0, forget gate irrelevant at t=1
           = i_1 * g_1

h_1^enc[1] = o_1 * tanh(C_1^enc[1])             →  (2, 6)
```

Layer 1 is done. `h_1^enc[1]` (shape `(2, 6)`) feeds layer 2 as its input for the same `t=1`.

---

### Decoder step t=1

Assume the encoder has fully run. We have:

```
H = array of shape (B=2, Tx=3, d=6)    # full encoder memory
    H[0, :, :] has all 3 positions meaningful  (Xlen[0]=3)
    H[1, :, :] has positions 0,1 meaningful, position 2 is padding  (Xlen[1]=2)

h_0^dec[l] = h_final^enc[l]    →  (2, 6)  for each l
C_0^dec[l] = C_final^enc[l]    →  (2, 6)  for each l
# example 1's final state was gathered at position idx=1 (Xlen[1]-1=1), not position 2
```

**Embed decoder input** (`Yin[:, 0]` = START token for both examples):

```
y_1 = Ey[Yin[:, 0]]    →  (2, e=4) = (2, 4)
      # both examples get the START embedding at t=1
```

**Decoder LSTM** — identical gate math to encoder, skipped for brevity.  
After layers 1 and 2:

```
s_1 := h_1^dec[2]    →  (2, d=6) = (2, 6)
```

**Attention:**

```
temp = s_1 @ W_a               (2, 6) @ (6, 6)  →  (2, 6)

e_1 = einsum('bd,btd->bt', temp, H)
    # for each b,j: dot(temp[b], H[b,j,:])
    →  (2, Tx=3) = (2, 3)

# Illustrative values before masking:
e_1 = [[ 2.1,  1.4, -0.3],    # example 0
       [ 1.8,  0.9, -0.5]]    # example 1
```

**Mask** (`Xlen = [3, 2]` → example 1's position 2 is padding):

```
Tx = 3
j_idx  = [0, 1, 2]                   # shape (Tx,)
Xlen_  = [[3], [2]]                  # shape (B, 1), after [:, None]
M      = (j_idx < Xlen_)             # (B, Tx) = (2, 3)
       = [[True,  True,  True ],
          [True,  True,  False]]

e_1_masked = e_1 + (M.astype(float) - 1) * 1e9

e_1_masked = [[ 2.1,   1.4,  -0.3  ],     # example 0: no change
              [ 1.8,   0.9,  -1e9  ]]     # example 1: position 2 → -inf
```

**Softmax:**

```
alpha_1 = softmax(e_1_masked, axis=1)    →  (2, 3)

alpha_1 ≈ [[0.62,  0.31,  0.07],    # example 0: spread over all 3 real positions
            [0.71,  0.29,  0.00]]   # example 1: zero weight on padding position
```

Each row sums exactly to 1.0. Example 1's last position is effectively zeroed.

**Context vector:**

```
z_1 = einsum('bt,btd->bd', alpha_1, H)    →  (2, d=6)
    # z_1[b] = alpha_1[b,0]*H[b,0,:] + alpha_1[b,1]*H[b,1,:] + alpha_1[b,2]*H[b,2,:]
```

**Fusion:**

```
concat    = np.concatenate([z_1, s_1], axis=1)   →  (2, 2d=12)
s_tilde_1 = tanh(concat @ W_c + b_c)
           # (2,12) @ (12,6) + (6,)
           →  (2, d=6)
```

**Output projection:**

```
logits_1 = s_tilde_1 @ Wy + by
         # (2, 6) @ (6, V=8) + (8,)
         →  (2, V=8)

p_1      = softmax(logits_1, axis=1)    →  (2, 8)   # rows sum to 1
y_hat_1  = argmax(p_1, axis=1)          →  (2,)     # one predicted token per example
```

---

### After all Ty=2 decoder steps

```
logits = stack([logits_1, logits_2], axis=1)    →  (B=2, Ty=2, V=8)

mask   = (Yout != PAD)                          →  (2, 2)

L      = sum(-log(p_t[b, Yout[b,t]]) * mask) / sum(mask)    →  scalar
```

This scalar is what gets differentiated through backpropagation to update
all the weights: `Ex`, `Ey`, every `W_ih`, `W_hh`, `b`, `W_a`, `W_c`, `Wy`, `by`.

---

## Summary — Unidirectional vs Bidirectional Trade-off

| Property | Unidirectional (this doc) | Bidirectional |
|---|---|---|
| Encoder passes per layer | 1 | 2 (fwd + bwd) |
| Encoder matmuls per batch | `Tx × L` | `Tx × L × 2` |
| `W_ih^enc[l>=2]` input width | `d` | `2d` |
| `H` shape | `(B, Tx, d)` | `(B, Tx, 2d)` → bridge → `(B, Tx, d)` |
| Handoff | gather only — zero matmuls | gather + concat + bridge matmul per layer |
| Bridge weights | none | `W_bridge_H`, `W_bridge_h[l]`, `W_bridge_C[l]` |
| Decoder | unchanged | unchanged |
| Attention / output | unchanged | unchanged |
| Implementation complexity | baseline | ~2× encoder code + bridge code |

The unidirectional encoder is strictly simpler: one weight set per layer,
one loop, no concatenation, no projection. The decoder and everything
downstream is identical either way.
