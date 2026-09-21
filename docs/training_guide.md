# Tensor Engine NMT — Training & Operations Guide

> This document covers the **operational** side of training: how to run it, understand
> the logs, resume correctly, manage checkpoints, and evaluate the model.
>
> **Short on time / slow hardware? Read
> [`short_session_training.md`](short_session_training.md) first** — it is the
> practical guide for training in bursts that stop cleanly.
>
> For the mathematical spec, see `docs/bidirectional_pipeline.md`.
> For architecture decisions, see `docs/implementation_plan.md`.
> For the backward-pass fixes and the current roadmap, see `docs/bugfix_report.md`.

---

## Table of Contents

1. [Training Environment](#1-training-environment)
2. [Checkpoint System](#2-checkpoint-system)
3. [Resume Mechanics](#3-resume-mechanics)
4. [Reading the Training Logs](#4-reading-the-training-logs)
5. [Loss Curve Interpretation](#5-loss-curve-interpretation)
6. [Epoch Tracking](#6-epoch-tracking)
7. [Short Sessions & Time Budgets](#7-short-sessions--time-budgets)
8. [Cloud Workflows](#8-cloud-workflows)
9. [Evaluation & Live Demo](#9-evaluation--live-demo)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Training Environment

### Supported backends

| Backend | How to activate | Notes |
|---|---|---|
| NumPy (CPU) | Default; no extra install | Works, but only the `laptop` preset is practical |
| CuPy (GPU) | `pip install cupy-cuda12x` (or `uv sync --extra gpu`) | Required for the full-size model |

The backend is selected automatically at import time; the log line
`[train] Backend: numpy` / `cupy` confirms which one is active.

### Measure it — do not trust a table

Throughput varies by more than 20× between CPUs, so the only meaningful number is
the one from your own machine:

```bash
uv run tensor-engine-nmt bench --preset gpu      # full-size model
uv run tensor-engine-nmt bench --preset laptop   # small model
```

It runs real forward/backward/Adam steps and prints measured tok/s, seconds per
step, and projected hours per epoch.

Reference measurements from the machine this project was developed on
(laptop CPU, `Backend: numpy`):

| | `--preset gpu` | `--preset laptop` |
|---|---|---|
| Parameters | 108,585,216 | 5,877,824 |
| Seconds per step | 9.25 s | 0.52 s |
| Throughput | 27 tok/s | 962 tok/s |
| **Hours per epoch** | **~950 h ≈ 40 days** | **0.58 h ≈ 35 min** |
| 10 epochs | ~400 days | 5.8 h |

> An older version of this document claimed "~1,700 tok/s (laptop)" for the NumPy
> backend and quoted 1,700 vs 1,500–1,650 for CPU vs GPU. That was simply wrong —
> it made CPU look as fast as a T4. The real gap is **50–100×**.

### Dataset

- **Source:** PhoMT (English → Vietnamese)
- **Training pairs:** 2,966,638
- **BPE vocabulary:** 30,839 tokens in `bpe_vocab/` (shared EN+VI).
  `cfg.V = 32000` is only an upper bound on the embedding table size; the loader
  raises a clear `ValueError` if `V` is ever smaller than the real vocabulary.
- **Cache file:** `PhoMT_dataset/train_cache_V30839_L30.pkl`. The name encodes the
  vocabulary size **and** `max_len` (and `max_pairs`, if set), because all three are
  baked into the cached content — see §2 of `short_session_training.md`.

### Architecture at a glance

```
108,585,216 parameters  (34 parameter tensors)

Encoder: 3-layer bidirectional LSTM
  - Embedding: V × e  =  32,000 × 512
  - Layer 0:   e × 4(d/2) = 512 × 2,048   (input → gates, per direction)
  - Layer 1,2: d × 4(d/2) = 1,024 × 2,048
  - Memory:    H ∈ (B, Tx, d) = batch × src_len × 1024

Attention: Luong "general"
  - W_a: d × d = 1,024 × 1,024

Decoder: 3-layer unidirectional LSTM
  - Embedding: V × e  =  32,000 × 512
  - Layer 0:   e × 4d  = 512 × 4,096
  - Layer 1,2: d × 4d  = 1,024 × 4,096
  - Fusion:    W_c: 2d × d = 2,048 × 1,024
  - Output:    Wy:  d × V  = 1,024 × 32,000
```

---

## 2. Checkpoint System

Every save involves **two files**:

```
checkpoints/
├── step_196000.npz    ← weights + epoch metadata, ~401 MB (compressed)
├── step_200000.npz
├── epoch_8.npz        ← end-of-epoch snapshot, ~401 MB
├── optim_state.npz    ← Adam state, ~733 MB, ALWAYS overwritten
├── state.txt          ← legacy epoch/step side file (see §3)
└── train_logs.txt     ← appended log lines
```

### Why two files?

Storing the Adam moments (`m`, `v`) inside every step checkpoint would roughly
triple storage. Instead there is exactly **one** `optim_state.npz`, overwritten on
every save.

That design has one hazard, which the code now defends against: a single shared
file could be silently paired with the wrong weights. So `optim_state.npz` is
**tagged with the checkpoint it was saved with**:

```
[model] WARNING: checkpoints/optim_state.npz holds the optimizer state saved with
'step_200000.npz', but you are loading 'step_196000.npz'. Those moments belong to
different weights, so they are not a valid resume point — starting the optimizer fresh.
```

### What is saved in each file

**`step_N.npz`**
- `p0` … `p33` — all 34 parameter tensors (encoder + decoder + attention + output)
- `meta_epoch`, `meta_step_in_epoch` — the exact position in the dataset

**`optim_state.npz`**
- `adam_t` — the Adam step counter (used for bias correction)
- `adam_m0…33`, `adam_v0…33` — first and second moment estimates
- `ckpt` — the basename of the checkpoint these moments belong to

### Atomic writes

Both files are written to a temp file and then `os.replace`d into position. A
session that is killed mid-write therefore leaves the *previous* good checkpoint
intact rather than a truncated one. A truncated file, if it ever occurs, is
detected on load and reported instead of crashing with `BadZipFile`.

### Critical rules

> 1. The `optim_state.npz` always belongs to the **most recently saved** step
>    checkpoint. Never delete that checkpoint while keeping the optimizer file.
> 2. If you copy a checkpoint somewhere as a backup, copy `optim_state.npz` with it.
> 3. Delete the *newest* checkpoint only if you also intend to restart the optimizer
>    cold from an older one.

### Limiting how many checkpoints you keep

Frequent checkpointing is what makes short sessions safe, but at ~401 MB each it
fills a disk quickly. Pass `--keep-last N` and older `step_*.npz` files are deleted
after every save:

```bash
uv run tensor-engine-nmt train --save-every 300 --keep-last 3
```

Only `step_*.npz` is pruned. `optim_state.npz` (which belongs to the newest one)
and `epoch_*.npz` (deliberate milestone snapshots) are never touched.

---

## 3. Resume Mechanics

### Automatic resume (default)

When `train.py` starts without `--resume`, it:

1. Globs `checkpoints/step_*.npz` and sorts them **numerically**
   (so `step_900.npz` ranks before `step_1000.npz`, unlike a lexicographic sort).
2. Loads the highest-step checkpoint and restores its Adam state.
3. Reads the epoch position **from inside that checkpoint**
   (`meta_epoch`, `meta_step_in_epoch`).
4. Recreates the deterministic batch order for that epoch
   (`seed = 42 + epoch`) and skips exactly the batches already trained.

### Manual resume

```bash
uv run tensor-engine-nmt train --resume checkpoints/step_46000.npz
```

### What a healthy resume looks like

```
[BPE] Loaded from bpe_vocab/  (31,622 merges, 30,839 tokens)
[train] Model parameters: 108,585,216
[train] Backend: cupy
[dataset] Loaded 2,966,638 pairs.
[model] Adam state restored (t=196000) from checkpoints/optim_state.npz
[model] Checkpoint loaded <- checkpoints/step_196000.npz
[model] Checkpoint metadata: {'epoch': 7, 'step_in_epoch': 23277}
[train] Auto-resumed from checkpoints/step_196000.npz (step 196000)
[train] Checkpoint metadata: epoch 8/30, step 23277 within epoch
Skipping first 23277 batches to resume mid-epoch...
```

The two lines that matter are `Adam state restored (t=...)` and
`Checkpoint metadata: epoch X/…, step Y within epoch`. Together they mean the
optimizer and the dataset position both line up with the weights.

### Legacy `state.txt`

Checkpoints written before the position was stored inside them have no
`meta_epoch`. For those, the loader falls back to `checkpoints/state.txt`. The side
file is still written for compatibility, but it is **no longer authoritative** —
which removes a real crash window, since it used to be written *after* the 733 MB
optimizer dump.

If neither is available, the epoch position is estimated from `global_step` and the
log says so explicitly.

### Warning signs of a bad resume

```
[model] Warning: no optim_state.npz found, optimizer starts fresh
[model] WARNING: ... holds the optimizer state saved with 'step_X.npz' ...
```

Either message means the optimizer resets to `t=0, m=0, v=0`. Expect a loss bump
of roughly 0.3–1.0 for the next few thousand steps while Adam re-warms its moments.
The model recovers, but the steps spent recovering are wasted.

### Changing hyperparameters between sessions

| Parameter | Safe to change mid-run? |
|---|---|
| `--lr`, `--min-tf`, `--k`, `--max-minutes`, `--keep-last` | ✅ Yes |
| `--max-tokens`, `--max-pairs`, `max_len`, `d`, `L`, `e`, `V` | ❌ No — `max_tokens`/`max_pairs`/`max_len` change the batching, so the mid-epoch skip lands in the wrong place; the structural ones must match the saved matrix shapes |

---

## 4. Reading the Training Logs

```
2026-09-19 09:38:10 | INFO | ep 8/30 | step  196700 | loss=2.5364 | lr=1.00e-04 |
gnorm=1.194 | ε=0.700 | tok/s=1,064 | ~2.31s/step, epoch≈15.8h
```

| Field | Meaning |
|---|---|
| `ep 8/30` | Current epoch / total epochs |
| `step 196700` | Global step count (cumulative, never resets) |
| `loss=2.5364` | Mean per-token NLL since the previous log line |
| `lr=1.00e-04` | Current learning rate (fixed; there is no scheduler) |
| `gnorm=1.194` | Gradient norm **before** clipping |
| `ε=0.700` | Teacher-forcing probability (1.0 = always feed the gold token) |
| `tok/s=1,064` | Target tokens processed per second |
| `~2.31s/step` | Measured seconds per optimiser step |
| `epoch≈15.8h` | Projected wall-clock time for a full epoch, from measurements so far |

The last two fields are what you use to plan a session — they did not exist in
earlier versions of the code.

### About `gnorm`

- Raw gradient norm **before** clipping; values above `clip_norm` (default **2.0**)
  are scaled down before the Adam update.
- Healthy range: roughly 0.5–2.5 with this configuration.
- Occasional spikes are normal (a hard batch). Sustained values far above
  `clip_norm` mean the clipping is doing all the work — lower the learning rate.

### About `ε` (teacher forcing)

```
ε = max(min_tf, k / (k + exp(global_step / k)))     k = 17,000, min_tf = 0.70
```

| global_step | Raw decay | Active `ε` with `min_tf=0.70` |
|---|---|---|
| 0 | 1.000 | 1.000 (pure teacher forcing) |
| 50,000 | 0.999 | 0.999 |
| 100,000 | 0.979 | 0.979 |
| 151,200 | 0.700 | **0.700** |
| 200,000+ | 0.690 → 0 | **0.700 — anchored, it stops decaying** |

`min_tf` is a floor, and with `k=17,000` the floor engages at around step 151,000
and then holds ε at 0.70 **forever**. This is deliberate for the full-size run
(decaying to 0 is known to cause error cascades with cross-entropy training), but
be aware of what it means: the model is trained with ≥70 % ground-truth inputs
while inference is 100 % free-running.

To let the schedule actually anneal, lower the floor and shorten the decay:

```bash
uv run tensor-engine-nmt train --k 4000 --min-tf 0.35
```

Since the fix in `docs/bugfix_report.md`, teacher forcing is also sampled
**per sequence** (not one coin-flip for the whole batch) and is seeded from
`global_step`, so a resumed run replays exactly the same decisions.

---

## 5. Loss Curve Interpretation

### Observed trajectory of the `checkpoints/` run

Taken from `checkpoints/train_logs.txt` (full-size model, `lr=1e-4`,
`max_tokens=4000`, ~24,600 steps per epoch):

| Step | Loss | Notes |
|---|---|---|
| 100 | 6.76 | Random initialisation |
| 1,000 | 5.12 | Fast initial descent |
| 5,000 | 3.94 | |
| 10,000 | 4.22 | Slower than step 5,000 — expected as easy sentences are fitted |
| 19,000 | 3.55 | |
| 19,100 | 3.95 | Transient spike, recovers within ~1,000 steps |
| 24,600 | 3.76 | End of epoch 1 |
| 24,700 | 3.11 | Epoch 2 begins — reshuffle, and the averaged loss restarts |
| 25,000 | 2.68 | |
| 196,000 | ~2.54 | End of the available run (epoch 8/30) |

At step 196,000 the model scored **corpus BLEU-4 = 18.68** (greedy) and **24.66**
(beam-4) on the first 200 length-filtered test sentences.

### Expectation, honestly stated

The loss was still drifting downward when the run stopped, and only ~8 of 30
epochs were done. This model is **under-trained, not over-trained**. Do not read
the BLEU numbers above as a ceiling.

| Milestone | Very rough expectation |
|---|---|
| End of epoch 2 | loss ~3.0–3.5 on reshuffle, BLEU high single digits |
| End of epoch 8 (where the checkpoint is) | loss ~2.5, BLEU ~19 / ~25 |
| End of epoch 15 | loss ~1.5–2.0, BLEU 25–32 |
| End of epoch 30 | loss ~1.2–1.6, BLEU 30–38 |

> These are extrapolations from one run, not measurements. BLEU-4 on Vietnamese is
> also unusually brittle — see §9.

### Normal vs. abnormal

| Pattern | Normal? | Likely cause |
|---|---|---|
| Gradual decrease with noise | Yes | Expected |
| A step change at an epoch boundary | Yes | Reshuffle + the averaged loss window restarts |
| Spike of +0.3…1.0 right after a resume | Sometimes | Optimizer state was not restored (check the log) or batching parameters changed |
| Spike that decays over ~1,000 steps | Yes | A hard batch, or Adam re-warming |
| Spike > +1.0 that never recovers | No | Check for NaN weights; roll back a checkpoint |
| Loss frozen for > 5,000 steps | Sometimes | Learning rate too high, or ε annealing issues — see §4 |

---

## 6. Epoch Tracking

One epoch is one full pass over the training pairs. The number of steps depends on
the dynamic token batching:

```
pairs_per_batch ≈ max_tokens / (average EN + VI tokens per pair)
steps_per_epoch  = len(dataset.pairs) / pairs_per_batch
```

With the full 2,966,638 pairs and `max_tokens=4000`, batches averaged ~121 pairs,
giving **~24,600 steps per epoch** — which is what the logs show. The number printed
at the start of each epoch comes from the checkpoint metadata (or, for old
checkpoints, from `state.txt`), not from an estimate.

At the end of each epoch the code saves `epoch_E.npz` with
`meta_epoch = E`, `meta_step_in_epoch = 0`, so the next run starts cleanly at the
top of epoch E+1.

---

## 7. Short Sessions & Time Budgets

If you cannot run for many hours at a time, use the wall-clock budget. It writes a
checkpoint and exits **cleanly**, so nothing is ever lost to a forced kill:

```bash
uv run tensor-engine-nmt train --max-minutes 120 --save-every 300 --keep-last 3
```

Re-running the identical command resumes exactly where it stopped — no `--resume`
needed.

| Flag | Why it matters for a short session |
|---|---|
| `--max-minutes N` | Stops cleanly at a checkpoint boundary instead of being killed |
| `--save-every N` | How much work a crash can cost. Size it from your measured speed: `N ≈ 900 / seconds_per_step` gives a checkpoint every ~15 min |
| `--keep-last N` | Stops frequent checkpoints filling the disk |
| `--preset laptop` | A ~5.9 M parameter model that a CPU can actually train |
| `--max-pairs N` | Trains on an evenly-strided sample of the corpus, so an epoch finishes |

The measure-first workflow, the laptop preset, the cloud session loop and a
worked weekly plan are all in
**[`short_session_training.md`](short_session_training.md)**.

---

## 8. Cloud Workflows

Both free options cap session length, so every run should be time-boxed to end
before the platform's hard limit:

```bash
uv run tensor-engine-nmt train --lr 1e-4 --max-minutes 630 --save-every 500 --keep-last 3
```

* **[COLAB_GUIDE.md](../COLAB_GUIDE.md)** — free T4, Google Drive for persistence.
* **[KAGGLE_GUIDE.md](../KAGGLE_GUIDE.md)** — free P100/T4×2 in 12-hour background
  runs, 30 h/week.

Both guides use the short-session recipe above. After every session, download the
newest `step_*.npz`, `optim_state.npz` and `train_logs.txt` and keep them together.

---

## 9. Evaluation & Live Demo

### Corpus BLEU-4

```bash
# Test set, beam search (default), appends to checkpoints/evaluation_result.txt
uv run tensor-engine-nmt evaluate --n 200 --verbose

# Faster: greedy decoding
uv run tensor-engine-nmt evaluate --n 200 --method greedy

# Overfitting check: a random sample of training sentences
uv run tensor-engine-nmt evaluate --split train --n 500 --random

# Evaluate a specific checkpoint (needed for --preset laptop, which writes elsewhere)
uv run tensor-engine-nmt evaluate --ckpt checkpoints_laptop/step_3120.npz --method greedy --n 200

# Disable the max_len filter (full, unfiltered test set)
uv run tensor-engine-nmt evaluate --no-filter
```

By default evaluation applies the **same `max_len` filter as training**, so the
score is measured on the distribution the model was trained on. The log states
exactly how many sentences were skipped.

Every run appends to `checkpoints/evaluation_result.txt`; summarise it with:

```bash
python analyze.py --eval-only
```

### BLEU-4 limitations for Vietnamese

Vietnamese pronoun choice is context-dependent ("anh", "chị", "em", "tôi" all map
to "I/you"). BLEU-4 gives 0.0 to a semantically correct translation that picks a
different pronoun from the reference.

| BLEU-4 | Interpretation |
|---|---|
| < 5 | Learning basic structure |
| 5–15 | Rough but partly intelligible |
| 15–25 | Good, human-readable |
| > 25 | Strong for this architecture |

Consider adding `chrF++` or `BERTScore` for a second opinion — BLEU alone cannot
distinguish "wrong" from "differently worded".

### Interactive demo

```bash
python demo.py                              # auto-loads the newest checkpoint
python demo.py --method both --beam-width 5 # compare greedy vs beam with latency
python demo.py --ckpt checkpoints_laptop/step_3120.npz
uv run tensor-engine-nmt translate          # CLI equivalent
```

In-session commands: `:mode beam|greedy|both`, `:beam N`, `:ckpt`, `:help`, `:quit`.

### Note: inference is cheap, training is not

On a CPU laptop the full-size model decodes a sentence in ~0.1 s (greedy) or
~1.4 s (beam-4), and BLEU on 200 sentences takes ~20 s. Only training needs a GPU.
Train in the cloud, then copy `step_N.npz` plus `bpe_vocab/` to your laptop to
evaluate and translate locally.

---

## 10. Troubleshooting

### "No checkpoints found"
`checkpoints/` is empty or the path is wrong. In Colab, confirm Drive is mounted and
`%cd` points at the project.

### Loss spike after resume
Check the log for `Adam state restored (t=N)`. If it says `starting the optimizer
fresh`, the moments were missing or belonged to a different checkpoint. Also
possible: you changed `--max-tokens`, `--max-pairs` or `max_len` between sessions,
so the mid-epoch skip is misaligned.

### `ValueError: hp.V=... is smaller than the loaded BPE vocabulary`
You are mixing directories — e.g. `--preset laptop` (V=8000, `bpe_vocab_laptop/`)
with the full `bpe_vocab/` (30,839 tokens). Use the preset's own directories, or
raise `V` to at least the real vocabulary size.

### `WARNING: using the legacy cache train_cache_V30839.pkl`
The cache predates the current naming scheme (which encodes `max_len` and
`max_pairs`). It loads, but if you have *raised* `max_len` since it was built the
longer sentences are no longer in it. Delete the file and re-run to rebuild.

### GPU out of memory
Lower the token budget first:
```bash
uv run tensor-engine-nmt train --max-tokens 2000
```

### Disk filling up
Add `--keep-last 3`, or delete old `step_*.npz` files by hand — keeping the newest
one **and** `optim_state.npz`. Each full-size checkpoint is ~401 MB and the
optimizer file is ~733 MB.

### Loss is NaN
1. Inspect the checkpoint for NaN: `np.load(path)['p0']`.
2. Did `gnorm` explode first? Lower `--lr` or `clip_norm`.
3. Roll back to the previous checkpoint and resume.

### `tok/s` suddenly drops during an epoch
Expected toward the end of an epoch: the interleaved buckets are exhausted at
different rates, so the tail of an epoch serves longer sentences. It recovers at
the next epoch boundary.

### `ImportError: cannot import name 'evaluate_bleu'`
Colab notebooks may be out of date. Use:
```python
from tensor_engine_nmt.evaluate import evaluate
bleu = evaluate(hp=hp, verbose=True)
```
(`evaluate_bleu` still exists as an alias.)
