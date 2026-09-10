# Tensor Engine NMT — Training & Operations Guide

> This document covers the **operational** side of training: how to run it, understand
> the logs, resume correctly, manage checkpoints, and evaluate the model.
> For the mathematical spec, see `docs/unidirectional_pipeline.md`.
> For architecture decisions, see `docs/implementation_plan.md`.

---

## Table of Contents

1. [Training Environment](#1-training-environment)
2. [Checkpoint System](#2-checkpoint-system)
3. [Resume Mechanics](#3-resume-mechanics)
4. [Reading the Training Logs](#4-reading-the-training-logs)
5. [Loss Curve Interpretation](#5-loss-curve-interpretation)
6. [Epoch Tracking](#6-epoch-tracking)
7. [Google Colab Workflow](#7-google-colab-workflow)
8. [Evaluation](#8-evaluation)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Training Environment

### Supported Backends

| Backend | How to activate | Expected speed |
|---|---|---|
| NumPy (CPU) | Default; no extra install | ~1,700 tok/s (laptop) |
| CuPy (GPU) | `pip install cupy-cuda12x` | ~1,500–1,650 tok/s (T4 Colab) |

The backend is selected automatically at import time:
```python
# backend.py
try:
    import cupy as xp   # GPU
except ImportError:
    import numpy as xp  # CPU fallback
```

The log line `[train] Backend: cupy` or `[train] Backend: numpy` confirms which is active.

### Dataset

- **Source:** PhoMT (English → Vietnamese)
- **Training pairs:** 2,966,638 sentence pairs
- **Vocabulary:** 30,839 BPE subword tokens (shared EN+VI)
- **Cache file:** `PhoMT_dataset/train_cache_V30839.pkl` (loaded in < 1 second after first run)

### Architecture at a glance

```
114,876,672 parameters total

Encoder: 3-layer unidirectional LSTM
  - Embedding: V × e  =  32,000 × 512
  - Layer 0:   e × 4d  = 512 × 4,096  (input → gates)
  - Layer 1,2: d × 4d  = 1,024 × 4,096
  - Memory:    H ∈ (B, Tx, d) = batch × src_len × 1024

Attention: Luong "general"
  - W_a: d × d = 1,024 × 1,024

Decoder: 3-layer unidirectional LSTM
  - Embedding: V × e  =  32,000 × 512
  - Layer 0:   e × 4d  = 512 × 4,096  (input → gates)
  - Layer 1,2: d × 4d  = 1,024 × 4,096
  - Fusion:    W_c: 2d × d = 2,048 × 1,024
  - Output:    Wy:  d × V  = 1,024 × 32,000
```

---

## 2. Checkpoint System

Every checkpoint involves **two files**:

```
checkpoints/
├── step_2000.npz      ← weights only, ~140 MB compressed
├── step_4000.npz
├── ...
├── step_N.npz         ← latest step checkpoint
├── epoch_1.npz        ← end-of-epoch snapshot
├── epoch_2.npz
└── optim_state.npz    ← Adam state, ALWAYS overwritten, ~270 MB
```

### Why two files?

Storing Adam state (`m`, `v`, `t` — same size as the weights, so ~270 MB) inside
every step checkpoint would multiply storage by 3×:
- Per checkpoint: 140 MB (weights) + 270 MB (Adam) = 410 MB
- Over 30 epochs at save_every=2000: ~695 checkpoints × 410 MB = **~285 GB** (with companion file: ~100 GB total)

With the companion-file approach:
- Each `step_N.npz` = ~160 MB
- `optim_state.npz` = ~320 MB, only ONE copy, always overwritten

### What is saved in each file

**`step_N.npz` contents:**
- `p0` … `p18` — all 19 parameter arrays (encoder + decoder weights)

**`optim_state.npz` contents:**
- `adam_t` — int64 step counter (the Adam `t`)
- `adam_m0` … `adam_m18` — first moment vectors
- `adam_v0` … `adam_v18` — second moment vectors

### Critical rule

> The `optim_state.npz` always corresponds to the **most recently saved** step
> checkpoint. Never delete the latest `step_N.npz` while keeping `optim_state.npz`,
> or you'll get a weight/optimizer mismatch.

### Epoch checkpoints

At the end of each epoch, an additional `epoch_E.npz` is saved. These are standalone
weight snapshots (also paired with the current `optim_state.npz`) and are useful
for rolling back to a clean epoch boundary.

---

## 3. Resume Mechanics

### Automatic resume (default)

When `train.py` starts with no `--resume` flag, it automatically:

1. Globs all `checkpoints/step_*.npz`
2. Sorts them **numerically** (not alphabetically) by step number
3. Loads the highest-step checkpoint
4. Loads `checkpoints/optim_state.npz` for Adam state
5. Computes `start_epoch` and `steps_done_in_epoch` from `global_step`
6. Skips already-completed batches in the current epoch via `iterate(start_idx=N)`

### Manual resume

```bash
uv run tensor-engine-nmt train --resume checkpoints/step_46000.npz
```

### What you should see in the log on a clean resume

```
[BPE] Loaded from bpe_vocab/  (31,622 merges, 30,839 tokens)
[train] Model parameters: 40,935,168
[train] Backend: cupy
[model] Checkpoint loaded <- checkpoints/step_46000.npz
[model] Adam state restored (t=46000) from checkpoints/optim_state.npz
[train] Auto-resumed from checkpoints/step_46000.npz (step 46000)
[dataset] Loaded 2,966,638 pairs.
[train] Epoch position: epoch 2/30, step 9647/46353 within epoch (46353 steps/epoch)

============================================================
  Epoch 2/30  |  global_step=46000
============================================================
  step   46100  loss=...
```

The key line is `Adam state restored (t=46000)` — this confirms no optimizer cold-start.

### Warning signs of a bad resume

```
[model] Warning: no optim_state.npz found, optimizer starts fresh
```
This means the optimizer resets to t=0, m=0, v=0. Expect a loss spike of ~0.5–1.0
for the next 2,000–5,000 steps while Adam re-calibrates its moments. The model
**will recover**, but it wastes training time.

---

## 4. Reading the Training Logs

Each log line format:
```
2026-09-09 17:48:11 | INFO | ep 2/30 | step 44800 | loss=2.1760 | lr=3.00e-04 | gnorm=0.557 | ε=1.000 | tok/s=1,624
```

| Field | Meaning |
|---|---|
| `ep 2/30` | Current epoch / total epochs |
| `step 44800` | Global step count (cumulative across all epochs) |
| `loss=2.1760` | Mean per-token NLL for the last `log_every=50` steps |
| `lr=3.00e-04` | Current learning rate (fixed; no scheduler) |
| `gnorm=0.557` | Gradient norm **before** clipping (clip_norm=5.0) |
| `ε=1.000` | Teacher forcing probability (1.0 = always use gold tokens) |
| `tok/s=1,624` | Target tokens processed per second (throughput metric) |

### About `gnorm`

- `gnorm` is the raw gradient norm **before** clipping
- Values > 5.0 are clipped to 5.0 before the Adam update
- Typical healthy range: 0.5–2.5
- Occasional spikes to 5–10 are normal (hard sentences)
- Sustained spikes > 10 over many steps indicate instability

### About `ε` (teacher forcing)

```
ε = k / (k + exp(global_step / k))     k = 2000
```

| global_step | ε |
|---|---|
| 0 | 1.000 (always use gold) |
| 20,000 | 0.999 |
| 46,000 | 0.999 |
| 100,000 | 0.994 |
| 500,000 | 0.796 |
| 1,390,590 (30 epochs) | ~0.5 |

With k=2000, ε decays very slowly — the model mostly uses teacher forcing
throughout training, which is appropriate for this dataset size.

---

## 5. Loss Curve Interpretation

### Observed training trajectory (from train_logs.txt)

| Step range | Loss | Notes |
|---|---|---|
| 100 | 6.25 | Random initialization |
| 5,400 | 3.47 | Fast descent (~0.5 loss/10k steps) |
| 5,500 | 4.30 | Session restart spike (old Adam bug — now fixed) |
| 19,300 | 2.97 | Best before second restart spike |
| 32,000 | 2.95 | Recovered |
| 36,100–38,300 | 3.3+ | Worst spike, gnorm=13.9 (conflicting sessions) |
| Epoch 1 end (~43,800) | 3.0–3.4 | Many restarts hurt final epoch 1 result |
| Epoch 2 start (44,000) | 2.67 | Model retained learning from epoch 1 |
| Step 46,400 | 2.03 | Current (epoch 2, ~21% in) |

### Expected future trajectory

| Milestone | Expected loss | Expected BLEU-4 |
|---|---|---|
| End of epoch 2 (~92,700) | ~1.7–1.9 | ~5–10 |
| End of epoch 5 (~230,000) | ~1.4–1.6 | ~12–18 |
| End of epoch 15 (~695,000) | ~1.1–1.3 | ~20–28 |
| End of epoch 30 (~1,390,590) | ~0.9–1.1 | ~28–35 |

> These are estimates. Actual BLEU depends heavily on beam search quality and
> tokenization match between hypothesis and reference.

### Normal vs. abnormal loss behavior

| Pattern | Normal? | Likely cause |
|---|---|---|
| Gradual decrease with noise | Yes | Expected |
| Spike then recovery over ~2,000 steps | Sometimes | Session restart without Adam state |
| Step decrease at epoch boundary | Yes | New epoch reshuffle, fresh learning rate reset |
| Sudden jump > +1.0, does not recover | No | Check for NaN weights; restart from previous checkpoint |
| Loss stuck / oscillating > 5,000 steps | Sometimes | lr may be too high; try `--lr 1e-4` |

---

## 6. Epoch Tracking

### How epochs work

Each epoch is one full pass through all 2,966,638 training pairs.

```
steps_per_epoch = 2,966,638 // effective_batch_size
```

With `max_tokens=4000` and average sentence length ~27 tokens:
```
effective_batch ≈ 4000 / 27 ≈ 148 pairs/batch
steps_per_epoch ≈ 2,966,638 / 148 ≈ 20,044 steps
```

> Note: The exact steps_per_epoch varies because `max_tokens` batching produces
> different batch sizes depending on sentence length. The value printed in the log
> (`[train] Epoch position: epoch X/30, step Y/Z within epoch`) reflects the
> actual computed value from `len(dataset.pairs) // hp.B`.

### Epoch boundaries

At the end of each epoch:
1. `epoch_E.npz` is saved to `checkpoints/`
2. `optim_state.npz` is updated
3. The dataset is re-shuffled (with a fresh random seed)
4. Training continues into epoch E+1

---

## 7. Google Colab Workflow

See `COLAB_GUIDE.md` for the full step-by-step Colab notebook setup.

### Quick reference — cell order

| Cell | Purpose | Run frequency |
|---|---|---|
| Cell 1 | Mount Google Drive | Every session |
| Cell 2 | Install dependencies (`cupy-cuda12x`, etc.) | Every session |
| Cell 3 | `%cd` into project directory | Every session |
| Cell 4 | Verify GPU + backend | Every session |
| Cell 5 | Train (auto-resumes) | Every session (the main loop) |
| Cell 6 | Manual resume from specific checkpoint | Only if needed |
| Cell 7 | Evaluate BLEU on test set | After training |
| Cell 8 | Interactive translation demo | After training |

### Colab session limits

- **Free Colab:** ~12 hours per session, GPU may be preempted
- **Colab Pro:** ~24 hours, more stable GPU access
- **Recommendation:** Run Cell 5 each session; auto-resume handles the rest

### Verifying a clean resume in Colab

After Cell 5 starts, look for:
```
[model] Adam state restored (t=XXXXX) from checkpoints/optim_state.npz
```
If you see this, the resume is correct. If you see the "starts fresh" warning,
the `optim_state.npz` was missing — the model will still train but expect a
small loss spike for the first ~2,000 steps.

---

## 8. Evaluation

### Running BLEU-4 evaluation

```bash
# Evaluate on test set (auto-finds latest checkpoint)
uv run tensor-engine-nmt evaluate

# Verbose mode (prints each translation)
uv run tensor-engine-nmt evaluate --verbose

# Evaluate specific checkpoint
uv run tensor-engine-nmt evaluate --ckpt checkpoints/step_46000.npz

# Evaluate only first 100 sentences (fast spot-check)
uv run tensor-engine-nmt evaluate --n 100
```

In Colab (Cell 7):
```python
from tensor_engine_nmt.evaluate import evaluate
bleu = evaluate(hp=hp, verbose=True)
print(f"Corpus BLEU-4: {bleu:.2f}")
```

### BLEU-4 limitations for Vietnamese

Vietnamese uses many pronouns and particles with no direct English equivalent
(e.g., "anh", "chị", "em", "tôi" all translate to "I/you" depending on context).
BLEU-4 penalises valid translations that differ from the reference's specific
pronoun choice, giving 0.0 sentence BLEU even for semantically correct translations.

**Practical guideline:**
- BLEU < 5: Model is learning basic structure
- BLEU 5–15: Rough but partially intelligible translations
- BLEU 15–25: Good quality, human-readable
- BLEU > 25: State-of-the-art range for this architecture

### Interactive translation

```bash
uv run tensor-engine-nmt translate
```

Type an English sentence and press Enter. The model outputs a Vietnamese translation
using beam search (width=4).

---

## 9. Troubleshooting

### "No checkpoints found"

```
FileNotFoundError: No checkpoints found in checkpoints/
```

The `checkpoints/` directory is empty or missing. Either training hasn't started yet,
or the directory path is wrong (Colab: check that Google Drive is mounted and `%cd`
points to the right folder).

### Loss spike after resume

A spike of +0.3 to +1.0 in the first 1,000–2,000 steps after a resume is caused
by the Adam optimizer re-warming its moment estimates. This is **now fixed** if
`optim_state.npz` is present. If you're seeing spikes on the fixed codebase, check:
1. Is `optim_state.npz` in `checkpoints/`?
2. Does the log show `Adam state restored (t=N)` (not the "starts fresh" warning)?

### GPU out of memory (OOM)

Reduce `max_tokens`:
```bash
uv run tensor-engine-nmt train --max-tokens 2000
```

Or reduce batch size manually in `config.py`.

### Loss is NaN

This indicates numerical overflow. Check:
1. Are there NaN values in the checkpoint? Load with `np.load()` and inspect
2. Did `gnorm` explode to thousands before the NaN? Reduce `lr` or lower `clip_norm`
3. Roll back to the previous checkpoint and resume

### tok/s is very low (< 300)

This is expected behavior at the end of a training epoch when the bucketed dataset
serves very long sentences. Speed naturally returns to normal at the start of the
next epoch after re-shuffling.

### "ImportError: cannot import name 'evaluate_bleu'"

The Colab guide may be out of date. The correct import is:
```python
from tensor_engine_nmt.evaluate import evaluate
bleu = evaluate(hp=hp, verbose=True)
```

Or use the alias (also works):
```python
from tensor_engine_nmt.evaluate import evaluate_bleu
bleu = evaluate_bleu(hp=hp, verbose=True)
```
