# Unidirectional Seq2Seq LSTM NMT — Implementation Plan (As-Built)

> **Status:** Implementation complete. This document reflects the **actual codebase**,
> not the original design sketch. Differences from the original plan are marked with a warning symbol.
> Cross-reference: `docs/unidirectional_pipeline.md` for the mathematical walkthrough.

---

## 1. Project Status

### 1.1 What has been built

| Path | Status | Notes |
|---|---|---|
| `src/tensor_engine_nmt/config.py` | Complete | Frozen HParams dataclass |
| `src/tensor_engine_nmt/backend.py` | Complete | NumPy / CuPy auto-selection |
| `src/tensor_engine_nmt/activations.py` | Complete | sigmoid, tanh, softmax + backward |
| `src/tensor_engine_nmt/init_weights.py` | Complete | Xavier uniform, Orthogonal (QR) |
| `src/tensor_engine_nmt/bpe.py` | Complete | Streaming BPE, saves `bpe_vocab/` |
| `src/tensor_engine_nmt/dataset.py` | Complete | Bucket loader + `.pkl` cache |
| `src/tensor_engine_nmt/encoder.py` | Complete | L-layer LSTM fwd + BPTT |
| `src/tensor_engine_nmt/attention.py` | Complete | Luong general fwd + backward |
| `src/tensor_engine_nmt/decoder.py` | Complete | L-layer LSTM fwd + BPTT |
| `src/tensor_engine_nmt/loss.py` | Complete | Masked cross-entropy fwd + bwd |
| `src/tensor_engine_nmt/model.py` | Complete | Seq2Seq wrapper, checkpoint save/load |
| `src/tensor_engine_nmt/optimizer.py` | Complete | Adam + gradient norm clip |
| `src/tensor_engine_nmt/train.py` | Complete | Full loop with auto-resume + epoch tracking |
| `src/tensor_engine_nmt/inference.py` | Complete | Greedy + beam search |
| `src/tensor_engine_nmt/evaluate.py` | Complete | BLEU-4 from scratch + evaluate_bleu alias |
| `src/tensor_engine_nmt/__init__.py` | Complete | CLI entry points for all sub-commands |
| `test/test_shapes.py` | 5/5 pass | All tensor shape assertions |
| `test/test_gradients.py` | Pass | Numerical gradient check (float32 tolerance) |
| `test/test_overfit.py` | Pass | Loss driven to < 0.05 on 8-sentence batch |
| `PhoMT_dataset/` | Present | ~3M EN/VI parallel sentence pairs |
| `bpe_vocab/` | Generated | 31,622 merge rules, 30,839 tokens |
| `checkpoints/` | Active | Step checkpoints + `optim_state.npz` |

### 1.2 Active Training

- **Current position:** Epoch 2/30, step ~46,400
- **Current loss:** ~2.03 (from train_logs.txt)
- **Training platform:** Google Colab T4 GPU via `cupy-cuda12x`
- **Backend confirmed:** cupy (GPU)

---

## 2. Actual Hyperparameters (config.py)

> CHANGED FROM PLAN: Architecture was scaled up after initial experiments confirmed the pipeline was correct.

| Symbol | Original Plan | Actual Value | Change |
|---|---|---|---|
| V | 32,000 | 32,000 | Shared BPE vocabulary size |
| e | 256 | 512 | CHANGED — Doubled embedding dimension for richer representations |
| d | 512 | 1,024 | CHANGED — Doubled for T4 VRAM budget |
| L | 2 | 3 | CHANGED — +1 layer for complex EN to VI grammar |
| B | 64 (fixed) | dynamic via max_tokens=4000 | CHANGED — Dynamic token batching replaces fixed batch size |
| k | 2,000 | 20,000.0 | CHANGED — Higher decay constant for smooth curriculum |
| lr | 3e-4 | 3e-4 | Unchanged |
| beta1 | 0.9 | 0.9 | Unchanged |
| beta2 | 0.999 | 0.999 | Unchanged |
| eps_adam | 1e-8 | 1e-8 | Unchanged |
| clip_norm | 5.0 | 5.0 | Unchanged |
| max_epochs | 30 | 30 | Unchanged |
| beam_width | 4 | 4 | Unchanged |
| max_decode_len | 150 | 100 | Aligned with Colab & inference budget |
| log_every | 50 | 100 | Reduced I/O frequency |
| save_every | 500 | 2,000 | Aligned with storage budget (~6 checkpoints per 12h) |

---

## 3. Module Layout (as-built)

```
src/tensor_engine_nmt/
├── __init__.py        <- CLI: train / translate / evaluate sub-commands
├── config.py          <- HParams frozen dataclass (single source of truth)
├── backend.py         <- xp = cupy | numpy (auto-selected at import time)
├── init_weights.py    <- Xavier uniform, Orthogonal via QR
├── activations.py     <- sigmoid, tanh, softmax + their backward functions
├── bpe.py             <- Streaming BPE tokenizer; saves bpe_vocab/
├── dataset.py         <- PhoMTDataset: bucket sort, .pkl cache, iterate(start_idx=)
├── encoder.py         <- 3-layer LSTM encoder (forward + BPTT)
├── attention.py       <- Luong general attention (forward + backward)
├── decoder.py         <- 3-layer LSTM decoder (forward + BPTT + teacher forcing)
├── loss.py            <- Masked cross-entropy (forward + backward)
├── model.py           <- Seq2Seq wrapper; save() / load() with Adam state companion
├── optimizer.py       <- Adam with global grad norm clipping
├── train.py           <- Training loop with auto-resume, epoch inference, skip_batches
├── inference.py       <- Greedy decode + beam search (Translator class)
└── evaluate.py        <- Corpus BLEU-4 (from scratch) + evaluate_bleu alias

test/
├── test_shapes.py       <- 5 shape assertion tests
├── test_gradients.py    <- Numerical gradient check
└── test_overfit.py      <- Overfit smoke test

docs/
├── implementation_plan.md     <- This file (as-built reference)
├── unidirectional_pipeline.md <- Mathematical spec (unchanged)
└── training_guide.md          <- Operational guide: resume, checkpoints, Colab
```

---

## 4. Key Design Decisions and Deviations from Plan

### 4.1 Dynamic Token Batching (dataset.py)

The original plan used a fixed `B=64` batch size. The actual implementation uses
`max_tokens=4000` dynamic batching: each batch is filled with as many sentence pairs
as possible without exceeding 4000 total target tokens. This:
- Automatically handles variable-length sequences more efficiently
- Prevents OOM on long sentences (which fixed batching can miss)
- Is the standard approach used by production NMT systems

### 4.2 Pre-tokenized Cache (dataset.py)

A critical performance optimization not in the original plan. All 2,966,638 sentence
pairs are BPE-tokenized on the first run and saved to:
```
PhoMT_dataset/train_cache_V30839.pkl
```
Subsequent runs (and all epoch restarts) load in under 1 second, eliminating what
was previously a 5+ minute startup per session.

### 4.3 Dual Checkpoint Files (model.py)

The original plan used a single `.npz` file per checkpoint. The actual implementation
splits into two:

- `checkpoints/step_N.npz` — model weights only (~160 MB compressed, one per saved step)
- `checkpoints/optim_state.npz` — Adam m, v, t arrays (~320 MB, always overwritten)

**Reason:** Saving Adam state inside every weight checkpoint would make each file
~480 MB. With save_every=500 over 30 epochs (~1,390,590 steps = ~2,780 checkpoints),
that would require ~1.3 TB of Google Drive storage. The companion file approach
keeps total storage at: (2,780 x 160 MB) + 320 MB = ~445 GB for weights + 320 MB
for optimizer = manageable if old checkpoints are periodically deleted.

### 4.4 Numerical Checkpoint Sorting (train.py, evaluate.py)

The original plan used `sorted(glob.glob(...))` which is alphabetical. This causes
`step_16000.npz` to sort before `step_8000.npz` (because '1' < '8').

Fixed with a numerical key:
```python
key = lambda p: int(re.search(r'step_(\d+)', p).group(1))
```

### 4.5 Correct Epoch Tracking on Resume (train.py)

The original plan assumed training always starts from epoch 0. When resuming from
step 56,000 on a 46,354-step-per-epoch dataset, the model is actually 21% into epoch 2.

The fix computes position from global_step:
```python
steps_per_epoch  = len(dataset.pairs) // hp.B_effective
start_epoch      = global_step // steps_per_epoch
steps_done_in_ep = global_step %  steps_per_epoch
# Passed to iterate(start_idx=steps_done_in_ep) to skip already-trained batches
```

### 4.6 All Decoder Layers Initialized from Encoder Final State

The original plan initialized only the first decoder layer from the encoder. The
actual implementation copies the encoder final hidden/cell state to ALL L decoder
layers. This was found to give faster convergence in early training.

---

## 5. Checkpoint Save/Load Contract

```python
# Saving (train.py calls this every save_every steps and at epoch end)
model.save("checkpoints/step_N.npz", optim=optim)
# Writes: checkpoints/step_N.npz  (weights p0..p18)
# Writes: checkpoints/optim_state.npz  (adam_t, adam_m0..m18, adam_v0..v18)

# Loading (on resume)
model.load("checkpoints/step_N.npz", optim=optim)
# Reads: checkpoints/step_N.npz  -> restores all weights
# Reads: checkpoints/optim_state.npz -> restores Adam state
# Prints: "[model] Adam state restored (t=N) from checkpoints/optim_state.npz"
# Or:     "[model] Warning: no optim_state.npz found, optimizer starts fresh"
```

**Important:** When resuming, always load from the step_*.npz that was saved in the
same session as the most recent optim_state.npz. If you manually delete checkpoints,
keep the most recent step_N.npz AND the optim_state.npz together.

---

## 6. Verification Results

```bash
pytest test/test_shapes.py test/test_overfit.py -v
```

| Test | Assertion | Result |
|---|---|---|
| test_encoder_shapes | H: (B,Tx,d), h_enc: (L,B,d) | PASS |
| test_attention_shapes | alpha: (B,Tx), z: (B,d) | PASS |
| test_decoder_shapes | logits: (B,Ty,V) | PASS |
| test_loss_shapes | loss: scalar, dlogits: (B,Ty,V) | PASS |
| test_full_forward_backward | All grads accumulate, no NaN | PASS |
| test_overfit | loss 2.0 to < 0.05 in 300 steps | PASS |
| test_gradients | Max relative error < 1e-3 (float32) | PASS |

---

## 7. Known Issues Resolved

| Issue | Root Cause | Fix Applied |
|---|---|---|
| Loss spikes on every resume | Adam m, v, t not saved | optim_state.npz companion file |
| Wrong epoch label on resume | start_epoch hardcoded to 0 | Computed from global_step // steps_per_epoch |
| Double-training mid-epoch data | No batch skip on resume | iterate(start_idx=N) |
| Checkpoint sort picks step_8000 over step_16000 | sorted() is alphabetical | Numerical regex key |
| 5-minute startup every run | BPE tokenization at runtime | .pkl pre-tokenized cache |
| Checkpoint storage explosion | Adam state in every .npz | Separate optim_state.npz, always overwritten |
| evaluate() not importable as evaluate_bleu | Name mismatch in guide | evaluate_bleu = evaluate alias added |

---

## 8. Dependencies

```toml
# pyproject.toml
dependencies = [
    "numpy>=1.24",
    "tqdm>=4.64",
    # cupy-cuda12x   # GPU -- install separately: pip install cupy-cuda12x
]
```

No PyTorch. No TensorFlow. No autograd. Pure NumPy/CuPy.
