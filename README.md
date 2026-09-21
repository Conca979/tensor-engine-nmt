# Tensor Engine NMT (Neural Machine Translation)

A complete Sequence-to-Sequence (Seq2Seq) Neural Machine Translation model built entirely from scratch in **NumPy** and **CuPy**.

This project implements an English-to-Vietnamese translation system using a stacked 3-layer **Bidirectional LSTM (BiLSTM)** encoder and a stacked 3-layer unidirectional LSTM decoder with Luong's General Attention mechanism. It features completely manual backpropagation through time (BPTT), a custom Byte-Pair Encoding (BPE) tokenizer, and a dynamic CPU/GPU backend — all without using modern deep learning frameworks like PyTorch or TensorFlow.

---

## 🚀 Key Features

- **Transparent Backend (`xp`)**: Automatically uses GPU acceleration via `CuPy` if available; otherwise falls back to CPU-based `NumPy`. One import switch covers the entire codebase.
- **Manual Backpropagation (BPTT)**: No autograd. Gradients are mathematically derived and manually propagated through the decoder, attention mechanism, and encoder.
- **Custom BPE Tokenizer**: A streaming, incremental Byte-Pair Encoding tokenizer written from scratch that handles the full 3-million-pair corpus without loading everything into memory.
- **Pre-tokenized Dataset Cache**: The dataset is tokenized once on the first run and saved to a `.pkl` cache file. Every subsequent epoch loads in under 1 second, eliminating the startup bottleneck.
- **Consistent Length Budget**: A single `max_len` hyperparameter controls the maximum BPE token count for both training (pairs exceeding the limit are filtered out) and evaluation (same filter applies, so BLEU is measured on the same distribution the model was trained on).
- **Smart Checkpoint Resume**: Training automatically finds and resumes from the latest checkpoint, sorted numerically — so `step_16000` is correctly ranked after `step_8000`. Optimizer (Adam) state is preserved in a companion `optim_state.npz`, which is tagged with the checkpoint it belongs to so it can never be silently paired with the wrong weights.
- **Deterministic Mid-Epoch Resumption**: The dataset shuffler uses an epoch-based random seed (`seed = 42 + epoch`) and the teacher-forcing sampler is seeded from `global_step`, so a resumed run replays exactly the same batches and the same teacher-forcing decisions. The position within the epoch is stored *inside* the checkpoint, so weights and dataset position cannot disagree after a crash.
- **Dual Checkpoint Files**: Model weights (`step_N.npz`) and Adam optimizer state (`optim_state.npz`) are stored separately, so the optimizer state is written once rather than per checkpoint. Both are written via temp-file + atomic rename, so a pre-empted session cannot leave a half-written checkpoint behind.
- **Dynamic Token Batching**: Instead of a fixed batch size, each batch is filled with as many sentence pairs as possible without exceeding `max_tokens` total tokens. This maximises GPU utilisation regardless of sentence length variance.
- **Beam Search Decoding**: The inference engine maintains multiple hypotheses and their respective LSTM states at every decode step, with length normalisation applied consistently to both beam pruning and final selection.

---

## 🧠 End-to-End Pipeline

### 1. Tokenization (`bpe.py`)
Raw English and Vietnamese text files are processed to build a **shared subword vocabulary**. The tokenizer learns rules to merge common character sequences (e.g., `"t" + "h" → "th"`) to represent out-of-vocabulary words effectively. The result is saved to `bpe_vocab/`.

### 2. Data Loading (`dataset.py`)
Raw sentences are converted into integer IDs using the BPE vocabulary **once**, then cached to disk as `train_cache_V<vocab_size>.pkl`. On all future runs and epochs, the cache loads in under 1 second.

The dataset also applies the **length budget** (`max_len=30`): any sentence pair where either the English or Vietnamese BPE token count exceeds `max_len` is dropped. This keeps training focused on short-to-medium sentences where a small-scale model can develop meaningful accuracy quickly.

### 3. Forward Pass
- **Encoder (`encoder.py`)**: English tokens are embedded and processed by a 3-layer **Bidirectional** LSTM. Each direction (forward `t=1→Tx` and backward `t=Tx→1`) has hidden size `d/2=512`. Their outputs are concatenated at every timestep to produce `H ∈ (B, Tx, d)`, where `d=1024`. The bidirectional design lets each encoder position see both past and future context, which is particularly beneficial for long-range syntactic dependencies in English.
- **Decoder (`decoder.py`)**: Initialised with the encoder's final concatenated state (already `d`-wide — no bridge projection needed), the decoder generates the Vietnamese sentence step-by-step using a **Teacher Forcing** schedule (inverse-sigmoid decay).
- **Attention (`attention.py`)**: At every decoder step, Luong's General Attention calculates alignment scores against `H` to focus on the relevant English tokens.
- **Loss (`loss.py`)**: Predictions are compared against the true Vietnamese tokens using Masked Cross-Entropy (padding tokens are excluded from the gradient).

### 4. Backward Pass (BPTT)
The gradient of the loss (`dlogits`) is pushed backward:
1. Through all decoder LSTM steps (BPTT).
2. Through the attention mechanism → gradient into encoder memory (`dH`).
3. Down through all encoder LSTM steps.

### 5. Optimization (`optimizer.py` & `model.py`)
Gradients are globally norm-clipped (`clip_norm=2.0`) and fed into Adam (with bias correction). Checkpoints are saved to `checkpoints/` every `save_every` steps.

---

## 💻 Installation (Local)

Ensure you have Python 3.10+ and `uv` installed.

```bash
# Clone the repository
git clone <repository_url>
cd tensor-engine-nmt

# Install dependencies (numpy, tqdm)
uv sync

# Optional: Install CuPy for GPU acceleration
# First, install the NVIDIA CUDA Toolkit 12.x from https://developer.nvidia.com/cuda-downloads
# Then:
uv add cupy-cuda12x
```

> **Note for Windows users**: `cupy` requires the NVIDIA CUDA Toolkit DLLs (e.g. `cublas64_12.dll`) — the display driver alone is NOT sufficient. See [COLAB_GUIDE.md](COLAB_GUIDE.md) for training on a free cloud GPU instead.

---

## ☁️ Training on Google Colab (Recommended)

Training on a personal laptop CPU is extremely slow. The recommended approach is to use a **free Google Colab T4 GPU**.

See the step-by-step migration guide: **[COLAB_GUIDE.md](COLAB_GUIDE.md)**

**Training in short bursts (limited GPU hours, or a laptop you can't leave running)?**
Read **[docs/short_session_training.md](docs/short_session_training.md)** first — it covers
measuring your own hardware, the `laptop` preset, time-boxed runs that stop cleanly, and
checkpoint discipline. Start by measuring, always:

```bash
uv run tensor-engine-nmt bench --preset gpu      # the full-size model
uv run tensor-engine-nmt bench --preset laptop   # a ~5.9M-param model
```

Measured on the laptop this project was developed on (no CUDA, `Backend: numpy`):

| Metric | `--preset gpu` (full model) | `--preset laptop` | GPU (Colab T4) |
|---|---|---|---|
| Parameters | 108,585,216 | 5,877,824 | 108,585,216 |
| Throughput | 27 tok/s | 962 tok/s | ~1,000–1,600 tok/s |
| Seconds per step | 9.25 s | 0.52 s | ~3–4 s |
| **Time per epoch** | **~40 days** | **~35 min** | **~6–8 hours** |
| 10 epochs | ~400 days | **5.8 h** | ~3 days |
| VRAM required | N/A (CPU) | N/A (CPU) | ~8–12 GB (`max_tokens=5000`) |

> The full-size model is simply not trainable on a CPU — that column is a
> measurement, not a guess. The laptop preset fits its whole 10-epoch schedule into
> an evening. Run `bench` on your own machine before planning anything.

---

## 🛠️ Usage (Local CLI)

### 0. Measure Your Machine (do this first)

```bash
uv run tensor-engine-nmt bench --preset gpu      # full-size model
uv run tensor-engine-nmt bench --preset laptop   # small model
```

Runs real training steps and prints measured tok/s, seconds per step, and projected
hours per epoch. It never touches your checkpoints, so it is always safe to run.
Use it to decide which preset and how long a session you can afford.

### 1. Train the Model

Start training from scratch or auto-resume from the latest checkpoint:
```bash
uv run tensor-engine-nmt train
```

*Options:*
- `--preset gpu|laptop`: Named hyperparameter bundle. `laptop` is a ~5.9M-param model on a strided 150k-pair subset with its own `bpe_vocab_laptop/` and `checkpoints_laptop/` (default: `gpu`).
- `--lr`: Learning rate (config default `1e-5`; the step-196k checkpoint used `1e-4`).
- `--max-tokens`: Dynamic token-level batch size limit (default: `5000`).
- `--k`, `--min-tf`: Teacher-forcing decay constant and floor (defaults `17000` / `0.70`).
- `--max-minutes N`: **Stop cleanly after N wall-clock minutes** — writes a checkpoint first. This is how you fit training into a short session.
- `--save-every N`, `--log-every N`: Checkpoint / log cadence (defaults `4000` / `100`).
- `--keep-last N`: After each save, delete older `step_*.npz` beyond the newest N (keeps the disk bounded).
- `--max-pairs N`: Cap the training set to an evenly-strided sample of N pairs.
- `--max-steps N`: Stop after N gradient steps.
- `--resume PATH`: Resume from a specific `.npz` checkpoint.

Example — a two-hour session that resumes automatically next time:
```bash
uv run tensor-engine-nmt train --preset laptop --max-minutes 120
```

The training log prints every `log_every` steps:
```
ep 8/30 | step 196700 | loss=2.5364 | lr=1.00e-04 | gnorm=1.194 | ε=0.700 | tok/s=1,064 | ~2.31s/step, epoch≈15.8h
[model] Checkpoint saved -> checkpoints/step_196000.npz
```

For a detailed explanation of each log field, the checkpoint system, and the resume
mechanics, see **[docs/training_guide.md](docs/training_guide.md)**. For short
sessions, see **[docs/short_session_training.md](docs/short_session_training.md)**.

### 2. Interactive Translation (Live Demo)
```bash
# Launch the rich interactive terminal demo (auto-loads latest checkpoint)
python demo.py

# Or via CLI
uv run tensor-engine-nmt translate
```
Type an English sentence and press Enter. The model uses Beam Search (width=4) to output a Vietnamese translation.
You can also compare Greedy vs. Beam Search side-by-side with latency metrics:
```bash
python demo.py --method both --beam-width 5
```

### 3. Evaluate (Corpus BLEU-4)

```bash
# Evaluate on test set (auto-finds latest checkpoint, applies max_len filter)
uv run tensor-engine-nmt evaluate --verbose

# Fast spot-check on first 100 filtered test sentences
uv run tensor-engine-nmt evaluate --n 100

# Diagnose overfitting: random sample of 500 training pairs
uv run tensor-engine-nmt evaluate --split train --n 500 --random --verbose

# Evaluate the full unfiltered test set (for reference; includes long sentences)
uv run tensor-engine-nmt evaluate --no-filter

# Evaluate a specific checkpoint
uv run tensor-engine-nmt evaluate --ckpt checkpoints/step_46000.npz --n 200
```

#### Evaluation Flags

| Flag | Default | Description |
|---|---|---|
| `--ckpt PATH` | Latest | Specific `.npz` checkpoint (auto-sorts numerically by step if omitted). |
| `--method` | `beam` | Decoding method: `beam` (high accuracy, width=4) or `greedy` (fast). |
| `--split` | `test` | Dataset split: `test`, `train`, or `val`. |
| `--n` | All | Evaluate up to N sentences (counted after length filtering). |
| `--random` | `False` | Shuffle with fixed seed (`42`) before picking N sentences. |
| `--verbose` | `False` | Print English source, hypothesis, reference, and sentence BLEU for every pair. |
| `--no-filter` | `False` | Disable the `max_len` filter and evaluate the full unfiltered split. |
| `--out PATH` | `checkpoints/evaluation_result.txt` | File to append detailed evaluation records and BLEU summary. |

#### Length Filter and BLEU Comparability

By default, `evaluate` applies the **same `max_len=30` BPE token filter** that the training dataset uses. Any test sentence where either the English or Vietnamese side exceeds `max_len` BPE tokens is skipped. The output log shows exactly how many were skipped:

```
[evaluate] Length filter (max_len=30): 998 kept / 7,896 total (6,898 skipped, 87.4%)
[evaluate] Corpus BLEU-4 (test, 998 sentences, max_len≤30): 10.20
```

This ensures BLEU is measured on the **same distribution the model was trained on**, giving a meaningful comparison. Use `--no-filter` if you need the full-corpus number for reporting purposes.

> **Diagnosing Overfitting vs. Underfitting**: If training loss drops and training BLEU (`--split train --n 500 --random`) is high while test BLEU is low, the model is overfitting. If both remain low and grow together, the model is still learning.

> **Evaluation Metrics Tradeoff**: BLEU-4 is built from scratch to maintain the "no external ML dependencies" philosophy. It is notoriously rigid for Vietnamese — translating "Are you ready?" as `"anh đã sẵn sàng chưa"` instead of the reference's `"bà đã sẵn sàng chưa"` yields sentence BLEU = 0.0 despite being semantically equivalent. Corpus BLEU is more robust. Consider `chrF++` (`pip install sacrebleu`) for character-level overlap, or `BERTScore` (`pip install bert-score`) for semantic similarity.

---

## ⚙️ Hyperparameters (config.py)

All numbers live in one frozen dataclass in [`src/tensor_engine_nmt/config.py`](src/tensor_engine_nmt/config.py). Nothing is hard-coded anywhere else.

Named bundles of overrides are in `PRESETS` and selected with `--preset`:

| Preset | Model | Use case |
|---|---|---|
| `gpu` (default) | `d=1024 L=3 e=512 V=32000 max_len=30 max_tokens=5000`, all 2.97M pairs, `save_every=4000` | The full-size model. Needs a GPU. |
| `laptop` | `d=256 L=2 e=128 V=8000 max_len=20 max_tokens=1000`, a strided 150k-pair subset, `save_every=100 keep_last=3`, own `bpe_vocab_laptop/` + `checkpoints_laptop/` | A ~5.9M-param model a CPU can actually train (~35 min/epoch). |

Run `uv run tensor-engine-nmt bench --preset <name>` to see what either costs on your machine.

| Parameter | Default | Tradeoff & Impact |
|---|---|---|
| `max_len` | `30` | **[Length Budget]** Maximum BPE token count for either the EN or VI side. Applied identically at training (filters dataset), evaluation (filters test pairs) and to the cache filename, so changing it can never silently reuse a shorter-sentence cache. Increase to cover longer sentences; decrease to speed up training on short-sentence accuracy. |
| `max_pairs` | `None` | **[Length Budget]** Cap the training set to an evenly-strided sample of N pairs (`None` = all 2.97M). Shrinks the cache, the epoch and the per-session cost. |
| `keep_last` | `None` | **[Checkpointing]** After each save, delete older `step_*.npz` beyond the newest N. Set this whenever `save_every` is small. |
| `d` | `1024` | **[Structural]** LSTM hidden size. For the **bidirectional encoder**, each direction uses `d/2=512`, concatenated to `d=1024`. The unidirectional decoder uses `d=1024` directly. Larger = more capacity but more VRAM and slower. |
| `L` | `3` | **[Structural]** Number of stacked LSTM layers (same for encoder and decoder). |
| `e` | `512` | **[Structural]** Embedding dimension. |
| `max_tokens` | `5000` | **[Training]** Dynamic token batching limit. Tune to fit your GPU VRAM. |
| `k` | `17,000` | **[Training]** Inverse-sigmoid teacher-forcing decay constant. Higher = stays in teacher-forcing mode longer. |
| `min_tf` | `0.70` | **[Training]** Teacher-forcing floor. Because of this floor ε reaches 0.70 at ~step 151,000 and then **stops decaying**; it never drops to 0.5. Lower `min_tf` (e.g. `0.4`) if you want the schedule to keep annealing. |
| `lr` | `1e-5` | **[Training]** Adam learning rate. Can be adjusted mid-run when resuming. (The step-196k run in `checkpoints/` used `1e-4`.) |
| `clip_norm` | `2.0` | **[Training]** Global gradient norm clip threshold. |
| `beam_width` | `4` | **[Inference]** Number of beam search hypotheses. |
| `max_decode_len` | `30` | **[Inference]** Hard cap on decoder output tokens. Keep equal to `max_len`. |

> **Resuming Rules:**
> ❌ **[Structural]** parameters (`d`, `L`, `e`) **cannot** be changed when resuming from a checkpoint — the saved matrix dimensions must match the code exactly.
> ✅ **[Training]** and **[Inference]** parameters can be changed freely when resuming.

---

## 🧪 Verification & Testing

```bash
# Every test is a plain script — no test runner required
python test/test_shapes.py
python test/test_gradients.py
python test/test_overfit.py
python test/test_smoke.py
python test/test_checkpoint.py
python test/test_short_session.py

# …or with pytest (needs the dev extra: uv sync --extra dev)
pytest test -v
```

| Test | What it verifies | Result |
|---|---|---|
| `test_shapes.py` | All tensor dimensions match the spec end-to-end | ✅ 5/5 pass |
| `test_gradients.py` | Analytical BPTT gradients match finite differences, on a batch with real padding and `Ty>1` | ✅ Pass |
| `test_overfit.py` | Model drives loss from ~2.0 to <0.05 on a tiny batch in 300 steps | ✅ Pass |
| `test_smoke.py` | BPE → cache → bucketed batching → forward/backward → Adam, on a 300-line corpus slice | ✅ Pass |
| `test_checkpoint.py` | Weights/Adam/epoch metadata round-trip; stale, truncated and mismatched checkpoints are rejected | ✅ Pass |
| `test_short_session.py` | Presets, `--max-minutes` clean stop + resume, `--keep-last` pruning, `--max-pairs` stride sampling | ✅ Pass |
| `verification/verify_bugs.py` | Regression gate for the two backward-pass bugs (see `verification/README.md`) | ✅ Pass |

> **On gradient checking in float32.** Perturbing a weight and differencing the
> real loss does *not* work here: the loss is ~2.07, its float32 ULP is ~2.4e-7,
> so with `ε=1e-3` the central difference is quantised in steps of ~1.2e-4 —
> larger than most gradients. `test_gradients.py` therefore checks a synthetic
> objective `J = <upstream, logits>` with unit-scale random upstream gradients,
> which puts the finite-difference signal 2–4 orders of magnitude above the
> noise floor. Entries that fall below the floor are reported as skipped rather
> than silently passing.

---

## 📁 Project Structure

```
tensor-engine-nmt/
├── src/tensor_engine_nmt/
│   ├── config.py        ← All hyperparameters + named presets (gpu / laptop)
│   ├── backend.py       ← Transparent NumPy/CuPy xp alias
│   ├── activations.py   ← sigmoid, tanh, softmax + derivatives
│   ├── init_weights.py  ← Glorot uniform, Orthogonal init
│   ├── bpe.py           ← Streaming BPE tokenizer
│   ├── dataset.py       ← Bucket-sorted loader + .pkl cache + max_len/max_pairs filter
│   ├── encoder.py       ← 3-layer BiLSTM encoder (fwd+bwd, padding-masked)
│   ├── attention.py     ← Luong general attention (forward + backward)
│   ├── decoder.py       ← Stacked LSTM decoder (forward + BPTT)
│   ├── loss.py          ← Masked cross-entropy loss
│   ├── model.py         ← Seq2Seq wrapper; atomic split weight/optimizer checkpoints
│   ├── optimizer.py     ← Adam + global grad norm clipping
│   ├── train.py         ← Training loop: auto-resume, time budgets, checkpoint pruning
│   ├── bench.py         ← Measure tok/s and hours-per-epoch on THIS machine
│   ├── inference.py     ← Greedy & beam search decode
│   └── evaluate.py      ← Corpus BLEU-4 with consistent max_len filter
├── test/
│   ├── test_shapes.py
│   ├── test_gradients.py
│   ├── test_overfit.py
│   ├── test_smoke.py           ← BPE → dataset → train step, on a 300-line slice
│   ├── test_checkpoint.py      ← save/load, stale & truncated checkpoint handling
│   └── test_short_session.py   ← presets, --max-minutes stop/resume, --keep-last
├── verification/
│   ├── README.md               ← Bug findings + before/after evidence
│   ├── verify_bugs.py          ← Regression gate for the backward-pass fixes
│   ├── verify_padding_leak.py  ← Padding must not change real-token encodings
│   └── compare_ab.py           ← Decode/BLEU A-B harness against a source tree
├── docs/
│   ├── implementation_plan.md      ← As-built architecture reference
│   ├── bidirectional_pipeline.md  ← Mathematical walkthrough
│   ├── naming_conventions.md      ← Tensor shapes, symbols, and naming rules
│   ├── commit_rules.md            ← Git commit rules and workflow standard
│   ├── short_session_training.md  ← Training in short bursts (START HERE if slow)
│   ├── bugfix_report.md           ← Backward-pass fixes, evidence and next steps
│   └── training_guide.md           ← Operational guide: resume, logs, troubleshooting
├── analyze.py           ← Training log & evaluation result analyser
├── PhoMT_dataset/       ← EN/VI parallel corpus (train / test splits)
├── bpe_vocab/           ← Auto-generated BPE vocab (merges.txt, vocab.json)
├── checkpoints/         ← step_N.npz (weights) + optim_state.npz (Adam state)
├── README.md
├── CONTRIBUTING.md      ← Contribution guidelines and conventional commit rules
├── COLAB_GUIDE.md       ← Step-by-step Google Colab training guide
└── pyproject.toml
```
