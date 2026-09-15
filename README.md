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
- **Smart Checkpoint Resume**: Training automatically finds and resumes from the latest checkpoint, sorted numerically — so `step_16000` is correctly ranked after `step_8000`. Optimizer (Adam) state is preserved in a companion `optim_state.npz` to prevent cold-start loss spikes on resume.
- **Deterministic Mid-Epoch Resumption**: The dataset shuffler uses an epoch-based random seed (`seed = 42 + epoch`). If a run crashes mid-epoch, the dataloader recreates the exact same shuffle order before skipping exactly the batches already trained, so there is zero data duplication or loss.
- **Dual Checkpoint Files**: Model weights (`step_N.npz`) and Adam optimizer state (`optim_state.npz`) are stored separately, reducing checkpoint storage from ~480 MB to ~160 MB per save step.
- **Dynamic Token Batching**: Instead of a fixed batch size, each batch is filled with as many sentence pairs as possible without exceeding `max_tokens` total tokens. This maximises GPU utilisation regardless of sentence length variance.
- **Beam Search Decoding**: The inference engine maintains multiple hypotheses and their respective LSTM states at every decode step.

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

Training on a personal laptop CPU is extremely slow. The recommended approach is to use a **free Google Colab T4 GPU**, which is 50–100× faster.

See the step-by-step migration guide: **[COLAB_GUIDE.md](COLAB_GUIDE.md)**

Expected performance with the current 3-layer `d=1024` architecture and `max_len=30` sentence filter:

| Metric | CPU (Laptop) | GPU (Colab T4) |
|---|---|---|
| Speed | ~15 tok/s | ~750–1,600 tok/s |
| Time per epoch (~20k steps) | — | ~6–8 hours |
| VRAM required | N/A | ~8–12 GB (`max_tokens=4000`) |

---

## 🛠️ Usage (Local CLI)

### 1. Train the Model

Start training from scratch or auto-resume from the latest checkpoint:
```bash
uv run tensor-engine-nmt train
```

*Options:*
- `--lr`: Learning rate (default: `1e-4`)
- `--max-tokens`: Dynamic token-level batch size limit (default: `4000`)
- `--min-tf`: Minimum teacher-forcing ratio floor (default: `0.70`)
- `--max-steps`: Stop after N gradient steps.
- `--resume PATH`: Resume from a specific `.npz` checkpoint.

The training log prints every `log_every` steps:
```
ep 2/30 | step  46000 | loss=2.03 | lr=1.00e-04 | gnorm=0.553 | ε=0.999 | tok/s=1,612
[model] Checkpoint saved → checkpoints/step_46000.npz
```

For a detailed explanation of each log field, the checkpoint system, and Colab workflow, see **[docs/training_guide.md](docs/training_guide.md)**.

### 2. Interactive Translation (Demo)
```bash
uv run tensor-engine-nmt translate
```
Type an English sentence and press Enter. The model uses Beam Search (width=4) to output a Vietnamese translation.

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
| `--split` | `test` | Dataset split: `test`, `train`, or `val`. |
| `--n` | All | Evaluate up to N sentences (counted after length filtering). |
| `--random` | `False` | Shuffle with fixed seed (`42`) before picking N sentences. |
| `--verbose` | `False` | Print English source, hypothesis, reference, and sentence BLEU for every pair. |
| `--no-filter` | `False` | Disable the `max_len` filter and evaluate the full unfiltered split. |

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

| Parameter | Default | Tradeoff & Impact |
|---|---|---|
| `max_len` | `30` | **[Length Budget]** Maximum BPE token count for either the EN or VI side. Applied identically at training (filters dataset) and evaluation (filters test pairs). Increase to cover longer sentences; decrease to speed up training on short-sentence accuracy. |
| `d` | `1024` | **[Structural]** LSTM hidden size. For the **bidirectional encoder**, each direction uses `d/2=512`, concatenated to `d=1024`. The unidirectional decoder uses `d=1024` directly. Larger = more capacity but more VRAM and slower. |
| `L` | `3` | **[Structural]** Number of stacked LSTM layers (same for encoder and decoder). |
| `e` | `512` | **[Structural]** Embedding dimension. |
| `max_tokens` | `4000` | **[Training]** Dynamic token batching limit. Tune to fit your GPU VRAM. |
| `k` | `17,000` | **[Training]** Inverse-sigmoid teacher-forcing decay constant. Higher = stays in teacher-forcing mode longer. At `k=17,000`, ε crosses 0.5 at ~step 200,000 (≈epoch 4). |
| `min_tf` | `0.70` | **[Training]** Teacher-forcing floor. Prevents the ratio from dropping below 70%, reducing exposure bias collapse. |
| `lr` | `1e-4` | **[Training]** Adam learning rate. Can be adjusted mid-run when resuming. |
| `clip_norm` | `2.0` | **[Training]** Global gradient norm clip threshold. |
| `beam_width` | `4` | **[Inference]** Number of beam search hypotheses. |
| `max_decode_len` | `30` | **[Inference]** Hard cap on decoder output tokens. Keep equal to `max_len`. |

> **Resuming Rules:**
> ❌ **[Structural]** parameters (`d`, `L`, `e`) **cannot** be changed when resuming from a checkpoint — the saved matrix dimensions must match the code exactly.
> ✅ **[Training]** and **[Inference]** parameters can be changed freely when resuming.

---

## 🧪 Verification & Testing

```bash
pytest test/test_shapes.py test/test_overfit.py test/test_gradients.py -v
```

| Test | What it verifies | Result |
|---|---|---|
| `test_shapes.py` | All tensor dimensions match the spec end-to-end | ✅ 5/5 pass |
| `test_overfit.py` | Model drives loss from ~2.0 to <0.05 on a tiny batch in 300 steps | ✅ Pass |
| `test_gradients.py` | Analytical BPTT gradients match numerical finite-difference | ✅ Pass (float32 tolerance) |

---

## 📁 Project Structure

```
tensor-engine-nmt/
├── src/tensor_engine_nmt/
│   ├── config.py        ← All hyperparameters (single source of truth)
│   ├── backend.py       ← Transparent NumPy/CuPy xp alias
│   ├── activations.py   ← sigmoid, tanh, softmax + derivatives
│   ├── init_weights.py  ← Glorot uniform, Orthogonal init
│   ├── bpe.py           ← Streaming BPE tokenizer
│   ├── dataset.py       ← Bucket-sorted loader + .pkl cache + max_len filter
│   ├── encoder.py       ← 3-layer BiLSTM encoder (fwd+bwd, d/2 each, concat to d)
│   ├── attention.py     ← Luong general attention (forward + backward)
│   ├── decoder.py       ← 3-layer unidirectional LSTM decoder (forward + BPTT)
│   ├── loss.py          ← Masked cross-entropy loss
│   ├── model.py         ← Seq2Seq wrapper; split weight / optimizer checkpoint
│   ├── optimizer.py     ← Adam + global grad norm clipping
│   ├── train.py         ← Training loop with auto-resume & mid-epoch skip
│   ├── inference.py     ← Greedy & beam search decode
│   └── evaluate.py      ← Corpus BLEU-4 with consistent max_len filter
├── test/
│   ├── test_shapes.py
│   ├── test_gradients.py
│   └── test_overfit.py
├── docs/
│   ├── implementation_plan.md      ← As-built architecture reference
│   ├── bidirectional_pipeline.md  ← Mathematical walkthrough
│   └── training_guide.md           ← Operational guide: resume, Colab, logs
├── analyze.py           ← Training log & evaluation result analyser
├── PhoMT_dataset/       ← EN/VI parallel corpus (train / test splits)
├── bpe_vocab/           ← Auto-generated BPE vocab (merges.txt, vocab.json)
├── checkpoints/         ← step_N.npz (weights) + optim_state.npz (Adam state)
├── README.md
├── COLAB_GUIDE.md       ← Step-by-step Google Colab training guide
└── pyproject.toml
```
