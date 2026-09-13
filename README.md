# Tensor Engine NMT (Neural Machine Translation)

A complete Sequence-to-Sequence (Seq2Seq) Neural Machine Translation model built entirely from scratch in **NumPy** and **CuPy**.

This project implements an English-to-Vietnamese translation system using a stacked 3-layer **Bidirectional LSTM** encoder and unidirectional LSTM decoder with Luong's General Attention mechanism. It features completely manual backpropagation through time (BPTT), a custom Byte-Pair Encoding (BPE) tokenizer, and a dynamic CPU/GPU backend — all without using modern deep learning frameworks like PyTorch or TensorFlow.

---

## 🚀 Key Features

- **Transparent Backend (`xp`)**: Automatically uses GPU acceleration via `CuPy` if available; otherwise, seamlessly falls back to CPU-based `NumPy`.
- **Manual Backpropagation (BPTT)**: No autograd! Gradients are mathematically derived and manually propagated back through the decoder, attention mechanism, and encoder.
- **Custom BPE Tokenizer**: An extremely fast, incremental Byte-Pair Encoding tokenizer written from scratch that handles large text corpora without exploding memory.
- **Pre-tokenized Dataset Cache**: The dataset is tokenized once on the first run and saved to a `.pkl` cache file. Every subsequent epoch loads in under 1 second, eliminating the startup bottleneck.
- **Smart Checkpoint Resume**: Training automatically finds and resumes from the latest checkpoint (sorted numerically, not alphabetically — so `step_16000` is correctly ranked after `step_8000`).
- **Deterministic Mid-Epoch Resumption**: The dataset shuffler uses an epoch-based random seed (`seed = 42 + epoch`). If your run crashes mid-epoch, the dataloader recreates the *exact* same random shuffle order for that epoch before skipping the exact number of batches you already trained on, ensuring zero data duplication or loss.
- **Batched Bucket Loading**: A custom data loader that streams text, bucket-sorts by length, and collates batches with dynamic `[PAD]` tokens.
- **Beam Search Decoding**: Integrated inference engine that maintains multiple hypotheses and their respective LSTM states for high-quality translations.
- **Adam Optimizer**: Complete with bias correction and global gradient norm clipping.

---

## 🧠 End-to-End Pipeline

The entire translation pipeline is modeled inside this repository. Here is how data flows through the system:

### 1. Tokenization (`bpe.py`)
Raw English and Vietnamese text files are processed to build a **shared subword vocabulary**. The tokenizer learns rules to merge common character sequences (e.g., `"t" + "h" → "th"`) to represent out-of-vocabulary words effectively. The result is saved to `bpe_vocab/`.

### 2. Data Loading (`dataset.py`)
Raw sentences are converted into integer IDs using the BPE vocabulary **once**, then cached to disk as `train_cache_V<vocab_size>.pkl`. On all future runs and epochs, the cache is loaded in under 1 second. The dataset also bucket-sorts sentences of similar lengths together to minimize padding waste.

### 3. Forward Pass
* **Encoder (`encoder.py`)**: The English tokens are embedded and processed by a 3-layer stacked LSTM. The final hidden states and the full sequence memory (`H`) are passed to the decoder.
* **Decoder (`decoder.py`)**: Initialized with the encoder's final state, the decoder generates the Vietnamese sentence step-by-step using **Teacher Forcing** during training.
* **Attention (`attention.py`)**: At every decoder step, Luong's Attention calculates alignment scores against the encoder's memory (`H`) to focus on the relevant English words.
* **Loss (`loss.py`)**: The model's predictions are compared against the true Vietnamese sentence using a Masked Cross-Entropy Loss function (ignoring padded tokens).

### 4. Backward Pass (BPTT)
The gradient of the loss with respect to the output logits (`dlogits`) is calculated and pushed backward in time:
* Through the decoder's LSTM steps.
* Through the attention mechanism to calculate the gradient for the encoder's memory (`dH`).
* Down through the encoder's LSTM steps.

### 5. Optimization (`optimizer.py` & `model.py`)
All computed gradients are collected. The `Adam` optimizer calculates moving averages of the gradients (moments), applies a global gradient norm clip to prevent exploding gradients, and updates the weights. Checkpoints are saved to `checkpoints/` at regular intervals.

---

## 💻 Installation (Local)

Ensure you have Python 3.10+ installed. This project uses `uv` for dependency management.

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

> **Note for Windows users**: `cupy` requires the NVIDIA CUDA Toolkit DLLs (e.g. `cublas64_12.dll`) to be installed separately. The display driver alone is NOT sufficient. See [COLAB_GUIDE.md](COLAB_GUIDE.md) for training on a free cloud GPU instead.

---

## ☁️ Training on Google Colab (Recommended)

Training on a personal laptop CPU is extremely slow. The recommended approach is to use a **free Google Colab T4 GPU**, which is ~50–80× faster.

See the step-by-step migration guide: **[COLAB_GUIDE.md](COLAB_GUIDE.md)**

Quick summary of what to expect on a T4 GPU for the massive 3-layer `d=1024` architecture:

| Metric | CPU (Laptop) | GPU (Colab T4) |
|---|---|---|
| Speed | ~15 tok/s | ~750 - 800 tok/s |
| Time to loss < 3.0 | 1+ month | ~34 hours |
| VRAM required | N/A | ~14 GB (`max_tokens=4000`) |

---

## 🛠️ Usage (Local CLI)

A unified command-line interface is provided to interact with the engine.

### 1. Train the Model
To start training from scratch (or auto-resume from the latest checkpoint):
```bash
uv run tensor-engine-nmt train
```
*Options:*
- `--lr`: Set the learning rate (default: `3e-4`)
- `--max-tokens`: Set the dynamic token-level batch size limit (default: `4000`)
- `--min-tf`: Set the minimum teacher-forcing ratio floor (default: `0.70`)
- `--max-steps`: Stop training after N gradient steps.
- `--resume`: Path to a specific `.npz` checkpoint file to resume from.

The training log prints every `log_every` steps:
```
step   2000  loss=4.71  gnorm=0.917  ε=1.000  tok/s=1,320
[model] Checkpoint saved → checkpoints/step_2000.npz
```

### 2. Interactive Translation (Demo)
Once you have saved checkpoints, run an interactive REPL:
```bash
uv run tensor-engine-nmt translate
```
Type an English sentence and press Enter — the model uses Beam Search to output a Vietnamese translation.

### 3. Evaluate (Corpus BLEU-4)
To calculate the BLEU-4 score on your test dataset:
```bash
uv run tensor-engine-nmt evaluate --verbose
```

> **Evaluation Metrics Tradeoff**: This codebase implements **BLEU-4 entirely from scratch** to maintain the "no external ML dependencies" philosophy of the project. While highly educational, BLEU is notoriously rigid for languages with varied pronouns and synonyms (like Vietnamese). It penalizes perfectly valid semantic translations that don't match the reference's exact n-grams (e.g., translating "Are you ready?" as "anh đã sẵn sàng chưa" instead of the reference's "bà đã sẵn sàng chưa" yields a sentence-level BLEU of 0.0). 
> 
> **Future Alternatives**: To assess the true semantic quality of your translations, consider running these external tools against your `test.vi` predictions:
> - **BERTScore** (`pip install bert-score`): The modern gold standard. Evaluates *semantic meaning* rather than exact word overlap using pre-trained embeddings.
> - **chrF++** (`pip install sacrebleu`): A statistical upgrade to BLEU that measures character n-gram overlap, making it much more resilient to synonyms and morphology.

---

## ⚙️ Hyperparameter Tuning Tradeoffs

Because the model architecture is defined purely in NumPy/CuPy, you have full control over the structural hyperparameters in `config.py`. If you want to start a completely new training run from scratch, here are the key parameters you can tune and their tradeoffs:

| Parameter | Default | Tradeoff & Impact |
|---|---|---|
| `d` (Hidden Dim) | `1024` | **[Structural]** Controls the internal memory capacity of the LSTM. Increasing to `2048` learns more complex patterns, but uses much more VRAM and slows training. |
| `L` (Layers) | `3` | **[Structural]** Number of stacked LSTM layers. `L=3` provides deep representations. `L=4` might overfit, while `L=1` is too shallow for complex grammar. |
| `e` (Embedding) | `512` | **[Structural]** Size of the word vectors. `512` gives rich semantic embeddings for complex bilingual vocabularies. |
| `max_tokens` | `4000` | **[Training]** Dynamic token batching limit. Tuned to maximize a 16GB T4 GPU. Increase this if using an A100 (40GB) to get massive speedups. |
| `k` (Teacher Forcing) | `17000.0` | **[Training]** Inverse-sigmoid decay constant. Controls how fast scheduled sampling introduces model predictions. |
| `min_tf` (Floor) | `0.70` | **[Training]** Minimum teacher forcing ratio floor. Prevents exposure bias collapse by guaranteeing at least 70% ground truth tokens. |
| `lr` (Learning Rate)| `3e-4` | **[Training]** Base learning rate for Adam. Can be adjusted mid-run if you see the loss plateau. |

> **Resuming Rules:** 
> ❌ You **cannot** change **[Structural]** parameters if you are resuming from an existing checkpoint. The matrix dimensions in the saved `.npz` file will strictly crash if they do not match the code! You must delete `checkpoints/` to change these.
> ✅ You **can** safely change **[Training]** parameters (`max_tokens`, `k`, `min_tf`, `lr`) at any time when resuming a checkpoint. For example, you can pause training, lower the learning rate or teacher-forcing floor, and hit play again!

---

## 🧪 Verification & Testing

The mathematical correctness of the manual gradients has been rigorously verified with a comprehensive test suite in the `test/` folder:

```bash
pytest test/test_shapes.py test/test_overfit.py
```

| Test | What it verifies | Result |
|---|---|---|
| `test_shapes.py` | All tensor dimensions match the spec end-to-end | ✅ 5/5 pass |
| `test_overfit.py` | Model can drive loss from ~2.0 to ~0.03 on a tiny batch | ✅ Pass |
| `test_gradients.py` | Analytical BPTT gradients match numerical finite-difference | ✅ Pass (float32 tolerance) |

---

## 📁 Project Structure

```
tensor-engine-nmt/
├── src/tensor_engine_nmt/
│   ├── config.py        ← All hyperparameters (HParams dataclass)
│   ├── backend.py       ← Transparent NumPy/CuPy xp alias
│   ├── activations.py   ← sigmoid, tanh, softmax + derivatives
│   ├── init_weights.py  ← Glorot uniform, Orthogonal init
│   ├── bpe.py           ← Incremental BPE tokenizer
│   ├── dataset.py       ← Bucket-sorted loader + .pkl cache
│   ├── encoder.py       ← 3-layer LSTM encoder (forward + BPTT)
│   ├── attention.py     ← Luong general attention (forward + backward)
│   ├── decoder.py       ← 3-layer LSTM decoder (forward + BPTT)
│   ├── loss.py          ← Masked cross-entropy loss
│   ├── model.py         ← Seq2Seq wrapper, checkpoint save/load
│   ├── optimizer.py     ← Adam + global grad norm clipping
│   ├── train.py         ← Training loop with auto-resume
│   ├── inference.py     ← Greedy & beam search decode
│   └── evaluate.py      ← Corpus BLEU-4 from scratch
├── test/
│   ├── test_shapes.py
│   ├── test_gradients.py
│   └── test_overfit.py
├── PhoMT_dataset/       ← EN/VI parallel corpus
├── bpe_vocab/           ← Auto-generated BPE vocab (merges.txt, vocab.json)
├── checkpoints/         ← Auto-saved model weights (.npz)
├── README.md
├── COLAB_GUIDE.md       ← Step-by-step Google Colab training guide
└── pyproject.toml
```
