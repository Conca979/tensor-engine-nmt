# Training on Kaggle (Background Runs & High-Speed GPU)

Kaggle offers **free 30 hours of GPU time per week**, and importantly, allows you to click "Save & Run All" to run your training loop in the **background for up to 12 hours** without needing to keep a browser tab open.

This guide explains how to migrate your `tensor-engine-nmt` training from Google Colab to Kaggle seamlessly, detailing the exact notebook cells and critical concerns you need to be aware of.

---

## 1. Setup Your Kaggle Notebook

1. Go to [Kaggle](https://www.kaggle.com/) and create an account.
2. Click **Create** (Top left) -> **New Notebook**.
3. In the Notebook editor, look at the **Session Options** panel on the right.
4. **Accelerator:** Change this from `None` to **GPU P100** (highly recommended for memory bandwidth) or **GPU T4 x2**.
5. **Internet:** Ensure the Internet toggle is turned **ON**.

---

## 2. Upload Your Dataset & Migrating Colab Checkpoints

Unlike Colab where you mount Google Drive directly, in Kaggle you attach data as a "Dataset". Since you are migrating from Colab, you need to bring both your dataset and your recent checkpoints.

1. **Download Checkpoints from Google Drive:** Go to your Google Drive and download your latest `step_XXXX.npz` (or `epoch_1.npz`) and the `optim_state.npz` file to your laptop.
2. **Create Kaggle Datasets:**
   - On the right panel in your Kaggle notebook, click **Add Input**.
   - Click **Upload a Dataset**. 
   - First, upload the `PhoMT_dataset` folder. Name it `phomt-dataset`.
   - Next, upload another dataset containing your downloaded `.npz` checkpoints. Name it `nmt-checkpoints`.
3. Kaggle will mount these read-only at `/kaggle/input/phomt-dataset/` and `/kaggle/input/nmt-checkpoints/`.

---

## 3. Repository Structure & What to Push to GitHub

Before running `git clone` on Kaggle, ensure your GitHub repository contains only source code and configuration files. Do **NOT** commit large datasets or checkpoint binaries to GitHub (upload them as Kaggle Datasets instead).

### Clean GitHub Repository Structure:
```text
tensor-engine-nmt/
├── src/
│   └── tensor_engine_nmt/
│       ├── __init__.py          # Package entry point & CLI dispatcher
│       ├── config.py            # Hyperparameter single source of truth
│       ├── backend.py           # Transparent CuPy (GPU) / NumPy (CPU) abstraction
│       ├── activations.py       # Activation functions & derivatives (softmax, tanh, etc.)
│       ├── init_weights.py      # Glorot Uniform & Orthogonal initializations
│       ├── bpe.py               # Custom BPE tokenizer & subword merges
│       ├── dataset.py           # Dynamic token batching & pickle caching
│       ├── encoder.py           # 3-layer LSTM encoder (forward + BPTT)
│       ├── attention.py         # Luong general attention (forward + backward)
│       ├── decoder.py           # 3-layer LSTM decoder with attention (forward + BPTT)
│       ├── loss.py              # Masked Cross-Entropy loss & gradient
│       ├── model.py             # Seq2Seq wrapper, model.save() & model.load()
│       ├── optimizer.py         # Adam optimizer with L2 gradient clipping
│       ├── train.py             # Training loop with automatic checkpoint resume
│       ├── inference.py         # Greedy & Beam search translation decoders
│       └── evaluate.py          # Multi-bleu / Corpus BLEU-4 evaluation
├── test/
│   ├── test_shapes.py           # Unit tests for tensor dimensions
│   ├── test_gradients.py        # Numerical gradient checks
│   └── test_overfit.py          # Smoke tests on synthetic batch
├── pyproject.toml               # Package dependencies & CLI command specification
├── .gitignore                   # Ignores .venv, cache, checkpoints, and datasets
├── README.md                    # Core documentation & mathematical formulation
├── COLAB_GUIDE.md               # Google Colab instructions
└── KAGGLE_GUIDE.md              # Kaggle migration guide
```

> [!IMPORTANT]
> **Files to keep OUT of GitHub (.gitignore):**
> * `PhoMT_dataset/` (~600 MB): Upload as Kaggle Dataset `phomt-dataset`.
> * `checkpoints/` (~140 MB per `.npz`): Upload as Kaggle Dataset `nmt-checkpoints`.
> * `bpe_vocab/` and `*.pkl` caches: Auto-generated at runtime.
> * `.venv/`: Virtual environment folder.

---

## 4. The Kaggle Cells (Detailed Breakdown)

In your Kaggle Notebook, create and run the following cells in order.

### Cell 1: Environment Setup & GPU Verification
**What this does:** Clones your codebase, installs dependencies including CuPy for GPU acceleration, and verifies that the `cupy` GPU backend is strictly active (preventing accidental CPU execution).

```python
import os

# 1. Clone the repository
!git clone <YOUR_GITHUB_REPO_URL_HERE> tensor-engine-nmt

# 2. Move into the project directory
os.chdir('tensor-engine-nmt')

# 3. Install CuPy for CUDA 12 and the package in editable mode
!pip install -q cupy-cuda12x
!pip install -q -e .

# 4. Strictly verify GPU is detected
from tensor_engine_nmt.backend import xp, BACKEND
print(f"Active Backend: {BACKEND}")
assert BACKEND == "cupy", "CRITICAL ERROR: CuPy is not active! Make sure GPU Accelerator is enabled in Kaggle Settings."
```

### Cell 2: Data & Checkpoint Preparation
**What this does:** Kaggle's `/kaggle/input/` directory is strictly **read-only**. Because our codebase creates the `bpe_vocab` folder and caches tokenized data, we must copy the dataset into the writable `/kaggle/working/` directory. We also create the `checkpoints/` folder and copy our migrated Colab weights there.

```python
# 1. Copy dataset to writable working directory
!cp -r /kaggle/input/phomt-dataset/PhoMT_dataset ./PhoMT_dataset

# 2. Setup checkpoints folder and migrate Colab weights
!mkdir -p checkpoints
!cp /kaggle/input/nmt-checkpoints/* ./checkpoints/

# Verify files are in place
!ls -lh checkpoints/
```

### Cell 3: Start Training Execution
**What this does:** Triggers the training loop. Since we placed the checkpoints in the `checkpoints/` folder, `train.py` will automatically find the latest `step_XXXX.npz` or `epoch_E.npz` and resume seamlessly without losing progress.

```python
# Run the training loop (using the verified GPU environment)
!tensor-engine-nmt train
```

---

## 5. Background Training (The Kaggle Superpower)

If you run Cell 3 interactively, the kernel will die if you close your laptop or browser tab. To train overnight in the background:

1. Look at the top right of the Kaggle interface and click the **Save Version** button.
2. Select **Save & Run All (Commit)**.
3. Click **Save**.

You can now completely close your browser and turn off your laptop. Kaggle will spin up a headless server, run your notebook from top to bottom, and train your model for **up to 12 hours**.

### Retrieving Your Checkpoints After a Background Run
1. Go to your Notebook's page.
2. Click on the **Output** tab under the finished version.
3. You will see your `checkpoints/` folder (along with `train_logs.txt`) perfectly preserved.
4. Click **Download** to save the new `.npz` files to your computer.

---

## 6. Critical Concerns & Limitations to Watch Out For

When using Kaggle, there are a few strict limitations you must architect around:

### 1. The 12-Hour Execution Limit
Kaggle background kernels are hard-killed exactly at the 12-hour mark. 
**Concern:** If the kernel is killed mid-epoch, you will rely on the last rolling checkpoint (`step_XXXX.npz`) saved via your `cfg.save_every` parameter. With our configured `save_every = 2000`, a checkpoint is saved approximately every ~2 hours on a GPU, so you never lose more than 2 hours of progress when the 12-hour timeout hits.

### 2. Disk Space Limits (`/kaggle/working/`)
Kaggle limits the `/kaggle/working/` directory to **20 GB of output**.
**Concern:** Each checkpoint is ~140 MB. With `save_every = 2000`, a 12-hour run (~12,000 steps) generates only ~6 checkpoints (~840 MB), plus `optim_state.npz` (~270 MB) and dataset cache (~200 MB) — well below 1.5 GB total! This guarantees you will never hit Kaggle's 20 GB ceiling. Avoid setting `save_every` too low (e.g., every 100 steps) which would spam hundreds of files and exhaust the disk.

### 3. RAM Limits (30 GB CPU RAM)
**Concern:** Our `PhoMTDataset` streams data line-by-line, which is highly memory efficient. However, `dataset.py` builds an in-memory cache to bucket-sort sequences. If the cache grows too large, Kaggle's CPU RAM might spike. Watch the resource monitor in the top right during the first epoch to ensure RAM stays below 30 GB.

### 4. The Multi-Account Quota Shuffle
Kaggle allows 30 hours of GPU time per week. If you plan to train for 72 hours, you will need at least 3 Kaggle accounts.
**Concern:** The migration process is manual. When Account A runs out of time, you must download the latest checkpoints to your laptop, upload them to a new Dataset on Account B, update the notebook on Account B to use the new Dataset, and launch the background run again. This "hot-swapping" creates downtime, so factor it into your schedule.
