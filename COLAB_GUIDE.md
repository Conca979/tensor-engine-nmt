# 🚀 Google Colab Training Guide

This guide walks you through migrating and training the Tensor Engine NMT model on **Google Colab**, which provides a **free NVIDIA GPU** (T4, 16GB VRAM) — a massive upgrade from running on your laptop CPU.

---

## 📋 Prerequisites

- A Google Account (for Google Drive and Colab)
- Your project code uploaded to **Google Drive** or **GitHub**

---

## Step 1: Upload Your Project Code

Choose **one** of the two options below:

### Option A — Via GitHub (Recommended)
If you have a GitHub repo, this is the cleanest approach. Push your local code to GitHub first:

```bash
# On your local machine
cd c:\Users\admin\projects\tensor-engine-nmt
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/tensor-engine-nmt.git
git push -u origin main
```

### Option B — Via Google Drive (No GitHub needed)
1. Open [Google Drive](https://drive.google.com/).
2. Create a new folder called `tensor-engine-nmt`.
3. Upload the following items from your local project into that folder:
   - The entire `src/` folder
   - The entire `test/` folder
   - `pyproject.toml`
   - `README.md`
4. **Do NOT upload** `.venv/`, `checkpoints/`, `bpe_vocab/`, `bpe_vocab_smoke/`, or the tokenized `.pkl` cache files — these will be regenerated on Colab.

---

## Step 2: Upload the Dataset to Google Drive

The PhoMT dataset is too large to re-download every time Colab restarts. Storing it in Drive ensures it persists.

1. Download the PhoMT dataset from the link in `PhoMT_dataset/download.txt`:
   **https://drive.google.com/drive/folders/1nSdw5HW35dBiYRJD3eFeTangkXtGE0ud**
2. Move the downloaded `PhoMT_dataset/` folder into your `tensor-engine-nmt/` folder on Google Drive.
3. Your final Drive structure should look like this:
   ```
   My Drive/
   └── tensor-engine-nmt/
       ├── src/
       ├── test/
       ├── PhoMT_dataset/
       │   ├── train/
       │   │   ├── train.en
       │   │   └── train.vi
       │   └── test/
       │       ├── test.en
       │       └── test.vi
       ├── pyproject.toml
       └── README.md
   ```

---

## Step 3: Create and Set Up Your Colab Notebook

1. Go to [https://colab.research.google.com/](https://colab.research.google.com/).
2. Click **New Notebook**.
3. At the top, click **Runtime → Change runtime type**.
4. Set **Hardware Accelerator** to **T4 GPU**.
5. Click **Save**.

---

## Step 4: Mount Google Drive and Install Dependencies

In the **first code cell**, paste and run the following:

```python
# ── Cell 1: Mount Drive and set up paths ──────────────────────────────────────
from google.colab import drive
drive.mount('/content/drive')

import os, sys

# Point to your project folder inside Drive
PROJECT_DIR = '/content/drive/MyDrive/tensor-engine-nmt'
os.chdir(PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))

print("Working directory:", os.getcwd())
print("Files:", os.listdir('.'))
```

In the **second cell**, install dependencies:

```python
# ── Cell 2: Install dependencies ──────────────────────────────────────────────
!pip install -q numpy tqdm cupy-cuda12x

# Verify GPU is available
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)
```

---

## Step 5: Verify the Backend Detects the GPU

```python
# ── Cell 3: Verify CuPy backend ───────────────────────────────────────────────
from tensor_engine_nmt.backend import xp
print(f"Active backend: {xp.__name__}")
# Should print: Active backend: cupy
```

If it prints `cupy`, you are running on GPU. If it prints `numpy`, go back to **Step 3** and confirm the T4 GPU runtime is selected.

---

## Step 6: Run the Full Training

```python
# ── Cell 4: Start training ────────────────────────────────────────────────────
from tensor_engine_nmt.config import HParams
from tensor_engine_nmt.bpe import load_or_train_bpe
from tensor_engine_nmt.dataset import PhoMTDataset
from tensor_engine_nmt.model import Seq2Seq
from tensor_engine_nmt.optimizer import Adam
from tensor_engine_nmt.train import train

hp = HParams(
    V=32000,    # Full vocabulary size
    e=512,      # Embedding dimension
    d=1024,     # LSTM hidden dimension (Increased capacity)
    L=3,        # Stacked LSTM layers (Increased capacity)
    max_tokens=4000, # Dynamic token batching (maximizes T4 16GB VRAM)
    k=17000.0,  # Inverse-sigmoid teacher forcing decay constant
    min_tf=0.70, # Minimum teacher forcing floor (scheduled sampling lower bound)
    lr=3e-4,    # Learning rate
    beta1=0.9, beta2=0.999, eps_adam=1e-8,
    clip_norm=5.0,
    max_epochs=30,
    beam_width=4,
    max_decode_len=100,
    log_every=100,
    save_every=2000,
    data_dir='PhoMT_dataset',
    bpe_dir='bpe_vocab',
    ckpt_dir='checkpoints',
    bpe_sample_lines=300000,
)

train(hp=hp)
```

> **First run:** The BPE tokenizer will train on 300,000 lines (~3–5 min) and then pre-tokenize the full dataset and cache it to `PhoMT_dataset/train_cache_V32000.pkl` (~5–10 min). **All subsequent runs and epoch restarts will skip this entirely.**

---

## Step 7: Save Checkpoints Back to Drive (Auto)

The training loop automatically saves `.npz` checkpoints to the `checkpoints/` directory. Since this lives inside your Google Drive folder, **they are automatically saved to Drive** and will not be lost when the Colab session ends. 

Additionally, it will automatically append logs to `checkpoints/train_logs.txt` and track your precise batch progress in `checkpoints/state.txt`.

To verify checkpoint saving is working:
```python
# ── Cell 5: List saved checkpoints ────────────────────────────────────────────
import glob, re

def sort_by_step(path):
    """Sort checkpoint paths numerically, not alphabetically."""
    match = re.search(r'step_(\d+)', path)
    return int(match.group(1)) if match else -1

ckpts = sorted(glob.glob('checkpoints/*.npz'), key=sort_by_step)
print(f"Found {len(ckpts)} checkpoints:")
for c in ckpts:
    print(f"  {c}")
```

---

## Step 8: Resuming After a Colab Disconnect

Colab sessions disconnect after ~12 hours of inactivity. Because we implemented robust state tracking, resuming is completely automatic!

To resume:
1. Re-run **Cells 1–3** (Mount Drive, Install dependencies, Verify GPU).
2. Re-run **Cell 4** (Start training).

`train.py` will automatically find the latest `.npz` checkpoint, read `state.txt` to find the exact epoch and batch you were on, and seamlessly resume training without skipping a single sentence. You do not need any extra code!

---

## Step 9: Run Evaluation

```python
# ── Cell 7: Evaluate BLEU score on test set ───────────────────────────────────
from tensor_engine_nmt.evaluate import evaluate

bleu = evaluate(hp=hp, verbose=True)
print(f"\nCorpus BLEU-4: {bleu:.2f}")
```

---

## Step 10: Interactive Translation Demo

```python
# ── Cell 8: Interactive translation demo ──────────────────────────────────────
from tensor_engine_nmt.inference import Translator
from tensor_engine_nmt.bpe import BPETokenizer
from tensor_engine_nmt.model import Seq2Seq
import glob

# Load the best checkpoint
ckpt = sorted(glob.glob('checkpoints/*.npz'))[-1]
model = Seq2Seq(hp)
model.load(ckpt)

bpe = BPETokenizer.load(hp.bpe_dir)
translator = Translator(model, bpe, hp)

sentences = [
    "Hello, how are you?",
    "The economy of Vietnam is growing rapidly.",
    "Machine learning is a subset of artificial intelligence.",
]

for sentence in sentences:
    translation = translator.translate(sentence, method='beam')
    print(f"EN: {sentence}")
    print(f"VI: {translation}")
    print()
```

---

## ⚠️ Important Colab Tips

| Tip | Detail |
|---|---|
| **Always store data in Drive** | Colab's `/content/` is wiped on disconnect. Only Drive persists. |
| **Run cells top-to-bottom** after a disconnect | After reconnecting, re-run Cell 1 (mount) and Cell 2 (install) before anything else. |
| **Set T4 GPU runtime** | Without GPU, CuPy will fail and training will be 50-100x slower. |
| **Don't close the tab** | Colab disconnects after ~90 min if the tab is closed or idle. |
| **Colab Pro is worth it** | For long runs (24h+), upgrading to Colab Pro gives A100 GPU access and longer sessions. |

---

## 📁 Recommended Final Drive Layout

```
My Drive/
└── tensor-engine-nmt/
    ├── src/                         ← Source code
    ├── test/                        ← Test suite
    ├── PhoMT_dataset/               ← Raw EN/VI text files
    │   ├── train_cache_V32000.pkl   ← Auto-generated after first run
    │   ├── train/
    │   └── test/
    ├── bpe_vocab/                   ← Auto-generated BPE vocabulary
    │   ├── merges.txt
    │   └── vocab.json
    ├── checkpoints/                 ← Auto-saved model weights (.npz)
    │   ├── train_logs.txt           ← Persistent training logs
    │   └── state.txt                ← Exact epoch tracking state
    ├── pyproject.toml
    └── README.md
```
