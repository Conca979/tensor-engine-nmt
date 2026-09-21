# 🚀 Google Colab Training Guide

This guide walks you through migrating and training the Tensor Engine NMT model on **Google Colab**, which provides a **free NVIDIA GPU** (T4, 16GB VRAM) — a massive upgrade from running on your laptop CPU.

> **🎯 The short-session mindset.** You do not need (and on free Colab you will not get) a 12-hour uninterrupted run. Training is **time-boxed and resumable**: you tell it `--max-minutes 180`, it writes a checkpoint, exits cleanly, and the next session picks up exactly where it left off. Ten 3-hour sessions beat one crashed 12-hour session. Everything below is built around that loop.

---

## 🧭 Short-Session Recipe (The Primary Workflow)

This is the whole loop. The detailed steps further down just set it up.

> Full background on these flags, the `--preset laptop` escape hatch, checkpoint
> discipline and how to read the throughput numbers:
> **[docs/short_session_training.md](docs/short_session_training.md)**.

**One session = pre-flight benchmark → time-boxed training run → download 3 files off the platform.**

```bash
# 0. In the session, measure speed FIRST (new subcommand)
!tensor-engine-nmt bench --steps 10

# 1. Train for a bounded slice of wall-clock time, checkpointing often
!tensor-engine-nmt train --lr 1e-4 --max-minutes 180 --save-every 500 --keep-last 3

# 2. After it stops, grab the newest step_*.npz, optim_state.npz, train_logs.txt
```

Then next session: re-run the same cell. It finds the latest checkpoint and continues. No extra code, no `--resume` needed unless you want a specific file.

| Knob | Value for short sessions | Why |
|---|---|---|
| `--max-minutes` | `180` (free Colab) / `540`–`660` (Pro) | Stops **cleanly** before the platform pulls the plug, so the final checkpoint is written |
| `--save-every` | `500` | ~28 min worst-case loss instead of ~3.7 hours (see below) |
| `--keep-last` | `3` | Deletes older `step_*.npz` so the disk doesn't fill up over many sessions |
| `--lr` | `1e-4` | **Always pass this explicitly** — the config default is `1e-5`, and the existing step-196k checkpoint was trained at `1e-4` |

> **Free Colab in practice:** disconnects are routine and you get no warning. Size the run to **~3 hours of actual compute** (`--max-minutes 180`) and treat every clean exit as a win. On Colab Pro you can safely stretch to `--max-minutes 540` (9 hours) or `--max-minutes 660` (11 hours).

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
!pip install -q -e .[gpu]

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

## Step 6: Pre-Flight Benchmark (Do This Every Session)

Before you commit to a long run, measure what this GPU is actually giving you today. `bench` runs a few real training steps and prints measured **tok/s**, **seconds/step** and an **estimated hours-per-epoch**:

```python
# ── Cell 4: Pre-flight benchmark ──────────────────────────────────────────────
!tensor-engine-nmt bench --steps 10
```

Read the printed **hours-per-epoch** and sanity-check it against your session budget:

- Budget **3 h** and the estimate says **2.9 h/epoch**? You will finish roughly one epoch. Fine — run it.
- Budget **3 h** and the estimate says **11 h/epoch**? Don't launch blind. Lower `--max-minutes` expectations, add `--save-every 300`, lower `--max-pairs`, or accept that you're doing a partial epoch (which is fine — checkpoints are mid-epoch safe).

You can also benchmark the small laptop configuration for comparison:

```python
!tensor-engine-nmt bench --preset laptop --steps 10
```

---

## Step 7: Run a Time-Boxed Training Session

The recommended way is the CLI, because the time-boxing flags live there:

```python
# ── Cell 5: Train for a bounded slice of wall-clock time ──────────────────────
# Stops at 180 minutes (3h), writes a checkpoint, exits cleanly.
# --save-every 500 keeps worst-case work-loss under ~30 minutes.
# --keep-last 3 protects your Drive quota across many sessions.
!tensor-engine-nmt train \
    --lr 1e-4 \
    --max-minutes 180 \
    --save-every 500 \
    --keep-last 3
```

**Why `--max-minutes` matters:** without it, a free Colab disconnect hard-kills the process and you lose everything since the last rolling checkpoint. With it, the run stops itself, flushes a checkpoint, and ends — so the very last thing that happened was a successful save.

**Why 500 and not 4000:** at ~1500 tok/s on a T4 with `max_tokens=5000`, one step is about 3.3 seconds, so the config default `save_every=4000` means **one checkpoint interval is roughly 3–4 hours**. A single pre-emption would wipe out an entire session's work. `--save-every 500` brings that down to ~28 minutes — a much better tradeoff, and with `--keep-last 3` it still costs only ~1.2 GB of disk.

**Other useful flags for short sessions:**

| Flag | Use it when |
|---|---|
| `--max-steps 1500` | You'd rather bound by steps than by clock time |
| `--max-pairs 400000` | You want a faster epoch (builds a smaller cache) from a strided sample of the corpus |
| `--k 8000` | You want teacher forcing to anneal faster, because you're doing fewer total steps |
| `--max-tokens 3000` | You're hitting VRAM limits (T4 has 16 GB) |
| `--resume checkpoints/step_196000.npz` | You want to resume from a specific file instead of the newest |
| `--preset laptop` | You're testing the pipeline on a machine without a big GPU |

### Option B — Pure Python (if you prefer tweaking `HParams` in the cell)

Everything above is also settable in Python. Note that `HParams` is constructed with the **corrected** values here, and `--lr 1e-4` is passed explicitly because `config.py` now defaults to `lr=1e-5`:

```python
# ── Cell 5b: Equivalent Python call ───────────────────────────────────────────
from tensor_engine_nmt.config import HParams
from tensor_engine_nmt.train import train

hp = HParams(
    V=30839,        # Real trained BPE vocab size (config.py's 32000 is just an upper bound)
    e=512,          # Embedding dimension
    d=1024,         # LSTM hidden dimension
    L=3,            # Stacked LSTM layers
    max_tokens=5000, # Dynamic token batching (maximizes T4 16GB VRAM)
    k=17000.0,      # Inverse-sigmoid teacher forcing decay constant
    min_tf=0.70,    # Minimum teacher forcing floor (scheduled sampling lower bound)
    lr=1e-4,        # Learning rate — matched to the existing step-196k checkpoint
    beta1=0.9, beta2=0.999, eps_adam=1e-8,
    clip_norm=2.0,
    max_epochs=30,
    beam_width=4,
    max_decode_len=30,
    log_every=100,
    save_every=500,  # Short-session value, NOT the 4000 config default
    data_dir='PhoMT_dataset',
    bpe_dir='bpe_vocab',
    ckpt_dir='checkpoints',
    bpe_sample_lines=300000,
)

train(hp=hp, max_minutes=180)   # 3-hour clean stop; add max_steps=1500 to bound by steps instead
```

> **Note:** `max_minutes` and `max_steps` are *arguments to `train()`*, not `HParams` fields — so Option B can time-box too, as shown above.

> **Vocabulary note:** `config.py` ships `V=32000`, but the BPE tokenizer actually trained to **30,839** tokens. `V` only needs to be **>=** the real vocab size, so anything from 30,839 upward is safe. Set it lower and the run stops immediately with a `ValueError`: `hp.V=..., is smaller than the loaded BPE vocabulary (30,839 tokens in bpe_vocab)` — no silent corruption, but no training either.
>
> **Don't mix the laptop preset with this block.** `--preset laptop` uses `V=8000` and its own `bpe_vocab_laptop/` folder precisely so those two vocabularies can never be confused. Use one or the other, not both.

> **First run:** The BPE tokenizer will train on 300,000 lines (~3–5 min) and then pre-tokenize the full dataset and cache it to `PhoMT_dataset/train_cache_V30839_L30.pkl` (~5–10 min). **All subsequent runs and epoch restarts will skip this entirely.**

---

## Step 8: Checkpoint Discipline (Download After EVERY Session)

The training loop saves `.npz` checkpoints to the `checkpoints/` directory. Since this lives inside your Google Drive folder, **they are automatically saved to Drive** and will not be lost when the Colab session ends.

It also appends logs to `checkpoints/train_logs.txt`, and writes `checkpoints/state.txt` as a **legacy fallback** — the authoritative epoch position now lives *inside* the checkpoint (see Step 9).

### The 3 essential files, every single session

| File | Size (full-size model) | What it is |
|---|---|---|
| `step_NNNNNN.npz` | ≈ **401 MB** | Model weights + the epoch/step position (`meta_epoch`, `meta_step_in_epoch`) |
| `optim_state.npz` | ≈ **733 MB** | Adam moments — the single shared companion file |
| `train_logs.txt` | tiny | Loss curve across all sessions |

Store them **off the platform** — your laptop, an external drive, a private cloud folder. Google Drive counts against your quota (401 MB + 733 MB per checkpoint adds up fast), and `/content/` is wiped on every disconnect.

> **⚠️ One shared `optim_state.npz`, tagged with its checkpoint.** It is *not* per-step. If you resume from an older `step_*.npz` while the companion `optim_state.npz` belongs to a *newer* checkpoint, training now prints a **loud warning** and starts the optimizer **fresh** rather than silently pairing mismatched moments. Always download weights and optimizer state **together**, from the same session.

To verify checkpoint saving is working:
```python
# ── Cell 6: List saved checkpoints ────────────────────────────────────────────
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

## Step 9: Resuming After a Colab Disconnect

Colab sessions disconnect without much warning. Because we implemented robust state tracking, resuming is completely automatic!

To resume:
1. Re-run **Cells 1–3** (Mount Drive, Install dependencies, Verify GPU).
2. Re-run **Cell 4** (pre-flight benchmark) — cheap, and confirms the checkpoint loads.
3. Re-run **Cell 5** (the training cell) with the same flags. It continues from the newest checkpoint.

> **📌 The epoch position is now stored inside the checkpoint.** `train.py` reads `meta_epoch` and `meta_step_in_epoch` directly out of the `.npz`, so **`state.txt` is no longer required for a correct resume**. It is still written alongside as a legacy fallback for older tooling, but the checkpoint is authoritative — you can delete `state.txt` and nothing breaks.

Normally you need no extra code at all. If you ever *do* want a specific file, pass it explicitly:

```python
!tensor-engine-nmt train --lr 1e-4 --max-minutes 180 --save-every 500 --keep-last 3 --resume checkpoints/step_196000.npz
```

---

## Step 10: Run Evaluation

Evaluation defaults to **Beam Search (`method="beam"`)** and automatically appends detailed sentence pairs and BLEU scores to `checkpoints/evaluation_result.txt`:

```python
# ── Cell 7: Evaluate BLEU score (Test & Train Sample) ─────────────────────────
from tensor_engine_nmt.evaluate import evaluate

# 1. Evaluate on test set (Beam Search by default; filters max_len <= 30)
bleu_test = evaluate(hp=hp, split="test", method="beam", max_sentences=200, verbose=True)
print(f"\nTest Corpus BLEU-4: {bleu_test:.2f}")

# 2. Overfitting check: Evaluate on a random sample of 500 training sentences
bleu_train = evaluate(hp=hp, split="train", max_sentences=500, random_sample=True, method="greedy", verbose=False)
print(f"\nTrain (Random 500 sample) BLEU-4: {bleu_train:.2f}")
```

Or run via terminal / CLI:
```bash
# Evaluate on test set with beam search (auto-logs to checkpoints/evaluation_result.txt)
!uv run tensor-engine-nmt evaluate --method beam --n 100 --verbose

# Run the analyzer on the logged evaluation results
!python analyze.py --eval-only
```

> **💡 You don't need Colab (or a GPU) to evaluate.** Evaluation and translation are cheap on CPU. Once a session has ended, just copy `step_NNNNNN.npz`, `optim_state.npz` and the `bpe_vocab/` folder onto your laptop and run this locally:
> ```bash
> # On your laptop — no GPU needed
> tensor-engine-nmt evaluate --method greedy --n 200
> python demo.py
> ```
> Greedy decoding over 200 sentences is a few minutes of pure CPU work. Use Colab GPU time for **training**; do your inspecting, BLEU checks and translation sanity passes at home.

---

## Step 11: Interactive Translation Demo

You can run the dedicated terminal live translation REPL right in Colab:
```bash
# Launch interactive REPL (auto-loads latest checkpoint)
!python demo.py --method both --beam-width 4
```

Or test translations programmatically in a notebook cell:
```python
# ── Cell 8: Interactive translation demo ──────────────────────────────────────
from tensor_engine_nmt.inference import Translator
from tensor_engine_nmt.bpe import BPETokenizer
from tensor_engine_nmt.model import Seq2Seq
from demo import find_latest_checkpoint

# Load the best checkpoint numerically
ckpt = find_latest_checkpoint("checkpoints")
print(f"Loading checkpoint: {ckpt}")
model = Seq2Seq(hp)
model.load(ckpt)

bpe = BPETokenizer.load(hp.bpe_dir)
translator = Translator(model, bpe, hp)

sentences = [
    "Hello, how are you?",
    "The economy of Vietnam is growing rapidly.",
    "He can water the horses .",
]

for sentence in sentences:
    translation = translator.translate(sentence, method='beam')
    print(f"EN: {sentence}")
    print(f"VI: {translation}")
    print()
```

---

## 🧰 Short-Session CLI Reference

Everything the time-boxed workflow uses, in one place.

**Training — `tensor-engine-nmt train`**

| Flag | Default | What it does |
|---|---|---|
| `--preset {gpu,laptop}` | `gpu` | `laptop` shrinks the model a lot (`d=256`, `L=2`, `e=128`, `V=8000`, `max_len=20`) and switches to a **separate** BPE / cache / checkpoint directory, so it never collides with GPU checkpoints |
| `--lr FLOAT` | `1e-5` (from `config.py`) | **Pass `1e-4` explicitly** to stay consistent with the existing step-196k checkpoint |
| `--max-tokens INT` | `5000` | Dynamic token batching budget |
| `--min-tf FLOAT` | `0.70` | Teacher-forcing floor |
| `--k FLOAT` | `17000` | Inverse-sigmoid teacher-forcing decay constant (smaller anneals faster) |
| `--max-steps INT` | — | Stop after N gradient steps |
| `--max-minutes FLOAT` | — | **Stop cleanly after N wall-clock minutes**: writes a checkpoint first, then exits. Resuming is automatic |
| `--save-every INT` | `4000` | Checkpoint every N steps — set to `500` for short sessions |
| `--log-every INT` | `100` | Log every N steps |
| `--keep-last INT` | — | After each save, delete older `step_*.npz` beyond the newest N (protects disk in long runs) |
| `--max-pairs INT` | — | Cap the training set to an evenly-strided sample of N pairs (builds a smaller, faster cache) |
| `--resume PATH` | auto | Resume from a specific `.npz` |

**Benchmark — `tensor-engine-nmt bench`**

```bash
tensor-engine-nmt bench [--preset {gpu,laptop}] [--steps INT] [--max-tokens INT] [--max-pairs INT]
```

Runs a few **real** training steps and prints measured **tok/s**, **seconds/step** and an estimated **hours-per-epoch** — so you can size a session *before* committing hours to it.

**Evaluate — `tensor-engine-nmt evaluate`**

```bash
tensor-engine-nmt evaluate --method greedy --n 200    # cheap on CPU, fine on your laptop
```

---

## ⚠️ Important Colab Tips

| Tip | Detail |
|---|---|
| **Always store data in Drive** | Colab's `/content/` is wiped on disconnect. Only Drive persists. |
| **Run cells top-to-bottom** after a disconnect | After reconnecting, re-run Cell 1 (mount) and Cell 2 (install) before anything else. |
| **Set T4 GPU runtime** | Without GPU, CuPy will fail and training will be 50-100x slower. |
| **Don't close the tab** | Colab disconnects after ~90 min if the tab is closed or idle. |
| **Stop before they stop you** | Size each run with `--max-minutes` (3 h on free tier, ~9–11 h on Pro) so the final checkpoint lands **before** the hard kill. `--max-minutes 660` = 11 hours, comfortably inside the 12-hour ceiling. |
| **Colab Pro is worth it** | For long runs, upgrading to Colab Pro gives A100 GPU access and longer sessions. |
| **Benchmark first, launch second** | `tensor-engine-nmt bench --steps 10` takes a minute and tells you whether the run fits your session at all. |

---

## 📁 Recommended Final Drive Layout

```
My Drive/
└── tensor-engine-nmt/
    ├── src/                            ← Source code
    ├── test/                           ← Test suite
    ├── PhoMT_dataset/                  ← Raw EN/VI text files
    │   ├── train_cache_V30839_L30.pkl  ← Auto-generated after first run
    │   ├── train/
    │   └── test/
    ├── bpe_vocab/                      ← Auto-generated BPE vocabulary
    │   ├── merges.txt
    │   └── vocab.json
    ├── checkpoints/                    ← Auto-saved model weights (.npz)
    │   ├── step_196000.npz             ← Weights + epoch/step position (≈401 MB)
    │   ├── optim_state.npz             ← Adam moments, shared companion (≈733 MB)
    │   ├── train_logs.txt              ← Persistent training logs
    │   └── state.txt                   ← Legacy fallback (checkpoint is authoritative)
    ├── pyproject.toml
    └── README.md
```

> **Disk tip:** `step_*.npz` (401 MB) plus `optim_state.npz` (733 MB) is ~1.1 GB *per kept checkpoint*. Always run with `--keep-last 3` (≈3.5 GB ceiling) and move older checkpoints off Drive to your laptop. Full-size model → full-size files; the `--preset laptop` model is dramatically smaller if you're tight on space.
