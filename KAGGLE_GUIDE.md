# Training on Kaggle (Background Runs & High-Speed GPU)

Kaggle offers **free 30 hours of GPU time per week**, and importantly, allows you to click "Save & Run All" to run your training loop in the **background for up to 12 hours** without needing to keep a browser tab open. Plan sessions around the fact that 12 hours is a hard kill, not a finish line — see the recipe below.

> [!TIP]
> ### ⏱️ The Short-Session Recipe (Start Here)
> Don't try to *fill* the 12-hour window — you'll get hard-killed mid-write. Instead, **stop yourself at ~10.5 hours** so the final checkpoint lands cleanly, and repeat across sessions. Training resumes automatically from the newest checkpoint.
>
> **In the notebook, before the long run:**
> ```bash
> # Pre-flight: measure tok/s and hours-per-epoch for THIS session
> !tensor-engine-nmt bench --steps 10
> ```
>
> **Then the real run — this is the command to memorize:**
> ```bash
> !tensor-engine-nmt train \
>     --lr 1e-4 \
>     --max-minutes 630 \
>     --save-every 500 \
>     --keep-last 3
> ```
>
> * `--max-minutes 630` = **10.5 hours** — comfortably inside Kaggle's 12-hour kill zone, with time left for the final checkpoint write.
> * `--save-every 500` — checkpoint every ~28 minutes instead of every ~3.7 hours.
> * `--keep-last 3` — auto-prune old `step_*.npz` so you never hit the 20 GB output ceiling.
> * `--lr 1e-4` — **always pass it.** The config default is now `1e-5`, but the existing step-196k checkpoint was trained at `1e-4`.
>
> **After every session:** download the newest `step_*.npz`, `optim_state.npz` and `train_logs.txt` (§5), then version-bump your `nmt-checkpoints` dataset (§2.6). That's the whole loop.
>
> Full background on these flags, the `--preset laptop` escape hatch, checkpoint discipline and how to read the throughput numbers: **[docs/short_session_training.md](docs/short_session_training.md)**.

> [!CAUTION]
> ### Mandatory Prerequisite: Phone Verification is Required for GPU Access
> Kaggle strictly requires a **one-time SMS Phone Verification** before allowing access to free GPUs (GPU P100 / GPU T4x2) and outbound internet.
> * **If your account is NOT verified:** The "Accelerator" setting is locked to **CPU only** (`None`), and the "Internet" toggle is locked to **OFF**. Because training our 108,585,216-parameter model on CPU is impossible (~15 tok/s = months per epoch), **you cannot train on Kaggle without phone verification**.
> * **If you cannot verify your phone number:** Do NOT use Kaggle. Use **[COLAB_GUIDE.md](COLAB_GUIDE.md)** instead! Google Colab provides free NVIDIA T4 GPUs connected to Google Drive without mandatory phone verification.
> * **To verify your Kaggle account:** Go to [kaggle.com/settings](https://www.kaggle.com/settings) -> scroll down to **Phone Verification** -> verify your phone number. Once verified, GPU and Internet access are unlocked immediately.

---

## 1. Setup Your Kaggle Notebook (Verified Accounts)

1. Go to [Kaggle](https://www.kaggle.com/) and create or sign into your verified account.
2. Click **Create** (Top left) -> **New Notebook**.
3. In the Notebook editor, look at the **Session options / Notebook options** panel on the right sidebar.
4. **Accelerator:** Select **GPU P100** (recommended for memory bandwidth) or **GPU T4 x2**.
5. **Internet:** Toggle to **ON** (`Internet on`).

---

## 2. Uploading & Updating Datasets on Kaggle (Step-by-Step)

Unlike Google Drive where you can simply drag-and-drop a raw folder tree, **Kaggle's browser uploader struggles with raw folders that contain subdirectories**. If you try to drag the raw `PhoMT_dataset/` folder into Kaggle, it will often fail, freeze, or flatten the folder structure.

Follow these exact steps to upload and update your data reliably.

---

### Step 2.1: The Golden Rule — Always Zip Using Forward Slashes (/)

When uploading any folder that has subdirectories or multiple files (like `PhoMT_dataset/` or `checkpoints/`), **compress it into a `.zip` archive first**.

> [!WARNING]
> **Avoid Windows Built-in "Compress to ZIP":**  
> Windows Explorer and PowerShell's `Compress-Archive` embed Windows backslashes (`\`) into the internal file paths of the `.zip`. Because Kaggle runs Linux, Kaggle's upload validator will reject the zip with:  
> `"... contains a forbidden character in name ('\')"`

**The Safe Way to Zip on Windows:**  
Use our included Python script which forces standard Linux-compatible forward slashes (`/`):
```bash
# To zip your codebase:
python zip_for_kaggle.py code

# To zip your dataset folder:
python zip_for_kaggle.py PhoMT_dataset

# To zip checkpoints:
python zip_for_kaggle.py checkpoints
```
*(On Linux/Mac, standard `zip -r PhoMT_dataset.zip PhoMT_dataset/` works natively).*

---

### Step 2.2: Uploading Your Source Code as a Dataset (No GitHub Needed)

You do not need to clone from GitHub. You can upload the codebase directly as a Kaggle Dataset:

1. **Package your code:** In your local project directory on your computer, run:
   ```bash
   python zip_for_kaggle.py code
   ```
   This generates **`tensor-engine-nmt-code.zip`** (~105 KB, containing `src/`, `test/`, `pyproject.toml`, etc., with clean Linux paths).
2. **Upload to Kaggle:**
   * Go to [kaggle.com/datasets](https://www.kaggle.com/datasets) -> click **+ New Dataset** (or in your notebook, **Add Input** -> **Upload** -> **New Dataset**).
   * Drag & drop `tensor-engine-nmt-code.zip`.
   * **Dataset Title:** Enter `tensor-engine-nmt-code` (mounts at `/kaggle/input/tensor-engine-nmt-code/`).
   * Set visibility to **Private** -> click **Create**.

---

### Step 2.3: Uploading the `PhoMT_dataset` (One-Time Setup)

1. **Package your dataset:** In your local project directory, run:
   ```bash
   python zip_for_kaggle.py PhoMT_dataset
   ```
2. In Kaggle, click **+ New Dataset** (or notebook **Add Input** -> **Upload** -> **New Dataset**).
3. Drag & drop `PhoMT_dataset.zip`.
4. **Dataset Title:** Enter `phomt-dataset` (mounts at `/kaggle/input/phomt-dataset/`).
5. Set visibility to **Private** -> click **Create**.

---

### Step 2.4: Uploading Your Migrated Colab Checkpoints

1. From Google Drive, download your latest checkpoints to your computer:
   * `step_66000.npz` (or whatever your latest step is)
   * `optim_state.npz`
   * `train_logs.txt`
   * `evaluation_result.txt` (if previously evaluated)
   * `state.txt` (optional — legacy only; the epoch position lives inside the `.npz` now)
2. Put these files together into a local folder named `checkpoints`.
3. In your terminal, run:
   ```bash
   python zip_for_kaggle.py checkpoints
   ```
4. On Kaggle, create your third dataset:
   * Go to **+ New Dataset** -> drag & drop `checkpoints.zip` (or the files directly).
   * **Dataset Title:** Enter `nmt-checkpoints` (mounts at `/kaggle/input/nmt-checkpoints/`).
   * Set to **Private** -> click **Create**.

---

### Step 2.5: Attach the 3 Datasets to Your Notebook

Open your training notebook on Kaggle. In the right-hand panel, click **Add Input** -> select **Your Work** / **My Datasets**, and click the **+** (Add) button next to all three:
1. `tensor-engine-nmt-code`
2. `phomt-dataset`
3. `nmt-checkpoints`

Verify that all three appear under the **Input** section on the right sidebar!

---

### Step 2.6: How to UPDATE Checkpoints with a "New Version" (Multi-Session Workflow)

When a time-boxed Kaggle run completes (it exits itself at `--max-minutes 630`, ~10.5 hours), you will have newer checkpoints (`step_88000.npz`, updated `optim_state.npz`, `train_logs.txt`, `evaluation_result.txt`). **You do NOT need to create a new dataset from scratch.** Instead, update the existing dataset with a new version:

1. Go to your dataset page:  
   `https://www.kaggle.com/datasets/<YOUR_KAGGLE_USERNAME>/nmt-checkpoints`
2. In the top-right corner of the dataset page, click the **New Version** button (or click the `...` menu -> **Upload New Version**).
3. Drag and drop your newly downloaded checkpoint files into the window.  
   *(They will overwrite old files with the same name and add the new `step_XXXX.npz`).*
4. In the **Version Notes** box, type a note (e.g. `Update to step 88000`).
5. Click **Create**.
6. **Refresh inside your Notebook:**
   * Return to your Kaggle Notebook.
   * In the right-hand **Session Options / Input** panel, find `nmt-checkpoints`.
   * Click the circular **Refresh (↻)** button next to it.
   * Kaggle will instantly mount the latest version into `/kaggle/input/nmt-checkpoints/` without needing to restart!

---

### Step 2.7: Permanent Speedup — Bundle `train_cache_*.pkl` to Save 2.5 Minutes Every Run

When you ran training for the first time, `dataset.py` spent ~2.5 minutes pre-tokenizing 2,966,638 sentence pairs and saved `train_cache_V30839_L30.pkl` into `/kaggle/working/PhoMT_dataset/`.

> **Cache naming:** the cache file is now `<split>_cache_V{vocab}_L{max_len}.pkl` — the `L30` is your `max_len` (BPE tokens per side). It is keyed on both values because the length filter is baked into the cached content. A legacy `train_cache_V30839.pkl` (no `_L`) still loads, but prints a warning; it was filtered at whatever `max_len` was active when it was built, so rebuild it if you've raised `max_len`.

Because `/kaggle/input/` datasets are read-only, this cache is stored in temporary session output. To make every future session and every secondary Kaggle account load in **2 seconds** instead of 2.5 minutes:

1. **Download the cache file from Kaggle:**
   * In your notebook's right-hand panel, look at **Output** (`/kaggle/working`).
   * Expand `PhoMT_dataset`.
   * Hover over `train_cache_V30839_L30.pkl`, click the three dots (`...`) -> **Download**.
2. **Place it in your local folder:**
   * Move the downloaded `train_cache_V30839_L30.pkl` into your local `PhoMT_dataset/` folder on your computer.
3. **Re-zip the dataset:**
   ```bash
   python zip_for_kaggle.py PhoMT_dataset
   ```
4. **Update `phomt-dataset` on Kaggle:**
   * Go to `https://www.kaggle.com/datasets/<YOUR_KAGGLE_USERNAME>/phomt-dataset`
   * Click **New Version**, drag and drop the updated `PhoMT_dataset.zip`, and click **Create**.
5. **Done!** From now on, whenever Cell 2 copies `PhoMT_dataset`, the pre-tokenized cache is already there. `dataset.py` will print:
   ```text
   [dataset] Loading pre-tokenized cache from PhoMT_dataset/train_cache_V30839_L30.pkl...
   [dataset] Loaded 2,966,638 pairs.
   ```
   and immediately launch GPU training in 2 seconds flat!

---

## 3. The 3 Attached Datasets & Kaggle Layout

Your notebook operates on the three attached input datasets:

```text
/kaggle/input/
├── tensor-engine-nmt-code/   <- Source code archive (tensor-engine-nmt-code.zip)
├── phomt-dataset/            <- Parallel EN/VI corpus (PhoMT_dataset/)
└── nmt-checkpoints/          <- Model weights (.npz, logs) + optional legacy state.txt
```

---

## 4. The Kaggle Cells (Detailed Breakdown)

In your GPU-enabled Kaggle Notebook (Accelerator set to **GPU P100** or **GPU T4 x2**, and **Internet ON**), run the following cells in order.

### Cell 1: Copy Source Code & Setup GPU Environment
**What this does:** Recursively locates your code directory in `/kaggle/input/` (handling Kaggle's `/kaggle/input/datasets/...` directory hierarchy automatically), copies it to writable `/kaggle/working/`, installs `cupy-cuda12x` and the package, and verifies that the `cupy` GPU backend is active.

```python
import os, sys

# 1. Recursively search for code directory containing pyproject.toml
code_dir = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'pyproject.toml' in files:
        code_dir = root
        break

if code_dir is None:
    print("Files found in /kaggle/input:")
    for r, d, f in os.walk('/kaggle/input'):
        print(r, d, f[:3])
    raise FileNotFoundError("Could not find pyproject.toml anywhere in /kaggle/input!")

print(f"Found code directory at: {code_dir}")

# 2. Copy extracted code to writable /kaggle/working/
!cp -r {code_dir}/* /kaggle/working/

# 3. Enter directory and install dependencies
os.chdir('/kaggle/working')
sys.path.insert(0, '/kaggle/working/src')
!pip install -q -e .[gpu]

# 4. Strictly verify GPU is detected
from tensor_engine_nmt.backend import BACKEND
print(f"Active Backend: {BACKEND}")
assert BACKEND == "cupy", "CRITICAL ERROR: CuPy is not active! Make sure GPU Accelerator is enabled in Settings."
```

---

### Cell 2: Data & Checkpoint Preparation
**What this does:** Recursively locates `phomt-dataset/` and your checkpoints in `/kaggle/input/`, copies them into the writable `/kaggle/working/` directory, and verifies the files.

```python
import os

# 1. Recursively locate PhoMT_dataset
dataset_src = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'phomt-dataset' in dirs:
        dataset_src = os.path.join(root, 'phomt-dataset')
        break
    elif os.path.basename(root) == 'phomt-dataset':
        dataset_src = root
        break

print(f"Found dataset at: {dataset_src}")
!cp -r {dataset_src} ./phomt-dataset

# 2. Recursively locate checkpoints (folder containing .npz files)
ckpt_src = None
for root, dirs, files in os.walk('/kaggle/input'):
    if any(f.endswith('.npz') for f in files):
        ckpt_src = root
        break

print(f"Found checkpoints at: {ckpt_src}")
!mkdir -p checkpoints
!cp {ckpt_src}/* ./checkpoints/

# 3. Recursively locate bpe_vocab (if not already copied with code)
if not os.path.exists("bpe_vocab"):
    bpe_src = None
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'vocab.json' in files and 'merges.txt' in files:
            bpe_src = root
            break
    if bpe_src:
        !mkdir -p bpe_vocab
        !cp {bpe_src}/* ./bpe_vocab/
        print(f"Loaded existing BPE vocab from: {bpe_src}")

# Verify files are in place
!ls -lh checkpoints/
!ls -lh bpe_vocab/
```

---

### Cell 3: Pre-Flight Benchmark (Run This EVERY Session)
**What this does:** Runs a handful of **real** training steps and prints measured **tok/s**, **seconds/step** and an estimated **hours-per-epoch**. Kaggle's P100 and T4x2 give different throughput, and you share the machine — so measure before you commit 10 hours.

```python
# Measure achievable throughput for THIS session/session-type
!tensor-engine-nmt bench --steps 10
```

**How to read the output:** take the printed *hours-per-epoch* and compare it to your `--max-minutes` budget.

| Printed estimate | What to do |
|---|---|
| ≤ 9 h/epoch | Great — a 630-minute run finishes an epoch. Launch as-is. |
| 10–20 h/epoch | Normal. You'll do a partial epoch per session; that's fine, checkpoints are mid-epoch safe. |
| > 30 h/epoch | Something is wrong (CPU backend? tiny T4 slice?) — check Cell 1's `BACKEND == "cupy"` assertion before launching. |

```python
# Sanity-check the benchmark actually ran on the GPU
from tensor_engine_nmt.backend import BACKEND
assert BACKEND == "cupy", "Benchmark ran on CPU — fix the accelerator before training!"
```

---

### Cell 4: Launch the Time-Boxed Background Training Run
**What this does:** Triggers the training loop with an explicit wall-clock budget. Since we placed the checkpoints in `checkpoints/`, `train.py` will automatically detect the latest `step_XXXX.npz` / `epoch_E.npz` and resume seamlessly without losing progress.

You can choose either of the two approaches below — **both do the same thing**:

#### Option A: Command Line (Recommended for Kaggle Background Runs)
Because our project was installed with `pip install -e .` in Cell 1, the `tensor-engine-nmt train` CLI entry point loads its defaults from [`src/tensor_engine_nmt/config.py`](file:///c:/Users/admin/projects/tensor-engine-nmt/src/tensor_engine_nmt/config.py) (`e=512`, `d=1024`, `L=3`, `max_tokens=5000`, `clip_norm=2.0`, `save_every=4000`). It runs as a clean subprocess that flushes logs directly to Kaggle's background logger.

**This is the command for the short-session workflow:**

```python
# ~10.5 hours, checkpointing every ~28 minutes, keeping the newest 3 checkpoints.
!tensor-engine-nmt train \
    --lr 1e-4 \
    --max-minutes 630 \
    --save-every 500 \
    --keep-last 3
```

**Why each flag:**

* **`--max-minutes 630` (10.5 h):** Kaggle hard-kills background kernels at exactly 12 hours, with no warning and no cleanup. Stopping ourselves at 10.5 h means the run writes its final checkpoint and exits cleanly, leaving 90 minutes of headroom. If a checkpoint write were interrupted by a hard kill you could lose the newest file entirely — this avoids that.
* **`--save-every 500`:** the old advice (`save_every=4000`) was sized for uninterrupted runs. On a T4 at ~1500 tok/s with `max_tokens=5000`, one step is ~3.3 s, so **4000 steps ≈ 3.7 hours between checkpoints**. Get pre-empted twice and you've thrown away an entire session. `500` steps cuts worst-case loss to **~28 minutes** — a far better tradeoff.
* **`--keep-last 3`:** without it, `save-every 500` over a 10.5-hour session (~9,500 steps) would leave **~19 weight files ≈ 7.6 GB** in `/kaggle/working/`. This prunes everything but the newest 3 after each save, holding you at ~1.9 GB.
* **`--lr 1e-4`:** **always pass this.** `config.py` now defaults to `lr=1e-5`, but your existing step-196000 checkpoint was trained at `1e-4` — passing it keeps the optimizer trajectory consistent.

**Other flags worth knowing:**

```python
# Bound by steps instead of the clock
!tensor-engine-nmt train --lr 1e-4 --max-steps 1500 --save-every 500 --keep-last 3

# Faster epochs on a limited session (builds a smaller cache from a strided sample)
!tensor-engine-nmt train --lr 1e-4 --max-minutes 630 --max-pairs 400000 --save-every 500

# Anneal teacher forcing faster, because you're doing fewer total steps
!tensor-engine-nmt train --lr 1e-4 --max-minutes 630 --k 8000 --save-every 500

# Resume from one specific file rather than the newest
!tensor-engine-nmt train --lr 1e-4 --max-minutes 630 --save-every 500 --resume checkpoints/step_196000.npz

# Full-size model needs VRAM headroom? shrink the batch budget
!tensor-engine-nmt train --lr 1e-4 --max-minutes 630 --max-tokens 3000 --save-every 500
```

#### Option B: Direct Python Call (Like Colab Guide)
If you prefer seeing and tweaking every hyperparameter directly inside your notebook cell without touching any Python files, you can call `train(hp=hp)` in pure Python. Note the corrected values — `V=30839` (the real BPE vocab size; `config.py`'s `32000` is just an upper bound and `V` only needs to be ≥ the true size), `max_tokens=5000`, `clip_norm=2.0`, `save_every=500`, and `lr=1e-4` passed explicitly:

```python
from tensor_engine_nmt.config import HParams
from tensor_engine_nmt.train import train

hp = HParams(
    V=30839,          # Real trained BPE vocab size (V must be >= this)
    e=512,            # Embedding dimension
    d=1024,           # LSTM hidden dimension (3 layers)
    L=3,              # Stacked LSTM layers
    max_tokens=5000,  # Dynamic token batching (maximizes 16GB GPU VRAM)
    k=17000.0,        # Teacher forcing inverse-sigmoid decay constant
    min_tf=0.70,      # Minimum teacher forcing floor (prevents exposure bias collapse)
    lr=1e-4,          # Learning rate — matches the existing step-196k checkpoint
    beta1=0.9, beta2=0.999, eps_adam=1e-8,
    clip_norm=2.0,
    max_epochs=30,
    beam_width=4,
    max_decode_len=30,
    log_every=100,
    save_every=500,   # Short-session value, NOT the 4000 config default
    data_dir='PhoMT_dataset',
    bpe_dir='bpe_vocab',
    ckpt_dir='checkpoints',
    bpe_sample_lines=300000,
)

train(hp=hp, max_minutes=630)   # 10.5-hour clean stop; max_steps=1500 to bound by steps instead
```

> **Note:** `max_minutes` and `max_steps` are *arguments to `train()`*, not `HParams` fields — so Option B can time-box too, as shown above. The remaining flags (`--keep-last`, `--max-pairs`, `--resume`) are `HParams` fields or handled by auto-resume.

---

### Cell 5: Evaluate BLEU Score & Translation Quality
**What this does:** Computes Corpus BLEU-4 on the test set using **Beam Search (`method="beam"`)** and automatically appends quantitative sentence-by-sentence evaluation to `checkpoints/evaluation_result.txt`.

You can run evaluation either via CLI or pure Python:

```python
# Option A: Run via CLI (Beam Search by default; filters max_len <= 30)
!tensor-engine-nmt evaluate --method beam --n 200 --verbose

# Run the analyzer to print a summary of recent evaluation logs:
!python analyze.py --eval-only
```

Or via Python:
```python
# Option B: Run via Python function
from tensor_engine_nmt.evaluate import evaluate

bleu_test = evaluate(hp=hp, split="test", method="beam", max_sentences=200, verbose=True)
print(f"\nCorpus BLEU-4 (Beam Search): {bleu_test:.2f}")
```

---

### Cell 6: Quick Translation / Interactive Demo
**What this does:** Test translations on custom English sentences or run the dedicated interactive REPL tool:

```python
from tensor_engine_nmt.inference import Translator
from tensor_engine_nmt.bpe import BPETokenizer
from tensor_engine_nmt.model import Seq2Seq
from demo import find_latest_checkpoint

# 1. Automatically pick the latest checkpoint numerically
ckpt = find_latest_checkpoint("checkpoints")
print(f"Loading checkpoint: {ckpt}")
model = Seq2Seq(hp)
model.load(ckpt)

bpe = BPETokenizer.load(hp.bpe_dir)
translator = Translator(model, bpe, hp)

# 2. Test translations
sentences = [
    "Hello, how are you?",
    "The economy of Vietnam is growing rapidly.",
    "He can water the horses .",
]

for s in sentences:
    greedy_vi = translator.translate(s, method="greedy")
    beam_vi = translator.translate(s, method="beam", beam_width=4)
    print(f"EN:     {s}")
    print(f"Greedy: {greedy_vi}")
    print(f"Beam-4: {beam_vi}")
    print("-" * 50)
```

*(You can also run `!python demo.py --method both --beam-width 4` in a terminal/console session for an interactive CLI).*

---

## 5. Background Training (The Kaggle Superpower)

### Interactive Session vs. Background Run:
* **Interactive Session (Running in the browser):** Great for testing and checking output in real-time. **Do NOT close your laptop or browser tab.** If disconnected, Kaggle kills interactive sessions after ~20–40 minutes of idle network time.
* **Background Run (`Save & Run All`):** The true superpower. You can completely close your browser and shut down your laptop. Kaggle spins up a dedicated headless VM that trains continuously until the **12-hour hard-kill ceiling** — which is a wall to stay well clear of, not a target to fill. We size runs at `--max-minutes 630` (10.5 h) so the process stops itself and saves cleanly first.

### How to Start a ~10.5-Hour Background Run:
1. If an interactive session is currently running, stop it first (click **Cancel Run** or the power icon `⏻` -> **Stop session**), because Kaggle allows only 1 GPU active per account.
2. Look at the top right of the Kaggle interface and click the black **Save Version** button.
3. Select **Save & Run All (Commit)**.
4. Click **Save**.
5. Once the bottom notification says `Running version X...`, you can safely close your laptop and walk away!

Because Cell 4 runs with `--max-minutes 630`, the kernel stops itself at ~10.5 hours, flushes its final checkpoint, and exits — **before** Kaggle's 12-hour hard kill. You get a clean finish, the version gets marked complete, and its output files are all intact and downloadable. A run that gets hard-killed at 12:00:00 is a run whose last several minutes of checkpoint writes you cannot trust.

---

### Mandatory Rule: Download Checkpoints from Output After EVERY Session!

> [!IMPORTANT]
> **Always download your latest checkpoint files as soon as any session finishes.**  
> Kaggle's `/kaggle/working/` directory is ephemeral. If you start a new interactive session or launch a new background run without saving your outputs, you risk losing your newly trained steps.

#### The Critical Files to Download Every Time:
1. **`step_XXXX.npz`** — The newest model weights (e.g. `step_88000.npz`). ≈ **401 MB**. Also carries the epoch/step position inside it.
2. **`optim_state.npz`** — The Adam optimizer moments, essential for continuous momentum. ≈ **733 MB**.
3. **`train_logs.txt`** — The training log with loss curves.
4. **`evaluation_result.txt`** — Quantitative BLEU-4 metrics and test set sample outputs (auto-appended during evaluation).
5. **`state.txt`** — *Optional, legacy only.* The epoch position now lives inside the checkpoint (see §6.5), so this is a fallback for older tooling rather than something a resume needs.

> [!WARNING]
> **`optim_state.npz` is a single shared file, tagged with the checkpoint it belongs to.** It is not per-step. If you resume from an **older** `step_*.npz` while the companion `optim_state.npz` came from a **newer** one, training prints a **loud warning and starts the optimizer fresh** instead of silently pairing mismatched Adam moments. Always download and store weights + optimizer state **as a matched pair from the same session** — never mix a `step_*.npz` from one version of the dataset with an `optim_state.npz` from another.

#### How to Retrieve Them:
* **After a Background Run:**
  1. Open your Notebook page on Kaggle.
  2. Click on the completed version under the **Versions** menu (or click the **Output** tab).
  3. Under `checkpoints/`, click **Download** to save the `.npz`, `.txt`, and log files to your computer.
* **From an Active Interactive Session:**
  1. In the right panel, open the **Output** section (`/kaggle/working`).
  2. Expand `checkpoints/`.
  3. Hover over the files or click the three dots (`...`) -> **Download**.

#### Next Step After Downloading:
Immediately go to your **`nmt-checkpoints`** dataset on Kaggle, click **Upload New Version**, drop the newly downloaded files in, and click **Create**. This guarantees your progress is permanently locked in and ready for the next 10.5-hour session or for switching to another Kaggle account! Also copy the pair (`step_*.npz` + `optim_state.npz`) onto your laptop as an off-platform backup — 481 MB per step is small enough that there is no excuse for having only one copy.

---

## 6. Critical Concerns & Limitations to Watch Out For

When using Kaggle, there are a few strict limitations you must architect around:

### 1. The 12-Hour Execution Limit — and Why We Stop at 10.5
Kaggle background kernels are hard-killed exactly at the 12-hour mark, without warning and without a graceful shutdown.

**This is why every run is launched with `--max-minutes 630` (10.5 hours).** The process watches the wall clock, and when the budget is exhausted it **writes a checkpoint and exits cleanly**. The last thing that happens in a session is a successful save — not a truncated file. Resuming is automatic: the next session finds the newest `step_*.npz` and continues.

**Concern:** if you *don't* pass `--max-minutes` and the kernel is killed mid-epoch, you fall back to the last rolling checkpoint saved via `save_every`. With the config default `save_every = 4000` — and at ~1500 tok/s on a T4 with `max_tokens=5000`, that's ~3.3 s/step — **one checkpoint interval is roughly 3.7 hours**. That is far too much work to leave to chance. Use `--save-every 500` (~28 minutes) and `--keep-last 3`.

### 2. Disk Space Limits (`/kaggle/working/`)
Kaggle limits the `/kaggle/working/` directory to **20 GB of output**.

**Concern:** the full-size model's `step_NNNNNN.npz` is ≈ **401 MB** and `optim_state.npz` is ≈ **733 MB**. So each *kept* checkpoint costs ~1.1 GB once you count the shared optimizer state.

* With the config default `save_every = 4000`, a 10.5-hour run (~11,000 steps) generates only ~3 weight files (~1.2 GB) + `optim_state.npz` (733 MB) + dataset cache (~200 MB) — very safe, but the flip side is those sparse checkpoints are 3.7 hours apart.
* With `--save-every 500` and **no pruning**, the same run would generate ~22 weight files (~8.8 GB) — plus the optimizer state, that's ~9.7 GB, and repeated sessions would push you toward the ceiling.
* With `--save-every 500 --keep-last 3`, you keep **3 × 401 MB + 733 MB ≈ 1.9 GB**, plus the cache. Frequent checkpoints at no disk cost.

**So: use `--save-every 500` together with `--keep-last 3`.** Never crank `save_every` down without also passing `--keep-last`.

### 3. RAM Limits (30 GB CPU RAM)
**Concern:** Our `PhoMTDataset` streams data line-by-line, which is highly memory efficient. However, `dataset.py` builds an in-memory cache to bucket-sort sequences. If the cache grows too large, Kaggle's CPU RAM might spike. Watch the resource monitor in the top right during the first epoch to ensure RAM stays below 30 GB.

### 4. The Multi-Account Quota Shuffle
Kaggle allows 30 hours of GPU time per week. If you plan to train for 72 hours, you will need at least 3 Kaggle accounts.
**Concern:** The migration process is manual. When Account A runs out of time, you must download the latest checkpoints to your laptop, upload them to a new Dataset on Account B, update the notebook on Account B to use the new Dataset, and launch the background run again. This "hot-swapping" creates downtime, so factor it into your schedule.

### 5. Epoch Position Lives Inside the Checkpoint (`state.txt` Is Legacy)
The epoch/step position is now stored **inside the `.npz`** as `meta_epoch` and `meta_step_in_epoch`. On resume, `train.py` reads those keys directly from the newest checkpoint.

**Consequence:** `state.txt` is **no longer required for a correct resume**. It is still written alongside as a fallback for older tooling and notebooks, but the checkpoint is authoritative — you can lose or delete `state.txt` and resume will still land on the exact step it left off. This matters when you're shuttling files between Kaggle accounts and datasets: if `state.txt` and `step_*.npz` ever disagree, trust the `.npz`.

### 6. Evaluation and Translation Are Cheap on CPU
You don't need to burn Kaggle GPU quota to look at your model. Copy a checkpoint plus the `bpe_vocab/` folder to your laptop and run locally:

```bash
# On your laptop — no GPU required
tensor-engine-nmt evaluate --method greedy --n 200
python demo.py
```

Greedy decoding over 200 sentences is a few minutes of ordinary CPU work. Use Kaggle's 30 GPU hours/week for **training**; do the inspecting and translation sanity checks at home.

### 7. When Kaggle Quota Runs Out: The `--preset laptop` Escape Hatch
Kaggle gives you 30 GPU hours/week; a full-size model wants far more than that. If you're out of quota and still want to make forward progress on your own machine, there's a small configuration built in:

```bash
tensor-engine-nmt bench  --preset laptop --steps 10
tensor-engine-nmt train  --preset laptop --lr 1e-4 --max-minutes 120
```

`--preset laptop` shrinks the model a lot (`d=256`, `L=2`, `e=128`, `V=8000`, `max_len=20`) and writes to its own `bpe_vocab_laptop/` and `checkpoints_laptop/` directories, so laptop runs can never collide with — or be auto-resumed into — your full-size GPU checkpoints. It already bundles sensible short-session settings (`max_pairs=150000`, `save_every=100`, `keep_last=3`, `k=6000`, `min_tf=0.50`), so you mainly need to supply `--lr` and `--max-minutes`. It is a different, much smaller model — useful for pipeline testing and for keeping momentum, not a drop-in replacement for the full-size run.

---

## 8. Short-Session CLI Reference

Everything the time-boxed workflow uses, in one place.

**Training — `tensor-engine-nmt train`**

| Flag | Default | What it does |
|---|---|---|
| `--preset {gpu,laptop}` | `gpu` | `laptop` shrinks the model a lot (`d=256`, `L=2`, `e=128`, `V=8000`, `max_len=20`) and uses its own `bpe_vocab_laptop/` + `checkpoints_laptop/` dirs |
| `--lr FLOAT` | `1e-5` (from `config.py`) | **Pass `1e-4` explicitly** to match the existing step-196k checkpoint |
| `--max-tokens INT` | `5000` | Dynamic token batching budget |
| `--min-tf FLOAT` | `0.70` | Teacher-forcing floor |
| `--k FLOAT` | `17000` | Inverse-sigmoid teacher-forcing decay constant (smaller anneals faster) |
| `--max-steps INT` | — | Stop after N gradient steps |
| `--max-minutes FLOAT` | — | **Stop cleanly after N wall-clock minutes**: writes a checkpoint first, then exits. Resuming is automatic |
| `--save-every INT` | `4000` | Checkpoint every N steps — use `500` for short sessions |
| `--log-every INT` | `100` | Log every N steps |
| `--keep-last INT` | — | After each save, delete older `step_*.npz` beyond the newest N (protects disk in long runs) |
| `--max-pairs INT` | — | Cap the training set to an evenly-strided sample of N pairs (builds a smaller cache) |
| `--resume PATH` | auto | Resume from a specific `.npz` |

**Benchmark — `tensor-engine-nmt bench`**

```bash
tensor-engine-nmt bench [--preset {gpu,laptop}] [--steps INT] [--max-tokens INT] [--max-pairs INT]
```

Runs a few **real** training steps and prints measured **tok/s**, **seconds/step** and an estimated **hours-per-epoch** — so you can size a session *before* committing 10 hours to it.
