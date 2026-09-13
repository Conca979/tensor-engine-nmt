# Training on Kaggle (Background Runs & High-Speed GPU)

Kaggle offers **free 30 hours of GPU time per week**, and importantly, allows you to click "Save & Run All" to run your training loop in the **background for up to 12 hours** without needing to keep a browser tab open.

> [!CAUTION]
> ### Mandatory Prerequisite: Phone Verification is Required for GPU Access
> Kaggle strictly requires a **one-time SMS Phone Verification** before allowing access to free GPUs (GPU P100 / GPU T4x2) and outbound internet.
> * **If your account is NOT verified:** The "Accelerator" setting is locked to **CPU only** (`None`), and the "Internet" toggle is locked to **OFF**. Because training our 114.9M parameter model on CPU is impossible (~15 tok/s = months per epoch), **you cannot train on Kaggle without phone verification**.
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
   * `state.txt`
   * `train_logs.txt`
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

When a 12-hour Kaggle run completes, you will have newer checkpoints (`step_88000.npz`, updated `optim_state.npz`, updated `state.txt`, `train_logs.txt`). **You do NOT need to create a new dataset from scratch.** Instead, update the existing dataset with a new version:

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

When you ran training for the first time, `dataset.py` spent ~2.5 minutes pre-tokenizing 2,966,638 sentence pairs and saved `train_cache_V30839.pkl` into `/kaggle/working/PhoMT_dataset/`.

Because `/kaggle/input/` datasets are read-only, this cache is stored in temporary session output. To make every future session and every secondary Kaggle account load in **2 seconds** instead of 2.5 minutes:

1. **Download the cache file from Kaggle:**
   * In your notebook's right-hand panel, look at **Output** (`/kaggle/working`).
   * Expand `PhoMT_dataset`.
   * Hover over `train_cache_V30839.pkl`, click the three dots (`...`) -> **Download**.
2. **Place it in your local folder:**
   * Move the downloaded `train_cache_V30839.pkl` into your local `PhoMT_dataset/` folder on your computer.
3. **Re-zip the dataset:**
   ```bash
   python zip_for_kaggle.py PhoMT_dataset
   ```
4. **Update `phomt-dataset` on Kaggle:**
   * Go to `https://www.kaggle.com/datasets/<YOUR_KAGGLE_USERNAME>/phomt-dataset`
   * Click **New Version**, drag and drop the updated `PhoMT_dataset.zip`, and click **Create**.
5. **Done!** From now on, whenever Cell 2 copies `PhoMT_dataset`, the pre-tokenized cache is already there. `dataset.py` will print:
   ```text
   [dataset] Loading pre-tokenized cache from PhoMT_dataset/train_cache_V30839.pkl...
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
└── nmt-checkpoints/          <- Model weights (.npz, state.txt, logs)
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
**What this does:** Recursively locates `PhoMT_dataset/` and your checkpoints in `/kaggle/input/`, copies them into the writable `/kaggle/working/` directory, and verifies the files.

```python
import os

# 1. Recursively locate PhoMT_dataset
dataset_src = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'PhoMT_dataset' in dirs:
        dataset_src = os.path.join(root, 'PhoMT_dataset')
        break
    elif os.path.basename(root) == 'PhoMT_dataset':
        dataset_src = root
        break

print(f"Found dataset at: {dataset_src}")
!cp -r {dataset_src} ./PhoMT_dataset

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

### Cell 3: Start Training Execution
**What this does:** Triggers the training loop. Since we placed the checkpoints in the `checkpoints/` folder, `train.py` will automatically detect the latest `step_XXXX.npz` or `epoch_E.npz` and resume seamlessly without losing progress.

You can choose either of the two approaches below — **both do the exact same thing**:

#### Option A: Quick Command Line (Recommended for Kaggle Background Runs)
Because our project was installed with `pip install -e .` in Cell 1, the `tensor-engine-nmt train` CLI entry point automatically loads all default hyperparameters directly from [`src/tensor_engine_nmt/config.py`](file:///c:/Users/admin/projects/tensor-engine-nmt/src/tensor_engine_nmt/config.py) (`e=512`, `d=1024`, `L=3`, `max_tokens=4000`, `save_every=2000`). It runs as a clean subprocess that flushes logs directly to Kaggle's background logger:

```python
# Run the training loop via CLI entrypoint
# (Optional flags: --min-tf 0.70, --lr 3e-4, --max-tokens 4000, --resume PATH)
!tensor-engine-nmt train --min-tf 0.70
```

#### Option B: Direct Python Call (Like Colab Guide)
If you prefer seeing and tweaking every hyperparameter directly inside your notebook cell without touching any Python files, you can call `train(hp=hp)` in pure Python:

```python
from tensor_engine_nmt.config import HParams
from tensor_engine_nmt.train import train

# Hyperparameters matching config.py exactly
hp = HParams(
    V=32000,          # Full vocabulary size
    e=512,            # Embedding dimension
    d=1024,           # LSTM hidden dimension (3 layers)
    L=3,              # Stacked LSTM layers
    max_tokens=4000,  # Dynamic token batching (maximizes 16GB GPU VRAM)
    k=17000.0,        # Teacher forcing inverse-sigmoid decay constant
    min_tf=0.70,      # Minimum teacher forcing floor (prevents exposure bias collapse)
    lr=3e-4,          # Learning rate
    beta1=0.9, beta2=0.999, eps_adam=1e-8,
    clip_norm=5.0,
    max_epochs=30,
    beam_width=4,
    max_decode_len=30,
    log_every=100,
    save_every=2000,
    data_dir='PhoMT_dataset',
    bpe_dir='bpe_vocab',
    ckpt_dir='checkpoints',
    bpe_sample_lines=300000,
)

train(hp=hp)
```

---

## 5. Background Training (The Kaggle Superpower)

### Interactive Session vs. Background Run:
* **Interactive Session (Running in the browser):** Great for testing and checking output in real-time. **Do NOT close your laptop or browser tab.** If disconnected, Kaggle kills interactive sessions after ~20–40 minutes of idle network time.
* **Background Run (`Save & Run All`):** The true superpower. You can completely close your browser and shut down your laptop. Kaggle spins up a dedicated headless VM that trains continuously for **up to 12 hours**.

### How to Start a 12-Hour Background Run:
1. If an interactive session is currently running, stop it first (click **Cancel Run** or the power icon `⏻` -> **Stop session**), because Kaggle allows only 1 GPU active per account.
2. Look at the top right of the Kaggle interface and click the black **Save Version** button.
3. Select **Save & Run All (Commit)**.
4. Click **Save**.
5. Once the bottom notification says `Running version X...`, you can safely close your laptop and walk away!

---

### Mandatory Rule: Download Checkpoints from Output After EVERY Session!

> [!IMPORTANT]
> **Always download your latest checkpoint files as soon as any session finishes.**  
> Kaggle's `/kaggle/working/` directory is ephemeral. If you start a new interactive session or launch a new background run without saving your outputs, you risk losing your newly trained steps.

#### The 4 Critical Files to Download Every Time:
1. **`step_XXXX.npz`** — The newest model weights (e.g. `step_88000.npz`).
2. **`optim_state.npz`** — The Adam optimizer moments (essential for continuous learning rate & momentum).
3. **`state.txt`** — The exact epoch and batch index for mid-epoch resume.
4. **`train_logs.txt`** — The training log with loss curves.

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
Immediately go to your **`nmt-checkpoints`** dataset on Kaggle, click **Upload New Version**, drop the newly downloaded files in, and click **Create**. This guarantees your progress is permanently locked in and ready for the next 12-hour session or for switching to another Kaggle account!

---

## 6. Critical Concerns & Limitations to Watch Out For

When using Kaggle, there are a few strict limitations you must architect around:

### 1. The 12-Hour Execution Limit
Kaggle background kernels are hard-killed exactly at the 12-hour mark. 
**Concern:** If the kernel is killed mid-epoch, you will rely on the last rolling checkpoint (`step_XXXX.npz`) saved via your `cfg.save_every` parameter. With our configured `save_every = 2000`, a checkpoint is saved approximately every ~2 hours on a GPU, so you never lose more than 2 hours of progress when the 12-hour timeout hits.

### 2. Disk Space Limits (`/kaggle/working/`)
Kaggle limits the `/kaggle/working/` directory to **20 GB of output**.
**Concern:** Each checkpoint is ~410 MB (with `e=512`). With `save_every = 2000`, a 12-hour run (~12,000 steps) generates only ~6 checkpoints (~2,460 MB), plus `optim_state.npz` (~720 MB) and dataset cache (~200 MB) — well below 3.5 GB total! This guarantees you will never hit Kaggle's 20 GB ceiling. Avoid setting `save_every` too low (e.g., every 100 steps) which would spam hundreds of files and exhaust the disk.

### 3. RAM Limits (30 GB CPU RAM)
**Concern:** Our `PhoMTDataset` streams data line-by-line, which is highly memory efficient. However, `dataset.py` builds an in-memory cache to bucket-sort sequences. If the cache grows too large, Kaggle's CPU RAM might spike. Watch the resource monitor in the top right during the first epoch to ensure RAM stays below 30 GB.

### 4. The Multi-Account Quota Shuffle
Kaggle allows 30 hours of GPU time per week. If you plan to train for 72 hours, you will need at least 3 Kaggle accounts.
**Concern:** The migration process is manual. When Account A runs out of time, you must download the latest checkpoints to your laptop, upload them to a new Dataset on Account B, update the notebook on Account B to use the new Dataset, and launch the background run again. This "hot-swapping" creates downtime, so factor it into your schedule.
