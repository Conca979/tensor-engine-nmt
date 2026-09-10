"""
train.py — Full training loop for the Seq2Seq NMT model.

Flow:
  1. Load (or train) BPE vocab
  2. Build model + Adam optimizer
  3. Resume from checkpoint if present
  4. Loop epochs → batches → forward → backward → optimizer.step
  5. Log every `cfg.log_every` steps
  6. Checkpoint every `cfg.save_every` steps

Run via:
    tensor-engine-nmt train [--max-steps N] [--resume PATH]
"""
import os
import time
import glob
import argparse
import numpy as np
import logging
import itertools

from .config import cfg, HParams
from .bpe import load_or_train_bpe
from .dataset import PhoMTDataset
from .model import Seq2Seq
from .optimizer import Adam


def train(hp: HParams = cfg, max_steps: int = None, resume: str = None) -> None:
    """
    Main training function.

    Parameters
    ----------
    hp        : HParams — hyperparameter config
    max_steps : int or None — stop after this many gradient steps (for smoke tests)
    resume    : str or None — path to a checkpoint .npz to resume from
    """
    # ── 0. Setup Logging ─────────────────────────────────────────────────────
    os.makedirs(hp.ckpt_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler()
        ],
        force=True
    )

    # ── 1. BPE ───────────────────────────────────────────────────────────────
    bpe = load_or_train_bpe(
        bpe_dir=hp.bpe_dir,
        data_dir=hp.data_dir,
        target_vocab=hp.V,
        sample_lines=hp.bpe_sample_lines,
    )

    # ── 2. Model + Optimizer ─────────────────────────────────────────────────
    model = Seq2Seq(hp)
    logging.info(f"[train] Model parameters: {model.param_count():,}")
    from .backend import xp
    logging.info(f"[train] Backend: {xp.__name__}")

    optim = Adam(model, hp)

    # ── 3. Resume from checkpoint ─────────────────────────────────────────────
    global_step = 0
    start_epoch = 0
    if resume and os.path.exists(resume):
        model.load(resume, optim=optim)
        # Parse step from filename if it follows the naming convention
        try:
            global_step = int(os.path.basename(resume).split("_")[1].replace(".npz", ""))
        except Exception:
            pass
        logging.info(f"[train] Resuming from step {global_step}")
    else:
        # Auto-find the latest checkpoint (sort numerically, not alphabetically)
        def _step_num(path):
            try: return int(os.path.basename(path).split("_")[1].replace(".npz", ""))
            except: return -1
        ckpts = sorted(glob.glob(os.path.join(hp.ckpt_dir, "step_*.npz")), key=_step_num)
        if ckpts:
            latest = ckpts[-1]
            model.load(latest, optim=optim)
            try:
                global_step = int(os.path.basename(latest).split("_")[1].replace(".npz", ""))
            except Exception:
                pass
            logging.info(f"[train] Auto-resumed from {latest} (step {global_step})")

    # ── 4. Dataset ───────────────────────────────────────────────────────────
    dataset = PhoMTDataset(bpe, data_dir=hp.data_dir, split="train", max_len=100)

    # ── 5. Infer epoch position from global_step ─────────────────────────────
    start_epoch = 0
    steps_done_in_ep = 0
    state_file = os.path.join(hp.ckpt_dir, "state.txt")

    if global_step > 0:
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    ep_str, step_str = f.read().strip().split(",")
                    start_epoch = int(ep_str)
                    steps_done_in_ep = int(step_str)
                logging.info(f"[train] Loaded exact epoch position: epoch {start_epoch + 1}/{hp.max_epochs}, step {steps_done_in_ep} within epoch")
            except Exception as e:
                logging.warning(f"[train] Could not read {state_file}, falling back to estimate.")
        
        if start_epoch == 0 and steps_done_in_ep == 0 and not os.path.exists(state_file):
            # Fallback to estimation for existing checkpoints without state.txt
            estimated_batch_size = getattr(hp, 'max_tokens', 4000) // 50
            if estimated_batch_size <= 0: estimated_batch_size = 64
            steps_per_epoch = len(dataset.pairs) // estimated_batch_size
            if steps_per_epoch > 0:
                start_epoch      = global_step // steps_per_epoch
                steps_done_in_ep = global_step %  steps_per_epoch
            logging.info(f"[train] Epoch position (estimated): epoch {start_epoch + 1}/{hp.max_epochs}, step {steps_done_in_ep}/{steps_per_epoch} within epoch")

    # Guard: already finished all epochs
    if start_epoch >= hp.max_epochs:
        logging.info(f"[train] All {hp.max_epochs} epochs already completed at step {global_step}. Nothing to do.")
        return

    # ── 6. Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, hp.max_epochs):
        logging.info(f"\n{'='*60}")
        logging.info(f"  Epoch {epoch + 1}/{hp.max_epochs}  |  global_step={global_step}")
        logging.info(f"{'='*60}")

        # Skip batches already completed in this epoch (only relevant on mid-epoch resume)
        skip_batches = steps_done_in_ep if epoch == start_epoch else 0
        steps_done_in_ep = 0   # only skip on the first resumed epoch

        epoch_loss   = 0.0
        epoch_steps  = 0
        epoch_tokens = 0
        t0 = time.time()

        batch_iterator = dataset.iterate(
            max_tokens=getattr(hp, 'max_tokens', 4000), 
            shuffle=True,
            seed=(42 + epoch)
        )
        if skip_batches > 0:
            logging.info(f"Skipping first {skip_batches} batches to resume mid-epoch...")
            batch_iterator = itertools.islice(batch_iterator, skip_batches, None)

        current_step_in_epoch = skip_batches

        for batch_idx, batch in enumerate(batch_iterator):
            current_step_in_epoch += 1
            X, Xlen, Yin, Yout, Ylen = (
                batch["X"], batch["Xlen"],
                batch["Yin"], batch["Yout"], batch["Ylen"]
            )

            # ── Forward ──────────────────────────────────────────────────────
            model.zero_grad()
            loss_val, _ = model.forward(X, Xlen, Yin, Yout, Ylen, global_step)

            # ── Backward ─────────────────────────────────────────────────────
            model.backward()

            # ── Optimiser step ────────────────────────────────────────────────
            grad_norm = optim.step()

            # ── Bookkeeping ───────────────────────────────────────────────────
            n_tokens    = int(Ylen.sum())
            epoch_loss   += loss_val
            epoch_steps  += 1
            epoch_tokens += n_tokens
            global_step  += 1

            # ── Logging ───────────────────────────────────────────────────────
            if global_step % hp.log_every == 0:
                elapsed   = time.time() - t0 + 1e-9
                tok_per_s = epoch_tokens / elapsed
                eps_i     = hp.k / (hp.k + np.exp(global_step / hp.k))
                avg_loss  = epoch_loss / epoch_steps
                log_msg = (
                    f"ep {epoch+1}/{hp.max_epochs} | "
                    f"step {global_step:7d} | "
                    f"loss={avg_loss:.4f} | "
                    f"lr={hp.lr:.2e} | "
                    f"gnorm={grad_norm:.3f} | "
                    f"ε={eps_i:.3f} | "
                    f"tok/s={tok_per_s:,.0f}"
                )
                logging.info(log_msg)
                
                # Manually open and close the file to force Google Drive to sync immediately
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
                with open(os.path.join(hp.ckpt_dir, "train_logs.txt"), "a", encoding="utf-8") as f:
                    f.write(f"{timestamp} | INFO | {log_msg}\n")
                    
                epoch_loss  = 0.0
                epoch_steps = 0
                t0 = time.time()
                epoch_tokens = 0

            # ── Checkpoint ────────────────────────────────────────────────────
            if global_step % hp.save_every == 0:
                ckpt_path = os.path.join(hp.ckpt_dir, f"step_{global_step}.npz")
                model.save(ckpt_path, optim=optim)
                with open(os.path.join(hp.ckpt_dir, "state.txt"), "w") as f:
                    f.write(f"{epoch},{current_step_in_epoch}")

            if max_steps and global_step >= max_steps:
                logging.info(f"[train] Reached max_steps={max_steps}. Stopping.")
                return

        # End of epoch
        final_ckpt = os.path.join(hp.ckpt_dir, f"epoch_{epoch + 1}.npz")
        model.save(final_ckpt, optim=optim)
        with open(os.path.join(hp.ckpt_dir, "state.txt"), "w") as f:
            f.write(f"{epoch + 1},0")


def main_train():
    parser = argparse.ArgumentParser(description="Train the Seq2Seq NMT model")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N steps")
    parser.add_argument("--resume",    type=str, default=None, help="Path to checkpoint .npz")
    parser.add_argument("--lr",         type=float, default=cfg.lr)
    parser.add_argument("--max-tokens", type=int,   default=getattr(cfg, 'max_tokens', 4000))
    args = parser.parse_args()

    import dataclasses
    
    hp = dataclasses.replace(cfg, lr=args.lr, max_tokens=args.max_tokens)
    train(hp=hp, max_steps=args.max_steps, resume=args.resume)
