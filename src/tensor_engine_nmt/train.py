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

from .config import cfg, HParams, PRESETS, apply_preset
from .bpe import load_or_train_bpe
from .dataset import PhoMTDataset
from .model import Seq2Seq
from .decoder import DecoderLSTM
from .optimizer import Adam


def prune_step_checkpoints(ckpt_dir: str, keep_last: int) -> None:
    """Delete the oldest `step_*.npz` files so only the newest `keep_last` remain.

    Only step checkpoints are touched — `optim_state.npz` belongs to the newest
    one and `epoch_*.npz` files are meant to be long-lived.
    """
    if not keep_last or keep_last <= 0:
        return
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.npz")), key=_checkpoint_step)
    for old in ckpts[:-keep_last]:
        try:
            os.remove(old)
            logging.info(f"[train] Pruned old checkpoint {os.path.basename(old)} (keep_last={keep_last})")
        except OSError as e:
            logging.warning(f"[train] Could not prune {old}: {e}")


def _checkpoint_step(path: str) -> int:
    try:
        return int(os.path.basename(path).split("_")[1].replace(".npz", ""))
    except Exception:
        return -1


def train(
    hp: HParams = cfg,
    max_steps: int = None,
    resume: str = None,
    max_minutes: float = None,
) -> None:
    """
    Main training function.

    Parameters
    ----------
    hp          : HParams — hyperparameter config
    max_steps   : int or None — stop after this many gradient steps (for smoke tests)
    resume      : str or None — path to a checkpoint .npz to resume from
    max_minutes : float or None — stop cleanly after this many wall-clock minutes.
                  A checkpoint is written before exiting, so re-running the same
                  command resumes exactly where this one stopped.  This is the
                  flag that makes short training sessions safe.
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

    # The embedding tables are sized hp.V, so every id the tokenizer can emit must
    # fit.  Getting this wrong is an IndexError deep inside the forward pass.
    if len(bpe.token2id) > hp.V:
        raise ValueError(
            f"hp.V={hp.V:,} is smaller than the loaded BPE vocabulary "
            f"({len(bpe.token2id):,} tokens in {hp.bpe_dir}). Embedding lookups "
            f"would run out of bounds. Use a preset whose bpe_dir matches its V, "
            f"or raise V to at least {len(bpe.token2id):,}."
        )

    # ── 2. Model + Optimizer ─────────────────────────────────────────────────
    model = Seq2Seq(hp)
    logging.info(f"[train] Model parameters: {model.param_count():,}")
    from .backend import xp
    logging.info(f"[train] Backend: {xp.__name__}")
    logging.info(f"[train] Config: d={hp.d} L={hp.L} e={hp.e} V={hp.V} max_len={hp.max_len} "
                 f"max_tokens={hp.max_tokens} lr={hp.lr:g} k={hp.k:g} min_tf={hp.min_tf:g} "
                 f"save_every={hp.save_every}")
    if max_minutes:
        logging.info(f"[train] Session budget: {max_minutes:g} minutes "
                     f"(will checkpoint and exit cleanly)")

    optim = Adam(model, hp)

    # ── 3. Resume from checkpoint ─────────────────────────────────────────────
    global_step = 0
    start_epoch = 0
    ckpt_meta: dict = {}

    _step_num = _checkpoint_step

    if resume and os.path.exists(resume):
        ckpt_meta = model.load(resume, optim=optim)
        # Parse step from filename if it follows the naming convention
        global_step = max(_step_num(resume), 0)
        logging.info(f"[train] Resuming from step {global_step}")
    else:
        # Auto-find the latest checkpoint (sort numerically, not alphabetically)
        ckpts = sorted(glob.glob(os.path.join(hp.ckpt_dir, "step_*.npz")), key=_step_num)
        if ckpts:
            latest = ckpts[-1]
            ckpt_meta = model.load(latest, optim=optim)
            global_step = max(_step_num(latest), 0)
            logging.info(f"[train] Auto-resumed from {latest} (step {global_step})")

    # ── 4. Dataset ───────────────────────────────────────────────────────────
    dataset = PhoMTDataset(bpe, data_dir=hp.data_dir, split="train",
                           max_len=hp.max_len, max_pairs=hp.max_pairs)

    # ── 5. Recover the position inside the epoch ──────────────────────────────
    steps_done_in_ep = 0
    state_file = os.path.join(hp.ckpt_dir, "state.txt")

    if global_step > 0:
        if "epoch" in ckpt_meta and "step_in_epoch" in ckpt_meta:
            # Authoritative: written atomically together with the weights.
            start_epoch      = ckpt_meta["epoch"]
            steps_done_in_ep = ckpt_meta["step_in_epoch"]
            logging.info(f"[train] Checkpoint metadata: epoch {start_epoch + 1}/{hp.max_epochs}, "
                         f"step {steps_done_in_ep} within epoch")
        elif os.path.exists(state_file):
            # Legacy checkpoints written before the position was stored inside them.
            try:
                with open(state_file, "r") as f:
                    ep_str, step_str = f.read().strip().split(",")
                    start_epoch = int(ep_str)
                    steps_done_in_ep = int(step_str)
                logging.info(f"[train] Loaded epoch position from {state_file}: "
                             f"epoch {start_epoch + 1}/{hp.max_epochs}, step {steps_done_in_ep} within epoch")
            except Exception as e:
                logging.warning(f"[train] Could not read {state_file} ({e}), falling back to estimate.")
        if start_epoch == 0 and steps_done_in_ep == 0:
            # Fallback to estimation for checkpoints with neither source of truth
            estimated_batch_size = hp.max_tokens // 50
            if estimated_batch_size <= 0:
                estimated_batch_size = 64
            steps_per_epoch = len(dataset.pairs) // estimated_batch_size
            if steps_per_epoch > 0:
                start_epoch      = global_step // steps_per_epoch
                steps_done_in_ep = global_step %  steps_per_epoch
            logging.info(f"[train] Epoch position (estimated): epoch {start_epoch + 1}/{hp.max_epochs}, "
                         f"step {steps_done_in_ep}/{steps_per_epoch} within epoch")

    # Guard: already finished all epochs
    if start_epoch >= hp.max_epochs:
        logging.info(f"[train] All {hp.max_epochs} epochs already completed at step {global_step}. Nothing to do.")
        return

    # ── 6. Training loop ─────────────────────────────────────────────────────
    deadline = (time.time() + max_minutes * 60) if max_minutes else None

    def _save_now(epoch_i: int, step_in_epoch: int) -> str:
        """Write a step checkpoint (+ legacy state.txt) at the current global_step."""
        ckpt_path = os.path.join(hp.ckpt_dir, f"step_{global_step}.npz")
        model.save(ckpt_path, optim=optim,
                   meta={"epoch": epoch_i, "step_in_epoch": step_in_epoch})
        # Legacy side file, kept so older tooling / notebooks that read state.txt
        # keep working.  The checkpoint's own metadata is authoritative.
        with open(os.path.join(hp.ckpt_dir, "state.txt"), "w") as f:
            f.write(f"{epoch_i},{step_in_epoch}")
        prune_step_checkpoints(hp.ckpt_dir, hp.keep_last)
        return ckpt_path

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
        # Cumulative counters for the epoch, used for the throughput / ETA estimate.
        ep_steps_total  = 0
        ep_pairs_total  = 0
        t0_epoch        = time.time()
        t0 = time.time()

        batch_iterator = dataset.iterate(
            max_tokens=hp.max_tokens, 
            shuffle=True,
            seed=(42 + epoch)
        )
        if skip_batches > 0:
            logging.info(f"Skipping first {skip_batches} batches to resume mid-epoch...")
            batch_iterator = itertools.islice(batch_iterator, skip_batches, None)

        current_step_in_epoch = skip_batches

        for batch_idx, batch in enumerate(batch_iterator):
            # ── Session budget: stop cleanly and leave a checkpoint ──────────
            if deadline is not None and time.time() >= deadline and global_step > 0:
                path = _save_now(epoch, current_step_in_epoch)
                logging.info(
                    f"[train] Session budget of {max_minutes:g} minutes reached at step "
                    f"{global_step} (epoch {epoch + 1}, batch {current_step_in_epoch}). "
                    f"Saved {os.path.basename(path)} — re-run the same command to resume."
                )
                return

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
            ep_steps_total += 1
            ep_pairs_total += int(X.shape[0])
            global_step  += 1

            # ── Logging ───────────────────────────────────────────────────────
            if global_step % hp.log_every == 0:
                elapsed   = time.time() - t0 + 1e-9
                tok_per_s = epoch_tokens / elapsed
                eps_i     = DecoderLSTM._teacher_forcing_prob(global_step, hp.k, hp.min_tf)
                avg_loss  = epoch_loss / epoch_steps

                # Estimate the length of a full epoch from what we have actually
                # measured, so a short session can be planned rather than guessed.
                eta_txt = ""
                sec_per_step = (time.time() - t0_epoch) / max(1, ep_steps_total)
                pairs_per_batch = ep_pairs_total / max(1, ep_steps_total)
                if pairs_per_batch > 0:
                    est_steps = len(dataset.pairs) / pairs_per_batch
                    hours = est_steps * sec_per_step / 3600.0
                    eta_txt = f" | ~{sec_per_step:.2f}s/step, epoch≈{hours:.1f}h"

                log_msg = (
                    f"ep {epoch+1}/{hp.max_epochs} | "
                    f"step {global_step:7d} | "
                    f"loss={avg_loss:.4f} | "
                    f"lr={hp.lr:.2e} | "
                    f"gnorm={grad_norm:.3f} | "
                    f"ε={eps_i:.3f} | "
                    f"tok/s={tok_per_s:,.0f}{eta_txt}"
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
                _save_now(epoch, current_step_in_epoch)

            if max_steps and global_step >= max_steps:
                logging.info(f"[train] Reached max_steps={max_steps}. Stopping.")
                return

        # End of epoch
        final_ckpt = os.path.join(hp.ckpt_dir, f"epoch_{epoch + 1}.npz")
        model.save(final_ckpt, optim=optim, meta={"epoch": epoch + 1, "step_in_epoch": 0})
        with open(os.path.join(hp.ckpt_dir, "state.txt"), "w") as f:
            f.write(f"{epoch + 1},0")
        logging.info(f"[train] Epoch {epoch + 1} complete in "
                     f"{(time.time() - t0_epoch) / 3600.0:.2f} h "
                     f"({ep_steps_total:,} steps, {ep_pairs_total:,} pairs)")


def main_train():
    parser = argparse.ArgumentParser(description="Train the Seq2Seq NMT model")
    parser.add_argument("--preset", type=str, default="gpu", choices=sorted(PRESETS),
                        help="Named hyperparameter bundle (default: gpu = full-size model)")
    parser.add_argument("--max-steps", type=int, default=None, help="Stop after N steps")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="Stop CLEANLY after N wall-clock minutes (checkpoints first). "
                             "Use this to fit training into a short session.")
    parser.add_argument("--resume",    type=str, default=None, help="Path to checkpoint .npz")
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--max-tokens", type=int,   default=None)
    parser.add_argument("--min-tf",     type=float, default=None, help="Minimum teacher-forcing ratio floor")
    parser.add_argument("--k",          type=float, default=None, help="Inverse-sigmoid teacher-forcing decay constant (smaller = anneal faster)")
    parser.add_argument("--save-every", type=int,   default=None, help="Checkpoint every N steps")
    parser.add_argument("--log-every",  type=int,   default=None, help="Log every N steps")
    parser.add_argument("--keep-last",  type=int,   default=None,
                        help="After each save, delete older step_*.npz beyond the newest N")
    parser.add_argument("--max-pairs",  type=int,   default=None,
                        help="Cap the training set to an evenly-strided sample of N pairs")
    args = parser.parse_args()

    # Start from the preset, then apply only the flags the user actually passed.
    hp = apply_preset(cfg, args.preset)
    explicit = {
        "lr":         args.lr,
        "max_tokens": args.max_tokens,
        "min_tf":     args.min_tf,
        "k":          args.k,
        "save_every": args.save_every,
        "log_every":  args.log_every,
        "keep_last":  args.keep_last,
        "max_pairs":  args.max_pairs,
    }
    overrides = {k: v for k, v in explicit.items() if v is not None}
    if overrides:
        import dataclasses
        hp = dataclasses.replace(hp, **overrides)

    train(hp=hp, max_steps=args.max_steps, resume=args.resume,
          max_minutes=args.max_minutes)
