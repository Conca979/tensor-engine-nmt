"""
bench.py — measure real training throughput before committing to a long run.

    tensor-engine-nmt bench [--preset gpu|laptop] [--steps 10] [--max-tokens N] [--max-pairs N]

Why this exists: "how long will training take on my machine" is the first
question anyone asks, and the honest answer depends on the machine, the preset
and the batch budget — not on a number in a README.  This runs a handful of real
forward/backward/optimiser steps on real batches and reports the measured rate,
the seconds per step and a projected time per epoch.

It deliberately does NOT load or write checkpoints, so it is safe to run at any
time and costs nothing but a few minutes.
"""
import os
import time

import numpy as np

from .config import cfg, apply_preset
from .bpe import load_or_train_bpe
from .dataset import PhoMTDataset
from .model import Seq2Seq
from .optimizer import Adam


def benchmark(
    hp=cfg,
    steps: int = 10,
    max_tokens: int = None,
    max_pairs: int = None,
) -> dict:
    """Run `steps` real training steps and report measured throughput."""
    import dataclasses
    overrides = {}
    if max_tokens is not None:
        overrides["max_tokens"] = max_tokens
    if max_pairs is not None:
        overrides["max_pairs"] = max_pairs
    if overrides:
        hp = dataclasses.replace(hp, **overrides)

    print("=" * 72)
    print("  Throughput benchmark")
    print("=" * 72)
    print(f"  Architecture : d={hp.d}  L={hp.L}  e={hp.e}  V={hp.V}  max_len={hp.max_len}")
    print(f"  Batching     : max_tokens={hp.max_tokens}  max_pairs={hp.max_pairs}")
    print(f"  Directories  : bpe_dir={hp.bpe_dir}  data_dir={hp.data_dir}")

    t0 = time.time()
    bpe = load_or_train_bpe(
        bpe_dir=hp.bpe_dir, data_dir=hp.data_dir,
        target_vocab=hp.V, sample_lines=hp.bpe_sample_lines,
    )
    print(f"  BPE          : {len(bpe.token2id):,} tokens ({time.time() - t0:.1f}s)")
    if len(bpe.token2id) > hp.V:
        raise ValueError(
            f"hp.V={hp.V:,} < BPE vocabulary {len(bpe.token2id):,} tokens in {hp.bpe_dir}. "
            f"Pick a preset whose bpe_dir matches its V."
        )

    t0 = time.time()
    dataset = PhoMTDataset(bpe, data_dir=hp.data_dir, split="train",
                           max_len=hp.max_len, max_pairs=hp.max_pairs)
    print(f"  Dataset      : {len(dataset.pairs):,} pairs ({time.time() - t0:.1f}s)")

    from .backend import xp, BACKEND
    model = Seq2Seq(hp)
    optim = Adam(model, hp)
    print(f"  Parameters   : {model.param_count():,}")
    print(f"  Backend      : {BACKEND}")
    print("-" * 72)

    if not dataset.pairs:
        raise RuntimeError("Dataset is empty — check max_len / max_pairs.")

    losses, times, tokens_seen, pairs_seen = [], [], 0, 0

    for i, batch in enumerate(dataset.iterate(max_tokens=hp.max_tokens, shuffle=True, seed=42)):
        if i >= steps:
            break
        model.zero_grad()
        t = time.time()
        loss_val, _ = model.forward(
            batch["X"], batch["Xlen"], batch["Yin"], batch["Yout"], batch["Ylen"], i
        )
        model.backward()
        optim.step()
        times.append(time.time() - t)

        pairs_seen += int(batch["X"].shape[0])
        tokens_seen += int(batch["Yin"].shape[0] * batch["Yin"].shape[1])
        losses.append(loss_val)

        if i == 0:
            print(f"  batch 1: B={batch['X'].shape[0]} Tx={batch['X'].shape[1]} "
                  f"Ty={batch['Yin'].shape[1]}  loss={loss_val:.4f}  {times[-1]:.2f}s")

    if not times:
        raise RuntimeError("No batches were produced — the dataset iterator yielded nothing.")

    # Warm-up step excluded: the first step pays for lazy BLAS/allocator setup.
    steady = times[1:] if len(times) > 1 else times
    sec_per_step = float(np.mean(steady))
    tok_per_s = tokens_seen / float(np.sum(times)) if tokens_seen else 0.0
    pairs_per_batch = pairs_seen / len(times)
    steps_per_epoch = len(dataset.pairs) / pairs_per_batch
    hours_per_epoch = steps_per_epoch * sec_per_step / 3600.0

    print("-" * 72)
    print(f"  Measured over {len(times)} steps ({pairs_seen:,} pairs, {tokens_seen:,} tokens)")
    print(f"  Steady-state      : {sec_per_step:.3f} s/step   ({1 / sec_per_step:.2f} steps/s)")
    print(f"  Throughput        : {tok_per_s:,.0f} tok/s")
    print(f"  Pairs per batch   : {pairs_per_batch:.1f}")
    print(f"  Steps per epoch   : {steps_per_epoch:,.0f}")
    print()
    print(f"  >>> ESTIMATED TIME PER EPOCH: {hours_per_epoch:.2f} hours "
          f"({hours_per_epoch / 24:.2f} days)")
    print()
    for n in (1, 2, 5, 10):
        total_h = hours_per_epoch * n
        print(f"      {n:2d} epoch(s): {total_h:8.1f} h  =  {total_h / 24:6.1f} days")
    print()
    print("  Plan a session with --max-minutes so the run stops cleanly:")
    suggested = max(10, int((2 * 3600) / sec_per_step))   # 2 hours of steps
    print(f"      --max-minutes 120 --save-every {max(10, suggested // 12)} --keep-last 3")
    print("=" * 72)

    return {
        "sec_per_step":    sec_per_step,
        "tok_per_s":       tok_per_s,
        "steps_per_epoch": steps_per_epoch,
        "hours_per_epoch": hours_per_epoch,
        "param_count":     model.param_count(),
        "pairs":           len(dataset.pairs),
    }


def main_bench():
    import argparse
    parser = argparse.ArgumentParser(
        description="Measure training throughput so you can size a session"
    )
    parser.add_argument("--preset", type=str, default="gpu", choices=["gpu", "laptop"])
    parser.add_argument("--steps", type=int, default=10, help="Measured steps (default 10)")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-pairs",  type=int, default=None)
    args = parser.parse_args()

    hp = apply_preset(cfg, args.preset)
    benchmark(hp, steps=args.steps, max_tokens=args.max_tokens, max_pairs=args.max_pairs)
