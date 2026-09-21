"""
test_short_session.py — the machinery that makes interrupted training safe.

    python test/test_short_session.py
    pytest test/test_short_session.py -v

Short sessions only work if three things hold:
  * a run can be told to stop CLEANLY after a wall-clock budget, leaving a
    checkpoint behind (never relying on the platform to kill it politely);
  * the next run resumes from that exact checkpoint and the exact batch;
  * frequent checkpoints do not fill the disk.

This exercises all three against a 300-line temp corpus, so it is fast and
touches nothing in the real project directories.
"""
import sys
import os
import glob
import shutil
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tensor_engine_nmt.config import HParams, cfg, PRESETS, apply_preset
from tensor_engine_nmt.train import train, prune_step_checkpoints
from tensor_engine_nmt.dataset import PhoMTDataset, _count_lines, BUCKET_BOUNDARIES

REAL_DATA = os.path.join(os.path.dirname(__file__), "..", "PhoMT_dataset")
N_LINES = 300


# ── presets ───────────────────────────────────────────────────────────────────

def test_presets_are_wellformed():
    """Every preset must produce a valid HParams without mutating the default."""
    before = cfg
    for name in PRESETS:
        hp = apply_preset(cfg, name)
        assert hp.d % 2 == 0, f"preset {name}: d must be even for the BiLSTM concat"
        assert hp.max_decode_len == hp.max_len, (
            f"preset {name}: max_decode_len ({hp.max_decode_len}) must equal "
            f"max_len ({hp.max_len}) so inference matches training"
        )
        assert hp.save_every > 0 and hp.log_every > 0
        assert cfg is before, "apply_preset mutated the global config"

    lap = apply_preset(cfg, "laptop")
    gpu = apply_preset(cfg, "gpu")
    assert lap.d < gpu.d and lap.L <= gpu.L and lap.V < gpu.V, "laptop must be smaller"
    assert lap.ckpt_dir != gpu.ckpt_dir, (
        "the laptop preset needs its own checkpoint dir so its differently-shaped "
        "weights can never be auto-resumed into a full-size run"
    )
    assert lap.bpe_dir != gpu.bpe_dir

    try:
        apply_preset(cfg, "nonsense")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown preset should raise KeyError")
    print("  [OK ] presets are well formed and isolated")


# ── checkpoint pruning ────────────────────────────────────────────────────────

def test_prune_keeps_newest_and_spares_other_files():
    def body(tmp):
        for step in (100, 200, 900, 1000, 2000):
            open(os.path.join(tmp, f"step_{step}.npz"), "wb").close()
        # These must never be pruned: optim_state.npz belongs to the newest
        # step checkpoint, and epoch_*.npz are deliberate long-lived snapshots.
        for other in ("optim_state.npz", "epoch_3.npz", "state.txt"):
            open(os.path.join(tmp, other), "wb").close()

        prune_step_checkpoints(tmp, 2)

        left = sorted(os.path.basename(p) for p in glob.glob(os.path.join(tmp, "step_*.npz")))
        assert left == ["step_1000.npz", "step_2000.npz"], left
        for other in ("optim_state.npz", "epoch_3.npz", "state.txt"):
            assert os.path.exists(os.path.join(tmp, other)), f"{other} was pruned"

        # Sorting must be numeric, not lexicographic (step_900 < step_1000).
        prune_step_checkpoints(tmp, 0)          # no-op
        assert len(glob.glob(os.path.join(tmp, "step_*.npz"))) == 2
        print("  [OK ] prune keeps the numerically newest N and spares optim/epoch files")

    _run(body)


# ── dataset capping ───────────────────────────────────────────────────────────

def test_max_pairs_stride_sampling_and_cache_key():
    def body(tmp):
        _make_tiny_corpus(tmp)
        from tensor_engine_nmt.bpe import load_or_train_bpe
        bpe = load_or_train_bpe(os.path.join(tmp, "bpe"), tmp, 300, N_LINES)

        full = PhoMTDataset(bpe, tmp, "train", max_len=30, max_pairs=None)
        capped = PhoMTDataset(bpe, tmp, "train", max_len=30, max_pairs=50)

        assert len(capped.pairs) <= 50, f"cap not enforced: {len(capped.pairs)}"
        assert len(capped.pairs) > 0, "cap produced an empty dataset"
        assert "N50" in os.path.basename(capped.cache_path), capped.cache_path
        assert full.cache_path != capped.cache_path, "cap must be part of the cache key"

        # A capped dataset must still be spread across the file, not the head of it:
        # stride sampling should not simply equal the first N pairs.
        assert capped.pairs != full.pairs[:len(capped.pairs)], (
            "cap looks like a head-truncation rather than a stride sample"
        )
        print(f"  [OK ] max_pairs caps to {len(capped.pairs)} pairs of {len(full.pairs)} "
              f"and changes the cache key")

    _run(body)


def test_count_lines():
    def body(tmp):
        p = os.path.join(tmp, "x.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(f"line {i}" for i in range(1234)) + "\n")
        assert _count_lines(p) == 1234
        print("  [OK ] line counting used by the stride sampler")

    _run(body)


# ── the actual short-session loop ─────────────────────────────────────────────

def test_time_budget_stops_cleanly_then_resumes():
    def body(tmp):
        _make_tiny_corpus(tmp)
        hp = HParams(
            V=300, e=16, d=32, L=1, B=4,
            max_tokens=200, max_len=30, max_decode_len=30,
            k=2000.0, lr=1e-3, clip_norm=2.0, min_tf=0.5,
            max_epochs=3, beam_width=2,
            log_every=2, save_every=1, keep_last=2,
            data_dir=tmp, bpe_dir=os.path.join(tmp, "bpe"),
            ckpt_dir=os.path.join(tmp, "ckpt"), bpe_sample_lines=N_LINES,
        )

        # ── Session 1: a tiny wall-clock budget ──────────────────────────────
        train(hp=hp, max_minutes=0.01)          # 0.6 s
        ckpts = glob.glob(os.path.join(hp.ckpt_dir, "step_*.npz"))
        assert ckpts, "the budget expired without writing a checkpoint"
        assert not glob.glob(os.path.join(hp.ckpt_dir, "epoch_*.npz")), (
            "the epoch finished — the budget was too generous for this assertion"
        )
        first_steps = sorted(_step(c) for c in ckpts)
        print(f"  [OK ] session 1 stopped cleanly at step {first_steps[-1]} "
              f"(checkpoints kept: {first_steps})")

        # keep_last=2 must have held the checkpoint count down
        assert len(first_steps) <= 2, first_steps

        # ── Session 2: identical call must resume, not restart ───────────────
        train(hp=hp, max_minutes=0.01)
        second_steps = sorted(_step(c) for c in
                              glob.glob(os.path.join(hp.ckpt_dir, "step_*.npz")))
        assert second_steps, "session 2 wrote no checkpoint"
        assert max(second_steps) > max(first_steps), (
            f"session 2 restarted instead of resuming: {first_steps} -> {second_steps}"
        )
        print(f"  [OK ] session 2 resumed and continued to step {second_steps[-1]}")

    _run(body)


def _step(path):
    return int(os.path.basename(path).split("_")[1].replace(".npz", ""))


# ── helpers ───────────────────────────────────────────────────────────────────

def _run(body):
    tmp = tempfile.mkdtemp(prefix="nmt_short_")
    try:
        body(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _make_tiny_corpus(root: str) -> None:
    os.makedirs(os.path.join(root, "train"), exist_ok=True)
    for lang in ("en", "vi"):
        src = os.path.join(REAL_DATA, "train", f"train.{lang}")
        dst = os.path.join(root, "train", f"train.{lang}")
        with open(src, "r", encoding="utf-8") as fin, \
             open(dst, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if i >= N_LINES:
                    break
                fout.write(line)


if __name__ == "__main__":
    print("Short-session checks\n" + "-" * 60)
    test_presets_are_wellformed()
    test_count_lines()
    test_prune_keeps_newest_and_spares_other_files()
    test_max_pairs_stride_sampling_and_cache_key()
    test_time_budget_stops_cleanly_then_resumes()
    print("-" * 60)
    print("All short-session checks passed.")
