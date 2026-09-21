"""
config.py — Single source of truth for all hyperparameters.

Every number in the entire codebase should trace back here.
Never hard-code a value anywhere else.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class HParams:
    # ── Vocabulary ──────────────────────────────────────────────────────────
    V: int = 32_000          # shared BPE vocabulary size
    PAD: int = 0             # padding token id
    UNK: int = 1             # unknown token id
    START: int = 2           # decoder start-of-sequence token id
    END: int = 3             # end-of-sequence token id

    # ── Architecture ────────────────────────────────────────────────────────
    e: int = 512             # embedding dimension
    d: int = 1024            # LSTM hidden size (per layer, one direction)
    L: int = 3               # number of stacked LSTM layers (encoder & decoder)

    # ── Length budget ────────────────────────────────────────────────────────
    # max_len is the SINGLE source of truth for sequence length filtering.
    # Applied at three places:
    #   1. dataset.py  — filters training pairs (both EN and VI sides)
    #   2. evaluate.py — filters eval pairs identically so metrics are comparable
    #   3. inference.py max_decode_len — hard cap on decoder output length
    # Rule: max_decode_len should always equal max_len so inference matches training.
    max_len: int = 30        # max BPE token count for either side (EN or VI)

    # ── Dataset size ────────────────────────────────────────────────────────
    # Cap the training set to an evenly-strided sample of N pairs.  None = all of
    # them.  Useful when the full corpus is more than you can get through: it
    # shrinks the tokenized cache, the epoch length and the per-session cost.
    max_pairs: Optional[int] = None

    # ── Training ────────────────────────────────────────────────────────────
    max_tokens: int = 5000   # maximum tokens per batch
    B: int = 64              # default batch size (used for unit testing and fixed-batch evaluation)
    k: float = 17_000.0     # inverse-sigmoid teacher-forcing decay constant
    min_tf: float = 0.7      # minimum teacher-forcing ratio floor (scheduled sampling lower bound)
    max_epochs: int = 30

    # ── Optimiser ───────────────────────────────────────────────────────────
    lr: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    eps_adam: float = 1e-8
    clip_norm: float = 2.0

    # ── Inference ───────────────────────────────────────────────────────────
    beam_width: int = 4
    max_decode_len: int = 30  # safety cap on decoder output — keep == max_len

    # ── Logging / checkpointing ──────────────────────────────────────────────
    log_every: int = 200
    save_every: int = 4000
    # After each save, keep only the newest N step_*.npz files (None = keep all).
    # Short-session training wants frequent saves; this stops them filling the disk.
    keep_last: Optional[int] = 2

    # ── Paths ────────────────────────────────────────────────────────────────
    data_dir: str = "PhoMT_dataset"
    bpe_dir: str = "bpe_vocab"
    ckpt_dir: str = "checkpoints"

    # ── BPE training ─────────────────────────────────────────────────────────
    bpe_sample_lines: int = 300_000   # lines sampled per language for BPE training


# Global default instance — import this everywhere.
cfg = HParams()


# ── Presets ───────────────────────────────────────────────────────────────────
#
# Named bundles of overrides, selected with `--preset`.  `gpu` is the tuned
# full-size configuration and changes nothing; `laptop` is a deliberately small
# model that a CPU can actually make progress on in short sessions.
#
# The laptop preset deliberately uses its OWN bpe_dir / ckpt_dir so its small
# checkpoints (different shapes) can never be confused with, or auto-resumed
# into, a full-size run.

PRESETS = {
    "gpu": {},
    "laptop": {
        "V": 8_000,              # smaller shared BPE vocabulary (trained on demand)
        "e": 128,                # embedding dimension
        "d": 256,                # LSTM hidden size  (encoder uses d/2 per direction)
        "L": 2,                  # stacked layers
        "max_len": 20,           # shorter sentences only
        "max_decode_len": 20,    # keep == max_len
        "max_tokens": 1_000,     # dynamic batch token budget
        "max_pairs": 150_000,    # evenly-strided sample of the corpus
        "max_epochs": 10,
        "lr": 3e-4,              # a small model tolerates a larger step
        "min_tf": 0.50,          # let the schedule actually anneal
        "k": 6_000.0,
        "clip_norm": 2.0,
        "log_every": 20,
        "save_every": 100,
        "keep_last": 3,
        "bpe_dir": "bpe_vocab_laptop",
        "ckpt_dir": "checkpoints_laptop",
        "bpe_sample_lines": 150_000,
    },
}


def apply_preset(hp: HParams, name: str) -> HParams:
    """Return a copy of `hp` with the named preset's overrides applied."""
    import dataclasses
    if name not in PRESETS:
        raise KeyError(f"Unknown preset {name!r}. Available: {sorted(PRESETS)}")
    return dataclasses.replace(hp, **PRESETS[name])
