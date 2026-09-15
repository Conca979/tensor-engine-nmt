"""
config.py — Single source of truth for all hyperparameters.

Every number in the entire codebase should trace back here.
Never hard-code a value anywhere else.
"""
from dataclasses import dataclass, field
from pathlib import Path


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

    # ── Training ────────────────────────────────────────────────────────────
    max_tokens: int = 4000   # maximum tokens per batch
    B: int = 64              # default batch size (used for unit testing and fixed-batch evaluation)
    k: float = 17_000.0     # inverse-sigmoid teacher-forcing decay constant
    min_tf: float = 0.7      # minimum teacher-forcing ratio floor (scheduled sampling lower bound)
    max_epochs: int = 30

    # ── Optimiser ───────────────────────────────────────────────────────────
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps_adam: float = 1e-8
    clip_norm: float = 2.0

    # ── Inference ───────────────────────────────────────────────────────────
    beam_width: int = 4
    max_decode_len: int = 30  # safety cap on decoder output — keep == max_len

    # ── Logging / checkpointing ──────────────────────────────────────────────
    log_every: int = 100
    save_every: int = 2000

    # ── Paths ────────────────────────────────────────────────────────────────
    data_dir: str = "PhoMT_dataset"
    bpe_dir: str = "bpe_vocab"
    ckpt_dir: str = "checkpoints"

    # ── BPE training ─────────────────────────────────────────────────────────
    bpe_sample_lines: int = 300_000   # lines sampled per language for BPE training


# Global default instance — import this everywhere.
cfg = HParams()
