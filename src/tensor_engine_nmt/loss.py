"""
loss.py — Masked cross-entropy loss (Stage 5).

Implements exactly:
    L = sum(-log(p[b,t, Yout[b,t]]) * mask_f) / sum(mask_f)

where mask = (Yout != PAD) zeros out padding positions.

The backward pass returns δlogits — the entry point for decoder BPTT.

Note: we use the log-sum-exp trick internally and do NOT call softmax
separately in the forward — this is numerically more stable.
"""
import numpy as np
from .backend import xp, f32
from .config import cfg


PAD_ID = cfg.PAD


def forward(
    logits: "xp.ndarray",   # (B, Ty, V)  float32
    Yout:   np.ndarray,     # (B, Ty)     int32
    Ylen:   np.ndarray,     # (B,)        int32  (for reference; mask is derived from Yout)
) -> tuple:
    """
    Compute masked cross-entropy loss.

    Returns
    -------
    loss   : float32 scalar — the mean per-token NLL
    p      : float32 (B, Ty, V) — the softmax probabilities (cached for backward)
    mask_f : float32 (B, Ty) — float mask (1.0 for real tokens, 0.0 for PAD)
    """
    B, Ty, V = logits.shape

    # ── Numerically stable softmax ─────────────────────────────────────────
    logits_max = logits.max(axis=-1, keepdims=True)      # (B, Ty, 1)
    shifted    = logits - logits_max                     # (B, Ty, V)
    ex         = xp.exp(shifted).astype(f32)
    p          = ex / ex.sum(axis=-1, keepdims=True)     # (B, Ty, V)

    # ── Padding mask ───────────────────────────────────────────────────────
    if isinstance(Yout, np.ndarray):
        Yout_xp = xp.asarray(Yout)
    else:
        Yout_xp = Yout
    mask   = (Yout_xp != PAD_ID)                        # (B, Ty) bool
    mask_f = mask.astype(f32)                            # (B, Ty) float32

    # ── Gather probability of the correct token ────────────────────────────
    B_idx  = xp.arange(B)[:, None]                      # (B, 1)
    T_idx  = xp.arange(Ty)[None, :]                     # (1, Ty)
    p_gold = p[B_idx, T_idx, Yout_xp]                   # (B, Ty)

    # Clip to avoid log(0)
    nll  = -xp.log(xp.clip(p_gold, 1e-9, 1.0))         # (B, Ty)
    loss = (nll * mask_f).sum() / (mask_f.sum() + 1e-9)  # scalar

    return float(loss), p, mask_f


def backward(
    p:       "xp.ndarray",   # (B, Ty, V)  softmax probabilities from forward
    Yout:    np.ndarray,     # (B, Ty)     int32
    mask_f:  "xp.ndarray",   # (B, Ty)     float32 mask
) -> "xp.ndarray":
    """
    Compute δL/δlogits.

    The gradient of cross-entropy + softmax is famously clean:
        δlogits[b,t,v] = (p[b,t,v] - 1{v == Yout[b,t]}) * mask_f[b,t] / N_tokens

    Shape: (B, Ty, V)  float32
    """
    B, Ty, V = p.shape
    N_tokens = mask_f.sum() + 1e-9

    if isinstance(Yout, np.ndarray):
        Yout_xp = xp.asarray(Yout)
    else:
        Yout_xp = Yout

    # δlogits = p  (everywhere)
    dlogits = p.copy()                              # (B, Ty, V)

    # Subtract 1 at the gold position
    B_idx = xp.arange(B)[:, None]
    T_idx = xp.arange(Ty)[None, :]
    dlogits[B_idx, T_idx, Yout_xp] -= 1.0          # (B, Ty, V)

    # Apply mask and normalise
    dlogits *= mask_f[:, :, None]                   # zero out PAD positions
    dlogits /= N_tokens                             # (B, Ty, V)

    return dlogits.astype(f32)
