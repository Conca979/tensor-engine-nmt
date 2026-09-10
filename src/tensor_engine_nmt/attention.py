"""
attention.py — Luong "general" attention mechanism.

Implements Stage 4 Step 3 from test/unidirectional_pipeline.md

Score function (Luong "general"):
    temp    = s_t @ W_a               (B, d)
    e_t[j]  = dot(temp[b], H[b,j,:]) for each encoder position j
    alpha_t = softmax(masked(e_t))    (B, Tx)
    z_t     = sum_j alpha_t[:,j] * H[:,j,:]   (B, d)

The attention module also handles the padding mask so that padded
encoder positions receive attention weight ≈ 0.
"""
import numpy as np
from .backend import xp, f32
from .config import cfg
from .activations import softmax
from .init_weights import glorot_uniform, zeros


class LuongAttention:
    """
    Luong "general" attention.

    Parameters
    ----------
    d : hidden dimension (from cfg.d)
    """

    def __init__(self, hp=cfg):
        self.hp = hp
        d = hp.d

        # W_a : (d, d)  — the bilinear weight matrix
        self.W_a  = glorot_uniform(d, d)
        self.dW_a = zeros(d, d)

        self._cache = None

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        s_t: "xp.ndarray",     # (B, d)  decoder top-layer hidden at step t
        H:   "xp.ndarray",     # (B, Tx, d) full encoder memory
        Xlen: np.ndarray,      # (B,) true encoder lengths (int32, CPU OK)
    ) -> tuple:
        """
        Compute context vector z_t for one decoder timestep.

        Returns
        -------
        z_t     : float32 (B, d)   — weighted encoder summary
        alpha_t : float32 (B, Tx)  — attention weights (rows sum to 1)
        cache   : dict for backward
        """
        B, Tx, d = H.shape

        # Score: e_t[b,j] = s_t[b] · (W_a · H[b,j])
        temp = s_t @ self.W_a                                   # (B, d)
        # einsum 'bd,btd->bt': for each (b,j): dot(temp[b], H[b,j,:])
        e_t = xp.einsum("bd,btd->bt", temp, H).astype(f32)     # (B, Tx)

        # Mask padded positions → -1e9 (→ 0 after softmax)
        # Xlen may be a numpy array; convert the comparison result to xp
        j_idx = xp.arange(Tx, dtype=xp.int32)                  # (Tx,)
        if isinstance(Xlen, np.ndarray):
            Xlen_xp = xp.asarray(Xlen)
        else:
            Xlen_xp = Xlen
        M = (j_idx[None, :] < Xlen_xp[:, None]).astype(f32)    # (B, Tx) float
        e_t = e_t + (M - 1.0) * 1e9                            # pad → ≈ -1e9

        alpha_t = softmax(e_t, axis=1)                          # (B, Tx)

        # Context vector: weighted sum of encoder states
        z_t = xp.einsum("bt,btd->bd", alpha_t, H).astype(f32)  # (B, d)

        self._cache = {
            "s_t": s_t,
            "H": H,
            "Xlen": Xlen,
            "temp": temp,
            "e_t": e_t,
            "alpha_t": alpha_t,
            "M": M,
            "B": B,
            "Tx": Tx,
        }

        return z_t, alpha_t

    # ── Backward ─────────────────────────────────────────────────────────────

    def backward(
        self,
        dz_t:    "xp.ndarray",   # (B, d) gradient into context vector
        ds_t_in: "xp.ndarray",   # (B, d) gradient into s_t from other paths
    ) -> tuple:
        """
        Backprop through the attention computation.

        Returns
        -------
        ds_t  : float32 (B, d)     — gradient into decoder hidden s_t
        dH_t  : float32 (B, Tx, d) — gradient into encoder memory H at this step
        """
        cache   = self._cache
        alpha_t = cache["alpha_t"]
        temp    = cache["temp"]
        s_t     = cache["s_t"]
        H       = cache["H"]
        B, Tx   = cache["B"], cache["Tx"]

        # ── Backward through z_t = einsum('bt,btd->bd', alpha_t, H) ─────────
        # dz_t : (B, d)
        # ∂L/∂alpha_t = einsum('bd,btd->bt', dz_t, H)
        dalpha_t = xp.einsum("bd,btd->bt", dz_t, H).astype(f32)   # (B, Tx)
        # ∂L/∂H from this path = einsum('bt,bd->btd', alpha_t, dz_t)
        dH = xp.einsum("bt,bd->btd", alpha_t, dz_t).astype(f32)   # (B, Tx, d)

        # ── Backward through alpha_t = softmax(e_t) ──────────────────────────
        # Softmax Jacobian shortcut: if s = softmax(x), then
        # dx = s * (dy - sum(s*dy, keepdims))
        sum_term = (alpha_t * dalpha_t).sum(axis=1, keepdims=True)  # (B, 1)
        de_t = alpha_t * (dalpha_t - sum_term)                      # (B, Tx)
        # Gradient is zero for masked positions (they have alpha≈0, de≈0 anyway)
        # but apply mask explicitly for cleanliness
        de_t = de_t * cache["M"]                                    # (B, Tx)

        # ── Backward through e_t = einsum('bd,btd->bt', temp, H) ────────────
        # ∂L/∂temp  = einsum('bt,btd->bd', de_t, H)
        dtemp = xp.einsum("bt,btd->bd", de_t, H).astype(f32)       # (B, d)
        # ∂L/∂H from score path = einsum('bd,bt->btd', temp, de_t)
        dH += xp.einsum("bd,bt->btd", temp, de_t).astype(f32)      # (B, Tx, d)

        # ── Backward through temp = s_t @ W_a ────────────────────────────────
        # ∂L/∂W_a = s_t.T @ dtemp   shape (d,d)
        self.dW_a += s_t.T @ dtemp
        # ∂L/∂s_t from attention path
        ds_t_attn = dtemp @ self.W_a.T                              # (B, d)

        # Total gradient into s_t
        ds_t = ds_t_in + ds_t_attn                                  # (B, d)

        return ds_t, dH

    # ── Parameter access ────────────────────────────────────────────────────

    def parameters(self):
        yield self.W_a, self.dW_a

    def zero_grad(self):
        self.dW_a[:] = 0.0
