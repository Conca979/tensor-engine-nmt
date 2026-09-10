"""
init_weights.py — Weight initialisation strategies.

All initialisers return float32 arrays and follow the row-vector convention:
  weight shape is (fan_in, fan_out).

Strategies used:
  - Xavier/Glorot uniform  → W_ih (input-to-hidden) — keeps activation variance ≈ 1
  - Orthogonal             → W_hh (hidden-to-hidden) — prevents vanishing/exploding in BPTT
  - Xavier uniform         → embedding tables, output projection Wy
  - Zeros                  → all biases
  - +1 on forget-gate bias → LSTM trick: start remembering everything
"""
import numpy as np
from .backend import xp, f32


def glorot_uniform(fan_in: int, fan_out: int) -> "xp.ndarray":
    """
    Glorot/Xavier uniform initialisation.
    U[-limit, limit]  where  limit = sqrt(6 / (fan_in + fan_out)).
    Shape: (fan_in, fan_out).
    """
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    W = np.random.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)
    return xp.asarray(W)


def orthogonal(rows: int, cols: int, gain: float = 1.0) -> "xp.ndarray":
    """
    Orthogonal initialisation via QR decomposition of a random normal matrix.
    If rows < cols, the result is a rectangular matrix with orthonormal rows.
    Shape: (rows, cols).

    Used for W_hh to give norm-preserving hidden-to-hidden transforms.
    """
    flat = np.random.randn(max(rows, cols), max(rows, cols)).astype(np.float32)
    Q, R = np.linalg.qr(flat)
    # Make the decomposition unique (ensure det > 0 to match PyTorch convention)
    d = np.diag(R)
    Q *= np.sign(d)
    W = Q[:rows, :cols] * gain
    return xp.asarray(W.astype(np.float32))


def xavier_uniform(fan_in: int, fan_out: int) -> "xp.ndarray":
    """Alias for glorot_uniform — used for embedding tables and Wy."""
    return glorot_uniform(fan_in, fan_out)


def zeros(*shape) -> "xp.ndarray":
    """Float32 zero tensor."""
    return xp.zeros(shape, dtype=f32)


def ones(*shape) -> "xp.ndarray":
    """Float32 ones tensor."""
    return xp.ones(shape, dtype=f32)


def init_lstm_weights(input_dim: int, hidden_dim: int):
    """
    Initialise one complete LSTM layer's weights.

    Returns:
        W_ih : (input_dim, 4*hidden_dim)  — Glorot uniform
        W_hh : (hidden_dim, 4*hidden_dim) — Orthogonal
        b    : (4*hidden_dim,)             — zeros, with forget-gate bias = +1
    """
    W_ih = glorot_uniform(input_dim, 4 * hidden_dim)
    W_hh = orthogonal(hidden_dim, 4 * hidden_dim)
    b = zeros(4 * hidden_dim)
    # Classic LSTM trick: initialise the forget-gate bias to +1
    # Gates are ordered [f, i, g, o] → forget gate occupies b[0:hidden_dim]
    b_np = xp.asnumpy(b) if hasattr(xp, 'asnumpy') else np.asarray(b)
    b_np[:hidden_dim] = 1.0
    b = xp.asarray(b_np.astype(np.float32))
    return W_ih, W_hh, b
