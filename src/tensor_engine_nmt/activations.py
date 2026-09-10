"""
activations.py — Numerically stable activation functions.

All functions:
  - accept and return float32 arrays of arbitrary shape
  - never upcast to float64
  - are safe for large-magnitude inputs
"""
from .backend import xp, f32


def sigmoid(x):
    """
    Element-wise sigmoid: σ(x) = 1 / (1 + exp(-x)).

    Safe for large |x|: splits on sign to avoid overflow in exp.
    Shape: same as input.
    """
    x = x.astype(f32)
    pos = x >= 0
    result = xp.empty_like(x)
    # For x >= 0: σ(x) = 1 / (1 + e^{-x})
    result[pos] = 1.0 / (1.0 + xp.exp(-x[pos]))
    # For x < 0: σ(x) = e^x / (1 + e^x)  — avoids exp(-x) overflow
    exp_x = xp.exp(x[~pos])
    result[~pos] = exp_x / (1.0 + exp_x)
    return result


def sigmoid_deriv(s):
    """
    Derivative of sigmoid given its OUTPUT s = σ(x).
    dσ/dx = s * (1 - s).
    Shape: same as s.
    """
    return s * (1.0 - s)


def tanh(x):
    """Element-wise tanh. Shape: same as input."""
    return xp.tanh(x.astype(f32))


def tanh_deriv(t):
    """
    Derivative of tanh given its OUTPUT t = tanh(x).
    d(tanh)/dx = 1 - t^2.
    Shape: same as t.
    """
    return (1.0 - t * t).astype(f32)


def softmax(x, axis: int = -1):
    """
    Numerically stable softmax along `axis`.

    Subtracts max before exp to prevent overflow.
    Result rows sum to 1.0 along `axis`.
    Shape: same as input.
    """
    x = x.astype(f32)
    x_max = x.max(axis=axis, keepdims=True)
    ex = xp.exp(x - x_max)
    return ex / ex.sum(axis=axis, keepdims=True)


def log_softmax(x, axis: int = -1):
    """
    Numerically stable log-softmax: log(softmax(x)).
    Used for computing cross-entropy without explicit softmax.
    Shape: same as input.
    """
    x = x.astype(f32)
    x_max = x.max(axis=axis, keepdims=True)
    shifted = x - x_max
    log_sum_exp = xp.log(xp.exp(shifted).sum(axis=axis, keepdims=True))
    return shifted - log_sum_exp
