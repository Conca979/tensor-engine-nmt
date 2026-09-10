"""
backend.py — Transparent NumPy / CuPy abstraction.

Import `xp` from here instead of importing numpy directly.
Swapping CPU ↔ GPU requires zero changes to any other module.
"""
try:
    import cupy as xp          # type: ignore[import]
    xp.cuda.Device(0).use()
    BACKEND = "cupy"
except (ImportError, Exception):
    import numpy as xp         # type: ignore[assignment]
    BACKEND = "numpy"

import numpy as np             # always available for type hints / CPU ops

__all__ = ["xp", "np", "BACKEND", "f32", "i32"]

# Convenient dtype aliases — always float32 / int32, never float64 / int64
f32 = xp.float32
i32 = xp.int32


def to_xp(arr):
    """Move a numpy array to the active backend (no-op if already there)."""
    if BACKEND == "cupy" and isinstance(arr, np.ndarray):
        return xp.asarray(arr)
    return arr


def to_np(arr) -> np.ndarray:
    """Move an array to CPU numpy (for I/O, BPE, etc.)."""
    if BACKEND == "cupy":
        return xp.asnumpy(arr)
    return arr
