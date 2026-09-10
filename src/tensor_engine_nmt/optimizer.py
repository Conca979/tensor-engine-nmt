"""
optimizer.py — Adam optimizer with global gradient norm clipping.

Algorithm:
  1. Compute global gradient norm across ALL parameters
  2. If norm > clip_norm, scale all gradients down
  3. Apply Adam update (bias-corrected first and second moment estimates)

Reference: Kingma & Ba (2015), "Adam: A Method for Stochastic Optimization"
"""
import numpy as np
from .backend import xp, f32
from .config import cfg


class Adam:
    """
    Adam optimizer with gradient clipping.

    Parameters
    ----------
    model   : Seq2Seq — provides .parameters() → [(param, grad), ...]
    hp      : HParams
    """

    def __init__(self, model, hp=cfg):
        self.model     = model
        self.hp        = hp
        self.t         = 0            # step counter (1-indexed for bias correction)

        # Initialise moment estimates — one pair per parameter tensor
        params = list(model.parameters())
        self.m = [xp.zeros_like(p) for (p, _) in params]   # first moment
        self.v = [xp.zeros_like(p) for (p, _) in params]   # second moment

    def step(self) -> float:
        """
        Perform one Adam update step.

        Returns the gradient norm (before clipping) for logging.
        """
        hp = self.hp
        self.t += 1
        t = self.t

        params_grads = list(self.model.parameters())

        # ── 1. Global gradient norm clip ─────────────────────────────────────
        global_norm_sq = sum(float((g ** 2).sum()) for (_, g) in params_grads)
        global_norm    = float(np.sqrt(global_norm_sq))

        if global_norm > hp.clip_norm:
            scale = hp.clip_norm / (global_norm + 1e-8)
            for _, g in params_grads:
                g *= scale

        # ── 2. Adam update ───────────────────────────────────────────────────
        b1, b2, eps = hp.beta1, hp.beta2, hp.eps_adam
        lr = hp.lr

        for idx, (param, grad) in enumerate(params_grads):
            m = self.m[idx]
            v = self.v[idx]

            m[:] = b1 * m + (1.0 - b1) * grad
            v[:] = b2 * v + (1.0 - b2) * grad ** 2

            # Bias-corrected estimates
            m_hat = m / (1.0 - b1 ** t)
            v_hat = v / (1.0 - b2 ** t)

            param -= lr * m_hat / (xp.sqrt(v_hat) + eps)

        return global_norm
