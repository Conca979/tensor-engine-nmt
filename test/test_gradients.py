"""
test_gradients.py — gradient correctness checks.

Run with either:
    python test/test_gradients.py
    pytest test/test_gradients.py -v

WHY THIS IS NOT A PLAIN LOSS-BASED FINITE-DIFFERENCE CHECK
----------------------------------------------------------
The obvious check — perturb a weight, difference the real loss, compare to the
analytic gradient — does not work in float32 at this model size.  The loss is
~2.07, whose float32 ULP is ~2.4e-7.  With EPS=1e-3 the central difference
    (L(+ε) − L(−ε)) / 2ε
is therefore quantised in steps of ~1.2e-4 — LARGER than most of the gradients
being measured.  That is why the previous version of this file reported
max_rel_err ≈ 1.0 on 21 of 25 tensors: it was measuring its own rounding noise,
and could not distinguish a correct backward pass from a broken one.

So instead we use a synthetic scalar objective
    J = <upstream, logits>,  upstream ~ N(0, 4)
with fixed random upstream gradients.  J and dJ/dθ are both O(1), the FD floor
drops to ~1e-4, and every remaining discrepancy is real.  Anything that breaks
the backward chain (misplaced handoff gradient, stale attention cache, missing
padding mask, wrong activation derivative) shows up here by a wide margin —
the bugs this file was written to catch produced relative errors of 0.4 to 1.7.

The checks use a batch with REAL padding (Xlen < Tx, Ylen < Ty) and Ty > 1,
because both of those conditions are needed to exercise the handoff, the
padding mask and the per-step attention cache.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tensor_engine_nmt.config import HParams
from tensor_engine_nmt.backend import xp

# ── Tiny config ───────────────────────────────────────────────────────────────
hp = HParams(
    V=8, e=4, d=6, L=2, B=2,
    k=100.0,
    lr=1e-3, beta1=0.9, beta2=0.999, eps_adam=1e-8, clip_norm=5.0,
    max_epochs=1, beam_width=2, max_decode_len=10,
    log_every=1, save_every=9999,
    data_dir="PhoMT_dataset", bpe_dir="bpe_vocab", ckpt_dir="checkpoints",
    bpe_sample_lines=1000,
)

# ── Batch: Tx=4 with row 1 padded (Xlen=3); Ty=3 with row 1 padded (Ylen=2) ───
Tx, Ty = 4, 3
X    = np.array([[4, 5, 6, 3], [4, 5, 3, 0]], dtype=np.int32)
Xlen = np.array([4, 3], dtype=np.int32)
Yin  = np.array([[2, 4, 5], [2, 6, 0]], dtype=np.int32)
Yout = np.array([[4, 5, 3], [6, 3, 0]], dtype=np.int32)
Ylen = np.array([3, 2], dtype=np.int32)

EPS       = 2e-2
TOL       = 3e-2      # relative tolerance on entries above the noise floor
NOISE_ULPS = 20.0     # ULP safety factor, see note above


def _param_names(model):
    """Names in exactly the order Seq2Seq.parameters() yields them."""
    names = ["enc.Ex"]
    for l in range(hp.L):
        names += [f"enc.W_ih_fwd[{l}]", f"enc.W_hh_fwd[{l}]", f"enc.b_fwd[{l}]",
                  f"enc.W_ih_bwd[{l}]", f"enc.W_hh_bwd[{l}]", f"enc.b_bwd[{l}]"]
    names += ["dec.Ey"]
    for l in range(hp.L):
        names += [f"dec.W_ih[{l}]", f"dec.W_hh[{l}]", f"dec.b[{l}]"]
    names += ["attn.W_a", "dec.W_c", "dec.b_c", "dec.Wy", "dec.by"]
    return names


def _reinit(model, seed=1234, scale=0.6):
    """Default init is nearly degenerate; spread the weights so gradients are large."""
    rng = np.random.default_rng(seed)
    for p, _ in model.parameters():
        p[...] = (rng.standard_normal(p.shape) * scale).astype(p.dtype)


def test_loss_directional_gradient():
    """dlogits from loss.backward must match a directional finite difference."""
    from tensor_engine_nmt import loss as loss_module

    rng = np.random.default_rng(3)
    logits = rng.standard_normal((hp.B, Ty, hp.V)).astype(np.float32) * 1.5
    v      = rng.standard_normal((hp.B, Ty, hp.V)).astype(np.float32)

    _, p, mask_f = loss_module.forward(logits, Yout, Ylen)
    dlogits = loss_module.backward(p, Yout, mask_f)

    l_plus,  _, _ = loss_module.forward(logits + EPS * v, Yout, Ylen)
    l_minus, _, _ = loss_module.forward(logits - EPS * v, Yout, Ylen)

    numeric  = (l_plus - l_minus) / (2 * EPS)
    analytic = float((dlogits * xp.asarray(v)).sum())

    denom = max(abs(numeric), abs(analytic), 1e-12)
    rel = abs(numeric - analytic) / denom
    print(f"  [{'OK ' if rel < 1e-3 else 'FAIL'}] loss directional derivative: "
          f"ana={analytic:+.6e} num={numeric:+.6e} rel={rel:.2e}")
    assert rel < 1e-3, f"loss.backward disagrees with finite differences (rel={rel:.2e})"


def test_full_model_gradients():
    """Every parameter's analytic gradient must match finite differences.

    Exercises encoder BPTT (both directions + handoff), decoder BPTT, the
    per-step attention cache, and the padded positions in X and Yout.
    """
    from tensor_engine_nmt.model import Seq2Seq

    np.random.seed(0)
    model = Seq2Seq(hp)
    _reinit(model)

    rng = np.random.default_rng(11)
    upstream = rng.standard_normal((hp.B, Ty, hp.V)).astype(np.float32) * 2.0
    up_x = xp.asarray(upstream)

    def objective():
        # NOTE: must not call zero_grad() here — that would destroy the analytic
        # gradients we are comparing against.
        _, logits = model.forward(X, Xlen, Yin, Yout, Ylen, global_step=-999999)
        return float((up_x * logits).sum())

    # Analytic gradient of J = <upstream, logits>, bypassing the loss module
    # (which test_loss_directional_gradient covers separately).
    model.zero_grad()
    model.forward(X, Xlen, Yin, Yout, Ylen, global_step=-999999)
    dH, dh_enc, dC_enc = model.decoder.backward(up_x)
    model.encoder.backward(dH, dh_enc, dC_enc)

    params = list(model.parameters())
    names = _param_names(model)
    assert len(names) == len(params), f"name/param mismatch: {len(names)} vs {len(params)}"

    j0 = objective()
    noise = NOISE_ULPS * np.finfo(np.float32).eps * max(abs(j0), 1.0) / (2 * EPS)

    idx_rng = np.random.default_rng(7)
    n_checked = n_failed = n_skipped = n_limited = 0
    worst = 0.0
    failures = []

    for name, (param, grad) in zip(names, params):
        ana = np.array(grad, dtype=np.float64, copy=True)
        flat = param.ravel()
        for idx in idx_rng.choice(flat.size, size=min(3, flat.size), replace=False):
            idx = int(idx)
            orig = float(flat[idx])

            flat[idx] = orig + EPS; j_plus = objective()
            flat[idx] = orig - EPS; j_minus = objective()
            flat[idx] = orig

            numeric = (j_plus - j_minus) / (2 * EPS)
            analytic = float(ana.ravel()[idx])

            mag = max(abs(numeric), abs(analytic))
            if mag < noise:
                n_skipped += 1
                continue
            n_checked += 1

            rel = abs(numeric - analytic) / max(mag, 1e-12)
            if mag > 10 * noise:
                worst = max(worst, rel)     # only well-conditioned entries count
            else:
                n_limited += 1

            if abs(numeric - analytic) > max(TOL * mag, noise):
                n_failed += 1
                failures.append(f"{name}[{idx}] ana={analytic:+.3e} num={numeric:+.3e} rel={rel:.2e}")

    print(f"  [{'OK ' if n_failed == 0 else 'FAIL'}] full model: J={j0:.3g}, "
          f"FD noise floor ~{noise:.1e}, {n_checked} entries checked "
          f"({n_skipped} below floor, {n_limited} noise-limited), "
          f"worst rel. err (well-conditioned) = {worst:.2e}")

    # Guard against the check silently degenerating into "no gradients at all".
    assert n_checked >= 30, f"only {n_checked} entries were above the noise floor"

    for f in failures[:10]:
        print(f"      {f}")
    assert not failures, f"{n_failed} of {n_checked} gradient entries FAILED finite differences"


def test_gradients_are_finite_and_nonzero():
    """No NaNs/Infs, and the backward pass actually produces gradient."""
    from tensor_engine_nmt.model import Seq2Seq

    np.random.seed(0)
    model = Seq2Seq(hp)
    model.zero_grad()
    model.forward(X, Xlen, Yin, Yout, Ylen, global_step=0)
    model.backward()

    any_nonzero = False
    for name, (param, grad) in zip(_param_names(model), model.parameters()):
        g = np.asarray(grad)
        assert np.isfinite(g).all(), f"{name} has non-finite gradients"
        if g.any():
            any_nonzero = True
    assert any_nonzero, "all gradients are zero after backward()"


if __name__ == "__main__":
    print("Gradient checks\n" + "-" * 60)
    test_loss_directional_gradient()
    test_full_model_gradients()
    test_gradients_are_finite_and_nonzero()
    print("-" * 60)
    print("All gradient checks passed.")
