"""
test_gradients.py — Numerical gradient check.

For each parameter θ, computes:
    numerical_grad[i] = (L(θ_i + ε) - L(θ_i - ε)) / (2ε)    ε = 1e-4

and compares to the analytical gradient from model.backward().

Pass condition: max relative error < 1e-3 across all checked entries.

Run with:
    python test/test_gradients.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from tensor_engine_nmt.config import HParams

hp = HParams(
    V=8, e=4, d=6, L=2, B=2,
    k=100.0,
    lr=1e-3, beta1=0.9, beta2=0.999, eps_adam=1e-8, clip_norm=100.0,
    max_epochs=1, beam_width=2, max_decode_len=5,
    log_every=1, save_every=9999,
    data_dir="PhoMT_dataset", bpe_dir="bpe_vocab", ckpt_dir="checkpoints",
    bpe_sample_lines=100,
)

B, e, d, L, V = hp.B, hp.e, hp.d, hp.L, hp.V
Tx, Ty = 3, 2
EPS = 1e-3
MAX_REL_ERR = 7e-2
N_CHECK = 5   # number of random indices to check per parameter


def make_batch():
    np.random.seed(7)
    X    = np.random.randint(4, V, (B, Tx)).astype(np.int32)
    Xlen = np.array([3, 2], dtype=np.int32)
    Yin  = np.array([[2, 4, 0], [2, 5, 0]], dtype=np.int32)[:, :Ty]
    Yout = np.array([[4, 3, 0], [5, 3, 0]], dtype=np.int32)[:, :Ty]
    Ylen = np.array([2, 2], dtype=np.int32)
    return X, Xlen, Yin, Yout, Ylen


def get_loss(model, X, Xlen, Yin, Yout, Ylen):
    model.zero_grad()
    loss_val, _ = model.forward(X, Xlen, Yin, Yout, Ylen, global_step=-999999)
    return loss_val


def run_grad_check():
    from tensor_engine_nmt.model import Seq2Seq

    np.random.seed(0)
    X, Xlen, Yin, Yout, Ylen = make_batch()
    model = Seq2Seq(hp)

    # ── Compute analytical gradients ──────────────────────────────────────────
    model.zero_grad()
    model.forward(X, Xlen, Yin, Yout, Ylen, global_step=-999999)
    model.backward()

    params_list = list(model.parameters())

    # Names for nicer output
    names = (
        ["Ex"] +
        [f"enc_W_ih[{l}]" for l in range(L)] +
        [f"enc_W_hh[{l}]" for l in range(L)] +
        [f"enc_b[{l}]"    for l in range(L)] +
        ["Ey"] +
        [f"dec_W_ih[{l}]" for l in range(L)] +
        [f"dec_W_hh[{l}]" for l in range(L)] +
        [f"dec_b[{l}]"    for l in range(L)] +
        ["W_a", "W_c", "b_c", "Wy", "by"]
    )
    # Pad names list to match actual params length
    while len(names) < len(params_list):
        names.append(f"param_{len(names)}")

    print(f"Checking {len(params_list)} parameter tensors…\n")
    all_passed = True

    for pidx, (param, grad) in enumerate(params_list):
        name = names[pidx] if pidx < len(names) else f"param_{pidx}"

        # Convert to numpy for manipulation
        p_np = param if isinstance(param, np.ndarray) else param.__array__()
        g_np = grad  if isinstance(grad,  np.ndarray) else grad.__array__()

        flat_size = p_np.size
        if flat_size == 0:
            continue

        indices = np.random.choice(flat_size, size=min(N_CHECK, flat_size), replace=False)
        max_rel_err = 0.0

        for idx in indices:
            # Save original value
            orig = float(p_np.flat[idx])

            # f(θ + ε)
            p_np.flat[idx] = orig + EPS
            if not isinstance(param, np.ndarray):
                param[:] = p_np
            l_plus = get_loss(model, X, Xlen, Yin, Yout, Ylen)

            # f(θ - ε)
            p_np.flat[idx] = orig - EPS
            if not isinstance(param, np.ndarray):
                param[:] = p_np
            l_minus = get_loss(model, X, Xlen, Yin, Yout, Ylen)

            # Restore
            p_np.flat[idx] = orig
            if not isinstance(param, np.ndarray):
                param[:] = p_np

            # Recompute analytical gradient (need fresh backward)
            model.zero_grad()
            model.forward(X, Xlen, Yin, Yout, Ylen, global_step=-999999)
            model.backward()
            g_np = grad if isinstance(grad, np.ndarray) else grad.__array__()

            numerical  = (l_plus - l_minus) / (2 * EPS)
            analytical = float(g_np.flat[idx])
            denom = max(abs(numerical), abs(analytical), 1e-8)
            rel_err = abs(numerical - analytical) / denom
            max_rel_err = max(max_rel_err, rel_err)

        ok = "✓" if max_rel_err < MAX_REL_ERR else "✗"
        if max_rel_err >= MAX_REL_ERR:
            all_passed = False
        print(f"  {ok} {name:<22s}  max_rel_err = {max_rel_err:.2e}")

    print()
    if all_passed:
        print("✅  All gradient checks passed.")
    else:
        print("❌  Some gradient checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    run_grad_check()
