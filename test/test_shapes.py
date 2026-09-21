"""
test_shapes.py — End-to-end shape assertions.

Uses tiny hyperparams matching the "Worked Example" in the spec:
    B=2, e=4, d=6, V=8, L=2, Tx=3, Ty=2

Runs a full forward pass on synthetic integer data and asserts every
intermediate tensor shape from the End-to-End Shape Checklist in
test/unidirectional_pipeline.md.

Run with:
    python -m pytest test/test_shapes.py -v
  or directly:
    python test/test_shapes.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

# ── Tiny config for testing ───────────────────────────────────────────────────
from tensor_engine_nmt.config import HParams

hp = HParams(
    V=8, e=4, d=6, L=2, B=2,
    k=100.0,
    lr=1e-3, beta1=0.9, beta2=0.999, eps_adam=1e-8, clip_norm=5.0,
    max_epochs=1, beam_width=2, max_decode_len=10,
    log_every=1, save_every=9999,
    data_dir="PhoMT_dataset", bpe_dir="bpe_vocab", ckpt_dir="checkpoints",
    bpe_sample_lines=1000,
)

B, e, d, L, V = hp.B, hp.e, hp.d, hp.L, hp.V
Tx, Ty = 3, 2


def make_batch():
    """Create a minimal synthetic batch."""
    np.random.seed(42)
    X    = np.random.randint(4, V, size=(B, Tx)).astype(np.int32)
    Xlen = np.array([3, 2], dtype=np.int32)
    Yin  = np.array([[2, 4, 0], [2, 5, 0]], dtype=np.int32)[:, :Ty]
    Yout = np.array([[4, 3, 0], [5, 3, 0]], dtype=np.int32)[:, :Ty]
    Ylen = np.array([2, 2], dtype=np.int32)
    return X, Xlen, Yin, Yout, Ylen


def test_encoder_shapes():
    from tensor_engine_nmt.encoder import EncoderLSTM
    X, Xlen, _, _, _ = make_batch()
    enc = EncoderLSTM(hp)
    H, h_enc_all, C_enc_all, cache = enc.forward(X, Xlen)

    # Spec: H : (B, Tx, d)
    assert H.shape == (B, Tx, d),        f"H shape: {H.shape}"
    # h_enc_all : (L, B, Tx, d)
    assert h_enc_all.shape == (L, B, Tx, d)
    assert C_enc_all.shape == (L, B, Tx, d)

    # Layer 0 input step 0: (B, e)
    x_t = cache["inp_cache"][0][0]
    assert x_t.shape == (B, e),          f"x_t shape: {x_t.shape}"

    # Per-step per-direction h: (B, d // 2)
    h_fwd_t = cache["h_cache_fwd"][0][1]
    assert h_fwd_t.shape == (B, d // 2), f"h_cache_fwd[0][1] shape: {h_fwd_t.shape}"

    print("[OK] encoder shapes OK")


def test_attention_shapes():
    from tensor_engine_nmt.encoder import EncoderLSTM
    from tensor_engine_nmt.attention import LuongAttention
    import numpy as np
    from tensor_engine_nmt.backend import xp, f32

    X, Xlen, _, _, _ = make_batch()
    enc = EncoderLSTM(hp)
    H, _, _, _ = enc.forward(X, Xlen)

    attn = LuongAttention(hp)
    s_t = xp.zeros((B, d), dtype=f32)
    z_t, alpha_t = attn.forward(s_t, H, Xlen)

    assert z_t.shape     == (B, d),      f"z_t shape: {z_t.shape}"
    assert alpha_t.shape == (B, Tx),     f"alpha_t shape: {alpha_t.shape}"

    # Rows must sum to 1
    row_sums = np.abs(float(alpha_t.sum(axis=1)[0]) - 1.0)
    assert row_sums < 1e-5, f"alpha_t row sum != 1: {alpha_t.sum(axis=1)}"

    # Padding position of example 1 must have ≈ 0 weight
    pad_weight = float(alpha_t[1, 2])
    assert pad_weight < 1e-6, f"Padding attention weight too large: {pad_weight}"

    print("[OK] attention shapes OK")


def test_decoder_shapes():
    from tensor_engine_nmt.encoder import EncoderLSTM
    from tensor_engine_nmt.decoder import DecoderLSTM

    X, Xlen, Yin, Yout, Ylen = make_batch()
    enc = EncoderLSTM(hp)
    H, h_enc_all, C_enc_all, _ = enc.forward(X, Xlen)

    dec = DecoderLSTM(hp)
    logits, cache = dec.forward(Yin, H, Xlen, h_enc_all, C_enc_all, global_step=0)

    # Spec: logits : (B, Ty, V)
    assert logits.shape == (B, Ty, V),   f"logits shape: {logits.shape}"

    # s_t : (B, d)
    s_t = cache["s_t_cache"][0]
    assert s_t.shape == (B, d),          f"s_t shape: {s_t.shape}"

    # e_t is masked raw scores before softmax: internal to attention
    # alpha_t : (B, Tx)
    alpha_t = cache["alpha_t_cache"][0]
    assert alpha_t.shape == (B, Tx),     f"alpha_t shape: {alpha_t.shape}"

    # z_t : (B, d)
    z_t = cache["z_t_cache"][0]
    assert z_t.shape == (B, d),          f"z_t shape: {z_t.shape}"

    # s_tilde : (B, d)
    s_tilde = cache["s_tilde_cache"][0]
    assert s_tilde.shape == (B, d),      f"s_tilde shape: {s_tilde.shape}"

    # y_t : (B, e)
    y_t = cache["y_emb_cache"][0]
    assert y_t.shape == (B, e),          f"y_t shape: {y_t.shape}"

    print("[OK] decoder shapes OK")


def test_loss_shapes():
    from tensor_engine_nmt.encoder import EncoderLSTM
    from tensor_engine_nmt.decoder import DecoderLSTM
    from tensor_engine_nmt import loss as loss_module
    import numpy as np

    X, Xlen, Yin, Yout, Ylen = make_batch()
    enc = EncoderLSTM(hp)
    H, h_enc_all, C_enc_all, _ = enc.forward(X, Xlen)
    dec = DecoderLSTM(hp)
    logits, _ = dec.forward(Yin, H, Xlen, h_enc_all, C_enc_all)

    loss_val, p, mask_f = loss_module.forward(logits, Yout, Ylen)

    # loss : scalar float
    assert isinstance(loss_val, float),  f"loss not scalar: {type(loss_val)}"
    assert loss_val > 0,                 f"loss should be > 0: {loss_val}"

    # p : (B, Ty, V)  rows sum to 1
    assert p.shape == (B, Ty, V),        f"p shape: {p.shape}"
    row_sums = float(p[0, 0, :].sum())
    assert abs(row_sums - 1.0) < 1e-5,  f"p row sum != 1: {row_sums}"

    # mask : (B, Ty)
    assert mask_f.shape == (B, Ty),      f"mask shape: {mask_f.shape}"

    # dlogits : (B, Ty, V)
    dlogits = loss_module.backward(p, Yout, mask_f)
    assert dlogits.shape == (B, Ty, V),  f"dlogits shape: {dlogits.shape}"

    print("[OK] loss shapes OK")


def test_full_forward_backward():
    """Run the entire model forward + backward without crashing."""
    from tensor_engine_nmt.model import Seq2Seq

    X, Xlen, Yin, Yout, Ylen = make_batch()
    model = Seq2Seq(hp)
    model.zero_grad()
    loss_val, logits = model.forward(X, Xlen, Yin, Yout, Ylen, global_step=0)
    model.backward()

    assert isinstance(loss_val, float)
    assert logits.shape == (B, Ty, V)

    # Check that at least one gradient is non-zero
    import numpy as np
    for (p, g) in model.parameters():
        g_np = g if isinstance(g, np.ndarray) else g.__array__()
        if g_np.any():
            print("[OK] full forward+backward OK (gradients non-zero)")
            return

    raise AssertionError("All gradients are zero after backward — something is wrong.")


if __name__ == "__main__":
    test_encoder_shapes()
    test_attention_shapes()
    test_decoder_shapes()
    test_loss_shapes()
    test_full_forward_backward()
    print("\nAll shape tests passed.")
