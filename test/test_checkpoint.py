"""
test_checkpoint.py — checkpoint save/load, resume safety, atomicity.

    python test/test_checkpoint.py
    pytest test/test_checkpoint.py -v

Covers the failure modes that matter for a training run that spans days:
  * weights, Adam moments and the position-in-epoch all round-trip
  * the single `optim_state.npz` is NOT silently paired with the wrong weights
  * a truncated (crash-interrupted) checkpoint is detected instead of loading garbage
  * an architecture mismatch is reported clearly
"""
import sys
import os
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tensor_engine_nmt.config import HParams
from tensor_engine_nmt.model import Seq2Seq
from tensor_engine_nmt.optimizer import Adam

hp = HParams(V=8, e=4, d=6, L=2, B=2, k=100.0, lr=1e-3, clip_norm=5.0,
             max_epochs=1, beam_width=2, max_decode_len=5,
             log_every=1, save_every=9999, bpe_sample_lines=100)

X    = np.array([[4, 5, 6, 3], [4, 5, 3, 0]], dtype=np.int32)
Xlen = np.array([4, 3], dtype=np.int32)
Yin  = np.array([[2, 4, 5], [2, 6, 0]], dtype=np.int32)
Yout = np.array([[4, 5, 3], [6, 3, 0]], dtype=np.int32)
Ylen = np.array([3, 2], dtype=np.int32)


def _one_step(model, optim):
    model.zero_grad()
    model.forward(X, Xlen, Yin, Yout, Ylen, global_step=0)
    model.backward()
    optim.step()


def _params(model):
    return [np.array(p, dtype=np.float64, copy=True) for p, _ in model.parameters()]


def _run(tmp, body):
    try:
        body(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_roundtrip_and_meta():
    def body(tmp):
        np.random.seed(0)
        model = Seq2Seq(hp)
        optim = Adam(model, hp)
        _one_step(model, optim)

        ckpt = os.path.join(tmp, "step_4000.npz")
        model.save(ckpt, optim=optim, meta={"epoch": 3, "step_in_epoch": 1200})
        assert os.path.exists(ckpt)
        assert not os.path.exists(ckpt + ".tmp.npz"), "temp file was not renamed away"

        before_w  = _params(model)
        before_m  = [np.array(m, copy=True) for m in optim.m]
        before_t  = optim.t

        np.random.seed(1)                      # different init
        model2 = Seq2Seq(hp)
        optim2 = Adam(model2, hp)
        meta = model2.load(ckpt, optim=optim2)

        assert meta == {"epoch": 3, "step_in_epoch": 1200}, meta
        assert optim2.t == before_t, f"optimizer step lost: {optim2.t} != {before_t}"
        for a, b in zip(before_w, _params(model2)):
            assert np.allclose(a, b, atol=0), "weights did not round-trip exactly"
        for a, b in zip(before_m, optim2.m):
            assert np.allclose(a, b, atol=0), "Adam first moment did not round-trip"
        print("  [OK ] weights + Adam state + epoch metadata round-trip")

    _run(tempfile.mkdtemp(prefix="nmt_ckpt_"), body)


def test_mismatched_optim_state_starts_fresh():
    """optim_state.npz belongs to a LATER checkpoint than the one being loaded."""
    def body(tmp):
        np.random.seed(0)
        model = Seq2Seq(hp)
        optim = Adam(model, hp)

        old = os.path.join(tmp, "step_4000.npz")
        _one_step(model, optim)
        model.save(old, optim=optim, meta={"epoch": 0, "step_in_epoch": 10})

        new = os.path.join(tmp, "step_8000.npz")
        for _ in range(5):
            _one_step(model, optim)
        model.save(new, optim=optim, meta={"epoch": 0, "step_in_epoch": 20})
        # optim_state.npz now belongs to step_8000.npz

        np.random.seed(1)
        model2 = Seq2Seq(hp)
        optim2 = Adam(model2, hp)
        model2.load(old, optim=optim2)          # resume from the OLDER checkpoint

        assert optim2.t == 0, (
            f"stale optimizer state was loaded (t={optim2.t}); it must not be paired "
            f"with different weights"
        )
        for m in optim2.m:
            assert not np.any(m), "stale Adam moments were loaded"
        print("  [OK ] mismatched optim_state.npz detected -> optimizer starts fresh")

    _run(tempfile.mkdtemp(prefix="nmt_ckpt_"), body)


def test_truncated_optim_state_does_not_crash():
    def body(tmp):
        np.random.seed(0)
        model = Seq2Seq(hp)
        optim = Adam(model, hp)
        _one_step(model, optim)
        ckpt = os.path.join(tmp, "step_4000.npz")
        model.save(ckpt, optim=optim, meta={"epoch": 0, "step_in_epoch": 5})

        # Simulate a session pre-empted mid-write of the companion file.
        optim_path = os.path.join(tmp, "optim_state.npz")
        with open(optim_path, "r+b") as f:
            f.truncate(os.path.getsize(optim_path) // 3)

        np.random.seed(1)
        model2 = Seq2Seq(hp)
        optim2 = Adam(model2, hp)
        model2.load(ckpt, optim=optim2)
        assert optim2.t == 0
        print("  [OK ] truncated optim_state.npz -> warning, fresh optimizer, no crash")

    _run(tempfile.mkdtemp(prefix="nmt_ckpt_"), body)


def test_truncated_weights_are_detected():
    def body(tmp):
        np.random.seed(0)
        model = Seq2Seq(hp)
        optim = Adam(model, hp)
        ckpt = os.path.join(tmp, "step_4000.npz")
        model.save(ckpt, optim=optim)
        with open(ckpt, "r+b") as f:
            f.truncate(os.path.getsize(ckpt) // 3)

        np.random.seed(1)
        model2 = Seq2Seq(hp)
        try:
            model2.load(ckpt)
        except Exception as e:
            print(f"  [OK ] truncated weights rejected ({type(e).__name__}: {e})")
            return
        raise AssertionError("a truncated checkpoint loaded without error")

    _run(tempfile.mkdtemp(prefix="nmt_ckpt_"), body)


def test_architecture_mismatch_is_clear():
    def body(tmp):
        np.random.seed(0)
        model = Seq2Seq(hp)
        ckpt = os.path.join(tmp, "step_4000.npz")
        model.save(ckpt)

        other = HParams(V=8, e=4, d=8, L=2, B=2, k=100.0, lr=1e-3, clip_norm=5.0,
                        max_epochs=1, beam_width=2, max_decode_len=5,
                        log_every=1, save_every=9999, bpe_sample_lines=100)
        np.random.seed(1)
        model2 = Seq2Seq(other)
        try:
            model2.load(ckpt)
        except ValueError as e:
            assert "shape" in str(e).lower()
            print(f"  [OK ] architecture mismatch reported: {e}")
            return
        raise AssertionError("a mismatched architecture loaded without error")

    _run(tempfile.mkdtemp(prefix="nmt_ckpt_"), body)


if __name__ == "__main__":
    print("Checkpoint checks\n" + "-" * 60)
    test_roundtrip_and_meta()
    test_mismatched_optim_state_starts_fresh()
    test_truncated_optim_state_does_not_crash()
    test_truncated_weights_are_detected()
    test_architecture_mismatch_is_clear()
    print("-" * 60)
    print("All checkpoint checks passed.")
