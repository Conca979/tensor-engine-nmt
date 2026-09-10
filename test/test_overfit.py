"""
test_overfit.py — Overfit smoke test.

Feed the model a single fixed batch of 4 examples and train for 300 steps.
Verify that the loss converges to near zero.

If this passes:
  - The forward pass is correct
  - The backward pass computes valid gradients
  - The Adam optimizer updates in the right direction

Run with:
    python test/test_overfit.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tensor_engine_nmt.config import HParams

# Slightly larger than shape test to have more capacity to overfit
hp = HParams(
    V=16, e=8, d=16, L=2, B=4,
    k=1e9,      # ε ≈ 1.0 always → pure teacher forcing for easier overfit
    lr=3e-3,    # higher LR to overfit faster
    beta1=0.9, beta2=0.999, eps_adam=1e-8, clip_norm=5.0,
    max_epochs=1, beam_width=2, max_decode_len=10,
    log_every=50, save_every=9999,
    data_dir="PhoMT_dataset", bpe_dir="bpe_vocab", ckpt_dir="checkpoints",
    bpe_sample_lines=100,
)

B, e, d, L, V = hp.B, hp.e, hp.d, hp.L, hp.V
Tx, Ty = 5, 4


def make_fixed_batch():
    """A fixed, deterministic batch of 4 sentence pairs."""
    np.random.seed(123)
    X    = np.random.randint(4, V, (B, Tx)).astype(np.int32)
    Xlen = np.array([5, 4, 5, 3], dtype=np.int32)
    # Decoder: Yin starts with START (2), Yout ends with END (3)
    Yin  = np.array([
        [2, 5,  8, 11],
        [2, 6,  9, 12],
        [2, 7, 10, 13],
        [2, 5,  6,  0],
    ], dtype=np.int32)[:, :Ty]
    Yout = np.array([
        [5,  8, 11, 3],
        [6,  9, 12, 3],
        [7, 10, 13, 3],
        [5,  6,  3, 0],
    ], dtype=np.int32)[:, :Ty]
    Ylen = np.array([4, 4, 4, 3], dtype=np.int32)
    return X, Xlen, Yin, Yout, Ylen


def test_overfit():
    from tensor_engine_nmt.model import Seq2Seq
    from tensor_engine_nmt.optimizer import Adam

    X, Xlen, Yin, Yout, Ylen = make_fixed_batch()
    model = Seq2Seq(hp)
    optim = Adam(model, hp)

    n_steps    = 300
    target_loss = 0.5   # loss should drop well below this

    losses = []
    for step in range(n_steps):
        model.zero_grad()
        loss_val, _ = model.forward(X, Xlen, Yin, Yout, Ylen, global_step=step)
        model.backward()
        optim.step()
        losses.append(loss_val)

        if (step + 1) % 50 == 0:
            print(f"  step {step+1:4d}/{n_steps}  loss = {loss_val:.6f}")

    final_loss = losses[-1]
    print(f"\nFinal loss after {n_steps} steps: {final_loss:.6f}  (target < {target_loss})")

    if final_loss < target_loss:
        print("✅  Overfit test PASSED — model can memorize a small batch.")
    else:
        print(f"❌  Overfit test FAILED — loss {final_loss:.4f} did not reach < {target_loss}.")
        sys.exit(1)


if __name__ == "__main__":
    test_overfit()
