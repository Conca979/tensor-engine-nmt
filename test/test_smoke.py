"""
test_smoke.py — fast end-to-end smoke test: BPE → dataset → model → optimizer.

    python test/test_smoke.py
    pytest test/test_smoke.py -v

Runs on a 300-line slice of the real corpus, copied into a temp directory, so it
exercises the full pipeline without tokenizing the 3M-pair training set.  It is
the only test that covers the data plumbing (cache, bucketing, dynamic batching,
collation) together with a real optimizer step.

Note on the previous version: it called `ds.iterate(batch_size=hp.B)`, but
iterate() does token-level dynamic batching and has no `batch_size` parameter,
so the test always died with a TypeError before doing anything.
"""
import sys
import os
import shutil
import tempfile
import time

# Force UTF-8 so Vietnamese output does not explode on a cp1252 console.
for _stream in ("stdout", "stderr"):
    s = getattr(sys, _stream, None)
    if s is not None and hasattr(s, "reconfigure"):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from tensor_engine_nmt.config import HParams
from tensor_engine_nmt.bpe import load_or_train_bpe
from tensor_engine_nmt.dataset import PhoMTDataset, verify_collation
from tensor_engine_nmt.model import Seq2Seq
from tensor_engine_nmt.optimizer import Adam

REAL_DATA = os.path.join(os.path.dirname(__file__), "..", "PhoMT_dataset")
N_LINES = 300
N_STEPS = 20


def _make_tiny_corpus(root: str) -> None:
    """Copy the first N_LINES of each training file into a throwaway corpus."""
    os.makedirs(os.path.join(root, "train"), exist_ok=True)
    for lang in ("en", "vi"):
        src = os.path.join(REAL_DATA, "train", f"train.{lang}")
        dst = os.path.join(root, "train", f"train.{lang}")
        with open(src, "r", encoding="utf-8") as fin, \
             open(dst, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if i >= N_LINES:
                    break
                fout.write(line)


def run_smoke() -> None:
    real_en = os.path.join(REAL_DATA, "train", "train.en")
    if not os.path.exists(real_en):
        print(f"SKIP: {real_en} not found — smoke test needs the corpus.")
        return

    root = tempfile.mkdtemp(prefix="nmt_smoke_")
    try:
        _make_tiny_corpus(root)

        hp = HParams(
            V=500, e=32, d=64, L=1, B=8,
            # Deliberately tiny token budget: 114 pairs must yield well over
            # N_STEPS batches so the optimiser actually gets exercised.
            max_tokens=200,
            k=2000.0, lr=1e-3, beta1=0.9, beta2=0.999, eps_adam=1e-8, clip_norm=5.0,
            max_epochs=1, beam_width=2, max_decode_len=30,
            log_every=5, save_every=9999,
            data_dir=root, bpe_dir=os.path.join(root, "bpe"),
            ckpt_dir=os.path.join(root, "ckpt"), bpe_sample_lines=N_LINES,
        )

        t0 = time.time()
        bpe = load_or_train_bpe(hp.bpe_dir, hp.data_dir, hp.V, hp.bpe_sample_lines)
        print(f"[smoke] BPE trained in {time.time() - t0:.1f}s, vocab={len(bpe.token2id):,}")

        probe = "Hello world how are you"
        ids = bpe.encode(probe, add_end=True)
        print(f"[smoke] round-trip: {probe!r} -> {len(ids)} ids -> {bpe.decode(ids)!r}")
        assert ids and bpe.decode(ids), "BPE round-trip produced nothing"

        ds = PhoMTDataset(bpe, hp.data_dir, "train", max_len=hp.max_len)
        print(f"[smoke] dataset: {len(ds.pairs):,} pairs after the max_len filter")
        assert ds.pairs, "dataset is empty after filtering — try a larger N_LINES"

        model = Seq2Seq(hp)
        optim = Adam(model, hp)
        print(f"[smoke] model parameters: {model.param_count():,}")

        losses = []
        t1 = time.time()
        for step, batch in enumerate(ds.iterate(max_tokens=hp.max_tokens, shuffle=False, seed=42)):
            if step >= N_STEPS:
                break
            if step == 0:
                verify_collation(batch)      # asserts dtypes, START position, END placement

            model.zero_grad()
            loss_val, _ = model.forward(
                batch["X"], batch["Xlen"], batch["Yin"], batch["Yout"], batch["Ylen"], step
            )
            model.backward()
            gnorm = optim.step()
            losses.append(loss_val)
            assert np.isfinite(loss_val), f"non-finite loss at step {step}"
            assert np.isfinite(gnorm), f"non-finite grad norm at step {step}"
            if (step + 1) % 5 == 0:
                print(f"[smoke]   step {step + 1:3d}  loss={loss_val:.4f}  gnorm={gnorm:.3f}")

        elapsed = time.time() - t1
        print(f"[smoke] {len(losses)} steps in {elapsed:.1f}s "
              f"({elapsed / max(1, len(losses)):.2f} s/step)")

        assert losses, "no batches were produced by the dataset"
        assert len(losses) >= 10, (
            f"only {len(losses)} batches produced — lower max_tokens so the "
            f"optimiser gets enough steps to be meaningfully tested"
        )

        first5 = float(np.mean(losses[:5]))
        best_after = float(np.min(losses[5:])) if len(losses) > 5 else float("inf")
        print(f"[smoke] first-5 avg loss = {first5:.4f}   best after step 5 = {best_after:.4f}")
        assert best_after < first5, (
            f"loss never improved: first-5 avg {first5:.4f} vs best {best_after:.4f}"
        )
        print("SMOKE TEST PASSED")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_smoke():
    run_smoke()


if __name__ == "__main__":
    run_smoke()
