# Backward-pass bug fixes — report and next steps

Date: 2026-02 (session)
Baseline checkpoint: `checkpoints/step_196000.npz` (108,585,216 params, epoch 8/30)

This document records what was broken, what was changed, the evidence, and what
to do next. Everything here is reproducible from `verification/` and `test/`.

---

## 1. What was fixed

### Bug A — encoder handoff gradient injected at the wrong timestep
`src/tensor_engine_nmt/encoder.py`, `EncoderLSTM.backward`

The decoder reads the handoff as `[ h_fwd @ t=Xlen-1 | h_bwd @ t=0 ]`
(`decoder.py:133-138`), but the backward pass injected the **whole** `(B, d)`
gradient vector at `t = Xlen-1`. The backward-direction half therefore landed on
the wrong timestep and the correct one (`t = 0`) received no gradient at all.

The file contradicted itself: the *cell*-state injection 25 lines below did split
the two halves correctly (`dC_enc_final[l][:, :d_half]` at `Xlen-1`,
`dC_enc_final[l][:, d_half:]` at `t == 0`). That inconsistency is what identified
it as an oversight rather than a design choice.

**Fix:** build a forward-half vector masked to `t = Xlen-1` and a backward-half
vector masked to `t = 0`, and inject each separately.

### Bug B — attention reused one cache for every decoder timestep
`src/tensor_engine_nmt/attention.py`, `src/tensor_engine_nmt/decoder.py`

`LuongAttention.forward` stored its per-step cache in a single `self._cache`,
overwritten on every call. `DecoderLSTM.forward` calls it once per output token,
but `DecoderLSTM.backward` calls `attention.backward` once per token **in
reverse** — and `backward` read `self._cache`, which by then held only the final
step's `alpha_t`, `temp` and `s_t`. Every step except the last backpropagated
attention against the wrong alignment weights, the wrong score vector and the
wrong hidden state.

**Fix:** `decoder.forward` collects each step's cache into `attn_caches`, stores
it in the decoder cache, and `decoder.backward` passes `attn_caches[t]` to
`attention.backward(dz, ds, cache)`. The parameter defaults to `self._cache`, so
single-step callers (inference, `test_shapes`) are unaffected.

### Bug C — trailing padding leaked into the backward encoder states
`src/tensor_engine_nmt/encoder.py`, `EncoderLSTM.forward` / `backward`

The backward LSTM swept `Tx-1 → 0` for every sequence regardless of `Xlen`, so
trailing PAD steps were processed *first* and their state flowed into the real
positions and into the decoder handoff. Consequences:

* the representation of a sentence depended on how much padding its batch
  happened to contain;
* at inference `B = 1` and there is no padding, so training and decoding saw
  systematically different encoder states.

**Fix:** zero `h_bwd`/`C_bwd` (and the stored cache entries) at `t >= Xlen`, and
drop the corresponding incoming gradients in the backward pass — the stored
values are what the graph consumes, so a gradient arriving at a padded step is a
gradient with respect to a constant zero.

### Minor correctness / robustness fixes

| Where | Fix |
|---|---|
| `dataset.collate_batch` | `Ty = max(len(s) + 1)` always appended an all-PAD column, costing every batch one wasted decoder step. Now `Ty = max(len(s))`. |
| `decoder.forward` | Teacher forcing drew **one** random number for the whole batch, so every sequence either got ground truth or none. Now per-sequence, and seeded from `global_step` so a resumed run replays identical decisions. |
| `inference.beam_decode` | Length normalisation was applied only to completed hypotheses while active beams were pruned on **raw** NLL — biasing the beam toward short outputs, the exact thing the penalty exists to prevent. Now both use `NLL / gen_len**alpha`. |
| `inference.beam_decode` | The repetition penalty was applied to `exp(logits)` (always positive → always a division), silently differing from `greedy_decode`'s documented rule. Now both apply it to the logits. |
| `inference` (both) | `PAD` and `START` are now banned from the output vocabulary. |
| `model.save/load` | Exactly one `optim_state.npz` exists per directory, so resuming from an **older** `step_*.npz` silently paired it with a newer step's Adam moments. The file is now tagged with its checkpoint and a mismatch starts the optimizer fresh with a loud warning. |
| `model.save` | `np.savez_compressed` writes straight to the destination — a pre-empted Colab/Kaggle session could leave a truncated multi-hundred-MB checkpoint. Both files are now written to a temp file and `os.replace`d. |
| `model.load` | A truncated `optim_state.npz` raised `BadZipFile` outside any guard; now caught. Added an explicit architecture-mismatch error instead of a raw broadcast failure. |
| `train.py` | The position within the epoch lived in a side file written **after** both the weights and the 733 MB optimizer state, leaving a multi-second crash window. It is now stored inside the checkpoint (`meta_epoch`, `meta_step_in_epoch`); `state.txt` is kept as a legacy fallback. |
| `train.py` | `PhoMTDataset(..., max_len=30)` was hard-coded, defeating the "single source of truth" design while `evaluate.py` used `hp.max_len`. Now `hp.max_len` everywhere, plus the `getattr(hp, 'x', <magic default>)` fallbacks were removed. |
| `dataset.py` | The tokenized cache was keyed on vocab size only, but `_stream_pairs` bakes the `max_len` filter into the cached content — raising `max_len` silently reused a truncated cache. Key is now `V{vocab}_L{max_len}`, new caches carry metadata, and a legacy cache triggers an explicit warning. |
| `test/test_smoke.py` | Called `ds.iterate(batch_size=...)`, which has never existed (iterate does token-level dynamic batching) — the test always died with `TypeError`. Rewritten to run on a 300-line corpus slice in a temp dir. |
| `test/test_gradients.py` | See §2. Rewritten. |

---

## 2. Why the old gradient test was worthless

`test/test_gradients.py` used `EPS = 1e-3` on a float32 model. The loss is ~2.07,
whose float32 ULP is ~2.4e-7, so the central difference
`(L(+ε) − L(−ε)) / 2ε` is quantised in steps of **1.2e-4** — larger than most of
the gradients being measured. It reported `max_rel_err = 1.00e+00` for 21 of 25
parameter tensors and `sys.exit(1)`: it was failing on its own rounding noise, and
the README advertised it as "✅ Pass".

The replacement measures a synthetic objective `J = <upstream, logits>` with
unit-scale random upstream gradients, so `J` and `dJ/dθ` are both O(1) and the FD
floor drops to ~1e-4. Entries below the floor are reported as skipped instead of
silently passing. It also uses a batch with real padding and `Ty > 1`, which is
what makes it sensitive to all three bugs.

**Proof the new test has teeth:** run against the pre-fix source it fails with
`27 of 71 gradient entries FAILED`, worst relative error `1.53`. Against the
fixed source: `worst rel. err (well-conditioned) = 5.24e-03`.

A/B of the fixes (`verification/compare_ab.py`, same checkpoint):

| | before | after |
|---|---|---|
| corpus BLEU-4, greedy, 200 filtered test sentences | 18.68 | **18.68** |
| corpus BLEU-4, beam-4, 100 filtered test sentences | 23.73 | **24.66** |
| `"Thank you very much."` (beam) | `- cảm cảm ơn .` | `- cảm ơn con rất nhiều .` |

Greedy output is **byte-identical**. That is expected and important: Bugs A and B
only touch the backward pass, and Bug C's mask is a no-op when `Tx == Xlen`, which
is always the case at inference. **So `step_196000.npz` remains fully usable for
translation** — the fixes do not invalidate it. They do mean that *continuing*
training will now produce correct gradients, and the encoder will build
representations that are consistent between training and inference.

---

## 3. Next steps, in priority order

### 3.1 Resume training from `step_196000.npz` (highest value)
The checkpoint is at epoch 8/30 with test BLEU-4 ≈ 18.7 (greedy) / 24.7 (beam),
and the training log shows the loss still drifting down (3.5 at step 19 k → 2.5 at
step 196 k) with ε pinned at its 0.70 floor. It is under-trained, not
over-trained. Continue the run — gradients are now correct for the first time.

**Decide the learning rate first.** `config.py` currently says `lr = 1e-5`
(uncommitted change), but the entire step-196 k run used `1e-4` and the README
still documents `1e-4`. Resuming at `1e-5` would be ~10× slower. Pass it
explicitly so the choice is visible:

```bash
# 1-step dry run: confirms resume, prints checkpoint metadata + Adam restore
uv run tensor-engine-nmt train --lr 1e-4 --max-tokens 300 --max-steps 196001
```

Then launch the real run. Watch for: the loss at steps 196–200 k should behave
differently from before, because ~3/4 of decoder steps previously received
corrupted attention gradients.

### 3.2 Fix the teacher-forcing schedule before the next long run
Two compounding problems:

* `min_tf = 0.70` means ε stops decaying at ~step 151,000 and **never** drops
  below 0.70. The model has been trained with ≥70 % ground-truth inputs for its
  entire life, while inference is 100 % free-running — a large exposure-bias gap.
* `k = 17,000` decays far too slowly relative to the 24,600-step epoch length.

Suggested: `k = 4_000`, `min_tf = 0.35`, so ε actually anneals within the
remaining epochs. Both are now CLI flags, so no file edit is needed:

```bash
uv run tensor-engine-nmt train --lr 1e-4 --k 4000 --min-tf 0.35 --max-tokens 4000
```

Expect a loss bump in the first few thousand steps: ε will fall below 0.70 for
the first time in the model's life, so it is being pushed into a regime it has
barely seen. That is the point — but watch that it recovers.

### 3.3 Add dropout
`docs/analysis_results.md` flags this and it is the single biggest architectural
gap. There is none. With 108 M parameters on a 3 M-pair corpus you will overfit
eventually. Variational dropout on the LSTM *outputs* (same mask across
timesteps) is the standard choice and is easy here: sample one mask per layer per
batch and apply it to `layer_output` / `h_dec`, then multiply the incoming
gradient by the same mask in backward. Do not add it before 3.1 — you want a
clean read on the fixes first.

### 3.4 Evaluate properly
`evaluate.py` only computes BLEU-4, which the README itself notes is brittle for
Vietnamese. Add `chrF++` and `BERTScore` as optional metrics (guarded imports) so
you can tell "wrong but reasonable paraphrase" from "actually broken". Also
consider banning `[UNK]` from BLEU inputs, or at least reporting the UNK rate —
it is currently counted as a token mismatch.

### 3.5 Lower-priority cleanups
* `dataset._bucket_id` buckets on the **English** length only; the batch also
  pads the Vietnamese side, so a long-VI/short-EN pair still wastes padding.
  Bucket on `max(len(en), len(vi))`.
* `encoder.backward` scatters embedding gradients through `np.add.at` on the CPU
  (`.asnumpy()` per timestep in both encoder and decoder). On GPU this forces a
  sync every step. Replace with a backend `add.at` equivalent or a scatter-add.
* `loss.forward` returns `float(loss)`, forcing a device sync per step.
* `backend.to_xp` / `to_np` are dead code.
* `checkpoints/` contains a 341 MB `dataset.npz` (a Kaggle bundle of the raw
  corpus) and a stray `validate_nb.py`. `__init__.py`'s auto-checkpoint glob
  picks up any `*.npz` in that directory; it happens to sort last because it has
  no `step_` number, but the glob should be `step_*.npz`.
* No unit test covers `bpe.py` (round-trip, UNK behaviour, `</w>` handling) or
  `evaluate.corpus_bleu` against a known reference value.

### 3.6 Only if you want more quality after all of the above
The 3-layer `d=1024` BiLSTM at ~110 M parameters is roughly Luong et al. (2015)
scale and should reach BLEU-4 in the mid-20s on this test set given enough
steps. If it plateaus there, the highest-leverage architectural change is input
feeding (concatenate the previous target embedding to the decoder input) plus
residual connections between LSTM layers — both are small changes to
`decoder.py` and both are known to help stacked LSTMs. A Transformer would beat
it, but that is what `torch_experiment/` already is.

---

## 4. How to reproduce the verification

```bash
.venv/Scripts/python.exe verification/verify_bugs.py        # backward-pass gate
.venv/Scripts/python.exe verification/verify_padding_leak.py
.venv/Scripts/python.exe test/test_gradients.py
.venv/Scripts/python.exe test/test_checkpoint.py
.venv/Scripts/python.exe test/test_smoke.py
pytest test -v                                             # 15 tests
```
