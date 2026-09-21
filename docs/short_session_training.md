# Training in Short Sessions

> **You are here because you cannot leave a machine training for 12 hours.**
> This guide is the practical answer: how to measure what your hardware can do,
> how to split training into sessions that stop *cleanly*, and how to make sure a
> killed session never costs you more than a few minutes of work.

Everything here is verified against the current codebase. Where it says "run this
and look for that", the output shown is real.

---

## 0. Pick your path first

| Your situation | Path | Realistic outcome |
|---|---|---|
| No GPU (laptop/desktop CPU), or you can only run 1–2 h at a time | **§2 Path A — laptop preset** | A small model that learns short sentences. BLEU-4 in the low double digits. It will **not** match the full-size model. |
| Can use free Colab/Kaggle, but sessions get cut at ~12 h | **§3 Path B — time-boxed cloud runs** | The real model (108.6 M params). This is how the existing `step_196000.npz` was trained. |
| Both | **Best option**: train on the cloud, evaluate and translate locally | `§5` explains why the laptop is perfectly good for everything *except* training. |

**Before anything else, measure.** Do not skip §1. Everything else in this guide
depends on the number it prints.

---

## 1. Step 0 — measure your machine (5 minutes)

```bash
uv run tensor-engine-nmt bench --preset gpu      # the full-size model
uv run tensor-engine-nmt bench --preset laptop   # the small model
```

`bench` runs a handful of **real** forward/backward/Adam steps on real batches
and prints measured throughput, seconds per step, and a projected hours-per-epoch.
It does not touch your checkpoints, so it is always safe to run.

### Measured on the machine this project was developed on

A laptop CPU with no CUDA (`Backend: numpy`), 10 measured steps per preset:

| | `--preset gpu` (full model) | `--preset laptop` |
|---|---|---|
| Parameters | 108,585,216 | **5,877,824** |
| Example batch | B=8, Tx=30, Ty=31 | B=24, Tx=20, Ty=21 |
| Seconds per step | 9.25 s | **0.52 s** |
| Throughput | 27 tok/s | **962 tok/s** |
| Pairs per batch | 8.0 | 24.1 |
| Steps per epoch | 370,830 | 4,005 |
| **Hours per epoch** | **~950 h ≈ 40 days** | **0.58 h ≈ 35 min** |
| 10 epochs | ~400 days | **5.8 h** |
| One-time setup | — | BPE 188 s + cache 30 s |

> The full-size row was measured with `--max-tokens 500` to keep the benchmark
> quick. Time per epoch is only weakly dependent on `max_tokens`: a larger batch
> means fewer, slower steps. At the default `max_tokens=5000` the same machine
> lands in the **20–40 days per epoch** range. Either way the conclusion is the
> same — see below.

**This is the whole argument of the guide in one table.** On this laptop:

* the full-size model needs **about a month per epoch** — 30 epochs is over two
  years of continuous compute. Training it locally is not a scheduling problem,
  it is impossible.
* the laptop preset needs **35 minutes per epoch** — the entire 10-epoch schedule
  fits in **under 6 hours**, which is three evenings.

That is why Path A exists.

### How to read it

* **`hours per epoch`** is the only number that matters for planning. Compare it
  to your session length:
  * `hours/epoch` ≤ 1/3 of your session → you will see meaningful progress per session.
  * `hours/epoch` ≈ your session length → one epoch per session; slow but workable.
  * `hours/epoch` ≫ your session length → **do not start**. Shrink the preset, cap
    the dataset with `--max-pairs`, or move to the cloud. You would be spending
    days of electricity to reach the loss the model had after its first hour.
* **`tok/s`** is target tokens per second. It is the number printed in the training
  log, so you can sanity-check that a real run matches the benchmark.
* The first measured step is excluded from the steady-state figure because it pays
  for lazy BLAS setup and memory allocation.

> **A note on the README's old CPU figure.** The README quotes "~15 tok/s" for CPU.
> `bench` is the authority for *your* machine — CPUs vary by more than 20×. Also
> note the full-size model has 108.6 M parameters and a 32 000 × 512 embedding
> table, so its cost is dominated by matrix multiplies, while the laptop preset is
> dominated by Python-level loop overhead.

---

## 2. Path A — train on the laptop in 1–2 hour bursts

### What the `laptop` preset changes

`--preset laptop` is not just "smaller". It is a complete, separate setup:

| Setting | `gpu` | `laptop` | Why |
|---|---|---|---|
| `d` (hidden) | 1024 | **256** | ~4× less compute per gate; 16× less in the attention matmuls |
| `L` (layers) | 3 | **2** | fewer sequential Python loop iterations per token |
| `e` (embedding) | 512 | **128** | smaller tables, faster lookups |
| `V` (vocab) | 32000 | **8000** | the output projection `d × V` is the single biggest matmul in the model |
| `max_len` | 30 | **20** | shorter sequences only |
| `max_tokens` | 5000 | **1000** | batches that fit comfortably in CPU cache |
| `max_pairs` | all 2.97 M | **150 000** | short epochs, small cache |
| `k` / `min_tf` | 17000 / 0.70 | **6000 / 0.50** | the teacher-forcing schedule actually anneals within 10 epochs |
| `save_every` | 4000 | **100** | a checkpoint every few minutes, not every few hours |
| `keep_last` | — | **3** | keeps disk usage bounded |

It also uses **its own directories** — `bpe_vocab_laptop/` and
`checkpoints_laptop/` — so a small checkpoint can never be auto-resumed into the
full-size model (the shapes differ, and the loader would refuse anyway).

### First run: what to expect

```bash
uv run tensor-engine-nmt train --preset laptop --max-minutes 120
```

On the very first run this does three one-time setup steps before training starts:

1. **Trains a BPE vocabulary** on 150 000 sampled lines → `bpe_vocab_laptop/`
   (188 s on the reference laptop).
2. **Counts corpus lines and builds a tokenized cache** at an even stride (it
   encodes every ~19th of 2 977 999 lines) →
   `PhoMT_dataset/train_cache_V7947_L20_N150000.pkl` (30 s, 96 519 pairs).
   The count lands below `max_pairs` because roughly 40 % of the sampled
   sentences are longer than `max_len=20` and get filtered out.
3. Then starts stepping.

> **Already done in this checkout.** `bpe_vocab_laptop/` (7 947 tokens) and
> `train_cache_V7947_L20_N150000.pkl` (96 519 pairs, 6.6 MB) are present, so your
> first `--preset laptop` run skips straight to training:
> ```
> [BPE] Loaded from bpe_vocab_laptop/  (7,662 merges, 7,947 tokens)
> [dataset] Loading pre-tokenized cache from PhoMT_dataset\train_cache_V7947_L20_N150000.pkl...
> [dataset] Loaded 96,519 pairs.
> ```
> Both are in `.gitignore`, so they exist on disk but are not committed. If you
> delete them, the next run rebuilds them from scratch.

Both are cached. Every later run loads them in seconds and jumps straight to
training.

> The stride matters. Corpus files are usually grouped by source or topic, so
> "take the first 150 000 lines" would silently train the model on one narrow slice
> of the data. The stride spreads the sample across the whole file.

### The 2-hour recipe

```bash
# Session 1 (and every session after — the same command)
uv run tensor-engine-nmt train --preset laptop --max-minutes 120
```

When the budget expires you will see:

```
[train] Session budget of 120 minutes reached at step 3120 (epoch 1, batch 3120).
        Saved step_3120.npz — re-run the same command to resume.
```

No `--resume` needed: the next run auto-finds `checkpoints_laptop/step_3120.npz`
and continues from the exact batch.

**Do not** change `--max-pairs`, `--max-tokens` or `max_len` between sessions.
The mid-epoch resume logic recreates the batch order from `(seed, epoch)` and
skips the batches already done; change the batching parameters and that skip
lands in the wrong place. Changing `--lr` or `--max-minutes` between sessions is
fine.

### Using the model you trained

```bash
# Evaluate (greedy is much faster than beam on CPU)
uv run tensor-engine-nmt evaluate --ckpt checkpoints_laptop/step_3120.npz --method greedy --n 200

# Interactive translation
python demo.py --ckpt checkpoints_laptop/step_3120.npz
```

### Setting expectations honestly

The laptop preset is a ~10 M parameter model trained on 5 % of the corpus with a
20-token length cap. Expect it to handle short, common sentences and to struggle
with anything long or rare. Use it to:
* learn and verify the whole pipeline end to end,
* get a working translator for short sentences,
* test code changes cheaply before spending GPU hours.

Do not expect it to reproduce the BLEU numbers of the full model in
`docs/bugfix_report.md`. If you need those, use Path B.

---

## 3. Path B — cloud GPU in time-boxed chunks

The cloud is where the real model gets trained. Both free options cap sessions
(Colab ~12 h, Kaggle 12 h per run / 30 h per week), so the goal is to make every
session end **on your terms, with a checkpoint already written** — never on the
platform's.

```bash
# Fit inside the session and stop cleanly a little before the hard limit
uv run tensor-engine-nmt train --lr 1e-4 --max-minutes 630 --save-every 500 --keep-last 3
```

* `--max-minutes 630` = 10.5 h. It writes a checkpoint and exits cleanly, so a
  hard kill at 12 h can never catch you mid-write.
* `--save-every 500` checkpoints roughly every 20–35 minutes on a T4. The old
  `save_every=4000` meant a pre-emption could cost you **3–4 hours**.
* `--keep-last 3` deletes older `step_*.npz` after each save. Each full-size
  checkpoint is ~401 MB and `optim_state.npz` is ~733 MB, so on Kaggle's 20 GB
  output limit you cannot keep everything.

Full setup instructions: **[COLAB_GUIDE.md](../COLAB_GUIDE.md)** and
**[KAGGLE_GUIDE.md](../KAGGLE_GUIDE.md)**.

### The session loop

1. Open the notebook, run the setup cells.
2. `tensor-engine-nmt bench --steps 10` — confirm hours-per-epoch still matches
   what you planned. (A different GPU will give a different number.)
3. Launch the time-boxed run.
4. When it exits, **download `step_*.npz`, `optim_state.npz` and `train_logs.txt`
   to your laptop.** Do this every session — cloud working directories are
   ephemeral.
5. Next session: upload them back and run the exact same command.

---

## 4. The six rules of short-session training

1. **Always pass `--max-minutes`.** A clean stop with a checkpoint costs seconds.
   A kill mid-write costs the session.
2. **Set `--save-every` from your measured speed**, not from a default. Target a
   checkpoint every 10–20 minutes:
   ```
   save_every ≈ (15 * 60) / seconds_per_step
   ```
   At 0.5 s/step that is ~1800; at 3 s/step it is ~300.
3. **Use `--keep-last 3`.** Otherwise frequent checkpoints fill the disk — at
   401 MB each, 100 checkpoints is 40 GB.
4. **Resume with the same command.** Auto-resume finds the newest
   `step_*.npz` and continues at the exact batch. Verify these three lines:
   ```
   [model] Adam state restored (t=196000) from checkpoints/optim_state.npz
   [model] Checkpoint loaded <- checkpoints/step_196000.npz
   [train] Checkpoint metadata: epoch 8/30, step 23277 within epoch
   ```
   If instead you see `starting the optimizer fresh`, the Adam moments were
   missing or belonged to a different checkpoint — expect a small loss bump while
   Adam re-warms. See §6.
5. **Keep one copy off the machine.** After every session, copy the newest
   `step_*.npz` **and** `optim_state.npz` somewhere else (Drive, an external disk,
   a Kaggle dataset). They are a matched pair — see rule 6.
6. **Never delete `optim_state.npz` while keeping an older `step_*.npz`.** There is
   exactly one optimizer file, and it always belongs to the *most recently saved*
   checkpoint. The loader now checks this: if the file is tagged with a different
   checkpoint it warns loudly and starts the optimizer fresh instead of silently
   pairing mismatched moments with your weights.

---

## 5. What is cheap and what is expensive

| Task | Full-size model on a CPU laptop | Verdict |
|---|---|---|
| Greedy translation of one sentence | ~0.1 s | Totally fine |
| Beam-4 translation of one sentence | ~1.4 s | Fine for a demo |
| BLEU-4 on 200 sentences, greedy | ~20 s | Fine |
| **One training step** | **seconds to minutes** | This is the only expensive thing |

So the sensible split is: **train on the cloud, evaluate and demo on the laptop.**
Copy `step_N.npz` + the `bpe_vocab/` folder to your machine and everything except
training works.

---

## 6. Reading the log in a short session

```
2026-09-20 23:37:26 | INFO | ep 3/10 | step 3120 | loss=4.8213 | lr=3.00e-04 |
gnorm=0.874 | ε=0.612 | tok/s=1,240 | ~0.81s/step, epoch≈3.4h
```

| Field | Meaning |
|---|---|
| `ep 3/10` | Epoch / total epochs |
| `step 3120` | Cumulative optimiser steps (never resets) |
| `loss` | Mean per-token NLL since the last log line |
| `lr` | Learning rate (fixed; no scheduler) |
| `gnorm` | Gradient norm **before** clipping |
| `ε` | Teacher-forcing probability — how often the decoder is fed ground truth |
| `tok/s` | Target tokens per second |
| `~0.81s/step` | Measured seconds per optimiser step this epoch |
| `epoch≈3.4h` | Projected time for a full epoch, from the measurements so far |

The last two fields are the ones to watch in a short session — they tell you
whether the session you are in will finish an epoch, and they update as the epoch
progresses (early batches of an epoch are less representative).

---

## 7. Troubleshooting

### "Skipping first N batches to resume mid-epoch..." takes a while
Expected. The dataset iterator has to regenerate and discard the batches already
trained on, because the order is deterministic and recreated from scratch. It is
pure data work — no model compute — and takes seconds to a couple of minutes.

### `WARNING: ... holds the optimizer state saved with 'step_8000.npz', but you are loading 'step_4000.npz'`
You are resuming from an older checkpoint while `optim_state.npz` belongs to a
newer one. This is the correct, safe behaviour: the moments belong to different
weights, so the optimizer starts fresh and you get a small loss bump. To resume
properly, load the checkpoint the optimizer file belongs to (i.e. the newest one).

### `WARNING: using the legacy cache train_cache_V30839.pkl`
You have a cache built before cache filenames included `max_len`/`max_pairs`.
It still loads, but if you have *raised* `max_len` since it was built, the longer
sentences are already gone from it. Delete the file and re-run to rebuild.

### Loss spikes by +0.3 to +1.0 right after a resume
Either the optimizer state was not restored (check for `Adam state restored` in
the log), or you changed `--max-tokens` / `--max-pairs` between sessions and the
mid-epoch skip is misaligned. The model recovers either way; it just wastes steps.

### `ValueError: hp.V=32000 is smaller than the loaded BPE vocabulary`
`V` must be at least as large as the tokenizer you loaded. This happens if you mix
directories — e.g. `--preset laptop` (V=8000, `bpe_vocab_laptop/`) with the full
`bpe_vocab/` (30 839 tokens). Use the preset's own directories, or raise `V`.

### Out of memory on the laptop
Lower the token budget, not the model:
```bash
uv run tensor-engine-nmt train --preset laptop --max-tokens 500 --max-minutes 120
```

### Disk filling up
You have too many checkpoints. Add `--keep-last 3`, or delete old `step_*.npz`
files manually — but **keep the newest one and `optim_state.npz`**.

---

## 8. A realistic one-week plan (2 h/night on a laptop)

| Night | Command | What you get |
|---|---|---|
| 1 | `bench --preset laptop` | The number that decides everything below |
| 2 | `train --preset laptop --max-minutes 120` | One-time BPE + cache build, then the first steps |
| 3–4 | same command | A few epochs, loss falling |
| 5 | `evaluate --method greedy --n 200` | Your first BLEU number |
| 6–7 | same train command | Continue; try `python demo.py` on sentences you care about |

If `bench` told you the laptop preset needs more than ~10 h per epoch, do not
follow this table — go to Path B. Spending a week of evenings to move the loss by
0.1 is worse than spending one evening setting up Colab.
