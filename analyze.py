"""
analyze.py — Professional Training & Evaluation Analysis Tool
=============================================================

Parses `checkpoints/train_logs.txt` and `evaluation_result.txt` to produce
a comprehensive diagnostic report covering:

  1. Training progression   — loss curves, gradient health, throughput
  2. Session analysis       — restart detection, cold-start spikes
  3. Phase-by-phase summary — per-epoch and cross-epoch trends
  4. Teacher forcing decay  — schedule verification against config
  5. BLEU-4 evaluation      — corpus and sentence-level statistics
  6. Translation quality    — qualitative sample analysis

Usage:
    python analyze.py                   # full report (logs + eval)
    python analyze.py --logs-only       # training analysis only
    python analyze.py --eval-only       # evaluation analysis only
    python analyze.py --top N           # show top-N translations (default 5)
"""

import sys
import os
import re
import math
import argparse
import statistics
from collections import Counter, defaultdict
from datetime import datetime

# ── UTF-8 safe output ─────────────────────────────────────────────────────────
sys.stdout.reconfigure(encoding="utf-8")

# ── Config ────────────────────────────────────────────────────────────────────
LOG_PATH  = os.path.join("checkpoints", "train_logs.txt")
EVAL_PATH = "evaluation_result.txt"

# Teacher-forcing schedule params (must match config.py)
K_TF = 17_000.0


# ══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════════════

def div(a: str, width: int = 70) -> None:
    """Print a section divider."""
    print(f"\n{'━' * width}")
    label = f"  {a}  "
    pad = (width - len(label)) // 2
    print(" " * pad + label)
    print("━" * width)


def sub(label: str, width: int = 70) -> None:
    print(f"\n  {'─' * (width - 4)}")
    print(f"  {label}")
    print(f"  {'─' * (width - 4)}")


def row(label: str, value, width: int = 30) -> None:
    print(f"  {label:<{width}} {value}")


def _ngrams(tokens: list, n: int) -> Counter:
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hypotheses: list, references: list, max_n: int = 4):
    """Corpus BLEU-4 with full breakdown."""
    clipped = [0] * max_n
    total   = [0] * max_n
    h_len = r_len = 0

    for hyp, ref in zip(hypotheses, references):
        h_len += len(hyp)
        r_len += len(ref)
        for n in range(1, max_n + 1):
            h_ng = _ngrams(hyp, n)
            r_ng = _ngrams(ref, n)
            clipped[n - 1] += sum(min(c, r_ng.get(ng, 0)) for ng, c in h_ng.items())
            total[n - 1]   += max(0, len(hyp) - n + 1)

    prec = []
    for n in range(max_n):
        prec.append(clipped[n] / total[n] if total[n] > 0 else 0.0)

    if any(p == 0 for p in prec):
        bleu = 0.0
    else:
        log_avg = sum(math.log(p) for p in prec) / max_n
        bp      = math.exp(min(0.0, 1.0 - r_len / max(1, h_len)))
        bleu    = 100.0 * bp * math.exp(log_avg)

    bp = math.exp(min(0.0, 1.0 - r_len / max(1, h_len)))
    return bleu, prec, bp, h_len, r_len


def tf_epsilon(step: int, k: float = K_TF) -> float:
    """Inverse-sigmoid teacher-forcing schedule: ε = k / (k + exp(step/k))."""
    try:
        return k / (k + math.exp(step / k))
    except OverflowError:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  Log parser
# ══════════════════════════════════════════════════════════════════════════════

_LOG_RE = re.compile(
    r"(?P<time>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \| INFO \| "
    r"ep (?P<ep>\d+)/(?P<max_ep>\d+) \| step\s+(?P<step>\d+) \| "
    r"loss=(?P<loss>[\d.]+) \| lr=(?P<lr>[\d.e+\-]+) \| "
    r"gnorm=(?P<gnorm>[\d.]+) \| ε=(?P<eps>[\d.]+) \| "
    r"tok/s=(?P<toks>[\d,]+)"
)


def _parse_logs(path: str) -> list:
    """
    Parse training log file.

    Handles two known formatting artefacts:
    - Concatenated lines (tok/s value runs directly into next timestamp)
    - Duplicate step entries (from session restarts mid-epoch)

    Returns a list of dicts sorted by (step, time), de-duplicated to keep the
    last record for each step (latest session's value wins).
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        raw = fh.read()

    # Split concatenated lines: find positions where a new timestamp starts
    # in the middle of a line and insert a newline there.
    lines = []
    for raw_line in raw.splitlines():
        parts = re.split(r"(?=\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", raw_line)
        for p in parts:
            p = p.strip()
            if p:
                lines.append(p)

    seen: dict = {}   # step → record (last-writer wins)
    for line in lines:
        m = _LOG_RE.search(line)
        if not m:
            continue
        d = m.groupdict()
        toks_str = d["toks"].replace(",", "")
        # Strip any timestamp digits that leaked into the tok/s field
        if len(toks_str) > 6:
            toks_str = toks_str[:4]
        try:
            record = {
                "time":  d["time"],
                "ep":    int(d["ep"]),
                "step":  int(d["step"]),
                "loss":  float(d["loss"]),
                "lr":    float(d["lr"]),
                "gnorm": float(d["gnorm"]),
                "eps":   float(d["eps"]),
                "toks":  int(toks_str),
            }
        except (ValueError, KeyError):
            continue
        seen[record["step"]] = record   # keep last occurrence

    return sorted(seen.values(), key=lambda r: r["step"])


# ══════════════════════════════════════════════════════════════════════════════
#  Training analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_logs() -> None:
    div("TRAINING LOG ANALYSIS")

    if not os.path.exists(LOG_PATH):
        print(f"  [ERROR] Log file not found: {LOG_PATH}")
        return

    logs = _parse_logs(LOG_PATH)
    if not logs:
        print("  [ERROR] No valid log records found.")
        return

    first  = logs[0]
    last   = logs[-1]
    best   = min(logs, key=lambda r: r["loss"])
    worst  = max(logs, key=lambda r: r["loss"])
    all_losses  = [r["loss"]  for r in logs]
    all_gnorms  = [r["gnorm"] for r in logs]
    all_toks    = [r["toks"]  for r in logs]

    # ── Overview ─────────────────────────────────────────────────────────────
    sub("Overview")
    row("Records parsed:",        len(logs))
    row("Training start:",        first["time"])
    row("Training end:",          last["time"])
    row("Step range:",            f"{first['step']:,} → {last['step']:,}")
    row("Epochs covered:",        sorted({r["ep"] for r in logs}))
    row("Initial loss:",          f"{first['loss']:.4f}  (step {first['step']:,})")
    row("Final loss:",            f"{last['loss']:.4f}  (step {last['step']:,})")
    row("Best loss (gall time):",  f"{best['loss']:.4f}  (step {best['step']:,}, epoch {best['ep']})")
    row("Worst loss (all time):", f"{worst['loss']:.4f}  (step {worst['step']:,})")
    row("Overall loss drop:",     f"{first['loss'] - last['loss']:+.4f}  ({(first['loss']-last['loss'])/first['loss']*100:.1f}% reduction)")

    # ── Loss statistics ───────────────────────────────────────────────────────
    sub("Loss Distribution")
    row("Mean loss:",    f"{statistics.mean(all_losses):.4f}")
    row("Median loss:",  f"{statistics.median(all_losses):.4f}")
    row("Std deviation:",f"{statistics.stdev(all_losses):.4f}")
    quartiles = statistics.quantiles(all_losses, n=4)
    row("Q1 / Q2 / Q3:", f"{quartiles[0]:.4f} / {quartiles[1]:.4f} / {quartiles[2]:.4f}")

    # ── Gradient health ───────────────────────────────────────────────────────
    sub("Gradient Norm Health")
    clip_thresh = 5.0
    clipped_count = sum(1 for g in all_gnorms if g > clip_thresh)
    spike_count   = sum(1 for g in all_gnorms if g > 8.0)
    row("Mean gnorm:",    f"{statistics.mean(all_gnorms):.3f}")
    row("Median gnorm:",  f"{statistics.median(all_gnorms):.3f}")
    row("Max gnorm:",     f"{max(all_gnorms):.3f}  (step {max(logs, key=lambda r: r['gnorm'])['step']:,})")
    row("Steps clipped (>5.0):", f"{clipped_count} / {len(logs)}  ({clipped_count/len(logs)*100:.1f}%)")
    row("Severe spikes (>8.0):", f"{spike_count} / {len(logs)}  ({spike_count/len(logs)*100:.1f}%)")
    if spike_count > 0:
        spike_steps = [r["step"] for r in logs if r["gnorm"] > 8.0]
        print(f"    Spike steps: {spike_steps}")

    # ── Throughput ────────────────────────────────────────────────────────────
    sub("Throughput")
    row("Mean tok/s:",   f"{statistics.mean(all_toks):,.0f}")
    row("Median tok/s:", f"{statistics.median(all_toks):,.0f}")
    row("Peak tok/s:",   f"{max(all_toks):,}  (step {max(logs, key=lambda r: r['toks'])['step']:,})")
    row("Min tok/s:",    f"{min(all_toks):,}  (step {min(logs, key=lambda r: r['toks'])['step']:,})")
    # Breakdown by percentile
    toks_sorted = sorted(all_toks)
    n = len(toks_sorted)
    p10 = toks_sorted[int(n * 0.10)]
    p90 = toks_sorted[int(n * 0.90)]
    row("P10 / P90 tok/s:", f"{p10:,} / {p90:,}")
    print(f"\n  Note: Low tok/s at the end of an epoch is normal — the bucket")
    print(f"  sorter serves increasingly longer sentences as the epoch progresses.")
    print(f"  Throughput resets to peak at the start of each new epoch (re-shuffle).")

    # ── Session restart detection ─────────────────────────────────────────────
    sub("Session Restart Detection")
    print("  (Restarts cause Adam cold-start → temporary loss spikes.)")
    print("  Detected by: tok/s drop + loss spike at the same boundary.\n")

    raw_with_dupes: dict = defaultdict(list)
    # Re-parse to catch all duplicates
    with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as fh:
        raw_text = fh.read()
    raw_lines = []
    for raw_line in raw_text.splitlines():
        parts = re.split(r"(?=\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", raw_line)
        for p in parts:
            if p.strip():
                raw_lines.append(p.strip())
    for line in raw_lines:
        m = _LOG_RE.search(line)
        if m:
            d = m.groupdict()
            raw_with_dupes[int(d["step"])].append(d["time"])

    restart_steps = sorted(s for s, times in raw_with_dupes.items() if len(times) > 1)
    if restart_steps:
        row("Restart-boundary steps:", len(restart_steps))
        for s in restart_steps[:10]:
            times = raw_with_dupes[s]
            print(f"    Step {s:,} appeared {len(times)}× → sessions: {', '.join(times)}")
        if len(restart_steps) > 10:
            print(f"    ... and {len(restart_steps) - 10} more.")
    else:
        print("  No duplicate step entries found — training ran continuously.")

    # Loss spikes: consecutive records where loss jumps > 0.5 NLL
    print()
    spike_events = []
    for i in range(1, len(logs)):
        delta = logs[i]["loss"] - logs[i - 1]["loss"]
        if delta > 0.5:
            spike_events.append((logs[i]["step"], delta, logs[i - 1]["loss"], logs[i]["loss"]))
    if spike_events:
        row("Loss spikes detected:", len(spike_events))
        for step, delta, before, after in spike_events:
            print(f"    Step {step:,}: {before:.4f} → {after:.4f}  (Δ={delta:+.4f})")
    else:
        print("  No significant loss spikes detected (threshold: +0.5 NLL).")

    # ── Per-epoch breakdown ───────────────────────────────────────────────────
    sub("Per-Epoch Breakdown")
    epochs: dict = defaultdict(list)
    for r in logs:
        epochs[r["ep"]].append(r)

    header = f"  {'Epoch':>5}  {'Steps':>12}  {'Loss Start':>10}  {'Loss End':>8}  {'Best':>7}  {'Mean gnorm':>10}  {'Mean tok/s':>10}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    for ep in sorted(epochs):
        entries = epochs[ep]
        losses  = [e["loss"]  for e in entries]
        gnorms  = [e["gnorm"] for e in entries]
        toks    = [e["toks"]  for e in entries]
        steps   = [e["step"]  for e in entries]
        print(
            f"  {ep:>5}  "
            f"{min(steps):>6,} – {max(steps):<6,}  "
            f"{entries[0]['loss']:>10.4f}  "
            f"{entries[-1]['loss']:>8.4f}  "
            f"{min(losses):>7.4f}  "
            f"{statistics.mean(gnorms):>10.3f}  "
            f"{statistics.mean(toks):>10,.0f}"
        )
        # Flag epochs with instability
        if max(gnorms) > 8.0:
            print(f"         ⚠  Epoch {ep}: max gnorm={max(gnorms):.2f} — gradient instability detected")
        if entries[-1]["loss"] > entries[0]["loss"]:
            print(f"         ⚠  Epoch {ep}: loss INCREASED this epoch ({entries[0]['loss']:.4f} → {entries[-1]['loss']:.4f})")

    # ── Teacher-forcing schedule ──────────────────────────────────────────────
    sub("Teacher-Forcing Schedule  (k = {:,.0f})".format(K_TF))
    print(f"  Formula: ε = k / (k + exp(step / k))\n")
    milestones = [
        (0,       "Training start"),
        (46_000,  "End of epoch 1 (approx)"),
        (100_000, "~2 epochs"),
        (200_000, "~4 epochs  ← 50/50 crossover with k=20,000"),
        (500_000, "~10 epochs ← ε ≈ 0"),
        (last["step"], f"Current step ({last['step']:,})"),
    ]
    for step, label in milestones:
        eps = tf_epsilon(step, K_TF)
        bar_len = int(eps * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  Step {step:>8,}  ε={eps:.4f}  [{bar}]  {label}")

    logged_eps = sorted(set(r["eps"] for r in logs))
    print(f"\n  Logged ε values in this run: {logged_eps}")
    if max(logged_eps) >= 0.999:
        print(f"  → Model is still almost entirely in teacher-forcing mode.")
        print(f"  → Transition to autonomous generation begins ~step 200,000.")


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation analysis
# ══════════════════════════════════════════════════════════════════════════════

def analyze_eval(top_n: int = 5) -> None:
    div("EVALUATION ANALYSIS")

    if not os.path.exists(EVAL_PATH):
        print(f"  [SKIP] {EVAL_PATH} not found.")
        return

    with open(EVAL_PATH, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()

    # ── Checkpoint info ───────────────────────────────────────────────────────
    ckpt_m = re.search(r"Checkpoint loaded <-\s*(.*)", text)
    step_m = re.search(r"step[_\s]+(\d+)", ckpt_m.group(1)) if ckpt_m else None
    print(f"\n  Checkpoint: {ckpt_m.group(1).strip() if ckpt_m else 'unknown'}")

    # ── Parse sentence entries ────────────────────────────────────────────────
    entry_re = re.compile(
        r"\[\s*(?P<idx>\d+)\]\s+EN:\s*(?P<en>.*?)\n\s+HY:\s*(?P<hy>.*?)\n\s+RE:\s*(?P<re>.*?)\n\s+BLEU:\s*(?P<bleu>[\d.]+)",
        re.DOTALL,
    )
    entries = [
        {
            "idx":  int(m.group("idx")),
            "en":   m.group("en").strip(),
            "hy":   m.group("hy").strip(),
            "re":   m.group("re").strip(),
            "bleu": float(m.group("bleu")),
        }
        for m in entry_re.finditer(text)
    ]

    n = len(entries)
    if n == 0:
        print("  [ERROR] No sentence entries found in evaluation file.")
        return

    # ── Corpus BLEU ───────────────────────────────────────────────────────────
    hyps = [e["hy"].split()          for e in entries]
    refs = [e["re"].lower().split()  for e in entries]
    c_bleu, prec, bp, h_len, r_len = corpus_bleu(hyps, refs)

    sub(f"Corpus BLEU-4  ({n:,} sentences)")
    row("Corpus BLEU-4:",     f"{c_bleu:.2f} / 100.00")
    row("Brevity Penalty:",   f"{bp:.4f}")
    row("Hypothesis tokens:", f"{h_len:,}")
    row("Reference tokens:",  f"{r_len:,}")
    row("Length ratio:",      f"{h_len/r_len:.4f}  ({'under' if h_len < r_len else 'over'}-generation)")
    print()
    row("1-gram precision:", f"{prec[0]*100:.2f}%  ({int(prec[0]*sum(max(0,len(h)-0) for h in hyps)):,} matches)")
    row("2-gram precision:", f"{prec[1]*100:.2f}%")
    row("3-gram precision:", f"{prec[2]*100:.2f}%")
    row("4-gram precision:", f"{prec[3]*100:.2f}%")

    # BLEU interpretation
    print()
    if c_bleu < 5:
        interp = "Early learning stage — basic structure emerging but largely unintelligible."
    elif c_bleu < 15:
        interp = "Rough but partially correct — recognisable translation fragments."
    elif c_bleu < 25:
        interp = "Good quality — human-readable output with minor errors."
    elif c_bleu < 35:
        interp = "Strong quality — approaching professional MT systems."
    else:
        interp = "Excellent — state-of-the-art for this architecture."
    print(f"  Interpretation: {interp}")

    # ── Sentence-level BLEU distribution ─────────────────────────────────────
    bleus = [e["bleu"] for e in entries]
    sub("Sentence-level BLEU Distribution")
    row("Mean sentence BLEU:",    f"{statistics.mean(bleus):.4f}")
    row("Median sentence BLEU:",  f"{statistics.median(bleus):.4f}")
    row("Std deviation:",         f"{statistics.stdev(bleus):.4f}")
    row("Max sentence BLEU:",     f"{max(bleus):.4f}  (sentence #{max(entries, key=lambda e: e['bleu'])['idx']})")
    print()

    zero_n    = sum(1 for b in bleus if b == 0.0)
    nonzero_n = n - zero_n
    buckets = [
        ("BLEU = 0.00",           sum(1 for b in bleus if b == 0.0)),
        ("BLEU 0.00 – 0.10",      sum(1 for b in bleus if 0.0 < b < 0.10)),
        ("BLEU 0.10 – 0.30",      sum(1 for b in bleus if 0.10 <= b < 0.30)),
        ("BLEU 0.30 – 0.60",      sum(1 for b in bleus if 0.30 <= b < 0.60)),
        ("BLEU ≥ 0.60",           sum(1 for b in bleus if b >= 0.60)),
    ]
    print("  Bucket distribution:")
    for label, count in buckets:
        pct  = count / n * 100
        bar  = "█" * int(pct / 2)
        print(f"    {label:<22} {count:>5} / {n}  ({pct:5.1f}%)  {bar}")

    print(f"\n  Note on zero-BLEU: Sentence BLEU-4 is 0.0 whenever there are no")
    print(f"  matching 4-grams with the reference, even if 1/2/3-gram matches exist.")
    print(f"  This is a well-known limitation of sentence-BLEU. {zero_n} sentences")
    print(f"  ({zero_n/n*100:.1f}%) score 0.0 at sentence-level but may be semantically correct.")

    # ── Length analysis ───────────────────────────────────────────────────────
    sub("Translation Length Analysis")
    hyp_lens = [len(e["hy"].split()) for e in entries]
    ref_lens  = [len(e["re"].split()) for e in entries]
    src_lens  = [len(e["en"].split()) for e in entries]
    row("Mean source length (EN):",    f"{statistics.mean(src_lens):.1f} tokens")
    row("Mean hypothesis length:",     f"{statistics.mean(hyp_lens):.1f} tokens")
    row("Mean reference length (VI):", f"{statistics.mean(ref_lens):.1f} tokens")
    row("Hyp/Ref length ratio:",       f"{statistics.mean(hyp_lens)/statistics.mean(ref_lens):.3f}")

    # Length-stratified BLEU
    print()
    print("  BLEU by source sentence length:")
    buckets_len = [(1, 10), (10, 20), (20, 35), (35, 60), (60, 9999)]
    for lo, hi in buckets_len:
        subset = [e for e in entries if lo <= len(e["en"].split()) < hi]
        if not subset:
            continue
        sub_bleu = statistics.mean(e["bleu"] for e in subset)
        label = f"    EN length {lo:>3}–{min(hi-1,999):<3}"
        print(f"  {label}  {len(subset):>5} sentences  mean BLEU = {sub_bleu:.4f}")

    # ── n-gram precision drill-down ───────────────────────────────────────────
    sub("N-gram Precision Drill-down")
    for n_val in [1, 2, 3, 4]:
        matched = total = 0
        for hyp, ref in zip(hyps, refs):
            h_ng = _ngrams(hyp, n_val)
            r_ng = _ngrams(ref, n_val)
            matched += sum(min(c, r_ng.get(ng, 0)) for ng, c in h_ng.items())
            total   += max(0, len(hyp) - n_val + 1)
        p = matched / total if total > 0 else 0.0
        bar = "█" * int(p * 40)
        print(f"  {n_val}-gram:  {p*100:5.2f}%  [{bar:<40}]  {matched:,} / {total:,}")

    # ── Top translations ──────────────────────────────────────────────────────
    sub(f"Top-{top_n} Translations by Sentence BLEU")
    ranked = sorted(entries, key=lambda e: e["bleu"], reverse=True)
    for i, e in enumerate(ranked[:top_n], 1):
        print(f"  #{i:02d}  [sent {e['idx']:>5}]  BLEU = {e['bleu']:.4f}")
        print(f"       EN : {e['en'][:110]}")
        print(f"       HY : {e['hy'][:110]}")
        print(f"       REF: {e['re'][:110]}")
        print()

    # ── Zero-BLEU with partial word overlap ──────────────────────────────────
    sub("Zero-BLEU Sentences with Highest Word Overlap (Exposure Bias Candidates)")
    print("  These sentences score BLEU-4 = 0.0 but have significant word-level")
    print("  overlap — likely caused by pronoun/particle mismatch in Vietnamese.\n")
    partial = []
    for e in entries:
        if e["bleu"] > 0.0:
            continue
        h_words = set(e["hy"].lower().split())
        r_words = set(e["re"].lower().split())
        overlap = len(h_words & r_words) / max(1, len(h_words))
        if overlap > 0.35 and len(h_words) >= 4:
            partial.append((overlap, e))
    partial.sort(key=lambda x: x[0], reverse=True)

    if not partial:
        print("  None found (threshold: ≥35% word overlap, ≥4 hypothesis words).")
    else:
        for i, (ov, e) in enumerate(partial[:top_n], 1):
            print(f"  #{i:02d}  [sent {e['idx']:>5}]  Word overlap = {ov*100:.1f}%  (BLEU-4 = 0.0)")
            print(f"       EN : {e['en'][:110]}")
            print(f"       HY : {e['hy'][:110]}")
            print(f"       REF: {e['re'][:110]}")
            print()

    # ── Unique word analysis ──────────────────────────────────────────────────
    sub("Vocabulary Coverage")
    all_hyp_words = [w for e in entries for w in e["hy"].lower().split()]
    all_ref_words = [w for e in entries for w in e["re"].lower().split()]
    hyp_vocab = set(all_hyp_words)
    ref_vocab  = set(all_ref_words)
    row("Unique words in hypotheses:", f"{len(hyp_vocab):,}")
    row("Unique words in references:", f"{len(ref_vocab):,}")
    row("Vocab coverage:",             f"{len(hyp_vocab & ref_vocab):,} / {len(ref_vocab):,}  ({len(hyp_vocab & ref_vocab)/len(ref_vocab)*100:.1f}%)")
    row("Hyp words not in ref:",       f"{len(hyp_vocab - ref_vocab):,}")

    top_hyp = Counter(all_hyp_words).most_common(10)
    top_ref  = Counter(all_ref_words).most_common(10)
    print(f"\n  Most common hypothesis words: {[w for w, _ in top_hyp]}")
    print(f"  Most common reference words:  {[w for w, _ in top_ref]}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Professional training & evaluation analyser for Tensor Engine NMT"
    )
    parser.add_argument("--logs-only", action="store_true", help="Run training log analysis only")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation analysis only")
    parser.add_argument("--top",       type=int, default=5,  help="Number of top/sample translations to show (default: 5)")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║         Tensor Engine NMT — Training & Evaluation Analyser          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    if not args.eval_only:
        analyze_logs()

    if not args.logs_only:
        analyze_eval(top_n=args.top)

    print("\n" + "━" * 70)
    print("  Analysis complete.")
    print("━" * 70 + "\n")


if __name__ == "__main__":
    main()
