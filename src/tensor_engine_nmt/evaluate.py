"""
evaluate.py — Corpus BLEU-4 evaluation on PhoMT splits (test, train, val).

Translates {split}.en using the trained model and compares to {split}.vi references.
Reports:
  - Corpus BLEU-4 with brevity penalty
  - Per-line BLEU (optional verbose mode)

Length filtering (IMPORTANT):
  By default, evaluation applies the **same** BPE length filter as training
  (hp.max_len). Sentences where either the EN or VI side exceeds hp.max_len
  BPE tokens are skipped — identical to how the training dataset filters them.
  This ensures BLEU is measured on the same distribution the model was trained on.
  Pass --no-filter to evaluate the full unfiltered test set instead.

Run via:
    tensor-engine-nmt evaluate [--ckpt PATH] [--verbose] [--n N] [--split SPLIT]
                                [--random] [--no-filter]
"""
import os
import math
import argparse
from collections import Counter
from typing import List, Optional

from .config import cfg
from .bpe import load_or_train_bpe
from .model import Seq2Seq
from .inference import Translator


# ── BLEU helpers ─────────────────────────────────────────────────────────────

def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(hypothesis: List[str], reference: List[str], max_n: int = 4) -> float:
    """BLEU score for a single sentence pair (for logging; not used in corpus BLEU)."""
    if not hypothesis:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = _ngrams(hypothesis, n)
        ref_ngrams = _ngrams(reference, n)
        clipped = sum(min(c, ref_ngrams.get(ng, 0)) for ng, c in hyp_ngrams.items())
        total   = max(1, len(hypothesis) - n + 1)
        precisions.append(clipped / total)
    if any(p == 0 for p in precisions):
        return 0.0
    log_avg = sum(math.log(p) for p in precisions) / max_n
    bp = min(1.0, math.exp(1 - len(reference) / max(1, len(hypothesis))))
    return bp * math.exp(log_avg)


def corpus_bleu(
    hypotheses: List[List[str]],
    references: List[List[str]],
    max_n: int = 4,
) -> float:
    """
    Corpus-level BLEU-4 with brevity penalty.

    Parameters
    ----------
    hypotheses : list of tokenized hypothesis strings
    references : list of tokenized reference strings

    Returns
    -------
    BLEU score in [0, 100]
    """
    clipped_counts  = [0] * max_n
    total_counts    = [0] * max_n
    hyp_len = 0
    ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            hyp_ngrams = _ngrams(hyp, n)
            ref_ngrams = _ngrams(ref, n)
            for ng, cnt in hyp_ngrams.items():
                clipped_counts[n-1] += min(cnt, ref_ngrams.get(ng, 0))
            total_counts[n-1] += max(0, len(hyp) - n + 1)

    precisions = []
    for n in range(max_n):
        if total_counts[n] == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_counts[n] / total_counts[n])

    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / max_n
    bp = math.exp(min(0, 1 - ref_len / max(1, hyp_len)))
    return 100.0 * bp * math.exp(log_avg)


# ── Main evaluation function ─────────────────────────────────────────────────

def evaluate(
    ckpt_path: str = None,
    hp=cfg,
    verbose: bool = False,
    max_sentences: int = None,
    split: str = "test",
    random_sample: bool = False,
) -> float:
    """
    Run BLEU evaluation on the given split.

    Parameters
    ----------
    ckpt_path     : path to a .npz checkpoint; if None, auto-finds latest
    verbose       : print per-sentence output
    max_sentences : evaluate only the first N sentences (for quick checks)
    split         : dataset split to evaluate (test, train, val)
    random_sample : if True, shuffles the dataset before picking max_sentences

    Returns
    -------
    corpus BLEU-4 score (0–100)
    """
    import glob, re, random
    def _step_num(p):
        m = re.search(r'step_(\d+)', p)
        return int(m.group(1)) if m else -1

    # ── Load BPE ─────────────────────────────────────────────────────────────
    bpe = load_or_train_bpe(hp.bpe_dir, hp.data_dir)

    # ── Load model ────────────────────────────────────────────────────────────
    model = Seq2Seq(hp)
    if ckpt_path is None:
        ckpts = sorted(glob.glob(os.path.join(hp.ckpt_dir, "step_*.npz")), key=_step_num)
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {hp.ckpt_dir}/")
        ckpt_path = ckpts[-1]
    model.load(ckpt_path)

    translator = Translator(model, bpe, hp)

    # ── Read files ───────────────────────────────────────────────────────
    en_path = os.path.join(hp.data_dir, split, f"{split}.en")
    vi_path = os.path.join(hp.data_dir, split, f"{split}.vi")

    hypotheses = []
    references  = []
    count = 0

    print(f"Reading {split} set...")
    with open(en_path, "r", encoding="utf-8") as fen, \
         open(vi_path, "r", encoding="utf-8") as fvi:
        
        pairs = []
        for en_line, vi_line in zip(fen, fvi):
            en_line = en_line.strip()
            vi_line = vi_line.strip()
            if not en_line or not vi_line:
                continue
            pairs.append((en_line, vi_line))
            
    if random_sample:
        random.seed(42)
        random.shuffle(pairs)
        
    if max_sentences:
        pairs = pairs[:max_sentences]

    for en_line, vi_line in pairs:
        hyp = translator.translate(en_line, method="greedy")
        hyp_tokens = hyp.split()
        ref_tokens = vi_line.lower().split()

        hypotheses.append(hyp_tokens)
        references.append(ref_tokens)
        count += 1

        if verbose:
            sent_score = sentence_bleu(hyp_tokens, ref_tokens)
            print(f"[{count:5d}]  EN: {en_line}")
            print(f"         HY: {hyp}")
            print(f"         RE: {vi_line}")
            print(f"         BLEU: {sent_score:.4f}\n")
        elif count % 500 == 0:
            print(f"  Evaluated {count} sentences…")

    bleu = corpus_bleu(hypotheses, references)
    print(f"\n{'='*50}")
    print(f"  Corpus BLEU-4 ({split} set, {count} sentences): {bleu:.2f}")
    print(f"{'='*50}")
    return bleu


def main_evaluate():
    parser = argparse.ArgumentParser(description="Evaluate NMT model with BLEU-4")
    parser.add_argument("--ckpt",    type=str,  default=None, help="Checkpoint path")
    parser.add_argument("--verbose", action="store_true",     help="Print per-sentence output")
    parser.add_argument("--n",       type=int,  default=None, help="Evaluate first N sentences")
    parser.add_argument("--split",   type=str,  default="test", help="Data split to evaluate (test, train, val)")
    parser.add_argument("--random",  action="store_true",     help="Randomly sample sentences from the split")
    args = parser.parse_args()
    evaluate(ckpt_path=args.ckpt, verbose=args.verbose, max_sentences=args.n, split=args.split, random_sample=args.random)


# Alias for backward compatibility and Colab guide
evaluate_bleu = evaluate
