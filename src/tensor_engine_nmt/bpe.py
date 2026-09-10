"""
bpe.py — Byte-Pair Encoding tokenizer (Sennrich et al., 2016).

Fast incremental implementation:
  - Vocab stored as tuple-of-symbols for O(1) indexing
  - After each merge, only pairs that TOUCH the merged bigram are updated
    → O(merge × avg_word_len) instead of O(merge × total_tokens)
  - On 5000 lines: ~30s for V=2000, ~3 min for V=32000

Trained jointly on EN + VI corpora (shared vocabulary).

Special token ids:
    PAD = 0,  UNK = 1,  START = 2,  END = 3
"""
import sys
import json
import os
import re
from collections import Counter, defaultdict
from typing import List, Optional

# Force UTF-8 so Vietnamese chars in progress prints don't crash cp1252.
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from .config import cfg


# ── Constants ─────────────────────────────────────────────────────────────────

WORD_END = "</w>"            # ASCII-safe end-of-word marker
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[START]", "[END]"]
PAD_ID, UNK_ID, START_ID, END_ID = 0, 1, 2, 3


# ── Helpers ───────────────────────────────────────────────────────────────────

def _word_to_chars(word: str) -> tuple:
    """Split a word into a tuple of characters, appending WORD_END to the last."""
    chars = list(word)
    if not chars:
        return (WORD_END,)
    chars[-1] = chars[-1] + WORD_END
    return tuple(chars)


def _get_pair_counts(vocab: dict) -> Counter:
    """
    Build full pair-frequency Counter from scratch.
    vocab: {tuple_of_symbols: frequency}
    """
    counts: Counter = Counter()
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            counts[(symbols[i], symbols[i + 1])] += freq
    return counts


def _update_pair_counts(
    counts: Counter,
    old_word: tuple,
    new_word: tuple,
    freq: int,
    pair_index: dict,
) -> None:
    """
    Incrementally update pair counts when old_word → new_word (after one merge).
    Only touches pairs that exist in old or new word — O(word_len).
    """
    # Remove counts from old word
    for i in range(len(old_word) - 1):
        pair = (old_word[i], old_word[i + 1])
        counts[pair] -= freq
        if counts[pair] <= 0:
            del counts[pair]
        if pair in pair_index and old_word in pair_index[pair]:
            pair_index[pair].discard(old_word)

    # Add counts from new word
    for i in range(len(new_word) - 1):
        pair = (new_word[i], new_word[i + 1])
        counts[pair] = counts.get(pair, 0) + freq
        pair_index.setdefault(pair, set()).add(new_word)


def _apply_merge(vocab: dict, best: tuple, counts: Counter, pair_index: dict) -> dict:
    """
    Apply one merge rule to the entire vocab, updating counts incrementally.
    Returns new vocab dict.
    """
    new_vocab = {}
    a, b = best
    merged = a + b

    # Collect words that contain the best pair (use index if available)
    if best in pair_index:
        affected = list(pair_index[best])
    else:
        affected = [w for w in vocab if best in zip(w, w[1:])]

    # Build replacement map for affected words
    replace_map = {}
    for word in affected:
        if word not in vocab:
            continue
        # Merge all occurrences of (a, b) in this word
        new_syms = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                new_syms.append(merged)
                i += 2
            else:
                new_syms.append(word[i])
                i += 1
        new_word = tuple(new_syms)
        replace_map[word] = new_word

    # Apply replacements and update counts
    for word, freq in vocab.items():
        if word in replace_map:
            new_word = replace_map[word]
            new_vocab[new_word] = new_vocab.get(new_word, 0) + freq
            _update_pair_counts(counts, word, new_word, freq, pair_index)
        else:
            new_vocab[word] = freq

    # Remove stale best-pair index entry
    if best in pair_index:
        del pair_index[best]

    return new_vocab


# ── BPE Tokenizer class ───────────────────────────────────────────────────────

class BPETokenizer:
    """
    Fast incremental BPE tokenizer with train, encode, decode, save, load.
    """

    def __init__(self):
        self.merges: List[tuple] = []
        self.token2id: dict = {}
        self.id2token: dict = {}
        self._merge_rank: dict = {}

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        files: List[str],
        target_vocab_size: int = cfg.V,
        sample_lines: int = cfg.bpe_sample_lines,
    ) -> None:
        """
        Train BPE jointly on all given text files.

        Streams up to `sample_lines` from each file. Uses incremental
        pair-count updates for speed.
        """
        print(f"[BPE] Reading up to {sample_lines:,} lines from {len(files)} file(s)...")

        # Step 1: word frequency table
        word_freq: Counter = Counter()
        for filepath in files:
            count = 0
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if count >= sample_lines:
                        break
                    for word in line.strip().split():
                        word_freq[word.lower()] += 1
                    count += 1
            print(f"  {filepath}: {count:,} lines read")

        # Step 2: init vocab as {tuple_of_chars: freq}
        vocab: dict = {}
        char_set: set = set()
        for word, freq in word_freq.items():
            if not word:
                continue
            chars = _word_to_chars(word)
            char_set.update(chars)
            vocab[chars] = vocab.get(chars, 0) + freq

        n_merges = max(0, target_vocab_size - 4 - len(char_set))
        print(f"[BPE] {len(char_set)} base chars. Learning {n_merges} merges...")

        # Step 3: initial pair counts + index
        counts   = _get_pair_counts(vocab)
        pair_index: dict = defaultdict(set)
        for word in vocab:
            for i in range(len(word) - 1):
                pair_index[(word[i], word[i + 1])].add(word)

        self.merges = []
        for step in range(n_merges):
            if not counts:
                break
            best = max(counts, key=lambda p: (counts[p], p))
            vocab = _apply_merge(vocab, best, counts, pair_index)
            self.merges.append(best)

            if (step + 1) % 1000 == 0:
                print(f"  {step+1}/{n_merges} merges done", flush=True)

        # Step 4: collect all tokens
        all_tokens: set = set()
        for word in vocab:
            all_tokens.update(word)
        all_tokens.update(char_set)

        self.token2id = {}
        self.id2token = {}
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.token2id[tok] = i
            self.id2token[i] = tok

        idx = len(SPECIAL_TOKENS)
        for tok in sorted(all_tokens):
            if tok not in self.token2id:
                self.token2id[tok] = idx
                self.id2token[idx] = tok
                idx += 1

        self._merge_rank = {m: i for i, m in enumerate(self.merges)}
        print(f"[BPE] Done. Final vocab size: {len(self.token2id):,}")

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _apply_merges_to_word(self, symbols: list) -> list:
        """Apply BPE merge rules greedily to a list of character symbols."""
        while True:
            best_rank = len(self.merges) + 1
            best_idx  = -1
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self._merge_rank.get(pair, len(self.merges) + 1)
                if rank < best_rank:
                    best_rank = rank
                    best_idx  = i
            if best_idx == -1:
                break
            a, b = symbols[best_idx], symbols[best_idx + 1]
            symbols = symbols[:best_idx] + [a + b] + symbols[best_idx + 2:]
        return symbols

    def encode(
        self,
        sentence: str,
        add_start: bool = False,
        add_end: bool = True,
    ) -> List[int]:
        """Encode a sentence string to a list of int32 token ids."""
        ids = []
        if add_start:
            ids.append(START_ID)
        for word in sentence.strip().lower().split():
            chars = list(_word_to_chars(word))
            tokens = self._apply_merges_to_word(chars)
            for tok in tokens:
                ids.append(self.token2id.get(tok, UNK_ID))
        if add_end:
            ids.append(END_ID)
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode a list of int ids back to a string."""
        tokens = []
        for id_ in ids:
            tok = self.id2token.get(id_, "[UNK]")
            if skip_special and tok in SPECIAL_TOKENS:
                continue
            tokens.append(tok)
        text = "".join(tokens).replace(WORD_END, " ").strip()
        return text

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "merges.txt"), "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")
        with open(os.path.join(directory, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)
        print(f"[BPE] Saved to {directory}/  ({len(self.merges):,} merges, {len(self.token2id):,} tokens)")

    @classmethod
    def load(cls, directory: str) -> "BPETokenizer":
        tok = cls()
        with open(os.path.join(directory, "merges.txt"), "r", encoding="utf-8") as f:
            tok.merges = []
            for line in f:
                parts = line.rstrip("\n").split(" ", 1)
                if len(parts) == 2:
                    tok.merges.append((parts[0], parts[1]))
        with open(os.path.join(directory, "vocab.json"), "r", encoding="utf-8") as f:
            tok.token2id = json.load(f)
        tok.id2token = {v: k for k, v in tok.token2id.items()}
        tok._merge_rank = {m: i for i, m in enumerate(tok.merges)}
        print(f"[BPE] Loaded from {directory}/  ({len(tok.merges):,} merges, {len(tok.token2id):,} tokens)")
        return tok


# ── Convenience function ──────────────────────────────────────────────────────

def load_or_train_bpe(
    bpe_dir: str = cfg.bpe_dir,
    data_dir: str = cfg.data_dir,
    target_vocab: int = cfg.V,
    sample_lines: int = cfg.bpe_sample_lines,
) -> BPETokenizer:
    """Load cached BPE vocab or train and save it."""
    if (os.path.exists(os.path.join(bpe_dir, "merges.txt")) and
            os.path.exists(os.path.join(bpe_dir, "vocab.json"))):
        return BPETokenizer.load(bpe_dir)

    train_en = os.path.join(data_dir, "train", "train.en")
    train_vi = os.path.join(data_dir, "train", "train.vi")
    tok = BPETokenizer()
    tok.train([train_en, train_vi], target_vocab_size=target_vocab, sample_lines=sample_lines)
    tok.save(bpe_dir)
    return tok
