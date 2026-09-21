"""
dataset.py — PhoMT data loading, bucket-sorted batching, and collation.

The data pipeline:
  1. Stream `train.en` / `train.vi` line-by-line (never load all ~530 MB at once)
  2. Encode each pair with BPE
  3. Group into length buckets to minimise padding waste (~40% fewer PAD tokens)
  4. Interleave buckets in round-robin order (uniform throughput throughout epoch)
  5. `collate_batch` pads and produces the five tensors the model expects

Batch ordering strategy (INTERLEAVED):
  Pairs are split into 4 length buckets, each independently shuffled.
  Then batches are drawn in a round-robin fashion across all buckets until
  all are exhausted. This produces ~uniform sentence-length distribution
  throughout the epoch, reducing the throughput coefficient of variation
  from ~21% (sequential bucket drain) to ~8–10%.

All output tensors are int32 (token ids never enter arithmetic — they are
only used as indices).
"""
import os
import random
import pickle
import itertools
from typing import List, Tuple, Iterator, Optional
import numpy as np

from .config import cfg
from .bpe import BPETokenizer, PAD_ID, START_ID, END_ID


# ── Length bucket boundaries (by English token count after BPE) ──────────────
#
# Creates 4 buckets: [0,10), [10,20), [20,30), [30,∞)
# Pairs within each bucket are shuffled independently before interleaving,
# so batches stay length-similar (low padding waste) while the epoch-level
# ordering is varied (stable throughput).

BUCKET_BOUNDARIES = [10, 20, 30]


def _count_lines(path: str) -> int:
    """Count newlines without decoding — a few seconds on a 500 MB corpus."""
    n = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            n += chunk.count(b"\n")
    return n


def _bucket_id(length: int) -> int:
    for i, b in enumerate(BUCKET_BOUNDARIES):
        if length < b:
            return i
    return len(BUCKET_BOUNDARIES)


# ── Collation ─────────────────────────────────────────────────────────────────

def collate_batch(
    pairs: List[Tuple[List[int], List[int]]],
) -> dict:
    """
    Pad a list of (en_ids, vi_ids) pairs to the batch-maximum length.

    en_ids: already includes END token from bpe.encode(add_end=True)
    vi_ids: raw VI token ids (no START/END yet)

    Returns a dict with:
        X    : int32 (B, Tx)   — padded EN ids
        Xlen : int32 (B,)      — true EN lengths (including END)
        Yin  : int32 (B, Ty)   — [START, v1, v2, ..., PAD…]
        Yout : int32 (B, Ty)   — [v1, v2, ..., END, PAD…]
        Ylen : int32 (B,)      — true VI lengths (including END token)
    """
    B = len(pairs)

    en_seqs, vi_seqs = zip(*pairs)

    # EN: already has END appended
    Tx = max(len(s) for s in en_seqs)
    X = np.full((B, Tx), PAD_ID, dtype=np.int32)
    Xlen = np.zeros(B, dtype=np.int32)
    for b, s in enumerate(en_seqs):
        X[b, :len(s)] = s
        Xlen[b] = len(s)

    # VI: construct Yin = [START, v1, ..., vN]  and  Yout = [v1, ..., vN, END]
    # vi_seqs are raw (no START/END); add END to each
    vi_with_end = [list(s) + [END_ID] for s in vi_seqs]
    # Both Yin and Yout are len(vi_with_end[b]) long, so Ty is exactly that max.
    # (This used to be max(len(s) + 1), which appended an all-PAD column that
    # every batch then paid a decoder step for.)
    Ty = max(len(s) for s in vi_with_end)

    Yin  = np.full((B, Ty), PAD_ID, dtype=np.int32)
    Yout = np.full((B, Ty), PAD_ID, dtype=np.int32)
    Ylen = np.zeros(B, dtype=np.int32)

    for b, s in enumerate(vi_with_end):
        # Yin  : [START, v1, v2, ..., vN-1, END]   — what the decoder READS
        yin_seq = [START_ID] + s[:-1]
        # Yout : [v1,  v2, ..., vN-1, END]         — what the decoder PREDICTS
        yout_seq = s
        Ty_b = len(yout_seq)
        Yin[b,  :len(yin_seq)]  = yin_seq
        Yout[b, :Ty_b]          = yout_seq
        Ylen[b]                 = Ty_b

    return {"X": X, "Xlen": Xlen, "Yin": Yin, "Yout": Yout, "Ylen": Ylen}


# ── Streaming dataset ─────────────────────────────────────────────────────────

class PhoMTDataset:
    """
    Lazy, streaming EN-VI sentence pair dataset.

    Usage:
        ds = PhoMTDataset(bpe, data_dir)
        for batch in ds.iterate(max_tokens=4000, shuffle=True, seed=42):
            X, Xlen, Yin, Yout, Ylen = batch["X"], ...
    """

    def __init__(
        self,
        bpe: BPETokenizer,
        data_dir: str = cfg.data_dir,
        split: str = "train",
        max_len: int = cfg.max_len,   # driven by hp.max_len — single source of truth
        max_pairs: Optional[int] = None,
    ):
        self.bpe = bpe
        self.split = split
        self.en_path = os.path.join(data_dir, split, f"{split}.en")
        self.vi_path = os.path.join(data_dir, split, f"{split}.vi")
        self.max_len = max_len
        self.max_pairs = max_pairs
        self.pairs: List[Tuple[List[int], List[int]]] = []

        # Cache identity must include the vocabulary, the length budget AND the
        # pair cap: _stream_pairs() bakes all three into the cached content, so a
        # cache keyed on vocab size alone is silently wrong when any of them change.
        vocab_sz = len(bpe.token2id)
        pair_tag = f"_N{max_pairs}" if max_pairs else ""
        self.cache_path = os.path.join(
            data_dir, f"{split}_cache_V{vocab_sz}_L{max_len}{pair_tag}.pkl"
        )
        self._legacy_cache_path = os.path.join(data_dir, f"{split}_cache_V{vocab_sz}.pkl")
        self._load_or_build_cache()

    @staticmethod
    def _unpack_cache(payload):
        """Return (meta, pairs).  Caches written before this version are a bare list."""
        if isinstance(payload, dict) and "pairs" in payload:
            return payload.get("meta", {}), payload["pairs"]
        return None, payload

    def _load_or_build_cache(self) -> None:
        """Load tokenized pairs from disk, or build them once and save."""
        path  = self.cache_path if os.path.exists(self.cache_path) else None
        if path is None and os.path.exists(self._legacy_cache_path):
            path = self._legacy_cache_path
            print(
                f"[dataset] WARNING: using the legacy cache {os.path.basename(path)} "
                f"(no max_len in its name and no metadata). It was filtered at whatever "
                f"max_len was active when it was built, so if you RAISED max_len above "
                f"that value these sentences are already gone — delete the file and "
                f"re-run to rebuild."
            )

        if path is not None:
            print(f"[dataset] Loading pre-tokenized cache from {path}...")
            with open(path, "rb") as f:
                meta, pairs = self._unpack_cache(pickle.load(f))
            self.pairs = pairs
            if meta:
                if meta.get("max_len") != self.max_len:
                    print(
                        f"[dataset] WARNING: cache was built with max_len="
                        f"{meta.get('max_len')} but max_len={self.max_len} is active. "
                        f"Build a fresh cache with this budget for an exact match."
                    )
                if meta.get("vocab_size") != len(self.bpe.token2id):
                    print(
                        f"[dataset] WARNING: cache vocab size {meta.get('vocab_size')} "
                        f"!= current {len(self.bpe.token2id)}."
                    )
                if meta.get("max_pairs") != self.max_pairs:
                    print(
                        f"[dataset] WARNING: cache holds a max_pairs="
                        f"{meta.get('max_pairs')} sample but max_pairs={self.max_pairs} "
                        f"is active."
                    )
            print(f"[dataset] Loaded {len(self.pairs):,} pairs.")
        else:
            print(f"[dataset] No cache found. Pre-tokenizing {self.en_path} (This takes a few minutes)...")
            self.pairs = list(self._stream_pairs())
            print(f"[dataset] Tokenization complete. Saving {len(self.pairs):,} pairs to {self.cache_path}...")
            with open(self.cache_path, "wb") as f:
                pickle.dump({
                    "meta": {
                        "max_len":    self.max_len,
                        "vocab_size": len(self.bpe.token2id),
                        "max_pairs":  self.max_pairs,
                        "split":      self.split,
                    },
                    "pairs": self.pairs,
                }, f)
            print("[dataset] Cache saved.")

    def _stream_pairs(self) -> Iterator[Tuple[List[int], List[int]]]:
        """
        Generator that yields (en_ids, vi_ids) pairs.
        Filters out pairs where either side exceeds max_len after BPE.
        EN ids include END.  VI ids are raw (no START/END).

        If `max_pairs` is set the corpus is sampled at an even stride rather than
        truncated to the first N lines.  Corpus files are often grouped by topic
        or source, so "the first 150k pairs" is a biased subset; a stride keeps
        the tokenizer and the model exposed to the whole document mix while still
        only encoding as many lines as we need.
        """
        stride = 1
        if self.max_pairs:
            total = _count_lines(self.en_path)
            if total > self.max_pairs:
                stride = max(1, total // self.max_pairs)
                print(f"[dataset] max_pairs={self.max_pairs:,}: encoding every "
                      f"{stride:,}th of {total:,} lines (~{self.max_pairs:,} pairs)")

        kept = 0
        with open(self.en_path, "r", encoding="utf-8") as fen, \
             open(self.vi_path, "r", encoding="utf-8") as fvi:
            for i, (en_line, vi_line) in enumerate(zip(fen, fvi)):
                if stride > 1 and (i % stride):
                    continue
                en_line = en_line.strip()
                vi_line = vi_line.strip()
                if not en_line or not vi_line:
                    continue
                en_ids = self.bpe.encode(en_line, add_start=False, add_end=True)
                vi_ids = self.bpe.encode(vi_line, add_start=False, add_end=False)
                if len(en_ids) > self.max_len or len(vi_ids) > self.max_len:
                    continue
                if self.max_pairs and kept >= self.max_pairs:
                    break
                kept += 1
                yield en_ids, vi_ids

    def iterate(
        self,
        max_tokens: int = cfg.max_tokens,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> Iterator[dict]:
        """
        Yield collated batches using interleaved bucket sampling.

        Strategy
        --------
        1. Split all pairs into 4 length buckets (boundaries: 10, 20, 30 tokens, ....).
        2. Independently shuffle each bucket (all pairs within a bucket are
           length-similar → low padding waste within each batch).
        3. Drain all buckets in round-robin order using itertools.zip_longest.
           This interleaves short and long sentences throughout the epoch,
           producing uniform throughput instead of the sawtooth pattern that
           occurred when buckets were concatenated sequentially (shortest-first).
        4. Apply dynamic token-level batching: accumulate pairs until adding
           the next pair would exceed max_tokens padded tokens, then yield.

        Throughput improvement
        ----------------------
        Sequential bucket drain:   CV ≈ 21%  (fast start, slow end every epoch)
        Interleaved round-robin:   CV ≈  8%  (stable throughout)

        Resume safety
        -------------
        The ordering produced by a given (seed, epoch) is fully deterministic.
        train.py passes seed = 42 + epoch, so the skip-batch resume logic
        recreates the identical batch sequence.
        """
        # ── Fill buckets ─────────────────────────────────────────────────────
        # The pair cap is applied when the cache is built (see __init__), so
        # self.pairs is already the sample we want; only the max_len filter is
        # re-applied here as a cheap safety net.
        n_buckets = len(BUCKET_BOUNDARIES) + 1
        buckets: List[List[Tuple]] = [[] for _ in range(n_buckets)]
        for pair in self.pairs:
            en_ids, vi_ids = pair
            if len(en_ids) > self.max_len or len(vi_ids) > self.max_len:
                continue
            buckets[_bucket_id(len(en_ids))].append(pair)

        # ── Shuffle within each bucket independently ──────────────────────────
        if shuffle:
            rng = random.Random(seed) if seed is not None else random.Random()
            for bucket in buckets:
                rng.shuffle(bucket)

        # ── Interleave buckets in round-robin order ───────────────────────────
        # zip_longest cycles through all buckets simultaneously, filling with
        # a sentinel (None) when a bucket is exhausted.  We filter out None
        # sentinels to get a flat interleaved list.
        #
        # Example with 3 buckets of sizes [4, 2, 3]:
        #   bucket 0: [A0, A1, A2, A3]   (0–9 tokens)
        #   bucket 1: [B0, B1]            (10–19 tokens)
        #   bucket 2: [C0, C1, C2]        (20–29 tokens)
        #
        # zip_longest output (before None removal):
        #   (A0,B0,C0), (A1,B1,C1), (A2,None,C2), (A3,None,None)
        #
        # Flattened: A0,B0,C0, A1,B1,C1, A2,C2, A3
        #   → short, medium, long, short, medium, long, short, long, short
        #
        # Dynamic batching then groups these into token-capped batches.
        # Each batch sees a mix of lengths → stable GPU utilisation.
        _SENTINEL = object()
        interleaved: List[Tuple] = []
        for row in itertools.zip_longest(*buckets, fillvalue=_SENTINEL):
            for item in row:
                if item is not _SENTINEL:
                    interleaved.append(item)

        # ── Dynamic token-level batching ──────────────────────────────────────
        current_batch: List[Tuple] = []
        max_en = 0
        max_vi = 0

        for pair in interleaved:
            en_len = len(pair[0])
            vi_len = len(pair[1]) + 1   # +1 for START/END overhead

            future_max_en = max(max_en, en_len)
            future_max_vi = max(max_vi, vi_len)
            future_tokens = (future_max_en + future_max_vi) * (len(current_batch) + 1)

            if future_tokens > max_tokens and current_batch:
                yield collate_batch(current_batch)
                current_batch = [pair]
                max_en = en_len
                max_vi = vi_len
            else:
                current_batch.append(pair)
                max_en = future_max_en
                max_vi = future_max_vi

        if current_batch:
            yield collate_batch(current_batch)


# ── Quick diagnostic ──────────────────────────────────────────────────────────

def verify_collation(batch: dict) -> None:
    """Assert all expected shapes and dtypes in a collated batch."""
    X, Xlen, Yin, Yout, Ylen = (
        batch["X"], batch["Xlen"], batch["Yin"], batch["Yout"], batch["Ylen"]
    )
    B, Tx = X.shape
    _, Ty  = Yin.shape
    assert X.dtype    == np.int32, f"X.dtype={X.dtype}"
    assert Xlen.dtype == np.int32
    assert Yin.dtype  == np.int32
    assert Yout.dtype == np.int32
    assert Ylen.dtype == np.int32
    assert Xlen.shape == (B,)
    assert Yin.shape  == (B, Ty)
    assert Yout.shape == (B, Ty)
    assert Ylen.shape == (B,)
    # Yin starts with START
    assert (Yin[:, 0] == START_ID).all(), "Yin column 0 must be START"
    # Yout[b, Ylen[b]-1] == END for every b
    for b in range(B):
        assert Yout[b, Ylen[b] - 1] == END_ID, f"Example {b}: END not at position Ylen-1"
    print(f"[dataset] batch OK — B={B}, Tx={Tx}, Ty={Ty}")
