"""
dataset.py — PhoMT data loading, bucket-sorted batching, and collation.

The data pipeline:
  1. Stream `train.en` / `train.vi` line-by-line (never load all ~530 MB at once)
  2. Encode each pair with BPE
  3. Group into length buckets to minimize padding waste (~40% fewer PAD tokens)
  4. Shuffle within each bucket and yield B-sized batches
  5. `collate_batch` pads and produces the five tensors the model expects

All output tensors are int32 (token ids never enter arithmetic — they are
only used as indices).
"""
import os
import random
import pickle
from typing import List, Tuple, Iterator, Optional
import numpy as np

from .config import cfg
from .bpe import BPETokenizer, PAD_ID, START_ID, END_ID


# ── Length bucket boundaries (by English token count after BPE) ──────────────

BUCKET_BOUNDARIES = [10, 20, 40, 80]   # creates 5 buckets: [0,10), [10,20), [20,40), [40,80), [80,∞)


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
    Ty = max(len(s) + 1 for s in vi_with_end)   # +1 for the START in Yin

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
        for batch in ds.iterate(batch_size=64, shuffle=True):
            X, Xlen, Yin, Yout, Ylen = batch["X"], ...
    """

    def __init__(
        self,
        bpe: BPETokenizer,
        data_dir: str = cfg.data_dir,
        split: str = "train",
        max_len: int = 100,
    ):
        self.bpe = bpe
        self.en_path = os.path.join(data_dir, split, f"{split}.en")
        self.vi_path = os.path.join(data_dir, split, f"{split}.vi")
        self.max_len = max_len
        self.pairs: List[Tuple[List[int], List[int]]] = []
        
        # Unique cache name based on vocab size to avoid stale token caches
        vocab_sz = len(bpe.token2id)
        self.cache_path = os.path.join(data_dir, f"{split}_cache_V{vocab_sz}.pkl")
        self._load_or_build_cache()

    def _load_or_build_cache(self) -> None:
        """Load tokenized pairs from disk, or build them once and save."""
        if os.path.exists(self.cache_path):
            print(f"[dataset] Loading pre-tokenized cache from {self.cache_path}...")
            with open(self.cache_path, "rb") as f:
                self.pairs = pickle.load(f)
            print(f"[dataset] Loaded {len(self.pairs):,} pairs.")
        else:
            print(f"[dataset] No cache found. Pre-tokenizing {self.en_path} (This takes a few minutes)...")
            self.pairs = list(self._stream_pairs())
            print(f"[dataset] Tokenization complete. Saving {len(self.pairs):,} pairs to {self.cache_path}...")
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.pairs, f)
            print("[dataset] Cache saved.")

    def _stream_pairs(self) -> Iterator[Tuple[List[int], List[int]]]:
        """
        Generator that yields (en_ids, vi_ids) pairs.
        Filters out pairs where either side exceeds max_len after BPE.
        EN ids include END.  VI ids are raw (no START/END).
        """
        with open(self.en_path, "r", encoding="utf-8") as fen, \
             open(self.vi_path, "r", encoding="utf-8") as fvi:
            for en_line, vi_line in zip(fen, fvi):
                en_line = en_line.strip()
                vi_line = vi_line.strip()
                if not en_line or not vi_line:
                    continue
                en_ids = self.bpe.encode(en_line, add_start=False, add_end=True)
                vi_ids = self.bpe.encode(vi_line, add_start=False, add_end=False)
                if len(en_ids) > self.max_len or len(vi_ids) > self.max_len:
                    continue
                yield en_ids, vi_ids

    def iterate(
        self,
        max_tokens: int = getattr(cfg, 'max_tokens', 4000),
        shuffle: bool = True,
        max_pairs: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Iterator[dict]:
        """
        Yield collated batches.

        Bucket-sorts each epoch's data so that batches have similar-length
        sequences. Uses token-level dynamic batching to group sentences
        such that the padded token count never exceeds `max_tokens`.
        """
        # Fill buckets using the in-memory cached pairs
        buckets: List[List[Tuple]] = [[] for _ in range(len(BUCKET_BOUNDARIES) + 1)]
        total = 0
        for pair in self.pairs:
            en_ids, vi_ids = pair
            bid = _bucket_id(len(en_ids))
            buckets[bid].append(pair)
            total += 1
            if max_pairs and total >= max_pairs:
                break

        # Shuffle within buckets using an explicit seed if provided
        if shuffle:
            rng = random.Random(seed) if seed is not None else random.Random()
            for bucket in buckets:
                rng.shuffle(bucket)

        # Interleave buckets into batch stream
        all_pairs: List[Tuple] = []
        for bucket in buckets:
            all_pairs.extend(bucket)

        # Group pairs into dynamic token-level batches
        current_batch = []
        max_en = 0
        max_vi = 0
        
        for pair in all_pairs:
            en_len = len(pair[0])
            vi_len = len(pair[1]) + 1  # +1 for START/END roughly
            
            future_max_en = max(max_en, en_len)
            future_max_vi = max(max_vi, vi_len)
            future_effective_tokens = (future_max_en + future_max_vi) * (len(current_batch) + 1)
            
            if future_effective_tokens > max_tokens and current_batch:
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
