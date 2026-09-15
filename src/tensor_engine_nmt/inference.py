"""
inference.py — Greedy and beam-search decoding.

Both modes share the same encoder forward pass.
The decoder runs one step at a time using DecoderLSTM.step().

Usage:
    model = Seq2Seq.load(...)
    translator = Translator(model, bpe)
    print(translator.translate("How are you?"))
"""
import numpy as np
import heapq
from typing import List, Tuple

from .backend import xp, f32
from .config import cfg
from .bpe import BPETokenizer, START_ID, END_ID, PAD_ID


class Translator:
    """
    Wraps a trained Seq2Seq model and BPE tokenizer for inference.
    """

    def __init__(self, model, bpe: BPETokenizer, hp=cfg):
        self.model = model
        self.bpe   = bpe
        self.hp    = hp

    def _encode_source(self, sentence: str):
        """
        Encode a source sentence and run the encoder.

        Returns
        -------
        H          : float32 (1, Tx, d) — encoder memory
        h_enc_all  : float32 (L, 1, Tx, d)
        C_enc_all  : float32 (L, 1, Tx, d)
        Xlen       : int32 (1,)
        """
        enc = self.model.encoder
        ids = self.bpe.encode(sentence, add_start=False, add_end=True)
        X    = xp.asarray(np.array([ids], dtype=np.int32))    # (1, Tx)
        Xlen = np.array([len(ids)], dtype=np.int32)           # (1,)
        H, h_enc_all, C_enc_all, _ = enc.forward(X, Xlen)
        return H, h_enc_all, C_enc_all, Xlen

    def _init_decoder_states(self, h_enc_all, C_enc_all, Xlen):
        """Perform Stage 3 handoff for a single sentence (B=1)."""
        hp = self.hp
        L  = hp.L
        d_half = hp.d // 2
        idx = Xlen[0] - 1
        # Forward half: last real encoder step (has seen full left context).
        # Backward half: t=0 is the backward LSTM's last computed step
        #                (has seen the full sentence from right to left).
        h_dec = [
            xp.concatenate([
                h_enc_all[l][0:1, idx:idx+1, :d_half].squeeze(1),
                h_enc_all[l][0:1, 0:1,       d_half:].squeeze(1),
            ], axis=1).copy()
            for l in range(L)
        ]
        C_dec = [
            xp.concatenate([
                C_enc_all[l][0:1, idx:idx+1, :d_half].squeeze(1),
                C_enc_all[l][0:1, 0:1,       d_half:].squeeze(1),
            ], axis=1).copy()
            for l in range(L)
        ]
        return h_dec, C_dec

    # ── Greedy decode ─────────────────────────────────────────────────────────

    def greedy_decode(self, sentence: str) -> str:
        """
        Greedy argmax decoding.

        At each step, pick the single most probable token.
        Stops at END or max_decode_len.

        Returns the decoded Vietnamese string.
        """
        hp = self.hp
        H, h_enc_all, C_enc_all, Xlen = self._encode_source(sentence)
        h_dec, C_dec = self._init_decoder_states(h_enc_all, C_enc_all, Xlen)

        token = xp.asarray(np.array([START_ID], dtype=np.int32))  # (1,)
        output_ids = []

        dec = self.model.decoder
        for _ in range(hp.max_decode_len):
            logits_t, h_dec, C_dec = dec.step(token, h_dec, C_dec, H, Xlen)
            token = xp.argmax(logits_t, axis=1).astype(xp.int32)   # (1,)
            tid = int(token[0])
            if tid == END_ID:
                break
            output_ids.append(tid)

        return self.bpe.decode(output_ids)

    # ── Beam search ───────────────────────────────────────────────────────────

    def beam_decode(self, sentence: str, beam_width: int = None) -> str:
        """
        Beam search decoding.

        Maintains `beam_width` hypotheses at each step.
        Each hypothesis carries its own decoder hidden/cell state.
        Returns the best complete hypothesis.

        Parameters
        ----------
        beam_width : int — if None, uses cfg.beam_width
        """
        hp = self.hp
        bw = beam_width or hp.beam_width
        H, h_enc_all, C_enc_all, Xlen = self._encode_source(sentence)
        h_dec, C_dec = self._init_decoder_states(h_enc_all, C_enc_all, Xlen)

        dec = self.model.decoder

        # Each beam entry: (neg_log_prob, token_sequence, h_dec, C_dec)
        # Using negative log prob so heapq (min-heap) gives the best (lowest NLL) first
        beams = [(0.0, [START_ID], h_dec, C_dec)]
        completed = []

        for step in range(hp.max_decode_len):
            candidates = []
            for neg_lp, seq, h, C in beams:
                last_token = xp.asarray(np.array([seq[-1]], dtype=np.int32))
                logits_t, h_new, C_new = dec.step(last_token, h, C, H, Xlen)

                # Softmax → log-probs
                lp = logits_t[0]           # (V,)
                lp_max = float(lp.max())
                exp_lp = xp.exp(lp - lp_max)
                probs  = exp_lp / exp_lp.sum()
                log_probs = xp.log(probs + 1e-9)

                # Top-bw tokens
                top_ids = xp.argsort(log_probs)[-bw:]  # ascending → take last bw

                for tid in top_ids:
                    tid_i  = int(tid)
                    new_lp = float(neg_lp) + float(-log_probs[tid_i])
                    new_seq = seq + [tid_i]

                    if tid_i == END_ID:
                        # Normalise by length and store
                        length_norm = len(new_seq)
                        completed.append((new_lp / length_norm, new_seq[1:-1]))
                    else:
                        candidates.append((new_lp, new_seq, h_new, C_new))

            if not candidates:
                break

            # Prune to top-bw
            candidates.sort(key=lambda x: x[0])
            beams = candidates[:bw]

            if len(completed) >= bw:
                break

        if not completed:
            # Fall back to the best beam
            best_seq = beams[0][1][1:] if beams else []
            return self.bpe.decode(best_seq)

        completed.sort(key=lambda x: x[0])
        return self.bpe.decode(completed[0][1])

    # ── High-level API ────────────────────────────────────────────────────────

    def translate(self, sentence: str, method: str = "beam") -> str:
        """
        Translate an English sentence to Vietnamese.

        Parameters
        ----------
        method : 'beam' or 'greedy'
        """
        if method == "beam":
            return self.beam_decode(sentence)
        return self.greedy_decode(sentence)

    def interactive(self) -> None:
        """Run an interactive translation REPL."""
        print("=== tensor-engine-nmt translate ===")
        print("Type an English sentence and press Enter. Ctrl-C to quit.\n")
        while True:
            try:
                src = input("EN> ").strip()
                if not src:
                    continue
                vi = self.translate(src)
                print(f"VI> {vi}\n")
            except (KeyboardInterrupt, EOFError):
                print("\nBye!")
                break
