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

    def greedy_decode(self, sentence: str, rep_penalty: float = 1.3) -> str:
        """
        Greedy argmax decoding with repetition penalty.

        At each step, pick the single most probable token.
        Stops at END or max_decode_len.

        Parameters
        ----------
        rep_penalty : float
            Divide the logit of any token that already appeared in the output
            by this factor (if logit > 0) or multiply (if logit < 0).
            1.0 = no penalty. 1.3 = moderate penalty (default).
            Reduces the attention-loop repetition common in early-stage models.

        Returns the decoded Vietnamese string.
        """
        hp = self.hp
        H, h_enc_all, C_enc_all, Xlen = self._encode_source(sentence)
        h_dec, C_dec = self._init_decoder_states(h_enc_all, C_enc_all, Xlen)

        token = xp.asarray(np.array([START_ID], dtype=np.int32))  # (1,)
        output_ids = []
        seen_ids = set()

        dec = self.model.decoder
        for _ in range(hp.max_decode_len):
            logits_t, h_dec, C_dec = dec.step(token, h_dec, C_dec, H, Xlen)

            # Apply repetition penalty: tokens already in output are down-scored.
            if rep_penalty != 1.0 and seen_ids:
                logits_np = (logits_t[0] if isinstance(logits_t, np.ndarray)
                             else xp.asnumpy(logits_t[0])).copy()
                for prev_id in seen_ids:
                    if logits_np[prev_id] > 0:
                        logits_np[prev_id] /= rep_penalty
                    else:
                        logits_np[prev_id] *= rep_penalty
                logits_t = xp.asarray(logits_np[np.newaxis, :])

            # PAD/START are structure, never output.
            logits_t = logits_t.copy()
            logits_t[:, PAD_ID]   = -1e30
            logits_t[:, START_ID] = -1e30
            token = xp.argmax(logits_t, axis=1).astype(xp.int32)   # (1,)
            tid = int(token[0])
            if tid == END_ID:
                break
            output_ids.append(tid)
            seen_ids.add(tid)

        return self.bpe.decode(output_ids)

    # ── Beam search ───────────────────────────────────────────────────────────

    def beam_decode(self, sentence: str, beam_width: int = None,
                    length_penalty: float = 0.7,
                    rep_penalty: float = 1.3) -> str:
        """
        Beam search decoding with length normalization and repetition penalty.

        Fixes vs previous version
        --------------------------
        1. Early-stop bug: previously stopped when len(completed) >= bw, which
           triggered on the first step because all 16 (bw×bw) expansions were
           checked and many hit END. Now stops only after max_decode_len steps
           or when all active beams have ended.

        2. Length normalization is now applied CONSISTENTLY: the same
           score = NLL / gen_len**alpha is used to prune the active beams and to
           rank completed hypotheses. Previously only completed hypotheses were
           normalized while active beams were pruned on raw NLL, which biased
           the beam toward short outputs — the exact thing the penalty exists to
           prevent.

        3. State aliasing: each candidate now carries deep-copied LSTM states,
           preventing multiple candidates from a single parent from sharing
           (and mutating) the same state arrays.

        4. Repetition penalty now uses greedy's documented rule (divide positive
           logits, multiply negative ones) and is applied to the logits rather
           than to exp(logits), so both decoders penalise identically.

        5. PAD and START are banned from the output vocabulary.

        Parameters
        ----------
        beam_width     : number of hypotheses to maintain (default cfg.beam_width)
        length_penalty : alpha for length normalization score = NLL / len^alpha.
                         0.0 = no normalization (biases short). 1.0 = full linear.
                         0.7 is standard (Wu et al. 2016 Google NMT). Default 0.7.
        rep_penalty    : repetition penalty factor (same as greedy_decode).
        """
        hp = self.hp
        bw = beam_width or hp.beam_width
        H, h_enc_all, C_enc_all, Xlen = self._encode_source(sentence)
        h_dec, C_dec = self._init_decoder_states(h_enc_all, C_enc_all, Xlen)

        dec = self.model.decoder

        def _copy_states(h, C):
            """Deep-copy LSTM state lists so beams don't share memory."""
            return (
                [xp.array(s, copy=True) for s in h],
                [xp.array(s, copy=True) for s in C],
            )

        # Beam entry: (cum_neg_log_prob, norm_score, token_seq, h_dec, C_dec, seen)
        # token_seq starts with START_ID; seen_ids excludes START.
        # norm_score = cum_neg_log_prob / gen_len**length_penalty, and is what we
        # rank by at BOTH pruning and completion.  Ranking the active beams by raw
        # NLL (as this used to) silently favours short hypotheses, which is exactly
        # what the length penalty is supposed to counteract.
        beams = [(0.0, 0.0, [START_ID], h_dec, C_dec, set())]
        completed = []

        def _norm(nlp, seq):
            """Length-normalised NLL.  seq includes START, so generated tokens
            are len(seq) - 1 (END included when the sequence has finished)."""
            return nlp / (max(1, len(seq) - 1) ** length_penalty)

        for step in range(hp.max_decode_len):
            if not beams:
                break

            candidates = []
            for neg_lp, _norm_lp, seq, h, C, seen in beams:
                last_token = xp.asarray(np.array([seq[-1]], dtype=np.int32))
                logits_t, h_new, C_new = dec.step(last_token, h, C, H, Xlen)

                # Repetition penalty on the LOGITS, with the same rule greedy
                # decoding documents (divide a positive logit, multiply a
                # negative one).  Applying it to exp(logits) instead quietly
                # changed the strength of the penalty between the two decoders.
                lp = logits_t[0]           # (V,)
                if rep_penalty != 1.0 and seen:
                    lp_np = (lp if isinstance(lp, np.ndarray) else xp.asnumpy(lp)).copy()
                    for prev_id in seen:
                        if lp_np[prev_id] > 0:
                            lp_np[prev_id] /= rep_penalty
                        else:
                            lp_np[prev_id] *= rep_penalty
                    lp = xp.asarray(lp_np)

                # PAD/START are structure, never output.
                lp = lp.copy() if hasattr(lp, "copy") else lp
                lp[PAD_ID]   = -1e30
                lp[START_ID] = -1e30

                exp_lp    = xp.exp(lp - lp.max())
                probs     = exp_lp / (exp_lp.sum() + 1e-9)
                log_probs = xp.log(probs + 1e-9)

                # Top-bw tokens
                top_ids = xp.argsort(log_probs)[-bw:]

                for tid in top_ids:
                    tid_i  = int(tid)
                    token_nlp = float(-log_probs[tid_i])
                    new_nlp = neg_lp + token_nlp
                    new_seq = seq + [tid_i]

                    if tid_i == END_ID:
                        # Store token_ids without START and END
                        completed.append((_norm(new_nlp, new_seq), new_seq[1:-1]))
                    else:
                        # Deep copy states so each candidate is independent
                        h_c, C_c = _copy_states(h_new, C_new)
                        new_seen = seen | {tid_i}
                        candidates.append(
                            (new_nlp, _norm(new_nlp, new_seq), new_seq, h_c, C_c, new_seen)
                        )

            if not candidates:
                # All beams ended with END this step
                break

            # Prune to top-bw active beams — on the normalised score
            candidates.sort(key=lambda x: x[1])
            beams = candidates[:bw]

        # If no completed hypothesis, fall back to the best active beam
        if not completed:
            if beams:
                best_seq = beams[0][2][1:]   # strip START
                return self.bpe.decode(best_seq)
            return ""

        completed.sort(key=lambda x: x[0])   # lowest normalized NLL = best
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
