"""
decoder.py — 2-layer stacked unidirectional LSTM decoder with Luong attention.

Implements Stages 3 & 4 from test/unidirectional_pipeline.md

Per decoder timestep t:
  1. Select input token (teacher forcing schedule)
  2. Embed: y_t = Ey[input_ids]                          (B, e)
  3. LSTM layer 1: gates → f,i,g,o → C,h               (B, d)
  4. LSTM layer 2: same                                  (B, d)
  5. s_t = h_dec[L-1]   (top-layer alias)               (B, d)
  6. Attention: temp = s_t @ W_a; e_t; alpha_t; z_t     (B, d)
  7. Fusion: concat = [z_t ; s_t]; s_tilde = tanh(concat @ W_c + b_c)  (B, d)
  8. Output: logits_t = s_tilde @ Wy + by               (B, V)

Teacher forcing schedule (Bengio et al., 2015):
    ε_i = k / (k + exp(i / k))    where i = global training step
    At t=0: always use START (Yin[:,0])
    At t>=1: with prob ε use ground truth Yin[:,t], else argmax from t-1
"""
import numpy as np
from .backend import xp, f32
from .config import cfg
from .activations import sigmoid, sigmoid_deriv, tanh, tanh_deriv
from .attention import LuongAttention
from .init_weights import init_lstm_weights, xavier_uniform, glorot_uniform, zeros


class DecoderLSTM:
    """
    Stacked unidirectional LSTM decoder with Luong general attention.
    """

    def __init__(self, hp=cfg):
        self.hp = hp
        V, e, d, L = hp.V, hp.e, hp.d, hp.L

        # ── Decoder embedding table ──────────────────────────────────────────
        # Ey : (V, e) — separate from Ex (shared vocab ≠ shared weights)
        self.Ey  = xavier_uniform(V, e)
        self.dEy = zeros(V, e)

        # ── LSTM weights — one set per layer ─────────────────────────────────
        self.W_ih  = []
        self.W_hh  = []
        self.b     = []
        self.dW_ih = []
        self.dW_hh = []
        self.db    = []

        for l in range(L):
            in_dim = e if l == 0 else d
            W_ih_l, W_hh_l, b_l = init_lstm_weights(in_dim, d)
            self.W_ih.append(W_ih_l)
            self.W_hh.append(W_hh_l)
            self.b.append(b_l)
            self.dW_ih.append(zeros(in_dim, 4 * d))
            self.dW_hh.append(zeros(d, 4 * d))
            self.db.append(zeros(4 * d))

        # ── Attention ─────────────────────────────────────────────────────────
        self.attention = LuongAttention(hp)

        # ── Fusion layer ─────────────────────────────────────────────────────
        # concat = [z_t ; s_t] → (B, 2d)
        # s_tilde = tanh(concat @ W_c + b_c)  → (B, d)
        self.W_c  = glorot_uniform(2 * d, d)
        self.b_c  = zeros(d)
        self.dW_c = zeros(2 * d, d)
        self.db_c = zeros(d)

        # ── Output projection ─────────────────────────────────────────────────
        # logits = s_tilde @ Wy + by  → (B, V)
        self.Wy  = glorot_uniform(d, V)
        self.by  = zeros(V)
        self.dWy = zeros(d, V)
        self.dby = zeros(V)

        self._cache = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _teacher_forcing_prob(global_step: int, k: float, min_tf: float = 0.0) -> float:
        """ε_i = max(min_tf, k / (k + exp(i/k)))"""
        raw_prob = float(k / (k + np.exp(global_step / k)))
        return max(min_tf, raw_prob)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        Yin:        np.ndarray,     # (B, Ty) int32 — ground-truth decoder inputs
        H:          "xp.ndarray",   # (B, Tx, d) encoder memory
        Xlen:       np.ndarray,     # (B,)  encoder true lengths
        h_enc_all:  "xp.ndarray",   # (L, B, Tx, d) encoder hidden states (for handoff)
        C_enc_all:  "xp.ndarray",   # (L, B, Tx, d) encoder cell states (for handoff)
        global_step: int = 0,
    ) -> tuple:
        """
        Run the decoder forward pass.

        Returns
        -------
        logits : float32 (B, Ty, V) — output logits stacked over all timesteps
        cache  : dict — everything needed for backward
        """
        hp = self.hp
        B, Ty = Yin.shape
        d, L, V, e = hp.d, hp.L, hp.V, hp.e

        if isinstance(Yin, np.ndarray):
            Yin_xp = xp.asarray(Yin)
        else:
            Yin_xp = Yin

        # ── Stage 3: Encoder→Decoder Handoff ─────────────────────────────────
        # Gather the final (real, non-pad) hidden/cell state per example
        Xlen_arr = Xlen if isinstance(Xlen, np.ndarray) else xp.asnumpy(Xlen)
        idx = Xlen_arr - 1                                         # (B,) 0-indexed last pos
        idx_xp = xp.asarray(idx)
        b_idx  = xp.arange(B)

        h_dec = []
        C_dec = []
        d_half = hp.d // 2
        for l in range(L):
            # h_enc_all[l] is (B, Tx, d).
            # Forward half  → last real token (idx) has seen the full left context.
            # Backward half → t=0 is where the backward LSTM *finishes* after
            #                 sweeping right-to-left; that state summarises the whole
            #                 sentence from right to left.
            h_fwd = h_enc_all[l][b_idx, idx_xp, :d_half]          # (B, d/2)
            h_bwd = h_enc_all[l][b_idx, 0,      d_half:]          # (B, d/2)
            C_fwd = C_enc_all[l][b_idx, idx_xp, :d_half]          # (B, d/2)
            C_bwd = C_enc_all[l][b_idx, 0,      d_half:]          # (B, d/2)
            h_dec.append(xp.concatenate([h_fwd, h_bwd], axis=1).copy())  # (B, d)
            C_dec.append(xp.concatenate([C_fwd, C_bwd], axis=1).copy())  # (B, d)

        # ── Teacher forcing probability for this batch ────────────────────────
        # Enforce the anchor threshold via hp.min_tf (prevent exposure bias trap)
        epsilon = self._teacher_forcing_prob(global_step, hp.k, getattr(hp, "min_tf", 0.7))

        # ── Initialise caches ─────────────────────────────────────────────────
        h_cache    = [[None] * (Ty + 1) for _ in range(L)]
        C_cache    = [[None] * (Ty + 1) for _ in range(L)]
        gates_cache= [[None] * Ty       for _ in range(L)]
        inp_cache  = [[None] * Ty       for _ in range(L)]

        for l in range(L):
            h_cache[l][0] = h_dec[l]
            C_cache[l][0] = C_dec[l]

        y_emb_cache    = []       # (B, e) per step
        input_ids_cache= []       # (B,)   int32 per step
        s_t_cache      = []       # (B, d) top-layer h per step (= s_t)
        z_t_cache      = []       # (B, d) context vector per step
        alpha_t_cache  = []       # (B, Tx) attention weights per step
        s_tilde_cache  = []       # (B, d) attentional h per step
        concat_cache   = []       # (B, 2d) before fusion per step
        logits_list    = []       # (B, V) per step

        y_hat_prev = None         # (B,) int32 argmax from previous step

        # ── Decoder loop: t = 0 … Ty-1 ───────────────────────────────────────
        for t in range(Ty):
            # Step 1: select input token ids
            if t == 0:
                input_ids = Yin_xp[:, 0]                 # always START
            else:
                use_gt = (np.random.rand() < epsilon)
                if use_gt:
                    input_ids = Yin_xp[:, t]
                else:
                    input_ids = y_hat_prev                # int32 (B,)

            input_ids_cache.append(input_ids)

            # Step 2: embed
            y_t = self.Ey[input_ids]                      # (B, e)
            y_emb_cache.append(y_t)

            # Step 3 & 4: LSTM layers
            for l in range(L):
                inp = y_t if l == 0 else h_dec[l - 1]    # (B, e or d)

                gates = inp @ self.W_ih[l] + h_dec[l] @ self.W_hh[l] + self.b[l]
                f_pre, i_pre, g_pre, o_pre = xp.split(gates, 4, axis=1)

                f = sigmoid(f_pre)
                i = sigmoid(i_pre)
                g = xp.tanh(g_pre)
                o = sigmoid(o_pre)

                C_dec[l] = f * C_dec[l] + i * g
                h_dec[l] = o * xp.tanh(C_dec[l])

                h_cache[l][t + 1]  = h_dec[l]
                C_cache[l][t + 1]  = C_dec[l]
                gates_cache[l][t]  = (f, i, g, o)
                inp_cache[l][t]    = inp

            # Step 5: top-layer alias
            s_t = h_dec[L - 1]                            # (B, d)
            s_t_cache.append(s_t)

            # Step 6: attention
            z_t, alpha_t = self.attention.forward(s_t, H, Xlen)
            z_t_cache.append(z_t)
            alpha_t_cache.append(alpha_t)

            # Step 7: fusion
            concat   = xp.concatenate([z_t, s_t], axis=1)  # (B, 2d)
            s_tilde  = xp.tanh(concat @ self.W_c + self.b_c)  # (B, d)
            concat_cache.append(concat)
            s_tilde_cache.append(s_tilde)

            # Step 8: output projection
            logits_t = s_tilde @ self.Wy + self.by         # (B, V)
            logits_list.append(logits_t)

            # Compute argmax for next step's free-running input
            y_hat_prev = xp.argmax(logits_t, axis=1).astype(xp.int32)  # (B,)

        # Stack logits: (B, Ty, V)
        logits = xp.stack(logits_list, axis=1).astype(f32)

        self._cache = {
            "Yin": Yin,
            "H": H,
            "Xlen": Xlen,
            "h_cache": h_cache,
            "C_cache": C_cache,
            "gates_cache": gates_cache,
            "inp_cache": inp_cache,
            "y_emb_cache": y_emb_cache,
            "input_ids_cache": input_ids_cache,
            "s_t_cache": s_t_cache,
            "z_t_cache": z_t_cache,
            "alpha_t_cache": alpha_t_cache,
            "s_tilde_cache": s_tilde_cache,
            "concat_cache": concat_cache,
            "logits_list": logits_list,
            "B": B,
            "Ty": Ty,
            # handoff indices for backward
            "idx_xp": idx_xp,
            "b_idx": b_idx,
        }

        return logits, self._cache

    # ── Backward pass (BPTT) ─────────────────────────────────────────────────

    def backward(self, dlogits: "xp.ndarray") -> tuple:
        """
        Backprop through the decoder.

        Parameters
        ----------
        dlogits : float32 (B, Ty, V)
            Gradient from loss.backward().

        Returns
        -------
        dH      : float32 (B, Tx, d) — gradient into encoder memory
        dh_enc  : float32 (L, B, d)  — gradient into encoder final hidden states
        dC_enc  : float32 (L, B, d)  — gradient into encoder final cell states
        """
        cache = self._cache
        hp    = self.hp
        B, Ty = cache["B"], cache["Ty"]
        L, d, V = hp.L, hp.d, hp.V
        Tx = cache["H"].shape[1]

        h_cache     = cache["h_cache"]
        C_cache     = cache["C_cache"]
        gates_cache = cache["gates_cache"]
        inp_cache   = cache["inp_cache"]

        # Accumulated gradient into full encoder memory H
        dH = xp.zeros_like(cache["H"])    # (B, Tx, d)

        # Hidden/cell gradients carried through decoder time (BPTT)
        dh_next = [xp.zeros((B, d), dtype=f32) for _ in range(L)]
        dC_next = [xp.zeros((B, d), dtype=f32) for _ in range(L)]

        # Embedding gradient (scatter-add in numpy)
        import numpy as np
        dEy_np = np.zeros((hp.V, hp.e), dtype=np.float32)

        # ── BPTT: t = Ty-1 … 0 ──────────────────────────────────────────────
        for t in reversed(range(Ty)):
            dlogits_t  = dlogits[:, t, :]              # (B, V)
            s_tilde_t  = cache["s_tilde_cache"][t]     # (B, d)
            concat_t   = cache["concat_cache"][t]      # (B, 2d)
            s_t        = cache["s_t_cache"][t]         # (B, d)
            z_t        = cache["z_t_cache"][t]         # (B, d)
            input_ids_t = cache["input_ids_cache"][t]  # (B,) int32

            # ── Step 8 backward: logits = s_tilde @ Wy + by ──────────────────
            self.dWy += s_tilde_t.T @ dlogits_t         # (d, V)
            self.dby += dlogits_t.sum(axis=0)           # (V,)
            ds_tilde = dlogits_t @ self.Wy.T            # (B, d)

            # ── Step 7 backward: s_tilde = tanh(concat @ W_c + b_c) ─────────
            # dtanh: d(tanh(x))/dx = 1 - tanh(x)^2
            dtanh_val = (1.0 - s_tilde_t ** 2) * ds_tilde   # (B, d)
            self.dW_c += concat_t.T @ dtanh_val              # (2d, d)
            self.db_c += dtanh_val.sum(axis=0)               # (d,)
            dconcat = dtanh_val @ self.W_c.T                 # (B, 2d)

            # Split dconcat → [dz_t_path, ds_t_from_fusion]
            dz_t_path      = dconcat[:, :d]                  # (B, d)
            ds_t_from_fusion = dconcat[:, d:]                # (B, d)

            # ── Step 6 backward: attention ───────────────────────────────────
            ds_t_total, dH_t = self.attention.backward(dz_t_path, ds_t_from_fusion)
            dH += dH_t                                       # accumulate into full H

            # ds_t_total is the gradient into s_t = h_dec[L-1]
            dh_next[L - 1] = dh_next[L - 1] + ds_t_total

            # ── Steps 4 & 3 backward: LSTM layers ────────────────────────────
            for l in reversed(range(L)):
                f, i, g, o = gates_cache[l][t]
                h_prev = h_cache[l][t]
                C_prev = C_cache[l][t]
                C_cur  = C_cache[l][t + 1]
                inp    = inp_cache[l][t]

                dh_out = dh_next[l]               # (B, d)

                tanh_C = xp.tanh(C_cur)
                do_pre = dh_out * tanh_C * sigmoid_deriv(o)
                dC_cur = dh_out * o * tanh_deriv(tanh_C) + dC_next[l]

                df_pre = dC_cur * C_prev * sigmoid_deriv(f)
                di_pre = dC_cur * g      * sigmoid_deriv(i)
                dg_pre = dC_cur * i      * tanh_deriv(g)
                dC_next[l] = dC_cur * f

                dgates = xp.concatenate([df_pre, di_pre, dg_pre, do_pre], axis=1)

                self.dW_ih[l] += inp.T @ dgates
                self.dW_hh[l] += h_prev.T @ dgates
                self.db[l]    += dgates.sum(axis=0)

                dh_next[l] = dgates @ self.W_hh[l].T
                dinp       = dgates @ self.W_ih[l].T

                if l == 0:
                    # dinp → embedding gradient (scatter-add)
                    dinp_np = dinp if isinstance(dinp, np.ndarray) else xp.asnumpy(dinp)
                    ids_np  = (input_ids_t if isinstance(input_ids_t, np.ndarray)
                               else xp.asnumpy(input_ids_t))
                    np.add.at(dEy_np, ids_np, dinp_np)
                else:
                    dh_next[l - 1] = dh_next[l - 1] + dinp

        # Copy embedding grad
        self.dEy += xp.asarray(dEy_np)

        # ── Backward through Stage 3 handoff (gather) ────────────────────────
        # The gradient of a gather is a scatter: distribute dh_next into the
        # correct positions in h_enc_all (the encoder's BPTT picks it up there).
        b_idx  = cache["b_idx"]
        idx_xp = cache["idx_xp"]

        dh_enc = xp.zeros((L, B, d), dtype=f32)
        dC_enc = xp.zeros((L, B, d), dtype=f32)
        for l in range(L):
            dh_enc[l] = dh_next[l]    # (B, d) → encoder BPTT receives this at t=Xlen[b]-1
            dC_enc[l] = dC_next[l]

        return dH, dh_enc, dC_enc

    # ── Single-step forward for inference ────────────────────────────────────

    def step(
        self,
        input_id: "xp.ndarray",   # (B,) int32
        h_dec:    list,            # [L × (B, d)] current hidden states
        C_dec:    list,            # [L × (B, d)] current cell states
        H:        "xp.ndarray",   # (B, Tx, d) encoder memory
        Xlen:     np.ndarray,
    ) -> tuple:
        """
        One inference step (no caching, no teacher forcing).

        Returns
        -------
        logits_t : (B, V)
        h_dec    : updated hidden states list
        C_dec    : updated cell states list
        """
        hp = self.hp
        L = hp.L

        y_t = self.Ey[input_id]                               # (B, e)

        for l in range(L):
            inp = y_t if l == 0 else h_dec[l - 1]
            gates = inp @ self.W_ih[l] + h_dec[l] @ self.W_hh[l] + self.b[l]
            f, i, g, o = xp.split(gates, 4, axis=1)
            f, i, o = sigmoid(f), sigmoid(i), sigmoid(o)
            g = xp.tanh(g)
            C_dec[l] = f * C_dec[l] + i * g
            h_dec[l] = o * xp.tanh(C_dec[l])

        s_t = h_dec[L - 1]
        z_t, _ = self.attention.forward(s_t, H, Xlen)
        concat  = xp.concatenate([z_t, s_t], axis=1)
        s_tilde = xp.tanh(concat @ self.W_c + self.b_c)
        logits_t = s_tilde @ self.Wy + self.by

        return logits_t, h_dec, C_dec

    # ── Parameter access ────────────────────────────────────────────────────

    def parameters(self):
        yield self.Ey, self.dEy
        for l in range(self.hp.L):
            yield self.W_ih[l], self.dW_ih[l]
            yield self.W_hh[l], self.dW_hh[l]
            yield self.b[l],    self.db[l]
        yield from self.attention.parameters()
        yield self.W_c,  self.dW_c
        yield self.b_c,  self.db_c
        yield self.Wy,   self.dWy
        yield self.by,   self.dby

    def zero_grad(self):
        self.dEy[:] = 0.0
        for l in range(self.hp.L):
            self.dW_ih[l][:] = 0.0
            self.dW_hh[l][:] = 0.0
            self.db[l][:]    = 0.0
        self.attention.zero_grad()
        self.dW_c[:] = 0.0
        self.db_c[:] = 0.0
        self.dWy[:]  = 0.0
        self.dby[:]  = 0.0
