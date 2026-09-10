"""
encoder.py — 2-layer stacked unidirectional LSTM encoder.

Implements Stage 2 and the gradient half of Stage 2 from:
    test/unidirectional_pipeline.md

Row-vector convention throughout:
    y = x @ W + b   where x is (B, in), W is (in, out), result is (B, out)

All weights are float32.  Integer tensors (X) are never used in arithmetic.
"""
import numpy as np
from .backend import xp, f32
from .config import cfg
from .activations import sigmoid, sigmoid_deriv, tanh, tanh_deriv
from .init_weights import init_lstm_weights, xavier_uniform, zeros


class EncoderLSTM:
    """
    Stacked unidirectional LSTM encoder.

    Parameters
    ----------
    V, e, d, L : from cfg — vocabulary size, embedding dim, hidden dim, n layers
    """

    def __init__(self, hp: "HParams" = cfg):
        self.hp = hp
        V, e, d, L = hp.V, hp.e, hp.d, hp.L

        # ── Embedding table ──────────────────────────────────────────────────
        # Ex : (V, e)  float32 — shared vocab, separate weights from decoder
        self.Ex = xavier_uniform(V, e)
        self.dEx = zeros(V, e)

        # ── LSTM weights — one set per layer ─────────────────────────────────
        # Layer 0: input_dim = e (receives embeddings)
        # Layer l: input_dim = d (receives previous layer's hidden state)
        self.W_ih = []   # list of (input_dim, 4d)
        self.W_hh = []   # list of (d, 4d)
        self.b    = []   # list of (4d,)
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

        # Cache (populated by forward, consumed by backward)
        self._cache = None

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray, Xlen: np.ndarray) -> tuple:
        """
        Run the encoder forward pass.

        Parameters
        ----------
        X    : int32 (B, Tx)   — padded encoder token ids
        Xlen : int32 (B,)      — true lengths (pre-pad) of each sequence

        Returns
        -------
        H          : float32 (B, Tx, d) — top-layer hidden states for attention
        h_enc_all  : float32 (L, B, Tx, d) — all layers' h, for BPTT + handoff
        C_enc_all  : float32 (L, B, Tx, d) — all layers' C, for BPTT + handoff
        cache      : dict — everything needed for backward
        """
        hp = self.hp
        B, Tx = X.shape
        d, L = hp.d, hp.L

        # Embedding lookup — (B, Tx, e)
        # We process one timestep at a time to avoid holding the full (B,Tx,e) matrix
        # (though we do cache x_emb for the backward pass)
        x_emb = self.Ex[X]       # (B, Tx, e)  — gather; float32

        # Initialise hidden and cell states per layer
        h = [xp.zeros((B, d), dtype=f32) for _ in range(L)]
        C = [xp.zeros((B, d), dtype=f32) for _ in range(L)]

        # Storage for BPTT: h_cache[l][t] and C_cache[l][t] store the state
        # *after* processing timestep t.  Index 0 → init (zeros).
        h_cache = [[None] * (Tx + 1) for _ in range(L)]
        C_cache = [[None] * (Tx + 1) for _ in range(L)]
        gates_cache = [[None] * Tx for _ in range(L)]
        inp_cache   = [[None] * Tx for _ in range(L)]
        for l in range(L):
            h_cache[l][0] = h[l]
            C_cache[l][0] = C[l]

        # Top-layer h at every t — used by the decoder's attention
        H = xp.zeros((B, Tx, d), dtype=f32)

        # Also store all-layer h and C for handoff (Stage 3) and BPTT
        h_enc_all = xp.zeros((L, B, Tx, d), dtype=f32)
        C_enc_all = xp.zeros((L, B, Tx, d), dtype=f32)

        # ── Loop over timesteps ───────────────────────────────────────────────
        for t in range(Tx):
            for l in range(L):
                inp = x_emb[:, t, :] if l == 0 else h[l - 1]   # (B, e or d)

                # Gate pre-activations: (B, 4d)
                gates = inp @ self.W_ih[l] + h[l] @ self.W_hh[l] + self.b[l]

                # Split into [f, i, g, o] each (B, d)
                f_pre, i_pre, g_pre, o_pre = xp.split(gates, 4, axis=1)

                f = sigmoid(f_pre)   # forget gate
                i = sigmoid(i_pre)   # input gate
                g = xp.tanh(g_pre)   # candidate
                o = sigmoid(o_pre)   # output gate

                C[l] = f * C[l] + i * g               # (B, d)
                h[l] = o * xp.tanh(C[l])              # (B, d)

                # Cache for BPTT
                h_cache[l][t + 1] = h[l]
                C_cache[l][t + 1] = C[l]
                # Store activated gate values (not pre-activations) — needed for derivs
                gates_cache[l][t] = (f, i, g, o)
                inp_cache[l][t]   = inp

                h_enc_all[l, :, t, :] = h[l]
                C_enc_all[l, :, t, :] = C[l]

            H[:, t, :] = h[L - 1]    # top layer output

        self._cache = {
            "X": X,
            "Xlen": Xlen,
            "x_emb": x_emb,
            "h_cache": h_cache,
            "C_cache": C_cache,
            "gates_cache": gates_cache,
            "inp_cache": inp_cache,
            "B": B,
            "Tx": Tx,
        }

        return H, h_enc_all, C_enc_all, self._cache

    # ── Backward pass (BPTT) ─────────────────────────────────────────────────

    def backward(self, dH: np.ndarray, dh_enc_final=None, dC_enc_final=None) -> None:
        """
        Backprop-through-time for the encoder.

        Parameters
        ----------
        dH : float32 (B, Tx, d)
            Gradient flowing from the decoder's attention into the top-layer
            encoder hidden states.  δL/δH.

        dh_enc_final : float32 (L, B, d) or None
            Gradient from the decoder handoff into the final h states per layer.

        dC_enc_final : float32 (L, B, d) or None
            Gradient from the decoder handoff into the final C states per layer.

        Side effects
        ------------
        Accumulates into self.dEx, self.dW_ih[l], self.dW_hh[l], self.db[l].
        """
        cache = self._cache
        hp = self.hp
        B, Tx = cache["B"], cache["Tx"]
        L, d = hp.L, hp.d

        h_cache     = cache["h_cache"]
        C_cache     = cache["C_cache"]
        gates_cache = cache["gates_cache"]
        inp_cache   = cache["inp_cache"]
        X           = cache["X"]

        # Gradient carriers from one timestep to the previous
        dh_next = [xp.zeros((B, d), dtype=f32) for _ in range(L)]
        dC_next = [xp.zeros((B, d), dtype=f32) for _ in range(L)]

        # Inject handoff gradients at the LAST MEANINGFUL timestep per example
        # (Stage 3 backward: gradient of the gather operation)
        if dh_enc_final is not None:
            Xlen = cache["Xlen"]
            for l in range(L):
                # dh_enc_final[l] is (B, d) — add to dh_next at t=Tx (last step)
                # We handle this by initialising dh_next[l] with it
                dh_next[l] = dh_next[l] + dh_enc_final[l]
                if dC_enc_final is not None:
                    dC_next[l] = dC_next[l] + dC_enc_final[l]

        # dEx accumulates via scatter-add
        # We'll use numpy for the scatter-add since xp.add.at may be slow on CuPy
        dEx_np = np.zeros_like(np.zeros((hp.V, hp.e), dtype=np.float32))

        # ── BPTT: t = Tx-1 … 0 ──────────────────────────────────────────────
        for t in reversed(range(Tx)):
            # Gradient into the top layer from H
            dh_next[L - 1] = dh_next[L - 1] + dH[:, t, :]

            for l in reversed(range(L)):
                f, i, g, o = gates_cache[l][t]
                h_prev = h_cache[l][t]         # h at t-1
                C_prev = C_cache[l][t]         # C at t-1
                C_cur  = C_cache[l][t + 1]     # C at t (current)
                inp    = inp_cache[l][t]

                dh_out = dh_next[l]            # (B, d)

                # h[l,t] = o * tanh(C[l,t])
                tanh_C   = xp.tanh(C_cur)
                do_pre   = dh_out * tanh_C * sigmoid_deriv(o)      # (B, d)
                dC_cur   = dh_out * o * tanh_deriv(tanh_C)         # (B, d)
                dC_cur   = dC_cur + dC_next[l]                     # add cell carry

                # C[l,t] = f * C[l,t-1] + i * g
                df_pre   = dC_cur * C_prev * sigmoid_deriv(f)      # (B, d)
                di_pre   = dC_cur * g      * sigmoid_deriv(i)      # (B, d)
                dg_pre   = dC_cur * i      * tanh_deriv(g)         # (B, d)
                dC_next[l] = dC_cur * f                            # carry to t-1

                # Concatenate gate gradients: (B, 4d)
                dgates = xp.concatenate([df_pre, di_pre, dg_pre, do_pre], axis=1)

                # Parameter gradients (accumulate)
                self.dW_ih[l] += inp.T @ dgates
                self.dW_hh[l] += h_prev.T @ dgates
                self.db[l]    += dgates.sum(axis=0)

                # Carry hidden gradient to previous timestep
                dh_next[l] = dgates @ self.W_hh[l].T    # (B, d)

                # Gradient to input of this layer
                dinp = dgates @ self.W_ih[l].T           # (B, e or d)

                if l == 0:
                    # dinp is the gradient into the embedding — scatter-add into dEx
                    dinp_np = dinp if isinstance(dinp, np.ndarray) else xp.asnumpy(dinp)
                    X_np    = X if isinstance(X, np.ndarray) else xp.asnumpy(X)
                    np.add.at(dEx_np, X_np[:, t], dinp_np)
                else:
                    # dinp feeds into h of the layer below at the SAME timestep
                    dh_next[l - 1] = dh_next[l - 1] + dinp

        # Copy accumulated dEx back to the xp backend
        self.dEx += xp.asarray(dEx_np)

    # ── Parameter access ────────────────────────────────────────────────────

    def parameters(self):
        """
        Yield (param, grad) pairs for the optimizer.
        Order: Ex, W_ih[0], W_hh[0], b[0], W_ih[1], W_hh[1], b[1], …
        """
        yield self.Ex, self.dEx
        for l in range(self.hp.L):
            yield self.W_ih[l], self.dW_ih[l]
            yield self.W_hh[l], self.dW_hh[l]
            yield self.b[l],    self.db[l]

    def zero_grad(self):
        """Zero all gradient arrays in-place."""
        self.dEx[:] = 0.0
        for l in range(self.hp.L):
            self.dW_ih[l][:] = 0.0
            self.dW_hh[l][:] = 0.0
            self.db[l][:]    = 0.0
