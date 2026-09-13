"""
encoder.py — Stacked Bidirectional LSTM encoder.

Implements a Bidirectional LSTM where the forward and backward directions
each have hidden dimension d/2, so their concatenated output is dimension d,
perfectly matching the decoder's expected dimension without requiring a projection layer.
"""
import numpy as np
from .backend import xp, f32
from .config import cfg
from .activations import sigmoid, sigmoid_deriv, tanh, tanh_deriv
from .init_weights import init_lstm_weights, xavier_uniform, zeros


class EncoderLSTM:
    """
    Stacked Bidirectional LSTM encoder.

    Parameters
    ----------
    hp : HParams
    """

    def __init__(self, hp: "HParams" = cfg):
        self.hp = hp
        V, e, d, L = hp.V, hp.e, hp.d, hp.L

        assert d % 2 == 0, "Hidden dimension d must be even for BiLSTM to concatenate to d"
        d_half = d // 2

        # ── Embedding table ──────────────────────────────────────────────────
        self.Ex = xavier_uniform(V, e)
        self.dEx = zeros(V, e)

        # ── LSTM weights ─────────────────────────────────────────────────────
        self.W_ih_fwd, self.W_hh_fwd, self.b_fwd = [], [], []
        self.dW_ih_fwd, self.dW_hh_fwd, self.db_fwd = [], [], []
        
        self.W_ih_bwd, self.W_hh_bwd, self.b_bwd = [], [], []
        self.dW_ih_bwd, self.dW_hh_bwd, self.db_bwd = [], [], []

        for l in range(L):
            in_dim = e if l == 0 else d
            
            # Forward weights
            W_ih, W_hh, b = init_lstm_weights(in_dim, d_half)
            self.W_ih_fwd.append(W_ih)
            self.W_hh_fwd.append(W_hh)
            self.b_fwd.append(b)
            self.dW_ih_fwd.append(zeros(in_dim, 4 * d_half))
            self.dW_hh_fwd.append(zeros(d_half, 4 * d_half))
            self.db_fwd.append(zeros(4 * d_half))
            
            # Backward weights
            W_ih, W_hh, b = init_lstm_weights(in_dim, d_half)
            self.W_ih_bwd.append(W_ih)
            self.W_hh_bwd.append(W_hh)
            self.b_bwd.append(b)
            self.dW_ih_bwd.append(zeros(in_dim, 4 * d_half))
            self.dW_hh_bwd.append(zeros(d_half, 4 * d_half))
            self.db_bwd.append(zeros(4 * d_half))

        self._cache = None

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, X: np.ndarray, Xlen: np.ndarray) -> tuple:
        hp = self.hp
        B, Tx = X.shape
        d, L = hp.d, hp.L
        d_half = d // 2

        x_emb = self.Ex[X]       # (B, Tx, e)

        # Storage for BPTT
        h_cache_fwd = [[None] * (Tx + 1) for _ in range(L)]
        C_cache_fwd = [[None] * (Tx + 1) for _ in range(L)]
        gates_fwd   = [[None] * Tx for _ in range(L)]
        
        h_cache_bwd = [[None] * (Tx + 1) for _ in range(L)]
        C_cache_bwd = [[None] * (Tx + 1) for _ in range(L)]
        gates_bwd   = [[None] * Tx for _ in range(L)]
        
        inp_cache   = [[None] * Tx for _ in range(L)]

        # Outputs
        H = xp.zeros((B, Tx, d), dtype=f32)
        h_enc_all = xp.zeros((L, B, Tx, d), dtype=f32)
        C_enc_all = xp.zeros((L, B, Tx, d), dtype=f32)
        
        layer_input = x_emb  # Layer 0 input is embeddings
        
        # ── Loop over layers ─────────────────────────────────────────────────
        for l in range(L):
            h_fwd = xp.zeros((B, d_half), dtype=f32)
            C_fwd = xp.zeros((B, d_half), dtype=f32)
            h_cache_fwd[l][0] = h_fwd
            C_cache_fwd[l][0] = C_fwd
            
            h_bwd = xp.zeros((B, d_half), dtype=f32)
            C_bwd = xp.zeros((B, d_half), dtype=f32)
            h_cache_bwd[l][Tx] = h_bwd
            C_cache_bwd[l][Tx] = C_bwd

            layer_output = xp.zeros((B, Tx, d), dtype=f32)
            
            # ── Forward direction: 0 to Tx-1 ─────────────────────────────────
            for t in range(Tx):
                inp = layer_input[:, t, :]
                inp_cache[l][t] = inp
                
                gates = inp @ self.W_ih_fwd[l] + h_fwd @ self.W_hh_fwd[l] + self.b_fwd[l]
                f_pre, i_pre, g_pre, o_pre = xp.split(gates, 4, axis=1)
                
                f = sigmoid(f_pre)
                i = sigmoid(i_pre)
                g = xp.tanh(g_pre)
                o = sigmoid(o_pre)
                
                C_fwd = f * C_fwd + i * g
                h_fwd = o * xp.tanh(C_fwd)
                
                h_cache_fwd[l][t + 1] = h_fwd
                C_cache_fwd[l][t + 1] = C_fwd
                gates_fwd[l][t] = (f, i, g, o)
                
                layer_output[:, t, :d_half] = h_fwd
                C_enc_all[l, :, t, :d_half] = C_fwd
            
            # ── Backward direction: Tx-1 down to 0 ───────────────────────────
            for t in reversed(range(Tx)):
                inp = layer_input[:, t, :]
                # (inp_cache already stored during forward direction pass)
                
                gates = inp @ self.W_ih_bwd[l] + h_bwd @ self.W_hh_bwd[l] + self.b_bwd[l]
                f_pre, i_pre, g_pre, o_pre = xp.split(gates, 4, axis=1)
                
                f = sigmoid(f_pre)
                i = sigmoid(i_pre)
                g = xp.tanh(g_pre)
                o = sigmoid(o_pre)
                
                C_bwd = f * C_bwd + i * g
                h_bwd = o * xp.tanh(C_bwd)
                
                h_cache_bwd[l][t] = h_bwd
                C_cache_bwd[l][t] = C_bwd
                gates_bwd[l][t] = (f, i, g, o)
                
                layer_output[:, t, d_half:] = h_bwd
                C_enc_all[l, :, t, d_half:] = C_bwd
                
            h_enc_all[l] = layer_output
            layer_input = layer_output  # Input for next layer is this layer's output
            
            if l == L - 1:
                H = layer_output  # Top layer output for attention
                
        self._cache = {
            "X": X,
            "Xlen": Xlen,
            "h_cache_fwd": h_cache_fwd,
            "C_cache_fwd": C_cache_fwd,
            "gates_fwd": gates_fwd,
            "h_cache_bwd": h_cache_bwd,
            "C_cache_bwd": C_cache_bwd,
            "gates_bwd": gates_bwd,
            "inp_cache": inp_cache,
            "B": B,
            "Tx": Tx,
        }

        return H, h_enc_all, C_enc_all, self._cache

    # ── Backward pass (BPTT) ─────────────────────────────────────────────────

    def backward(self, dH: np.ndarray, dh_enc_final=None, dC_enc_final=None) -> None:
        cache = self._cache
        hp = self.hp
        B, Tx = cache["B"], cache["Tx"]
        L, d = hp.L, hp.d
        d_half = d // 2

        h_cache_fwd = cache["h_cache_fwd"]
        C_cache_fwd = cache["C_cache_fwd"]
        gates_fwd   = cache["gates_fwd"]
        
        h_cache_bwd = cache["h_cache_bwd"]
        C_cache_bwd = cache["C_cache_bwd"]
        gates_bwd   = cache["gates_bwd"]
        
        inp_cache   = cache["inp_cache"]
        X           = cache["X"]
        Xlen        = cache["Xlen"]

        # Gradients wrt the output of each layer. 
        dLayer_output = [xp.zeros((B, Tx, d), dtype=f32) for _ in range(L)]
        dLayer_output[L - 1] = dLayer_output[L - 1] + dH

        # Inject handoff gradients at the last meaningful step (Stage 3 backward)
        if dh_enc_final is not None:
            Xlen_arr = xp.asarray(Xlen)
            mask = (xp.arange(Tx)[None, :] == (Xlen_arr[:, None] - 1)) # (B, Tx)
            for l in range(L):
                dLayer_output[l] = dLayer_output[l] + dh_enc_final[l][:, None, :] * mask[:, :, None]

        dEx_np = np.zeros((hp.V, hp.e), dtype=np.float32)

        # ── BPTT layer by layer from top to bottom ───────────────────────────
        for l in reversed(range(L)):
            
            # ── 1. Forward direction BPTT: time flows Tx-1 down to 0 ─────────
            dh_next_fwd = xp.zeros((B, d_half), dtype=f32)
            dC_next_fwd = xp.zeros((B, d_half), dtype=f32)
            
            for t in reversed(range(Tx)):
                dh_out = dLayer_output[l][:, t, :d_half] + dh_next_fwd
                
                f, i, g, o = gates_fwd[l][t]
                h_prev = h_cache_fwd[l][t]         
                C_prev = C_cache_fwd[l][t]         
                C_cur  = C_cache_fwd[l][t + 1]     
                inp    = inp_cache[l][t]
                
                tanh_C = xp.tanh(C_cur)
                do_pre = dh_out * tanh_C * sigmoid_deriv(o)
                dC_cur = dh_out * o * tanh_deriv(tanh_C) + dC_next_fwd
                
                if dC_enc_final is not None:
                    # Inject cell handoff gradient
                    mask_t = (xp.asarray(Xlen) - 1 == t) # (B,)
                    dC_cur = dC_cur + dC_enc_final[l][:, :d_half] * mask_t[:, None]
                
                df_pre = dC_cur * C_prev * sigmoid_deriv(f)
                di_pre = dC_cur * g * sigmoid_deriv(i)
                dg_pre = dC_cur * i * tanh_deriv(g)
                dC_next_fwd = dC_cur * f
                
                dgates = xp.concatenate([df_pre, di_pre, dg_pre, do_pre], axis=1)
                
                self.dW_ih_fwd[l] += inp.T @ dgates
                self.dW_hh_fwd[l] += h_prev.T @ dgates
                self.db_fwd[l]    += dgates.sum(axis=0)
                
                dh_next_fwd = dgates @ self.W_hh_fwd[l].T
                dinp_fwd = dgates @ self.W_ih_fwd[l].T
                
                if l == 0:
                    dinp_np = dinp_fwd if isinstance(dinp_fwd, np.ndarray) else xp.asnumpy(dinp_fwd)
                    X_np    = X if isinstance(X, np.ndarray) else xp.asnumpy(X)
                    np.add.at(dEx_np, X_np[:, t], dinp_np)
                else:
                    dLayer_output[l - 1][:, t, :] = dLayer_output[l - 1][:, t, :] + dinp_fwd
                    
            # ── 2. Backward direction BPTT: time flows 0 up to Tx-1 ──────────
            dh_next_bwd = xp.zeros((B, d_half), dtype=f32)
            dC_next_bwd = xp.zeros((B, d_half), dtype=f32)
            
            for t in range(Tx):
                dh_out = dLayer_output[l][:, t, d_half:] + dh_next_bwd
                
                f, i, g, o = gates_bwd[l][t]
                h_prev = h_cache_bwd[l][t + 1] 
                C_prev = C_cache_bwd[l][t + 1]
                C_cur  = C_cache_bwd[l][t]
                inp    = inp_cache[l][t]
                
                tanh_C = xp.tanh(C_cur)
                do_pre = dh_out * tanh_C * sigmoid_deriv(o)
                dC_cur = dh_out * o * tanh_deriv(tanh_C) + dC_next_bwd
                
                if dC_enc_final is not None:
                    # Inject cell handoff gradient
                    mask_t = (xp.asarray(Xlen) - 1 == t) # (B,)
                    dC_cur = dC_cur + dC_enc_final[l][:, d_half:] * mask_t[:, None]
                
                df_pre = dC_cur * C_prev * sigmoid_deriv(f)
                di_pre = dC_cur * g * sigmoid_deriv(i)
                dg_pre = dC_cur * i * tanh_deriv(g)
                dC_next_bwd = dC_cur * f
                
                dgates = xp.concatenate([df_pre, di_pre, dg_pre, do_pre], axis=1)
                
                self.dW_ih_bwd[l] += inp.T @ dgates
                self.dW_hh_bwd[l] += h_prev.T @ dgates
                self.db_bwd[l]    += dgates.sum(axis=0)
                
                dh_next_bwd = dgates @ self.W_hh_bwd[l].T
                dinp_bwd = dgates @ self.W_ih_bwd[l].T
                
                if l == 0:
                    dinp_np = dinp_bwd if isinstance(dinp_bwd, np.ndarray) else xp.asnumpy(dinp_bwd)
                    X_np    = X if isinstance(X, np.ndarray) else xp.asnumpy(X)
                    np.add.at(dEx_np, X_np[:, t], dinp_np)
                else:
                    dLayer_output[l - 1][:, t, :] = dLayer_output[l - 1][:, t, :] + dinp_bwd

        # Copy accumulated dEx back to the xp backend
        self.dEx += xp.asarray(dEx_np)

    # ── Parameter access ────────────────────────────────────────────────────

    def parameters(self):
        """
        Yield (param, grad) pairs for the optimizer.
        """
        yield self.Ex, self.dEx
        for l in range(self.hp.L):
            yield self.W_ih_fwd[l], self.dW_ih_fwd[l]
            yield self.W_hh_fwd[l], self.dW_hh_fwd[l]
            yield self.b_fwd[l],    self.db_fwd[l]
            yield self.W_ih_bwd[l], self.dW_ih_bwd[l]
            yield self.W_hh_bwd[l], self.dW_hh_bwd[l]
            yield self.b_bwd[l],    self.db_bwd[l]

    def zero_grad(self):
        """Zero all gradient arrays in-place."""
        self.dEx[:] = 0.0
        for l in range(self.hp.L):
            self.dW_ih_fwd[l][:] = 0.0
            self.dW_hh_fwd[l][:] = 0.0
            self.db_fwd[l][:]    = 0.0
            
            self.dW_ih_bwd[l][:] = 0.0
            self.dW_hh_bwd[l][:] = 0.0
            self.db_bwd[l][:]    = 0.0
