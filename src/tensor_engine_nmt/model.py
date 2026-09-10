"""
model.py — Seq2Seq wrapper.

Wires EncoderLSTM + DecoderLSTM together into a single object:
  - forward()  : encoder → handoff → decoder → logits + loss
  - backward() : loss.backward → decoder.backward → encoder.backward
  - parameters(): flat list of (param, grad) for Adam
  - save/load  : checkpoint via np.savez / np.load
"""
import os
import numpy as np
from .backend import xp, f32
from .config import cfg
from .encoder import EncoderLSTM
from .decoder import DecoderLSTM
from . import loss as loss_module


class Seq2Seq:
    """
    Complete Seq2Seq NMT model.

    Parameters
    ----------
    hp : HParams — defaults to cfg
    """

    def __init__(self, hp=cfg):
        self.hp      = hp
        self.encoder = EncoderLSTM(hp)
        self.decoder = DecoderLSTM(hp)
        # State kept between forward and backward
        self._fwd_state = None

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        X:            np.ndarray,   # (B, Tx) int32
        Xlen:         np.ndarray,   # (B,)    int32
        Yin:          np.ndarray,   # (B, Ty) int32
        Yout:         np.ndarray,   # (B, Ty) int32
        Ylen:         np.ndarray,   # (B,)    int32
        global_step:  int = 0,
    ) -> tuple:
        """
        Run the full forward pass.

        Returns
        -------
        loss   : float  — scalar mean per-token NLL
        logits : float32 (B, Ty, V) — raw output scores (for inspect / eval)
        """
        # ── Encoder ──────────────────────────────────────────────────────────
        H, h_enc_all, C_enc_all, enc_cache = self.encoder.forward(X, Xlen)

        # ── Decoder ──────────────────────────────────────────────────────────
        logits, dec_cache = self.decoder.forward(
            Yin, H, Xlen, h_enc_all, C_enc_all, global_step
        )

        # ── Loss ─────────────────────────────────────────────────────────────
        loss_val, p, mask_f = loss_module.forward(logits, Yout, Ylen)

        self._fwd_state = {
            "logits": logits,
            "Yout":   Yout,
            "p":      p,
            "mask_f": mask_f,
            "h_enc_all": h_enc_all,
            "C_enc_all": C_enc_all,
        }

        return loss_val, logits

    # ── Backward ─────────────────────────────────────────────────────────────

    def backward(self) -> None:
        """
        Run the full backward pass using the cached forward state.
        Accumulates gradients into all parameter .grad arrays.
        """
        state = self._fwd_state

        # ── Loss backward ────────────────────────────────────────────────────
        dlogits = loss_module.backward(state["p"], state["Yout"], state["mask_f"])
        # dlogits: (B, Ty, V)

        # ── Decoder backward ─────────────────────────────────────────────────
        dH, dh_enc, dC_enc = self.decoder.backward(dlogits)
        # dH: (B, Tx, d),  dh_enc/dC_enc: (L, B, d)

        # ── Encoder backward ─────────────────────────────────────────────────
        self.encoder.backward(dH, dh_enc, dC_enc)

    # ── Utility ──────────────────────────────────────────────────────────────

    def zero_grad(self) -> None:
        """Zero all gradient arrays."""
        self.encoder.zero_grad()
        self.decoder.zero_grad()

    def parameters(self):
        """
        Yield (param, grad) tuples for the Adam optimizer.
        Encoder params first, then decoder params.
        """
        yield from self.encoder.parameters()
        yield from self.decoder.parameters()

    # ── Checkpointing ────────────────────────────────────────────────────────

    def save(self, path: str, optim=None) -> None:
        """Save model weights to `path` and Adam state to a companion file.

        Adam state is saved separately to `<ckpt_dir>/optim_state.npz` and
        is always overwritten — so it only occupies space once, not per checkpoint.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        arrays = {}
        for idx, (param, _) in enumerate(self.parameters()):
            p_np = param if isinstance(param, np.ndarray) else xp.asnumpy(param)
            arrays[f"p{idx}"] = p_np
        np.savez_compressed(path, **arrays)
        print(f"[model] Checkpoint saved -> {path}")

        # Save Adam state to a single companion file (always overwritten, not duplicated)
        if optim is not None:
            optim_path = os.path.join(os.path.dirname(path) or ".", "optim_state.npz")
            opt_arrays = {"adam_t": np.array([optim.t], dtype=np.int64)}
            for idx, m in enumerate(optim.m):
                opt_arrays[f"adam_m{idx}"] = m if isinstance(m, np.ndarray) else xp.asnumpy(m)
            for idx, v in enumerate(optim.v):
                opt_arrays[f"adam_v{idx}"] = v if isinstance(v, np.ndarray) else xp.asnumpy(v)
            np.savez_compressed(optim_path, **opt_arrays)


    def load(self, path: str, optim=None) -> None:
        """Load model weights from `path` and Adam state from companion optim_state.npz."""
        data = np.load(path)
        params = list(self.parameters())
        for idx, (param, _) in enumerate(params):
            key = f"p{idx}"
            if key not in data:
                raise KeyError(f"Checkpoint missing key '{key}'")
            p_np = data[key].astype(np.float32)
            if isinstance(param, np.ndarray):
                param[:] = p_np
            else:
                param[:] = xp.asarray(p_np)

        # Restore Adam state from companion file if available
        if optim is not None:
            optim_path = os.path.join(os.path.dirname(path) or ".", "optim_state.npz")
            if os.path.exists(optim_path):
                opt = np.load(optim_path)
                optim.t = int(opt["adam_t"][0])
                for idx in range(len(optim.m)):
                    optim.m[idx][:] = xp.asarray(opt[f"adam_m{idx}"].astype(np.float32))
                    optim.v[idx][:] = xp.asarray(opt[f"adam_v{idx}"].astype(np.float32))
                print(f"[model] Adam state restored (t={optim.t}) from {optim_path}")
            else:
                print(f"[model] Warning: no optim_state.npz found, optimizer starts fresh")
        print(f"[model] Checkpoint loaded <- {path}")


    def param_count(self) -> int:
        """Total number of trainable scalar parameters."""
        return sum(p.size for (p, _) in self.parameters())
