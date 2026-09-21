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

    @staticmethod
    def _atomic_savez(path: str, arrays: dict) -> None:
        """Write `arrays` to `path` via a temp file + rename.

        np.savez writes straight to the destination, so a process killed
        mid-write leaves a truncated .npz behind — which is exactly what happens
        to a multi-hundred-MB checkpoint on a Colab/Kaggle session that gets
        pre-empted.  A rename is atomic on both POSIX and Windows.
        """
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)

    def save(self, path: str, optim=None, meta: dict = None) -> None:
        """Save model weights to `path` and Adam state to a companion file.

        Adam state is saved separately to `<ckpt_dir>/optim_state.npz` and
        is always overwritten — so it only occupies space once, not per checkpoint.

        `meta` (e.g. {"epoch": 3, "step_in_epoch": 1200}) is stored INSIDE the
        checkpoint rather than in a side file, so the weights and the position in
        the dataset can never disagree after a crash.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        arrays = {}
        for idx, (param, _) in enumerate(self.parameters()):
            p_np = param if isinstance(param, np.ndarray) else xp.asnumpy(param)
            arrays[f"p{idx}"] = p_np
        for k, v in (meta or {}).items():
            arrays[f"meta_{k}"] = np.array([v], dtype=np.int64)
        self._atomic_savez(path, arrays)
        print(f"[model] Checkpoint saved -> {path}")

        # Save Adam state to a single companion file (always overwritten, not duplicated).
        # We tag it with the checkpoint it belongs to: there is only ONE such file
        # per directory, so resuming from an older step_*.npz would otherwise
        # silently pair those weights with a newer step's moments.
        if optim is not None:
            optim_path = os.path.join(os.path.dirname(path) or ".", "optim_state.npz")
            opt_arrays = {
                "adam_t": np.array([optim.t], dtype=np.int64),
                "ckpt":   np.array([os.path.basename(path)]),
            }
            for idx, m in enumerate(optim.m):
                opt_arrays[f"adam_m{idx}"] = m if isinstance(m, np.ndarray) else xp.asnumpy(m)
            for idx, v in enumerate(optim.v):
                opt_arrays[f"adam_v{idx}"] = v if isinstance(v, np.ndarray) else xp.asnumpy(v)
            self._atomic_savez(optim_path, opt_arrays)


    def load(self, path: str, optim=None) -> dict:
        """Load weights from `path`, plus Adam state from companion optim_state.npz.

        Returns the `meta` dict that was passed to save() (empty for checkpoints
        written by older versions).
        """
        data = np.load(path)
        params = list(self.parameters())
        for idx, (param, _) in enumerate(params):
            key = f"p{idx}"
            if key not in data:
                raise KeyError(f"Checkpoint missing key '{key}'")
            p_np = data[key].astype(np.float32)
            if p_np.shape != tuple(param.shape):
                raise ValueError(
                    f"Checkpoint '{key}' has shape {p_np.shape} but the model expects "
                    f"{tuple(param.shape)} — architecture parameters (d, L, e, V) must "
                    f"match the checkpoint."
                )
            if isinstance(param, np.ndarray):
                param[:] = p_np
            else:
                param[:] = xp.asarray(p_np)

        meta = {k[len("meta_"):]: int(data[k][0]) for k in data.files
                if k.startswith("meta_")}

        # Restore Adam state from companion file if available
        if optim is not None:
            optim_path = os.path.join(os.path.dirname(path) or ".", "optim_state.npz")
            if not os.path.exists(optim_path):
                print("[model] Warning: no optim_state.npz found, optimizer starts fresh")
            else:
                restored = False
                try:
                    with np.load(optim_path) as opt:
                        owner = str(opt["ckpt"][0]) if "ckpt" in opt.files else None
                        n_saved = sum(1 for k in opt.files if k.startswith("adam_m"))
                        if owner is not None and owner != os.path.basename(path):
                            print(
                                f"[model] WARNING: {optim_path} holds the optimizer state "
                                f"saved with '{owner}', but you are loading '{os.path.basename(path)}'. "
                                f"Those moments belong to different weights, so they are not a "
                                f"valid resume point — starting the optimizer fresh."
                            )
                        elif n_saved != len(optim.m):
                            print(
                                f"[model] WARNING: optimizer state has {n_saved} tensors but the "
                                f"model has {len(optim.m)} — architecture changed? "
                                f"Starting the optimizer fresh."
                            )
                        else:
                            optim.t = int(opt["adam_t"][0])
                            for idx in range(len(optim.m)):
                                optim.m[idx][:] = xp.asarray(opt[f"adam_m{idx}"].astype(np.float32))
                                optim.v[idx][:] = xp.asarray(opt[f"adam_v{idx}"].astype(np.float32))
                            restored = True
                except Exception as e:
                    # Covers a file truncated by a crash mid-write (BadZipFile),
                    # missing keys, or a shape change.
                    print(f"[model] WARNING: could not read {optim_path} "
                          f"({type(e).__name__}: {e}) — starting the optimizer fresh")

                if restored:
                    print(f"[model] Adam state restored (t={optim.t}) from {optim_path}")
                else:
                    optim.t = 0
                    for idx in range(len(optim.m)):
                        optim.m[idx][:] = 0.0
                        optim.v[idx][:] = 0.0
        print(f"[model] Checkpoint loaded <- {path}")
        if meta:
            print(f"[model] Checkpoint metadata: {meta}")
        return meta


    def param_count(self) -> int:
        """Total number of trainable scalar parameters."""
        return sum(p.size for (p, _) in self.parameters())
