# Project Naming Conventions & Notation Standard

This document establishes the official naming conventions, mathematical symbols, tensor shapes, and file structure rules used across `tensor-engine-nmt`.

Because this codebase implements deep neural network primitives (LSTMs, multi-head attention, BPTT, and Adam) from scratch without high-level frameworks, consistent variable naming and dimension notation are critical to preventing shape mismatch bugs and gradient leaks.

---

## 1. Tensor Dimension & Shape Symbols

Across all documentation, docstrings, and tensor operations, dimensions are consistently represented by single-letter or short abbreviations:

| Symbol | Description | Typical Value (GPU) | Typical Value (Laptop) |
|---|---|---|---|
| `B` | Batch size (number of sequences in a mini-batch) | Dynamic (token-bucketed) | `16`–`32` |
| `Tx` | Source sequence length (timesteps in encoder, padded) | Dynamic (`<= max_len`) | `<= 30` |
| `Ty` | Target sequence length (timesteps in decoder, padded) | Dynamic (`<= max_len`) | `<= 30` |
| `d` | Hidden state dimension (per layer, per direction) | `1024` | `256` |
| `d_half` | Half of hidden state dimension (`d // 2`) | `512` | `128` |
| `e` | Word embedding dimension | `512` | `128` |
| `L` | Number of stacked recurrent layers (encoder & decoder) | `3` | `2` |
| `V` | Vocabulary size (shared BPE vocabulary) | `30,839` | `8,000` |
| `d_ff` | Feed-forward intermediate expansion (Transformer) | `2048` | — |
| `nhead` | Number of attention heads (Transformer) | `8` | — |

---

## 2. Sequence & Batch Variables

| Variable | Shape | Type | Description |
|---|---|---|---|
| `X` | `(B, Tx)` | `int32` | Source token IDs for the encoder, padded with `PAD_ID` (`0`). |
| `Xlen` | `(B,)` | `int32` | Actual (unpadded) token count for each source sentence. |
| `Yin` | `(B, Ty)` | `int32` | Decoder input tokens (begins with `START_ID` / `BOS`, followed by target tokens). |
| `Yout` | `(B, Ty)` | `int32` | Decoder target ground-truth labels for cross-entropy loss (ends with `END_ID` / `EOS`). |
| `Ylen` | `(B,)` | `int32` | Actual (unpadded) token count for each target sentence including `END_ID`. |

---

## 3. Recurrent States & Hidden Tensors

### Encoder
| Variable | Shape | Description |
|---|---|---|
| `H` | `(B, Tx, 2*d)` or `(B, Tx, d)` | Top-layer encoder contextual memory output exposed to Attention. |
| `h_enc_all` | `(L, B, Tx, d)` | Hidden states across all `L` layers and all `Tx` timesteps (cached for BPTT). |
| `C_enc_all` | `(L, B, Tx, d)` | Cell states across all `L` layers and all `Tx` timesteps (cached for BPTT). |
| `h_fwd`, `h_bwd` | `(B, d_half)` | Forward and backward slices of a layer's hidden vector. |

### Decoder
| Variable | Structure | Description |
|---|---|---|
| `h_dec` | `list` of `L` arrays `(B, d)` | Current hidden state vector for each decoder layer at timestep `t`. |
| `C_dec` | `list` of `L` arrays `(B, d)` | Current cell state vector for each decoder layer at timestep `t`. |
| `c_t` | `(B, d)` or `(B, 2*d)` | Context vector produced by Attention over encoder memory `H`. |
| `h_tilde` | `(B, d)` | Attentional hidden state combining `[h_dec; c_t]` through `tanh(W_c @ ...)`. |
| `logits_t` | `(B, V)` | Unnormalized vocabulary logits before softmax at step `t`. |

---

## 4. Parameter & Weight Matrices

LSTM gates are packed into unified matrices for maximum vectorization:
The gates are arranged in order: **Input (`i`)**, **Forget (`f`)**, **Candidate/Cell (`c` or `g`)**, and **Output (`o`)**.

| Parameter | Shape | Role |
|---|---|---|
| `W_embed` | `(V, e)` | Token embedding lookup table (shared across source and target). |
| `W_xh` | `(e, 4*d)` (layer 1) or `(d, 4*d)` (layers 2+) | Input-to-hidden projection containing concatenated gates `[i, f, c, o]`. |
| `W_hh` | `(d, 4*d)` | Hidden-to-hidden recurrent transition matrix. |
| `b_h` | `(4*d,)` | Bias vector for the four LSTM gates. |
| `W_a` | `(d, d)` | Attention bilinear weight matrix (Luong General Attention: `score = h_dec @ W_a @ H^T`). |
| `W_c` | `(d + enc_dim, d)` | Context projection matrix mapping concatenated context and decoder state to `h_tilde`. |
| `W_s` | `(d, V)` | Final linear projection from `h_tilde` to vocabulary logits. |
| `b_s` | `(V,)` | Vocabulary output bias. |

---

## 5. Gradients & Backpropagation Naming

Gradients strictly follow the mathematical derivative prefix convention `d<Variable>`:

```text
d<Var> = ∂Loss / ∂<Var>
```

- **`dW_xh`, `dW_hh`, `db_h`**: Parameter gradients accumulated during BPTT.
- **`dh_next`, `dC_next`**: Temporal gradients passed backward from timestep `t+1` to `t`.
- **`dH`**: Gradient backpropagated from Attention into encoder memory `H` (`(B, Tx, d)`).
- **`dX`**: Gradient backpropagated into input word embeddings.
- **`gnorm`**: Global gradient L2-norm across all parameters before clipping:
  ```text
  gnorm = sqrt(sum(norm(grad)^2 for grad in all_grads))
  ```

---

## 6. Special Tokens & Reserved IDs

Special tokens are defined in `config.py` and must never be hardcoded elsewhere:

| Constant | ID | String Literal | Purpose |
|---|---|---|---|
| `PAD_ID` | `0` | `<pad>` | Sequence padding; masked out of loss and attention. |
| `UNK_ID` | `1` | `<unk>` | Out-of-vocabulary fallback. |
| `START_ID` / `BOS_ID` | `2` | `<s>` or `START` | Forced prefix fed to the decoder at step `t = 0`. |
| `END_ID` / `EOS_ID` | `3` | `</s>` or `END` | Signals sequence termination; stops greedy & beam decode. |
| Boundary marker | — | `</w>` | Suffix used by BPE tokenizer to denote word ends. |

---

## 7. Numerical Backend & Type Abstractions

To allow zero-code switching between GPU (CuPy) and CPU (NumPy), `src/tensor_engine_nmt/backend.py` defines:

- **`xp`**: Active array module (`cupy` if GPU available, otherwise `numpy`). Import `from .backend import xp`.
- **`np`**: Always standard host CPU `numpy` (reserved for dataset loading, I/O, and shape calculations).
- **`to_xp(arr)`**: Converts a CPU NumPy array to the active device array (no-op if already on device).
- **`to_np(arr)`**: Explicitly transfers device tensor back to host CPU NumPy.
- **`f32`**: Alias for `xp.float32`. All float tensors in the project MUST use `f32` (never `float64`).
- **`i32`**: Alias for `xp.int32`. All integer index and token tensors MUST use `i32` (never `int64`).

---

## 8. File & Artifact Naming Schemes

| Artifact Pattern | Example | Description |
|---|---|---|
| `{split}_cache_V{vocab}_L{max_len}.pkl` | `train_cache_V30839_L30.pkl` | Pre-tokenized, bucket-sorted dataset cache. Encodes vocabulary size and length cutoff. |
| `step_{step}.npz` | `step_196000.npz` | Model parameter checkpoint (NumPy archive containing all layer weights). |
| `step_{step}_optim.npz` | `step_196000_optim.npz` | Adam optimizer state (first moment `m` and second moment `v` arrays). |
| `step_{step}.pt` | `step_6000.pt` | PyTorch Transformer checkpoint bundle (weights, optimizer, scheduler, scaler). |
| `state.txt` | `8,15323` | Atomic checkpoint pointer tracking `epoch,batch_idx` for safe resumption. |
| `train_logs.txt` | `train_logs.txt` | Structured tabular log file appended every `log_every` steps. |
| `evaluation_result.txt` | `evaluation_result.txt` | Human-readable BLEU evaluation output with translated sample comparisons. |

---

## 9. Code Style & Indentation Rules

- **Indentation:** Exactly **2 spaces** throughout the codebase (Python and Markdown).
- **Function Naming:**
  - Forward operations: `forward(...)` or `step(...)`
  - Backward operations: `backward(...)`
  - Private helper functions: prefix with a single leading underscore `_` (e.g., `_word_to_chars()`, `_init_weights()`).
- **Imports:**
  - Hyperparameters: always import `cfg` from `tensor_engine_nmt.config`. Never hardcode numbers.
  - Backend: always import `xp`, `f32`, `i32` from `tensor_engine_nmt.backend`. Never import `cupy` directly inside model layers.
