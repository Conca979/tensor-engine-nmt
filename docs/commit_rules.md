# Git Commit Rules & Workflow Standard

This document outlines the official git commit rules and conventions for `tensor-engine-nmt`. All contributors and automated assistants must follow these standards to keep the repository history clean, semantic, and easily traceable.

---

## 1. Commit Message Format

Commit messages strictly follow the [Conventional Commits v1.0.0](https://www.conventionalcommits.org/) specification:

```text
<type>(<scope>): <short imperative summary>

[optional body explaining motivation, technical context, and impact]

[optional footer(s), e.g. BREAKING CHANGE: ..., Closes #123]
```

### Example
```text
fix(model): align BiLSTM state handoff and backward gradient injection

Initialize the decoder backward hidden state from t=0 instead of
t=Xlen[b]-1, ensuring the decoder receives the complete right-to-left
sentence context. Correct the corresponding BPTT backward gradient
injection index.
```

---

## 2. Commit Types

| Type | Description | When to use |
|---|---|---|
| `feat` | New feature | Adding a new capability (e.g., beam search, short-session timer, bucket sampler). |
| `fix` | Bug fix | Correcting unintended behavior, mathematical bugs, shape mismatches, or leaks. |
| `docs` | Documentation | Changes only to documentation files (`.md`, docstrings, guides). |
| `test` | Tests | Adding, modifying, or refactoring unit tests, gradient checks, or smoke tests. |
| `perf` | Performance | Optimizations that improve tok/s, reduce VRAM, or speed up array operations. |
| `refactor` | Refactoring | Code restructuring without adding features or fixing bugs. |
| `chore` | Maintenance | Packaging, dependencies, `pyproject.toml`, `.gitignore`, or tooling updates. |

---

## 3. Standard Scopes

Scopes categorize the component or subsystem modified:

### Neural Engine & Architecture
- `(model)`: Top-level `Seq2Seq` coordinator, parameter serialization, state handoff
- `(encoder)`: Forward/backward BiLSTM encoder layers and unrolled state gathering
- `(decoder)`: LSTM decoder layers, step-by-step recurrence, cell states
- `(attention)`: Luong general attention, alignment scores, context vectors
- `(activations)`: Softmax, Sigmoid, Tanh, and their analytical derivatives
- `(loss)`: Masked cross-entropy loss, token normalization
- `(backend)`: NumPy/CuPy abstraction layer (`xp`), dtype aliases (`f32`, `i32`)

### Data & Tokenization
- `(bpe)`: BPE tokenizer, vocabulary loading, merge rules, encoding/decoding
- `(dataset)`: Length bucketing, streaming generator, batch collation, cache loading
- `(config)`: Hyperparameters, `HParams` dataclass, preset configurations

### Training & Evaluation
- `(train)`: Main training loop, scheduled sampling (teacher forcing), learning rate
- `(optimizer)`: Adam optimizer, gradient norm clipping, moment updates
- `(inference)`: Greedy decoding, beam search, length normalization
- `(evaluate)`: BLEU-4 corpus calculation, split evaluation, metric reporting

### Tools & Deployment
- `(tools)`: Standalone utilities (`analyze.py`, `demo.py`, `zip_for_kaggle.py`)
- `(bench)`: Benchmarking tool measuring hardware throughput and epoch estimates
- `(project)`: Project configuration, environment setup, packaging

---

## 4. Subject Line Formatting Rules

1. **Imperative mood:** Write as an instruction.
   - Good: `feat(inference): add length normalization to beam search`
   - Bad: `feat(inference): added length normalization` / `adds length normalization`
2. **Lowercase:** Start the subject with a lowercase letter after the colon.
   - Good: `fix(dataset): handle empty lines in parallel corpus`
   - Bad: `fix(dataset): Handle empty lines in parallel corpus`
3. **No trailing period:** Do not put a period (`.`) at the end of the subject line.
4. **Character limit:** Keep the subject line under **72 characters**.

---

## 5. Body Guidelines

A body is recommended for non-trivial changes (`feat`, `fix`, `perf`):
- **Explain "Why", not just "What":** State the underlying problem, mathematical justification, or architectural reason for the change.
- **Quantify impact where possible:** (e.g., *"Reduced peak VRAM by 1.8 GB"*, *"Boosted throughput from 900 to 1,400 tok/s"*).
- Wrap body paragraphs at **72–80 characters**.

---

## 6. Repository Hygiene & Guardrails

To protect repository storage and maintain git performance, **NEVER commit**:

| Forbidden Artifact | Reason | Correct Handling |
|---|---|---|
| `*.npz` / `*.pt` | Heavy binary weights (100 MB–2 GB) | Saved to `checkpoints/` (ignored by git). Upload to Kaggle Dataset or Drive. |
| `*.pkl` | Large preprocessed dataset caches (300 MB+) | Saved to `PhoMT_dataset/` (ignored by git). Rebuilt at runtime or bundled via zip script. |
| `PhoMT_dataset/` | Raw parallel corpus (500 MB+) | Downloaded on target machine or attached as external dataset. |
| `*.zip` | Archive files | Generated on-demand via `python zip_for_kaggle.py`. |
| `.venv/` / `venv/` | Python virtual environments | Managed locally via `uv` or `pip`. |
| `*.log` / `train_logs.txt` | Active training logs | Kept in `checkpoints/` or project root (ignored by git). |

---

## 7. Pre-Commit Checklist

Before staging and committing your changes:

```powershell
# 1. Verify tensor shapes and forward/backward dimensions
uv run python test/test_shapes.py

# 2. Verify numerical stability with a smoke test run
uv run python test/test_smoke.py

# 3. Check git status to ensure no forbidden binary artifacts are staged
git status --short
```
