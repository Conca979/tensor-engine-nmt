# Project Structure & Architecture

This document provides a comprehensive overview of the `tensor-engine-nmt` project structure. This codebase is a from-scratch implementation of an English-to-Vietnamese Neural Machine Translation (NMT) system, utilizing a Bidirectional LSTM (BiLSTM) with Attention. It is built entirely using NumPy (for CPU) and CuPy (for GPU acceleration), avoiding high-level frameworks like PyTorch or TensorFlow.

## Directory Structure

```text
tensor-engine-nmt/
├── src/
│   └── tensor_engine_nmt/  # Main source code package
│       ├── __init__.py     # CLI entry point
│       ├── activations.py  # Activation functions (Softmax, Sigmoid, etc.)
│       ├── attention.py    # Attention mechanism (Luong / Dot-product)
│       ├── backend.py      # Abstracted NumPy/CuPy backend (xp)
│       ├── bpe.py          # Byte Pair Encoding tokenizer
│       ├── config.py       # Global hyperparameters and configuration
│       ├── dataset.py      # Data loading, length bucketing, and batching
│       ├── decoder.py      # LSTM Decoder with Attention
│       ├── encoder.py      # Bidirectional LSTM Encoder
│       ├── evaluate.py     # Corpus BLEU-4 evaluation metric
│       ├── inference.py    # Beam search and translation generation
│       ├── init_weights.py # Weight initialization (Xavier, etc.)
│       ├── loss.py         # Masked Cross-Entropy Loss
│       ├── model.py        # Top-level Seq2Seq architecture; atomic checkpoints
│       ├── optimizer.py    # Adam Optimizer implementation
│       ├── bench.py        # Measures tok/s and hours-per-epoch on the local machine
│       └── train.py        # Training loop, time budgets and scheduled sampling
├── docs/                   # Documentation and Guides
│   ├── bidirectional_pipeline.md
│   ├── bugfix_report.md      # Backward-pass fixes, evidence, next steps
│   ├── commit_rules.md       # Conventional commits rules and git workflow standard
│   ├── naming_conventions.md # Official tensor naming, dimensions, and symbols
│   ├── project_structure.md
│   ├── short_session_training.md  # Training in short bursts (slow hardware / limited GPU hours)
│   ├── Teacher_Forcing_in_NMT_Google_Production.md
│   └── training_guide.md
├── test/                   # Test suite (runnable as scripts or via pytest)
│   ├── test_shapes.py
│   ├── test_gradients.py   # Finite-difference gate for the manual BPTT
│   ├── test_overfit.py
│   ├── test_smoke.py       # BPE → cache → batching → train step
│   ├── test_checkpoint.py  # save/load, stale & truncated checkpoint handling
│   └── test_short_session.py  # presets, --max-minutes stop/resume, --keep-last
├── verification/           # Targeted diagnostics + before/after evidence
│   ├── verify_bugs.py
│   ├── verify_padding_leak.py
│   └── compare_ab.py
├── bpe_vocab/              # Saved BPE vocabularies and merges
├── checkpoints/            # Saved model checkpoints and training logs
├── PhoMT_dataset/          # (Ignored) Dataset files
├── demo.py                 # Live interactive translation REPL with beam/greedy decoding
├── analyze.py              # Script to analyze training logs and evaluation results
├── zip_for_kaggle.py       # Deployment script to bundle code for Kaggle
├── COLAB_GUIDE.md          # Guide for training/evaluating on Google Colab
├── KAGGLE_GUIDE.md         # Guide for training on Kaggle
└── README.md               # Quickstart and project overview
```

---

## Core Components (File Descriptions)

### 1. Infrastructure & Core
- **`backend.py`**: The bridge between CPU (`numpy`) and GPU (`cupy`). It dynamically exports the active backend as `xp`, allowing the entire codebase to run on either CPU or GPU seamlessly.
- **`config.py`**: Defines `cfg`, the singleton configuration object holding all hyperparameters (dimensions, learning rate, vocabulary size, batch size, etc.).
- **`init_weights.py`**: Contains routines for initializing neural network weights (e.g., Xavier uniform initialization) to ensure stable training gradients.

### 2. Data Pipeline
- **`bpe.py`**: Implements a highly optimized, from-scratch Byte Pair Encoding (BPE) tokenizer. It handles training the vocabulary, merging tokens, and encoding/decoding sequences.
- **`dataset.py`**: Handles loading the PhoMT dataset. Crucially, it implements **Length Bucketing**. Sentences are grouped by length before batching to minimize padding tokens, which drastically improves both memory efficiency and training speed.

### 3. Neural Architecture
- **`activations.py`**: Contains forward and backward passes for activation functions (Softmax, Tanh, Sigmoid).
- **`encoder.py`**: Implements a Bidirectional LSTM (BiLSTM). It processes the source sentence forward and backward simultaneously, concatenating the hidden states to create rich, context-aware token representations.
- **`attention.py`**: Implements the attention mechanism, allowing the decoder to focus on specific parts of the encoder's output when predicting each word.
- **`decoder.py`**: Implements the LSTM Decoder. At each time step, it takes the previous word, its previous hidden state, and the attention context vector to predict the next word.
- **`model.py`**: The `Seq2Seq` class. It orchestrates the Encoder, Decoder, and word embeddings, managing the full forward pass, backward pass, and parameter serialization (saving/loading checkpoints).

### 4. Training & Optimization
- **`optimizer.py`**: Implements the Adam optimizer from scratch. It manages gradient updates, moment estimates, and gradient clipping to prevent exploding gradients.
- **`loss.py`**: Calculates the Masked Cross-Entropy Loss, ensuring that padded `<PAD>` tokens do not contribute to the loss or gradients.
- **`train.py`**: The main training loop. It handles data iteration, forward/backward passes, optimizer steps, logging, checkpointing, and dynamically manages **Teacher Forcing** (scheduled sampling).

### 5. Evaluation & Inference
- **`inference.py`**: Contains the `Translator` class, implementing **Beam Search** (with length normalization and repetition penalty) and Greedy search to find high-probability translations.
- **`evaluate.py`**: Implements length-filtered Corpus BLEU-4 evaluation (defaulting to Beam Search) and auto-appends structured logs to `checkpoints/evaluation_result.txt`.
- **`demo.py`**: Interactive live translation terminal application with automatic checkpoint detection, interactive commands (`:mode`, `:beam`, `:ckpt`), and latency metrics.

---

## References & Further Reading

The `docs/` folder and root guides contain extensive documentation explaining the mathematical and engineering decisions behind the project:

- **[README.md](../README.md)**: The primary entry point. Contains the architectural overview, installation instructions, and quickstart commands.
- **[naming_conventions.md](naming_conventions.md)**: Official guide for tensor shapes (`B`, `Tx`, `Ty`, `d`), mathematical notation, LSTM gate weights, backpropagation variables, and file naming conventions.
- **[commit_rules.md](commit_rules.md)**: Repository git commit conventions, allowed types/scopes, and forbidden artifact guardrails.
- **[bidirectional_pipeline.md](bidirectional_pipeline.md)**: A deep dive into how the Bidirectional LSTM encoder works, including the mathematics of concatenating forward and backward states and the impact on the attention mechanism.
- **[Teacher_Forcing_in_NMT_Google_Production.md](Teacher_Forcing_in_NMT_Google_Production.md)**: Explains the concept of Scheduled Sampling / Teacher Forcing, why it's critical to prevent "exposure bias," and how the inverse-sigmoid decay schedule is implemented.
- **[training_guide.md](training_guide.md)**: Comprehensive guide on how to train the model, manage checkpoints, resume interrupted runs, and diagnose common training issues (like loss spikes).
- **[COLAB_GUIDE.md](../COLAB_GUIDE.md)** & **[KAGGLE_GUIDE.md](../KAGGLE_GUIDE.md)**: Step-by-step instructions for running the training pipeline on free cloud GPUs.
