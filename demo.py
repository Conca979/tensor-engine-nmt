"""
demo.py — Interactive Live Demo Translation on Terminal.

Features:
  - Automatically locates and loads the latest trained checkpoint (*.npz)
  - Auto-initializes Byte-Pair Encoding (BPE) vocabulary and model architecture
  - Detects active compute device (CuPy GPU or NumPy CPU)
  - Interactive REPL with support for Beam Search and Greedy decoding
  - Live comparison mode (Greedy vs. Beam Search side-by-side with latency)
  - In-session commands to change beam width, decode mode, or list checkpoints

Usage:
    python demo.py
    python demo.py --ckpt checkpoints/step_34000.npz
    python demo.py --method both --beam-width 5
    uv run tensor-engine-nmt translate
"""
import os
import re
import sys
import time
import glob
import argparse
from pathlib import Path

# Ensure src/ is on the python search path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tensor_engine_nmt.config import cfg, HParams
from tensor_engine_nmt.backend import BACKEND, xp
from tensor_engine_nmt.bpe import load_or_train_bpe
from tensor_engine_nmt.model import Seq2Seq
from tensor_engine_nmt.inference import Translator


# ── ANSI Styling ─────────────────────────────────────────────────────────────
class Color:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    MAGENTA = "\033[95m"
    BLUE    = "\033[94m"
    RED     = "\033[91m"


def find_latest_checkpoint(ckpt_dir: str = "checkpoints") -> str | None:
    """Find the checkpoint with the highest numerical step in ckpt_dir."""
    if not os.path.exists(ckpt_dir):
        return None

    candidates = []
    for f in glob.glob(os.path.join(ckpt_dir, "*.npz")):
        # Ignore optimizer state files
        if "optim" in os.path.basename(f).lower():
            continue
        m = re.search(r"step_(\d+)", os.path.basename(f))
        step = int(m.group(1)) if m else -1
        candidates.append((step, f))

    if not candidates:
        return None

    # Sort numerically by step number
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def list_checkpoints(ckpt_dir: str = "checkpoints") -> list[tuple[int, str, float]]:
    """Return all valid checkpoints sorted by step with file size in MB."""
    if not os.path.exists(ckpt_dir):
        return []

    ckpts = []
    for f in glob.glob(os.path.join(ckpt_dir, "*.npz")):
        if "optim" in os.path.basename(f).lower():
            continue
        m = re.search(r"step_(\d+)", os.path.basename(f))
        step = int(m.group(1)) if m else -1
        sz_mb = os.path.getsize(f) / (1024 * 1024)
        ckpts.append((step, f, sz_mb))

    ckpts.sort(key=lambda x: x[0])
    return ckpts


class LiveDemoApp:
    def __init__(self, ckpt_path: str | None = None, method: str = "beam", beam_width: int = 4):
        self.method = method
        self.beam_width = beam_width
        self.hp = cfg
        self.ckpt_path = ckpt_path

        self._init_system()

    def _init_system(self):
        print(f"\n{Color.CYAN}╔══════════════════════════════════════════════════════════════════╗{Color.RESET}")
        print(f"{Color.CYAN}║      Tensor Engine NMT — Interactive Live Translation Demo       ║{Color.RESET}")
        print(f"{Color.CYAN}╚══════════════════════════════════════════════════════════════════╝{Color.RESET}\n")

        # 1. Device Info
        dev_color = Color.GREEN if BACKEND == "cupy" else Color.YELLOW
        dev_desc = "GPU (CuPy acceleration active)" if BACKEND == "cupy" else "CPU (NumPy fallback)"
        print(f"  {Color.BOLD}Compute Device :{Color.RESET} {dev_color}{dev_desc}{Color.RESET}")

        # 2. Vocabulary Info
        print(f"  {Color.BOLD}Vocabulary     :{Color.RESET} Loading BPE vocab from '{self.hp.bpe_dir}'...")
        self.bpe = load_or_train_bpe(self.hp.bpe_dir, self.hp.data_dir)
        vocab_sz = len(self.bpe.token2id)
        print(f"  {Color.BOLD}Total Subwords :{Color.RESET} {Color.GREEN}{vocab_sz:,} tokens{Color.RESET}")

        # 3. Model Architecture
        print(f"  {Color.BOLD}Architecture   :{Color.RESET} 3-layer BiLSTM Encoder + 3-layer Decoder (d={self.hp.d}, e={self.hp.e})")
        print(f"  {Color.BOLD}Attention      :{Color.RESET} Luong General Attention ({self.hp.d}x{self.hp.d})")

        # 4. Checkpoint Auto-Discovery
        if not self.ckpt_path:
            self.ckpt_path = find_latest_checkpoint(self.hp.ckpt_dir)

        print(f"  {Color.BOLD}Checkpoint     :{Color.RESET} ", end="")
        self.model = Seq2Seq(self.hp)

        if self.ckpt_path and os.path.exists(self.ckpt_path):
            sz_mb = os.path.getsize(self.ckpt_path) / (1024 * 1024)
            step_m = re.search(r"step_(\d+)", os.path.basename(self.ckpt_path))
            step_tag = f"Step {step_m.group(1)}" if step_m else "Custom Checkpoint"
            print(f"{Color.GREEN}{self.ckpt_path} [{step_tag}, {sz_mb:.1f} MB]{Color.RESET}")
            print(f"  {Color.DIM}Loading weights into tensor engine...{Color.RESET}", end="", flush=True)
            self.model.load(self.ckpt_path)
            print(f" {Color.GREEN}Loaded successfully!{Color.RESET}\n")
        else:
            print(f"{Color.RED}[None found in '{self.hp.ckpt_dir}']{Color.RESET}")
            print(f"  {Color.YELLOW}Notice: Using randomly initialized weights. Download or save a .npz checkpoint into 'checkpoints/' to get real translations.{Color.RESET}\n")

        self.translator = Translator(self.model, self.bpe, self.hp)

    def print_help(self):
        print(f"\n{Color.BOLD}Interactive Commands:{Color.RESET}")
        print(f"  {Color.CYAN}:mode beam{Color.RESET}    — Use Beam Search decoding (default)")
        print(f"  {Color.CYAN}:mode greedy{Color.RESET}  — Use Greedy argmax decoding (fastest)")
        print(f"  {Color.CYAN}:mode both{Color.RESET}    — Compare both Beam Search & Greedy side-by-side")
        print(f"  {Color.CYAN}:beam <N>{Color.RESET}     — Change beam width (e.g. :beam 5)")
        print(f"  {Color.CYAN}:ckpt{Color.RESET}         — List all saved checkpoints in checkpoints/")
        print(f"  {Color.CYAN}:help{Color.RESET}         — Show this help message")
        print(f"  {Color.CYAN}:quit{Color.RESET} (or q)  — Exit the demo\n")

    def run(self):
        print(f"{Color.DIM}━" * 68 + f"{Color.RESET}")
        print(f"  Ready! Type an English sentence and press Enter.")
        print(f"  Current mode: {Color.BOLD}{self.method.upper()}{Color.RESET} (beam_width={self.beam_width}) | Type {Color.CYAN}:help{Color.RESET} for options")
        print(f"{Color.DIM}━" * 68 + f"{Color.RESET}\n")

        while True:
            try:
                prompt_str = f"{Color.BOLD}{Color.BLUE}EN >{Color.RESET} "
                src = input(prompt_str).strip()

                if not src:
                    continue

                # In-session commands
                if src.startswith(":"):
                    self._handle_command(src)
                    continue

                if src.lower() in ("quit", "exit", "q"):
                    print(f"\n{Color.CYAN}Goodbye! 👋{Color.RESET}")
                    break

                self._translate_and_display(src)

            except (KeyboardInterrupt, EOFError):
                print(f"\n\n{Color.CYAN}Exiting demo. Goodbye! 👋{Color.RESET}")
                break

    def _handle_command(self, cmd_str: str):
        parts = cmd_str[1:].strip().split()
        if not parts:
            return

        cmd = parts[0].lower()

        if cmd in ("help", "h"):
            self.print_help()

        elif cmd == "mode" and len(parts) > 1:
            m = parts[1].lower()
            if m in ("beam", "greedy", "both"):
                self.method = m
                print(f"  {Color.GREEN}✓ Decoding mode set to: {m.upper()}{Color.RESET}\n")
            else:
                print(f"  {Color.RED}Invalid mode. Choose 'beam', 'greedy', or 'both'.{Color.RESET}\n")

        elif cmd == "beam" and len(parts) > 1:
            try:
                bw = int(parts[1])
                if bw < 1:
                    raise ValueError
                self.beam_width = bw
                print(f"  {Color.GREEN}✓ Beam width set to: {bw}{Color.RESET}\n")
            except ValueError:
                print(f"  {Color.RED}Please provide a positive integer, e.g. :beam 4{Color.RESET}\n")

        elif cmd in ("ckpt", "checkpoints"):
            ckpts = list_checkpoints(self.hp.ckpt_dir)
            if not ckpts:
                print(f"  {Color.YELLOW}No checkpoints found in '{self.hp.ckpt_dir}/'{Color.RESET}\n")
            else:
                print(f"\n  {Color.BOLD}Available Checkpoints:{Color.RESET}")
                for step, p, sz in ckpts:
                    is_cur = " (ACTIVE)" if p == self.ckpt_path else ""
                    print(f"    - {os.path.basename(p):<24} [Step {step:>6}]  {sz:>5.1f} MB{Color.GREEN}{is_cur}{Color.RESET}")
                print()

        elif cmd == "quit" or cmd == "q":
            raise KeyboardInterrupt

        else:
            print(f"  {Color.RED}Unknown command '{cmd_str}'. Type :help for commands.{Color.RESET}\n")

    def _translate_and_display(self, src: str):
        # Subword tokenization preview
        tokens = self.bpe.encode(src, add_start=False, add_end=True)
        token_count = len(tokens)

        if self.method in ("greedy", "beam"):
            t0 = time.perf_counter()
            if self.method == "beam":
                hyp = self.translator.beam_decode(src, beam_width=self.beam_width)
            else:
                hyp = self.translator.greedy_decode(src)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            hyp_words = len(hyp.split())

            print(f"{Color.BOLD}{Color.GREEN}VI >{Color.RESET} {Color.BOLD}{hyp}{Color.RESET}")
            print(f"     {Color.DIM}[{self.method.upper()} | in: {token_count} BPE tok | out: {hyp_words} words | {elapsed_ms:.1f} ms]{Color.RESET}\n")

        elif self.method == "both":
            # Greedy
            t0 = time.perf_counter()
            hyp_greedy = self.translator.greedy_decode(src)
            t_greedy = (time.perf_counter() - t0) * 1000.0

            # Beam Search
            t0 = time.perf_counter()
            hyp_beam = self.translator.beam_decode(src, beam_width=self.beam_width)
            t_beam = (time.perf_counter() - t0) * 1000.0

            print(f"{Color.BOLD}{Color.YELLOW}Greedy >{Color.RESET} {hyp_greedy}  {Color.DIM}({t_greedy:.1f} ms){Color.RESET}")
            print(f"{Color.BOLD}{Color.GREEN}Beam   >{Color.RESET} {Color.BOLD}{hyp_beam}{Color.RESET}  {Color.DIM}(width={self.beam_width} | {t_beam:.1f} ms){Color.RESET}\n")


def main():
    parser = argparse.ArgumentParser(description="Live Interactive Translation REPL for Tensor Engine NMT")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to specific .npz checkpoint")
    parser.add_argument("--method", type=str, default="beam", choices=["beam", "greedy", "both"], help="Decoding method")
    parser.add_argument("--beam-width", type=int, default=4, help="Beam width for beam search")
    args = parser.parse_args()

    app = LiveDemoApp(ckpt_path=args.ckpt, method=args.method, beam_width=args.beam_width)
    app.run()


if __name__ == "__main__":
    main()
