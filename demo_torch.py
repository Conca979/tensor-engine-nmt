"""
demo_torch.py — Root launcher for the PyTorch Transformer Translation Demo.
===========================================================================

Convenience wrapper to run torch_experiment/demo.py from the project root.

Usage:
    python demo_torch.py
    python demo_torch.py --method both --beam-width 4
    python demo_torch.py --text "He can water the horses ."
"""
import os
import sys

script_path = os.path.join(os.path.dirname(__file__), "torch_experiment", "demo.py")

if __name__ == "__main__":
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found.")
        sys.exit(1)

    with open(script_path, "rb") as f:
        code = compile(f.read(), script_path, "exec")
        exec(code, {"__name__": "__main__", "__file__": script_path})
