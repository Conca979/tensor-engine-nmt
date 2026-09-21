"""
tensor_engine_nmt — English → Vietnamese NMT engine.

CLI entry points:
    tensor-engine-nmt train      Train from scratch or resume
    tensor-engine-nmt translate  Interactive translation REPL
    tensor-engine-nmt evaluate   Corpus BLEU-4 on test set

All heavy work is delegated to the relevant module.
"""
import sys


def main() -> None:
    """Dispatch to sub-commands based on sys.argv[1]."""
    if len(sys.argv) < 2:
        _print_help()
        sys.exit(0)

    cmd = sys.argv.pop(1)     # remove subcommand so argparse in sub-module sees clean argv

    if cmd == "train":
        from .train import main_train
        main_train()

    elif cmd == "bench":
        from .bench import main_bench
        main_bench()

    elif cmd == "translate":
        import glob
        import os
        import re
        from .config import cfg
        from .bpe import load_or_train_bpe
        from .model import Seq2Seq
        from .inference import Translator

        bpe   = load_or_train_bpe()
        model = Seq2Seq(cfg)
        ckpts = []
        for f in glob.glob(os.path.join(cfg.ckpt_dir, "*.npz")):
            base = os.path.basename(f)
            # Skip the Adam companion file and anything that is not a checkpoint
            # (e.g. a dataset.npz bundle dropped in the same directory).
            if "optim" in base.lower():
                continue
            m = re.search(r"(?:step|epoch)_(\d+)", base)
            if not m:
                continue
            # A step checkpoint wins over an epoch checkpoint with the same number.
            ckpts.append((int(m.group(1)), base.startswith("step_"), f))
        ckpts.sort(key=lambda x: (x[0], x[1]))
        if ckpts:
            print(f"[model] Loaded checkpoint: {ckpts[-1][2]}")
            model.load(ckpts[-1][2])
        else:
            print("[WARNING] No checkpoint found in 'checkpoints/' — using untrained weights.")
        Translator(model, bpe, cfg).interactive()

    elif cmd == "evaluate":
        from .evaluate import main_evaluate
        main_evaluate()

    else:
        print(f"Unknown command: {cmd!r}")
        _print_help()
        sys.exit(1)


def _print_help() -> None:
    print(
        "tensor-engine-nmt — EN→VI NMT\n"
        "\n"
        "Commands:\n"
        "  bench      Measure tok/s and hours-per-epoch on THIS machine (start here)\n"
        "  train      Train the model\n"
        "               --preset gpu|laptop   --max-minutes N     --save-every N\n"
        "               --keep-last N         --max-pairs N       --lr LR\n"
        "               --k K                 --min-tf TF         --max-tokens N\n"
        "               --max-steps N         --resume PATH\n"
        "  translate  Interactive translation REPL\n"
        "  evaluate   Corpus BLEU-4 evaluation (--ckpt PATH  --verbose  --n N  --split SPLIT  --random)\n"
    )
