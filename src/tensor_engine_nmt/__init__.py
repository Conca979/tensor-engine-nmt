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

    elif cmd == "translate":
        import glob
        import os
        from .config import cfg
        from .bpe import load_or_train_bpe
        from .model import Seq2Seq
        from .inference import Translator

        bpe   = load_or_train_bpe()
        model = Seq2Seq(cfg)
        ckpts = sorted(glob.glob(os.path.join(cfg.ckpt_dir, "*.npz")))
        if ckpts:
            model.load(ckpts[-1])
        else:
            print("[WARNING] No checkpoint found — using untrained weights.")
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
        "  train      Train the model (--max-steps N  --resume PATH  --lr LR  --min-tf TF)\n"
        "  translate  Interactive translation REPL\n"
        "  evaluate   Corpus BLEU-4 evaluation (--ckpt PATH  --verbose  --n N  --split SPLIT  --random)\n"
    )
