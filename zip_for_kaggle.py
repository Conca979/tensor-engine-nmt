"""
zip_for_kaggle.py — Create Kaggle-compatible ZIP archives using POSIX forward slashes (/).

Windows default zip utilities (like PowerShell Compress-Archive or File Explorer)
embed Windows backslashes (\\) into file paths inside the archive. Kaggle runs Linux
and rejects these archives with:
    "contains a forbidden character in name ('\\')"

Presets:
    python zip_for_kaggle.py code          -> tensor-engine-nmt-code.zip
    python zip_for_kaggle.py torch         -> phomt-transformer-assets.zip (for PyTorch Transformer)
    python zip_for_kaggle.py PhoMT_dataset -> PhoMT_dataset.zip (for NumPy/CuPy BiLSTM)
    python zip_for_kaggle.py checkpoints   -> checkpoints.zip

Custom multi-item command:
    python zip_for_kaggle.py -o custom.zip folder1 file2.pkl file3.txt
    python zip_for_kaggle.py bpe_vocab PhoMT_dataset/train_cache_V30839.pkl PhoMT_dataset/test/test.en PhoMT_dataset/test/test.vi
"""
import os
import sys
import glob
import zipfile


def zip_code():
    out_zip = "tensor-engine-nmt-code.zip"
    items = ["src", "test", "bpe_vocab", "demo.py", "demo_torch.py", "analyze.py", "pyproject.toml", "README.md", "COLAB_GUIDE.md", "KAGGLE_GUIDE.md"]
    print(f"Packaging codebase into {out_zip}...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in items:
            if os.path.isdir(item):
                for root, dirs, files in os.walk(item):
                    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".venv", ".git")]
                    for f in files:
                        if f.endswith((".pyc", ".pyo")):
                            continue
                        full = os.path.join(root, f)
                        arcname = os.path.relpath(full).replace("\\", "/")
                        zf.write(full, arcname=arcname)
            elif os.path.isfile(item):
                zf.write(item, arcname=item.replace("\\", "/"))
    print(f"Success! Created {out_zip} (All paths verified with POSIX '/')")


def zip_torch_assets(out_zip="phomt-transformer-assets.zip"):
    """Bundle all assets needed for PyTorch Transformer on Kaggle."""
    print(f"Packaging PyTorch Transformer assets into '{out_zip}'...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. BPE vocabulary folder
        bpe_dir = "bpe_vocab"
        if os.path.exists(bpe_dir):
            for root, dirs, files in os.walk(bpe_dir):
                for f in files:
                    full = os.path.join(root, f)
                    arcname = os.path.relpath(full).replace("\\", "/")
                    print(f"  + {full} -> {arcname}")
                    zf.write(full, arcname=arcname)
        else:
            print(f"  ! Warning: '{bpe_dir}' not found.")

        # 2. Tokenized train cache (check PhoMT_dataset/ then root)
        cache_candidates = glob.glob("PhoMT_dataset/*cache*.pkl") + glob.glob("*cache*.pkl")
        if cache_candidates:
            cache_file = cache_candidates[0]
            arcname = os.path.basename(cache_file)
            print(f"  + {cache_file} -> {arcname}")
            zf.write(cache_file, arcname=arcname)
        else:
            print("  ! Warning: train cache .pkl not found.")

        # 3. Test files (test.en and test.vi)
        for lang in ["en", "vi"]:
            test_candidates = [
                f"PhoMT_dataset/test/test.{lang}",
                f"PhoMT_dataset/test.{lang}",
                f"test.{lang}",
            ]
            found = False
            for cand in test_candidates:
                if os.path.exists(cand):
                    arcname = f"test.{lang}"
                    print(f"  + {cand} -> {arcname}")
                    zf.write(cand, arcname=arcname)
                    found = True
                    break
            if not found:
                print(f"  ! Warning: test.{lang} not found.")

    print(f"Success! Created {out_zip} (Ready to upload to Kaggle as 'phomt-transformer-assets')")


def zip_custom_items(items, out_zip="bundle.zip"):
    """Package arbitrary list of files and folders into one POSIX-clean zip archive."""
    print(f"Packaging {len(items)} items into '{out_zip}'...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in items:
            item = os.path.normpath(item)
            if not os.path.exists(item):
                print(f"  ! Warning: '{item}' does not exist, skipping.")
                continue

            if os.path.isdir(item):
                for root, dirs, files in os.walk(item):
                    dirs[:] = [d for d in dirs if d not in ("__pycache__", ".venv", ".git")]
                    for f in files:
                        if f.endswith((".pyc", ".pyo")):
                            continue
                        full = os.path.join(root, f)
                        rel = os.path.relpath(full, os.path.dirname(item) or ".")
                        arcname = rel.replace("\\", "/")
                        print(f"  + {full} -> {arcname}")
                        zf.write(full, arcname=arcname)
            elif os.path.isfile(item):
                arcname = os.path.basename(item)
                print(f"  + {item} -> {arcname}")
                zf.write(item, arcname=arcname)

    print(f"Success! Created {out_zip} (All paths verified with POSIX '/')")


def zip_folder(folder_path, out_zip=None):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    folder_name = os.path.normpath(folder_path).split(os.sep)[-1]
    if out_zip is None:
        out_zip = f"{folder_name}.zip"
    print(f"Packaging '{folder_path}' into '{out_zip}'...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder_path):
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".venv", ".git")]
            for f in files:
                if f.endswith((".pyc", ".pyo")):
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, os.path.dirname(folder_path) or ".")
                arcname = rel.replace("\\", "/")
                zf.write(full, arcname=arcname)
        
        if folder_name == "PhoMT_dataset":
            for pkl_file in glob.glob("*_cache_*.pkl"):
                arcname = f"{folder_name}/{pkl_file}"
                print(f"  + Including {pkl_file} -> {arcname}")
                zf.write(pkl_file, arcname=arcname)

    print(f"Success! Created {out_zip} (All paths verified with POSIX '/')")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "code":
        zip_code()
    elif args[0] in ("torch", "torch_assets", "transformer"):
        out_name = args[1] if len(args) > 1 else "phomt-transformer-assets.zip"
        zip_torch_assets(out_name)
    elif len(args) == 1 and os.path.isdir(args[0]):
        zip_folder(args[0])
    else:
        out_zip = "phomt-transformer-assets.zip" if any("cache" in a.lower() for a in args) else "bundle.zip"
        items = []
        i = 0
        while i < len(args):
            if args[i] in ("-o", "--out") and i + 1 < len(args):
                out_zip = args[i + 1]
                i += 2
            else:
                items.append(args[i])
                i += 1
        zip_custom_items(items, out_zip)

