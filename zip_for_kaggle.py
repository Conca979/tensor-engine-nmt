"""
zip_for_kaggle.py — Create Kaggle-compatible ZIP archives using POSIX forward slashes (/).

Windows default zip utilities (like PowerShell Compress-Archive or File Explorer)
embed Windows backslashes (\\) into file paths inside the archive. Kaggle runs Linux
and rejects these archives with:
    "contains a forbidden character in name ('\\')"

Usage:
    python zip_for_kaggle.py code
    python zip_for_kaggle.py PhoMT_dataset
    python zip_for_kaggle.py checkpoints
"""
import os
import sys
import zipfile


def zip_code():
    out_zip = "tensor-engine-nmt-code.zip"
    items = ["src", "test", "bpe_vocab", "pyproject.toml", "requirements.txt", "README.md", "COLAB_GUIDE.md", "KAGGLE_GUIDE.md"]
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
                # Keep top folder name in arcname (e.g. PhoMT_dataset/train/...)
                rel = os.path.relpath(full, os.path.dirname(folder_path) or ".")
                arcname = rel.replace("\\", "/")
                zf.write(full, arcname=arcname)
    print(f"Success! Created {out_zip} (All paths verified with POSIX '/')")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "code":
        zip_code()
    else:
        target = sys.argv[1]
        zip_folder(target)
