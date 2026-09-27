"""
DeepRetail — Import a Roboflow *folder-format* (classification) export into
data/classifier_dataset/{train,valid,test}/<class_name>/

Why this exists:
    Roboflow folder names ("Haka_Noodles_Masala") must match PRODUCT_CLASSES in
    src/config.py ("chings_hakka"), because the classifier takes its label names
    from the folder names and the DB / shelf slots use PRODUCT_CLASSES.

Usage:
    python src/data_tools/prepare_classifier_dataset.py --src path/to/export.zip
    python src/data_tools/prepare_classifier_dataset.py --src path/to/export_folder
    python src/data_tools/prepare_classifier_dataset.py --src ... --clean   (wipe old data first)
"""

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import CLASSIFIER_DATASET_DIR, PRODUCT_CLASSES

# Roboflow folder name (lower-cased) → PRODUCT_CLASSES name.
# Add entries here if you rename products in Roboflow.
ALIASES: dict[str, str] = {
    "haka_noodles_masala": "chings_hakka",
    "hakka_noodles_masala": "chings_hakka",
    "veg_manchurian_masala_mix": "chings_manchurian",
    "homelite_match_box": "homelite_matchbox",
    "homelite_matchbox": "homelite_matchbox",
    "vaseline": "vaseline_jelly",
}

SPLIT_ALIASES = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def map_class(folder_name: str) -> str | None:
    key = folder_name.strip().lower()
    if key in PRODUCT_CLASSES:
        return key
    return ALIASES.get(key)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Roboflow folder export (.zip or directory)")
    p.add_argument("--clean", action="store_true", help="Delete existing train/valid/test first")
    args = p.parse_args()

    src = Path(args.src).expanduser().resolve()
    tmp = None
    if src.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp.name)
        root = Path(tmp.name)
    else:
        root = src
    if not root.is_dir():
        sys.exit(f"[Prepare] Not found: {src}")

    if args.clean:
        for s in ("train", "valid", "test"):
            shutil.rmtree(CLASSIFIER_DATASET_DIR / s, ignore_errors=True)

    counts: dict[tuple[str, str], int] = {}
    unmapped: set[str] = set()
    for split_dir in sorted(d for d in root.rglob("*") if d.is_dir() and d.name.lower() in SPLIT_ALIASES):
        split = SPLIT_ALIASES[split_dir.name.lower()]
        for cls_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            cls = map_class(cls_dir.name)
            if cls is None:
                unmapped.add(cls_dir.name)
                continue
            out = CLASSIFIER_DATASET_DIR / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for img in cls_dir.iterdir():
                if img.suffix.lower() in IMG_EXTS:
                    shutil.copy2(img, out / img.name)
                    counts[(split, cls)] = counts.get((split, cls), 0) + 1

    if tmp:
        tmp.cleanup()

    if not counts:
        sys.exit("[Prepare] No images copied. Expected <export>/train|valid|test/<class>/*.jpg")

    print(f"[Prepare] Dataset written to {CLASSIFIER_DATASET_DIR}")
    for split in ("train", "valid", "test"):
        row = {c: counts.get((split, c), 0) for c in PRODUCT_CLASSES}
        print(f"  {split:6s} " + "  ".join(f"{c}={n}" for c, n in row.items()))
    missing = [c for c in PRODUCT_CLASSES if counts.get(("train", c), 0) == 0]
    if missing:
        print(f"[WARN] No training images for: {missing}")
    if unmapped:
        print(f"[WARN] Skipped folders with no mapping: {sorted(unmapped)} "
              "→ add them to ALIASES in this script.")


if __name__ == "__main__":
    main()
