"""
DeepRetail — YOLO Dataset YAML Generator

Creates data/yolo_dataset/data.yaml from the existing image/label folder structure.
Also optionally splits images+labels into train/val sets.

Usage:
    python src/data_tools/create_yolo_yaml.py
    python src/data_tools/create_yolo_yaml.py --split 0.8
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import YOLO_DATASET_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=float, default=0.8,
                   help="Train fraction (default 0.8 → 80% train, 20% val)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def split_dataset(split: float, seed: int) -> None:
    """
    Takes images from data/yolo_dataset/images/ and labels from
    data/yolo_dataset/labels/ and splits into train/val subdirectories.
    """
    img_src = YOLO_DATASET_DIR / "images"
    lbl_src = YOLO_DATASET_DIR / "labels"

    # Check if already split
    if (img_src / "train").exists() and any((img_src / "train").glob("*.jpg")):
        print("[YOLO YAML] Dataset already split into train/val. Skipping split.")
        return

    images = sorted(list(img_src.glob("*.jpg")) + list(img_src.glob("*.png")))
    if not images:
        print(f"[WARN] No images found in {img_src}. "
              "Place annotated images there before running this script.")
        return

    random.seed(seed)
    random.shuffle(images)
    n_train = int(len(images) * split)
    train_imgs = images[:n_train]
    val_imgs = images[n_train:]

    for subset, imgs in [("train", train_imgs), ("val", val_imgs)]:
        (img_src / subset).mkdir(exist_ok=True)
        (lbl_src / subset).mkdir(exist_ok=True)
        for img in imgs:
            shutil.move(str(img), str(img_src / subset / img.name))
            lbl = lbl_src / img.with_suffix(".txt").name
            if lbl.exists():
                shutil.move(str(lbl), str(lbl_src / subset / lbl.name))

    print(f"[YOLO YAML] Split: {len(train_imgs)} train / {len(val_imgs)} val")


def create_yaml() -> Path:
    yaml_path = YOLO_DATASET_DIR / "data.yaml"
    yaml_content = f"""# DeepRetail YOLO detection dataset
# 2 classes: product (any product, for localization) and hand
# YOLO does NOT identify product types — classification is Stage 3's job

path: {YOLO_DATASET_DIR.resolve().as_posix()}
train: images/train
val:   images/val

nc: 2
names:
  0: product
  1: hand
"""
    yaml_path.write_text(yaml_content)
    print(f"[YOLO YAML] Written to {yaml_path}")
    return yaml_path


def main() -> None:
    args = parse_args()
    YOLO_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    (YOLO_DATASET_DIR / "images").mkdir(exist_ok=True)
    (YOLO_DATASET_DIR / "labels").mkdir(exist_ok=True)
    split_dataset(args.split, args.seed)
    create_yaml()
    print("[YOLO YAML] Done. Run: python src/detection/train_yolo.py")


if __name__ == "__main__":
    main()
