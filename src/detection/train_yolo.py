"""
DeepRetail — YOLOv8n Training Script
YOLO detects only 2 classes: 0=product, 1=hand
It does NOT identify product types — that is the classifier's job.

Usage:
    python src/detection/train_yolo.py
    python src/detection/train_yolo.py --epochs 100 --batch 16

Prerequisites:
    - Annotated dataset in YOLO format at data/yolo_dataset/
    - data/yolo_dataset/data.yaml describing classes and splits
    - Run: python src/data_tools/create_yolo_yaml.py  to generate data.yaml

Output:
    models/yolo_best.pt
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    YOLO_DATASET_DIR, MODELS_DIR, YOLO_MODEL_PATH,
    YOLO_EPOCHS, YOLO_BATCH, YOLO_IMGSZ,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8n for DeepRetail detection")
    p.add_argument("--epochs", type=int, default=YOLO_EPOCHS)
    p.add_argument("--batch", type=int, default=YOLO_BATCH)
    p.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    p.add_argument("--device", type=str, default="",
                   help="Training device: '' = auto, '0' = GPU 0, 'cpu' = CPU")
    p.add_argument("--resume", action="store_true",
                   help="Resume from last checkpoint")
    return p.parse_args()


def build_data_yaml() -> Path:
    """Auto-generate data.yaml if not present."""
    yaml_path = YOLO_DATASET_DIR / "data.yaml"
    if yaml_path.exists():
        print(f"[YOLO Train] Using existing {yaml_path}")
        return yaml_path

    # Check dataset structure
    train_img = YOLO_DATASET_DIR / "images" / "train"
    val_img = YOLO_DATASET_DIR / "images" / "val"
    if not train_img.exists():
        raise FileNotFoundError(
            f"YOLO dataset not found at {YOLO_DATASET_DIR}.\n"
            "Expected structure:\n"
            "  data/yolo_dataset/images/train/*.jpg\n"
            "  data/yolo_dataset/images/val/*.jpg\n"
            "  data/yolo_dataset/labels/train/*.txt\n"
            "  data/yolo_dataset/labels/val/*.txt\n"
            "Annotate images using LabelImg or Roboflow (YOLO format), "
            "with classes: 0=product, 1=hand"
        )

    yaml_content = f"""# DeepRetail YOLO dataset config
# Classes: 0=product, 1=hand (NEVER product identity)
path: {YOLO_DATASET_DIR.resolve()}
train: images/train
val: images/val

nc: 2
names:
  0: product
  1: hand
"""
    yaml_path.write_text(yaml_content)
    print(f"[YOLO Train] Generated {yaml_path}")
    return yaml_path


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    data_yaml = build_data_yaml()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[YOLO Train] Starting fine-tune of yolov8n.pt")
    print(f"  Data:   {data_yaml}")
    print(f"  Epochs: {args.epochs}  Batch: {args.batch}  imgsz: {args.imgsz}")

    model = YOLO("yolov8n.pt")  # downloads pretrained if not cached

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if args.device else None,
        project=str(MODELS_DIR / "yolo_runs"),
        name="train",
        exist_ok=True,
        resume=args.resume,
        patience=20,
        save=True,
        plots=True,
        verbose=True,
    )

    # Copy best weights to canonical location
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    if best_src.exists():
        shutil.copy(best_src, YOLO_MODEL_PATH)
        print(f"\n[YOLO Train] Best weights saved → {YOLO_MODEL_PATH}")
    else:
        print(f"[WARNING] best.pt not found at {best_src}. Check training run.")

    # Quick validation
    print("\n[YOLO Train] Running validation on val split...")
    metrics = model.val(data=str(data_yaml))
    print(f"  mAP50 (product): check results above")
    print(f"  Full metrics:    {results.save_dir}")


if __name__ == "__main__":
    main()
