"""
DeepRetail — Import a Roboflow "YOLOv8" export into data/yolo_dataset/

The detector has exactly two classes: 0 = product, 1 = hand.
You can annotate in Roboflow with any class names; this script remaps them:
    any class whose name contains "hand"  → 1 (hand)
    every other class                     → 0 (product)
So it is fine to label boxes as "vaseline", "matchbox", ... or just "product".

Usage:
    python src/data_tools/import_yolo_dataset.py --src path/to/roboflow_yolov8.zip
    python src/data_tools/import_yolo_dataset.py --src path/to/export_folder --clean
Then:
    python src/detection/train_yolo.py
"""

import argparse
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import YOLO_DATASET_DIR

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "val"}


def read_names(data_yaml: Path) -> list[str]:
    text = data_yaml.read_text(encoding="utf-8")
    try:
        import yaml  # installed with ultralytics
        names = yaml.safe_load(text).get("names", [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names)]
        return [str(n) for n in names]
    except ImportError:
        m = re.search(r"names:\s*\[(.*?)\]", text, re.S)
        if m:
            return [n.strip().strip("'\"") for n in m.group(1).split(",") if n.strip()]
        return re.findall(r"^\s*(?:\d+:|-)\s*['\"]?([^'\"\n]+)['\"]?\s*$",
                          text.split("names:", 1)[-1], re.M)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Roboflow YOLOv8 export (.zip or folder)")
    p.add_argument("--clean", action="store_true", help="Delete existing images/labels first")
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

    yamls = list(root.rglob("data.yaml"))
    if not yamls:
        sys.exit("[YOLO import] data.yaml not found — export from Roboflow in 'YOLOv8' format.")
    root = yamls[0].parent
    names = read_names(yamls[0])
    remap = {i: (1 if "hand" in n.lower() else 0) for i, n in enumerate(names)}
    print(f"[YOLO import] Roboflow classes: {names}")
    print(f"[YOLO import] Remap → " + ", ".join(
        f"{n}→{'hand' if remap[i] else 'product'}" for i, n in enumerate(names)))

    if args.clean:
        shutil.rmtree(YOLO_DATASET_DIR / "images", ignore_errors=True)
        shutil.rmtree(YOLO_DATASET_DIR / "labels", ignore_errors=True)

    counts = {"train": 0, "val": 0}
    boxes = {0: 0, 1: 0}
    for split_name, out_split in SPLITS.items():
        img_dir = root / split_name / "images"
        lbl_dir = root / split_name / "labels"
        if not img_dir.is_dir():
            continue
        out_img = YOLO_DATASET_DIR / "images" / out_split
        out_lbl = YOLO_DATASET_DIR / "labels" / out_split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        for img in img_dir.iterdir():
            if img.suffix.lower() not in IMG_EXTS:
                continue
            shutil.copy2(img, out_img / img.name)
            lines_out = []
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                for line in lbl.read_text().splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls = remap.get(int(parts[0]), 0)
                    if len(parts) > 5:  # polygon export → convert to bbox
                        xs = [float(v) for v in parts[1::2]]
                        ys = [float(v) for v in parts[2::2]]
                        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                        parts = [str(cls), f"{(x1+x2)/2:.6f}", f"{(y1+y2)/2:.6f}",
                                 f"{x2-x1:.6f}", f"{y2-y1:.6f}"]
                    else:
                        parts[0] = str(cls)
                    lines_out.append(" ".join(parts))
                    boxes[cls] += 1
            (out_lbl / (img.stem + ".txt")).write_text("\n".join(lines_out))
            counts[out_split] += 1

    if tmp:
        tmp.cleanup()

    from src.data_tools.create_yolo_yaml import create_yaml
    create_yaml()
    print(f"[YOLO import] images: train={counts['train']} val={counts['val']} | "
          f"boxes: product={boxes[0]} hand={boxes[1]}")
    if counts["val"] == 0:
        print("[WARN] No valid/test split in the export — generate one in Roboflow "
              "(e.g. 80/20) or training will have nothing to validate on.")
    if boxes[1] == 0:
        print("[WARN] No 'hand' boxes found. Pipeline still works (hands are not "
              "tracked), but labelling hands reduces hand-as-product false positives.")


if __name__ == "__main__":
    main()
