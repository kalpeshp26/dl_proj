"""
DeepRetail — FREE auto-labelling for the detector (no Roboflow credits, no manual boxes).

Runs the zero-shot YOLO-World model (text prompts in src/config.py → WORLD_PROMPTS)
over your demo video(s) or image folder(s), and writes a ready-to-train YOLO dataset:

    data/yolo_dataset/images/{train,val}/*.jpg
    data/yolo_dataset/labels/{train,val}/*.txt      (0 = product, 1 = hand)
    data/yolo_dataset/data.yaml
    data/yolo_dataset/previews/*.jpg                (contact sheets — LOOK AT THESE)

Then:  python src/detection/train_yolo.py   → models/yolo_best.pt (fast YOLOv8n)

Usage:
    python src/data_tools/auto_label.py --src demo_video.mp4
    python src/data_tools/auto_label.py --src video1.mp4 video2.mp4 --every 10
    python src/data_tools/auto_label.py --src path/to/frames_folder
    python src/data_tools/auto_label.py --src demo.mp4 --conf 0.15 --roboflow-zip
        (--roboflow-zip also writes data/yolo_dataset/autolabel_for_roboflow.zip that
         you can upload to Roboflow as an already-annotated dataset and just fix boxes)

Validation split: the LAST 20% of each video (not random frames), so val frames are
not near-duplicates of train frames.
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import YOLO_DATASET_DIR, WORLD_CONF_THRESH, WORLD_MAX_BOX_FRAC
from src.detection.yolo_infer import load_world_model, run_detector, draw_detections

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp", ".webm"}
CLS_ID = {"product": 0, "hand": 1}


def iter_frames(src: Path, every: int, max_frames: int):
    """Yield (name, frame_bgr, position_0_to_1) from a video file or an image folder."""
    if src.is_dir():
        files = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS)
        files = files[::max(1, every // 5)] if every > 5 else files
        files = files[:max_frames]
        for i, f in enumerate(files):
            img = cv2.imread(str(f))
            if img is not None:
                yield f"{src.name}_{f.stem}", img, i / max(1, len(files) - 1)
        return
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"[AutoLabel] Cannot open {src}")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = saved = 0
    stem = "".join(c if c.isalnum() else "_" for c in src.stem)[:40]
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            yield f"{stem}_f{idx:05d}", frame, idx / total
            saved += 1
        idx += 1
    cap.release()


def to_yolo_lines(dets, w: int, h: int) -> list[str]:
    lines = []
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        lines.append(f"{CLS_ID[d.class_name]} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} "
                     f"{(x2-x1)/w:.6f} {(y2-y1)/h:.6f}")
    return lines


def contact_sheet(items: list[np.ndarray], path: Path, cols: int = 4, tile: int = 320) -> None:
    tiles = []
    for im in items:
        h, w = im.shape[:2]
        s = tile / max(h, w)
        im = cv2.resize(im, (int(w * s), int(h * s)))
        canvas = np.full((tile, tile, 3), 30, np.uint8)
        canvas[:im.shape[0], :im.shape[1]] = im
        tiles.append(canvas)
    while len(tiles) % cols:
        tiles.append(np.full((tile, tile, 3), 30, np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.vstack(rows))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", nargs="+", required=True, help="Video file(s) and/or image folder(s)")
    p.add_argument("--every", type=int, default=10,
                   help="Keep every Nth video frame (default 10 ≈ 3 fps for 30 fps video)")
    p.add_argument("--max-per-source", type=int, default=300)
    p.add_argument("--conf", type=float, default=WORLD_CONF_THRESH,
                   help="Detection threshold (lower = more boxes, more false positives)")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--keep-empty", action="store_true",
                   help="Keep frames with no detections (as background examples)")
    p.add_argument("--no-clean", action="store_true", help="Add to the existing dataset")
    p.add_argument("--roboflow-zip", action="store_true",
                   help="Also write a zip you can upload to Roboflow for review")
    args = p.parse_args()

    if not args.no_clean:
        for sub in ("images", "labels", "previews"):
            shutil.rmtree(YOLO_DATASET_DIR / sub, ignore_errors=True)
    for split in ("train", "val"):
        (YOLO_DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    prev_dir = YOLO_DATASET_DIR / "previews"
    prev_dir.mkdir(parents=True, exist_ok=True)

    print("[AutoLabel] Loading YOLO-World (first run downloads the weights) ...")
    model, class_map = load_world_model()

    stats = {"train": 0, "val": 0, "product": 0, "hand": 0, "empty": 0}
    preview_buf: list[np.ndarray] = []
    sheet_no = 0

    for src_str in args.src:
        src = Path(src_str).expanduser()
        if not src.exists():
            print(f"[AutoLabel] Not found: {src}")
            continue
        if src.is_file() and src.suffix.lower() not in VID_EXTS:
            print(f"[AutoLabel] Skipping {src} (not a video or folder)")
            continue
        print(f"[AutoLabel] Processing {src} ...")
        for name, frame, pos in iter_frames(src, args.every, args.max_per_source):
            h, w = frame.shape[:2]
            dets = run_detector(model, frame, class_map, conf=args.conf,
                                max_box_frac=WORLD_MAX_BOX_FRAC)
            if not dets and not args.keep_empty:
                stats["empty"] += 1
                continue
            split = "val" if pos >= 1 - args.val_frac else "train"
            cv2.imwrite(str(YOLO_DATASET_DIR / "images" / split / f"{name}.jpg"), frame)
            (YOLO_DATASET_DIR / "labels" / split / f"{name}.txt").write_text(
                "\n".join(to_yolo_lines(dets, w, h)))
            stats[split] += 1
            for d in dets:
                stats[d.class_name] += 1

            vis = draw_detections(frame, dets)
            cv2.putText(vis, name[-18:], (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.5, w / 1200), (255, 255, 255), 2)
            preview_buf.append(vis)
            if len(preview_buf) == 16:
                sheet_no += 1
                contact_sheet(preview_buf, prev_dir / f"sheet_{sheet_no:03d}.jpg")
                preview_buf = []
    if preview_buf:
        sheet_no += 1
        contact_sheet(preview_buf, prev_dir / f"sheet_{sheet_no:03d}.jpg")

    from src.data_tools.create_yolo_yaml import create_yaml
    create_yaml()

    print(f"\n[AutoLabel] Done. images: train={stats['train']} val={stats['val']} | "
          f"boxes: product={stats['product']} hand={stats['hand']} | "
          f"frames with no detections skipped: {stats['empty']}")
    print(f"[AutoLabel] CHECK the preview sheets: {prev_dir}")
    print("  Green = product, blue = hand. If products are missed → lower --conf or edit\n"
          "  WORLD_PROMPTS in src/config.py. If the cloth/table gets boxed → raise --conf.")

    if args.roboflow_zip:
        zpath = YOLO_DATASET_DIR / "autolabel_for_roboflow.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(YOLO_DATASET_DIR / "data.yaml", "data.yaml")
            for split, rf_split in (("train", "train"), ("val", "valid")):
                for sub in ("images", "labels"):
                    for f in (YOLO_DATASET_DIR / sub / split).iterdir():
                        z.write(f, f"{rf_split}/{sub}/{f.name}")
        print(f"[AutoLabel] Roboflow upload zip → {zpath}")

    if stats["train"] == 0:
        print("[WARN] Nothing was labelled — try --conf 0.05 and check the prompts.")
    else:
        print("\nNext:  python src/detection/train_yolo.py")


if __name__ == "__main__":
    main()
