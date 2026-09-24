"""
DeepRetail — Image Capture Tool
Usage:
    python src/data_tools/capture_images.py --class coke_500ml
    python src/data_tools/capture_images.py --class background --limit 50

Controls:
    SPACE  — save current frame
    n      — cycle to next class (if --class not provided)
    q      — quit

Images are saved to: data/raw_images/<class_name>/
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import PRODUCT_CLASSES, CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, RAW_IMAGES_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Capture training images for DeepRetail")
    p.add_argument("--class", dest="product_class", type=str, default=None,
                   help="Product class to capture (must match PRODUCT_CLASSES in config.py)")
    p.add_argument("--camera", type=int, default=CAMERA_INDEX,
                   help="OpenCV camera index (default: 0)")
    p.add_argument("--limit", type=int, default=300,
                   help="Stop after this many captures for this class")
    p.add_argument("--all", action="store_true",
                   help="Cycle through ALL product classes + background interactively")
    return p.parse_args()


def capture_for_class(
    cap: cv2.VideoCapture,
    cls_name: str,
    limit: int,
) -> int:
    """Capture images for one class. Returns number of images saved."""
    save_dir = RAW_IMAGES_DIR / cls_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Count already-saved images to continue numbering
    existing = len(list(save_dir.glob("*.jpg")))
    saved = 0

    print(f"\n[Capture] Class: '{cls_name}' | Target: {limit} | "
          f"Already exists: {existing}")
    print("  SPACE = save frame | q = quit/next class")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read from camera.")
            break

        # Overlay HUD
        display = frame.copy()
        total = existing + saved
        cv2.putText(display, f"Class: {cls_name}  Saved: {saved}/{limit}  Total: {total}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=save | q=done",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.imshow("DeepRetail — Image Capture", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            fname = save_dir / f"{cls_name}_{existing + saved:04d}.jpg"
            cv2.imwrite(str(fname), frame)
            saved += 1
            print(f"  Saved: {fname.name} ({saved}/{limit})")
            if saved >= limit:
                print(f"[Done] Reached limit of {limit} for '{cls_name}'")
                break
        elif key == ord("q"):
            print(f"[Done] Quit after {saved} captures for '{cls_name}'")
            break

    return saved


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.camera}")
        sys.exit(1)

    classes_to_capture: list[str]
    if args.all:
        classes_to_capture = PRODUCT_CLASSES + ["background"]
    elif args.product_class:
        classes_to_capture = [args.product_class]
    else:
        # Interactive class selection
        print("Available classes:")
        for i, c in enumerate(PRODUCT_CLASSES + ["background"]):
            print(f"  {i}: {c}")
        idx = int(input("Enter class index: "))
        classes_to_capture = [(PRODUCT_CLASSES + ["background"])[idx]]

    total_saved = 0
    for cls in classes_to_capture:
        n = capture_for_class(cap, cls, args.limit)
        total_saved += n

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[Summary] Total images saved this session: {total_saved}")
    print(f"[Summary] Images stored in: {RAW_IMAGES_DIR}")


if __name__ == "__main__":
    main()
