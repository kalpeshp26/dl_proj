"""
DeepRetail — Zone Calibration Utility
Usage:
    python calibrate.py

Steps:
    1. Captures a single frame from the webcam (or loads an image).
    2. User clicks/drags rectangles to define:
       - shelf_roi     (overall shelf region)
       - basket_roi    (basket / cart region)
       - shelf_slots   (individual slot ROIs, one per product)
    3. Saves results to config.json in the project root.

Controls (during ROI selection):
    Each zone uses cv2.selectROI.
    Press ENTER or SPACE to confirm an ROI.
    Press ESC to skip/cancel a zone.
"""

import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
    PRODUCT_CLASSES, CONFIG_JSON_PATH, save_zones, load_zones,
)


def capture_frame(camera_index: int) -> "cv2.Mat":
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    for _ in range(10):          # drain buffer
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot capture from camera {camera_index}")
    return frame


def select_roi_labeled(frame: "cv2.Mat", label: str) -> list[int] | None:
    """Show frame, let user drag an ROI, return [x1,y1,x2,y2] or None if skipped."""
    display = frame.copy()
    cv2.putText(display, f"Select: {label}  (ENTER=confirm, ESC=skip)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("DeepRetail — Calibration", display)
    cv2.waitKey(200)

    r = cv2.selectROI("DeepRetail — Calibration", display,
                      fromCenter=False, showCrosshair=True)
    x, y, w, h = r
    if w == 0 or h == 0:
        print(f"  [skip] {label}")
        return None
    roi = [int(x), int(y), int(x + w), int(y + h)]
    print(f"  [set]  {label} = {roi}")
    return roi


def draw_zones(frame: "cv2.Mat", zones: dict) -> "cv2.Mat":
    vis = frame.copy()
    colors = {"shelf_roi": (0, 255, 0), "basket_roi": (0, 0, 255)}
    slot_color = (255, 165, 0)

    for key, color in colors.items():
        roi = zones.get(key)
        if roi:
            cv2.rectangle(vis, (roi[0], roi[1]), (roi[2], roi[3]), color, 2)
            cv2.putText(vis, key, (roi[0] + 4, roi[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for slot_id, slot in zones.get("shelf_slots", {}).items():
        roi = slot.get("roi")
        if roi:
            cv2.rectangle(vis, (roi[0], roi[1]), (roi[2], roi[3]), slot_color, 2)
            label = f"{slot_id}: {slot.get('expected_product', '?')}"
            cv2.putText(vis, label, (roi[0] + 4, roi[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, slot_color, 1)
    return vis


def main() -> None:
    print("=" * 60)
    print("DeepRetail — Zone Calibration")
    print("=" * 60)

    # ── Load frame ───────────────────────────────────────────────
    use_image = input("Use existing image file? (y/N): ").strip().lower() == "y"
    if use_image:
        img_path = input("Image path: ").strip()
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[ERROR] Cannot load {img_path}")
            sys.exit(1)
    else:
        print(f"Capturing from camera {CAMERA_INDEX}...")
        frame = capture_frame(CAMERA_INDEX)
        save_snap = str(Path("calibration_snapshot.jpg"))
        cv2.imwrite(save_snap, frame)
        print(f"Snapshot saved → {save_snap}")

    zones: dict = load_zones()

    # ── Shelf ROI ────────────────────────────────────────────────
    print("\n[1/3] Define the SHELF region (where products rest)")
    shelf = select_roi_labeled(frame, "shelf_roi")
    if shelf:
        zones["shelf_roi"] = shelf

    # ── Basket ROI ───────────────────────────────────────────────
    print("\n[2/3] Define the BASKET region (cart/tray)")
    basket = select_roi_labeled(frame, "basket_roi")
    if basket:
        zones["basket_roi"] = basket

    # ── Shelf slots ──────────────────────────────────────────────
    print(f"\n[3/3] Define {len(PRODUCT_CLASSES)} shelf slot(s) (one per product class)")
    print("      Press ESC to skip remaining slots.")
    if "shelf_slots" not in zones:
        zones["shelf_slots"] = {}

    for i, cls_name in enumerate(PRODUCT_CLASSES):
        slot_id = f"slot_{i}"
        print(f"  Slot {i}: expected product = '{cls_name}'")
        roi = select_roi_labeled(frame, f"{slot_id} ({cls_name})")
        if roi is None:
            print("  Stopping slot definition.")
            break
        threshold = int(input(f"  Low-stock threshold for {cls_name} [default 3]: ").strip() or "3")
        zones["shelf_slots"][slot_id] = {
            "roi": roi,
            "expected_product": cls_name,
            "low_stock_threshold": threshold,
        }

    cv2.destroyAllWindows()

    # ── Save ─────────────────────────────────────────────────────
    save_zones(zones)
    print(f"\n[Saved] Zones written to {CONFIG_JSON_PATH}")

    # ── Preview ──────────────────────────────────────────────────
    preview = draw_zones(frame, zones)
    cv2.imwrite("calibration_preview.jpg", preview)
    cv2.imshow("DeepRetail — Zone Preview (press any key to close)", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("[Done] Calibration complete. Run 'python src/pipeline/run_pipeline.py' next.")


if __name__ == "__main__":
    main()
