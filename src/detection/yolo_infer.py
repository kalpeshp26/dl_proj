"""
DeepRetail — YOLO Inference Wrapper (Stage 1)

Fixed output contract — every consumer in the pipeline depends on this:
    List[DetectionResult]

Where DetectionResult is a dataclass with:
    class_name : str   — "product" or "hand" ONLY
    conf       : float — confidence [0, 1]
    xyxy       : list[float] — [x1, y1, x2, y2] in pixels

YOLO NEVER outputs product identity. That is Stage 3's job.
"""

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    YOLO_MODEL_PATH, YOLO_CONF_THRESH, YOLO_IOU_THRESH,
    YOLO_CLASSES, LOG_DETECTION, DETECTOR_BACKEND,
    WORLD_MODEL, WORLD_PROMPTS, WORLD_CONF_THRESH, WORLD_MAX_BOX_FRAC,
)


# ─── Output contract ──────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """One detected object from Stage 1."""
    class_name: str          # "product" or "hand"
    conf: float              # detection confidence
    xyxy: list[float]        # [x1, y1, x2, y2]

    @property
    def cx(self) -> float:
        return (self.xyxy[0] + self.xyxy[2]) / 2

    @property
    def cy(self) -> float:
        return (self.xyxy[1] + self.xyxy[3]) / 2

    @property
    def area(self) -> float:
        return (self.xyxy[2] - self.xyxy[0]) * (self.xyxy[3] - self.xyxy[1])

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Return the cropped BGR patch for this detection."""
        x1, y1, x2, y2 = [int(v) for v in self.xyxy]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2]

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Shared helpers (also used by src/data_tools/auto_label.py) ──────────────

def load_world_model():
    """
    Load YOLO-World and set the text prompts from config.
    Returns (model, class_map) where class_map[i] is "product" or "hand".
    """
    try:
        from ultralytics import YOLOWorld
    except ImportError:
        raise ImportError("Your ultralytics version has no YOLOWorld. "
                          "Run: pip install -U ultralytics")
    model = YOLOWorld(WORLD_MODEL)       # downloads weights on first use
    prompts = list(WORLD_PROMPTS)
    try:
        model.set_classes(prompts)       # needs the CLIP text encoder (auto-installed)
    except Exception as e:
        raise RuntimeError(
            f"YOLO-World could not load its text encoder: {e}\n"
            "Fix: pip install git+https://github.com/ultralytics/CLIP.git "
            "(needs git installed), then run again."
        ) from e
    return model, [WORLD_PROMPTS[p] for p in prompts]


def run_detector(model, frame: np.ndarray, class_map: list[str], conf: float,
                 iou: float = YOLO_IOU_THRESH, max_box_frac: float = 1.0,
                 imgsz: int = 384
                 ) -> list["DetectionResult"]:
    """Run an ultralytics model and map its classes to product/hand."""
    h, w = frame.shape[:2]
    results = model.predict(source=frame, conf=conf, iou=iou,
                            imgsz=imgsz,
                            agnostic_nms=False,
                            verbose=False, stream=False)
    out: list[DetectionResult] = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls_idx = int(box.cls[0].item())
            if cls_idx >= len(class_map):
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            if (x2 - x1) * (y2 - y1) > max_box_frac * w * h:
                continue                 # background / table / big bag
            out.append(DetectionResult(
                class_name=class_map[cls_idx],
                conf=round(float(box.conf[0].item()), 4),
                xyxy=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            ))
    return out


# ─── Inference class ─────────────────────────────────────────────────────────

class YOLODetector:
    """
    Thin wrapper around YOLOv8n for DeepRetail.
    Loads model once; call .detect(frame) per frame.
    """

    def __init__(
        self,
        model_path: Path = YOLO_MODEL_PATH,
        conf_thresh: float = YOLO_CONF_THRESH,
        iou_thresh: float = YOLO_IOU_THRESH,
        device: str = "",
        log: bool = True,
        backend: str = DETECTOR_BACKEND,
    ) -> None:
        """
        backend:
            "trained" — your fine-tuned models/yolo_best.pt (classes: product, hand)
            "world"   — zero-shot YOLO-World with text prompts (no training needed)
            "auto"    — "trained" if the weights exist, otherwise "world"
        """
        self._log = log
        self._iou = iou_thresh
        self._frame_count = 0

        try:
            from ultralytics import YOLO as _YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        if backend == "auto":
            backend = "trained" if model_path.exists() else "world"
        self.backend = backend

        if backend == "trained":
            if not model_path.exists():
                raise FileNotFoundError(
                    f"YOLO weights not found: {model_path}\n"
                    "Train first:  python src/detection/train_yolo.py\n"
                    "or use the zero-shot detector:  --detector world"
                )
            print(f"[YOLODetector] Loading trained weights from {model_path}")
            self._model = _YOLO(str(model_path))
            self._conf = conf_thresh
            self._class_map = list(YOLO_CLASSES)
            self._max_box_frac = 1.0
        elif backend == "world":
            self._model, self._class_map = load_world_model()
            self._conf = WORLD_CONF_THRESH
            self._max_box_frac = WORLD_MAX_BOX_FRAC
            print(f"[YOLODetector] Zero-shot YOLO-World ({WORLD_MODEL}) — "
                  f"prompts: {list(WORLD_PROMPTS)}")
        else:
            raise ValueError(f"Unknown detector backend: {backend!r}")

        if device:
            self._model.to(device)

        # Ensure log directory exists
        LOG_DETECTION.parent.mkdir(parents=True, exist_ok=True)

    def detect(
        self, frame: np.ndarray, frame_id: Optional[int] = None
    ) -> list[DetectionResult]:
        """
        Run inference on a BGR frame.
        Returns a list of DetectionResult (product and/or hand detections).
        """
        if frame_id is None:
            frame_id = self._frame_count
        self._frame_count += 1

        detections = run_detector(self._model, frame, self._class_map,
                                  conf=self._conf, iou=self._iou,
                                  max_box_frac=self._max_box_frac)

        if self._log:
            self._write_log(frame_id, detections)

        return detections

    def _write_log(self, frame_id: int, detections: list[DetectionResult]) -> None:
        entry = {
            "ts": time.time(),
            "frame_id": frame_id,
            "detections": [d.to_dict() for d in detections],
        }
        with open(LOG_DETECTION, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ─── Annotation helper ───────────────────────────────────────────────────────

def draw_detections(
    frame: np.ndarray,
    detections: list[DetectionResult],
    show_conf: bool = True,
) -> np.ndarray:
    """Draw bounding boxes on frame. Returns annotated copy."""
    vis = frame.copy()
    colors = {"product": (0, 255, 0), "hand": (255, 100, 0)}
    for d in detections:
        color = colors.get(d.class_name, (200, 200, 200))
        x1, y1, x2, y2 = [int(v) for v in d.xyxy]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{d.class_name} {d.conf:.2f}" if show_conf else d.class_name
        cv2.putText(vis, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return vis


# ─── Quick smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run YOLO inference on webcam or video")
    parser.add_argument("--source", type=str, default="0",
                        help="Camera index (int) or video file path")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--detector", choices=["auto", "trained", "world"], default=DETECTOR_BACKEND)
    args = parser.parse_args()

    detector = YOLODetector(log=not args.no_log, backend=args.detector)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    print("[YOLOInfer] Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector.detect(frame)
        vis = draw_detections(frame, dets)
        fps_text = f"Detections: {len(dets)}"
        cv2.putText(vis, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("YOLO Stage 1", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
