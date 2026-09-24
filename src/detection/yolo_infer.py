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
    YOLO_CLASSES, LOG_DETECTION,
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
    ) -> None:
        self._log = log
        self._conf = conf_thresh
        self._iou = iou_thresh
        self._frame_count = 0

        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {model_path}\n"
                "Train first:  python src/detection/train_yolo.py"
            )

        try:
            from ultralytics import YOLO as _YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        print(f"[YOLODetector] Loading weights from {model_path}")
        self._model = _YOLO(str(model_path))
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

        results = self._model.predict(
            source=frame,
            conf=self._conf,
            iou=self._iou,
            verbose=False,
            stream=False,
        )

        detections: list[DetectionResult] = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_idx = int(box.cls[0].item())
                if cls_idx >= len(YOLO_CLASSES):
                    continue
                cls_name = YOLO_CLASSES[cls_idx]
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(DetectionResult(
                    class_name=cls_name,
                    conf=round(conf, 4),
                    xyxy=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                ))

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
    args = parser.parse_args()

    detector = YOLODetector(log=not args.no_log)

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
