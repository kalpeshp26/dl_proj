"""
DeepRetail — Main Pipeline Runner (Stages 1–5)

Connects:
    Camera / video file
    → Stage 1: YOLODetector
    → Stage 2: Tracker
    → Stage 3: ProductClassifier (crops of product detections)
    → Stage 3b: Open-set check (embedded in ProductClassifier)
    → Stage 4: ZoneManager
    → Stage 5: StateMachineManager

Events (pick/return/anomaly) are:
    - Printed to console
    - Logged to logs/events.jsonl
    - Optionally pushed to a callback (used by the backend in Phase 2)

Usage:
    python src/pipeline/run_pipeline.py
    python src/pipeline/run_pipeline.py --source path/to/video.mp4
    python src/pipeline/run_pipeline.py --no-display      (headless)
    python src/pipeline/run_pipeline.py --mock            (no models needed, for testing)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS,
    YOLO_MODEL_PATH, CLASSIFIER_MODEL_PATH, ensure_dirs,
)
from src.detection.yolo_infer import DetectionResult, YOLODetector, draw_detections
from src.tracking.tracker import Tracker, Track
from src.classification.infer_classifier import ProductClassifier, ClassificationResult
from src.logic.zones import ZoneManager
from src.logic.state_machine import StateMachineManager, RetailEvent


# ─── Shared pipeline state (readable by backend/dashboard) ───────────────────

class PipelineState:
    """Thread-safe-ish shared state snapshot. Backend reads this."""
    def __init__(self):
        self.frame: Optional[np.ndarray] = None
        self.annotated_frame: Optional[np.ndarray] = None
        self.detections: list[dict] = []
        self.tracks: list[dict] = []
        self.classifications: dict[int, dict] = {}  # track_id → ClassificationResult.to_dict()
        self.events: list[dict] = []                 # last N events
        self.track_states: dict[int, str] = {}
        self.fps: float = 0.0
        self.frame_id: int = 0
        self.running: bool = False
        self._max_events: int = 100

    def push_event(self, event: RetailEvent):
        self.events.append(event.to_dict())
        if len(self.events) > self._max_events:
            self.events = self.events[-self._max_events:]


# Global singleton — imported by backend
pipeline_state = PipelineState()


# ─── Video source abstraction (supports webcam + video file) ─────────────────

class VideoSource:
    """
    Unified camera abstraction.
    source=int → webcam | source=str → video file (replay fallback)
    """

    def __init__(self, source, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        self._cap = cv2.VideoCapture(source)
        if isinstance(source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._source = source
        self._is_file = isinstance(source, str)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        ret, frame = self._cap.read()
        if not ret and self._is_file:
            # Loop video file for replay mode
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        return ret, frame

    def release(self):
        self._cap.release()

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def is_file(self) -> bool:
        return self._is_file


# ─── Annotation helpers ───────────────────────────────────────────────────────

def annotate_frame(
    frame: np.ndarray,
    detections: list[DetectionResult],
    tracks: list[Track],
    classifications: dict[int, ClassificationResult],
    zone_manager: ZoneManager,
    sm_manager: StateMachineManager,
    recent_events: list[dict],
) -> np.ndarray:
    vis = frame.copy()

    # Draw zone ROIs
    sr = zone_manager.shelf_roi
    br = zone_manager.basket_roi
    cv2.rectangle(vis, (sr[0], sr[1]), (sr[2], sr[3]), (0, 200, 0), 1)
    cv2.putText(vis, "SHELF", (sr[0] + 4, sr[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    cv2.rectangle(vis, (br[0], br[1]), (br[2], br[3]), (0, 0, 200), 1)
    cv2.putText(vis, "BASKET", (br[0] + 4, br[1] + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)

    # Draw confirmed tracks with classification
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.box]
        clf = classifications.get(t.track_id)
        state = sm_manager.get_state(t.track_id)
        state_str = state.value if state else "?"

        # Color by state
        state_colors = {
            "SHELF": (0, 200, 0),
            "HELD": (0, 165, 255),
            "BASKET": (0, 0, 200),
            "ANOMALY": (0, 0, 255),
        }
        color = state_colors.get(state_str, (200, 200, 200))

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        if clf:
            product_label = clf.class_name
            if clf.is_unknown:
                product_label = "UNKNOWN"
                color = (0, 0, 200)
            label = f"#{t.track_id} {product_label} ({clf.conf:.0%}) [{state_str}]"
        else:
            label = f"#{t.track_id} [detecting...] [{state_str}]"

        cv2.putText(vis, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw recent events
    for i, ev in enumerate(recent_events[-5:]):
        txt = f"[{ev['event'].upper()}] {ev.get('product', '?')} (id={ev['track_id']})"
        cv2.putText(vis, txt, (10, 30 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    return vis


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class Pipeline:
    def __init__(
        self,
        source=CAMERA_INDEX,
        display: bool = True,
        anomaly_enabled: bool = False,
        mock_mode: bool = False,
        event_callbacks: Optional[list[Callable]] = None,
    ) -> None:
        self.display = display
        self.mock_mode = mock_mode

        ensure_dirs()

        if not mock_mode:
            self._detector = YOLODetector()
            self._classifier = ProductClassifier()
        else:
            self._detector = None
            self._classifier = None
            print("[Pipeline] MOCK MODE — no models loaded, deterministic outputs")

        self._tracker = Tracker()
        self._zones = ZoneManager()
        self._sm = StateMachineManager(anomaly_enabled=anomaly_enabled)
        self._sm.register_callback(pipeline_state.push_event)

        if event_callbacks:
            for cb in event_callbacks:
                self._sm.register_callback(cb)

        self._source = VideoSource(source)
        self._clf_cache: dict[int, ClassificationResult] = {}   # track_id → last result

        pipeline_state.running = True

    def run(self) -> None:
        """Main loop — blocks until 'q' pressed or stream ends."""
        frame_id = 0
        t_prev = time.time()

        try:
            while True:
                ret, frame = self._source.read()
                if not ret:
                    print("[Pipeline] Stream ended.")
                    break

                frame_id += 1
                pipeline_state.frame_id = frame_id
                pipeline_state.frame = frame.copy()

                # ── Stage 1: Detection ────────────────────────────
                if not self.mock_mode:
                    detections = self._detector.detect(frame, frame_id)
                else:
                    detections = self._mock_detections(frame_id)

                pipeline_state.detections = [d.to_dict() for d in detections]

                # ── Stage 2: Tracking ─────────────────────────────
                tracks = self._tracker.update(detections)
                pipeline_state.tracks = [t.to_dict() for t in tracks]

                # ── Stage 3: Classify each confirmed track ────────
                for t in tracks:
                    if not self.mock_mode:
                        crop = DetectionResult(
                            class_name="product", conf=t.conf, xyxy=t.box
                        ).crop(frame)
                        if crop.size > 0:
                            clf_result = self._classifier.classify(
                                crop, frame_id=frame_id, track_id=t.track_id
                            )
                            self._clf_cache[t.track_id] = clf_result
                    else:
                        from src.classification.infer_classifier import ClassificationResult
                        self._clf_cache[t.track_id] = ClassificationResult(
                            class_name="mock_product",
                            class_idx=0, conf=0.95,
                            embedding=np.zeros(576),
                            is_unknown=False,
                        )

                pipeline_state.classifications = {
                    tid: r.to_dict()
                    for tid, r in self._clf_cache.items()
                }

                # ── Stage 4: Zone membership ──────────────────────
                tracks_with_zones = []
                for t in tracks:
                    zone = self._zones.classify_location(t.cx, t.cy)
                    clf = self._clf_cache.get(t.track_id)
                    tracks_with_zones.append({
                        "track_id": t.track_id,
                        "zone": zone,
                        "product": clf.class_name if clf else "UNKNOWN",
                        "conf": clf.conf if clf else 0.0,
                    })

                # ── Stage 5: State machine ────────────────────────
                events = self._sm.process_frame(tracks_with_zones)
                pipeline_state.track_states = self._sm.get_all_states()

                # ── Annotate + display ────────────────────────────
                if self.display:
                    vis = annotate_frame(
                        frame, detections, tracks,
                        self._clf_cache, self._zones, self._sm,
                        pipeline_state.events,
                    )
                    # FPS
                    now = time.time()
                    fps = 1.0 / max(now - t_prev, 0.001)
                    t_prev = now
                    pipeline_state.fps = round(fps, 1)
                    cv2.putText(vis, f"FPS: {fps:.1f}", (vis.shape[1] - 120, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                    pipeline_state.annotated_frame = vis
                    cv2.imshow("DeepRetail Pipeline", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Throttle to TARGET_FPS if running faster
                elapsed = time.time() - t_prev
                sleep_time = (1.0 / TARGET_FPS) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            self._source.release()
            cv2.destroyAllWindows()
            pipeline_state.running = False
            print("[Pipeline] Stopped.")

    def _mock_detections(self, frame_id: int) -> list[DetectionResult]:
        """Deterministic mock detections for unit testing without hardware."""
        # Simulate a product moving across the frame
        x = 100 + (frame_id * 3) % 900
        y = 200
        return [DetectionResult(class_name="product", conf=0.92,
                                xyxy=[float(x), float(y), float(x + 100), float(y + 100)])]


# ─── CLI entry point ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run DeepRetail pipeline")
    p.add_argument("--source", type=str, default=str(CAMERA_INDEX),
                   help="Camera index or video file path")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--anomaly", action="store_true",
                   help="Enable Phase 2 anomaly detection")
    p.add_argument("--mock", action="store_true",
                   help="Mock mode (no models needed)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    pipeline = Pipeline(
        source=source,
        display=not args.no_display,
        anomaly_enabled=args.anomaly,
        mock_mode=args.mock,
    )
    pipeline.run()
