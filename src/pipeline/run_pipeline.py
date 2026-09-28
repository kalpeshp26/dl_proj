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

    → Phase 2: ShelfMonitor (misplaced products) + UNKNOWN_PRODUCT alerts

Events (pick/return/anomaly) and alerts are:
    - Printed to console and logged to logs/*.jsonl
    - Pushed to the FastAPI backend over HTTP (src/pipeline/api_bridge.py),
      together with the annotated frame and FPS/status for the dashboard

Usage (start the backend first: uvicorn src.backend.main:app --port 8000):
    python -m src.pipeline.run_pipeline
    python -m src.pipeline.run_pipeline --source path/to/video.mp4
    python -m src.pipeline.run_pipeline --no-display      (no OpenCV window)
    python -m src.pipeline.run_pipeline --anomaly         (enable concealment alerts)
    python -m src.pipeline.run_pipeline --mock            (no models/camera: scripted demo)
    python -m src.pipeline.run_pipeline --no-backend      (standalone, window only)
    python -m src.pipeline.run_pipeline --detector world  (zero-shot detector, no YOLO training)
"""

import argparse
import json
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS, PRODUCT_CLASSES,
    CLASSIFY_EVERY_N_FRAMES, CLASS_VOTE_WINDOW, UNKNOWN_ALERT_FRAMES,
    DEFAULT_SESSION_ID, FRAME_PUSH_INTERVAL_SEC, STATUS_PUSH_INTERVAL_SEC,
    API_BASE_URL, LOG_ALERTS,
    YOLO_MODEL_PATH, CLASSIFIER_MODEL_PATH, ensure_dirs,
)
from src.detection.yolo_infer import DetectionResult, YOLODetector, draw_detections
from src.tracking.tracker import Tracker, Track
# The classifier (torch) is imported lazily in Pipeline.__init__ so that
# --mock mode works on machines without torch installed.
if TYPE_CHECKING:
    from src.classification.infer_classifier import ClassificationResult
from src.logic.zones import ZoneManager
from src.logic.state_machine import StateMachineManager, RetailEvent
from src.logic.shelf_monitor import ShelfMonitor
from src.pipeline.api_bridge import ApiBridge


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
        self.session_id: str = ""
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
        if isinstance(source, int) and sys.platform.startswith("win"):
            self._cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(source)
        self._source = source
        self._is_file = isinstance(source, str)

    @property
    def is_opened(self) -> bool:
        return self._cap.isOpened()

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
    classifications: "dict[int, TrackLabel]",
    zone_manager: ZoneManager,
    sm_manager: StateMachineManager,
    recent_events: list[dict],
) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Draw zone ROIs
    sr = zone_manager.shelf_roi
    br = zone_manager.basket_roi

    # 1. Desk / Table Staging Zone (outside bag)
    cv2.rectangle(vis, (sr[0], sr[1]), (sr[2], sr[3]), (160, 160, 160), 1)
    banner_h = min(24, max(14, (sr[3] - sr[1]) // 10))
    cv2.rectangle(vis, (sr[0], sr[1]), (sr[2], sr[1] + banner_h), (50, 50, 50), -1)
    cv2.putText(vis, "TABLE / DESK (OUTSIDE BAG)", (sr[0] + 6, sr[1] + banner_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    # 2. Shopping Bag / Cart Zone (items placed here are added to bill)
    cv2.rectangle(vis, (br[0], br[1]), (br[2], br[3]), (255, 180, 0), 2)
    b_banner_h = min(24, max(14, (br[3] - br[1]) // 10))
    cv2.rectangle(vis, (br[0], br[1]), (br[2], br[1] + b_banner_h), (200, 120, 0), -1)
    cv2.putText(vis, "SHOPPING BAG (PUT ITEMS HERE TO BUY)", (br[0] + 6, br[1] + b_banner_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Draw shelf slots with their expected product ONLY if configured
    slots = zone_manager.all_slots()
    if slots:
        for slot_id, slot in slots.items():
            x1, y1, x2, y2 = [int(v) for v in slot["roi"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 120, 0), 1)
            cv2.putText(vis, str(slot.get("expected_product", slot_id)), (x1 + 4, y2 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 160, 0), 1)

    # Draw confirmed tracks with classification and state
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.box]
        clf = classifications.get(t.track_id)
        state = sm_manager.get_state(t.track_id)
        state_str = state.value if state else "?"

        if state_str == "BASKET":
            state_label = "IN BAG -> IN CART"
            color = (0, 230, 0)       # Green
        elif state_str == "SHELF":
            state_label = "ON TABLE"
            color = (255, 200, 60)    # Cyan / Blue
        elif state_str == "HELD":
            state_label = "MOVING"
            color = (0, 165, 255)     # Orange
        elif state_str == "ANOMALY":
            state_label = "ANOMALY"
            color = (0, 0, 255)       # Red
        else:
            state_label = state_str
            color = (200, 200, 200)

        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        if clf:
            product_label = clf.class_name
            if clf.is_unknown:
                product_label = "UNKNOWN"
                color = (0, 0, 200)
            label = f"#{t.track_id} {product_label} ({clf.conf:.0%}) [{state_label}]"
        else:
            label = f"#{t.track_id} [detecting...] [{state_label}]"

        # Background pill behind label for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(y1 - 6, th + 8)
        cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 6, ty + 2), (20, 20, 20), -1)
        cv2.putText(vis, label, (x1 + 3, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Top banner with quick instruction
    top_bar = vis.copy()
    cv2.rectangle(top_bar, (0, 0), (w, 24), (20, 20, 20), -1)
    cv2.addWeighted(top_bar, 0.65, vis, 0.35, 0, vis)
    cv2.putText(vis, "BAG: Put items in Bag to add to Bill | Lift out of Bag to remove",
                (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

    # Draw recent events on a translucent panel (top-right, under the FPS)
    recent = recent_events[-5:]
    if recent:
        pw, ph = 360, 14 + 22 * len(recent)
        x0, y0 = vis.shape[1] - pw - 10, 32
        overlay = vis.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)
        for i, ev in enumerate(recent):
            action_sym = "+" if ev['event'] == 'pick' else ("-" if ev['event'] == 'return' else "!")
            txt = f"[{action_sym} {ev['event'].upper()}] {ev.get('product', '?')} (#{ev['track_id']})"
            ev_color = (0, 255, 0) if ev['event'] == 'pick' else (
                (255, 200, 0) if ev['event'] == 'return' else (0, 0, 255)
            )
            cv2.putText(vis, txt, (x0 + 8, y0 + 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ev_color, 1)

    return vis


# ─── Per-track label smoothing ───────────────────────────────────────────────

class TrackLabel:
    """Majority-voted product identity for one track (reduces label flicker)."""

    def __init__(self, window: int = CLASS_VOTE_WINDOW) -> None:
        self._names: deque[str] = deque(maxlen=window)
        self._confs: deque[float] = deque(maxlen=window)
        self.unknown_streak = 0
        self.unknown_alerted = False
        self.class_name = "UNKNOWN"
        self.conf = 0.0
        self.is_unknown = True

    def add(self, name: str, conf: float) -> None:
        self._names.append(name)
        self._confs.append(conf)
        winner, _ = Counter(self._names).most_common(1)[0]
        confs = [c for n, c in zip(self._names, self._confs) if n == winner]
        self.class_name = winner
        self.conf = sum(confs) / len(confs)
        self.is_unknown = winner == "UNKNOWN"
        self.unknown_streak = self.unknown_streak + 1 if self.is_unknown else 0

    def to_dict(self) -> dict:
        return {"class_name": self.class_name, "conf": round(self.conf, 4),
                "is_unknown": self.is_unknown}


# ─── Mock scenario (no camera, no models) ────────────────────────────────────

class MockScenario:
    """
    Scripted demo that exercises the whole stack without a camera or weights:
    each cycle one product (from PRODUCT_CLASSES) moves shelf-slot → basket
    (PICK); every third cycle an item is taken back from the basket to its slot
    (RETURN). Coordinates come from the calibrated/default zones, so the cart,
    inventory and low-stock alerts on the dashboard all react.
    """
    CYCLE = 100

    def __init__(self, zones: ZoneManager) -> None:
        self._zones = zones
        slots = zones.all_slots()
        self._slot_centre = {}
        for slot in slots.values():
            x1, y1, x2, y2 = slot["roi"]
            self._slot_centre[slot.get("expected_product")] = ((x1 + x2) / 2, (y1 + y2) / 2)
        sx1, sy1, sx2, sy2 = zones.shelf_roi
        bx1, by1, bx2, by2 = zones.basket_roi
        self._shelf_default = ((sx1 + sx2) / 2, (sy1 + sy2) / 2)
        self._basket = ((bx1 + bx2) / 2, (by1 + by2) / 2)
        self.current_product = PRODUCT_CLASSES[0]

    @staticmethod
    def _lerp(a, b, t):
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    def detections(self, frame_id: int) -> list[DetectionResult]:
        cycle, f = divmod(frame_id, self.CYCLE)
        is_return = cycle % 3 == 2
        # a return takes back the product picked in the previous cycle
        pick_no = (cycle - 1 if is_return else cycle)
        pick_no -= (pick_no + 1) // 3            # count pick cycles only
        product = PRODUCT_CLASSES[pick_no % len(PRODUCT_CLASSES)]
        self.current_product = product
        slot = self._slot_centre.get(product, self._shelf_default)
        start, end = (self._basket, slot) if is_return else (slot, self._basket)
        if f < 20:
            cx, cy = start
        elif f < 45:
            cx, cy = self._lerp(start, end, (f - 20) / 25)
        elif f < 70:
            cx, cy = end
        else:
            return []          # object leaves view; track expires
        half = 45
        return [DetectionResult(class_name="product", conf=0.92,
                                xyxy=[cx - half, cy - half, cx + half, cy + half])]

    def frame(self) -> np.ndarray:
        img = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 40, dtype=np.uint8)
        cv2.putText(img, "MOCK MODE - scripted demo (no camera / models)",
                    (20, FRAME_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
        return img


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class Pipeline:
    def __init__(
        self,
        source=CAMERA_INDEX,
        display: bool = True,
        anomaly_enabled: bool = False,
        mock_mode: bool = False,
        event_callbacks: Optional[list[Callable]] = None,
        session_id: str = DEFAULT_SESSION_ID,
        use_backend: bool = True,
        api_url: str = API_BASE_URL,
        detector_backend: str = "auto",
    ) -> None:
        self.display = display
        self.mock_mode = mock_mode
        self.session_id = session_id

        ensure_dirs()

        self._zones = ZoneManager()
        self._current_source = source
        if not mock_mode:
            from src.classification.infer_classifier import ProductClassifier
            self._detector = YOLODetector(backend=detector_backend)
            self._classifier = ProductClassifier()
            self._mock = None
            self._source = VideoSource(source)
            if not self._source.is_opened:
                raise RuntimeError(
                    f"Could not open video source {source!r}. For a webcam try another "
                    "index (--source 1) or set CAMERA_INDEX in src/config.py."
                )
        else:
            self._detector = None
            self._classifier = None
            self._mock = MockScenario(self._zones)
            self._source = None
            print("[Pipeline] MOCK MODE — scripted detections, no camera or models needed")

        self._tracker = Tracker()
        self._sm = StateMachineManager(session_id=session_id, anomaly_enabled=anomaly_enabled)
        self._sm.register_callback(pipeline_state.push_event)

        self._bridge: Optional[ApiBridge] = ApiBridge(api_url) if use_backend else None
        if self._bridge:
            self._sm.register_callback(self._bridge.send_event)
        if event_callbacks:
            for cb in event_callbacks:
                self._sm.register_callback(cb)

        self._shelf_monitor = ShelfMonitor(self._zones, alert_callback=self._emit_alert_remote)
        self._labels: dict[int, TrackLabel] = {}
        pipeline_state.session_id = session_id
        pipeline_state.running = True
        print(f"[Pipeline] Session: {session_id}")

    def switch_source(self, target_src) -> bool:
        """Switch video source dynamically (e.g. from camera 0 to uploaded video)."""
        if not self._source or self.mock_mode:
            return False
        if str(target_src) == str(self._current_source):
            return True
        try:
            print(f"[Pipeline] Switching video source to: {target_src}")
            new_source = VideoSource(target_src)
            if new_source.is_opened:
                if self._source:
                    self._source.release()
                self._source = new_source
                self._current_source = target_src
                self._tracker = Tracker()
                self._labels.clear()
                print(f"[Pipeline] Successfully switched source to: {target_src}")
                return True
            else:
                print(f"[Pipeline] Failed to open source {target_src}")
                return False
        except Exception as e:
            print(f"[Pipeline] Error switching source: {e}")
            return False

    # ── alerts ────────────────────────────────────────────────────
    def _emit_alert_remote(self, alert: dict) -> None:
        """ShelfMonitor already logs + prints; just forward to the backend."""
        if self._bridge:
            self._bridge.send_alert(alert)

    def _emit_alert(self, alert: dict) -> None:
        LOG_ALERTS.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_ALERTS, "a") as f:
            f.write(json.dumps(alert) + "\n")
        print(f"[ALERT] {alert['type']}: {alert['message']}")
        self._emit_alert_remote(alert)

    # ── main loop ─────────────────────────────────────────────────
    def run(self, max_frames: Optional[int] = None) -> None:
        """Main loop — blocks until 'q' pressed, Ctrl+C, the stream ends or max_frames."""
        frame_id = 0
        t_prev = time.time()
        fps_ema = 0.0
        last_frame_push = last_status_push = 0.0

        try:
            while True:
                t_loop = time.time()
                if self._mock:
                    frame = self._mock.frame()
                else:
                    assert self._source is not None  # always set when not mock_mode
                    ret, frame = self._source.read()
                    if not ret:
                        print("[Pipeline] Stream ended.")
                        break

                frame_id += 1
                if max_frames is not None and frame_id > max_frames:
                    break
                assert frame is not None  # guaranteed: mock returns ndarray; VideoCapture guarded by `if not ret: break`
                pipeline_state.frame_id = frame_id
                pipeline_state.frame = frame

                # ── Stage 1: Detection ────────────────────────────
                if self._mock:
                    detections = self._mock.detections(frame_id)
                else:
                    assert self._detector is not None  # always set when not mock_mode
                    detections = self._detector.detect(frame, frame_id)
                pipeline_state.detections = [d.to_dict() for d in detections]

                # ── Stage 2: Tracking ─────────────────────────────
                tracks = self._tracker.update(detections)
                pipeline_state.tracks = [t.to_dict() for t in tracks]
                live_ids = {t.track_id for t in tracks}
                for tid in list(self._labels):
                    if tid not in live_ids:
                        del self._labels[tid]

                # ── Stage 3: Classify (every N frames) + vote ─────
                for t in tracks:
                    label = self._labels.setdefault(t.track_id, TrackLabel())
                    if self._mock:
                        label.add(self._mock.current_product, 0.95)
                        continue
                    skip_n = 10 if (label.class_name != "UNKNOWN" and label.conf > 0.90) else CLASSIFY_EVERY_N_FRAMES
                    if (frame_id + t.track_id) % skip_n and label._names:
                        continue
                    crop = DetectionResult(class_name="product", conf=t.conf,
                                           xyxy=t.box).crop(frame)
                    if crop.size == 0:
                        continue

                    # Direct prompt resolution from YOLO-World zero-shot prompt
                    t_prompt = getattr(t, "prompt", "").lower()
                    direct_product = None
                    if any(k in t_prompt for k in ("hakka", "noodle")):
                        direct_product = "chings_hakka"
                    elif any(k in t_prompt for k in ("manchurian", "soup")):
                        direct_product = "chings_manchurian"
                    elif any(k in t_prompt for k in ("matchbox", "homelite")):
                        direct_product = "homelite_matchbox"
                    elif any(k in t_prompt for k in ("vaseline", "petroleum jelly")):
                        direct_product = "vaseline_jelly"

                    assert self._classifier is not None  # always set when not mock_mode
                    r = self._classifier.classify(crop, frame_id=frame_id, track_id=t.track_id)
                    final_name = direct_product if direct_product else ("UNKNOWN" if r.is_unknown else r.class_name)
                    final_conf = 0.95 if direct_product else r.conf
                    label.add(final_name, final_conf)

                    if (label.unknown_streak >= UNKNOWN_ALERT_FRAMES
                            and not label.unknown_alerted):
                        label.unknown_alerted = True
                        self._emit_alert({
                            "type": "UNKNOWN_PRODUCT",
                            "message": (f"Untrained / unknown object in view "
                                        f"(track #{t.track_id}). Not added to cart."),
                            "timestamp": time.time(),
                            "details": {"track_id": t.track_id,
                                        "openset_dist": round(float(r.openset_dist), 4)},
                        })

                pipeline_state.classifications = {
                    tid: lab.to_dict() for tid, lab in self._labels.items()
                }

                # ── Stage 4: Zone membership & Dynamic Source ─────
                if frame_id % 30 == 0:
                    self._zones.reload()
                    from src.config import load_active_source
                    desired_src = load_active_source()
                    if str(desired_src) != str(self._current_source):
                        self.switch_source(desired_src)
                tracks_with_zones = []
                for t in tracks:
                    lab = self._labels.get(t.track_id)
                    tracks_with_zones.append({
                        "track_id": t.track_id,
                        "zone": self._zones.classify_location(t.cx, t.cy),
                        "product": lab.class_name if lab else "UNKNOWN",
                        "conf": lab.conf if lab else 0.0,
                        "cx": t.cx,
                        "cy": t.cy,
                    })

                # ── Stage 5: State machine + shelf monitor ────────
                self._sm.process_frame(tracks_with_zones)
                pipeline_state.track_states = self._sm.get_all_states()
                self._shelf_monitor.update(tracks_with_zones)

                # ── FPS ───────────────────────────────────────────
                now = time.time()
                inst = 1.0 / max(now - t_prev, 1e-3)
                fps_ema = inst if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst
                t_prev = now
                pipeline_state.fps = round(fps_ema, 1)

                # ── Annotate (always: the dashboard needs it) ─────
                vis = annotate_frame(
                    frame, detections, tracks, self._labels,
                    self._zones, self._sm, pipeline_state.events,
                )
                cv2.putText(vis, f"FPS: {fps_ema:.1f}", (vis.shape[1] - 120, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                pipeline_state.annotated_frame = vis

                if self._bridge:
                    if now - last_frame_push >= FRAME_PUSH_INTERVAL_SEC:
                        self._bridge.send_frame(vis)
                        last_frame_push = now
                    if now - last_status_push >= STATUS_PUSH_INTERVAL_SEC:
                        self._bridge.send_status({
                            "running": True, "fps": pipeline_state.fps,
                            "frame_id": frame_id, "active_tracks": len(tracks),
                            "session_id": self.session_id,
                        })
                        last_status_push = now

                if self.display:
                    cv2.imshow("DeepRetail Pipeline", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Throttle to TARGET_FPS (always in mock mode / file replay)
                sleep_time = (1.0 / TARGET_FPS) - (time.time() - t_loop)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[Pipeline] Interrupted.")
        finally:
            if self._source:
                self._source.release()
            if self.display:
                cv2.destroyAllWindows()
            pipeline_state.running = False
            if self._bridge:
                self._bridge.send_status({
                    "running": False, "fps": 0.0, "frame_id": frame_id,
                    "active_tracks": 0, "session_id": self.session_id,
                })
                self._bridge.close()
            print("[Pipeline] Stopped.")


# ─── CLI entry point ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run DeepRetail pipeline")
    p.add_argument("--source", type=str, default=str(CAMERA_INDEX),
                   help="Camera index or video file path")
    p.add_argument("--no-display", action="store_true", help="Don't open an OpenCV window")
    p.add_argument("--anomaly", action="store_true",
                   help="Enable Phase 2 anomaly (concealment) detection")
    p.add_argument("--mock", action="store_true",
                   help="Scripted demo — no camera or model weights needed")
    p.add_argument("--session", type=str, default=DEFAULT_SESSION_ID,
                   help=f"Cart session id shown on the dashboard (default: {DEFAULT_SESSION_ID})")
    p.add_argument("--no-backend", action="store_true",
                   help="Don't push events/frames to the FastAPI backend")
    p.add_argument("--api-url", type=str, default=API_BASE_URL)
    p.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    p.add_argument("--detector", choices=["auto", "trained", "world"], default="auto",
                   help="auto = models/yolo_best.pt if it exists, else zero-shot YOLO-World")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    pipeline = Pipeline(
        source=source,
        display=not args.no_display,
        anomaly_enabled=args.anomaly,
        mock_mode=args.mock,
        session_id=args.session,
        use_backend=not args.no_backend,
        api_url=args.api_url,
        detector_backend=args.detector,
    )
    pipeline.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
