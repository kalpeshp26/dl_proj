"""
DeepRetail — Custom IoU/Centroid Tracker (Stage 2)

Design:
    - Hungarian assignment (scipy.optimize.linear_sum_assignment) for
      maximum-weight matching between current detections and existing tracks.
    - Primary cost: IoU overlap (higher = better match)
    - Fallback cost: Euclidean centroid distance (when IoU is 0)
    - Tracks are confirmed after TRACK_MIN_HITS detections.
    - Tracks expire after TRACK_MAX_AGE consecutive unseen frames.

Fixed output contract (per-track dict fed to Stage 4/5):
    {
        "track_id"  : int,
        "class_name": str,         # "product" (always — hands are not tracked)
        "box"       : [x1,y1,x2,y2],
        "conf"      : float,
        "cx"        : float,
        "cy"        : float,
        "age"       : int,         # frames since first seen
        "hits"      : int,         # consecutive confirmation frames
        "is_confirmed": bool,
    }
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    TRACK_IOU_THRESH, TRACK_MAX_AGE, TRACK_MIN_HITS,
    TRACK_CENTROID_MAX_DIST, LOG_TRACKING,
)
from src.detection.yolo_infer import DetectionResult


# ─── Track data structure ────────────────────────────────────────────────────

@dataclass
class Track:
    track_id: int
    class_name: str
    box: list[float]           # [x1, y1, x2, y2]
    conf: float
    age: int = 1               # frames since creation
    hits: int = 1              # consecutive matched frames
    frames_since_update: int = 0
    box_history: list = field(default_factory=list)
    prompt: str = ""

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= TRACK_MIN_HITS

    def update(self, det: DetectionResult) -> None:
        self.box = det.xyxy
        self.conf = det.conf
        self.hits += 1
        self.frames_since_update = 0
        self.age += 1
        if hasattr(det, "prompt") and det.prompt:
            self.prompt = det.prompt
        self.box_history.append(list(det.xyxy))
        if len(self.box_history) > 30:
            self.box_history = self.box_history[-30:]

    def mark_missed(self) -> None:
        self.frames_since_update += 1
        self.age += 1

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "box": [round(v, 1) for v in self.box],
            "conf": round(self.conf, 4),
            "cx": round(self.cx, 1),
            "cy": round(self.cy, 1),
            "age": self.age,
            "hits": self.hits,
            "is_confirmed": self.is_confirmed,
        }


# ─── IoU helpers ─────────────────────────────────────────────────────────────

def _iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area
    return inter_area / (union_area + 1e-6)


def _centroid_dist(t: Track, d: DetectionResult) -> float:
    dcx = (d.xyxy[0] + d.xyxy[2]) / 2
    dcy = (d.xyxy[1] + d.xyxy[3]) / 2
    return float(np.sqrt((t.cx - dcx) ** 2 + (t.cy - dcy) ** 2))


def _build_cost_matrix(
    tracks: list[Track],
    detections: list[DetectionResult],
) -> np.ndarray:
    """
    Cost matrix for Hungarian assignment.
    Cost = 1 - IoU (lower = better match).
    If IoU is 0, use normalized centroid distance as fallback.
    """
    n_tracks = len(tracks)
    n_dets = len(detections)
    cost = np.ones((n_tracks, n_dets), dtype=float)

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            iou = _iou(track.box, det.xyxy)
            if iou > 0:
                cost[i, j] = 1.0 - iou
            else:
                # Normalized centroid fallback — map to [0.9, 1.9] range
                # so IoU matches always beat centroid-only matches
                cd = _centroid_dist(track, det)
                cost[i, j] = 1.0 + min(cd / TRACK_CENTROID_MAX_DIST, 1.0)
    return cost


# ─── Tracker ─────────────────────────────────────────────────────────────────

class Tracker:
    """
    Multi-object tracker using Hungarian assignment.
    Tracks only "product" class objects (hands are excluded).
    """

    def __init__(
        self,
        iou_thresh: float = TRACK_IOU_THRESH,
        max_age: int = TRACK_MAX_AGE,
        min_hits: int = TRACK_MIN_HITS,
        log: bool = True,
    ) -> None:
        self._iou_thresh = iou_thresh
        self._max_age = max_age
        self._min_hits = min_hits
        self._log = log

        self._tracks: list[Track] = []
        self._next_id: int = 1
        self._frame_count: int = 0

        LOG_TRACKING.parent.mkdir(parents=True, exist_ok=True)

    @property
    def active_tracks(self) -> list[Track]:
        """Return confirmed, recently-updated tracks."""
        return [t for t in self._tracks if t.is_confirmed and t.frames_since_update <= 1]

    def update(self, detections: list[DetectionResult]) -> list[Track]:
        """
        Process one frame's detections.
        Returns all currently confirmed, active tracks.
        """
        self._frame_count += 1

        # Only track "product" detections
        product_dets = [d for d in detections if d.class_name == "product"]

        if len(self._tracks) == 0:
            # No existing tracks — create fresh tracks for all detections
            for det in product_dets:
                self._spawn(det)
        elif len(product_dets) == 0:
            # No detections — age all tracks
            for t in self._tracks:
                t.mark_missed()
        else:
            # Hungarian assignment
            cost = _build_cost_matrix(self._tracks, product_dets)
            track_indices, det_indices = linear_sum_assignment(cost)

            matched_tracks = set()
            matched_dets = set()

            for ti, di in zip(track_indices, det_indices):
                c = cost[ti, di]
                # Accept match if IoU-based cost < (1 - thresh), i.e. IoU > thresh
                # Centroid-based costs are always ≥ 1.0, so they only match if
                # no better IoU match exists AND centroid is within range
                if c < (1.0 - self._iou_thresh):
                    self._tracks[ti].update(product_dets[di])
                    matched_tracks.add(ti)
                    matched_dets.add(di)
                elif c < 1.5:
                    # Centroid fallback: accept if within TRACK_CENTROID_MAX_DIST/2
                    cd = _centroid_dist(self._tracks[ti], product_dets[di])
                    if cd < TRACK_CENTROID_MAX_DIST / 2:
                        self._tracks[ti].update(product_dets[di])
                        matched_tracks.add(ti)
                        matched_dets.add(di)

            # Miss unmatched tracks
            for ti, track in enumerate(self._tracks):
                if ti not in matched_tracks:
                    track.mark_missed()

            # Spawn new tracks for unmatched detections
            for di, det in enumerate(product_dets):
                if di not in matched_dets:
                    self._spawn(det)

        # Prune dead tracks
        self._tracks = [t for t in self._tracks if t.frames_since_update <= self._max_age]

        active = self.active_tracks
        if self._log:
            self._write_log(active)
        return active

    def _spawn(self, det: DetectionResult) -> Track:
        t = Track(
            track_id=self._next_id,
            class_name=det.class_name,
            box=list(det.xyxy),
            conf=det.conf,
            box_history=[list(det.xyxy)],
            prompt=getattr(det, "prompt", ""),
        )
        self._tracks.append(t)
        self._next_id += 1
        return t

    def get_track(self, track_id: int) -> Optional[Track]:
        for t in self._tracks:
            if t.track_id == track_id:
                return t
        return None

    def _write_log(self, tracks: list[Track]) -> None:
        entry = {
            "ts": time.time(),
            "frame_id": self._frame_count,
            "active_tracks": [t.to_dict() for t in tracks],
        }
        with open(LOG_TRACKING, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ─── Smoke test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import cv2
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.detection.yolo_infer import YOLODetector, draw_detections

    detector = YOLODetector()
    tracker = Tracker()

    source = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
    cap = cv2.VideoCapture(source)
    print("[Tracker] Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector.detect(frame)
        tracks = tracker.update(dets)

        vis = frame.copy()
        # Draw YOLO boxes
        vis = draw_detections(vis, dets, show_conf=False)
        # Draw track IDs
        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t.box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(vis, f"ID:{t.track_id}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(vis, f"Tracks: {len(tracks)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow("Stage 2 — Tracker", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
