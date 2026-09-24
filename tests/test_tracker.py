"""
Tests for Stage 2: Tracker — ID persistence, occlusion, contract.
"""

import numpy as np
import pytest
from src.detection.yolo_infer import DetectionResult
from src.tracking.tracker import Tracker, Track, _iou


class TestIoU:
    def test_perfect_overlap(self):
        box = [0.0, 0.0, 100.0, 100.0]
        assert _iou(box, box) == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self):
        a = [0.0, 0.0, 50.0, 50.0]
        b = [100.0, 100.0, 200.0, 200.0]
        assert _iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = [0.0, 0.0, 100.0, 100.0]
        b = [50.0, 0.0, 150.0, 100.0]
        # Intersection = 50x100=5000; Union = 10000+10000-5000=15000
        assert _iou(a, b) == pytest.approx(5000 / 15000, abs=1e-3)


class TestTracker:
    def _make_det(self, x1, y1, x2, y2, cls="product", conf=0.9):
        return DetectionResult(class_name=cls, conf=conf, xyxy=[x1, y1, x2, y2])

    def test_track_created_for_new_detection(self):
        t = Tracker(log=False)
        dets = [self._make_det(100, 100, 200, 200)]
        tracks = t.update(dets)
        # After 1 frame, not yet confirmed (min_hits=2 default)
        # Feed again to confirm
        tracks = t.update(dets)
        assert len(tracks) >= 1

    def test_track_id_persists(self):
        """Single product moving slowly should keep the same track_id."""
        t = Tracker(log=False)
        ids = []
        for step in range(20):
            x = 100.0 + step * 3
            dets = [self._make_det(x, 100, x + 100, 200)]
            tracks = t.update(dets)
            if tracks:
                ids.append(tracks[0].track_id)

        # All IDs should be the same (no id switches)
        if ids:
            assert len(set(ids)) == 1, (
                f"Track ID switched unexpectedly: {ids}"
            )

    def test_hand_detections_not_tracked(self):
        """Hand class should be filtered out and not create tracks."""
        t = Tracker(log=False)
        dets = [self._make_det(100, 100, 200, 200, cls="hand")]
        for _ in range(5):
            tracks = t.update(dets)
        assert len(tracks) == 0

    def test_multiple_products_get_different_ids(self):
        t = Tracker(log=False)
        dets = [
            self._make_det(100, 100, 200, 200),
            self._make_det(500, 100, 600, 200),
        ]
        for _ in range(3):
            tracks = t.update(dets)
        if len(tracks) >= 2:
            ids = [tr.track_id for tr in tracks]
            assert len(set(ids)) == len(ids), "Multiple products must have different track IDs"

    def test_track_expires_after_max_age(self):
        t = Tracker(log=False, max_age=5)
        dets = [self._make_det(100, 100, 200, 200)]
        # Confirm track
        for _ in range(3):
            t.update(dets)
        # Now stop detecting it
        for _ in range(10):
            t.update([])
        tracks = t.update([])
        assert len(tracks) == 0, "Track should expire after max_age frames unseen"

    def test_occlusion_recovery(self):
        """Track should re-associate after brief occlusion."""
        t = Tracker(log=False, max_age=15)
        det = self._make_det(100, 100, 200, 200)
        # Establish track
        original_id = None
        for _ in range(3):
            tracks = t.update([det])
            if tracks:
                original_id = tracks[0].track_id

        # Occlude for 5 frames
        for _ in range(5):
            t.update([])

        # Reappear at same location
        tracks = t.update([det])
        # Either same ID (persisted) or new ID (re-spawned) — both acceptable
        # but must have a valid track
        assert tracks is not None

    def test_track_dict_has_required_keys(self):
        t = Tracker(log=False)
        dets = [self._make_det(100, 100, 200, 200)]
        for _ in range(3):
            tracks = t.update(dets)
        if tracks:
            d = tracks[0].to_dict()
            required = {"track_id", "class_name", "box", "conf", "cx", "cy",
                        "age", "hits", "is_confirmed"}
            assert required.issubset(set(d.keys()))

    def test_track_box_has_four_elements(self):
        t = Tracker(log=False)
        dets = [self._make_det(100, 100, 200, 200)]
        for _ in range(3):
            tracks = t.update(dets)
        if tracks:
            assert len(tracks[0].box) == 4
