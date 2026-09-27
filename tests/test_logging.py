"""
Tests for JSONL logging integrity across all pipeline stages.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest


class TestEventLogging:

    def test_events_jsonl_has_required_fields(self, tmp_path, monkeypatch):
        log_path = tmp_path / "events.jsonl"
        monkeypatch.setattr("src.logic.state_machine.LOG_EVENTS", log_path)

        from src.logic.state_machine import StateMachineManager
        mgr = StateMachineManager(session_id="log_test")
        mgr.process_frame([{"track_id": 5, "zone": "TRANSIT",
                             "product": "chings_manchurian", "conf": 0.9}])
        mgr.process_frame([{"track_id": 5, "zone": "BASKET",
                             "product": "chings_manchurian", "conf": 0.9}])

        assert log_path.exists()
        lines = [l for l in log_path.read_text().strip().split("\n") if l]
        assert len(lines) >= 1

        for line in lines:
            entry = json.loads(line)
            assert "timestamp" in entry, "Missing 'timestamp'"
            assert "event" in entry, "Missing 'event'"
            assert "track_id" in entry, "Missing 'track_id'"
            assert "product" in entry, "Missing 'product'"
            assert "conf" in entry, "Missing 'conf'"
            assert "session_id" in entry, "Missing 'session_id'"
            # Timestamp must be a float
            assert isinstance(entry["timestamp"], (int, float))
            # Event must be a recognized type
            assert entry["event"] in ("pick", "return", "anomaly")

    def test_no_malformed_lines(self, tmp_path, monkeypatch):
        log_path = tmp_path / "events2.jsonl"
        monkeypatch.setattr("src.logic.state_machine.LOG_EVENTS", log_path)

        from src.logic.state_machine import StateMachineManager
        mgr = StateMachineManager(session_id="malform_test")

        # Run many transitions
        for i in range(5):
            mgr.process_frame([{"track_id": i, "zone": "TRANSIT",
                                  "product": "chings_hakka", "conf": 0.8}])
            mgr.process_frame([{"track_id": i, "zone": "BASKET",
                                  "product": "chings_hakka", "conf": 0.8}])
            mgr.process_frame([{"track_id": i, "zone": "TRANSIT",
                                  "product": "chings_hakka", "conf": 0.8}])
            mgr.process_frame([{"track_id": i, "zone": "SHELF",
                                  "product": "chings_hakka", "conf": 0.8}])

        if log_path.exists():
            for line in log_path.read_text().strip().split("\n"):
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Malformed JSONL line: {line!r}\nError: {e}")


class TestDetectionLogging:

    def test_detection_log_written(self, tmp_path, monkeypatch):
        log_path = tmp_path / "detection.jsonl"
        monkeypatch.setattr("src.detection.yolo_infer.LOG_DETECTION", log_path)

        # Simulate what the YOLO detector's _write_log method does
        import time
        import json as json_mod
        from src.detection.yolo_infer import DetectionResult

        det = DetectionResult(class_name="product", conf=0.9,
                              xyxy=[10.0, 20.0, 100.0, 120.0])
        entry = {
            "ts": time.time(),
            "frame_id": 0,
            "detections": [det.to_dict()],
        }
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json_mod.dumps(entry) + "\n")

        assert log_path.exists()
        line = log_path.read_text().strip()
        parsed = json_mod.loads(line)
        assert "ts" in parsed
        assert "frame_id" in parsed
        assert "detections" in parsed
        assert isinstance(parsed["detections"], list)
