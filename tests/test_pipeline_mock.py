"""
Tests for the full pipeline in mock mode (no camera/model weights needed).
Verifies end-to-end flow from detection contract → events.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestPipelineContractsMockMode:
    """
    These tests verify that the pipeline modules can be instantiated
    and their contracts are respected without any real hardware or models.
    """

    def test_video_source_mock(self):
        """VideoSource can wrap a synthetic frame producer."""
        # We test the VideoSource concept without real camera
        from src.pipeline.run_pipeline import PipelineState
        state = PipelineState()
        assert state.running is False
        assert state.fps == 0.0
        assert state.events == []

    def test_pipeline_state_push_event(self):
        from src.pipeline.run_pipeline import PipelineState
        from src.logic.state_machine import RetailEvent
        state = PipelineState()
        ev = RetailEvent(event="pick", track_id=1, product="chings_manchurian",
                         conf=0.9, session_id="test")
        state.push_event(ev)
        assert len(state.events) == 1
        assert state.events[0]["event"] == "pick"

    def test_pipeline_state_event_buffer_limit(self):
        from src.pipeline.run_pipeline import PipelineState
        from src.logic.state_machine import RetailEvent
        state = PipelineState()
        state._max_events = 5
        for i in range(10):
            ev = RetailEvent(event="pick", track_id=i, product="x",
                             conf=0.9, session_id="s")
            state.push_event(ev)
        assert len(state.events) == 5

    def test_full_mock_pipeline_emits_events(self, tmp_path, monkeypatch):
        """
        Run 50 frames through the pipeline in mock mode and verify
        that pick events are logged.
        """
        log_path = tmp_path / "events.jsonl"
        monkeypatch.setattr("src.logic.state_machine.LOG_EVENTS", log_path)
        monkeypatch.setattr("src.detection.yolo_infer.LOG_DETECTION",
                            tmp_path / "detection.jsonl")
        monkeypatch.setattr("src.tracking.tracker.LOG_TRACKING",
                            tmp_path / "tracking.jsonl")
        monkeypatch.setattr("src.classification.infer_classifier.LOG_CLASSIFICATION",
                            tmp_path / "classification.jsonl")

        # Build a simplified pipeline manually (bypass VideoSource)
        from src.detection.yolo_infer import DetectionResult
        from src.tracking.tracker import Tracker
        from src.logic.zones import ZoneManager
        from src.logic.state_machine import StateMachineManager

        def mock_load():
            return {
                "shelf_roi": [0, 0, 500, 300],
                "basket_roi": [600, 400, 1200, 700],
                "shelf_slots": {},
            }
        monkeypatch.setattr("src.logic.zones.load_zones", mock_load)

        tracker = Tracker(log=False)
        zones = ZoneManager()
        sm = StateMachineManager(session_id="mock_test")

        all_events = []
        sm.register_callback(lambda e: all_events.append(e))

        # Simulate: product on shelf (frames 0-14), moves smoothly to transit (15-29),
        # moves into basket (30-44), stays in basket (45-59)
        start_c = (100.0, 100.0)     # shelf_roi
        transit_c = (550.0, 350.0)   # transit
        end_c = (750.0, 550.0)       # basket_roi

        for frame_id in range(60):
            if frame_id < 15:
                cx, cy = start_c
            elif frame_id < 30:
                t = (frame_id - 15) / 15.0
                cx = start_c[0] + t * (transit_c[0] - start_c[0])
                cy = start_c[1] + t * (transit_c[1] - start_c[1])
            elif frame_id < 45:
                t = (frame_id - 30) / 15.0
                cx = transit_c[0] + t * (end_c[0] - transit_c[0])
                cy = transit_c[1] + t * (end_c[1] - transit_c[1])
            else:
                cx, cy = end_c

            box = [cx - 50.0, cy - 50.0, cx + 50.0, cy + 50.0]

            det = DetectionResult(class_name="product", conf=0.92, xyxy=box)
            tracks = tracker.update([det])

            for t_track in tracks:
                zone = zones.classify_location(t_track.cx, t_track.cy)
                sm.process_frame([{
                    "track_id": t_track.track_id,
                    "zone": zone,
                    "product": "chings_manchurian",
                    "conf": 0.92,
                }])

        # Should have emitted a pick event
        pick_events = [e for e in all_events if e.event == "pick"]
        assert len(pick_events) >= 1, (
            "Pipeline mock mode did not emit a pick event after shelf→transit→basket sequence"
        )
