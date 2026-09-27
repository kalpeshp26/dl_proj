"""
Tests for:
    - ApiBridge: events / status / frames reach an HTTP backend from a background thread
    - Mock pipeline scenario: produces PICK and RETURN events for real product classes
    - State machine: a track first seen inside the basket must not create a second PICK
No camera, model weights, torch or FastAPI required.
"""

import http.server
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest


class _Recorder(http.server.BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        _Recorder.received.append((self.path, body))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


@pytest.fixture()
def http_backend():
    _Recorder.received = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def _wait_for(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


class TestApiBridge:
    def test_event_status_frame_delivered(self, http_backend):
        from src.pipeline.api_bridge import ApiBridge
        from src.logic.state_machine import RetailEvent

        bridge = ApiBridge(http_backend)
        bridge.send_event(RetailEvent(event="pick", track_id=1, product="chings_hakka",
                                      conf=0.9, session_id="demo_session"))
        bridge.send_status({"running": True, "fps": 12.0, "frame_id": 5,
                            "active_tracks": 1, "session_id": "demo_session"})
        bridge.send_frame(np.zeros((720, 1280, 3), dtype=np.uint8))

        paths = lambda: {p for p, _ in _Recorder.received}
        assert _wait_for(lambda: {"/internal/event", "/internal/status",
                                  "/internal/frame"} <= paths())
        bridge.close()

        ev = json.loads(next(b for p, b in _Recorder.received if p == "/internal/event"))
        assert ev["event"] == "pick" and ev["product"] == "chings_hakka"
        frame = next(b for p, b in _Recorder.received if p == "/internal/frame")
        assert frame[:2] == b"\xff\xd8"  # JPEG magic

    def test_backend_down_does_not_block(self):
        from src.pipeline.api_bridge import ApiBridge
        bridge = ApiBridge("http://127.0.0.1:9", timeout=0.2)  # nothing listens here
        t0 = time.time()
        for _ in range(50):
            bridge.send_alert({"type": "X", "message": "m", "timestamp": 0})
        assert time.time() - t0 < 0.5  # enqueue is non-blocking
        bridge.close(flush_timeout=0.1)


class TestMockPipeline:
    def test_mock_emits_pick_and_return(self, tmp_path, monkeypatch):
        import src.pipeline.run_pipeline as rp
        from src.config import PRODUCT_CLASSES

        monkeypatch.setattr("src.logic.state_machine.LOG_EVENTS", tmp_path / "events.jsonl")
        monkeypatch.setattr("src.tracking.tracker.LOG_TRACKING", tmp_path / "tracking.jsonl")
        monkeypatch.setattr("src.logic.shelf_monitor.LOG_ALERTS", tmp_path / "alerts.jsonl")
        monkeypatch.setattr(rp, "TARGET_FPS", 10_000)

        events = []
        p = rp.Pipeline(mock_mode=True, display=False, use_backend=False,
                        event_callbacks=[events.append])
        p.run(max_frames=3 * rp.MockScenario.CYCLE + 5)

        kinds = [e.event for e in events]
        assert kinds.count("pick") >= 2
        assert "return" in kinds
        assert all(e.product in PRODUCT_CLASSES for e in events)
        assert all(e.session_id == "demo_session" for e in events)


class TestReacquiredBasketTrack:
    def test_first_seen_in_basket_is_not_a_pick(self):
        from src.logic.state_machine import StateMachineManager
        mgr = StateMachineManager(session_id="t")
        evs = []
        for _ in range(5):
            evs += mgr.process_frame([{"track_id": 7, "zone": "BASKET",
                                       "product": "vaseline_jelly", "conf": 0.9}])
        assert evs == []

    def test_basket_to_shelf_is_a_return(self):
        from src.logic.state_machine import StateMachineManager
        mgr = StateMachineManager(session_id="t")
        evs = []
        for zone in ["BASKET", "TRANSIT", "SHELF"]:
            evs += mgr.process_frame([{"track_id": 7, "zone": zone,
                                       "product": "vaseline_jelly", "conf": 0.9}])
        assert [e.event for e in evs] == ["return"]
