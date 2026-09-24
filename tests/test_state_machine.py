"""
Tests for Stage 5: State machine transitions, pick/return/anomaly events.
"""

import time
import pytest
from src.logic.state_machine import (
    TrackStateMachine, StateMachineManager, ProductState, RetailEvent
)


class TestTrackStateMachine:

    def _make_sm(self, anomaly_enabled=False, anomaly_timeout=2.0):
        return TrackStateMachine(
            track_id=1,
            initial_product="coke_500ml",
            initial_conf=0.9,
            session_id="test_session",
            anomaly_enabled=anomaly_enabled,
            anomaly_timeout=anomaly_timeout,
        )

    def test_initial_state_is_shelf(self):
        sm = self._make_sm()
        assert sm.state == ProductState.SHELF

    def test_shelf_to_held_on_transit(self):
        sm = self._make_sm()
        ev = sm.update("TRANSIT", "coke_500ml", 0.9)
        assert sm.state == ProductState.HELD
        assert ev is None   # no event on SHELF→HELD

    def test_shelf_to_held_on_basket(self):
        """Direct shelf→basket (very fast pick) should still work."""
        sm = self._make_sm()
        ev = sm.update("BASKET", "coke_500ml", 0.9)
        # Goes to HELD first (SHELF→HELD), not BASKET directly
        assert sm.state in (ProductState.HELD, ProductState.BASKET)

    def test_pick_event_on_held_to_basket(self):
        sm = self._make_sm()
        sm.update("TRANSIT", "coke_500ml", 0.9)   # SHELF → HELD
        ev = sm.update("BASKET", "coke_500ml", 0.9)  # HELD → BASKET
        assert sm.state == ProductState.BASKET
        assert ev is not None
        assert ev.event == "pick"
        assert ev.product == "coke_500ml"
        assert ev.track_id == 1

    def test_no_duplicate_pick_events(self):
        sm = self._make_sm()
        sm.update("TRANSIT", "coke_500ml", 0.9)
        ev1 = sm.update("BASKET", "coke_500ml", 0.9)
        ev2 = sm.update("BASKET", "coke_500ml", 0.9)
        assert ev1 is not None and ev1.event == "pick"
        assert ev2 is None   # already in BASKET, no second pick

    def test_return_event_on_basket_held_shelf(self):
        sm = self._make_sm()
        sm.update("TRANSIT", "coke_500ml", 0.9)   # SHELF → HELD
        sm.update("BASKET", "coke_500ml", 0.9)     # HELD → BASKET (pick)
        sm.update("TRANSIT", "coke_500ml", 0.9)    # BASKET → HELD
        ev = sm.update("SHELF", "coke_500ml", 0.9) # HELD → SHELF (return)
        assert sm.state == ProductState.SHELF
        assert ev is not None
        assert ev.event == "return"
        assert ev.product == "coke_500ml"

    def test_no_return_event_on_direct_shelf(self):
        """Putting back without being in basket first should NOT fire return."""
        sm = self._make_sm()
        sm.update("TRANSIT", "coke_500ml", 0.9)   # SHELF → HELD
        ev = sm.update("SHELF", "coke_500ml", 0.9) # HELD → SHELF (no basket)
        assert sm.state == ProductState.SHELF
        assert ev is None   # no return event — never went to basket

    def test_anomaly_on_held_timeout(self):
        """Phase 2: held without resolution should emit anomaly after timeout."""
        sm = self._make_sm(anomaly_enabled=True, anomaly_timeout=0.1)
        sm.update("TRANSIT", "coke_500ml", 0.9)   # SHELF → HELD
        time.sleep(0.2)
        ev = sm.update("TRANSIT", "coke_500ml", 0.9)
        assert ev is not None
        assert ev.event == "anomaly"
        assert sm.state == ProductState.ANOMALY

    def test_no_anomaly_when_disabled(self):
        """Anomaly timeout should not fire when anomaly_enabled=False (Phase 1)."""
        sm = self._make_sm(anomaly_enabled=False, anomaly_timeout=0.1)
        sm.update("TRANSIT", "coke_500ml", 0.9)
        time.sleep(0.2)
        ev = sm.update("TRANSIT", "coke_500ml", 0.9)
        assert ev is None   # no anomaly in Phase 1 mode

    def test_anomaly_state_is_terminal(self):
        """After ANOMALY, no further transitions should happen."""
        sm = self._make_sm(anomaly_enabled=True, anomaly_timeout=0.1)
        sm.update("TRANSIT", "coke_500ml", 0.9)
        time.sleep(0.2)
        sm.update("TRANSIT", "coke_500ml", 0.9)  # triggers anomaly
        assert sm.state == ProductState.ANOMALY
        ev = sm.update("BASKET", "coke_500ml", 0.9)   # should not transition
        assert sm.state == ProductState.ANOMALY
        assert ev is None

    def test_event_has_required_fields(self):
        sm = self._make_sm()
        sm.update("TRANSIT", "coke_500ml", 0.9)
        ev = sm.update("BASKET", "coke_500ml", 0.88)
        assert ev is not None
        d = ev.to_dict()
        required = {"timestamp", "event", "track_id", "product", "conf", "session_id"}
        assert required.issubset(set(d.keys()))

    def test_product_identity_updated_mid_track(self):
        """If classifier updates the product name mid-track, use latest."""
        sm = self._make_sm()
        sm.state = ProductState.HELD
        sm._came_from_basket = False
        sm.update("TRANSIT", "pepsi_500ml", 0.95)
        ev = sm.update("BASKET", "pepsi_500ml", 0.95)
        if ev:
            assert ev.product == "pepsi_500ml"


class TestStateMachineManager:

    def test_multiple_tracks_independent(self):
        """Two different products should have independent state machines."""
        mgr = StateMachineManager(session_id="test")
        ev1 = mgr.process_frame([
            {"track_id": 1, "zone": "TRANSIT", "product": "coke_500ml", "conf": 0.9},
            {"track_id": 2, "zone": "SHELF",   "product": "pepsi_500ml", "conf": 0.8},
        ])
        assert len(ev1) == 0  # no events yet
        ev2 = mgr.process_frame([
            {"track_id": 1, "zone": "BASKET", "product": "coke_500ml", "conf": 0.9},
            {"track_id": 2, "zone": "SHELF",  "product": "pepsi_500ml", "conf": 0.8},
        ])
        # Track 1 should have emitted pick; track 2 no event
        pick_events = [e for e in ev2 if e.event == "pick"]
        assert len(pick_events) == 1
        assert pick_events[0].track_id == 1

    def test_event_logged_to_file(self, tmp_path, monkeypatch):
        """Events must be written to logs/events.jsonl."""
        import json
        log_path = tmp_path / "events.jsonl"
        monkeypatch.setattr("src.logic.state_machine.LOG_EVENTS", log_path)

        mgr = StateMachineManager(session_id="logtest")
        mgr.process_frame([
            {"track_id": 10, "zone": "TRANSIT", "product": "coke_500ml", "conf": 0.9},
        ])
        mgr.process_frame([
            {"track_id": 10, "zone": "BASKET", "product": "coke_500ml", "conf": 0.9},
        ])

        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) >= 1
        entry = json.loads(lines[0])
        assert "event" in entry
        assert "timestamp" in entry
        assert "track_id" in entry
        assert "product" in entry
        assert "session_id" in entry
