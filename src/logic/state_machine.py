"""
DeepRetail — Product Interaction State Machine (Stage 5)

One state machine instance per track_id. Pure Python, no DL.

States:
    SHELF   — product is resting in the shelf zone
    HELD    — product is in transit (near hand / between zones)
    BASKET  — product has been placed in the basket

Transitions (Phase 1):
    SHELF → HELD           : product leaves shelf zone
    HELD  → BASKET         : product enters basket zone → emit PICK event
    HELD  → SHELF          : product returns to shelf zone (no event)
    BASKET → HELD          : product lifted from basket
    HELD  → SHELF (from BASKET path) → emit RETURN event

Transitions (Phase 2, enabled by set anomaly_enabled=True):
    HELD → ANOMALY         : held longer than ANOMALY_TIMEOUT_SEC without resolution
                             → emit ANOMALY event ("Unregistered Item")

Events emitted:
    {
        "timestamp" : float (Unix),
        "event"     : "pick" | "return" | "anomaly",
        "track_id"  : int,
        "product"   : str,
        "conf"      : float,   # classifier confidence at event time
        "session_id": str,
    }
"""

import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import ANOMALY_TIMEOUT_SEC, LOG_EVENTS


# ─── States ───────────────────────────────────────────────────────────────────

class ProductState(str, Enum):
    SHELF = "SHELF"
    HELD = "HELD"
    BASKET = "BASKET"
    ANOMALY = "ANOMALY"     # terminal state — anomaly emitted


# ─── Event ────────────────────────────────────────────────────────────────────

@dataclass
class RetailEvent:
    event: str               # "pick" | "return" | "anomaly"
    track_id: int
    product: str
    conf: float
    session_id: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "event": self.event,
            "track_id": self.track_id,
            "product": self.product,
            "conf": round(self.conf, 4),
            "session_id": self.session_id,
        }


# ─── Per-track state machine ──────────────────────────────────────────────────

class TrackStateMachine:
    """Manages state for a single tracked product."""

    def __init__(
        self,
        track_id: int,
        initial_product: str = "UNKNOWN",
        initial_conf: float = 0.0,
        session_id: str = "",
        anomaly_enabled: bool = False,
        anomaly_timeout: float = ANOMALY_TIMEOUT_SEC,
    ) -> None:
        self.track_id = track_id
        self.product = initial_product
        self.conf = initial_conf
        self.session_id = session_id
        self.anomaly_enabled = anomaly_enabled
        self.anomaly_timeout = anomaly_timeout

        self.state = ProductState.SHELF
        self._held_since: Optional[float] = None
        self._came_from_basket: bool = False   # track BASKET→HELD→SHELF path

    def update(
        self,
        zone: str,               # "SHELF", "BASKET", "TRANSIT"
        product: str,
        conf: float,
    ) -> Optional[RetailEvent]:
        """
        Feed zone membership and latest classification.
        Returns a RetailEvent if a transition emits one, else None.
        """
        # Update product identity (use latest classifier output)
        if product != "UNKNOWN":
            self.product = product
            self.conf = conf

        event: Optional[RetailEvent] = None

        if self.state == ProductState.SHELF:
            if zone in ("TRANSIT", "BASKET"):
                # Product lifted — transition to HELD
                self.state = ProductState.HELD
                self._held_since = time.time()
                self._came_from_basket = False

        elif self.state == ProductState.HELD:
            if zone == "BASKET":
                # Product placed in basket — PICK event
                self.state = ProductState.BASKET
                self._held_since = None
                self._came_from_basket = False
                event = RetailEvent(
                    event="pick",
                    track_id=self.track_id,
                    product=self.product,
                    conf=self.conf,
                    session_id=self.session_id,
                )

            elif zone == "SHELF":
                if self._came_from_basket:
                    # BASKET → HELD → SHELF — RETURN event
                    event = RetailEvent(
                        event="return",
                        track_id=self.track_id,
                        product=self.product,
                        conf=self.conf,
                        session_id=self.session_id,
                    )
                # Regardless, go back to SHELF state
                self.state = ProductState.SHELF
                self._held_since = None
                self._came_from_basket = False

            elif zone == "TRANSIT":
                # Still being held — check anomaly timeout (Phase 2)
                if (
                    self.anomaly_enabled
                    and self._held_since is not None
                    and (time.time() - self._held_since) > self.anomaly_timeout
                ):
                    # Emit anomaly and move to terminal state
                    held_duration = time.time() - self._held_since
                    confidence = round(self.conf * min(1.0, 15.0 / held_duration), 4)
                    event = RetailEvent(
                        event="anomaly",
                        track_id=self.track_id,
                        product=self.product,
                        conf=confidence,
                        session_id=self.session_id,
                    )
                    self.state = ProductState.ANOMALY
                    self._held_since = None

        elif self.state == ProductState.BASKET:
            if zone in ("TRANSIT", "SHELF"):
                # Product lifted from basket
                self.state = ProductState.HELD
                self._held_since = time.time()
                self._came_from_basket = True

        # ANOMALY is terminal — no more transitions
        return event


# ─── Global state machine manager ────────────────────────────────────────────

class StateMachineManager:
    """
    Manages one TrackStateMachine per active track_id.
    Call .process_frame() each pipeline tick.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        anomaly_enabled: bool = False,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.anomaly_enabled = anomaly_enabled
        self._machines: dict[int, TrackStateMachine] = {}
        self._event_callbacks: list = []   # callables receiving RetailEvent

        LOG_EVENTS.parent.mkdir(parents=True, exist_ok=True)

    def register_callback(self, fn) -> None:
        """Register a callable(event: RetailEvent) to be called on each event."""
        self._event_callbacks.append(fn)

    def process_frame(
        self,
        tracks_with_zones: list[dict],
    ) -> list[RetailEvent]:
        """
        Process all active tracks for this frame.

        tracks_with_zones: list of dicts, each containing:
            {
                "track_id": int,
                "zone"    : str,       # "SHELF"|"BASKET"|"TRANSIT"
                "product" : str,
                "conf"    : float,
            }

        Returns list of any events emitted this frame.
        """
        events: list[RetailEvent] = []
        active_ids = set()

        for t in tracks_with_zones:
            tid = t["track_id"]
            active_ids.add(tid)

            # Create new machine for first-seen tracks
            if tid not in self._machines:
                self._machines[tid] = TrackStateMachine(
                    track_id=tid,
                    initial_product=t.get("product", "UNKNOWN"),
                    initial_conf=t.get("conf", 0.0),
                    session_id=self.session_id,
                    anomaly_enabled=self.anomaly_enabled,
                )

            event = self._machines[tid].update(
                zone=t["zone"],
                product=t.get("product", "UNKNOWN"),
                conf=t.get("conf", 0.0),
            )
            if event:
                events.append(event)
                self._log_event(event)
                for cb in self._event_callbacks:
                    try:
                        cb(event)
                    except Exception as e:
                        print(f"[StateMachine] Callback error: {e}")

        # Check anomaly timeouts for tracks no longer visible (Phase 2)
        if self.anomaly_enabled:
            for tid, machine in list(self._machines.items()):
                if tid not in active_ids and machine.state == ProductState.HELD:
                    if (machine._held_since is not None and
                            (time.time() - machine._held_since) > machine.anomaly_timeout):
                        conf = round(machine.conf * 0.6, 4)  # penalize for lost track
                        event = RetailEvent(
                            event="anomaly",
                            track_id=tid,
                            product=machine.product,
                            conf=conf,
                            session_id=self.session_id,
                        )
                        events.append(event)
                        self._log_event(event)
                        machine.state = ProductState.ANOMALY
                        for cb in self._event_callbacks:
                            try:
                                cb(event)
                            except Exception:
                                pass

        return events

    def get_state(self, track_id: int) -> Optional[ProductState]:
        m = self._machines.get(track_id)
        return m.state if m else None

    def get_all_states(self) -> dict[int, str]:
        return {tid: m.state.value for tid, m in self._machines.items()}

    def _log_event(self, event: RetailEvent) -> None:
        """Write event to JSONL log."""
        with open(LOG_EVENTS, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")
        print(f"[EVENT] {event.event.upper():8s} | "
              f"product={event.product:20s} | "
              f"track_id={event.track_id} | "
              f"conf={event.conf:.2f} | "
              f"session={event.session_id}")

    def new_session(self) -> str:
        """Start a fresh session (clears all state machines)."""
        self.session_id = str(uuid.uuid4())[:8]
        self._machines.clear()
        return self.session_id
