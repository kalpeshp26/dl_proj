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
        self.pick_emitted: bool = False        # True once item has been billed
        self._shelf_streak: int = 0            # Debounce shelf returns
        self._picked_product: Optional[str] = initial_product if initial_product != "UNKNOWN" else None
        self._pick_time: Optional[float] = None  # wall-clock time of last pick event

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
            if self.state == ProductState.BASKET and not self._picked_product:
                self._picked_product = product

        event: Optional[RetailEvent] = None

        if self.state == ProductState.SHELF:
            if zone == "BASKET":
                # Item placed directly or swiftly into basket
                self.state = ProductState.BASKET
                self._shelf_streak = 0
                self._held_since = None
                self._came_from_basket = False
                if self.product != "UNKNOWN" and not self.pick_emitted:
                    self.pick_emitted = True
                    self._picked_product = self.product
                    self._pick_time = time.time()
                    event = RetailEvent(
                        event="pick",
                        track_id=self.track_id,
                        product=self.product,
                        conf=self.conf,
                        session_id=self.session_id,
                    )
            elif zone == "TRANSIT":
                # Product lifted — transition to HELD
                self.state = ProductState.HELD
                self._held_since = time.time()
                self._came_from_basket = False
                self._shelf_streak = 0

        elif self.state == ProductState.HELD:
            if zone == "BASKET":
                # Product placed in basket
                self.state = ProductState.BASKET
                self._held_since = None
                self._came_from_basket = False
                self._shelf_streak = 0
                if self.product != "UNKNOWN" and not self.pick_emitted:
                    self.pick_emitted = True
                    self._picked_product = self.product
                    self._pick_time = time.time()
                    event = RetailEvent(
                        event="pick",
                        track_id=self.track_id,
                        product=self.product,
                        conf=self.conf,
                        session_id=self.session_id,
                    )

            elif zone == "SHELF":
                if self._came_from_basket and self.pick_emitted:
                    # BASKET → HELD → SHELF — RETURN event
                    # Cooldown: don't fire return within 3s of the pick
                    if self._pick_time is None or (time.time() - self._pick_time) >= 3.0:
                        ret_prod = self.product if self.product != "UNKNOWN" else (self._picked_product or "UNKNOWN")
                        event = RetailEvent(
                            event="return",
                            track_id=self.track_id,
                            product=ret_prod,
                            conf=max(self.conf, 0.85),
                            session_id=self.session_id,
                        )
                        self.pick_emitted = False
                # Go back to SHELF state
                self.state = ProductState.SHELF
                self._held_since = None
                self._came_from_basket = False

            elif zone == "TRANSIT":
                self._shelf_streak = 0
                # Still being held — check anomaly timeout (Phase 2)
                if (
                    self.anomaly_enabled
                    and self._held_since is not None
                    and (time.time() - self._held_since) > self.anomaly_timeout
                ):
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
            if zone == "BASKET":
                self._shelf_streak = 0
                # If it was previously UNKNOWN or pending pick,
                # emit the PICK event immediately!
                if self.product != "UNKNOWN" and not self.pick_emitted:
                    self.pick_emitted = True
                    self._picked_product = self.product
                    self._pick_time = time.time()
                    event = RetailEvent(
                        event="pick",
                        track_id=self.track_id,
                        product=self.product,
                        conf=self.conf,
                        session_id=self.session_id,
                    )
            elif zone == "SHELF":
                # Direct transition from BASKET to SHELF
                if self.pick_emitted:
                    # Cooldown: don't fire return within 3s of the pick
                    if self._pick_time is None or (time.time() - self._pick_time) >= 3.0:
                        ret_prod = self.product if self.product != "UNKNOWN" else (self._picked_product or "UNKNOWN")
                        event = RetailEvent(
                            event="return",
                            track_id=self.track_id,
                            product=ret_prod,
                            conf=max(self.conf, 0.85),
                            session_id=self.session_id,
                        )
                        self.pick_emitted = False
                self.state = ProductState.SHELF
                self._held_since = None
                self._came_from_basket = False
            elif zone == "TRANSIT":
                # Product lifted from basket into transit
                self.state = ProductState.HELD
                self._held_since = time.time()
                self._came_from_basket = True
                self._shelf_streak = 0

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
        self._settled_basket_items: list[dict] = []  # items in basket {product, cx, cy, last_seen, track_id}

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
        """
        events: list[RetailEvent] = []
        active_ids = set()
        now = time.time()

        for t in tracks_with_zones:
            tid = t["track_id"]
            active_ids.add(tid)
            cx, cy = t.get("cx", 0.0), t.get("cy", 0.0)
            prod = t.get("product", "UNKNOWN")

            # Create new machine for first-seen tracks
            if tid not in self._machines:
                self._machines[tid] = TrackStateMachine(
                    track_id=tid,
                    initial_product=prod,
                    initial_conf=t.get("conf", 0.0),
                    session_id=self.session_id,
                    anomaly_enabled=self.anomaly_enabled,
                )
                if t["zone"] == "BASKET":
                    # Check if this is a track re-acquisition of an item already billed at this spot
                    reacquired = False
                    for item in self._settled_basket_items:
                        dist = ((item["cx"] - cx)**2 + (item["cy"] - cy)**2)**0.5
                        if (item["product"] == prod or prod == "UNKNOWN") and dist < 120.0 and (now - item["last_seen"]) < 6.0:
                            reacquired = True
                            item["last_seen"] = now
                            item["track_id"] = tid
                            item["cx"] = cx
                            item["cy"] = cy
                            break

                    self._machines[tid].state = ProductState.BASKET
                    # Only set _came_from_basket if we are truly re-acquiring an already-billed item;
                    # brand-new tracks appearing in basket are just picked items, not returns.
                    self._machines[tid]._came_from_basket = reacquired
                    self._machines[tid].pick_emitted = reacquired

            # Update last_seen for settled basket items
            if t["zone"] == "BASKET":
                for item in self._settled_basket_items:
                    if item.get("track_id") == tid:
                        item["last_seen"] = now
                        item["cx"] = cx
                        item["cy"] = cy

            event = self._machines[tid].update(
                zone=t["zone"],
                product=t.get("product", "UNKNOWN"),
                conf=t.get("conf", 0.0),
            )
            if event:
                events.append(event)
                self._log_event(event)

                if event.event == "pick":
                    self._settled_basket_items.append({
                        "track_id": tid,
                        "product": event.product,
                        "cx": cx,
                        "cy": cy,
                        "last_seen": now,
                    })
                elif event.event == "return":
                    found_idx = -1
                    for idx, it in enumerate(self._settled_basket_items):
                        if it["product"] == event.product or it.get("track_id") == tid:
                            found_idx = idx
                            break
                    if found_idx >= 0:
                        self._settled_basket_items.pop(found_idx)

                for cb in self._event_callbacks:
                    try:
                        cb(event)
                    except Exception as e:
                        print(f"[StateMachine] Callback error: {e}")

        # Re-identification putback detection:
        # If an item was billed in the basket, but has now disappeared from the basket,
        # and an item of that product appears on the SHELF (table), emit return even if
        # hand occlusion changed the track ID.
        # Guard: only fire if the basket item has been gone for at least 2.5s AND the
        # shelf item is far from the basket item's last known position (>80px) —
        # this prevents brief zone misclassifications at low FPS from looking like putbacks.
        active_basket_tids = {t["track_id"] for t in tracks_with_zones if t["zone"] == "BASKET"}
        for t in tracks_with_zones:
            if t["zone"] == "SHELF":
                tid = t["track_id"]
                prod = t.get("product", "UNKNOWN")
                machine = self._machines.get(tid)
                if machine and machine.state == ProductState.SHELF and not machine._came_from_basket:
                    for s_idx, item in enumerate(self._settled_basket_items):
                        if (prod != "UNKNOWN" and item["product"] == prod) or (prod != "UNKNOWN" and item["product"] == "UNKNOWN"):
                            if item["track_id"] not in active_basket_tids:
                                time_gone = now - item["last_seen"]
                                shelf_cx, shelf_cy = t.get("cx", 0.0), t.get("cy", 0.0)
                                dist_from_basket = ((item["cx"] - shelf_cx)**2 + (item["cy"] - shelf_cy)**2)**0.5
                                # Require item to have been absent from basket for ≥2.5s
                                # AND the shelf position to be meaningfully away from the basket position
                                if time_gone < 2.5 or dist_from_basket < 80.0:
                                    continue  # too soon / too close — likely a zone flicker, not a real putback
                                ret_prod = item["product"] if item["product"] != "UNKNOWN" else prod
                                ret_event = RetailEvent(
                                    event="return",
                                    track_id=tid,
                                    product=ret_prod,
                                    conf=max(t.get("conf", 0.85), 0.85),
                                    session_id=self.session_id,
                                )
                                events.append(ret_event)
                                self._log_event(ret_event)
                                self._settled_basket_items.pop(s_idx)
                                machine._came_from_basket = False
                                machine.pick_emitted = False
                                for cb in self._event_callbacks:
                                    try:
                                        cb(ret_event)
                                    except Exception as e:
                                        print(f"[StateMachine] Callback error: {e}")
                                break

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
