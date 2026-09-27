"""
DeepRetail — Shelf Monitoring & Misplaced Product Detection (Phase 2)

Responsibilities:
    1. Count classified products per shelf slot every N frames
       (sanity cross-check; event-driven picks/returns are the DB source of truth)
    2. Detect misplaced products (wrong product in a slot for K consecutive frames)
    3. Emit LOW_STOCK and MISPLACED_PRODUCT alerts

This module is called from run_pipeline.py every frame.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    MISPLACED_CONSECUTIVE_FRAMES, SHELF_COUNT_INTERVAL_FRAMES,
    LOG_ALERTS,
)
from src.logic.zones import ZoneManager


class ShelfMonitor:
    """
    Tracks per-slot product counts and detects anomalies.
    Call .update() each pipeline frame.
    """

    def __init__(
        self,
        zone_manager: ZoneManager,
        alert_callback: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self._zones = zone_manager
        self._alert_cb = alert_callback
        self._frame_count = 0

        # Per-slot: consecutive frames where wrong product observed
        self._misplace_counters: dict[str, int] = defaultdict(int)
        # Per-slot: last detected product in that slot
        self._slot_last_product: dict[str, str] = {}

        # Alert dedup: avoid re-firing the same alert every frame
        self._active_alerts: set[str] = set()

        LOG_ALERTS.parent.mkdir(parents=True, exist_ok=True)

    def update(
        self,
        tracks_with_zones: list[dict],
        db_low_stock_check: Optional[Callable[[], list[dict]]] = None,
    ) -> list[dict]:
        """
        Process one frame. Returns list of new alert dicts emitted this frame.

        tracks_with_zones: same structure as used by state machine:
            [{"track_id", "zone", "product", "conf"}, ...]

        db_low_stock_check: optional callable → list of low-stock products from DB.
            Call periodically to avoid per-frame DB hits.
        """
        self._frame_count += 1
        alerts: list[dict] = []

        # Build per-slot product observations for this frame
        slot_observations: dict[str, list[str]] = defaultdict(list)

        for t in tracks_with_zones:
            if t["zone"] != "SHELF" or t["product"] == "UNKNOWN":
                continue
            # Which slot does this track's centroid belong to?
            # We need cx/cy — these come from the pipeline when assembling the dict.
            # Fallback: skip if not provided
            cx = t.get("cx")
            cy = t.get("cy")
            if cx is None or cy is None:
                continue
            slot_id = self._zones.get_slot(cx, cy)
            if slot_id:
                slot_observations[slot_id].append(t["product"])

        # ── Misplaced product detection ───────────────────────────
        for slot_id, slot in self._zones.all_slots().items():
            expected = slot.get("expected_product")
            if not expected:
                continue

            detected_products = slot_observations.get(slot_id, [])
            if not detected_products:
                # Nothing in slot — reset misplace counter
                self._misplace_counters[slot_id] = 0
                continue

            # Use majority product in slot this frame
            from collections import Counter
            majority_product = Counter(detected_products).most_common(1)[0][0]
            self._slot_last_product[slot_id] = majority_product

            if majority_product != expected:
                self._misplace_counters[slot_id] += 1
                if (self._misplace_counters[slot_id] >= MISPLACED_CONSECUTIVE_FRAMES
                        and slot_id not in self._active_alerts):
                    alert = {
                        "type": "MISPLACED_PRODUCT",
                        "message": (f"Wrong product in {slot_id}: "
                                    f"expected '{expected}', detected '{majority_product}'"),
                        "timestamp": time.time(),
                        "details": {
                            "slot_id": slot_id,
                            "expected": expected,
                            "detected": majority_product,
                            "consecutive_frames": self._misplace_counters[slot_id],
                        },
                    }
                    alerts.append(alert)
                    self._active_alerts.add(slot_id)
                    self._fire_alert(alert)
            else:
                # Correct product — reset counter and clear active alert
                self._misplace_counters[slot_id] = 0
                self._active_alerts.discard(slot_id)

        # ── Low-stock check (every N frames) ─────────────────────
        if (db_low_stock_check is not None
                and self._frame_count % SHELF_COUNT_INTERVAL_FRAMES == 0):
            low = db_low_stock_check()
            for product in low:
                key = f"LOW_STOCK_{product['name']}"
                if key not in self._active_alerts:
                    alert = {
                        "type": "LOW_STOCK",
                        "message": (f"Low stock: {product['name']} "
                                    f"({product['stock']} remaining, "
                                    f"threshold={product['low_stock_threshold']})"),
                        "timestamp": time.time(),
                        "details": product,
                    }
                    alerts.append(alert)
                    self._active_alerts.add(key)
                    self._fire_alert(alert)
                # If stock recovered above threshold — clear alert
            current_low_names = {p["name"] for p in low}
            for key in list(self._active_alerts):
                if key.startswith("LOW_STOCK_"):
                    name = key.removeprefix("LOW_STOCK_")
                    if name not in current_low_names:
                        self._active_alerts.discard(key)

        return alerts

    def _fire_alert(self, alert: dict) -> None:
        """Log alert and call callback."""
        with open(LOG_ALERTS, "a") as f:
            f.write(json.dumps(alert) + "\n")
        if self._alert_cb:
            try:
                self._alert_cb(alert)
            except Exception as e:
                print(f"[ShelfMonitor] Alert callback error: {e}")
        print(f"[ALERT] {alert['type']}: {alert['message']}")
