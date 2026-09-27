"""
DeepRetail — Zone / ROI Logic (Stage 4)

Pure Python, no DL. Answers questions like:
    - Is centroid (cx, cy) inside the shelf ROI?
    - Is centroid inside the basket ROI?
    - Which shelf slot does this centroid belong to?

All zone coordinates come from config.json (written by calibrate.py).
No pixel coordinates are hard-coded here.
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import load_zones


# ─── ROI helpers ─────────────────────────────────────────────────────────────

def point_in_roi(cx: float, cy: float, roi: list[int]) -> bool:
    """Return True if (cx, cy) is inside [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = roi
    return x1 <= cx <= x2 and y1 <= cy <= y2


def iou_roi(box: list[float], roi: list[int]) -> float:
    """IoU between a bounding box and a zone ROI."""
    ax1, ay1, ax2, ay2 = box
    bx1, by1, bx2, by2 = roi
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_box = (ax2 - ax1) * (ay2 - ay1)
    area_roi = (bx2 - bx1) * (by2 - by1)
    return inter / (area_box + area_roi - inter + 1e-6)


# ─── Zone manager ─────────────────────────────────────────────────────────────

class ZoneManager:
    """
    Loads zone config once and answers spatial membership queries.
    Reload with .reload() after re-calibration.
    """

    def __init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        zones = load_zones()
        self._shelf_roi: list[int] = zones.get("shelf_roi", [0, 0, 9999, 9999])
        self._basket_roi: list[int] = zones.get("basket_roi", [0, 0, 0, 0])
        self._slots: dict = zones.get("shelf_slots", {})

    # ── Public query interface ────────────────────────────────────

    def in_shelf(self, cx: float, cy: float) -> bool:
        return point_in_roi(cx, cy, self._shelf_roi)

    def in_basket(self, cx: float, cy: float) -> bool:
        return point_in_roi(cx, cy, self._basket_roi)

    def in_neither(self, cx: float, cy: float) -> bool:
        """True when the object is between shelf and basket ('in transit')."""
        return not self.in_shelf(cx, cy) and not self.in_basket(cx, cy)

    def get_slot(self, cx: float, cy: float) -> Optional[str]:
        """
        Return slot_id if centroid falls inside a configured shelf slot.
        Returns None if no slot matches.
        """
        for slot_id, slot in self._slots.items():
            roi = slot.get("roi")
            if roi and point_in_roi(cx, cy, roi):
                return slot_id
        return None

    def expected_product_for_slot(self, slot_id: str) -> Optional[str]:
        slot = self._slots.get(slot_id, {})
        return slot.get("expected_product")

    def low_stock_threshold_for_slot(self, slot_id: str) -> int:
        slot = self._slots.get(slot_id, {})
        return int(slot.get("low_stock_threshold", 3))

    def all_slots(self) -> dict:
        return dict(self._slots)

    @property
    def shelf_roi(self) -> list[int]:
        return self._shelf_roi

    @property
    def basket_roi(self) -> list[int]:
        return self._basket_roi

    def classify_location(self, cx: float, cy: float) -> str:
        """
        Returns one of: "SHELF", "BASKET", "TRANSIT"
        Used by the state machine to determine zone membership.
        """
        if self.in_shelf(cx, cy):
            return "SHELF"
        if self.in_basket(cx, cy):
            return "BASKET"
        return "TRANSIT"
