"""
Tests for shelf monitoring: low-stock alerts and misplaced product detection.
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logic.shelf_monitor import ShelfMonitor
from src.config import MISPLACED_CONSECUTIVE_FRAMES


@pytest.fixture()
def zone_mgr(monkeypatch):
    def mock_load():
        return {
            "shelf_roi": [0, 0, 500, 300],
            "basket_roi": [600, 400, 1200, 700],
            "shelf_slots": {
                "slot_0": {
                    "roi": [0, 0, 250, 300],
                    "expected_product": "chings_manchurian",
                    "low_stock_threshold": 3,
                },
            },
        }
    monkeypatch.setattr("src.logic.zones.load_zones", mock_load)
    from src.logic.zones import ZoneManager
    return ZoneManager()


class TestMisplacedProductDetection:

    def test_no_alert_for_correct_product(self, zone_mgr):
        fired = []
        monitor = ShelfMonitor(zone_mgr, alert_callback=lambda a: fired.append(a))
        for _ in range(MISPLACED_CONSECUTIVE_FRAMES + 5):
            monitor.update([{
                "track_id": 1, "zone": "SHELF",
                "product": "chings_manchurian", "conf": 0.9,
                "cx": 100.0, "cy": 100.0,
            }])
        misplaced = [a for a in fired if a["type"] == "MISPLACED_PRODUCT"]
        assert len(misplaced) == 0

    def test_alert_fires_after_k_frames(self, zone_mgr):
        fired = []
        monitor = ShelfMonitor(zone_mgr, alert_callback=lambda a: fired.append(a))
        for _ in range(MISPLACED_CONSECUTIVE_FRAMES + 1):
            monitor.update([{
                "track_id": 1, "zone": "SHELF",
                "product": "chings_hakka",   # wrong product
                "conf": 0.9,
                "cx": 100.0, "cy": 100.0,
            }])
        misplaced = [a for a in fired if a["type"] == "MISPLACED_PRODUCT"]
        assert len(misplaced) >= 1
        assert misplaced[0]["details"]["expected"] == "chings_manchurian"
        assert misplaced[0]["details"]["detected"] == "chings_hakka"

    def test_no_flicker_false_positive(self, zone_mgr):
        """Alert should NOT fire for fewer than K frames."""
        fired = []
        monitor = ShelfMonitor(zone_mgr, alert_callback=lambda a: fired.append(a))
        for _ in range(MISPLACED_CONSECUTIVE_FRAMES - 1):
            monitor.update([{
                "track_id": 1, "zone": "SHELF",
                "product": "chings_hakka",
                "conf": 0.9,
                "cx": 100.0, "cy": 100.0,
            }])
        misplaced = [a for a in fired if a["type"] == "MISPLACED_PRODUCT"]
        assert len(misplaced) == 0, "Flicker suppression failed — alert fired too early"

    def test_alert_clears_when_correct_product_returns(self, zone_mgr):
        fired = []
        monitor = ShelfMonitor(zone_mgr, alert_callback=lambda a: fired.append(a))
        # Trigger misplacement
        for _ in range(MISPLACED_CONSECUTIVE_FRAMES + 1):
            monitor.update([{
                "track_id": 1, "zone": "SHELF",
                "product": "chings_hakka",
                "conf": 0.9, "cx": 100.0, "cy": 100.0,
            }])
        # Put correct product back
        monitor.update([{
            "track_id": 1, "zone": "SHELF",
            "product": "chings_manchurian",
            "conf": 0.9, "cx": 100.0, "cy": 100.0,
        }])
        # Slot should now be cleared (active_alerts should not contain slot_0)
        assert "slot_0" not in monitor._active_alerts


class TestLowStockAlert:

    def test_low_stock_fires_from_db_callback(self, zone_mgr):
        fired = []
        monitor = ShelfMonitor(zone_mgr, alert_callback=lambda a: fired.append(a))

        # Simulate a DB callback returning low-stock products
        def mock_low_stock():
            return [{"name": "chings_manchurian", "stock": 2, "low_stock_threshold": 3,
                     "product_id": 1}]

        # Need to trigger on the right frame interval
        from src.config import SHELF_COUNT_INTERVAL_FRAMES
        for i in range(SHELF_COUNT_INTERVAL_FRAMES + 1):
            monitor.update([], db_low_stock_check=mock_low_stock)

        low_alerts = [a for a in fired if a["type"] == "LOW_STOCK"]
        assert len(low_alerts) >= 1
        assert "chings_manchurian" in low_alerts[0]["message"]
