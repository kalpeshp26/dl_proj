"""
Tests for Stage 4: Zone logic — ROI membership, slot detection.
"""

import pytest
from src.logic.zones import ZoneManager, point_in_roi


class TestPointInRoi:
    def test_point_inside(self):
        assert point_in_roi(50, 50, [0, 0, 100, 100]) is True

    def test_point_on_edge(self):
        assert point_in_roi(0, 0, [0, 0, 100, 100]) is True
        assert point_in_roi(100, 100, [0, 0, 100, 100]) is True

    def test_point_outside(self):
        assert point_in_roi(200, 200, [0, 0, 100, 100]) is False

    def test_point_negative_coords(self):
        assert point_in_roi(-1, 50, [0, 0, 100, 100]) is False


class TestZoneManager:
    @pytest.fixture()
    def zone_mgr(self, monkeypatch):
        """ZoneManager with patched load_zones returning deterministic config."""
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
                    "slot_1": {
                        "roi": [250, 0, 500, 300],
                        "expected_product": "chings_hakka",
                        "low_stock_threshold": 3,
                    },
                },
            }
        monkeypatch.setattr("src.logic.zones.load_zones", mock_load)
        return ZoneManager()

    def test_point_in_shelf(self, zone_mgr):
        assert zone_mgr.in_shelf(100, 100) is True

    def test_point_not_in_shelf(self, zone_mgr):
        assert zone_mgr.in_shelf(700, 600) is False

    def test_point_in_basket(self, zone_mgr):
        assert zone_mgr.in_basket(800, 500) is True

    def test_point_not_in_basket(self, zone_mgr):
        assert zone_mgr.in_basket(100, 100) is False

    def test_in_neither_is_transit(self, zone_mgr):
        # 550, 350 is outside both shelf and basket
        assert zone_mgr.in_neither(550, 350) is True

    def test_classify_location_shelf(self, zone_mgr):
        assert zone_mgr.classify_location(100, 100) == "SHELF"

    def test_classify_location_basket(self, zone_mgr):
        assert zone_mgr.classify_location(800, 500) == "BASKET"

    def test_classify_location_transit(self, zone_mgr):
        assert zone_mgr.classify_location(550, 350) == "TRANSIT"

    def test_get_slot_returns_correct_slot(self, zone_mgr):
        # (100, 100) is in slot_0 [0,0,250,300]
        assert zone_mgr.get_slot(100, 100) == "slot_0"

    def test_get_slot_returns_none_outside(self, zone_mgr):
        assert zone_mgr.get_slot(700, 600) is None

    def test_expected_product_for_slot(self, zone_mgr):
        assert zone_mgr.expected_product_for_slot("slot_0") == "chings_manchurian"
        assert zone_mgr.expected_product_for_slot("slot_1") == "chings_hakka"

    def test_low_stock_threshold(self, zone_mgr):
        assert zone_mgr.low_stock_threshold_for_slot("slot_0") == 3
