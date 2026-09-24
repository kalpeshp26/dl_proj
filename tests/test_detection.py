"""
Tests for Stage 1: YOLO output contract.
These tests verify the contract without requiring trained weights.
"""

import numpy as np
import pytest
from src.detection.yolo_infer import DetectionResult


class TestDetectionResultContract:
    """Verify the Stage 1 → Stage 2 output contract."""

    def test_required_fields_present(self):
        d = DetectionResult(class_name="product", conf=0.92, xyxy=[10.0, 20.0, 110.0, 120.0])
        assert hasattr(d, "class_name")
        assert hasattr(d, "conf")
        assert hasattr(d, "xyxy")

    def test_class_name_is_product_or_hand_only(self):
        """YOLO must never output product identity."""
        valid_classes = {"product", "hand"}
        for cls in valid_classes:
            d = DetectionResult(class_name=cls, conf=0.9, xyxy=[0.0, 0.0, 100.0, 100.0])
            assert d.class_name in valid_classes

    def test_xyxy_has_four_elements(self):
        d = DetectionResult(class_name="product", conf=0.8, xyxy=[5.0, 10.0, 50.0, 80.0])
        assert len(d.xyxy) == 4

    def test_conf_in_range(self):
        d = DetectionResult(class_name="hand", conf=0.75, xyxy=[0.0, 0.0, 100.0, 100.0])
        assert 0.0 <= d.conf <= 1.0

    def test_centroid_properties(self):
        d = DetectionResult(class_name="product", conf=0.9, xyxy=[100.0, 200.0, 200.0, 300.0])
        assert d.cx == pytest.approx(150.0)
        assert d.cy == pytest.approx(250.0)

    def test_area_property(self):
        d = DetectionResult(class_name="product", conf=0.9, xyxy=[0.0, 0.0, 100.0, 100.0])
        assert d.area == pytest.approx(10000.0)

    def test_crop_returns_array(self, dummy_frame):
        d = DetectionResult(class_name="product", conf=0.9,
                            xyxy=[100.0, 100.0, 300.0, 300.0])
        crop = d.crop(dummy_frame)
        assert isinstance(crop, np.ndarray)
        assert crop.ndim == 3
        assert crop.shape[2] == 3   # BGR

    def test_crop_clamps_to_frame_boundary(self, dummy_frame):
        """Out-of-bounds boxes should be clamped, not raise errors."""
        d = DetectionResult(class_name="product", conf=0.9,
                            xyxy=[-50.0, -50.0, 2000.0, 2000.0])
        crop = d.crop(dummy_frame)
        assert crop.shape[0] > 0 and crop.shape[1] > 0

    def test_to_dict_keys(self):
        d = DetectionResult(class_name="product", conf=0.85, xyxy=[0.0, 0.0, 50.0, 50.0])
        d_dict = d.to_dict()
        assert "class_name" in d_dict
        assert "conf" in d_dict
        assert "xyxy" in d_dict

    def test_product_identity_not_in_class_name(self):
        """Explicitly verify YOLO contract: product names must not appear."""
        product_names = ["coke", "pepsi", "lays", "oreo", "kitkat"]
        d = DetectionResult(class_name="product", conf=0.9, xyxy=[0.0, 0.0, 100.0, 100.0])
        for name in product_names:
            assert name not in d.class_name.lower(), (
                f"YOLO output contract violated: class_name contains '{name}'. "
                "YOLO must only output 'product' or 'hand'."
            )
