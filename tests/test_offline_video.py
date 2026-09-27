"""
Test for the offline video processor (uploaded recording → bill).

The detector and classifier are replaced by fakes, so no weights, torch or
ultralytics are needed; tracking, zones, voting and the state machine are real.
A synthetic video moves one product from the shelf ROI to the basket ROI.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
cv2 = pytest.importorskip("cv2")


def _write_video(path: Path, n: int = 90) -> None:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
    for _ in range(n):
        vw.write(np.zeros((480, 640, 3), np.uint8))
    vw.release()


@pytest.fixture()
def patched(tmp_path, monkeypatch):
    from src.backend import db as db_mod
    monkeypatch.setattr(db_mod, "DATABASE_PATH", tmp_path / "t.db")
    db_mod.init_db()

    # default zones: shelf x∈[20,310], basket x∈[330,620]
    monkeypatch.setattr("src.logic.zones.load_zones", lambda: {
        "shelf_roi": [20, 20, 310, 460], "basket_roi": [330, 20, 620, 460], "shelf_slots": {}})

    from src.detection.yolo_infer import DetectionResult
    import src.pipeline.offline_video_processor as ovp

    class FakeDetector:
        def __init__(self, *a, **k): pass
        def detect(self, frame, frame_id):
            # product walks shelf → gap → basket, then sits in the basket
            cx = min(150 + frame_id * 6, 480)
            return [DetectionResult("product", 0.9, [cx - 30, 200, cx + 30, 260])]

    class FakeResult:
        class_name, conf, is_unknown, openset_dist = "chings_hakka", 0.95, False, 0.1

    class FakeClassifier:
        def __init__(self, *a, **k): pass
        def classify(self, crop, frame_id=None, track_id=None): return FakeResult()

    monkeypatch.setattr(ovp, "YOLODetector", FakeDetector)
    fake_mod = types.ModuleType("src.classification.infer_classifier")
    fake_mod.ProductClassifier = FakeClassifier
    monkeypatch.setitem(sys.modules, "src.classification.infer_classifier", fake_mod)
    return ovp, db_mod


def test_pick_is_billed_and_store_untouched(patched, tmp_path):
    ovp, db_mod = patched
    video = tmp_path / "shop.avi"
    _write_video(video)
    stock_before = db_mod.get_product_by_name("chings_hakka")["stock"]

    progress = []
    res = ovp.analyze_shopping_video_offline(str(video), progress_callback=lambda *a: progress.append(a))

    assert res["status"] == "completed"
    assert res["total_picks"] == 1 and res["total_returns"] == 0
    bill = res["bill"]
    assert [i["product"] for i in bill["items"]] == ["chings_hakka"]
    price = db_mod.get_product_by_name("chings_hakka")["price"]
    assert bill["subtotal"] == pytest.approx(price)
    assert bill["grand_total"] == pytest.approx(round(price * (1 + bill["tax_rate"]), 2))
    assert bill["items"][0]["name"] == "Chings Hakka"
    assert progress[-1][2] == 100.0
    # offline analysis produces an invoice only; live stock is not changed
    assert db_mod.get_product_by_name("chings_hakka")["stock"] == stock_before


def test_missing_file_raises(patched):
    ovp, _ = patched
    with pytest.raises(FileNotFoundError):
        ovp.analyze_shopping_video_offline("/nope/missing.mp4")
