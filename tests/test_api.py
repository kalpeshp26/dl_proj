"""
Tests for FastAPI endpoints: cart, inventory, alerts, analytics.
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Test client with isolated DB."""
    db_path = tmp_path / "api_test.db"
    monkeypatch.setattr("src.backend.db.DATABASE_PATH", db_path)

    from src.backend import db as db_mod
    db_mod.DATABASE_PATH = db_path

    # Patch the db module used inside main.py
    import importlib
    import src.backend.main as main_mod
    # Re-patch main's reference to DATABASE_PATH via db_mod
    monkeypatch.setattr(main_mod, "init_db", db_mod.init_db)
    monkeypatch.setattr(main_mod, "get_all_products", db_mod.get_all_products)
    monkeypatch.setattr(main_mod, "get_cart", db_mod.get_cart)
    monkeypatch.setattr(main_mod, "get_cart_total", db_mod.get_cart_total)
    monkeypatch.setattr(main_mod, "get_low_stock_products", db_mod.get_low_stock_products)
    monkeypatch.setattr(main_mod, "get_transactions", db_mod.get_transactions)
    monkeypatch.setattr(main_mod, "record_pick", db_mod.record_pick)
    monkeypatch.setattr(main_mod, "record_return", db_mod.record_return)
    monkeypatch.setattr(main_mod, "record_anomaly", db_mod.record_anomaly)

    db_mod.init_db()
    main_mod._alerts.clear()
    main_mod._low_stock_alerted.clear()
    monkeypatch.setattr(main_mod, "_pipeline_status", {})
    monkeypatch.setattr(main_mod, "_pipeline_status_ts", 0.0)
    monkeypatch.setattr(main_mod, "_latest_frame", None)

    from fastapi.testclient import TestClient
    from src.backend.main import app
    return TestClient(app)


class TestCartEndpoint:

    def test_empty_cart(self, client):
        r = client.get("/cart/new_session_123")
        assert r.status_code == 200
        data = r.json()
        assert "items" in data
        assert data["total"] == 0.0

    def test_cart_after_pick(self, client, tmp_path, monkeypatch):
        db_path = tmp_path / "api_test.db"
        monkeypatch.setattr("src.backend.db.DATABASE_PATH", db_path)
        from src.backend import db as db_mod
        db_mod.DATABASE_PATH = db_path
        db_mod.init_db()
        db_mod.record_pick("chings_manchurian", "session_abc")
        r = client.get("/cart/session_abc")
        assert r.status_code == 200


class TestInventoryEndpoint:

    def test_inventory_returns_products(self, client):
        r = client.get("/inventory")
        assert r.status_code == 200
        data = r.json()
        assert "products" in data
        assert len(data["products"]) > 0

    def test_inventory_has_required_fields(self, client):
        r = client.get("/inventory")
        products = r.json()["products"]
        for p in products:
            assert "product_id" in p
            assert "name" in p
            assert "stock" in p
            assert "price" in p


class TestAlertsEndpoint:

    def test_alerts_returns_list(self, client):
        r = client.get("/alerts")
        assert r.status_code == 200
        data = r.json()
        assert "alerts" in data
        assert isinstance(data["alerts"], list)


class TestAnalyticsEndpoints:

    def test_top_products_endpoint(self, client):
        r = client.get("/analytics/top-products")
        assert r.status_code == 200
        data = r.json()
        assert "top_products" in data

    def test_recommendations_endpoint(self, client):
        r = client.get("/analytics/recommendations/1")
        assert r.status_code in (200, 404)

    def test_recommendations_404_for_invalid_id(self, client):
        r = client.get("/analytics/recommendations/99999")
        assert r.status_code == 404


class TestPipelineStatusEndpoint:

    def test_pipeline_status(self, client):
        r = client.get("/pipeline/status")
        assert r.status_code == 200
        data = r.json()
        assert "running" in data
        assert "fps" in data


class TestPipelineBridgeEndpoints:
    """Endpoints the pipeline process pushes to (see src/pipeline/api_bridge.py)."""

    def test_pick_event_updates_cart_and_stock(self, client):
        before = {p["name"]: p["stock"] for p in client.get("/inventory").json()["products"]}
        r = client.post("/internal/event", json={
            "event": "pick", "product": "chings_hakka", "track_id": 1,
            "conf": 0.9, "session_id": "demo_session", "timestamp": 0.0})
        assert r.status_code == 200 and r.json()["recorded"] is True
        cart = client.get("/cart/demo_session").json()
        assert [i["name"] for i in cart["items"]] == ["chings_hakka"]
        after = {p["name"]: p["stock"] for p in client.get("/inventory").json()["products"]}
        assert after["chings_hakka"] == before["chings_hakka"] - 1

    def test_unknown_product_event_is_not_recorded(self, client):
        r = client.post("/internal/event", json={
            "event": "pick", "product": "not_a_product", "track_id": 1,
            "conf": 0.9, "session_id": "s1"})
        assert r.json()["recorded"] is False
        assert client.get("/cart/s1").json()["items"] == []

    def test_low_stock_alert_fires_once(self, client):
        for _ in range(5):
            client.post("/internal/event", json={
                "event": "pick", "product": "vaseline_jelly", "track_id": 1,
                "conf": 0.9, "session_id": "s2"})
        low = [a for a in client.get("/alerts").json()["alerts"] if a["type"] == "LOW_STOCK"]
        assert len(low) == 1

    def test_status_and_frame_roundtrip(self, client):
        assert client.get("/pipeline/status").json()["running"] is False
        client.post("/internal/status", json={"running": True, "fps": 14.2, "frame_id": 9,
                                               "active_tracks": 2, "session_id": "demo_session"})
        st = client.get("/pipeline/status").json()
        assert st["running"] is True and st["session_id"] == "demo_session"

        assert client.get("/frame/latest").status_code == 204
        jpeg = b"\xff\xd8fakejpeg\xff\xd9"
        client.post("/internal/frame", content=jpeg, headers={"Content-Type": "image/jpeg"})
        r = client.get("/frame/latest")
        assert r.status_code == 200 and r.content == jpeg

    def test_alert_clear_and_reset(self, client):
        client.post("/internal/alert", json={"type": "MISPLACED_PRODUCT", "message": "x",
                                              "timestamp": 0.0, "details": {}})
        assert len(client.get("/alerts").json()["alerts"]) == 1
        assert client.post("/alerts/clear").status_code == 200
        assert client.get("/alerts").json()["alerts"] == []

        client.post("/internal/event", json={"event": "pick", "product": "chings_hakka",
                                              "track_id": 1, "conf": 0.9, "session_id": "s3"})
        assert client.post("/admin/reset").status_code == 200
        assert client.get("/cart/s3").json()["items"] == []


class TestHealthEndpoints:

    def test_root_endpoint(self, client):
        r = client.get("/")
        assert r.status_code == 200
        data = r.json()
        assert data["service"] == "DeepRetail API"
        assert data["status"] == "online"

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

