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
        db_mod.record_pick("coke_500ml", "session_abc")
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
