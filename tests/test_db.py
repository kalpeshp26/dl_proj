"""
Tests for inventory DB: picks, returns, cart, low-stock, concurrency.
"""

import threading
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Isolated in-memory-style DB using a temp path."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.backend.db.DATABASE_PATH", db_path)
    from src.backend import db as db_mod
    db_mod.DATABASE_PATH = db_path
    db_mod.init_db()
    yield db_mod


class TestInventoryDB:

    def test_init_seeds_products(self, db):
        products = db.get_all_products()
        assert len(products) > 0

    def test_pick_decrements_stock(self, db):
        products = db.get_all_products()
        p = products[0]
        initial_stock = p["stock"]
        db.record_pick(p["name"], "session_1")
        updated = db.get_product_by_name(p["name"])
        assert updated["stock"] == max(0, initial_stock - 1)

    def test_return_increments_stock(self, db):
        products = db.get_all_products()
        p = products[0]
        db.record_pick(p["name"], "session_1")
        after_pick = db.get_product_by_name(p["name"])
        db.record_return(p["name"], "session_1")
        after_return = db.get_product_by_name(p["name"])
        assert after_return["stock"] == after_pick["stock"] + 1

    def test_pick_unknown_product_returns_none(self, db):
        result = db.record_pick("nonexistent_product_xyz", "session_1")
        assert result is None

    def test_cart_reflects_picks(self, db):
        products = db.get_all_products()
        p = products[0]
        db.record_pick(p["name"], "cart_session")
        cart = db.get_cart("cart_session")
        assert len(cart) == 1
        assert cart[0]["name"] == p["name"]
        assert cart[0]["quantity"] == 1

    def test_cart_net_quantity_after_return(self, db):
        products = db.get_all_products()
        p = products[0]
        db.record_pick(p["name"], "cart_session2")
        db.record_pick(p["name"], "cart_session2")
        db.record_return(p["name"], "cart_session2")
        cart = db.get_cart("cart_session2")
        qty = next((i["quantity"] for i in cart if i["name"] == p["name"]), 0)
        assert qty == 1   # 2 picks - 1 return = 1

    def test_cart_empty_after_all_returned(self, db):
        products = db.get_all_products()
        p = products[0]
        db.record_pick(p["name"], "cart_session3")
        db.record_return(p["name"], "cart_session3")
        cart = db.get_cart("cart_session3")
        qty = next((i["quantity"] for i in cart if i["name"] == p["name"]), 0)
        assert qty == 0

    def test_low_stock_triggered(self, db):
        products = db.get_all_products()
        p = products[0]
        # Force stock to 0
        with db.get_conn() as conn:
            conn.execute("UPDATE products SET stock=1, low_stock_threshold=3 WHERE name=?",
                         (p["name"],))
        low = db.get_low_stock_products()
        names = [r["name"] for r in low]
        assert p["name"] in names

    def test_low_stock_not_triggered_above_threshold(self, db):
        products = db.get_all_products()
        p = products[0]
        with db.get_conn() as conn:
            conn.execute("UPDATE products SET stock=10, low_stock_threshold=3 WHERE name=?",
                         (p["name"],))
        low = db.get_low_stock_products()
        names = [r["name"] for r in low]
        assert p["name"] not in names

    def test_concurrent_picks_no_duplicate_transactions(self, db):
        """Multiple threads picking simultaneously should not lose transactions."""
        products = db.get_all_products()
        p = products[0]
        n_threads = 5

        with db.get_conn() as conn:
            conn.execute("UPDATE products SET stock=20 WHERE name=?", (p["name"],))

        errors = []
        def pick():
            try:
                db.record_pick(p["name"], "concurrent_test")
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=pick) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent picks caused errors: {errors}"
        txns = db.get_transactions(limit=100, session_id="concurrent_test", action="pick")
        assert len(txns) == n_threads


class TestInventoryConsistency:

    def test_20_picks_and_returns_zero_drift(self, db):
        """Run 20 mixed pick/return and verify stock matches ground truth."""
        products = db.get_all_products()
        p = products[0]
        initial_stock = p["stock"]

        # 10 picks, 5 returns → net change = -5
        for _ in range(10):
            db.record_pick(p["name"], "consistency_test")
        for _ in range(5):
            db.record_return(p["name"], "consistency_test")

        final = db.get_product_by_name(p["name"])
        expected = max(0, initial_stock - 5)
        assert final["stock"] == expected, (
            f"Stock drift detected: expected {expected}, got {final['stock']}"
        )
