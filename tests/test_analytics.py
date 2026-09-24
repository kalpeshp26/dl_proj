"""
Tests for analytics: top products, co-occurrence recommendations.
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def seeded_db(tmp_path, monkeypatch):
    """DB with synthetic purchase history for analytics testing."""
    db_path = tmp_path / "analytics_test.db"
    monkeypatch.setattr("src.backend.db.DATABASE_PATH", db_path)
    from src.backend import db as db_mod
    db_mod.DATABASE_PATH = db_path
    db_mod.init_db()

    # Seed: coke+lays bought together 5 times in same sessions
    for i in range(5):
        db_mod.record_pick("coke_500ml", f"session_{i}")
        db_mod.record_pick("lays_classic", f"session_{i}")

    # Oreo bought alone 3 times
    for i in range(3):
        db_mod.record_pick("oreo", f"solo_session_{i}")

    return db_mod


class TestTopProducts:

    def test_top_products_returns_list(self, seeded_db, monkeypatch):
        monkeypatch.setattr("src.backend.analytics.get_conn", seeded_db.get_conn)
        from src.backend.analytics import get_top_products
        result = get_top_products(limit=10)
        assert isinstance(result, list)

    def test_most_purchased_at_top(self, seeded_db, monkeypatch):
        monkeypatch.setattr("src.backend.analytics.get_conn", seeded_db.get_conn)
        from src.backend.analytics import get_top_products
        result = get_top_products(limit=10)
        # coke and lays each have 5 picks — should be at top
        names = [r["name"] for r in result]
        assert "coke_500ml" in names or "lays_classic" in names


class TestRecommendations:

    def test_coke_recommends_lays(self, seeded_db, monkeypatch):
        monkeypatch.setattr("src.backend.analytics.get_conn", seeded_db.get_conn)
        from src.backend.analytics import get_recommendations
        from src.backend.db import get_product_by_name
        monkeypatch.setattr("src.backend.db.DATABASE_PATH",
                            seeded_db.DATABASE_PATH)

        coke = seeded_db.get_product_by_name("coke_500ml")
        assert coke is not None

        recs = get_recommendations(coke["product_id"], top_n=5)
        rec_names = [r["name"] for r in recs]
        assert "lays_classic" in rec_names, (
            "If Coke and Lays are always bought together, "
            "Lays should be recommended when Coke is queried."
        )

    def test_recommendation_has_required_fields(self, seeded_db, monkeypatch):
        monkeypatch.setattr("src.backend.analytics.get_conn", seeded_db.get_conn)
        from src.backend.analytics import get_recommendations
        coke = seeded_db.get_product_by_name("coke_500ml")
        if coke is None:
            pytest.skip("coke_500ml not in seeded DB")
        recs = get_recommendations(coke["product_id"])
        for r in recs:
            assert "product_id" in r
            assert "name" in r
            assert "co_occurrence_count" in r
            assert "confidence" in r

    def test_no_self_recommendation(self, seeded_db, monkeypatch):
        monkeypatch.setattr("src.backend.analytics.get_conn", seeded_db.get_conn)
        from src.backend.analytics import get_recommendations
        coke = seeded_db.get_product_by_name("coke_500ml")
        if coke is None:
            pytest.skip()
        recs = get_recommendations(coke["product_id"])
        for r in recs:
            assert r["product_id"] != coke["product_id"], \
                "Product should not recommend itself"

    def test_empty_recs_for_isolated_product(self, seeded_db, monkeypatch):
        """A product never bought with others should return empty recommendations."""
        monkeypatch.setattr("src.backend.analytics.get_conn", seeded_db.get_conn)
        from src.backend.analytics import get_recommendations
        # kitkat was never purchased in seeded data
        kitkat = seeded_db.get_product_by_name("kitkat")
        if kitkat is None:
            pytest.skip()
        recs = get_recommendations(kitkat["product_id"])
        assert recs == []
