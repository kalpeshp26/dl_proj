"""
Tests for Stage 8 — demand forecasting and the simulated-history seeder.
No camera, weights or torch required.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.forecasting import forecast as fc


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    from src.backend import db as db_mod
    monkeypatch.setattr(db_mod, "DATABASE_PATH", db_path)
    db_mod.init_db()
    yield db_mod


class TestSES:

    def test_constant_series_forecasts_constant(self):
        y = np.full(30, 4.0)
        for a in (0.1, 0.5, 0.9):
            assert fc.ses_level(y, a) == pytest.approx(4.0)

    def test_alpha_one_is_naive(self):
        y = np.array([1, 5, 2, 8, 3], dtype=float)
        assert fc.ses_level(y, 1.0) == pytest.approx(3.0)

    def test_manual_recursion(self):
        y = np.array([2.0, 4.0, 6.0])
        # l0=2, l1=.5*4+.5*2=3, l2=.5*6+.5*3=4.5
        assert fc.ses_level(y, 0.5) == pytest.approx(4.5)

    def test_empty(self):
        assert fc.ses_level(np.zeros(0), 0.3) == 0.0

    def test_select_alpha_prefers_high_alpha_for_level_shift(self):
        y = np.array([1.0] * 20 + [10.0] * 20)
        alpha, mae, errs = fc.select_alpha(y, alphas=(0.1, 0.7), window=14)
        assert alpha == 0.7
        assert len(errs) == 14


class TestReorderAdvice:

    def test_formulas(self):
        r = fc.reorder_advice(d_hat=5.0, sigma=2.0, stock=10, lead=2, horizon=7, z=1.65)
        safety = 1.65 * 2.0 * np.sqrt(2)
        assert r["safety_stock"] == pytest.approx(safety, abs=0.01)
        assert r["reorder_point"] == pytest.approx(10 + safety, abs=0.01)
        assert r["order_qty"] == int(np.ceil(5 * 9 + safety - 10))
        assert r["days_to_stockout"] == 2.0
        assert r["reorder_now"] is True

    def test_no_demand_no_order(self):
        r = fc.reorder_advice(d_hat=0.0, sigma=0.0, stock=6)
        assert r["order_qty"] == 0
        assert r["days_to_stockout"] is None
        assert r["reorder_now"] is False

    def test_overstocked_orders_zero(self):
        r = fc.reorder_advice(d_hat=1.0, sigma=0.0, stock=500)
        assert r["order_qty"] == 0


class TestForecastSeries:

    def test_short_history_falls_back_to_mean(self):
        r = fc.forecast_series(np.array([2.0, 4.0, 6.0]), stock=10)
        assert r["method"] == "mean"
        assert r["forecast"] == pytest.approx(4.0)
        assert r["alpha"] is None

    def test_long_history_uses_ses_and_scores_baselines(self):
        rng = np.random.default_rng(0)
        y = rng.poisson(6, 60).astype(float)
        r = fc.forecast_series(y, stock=20)
        assert r["method"] == "ses"
        assert r["alpha"] in (0.1, 0.2, 0.3, 0.5, 0.7)
        for k in ("mae_ses", "mae_naive", "mae_ma"):
            assert r[k] is not None and r[k] >= 0
        # on stationary Poisson noise SES should beat the naive forecast
        assert r["mae_ses"] <= r["mae_naive"]


class TestDatabaseIntegration:

    def test_daily_demand_excludes_today_and_fills_gaps(self, db):
        p = db.get_all_products()[0]
        today = date(2026, 3, 10)
        with db.get_conn() as c:
            rows = [
                ("2026-03-07T10:00:00", "pick"), ("2026-03-07T11:00:00", "pick"),
                ("2026-03-07T12:00:00", "return"),
                ("2026-03-09T10:00:00", "pick"),
                ("2026-03-10T09:00:00", "pick"),          # today → excluded
            ]
            for ts, act in rows:
                c.execute("INSERT INTO transactions (timestamp, product_id, action, session_id)"
                          " VALUES (?, ?, ?, 's')", (ts, p["product_id"], act))
        days, y = fc.daily_demand(p["product_id"], today=today)
        assert days == ["2026-03-07", "2026-03-08", "2026-03-09"]
        assert list(y) == [1.0, 0.0, 1.0]

    def test_anomalies_are_not_demand(self, db):
        p = db.get_all_products()[0]
        db.record_anomaly(p["name"], "s")
        _, y = fc.daily_demand(p["product_id"], today=date.today() + timedelta(days=1))
        assert y.sum() == 0

    def test_seed_forecast_and_clear(self, db):
        from src.data_tools import seed_history
        stock_before = {p["name"]: p["stock"] for p in db.get_all_products()}
        n = seed_history.seed(days=30, rng_seed=1)
        assert n > 0
        # stock is untouched by simulated history
        assert {p["name"]: p["stock"] for p in db.get_all_products()} == stock_before
        rows = fc.forecast_all()
        assert len(rows) == len(stock_before)
        assert all(r["history_days"] == 30 and r["method"] == "ses" for r in rows)
        assert seed_history.clear_simulated() == n
        assert all(r["history_days"] == 0 for r in fc.forecast_all())

    def test_seed_is_reproducible(self, db):
        from src.data_tools import seed_history
        a = seed_history.seed(days=20, rng_seed=5, today=date(2026, 1, 1))
        fa = [r["forecast"] for r in fc.forecast_all(today=date(2026, 1, 1))]
        b = seed_history.seed(days=20, rng_seed=5, today=date(2026, 1, 1))
        fb = [r["forecast"] for r in fc.forecast_all(today=date(2026, 1, 1))]
        assert a == b and fa == fb
