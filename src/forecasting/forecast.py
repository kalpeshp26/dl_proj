"""
DeepRetail — Demand Forecasting (Stage 8)

Turns the pick/return transactions recorded by the vision pipeline into a
daily demand forecast and replenishment advice per product.

Method
    y_t      daily net demand = picks − returns on day t (clipped at 0);
             the current, incomplete day is excluded.
    SES      l_t = α·y_t + (1 − α)·l_{t−1},  forecast d̂ = l_T
    α        chosen from FORECAST_ALPHAS by a rolling-origin, one-step-ahead
             backtest over the last FORECAST_BACKTEST_DAYS days (min MAE).
    baselines naive (yesterday) and 7-day moving average, scored on the
             same backtest.
    ROP      d̂·L + z·σ·√L          (σ = std of the SES backtest errors)
    Q        max(0, ⌈d̂·(L+H) + z·σ·√L − S⌉)
    cover    S / d̂  days until stock-out

With fewer than FORECAST_MIN_HISTORY_DAYS days of history the forecast
falls back to the historical mean (σ = std of the history).

Only NumPy + sqlite — no torch — so it also runs in the cloud build.

CLI:
    python -m src.forecasting.forecast            # print the forecast table
"""

from __future__ import annotations

import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    FORECAST_ALPHAS, FORECAST_BACKTEST_DAYS, FORECAST_MIN_HISTORY_DAYS,
    FORECAST_LEAD_TIME_DAYS, FORECAST_HORIZON_DAYS, FORECAST_SERVICE_Z,
    FORECAST_MA_WINDOW,
)


# ─── Core maths (pure functions, unit-tested) ────────────────────────────────

def ses_level(y: np.ndarray, alpha: float) -> float:
    """Final SES level after running over y. Level is initialised to y[0]."""
    if len(y) == 0:
        return 0.0
    level = float(y[0])
    for v in y[1:]:
        level = alpha * float(v) + (1.0 - alpha) * level
    return level


def _one_step_errors(y: np.ndarray, predict, window: int) -> np.ndarray:
    """Rolling-origin backtest: for each of the last `window` days t, fit on
    y[:t] and forecast y[t]. Returns the error vector (actual − forecast)."""
    n = len(y)
    start = max(1, n - window)
    errs = [float(y[t]) - float(predict(y[:t])) for t in range(start, n)]
    return np.asarray(errs, dtype=float)


def naive_forecast(hist: np.ndarray) -> float:
    return float(hist[-1]) if len(hist) else 0.0


def moving_average_forecast(hist: np.ndarray, window: int = FORECAST_MA_WINDOW) -> float:
    return float(np.mean(hist[-window:])) if len(hist) else 0.0


def select_alpha(y: np.ndarray, alphas=FORECAST_ALPHAS,
                 window: int = FORECAST_BACKTEST_DAYS) -> tuple[float, float, np.ndarray]:
    """Return (best_alpha, best_mae, best_errors). Ties go to the smaller α."""
    best = None
    for a in alphas:
        errs = _one_step_errors(y, lambda h, a=a: ses_level(h, a), window)
        mae = float(np.mean(np.abs(errs))) if len(errs) else math.inf
        if best is None or mae < best[1] - 1e-12:
            best = (a, mae, errs)
    return best


def reorder_advice(d_hat: float, sigma: float, stock: int,
                   lead: int = FORECAST_LEAD_TIME_DAYS,
                   horizon: int = FORECAST_HORIZON_DAYS,
                   z: float = FORECAST_SERVICE_Z) -> dict:
    safety = z * sigma * math.sqrt(lead)
    rop = d_hat * lead + safety
    order_qty = max(0, math.ceil(d_hat * (lead + horizon) + safety - stock))
    days_left = (stock / d_hat) if d_hat > 1e-9 else None
    return {
        "safety_stock": round(safety, 2),
        "reorder_point": round(rop, 2),
        "order_qty": int(order_qty),
        "days_to_stockout": round(days_left, 1) if days_left is not None else None,
        "reorder_now": bool(stock <= rop),
    }


def forecast_series(y: np.ndarray, stock: int) -> dict:
    """Forecast one product from its daily demand history y."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < FORECAST_MIN_HISTORY_DAYS:
        d_hat = float(np.mean(y)) if n else 0.0
        sigma = float(np.std(y)) if n > 1 else 0.0
        out = {"method": "mean", "alpha": None, "forecast": round(d_hat, 2),
               "sigma": round(sigma, 2), "history_days": n,
               "mae_ses": None, "mae_naive": None, "mae_ma": None}
    else:
        alpha, mae, errs = select_alpha(y)
        d_hat = ses_level(y, alpha)
        sigma = float(np.std(errs)) if len(errs) > 1 else 0.0
        e_naive = _one_step_errors(y, naive_forecast, FORECAST_BACKTEST_DAYS)
        e_ma = _one_step_errors(y, moving_average_forecast, FORECAST_BACKTEST_DAYS)
        out = {"method": "ses", "alpha": alpha, "forecast": round(d_hat, 2),
               "sigma": round(sigma, 2), "history_days": n,
               "mae_ses": round(mae, 2),
               "mae_naive": round(float(np.mean(np.abs(e_naive))), 2),
               "mae_ma": round(float(np.mean(np.abs(e_ma))), 2)}
    out.update(reorder_advice(d_hat, sigma, stock))
    return out


# ─── Database access ─────────────────────────────────────────────────────────

def daily_demand(product_id: int, today: Optional[date] = None,
                 conn=None) -> tuple[list[str], np.ndarray]:
    """Daily net demand for one product from its first transaction up to
    yesterday (today's partial day is excluded). Missing days count as 0."""
    from src.backend.db import get_conn
    today = today or date.today()
    q = """
        SELECT substr(timestamp, 1, 10) AS day,
               SUM(CASE WHEN action = 'pick' THEN 1
                        WHEN action = 'return' THEN -1 ELSE 0 END) AS net
        FROM transactions
        WHERE product_id = ? AND action IN ('pick', 'return')
        GROUP BY day ORDER BY day
    """
    if conn is None:
        with get_conn() as c:
            rows = c.execute(q, (product_id,)).fetchall()
    else:
        rows = conn.execute(q, (product_id,)).fetchall()

    by_day = {}
    for r in rows:
        try:
            d = datetime.strptime(r["day"], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if d < today:
            by_day[d] = max(0, int(r["net"] or 0))
    if not by_day:
        return [], np.zeros(0)
    first, last = min(by_day), today - timedelta(days=1)
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    return [d.isoformat() for d in days], np.array([by_day.get(d, 0) for d in days], dtype=float)


def forecast_all(today: Optional[date] = None) -> list[dict]:
    """Forecast + reorder advice for every product in the catalog."""
    from src.backend.db import get_conn
    results = []
    with get_conn() as conn:
        products = [dict(r) for r in conn.execute("SELECT * FROM products ORDER BY name")]
        for p in products:
            _, y = daily_demand(p["product_id"], today=today, conn=conn)
            r = forecast_series(y, int(p["stock"]))
            r.update({"product_id": p["product_id"], "name": p["name"],
                      "stock": int(p["stock"])})
            results.append(r)
    return results


def product_history(name: str, today: Optional[date] = None) -> Optional[dict]:
    """Daily history of one product plus its flat SES forecast for the horizon."""
    from src.backend.db import get_product_by_name
    p = get_product_by_name(name)
    if p is None:
        return None
    today = today or date.today()
    days, y = daily_demand(p["product_id"], today=today)
    fc = forecast_series(y, int(p["stock"]))
    future = [(today + timedelta(days=i)).isoformat() for i in range(FORECAST_HORIZON_DAYS)]
    return {"name": name, "days": days, "demand": [float(v) for v in y],
            "forecast_days": future, "forecast": [fc["forecast"]] * len(future),
            "summary": fc}


if __name__ == "__main__":
    rows = forecast_all()
    hdr = f"{'product':<20}{'hist':>5}{'α':>6}{'d̂/day':>8}{'MAE ses':>9}{'naive':>7}{'7d-MA':>7}{'stock':>6}{'ROP':>7}{'order':>6}{'days':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        fmt = lambda v: "-" if v is None else v
        print(f"{r['name']:<20}{r['history_days']:>5}{fmt(r['alpha']):>6}{r['forecast']:>8}"
              f"{fmt(r['mae_ses']):>9}{fmt(r['mae_naive']):>7}{fmt(r['mae_ma']):>7}"
              f"{r['stock']:>6}{r['reorder_point']:>7}{r['order_qty']:>6}{fmt(r['days_to_stockout']):>7}")
