"""
DeepRetail — Seed simulated sales history for the forecasting demo (Stage 8)

A live demo only produces a few minutes of transactions, which is not enough
to forecast from. This script inserts N days of *simulated* pick/return rows
so the forecast panel has something to work with.

    demand_t ~ Poisson( base · (1 + trend·t/N) · weekly[t mod 7] )
    returns  ~ Binomial(demand_t, 0.05)

Every simulated row has a session_id starting with "sim_" (SIM_SESSION_PREFIX),
so it can be told apart from real data and removed with --clear. Product
stock is NOT changed.

Usage:
    python src/data_tools/seed_history.py                 # 60 days, seed 42
    python src/data_tools/seed_history.py --days 90 --seed 7
    python src/data_tools/seed_history.py --clear         # remove simulated rows
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.backend.db import get_conn, init_db, _lock
from src.config import SIM_SESSION_PREFIX

# Mean units/day per product (anything not listed gets DEFAULT_BASE).
BASE_DEMAND = {
    "chings_manchurian": 5.0,
    "chings_hakka": 7.0,
    "homelite_matchbox": 9.0,
    "vaseline_jelly": 3.0,
}
DEFAULT_BASE = 4.0
TREND = 0.15                                   # +15% over the whole period
WEEKLY = [0.9, 0.85, 0.9, 1.0, 1.1, 1.35, 1.25]  # Mon … Sun
RETURN_RATE = 0.05
SHOPPERS_PER_DAY = 12


def clear_simulated() -> int:
    with _lock, get_conn() as conn:
        cur = conn.execute("DELETE FROM transactions WHERE session_id LIKE ?",
                           (SIM_SESSION_PREFIX + "%",))
        return cur.rowcount


def seed(days: int = 60, rng_seed: int = 42, today: date | None = None) -> int:
    """Insert `days` days of simulated history ending yesterday. Returns rows inserted."""
    init_db()
    clear_simulated()
    rng = np.random.default_rng(rng_seed)
    today = today or date.today()
    rows: list[tuple] = []

    with get_conn() as conn:
        products = [dict(r) for r in conn.execute("SELECT product_id, name FROM products")]

    for i in range(days):
        d = today - timedelta(days=days - i)
        season = WEEKLY[d.weekday()]
        for p in products:
            base = BASE_DEMAND.get(p["name"], DEFAULT_BASE)
            lam = base * (1 + TREND * i / max(days - 1, 1)) * season
            n_pick = int(rng.poisson(lam))
            n_ret = int(rng.binomial(n_pick, RETURN_RATE)) if n_pick else 0
            for k in range(n_pick + n_ret):
                action = "pick" if k < n_pick else "return"
                ts = datetime(d.year, d.month, d.day, int(rng.integers(9, 22)),
                              int(rng.integers(0, 60)), int(rng.integers(0, 60)))
                sid = f"{SIM_SESSION_PREFIX}{d:%Y%m%d}_{int(rng.integers(SHOPPERS_PER_DAY)):02d}"
                rows.append((ts.strftime("%Y-%m-%dT%H:%M:%S"), p["product_id"], action, sid))

    with _lock, get_conn() as conn:
        conn.executemany(
            "INSERT INTO transactions (timestamp, product_id, action, session_id) VALUES (?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Seed simulated sales history (Stage 8 demo)")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--clear", action="store_true", help="remove simulated rows and exit")
    args = ap.parse_args()
    if args.clear:
        print(f"[seed_history] removed {clear_simulated()} simulated rows")
        return
    n = seed(args.days, args.seed)
    print(f"[seed_history] inserted {n} simulated rows over {args.days} days "
          f"(session_id '{SIM_SESSION_PREFIX}*'; stock unchanged)")


if __name__ == "__main__":
    main()
