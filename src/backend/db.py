"""
DeepRetail — SQLite Database Layer

Schema:
    products     — product catalog + current stock
    transactions — pick/return events (source of truth for inventory)

All mutations are wrapped in DB transactions to prevent race conditions.
"""

import json
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import DATABASE_PATH, PRODUCT_CLASSES, LOW_STOCK_DEFAULT_THRESHOLD


_lock = threading.Lock()   # serialize writes from pipeline + API


@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # allows concurrent reads during writes
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─── Schema ───────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS products (
    product_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    NOT NULL UNIQUE,
    category     TEXT    NOT NULL DEFAULT 'general',
    price        REAL    NOT NULL DEFAULT 0.0,
    stock        INTEGER NOT NULL DEFAULT 0,
    shelf_slot   TEXT,
    low_stock_threshold INTEGER NOT NULL DEFAULT 3
);

CREATE TABLE IF NOT EXISTS transactions (
    txn_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp  TEXT    NOT NULL,
    product_id INTEGER NOT NULL,
    action     TEXT    NOT NULL,   -- 'pick' | 'return' | 'anomaly'
    session_id TEXT    NOT NULL,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE INDEX IF NOT EXISTS idx_txn_session ON transactions(session_id);
CREATE INDEX IF NOT EXISTS idx_txn_product ON transactions(product_id);
CREATE INDEX IF NOT EXISTS idx_txn_ts      ON transactions(timestamp);
"""


def init_db() -> None:
    """Create tables and seed default products if empty."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript(DDL)

    # Seed product catalog from config if table is empty
    with get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        if count == 0:
            _seed_products(conn)


def _seed_products(conn: sqlite3.Connection) -> None:
    """Insert one row per product class from config."""
    default_prices = {
        "coke_500ml": 40.0,
        "pepsi_500ml": 38.0,
        "lays_classic": 20.0,
        "lays_spicy": 20.0,
        "oreo": 30.0,
        "kurkure": 20.0,
        "5star": 10.0,
        "kitkat": 30.0,
    }
    default_slots = {
        f: f"slot_{i}" for i, f in enumerate(PRODUCT_CLASSES)
    }
    for i, cls in enumerate(PRODUCT_CLASSES):
        conn.execute(
            """
            INSERT OR IGNORE INTO products
            (name, category, price, stock, shelf_slot, low_stock_threshold)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                cls,
                "snacks" if any(k in cls for k in ["lays", "kurkure", "oreo"]) else "beverages",
                default_prices.get(cls, 20.0),
                10,    # default initial stock
                default_slots.get(cls, f"slot_{i}"),
                LOW_STOCK_DEFAULT_THRESHOLD,
            ),
        )
    print(f"[DB] Seeded {len(PRODUCT_CLASSES)} products.")


# ─── Product queries ──────────────────────────────────────────────────────────

def get_all_products() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM products ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def get_product_by_name(name: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM products WHERE name = ?", (name,)
        ).fetchone()
    return dict(row) if row else None


def get_product_by_id(product_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM products WHERE product_id = ?", (product_id,)
        ).fetchone()
    return dict(row) if row else None


# ─── Stock mutations ──────────────────────────────────────────────────────────

def record_pick(product_name: str, session_id: str) -> Optional[int]:
    """
    Decrement stock and insert a 'pick' transaction.
    Returns txn_id or None if product not found.
    Atomic — uses a single DB transaction.
    """
    with _lock:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT product_id, stock FROM products WHERE name = ?",
                (product_name,)
            ).fetchone()
            if row is None:
                return None
            product_id, stock = row["product_id"], row["stock"]
            new_stock = max(0, stock - 1)
            conn.execute(
                "UPDATE products SET stock = ? WHERE product_id = ?",
                (new_stock, product_id),
            )
            cur = conn.execute(
                """
                INSERT INTO transactions (timestamp, product_id, action, session_id)
                VALUES (?, ?, 'pick', ?)
                """,
                (time.strftime("%Y-%m-%dT%H:%M:%S"), product_id, session_id),
            )
            return cur.lastrowid


def record_return(product_name: str, session_id: str) -> Optional[int]:
    """
    Increment stock and insert a 'return' transaction.
    Returns txn_id or None if product not found.
    """
    with _lock:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT product_id, stock FROM products WHERE name = ?",
                (product_name,)
            ).fetchone()
            if row is None:
                return None
            product_id, stock = row["product_id"], row["stock"]
            conn.execute(
                "UPDATE products SET stock = ? WHERE product_id = ?",
                (stock + 1, product_id),
            )
            cur = conn.execute(
                """
                INSERT INTO transactions (timestamp, product_id, action, session_id)
                VALUES (?, ?, 'return', ?)
                """,
                (time.strftime("%Y-%m-%dT%H:%M:%S"), product_id, session_id),
            )
            return cur.lastrowid


def record_anomaly(product_name: str, session_id: str) -> Optional[int]:
    """Log anomaly transaction (no stock change — product status unclear)."""
    with _lock:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT product_id FROM products WHERE name = ?", (product_name,)
            ).fetchone()
            product_id = row["product_id"] if row else None
            if product_id is None:
                # Log with product_id=0 for unknown products
                cur = conn.execute(
                    """
                    INSERT INTO transactions (timestamp, product_id, action, session_id)
                    VALUES (?, 0, 'anomaly', ?)
                    """,
                    (time.strftime("%Y-%m-%dT%H:%M:%S"), session_id),
                )
            else:
                cur = conn.execute(
                    """
                    INSERT INTO transactions (timestamp, product_id, action, session_id)
                    VALUES (?, ?, 'anomaly', ?)
                    """,
                    (time.strftime("%Y-%m-%dT%H:%M:%S"), product_id, session_id),
                )
            return cur.lastrowid


# ─── Cart / session queries ───────────────────────────────────────────────────

def get_cart(session_id: str) -> list[dict]:
    """
    Return the current virtual cart for a session.
    Cart = picks − returns for this session.
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT p.name, p.price,
                   SUM(CASE WHEN t.action='pick'   THEN 1 ELSE 0 END) AS picks,
                   SUM(CASE WHEN t.action='return' THEN 1 ELSE 0 END) AS returns
            FROM transactions t
            JOIN products p ON t.product_id = p.product_id
            WHERE t.session_id = ?
            GROUP BY t.product_id
            """,
            (session_id,),
        ).fetchall()

    cart = []
    total = 0.0
    for r in rows:
        qty = r["picks"] - r["returns"]
        if qty > 0:
            item = {
                "name": r["name"],
                "quantity": qty,
                "unit_price": r["price"],
                "subtotal": round(r["price"] * qty, 2),
            }
            cart.append(item)
            total += item["subtotal"]

    return cart


def get_cart_total(session_id: str) -> float:
    items = get_cart(session_id)
    return round(sum(i["subtotal"] for i in items), 2)


# ─── Low-stock check ──────────────────────────────────────────────────────────

def get_low_stock_products() -> list[dict]:
    """Return products at or below their low_stock_threshold."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM products
            WHERE stock <= low_stock_threshold
            ORDER BY stock ASC
            """
        ).fetchall()
    return [dict(r) for r in rows]


# ─── Transactions ─────────────────────────────────────────────────────────────

def get_transactions(
    limit: int = 100,
    session_id: Optional[str] = None,
    action: Optional[str] = None,
) -> list[dict]:
    query = """
        SELECT t.*, p.name AS product_name, p.price
        FROM transactions t
        LEFT JOIN products p ON t.product_id = p.product_id
    """
    params: list = []
    conditions: list[str] = []
    if session_id:
        conditions.append("t.session_id = ?")
        params.append(session_id)
    if action:
        conditions.append("t.action = ?")
        params.append(action)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY t.txn_id DESC LIMIT ?"
    params.append(limit)

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print("[DB] Initialized. Products:")
    for p in get_all_products():
        print(f"  {p['name']:20s}  stock={p['stock']}  slot={p['shelf_slot']}")
