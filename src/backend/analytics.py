"""
DeepRetail — Recommendation & Analytics Engine

No DL required. Uses simple co-occurrence counting over transaction history.

Implemented:
    - Most-purchased products (GROUP BY + COUNT)
    - Frequently-bought-together (pairwise co-occurrence per session)
    - Product recommendations for a given product_id
"""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.backend.db import get_conn


# ─── Top products ─────────────────────────────────────────────────────────────

def get_top_products(limit: int = 10) -> list[dict]:
    """
    Most-purchased products across all sessions.
    Returns: [{product_id, name, pick_count}, ...]
    """
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT t.product_id, p.name,
                   COUNT(*) AS pick_count
            FROM transactions t
            JOIN products p ON t.product_id = p.product_id
            WHERE t.action = 'pick'
            GROUP BY t.product_id
            ORDER BY pick_count DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


# ─── Co-occurrence matrix ─────────────────────────────────────────────────────

def _build_cooccurrence() -> dict[int, dict[int, int]]:
    """
    Build a pairwise co-occurrence count from transaction history.
    Two products co-occur if they both appear as 'pick' in the same session.
    """
    # Group picks by session
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT session_id, product_id
            FROM transactions
            WHERE action = 'pick'
            ORDER BY session_id
            """
        ).fetchall()

    sessions: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        sessions[row["session_id"]].add(row["product_id"])

    cooc: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for product_ids in sessions.values():
        product_list = sorted(product_ids)
        for i in range(len(product_list)):
            for j in range(i + 1, len(product_list)):
                a, b = product_list[i], product_list[j]
                cooc[a][b] += 1
                cooc[b][a] += 1

    return cooc


def get_recommendations(product_id: int, top_n: int = 5) -> list[dict]:
    """
    Given a product_id, return the top_n most co-purchased products.
    Returns: [{product_id, name, co_occurrence_count, confidence}, ...]

    confidence = co_occurrence_count / total_picks_of_input_product
    """
    cooc = _build_cooccurrence()

    # Total picks for the query product (denominator for confidence)
    with get_conn() as conn:
        total_row = conn.execute(
            """
            SELECT COUNT(*) AS cnt FROM transactions
            WHERE product_id = ? AND action = 'pick'
            """,
            (product_id,),
        ).fetchone()
        total_picks = total_row["cnt"] if total_row else 1

    related = cooc.get(product_id, {})
    if not related:
        return []

    # Sort by co-occurrence count descending
    sorted_related = sorted(related.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Fetch product names
    recs = []
    with get_conn() as conn:
        for pid, count in sorted_related:
            row = conn.execute(
                "SELECT product_id, name FROM products WHERE product_id = ?",
                (pid,),
            ).fetchone()
            if row:
                recs.append({
                    "product_id": row["product_id"],
                    "name": row["name"],
                    "co_occurrence_count": count,
                    "confidence": round(count / max(total_picks, 1), 4),
                })
    return recs


# ─── CLI smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.backend.db import init_db

    init_db()
    print("Top products:")
    for p in get_top_products():
        print(f"  {p}")

    print("\nRecommendations for product_id=1:")
    for r in get_recommendations(1):
        print(f"  {r}")
