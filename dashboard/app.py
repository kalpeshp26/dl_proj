"""
DeepRetail — Streamlit Dashboard (Stage 7)

Layout:
    LEFT    : live annotated camera feed
    TOP-RIGHT : cart + total
    MID-RIGHT : AI activity log / recent events
    BOT-RIGHT : inventory / store status
    BOTTOM   : alerts banner (low-stock, misplaced, anomaly, unknown)

The dashboard reads from the FastAPI backend (HTTP polling + optional WS).
Refresh interval: ~300ms via st.rerun() loop.

Run:
    streamlit run dashboard/app.py

Requires backend to be running:
    uvicorn src.backend.main:app --host 127.0.0.1 --port 8000
"""

import sys
import time
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import API_HOST, API_PORT

BASE_URL = f"http://{API_HOST}:{API_PORT}"
REFRESH_MS = 300

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepRetail — Autonomous Checkout",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def api_get(endpoint: str, default=None):
    try:
        r = requests.get(f"{BASE_URL}{endpoint}", timeout=1.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return default


def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = "demo_session"
    return st.session_state.session_id


# ─── Try import pipeline state (if running in same process) ──────────────────
_pipeline_state = None
try:
    from src.pipeline.run_pipeline import pipeline_state as _ps
    _pipeline_state = _ps
except Exception:
    pass


def get_annotated_frame():
    """Get latest annotated frame from pipeline state or None."""
    if _pipeline_state and _pipeline_state.annotated_frame is not None:
        return _pipeline_state.annotated_frame
    return None


# ─── Sidebar: session controls ────────────────────────────────────────────────
with st.sidebar:
    st.header("Session Controls")
    session_id = st.text_input("Session ID", value=get_session_id())
    st.session_state.session_id = session_id

    if st.button("🔄 New Session"):
        st.session_state.session_id = f"session_{int(time.time())}"
        st.rerun()

    if st.button("🗑️ Clear Alerts"):
        api_get("/alerts/clear")

    st.markdown("---")
    backend_status = api_get("/pipeline/status", {})
    is_running = backend_status.get("running", False) if backend_status else False
    status_color = "🟢" if is_running else "🔴"
    st.markdown(f"{status_color} Pipeline: {'RUNNING' if is_running else 'STOPPED'}")
    if backend_status:
        st.caption(f"FPS: {backend_status.get('fps', 0):.1f} | "
                   f"Frame: {backend_status.get('frame_id', 0)} | "
                   f"Tracks: {backend_status.get('active_tracks', 0)}")

    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_interval = st.slider("Refresh (ms)", 200, 2000, REFRESH_MS, step=100)


# ─── Main layout ──────────────────────────────────────────────────────────────
st.title("🛒 DeepRetail — Autonomous Checkout Intelligence")

col_left, col_right = st.columns([3, 2])

# ────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN: Live camera feed
# ────────────────────────────────────────────────────────────────────────────
with col_left:
    st.subheader("📷 Live Feed")
    feed_placeholder = st.empty()

    frame = get_annotated_frame()
    if frame is not None:
        import cv2
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        feed_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
    else:
        feed_placeholder.info(
            "📡 Waiting for pipeline frame...\n\n"
            "Run: `python src/pipeline/run_pipeline.py --no-display`"
        )

# ────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN: Cart + Events + Inventory
# ────────────────────────────────────────────────────────────────────────────
with col_right:

    # ── Cart ─────────────────────────────────────────────────────
    st.subheader("🛒 Current Cart")
    cart_data = api_get(f"/cart/{session_id}", {"items": [], "total": 0.0})
    if cart_data and cart_data.get("items"):
        cart_container = st.container()
        with cart_container:
            for item in cart_data["items"]:
                st.markdown(
                    f"**{item['name']}** × {item['quantity']} "
                    f"— ₹{item['unit_price']:.2f} ea "
                    f"= **₹{item['subtotal']:.2f}**"
                )
        total = cart_data.get("total", 0.0)
        st.markdown(f"### Total: ₹{total:.2f}")
    else:
        st.info("Cart is empty. Pick items from shelf to begin.")

    st.markdown("---")

    # ── AI Activity Log ──────────────────────────────────────────
    st.subheader("🤖 AI Activity Log")
    events_data = api_get(f"/events?limit=10&session_id={session_id}", {"events": []})
    events = events_data.get("events", []) if events_data else []

    if events:
        for ev in events[:8]:
            action = ev.get("action", ev.get("event", "?"))
            product = ev.get("product_name", ev.get("product", "?"))
            ts = ev.get("timestamp", "")
            if isinstance(ts, str) and "T" in ts:
                ts_disp = ts.split("T")[-1][:8]
            else:
                ts_disp = str(ts)[:8]

            icon = {"pick": "🟢", "return": "🔵", "anomaly": "🔴"}.get(action, "⚪")
            st.markdown(f"{icon} `{ts_disp}` **{action.upper()}** — {product}")
    else:
        st.caption("No activity yet this session.")

    st.markdown("---")

    # ── Store Inventory ──────────────────────────────────────────
    st.subheader("📦 Store Inventory")
    inventory_data = api_get("/inventory", {"products": [], "low_stock": []})
    if inventory_data and inventory_data.get("products"):
        import pandas as pd
        products = inventory_data["products"]
        df = pd.DataFrame([{
            "Product": p["name"],
            "Stock": p["stock"],
            "Slot": p.get("shelf_slot", "-"),
            "Price (₹)": p["price"],
            "Min Stock": p["low_stock_threshold"],
        } for p in products])

        # Highlight low-stock rows
        low_names = {p["name"] for p in inventory_data.get("low_stock", [])}

        def highlight_low(row):
            if row["Product"] in low_names:
                return ["background-color: #ffcccc"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df.style.apply(highlight_low, axis=1),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Inventory not available.")


# ─── Alerts Banner (full-width bottom) ────────────────────────────────────────
st.markdown("---")
st.subheader("🚨 Alerts")

alerts_data = api_get("/alerts", {"alerts": []})
alerts = alerts_data.get("alerts", []) if alerts_data else []

if alerts:
    recent_alerts = alerts[-6:]
    cols = st.columns(min(len(recent_alerts), 3))
    alert_icons = {
        "LOW_STOCK": "📉",
        "MISPLACED_PRODUCT": "⚠️",
        "ANOMALY": "🔴",
        "UNKNOWN_PRODUCT": "❓",
    }
    for i, alert in enumerate(recent_alerts):
        col = cols[i % len(cols)]
        with col:
            icon = alert_icons.get(alert.get("type", ""), "⚠️")
            st.error(f"{icon} **{alert.get('type', 'ALERT')}**\n\n{alert.get('message', '')}")
else:
    st.success("✅ No active alerts. All systems normal.")

# ─── Recommendations (bottom) ─────────────────────────────────────────────────
with st.expander("💡 Recommendations & Analytics"):
    tab1, tab2 = st.tabs(["Top Products", "Recommendations"])

    with tab1:
        top_data = api_get("/analytics/top-products?limit=8", {"top_products": []})
        top_products = top_data.get("top_products", []) if top_data else []
        if top_products:
            import pandas as pd
            df_top = pd.DataFrame([{
                "Product": p["name"],
                "Times Purchased": p["pick_count"],
            } for p in top_products])
            st.bar_chart(df_top.set_index("Product"))
        else:
            st.caption("No purchase history yet.")

    with tab2:
        inventory_d = api_get("/inventory", {"products": []})
        product_names = [p["name"] for p in (inventory_d or {}).get("products", [])]
        if product_names:
            selected = st.selectbox("Select product:", product_names)
            # Get product_id
            all_prods = (inventory_d or {}).get("products", [])
            sel_id = next((p["product_id"] for p in all_prods if p["name"] == selected), None)
            if sel_id:
                recs_data = api_get(
                    f"/analytics/recommendations/{sel_id}",
                    {"recommendations": []}
                )
                recs = recs_data.get("recommendations", []) if recs_data else []
                if recs:
                    for r in recs:
                        st.markdown(
                            f"🔗 **{r['name']}** — "
                            f"bought together {r['co_occurrence_count']} times "
                            f"(confidence: {r['confidence']:.0%})"
                        )
                else:
                    st.caption("No co-purchase data yet.")
        else:
            st.caption("No products loaded.")

# ─── Auto-refresh ─────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_interval / 1000.0)
    st.rerun()
