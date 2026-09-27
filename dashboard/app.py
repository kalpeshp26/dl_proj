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
from src.config import API_BASE_URL, DEFAULT_SESSION_ID

BASE_URL = API_BASE_URL
REFRESH_MS = 300

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepRetail — Autonomous Checkout",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─── Helpers ──────────────────────────────────────────────────────────────────
_session = requests.Session()


def api_get(endpoint: str, default=None):
    try:
        r = _session.get(f"{BASE_URL}{endpoint}", timeout=0.8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return default


def api_post(endpoint: str) -> bool:
    try:
        r = _session.post(f"{BASE_URL}{endpoint}", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def get_annotated_frame() -> bytes | None:
    """Latest annotated JPEG pushed by the pipeline to the backend (None if not running)."""
    try:
        r = _session.get(f"{BASE_URL}/frame/latest", timeout=0.8)
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:
        pass
    return None


backend_status = api_get("/pipeline/status", None)
backend_up = backend_status is not None
backend_status = backend_status or {}

# Default the cart view to whatever session the pipeline is writing to
if "session_id" not in st.session_state:
    sid = backend_status.get("session_id")
    st.session_state.session_id = sid if sid and sid not in ("none", "unknown") else DEFAULT_SESSION_ID


# ─── Sidebar: session controls ────────────────────────────────────────────────
with st.sidebar:
    st.header("System & Zones")
    session_id = st.text_input("Session ID", value=st.session_state.get("session_id", DEFAULT_SESSION_ID),
                               help="Must match the pipeline's --session (default: demo_session)")
    st.session_state.session_id = session_id

    # Bag placement zone selector
    from src.config import load_zones, save_zones
    current_zones = load_zones()
    current_basket = current_zones.get("basket_roi", [330, 20, 620, 460])
    
    zone_presets = {
        "Right Side (Standard)": {
            "shelf_roi": [20, 20, 310, 460],
            "basket_roi": [330, 20, 620, 460],
            "shelf_slots": {},
        },
        "Left Side": {
            "shelf_roi": [330, 20, 620, 460],
            "basket_roi": [20, 20, 310, 460],
            "shelf_slots": {},
        },
        "Bottom Half": {
            "shelf_roi": [20, 20, 620, 240],
            "basket_roi": [20, 250, 620, 460],
            "shelf_slots": {},
        },
    }

    # Determine default index
    default_idx = 0
    if current_basket[0] < 50:
        default_idx = 1 if current_basket[1] < 100 else 2

    chosen_preset = st.selectbox(
        "🛍️ Shopping Bag Position on Camera",
        list(zone_presets.keys()),
        index=default_idx,
        help="Choose which side of your camera view your shopping bag is placed."
    )
    if st.button("Apply Bag Position"):
        save_zones(zone_presets[chosen_preset])
        st.success(f"Bag position updated to: {chosen_preset}")

    st.markdown("---")
    if st.button("🗑️ Clear Alerts"):
        api_post("/alerts/clear")
    if st.button("♻️ Reset Inventory & Carts",
                 help="Restores stock from data/inventory.csv — use between demo runs"):
        api_post("/admin/reset")
        st.session_state.pop("receipt", None)
        st.success("Inventory restored to initial stock!")

    st.markdown("---")
    is_running = backend_status.get("running", False)
    if not backend_up:
        st.markdown("🔴 Backend: OFFLINE")
        st.caption("Start it: `uvicorn src.backend.main:app --port 8000`")
    else:
        st.markdown(f"{'🟢' if is_running else '🔴'} Pipeline: {'RUNNING' if is_running else 'STOPPED'}")
        st.caption(f"FPS: {backend_status.get('fps', 0):.1f} | "
                   f"Frame: {backend_status.get('frame_id', 0)} | "
                   f"Tracks: {backend_status.get('active_tracks', 0)}")
        pipe_session = backend_status.get("session_id")
        if is_running and pipe_session and pipe_session != session_id:
            st.warning(f"Pipeline is writing to session '{pipe_session}'")

    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_interval = st.slider("Refresh (ms)", 200, 2000, REFRESH_MS, step=100)


# ─── Main layout ──────────────────────────────────────────────────────────────
st.title("🛒 DeepRetail — Autonomous Checkout")

col_left, col_right = st.columns([3, 2])

# ────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN: Live camera feed
# ────────────────────────────────────────────────────────────────────────────
with col_left:
    st.subheader("📷 Live Camera Feed")
    feed_placeholder = st.empty()

    frame_jpeg = get_annotated_frame()
    if frame_jpeg is not None:
        import base64
        b64 = base64.b64encode(frame_jpeg).decode()
        feed_placeholder.markdown(
            f'<img src="data:image/jpeg;base64,{b64}" style="width:100%; border-radius:8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);" />',
            unsafe_allow_html=True,
        )
    else:
        feed_placeholder.info(
            "📡 Waiting for pipeline frame...\n\n"
            "Live camera is initializing. Please wait a moment."
        )

# ────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN: Cart + Checkout + Inventory
# ────────────────────────────────────────────────────────────────────────────
with col_right:

    # ── Cart / Live Bill ─────────────────────────────────────────
    st.subheader("🧾 Current Bill & Cart")
    cart_data = api_get(f"/cart/{session_id}", {"items": [], "total": 0.0})
    items = cart_data.get("items", []) if cart_data else []
    total = cart_data.get("total", 0.0) if cart_data else 0.0

    if items:
        import pandas as pd
        cart_rows = [{
            "Item": it["name"],
            "Qty": it["quantity"],
            "Price": f"₹{it['unit_price']:.2f}",
            "Subtotal": f"₹{it['subtotal']:.2f}",
        } for it in items]
        st.dataframe(pd.DataFrame(cart_rows), width="stretch", hide_index=True)

        tax = round(total * 0.05, 2)
        grand_total = round(total + tax, 2)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Items Total:** ₹{total:.2f}")
            st.markdown(f"**GST (5%):** ₹{tax:.2f}")
        with c2:
            st.markdown(f"### Grand Total: ₹{grand_total:.2f}")

        col_pay, col_new = st.columns([3, 2])
        with col_pay:
            if st.button("💳 Checkout & Print Bill", type="primary", key="checkout_btn"):
                st.session_state.receipt = {
                    "session_id": session_id,
                    "items": cart_rows,
                    "total": total,
                    "tax": tax,
                    "grand_total": grand_total,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
        with col_new:
            if st.button("🔄 New Customer"):
                new_sid = f"cust_{int(time.time()) % 10000}"
                st.session_state.session_id = new_sid
                st.session_state.pop("receipt", None)
                st.rerun()

    else:
        st.info("🛍️ **Your shopping bag is empty.**\n\nPut items into the **Shopping Bag** area on the camera to add them to your bill. Remove them from the bag to cancel.")

    # Show Final Receipt if checked out
    if "receipt" in st.session_state:
        rc = st.session_state.receipt
        st.success("✅ **PAYMENT COMPLETED — AUTONOMOUS CHECKOUT**")
        st.markdown(f"""
        ```text
        ==================================================
                    DEEPRETAIL OFFICIAL RECEIPT
        Invoice #: INV-{rc['session_id'][-6:].upper()}
        Date/Time: {rc['time']}
        --------------------------------------------------
        Item                      Qty    Price     Subtotal
        --------------------------------------------------
        """ + "\n".join([
            f"{it['Item']:<24}  {it['Qty']:>3}   {it['Price']:>7}  {it['Subtotal']:>10}"
            for it in rc['items']
        ]) + f"""
        --------------------------------------------------
        Subtotal:                                ₹{rc['total']:>8.2f}
        Taxes (5% GST):                          ₹{rc['tax']:>8.2f}
        TOTAL PAID:                              ₹{rc['grand_total']:>8.2f}
        ==================================================
        Thank you for shopping with DeepRetail!
        ```
        """)

    st.markdown("---")

    # ── AI Activity Log ──────────────────────────────────────────
    st.subheader("🤖 Live Activity")
    events_data = api_get(f"/events?limit=8&session_id={session_id}", {"events": []})
    events = events_data.get("events", []) if events_data else []

    if events:
        for ev in events[:6]:
            action = ev.get("action", ev.get("event", "?"))
            product = ev.get("product_name", ev.get("product", "?"))
            ts = ev.get("timestamp", "")
            if isinstance(ts, str) and "T" in ts:
                ts_disp = ts.split("T")[-1][:8]
            else:
                ts_disp = str(ts)[:8]

            icon = {"pick": "🟢 Added", "return": "🔵 Removed", "anomaly": "🔴 Alert"}.get(action, "⚪")
            st.markdown(f"`{ts_disp}` **{icon}** — {product}")
    else:
        st.caption("No movement detected yet this session.")

    st.markdown("---")

    # ── Store Inventory ──────────────────────────────────────────
    st.subheader("📦 Live Store Inventory")
    inventory_data = api_get("/inventory", {"products": [], "low_stock": []})
    if inventory_data and inventory_data.get("products"):
        import pandas as pd
        products = inventory_data["products"]
        low_names = {p["name"] for p in inventory_data.get("low_stock", [])}

        df = pd.DataFrame([{
            "Product": p["name"],
            "In Stock": p["stock"],
            "Price": f"₹{p['price']:.2f}",
            "Status": "⚠️ Low Stock" if p["name"] in low_names else "🟢 In Stock",
        } for p in products])

        st.dataframe(
            df,
            width="stretch",
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
    tab1, tab2, tab3 = st.tabs(["Top Products", "Recommendations", "📈 Demand Forecast"])

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

    with tab3:
        import pandas as pd
        fc_data = api_get("/forecast", {"products": []}) or {"products": []}
        fc_rows = fc_data.get("products", [])
        if not fc_rows or all(r["history_days"] == 0 for r in fc_rows):
            st.caption("No sales history yet. For a demo run: "
                       "`python src/data_tools/seed_history.py` (simulated, removable with --clear).")
        else:
            df_fc = pd.DataFrame([{
                "Product": r["name"],
                "Forecast / day": r["forecast"],
                "Stock": r["stock"],
                "Days left": r["days_to_stockout"],
                "Reorder point": r["reorder_point"],
                "Order qty": r["order_qty"],
                "Reorder?": "🔴 now" if r["reorder_now"] else "🟢 ok",
                "α": r["alpha"],
                "MAE SES / naive / 7d": (f"{r['mae_ses']} / {r['mae_naive']} / {r['mae_ma']}"
                                         if r["mae_ses"] is not None else "—"),
            } for r in fc_rows])
            st.dataframe(df_fc, hide_index=True, use_container_width=True)
            pick = st.selectbox("History + forecast for:", [r["name"] for r in fc_rows], key="fc_pick")
            hist = api_get(f"/forecast/{pick}/history", None)
            if hist and hist.get("days"):
                series = pd.concat([
                    pd.DataFrame({"day": hist["days"], "Actual": hist["demand"]}),
                    pd.DataFrame({"day": hist["forecast_days"], "Forecast": hist["forecast"]}),
                ]).set_index("day")
                st.line_chart(series)

# ─── Auto-refresh ─────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(refresh_interval / 1000.0)
    st.rerun()
