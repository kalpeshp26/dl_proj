"""
DeepRetail — FastAPI Backend (Stage 6)

Runs alongside the pipeline (in the same process or as a separate server).
The pipeline writes to pipeline_state (shared in-process object).
The backend reads pipeline_state and the SQLite DB to serve the dashboard.

Endpoints:
    GET  /cart/{session_id}
    GET  /inventory
    GET  /alerts
    GET  /analytics/top-products
    GET  /analytics/recommendations/{product_id}
    GET  /pipeline/status
    GET  /events                (latest N events)
    WS   /ws/live               (real-time event stream to dashboard)

Start:
    uvicorn src.backend.main:app --host 127.0.0.1 --port 8000 --reload
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.backend.db import (
    init_db, get_all_products, get_cart, get_cart_total,
    get_low_stock_products, get_transactions,
    record_pick, record_return, record_anomaly,
)
from src.backend.analytics import get_top_products, get_recommendations
from src.backend.schemas import (
    CartResponse, CartItem, InventoryResponse, ProductInfo,
    AlertsResponse, AlertItem, EventItem,
    AnalyticsTopResponse, TopProductItem,
    AnalyticsRecsResponse, RecommendationItem,
    PipelineStatusResponse,
)
from src.config import LOG_ALERTS

# Import shared pipeline state (None if pipeline not running in same process)
try:
    from src.pipeline.run_pipeline import pipeline_state
except Exception:
    pipeline_state = None  # type: ignore

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    print("[Backend] Database initialized.")
    yield

app = FastAPI(
    title="DeepRetail API",
    description="Autonomous checkout and retail intelligence backend",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory alert buffer (appended by pipeline callbacks)
_alerts: list[dict] = []
# WebSocket connection manager
_ws_clients: list[WebSocket] = []



# ─── Cart ─────────────────────────────────────────────────────────────────────

@app.get("/cart/{session_id}", response_model=CartResponse)
def get_cart_endpoint(session_id: str):
    items_raw = get_cart(session_id)
    items = [CartItem(**i) for i in items_raw]
    total = round(sum(i.subtotal for i in items), 2)
    return CartResponse(session_id=session_id, items=items, total=total)


# ─── Inventory ────────────────────────────────────────────────────────────────

@app.get("/inventory", response_model=InventoryResponse)
def get_inventory():
    all_products = [ProductInfo(**p) for p in get_all_products()]
    low_stock = [ProductInfo(**p) for p in get_low_stock_products()]
    return InventoryResponse(products=all_products, low_stock=low_stock)


# ─── Alerts ───────────────────────────────────────────────────────────────────

@app.get("/alerts", response_model=AlertsResponse)
def get_alerts():
    return AlertsResponse(alerts=[AlertItem(**a) for a in _alerts[-50:]])


@app.post("/alerts/clear")
def clear_alerts():
    _alerts.clear()
    return {"status": "cleared"}


# ─── Events ───────────────────────────────────────────────────────────────────

@app.get("/events")
def get_events(limit: int = 50, session_id: Optional[str] = None):
    txns = get_transactions(limit=limit, session_id=session_id)
    return {"events": txns}


# ─── Analytics ────────────────────────────────────────────────────────────────

@app.get("/analytics/top-products", response_model=AnalyticsTopResponse)
def top_products_endpoint(limit: int = 10):
    results = get_top_products(limit=limit)
    return AnalyticsTopResponse(
        top_products=[TopProductItem(**r) for r in results]
    )


@app.get("/analytics/recommendations/{product_id}", response_model=AnalyticsRecsResponse)
def recommendations_endpoint(product_id: int, top_n: int = 5):
    from src.backend.db import get_product_by_id
    product = get_product_by_id(product_id)
    if not product:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
    recs = get_recommendations(product_id=product_id, top_n=top_n)
    return AnalyticsRecsResponse(
        product_id=product_id,
        product_name=product["name"],
        recommendations=[RecommendationItem(**r) for r in recs],
    )


# ─── Pipeline status ──────────────────────────────────────────────────────────

@app.get("/pipeline/status", response_model=PipelineStatusResponse)
def pipeline_status():
    if pipeline_state:
        return PipelineStatusResponse(
            running=pipeline_state.running,
            fps=pipeline_state.fps,
            frame_id=pipeline_state.frame_id,
            active_tracks=len(pipeline_state.tracks),
            session_id=getattr(pipeline_state, "session_id", "unknown"),
        )
    return PipelineStatusResponse(
        running=False, fps=0.0, frame_id=0, active_tracks=0, session_id="none"
    )


# ─── Internal endpoints (called by pipeline) ─────────────────────────────────

@app.post("/internal/event")
async def receive_pipeline_event(event: dict):
    """
    Called by the pipeline (direct function call or HTTP POST) to
    record a pick/return/anomaly event in the DB and push to WS clients.
    """
    action = event.get("event", "")
    product = event.get("product", "UNKNOWN")
    session_id = event.get("session_id", "default")

    if action == "pick" and product != "UNKNOWN":
        record_pick(product, session_id)
    elif action == "return" and product != "UNKNOWN":
        record_return(product, session_id)
    elif action == "anomaly":
        record_anomaly(product, session_id)
        alert = AlertItem(
            type="ANOMALY",
            message=f"Unregistered item detected: {product}. Staff verification required.",
            timestamp=event.get("timestamp", time.time()),
            details=event,
        )
        _alerts.append(alert.model_dump())
        _write_alert_log(alert.model_dump())

    # Push to WebSocket clients
    await _broadcast_ws(event)
    return {"status": "ok"}


@app.post("/internal/alert")
async def receive_alert(alert: dict):
    """Receive a structured alert from the pipeline (low-stock, misplaced, etc.)."""
    _alerts.append(alert)
    _write_alert_log(alert)
    await _broadcast_ws({"type": "alert", "data": alert})
    return {"status": "ok"}


# ─── WebSocket live feed ──────────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            # Keep connection alive; server pushes data via _broadcast_ws
            data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            # Client can send {"cmd": "ping"} for keepalive
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


async def _broadcast_ws(data: dict) -> None:
    """Push a message to all connected WebSocket clients."""
    if not _ws_clients:
        return
    msg = json.dumps(data)
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _write_alert_log(alert: dict) -> None:
    LOG_ALERTS.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_ALERTS, "a") as f:
        f.write(json.dumps(alert) + "\n")


# ─── Pipeline event callback (used when pipeline runs in same process) ────────

def pipeline_event_callback(event) -> None:
    """
    Synchronous callback registered with StateMachineManager.
    Dispatches event to DB and WS without running the async loop directly.
    Handles the case where the event loop may not be running.
    """
    import asyncio
    action = event.event if hasattr(event, "event") else event.get("event", "")
    product = event.product if hasattr(event, "product") else event.get("product", "UNKNOWN")
    session_id = (event.session_id if hasattr(event, "session_id")
                  else event.get("session_id", "default"))
    ev_dict = event.to_dict() if hasattr(event, "to_dict") else event

    if action == "pick" and product != "UNKNOWN":
        record_pick(product, session_id)
    elif action == "return" and product != "UNKNOWN":
        record_return(product, session_id)
    elif action == "anomaly":
        record_anomaly(product, session_id)
        alert = {
            "type": "ANOMALY",
            "message": f"Unregistered item: {product}. Staff verification required.",
            "timestamp": ev_dict.get("timestamp", time.time()),
            "details": ev_dict,
        }
        _alerts.append(alert)
        _write_alert_log(alert)
