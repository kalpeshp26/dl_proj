"""
DeepRetail — FastAPI Backend (Stage 6)

Runs as its own process. The pipeline (src/pipeline/run_pipeline.py) pushes
events, alerts, status and the annotated frame here over HTTP
(src/pipeline/api_bridge.py); the dashboard reads everything from here.

Endpoints:
    GET  /cart/{session_id}
    GET  /inventory
    GET  /alerts
    GET  /analytics/top-products
    GET  /analytics/recommendations/{product_id}
    GET  /forecast              (Stage 8: demand forecast + reorder advice)
    GET  /forecast/{name}/history
    POST /video/upload | /video/analyze-offline, GET /video/job-status/{id}
    GET  /pipeline/status
    GET  /events                (latest N events)
    GET  /frame/latest          (latest annotated JPEG from the pipeline)
    POST /admin/reset           (restore inventory from data/inventory.csv, clear carts)
    WS   /ws/live               (real-time event stream to dashboard)

    POST /internal/event | /internal/alert | /internal/status | /internal/frame
                                (called by the pipeline)

Start:
    uvicorn src.backend.main:app --host 127.0.0.1 --port 8000 --reload
"""

import asyncio
import json
import sys
import time
import uuid
import threading
from pathlib import Path
from typing import Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Response, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.backend.db import (
    init_db, get_all_products, get_cart, get_cart_total,
    get_low_stock_products, get_transactions,
    record_pick, record_return, record_anomaly,
    get_product_by_name, reset_inventory,
)
from src.backend.analytics import get_top_products, get_recommendations
from src.backend.schemas import (
    CartResponse, CartItem, InventoryResponse, ProductInfo,
    AlertsResponse, AlertItem, EventItem,
    AnalyticsTopResponse, TopProductItem,
    AnalyticsRecsResponse, RecommendationItem,
    PipelineStatusResponse, ForecastItem, ForecastResponse,
)
from src.config import LOG_ALERTS

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# Latest state pushed by the pipeline process (see src/pipeline/api_bridge.py)
_pipeline_status: dict = {}
_pipeline_status_ts: float = 0.0
_latest_frame: Optional[bytes] = None
_latest_frame_ts: float = 0.0
PIPELINE_STALE_SEC = 3.0

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

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# In-memory alert buffer (appended by pipeline callbacks)
_alerts: list[dict] = []
_low_stock_alerted: set[str] = set()   # dedup LOW_STOCK alerts per product
# WebSocket connection manager
_ws_clients: list[WebSocket] = []


# ─── Health & Metadata ────────────────────────────────────────────────────────

@app.get("/")
def root_endpoint(request: Request):
    accept = request.headers.get("accept", "")
    index_file = STATIC_DIR / "index.html"
    if "text/html" in accept and index_file.exists():
        return FileResponse(index_file)
    return {
        "service": "DeepRetail API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
    }


@app.get("/kiosk")
def kiosk_endpoint():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"service": "DeepRetail Kiosk", "status": "online"}


@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": time.time()}


# ─── Cart & Checkout ──────────────────────────────────────────────────────────

@app.get("/cart/{session_id}", response_model=CartResponse)
def get_cart_endpoint(session_id: str):
    items_raw = get_cart(session_id)
    items = [CartItem(**i) for i in items_raw]
    total = round(sum(i.subtotal for i in items), 2)
    return CartResponse(session_id=session_id, items=items, total=total)


@app.post("/cart/checkout/{session_id}")
def checkout_cart(session_id: str):
    items_raw = get_cart(session_id)
    if not items_raw:
        return {"status": "empty", "message": "Cart is empty"}
    items = [CartItem(**i) for i in items_raw]
    subtotal = round(sum(i.subtotal for i in items), 2)
    tax = round(subtotal * 0.05, 2)
    grand_total = round(subtotal + tax, 2)
    inv_num = f"INV-{int(time.time()) % 100000:05d}"

    # Clear active cart for session
    import sqlite3
    from src.config import DATABASE_PATH
    with sqlite3.connect(DATABASE_PATH) as con:
        con.execute("DELETE FROM transactions WHERE session_id = ?", (session_id,))
        con.commit()

    return {
        "status": "success",
        "invoice_number": inv_num,
        "session_id": session_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "items": [i.model_dump() for i in items],
        "subtotal": subtotal,
        "tax": tax,
        "grand_total": grand_total,
    }


@app.post("/zones/preset")
def set_zone_preset(preset_data: dict):
    from src.config import save_zones
    save_zones(preset_data)
    return {"status": "ok", "zones": preset_data}


# ─── Video Upload & Dynamic Source Management ────────────────────────────────

@app.post("/video/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file to data/uploads and automatically set it as the active pipeline source."""
    from src.config import UPLOADS_DIR, save_active_source
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    clean_name = Path(file.filename or "upload.mp4").name
    target_path = UPLOADS_DIR / clean_name
    content = await file.read()
    with open(target_path, "wb") as f:
        f.write(content)
    save_active_source(str(target_path))
    return {
        "status": "success",
        "filename": clean_name,
        "path": str(target_path),
        "size_mb": round(len(content) / (1024 * 1024), 2),
        "active": True,
    }


@app.post("/pipeline/source")
def set_pipeline_source(payload: dict):
    """Switch active video source (e.g. 0 for webcam or path to uploaded video)."""
    from src.config import save_active_source
    src = payload.get("source", 0)
    save_active_source(src)
    return {"status": "ok", "source": src}


@app.get("/pipeline/source")
def get_pipeline_source():
    """Return the current active video source."""
    from src.config import load_active_source
    return {"source": load_active_source()}


@app.get("/videos")
def list_uploaded_videos():
    """List available uploaded video recordings."""
    from src.config import UPLOADS_DIR, load_active_source
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    active_src = load_active_source()
    files = []
    for p in sorted(UPLOADS_DIR.glob("*")):
        if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            files.append({
                "filename": p.name,
                "path": str(p),
                "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                "is_active": str(active_src) == str(p),
            })
    return {"videos": files, "active_source": active_src}


# ─── Offline Batch Shopping Video Analysis ────────────────────────────────────

_offline_jobs: dict[str, dict] = {}


def _run_offline_analysis_thread(job_id: str, video_path: str):
    from src.pipeline.offline_video_processor import analyze_shopping_video_offline
    _offline_jobs[job_id]["status"] = "processing"

    def on_progress(cur, tot, pct):
        _offline_jobs[job_id]["current_frame"] = cur
        _offline_jobs[job_id]["total_frames"] = tot
        _offline_jobs[job_id]["progress"] = pct

    try:
        res = analyze_shopping_video_offline(video_path, progress_callback=on_progress)
        _offline_jobs[job_id]["status"] = "completed"
        _offline_jobs[job_id]["progress"] = 100.0
        _offline_jobs[job_id]["result"] = res
    except Exception as e:
        import traceback
        traceback.print_exc()
        _offline_jobs[job_id]["status"] = "failed"
        _offline_jobs[job_id]["error"] = str(e)


@app.post("/video/analyze-offline")
def start_offline_video_analysis(payload: dict):
    """
    Start background offline batch analysis of a shopping video file.
    Returns job_id to poll progress and retrieve finalized bill.
    """
    from src.config import UPLOADS_DIR
    import importlib.util
    missing = [m for m in ("cv2", "torch", "ultralytics") if importlib.util.find_spec(m) is None]
    if missing:
        # e.g. the lightweight cloud build (requirements.cloud.txt) has no vision stack
        raise HTTPException(
            status_code=503,
            detail=f"Offline video analysis needs the full install (missing: {', '.join(missing)}). "
                   "Run it locally with requirements.txt.",
        )
    raw = payload.get("video_path") or payload.get("path") or payload.get("filename")
    if not raw:
        raise HTTPException(status_code=400, detail="Provide 'filename' of an uploaded video")
    # Only files inside data/uploads/ may be analysed (no arbitrary paths from the client)
    uploads = UPLOADS_DIR.resolve()
    candidate = Path(raw)
    candidate = (uploads / candidate.name) if not candidate.is_absolute() else candidate.resolve()
    if uploads not in candidate.parents or not candidate.exists():
        raise HTTPException(status_code=400, detail=f"Video not found in uploads: {Path(raw).name}")
    vpath = str(candidate)

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    _offline_jobs[job_id] = {
        "job_id": job_id,
        "video_path": vpath,
        "filename": Path(vpath).name,
        "status": "queued",
        "progress": 0.0,
        "current_frame": 0,
        "total_frames": 0,
        "result": None,
        "error": None,
        "start_time": time.time(),
    }
    t = threading.Thread(target=_run_offline_analysis_thread, args=(job_id, vpath), daemon=True)
    t.start()
    return {"job_id": job_id, "status": "processing", "filename": Path(vpath).name}


@app.get("/video/job-status/{job_id}")
def get_offline_analysis_status(job_id: str):
    """Poll status of offline shopping video analysis."""
    job = _offline_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


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


@app.api_route("/alerts/clear", methods=["GET", "POST"])
def clear_alerts():
    _alerts.clear()
    _low_stock_alerted.clear()
    return {"status": "cleared"}


@app.post("/admin/reset")
def admin_reset():
    """Restore stock from data/inventory.csv and wipe carts/history (between demo runs)."""
    reset_inventory()
    _alerts.clear()
    _low_stock_alerted.clear()
    return {"status": "reset"}


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


# ─── Demand forecasting (Stage 8) ─────────────────────────────────────────────

@app.get("/forecast", response_model=ForecastResponse)
def forecast_endpoint():
    """Daily demand forecast + reorder advice for every product."""
    from src.forecasting.forecast import forecast_all
    return ForecastResponse(products=[ForecastItem(**r) for r in forecast_all()])


@app.get("/forecast/{product_name}/history")
def forecast_history_endpoint(product_name: str):
    """Daily demand history of one product plus its forecast for the horizon."""
    from src.forecasting.forecast import product_history
    h = product_history(product_name)
    if h is None:
        raise HTTPException(status_code=404, detail=f"Product {product_name!r} not found")
    return h


# ─── Pipeline status ──────────────────────────────────────────────────────────

@app.get("/pipeline/status", response_model=PipelineStatusResponse)
def pipeline_status():
    fresh = (time.time() - _pipeline_status_ts) < PIPELINE_STALE_SEC
    if _pipeline_status and fresh:
        return PipelineStatusResponse(
            running=bool(_pipeline_status.get("running", False)),
            fps=float(_pipeline_status.get("fps", 0.0)),
            frame_id=int(_pipeline_status.get("frame_id", 0)),
            active_tracks=int(_pipeline_status.get("active_tracks", 0)),
            session_id=str(_pipeline_status.get("session_id", "unknown")),
        )
    return PipelineStatusResponse(
        running=False, fps=0.0,
        frame_id=int(_pipeline_status.get("frame_id", 0)) if _pipeline_status else 0,
        active_tracks=0,
        session_id=str(_pipeline_status.get("session_id", "none")) if _pipeline_status else "none",
    )


@app.get("/frame/latest")
def latest_frame():
    """Latest annotated frame (JPEG). 204 if the pipeline hasn't sent one recently."""
    if _latest_frame is None or (time.time() - _latest_frame_ts) > PIPELINE_STALE_SEC:
        return Response(status_code=204)
    return Response(content=_latest_frame, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


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

    recorded = None
    if action == "pick" and product != "UNKNOWN":
        recorded = record_pick(product, session_id)
    elif action == "return" and product != "UNKNOWN":
        recorded = record_return(product, session_id)
    elif action == "anomaly":
        record_anomaly(product, session_id)
        alert = AlertItem(
            type="ANOMALY",
            message=f"Unregistered item detected: {product}. Staff verification required.",
            timestamp=event.get("timestamp", time.time()),
            details=event,
        )
        _add_alert(alert.model_dump())

    if action in ("pick", "return") and product != "UNKNOWN" and recorded is None:
        print(f"[Backend] WARNING: '{product}' is not in the products table — check that "
              "classifier folder names match PRODUCT_CLASSES / data/inventory.csv")

    if action in ("pick", "return") and product != "UNKNOWN":
        _check_low_stock(product)

    # Push to WebSocket clients
    await _broadcast_ws(event)
    return {"status": "ok", "recorded": recorded is not None}


@app.post("/internal/status")
async def receive_status(status: dict):
    global _pipeline_status, _pipeline_status_ts
    _pipeline_status = status
    _pipeline_status_ts = time.time()
    return {"status": "ok"}


@app.post("/internal/frame")
async def receive_frame(request: Request):
    global _latest_frame, _latest_frame_ts
    _latest_frame = await request.body()
    _latest_frame_ts = time.time()
    return {"status": "ok"}


@app.post("/internal/alert")
async def receive_alert(alert: dict):
    """Receive a structured alert from the pipeline (misplaced, unknown, etc.)."""
    _alerts.append(alert)
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

def _add_alert(alert: dict) -> None:
    _alerts.append(alert)
    _write_alert_log(alert)


def _check_low_stock(product_name: str) -> None:
    """Fire LOW_STOCK once when stock drops to the threshold; re-arm when restocked."""
    p = get_product_by_name(product_name)
    if not p:
        return
    if p["stock"] <= p["low_stock_threshold"]:
        if product_name not in _low_stock_alerted:
            _low_stock_alerted.add(product_name)
            _add_alert({
                "type": "LOW_STOCK",
                "message": (f"Low stock: {product_name} — {p['stock']} left "
                            f"(threshold {p['low_stock_threshold']})"),
                "timestamp": time.time(),
                "details": p,
            })
    else:
        _low_stock_alerted.discard(product_name)


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
        _check_low_stock(product)
    elif action == "return" and product != "UNKNOWN":
        record_return(product, session_id)
        _check_low_stock(product)
    elif action == "anomaly":
        record_anomaly(product, session_id)
        _add_alert({
            "type": "ANOMALY",
            "message": f"Unregistered item: {product}. Staff verification required.",
            "timestamp": ev_dict.get("timestamp", time.time()),
            "details": ev_dict,
        })
