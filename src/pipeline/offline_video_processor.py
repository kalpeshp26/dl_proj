"""
Offline Video Processor for DeepRetail

Performs batch / offline video analysis on recorded shopping sessions,
tracking all product picks and returns through the entire video,
and generates a finalized invoice bill and audit timeline.
"""

import time
import uuid
from pathlib import Path
from typing import Callable, Optional
import cv2

from src.zones.zone_manager import ZoneManager
from src.tracking.tracker import Tracker
from src.detection.yolo_detector import YOLODetector
from src.classification.infer_classifier import ProductClassifier
from src.logic.state_machine import StateMachineManager, RetailEvent
from src.backend.db import get_product_by_name


def analyze_shopping_video_offline(
    video_path: str,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> dict:
    """
    Process an entire video file from start to finish.
    Returns complete final bill, itemized picks & returns, timeline, and statistics.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total_frames <= 0:
        total_frames = 1  # avoid division by zero

    duration_sec = total_frames / max(fps, 1.0)

    # Initialize components
    zone_manager = ZoneManager()
    detector = YOLODetector(backend="auto")
    classifier = ProductClassifier()
    tracker = Tracker()
    session_id = f"offline_{uuid.uuid4().hex[:8]}"
    sm = StateMachineManager(session_id=session_id, anomaly_enabled=False)

    # State tracking
    cart: dict[str, dict] = {}
    returns: dict[str, dict] = {}
    events_log: list[dict] = []
    labels: dict[int, dict] = {}

    current_frame_idx = [0]

    def on_event(ev: RetailEvent):
        pname = ev.product
        prod_info = get_product_by_name(pname) or {
            "name": pname.replace("_", " ").title(),
            "price": 3.99,
        }
        sec = current_frame_idx[0] / max(fps, 1.0)
        mm = int(sec // 60)
        ss = int(sec % 60)
        time_str = f"{mm:02d}:{ss:02d}"

        if ev.event == "pick":
            if pname not in cart:
                cart[pname] = {
                    "product": pname,
                    "name": prod_info.get("name", pname),
                    "unit_price": float(prod_info.get("price", 0.0)),
                    "quantity": 0,
                }
            cart[pname]["quantity"] += 1
            events_log.append({
                "time": time_str,
                "timestamp_sec": round(sec, 2),
                "event": "pick",
                "product": pname,
                "name": prod_info.get("name", pname),
                "price": float(prod_info.get("price", 0.0)),
                "conf": round(ev.conf, 2),
                "track_id": ev.track_id,
            })
        elif ev.event == "return":
            if pname in cart and cart[pname]["quantity"] > 0:
                cart[pname]["quantity"] -= 1
                if cart[pname]["quantity"] == 0:
                    del cart[pname]
            if pname not in returns:
                returns[pname] = {
                    "product": pname,
                    "name": prod_info.get("name", pname),
                    "unit_price": float(prod_info.get("price", 0.0)),
                    "quantity": 0,
                }
            returns[pname]["quantity"] += 1
            events_log.append({
                "time": time_str,
                "timestamp_sec": round(sec, 2),
                "event": "return",
                "product": pname,
                "name": prod_info.get("name", pname),
                "price": float(prod_info.get("price", 0.0)),
                "conf": round(ev.conf, 2),
                "track_id": ev.track_id,
            })

    sm.register_callback(on_event)

    frame_idx = 0
    t_start = time.time()

    # Step frame-by-frame: we sample every 2 frames for fast offline computation while maintaining track continuity
    frame_step = 2 if total_frames > 200 else 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            current_frame_idx[0] = frame_idx

            if frame_idx % frame_step != 0:
                continue

            # Stage 1: Detection
            detections = detector.detect(frame, frame_idx)

            # Stage 2: Tracking
            tracks = tracker.update(detections)

            # Clean stale labels
            live_ids = {t.track_id for t in tracks}
            for tid in list(labels):
                if tid not in live_ids:
                    del labels[tid]

            # Stage 3: Classification
            for t in tracks:
                lbl = labels.setdefault(t.track_id, {"product": "UNKNOWN", "conf": 0.0, "votes": {}})
                if (frame_idx + t.track_id) % 4 == 0 or lbl["product"] == "UNKNOWN":
                    x1, y1, x2, y2 = [int(v) for v in t.box]
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                        crop = frame[y1:y2, x1:x2]
                        res = classifier.predict(crop)
                        if not res.is_unknown and res.conf >= 0.35:
                            v = lbl["votes"].setdefault(res.class_name, 0.0)
                            lbl["votes"][res.class_name] = v + res.conf
                            best = max(lbl["votes"].items(), key=lambda kv: kv[1])
                            lbl["product"] = best[0]
                            lbl["conf"] = res.conf

            # Stage 4: Assign zones and feed to state machine
            tracks_with_zones = []
            for t in tracks:
                lbl = labels.get(t.track_id, {"product": "UNKNOWN", "conf": 0.0})
                zone = zone_manager.get_zone(t.centroid)
                tracks_with_zones.append({
                    "track_id": t.track_id,
                    "zone": zone,
                    "cx": t.centroid[0],
                    "cy": t.centroid[1],
                    "product": lbl["product"],
                    "conf": lbl["conf"],
                })

            sm.process_frame(tracks_with_zones)

            if progress_callback and frame_idx % 10 == 0:
                pct = round((frame_idx / total_frames) * 100, 1)
                progress_callback(frame_idx, total_frames, pct)

    finally:
        cap.release()

    elapsed = round(time.time() - t_start, 2)
    if progress_callback:
        progress_callback(total_frames, total_frames, 100.0)

    # Compute final itemized bill
    items_list = []
    subtotal = 0.0
    for p, it in cart.items():
        if it["quantity"] > 0:
            line_tot = round(it["quantity"] * it["unit_price"], 2)
            subtotal += line_tot
            items_list.append({
                "product": p,
                "name": it["name"],
                "quantity": it["quantity"],
                "unit_price": it["unit_price"],
                "line_total": line_tot,
            })

    subtotal = round(subtotal, 2)
    tax = round(subtotal * 0.08, 2)
    grand_total = round(subtotal + tax, 2)

    returns_list = []
    for p, it in returns.items():
        if it["quantity"] > 0:
            returns_list.append({
                "product": p,
                "name": it["name"],
                "quantity": it["quantity"],
                "unit_price": it["unit_price"],
                "deducted_total": round(it["quantity"] * it["unit_price"], 2),
            })

    invoice_id = f"INV-VID-{uuid.uuid4().hex[:6].upper()}"

    return {
        "status": "completed",
        "video_filename": path.name,
        "total_frames": total_frames,
        "frames_processed": frame_idx,
        "duration_sec": round(duration_sec, 2),
        "analysis_time_sec": elapsed,
        "fps_analysis": round(frame_idx / max(elapsed, 0.01), 1),
        "total_picks": sum(it["quantity"] for it in cart.values()) + sum(it["quantity"] for it in returns.values()),
        "total_returns": sum(it["quantity"] for it in returns.values()),
        "net_items_count": sum(it["quantity"] for it in items_list),
        "bill": {
            "invoice_id": invoice_id,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "items": items_list,
            "returns": returns_list,
            "subtotal": subtotal,
            "tax": tax,
            "grand_total": grand_total,
        },
        "timeline": events_log,
    }
