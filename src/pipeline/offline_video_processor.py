"""
DeepRetail — Offline Video Processor

Runs a recorded shopping video through the same Stages 1–5 as the live
pipeline (src/pipeline/run_pipeline.py) as fast as the CPU allows, and returns
an itemised bill, returns list and event timeline. Called from the backend's
POST /video/analyze-offline in a background thread.

It does NOT write to the store's cart or stock: the result is a standalone
invoice for the uploaded recording.

Notes
  * Needs the full install (torch, ultralytics, opencv) plus trained classifier
    weights; the lightweight cloud build returns 503 before reaching this file.
  * The state machine's re-acquisition window uses wall-clock time, and the
    video is processed faster than real time, so anomaly detection is off.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable, Optional

import cv2

from src.config import CLASSIFY_EVERY_N_FRAMES, BILL_TAX_RATE
from src.detection.yolo_infer import DetectionResult, YOLODetector
from src.tracking.tracker import Tracker
from src.logic.zones import ZoneManager
from src.logic.state_machine import StateMachineManager, RetailEvent
from src.pipeline.run_pipeline import TrackLabel
from src.backend.db import get_product_by_name


def _display_name(class_name: str) -> str:
    return class_name.replace("_", " ").title()


def _fmt_time(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def analyze_shopping_video_offline(
    video_path: str,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> dict:
    """Process a whole video file; return bill, returns, timeline and stats."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    total_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_sec = total_frames / max(fps, 1.0)

    # ── Same components as the live pipeline ─────────────────────────────
    from src.classification.infer_classifier import ProductClassifier  # torch, lazy
    zones = ZoneManager()
    detector = YOLODetector(log=False)          # trained weights if present, else YOLO-World
    classifier = ProductClassifier(log=False)
    tracker = Tracker(log=False)
    session_id = f"offline_{uuid.uuid4().hex[:8]}"
    sm = StateMachineManager(session_id=session_id, anomaly_enabled=False)

    cart: dict[str, dict] = {}
    returns: dict[str, dict] = {}
    timeline: list[dict] = []
    labels: dict[int, TrackLabel] = {}
    frame_idx = 0
    price_cache: dict[str, Optional[float]] = {}

    def price_of(name: str) -> Optional[float]:
        if name not in price_cache:
            p = get_product_by_name(name)
            price_cache[name] = float(p["price"]) if p else None
        return price_cache[name]

    def on_event(ev: RetailEvent) -> None:
        if ev.event not in ("pick", "return") or ev.product == "UNKNOWN":
            return
        price = price_of(ev.product)
        if price is None:
            print(f"[Offline] '{ev.product}' is not in the catalog — skipped")
            return
        sec = frame_idx / max(fps, 1.0)
        line = {"product": ev.product, "name": _display_name(ev.product),
                "unit_price": price, "quantity": 0}
        if ev.event == "pick":
            cart.setdefault(ev.product, dict(line))["quantity"] += 1
        else:
            if ev.product in cart:
                cart[ev.product]["quantity"] -= 1
                if cart[ev.product]["quantity"] <= 0:
                    del cart[ev.product]
            returns.setdefault(ev.product, dict(line))["quantity"] += 1
        timeline.append({
            "time": _fmt_time(sec), "timestamp_sec": round(sec, 2),
            "event": ev.event, "product": ev.product,
            "name": _display_name(ev.product), "price": price,
            "conf": round(float(ev.conf), 2), "track_id": ev.track_id,
        })

    sm.register_callback(on_event)
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            detections = detector.detect(frame, frame_idx)          # Stage 1
            tracks = tracker.update(detections)                      # Stage 2 (products only)

            live = {t.track_id for t in tracks}
            for tid in list(labels):
                if tid not in live:
                    del labels[tid]

            for t in tracks:                                         # Stage 3 / 3b
                lab = labels.setdefault(t.track_id, TrackLabel())
                if (frame_idx + t.track_id) % CLASSIFY_EVERY_N_FRAMES and lab._names:
                    continue
                crop = DetectionResult(class_name="product", conf=t.conf, xyxy=t.box).crop(frame)
                if crop.size == 0:
                    continue
                t_prompt = getattr(t, "prompt", "").lower()
                direct_product = None
                if any(k in t_prompt for k in ("hakka", "noodle")):
                    direct_product = "chings_hakka"
                elif any(k in t_prompt for k in ("manchurian", "soup")):
                    direct_product = "chings_manchurian"
                elif any(k in t_prompt for k in ("matchbox", "homelite")):
                    direct_product = "homelite_matchbox"
                elif any(k in t_prompt for k in ("vaseline", "petroleum jelly")):
                    direct_product = "vaseline_jelly"

                r = classifier.classify(crop, frame_id=frame_idx, track_id=t.track_id)
                final_name = direct_product if direct_product else ("UNKNOWN" if r.is_unknown else r.class_name)
                final_conf = 0.95 if direct_product else r.conf
                lab.add(final_name, final_conf)

            sm.process_frame([{                                      # Stages 4 + 5
                "track_id": t.track_id,
                "zone": zones.classify_location(t.cx, t.cy),
                "product": labels[t.track_id].class_name if t.track_id in labels else "UNKNOWN",
                "conf": labels[t.track_id].conf if t.track_id in labels else 0.0,
                "cx": t.cx, "cy": t.cy,
            } for t in tracks])

            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx, total_frames,
                                  min(99.0, round(frame_idx / total_frames * 100, 1)))
    finally:
        cap.release()

    elapsed = round(time.time() - t_start, 2)
    if progress_callback:
        progress_callback(total_frames, total_frames, 100.0)

    items = []
    for p, it in cart.items():
        if it["quantity"] > 0:
            items.append({**it, "line_total": round(it["quantity"] * it["unit_price"], 2)})
    subtotal = round(sum(i["line_total"] for i in items), 2)
    tax = round(subtotal * BILL_TAX_RATE, 2)
    returns_list = [{**it, "deducted_total": round(it["quantity"] * it["unit_price"], 2)}
                    for it in returns.values() if it["quantity"] > 0]
    n_picks = sum(1 for e in timeline if e["event"] == "pick")
    n_returns = sum(1 for e in timeline if e["event"] == "return")

    return {
        "status": "completed",
        "video_filename": path.name,
        "total_frames": total_frames,
        "frames_processed": frame_idx,
        "duration_sec": round(duration_sec, 2),
        "analysis_time_sec": elapsed,
        "fps_analysis": round(frame_idx / max(elapsed, 0.01), 1),
        "total_picks": n_picks,
        "total_returns": n_returns,
        "net_items_count": sum(i["quantity"] for i in items),
        "bill": {
            "invoice_id": f"INV-VID-{uuid.uuid4().hex[:6].upper()}",
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "items": items,
            "returns": returns_list,
            "subtotal": subtotal,
            "tax_rate": BILL_TAX_RATE,
            "tax": tax,
            "grand_total": round(subtotal + tax, 2),
        },
        "timeline": timeline,
    }
