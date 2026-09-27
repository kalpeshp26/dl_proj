"""
DeepRetail — Pipeline → Backend bridge.

The pipeline, the FastAPI backend and the Streamlit dashboard run as three
separate processes, so they cannot share Python objects. This bridge pushes
everything the backend needs over HTTP, from a background thread, so a slow or
missing backend never stalls the camera loop.

    events  (pick / return / anomaly)  → POST /internal/event
    alerts  (misplaced / unknown)      → POST /internal/alert
    status  (fps, frame id, tracks)    → POST /internal/status
    frame   (annotated JPEG)           → POST /internal/frame

Only the standard library is used (urllib), so no extra dependency.
"""

import json
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

import cv2
import numpy as np

from src.config import API_BASE_URL, FRAME_JPEG_QUALITY


class ApiBridge:
    def __init__(self, base_url: str = API_BASE_URL, timeout: float = 1.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._q: "queue.Queue[tuple[str, dict]]" = queue.Queue(maxsize=500)
        self._latest_frame: Optional[bytes] = None
        self._latest_status: Optional[dict] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._connected: Optional[bool] = None
        self._thread = threading.Thread(target=self._worker, daemon=True, name="api-bridge")
        self._thread.start()

    # ── Public API (called from the pipeline loop; never blocks) ──────────
    def send_event(self, event) -> None:
        payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
        self._put("/internal/event", payload)

    def send_alert(self, alert: dict) -> None:
        self._put("/internal/alert", alert)

    def send_status(self, status: dict) -> None:
        with self._lock:
            self._latest_status = status

    def send_frame(self, frame_bgr: np.ndarray, max_width: int = 960) -> None:
        h, w = frame_bgr.shape[:2]
        if w > max_width:
            frame_bgr = cv2.resize(frame_bgr, (max_width, int(h * max_width / w)))
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY])
        if ok:
            with self._lock:
                self._latest_frame = buf.tobytes()   # only the newest frame matters

    def close(self, flush_timeout: float = 2.0) -> None:
        deadline = time.time() + flush_timeout
        while not self._q.empty() and time.time() < deadline:
            time.sleep(0.05)
        self._stop.set()
        self._thread.join(timeout=1.0)

    # ── Internals ─────────────────────────────────────────────────────────
    def _put(self, path: str, payload: dict) -> None:
        try:
            self._q.put_nowait((path, payload))
        except queue.Full:
            pass  # backend is down for a long time; drop rather than block

    def _post(self, path: str, body: bytes, content_type: str) -> bool:
        req = urllib.request.Request(self.base_url + path, data=body, method="POST",
                                     headers={"Content-Type": content_type})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                ok = 200 <= r.status < 300
        except (urllib.error.URLError, OSError, TimeoutError):
            ok = False
        if ok != self._connected:
            self._connected = ok
            print(f"[Bridge] Backend {'connected' if ok else 'NOT reachable'} at {self.base_url}"
                  + ("" if ok else " — start it with: uvicorn src.backend.main:app --port 8000"))
        return ok

    def _worker(self) -> None:
        while not self._stop.is_set():
            # 1. Events/alerts first (they change the cart). Retry briefly if backend is down.
            try:
                path, payload = self._q.get(timeout=0.05)
                body = json.dumps(payload, default=str).encode()
                for attempt in range(3):
                    if self._post(path, body, "application/json"):
                        break
                    time.sleep(0.5 * (attempt + 1))
            except queue.Empty:
                pass

            # 2. Latest status + frame (lossy, best effort)
            with self._lock:
                status, self._latest_status = self._latest_status, None
                frame, self._latest_frame = self._latest_frame, None
            if status is not None:
                self._post("/internal/status", json.dumps(status).encode(), "application/json")
            if frame is not None:
                self._post("/internal/frame", frame, "image/jpeg")
