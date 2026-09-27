"""
DeepRetail — Central Configuration
All thresholds, paths, and constants live here. No magic numbers anywhere else.
If a value needs tuning, change it here only.
"""

import json
import os
from pathlib import Path

# ─── Project root ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ─── Product classes (user fills this with their physical products) ───────────
# These must exactly match the subdirectory names in data/classifier_dataset/
PRODUCT_CLASSES: list[str] = [
    "chings_manchurian",
    "chings_hakka",
    "homelite_matchbox",
    "vaseline_jelly",
]

# ─── YOLO ────────────────────────────────────────────────────────────────────
# YOLO only detects 2 classes: product and hand — never product identity
YOLO_CLASSES: list[str] = ["product", "hand"]
YOLO_CONF_THRESH: float = 0.40          # minimum detection confidence
YOLO_IOU_THRESH: float = 0.45          # NMS IoU threshold
YOLO_IMGSZ: int = 640
YOLO_EPOCHS: int = 50
YOLO_BATCH: int = 8

# ─── Zero-shot detector (YOLO-World) — no annotation needed ──────────────────
# Used (a) by src/data_tools/auto_label.py to auto-annotate frames for training
# and (b) at runtime when models/yolo_best.pt doesn't exist (or --detector world).
# Weights download automatically on first use (~25 MB for "s").
WORLD_MODEL: str = "yolov8s-worldv2.pt"
# text prompt → DeepRetail detector class. Tune the prompts if things are missed.
WORLD_PROMPTS: dict[str, str] = {
    # Manchurian soup & seasoning pouches:
    "veg manchurian packet": "product",
    "manchurian packet": "product",
    "chings veg manchurian": "product",
    "chings manchurian soup": "product",
    "veg manchurian soup": "product",
    "manchurian soup packet": "product",
    "soup packet": "product",
    "soup sachet": "product",
    "green soup packet": "product",
    "chings packet": "product",
    "chinese soup packet": "product",

    # Hakka noodles & masala pouches:
    "hakka noodles packet": "product",
    "chings hakka": "product",
    "orange noodles packet": "product",
    "noodles packet": "product",
    "noodle masala packet": "product",

    # Matchbox:
    "homelite matchbox": "product",
    "matchbox": "product",
    "small matchbox": "product",
    "yellow matchbox": "product",

    # Vaseline:
    "vaseline petroleum jelly": "product",
    "vaseline jar": "product",
    "blue plastic jar": "product",
    "petroleum jelly": "product",

    # Generic packages / grocery items:
    "snack sachet packet": "product",
    "spice packet": "product",
    "food packet": "product",
    "food pouch": "product",
    "seasoning packet": "product",
    "small box": "product",
    "package": "product",
    "product": "product",
    "item": "product",
    "small plastic jar": "product",
    "jar container": "product",
    "small cardboard box": "product",
    "grocery item": "product",
    "retail merchandise": "product",
    "object held in hand": "product",

    # Hands:
    "person hand": "hand",
    "human hand": "hand",
    "hand holding object": "hand",
    "hand": "hand",
}
WORLD_CONF_THRESH: float = 0.045       # High sensitivity for tilted/held products
WORLD_MAX_BOX_FRAC: float = 0.40       # Allow close-up objects (up to 40% of camera frame)
DETECTOR_BACKEND: str = "auto"         # "auto" = trained weights if present, else YOLO-World

# ─── Classifier ──────────────────────────────────────────────────────────────
CLASSIFIER_IMG_SIZE: int = 224          # MobileNetV3 input size
CLASSIFIER_BATCH: int = 32
CLASSIFIER_EPOCHS_FROZEN: int = 5       # backbone frozen, head trains
CLASSIFIER_EPOCHS_FINETUNE: int = 15    # full model fine-tuned
CLASSIFIER_LR_HEAD: float = 1e-3
CLASSIFIER_LR_FINETUNE: float = 1e-4
CLASSIFIER_WEIGHT_DECAY: float = 1e-4
CLASSIFIER_VAL_SPLIT: float = 0.2      # only used if the dataset has no train/valid folders
CLASSIFIER_NUM_WORKERS: int = 0        # 0 = safe on Windows; raise to 2-4 on Linux/macOS

# Pipeline-side classification smoothing
CLASSIFY_EVERY_N_FRAMES: int = 2       # re-classify each track every N frames (CPU saver)
CLASS_VOTE_WINDOW: int = 3             # majority vote over last N predictions per track (fast response)
UNKNOWN_ALERT_FRAMES: int = 8          # consecutive UNKNOWN votes before UNKNOWN_PRODUCT alert

# ─── Open-set rejection ──────────────────────────────────────────────────────
# Cosine distance to nearest class centroid. Above = UNKNOWN.
# Lowered to 0.30: products not in training (e.g. Sunsilk) typically score
# 0.35–0.55 from the nearest centroid; trained products score <0.25.
OPENSET_DIST_THRESH: float = 0.30
EMBEDDING_DIM: int = 1024              # MobileNetV3-Small penultimate layer dim (classifier[:-1])

# ─── Tracker ─────────────────────────────────────────────────────────────────
TRACK_IOU_THRESH: float = 0.30         # minimum IoU to associate detection to track
TRACK_MAX_AGE: int = 15                # frames a track survives without detection
TRACK_MIN_HITS: int = 2                # detections before track is confirmed
TRACK_CENTROID_MAX_DIST: float = 150.0 # fallback centroid distance (pixels)

# ─── Camera / video ──────────────────────────────────────────────────────────
CAMERA_INDEX: int = 0
FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
TARGET_FPS: int = 15                   # pipeline target FPS (throttle if GPU absent)

# ─── State machine ───────────────────────────────────────────────────────────
# Phase 2: anomaly timeout — if HELD > this many seconds without resolution
ANOMALY_TIMEOUT_SEC: float = 10.0

# ─── Shelf/misplacement ──────────────────────────────────────────────────────
# K consecutive frames the wrong product must appear before alert fires
MISPLACED_CONSECUTIVE_FRAMES: int = 10
# Frames between periodic shelf-count sanity checks
SHELF_COUNT_INTERVAL_FRAMES: int = 30

# ─── Inventory / low-stock ───────────────────────────────────────────────────
LOW_STOCK_DEFAULT_THRESHOLD: int = 3

# ─── Demand forecasting (Stage 8) ────────────────────────────────────────────
# Simple Exponential Smoothing on daily net demand (picks − returns) per product.
FORECAST_ALPHAS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.7)  # α grid, picked by backtest
FORECAST_BACKTEST_DAYS: int = 14       # rolling-origin one-step-ahead backtest window
FORECAST_MIN_HISTORY_DAYS: int = 7     # below this, fall back to the historical mean
FORECAST_LEAD_TIME_DAYS: int = 2       # L — supplier lead time
FORECAST_HORIZON_DAYS: int = 7         # H — days of cover an order should buy
FORECAST_SERVICE_Z: float = 1.65       # z — ≈95% cycle service level
FORECAST_MA_WINDOW: int = 7            # moving-average baseline window
SIM_SESSION_PREFIX: str = "sim_"       # session_id prefix for simulated history rows

# Tax on the bill (same 5% the kiosk UI applies to the live cart in static/app.js)
BILL_TAX_RATE: float = float(os.getenv("BILL_TAX_RATE", "0.05"))

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_DIRECTORY: Path = ROOT_DIR / "logs"
LOG_DETECTION: Path = LOG_DIRECTORY / "detection.jsonl"
LOG_TRACKING: Path = LOG_DIRECTORY / "tracking.jsonl"
LOG_CLASSIFICATION: Path = LOG_DIRECTORY / "classification.jsonl"
LOG_EVENTS: Path = LOG_DIRECTORY / "events.jsonl"
LOG_ALERTS: Path = LOG_DIRECTORY / "alerts.jsonl"

# ─── Model paths ─────────────────────────────────────────────────────────────
MODELS_DIR: Path = ROOT_DIR / "models"
YOLO_MODEL_PATH: Path = MODELS_DIR / "yolo_best.pt"
CLASSIFIER_MODEL_PATH: Path = MODELS_DIR / "classifier_best.pt"
CLASS_CENTROIDS_PATH: Path = MODELS_DIR / "class_centroids.npy"

# ─── Data paths ──────────────────────────────────────────────────────────────
DATA_DIR: Path = ROOT_DIR / "data"
RAW_IMAGES_DIR: Path = DATA_DIR / "raw_images"
YOLO_DATASET_DIR: Path = DATA_DIR / "yolo_dataset"
CLASSIFIER_DATASET_DIR: Path = DATA_DIR / "classifier_dataset"
# Preferred layout (created by src/data_tools/prepare_classifier_dataset.py):
#   classifier_dataset/train/<class>/  valid/<class>/  test/<class>/
# A flat layout classifier_dataset/<class>/ still works (random split is used).
CLASSIFIER_TRAIN_DIR: Path = CLASSIFIER_DATASET_DIR / "train"
CLASSIFIER_VALID_DIR: Path = CLASSIFIER_DATASET_DIR / "valid"
CLASSIFIER_TEST_DIR: Path = CLASSIFIER_DATASET_DIR / "test"

# ─── Backend / DB ────────────────────────────────────────────────────────────
DATABASE_PATH: Path = ROOT_DIR / "deepretail.db"
# Product catalog (price, starting stock, threshold, slot) — edit this CSV,
# then run:  python -m src.backend.db --reset
INVENTORY_CSV_PATH: Path = DATA_DIR / "inventory.csv"
API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_BASE_URL: str = os.getenv("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")

# Pipeline -> backend bridge (pipeline and backend run as separate processes)
DEFAULT_SESSION_ID: str = os.getenv("DEFAULT_SESSION_ID", "demo_session")
FRAME_PUSH_INTERVAL_SEC: float = 0.2       # push annotated frame to backend ~5 fps
FRAME_JPEG_QUALITY: int = 70
STATUS_PUSH_INTERVAL_SEC: float = 0.5

# ─── Runtime zone config (loaded from config.json if calibrated) ─────────────
CONFIG_JSON_PATH: Path = ROOT_DIR / "config.json"

_DEFAULT_ZONES = {
    # Optimized for webcam (640x480 standard / scalable):
    # Left zone: Desk / Table Staging Area (outside bag)
    # Right zone: Shopping Bag / Basket Area
    "shelf_roi": [20, 20, 310, 460],     # [x1, y1, x2, y2] Table / Scan area
    "basket_roi": [330, 20, 620, 460],   # [x1, y1, x2, y2] Shopping Bag area
    "shelf_slots": {},                   # No rigid slots needed for desk shopping
}


def load_zones() -> dict:
    """
    Load zone/ROI configuration from config.json (written by calibrate.py).
    Falls back to _DEFAULT_ZONES if the file does not exist.
    """
    if CONFIG_JSON_PATH.exists():
        try:
            with open(CONFIG_JSON_PATH, "r") as f:
                data = json.load(f)
            return data.get("zones", _DEFAULT_ZONES)
        except Exception:
            return _DEFAULT_ZONES
    return _DEFAULT_ZONES


def save_zones(zones: dict) -> None:
    """Persist calibrated zones to config.json."""
    existing: dict = {}
    if CONFIG_JSON_PATH.exists():
        with open(CONFIG_JSON_PATH, "r") as f:
            existing = json.load(f)
    existing["zones"] = zones
    with open(CONFIG_JSON_PATH, "w") as f:
        json.dump(existing, f, indent=2)


UPLOADS_DIR: Path = DATA_DIR / "uploads"


def load_active_source():
    """Load active video source (int camera index or str video file path)."""
    if CONFIG_JSON_PATH.exists():
        try:
            with open(CONFIG_JSON_PATH, "r") as f:
                data = json.load(f)
            src = data.get("source", 0)
            if isinstance(src, str) and src.isdigit():
                return int(src)
            return src
        except Exception:
            return 0
    return 0


def save_active_source(source) -> None:
    """Save active video source to config.json."""
    existing: dict = {}
    if CONFIG_JSON_PATH.exists():
        try:
            with open(CONFIG_JSON_PATH, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing["source"] = source
    with open(CONFIG_JSON_PATH, "w") as f:
        json.dump(existing, f, indent=2)


def ensure_dirs() -> None:
    """Create all required runtime directories if they don't exist."""
    for d in [LOG_DIRECTORY, MODELS_DIR, DATA_DIR, RAW_IMAGES_DIR,
              YOLO_DATASET_DIR, CLASSIFIER_DATASET_DIR, UPLOADS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
