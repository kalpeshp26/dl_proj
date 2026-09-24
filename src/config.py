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
    "coke_500ml",
    "pepsi_500ml",
    "lays_classic",
    "lays_spicy",
    "oreo",
    "kurkure",
    "5star",
    "kitkat",
]

# ─── YOLO ────────────────────────────────────────────────────────────────────
# YOLO only detects 2 classes: product and hand — never product identity
YOLO_CLASSES: list[str] = ["product", "hand"]
YOLO_CONF_THRESH: float = 0.40          # minimum detection confidence
YOLO_IOU_THRESH: float = 0.45          # NMS IoU threshold
YOLO_IMGSZ: int = 640
YOLO_EPOCHS: int = 50
YOLO_BATCH: int = 8

# ─── Classifier ──────────────────────────────────────────────────────────────
CLASSIFIER_IMG_SIZE: int = 224          # MobileNetV3 input size
CLASSIFIER_BATCH: int = 32
CLASSIFIER_EPOCHS_FROZEN: int = 5       # backbone frozen, head trains
CLASSIFIER_EPOCHS_FINETUNE: int = 15    # full model fine-tuned
CLASSIFIER_LR_HEAD: float = 1e-3
CLASSIFIER_LR_FINETUNE: float = 1e-4
CLASSIFIER_WEIGHT_DECAY: float = 1e-4
CLASSIFIER_VAL_SPLIT: float = 0.2

# ─── Open-set rejection ──────────────────────────────────────────────────────
# Cosine distance to nearest class centroid. Above = UNKNOWN.
# Tune empirically using held-out non-trained objects.
OPENSET_DIST_THRESH: float = 0.35
EMBEDDING_DIM: int = 576               # MobileNetV3-Small penultimate dim

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

# ─── Backend / DB ────────────────────────────────────────────────────────────
DATABASE_PATH: Path = ROOT_DIR / "deepretail.db"
API_HOST: str = "127.0.0.1"
API_PORT: int = 8000

# ─── Runtime zone config (loaded from config.json if calibrated) ─────────────
CONFIG_JSON_PATH: Path = ROOT_DIR / "config.json"

_DEFAULT_ZONES = {
    "shelf_roi": [100, 50, 1100, 400],    # [x1, y1, x2, y2] in pixels
    "basket_roi": [100, 450, 600, 700],
    "shelf_slots": {
        # slot_id → {roi, expected_product, low_stock_threshold}
        "slot_0": {
            "roi": [100, 50, 350, 400],
            "expected_product": "coke_500ml",
            "low_stock_threshold": 3,
        },
        "slot_1": {
            "roi": [350, 50, 600, 400],
            "expected_product": "pepsi_500ml",
            "low_stock_threshold": 3,
        },
        "slot_2": {
            "roi": [600, 50, 850, 400],
            "expected_product": "lays_classic",
            "low_stock_threshold": 3,
        },
        "slot_3": {
            "roi": [850, 50, 1100, 400],
            "expected_product": "oreo",
            "low_stock_threshold": 3,
        },
    },
}


def load_zones() -> dict:
    """
    Load zone/ROI configuration from config.json (written by calibrate.py).
    Falls back to _DEFAULT_ZONES if the file does not exist.
    """
    if CONFIG_JSON_PATH.exists():
        with open(CONFIG_JSON_PATH, "r") as f:
            data = json.load(f)
        return data.get("zones", _DEFAULT_ZONES)
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


def ensure_dirs() -> None:
    """Create all required runtime directories if they don't exist."""
    for d in [LOG_DIRECTORY, MODELS_DIR, DATA_DIR, RAW_IMAGES_DIR,
              YOLO_DATASET_DIR, CLASSIFIER_DATASET_DIR]:
        d.mkdir(parents=True, exist_ok=True)
