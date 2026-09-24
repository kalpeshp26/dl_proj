# DeepRetail — Deep Learning-Based Autonomous Checkout & Retail Intelligence

> Single-camera, real-time retail intelligence system combining product detection, fine-grained classification, open-set recognition, object tracking, rule-based interaction logic, inventory management, and a live dashboard.

---

## Architecture

```
WEBCAM
  ↓
Stage 1 — YOLOv8n Detection          (product + hand localization ONLY)
  ↓
Stage 2 — Custom IoU/Centroid Tracker (Hungarian matching, persistent IDs)
  ↓
Stage 3 — MobileNetV3-Small Classifier (product identity from crop)
  ↓
Stage 3b — Open-Set Rejection          (cosine distance to class centroids)
  ↓
Stage 4 — Zone Logic                   (shelf ROI, basket ROI, slot ROIs)
  ↓
Stage 5 — Rule-Based State Machine     (SHELF→HELD→BASKET = pick, etc.)
  ↓
Stage 6 — FastAPI + SQLite             (cart, inventory, analytics, alerts)
  ↓
Stage 7 — Streamlit Dashboard          (live feed, cart, inventory, alerts)
```

**Key principle:** YOLO only localizes ("is there a product here?"). Product identity always comes from Stage 3. No LSTM or Transformer for action recognition — the state machine is rule-based.

---

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.10+ |
| Detection | Ultralytics YOLOv8n |
| DL Framework | PyTorch + torchvision |
| Classifier | MobileNetV3-Small (ImageNet pretrained) |
| Tracker | Custom IoU+centroid (scipy Hungarian) |
| Backend | FastAPI + Uvicorn |
| Database | SQLite |
| Dashboard | Streamlit |
| Video I/O | OpenCV |
| Logging | JSONL |

---

## Installation

### 1. Clone / unzip the project

```bash
cd deepretail
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> On a CPU-only machine PyTorch will be installed automatically. For CUDA, install the appropriate torch version from https://pytorch.org/get-started/locally/ before running the above command.

---

## Dataset Preparation

### Product classes

Edit `src/config.py` and set `PRODUCT_CLASSES` to your 5–8 physical products:

```python
PRODUCT_CLASSES = [
    "coke_500ml",
    "pepsi_500ml",
    "lays_classic",
    ...
]
```

### Collect product images (classifier dataset)

```bash
# Capture one class at a time
python src/data_tools/capture_images.py --class coke_500ml

# Or cycle through all classes interactively
python src/data_tools/capture_images.py --all

# Controls: SPACE = save frame, q = quit/next class
```

Target: **150–300 images per class** + ~50 background/empty images.

---

## YOLO Annotation

1. Annotate **200–300 images** using [LabelImg](https://github.com/HumanSignal/labelImg) or [Roboflow](https://roboflow.com):
   - Class **0** = `product` (any product regardless of type)
   - Class **1** = `hand`
   - Export in **YOLO format**

2. Place:
   ```
   data/yolo_dataset/images/  ← all annotated images
   data/yolo_dataset/labels/  ← corresponding .txt label files
   ```

3. Generate `data.yaml` and split into train/val:
   ```bash
   python src/data_tools/create_yolo_yaml.py
   ```

---

## YOLO Training

```bash
python src/detection/train_yolo.py
# Optional: python src/detection/train_yolo.py --epochs 100 --batch 16
```

Output: `models/yolo_best.pt`

---

## Classifier Training

```bash
python src/classification/train_classifier.py
# Optional: --epochs-frozen 5 --epochs-finetune 20
```

Output: `models/classifier_best.pt`, `models/class_indices.json`, `logs/classifier_eval.json`

---

## Open-Set Centroid Generation

Run after classifier training to compute class centroids for unknown-product rejection:

```bash
python src/classification/openset.py
```

Output: `models/class_centroids.npy`, `models/centroids_meta.json`

Tune `OPENSET_DIST_THRESH` in `src/config.py` if:
- Too many known products flagged as UNKNOWN → increase threshold
- Unknown objects not rejected → decrease threshold

---

## Calibration

Set up your shelf + basket ROIs by pointing the camera at your physical setup:

```bash
python calibrate.py
```

Follow the interactive prompts to click ROIs for:
1. Shelf region
2. Basket region
3. Each shelf slot (mapped to expected product)

Output: `config.json` in the project root.

---

## Running the Pipeline

```bash
# Live webcam (default)
python src/pipeline/run_pipeline.py

# Video file (replay/fallback mode)
python src/pipeline/run_pipeline.py --source path/to/recording.mp4

# Headless (no display window)
python src/pipeline/run_pipeline.py --no-display

# Enable Phase 2 anomaly detection
python src/pipeline/run_pipeline.py --anomaly

# Mock mode (no models needed — for testing)
python src/pipeline/run_pipeline.py --mock
```

---

## Running FastAPI Backend

```bash
uvicorn src.backend.main:app --host 127.0.0.1 --port 8000 --reload
```

API docs: http://127.0.0.1:8000/docs

Key endpoints:
```
GET  /cart/{session_id}
GET  /inventory
GET  /alerts
GET  /analytics/top-products
GET  /analytics/recommendations/{product_id}
GET  /pipeline/status
WS   /ws/live
```

---

## Running Streamlit Dashboard

Start the backend first, then:

```bash
streamlit run dashboard/app.py
```

Dashboard opens at http://localhost:8501

---

## Running Tests

```bash
# All tests (no camera/weights required)
pytest

# Specific test file
pytest tests/test_state_machine.py -v

# With coverage
pytest --cov=src --cov-report=term-missing
```

Hardware-dependent tests (YOLO accuracy, classifier accuracy, live pick/return) must be run manually using the physical setup per the pass/fail criteria in `DeepRetail_Implementation_Guide.md`.

---

## Demo Procedure

Ensure all six scenarios work from one session (no restarts between them):

| # | Scenario | Expected |
|---|---|---|
| 1 | Pick product → place in basket | Cart updates, inventory decrements |
| 2 | Pick product → return to shelf | Cart removes item, inventory restored |
| 3 | Pick product → conceal/walk off (>10s) | `ANOMALY` alert fired |
| 4 | Present untrained object | `UNKNOWN PRODUCT` alert, no cart update |
| 5 | Place wrong product in a shelf slot | `MISPLACED_PRODUCT` alert after K frames |
| 6 | Remove items until below threshold | `LOW_STOCK` alert fires |

---

## Configuration Reference

All tunable values are in `src/config.py`:

| Constant | Default | Description |
|---|---|---|
| `PRODUCT_CLASSES` | 8 classes | Your physical product names |
| `YOLO_CONF_THRESH` | 0.40 | YOLO minimum detection confidence |
| `OPENSET_DIST_THRESH` | 0.35 | Cosine distance threshold for UNKNOWN |
| `TRACK_IOU_THRESH` | 0.30 | Minimum IoU to associate detection to track |
| `TRACK_MAX_AGE` | 15 | Frames before a track expires |
| `ANOMALY_TIMEOUT_SEC` | 10.0 | Seconds held without basket → anomaly |
| `MISPLACED_CONSECUTIVE_FRAMES` | 10 | Frames before misplacement alert fires |
| `LOW_STOCK_DEFAULT_THRESHOLD` | 3 | Default stock level for low-stock alert |
| `CAMERA_INDEX` | 0 | OpenCV camera index |
| `TARGET_FPS` | 15 | Pipeline target frame rate |

---

## Project Structure

```
deepretail/
├── data/
│   ├── raw_images/<class_name>/         # collected images
│   ├── yolo_dataset/images+labels/      # annotated YOLO dataset
│   └── classifier_dataset/<class_name>/ # classifier training images
├── models/
│   ├── yolo_best.pt                     # trained YOLO weights
│   ├── classifier_best.pt               # trained MobileNetV3 weights
│   ├── class_centroids.npy              # open-set centroids
│   └── class_indices.json               # class name ↔ index map
├── src/
│   ├── config.py                        # all constants and thresholds
│   ├── detection/yolo_infer.py          # Stage 1
│   ├── detection/train_yolo.py
│   ├── tracking/tracker.py              # Stage 2
│   ├── classification/train_classifier.py # Stage 3 training
│   ├── classification/infer_classifier.py # Stage 3 inference
│   ├── classification/openset.py        # Stage 3b
│   ├── logic/zones.py                   # Stage 4
│   ├── logic/state_machine.py           # Stage 5
│   ├── logic/shelf_monitor.py           # Phase 2 shelf monitoring
│   ├── backend/main.py                  # Stage 6 FastAPI
│   ├── backend/db.py                    # SQLite layer
│   ├── backend/schemas.py               # Pydantic models
│   ├── backend/analytics.py             # recommendations engine
│   ├── pipeline/run_pipeline.py         # main pipeline runner
│   └── data_tools/capture_images.py     # data collection
├── dashboard/app.py                     # Stage 7 Streamlit
├── calibrate.py                         # zone calibration
├── logs/                                # JSONL logs per stage
├── tests/                               # pytest suite
├── requirements.txt
└── README.md
```

---

## Troubleshooting

**YOLO weights not found**
→ Run `python src/detection/train_yolo.py`

**Classifier weights not found**
→ Run `python src/classification/train_classifier.py`

**No centroids file**
→ Run `python src/classification/openset.py` after classifier training

**Everything shows UNKNOWN**
→ Decrease `OPENSET_DIST_THRESH` in `src/config.py`

**Too many anomaly false positives**
→ Increase `ANOMALY_TIMEOUT_SEC`

**Misplacement alerts firing for brief movements**
→ Increase `MISPLACED_CONSECUTIVE_FRAMES`

**Camera not opening**
→ Change `CAMERA_INDEX` in config.py; verify with `python -c "import cv2; cap=cv2.VideoCapture(0); print(cap.isOpened())"`

**Dashboard shows no frame**
→ Run pipeline in a separate terminal with `--no-display`; ensure backend is running

**SQLite locked error**
→ Restart the backend; WAL mode handles most concurrency but some edge cases may require a fresh DB

---

## Logging

Every pipeline stage writes to `logs/`:
- `logs/detection.jsonl` — per-frame YOLO detections
- `logs/tracking.jsonl` — per-frame track states  
- `logs/classification.jsonl` — per-crop classifier outputs
- `logs/events.jsonl` — pick/return/anomaly events
- `logs/alerts.jsonl` — low-stock/misplaced/anomaly alerts

Logs allow full session replay for debugging without live camera.
