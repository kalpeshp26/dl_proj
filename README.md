# DeepRetail — Deep Learning-Based Autonomous Checkout & Retail Intelligence

> Single-camera, real-time retail intelligence: product detection, fine-grained classification, open-set (unknown product) rejection, multi-object tracking, rule-based pick/return logic, live inventory, alerts and a dashboard.

Built by students of the Department of CSE (AI & ML), Vishwakarma Institute of Technology, Pune.

---

## Architecture

```
WEBCAM / VIDEO FILE
  ↓
Stage 1 — YOLOv8n Detection            (localizes "product" and "hand" only)
  ↓
Stage 2 — IoU/Centroid Tracker          (Hungarian matching, persistent IDs)
  ↓
Stage 3 — MobileNetV3-Small Classifier  (product identity from the crop, majority-voted per track)
  ↓
Stage 3b — Open-Set Rejection           (cosine distance to class centroids → UNKNOWN)
  ↓
Stage 4 — Zone Logic                    (shelf ROI, basket ROI, per-product shelf slots)
  ↓
Stage 5 — Rule-Based State Machine      (SHELF→HELD→BASKET = pick, BASKET→HELD→SHELF = return)
        + Shelf Monitor                 (misplaced product) + UNKNOWN_PRODUCT alerts
  ↓  HTTP (events, alerts, status, annotated frame)
Stage 6 — FastAPI + SQLite              (cart, inventory, low-stock, analytics)
  ↓
Stage 7 — Streamlit Dashboard           (live feed, cart, inventory, alerts)
```

The pipeline, backend and dashboard are **three separate processes**. The pipeline pushes everything to the backend through a non-blocking HTTP bridge (`src/pipeline/api_bridge.py`), and the dashboard reads only from the backend.

**Key principle:** YOLO only answers "is there a product here?". Product identity always comes from Stage 3. Action recognition is a rule-based state machine, not an LSTM/Transformer.

### Products in this build

| Class name | Product | Shelf slot |
|---|---|---|
| `chings_manchurian` | Ching's Veg Manchurian Masala Mix | slot_0 |
| `chings_hakka` | Ching's Hakka Noodles Masala | slot_1 |
| `homelite_matchbox` | Homelite Matchbox | slot_2 |
| `vaseline_jelly` | Vaseline Petroleum Jelly | slot_3 |

Class names must match in three places: `PRODUCT_CLASSES` in `src/config.py`, the folder names in `data/classifier_dataset/`, and `data/inventory.csv`.

---

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.10 – 3.12 |
| Detection | Ultralytics YOLOv8n |
| DL framework | PyTorch + torchvision |
| Classifier | MobileNetV3-Small (ImageNet pretrained, fine-tuned) |
| Tracker | Custom IoU + centroid (SciPy Hungarian) |
| Backend | FastAPI + Uvicorn, SQLite |
| Dashboard | Streamlit |
| Video I/O | OpenCV |
| Logging | JSONL per stage |

---

## Quick start (2 minutes, no camera or models)

```bash
python -m venv venv
# Windows: venv\Scripts\activate      macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

python run_demo.py --mock --reset
```

This starts the backend and the dashboard (http://localhost:8501) and runs a **scripted mock scenario**. Products move from their shelf slot to the basket and back, so the cart, inventory, activity log and low-stock alerts all update. Use it to check the setup, or as a backup during a presentation.

> For an NVIDIA GPU, install `torch`/`torchvision` from https://pytorch.org/get-started/locally/ before `pip install -r requirements.txt`.

---

## Full setup

### 1. Classifier dataset (included)

`data/classifier_dataset/` already holds the four-product dataset (1,182 frames at 224×224, split train/valid/test). It was exported from Roboflow in folder format and imported with:

```bash
python src/data_tools/prepare_classifier_dataset.py --src Four_Product_Video_Frames.zip --clean
```

The script renames Roboflow folders (e.g. `Haka_Noodles_Masala`) to the class names above. To add products, extend `ALIASES` in the script, `PRODUCT_CLASSES` and `data/inventory.csv`.

To capture new images from the webcam instead:
```bash
python src/data_tools/capture_images.py --class chings_hakka     # SPACE = save, q = next
```

### 2. Train the classifier + open-set centroids

```bash
python src/classification/train_classifier.py
python src/classification/openset.py
```

Outputs: `models/classifier_best.pt`, `models/class_indices.json`, `models/class_centroids.npy`, `logs/classifier_eval.json` (val + test accuracy, confusion matrices).

### 3. Detector dataset (free auto-labelling, no manual boxes)

YOLO needs **bounding boxes**, which the classifier dataset doesn't have. You don't have to draw them yourself:

1. Record a 2–3 minute video **from the actual demo camera position**, showing the shelf, the basket, and hands picking and placing each product.
2. Auto-label it locally with the free zero-shot **YOLO-World** model:
   ```bash
   python src/data_tools/auto_label.py --src my_demo_video.mp4
   ```
   This writes `data/yolo_dataset/` (train = first 80% of the video, val = last 20%) plus **preview sheets** in `data/yolo_dataset/previews/`. Open a few to check the boxes: green is product, blue is hand.
   - Products missed → `--conf 0.05`, or edit `WORLD_PROMPTS` in `src/config.py`
   - Cloth or table boxed → `--conf 0.2`
   - For hand-checking the boxes, add `--roboflow-zip` and upload the zip to Roboflow as an already-annotated dataset
3. Train the fast detector:
   ```bash
   python src/detection/train_yolo.py            # → models/yolo_best.pt
   ```

**Shortcut:** skip training entirely and run the zero-shot detector live with `--detector world`. It's slower (roughly 5–10 FPS on a laptop CPU) and less precise, but needs no dataset. Without `models/yolo_best.pt`, the pipeline uses it automatically.

**Manual alternative:** label ~100 frames in Roboflow (classes `product`, `hand`), export as YOLOv8, then run:
```bash
python src/data_tools/import_yolo_dataset.py --src roboflow_yolov8.zip --clean
```

### 4. Calibrate zones

Point the camera at the shelf and basket, then run:

```bash
python calibrate.py
```

Click the shelf ROI, the basket ROI and one ROI per shelf slot. This writes `config.json`. Without it, the defaults in `src/config.py` are used (for 1280×720 frames).

### 5. Run the live demo

```bash
python run_demo.py --reset --anomaly
```

Or run each process in its own terminal:

```bash
uvicorn src.backend.main:app --host 127.0.0.1 --port 8000
streamlit run dashboard/app.py
python -m src.pipeline.run_pipeline --anomaly          # add --no-display to hide the OpenCV window
```

Pipeline options: `--detector world` (zero-shot, no YOLO training), `--source 1` (another webcam) or `--source demo.mp4` (replay a recording, which is a good fallback if the live camera misbehaves). Other flags: `--session NAME`, `--mock`, `--no-backend`, `--max-frames N`.

API docs: http://127.0.0.1:8000/docs

---

## Demo procedure

Run all six scenarios in one session. Use **♻️ Reset inventory & carts** in the dashboard sidebar between rehearsals.

| # | Scenario | Expected |
|---|---|---|
| 1 | Pick product → place in basket | Cart updates, stock decrements |
| 2 | Take product from basket → back to its slot | Item leaves cart, stock restored |
| 3 | Pick product → hold/conceal > 10 s (run with `--anomaly`) | `ANOMALY` alert |
| 4 | Show an object that isn't one of the 4 products | `UNKNOWN_PRODUCT` alert, no cart update |
| 5 | Put a product in the wrong shelf slot | `MISPLACED_PRODUCT` alert after 10 frames |
| 6 | Pick one product repeatedly until stock ≤ 3 | `LOW_STOCK` alert |

---

## Running tests

```bash
pytest                                 # no camera or weights required
pytest tests/test_state_machine.py -v
pytest --cov=src --cov-report=term-missing
```

Hardware-dependent checks (YOLO accuracy, classifier accuracy, live pick/return) are manual. See the pass/fail criteria in `IMPLEMENTATION_GUIDE.md`.

---

## Known limitations

- **Classifier test accuracy is optimistic.** Each product was recorded as one video and the frames were split randomly, so test frames are near-duplicates of training frames. For an honest number, record one more short video per product (different place/lighting) and evaluate on it.
- The classifier was trained on hand-held close-ups. At runtime it sees YOLO crops from the demo camera, so adding a few crops from the real setup improves robustness.
- Open-set rejection is threshold-based (`OPENSET_DIST_THRESH`). Tune it with a couple of non-product objects.
- Single camera, single basket, one shopper at a time.

---

## Configuration Reference

All tunable values are in `src/config.py`:

| Constant | Default | Description |
|---|---|---|
| `PRODUCT_CLASSES` | 4 classes | Product class names (must match dataset folders + inventory.csv) |
| `YOLO_CONF_THRESH` | 0.40 | YOLO minimum detection confidence |
| `OPENSET_DIST_THRESH` | 0.35 | Cosine distance threshold for UNKNOWN |
| `TRACK_IOU_THRESH` | 0.30 | Minimum IoU to associate detection to track |
| `TRACK_MAX_AGE` | 15 | Frames before a track expires |
| `ANOMALY_TIMEOUT_SEC` | 10.0 | Seconds held without basket → anomaly |
| `MISPLACED_CONSECUTIVE_FRAMES` | 10 | Frames before misplacement alert fires |
| `LOW_STOCK_DEFAULT_THRESHOLD` | 3 | Default stock level for low-stock alert |
| `CAMERA_INDEX` | 0 | OpenCV camera index |
| `TARGET_FPS` | 15 | Pipeline target frame rate |
| `CLASSIFY_EVERY_N_FRAMES` | 2 | Re-classify each track every N frames |
| `CLASS_VOTE_WINDOW` | 7 | Majority vote window for a track's label |
| `UNKNOWN_ALERT_FRAMES` | 8 | Consecutive UNKNOWN votes before an alert |
| `DEFAULT_SESSION_ID` | demo_session | Cart session used by pipeline + dashboard |

Prices, starting stock and low-stock thresholds are in `data/inventory.csv`.

---

## Project Structure

```
DeepRetail/
├── run_demo.py                          # one command: backend + dashboard + pipeline
├── calibrate.py                         # click shelf / basket / slot ROIs → config.json
├── data/
│   ├── inventory.csv                    # product catalog: price, stock, threshold, slot
│   ├── classifier_dataset/{train,valid,test}/<class>/   # classifier images (included)
│   ├── yolo_dataset/                    # detector images + labels (import from Roboflow)
│   └── raw_images/                      # webcam captures
├── models/                              # trained weights (generated, git-ignored)
├── src/
│   ├── config.py                        # all constants and thresholds
│   ├── detection/          yolo_infer.py, train_yolo.py
│   ├── tracking/           tracker.py
│   ├── classification/     train_classifier.py, infer_classifier.py, openset.py
│   ├── logic/              zones.py, state_machine.py, shelf_monitor.py
│   ├── pipeline/           run_pipeline.py, api_bridge.py
│   ├── backend/            main.py (FastAPI), db.py (SQLite), analytics.py, schemas.py
│   └── data_tools/         auto_label.py, prepare_classifier_dataset.py, import_yolo_dataset.py,
│                           capture_images.py, create_yolo_yaml.py
├── dashboard/app.py                     # Streamlit dashboard
├── logs/                                # JSONL logs per stage (git-ignored)
├── tests/                               # pytest suite
├── start_mock_demo.bat                  # 1-click Windows launcher for mock demo
├── start_live_demo.bat                  # 1-click Windows launcher for live webcam demo
├── run_tests.bat                        # 1-click Windows test suite launcher
├── Dockerfile                           # Production container build
├── docker-compose.yml                   # Multi-container orchestration (backend + dashboard)
├── .env.example                         # Environment configuration template
├── IMPLEMENTATION_GUIDE.md
└── requirements.txt
```

---

## Deployment & Production Options

### Option 1: 1-Click Windows Launchers
- **`start_mock_demo.bat`**: Double-click to instantly run the FastAPI backend, Streamlit dashboard, and scripted mock scenario with zero configuration or camera requirements.
- **`start_live_demo.bat`**: Double-click to launch live retail intelligence with webcam detection and tracking.
- **`run_tests.bat`**: Double-click to execute the automated pytest suite (108 unit/integration tests).

### Option 2: Docker & Docker Compose
To deploy the backend and dashboard in isolated containers:
```bash
docker-compose up --build
```
- **FastAPI Backend**: `http://localhost:8000` (Docs at `http://localhost:8000/docs`, Health check at `http://localhost:8000/health`)
- **Streamlit Dashboard**: `http://localhost:8501`

### Option 3: Manual Multi-Process Production
```bash
# 1. Start Backend
uvicorn src.backend.main:app --host 0.0.0.0 --port 8000

# 2. Start Dashboard
streamlit run dashboard/app.py --server.port 8501

# 3. Start Pipeline
python -m src.pipeline.run_pipeline --source 0 --anomaly
```


---

## Troubleshooting

| Problem | Fix |
|---|---|
| `YOLO weights not found` | Auto-label (step 3) + `train_yolo.py`, or run with `--detector world` |
| YOLO-World error about CLIP / text encoder | `pip install git+https://github.com/ultralytics/CLIP.git` (needs git installed) |
| `Classifier weights not found` | `python src/classification/train_classifier.py` |
| No centroids / open-set disabled | `python src/classification/openset.py` after training |
| Dashboard says *Backend OFFLINE* | Start `uvicorn src.backend.main:app --port 8000` (or use `run_demo.py`) |
| Dashboard feed says *Waiting for pipeline frame* | Pipeline isn't running, or was started with `--no-backend` |
| Pick detected in the OpenCV window but cart empty | Session mismatch (sidebar warns you) or class name not in `inventory.csv` (backend prints a warning) |
| Everything shows UNKNOWN | Increase `OPENSET_DIST_THRESH` |
| Unknown objects accepted as products | Decrease `OPENSET_DIST_THRESH` |
| Too many anomaly false positives | Increase `ANOMALY_TIMEOUT_SEC` |
| Misplacement alerts for brief movements | Increase `MISPLACED_CONSECUTIVE_FRAMES` |
| Camera not opening | `--source 1`, or change `CAMERA_INDEX` in `src/config.py` |
| Port 8000 in use | Change `API_PORT` in `src/config.py` |

---

## Logging

Every stage writes to `logs/`: `detection.jsonl`, `tracking.jsonl`, `classification.jsonl`, `events.jsonl`, `alerts.jsonl`, and `classifier_eval.json`. Together they allow full session replay for debugging without a live camera.
