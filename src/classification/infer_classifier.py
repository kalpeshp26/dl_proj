"""
DeepRetail — Classifier Inference (Stage 3)

Fixed output contract:
    ClassificationResult(
        class_name : str    — product name or "UNKNOWN"
        class_idx  : int    — numeric class index (-1 if UNKNOWN)
        conf       : float  — softmax confidence
        embedding  : np.ndarray shape (EMBEDDING_DIM,)
        is_unknown : bool
    )

Usage:
    from src.classification.infer_classifier import ProductClassifier
    clf = ProductClassifier()
    result = clf.classify(crop_bgr_image)
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    CLASSIFIER_MODEL_PATH, CLASS_CENTROIDS_PATH, MODELS_DIR,
    CLASSIFIER_IMG_SIZE, OPENSET_DIST_THRESH, EMBEDDING_DIM,
    LOG_CLASSIFICATION,
)


# ─── Output contract ──────────────────────────────────────────────────────────

@dataclass
class ClassificationResult:
    class_name: str               # product name or "UNKNOWN"
    class_idx: int                # -1 for UNKNOWN
    conf: float                   # softmax score of top-1 class
    embedding: np.ndarray = field(repr=False)  # penultimate embedding
    is_unknown: bool = False
    openset_dist: float = 0.0     # cosine distance to nearest centroid

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "class_idx": self.class_idx,
            "conf": round(self.conf, 4),
            "is_unknown": self.is_unknown,
            "openset_dist": round(self.openset_dist, 4),
        }


# ─── Preprocessing ────────────────────────────────────────────────────────────

def get_inference_transform(img_size: int = CLASSIFIER_IMG_SIZE) -> T.Compose:
    return T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),   # must match get_val_transforms() in training
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ─── Embedding extractor (penultimate layer hook) ────────────────────────────

class EmbeddingExtractor(nn.Module):
    """
    Wraps MobileNetV3-Small; forward() returns (logits, embedding).
    The embedding is extracted from the penultimate representation
    (output of features + avgpool, before the final Linear classifier).
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        # MobileNetV3 structure: .features, .avgpool, .classifier (Sequential)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        # All classifier layers EXCEPT the last linear
        self.pre_head = backbone.classifier[:-1]   # Hardswish + Dropout
        self.head = backbone.classifier[-1]         # final Linear

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        emb = self.pre_head(x)          # (batch, EMBEDDING_DIM)
        logits = self.head(emb)         # (batch, num_classes)
        return logits, emb


# ─── Classifier ──────────────────────────────────────────────────────────────

class ProductClassifier:
    """
    Loads trained MobileNetV3-Small weights and performs:
      1. Top-1 softmax classification
      2. Embedding extraction
      3. Open-set check via Stage 3b (cosine distance)

    Open-set logic can optionally delegate to openset.py, but
    for inference simplicity we replicate the distance check here
    using pre-loaded centroids.
    """

    def __init__(
        self,
        model_path: Path = CLASSIFIER_MODEL_PATH,
        centroids_path: Optional[Path] = CLASS_CENTROIDS_PATH,
        openset_thresh: float = OPENSET_DIST_THRESH,
        device: str = "",
        log: bool = True,
    ) -> None:
        self._log = log
        self._thresh = openset_thresh
        self._frame_count = 0

        # ── Device ───────────────────────────────────────────────
        if device:
            self._device = torch.device(device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        # ── Load class index map ──────────────────────────────────
        idx_path = MODELS_DIR / "class_indices.json"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"Class index map not found: {idx_path}\n"
                "Train the classifier first: python src/classification/train_classifier.py"
            )
        with open(idx_path) as f:
            idx_data = json.load(f)
        self._idx_to_class: dict[int, str] = {
            int(k): v for k, v in idx_data["idx_to_class"].items()
        }
        self._num_classes = len(self._idx_to_class)

        # ── Load model ───────────────────────────────────────────
        if not model_path.exists():
            raise FileNotFoundError(
                f"Classifier weights not found: {model_path}\n"
                "Train first: python src/classification/train_classifier.py"
            )
        import torchvision.models as models
        backbone = models.mobilenet_v3_small(weights=None)
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, self._num_classes)
        backbone.load_state_dict(
            torch.load(model_path, map_location=self._device)
        )
        self._model = EmbeddingExtractor(backbone).to(self._device)
        self._model.eval()
        print(f"[Classifier] Loaded {model_path.name} ({self._num_classes} classes) on {self._device}")

        # ── Load centroids (optional — needed for open-set) ──────
        self._centroids: Optional[np.ndarray] = None
        if centroids_path and centroids_path.exists():
            self._centroids = np.load(str(centroids_path))  # (num_classes, dim)
            print(f"[Classifier] Loaded class centroids {self._centroids.shape}")
        else:
            print("[Classifier] WARNING: No class centroids found. "
                  "Open-set rejection disabled. "
                  "Run: python src/classification/openset.py  to compute centroids.")

        self._transform = get_inference_transform()
        LOG_CLASSIFICATION.parent.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def classify(
        self,
        crop_bgr: np.ndarray,
        frame_id: Optional[int] = None,
        track_id: Optional[int] = None,
    ) -> ClassificationResult:
        """
        Classify a BGR crop from the camera.
        Returns ClassificationResult with product name or UNKNOWN.
        """
        if frame_id is None:
            frame_id = self._frame_count
        self._frame_count += 1

        if crop_bgr is None or crop_bgr.size == 0:
            return ClassificationResult(
                class_name="UNKNOWN", class_idx=-1,
                conf=0.0, embedding=np.zeros(EMBEDDING_DIM),
                is_unknown=True, openset_dist=1.0,
            )

        # BGR → RGB
        crop_rgb = crop_bgr[:, :, ::-1].copy()

        # Multi-angle / multi-orientation batch for robust recognition from any angle:
        # 1. Normal orientation
        # 2. Horizontal flip (mirrored / reverse view)
        # 3. 90 deg clockwise (vertical/sideways)
        # 4. 90 deg counter-clockwise (vertical/sideways)
        views = [
            crop_rgb,
            cv2.flip(crop_rgb, 1),
            cv2.rotate(crop_rgb, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(crop_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE),
        ]
        tensors = torch.stack([self._transform(v) for v in views]).to(self._device)

        with torch.no_grad():
            logits, emb = self._model(tensors)
            probs = torch.softmax(logits, dim=1)

        # Pick the orientation that gives the highest softmax confidence
        max_confs, max_indices = probs.max(dim=1)
        best_view_idx = int(max_confs.argmax().item())

        conf = float(max_confs[best_view_idx].item())
        pred_idx = int(max_indices[best_view_idx].item())
        emb_np = emb[best_view_idx].cpu().numpy()

        # Open-set check using the best-aligned orientation
        openset_dist = 0.0
        is_unknown = False
        if self._centroids is not None:
            openset_dist = self._cosine_dist_to_nearest(emb_np)
            eff_thresh = max(self._thresh, 0.52) if conf >= 0.85 else self._thresh
            is_unknown = openset_dist > eff_thresh

        class_name = "UNKNOWN" if is_unknown else self._idx_to_class.get(pred_idx, "UNKNOWN")
        class_idx = -1 if is_unknown else pred_idx

        result = ClassificationResult(
            class_name=class_name,
            class_idx=class_idx,
            conf=round(conf, 4),
            embedding=emb_np,
            is_unknown=is_unknown,
            openset_dist=round(float(openset_dist), 4),
        )

        if self._log:
            self._write_log(frame_id, track_id, result)

        return result

    def _cosine_dist_to_nearest(self, emb: np.ndarray) -> float:
        """Cosine distance (1 − similarity) to the nearest class centroid."""
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        centroids_norm = self._centroids / (
            np.linalg.norm(self._centroids, axis=1, keepdims=True) + 1e-8
        )
        sims = centroids_norm @ emb_norm          # (num_classes,)
        return float(1.0 - sims.max())

    def _write_log(
        self,
        frame_id: int,
        track_id: Optional[int],
        result: ClassificationResult,
    ) -> None:
        entry = {
            "ts": time.time(),
            "frame_id": frame_id,
            "track_id": track_id,
            **result.to_dict(),
        }
        with open(LOG_CLASSIFICATION, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_embedding(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Return raw embedding for a crop (used by openset.py centroid computation)."""
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros(EMBEDDING_DIM)
        crop_rgb = crop_bgr[:, :, ::-1].copy()
        tensor = self._transform(crop_rgb).unsqueeze(0).to(self._device)
        with torch.no_grad():
            _, emb = self._model(tensor)
        return emb[0].cpu().numpy()
