"""
DeepRetail — Open-Set / Unknown Product Detection (Stage 3b)

After classifier training, run this script once to:
    1. Pass all training-set images through the embedding extractor
    2. Compute per-class centroid in embedding space
    3. Save centroids to models/class_centroids.npy

At inference, cosine distance from a new embedding to its nearest centroid
determines whether the product is KNOWN or UNKNOWN.

Usage:
    python src/classification/openset.py
    python src/classification/openset.py --thresh 0.40   (evaluate threshold)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import (
    CLASSIFIER_DATASET_DIR, MODELS_DIR, CLASS_CENTROIDS_PATH,
    OPENSET_DIST_THRESH, EMBEDDING_DIM,
)


def load_classifier_for_embedding():
    """Load classifier in embedding-extraction mode (no centroid needed yet)."""
    # Import here to avoid circular deps before weights exist
    from src.classification.infer_classifier import ProductClassifier
    return ProductClassifier(centroids_path=None, log=False)


def compute_centroids(clf) -> dict[str, np.ndarray]:
    """
    Compute the mean embedding (centroid) for each product class
    from the training images.

    Returns: {class_name: centroid_ndarray}
    """
    class_dirs = [d for d in CLASSIFIER_DATASET_DIR.iterdir() if d.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(
            f"No class subdirectories in {CLASSIFIER_DATASET_DIR}. "
            "Run data collection first."
        )

    centroids: dict[str, list[np.ndarray]] = {}
    total_images = 0

    for cls_dir in sorted(class_dirs):
        cls_name = cls_dir.name
        images = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        if not images:
            print(f"  [skip] {cls_name}: no images found")
            continue

        embeddings_for_class: list[np.ndarray] = []
        print(f"  Processing {cls_name}: {len(images)} images")

        for img_path in tqdm(images, desc=f"  {cls_name}", leave=False):
            try:
                img_bgr = _load_bgr(img_path)
                emb = clf.get_embedding(img_bgr)
                embeddings_for_class.append(emb)
                total_images += 1
            except Exception as e:
                print(f"    [warn] Cannot load {img_path.name}: {e}")

        if embeddings_for_class:
            centroids[cls_name] = embeddings_for_class

    print(f"\n[Open-Set] Processed {total_images} images across {len(centroids)} classes")
    return {k: np.array(v) for k, v in centroids.items()}


def _load_bgr(path: Path) -> np.ndarray:
    """Load image as BGR numpy array (compatible with OpenCV pipeline)."""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"Cannot read {path}")
    return img


def save_centroids(class_embeddings: dict[str, np.ndarray]) -> None:
    """
    Compute mean centroid per class and save to class_centroids.npy.
    Also saves a metadata JSON for human inspection.
    """
    # Load class_to_idx to maintain consistent ordering
    idx_path = MODELS_DIR / "class_indices.json"
    if not idx_path.exists():
        raise FileNotFoundError(
            f"class_indices.json not found at {idx_path}. "
            "Run classifier training first."
        )
    with open(idx_path) as f:
        idx_data = json.load(f)
    class_to_idx = idx_data["class_to_idx"]

    num_classes = len(class_to_idx)
    centroids = np.zeros((num_classes, EMBEDDING_DIM), dtype=np.float32)

    for cls_name, embs in class_embeddings.items():
        if cls_name not in class_to_idx:
            print(f"  [warn] '{cls_name}' not in class_to_idx, skipping")
            continue
        idx = class_to_idx[cls_name]
        mean_emb = embs.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        centroids[idx] = mean_emb / (norm + 1e-8)   # unit-length centroid
        print(f"  Centroid [{idx}] {cls_name}: mean of {len(embs)} embeddings, norm={norm:.4f}")

    np.save(str(CLASS_CENTROIDS_PATH), centroids)
    print(f"\n[Open-Set] Centroids saved → {CLASS_CENTROIDS_PATH} (shape: {centroids.shape})")

    # Human-readable metadata
    meta = {
        "num_classes": num_classes,
        "embedding_dim": EMBEDDING_DIM,
        "class_to_idx": class_to_idx,
        "openset_dist_thresh": OPENSET_DIST_THRESH,
        "per_class_sample_counts": {
            k: int(len(v)) for k, v in class_embeddings.items()
        },
    }
    meta_path = MODELS_DIR / "centroids_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Open-Set] Metadata → {meta_path}")


def evaluate_threshold(
    clf,
    class_embeddings: dict[str, np.ndarray],
    centroids: np.ndarray,
    class_to_idx: dict[str, int],
    threshold: float,
) -> None:
    """
    Print per-class false-reject rates (known objects flagged as UNKNOWN).
    Use held-out data for unknown-object true-positive evaluation.
    """
    print(f"\n[Threshold Eval] threshold = {threshold}")
    print(f"  False-reject check (should be ≤10% per class):")

    total, rejected = 0, 0
    for cls_name, embs in class_embeddings.items():
        if cls_name not in class_to_idx:
            continue
        idx = class_to_idx[cls_name]
        cls_rejected = 0
        for emb in embs:
            dist = _cosine_dist_to_nearest(emb, centroids)
            if dist > threshold:
                cls_rejected += 1
        rate = cls_rejected / len(embs) if embs.size else 0
        flag = " ← HIGH" if rate > 0.10 else ""
        print(f"    {cls_name:20s}  reject_rate={rate:.3f} ({cls_rejected}/{len(embs)}){flag}")
        total += len(embs)
        rejected += cls_rejected

    overall = rejected / total if total else 0
    print(f"  Overall false-reject rate: {overall:.3f} ({rejected}/{total})")
    if overall > 0.10:
        print("  [!] Consider INCREASING threshold to reduce false rejects.")
    else:
        print("  [OK] False-reject rate within target (≤10%).")


def _cosine_dist_to_nearest(emb: np.ndarray, centroids: np.ndarray) -> float:
    emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
    sims = centroids @ emb_norm
    return float(1.0 - sims.max())


def main() -> None:
    p = argparse.ArgumentParser(description="Compute open-set class centroids")
    p.add_argument("--thresh", type=float, default=OPENSET_DIST_THRESH,
                   help="Threshold to evaluate (doesn't change config)")
    p.add_argument("--no-eval", action="store_true",
                   help="Skip threshold evaluation (just compute centroids)")
    args = p.parse_args()

    print("[Open-Set] Loading classifier for embedding extraction...")
    clf = load_classifier_for_embedding()

    print("[Open-Set] Computing embeddings from training set...")
    class_embeddings = compute_centroids(clf)

    save_centroids(class_embeddings)

    if not args.no_eval:
        idx_path = MODELS_DIR / "class_indices.json"
        with open(idx_path) as f:
            class_to_idx = json.load(f)["class_to_idx"]
        centroids = np.load(str(CLASS_CENTROIDS_PATH))
        evaluate_threshold(clf, class_embeddings, centroids, class_to_idx, args.thresh)


if __name__ == "__main__":
    main()
