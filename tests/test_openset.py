"""
Tests for Stage 3b: Open-set detection using cosine distance.
Tests work without trained weights — uses synthetic embeddings.
"""

import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classification.openset import _cosine_dist_to_nearest


class TestCosineDistance:

    def test_identical_vector_zero_distance(self):
        v = np.array([1.0, 0.0, 0.0])
        centroids = np.array([[1.0, 0.0, 0.0]])
        dist = _cosine_dist_to_nearest(v, centroids)
        assert dist == pytest.approx(0.0, abs=1e-5)

    def test_orthogonal_vectors_distance_one(self):
        v = np.array([1.0, 0.0, 0.0])
        centroids = np.array([[0.0, 1.0, 0.0]])
        dist = _cosine_dist_to_nearest(v, centroids)
        assert dist == pytest.approx(1.0, abs=1e-5)

    def test_opposite_vectors_distance_two(self):
        v = np.array([1.0, 0.0, 0.0])
        centroids = np.array([[-1.0, 0.0, 0.0]])
        dist = _cosine_dist_to_nearest(v, centroids)
        assert dist == pytest.approx(2.0, abs=1e-5)

    def test_nearest_centroid_selected(self):
        v = np.array([1.0, 0.0, 0.0])
        centroids = np.array([
            [0.0, 1.0, 0.0],  # dist=1.0
            [0.9, 0.1, 0.0],  # dist≈0.04 (closer)
            [-1.0, 0.0, 0.0], # dist=2.0
        ])
        dist = _cosine_dist_to_nearest(v, centroids)
        # Should pick the nearest centroid (approx dist < 0.12 after normalization)
        assert dist < 0.12, f"Expected nearest centroid to be selected, got dist={dist}"

    def test_known_product_below_threshold(self):
        """A product from training should have distance below threshold."""
        from src.config import OPENSET_DIST_THRESH
        # Simulate: centroid is unit [1,0,...,0], product embedding is very close
        dim = 576
        centroid = np.zeros(dim)
        centroid[0] = 1.0
        embedding = centroid + np.random.randn(dim) * 0.01  # tiny perturbation
        embedding /= np.linalg.norm(embedding)

        centroids = centroid.reshape(1, -1)
        dist = _cosine_dist_to_nearest(embedding, centroids)
        assert dist < OPENSET_DIST_THRESH, (
            f"Known product flagged as UNKNOWN (dist={dist:.4f} > {OPENSET_DIST_THRESH})"
        )

    def test_unknown_product_above_threshold(self):
        """A random vector (unknown product) should have high distance."""
        from src.config import OPENSET_DIST_THRESH
        dim = 576
        np.random.seed(99)
        # Centroid points in +x direction
        centroid = np.zeros(dim)
        centroid[0] = 1.0
        centroids = centroid.reshape(1, -1)

        # Random orthogonal-ish vector
        unknown_emb = np.random.randn(dim)
        unknown_emb /= np.linalg.norm(unknown_emb)

        dist = _cosine_dist_to_nearest(unknown_emb, centroids)
        # In high-dim space, random vectors are nearly orthogonal → dist ≈ 1.0
        assert dist > 0.3, (
            f"Unknown product NOT rejected (dist={dist:.4f}). "
            "Threshold may be too aggressive."
        )

    def test_batch_of_known_products_low_false_reject(self):
        """False-reject rate on known products should be ≤ 10%."""
        from src.config import OPENSET_DIST_THRESH
        dim = 576
        np.random.seed(42)

        num_classes = 8
        # Synthetic centroids: random unit vectors (realistic high-dim geometry)
        raw_centroids = np.random.randn(num_classes, dim).astype(np.float32)
        centroids = raw_centroids / np.linalg.norm(raw_centroids, axis=1, keepdims=True)

        # Generate 20 "known" product samples per class:
        # Small perturbations from centroid (tighter cluster than the default threshold)
        # Use perturbation scale 0.02 → products stay close to their centroid
        false_rejects = 0
        total = 0
        for i in range(num_classes):
            for _ in range(20):
                noise = np.random.randn(dim).astype(np.float32) * 0.02
                emb = centroids[i] + noise
                emb /= np.linalg.norm(emb)
                dist = _cosine_dist_to_nearest(emb, centroids)
                if dist > OPENSET_DIST_THRESH:
                    false_rejects += 1
                total += 1

        rate = false_rejects / total
        assert rate <= 0.10, (
            f"False-reject rate {rate:.3f} exceeds 10% target. "
            "Decrease OPENSET_DIST_THRESH or increase perturbation tightness."
        )
