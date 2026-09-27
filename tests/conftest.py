"""
DeepRetail — pytest configuration and shared fixtures.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Make sure the package root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(scope="session")
def tmp_project(tmp_path_factory):
    """A temporary directory tree mimicking the project structure."""
    base = tmp_path_factory.mktemp("deepretail_test")
    (base / "models").mkdir()
    (base / "logs").mkdir()
    (base / "data").mkdir()
    return base


@pytest.fixture()
def dummy_frame():
    """A 720×1280 BGR frame filled with random noise."""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture()
def dummy_crop():
    """A 224×224 BGR crop (classifier input size)."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture()
def dummy_embedding():
    """Random unit-length embedding vector."""
    v = np.random.randn(576).astype(np.float32)
    return v / np.linalg.norm(v)
