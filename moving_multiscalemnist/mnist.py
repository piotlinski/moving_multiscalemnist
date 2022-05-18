"""Tools for loading MNIST dataset files."""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def load_images(data_dir: Path, images_file: str) -> np.ndarray:
    """Load data from image file."""
    with data_dir.joinpath(images_file).open() as fp:
        loaded = np.fromfile(file=fp, dtype=np.uint8)
        return loaded[16:].reshape((-1, 28, 28))


def load_labels(data_dir: Path, labels_file: str) -> np.ndarray:
    """Load data from labels file."""
    with data_dir.joinpath(labels_file).open() as fp:
        loaded = np.fromfile(file=fp, dtype=np.uint8)
        return loaded[8:]


def fetch_mnist(data_dir: str = "mnist") -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load MNIST dataset."""
    path = Path(data_dir)
    return {
        "train": (
            load_images(path, "train-images"),
            load_labels(path, "train-labels"),
        ),
        "test": (
            load_images(path, "test-images"),
            load_labels(path, "test-labels"),
        ),
    }
