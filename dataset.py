"""Local MNIST IDX loading and deterministic stratified splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    split_id: str


def _stratified_indices(labels: np.ndarray, validation_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    classes = np.unique(labels)
    per_class = validation_size // len(classes)
    remainder = validation_size % len(classes)

    for position, label in enumerate(classes):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = per_class + (1 if position < remainder else 0)
        validation_indices.extend(indices[:count].tolist())
        train_indices.extend(indices[count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return np.asarray(train_indices), np.asarray(validation_indices)


def _find_file(data_dir: Path, names: tuple[str, ...], description: str) -> Path:
    candidates = [data_dir / name for name in names]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = " or ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find MNIST {description}. Expected one of: {expected}"
    )


def _read_idx_labels(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        header = file.read(8)
        if len(header) != 8:
            raise ValueError(f"MNIST label file is truncated: {path}")
        magic, size = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label magic number in {path}: {magic}")
        labels = np.frombuffer(file.read(), dtype=np.uint8)
    if len(labels) != size:
        raise ValueError(
            f"MNIST label count mismatch in {path}: header={size}, data={len(labels)}"
        )
    return labels


def _read_idx_images(path: Path) -> np.ndarray:
    with path.open("rb") as file:
        header = file.read(16)
        if len(header) != 16:
            raise ValueError(f"MNIST image file is truncated: {path}")
        magic, size, rows, columns = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image magic number in {path}: {magic}")
        image_data = np.frombuffer(file.read(), dtype=np.uint8)
    expected_size = size * rows * columns
    if len(image_data) != expected_size:
        raise ValueError(
            f"MNIST image count mismatch in {path}: header={expected_size}, "
            f"data={len(image_data)}"
        )
    return image_data.reshape(size, rows, columns)


def _load_idx_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images_path = _find_file(
        data_dir,
        (
            "train-images.idx3-ubyte",
            "train-images-idx3-ubyte/train-images-idx3-ubyte",
            "train-images-idx3-ubyte",
        ),
        "training images",
    )
    train_labels_path = _find_file(
        data_dir,
        (
            "train-labels.idx1-ubyte",
            "train-labels-idx1-ubyte/train-labels-idx1-ubyte",
            "train-labels-idx1-ubyte",
        ),
        "training labels",
    )
    test_images_path = _find_file(
        data_dir,
        (
            "t10k-images.idx3-ubyte",
            "t10k-images-idx3-ubyte/t10k-images-idx3-ubyte",
            "t10k-images-idx3-ubyte",
        ),
        "test images",
    )
    test_labels_path = _find_file(
        data_dir,
        (
            "t10k-labels.idx1-ubyte",
            "t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte",
            "t10k-labels-idx1-ubyte",
        ),
        "test labels",
    )
    x_train = _read_idx_images(train_images_path)
    y_train = _read_idx_labels(train_labels_path)
    x_test = _read_idx_images(test_images_path)
    y_test = _read_idx_labels(test_labels_path)
    if len(x_train) != len(y_train):
        raise ValueError("MNIST training image and label counts do not match")
    if len(x_test) != len(y_test):
        raise ValueError("MNIST test image and label counts do not match")
    return x_train, y_train, x_test, y_test


def load_mnist(dataset_seed: int = 2024, data_dir: str | Path = "data") -> DatasetBundle:
    """Load local MNIST IDX files and create the fixed 50k/10k/10k split."""
    data_path = Path(data_dir).expanduser()
    x_full, y_full, x_test, y_test = _load_idx_dataset(data_path)
    if len(x_full) != 60_000 or len(x_test) != 10_000:
        raise ValueError(
            "This experiment requires the standard MNIST sizes: "
            "60,000 training images and 10,000 test images"
        )
    train_indices, validation_indices = _stratified_indices(
        y_full, validation_size=10_000, seed=dataset_seed
    )

    def prepare(images: np.ndarray) -> np.ndarray:
        return images.astype("float32")[..., np.newaxis] / 255.0

    return DatasetBundle(
        x_train=prepare(x_full[train_indices]),
        y_train=y_full[train_indices].astype("int64"),
        x_validation=prepare(x_full[validation_indices]),
        y_validation=y_full[validation_indices].astype("int64"),
        x_test=prepare(x_test),
        y_test=y_test.astype("int64"),
        split_id=f"mnist-stratified-50000-10000-seed-{dataset_seed}",
    )


def concatenate_development_data(bundle: DatasetBundle) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([bundle.x_train, bundle.x_validation], axis=0),
        np.concatenate([bundle.y_train, bundle.y_validation], axis=0),
    )
