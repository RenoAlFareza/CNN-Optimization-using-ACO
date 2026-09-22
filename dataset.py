"""MNIST loading and deterministic stratified splitting."""

from __future__ import annotations

from dataclasses import dataclass
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


def load_mnist(dataset_seed: int = 2024) -> DatasetBundle:
    """Load MNIST through Keras and create the fixed 50k/10k/10k split."""
    try:
        from tensorflow.keras.datasets import mnist
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required to load MNIST. Install it with "
            "`pip install tensorflow` before running a CNN experiment."
        ) from error

    (x_full, y_full), (x_test, y_test) = mnist.load_data()
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
