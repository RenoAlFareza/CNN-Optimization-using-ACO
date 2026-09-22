"""Candidate CNN evaluation and deterministic trial metadata."""

from __future__ import annotations

import platform
import time
from typing import Any

import numpy as np

from cnn_model import build_model
from config import ExperimentConfig
from dataset import DatasetBundle
from evaluation_contract import EvaluationResult, candidate_id, canonical_json, trial_seed


__all__ = [
    "EvaluationResult",
    "candidate_id",
    "canonical_json",
    "trial_seed",
    "evaluate_candidate",
    "train_final_model",
]


def environment_metadata() -> dict[str, str]:
    metadata = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    try:
        import tensorflow as tf

        metadata["tensorflow_version"] = tf.__version__
        metadata["keras_version"] = str(tf.keras.__version__)
    except (ImportError, AttributeError):
        metadata["tensorflow_version"] = "unavailable"
        metadata["keras_version"] = "unavailable"
    return metadata


def seed_tensorflow(seed: int, deterministic: bool) -> dict[str, Any]:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    warning = ""
    effective = False
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
            effective = True
        except (AttributeError, RuntimeError) as error:
            warning = str(error)
    return {
        "determinism_requested": deterministic,
        "determinism_effective": effective if deterministic else False,
        "determinism_warning": warning,
    }


def _history_best(history: dict[str, list[float]]) -> tuple[int, float, float]:
    accuracies = history.get("val_sparse_categorical_accuracy", history.get("val_accuracy", []))
    losses = history.get("val_loss", [])
    if not accuracies or len(losses) != len(accuracies):
        raise ValueError("Training history did not contain validation accuracy")
    best = min(
        range(len(accuracies)),
        key=lambda index: (-float(accuracies[index]), float(losses[index]), index),
    )
    return best + 1, float(accuracies[best]), float(losses[best])


def evaluate_candidate(
    configuration: dict[str, Any],
    experiment: ExperimentConfig,
    data: DatasetBundle,
) -> EvaluationResult:
    seed = trial_seed(experiment.run_seed, experiment.mode, configuration)
    started = time.perf_counter()
    try:
        seed_tensorflow(seed, experiment.deterministic)
        model, effective_learning_rate = build_model(configuration, experiment.mode)
        import tensorflow as tf

        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_sparse_categorical_accuracy",
            mode="max",
            patience=experiment.early_stopping_patience,
            min_delta=0.0,
            restore_best_weights=True,
        )
        history = model.fit(
            data.x_train,
            data.y_train,
            validation_data=(data.x_validation, data.y_validation),
            epochs=experiment.max_epochs,
            batch_size=configuration["batch_size"],
            callbacks=[callback],
            verbose=0,
        )
        history_dict = {key: [float(value) for value in values] for key, values in history.history.items()}
        best_epoch, best_accuracy, best_loss = _history_best(history_dict)
        train_accuracy = float(history_dict["sparse_categorical_accuracy"][best_epoch - 1])
        warning = ""
        if experiment.mode in {"paper_literal", "paper_conventional"} and configuration["loss"] not in {
            "sparse_categorical_crossentropy",
            "categorical_crossentropy",
        }:
            warning = "nonstandard_loss_for_multiclass_sparse_labels"
        return EvaluationResult(
            status="success",
            fitness=best_accuracy,
            train_accuracy=train_accuracy,
            validation_accuracy=best_accuracy,
            validation_loss=best_loss,
            best_epoch=best_epoch,
            training_time_seconds=time.perf_counter() - started,
            effective_learning_rate=effective_learning_rate,
            semantic_warning=warning,
            metadata=environment_metadata(),
        )
    except Exception as error:  # A failed candidate must not stop the colony.
        return EvaluationResult(
            status="failed",
            fitness=None,
            train_accuracy=None,
            validation_accuracy=None,
            validation_loss=None,
            best_epoch=None,
            training_time_seconds=time.perf_counter() - started,
            failure_reason=f"{type(error).__name__}: {error}",
            metadata=environment_metadata(),
        )


def train_final_model(
    configuration: dict[str, Any],
    experiment: ExperimentConfig,
    data: DatasetBundle,
    best_epoch: int,
    output_path: str,
) -> dict[str, Any]:
    """Train a new model on 60,000 development images and evaluate once on test."""
    seed = trial_seed(experiment.run_seed, experiment.mode, configuration)
    seed_tensorflow(seed, experiment.deterministic)
    model, effective_learning_rate = build_model(configuration, experiment.mode)
    x_development = np.concatenate([data.x_train, data.x_validation], axis=0)
    y_development = np.concatenate([data.y_train, data.y_validation], axis=0)
    model.fit(
        x_development,
        y_development,
        epochs=best_epoch,
        batch_size=configuration["batch_size"],
        shuffle=True,
        verbose=0,
    )
    loss, accuracy = model.evaluate(data.x_test, data.y_test, verbose=0)
    model.save(output_path)
    return {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "best_epoch": best_epoch,
        "effective_learning_rate": effective_learning_rate,
        "model_path": output_path,
    }
