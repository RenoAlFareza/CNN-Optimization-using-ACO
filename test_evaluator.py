"""Evaluator adapter tests using a lightweight TensorFlow test double."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from config import ExperimentConfig
from dataset import DatasetBundle, load_mnist
from evaluator import evaluate_candidate


class _LearningRate:
    def __init__(self, value: float) -> None:
        self.value = value

    def numpy(self) -> float:
        return self.value


class _Optimizer:
    def __init__(self, learning_rate: float = 0.001, **_: object) -> None:
        self.learning_rate = _LearningRate(learning_rate)


class _Layer:
    def __init__(self, name: str, *args: object, **kwargs: object) -> None:
        self.name = name
        self.args = args
        self.kwargs = kwargs


class _FakeModel:
    def __init__(self, history: dict[str, list[float]] | None = None, fit_error: Exception | None = None) -> None:
        self.history = history or {
            "sparse_categorical_accuracy": [0.70, 0.80, 0.81],
            "val_sparse_categorical_accuracy": [0.60, 0.80, 0.80],
            "val_loss": [0.90, 0.30, 0.40],
        }
        self.fit_error = fit_error
        self.compile_kwargs: dict[str, object] = {}
        self.fit_kwargs: dict[str, object] = {}

    def compile(self, **kwargs: object) -> None:
        self.compile_kwargs = kwargs

    def fit(self, *args: object, **kwargs: object) -> types.SimpleNamespace:
        self.fit_kwargs = kwargs
        if self.fit_error is not None:
            raise self.fit_error
        return types.SimpleNamespace(history=self.history)


def _tensorflow_double(model: _FakeModel | None = None) -> types.ModuleType:
    model = model or _FakeModel()

    class _Sequential:
        def __init__(self, layers: list[_Layer]) -> None:
            self.layers = layers

        def compile(self, **kwargs: object) -> None:
            model.compile(**kwargs)
            self.compile_kwargs = model.compile_kwargs

        def fit(self, *args: object, **kwargs: object) -> types.SimpleNamespace:
            return model.fit(*args, **kwargs)

    class _EarlyStopping:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    tf = types.ModuleType("tensorflow")
    tf.__version__ = "test-tensorflow"
    tf.keras = types.SimpleNamespace(
        __version__="test-keras",
        Sequential=_Sequential,
        layers=types.SimpleNamespace(
            Input=lambda *args, **kwargs: _Layer("Input", *args, **kwargs),
            Conv2D=lambda *args, **kwargs: _Layer("Conv2D", *args, **kwargs),
            MaxPooling2D=lambda *args, **kwargs: _Layer("MaxPooling2D", *args, **kwargs),
            Dropout=lambda *args, **kwargs: _Layer("Dropout", *args, **kwargs),
            Flatten=lambda *args, **kwargs: _Layer("Flatten", *args, **kwargs),
            BatchNormalization=lambda *args, **kwargs: _Layer("BatchNormalization", *args, **kwargs),
            Dense=lambda *args, **kwargs: _Layer("Dense", *args, **kwargs),
        ),
        optimizers=types.SimpleNamespace(
            Adam=_Optimizer,
            RMSprop=_Optimizer,
            SGD=_Optimizer,
            Adadelta=_Optimizer,
            Adagrad=_Optimizer,
            Adamax=_Optimizer,
            Ftrl=_Optimizer,
            Nadam=_Optimizer,
        ),
        callbacks=types.SimpleNamespace(EarlyStopping=_EarlyStopping),
        utils=types.SimpleNamespace(set_random_seed=lambda seed: setattr(tf, "seed", seed)),
    )
    tf.config = types.SimpleNamespace(
        experimental=types.SimpleNamespace(enable_op_determinism=lambda: setattr(tf, "deterministic", True))
    )
    return tf


def _module_patch(tf: types.ModuleType):
    keras = types.ModuleType("tensorflow.keras")
    keras.__version__ = tf.keras.__version__
    keras.datasets = getattr(tf.keras, "datasets", types.SimpleNamespace())
    datasets = types.ModuleType("tensorflow.keras.datasets")
    datasets.mnist = getattr(keras.datasets, "mnist", None)
    return patch.dict(
        sys.modules,
        {"tensorflow": tf, "tensorflow.keras": keras, "tensorflow.keras.datasets": datasets},
    )


class DatasetAndModelTests(unittest.TestCase):
    def test_load_mnist_normalizes_and_reuses_deterministic_stratified_split(self) -> None:
        labels = np.tile(np.arange(10, dtype=np.int64), 6000)
        images = np.zeros((60000, 28, 28), dtype=np.uint8)
        test_images = np.zeros((10000, 28, 28), dtype=np.uint8)
        mnist = types.SimpleNamespace(load_data=lambda: ((images, labels), (test_images, labels[:10000])))
        tf = _tensorflow_double()
        tf.keras.datasets = types.SimpleNamespace(mnist=mnist)

        with _module_patch(tf):
            first = load_mnist()
            second = load_mnist()

        self.assertEqual(first.x_train.shape, (50000, 28, 28, 1))
        self.assertEqual(first.x_validation.shape, (10000, 28, 28, 1))
        self.assertEqual(first.x_test.shape, (10000, 28, 28, 1))
        self.assertEqual(first.x_train.dtype, np.float32)
        self.assertEqual(float(first.x_train.max()), 0.0)
        self.assertEqual(first.split_id, "mnist-stratified-50000-10000-seed-2024")
        self.assertTrue(np.array_equal(first.y_validation, second.y_validation))
        self.assertTrue(np.array_equal(first.y_train, second.y_train))
        self.assertEqual(np.bincount(first.y_train, minlength=10).tolist(), [5000] * 10)
        self.assertEqual(np.bincount(first.y_validation, minlength=10).tolist(), [1000] * 10)

    def test_build_model_keeps_fixed_architecture_and_improved_loss(self) -> None:
        from cnn_model import build_model

        tf = _tensorflow_double()
        with _module_patch(tf):
            model, learning_rate = build_model(
                {
                    "dropout_1": 0.1,
                    "dropout_2": 0.2,
                    "dense_units_1": 64,
                    "dense_units_2": 32,
                    "activation": "relu",
                    "optimizer": "adam",
                    "learning_rate": 0.01,
                    "loss": "mean_absolute_error",
                },
                "improved",
            )

        self.assertEqual([layer.name for layer in model.layers], [
            "Input", "Conv2D", "MaxPooling2D", "Dropout", "Conv2D", "MaxPooling2D",
            "Flatten", "BatchNormalization", "Dense", "Dropout", "Dense", "Dense",
        ])
        self.assertEqual(model.compile_kwargs["loss"], "sparse_categorical_crossentropy")
        self.assertEqual(learning_rate, 0.01)

    def test_evaluate_candidate_records_best_epoch_and_metadata(self) -> None:
        model = _FakeModel()
        tf = _tensorflow_double(model)
        data = DatasetBundle(
            np.zeros((2, 28, 28, 1), dtype=np.float32), np.array([0, 1]),
            np.zeros((1, 28, 28, 1), dtype=np.float32), np.array([0]),
            np.zeros((1, 28, 28, 1), dtype=np.float32), np.array([0]),
            "split-test",
        )
        config = {
            "dropout_1": 0.1,
            "dropout_2": 0.2,
            "dense_units_1": 64,
            "dense_units_2": 32,
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.01,
            "batch_size": 2,
        }
        experiment = ExperimentConfig(mode="improved", max_epochs=5, deterministic=True)

        with _module_patch(tf):
            result = evaluate_candidate(config, experiment, data)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.fitness, 0.80)
        self.assertEqual(result.validation_loss, 0.30)
        self.assertEqual(result.best_epoch, 2)
        self.assertEqual(result.train_accuracy, 0.80)
        self.assertEqual(result.effective_learning_rate, 0.01)
        self.assertEqual(result.metadata["split_id"], "split-test")
        self.assertEqual(result.metadata["determinism_effective"], True)
        self.assertEqual(model.fit_kwargs["callbacks"][0].kwargs, {
            "monitor": "val_sparse_categorical_accuracy",
            "mode": "max",
            "patience": 2,
            "min_delta": 0.0,
            "restore_best_weights": True,
        })

    def test_nonstandard_paper_loss_warns_when_training_succeeds(self) -> None:
        model = _FakeModel()
        tf = _tensorflow_double(model)
        data = DatasetBundle(
            np.zeros((1, 28, 28, 1), dtype=np.float32), np.array([0]),
            np.zeros((1, 28, 28, 1), dtype=np.float32), np.array([0]),
            np.zeros((1, 28, 28, 1), dtype=np.float32), np.array([0]),
            "split-test",
        )
        configuration = {
            "dropout_1": 0.1, "dropout_2": 0.2, "dense_units_1": 64, "dense_units_2": 32,
            "activation": "relu", "optimizer": "adam", "loss": "mean_absolute_error", "batch_size": 1,
        }

        with _module_patch(tf):
            result = evaluate_candidate(configuration, ExperimentConfig(mode="paper_literal"), data)

        self.assertEqual(result.status, "success")
        self.assertEqual(result.semantic_warning, "nonstandard_loss_for_multiclass_sparse_labels")

    def test_fit_errors_fail_without_accuracy_zero(self) -> None:
        model = _FakeModel(fit_error=ValueError("incompatible labels"))
        tf = _tensorflow_double(model)
        data = DatasetBundle(
            np.zeros((1, 28, 28, 1), dtype=np.float32), np.array([0]),
            np.zeros((1, 28, 28, 1), dtype=np.float32), np.array([0]),
            np.zeros((1, 28, 28, 1), dtype=np.float32), np.array([0]),
            "split-test",
        )
        configuration = {
            "dropout_1": 0.1, "dropout_2": 0.2, "dense_units_1": 64, "dense_units_2": 32,
            "activation": "relu", "optimizer": "adam", "loss": "sparse_categorical_crossentropy", "batch_size": 1,
        }

        with _module_patch(tf):
            result = evaluate_candidate(configuration, ExperimentConfig(mode="paper_literal"), data)

        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.fitness)
        self.assertEqual(result.semantic_warning, "")
        self.assertIn("ValueError: incompatible labels", result.failure_reason)


if __name__ == "__main__":
    unittest.main()
