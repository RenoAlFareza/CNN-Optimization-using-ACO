"""Behavior tests for post-aggregate final retraining and test evaluation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import numpy as np

from dataset import DatasetBundle
from evaluation_contract import EvaluationResult
from experiment_runner import (
    AggregateCandidate,
    ExperimentReport,
    ModeExperimentResult,
    SeedRunResult,
)
from config import ExperimentConfig
from evaluator import train_final_model
from final_evaluation import run_final_evaluations


def data_bundle() -> DatasetBundle:
    return DatasetBundle(
        x_train=np.zeros((50000, 1), dtype=np.float32),
        y_train=np.zeros(50000, dtype=np.int64),
        x_validation=np.zeros((10000, 1), dtype=np.float32),
        y_validation=np.zeros(10000, dtype=np.int64),
        x_test=np.zeros((10000, 1), dtype=np.float32),
        y_test=np.zeros(10000, dtype=np.int64),
        split_id="test-split",
    )


def aggregate_report() -> ExperimentReport:
    configuration = {
        "dense_units_1": 64,
        "dense_units_2": 32,
        "dropout_1": 0.3,
        "dropout_2": 0.3,
        "batch_size": 32,
        "activation": "relu",
        "optimizer": "adam",
        "learning_rate": 0.001,
    }
    candidate_id = "aggregate-candidate"
    confirmation = [
        EvaluationResult(
            status="success",
            fitness=0.9,
            train_accuracy=0.91,
            validation_accuracy=0.9,
            validation_loss=0.1,
            best_epoch=3,
            training_time_seconds=1.0,
        ),
        EvaluationResult(
            status="success",
            fitness=0.8,
            train_accuracy=0.82,
            validation_accuracy=0.8,
            validation_loss=0.2,
            best_epoch=4,
            training_time_seconds=1.1,
        ),
    ]
    aggregate = AggregateCandidate(candidate_id, configuration, confirmation)
    seed_runs = [
        SeedRunResult(
            mode="improved",
            family="improved",
            seed=42,
            status="success",
            runtime_seconds=1.0,
            global_best_candidate_id=candidate_id,
            global_best_configuration=configuration,
            best_validation_accuracy=0.9,
            best_epoch=3,
        ),
        SeedRunResult(
            mode="improved",
            family="improved",
            seed=43,
            status="success",
            runtime_seconds=1.0,
            global_best_candidate_id=candidate_id,
            global_best_configuration=configuration,
            best_validation_accuracy=0.8,
            best_epoch=4,
        ),
    ]
    return ExperimentReport(
        mode_results=[
            ModeExperimentResult(
                mode="improved",
                family="improved",
                interpretation="project_improvement",
                seed_runs=seed_runs,
                aggregate=aggregate,
                confirmation_evaluations=[
                    {
                        "candidate_id": candidate_id,
                        "configuration": configuration,
                        "seed": 42,
                        "status": "success",
                        "validation_accuracy": 0.9,
                        "validation_loss": 0.1,
                        "training_time_seconds": 1.0,
                        "best_epoch": 3,
                        "failure_reason": "",
                        "metadata": {},
                    },
                    {
                        "candidate_id": candidate_id,
                        "configuration": configuration,
                        "seed": 43,
                        "status": "success",
                        "validation_accuracy": 0.8,
                        "validation_loss": 0.2,
                        "training_time_seconds": 1.1,
                        "best_epoch": 4,
                        "failure_reason": "",
                        "metadata": {},
                    },
                ],
            )
        ]
    )


class FinalEvaluationTests(unittest.TestCase):
    def test_train_final_model_uses_development_data_without_early_stopping(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.fit_calls: list[dict[str, Any]] = []
                self.evaluate_calls = 0
                self.saved_path: str | None = None

            def fit(self, *args, **kwargs):
                self.fit_calls.append({"args": args, "kwargs": kwargs})

            def evaluate(self, *args, **kwargs):
                self.evaluate_calls += 1
                return 0.2, 0.9

            def save(self, path):
                self.saved_path = path

        model = FakeModel()
        configuration = aggregate_report().mode_results[0].aggregate.configuration
        experiment = ExperimentConfig(
            mode="improved",
            ants=20,
            iterations=20,
            max_epochs=10,
            run_seed=42,
            budget_name="main",
            search_space={key: [value] for key, value in configuration.items()},
        )

        with TemporaryDirectory() as directory, patch(
            "evaluator.seed_tensorflow", return_value={}
        ), patch(
            "evaluator.build_model", return_value=(model, 0.001)
        ):
            metrics = train_final_model(
                configuration,
                experiment,
                data_bundle(),
                best_epoch=3,
                output_path=str(Path(directory) / "best_model.keras"),
            )

        fit_call = model.fit_calls[0]
        self.assertEqual(len(fit_call["args"][0]), 60000)
        self.assertEqual(len(fit_call["args"][1]), 60000)
        self.assertEqual(fit_call["kwargs"]["epochs"], 3)
        self.assertTrue(fit_call["kwargs"]["shuffle"])
        self.assertNotIn("validation_data", fit_call["kwargs"])
        self.assertNotIn("callbacks", fit_call["kwargs"])
        self.assertEqual(model.evaluate_calls, 1)
        self.assertEqual(model.saved_path, str(Path(directory) / "best_model.keras"))
        self.assertEqual(metrics["test_accuracy"], 0.9)

    def test_retrains_new_model_on_development_data_and_tests_once_per_seed(self) -> None:
        calls: list[dict[str, Any]] = []

        def trainer(configuration, experiment, data, best_epoch, output_path):
            calls.append(
                {
                    "configuration": configuration,
                    "seed": experiment.run_seed,
                    "train_size": len(data.x_train) + len(data.x_validation),
                    "test_size": len(data.x_test),
                    "best_epoch": best_epoch,
                    "output_path": output_path,
                }
            )
            return {
                "test_loss": 0.1,
                "test_accuracy": 0.95,
                "best_epoch": best_epoch,
                "model_path": output_path,
            }

        with TemporaryDirectory() as directory:
            result = run_final_evaluations(
                report=aggregate_report(),
                data=data_bundle(),
                output_dir=directory,
                trainer=trainer,
            )

            self.assertEqual(result.status, "success")
            self.assertEqual(result.test_evaluations, 2)
            self.assertEqual([call["seed"] for call in calls], [42, 43])
            self.assertEqual([call["train_size"] for call in calls], [60000, 60000])
            self.assertEqual([call["test_size"] for call in calls], [10000, 10000])
            self.assertEqual([call["best_epoch"] for call in calls], [3, 4])
            self.assertTrue(all(Path(call["output_path"]).parent.exists() for call in calls))
            self.assertTrue((Path(directory) / "final_evaluation_report.json").exists())

            config_files = list(Path(directory).glob("improved/main/*/best_config.json"))
            self.assertEqual(len(config_files), 2)
            payload = json.loads(config_files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["dataset_split_id"], "test-split")
            self.assertIn("selected_configuration", payload)

    def test_no_aggregate_configuration_fails_without_testing(self) -> None:
        report = ExperimentReport(
            mode_results=[
                ModeExperimentResult(
                    mode="improved",
                    family="improved",
                    interpretation="project_improvement",
                    seed_runs=[],
                    aggregate=None,
                )
            ]
        )
        calls = 0

        def trainer(*args, **kwargs):
            nonlocal calls
            calls += 1
            raise AssertionError("final trainer must not be called")

        with TemporaryDirectory() as directory:
            result = run_final_evaluations(
                report=report,
                data=data_bundle(),
                output_dir=directory,
                trainer=trainer,
            )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.test_evaluations, 0)
        self.assertEqual(calls, 0)

    def test_invalid_confirmation_epoch_fails_that_seed_without_testing(self) -> None:
        report = aggregate_report()
        report.mode_results[0].confirmation_evaluations[1]["best_epoch"] = None
        calls = 0

        def trainer(*args, **kwargs):
            nonlocal calls
            calls += 1
            return {"test_accuracy": 0.9}

        with TemporaryDirectory() as directory:
            result = run_final_evaluations(
                report=report,
                data=data_bundle(),
                output_dir=directory,
                trainer=trainer,
            )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.test_evaluations, 1)
        self.assertEqual(calls, 1)
        self.assertEqual(result.mode_results[0]["seed_results"][1]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
