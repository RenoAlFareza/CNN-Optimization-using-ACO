"""Behavior tests for repeated primary runs and validation-only aggregation."""

from __future__ import annotations

import json
import unittest
from tempfile import TemporaryDirectory
from typing import Any

from evaluation_contract import EvaluationResult
from experiment_runner import (
    AggregateCandidate,
    aggregate_candidates,
    run_experiment_plan,
    run_primary_experiments,
)


def successful_result(
    validation_accuracy: float,
    validation_loss: float,
    training_time: float = 1.0,
) -> EvaluationResult:
    return EvaluationResult(
        status="success",
        fitness=validation_accuracy,
        train_accuracy=validation_accuracy,
        validation_accuracy=validation_accuracy,
        validation_loss=validation_loss,
        best_epoch=2,
        training_time_seconds=training_time,
    )


class ExperimentRunnerTests(unittest.TestCase):
    def test_primary_runner_executes_both_modes_for_each_seed(self) -> None:
        calls: list[tuple[str, int]] = []

        def evaluator_factory(experiment):
            def evaluator(configuration: dict[str, Any], _experiment):
                calls.append((experiment.mode, experiment.run_seed))
                return successful_result(0.8, 0.2)

            return evaluator

        report = run_experiment_plan(
            modes=("paper_conventional", "improved"),
            seeds=(42, 43, 44),
            budget={"ants": 1, "iterations": 1, "max_epochs": 1},
            budget_name="test",
            evaluator_factory=evaluator_factory,
        )

        self.assertEqual(len(report.mode_results), 2)
        self.assertEqual(
            [len(result.seed_runs) for result in report.mode_results],
            [3, 3],
        )
        self.assertTrue(all(run.status == "success" for result in report.mode_results for run in result.seed_runs))
        self.assertTrue(all(result.aggregate is not None for result in report.mode_results))
        self.assertGreaterEqual(len(calls), 12)
        self.assertEqual(
            report.mode_results[0].aggregate_evaluation_count,
            report.mode_results[0].aggregate_candidate_count * 3,
        )
        self.assertEqual(report.test_evaluations, 0)
        self.assertTrue(
            all(run.effective_ants == 1 for result in report.mode_results for run in result.seed_runs)
        )
        self.assertTrue(
            all(result.aggregate_runtime_seconds >= 0 for result in report.mode_results)
        )
        self.assertTrue(
            all(result.aggregate_failed_trials == 0 for result in report.mode_results)
        )
        self.assertEqual(report.runtime_summary["candidate_evaluation_count"], 6)
        self.assertIn("primary_runtime_seconds", report.runtime_summary)
        self.assertTrue(
            all("mean_candidate_runtime_seconds" in run.runtime_summary for result in report.mode_results for run in result.seed_runs)
        )

    def test_aggregate_candidates_are_evaluated_on_every_seed_before_selection(self) -> None:
        evaluations: list[tuple[str, int]] = []

        def evaluator_factory(experiment):
            def evaluator(configuration: dict[str, Any], _experiment):
                evaluations.append((configuration["choice"], experiment.run_seed))
                score = 0.9 if configuration["choice"] == "a" else 0.8
                return successful_result(score, 1.0 - score)

            return evaluator

        report = run_experiment_plan(
            modes=("improved",),
            seeds=(42, 43),
            budget={"ants": 1, "iterations": 1, "max_epochs": 1},
            budget_name="test",
            evaluator_factory=evaluator_factory,
            search_spaces={"improved": {"choice": ["a", "b"]}},
        )

        aggregate = report.mode_results[0].aggregate
        self.assertIsNotNone(aggregate)
        self.assertIn(aggregate.configuration["choice"], {"a", "b"})
        self.assertEqual(aggregate.evaluation_count, 2)
        selected = {
            run.global_best_configuration["choice"]
            for run in report.mode_results[0].seed_runs
            if run.global_best_configuration is not None
        }
        self.assertTrue(selected)
        self.assertEqual(
            set(evaluations),
            {(choice, seed) for choice in selected for seed in (42, 43)},
        )
        self.assertEqual(report.test_evaluations, 0)

    def test_aggregate_tie_breaks_mean_loss_then_time_then_candidate_id(self) -> None:
        candidates = [
            AggregateCandidate(
                candidate_id="b",
                configuration={"choice": "b"},
                evaluations=[successful_result(0.8, 0.2, 2.0)],
            ),
            AggregateCandidate(
                candidate_id="a",
                configuration={"choice": "a"},
                evaluations=[successful_result(0.8, 0.1, 2.0)],
            ),
        ]

        selected = aggregate_candidates(candidates, seeds=(42,))

        self.assertEqual(selected.candidate_id, "a")
        self.assertEqual(selected.mean_validation_accuracy, 0.8)
        self.assertEqual(selected.mean_validation_loss, 0.1)

    def test_report_can_be_written_without_test_metrics(self) -> None:
        def evaluator_factory(experiment):
            return lambda configuration, _experiment: successful_result(0.8, 0.2)

        with TemporaryDirectory() as directory:
            report = run_experiment_plan(
                modes=("improved",),
                seeds=(42,),
                budget={"ants": 1, "iterations": 1, "max_epochs": 1},
                budget_name="test",
                evaluator_factory=evaluator_factory,
                output_dir=directory,
                evaluator_type="synthetic",
        )
            report_path = report.output_path
            self.assertIsNotNone(report_path)
            with open(report_path, encoding="utf-8") as report_file:
                payload = json.load(report_file)
            self.assertEqual(payload["test_evaluations"], 0)
            self.assertNotIn("test_accuracy", payload["mode_results"][0]["aggregate"])
            self.assertIn("pipeline comparison", payload["comparison_note"])
            self.assertEqual(payload["evaluator_type"], "synthetic")
            self.assertIn("aggregate_runtime_seconds", payload["mode_results"][0])
            self.assertTrue(payload["mode_results"][0]["confirmation_evaluations"])

    def test_aggregate_confirmation_reuses_same_seed_cache_when_enabled(self) -> None:
        calls = 0

        def evaluator_factory(experiment):
            def evaluator(configuration, _experiment):
                nonlocal calls
                calls += 1
                return successful_result(0.8, 0.2)

            return evaluator

        report = run_experiment_plan(
            modes=("improved",),
            seeds=(42, 43),
            budget={"ants": 1, "iterations": 1, "max_epochs": 1},
            budget_name="test",
            evaluator_factory=evaluator_factory,
            search_spaces={"improved": {"choice": ["only"]}},
            cache_enabled=True,
        )

        result = report.mode_results[0]
        self.assertEqual(calls, 2)
        self.assertEqual(result.aggregate_cache_hits, 2)
        self.assertEqual(report.test_evaluations, 0)

    def test_primary_runner_requires_an_evaluator_factory(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires an evaluator_factory"):
            run_primary_experiments(evaluator_factory=None)

    def test_primary_runner_rejects_budget_other_than_20_by_5_by_5(self) -> None:
        with self.assertRaisesRegex(ValueError, "20x5x5"):
            run_primary_experiments(
                evaluator_factory=lambda _experiment: lambda _configuration, _config: successful_result(0.8, 0.2),
                budget={"ants": 20, "iterations": 4, "max_epochs": 5},
            )


if __name__ == "__main__":
    unittest.main()
