"""Fast mechanics tests; these do not require TensorFlow."""

from __future__ import annotations

import numpy as np
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ACO import synthetic_evaluator
from aco_optimizer import ACOOptimizer
from config import ExperimentConfig
from evaluation_contract import EvaluationResult, candidate_id, trial_seed


class ACOTest(unittest.TestCase):
    def test_trial_result_supports_success_failed_and_cached(self) -> None:
        successful = EvaluationResult(
            status="success",
            fitness=0.9,
            train_accuracy=0.95,
            validation_accuracy=0.9,
            validation_loss=0.2,
            best_epoch=2,
            training_time_seconds=1.5,
        )
        cached = EvaluationResult(
            status="cached",
            fitness=successful.fitness,
            train_accuracy=successful.train_accuracy,
            validation_accuracy=successful.validation_accuracy,
            validation_loss=successful.validation_loss,
            best_epoch=successful.best_epoch,
            training_time_seconds=successful.training_time_seconds,
            metadata={"source": "cache"},
        )
        failed = EvaluationResult(
            status="failed",
            fitness=None,
            train_accuracy=None,
            validation_accuracy=None,
            validation_loss=None,
            best_epoch=None,
            training_time_seconds=0.25,
            failure_reason="RuntimeError: synthetic failure",
        )

        self.assertEqual(successful.status, "success")
        self.assertEqual(cached.status, "cached")
        self.assertEqual(cached.metadata, {"source": "cache"})
        self.assertIsNone(failed.fitness)
        self.assertEqual(failed.failure_reason, "RuntimeError: synthetic failure")

    def test_candidate_id_and_trial_seed_are_canonical(self) -> None:
        first = {"activation": "relu", "batch_size": 32}
        reordered = {"batch_size": 32, "activation": "relu"}

        self.assertEqual(candidate_id(first), candidate_id(reordered))
        self.assertEqual(
            trial_seed(42, "improved", first),
            trial_seed(42, "improved", reordered),
        )
        self.assertNotEqual(trial_seed(42, "improved", first), trial_seed(43, "improved", first))
        self.assertNotEqual(trial_seed(42, "improved", first), trial_seed(42, "paper_literal", first))

    def test_probability_rows_sum_to_one(self) -> None:
        optimizer = ACOOptimizer(
            ExperimentConfig(mode="improved", ants=1, iterations=1, max_epochs=1),
            synthetic_evaluator,
        )
        probabilities = optimizer._probabilities()
        for index, count in enumerate(optimizer.option_counts):
            self.assertTrue(np.isclose(probabilities[index, :count].sum(), 1.0))
            self.assertTrue(np.all(probabilities[index, :count] > 0.0))


    def test_synthetic_smoke_all_modes(self) -> None:
        for mode in ("paper_literal", "paper_conventional", "improved"):
            optimizer = ACOOptimizer(
                ExperimentConfig(mode=mode, ants=2, iterations=2, max_epochs=1),
                synthetic_evaluator,
            )
            best = optimizer.run()
            self.assertIsNotNone(best.result.fitness)
            self.assertTrue(np.all(optimizer.pheromone >= optimizer.config.tau_min))
            self.assertEqual(len(optimizer.trial_rows), 4)
            for row in optimizer.trial_rows:
                for name, options in optimizer.config.search_space.items():
                    self.assertIn(row[name], options)
            for row in optimizer.pheromone_history:
                if row["hyperparameter"] in optimizer.config.search_space:
                    self.assertGreaterEqual(row["pheromone"], optimizer.config.tau_min)

    def test_synthetic_evaluator_is_injected_without_tensorflow(self) -> None:
        calls: list[dict[str, object]] = []

        def evaluator(configuration: dict[str, object], experiment: ExperimentConfig) -> EvaluationResult:
            calls.append(configuration)
            return synthetic_evaluator(configuration, experiment)

        optimizer = ACOOptimizer(
            ExperimentConfig(mode="improved", ants=2, iterations=1, max_epochs=1),
            evaluator,
        )
        optimizer.run()

        self.assertEqual(len(calls), 2)

    def test_cache_reuses_successful_evaluations_and_logs_cache_hits(self) -> None:
        calls: list[dict[str, object]] = []

        def evaluator(configuration: dict[str, object], experiment: ExperimentConfig) -> EvaluationResult:
            calls.append(configuration)
            return EvaluationResult(
                status="success",
                fitness=0.8,
                train_accuracy=0.9,
                validation_accuracy=0.8,
                validation_loss=0.2,
                best_epoch=1,
                training_time_seconds=0.1,
            )

        config = ExperimentConfig(
            mode="improved",
            ants=1,
            iterations=2,
            max_epochs=1,
            cache_enabled=True,
            search_space={"choice": ["only"]},
        )
        optimizer = ACOOptimizer(config, evaluator)
        optimizer.run()

        self.assertEqual(len(calls), 1)
        self.assertEqual([row["status"] for row in optimizer.trial_rows], ["success", "cached"])
        self.assertEqual([row["cache_hit"] for row in optimizer.trial_rows], [False, True])

    def test_failed_candidates_do_not_reinforce_and_all_failures_raise(self) -> None:
        def failing_evaluator(configuration: dict[str, object], experiment: ExperimentConfig) -> EvaluationResult:
            return EvaluationResult(
                status="failed",
                fitness=None,
                train_accuracy=None,
                validation_accuracy=None,
                validation_loss=None,
                best_epoch=None,
                training_time_seconds=0.1,
                failure_reason="synthetic failure",
            )

        config = ExperimentConfig(
            mode="paper_conventional",
            ants=1,
            iterations=1,
            max_epochs=1,
            rho=0.25,
            tau_0=1.0,
            search_space={"choice": ["a"]},
        )
        optimizer = ACOOptimizer(config, failing_evaluator)

        with self.assertRaisesRegex(RuntimeError, "All candidate evaluations failed"):
            optimizer.run()

        self.assertEqual(float(optimizer.pheromone[0, 0]), 0.75)
        self.assertEqual(optimizer.trial_rows[0]["status"], "failed")
        self.assertIsNone(optimizer.trial_rows[0]["fitness"])

    def test_logs_write_trial_and_pheromone_history(self) -> None:
        optimizer = ACOOptimizer(
            ExperimentConfig(mode="improved", ants=1, iterations=1, max_epochs=1),
            synthetic_evaluator,
        )
        optimizer.run()

        with TemporaryDirectory() as directory:
            optimizer.write_logs(directory)
            trial_log = Path(directory) / "aco_trials.csv"
            pheromone_log = Path(directory) / "pheromone_history.csv"
            self.assertTrue(trial_log.exists())
            self.assertTrue(pheromone_log.exists())
            self.assertIn("candidate_id", trial_log.read_text(encoding="utf-8"))
            self.assertIn("phase", pheromone_log.read_text(encoding="utf-8"))

    def test_same_seed_produces_same_best_candidate(self) -> None:
        def run() -> tuple[str, list[dict[str, object]]]:
            optimizer = ACOOptimizer(
                ExperimentConfig(mode="improved", ants=2, iterations=2, max_epochs=1),
                synthetic_evaluator,
            )
            best = optimizer.run()
            return best.candidate_id, optimizer.trial_rows

        first_id, first_rows = run()
        second_id, second_rows = run()
        self.assertEqual(first_id, second_id)
        self.assertEqual(
            [row["candidate_id"] for row in first_rows],
            [row["candidate_id"] for row in second_rows],
        )

    def test_empty_search_space_is_rejected(self) -> None:
        config = ExperimentConfig(
            mode="improved",
            ants=1,
            iterations=1,
            max_epochs=1,
            search_space={"choice": []},
        )
        with self.assertRaisesRegex(ValueError, "at least one option"):
            ACOOptimizer(config, synthetic_evaluator)
