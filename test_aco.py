"""Fast mechanics tests; these do not require TensorFlow."""

from __future__ import annotations

import numpy as np
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ACO import synthetic_evaluator
from aco_optimizer import ACOOptimizer
from config import (
    ExperimentConfig,
    LOG_HYPERPARAMETERS,
    budget_values,
    get_search_space,
    mode_interpretation,
    paper_label,
)
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

    def test_improved_search_space_has_exactly_5184_configurations_and_no_loss_dimension(self) -> None:
        config = ExperimentConfig(mode="improved", ants=1, iterations=1, max_epochs=1)

        self.assertEqual(
            5184,
            np.prod([len(options) for options in config.search_space.values()]),
        )
        self.assertNotIn("loss", config.search_space)
        self.assertEqual(config.rho, 0.25)
        self.assertEqual(config.q, 1.0)

    def test_improved_mode_requires_fixed_rho_and_q(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires rho=0.25"):
            ExperimentConfig(mode="improved", rho=0.2)
        with self.assertRaisesRegex(ValueError, "requires q=1.0"):
            ExperimentConfig(mode="improved", q=2.0)
        with self.assertRaisesRegex(ValueError, "does not sample loss"):
            ExperimentConfig(
                mode="improved",
                search_space={"choice": ["a"], "loss": ["unexpected"]},
            )


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

        self.assertEqual(optimizer.run_status, "failed")
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

    def test_paper_conventional_evaporates_once_then_reinforces_each_valid_ant(self) -> None:
        config = ExperimentConfig(
            mode="paper_conventional",
            ants=2,
            iterations=1,
            max_epochs=1,
            search_space={"choice": ["only"]},
        )
        optimizer = ACOOptimizer(config, synthetic_evaluator)

        optimizer.run()

        # One evaporation, followed by +0.5 for each of two valid ants.
        self.assertEqual(float(optimizer.pheromone[0, 0]), 1.75)
        phases = [row["phase"] for row in optimizer.pheromone_history]
        self.assertEqual(phases, ["before_sampling", "after_update"])
        self.assertEqual(
            [row["update_step"] for row in optimizer.pheromone_history],
            [0, 1],
        )

    def test_paper_conventional_failed_ant_does_not_reinforce(self) -> None:
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
            ants=2,
            iterations=1,
            max_epochs=1,
            search_space={"choice": ["only"]},
        )
        optimizer = ACOOptimizer(config, failing_evaluator)

        with self.assertRaisesRegex(RuntimeError, "All candidate evaluations failed"):
            optimizer.run()

        self.assertEqual(float(optimizer.pheromone[0, 0]), 0.75)

    def test_paper_search_space_has_eight_options_per_dimension(self) -> None:
        search_space = get_search_space("paper_conventional")

        self.assertEqual(len(search_space), 8)
        self.assertTrue(all(len(options) == 8 for options in search_space.values()))
        self.assertIn("linear", search_space["activation"])

    def test_paper_conventional_requires_journal_evaporation_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "require rho=0.25"):
            ExperimentConfig(mode="paper_conventional", rho=0.2)

    def test_named_budgets_expose_smoke_and_pilot_protocols(self) -> None:
        self.assertEqual(
            budget_values("smoke"),
            {"ants": 2, "iterations": 2, "max_epochs": 2},
        )
        self.assertEqual(
            budget_values("pilot"),
            {"ants": 10, "iterations": 10, "max_epochs": 10},
        )

    def test_conventional_ants_share_iteration_start_probabilities(self) -> None:
        optimizer = ACOOptimizer(
            ExperimentConfig(
                mode="paper_conventional",
                ants=2,
                iterations=1,
                max_epochs=1,
                search_space={"choice": ["a", "b"]},
            ),
            synthetic_evaluator,
        )

        optimizer.run()

        probabilities = [row["selection_probabilities"] for row in optimizer.trial_rows]
        self.assertEqual(probabilities[0], probabilities[1])

    def test_improved_log_keeps_fixed_loss_and_paper_mode_schema(self) -> None:
        improved = ACOOptimizer(
            ExperimentConfig(
                mode="improved",
                ants=1,
                iterations=1,
                max_epochs=1,
                search_space={"choice": ["only"]},
            ),
            synthetic_evaluator,
        )
        paper = ACOOptimizer(
            ExperimentConfig(
                mode="paper_conventional",
                ants=1,
                iterations=1,
                max_epochs=1,
                search_space={"choice": ["only"]},
            ),
            synthetic_evaluator,
        )

        improved.run()
        paper.run()

        self.assertEqual(
            improved.trial_rows[0]["loss"], "sparse_categorical_crossentropy"
        )
        self.assertEqual(improved.trial_rows[0].keys(), paper.trial_rows[0].keys())
        self.assertEqual(
            [improved.trial_rows[0][name] for name in LOG_HYPERPARAMETERS],
            [paper.trial_rows[0][name] for name in LOG_HYPERPARAMETERS],
        )

    def test_improved_reinforces_only_iteration_best_by_validation_accuracy(self) -> None:
        def scored_evaluator(configuration: dict[str, object], experiment: ExperimentConfig) -> EvaluationResult:
            fitness = 0.9 if configuration["choice"] == "b" else 0.4
            return EvaluationResult(
                status="success",
                fitness=fitness,
                train_accuracy=fitness,
                validation_accuracy=fitness,
                validation_loss=1.0 - fitness,
                best_epoch=1,
                training_time_seconds=0.1,
            )

        optimizer = ACOOptimizer(
            ExperimentConfig(
                mode="improved",
                ants=2,
                iterations=1,
                max_epochs=1,
                run_seed=42,
                search_space={"choice": ["a", "b"]},
            ),
            scored_evaluator,
        )

        best = optimizer.run()

        self.assertEqual(best.configuration["choice"], "b")
        self.assertAlmostEqual(float(optimizer.pheromone[0, 0]), 0.75)
        self.assertAlmostEqual(float(optimizer.pheromone[0, 1]), 1.65)
        self.assertEqual(
            sum(row["is_iteration_best"] for row in optimizer.trial_rows),
            1,
        )

    def test_improved_reinforcement_uses_validation_accuracy_explicitly(self) -> None:
        def evaluator(configuration: dict[str, object], experiment: ExperimentConfig) -> EvaluationResult:
            validation_accuracy = 0.6 if configuration["choice"] == "b" else 0.4
            fitness = 0.99 if configuration["choice"] == "b" else 0.01
            return EvaluationResult(
                status="success",
                fitness=fitness,
                train_accuracy=fitness,
                validation_accuracy=validation_accuracy,
                validation_loss=1.0 - validation_accuracy,
                best_epoch=1,
                training_time_seconds=0.1,
            )

        optimizer = ACOOptimizer(
            ExperimentConfig(
                mode="improved",
                ants=2,
                iterations=1,
                max_epochs=1,
                run_seed=42,
                search_space={"choice": ["a", "b"]},
            ),
            evaluator,
        )

        optimizer.run()

        self.assertAlmostEqual(float(optimizer.pheromone[0, 1]), 1.35)

    def test_failed_trial_with_accuracy_cannot_become_improved_best(self) -> None:
        def failing_evaluator(configuration: dict[str, object], experiment: ExperimentConfig) -> EvaluationResult:
            return EvaluationResult(
                status="failed",
                fitness=None,
                train_accuracy=None,
                validation_accuracy=0.99,
                validation_loss=0.01,
                best_epoch=None,
                training_time_seconds=0.1,
                failure_reason="synthetic failure",
            )

        optimizer = ACOOptimizer(
            ExperimentConfig(
                mode="improved",
                ants=1,
                iterations=1,
                max_epochs=1,
                search_space={"choice": ["only"]},
            ),
            failing_evaluator,
        )

        with self.assertRaisesRegex(RuntimeError, "All candidate evaluations failed"):
            optimizer.run()

        self.assertEqual(optimizer.run_status, "failed")

    def test_linear_identifier_preserves_the_paper_label(self) -> None:
        self.assertEqual(paper_label("activation", "linear"), "linier")
        self.assertEqual(paper_label("activation", "relu"), "relu")

    def test_paper_literal_updates_after_each_ant(self) -> None:
        optimizer = ACOOptimizer(
            ExperimentConfig(
                mode="paper_literal",
                ants=2,
                iterations=1,
                max_epochs=1,
                search_space={"choice": ["only"]},
            ),
            synthetic_evaluator,
        )

        optimizer.run()

        # Each ant applies 0.75 evaporation and +0.5 reinforcement.
        self.assertAlmostEqual(float(optimizer.pheromone[0, 0]), 1.4375)
        self.assertEqual(
            [row["phase"] for row in optimizer.pheromone_history],
            ["before_sampling", "after_update", "before_sampling", "after_update"],
        )
        self.assertEqual(
            [row["update_step"] for row in optimizer.pheromone_history],
            [0, 1, 2, 3],
        )
        self.assertEqual(
            optimizer.pheromone_history[1]["pheromone"],
            optimizer.pheromone_history[2]["pheromone"],
        )

    def test_paper_literal_next_ant_samples_from_updated_probabilities(self) -> None:
        optimizer = ACOOptimizer(
            ExperimentConfig(
                mode="paper_literal",
                ants=2,
                iterations=1,
                max_epochs=1,
                search_space={"choice": ["a", "b"]},
            ),
            synthetic_evaluator,
        )

        optimizer.run()

        self.assertNotEqual(
            optimizer.trial_rows[0]["selection_probabilities"],
            optimizer.trial_rows[1]["selection_probabilities"],
        )

    def test_paper_literal_failed_ant_evaporates_without_reinforcement(self) -> None:
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

        optimizer = ACOOptimizer(
            ExperimentConfig(
                mode="paper_literal",
                ants=2,
                iterations=1,
                max_epochs=1,
                search_space={"choice": ["only"]},
            ),
            failing_evaluator,
        )

        with self.assertRaisesRegex(RuntimeError, "All candidate evaluations failed"):
            optimizer.run()

        self.assertAlmostEqual(float(optimizer.pheromone[0, 0]), 0.5625)
        self.assertEqual(len(optimizer.pheromone_history), 4)

    def test_literal_mode_exposes_diagnostic_budget_and_interpretation(self) -> None:
        self.assertEqual(
            budget_values("diagnostic"),
            {"ants": 20, "iterations": 50, "max_epochs": 10},
        )
        self.assertIn("not_verified_original_code", mode_interpretation("paper_literal"))

    def test_budget_and_interpretation_are_recorded_in_literal_logs(self) -> None:
        optimizer = ACOOptimizer(
            ExperimentConfig(
                mode="paper_literal",
                budget_name="diagnostic",
                ants=1,
                iterations=1,
                max_epochs=1,
                search_space={"choice": ["only"]},
            ),
            synthetic_evaluator,
        )

        optimizer.run()

        self.assertEqual(optimizer.trial_rows[0]["budget"], "diagnostic")
        self.assertIn(
            "not_verified_original_code",
            optimizer.trial_rows[0]["mode_interpretation"],
        )
        self.assertEqual(optimizer.pheromone_history[0]["budget"], "diagnostic")

    def test_successful_run_and_logs_have_status_and_effective_budget(self) -> None:
        optimizer = ACOOptimizer(
            ExperimentConfig(
                mode="paper_literal",
                budget_name="diagnostic",
                ants=1,
                iterations=1,
                max_epochs=3,
                search_space={"choice": ["only"]},
            ),
            synthetic_evaluator,
        )

        optimizer.run()

        self.assertEqual(optimizer.run_status, "success")
        self.assertEqual(optimizer.trial_rows[0]["effective_ants"], 1)
        self.assertEqual(optimizer.trial_rows[0]["effective_iterations"], 1)
        self.assertEqual(optimizer.trial_rows[0]["effective_max_epochs"], 3)

        with TemporaryDirectory() as directory:
            optimizer.write_logs(directory)
            trial_log = (Path(directory) / "aco_trials.csv").read_text(encoding="utf-8")
            history_log = (Path(directory) / "pheromone_history.csv").read_text(encoding="utf-8")
            self.assertIn("run_status", trial_log)
            self.assertIn("success", trial_log)
            self.assertIn("run_status", history_log)
