"""Tests for experiment analysis and visualization outputs."""

from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from analysis_report import analyze_experiment


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class AnalysisReportTests(unittest.TestCase):
    def test_analysis_summarizes_trials_and_writes_four_plots_and_report(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "improved" / "seed_42"
            run_dir.mkdir(parents=True)
            write_csv(
                run_dir / "aco_trials.csv",
                [
                    {
                        "iteration": "1",
                        "mode": "improved",
                        "seed": "42",
                        "status": "success",
                        "fitness": "0.8",
                        "validation_accuracy": "0.8",
                        "validation_loss": "0.2",
                        "training_time_seconds": "1.0",
                        "cache_hit": "False",
                    },
                    {
                        "iteration": "1",
                        "mode": "improved",
                        "seed": "42",
                        "status": "failed",
                        "fitness": "",
                        "validation_accuracy": "",
                        "validation_loss": "",
                        "training_time_seconds": "0.5",
                        "cache_hit": "False",
                    },
                    {
                        "iteration": "2",
                        "mode": "improved",
                        "seed": "42",
                        "status": "cached",
                        "fitness": "0.85",
                        "validation_accuracy": "0.85",
                        "validation_loss": "0.15",
                        "training_time_seconds": "0.0",
                        "cache_hit": "True",
                    },
                ],
            )
            write_csv(
                run_dir / "pheromone_history.csv",
                [
                    {
                        "iteration": "1",
                        "phase": "after_update",
                        "hyperparameter": "optimizer",
                        "option_value": '"adam"',
                        "pheromone": "1.8",
                        "probability": "0.7",
                    },
                    {
                        "iteration": "2",
                        "phase": "after_update",
                        "hyperparameter": "optimizer",
                        "option_value": '"adam"',
                        "pheromone": "2.0",
                        "probability": "0.8",
                    },
                ],
            )
            (root / "primary_experiment_report.json").write_text(
                json.dumps(
                    {
                        "evaluator_type": "synthetic",
                        "test_evaluations": 0,
                        "comparison_note": "pipeline comparison",
                        "mode_results": [
                            {
                                "mode": "improved",
                                "budget_name": "smoke",
                                "effective_ants": 2,
                                "effective_iterations": 2,
                                "effective_max_epochs": 2,
                                "seed_runs": [],
                                "aggregate": None,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = analyze_experiment(str(root))

            self.assertEqual(result["evaluator_type"], "synthetic")
            self.assertEqual(result["trial_summary"]["failed_trials"], 1)
            self.assertEqual(result["trial_summary"]["cache_hits"], 1)
            self.assertEqual(result["trial_summary"]["best_validation_accuracy"], 0.85)
            self.assertIn("pipeline comparison", result["methodological_notes"])
            for name in (
                "convergence.png",
                "pheromone_evolution.png",
                "probability_evolution.png",
                "runtime_summary.png",
                "analysis_report.md",
                "analysis_summary.json",
            ):
                self.assertTrue((root / "analysis" / name).exists(), name)

    def test_empty_input_fails_with_actionable_message(self) -> None:
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "aco_trials.csv"):
                analyze_experiment(directory)

    def test_single_run_uses_run_summary_provenance(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "improved" / "smoke" / "run"
            run_dir.mkdir(parents=True)
            write_csv(
                run_dir / "aco_trials.csv",
                [{
                    "iteration": "1",
                    "mode": "improved",
                    "seed": "42",
                    "status": "success",
                    "fitness": "0.8",
                    "validation_accuracy": "0.8",
                    "validation_loss": "0.2",
                    "training_time_seconds": "1.0",
                    "cache_hit": "False",
                }],
            )
            (run_dir / "run_summary.json").write_text(
                json.dumps({
                    "run_id": "improved-smoke-seed-42",
                    "evaluator_type": "synthetic",
                    "runtime_seconds": 1.2,
                    "environment": {"python_version": "test"},
                }),
                encoding="utf-8",
            )

            result = analyze_experiment(directory)

            self.assertEqual(result["evaluator_type"], "synthetic")
            self.assertEqual(result["environment"]["python_version"], "test")
            self.assertEqual(result["per_run"]["improved-smoke-seed-42"]["runtime_seconds"], 1.2)


if __name__ == "__main__":
    unittest.main()
