"""Runtime projection tests that do not require TensorFlow or a GPU."""

from __future__ import annotations

import unittest

from runtime_calibration import _project_runtime


class RuntimeCalibrationTests(unittest.TestCase):
    def test_projects_tuning_confirmation_and_final_workflow(self) -> None:
        estimate = _project_runtime(
            candidate_seconds=[10.0, 20.0],
            epochs=[5, 5],
            calibration_budget={"ants": 20, "iterations": 1, "max_epochs": 5},
            primary_budget={"ants": 20, "iterations": 5, "max_epochs": 5},
            mode_count=2,
            seed_count=3,
            best_epochs=[2, 2],
        )

        self.assertEqual(estimate["tuning_candidate_evaluations"], 600)
        self.assertEqual(estimate["confirmation_evaluations_upper_bound"], 18)
        self.assertEqual(estimate["confirmation_fresh_evaluations_estimate"], 12)
        self.assertEqual(estimate["final_models"], 6)
        self.assertEqual(estimate["estimated_candidate_seconds_at_primary_budget"], 15.0)
        self.assertEqual(estimate["tuning_runtime_seconds"], 9000.0)
        self.assertEqual(estimate["confirmation_runtime_seconds"], 180.0)
        self.assertEqual(estimate["final_retraining_runtime_seconds"], 43.2)
        self.assertEqual(estimate["full_workflow_total_seconds"], 9223.2)

    def test_small_gpu_rate_projects_below_two_hours_without_changing_budget(self) -> None:
        estimate = _project_runtime(
            candidate_seconds=[8.0] * 40,
            epochs=[4] * 40,
            calibration_budget={"ants": 20, "iterations": 1, "max_epochs": 5},
            primary_budget={"ants": 20, "iterations": 5, "max_epochs": 5},
            mode_count=2,
            seed_count=3,
        )

        self.assertLess(estimate["full_workflow_total_seconds"], 2 * 60 * 60)


if __name__ == "__main__":
    unittest.main()
