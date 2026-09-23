"""Small real-CNN calibration and budget-aware runtime projection."""

from __future__ import annotations

import json
import csv
from pathlib import Path
from statistics import median
import time
from typing import Any

from dataset import load_mnist
from evaluator import evaluate_candidate, environment_metadata
from experiment_runner import DEFAULT_PRIMARY_MODES, run_experiment_plan


def _successful_candidate_durations(report) -> list[float]:
    durations: list[float] = []
    for mode_result in report.mode_results:
        for seed_run in mode_result.seed_runs:
            if not seed_run.output_dir:
                continue
            path = Path(seed_run.output_dir) / "aco_trials.csv"
            with path.open(newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    if (
                        row.get("cache_hit", "").lower() != "true"
                        and row.get("status") != "failed"
                    ):
                        durations.append(float(row.get("training_time_seconds") or 0.0))
    return durations


def _epochs_from_report(report) -> list[int]:
    values: list[int] = []
    for mode_result in report.mode_results:
        for seed_run in mode_result.seed_runs:
            path = Path(seed_run.output_dir or "") / "aco_trials.csv"
            if not path.is_file():
                continue
            with path.open(newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    value = row.get("actual_epochs_completed")
                    if (
                        value
                        and row.get("cache_hit", "").lower() != "true"
                        and row.get("status") != "failed"
                    ):
                        values.append(int(value))
    return values


def _best_epochs_from_report(report) -> list[int]:
    values: list[int] = []
    for mode_result in report.mode_results:
        for seed_run in mode_result.seed_runs:
            path = Path(seed_run.output_dir or "") / "aco_trials.csv"
            for row in _read_trial_rows(path):
                value = row.get("best_epoch")
                if (
                    value
                    and row.get("cache_hit", "").lower() != "true"
                    and row.get("status") != "failed"
                ):
                    values.append(int(value))
    return values


def _project_runtime(
    candidate_seconds: list[float],
    epochs: list[int],
    calibration_budget: dict[str, int],
    primary_budget: dict[str, int],
    mode_count: int,
    seed_count: int,
    best_epochs: list[int] | None = None,
) -> dict[str, Any]:
    mean_seconds = sum(candidate_seconds) / len(candidate_seconds)
    median_seconds = median(candidate_seconds)
    mean_epochs = sum(epochs) / len(epochs) if epochs else calibration_budget["max_epochs"]
    best_epochs = best_epochs or epochs
    epoch_adjusted_candidate_seconds = max(
        mean_seconds,
        median_seconds,
    ) * (primary_budget["max_epochs"] / max(mean_epochs, 1.0))
    evaluations_per_mode = primary_budget["ants"] * primary_budget["iterations"] * seed_count
    tuning_evaluations = evaluations_per_mode * mode_count
    tuning_seconds = epoch_adjusted_candidate_seconds * tuning_evaluations

    # Each mode contributes up to three seed-best configurations, evaluated on
    # all three seeds. Cache can reuse the evaluation on the originating seed.
    max_confirmation_evaluations = mode_count * seed_count * seed_count
    cached_confirmation_evaluations = mode_count * seed_count
    fresh_confirmation_evaluations = max(
        0, max_confirmation_evaluations - cached_confirmation_evaluations
    )
    confirmation_seconds = (
        epoch_adjusted_candidate_seconds * fresh_confirmation_evaluations
    )

    # Final retraining uses 60k development images, six models, and the selected
    # candidate's best epoch. Candidate runtime includes its actual epochs, so
    # scale by data volume and by the observed best/actual epoch ratio rather
    # than multiplying by the epoch count a second time.
    final_models = mode_count * seed_count
    mean_best_epochs = (
        sum(best_epochs) / len(best_epochs) if best_epochs else mean_epochs
    )
    final_seconds = (
        max(mean_seconds, median_seconds)
        * (60_000 / 50_000)
        * (mean_best_epochs / max(mean_epochs, 1.0))
        * final_models
    )
    return {
        "mean_candidate_seconds": mean_seconds,
        "median_candidate_seconds": median_seconds,
        "mean_actual_epochs": mean_epochs,
        "mean_best_epoch": mean_best_epochs,
        "estimated_candidate_seconds_at_primary_budget": epoch_adjusted_candidate_seconds,
        "tuning_candidate_evaluations": tuning_evaluations,
        "tuning_runtime_seconds": tuning_seconds,
        "confirmation_evaluations_upper_bound": max_confirmation_evaluations,
        "confirmation_cache_hits_assumed": cached_confirmation_evaluations,
        "confirmation_fresh_evaluations_estimate": fresh_confirmation_evaluations,
        "confirmation_runtime_seconds": confirmation_seconds,
        "final_models": final_models,
        "final_retraining_runtime_seconds": final_seconds,
        "primary_total_seconds": tuning_seconds + confirmation_seconds,
        "full_workflow_total_seconds": tuning_seconds + confirmation_seconds + final_seconds,
    }


def run_runtime_calibration(
    data_dir: str,
    output_dir: str,
    calibration_budget: dict[str, int],
    primary_budget: dict[str, int],
) -> dict[str, Any]:
    """Train a small calibration batch for both modes on seed 42 only."""
    if calibration_budget["ants"] != 20 or calibration_budget["iterations"] != 1:
        raise ValueError("Runtime calibration requires 20 ants and one iteration")
    if primary_budget["ants"] != 20:
        raise ValueError("Primary budget must retain the paper's 20 ants")

    started = time.perf_counter()
    data = load_mnist(2024, data_dir)

    def evaluator_factory(experiment):
        return lambda candidate, _experiment: evaluate_candidate(candidate, experiment, data)

    report = run_experiment_plan(
        modes=DEFAULT_PRIMARY_MODES,
        seeds=(42,),
        budget=calibration_budget,
        budget_name="runtime_calibration",
        evaluator_factory=evaluator_factory,
        output_dir=str(Path(output_dir) / "runtime_calibration"),
        cache_enabled=True,
        evaluator_type="cnn_mnist",
    )
    durations = _successful_candidate_durations(report)
    if not durations:
        raise RuntimeError("Runtime calibration produced no non-cache candidate timings")
    epochs = _epochs_from_report(report)
    best_epochs = _best_epochs_from_report(report)
    training_rows = [
        row
        for mode_result in report.mode_results
        for seed_run in mode_result.seed_runs
        if seed_run.output_dir
        for row in _read_trial_rows(Path(seed_run.output_dir) / "aco_trials.csv")
        if not row.get("cache_hit", "").lower() == "true"
        and row.get("status") != "failed"
    ]
    all_trial_count = sum(run.trial_count for mode in report.mode_results for run in mode.seed_runs)
    cache_hits = sum(run.cache_hits for mode in report.mode_results for run in mode.seed_runs)
    failed_trials = sum(run.failed_trials for mode in report.mode_results for run in mode.seed_runs)
    estimate = _project_runtime(
        durations,
        epochs,
        calibration_budget,
        primary_budget,
        mode_count=len(DEFAULT_PRIMARY_MODES),
        seed_count=3,
        best_epochs=best_epochs,
    )
    estimate["calibration_mean_best_epoch"] = (
        sum(best_epochs) / len(best_epochs) if best_epochs else None
    )
    result = {
        "status": "completed",
        "calibration_budget": calibration_budget,
        "primary_budget": primary_budget,
        "seeds_used_for_calibration": [42],
        "modes": list(DEFAULT_PRIMARY_MODES),
        "candidate_count": all_trial_count,
        "trained_candidate_count": len(durations),
        "candidate_runtime_seconds": durations,
        "mean_candidate_runtime_seconds": sum(durations) / len(durations),
        "median_candidate_runtime_seconds": median(durations),
        "actual_epochs_completed": epochs,
        "best_epochs": best_epochs,
        "epochs_per_candidate": {
            str(index + 1): epoch for index, epoch in enumerate(epochs)
        },
        "cache_hits": cache_hits,
        "cache_hit_rate": cache_hits / all_trial_count if all_trial_count else 0.0,
        "failed_trials": failed_trials,
        "failed_trial_rate": failed_trials / all_trial_count if all_trial_count else 0.0,
        "gpu_available": report.environment.get("gpu_available", "unknown"),
        "gpu_devices": report.environment.get("gpu_devices", "unknown"),
        "gpu_utilization": report.environment.get("gpu_utilization", "not_sampled"),
        "actual_epochs_by_candidate": [
            int(row["actual_epochs_completed"])
            if row.get("actual_epochs_completed")
            else None
            for row in training_rows
        ],
        "calibration_runtime_seconds": time.perf_counter() - started,
        "environment": report.environment,
        "estimate": estimate,
    }
    path = Path(output_dir) / "runtime_calibration" / "runtime_calibration_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["report_path"] = str(path)
    return result


def _read_trial_rows(path: Path) -> list[dict[str, str]]:
    import csv

    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))
