"""Final model retraining after validation-only aggregate selection."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from config import ExperimentConfig
from dataset import DatasetBundle
from evaluator import environment_metadata, train_final_model
from experiment_runner import ExperimentReport


FinalTrainer = Callable[[dict[str, Any], ExperimentConfig, DatasetBundle, int, str], dict[str, Any]]


@dataclass
class FinalEvaluationResult:
    status: str
    mode_results: list[dict[str, Any]] = field(default_factory=list)
    test_evaluations: int = 0
    runtime_seconds: float = 0.0
    primary_runtime_seconds: float = 0.0
    total_workflow_runtime_seconds: float = 0.0
    output_path: str | None = None
    failure_reason: str = ""


def _configuration_for_aggregate(report_mode) -> dict[str, Any]:
    if report_mode.aggregate is None:
        raise ValueError("Aggregate configuration is required before final retraining")
    return dict(report_mode.aggregate.configuration)


def _aggregate_best_epoch(report_mode, seed: int) -> int:
    """Use the epoch found for the selected aggregate candidate on this seed."""
    candidate_id = report_mode.aggregate.candidate_id
    matches = [
        record
        for record in report_mode.confirmation_evaluations
        if record["candidate_id"] == candidate_id
        and record["seed"] == seed
        and record["status"] in {"success", "cached"}
    ]
    if len(matches) != 1 or matches[0].get("best_epoch") is None:
        raise ValueError("No valid aggregate confirmation epoch is available for final retraining")
    return int(matches[0]["best_epoch"])


def _write_config(
    path: Path,
    mode: str,
    seed: int,
    configuration: dict[str, Any],
    best_epoch: int,
    data: DatasetBundle,
    run_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "mode": mode,
                "seed": seed,
                "run_id": run_id,
                "selected_configuration": configuration,
                "best_epoch": best_epoch,
                "dataset_split_id": data.split_id,
                "training_development_size": len(data.x_train) + len(data.x_validation),
                "test_size": len(data.x_test),
                "test_used_for_selection": False,
                "environment": environment_metadata(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_report(result: FinalEvaluationResult, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "final_evaluation_report.json"
    path.write_text(
        json.dumps(
            {
                "status": result.status,
                "test_evaluations": result.test_evaluations,
                "runtime_seconds": result.runtime_seconds,
                "primary_runtime_seconds": result.primary_runtime_seconds,
                "total_workflow_runtime_seconds": result.total_workflow_runtime_seconds,
                "failure_reason": result.failure_reason,
                "mode_results": result.mode_results,
                "environment": environment_metadata(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    result.output_path = str(path)
    return str(path)


def run_final_evaluations(
    report: ExperimentReport,
    data: DatasetBundle,
    output_dir: str,
    trainer: FinalTrainer = train_final_model,
) -> FinalEvaluationResult:
    """Retrain aggregate-selected configurations and test once per seed.

    The report must already contain aggregate configurations selected from
    validation evidence. This function has no selection logic and never uses
    test metrics to select a configuration.
    """
    started = time.perf_counter()
    root = Path(output_dir)
    result = FinalEvaluationResult(status="success")
    result.primary_runtime_seconds = float(
        report.runtime_summary.get("primary_runtime_seconds", 0.0)
    )

    if not report.mode_results or any(item.aggregate is None for item in report.mode_results):
        result.status = "failed"
        result.failure_reason = "No valid aggregate configuration is available for final retraining"
        result.runtime_seconds = time.perf_counter() - started
        result.total_workflow_runtime_seconds = (
            result.primary_runtime_seconds + result.runtime_seconds
        )
        _write_report(result, root)
        return result

    for mode_result in report.mode_results:
        configuration = _configuration_for_aggregate(mode_result)
        mode_record: dict[str, Any] = {
            "mode": mode_result.mode,
            "budget_name": mode_result.budget_name,
            "effective_ants": mode_result.effective_ants,
            "effective_iterations": mode_result.effective_iterations,
            "effective_max_epochs": mode_result.effective_max_epochs,
            "status": "success",
            "seed_results": [],
        }
        for seed_run in mode_result.seed_runs:
            seed_record: dict[str, Any] = {
                "seed": seed_run.seed,
                "run_id": seed_run.run_id,
                "runtime_seconds": 0.0,
                "status": "success",
                "test_evaluation_performed": False,
            }
            try:
                best_epoch = _aggregate_best_epoch(mode_result, seed_run.seed)
                config = ExperimentConfig(
                    mode=mode_result.mode,
                    ants=mode_result.effective_ants,
                    iterations=mode_result.effective_iterations,
                    max_epochs=mode_result.effective_max_epochs,
                    run_seed=seed_run.seed,
                    budget_name=mode_result.budget_name,
                    search_space={
                        key: [value]
                        for key, value in configuration.items()
                        if key != "loss"
                    },
                )
                model_path = (
                    root / mode_result.mode / mode_result.budget_name
                    / (seed_run.run_id or f"seed_{seed_run.seed}") / "best_model.keras"
                )
                config_path = model_path.parent / "best_config.json"
                _write_config(
                    config_path,
                    mode_result.mode,
                    seed_run.seed,
                    configuration,
                    best_epoch,
                    data,
                    seed_run.run_id or f"{mode_result.mode}-{mode_result.budget_name}-seed-{seed_run.seed}",
                )
                final_metrics = trainer(
                    configuration,
                    config,
                    data,
                    best_epoch,
                    str(model_path),
                )
                seed_record.update(final_metrics)
                seed_record["runtime_seconds"] = float(
                    final_metrics.get("runtime_seconds", 0.0)
                )
                seed_record["test_evaluation_performed"] = True
                result.test_evaluations += 1
            except Exception as error:
                seed_record["status"] = "failed"
                seed_record["failure_reason"] = f"{type(error).__name__}: {error}"
                mode_record["status"] = "failed"
                result.status = "failed"
            mode_record["seed_results"].append(seed_record)
        result.mode_results.append(mode_record)

    result.runtime_seconds = time.perf_counter() - started
    result.total_workflow_runtime_seconds = (
        result.primary_runtime_seconds + result.runtime_seconds
    )
    if result.status == "failed" and not result.failure_reason:
        result.failure_reason = "One or more final seed evaluations failed"
    _write_report(result, root)
    return result
