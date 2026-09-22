"""Repeated primary runs and validation-only aggregate confirmation."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from aco_optimizer import ACOOptimizer, Evaluator
from config import ExperimentConfig, family_for_mode, get_search_space, mode_interpretation
from evaluation_contract import EvaluationResult


DEFAULT_PRIMARY_MODES = ("paper_conventional", "improved")
DEFAULT_PRIMARY_SEEDS = (42, 43, 44)


def environment_metadata() -> dict[str, str]:
    import numpy as np

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


def _failed_result(error: Exception) -> EvaluationResult:
    return EvaluationResult(
        status="failed",
        fitness=None,
        train_accuracy=None,
        validation_accuracy=None,
        validation_loss=None,
        best_epoch=None,
        training_time_seconds=0.0,
        failure_reason=f"{type(error).__name__}: {error}",
    )


@dataclass
class SeedRunResult:
    mode: str
    family: str
    seed: int
    status: str
    runtime_seconds: float
    run_id: str = ""
    global_best_candidate_id: str | None = None
    global_best_configuration: dict[str, Any] | None = None
    best_validation_accuracy: float | None = None
    best_validation_loss: float | None = None
    best_epoch: int | None = None
    failed_trials: int = 0
    cache_hits: int = 0
    trial_count: int = 0
    output_dir: str | None = None
    failure_reason: str = ""
    budget_name: str = "main"
    effective_ants: int = 20
    effective_iterations: int = 20
    effective_max_epochs: int = 10
    best_fitness: float | None = None
    best_training_accuracy: float | None = None
    best_training_time_seconds: float | None = None
    best_effective_learning_rate: float | None = None
    best_semantic_warning: str = ""
    best_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregateCandidate:
    candidate_id: str
    configuration: dict[str, Any]
    evaluations: list[EvaluationResult]
    mean_validation_accuracy: float = field(init=False)
    mean_validation_loss: float = field(init=False)
    mean_training_time: float = field(init=False)
    evaluation_count: int = field(init=False)
    failed_evaluation_count: int = field(init=False)
    cache_hit_count: int = field(init=False)

    def __post_init__(self) -> None:
        valid = [
            result
            for result in self.evaluations
            if result.status in {"success", "cached"}
            and result.fitness is not None
            and result.validation_accuracy is not None
        ]
        self.evaluation_count = len(valid)
        self.failed_evaluation_count = sum(
            result.status == "failed" for result in self.evaluations
        )
        self.cache_hit_count = sum(
            result.status == "cached" for result in self.evaluations
        )
        if not valid:
            raise ValueError("Aggregate candidate requires at least one valid evaluation")
        self.mean_validation_accuracy = sum(
            float(result.validation_accuracy) for result in valid
        ) / len(valid)
        self.mean_validation_loss = sum(
            float(result.validation_loss)
            if result.validation_loss is not None
            else float("inf")
            for result in valid
        ) / len(valid)
        self.mean_training_time = sum(
            float(result.training_time_seconds) for result in valid
        ) / len(valid)


@dataclass
class ModeExperimentResult:
    mode: str
    family: str
    interpretation: str
    seed_runs: list[SeedRunResult]
    aggregate: AggregateCandidate | None = None
    aggregate_candidate_count: int = 0
    aggregate_evaluation_count: int = 0
    aggregate_runtime_seconds: float = 0.0
    aggregate_failed_trials: int = 0
    aggregate_cache_hits: int = 0
    budget_name: str = "main"
    effective_ants: int = 20
    effective_iterations: int = 20
    effective_max_epochs: int = 10
    aggregate_status: str = "success"
    aggregate_failure_reason: str = ""
    confirmation_evaluations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExperimentReport:
    mode_results: list[ModeExperimentResult]
    test_evaluations: int = 0
    output_path: str | None = None
    comparison_note: str = (
        "paper_conventional versus improved is a pipeline comparison: "
        "their search spaces differ and the result is not a controlled "
        "pheromone-update ablation."
    )
    evaluator_type: str = "cnn_mnist"
    environment: dict[str, str] = field(default_factory=environment_metadata)


def _aggregate_sort_key(candidate: AggregateCandidate) -> tuple[float, float, float, str]:
    return (
        -candidate.mean_validation_accuracy,
        candidate.mean_validation_loss,
        candidate.mean_training_time,
        candidate.candidate_id,
    )


def aggregate_candidates(
    candidates: Iterable[AggregateCandidate],
    seeds: Iterable[int],
) -> tuple[AggregateCandidate, int, int]:
    """Select using validation metrics only, after all requested seed evaluations."""
    candidates = list(candidates)
    expected_count = len(tuple(seeds))
    complete = [
        candidate
        for candidate in candidates
        if candidate.evaluation_count == expected_count
    ]
    if not complete:
        raise RuntimeError("No aggregate candidate has valid evaluations for every seed")
    return min(complete, key=_aggregate_sort_key)


def _configuration_from_seed_run(run: SeedRunResult) -> dict[str, Any]:
    if run.global_best_configuration is None:
        raise ValueError("A successful seed run must have a global-best configuration")
    return dict(run.global_best_configuration)


def _run_one_seed(
    mode: str,
    seed: int,
    budget: dict[str, int],
    budget_name: str,
    evaluator_factory: Callable[[ExperimentConfig], Evaluator],
    search_space: dict[str, list[Any]],
    output_dir: Path | None,
    cache_enabled: bool,
    evaluator_type: str = "cnn_mnist",
) -> tuple[SeedRunResult, dict[str, EvaluationResult]]:
    config = ExperimentConfig(
        mode=mode,
        ants=budget["ants"],
        iterations=budget["iterations"],
        max_epochs=budget["max_epochs"],
        run_seed=seed,
        budget_name=budget_name,
        cache_enabled=cache_enabled,
        search_space=search_space,
    )
    try:
        evaluator = evaluator_factory(config)
    except Exception as error:
        evaluator = lambda _configuration, _experiment, factory_error=error: _failed_result(
            factory_error
        )
    optimizer = ACOOptimizer(config, evaluator)
    started = time.perf_counter()
    run_id = (
        f"{mode}-{budget_name}-ants{budget['ants']}-"
        f"iterations{budget['iterations']}-epochs{budget['max_epochs']}-seed-{seed}"
    )
    run_output = output_dir / mode / budget_name / run_id if output_dir else None
    try:
        best = optimizer.run()
        status = "success"
        failure_reason = ""
    except Exception as error:
        best = None
        status = "failed"
        failure_reason = f"{type(error).__name__}: {error}"
        optimizer.run_status = "failed"
    finally:
        if run_output is not None:
            optimizer.write_logs(str(run_output))
            run_output.mkdir(parents=True, exist_ok=True)
            (run_output / "run_summary.json").write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "mode": mode,
                        "budget_name": budget_name,
                        "seed": seed,
                        "runtime_seconds": time.perf_counter() - started,
                        "run_status": optimizer.run_status,
                        "failure_reason": failure_reason,
                        "evaluator_type": evaluator_type,
                        "trial_count": len(optimizer.trial_rows),
                        "failed_trials": sum(
                            row["status"] == "failed" for row in optimizer.trial_rows
                        ),
                        "cache_hits": sum(
                            bool(row["cache_hit"]) for row in optimizer.trial_rows
                        ),
                        "environment": environment_metadata(),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

    rows = optimizer.trial_rows
    seed_result = SeedRunResult(
        mode=mode,
        family=family_for_mode(mode),
        seed=seed,
        run_id=(
            f"{mode}-{budget_name}-ants{budget['ants']}-"
            f"iterations{budget['iterations']}-epochs{budget['max_epochs']}-seed-{seed}"
        ),
        status=status,
        runtime_seconds=time.perf_counter() - started,
        global_best_candidate_id=best.candidate_id if best else None,
        global_best_configuration=dict(best.configuration) if best else None,
        best_validation_accuracy=best.result.validation_accuracy if best else None,
        best_validation_loss=best.result.validation_loss if best else None,
        best_epoch=best.result.best_epoch if best else None,
        best_fitness=best.result.fitness if best else None,
        best_training_accuracy=best.result.train_accuracy if best else None,
        best_training_time_seconds=best.result.training_time_seconds if best else None,
        best_effective_learning_rate=best.result.effective_learning_rate if best else None,
        best_semantic_warning=best.result.semantic_warning if best else "",
        best_metadata=(best.result.metadata or {}) if best else {},
        failed_trials=sum(row["status"] == "failed" for row in rows),
        cache_hits=sum(bool(row["cache_hit"]) for row in rows),
        trial_count=len(rows),
        output_dir=str(run_output) if run_output else None,
        failure_reason=failure_reason,
        budget_name=budget_name,
        effective_ants=budget["ants"],
        effective_iterations=budget["iterations"],
        effective_max_epochs=budget["max_epochs"],
    )
    return seed_result, optimizer.evaluation_results


def _confirm_candidates(
    mode: str,
    seed_runs: list[SeedRunResult],
    seeds: tuple[int, ...],
    budget: dict[str, int],
    budget_name: str,
    evaluator_factory: Callable[[ExperimentConfig], Evaluator],
    search_space: dict[str, list[Any]],
    cache_enabled: bool,
    seed_caches: dict[int, dict[str, EvaluationResult]],
) -> tuple[AggregateCandidate | None, int, int, float, int, int, list[dict[str, Any]]]:
    configurations: dict[str, dict[str, Any]] = {}
    for run in seed_runs:
        if run.status == "success" and run.global_best_candidate_id:
            configurations[run.global_best_candidate_id] = _configuration_from_seed_run(run)

    candidates: list[AggregateCandidate] = []
    all_evaluations: list[EvaluationResult] = []
    confirmation_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for identifier, configuration in sorted(configurations.items()):
        evaluations: list[EvaluationResult] = []
        for seed in seeds:
            config = ExperimentConfig(
                mode=mode,
                ants=budget["ants"],
                iterations=budget["iterations"],
                max_epochs=budget["max_epochs"],
                run_seed=seed,
                budget_name=budget_name,
                cache_enabled=cache_enabled,
                search_space=search_space,
            )
            cached_result = seed_caches.get(seed, {}).get(identifier)
            if cache_enabled and cached_result is not None:
                result = replace(cached_result, status="cached")
            else:
                try:
                    evaluator = evaluator_factory(config)
                    result = evaluator(configuration, config)
                except Exception as error:
                    result = _failed_result(error)
            evaluations.append(result)
            all_evaluations.append(result)
            confirmation_records.append(
                {
                    "candidate_id": identifier,
                    "run_id": (
                        f"{mode}-{budget_name}-ants{budget['ants']}-"
                        f"iterations{budget['iterations']}-epochs{budget['max_epochs']}-seed-{seed}"
                    ),
                    "configuration": configuration,
                    "seed": seed,
                    "status": result.status,
                    "fitness": result.fitness,
                    "train_accuracy": result.train_accuracy,
                    "validation_accuracy": result.validation_accuracy,
                    "validation_loss": result.validation_loss,
                    "training_time_seconds": result.training_time_seconds,
                    "best_epoch": result.best_epoch,
                    "failure_reason": result.failure_reason,
                    "effective_learning_rate": result.effective_learning_rate,
                    "semantic_warning": result.semantic_warning,
                    "cache_hit": result.status == "cached",
                    "budget_name": budget_name,
                    "effective_ants": budget["ants"],
                    "effective_iterations": budget["iterations"],
                    "effective_max_epochs": budget["max_epochs"],
                    "metadata": result.metadata or {},
                }
            )
        valid = [
            result
            for result in evaluations
            if result.status in {"success", "cached"}
            and result.fitness is not None
            and result.validation_accuracy is not None
        ]
        if len(valid) == len(seeds):
            candidates.append(AggregateCandidate(identifier, configuration, evaluations))

    aggregate = aggregate_candidates(candidates, seeds) if candidates else None
    return (
        aggregate,
        len(configurations),
        len(configurations) * len(seeds),
        time.perf_counter() - started,
        sum(result.status == "failed" for result in all_evaluations),
        sum(result.status == "cached" for result in all_evaluations),
        confirmation_records,
    )


def _aggregate_payload(candidate: AggregateCandidate | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "candidate_id": candidate.candidate_id,
        "configuration": candidate.configuration,
        "mean_validation_accuracy": candidate.mean_validation_accuracy,
        "mean_validation_loss": candidate.mean_validation_loss,
        "mean_training_time": candidate.mean_training_time,
        "evaluation_count": candidate.evaluation_count,
        "failed_evaluation_count": candidate.failed_evaluation_count,
        "cache_hit_count": candidate.cache_hit_count,
    }


def write_report(report: ExperimentReport, output_dir: str) -> str:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "primary_experiment_report.json"
    payload = {
        "test_evaluations": report.test_evaluations,
        "comparison_note": report.comparison_note,
        "evaluator_type": report.evaluator_type,
        "environment": report.environment,
        "mode_results": [
            {
                "mode": result.mode,
                "family": result.family,
                "interpretation": result.interpretation,
                "seed_runs": [asdict(run) for run in result.seed_runs],
                "aggregate": _aggregate_payload(result.aggregate),
                "aggregate_candidate_count": result.aggregate_candidate_count,
                "aggregate_evaluation_count": result.aggregate_evaluation_count,
                "aggregate_runtime_seconds": result.aggregate_runtime_seconds,
                "aggregate_failed_trials": result.aggregate_failed_trials,
                "aggregate_cache_hits": result.aggregate_cache_hits,
                "budget_name": result.budget_name,
                "effective_ants": result.effective_ants,
                "effective_iterations": result.effective_iterations,
                "effective_max_epochs": result.effective_max_epochs,
                "aggregate_status": result.aggregate_status,
                "aggregate_failure_reason": result.aggregate_failure_reason,
                "confirmation_evaluations": result.confirmation_evaluations,
            }
            for result in report.mode_results
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    report.output_path = str(path)
    return str(path)


def run_primary_experiments(
    modes: tuple[str, ...] = DEFAULT_PRIMARY_MODES,
    seeds: tuple[int, ...] = DEFAULT_PRIMARY_SEEDS,
    budget: dict[str, int] | None = None,
    budget_name: str = "main",
    evaluator_factory: Callable[[ExperimentConfig], Evaluator] | None = None,
    search_spaces: dict[str, dict[str, list[Any]]] | None = None,
    output_dir: str | None = None,
    cache_enabled: bool = False,
    evaluator_type: str = "cnn_mnist",
) -> ExperimentReport:
    """Run primary modes, then confirm global-best candidates on every seed.

    This function deliberately has no test-set input or test evaluator. Test
    evaluation belongs to the final-retraining ticket after this report.
    """
    if not modes or not seeds:
        raise ValueError("At least one mode and one seed are required")
    if evaluator_factory is None:
        raise ValueError("Primary experiment requires an evaluator_factory")
    budget = budget or {"ants": 20, "iterations": 20, "max_epochs": 10}
    if tuple(modes) != DEFAULT_PRIMARY_MODES:
        raise ValueError("Primary experiment requires paper_conventional and improved modes")
    if tuple(seeds) != DEFAULT_PRIMARY_SEEDS:
        raise ValueError("Primary experiment requires seeds 42, 43, and 44")
    if budget != {"ants": 20, "iterations": 20, "max_epochs": 10}:
        raise ValueError("Primary experiment requires the main 20x20x10 budget")
    if budget_name != "main":
        raise ValueError("Primary experiment requires budget_name='main'")
    return _run_experiment_plan(
        modes=tuple(modes),
        seeds=tuple(seeds),
        budget=budget,
        budget_name=budget_name,
        evaluator_factory=evaluator_factory,
        search_spaces=search_spaces,
        output_dir=output_dir,
        cache_enabled=cache_enabled,
        evaluator_type=evaluator_type,
    )


def run_experiment_plan(
    **kwargs: Any,
) -> ExperimentReport:
    """Run a small/custom plan for tests and non-primary diagnostics."""
    return _run_experiment_plan(**kwargs)


def _run_experiment_plan(
    modes: tuple[str, ...],
    seeds: tuple[int, ...],
    budget: dict[str, int],
    budget_name: str,
    evaluator_factory: Callable[[ExperimentConfig], Evaluator],
    search_spaces: dict[str, dict[str, list[Any]]] | None = None,
    output_dir: str | None = None,
    cache_enabled: bool = False,
    evaluator_type: str = "cnn_mnist",
) -> ExperimentReport:
    evaluator_factory = evaluator_factory or (
        lambda _config: (_ for _ in ()).throw(
            ValueError("An evaluator_factory is required for repeated experiments")
        )
    )
    seeds = tuple(seeds)
    root = Path(output_dir) if output_dir else None
    mode_results: list[ModeExperimentResult] = []

    for mode in modes:
        search_space = (
            search_spaces[mode]
            if search_spaces and mode in search_spaces
            else get_search_space(mode)
        )
        seed_run_pairs = [
            _run_one_seed(
                mode,
                seed,
                budget,
                budget_name,
                evaluator_factory,
                search_space,
                root,
                cache_enabled,
                evaluator_type,
            )
            for seed in seeds
        ]
        seed_runs = [pair[0] for pair in seed_run_pairs]
        seed_caches = {seed: pair[1] for seed, pair in zip(seeds, seed_run_pairs)}
        (
            aggregate,
            aggregate_candidate_count,
            aggregate_evaluation_count,
            aggregate_runtime_seconds,
            aggregate_failed_trials,
            aggregate_cache_hits,
            confirmation_evaluations,
        ) = _confirm_candidates(
            mode,
            seed_runs,
            seeds,
            budget,
            budget_name,
            evaluator_factory,
            search_space,
            cache_enabled,
            seed_caches,
        )
        mode_results.append(
            ModeExperimentResult(
                mode=mode,
                family=family_for_mode(mode),
                interpretation=mode_interpretation(mode),
                seed_runs=seed_runs,
                aggregate=aggregate,
                aggregate_candidate_count=aggregate_candidate_count,
                aggregate_evaluation_count=aggregate_evaluation_count,
                aggregate_runtime_seconds=aggregate_runtime_seconds,
                aggregate_failed_trials=aggregate_failed_trials,
                aggregate_cache_hits=aggregate_cache_hits,
                budget_name=budget_name,
                effective_ants=budget["ants"],
                effective_iterations=budget["iterations"],
                effective_max_epochs=budget["max_epochs"],
                aggregate_status="success" if aggregate is not None else "failed",
                aggregate_failure_reason=(
                    "No candidate received valid validation results for every seed"
                    if aggregate is None
                    else ""
                ),
                confirmation_evaluations=confirmation_evaluations,
            )
        )

    report = ExperimentReport(
        mode_results=mode_results,
        evaluator_type=evaluator_type,
        environment=environment_metadata(),
    )
    if output_dir:
        write_report(report, output_dir)
    return report
