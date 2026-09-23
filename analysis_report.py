"""Auditable summaries, plots, and academic notes for ACO experiments."""

from __future__ import annotations

import csv
import json
import platform
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _number(value: str | None) -> float | None:
    if value in {None, "", "None", "nan", "NaN"}:
        return None
    return float(value)


def _bool(value: str | None) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _trial_files(root: Path) -> list[Path]:
    files = sorted(root.glob("**/aco_trials.csv"))
    if not files:
        raise FileNotFoundError(f"No aco_trials.csv found under {root}")
    return files


def _load_trials(root: Path) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for path in _trial_files(root):
        for row in _read_csv(path):
            row["source_path"] = str(path)
            row["run_key"] = row.get("run_id") or str(path.parent)
            row["iteration_number"] = int(row.get("iteration", 0))
            row["fitness_number"] = _number(row.get("fitness"))
            row["validation_accuracy_number"] = _number(row.get("validation_accuracy"))
            row["validation_loss_number"] = _number(row.get("validation_loss"))
            row["training_time_number"] = _number(row.get("training_time_seconds")) or 0.0
            row["cache_hit_bool"] = _bool(row.get("cache_hit"))
            trials.append(row)
    return trials


def _load_pheromone(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("**/pheromone_history.csv")):
        for row in _read_csv(path):
            row["iteration_number"] = int(row.get("iteration", 0))
            row["pheromone_number"] = _number(row.get("pheromone")) or 0.0
            row["probability_number"] = _number(row.get("probability")) or 0.0
            row["run_key"] = row.get("run_id") or str(path.parent)
            rows.append(row)
    return rows


def _load_primary_report(root: Path) -> dict[str, Any]:
    path = root / "primary_experiment_report.json"
    if not path.exists():
        return {"evaluator_type": "unknown", "mode_results": [], "test_evaluations": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_final_report(root: Path) -> dict[str, Any]:
    path = root / "final_evaluation_report.json"
    if not path.exists():
        return {"status": "not_present", "mode_results": [], "test_evaluations": 0}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run_summaries(root: Path) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("**/run_summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        summaries[payload.get("run_id", str(path.parent))] = payload
    return summaries


def _selected_run_ids(
    primary: dict[str, Any], run_summaries: dict[str, dict[str, Any]]
) -> set[str]:
    report_run_ids = {
        seed_run.get("run_id")
        for mode_result in primary.get("mode_results", [])
        for seed_run in mode_result.get("seed_runs", [])
        if seed_run.get("run_id")
    }
    if report_run_ids:
        return report_run_ids
    return set(run_summaries)


def _filter_run_artifacts(
    rows: list[dict[str, Any]], selected_run_ids: set[str]
) -> list[dict[str, Any]]:
    if not selected_run_ids:
        return rows
    return [row for row in rows if row.get("run_key") in selected_run_ids]


def _analysis_environment() -> dict[str, str]:
    import numpy as np

    environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    try:
        import tensorflow as tf

        environment["tensorflow_version"] = tf.__version__
        environment["keras_version"] = str(tf.keras.__version__)
    except (ImportError, AttributeError):
        environment["tensorflow_version"] = "unavailable"
        environment["keras_version"] = "unavailable"
    return environment


def _final_summary(final: dict[str, Any]) -> dict[str, Any]:
    seed_results = [
        seed_result
        for mode_result in final.get("mode_results", [])
        for seed_result in mode_result.get("seed_results", [])
    ]
    return {
        "status": final.get("status", "not_present"),
        "test_evaluations": final.get("test_evaluations", 0),
        "runtime_seconds": final.get("runtime_seconds", 0.0),
        "primary_runtime_seconds": final.get("primary_runtime_seconds", 0.0),
        "total_workflow_runtime_seconds": final.get(
            "total_workflow_runtime_seconds", 0.0
        ),
        "per_seed_runtime_seconds": {
            f"{mode_result.get('mode')}/seed_{seed_result.get('seed')}": seed_result.get(
                "runtime_seconds", 0.0
            )
            for mode_result in final.get("mode_results", [])
            for seed_result in mode_result.get("seed_results", [])
        },
        "failed_seeds": sum(item.get("status") == "failed" for item in seed_results),
        "test_accuracies": {
            f"{mode_result.get('mode')}/seed_{seed_result.get('seed')}": seed_result.get("test_accuracy")
            for mode_result in final.get("mode_results", [])
            for seed_result in mode_result.get("seed_results", [])
            if seed_result.get("test_accuracy") is not None
        },
    }


def _methodological_notes(primary: dict[str, Any]) -> list[str]:
    notes = [
        "Journal facts: MNIST, 20 ants, pheromone-only selection, rho=0.25, constant +0.5 reinforcement, and the eight-dimensional paper search space.",
        "Interpretations: paper_literal updates after every ant; paper_conventional updates once per iteration. Neither is evidence of the authors' unavailable original code.",
        "Development decisions: fixed CNN architecture, deterministic 50,000/10,000 validation split, validation fitness, caching, early stopping, repeated seeds, and untouched final test evaluation.",
        "Smoke and pilot runs validate implementation behavior and are not primary scientific results.",
        "paper_conventional versus improved is a pipeline comparison, not a controlled pheromone-update ablation, because their search spaces differ.",
        "Budget provenance: 20 ants follows the explicit journal setting; the selected iteration count and maximum epoch count are project computational-budget decisions because the article does not specify them.",
        "The reduced budget targets a feasible reproducible three-seed experiment on available hardware; it does not claim literal replication of the article's runtime.",
    ]
    comparison_note = primary.get("comparison_note")
    if comparison_note:
        notes.append(str(comparison_note))
    return notes


def _summary(trials: list[dict[str, Any]], primary: dict[str, Any]) -> dict[str, Any]:
    valid = [row for row in trials if row["validation_accuracy_number"] is not None]
    by_run_iteration: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in valid:
        by_run_iteration[(row["run_key"], row["iteration_number"])].append(
            row["validation_accuracy_number"]
        )
    best_by_run_iteration = {
        f"{run_key}/iteration_{iteration}": max(values)
        for (run_key, iteration), values in sorted(by_run_iteration.items())
    }
    average_by_run_iteration = {
        f"{run_key}/iteration_{iteration}": sum(values) / len(values)
        for (run_key, iteration), values in sorted(by_run_iteration.items())
    }
    per_run: dict[str, dict[str, Any]] = {}
    for row in trials:
        key = row["run_key"]
        item = per_run.setdefault(
            key,
            {"mode": row.get("mode", "unknown"), "seed": row.get("seed", "unknown"),
             "trial_count": 0, "failed_trials": 0, "cache_hits": 0,
             "runtime_seconds": 0.0, "best_validation_accuracy": None,
             "budget_name": row.get("budget", "unknown"),
             "effective_ants": row.get("effective_ants"),
             "effective_iterations": row.get("effective_iterations"),
             "effective_max_epochs": row.get("effective_max_epochs")},
        )
        item["trial_count"] += 1
        item["failed_trials"] += row["status"] == "failed"
        item["cache_hits"] += row["cache_hit_bool"]
        item["runtime_seconds"] += row["training_time_number"]
        accuracy = row["validation_accuracy_number"]
        if accuracy is not None:
            item["best_validation_accuracy"] = max(
                accuracy, item["best_validation_accuracy"] or accuracy
            )
    environment: dict[str, Any] = _analysis_environment()
    confirmation_failed = 0
    confirmation_cache_hits = 0
    environment_by_run: dict[str, dict[str, Any]] = {}
    for mode_result in primary.get("mode_results", []):
        for seed_run in mode_result.get("seed_runs", []):
            key = seed_run.get("run_id") or (
                f"{mode_result.get('mode', 'unknown')}-"
                f"{mode_result.get('budget_name', 'unknown')}-"
                f"seed-{seed_run.get('seed', 'unknown')}"
            )
            if key in per_run and seed_run.get("runtime_seconds") is not None:
                per_run[key]["runtime_seconds"] = float(seed_run["runtime_seconds"])
            metadata = seed_run.get("best_metadata") or {}
            for name in ("python_version", "platform", "numpy_version", "tensorflow_version", "keras_version"):
                if metadata.get(name) is not None:
                    environment[name] = metadata[name]
            environment_by_run[key] = {
                name: metadata.get(name, "unavailable")
                for name in ("python_version", "platform", "numpy_version", "tensorflow_version", "keras_version")
            }
        for record in mode_result.get("confirmation_evaluations", []):
            confirmation_failed += record.get("status") == "failed"
            confirmation_cache_hits += bool(record.get("cache_hit"))
    for mode_result in primary.get("mode_results", []):
        budget = mode_result.get("budget_name", "unknown")
        for seed_run in mode_result.get("seed_runs", []):
            run_key = seed_run.get("run_id") or (
                f"{mode_result.get('mode', 'unknown')}-{budget}-"
                f"seed-{seed_run.get('seed', 'unknown')}"
            )
            if run_key in per_run:
                per_run[run_key]["budget_name"] = budget
                per_run[run_key]["effective_ants"] = mode_result.get("effective_ants")
                per_run[run_key]["effective_iterations"] = mode_result.get("effective_iterations")
                per_run[run_key]["effective_max_epochs"] = mode_result.get("effective_max_epochs")
    total_evaluations = len(trials) + sum(
        len(mode_result.get("confirmation_evaluations", []))
        for mode_result in primary.get("mode_results", [])
    )
    total_cache_hits = sum(row["cache_hit_bool"] for row in trials) + confirmation_cache_hits
    split_ids = sorted(
        {
            str(metadata.get("split_id"))
            for mode_result in primary.get("mode_results", [])
            for seed_run in mode_result.get("seed_runs", [])
            for metadata in [seed_run.get("best_metadata") or {}]
            if metadata.get("split_id") is not None
        }
    )
    return {
        "evaluator_type": primary.get("evaluator_type", "unknown"),
        "trial_count": len(trials),
        "successful_trials": sum(row["status"] in {"success", "cached"} for row in trials),
        "failed_trials": sum(row["status"] == "failed" for row in trials),
        "cache_hits": sum(row["cache_hit_bool"] for row in trials),
        "cache_hit_rate": total_cache_hits / total_evaluations if total_evaluations else 0.0,
        "runtime_seconds": sum(item["runtime_seconds"] for item in per_run.values()),
        "confirmation_runtime_seconds": sum(
            item.get("aggregate_runtime_seconds", 0.0)
            for item in primary.get("mode_results", [])
        ),
        "primary_runtime_seconds": primary.get("runtime_summary", {}).get(
            "primary_runtime_seconds", 0.0
        ),
        "primary_candidate_evaluation_count": primary.get("runtime_summary", {}).get(
            "candidate_evaluation_count", len(trials)
        ),
        "primary_confirmation_evaluation_count": primary.get(
            "runtime_summary", {}
        ).get("confirmation_evaluation_count", 0),
        "best_validation_accuracy": max(
            (row["validation_accuracy_number"] for row in valid), default=None
        ),
        "best_validation_accuracy_by_run_iteration": best_by_run_iteration,
        "average_validation_accuracy_by_run_iteration": average_by_run_iteration,
        "per_run": per_run,
        "test_evaluations": primary.get("test_evaluations", 0),
        "mode_count": len(primary.get("mode_results", [])),
        "environment": environment,
        "environment_by_run": environment_by_run,
        "confirmation_failed_trials": confirmation_failed,
        "confirmation_cache_hits": confirmation_cache_hits,
        "dataset_protocol": {
            "dataset": "MNIST",
            "dataset_seed": 2024,
            "train_size": 50000,
            "validation_size": 10000,
            "test_size": 10000,
            "test_used_for_selection": False,
            "split_ids": split_ids,
        },
    }


def _plot(root: Path, trials: list[dict[str, Any]], pheromone: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to generate analysis plots") from error

    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots()
    by_run: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for row in trials:
        if row["validation_accuracy_number"] is not None:
            key = row["run_key"]
            by_run[key].append((row["iteration_number"], row["validation_accuracy_number"], row["validation_accuracy_number"]))
    for key, values in sorted(by_run.items()):
        grouped: dict[int, list[float]] = defaultdict(list)
        for iteration, _, accuracy in values:
            grouped[iteration].append(accuracy)
        x = sorted(grouped)
        axis.plot(x, [max(grouped[i]) for i in x], marker="o", label=key)
        axis.plot(
            x,
            [sum(grouped[i]) / len(grouped[i]) for i in x],
            linestyle="--",
            alpha=0.7,
            label=f"{key} average",
        )
    axis.set(xlabel="Iteration", ylabel="Validation accuracy", title="ACO convergence")
    if by_run:
        axis.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(output / "convergence.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots()
    pheromone_series = _group_history(pheromone, "pheromone_number")
    for key, values in pheromone_series.items():
        axis.plot(values[0], values[1], label=key)
    axis.set(xlabel="Iteration", ylabel="Pheromone", title="Pheromone evolution")
    if 0 < len(pheromone_series) <= 20:
        axis.legend(fontsize="small")
    figure.subplots_adjust(bottom=0.3)
    figure.savefig(output / "pheromone_evolution.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots()
    probability_series = _group_history(pheromone, "probability_number")
    for key, values in probability_series.items():
        axis.plot(values[0], values[1], label=key)
    axis.set(xlabel="Iteration", ylabel="Selection probability", title="Probability evolution")
    if 0 < len(probability_series) <= 20:
        axis.legend(fontsize="small")
    figure.subplots_adjust(bottom=0.3)
    figure.savefig(output / "probability_evolution.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots()
    per_run = summary["per_run"]
    labels = list(per_run)
    extra_labels: list[str] = []
    extra_values: list[float] = []
    if summary.get("confirmation_runtime_seconds"):
        extra_labels.append("confirmation")
        extra_values.append(summary["confirmation_runtime_seconds"])
    if summary.get("final_evaluation", {}).get("runtime_seconds"):
        extra_labels.append("final_evaluation")
        extra_values.append(summary["final_evaluation"]["runtime_seconds"])
    plot_labels = labels + extra_labels
    plot_values = [per_run[key]["runtime_seconds"] for key in labels]
    plot_values.extend(extra_values)
    axis.bar(plot_labels, plot_values)
    axis.set_ylabel("Runtime (seconds)")
    axis.set_title("Runtime per run and evaluation stage")
    axis.tick_params(axis="x", labelrotation=45)
    figure.subplots_adjust(bottom=0.3)
    figure.savefig(output / "runtime_summary.png", dpi=150)
    plt.close(figure)


def _group_history(rows: list[dict[str, Any]], metric: str) -> dict[str, tuple[list[int], list[float]]]:
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (
            f"{row.get('mode', '')}:"
            f"{row.get('hyperparameter', '')}:{row.get('phase', '')}"
        )
        x_value = int(row.get("update_step", row["iteration_number"]))
        grouped[key][x_value].append(row[metric])
    return {
        key: (
            sorted(values),
            [sum(values[iteration]) / len(values[iteration]) for iteration in sorted(values)],
        )
        for key, values in grouped.items()
    }


def _write_markdown(root: Path, summary: dict[str, Any], notes: list[str], primary: dict[str, Any], final: dict[str, Any]) -> None:
    output = root / "analysis"
    lines = [
        "# ACO-CNN experiment analysis",
        "",
        f"Evaluator type: `{summary['evaluator_type']}`",
        f"Trial count: `{summary['trial_count']}`",
        f"Best validation accuracy: `{summary['best_validation_accuracy']}`",
        f"Per-run summaries: `{len(summary['per_run'])}`",
        f"Failed trials: `{summary['failed_trials']}`",
        f"Cache hits: `{summary['cache_hits']}`",
        f"Cache hit rate: `{summary['cache_hit_rate']:.3f}`",
        f"Test evaluations recorded by search report: `{summary['test_evaluations']}`",
        "",
        "## Methodological notes",
        "",
    ]
    lines.extend(f"- {note}" for note in notes)
    lines.extend(["", "## Final evaluation", ""])
    if final.get("status") != "not_present":
        lines.append(f"- Final evaluation status: `{final.get('status')}`")
        lines.append(f"- Final test evaluations: `{final.get('test_evaluations', 0)}`")
        for mode_result in final.get("mode_results", []):
            for seed_result in mode_result.get("seed_results", []):
                if "test_accuracy" in seed_result:
                    lines.append(
                        f"- `{mode_result.get('mode')}/seed_{seed_result.get('seed')}` test accuracy: `{seed_result['test_accuracy']}`"
                    )
                elif seed_result.get("status") == "failed":
                    lines.append(
                        f"- `{mode_result.get('mode')}/seed_{seed_result.get('seed')}` final evaluation failed: "
                        f"`{seed_result.get('failure_reason', 'unknown failure')}`"
                    )
    else:
        lines.append("- Final evaluation report is not present; analysis contains search results only.")
    lines.extend(["", "## Environment and protocol", ""])
    lines.append(f"- Evaluator type: `{summary['evaluator_type']}`. Runs remain grouped by budget-aware `run_id` in the summary and source logs.")
    lines.append(f"- Search runtime across mode/seed runs: `{summary.get('runtime_seconds', 0.0)}` seconds.")
    lines.append(
        f"- Primary workflow runtime (tuning plus confirmation): `{summary.get('primary_runtime_seconds', 0.0)}` seconds; "
        f"candidate evaluations: `{summary.get('primary_candidate_evaluation_count', 0)}`; "
        f"confirmation evaluations: `{summary.get('primary_confirmation_evaluation_count', 0)}`."
    )
    for name, value in sorted(summary.get("environment", {}).items()):
        lines.append(f"- {name}: `{value}`")
    lines.append(
        f"- Confirmation failed trials: `{summary.get('confirmation_failed_trials', 0)}`; "
        f"confirmation cache hits: `{summary.get('confirmation_cache_hits', 0)}`."
    )
    lines.append(
        f"- Confirmation runtime: `{summary.get('confirmation_runtime_seconds', 0.0)}` seconds."
    )
    for mode_result in primary.get("mode_results", []):
        lines.append(
            f"- `{mode_result.get('mode')}` budget: `{mode_result.get('budget_name')}`; "
            f"effective `{mode_result.get('effective_ants')} ants × {mode_result.get('effective_iterations')} iterations × {mode_result.get('effective_max_epochs')} epochs`."
        )
        for seed_run in mode_result.get("seed_runs", []):
            metadata = seed_run.get("best_metadata", {})
            if metadata:
                lines.append(
                    f"- `{mode_result.get('mode')}/seed_{seed_run.get('seed')}` split `{metadata.get('split_id', 'unknown')}`."
                )
    lines.append("- Final policy: retrain on 60,000 development images for the confirmation best epoch, then evaluate once on the untouched 10,000-image test set.")
    dataset_protocol = summary.get("dataset_protocol", {})
    lines.append(
        f"- Dataset protocol: `{dataset_protocol.get('dataset')}`, seed `{dataset_protocol.get('dataset_seed')}`, "
        f"split `{dataset_protocol.get('train_size')}/{dataset_protocol.get('validation_size')}/{dataset_protocol.get('test_size')}` "
        f"(train/validation/test), split IDs `{dataset_protocol.get('split_ids', [])}`."
    )
    final_summary = summary.get("final_evaluation", {})
    lines.append(
        f"- Final evaluation status: `{final_summary.get('status', 'not_present')}`; "
        f"runtime: `{final_summary.get('runtime_seconds', 0.0)}` seconds; "
        f"failed seeds: `{final_summary.get('failed_seeds', 0)}`."
    )
    lines.append("- The untouched test set is an evaluation-only dataset and must not be used to select configurations.")
    (output / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_experiment(root_dir: str) -> dict[str, Any]:
    root = Path(root_dir)
    primary = _load_primary_report(root)
    final = _load_final_report(root)
    run_summaries = _load_run_summaries(root)
    selected_run_ids = _selected_run_ids(primary, run_summaries)
    trials = _filter_run_artifacts(_load_trials(root), selected_run_ids)
    pheromone = _filter_run_artifacts(_load_pheromone(root), selected_run_ids)
    summary = _summary(trials, primary)
    summary["run_summaries"] = run_summaries
    if summary["evaluator_type"] == "unknown":
        evaluator_types = {
            payload.get("evaluator_type")
            for payload in run_summaries.values()
            if payload.get("evaluator_type")
        }
        if len(evaluator_types) == 1:
            summary["evaluator_type"] = evaluator_types.pop()
    summary["environment"] = {
        **summary.get("environment", {}),
        **primary.get("environment", {}),
    }
    for run_id, run_summary in run_summaries.items():
        summary["environment_by_run"][run_id] = run_summary.get("environment", {})
        summary["environment"].update(run_summary.get("environment", {}))
        item = summary["per_run"].setdefault(
            run_id,
            {
                "mode": run_summary.get("mode", "unknown"),
                "seed": run_summary.get("seed", "unknown"),
                "trial_count": run_summary.get("trial_count", 0),
                "failed_trials": run_summary.get("failed_trials", 0),
                "cache_hits": run_summary.get("cache_hits", 0),
                "runtime_seconds": 0.0,
                "best_validation_accuracy": None,
                "budget_name": run_summary.get("budget_name", "unknown"),
            },
        )
        item["runtime_seconds"] = run_summary.get(
            "runtime_seconds", item["runtime_seconds"]
        )
        item["failed_trials"] = run_summary.get("failed_trials", item["failed_trials"])
        item["cache_hits"] = run_summary.get("cache_hits", item["cache_hits"])
        for field in ("effective_ants", "effective_iterations", "effective_max_epochs"):
            if field in run_summary:
                item[field] = run_summary[field]
    if run_summaries:
        summary["trial_count"] = sum(
            item["trial_count"] for item in summary["per_run"].values()
        )
        summary["failed_trials"] = sum(
            item["failed_trials"] for item in summary["per_run"].values()
        )
        summary["cache_hits"] = sum(
            item["cache_hits"] for item in summary["per_run"].values()
        )
        summary["successful_trials"] = summary["trial_count"] - summary["failed_trials"]
        summary["runtime_seconds"] = sum(
            item["runtime_seconds"] for item in summary["per_run"].values()
        )
        total_evaluations = summary["trial_count"] + sum(
            len(mode_result.get("confirmation_evaluations", []))
            for mode_result in primary.get("mode_results", [])
        )
        summary["cache_hit_rate"] = (
            (summary["cache_hits"] + summary["confirmation_cache_hits"])
            / total_evaluations
            if total_evaluations
            else 0.0
        )
    summary["final_evaluation"] = _final_summary(final)
    notes = _methodological_notes(primary)
    _plot(root, trials, pheromone, summary)
    _write_markdown(root, summary, notes, primary, final)
    result = {
        **summary,
        "trial_summary": summary,
        "methodological_notes": "\n".join(notes),
        "final_evaluation": final,
    }
    (root / "analysis" / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
