"""Command-line entry point for the ACO-CNN experiments."""

from __future__ import annotations

import argparse
import time
from typing import Any

from aco_optimizer import ACOOptimizer
from config import ExperimentConfig, budget_values, get_search_space
from evaluation_contract import EvaluationResult


def synthetic_evaluator(configuration: dict[str, Any], experiment: ExperimentConfig) -> EvaluationResult:
    """Small deterministic evaluator for testing ACO mechanics without TensorFlow."""
    score = 0.5
    score += 0.02 if configuration.get("activation") == "relu" else 0.0
    score += 0.02 if configuration.get("optimizer") == "adam" else 0.0
    score += 0.01 if configuration.get("dense_units_1") in {64, 128} else 0.0
    return EvaluationResult(
        status="success",
        fitness=score,
        train_accuracy=min(score + 0.01, 1.0),
        validation_accuracy=score,
        validation_loss=1.0 - score,
        best_epoch=1,
        training_time_seconds=0.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run discrete ACO for CNN hyperparameter tuning.")
    parser.add_argument("--mode", choices=["paper_literal", "paper_conventional", "improved"], default="improved")
    parser.add_argument(
        "--budget",
        choices=["smoke", "pilot", "main", "diagnostic"],
        default="smoke",
    )
    parser.add_argument("--ants", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Test ACO without TensorFlow/MNIST.")
    parser.add_argument("--output-dir", default="experiments")
    parser.add_argument(
        "--primary",
        action="store_true",
        help="Run paper-conventional and improved modes across primary seeds.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--final",
        action="store_true",
        help="Run primary validation search, then retrain and test final models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budget = budget_values(args.budget)

    if args.primary or args.final:
        from experiment_runner import run_primary_experiments

        if args.final and args.synthetic:
            raise ValueError("--final requires the real CNN/MNIST evaluator; omit --synthetic")
        if args.budget != "main" or tuple(args.seeds) != (42, 43, 44):
            raise ValueError("--primary/--final requires main budget and seeds 42 43 44")
        if any(value is not None for value in (args.ants, args.iterations, args.max_epochs)):
            raise ValueError("--primary uses the fixed main budget without overrides")

        if args.synthetic:
            evaluator_factory = lambda _config: synthetic_evaluator
        else:
            from dataset import load_mnist
            from evaluator import evaluate_candidate

            data = load_mnist(2024)
            evaluator_factory = lambda config: (
                lambda candidate, _experiment: evaluate_candidate(candidate, config, data)
            )

        report = run_primary_experiments(
            seeds=tuple(args.seeds),
            budget=budget,
            budget_name=args.budget,
            evaluator_factory=evaluator_factory,
            output_dir=args.output_dir,
            cache_enabled=args.cache,
            evaluator_type="synthetic" if args.synthetic else "cnn_mnist",
        )
        if args.final:
            from final_evaluation import run_final_evaluations

            final_result = run_final_evaluations(
                report=report,
                data=data,
                output_dir=args.output_dir,
            )
            print(f"final_status={final_result.status}")
            print(f"final_test_evaluations={final_result.test_evaluations}")
            print(f"final_report={final_result.output_path}")
            return
        print(f"primary_modes={len(report.mode_results)} seeds={tuple(args.seeds)}")
        print(f"test_evaluations={report.test_evaluations}")
        print(f"report={report.output_path}")
        return

    config = ExperimentConfig(
        mode=args.mode,
        ants=args.ants if args.ants is not None else budget["ants"],
        iterations=args.iterations if args.iterations is not None else budget["iterations"],
        max_epochs=args.max_epochs if args.max_epochs is not None else budget["max_epochs"],
        run_seed=args.seed,
        cache_enabled=args.cache,
        budget_name=args.budget,
        search_space=get_search_space(args.mode),
        output_dir=args.output_dir,
    )

    if args.synthetic:
        evaluator = lambda candidate, experiment: synthetic_evaluator(candidate, experiment)
    else:
        from dataset import load_mnist
        from evaluator import evaluate_candidate

        data = load_mnist(config.dataset_seed)
        evaluator = lambda candidate, experiment: evaluate_candidate(candidate, experiment, data)

    started = time.perf_counter()
    optimizer = ACOOptimizer(config, evaluator)
    try:
        best = optimizer.run()
    except Exception:
        optimizer.run_status = "failed"
        raise
    finally:
        optimizer.write_logs(config.output_dir)
    elapsed = time.perf_counter() - started
    print(f"mode={config.mode} seed={config.run_seed} runtime_seconds={elapsed:.3f}")
    print(f"best_fitness={best.result.fitness:.6f}")
    print(f"best_configuration={best.configuration}")


if __name__ == "__main__":
    main()
