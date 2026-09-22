"""Command-line entry point for the ACO-CNN experiments."""

from __future__ import annotations

import argparse
import math
import time
from typing import Any

from aco_optimizer import ACOOptimizer
from config import ExperimentConfig, get_search_space
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
    parser.add_argument("--ants", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Test ACO without TensorFlow/MNIST.")
    parser.add_argument("--output-dir", default="experiments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        mode=args.mode,
        ants=args.ants,
        iterations=args.iterations,
        max_epochs=args.max_epochs,
        run_seed=args.seed,
        cache_enabled=args.cache,
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
    best = optimizer.run()
    optimizer.write_logs(config.output_dir)
    elapsed = time.perf_counter() - started
    print(f"mode={config.mode} seed={config.run_seed} runtime_seconds={elapsed:.3f}")
    print(f"best_fitness={best.result.fitness:.6f}")
    print(f"best_configuration={best.configuration}")


if __name__ == "__main__":
    main()
