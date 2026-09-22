"""Discrete pheromone ACO engine used by all three experiment modes."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from config import ExperimentConfig, family_for_mode, mode_interpretation, paper_label
from evaluation_contract import EvaluationResult, candidate_id, canonical_json


Evaluator = Callable[[dict[str, Any], ExperimentConfig], EvaluationResult]


@dataclass
class CandidateResult:
    configuration: dict[str, Any]
    result: EvaluationResult
    candidate_id: str
    ant_id: int


class ACOOptimizer:
    def __init__(self, config: ExperimentConfig, evaluator: Evaluator):
        self.config = config
        self.evaluator = evaluator
        self.parameter_names = list(config.search_space)
        self.option_counts = [len(config.search_space[name]) for name in self.parameter_names]
        if not self.parameter_names or any(count == 0 for count in self.option_counts):
            raise ValueError("search_space must contain at least one option per hyperparameter")
        self.pheromone = np.full(
            (len(self.parameter_names), max(self.option_counts)),
            config.tau_0,
            dtype=float,
        )
        self.pheromone_history: list[dict[str, Any]] = []
        self.trial_rows: list[dict[str, Any]] = []
        self.cache: dict[str, EvaluationResult] = {}
        self.rng = np.random.default_rng(config.run_seed)
        self.global_best: CandidateResult | None = None
        self.run_status = "pending"

    def _probabilities(self) -> np.ndarray:
        probabilities = np.zeros_like(self.pheromone)
        for index, count in enumerate(self.option_counts):
            row = np.maximum(self.pheromone[index, :count], self.config.tau_min)
            probabilities[index, :count] = row / row.sum()
        return probabilities

    def _sample(self) -> tuple[dict[str, Any], list[int], np.ndarray]:
        probabilities = self._probabilities()
        indices: list[int] = []
        configuration: dict[str, Any] = {}
        for parameter_index, name in enumerate(self.parameter_names):
            count = self.option_counts[parameter_index]
            option_index = int(self.rng.choice(count, p=probabilities[parameter_index, :count]))
            indices.append(option_index)
            configuration[name] = self.config.search_space[name][option_index]
        return configuration, indices, probabilities

    def _record_pheromone(self, iteration: int, ant_id: int | None, phase: str) -> None:
        probabilities = self._probabilities()
        update_step = len({row["update_step"] for row in self.pheromone_history})
        for parameter_index, name in enumerate(self.parameter_names):
            for option_index, option in enumerate(self.config.search_space[name]):
                self.pheromone_history.append(
                    {
                        "run_id": f"{self.config.mode}-seed-{self.config.run_seed}",
                        "mode": self.config.mode,
                        "family": family_for_mode(self.config.mode),
                        "mode_interpretation": mode_interpretation(self.config.mode),
                        "budget": self.config.budget_name,
                        "effective_ants": self.config.ants,
                        "effective_iterations": self.config.iterations,
                        "effective_max_epochs": self.config.max_epochs,
                        "seed": self.config.run_seed,
                        "iteration": iteration,
                        "ant_id": "" if ant_id is None else ant_id,
                        "update_step": update_step,
                        "phase": phase,
                        "hyperparameter": name,
                        "option_index": option_index,
                        "option_value": json.dumps(option),
                        "paper_label": json.dumps(paper_label(name, option)),
                        "pheromone": float(self.pheromone[parameter_index, option_index]),
                        "probability": float(probabilities[parameter_index, option_index]),
                    }
                )

    def _evaporate(self) -> None:
        self.pheromone *= 1.0 - self.config.rho
        self._clamp()

    def _clamp(self) -> None:
        for index, count in enumerate(self.option_counts):
            self.pheromone[index, :count] = np.maximum(
                self.pheromone[index, :count], self.config.tau_min
            )
            if count < self.pheromone.shape[1]:
                self.pheromone[index, count:] = self.config.tau_min

    def _reinforce(self, indices: list[int], amount: float) -> None:
        for parameter_index, option_index in enumerate(indices):
            self.pheromone[parameter_index, option_index] += amount
        self._clamp()

    @staticmethod
    def _is_better(candidate: CandidateResult, incumbent: CandidateResult | None) -> bool:
        if candidate.result.fitness is None:
            return False
        if incumbent is None or incumbent.result.fitness is None:
            return True
        candidate_key = (
            -candidate.result.fitness,
            candidate.result.validation_loss if candidate.result.validation_loss is not None else float("inf"),
            candidate.result.training_time_seconds,
            candidate.ant_id,
        )
        incumbent_key = (
            -incumbent.result.fitness,
            incumbent.result.validation_loss if incumbent.result.validation_loss is not None else float("inf"),
            incumbent.result.training_time_seconds,
            incumbent.ant_id,
        )
        return candidate_key < incumbent_key

    def _evaluate(
        self,
        configuration: dict[str, Any],
        ant_id: int,
        iteration: int,
        probabilities: np.ndarray,
    ) -> CandidateResult:
        key = candidate_id(configuration)
        cache_key = self._cache_key(configuration)
        cache_hit = False
        if self.config.cache_enabled and cache_key in self.cache:
            result = self.cache[cache_key]
            cache_hit = True
            result = replace(result, status="cached")
        else:
            result = self.evaluator(configuration, self.config)
            if self.config.cache_enabled and result.fitness is not None:
                self.cache[cache_key] = replace(result)

        row = {
            "run_id": f"{self.config.mode}-seed-{self.config.run_seed}",
            "mode": self.config.mode,
            "family": family_for_mode(self.config.mode),
            "mode_interpretation": mode_interpretation(self.config.mode),
            "budget": self.config.budget_name,
            "effective_ants": self.config.ants,
            "effective_iterations": self.config.iterations,
            "effective_max_epochs": self.config.max_epochs,
            "seed": self.config.run_seed,
            "iteration": iteration,
            "ant_id": ant_id,
            "candidate_id": key,
            **configuration,
            "fitness": result.fitness,
            "train_accuracy": result.train_accuracy,
            "validation_accuracy": result.validation_accuracy,
            "validation_loss": result.validation_loss,
            "best_epoch": result.best_epoch,
            "training_time_seconds": result.training_time_seconds,
            "status": result.status,
            "failure_reason": result.failure_reason,
            "semantic_warning": result.semantic_warning,
            "effective_learning_rate": result.effective_learning_rate,
            "cache_hit": cache_hit,
            "selection_probabilities": json.dumps(
                {
                    name: probabilities[index, :count].tolist()
                    for index, (name, count) in enumerate(
                        zip(self.parameter_names, self.option_counts)
                    )
                },
                sort_keys=True,
            ),
            "is_iteration_best": False,
            "is_global_best": False,
        }
        if self.config.mode in {"paper_literal", "paper_conventional"}:
            row["paper_label_activation"] = paper_label(
                "activation", configuration.get("activation", "")
            )
        self.trial_rows.append(row)
        return CandidateResult(configuration, result, key, ant_id)

    def _cache_key(self, configuration: dict[str, Any]) -> str:
        """Build a run-policy-aware key without changing the public candidate ID."""
        return canonical_json(
            {
                "mode": self.config.mode,
                "run_seed": self.config.run_seed,
                "configuration": configuration,
                "dataset_seed": self.config.dataset_seed,
                "max_epochs": self.config.max_epochs,
                "early_stopping_patience": self.config.early_stopping_patience,
                "deterministic": self.config.deterministic,
            }
        )

    def run(self) -> CandidateResult:
        for iteration in range(1, self.config.iterations + 1):
            iteration_results: list[CandidateResult] = []
            if self.config.mode == "paper_literal":
                for ant_id in range(1, self.config.ants + 1):
                    self._record_pheromone(iteration, ant_id, "before_sampling")
                    configuration, indices, probabilities = self._sample()
                    candidate = self._evaluate(
                        configuration, ant_id, iteration, probabilities
                    )
                    iteration_results.append(candidate)
                    self._evaporate()
                    if candidate.result.fitness is not None:
                        self._reinforce(indices, self.config.reinforcement)
                    self._record_pheromone(iteration, ant_id, "after_update")
            else:
                sampled: list[tuple[dict[str, Any], list[int]]] = []
                self._record_pheromone(iteration, None, "before_sampling")
                for ant_id in range(1, self.config.ants + 1):
                    configuration, indices, probabilities = self._sample()
                    sampled.append((configuration, indices))
                    iteration_results.append(
                        self._evaluate(
                            configuration, ant_id, iteration, probabilities
                        )
                    )
                valid = [item for item in iteration_results if item.result.fitness is not None]
                iteration_best = min(valid, key=lambda item: (
                    -item.result.fitness,
                    item.result.validation_loss if item.result.validation_loss is not None else float("inf"),
                    item.result.training_time_seconds,
                    item.ant_id,
                )) if valid else None
                self._evaporate()
                if self.config.mode == "improved":
                    if iteration_best is not None:
                        best_indices = next(
                            indices for candidate, indices in sampled
                            if candidate_id(candidate) == iteration_best.candidate_id
                        )
                        self._reinforce(
                            best_indices,
                            self.config.q * float(iteration_best.result.fitness),
                        )
                else:
                    for candidate, indices in zip(iteration_results, (item[1] for item in sampled)):
                        if candidate.result.fitness is not None:
                            self._reinforce(indices, self.config.reinforcement)
                self._record_pheromone(iteration, None, "after_update")

            valid = [item for item in iteration_results if item.result.fitness is not None]
            iteration_best = min(valid, key=lambda item: (
                -item.result.fitness,
                item.result.validation_loss if item.result.validation_loss is not None else float("inf"),
                item.result.training_time_seconds,
                item.ant_id,
            )) if valid else None
            if iteration_best is not None:
                for row in self.trial_rows[-len(iteration_results):]:
                    row["is_iteration_best"] = row["candidate_id"] == iteration_best.candidate_id
                if self._is_better(iteration_best, self.global_best):
                    self.global_best = iteration_best
                    for row in self.trial_rows:
                        row["is_global_best"] = row["candidate_id"] == self.global_best.candidate_id

        if self.global_best is None:
            self.run_status = "failed"
            raise RuntimeError("All candidate evaluations failed; no global best exists")
        self.run_status = "success"
        return self.global_best

    def write_logs(self, output_dir: str) -> None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for row in self.trial_rows:
            row["run_status"] = self.run_status
        for row in self.pheromone_history:
            row["run_status"] = self.run_status
        self._write_csv(directory / "aco_trials.csv", self.trial_rows)
        self._write_csv(directory / "pheromone_history.csv", self.pheromone_history)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
