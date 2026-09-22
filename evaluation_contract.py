"""Dependency-free contract shared by ACO and candidate evaluators."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal


TrialStatus = Literal["success", "failed", "cached"]


def canonical_json(configuration: dict[str, Any]) -> str:
    """Serialize a candidate independently of dictionary insertion order."""
    return json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)


def candidate_id(configuration: dict[str, Any]) -> str:
    """Return the stable identifier for a canonical candidate configuration."""
    return hashlib.sha256(canonical_json(configuration).encode("utf-8")).hexdigest()[:16]


def trial_seed(run_seed: int, mode: str, configuration: dict[str, Any]) -> int:
    """Return a stable seed that does not depend on ant or iteration placement."""
    payload = f"{run_seed}|{mode}|{canonical_json(configuration)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass
class EvaluationResult:
    """The result of one candidate trial at the optimizer/evaluator seam."""

    status: TrialStatus
    fitness: float | None
    train_accuracy: float | None
    validation_accuracy: float | None
    validation_loss: float | None
    best_epoch: int | None
    training_time_seconds: float
    failure_reason: str = ""
    semantic_warning: str = ""
    effective_learning_rate: float | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.status not in {"success", "failed", "cached"}:
            raise ValueError(f"Unsupported trial status: {self.status}")
        if self.status == "failed" and self.fitness is not None:
            raise ValueError("A failed trial cannot have fitness")
        if self.status in {"success", "cached"} and self.fitness is None:
            raise ValueError(f"A {self.status} trial must have fitness")
