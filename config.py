"""Configuration and search spaces for the ACO-CNN experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PAPER_SEARCH_SPACE: dict[str, list[Any]] = {
    "dense_units_1": [16, 32, 64, 128, 256, 512, 1024, 2048],
    "dense_units_2": [16, 32, 64, 128, 256, 512, 1024, 2048],
    "dropout_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "dropout_2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "batch_size": [16, 32, 64, 128, 256, 512, 1024, 2048],
    "activation": [
        "relu",
        "sigmoid",
        "softplus",
        "softsign",
        "tanh",
        "selu",
        "gelu",
        "linear",
    ],
    "optimizer": [
        "adam",
        "rmsprop",
        "sgd",
        "adadelta",
        "adagrad",
        "adamax",
        "ftrl",
        "nadam",
    ],
    "loss": [
        "sparse_categorical_crossentropy",
        "categorical_crossentropy",
        "binary_crossentropy",
        "mean_absolute_error",
        "mean_squared_error",
        "squared_hinge",
        "categorical_hinge",
        "cosine_similarity",
    ],
}

IMPROVED_SEARCH_SPACE: dict[str, list[Any]] = {
    "dense_units_1": [32, 64, 128, 256],
    "dense_units_2": [32, 64, 128, 256],
    "dropout_1": [0.1, 0.3, 0.5],
    "dropout_2": [0.1, 0.3, 0.5],
    "batch_size": [32, 64, 128],
    "activation": ["relu", "tanh"],
    "optimizer": ["adam", "rmsprop"],
    "learning_rate": [1e-4, 1e-3, 1e-2],
}

EXPERIMENT_BUDGETS: dict[str, dict[str, int]] = {
    "smoke": {"ants": 2, "iterations": 2, "max_epochs": 2},
    "pilot": {"ants": 20, "iterations": 1, "max_epochs": 5},
    "main": {"ants": 20, "iterations": 5, "max_epochs": 5},
    "diagnostic": {"ants": 20, "iterations": 50, "max_epochs": 10},
}

MODE_INTERPRETATIONS = {
    "paper_literal": "literal_pseudocode_interpretation_not_verified_original_code",
    "paper_conventional": "iteration_based_paper_interpretation",
    "improved": "project_improvement",
}

LOG_HYPERPARAMETERS = (
    "dense_units_1",
    "dense_units_2",
    "dropout_1",
    "dropout_2",
    "batch_size",
    "activation",
    "optimizer",
    "loss",
    "learning_rate",
)


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str = "improved"
    ants: int = 2
    iterations: int = 2
    max_epochs: int = 2
    run_seed: int = 42
    dataset_seed: int = 2024
    rho: float = 0.25
    tau_0: float = 1.0
    tau_min: float = 1e-12
    reinforcement: float = 0.5
    q: float = 1.0
    early_stopping_patience: int = 2
    cache_enabled: bool = False
    deterministic: bool = True
    output_dir: str = "experiments"
    model_dir: str = "models"
    budget_name: str = "custom"
    search_space: dict[str, list[Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"paper_literal", "paper_conventional", "improved"}:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.ants < 1 or self.iterations < 1 or self.max_epochs < 1:
            raise ValueError("ants, iterations, and max_epochs must be positive")
        if not 0 <= self.rho < 1:
            raise ValueError("rho must be in [0, 1)")
        if self.mode in {"paper_literal", "paper_conventional"} and self.rho != 0.25:
            raise ValueError("paper-faithful modes require rho=0.25")
        if self.mode == "improved" and self.rho != 0.25:
            raise ValueError("improved mode requires rho=0.25")
        if self.mode == "improved" and self.q != 1.0:
            raise ValueError("improved mode requires q=1.0")
        if self.mode == "improved" and "loss" in self.search_space:
            raise ValueError("improved mode does not sample loss")
        if self.tau_0 <= 0 or self.tau_min <= 0:
            raise ValueError("tau_0 and tau_min must be positive")
        if not self.search_space:
            object.__setattr__(
                self,
                "search_space",
                get_search_space(self.mode),
            )


def get_search_space(mode: str) -> dict[str, list[Any]]:
    """Return a copy so a caller cannot mutate the module-level definitions."""
    if mode in {"paper_literal", "paper_conventional"}:
        source = PAPER_SEARCH_SPACE
    elif mode == "improved":
        source = IMPROVED_SEARCH_SPACE
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return {name: list(values) for name, values in source.items()}


def family_for_mode(mode: str) -> str:
    if mode in {"paper_literal", "paper_conventional"}:
        return "paper_faithful"
    if mode == "improved":
        return "improved"
    raise ValueError(f"Unsupported mode: {mode}")


def budget_values(name: str) -> dict[str, int]:
    try:
        return dict(EXPERIMENT_BUDGETS[name])
    except KeyError as error:
        raise ValueError(f"Unsupported experiment budget: {name}") from error


def mode_interpretation(mode: str) -> str:
    try:
        return MODE_INTERPRETATIONS[mode]
    except KeyError as error:
        raise ValueError(f"Unsupported mode: {mode}") from error


def paper_label(parameter_name: str, value: Any) -> Any:
    """Return the source-paper spelling where it differs from the identifier."""
    if parameter_name == "activation" and value == "linear":
        return "linier"
    return value
