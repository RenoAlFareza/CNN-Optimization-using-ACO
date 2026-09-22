# 04: Paper-conventional ACO mode

**What to build:** Provide the primary paper-based ACO mode in which all ants are sampled and evaluated before one evaporation and constant reinforcement update per iteration.

**Blocked by:** 02: Generic ACO engine with logging and cache; 03: CNN-MNIST candidate evaluator

**Status:** completed

- [x] The mode exposes the journal's eight-dimensional, eight-option search space and preserves the paper label `linier` as metadata while using `linear` for execution.
- [x] Learning rate is not searched and the effective framework-default learning rate is recorded.
- [x] All ants use the pheromone state at iteration start for sampling.
- [x] Pheromone evaporates exactly once per iteration with rho 0.25.
- [x] Each valid ant reinforces every selected option by the constant amount 0.5; failed trials do not reinforce.
- [x] The mode is selectable from the command line and supports smoke and pilot budgets.
- [x] Logs make the iteration-level sampling and update lifecycle auditable.
