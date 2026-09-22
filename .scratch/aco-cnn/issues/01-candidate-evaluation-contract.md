# 01: Candidate evaluation contract and smoke harness

**What to build:** Define the boundary between the ACO optimizer and a candidate evaluator so synthetic and CNN/MNIST evaluators return a consistent trial result. Provide a fast smoke harness that verifies the boundary without TensorFlow.

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [x] A trial result supports `success`, `failed`, and `cached` semantics, including fitness, metrics, epoch, timing, and failure metadata.
- [x] Canonical candidate IDs and trial seeds are stable for the same mode, run seed, and configuration, and do not depend on iteration or ant ID.
- [x] A synthetic evaluator can be injected into the optimizer boundary without importing TensorFlow or loading MNIST.
- [x] Smoke checks verify that every generated candidate contains one value for each search-space hyperparameter.
- [x] Smoke checks verify that each selection-probability row sums to one and pheromone never falls below the configured minimum.

## Comments

Implemented the dependency-free `evaluation_contract.py` seam, moved optimizer imports to that contract, and added fast unittest coverage for trial statuses, canonical identities, candidate completeness, probability normalization, and pheromone floors. The synthetic CLI smoke run also passes for the improved mode. The workspace does not contain a `.git` directory, so no commit or Git-based diff review could be performed.
