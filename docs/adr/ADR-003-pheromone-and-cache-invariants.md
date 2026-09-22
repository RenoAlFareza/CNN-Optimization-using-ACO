# ADR-003: Pheromone, seed, and cache invariants

## Status

Accepted

## Context

The experiment compares stochastic candidate evaluations and must remain auditable while avoiding unnecessary repeated training.

## Decision

Initialize pheromone uniformly at 1.0, clamp it to configurable `tau_min` after updates, and derive probabilities by row normalization. Derive trial seeds and cache keys from mode, run seed, canonical configuration, split identifier, epoch policy, and training policy; never use iteration or ant ID. A cache hit is a valid evaluation for ranking and pheromone updates.

## Consequences

Repeated configurations can be avoided without sharing stochastic results across runs. Pheromone probability rows remain valid and nonzero within floating-point limits.
