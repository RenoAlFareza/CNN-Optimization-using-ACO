# ADR-001: Provide literal, conventional, and improved ACO modes

## Status

Accepted

## Context

The source article places pheromone operations inside an ant loop but does not fully specify update timing or which ants reinforce. A single implementation would hide an important interpretation choice.

## Decision

Expose `paper_literal`, `paper_conventional`, and `improved` as separate modes. The first preserves the literal ant-by-ant interpretation, the second batches updates per iteration, and the third uses iteration-best fitness-weighted reinforcement.

## Consequences

The ambiguity is testable and visible in logs. The primary comparison is between paper-conventional and improved; paper-literal is diagnostic. The modes must not be described as identical implementations of the paper.
