# ADR-002: Fixed validation split and untouched final test

## Status

Accepted

## Context

ACO repeatedly evaluates candidate CNNs. Using the test set during search would make it part of hyperparameter tuning.

## Decision

Use a deterministic stratified 50,000/10,000 split from the official MNIST training set with dataset seed 2024. Use validation accuracy for ACO fitness. After tuning, retrain a new model on all 60,000 development images for the selected best epoch and evaluate once on the official 10,000-image test set.

## Consequences

The final test metric remains independent of candidate selection. The final retraining does not use early stopping because no validation set remains.
