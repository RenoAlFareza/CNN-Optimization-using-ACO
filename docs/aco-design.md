# ACO-CNN experiment design

## Provenance and scope

The design is inspired by Purnomo et al., “Metaheuristics Approach for Hyperparameter Tuning of Convolutional Neural Network,” Jurnal RESTI 8(3), 2024, DOI 10.29207/resti.v8i3.5730.

The article reports MNIST, 20 ants, pheromone-only selection, evaporation rate `rho = 0.25`, constant reinforcement `+0.5`, and eight hyperparameters with eight options each. It does not fully specify initial pheromone, data split, epoch policy, stopping criterion, update timing, or which ants reinforce. Those omissions are not silently presented as facts.

## Fixed CNN implemented here

```text
Input 28x28x1
Conv2D(32, 3x3, relu, padding=same)
MaxPooling2D(2x2)
Dropout(dropout_1)
Conv2D(64, 3x3, relu, padding=same)
MaxPooling2D(2x2)
Flatten
BatchNormalization
Dense(dense_units_1, activation)
Dropout(dropout_2)
Dense(dense_units_2, activation)
Dense(10, softmax)
```

This is an explicit project design, not a claim that the article specifies these filter counts or layer details.

## Dataset protocol

MNIST is loaded from the local IDX files in `data/` (or the CLI `--data-dir`). The official 60,000 training images are split once, stratified and deterministically using `dataset_seed = 2024`, into 50,000 training and 10,000 validation images. The official 10,000-image test set remains untouched until final evaluation.

## Search spaces

### Paper-faithful modes

The eight dimensions from the article are retained:

```python
{
    "dense_units_1": [16, 32, 64, 128, 256, 512, 1024, 2048],
    "dense_units_2": [16, 32, 64, 128, 256, 512, 1024, 2048],
    "dropout_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "dropout_2": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "batch_size": [16, 32, 64, 128, 256, 512, 1024, 2048],
    "activation": ["relu", "sigmoid", "softplus", "softsign", "tanh", "selu", "gelu", "linear"],
    "optimizer": ["adam", "rmsprop", "sgd", "adadelta", "adagrad", "adamax", "ftrl", "nadam"],
    "loss": ["sparse_categorical_crossentropy", "categorical_crossentropy", "binary_crossentropy", "mean_absolute_error", "mean_squared_error", "squared_hinge", "categorical_hinge", "cosine_similarity"],
}
```

The paper spells the last activation `linier`; the implementation identifier is `linear`, while the paper label is retained in metadata.

### Improved mode

```python
{
    "dense_units_1": [32, 64, 128, 256],
    "dense_units_2": [32, 64, 128, 256],
    "dropout_1": [0.1, 0.3, 0.5],
    "dropout_2": [0.1, 0.3, 0.5],
    "batch_size": [32, 64, 128],
    "activation": ["relu", "tanh"],
    "optimizer": ["adam", "rmsprop"],
    "learning_rate": [1e-4, 1e-3, 1e-2],
}
```

This has `5,184` configurations. The loss is fixed to sparse categorical crossentropy.

## Pheromone and update rules

All modes initialize every option to `tau_0 = 1.0`. This is an implementation assumption because the article does not state its initial value. Probabilities are always normalized pheromone values and use `tau_min = 1e-12` by default.

### `paper_literal`

For each ant: sample from the current pheromone, evaluate, evaporate all pheromone once, then reinforce every option selected by that ant by `+0.5`. The next ant sees the updated pheromone. This is a literal interpretation of the paper pseudocode, not evidence about the authors' original code.

### `paper_conventional`

Sample all ants using the pheromone at the start of an iteration, evaluate all, evaporate once, then reinforce every successful/cached ant's selected options by `+0.5`.

### `improved`

Sample and evaluate all ants, select the iteration-best valid candidate, evaporate once, and reinforce only its selected options with:

```text
delta_tau = Q * validation_accuracy
Q = 1.0
```

## Training and selection

Candidate fitness is validation accuracy. Early stopping monitors Keras `val_accuracy` with `mode = max`, `patience = 2`, `min_delta = 0.0`, and `restore_best_weights = True`, and records `best_epoch` and actual epochs completed; validation-loss and training-time tie-breaks make selection deterministic. Failed trials have no fitness and do not reinforce pheromone. Cached successful results are valid trials and can reinforce according to the active mode.

Trial and pheromone logs include the requested budget name, effective ant/iteration/epoch counts, and run-level status (`success` or `failed`). This keeps CLI budget overrides auditable.

Paper modes use each optimizer's framework default learning rate. Improved mode searches learning rate explicitly. Candidate seeds use a stable SHA-256 hash of run seed, mode, and canonical configuration; iteration and ant ID are not included.

## Experiment hierarchy

```text
Smoke:    2 ants x 2 iterations x 2 max epochs
Pilot:   20 ants x 1 iteration x 5 max epochs, seed 42 per primary mode for runtime calibration
Main:    20 ants x 5 iterations x 5 max epochs, seeds 42/43/44
Diagnostic: 20 ants x 50 iterations x 10 max epochs, optional for paper-literal
```

Twenty ants follows the explicit ant count in the article. The five-iteration and five-epoch limits are project computational-budget decisions: the article does not specify either value. The reduced main budget targets a feasible, reproducible multi-seed experiment on available hardware; it is not a claim to reproduce the paper's runtime literally. Run `--calibrate-runtime` on the target GPU before the full primary/final workflow. The calibration report estimates candidate, seed, mode, confirmation, final-evaluation, and total workflow runtime. If its projected full-workflow runtime exceeds two hours, lower the epoch limit to four first, then the iteration limit to four; keep 20 ants and three seeds. If it is substantially below two hours, do not increase iterations until the estimate is reviewed.

The primary comparison is `paper_conventional` versus `improved`. `paper_literal` is diagnostic unless resources permit a supplementary three-seed run.

The repeated-run runner executes both primary modes for seeds 42, 43, and 44 using the main budget, then confirms every per-seed global-best candidate on every seed using validation results only. Its report records per-candidate, per-seed, per-mode, and primary-total runtime; failed trials; cache hits; effective budget; configuration metadata; and the explicit pipeline-comparison caveat. It does not accept a test evaluator; final test evaluation belongs to the final-retraining stage. Candidate logs include actual epochs completed when training history provides them.

The `--calibrate-runtime` command runs 20 candidates for one iteration on each primary mode with seed 42 and cache enabled. Its JSON report records candidate mean/median runtime, actual epochs, cache-hit and failure rates, detected GPU devices (utilization is marked not sampled), and estimates for tuning, aggregate confirmation, six final models, and total workflow. `--primary` and `--final` require this report in the output directory and refuse to start if projected total runtime exceeds two hours. This gate prevents an uncalibrated long run; it does not guarantee runtime under workload variation.

The analysis tool reads trial/pheromone CSVs and the primary/final JSON reports, then writes `analysis_report.md`, `analysis_summary.json`, and four plots: convergence, pheromone evolution, probability evolution, and runtime/trial outcomes. It preserves the evaluator type (`synthetic` versus `cnn_mnist`) and explicitly states that smoke/pilot outputs are implementation validation rather than primary scientific results.

After tuning, a new model is trained on all 60,000 training-development images for the selected `best_epoch` and evaluated once on the untouched test set. Test accuracy never selects a configuration.

## Interpretation caveat

The primary modes do not share the same search space: paper-conventional has `8^8 = 16,777,216` possible configurations, while improved has `5,184` and adds learning rate. Their comparison is therefore a pipeline comparison, not a controlled ablation isolating only the pheromone update rule.
