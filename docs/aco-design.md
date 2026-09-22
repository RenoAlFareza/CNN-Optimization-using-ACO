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

MNIST is loaded from the TensorFlow/Keras dataset API. The official 60,000 training images are split once, stratified and deterministically using `dataset_seed = 2024`, into 50,000 training and 10,000 validation images. The official 10,000-image test set remains untouched until final evaluation.

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

Candidate fitness is validation accuracy. Early stopping monitors validation accuracy with `patience = 2`, restores best weights, and records `best_epoch`; validation-loss and training-time tie-breaks make selection deterministic. Failed trials have no fitness and do not reinforce pheromone. Cached successful results are valid trials and can reinforce according to the active mode.

Trial and pheromone logs include the requested budget name, effective ant/iteration/epoch counts, and run-level status (`success` or `failed`). This keeps CLI budget overrides auditable.

Paper modes use each optimizer's framework default learning rate. Improved mode searches learning rate explicitly. Candidate seeds use a stable SHA-256 hash of run seed, mode, and canonical configuration; iteration and ant ID are not included.

## Experiment hierarchy

```text
Smoke:    2 ants x 2 iterations x 2 max epochs
Pilot:   10 ants x 10 iterations x 10 max epochs
Main:    20 ants x 20 iterations x 10 max epochs, seeds 42/43/44
Diagnostic: 20 ants x 50 iterations x 10 max epochs, optional for paper-literal
```

The primary comparison is `paper_conventional` versus `improved`. `paper_literal` is diagnostic unless resources permit a supplementary three-seed run.

The repeated-run runner executes both primary modes for seeds 42, 43, and 44 using the main budget, then confirms every per-seed global-best candidate on every seed using validation results only. Its report records per-seed and aggregate runtime, failed trials, cache hits, effective budget, configuration metadata, and the explicit pipeline-comparison caveat. It does not accept a test evaluator; final test evaluation belongs to the final-retraining stage.

The analysis tool reads trial/pheromone CSVs and the primary/final JSON reports, then writes `analysis_report.md`, `analysis_summary.json`, and four plots: convergence, pheromone evolution, probability evolution, and runtime/trial outcomes. It preserves the evaluator type (`synthetic` versus `cnn_mnist`) and explicitly states that smoke/pilot outputs are implementation validation rather than primary scientific results.

After tuning, a new model is trained on all 60,000 training-development images for the selected `best_epoch` and evaluated once on the untouched test set. Test accuracy never selects a configuration.

## Interpretation caveat

The primary modes do not share the same search space: paper-conventional has `8^8 = 16,777,216` possible configurations, while improved has `5,184` and adds learning rate. Their comparison is therefore a pipeline comparison, not a controlled ablation isolating only the pheromone update rule.
