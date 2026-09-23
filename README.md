# CNN Optimization using ACO

This project uses discrete Ant Colony Optimization (ACO) to search CNN
hyperparameters for MNIST classification. The convolutional architecture is
fixed; ACO searches dense-layer sizes, dropout, batch size, activation,
optimizer, and (in `improved` mode) learning rate.

## Requirements

- Python 3.10 or newer
- TensorFlow 2.13 or newer
- A CPU works for smoke tests and pilots. A CUDA-capable GPU is recommended
  for the full primary experiment.

Place the four standard MNIST IDX files in `data/`. The loader accepts either
flat filenames (`train-images.idx3-ubyte`, etc.) or the nested layout from the
provided notebook (`train-images-idx3-ubyte/train-images-idx3-ubyte`, etc.).
The dataset is intentionally ignored by Git because it is local input data.
If the files are stored elsewhere, pass the location with
`--data-dir PATH_TO_DATA`.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Verify the code

Run the tests without training a real CNN:

```bash
python -m unittest discover -q
```

Run a synthetic ACO smoke test:

```bash
python ACO.py --synthetic --mode improved --budget smoke --seed 42 \
  --cache --output-dir experiments/synthetic_smoke
```

## Run a real CNN smoke test

Omit `--synthetic` to load and train on MNIST:

```bash
python ACO.py --mode improved --budget smoke --seed 42 \
  --data-dir data --cache --output-dir experiments/real_smoke
```

The smoke run is only an implementation check, not a scientific result.

## Calibrate runtime on the target GPU

Before starting the primary experiment, run a small real-CNN calibration. It
trains 20 candidates for one iteration on each primary mode, using seed 42,
cache, and up to five epochs:

```bash
python ACO.py --calibrate-runtime --data-dir data \
  --output-dir experiments/main
```

This calibration is not part of the scientific primary runtime. It reports
observed candidate runtimes, actual epochs, failures, cache behavior, GPU
availability, and estimates for tuning, confirmation, and final evaluation.
Use the same `--output-dir` for calibration and the subsequent primary/final
command so the planning gate can find the report. The primary/full-workflow
commands refuse to start if the projected total is over two hours. This is a
planning gate, not a runtime guarantee; actual workload can vary.

## Run the primary experiment

The primary protocol compares `paper_conventional` and `improved` using seeds
42, 43, and 44 with a budget of 20 ants x 5 iterations x 5 maximum epochs:

```bash
python ACO.py --primary --budget main --seeds 42 43 44 \
  --cache --output-dir experiments/main
```

`--primary` requires a current runtime calibration report under the same output
directory. It refuses to start when the projected full workflow exceeds two
hours. Recalibrate after changing hardware, TensorFlow/CUDA, or budget.

Each ant evaluates one hyperparameter configuration by training a CNN and
measuring validation accuracy. The test set is not used during selection.
Twenty ants follows the explicit paper setting. Five iterations and five
maximum epochs are project budget decisions because the paper does not specify
them. This feasible multi-seed budget is not a claim to reproduce the paper's
runtime literally.

## Run primary search and final evaluation

To run validation search followed by final retraining and one untouched-test
evaluation per mode and seed:

```bash
python ACO.py --final --budget main --seeds 42 43 44 \
  --cache --output-dir experiments/main
```

The final stage retrains the selected configuration on the 60,000 development
images and evaluates once on the 10,000-image MNIST test set.

## Analyze results

After an experiment has produced CSV/JSON artifacts:

```bash
python ACO.py --analyze --output-dir experiments/main
```

Analysis outputs include convergence, pheromone and probability plots, runtime
summaries, `analysis_summary.json`, and `analysis_report.md`.

## Search-space note

The paper-conventional search space contains `8^8 = 16,777,216` possible
configurations. ACO does not exhaustively train all of them. With the main
budget, it evaluates at most 100 candidates per seed and uses their validation
results to guide subsequent pheromone sampling. Therefore the reported result
is the best configuration observed within the configured budget, not a proof of
the global optimum.
