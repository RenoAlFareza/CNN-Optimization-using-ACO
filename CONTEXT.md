# ACO-CNN domain context

## Purpose

This project studies Ant Colony Optimization (ACO) for selecting CNN hyperparameters for multiclass image classification. The initial benchmark is MNIST.

## Domain glossary

- **Ant**: one ACO search agent that samples one complete hyperparameter configuration.
- **Hyperparameter**: a CNN training or dense-layer setting selected from a finite candidate set.
- **Candidate / solution**: one complete assignment of values to every hyperparameter in a search space.
- **Search space**: the ordered candidate values available for each hyperparameter.
- **Pheromone**: a non-negative numerical preference attached to one option of one hyperparameter. Each hyperparameter has its own pheromone vector.
- **Selection probability**: the normalized pheromone value used to sample an option. For option `i` of hyperparameter `k`, `P(i,k) = tau(i,k) / sum_j tau(j,k)`.
- **Fitness**: validation accuracy of a successfully evaluated candidate. Higher fitness is better.
- **Iteration**: one ACO cycle. Depending on the mode, ants are updated one at a time or all ants are updated together.
- **Paper-faithful**: a mode intended to preserve the search space and constant pheromone reinforcement described by Purnomo et al. (2024), while explicitly recording details that the paper does not specify.
- **Improved**: the development mode with an explicit validation protocol, a smaller search space, iteration-best reinforcement, and fitness-weighted pheromone reinforcement.
- **Experiment family**: the report grouping that places `paper_literal` and `paper_conventional` under `paper-faithful`, and `improved` under `improved`.
- **Trial**: one candidate evaluation, including a successful training, a failed training, or a cache hit.
- **Final model**: a new CNN trained after ACO finishes, using the selected configuration and the training-development set, then evaluated once on the untouched test set.

## Boundaries

The convolutional architecture is fixed by this project. ACO changes dense-layer widths, dropout values, batch size, activation, optimizer, loss where applicable, and learning rate in improved mode. It is not a general neural architecture search system.

The test set is never used to calculate ACO fitness or select a configuration.

## Primary computational budget

The primary comparison uses `paper_conventional` and `improved`, with seeds
42, 43, and 44, and a budget of 20 ants × 5 iterations × at most 5 epochs.
The 20-ant count is explicit in the source paper; the iteration and epoch
limits are implementation decisions because the paper does not specify them.
The theoretical maximum is 600 tuning trial records across both modes and all
seeds. Runtime calibration must be performed on the target hardware before
primary/final execution; if the projected full workflow exceeds two hours,
reduce epochs to 4 first and iterations to 4 second, while retaining 20 ants
and all three seeds.
