# 05: Paper-literal diagnostic ACO mode

**What to build:** Provide the diagnostic mode that follows the journal pseudocode literally as an ant-by-ant lifecycle, making the consequences of per-ant pheromone updates visible without presenting it as the authors' confirmed implementation.

**Blocked by:** 02: Generic ACO engine with logging and cache; 03: CNN-MNIST candidate evaluator

**Status:** ready-for-agent

- [ ] Each ant samples from the current pheromone, is evaluated, triggers evaporation, and then reinforces its selected valid options by 0.5.
- [ ] The next ant samples from pheromone after the previous ant's update.
- [ ] Failed ants trigger the mode's evaporation lifecycle but do not receive reinforcement.
- [ ] Pheromone history records before-sampling and after-update snapshots for each ant and update step.
- [ ] The mode is selectable from the command line and supports smoke, pilot, and diagnostic budgets.
- [ ] Documentation and metadata identify this as a literal pseudocode interpretation, not a verified reproduction of the authors' code.
