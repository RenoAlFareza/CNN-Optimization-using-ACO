# 06: Improved fitness-weighted ACO mode

**What to build:** Provide the development ACO mode with the smaller explicit search space, fixed sparse categorical crossentropy, explicit learning-rate search, and iteration-best fitness-weighted reinforcement.

**Blocked by:** 02: Generic ACO engine with logging and cache; 03: CNN-MNIST candidate evaluator

**Status:** completed

- [x] The improved search space contains exactly 5,184 configurations across dense units, dropout, batch size, activation, optimizer, and learning rate.
- [x] Loss is fixed to sparse categorical crossentropy and is not sampled by ACO.
- [x] All ants are sampled and evaluated before pheromone update.
- [x] Pheromone evaporates once per iteration with rho 0.25.
- [x] Only the valid iteration-best candidate is reinforced.
- [x] Reinforcement equals Q multiplied by validation accuracy, with Q equal to 1.0.
- [x] The mode is selectable from the command line and produces the same trial and pheromone log schema as the paper-conventional mode.
- [x] Smoke and pilot runs verify that probability and pheromone behavior changes according to the improved update rule.
