# 07: Final retraining and test evaluation

**What to build:** After aggregate configuration selection, train a new final CNN on all 60,000 training-development images for the selected best epoch and evaluate it once on the untouched 10,000-image test set.

**Blocked by:** 08: Repeated runs and aggregate confirmation

**Status:** ready-for-agent

- [ ] Final retraining creates a new model instead of reusing a candidate checkpoint.
- [ ] The final model uses the selected configuration and all 60,000 training-development images.
- [ ] Final training uses the selected best epoch, shuffling, and no early stopping because no validation set remains.
- [ ] Test evaluation is performed once per selected final model after tuning and never feeds back into selection.
- [ ] Per-seed final model artifacts and selected configuration metadata are saved.
- [ ] A run with no valid selected configuration is marked failed and does not perform final retraining.
