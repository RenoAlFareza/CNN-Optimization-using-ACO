# 08: Repeated runs and aggregate confirmation

**What to build:** Run the primary paper-conventional and improved experiments across seeds 42, 43, and 44, report each seed independently, and select an aggregate configuration using validation evidence only before any final test evaluation.

**Blocked by:** 04: Paper-conventional ACO mode; 06: Improved fitness-weighted ACO mode

**Status:** completed

- [x] Each primary mode runs with 20 ants, 20 iterations, and the agreed maximum-epoch policy for seeds 42, 43, and 44.
- [x] Each seed produces an individual global-best configuration and validation result.
- [x] Candidate configurations from the per-seed global-best results are evaluated across the same seed set during aggregate confirmation.
- [x] Aggregate selection uses mean validation accuracy, then mean validation loss, mean training time, and canonical candidate ID as tie-breaks.
- [x] Test accuracy is not used for aggregate selection and no final test evaluation occurs before aggregate selection.
- [x] Per-seed and aggregate results include runtime, failed trials, cache hits, validation metrics, and configuration metadata.
- [x] The output explicitly labels paper-conventional versus improved as a pipeline comparison because their search spaces differ.
