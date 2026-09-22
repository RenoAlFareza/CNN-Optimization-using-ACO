# 03: CNN-MNIST candidate evaluator

**What to build:** Connect a candidate configuration to the fixed CNN architecture and MNIST data protocol so a candidate can be trained, scored on validation data, and returned through the common evaluation contract.

**Blocked by:** 01: Candidate evaluation contract and smoke harness

**Status:** ready-for-human

- [x] MNIST is normalized as grayscale 28×28×1 input.
- [x] The official training data is split once, stratified and deterministically, into 50,000 training and 10,000 validation images using dataset seed 2024; the official 10,000-image test set remains untouched.
- [x] The fixed CNN architecture has the agreed convolutional, pooling, dropout, flatten, batch-normalization, dense, and 10-class softmax layers.
- [x] Candidate fitness is validation accuracy, with validation loss, training accuracy, effective learning rate, timing, and best epoch recorded.
- [x] Early stopping monitors validation accuracy with patience 2, zero minimum delta, and restored best weights.
- [x] Best epoch uses validation loss as a deterministic tie-break when validation accuracy ties.
- [x] Compile or fit errors become failed trials with a failure reason rather than accuracy zero.
- [x] Paper loss candidates that run technically but are nonstandard for sparse multiclass labels receive a semantic warning.
- [x] Environment and determinism metadata are available for an actual CNN evaluation.

## Comments

Implemented CNN/MNIST evaluation through the shared `EvaluationResult` contract. Added seam tests using a TensorFlow test double because TensorFlow is not installed in this environment; the real adapter remains dependency-gated with a clear runtime error. Full test discovery passes: 15 tests.
