# Review and verification

Reviewed on 2026-11-11 as part of the documentation and diagnostic-report update.

## What was checked

- Dataset checksums, IDX dimensions and label ranges, and disjoint reproducible stratified partitions.
- Hidden-layer He initialization, dense-layer dimensions, and stable Softmax/cross-entropy.
- The combined mean-loss derivative and full-network finite-difference gradient checks.
- Sample-weighted metrics, partial mini-batches, reproducible shuffling, and minimum-validation-loss checkpoint selection.
- Checkpoint serialization and identical predictions after loading.
- Known ROC AUC values, tied scores, perfect/reversed rankings, absent classes, and invalid scores.
- Report generation from the existing MNIST checkpoint, including visual inspection of the exported figures.

The local suite passed 24 tests. The original model still scores 97.90% accuracy and 0.0727768813 cross-entropy on the official test data. No model-training mathematics or stored baseline parameters were changed during this update. This is a scoped engineering review supported by tests, not a proof that no undiscovered defects exist.

## Corrections made

Previously, training validated labels one mini-batch at a time. If an invalid label appeared in a later batch, earlier batches could already have changed the weights. Training now validates the complete label array and input matrix before any updates. A regression test verifies that invalid training labels leave all parameters unchanged.

Batch sizes and epoch counts now reject noninteger values explicitly. The training CLI checks model construction and the learning rate before downloading files or creating run artifacts. The evaluation CLI rejects nonpositive batch sizes and incompatible report options early.

## Limitations and next experiments

The baseline has one seed and no controlled local comparison with another architecture. Published MNIST comparisons in the README are historical context, not equivalent training protocols. Potential future work includes validation-only comparisons of learning-rate schedules, regularization, augmentation, and several seeds. The existing test diagnostics must not be treated as fresh held-out evidence when guiding future changes.

Softmax output has not been calibrated. The MLP does not explicitly encode spatial locality, and performance on external photographs or differently centered/inverted digits is unmeasured. The current checkpoint format supports inference and fresh training, not exact interrupted-run resumption.
