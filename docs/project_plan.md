# Implementation plan

The implementation follows four phases. Read each plain-language goal first, then study its technical details.

## Phase 1: Prepare the images

Goal: turn handwritten-digit images into numbers the network can process.

- Download and validate the original MNIST gzip IDX files.
- Flatten each 28×28 image into 784 values, convert to float32, and divide by 255.
- Keep integer labels from zero to nine. Cross-entropy also accepts one-hot labels.
- Split only the official training data into training and validation using a reproducible stratified split; preserve the official test data.
- Verify shapes, label ranges, reproducibility, and malformed-file detection.

Implementation: `src/data/mnist.py`.

## Phase 2: Build the prediction path

Goal: combine pixels through successive layers to score all ten digits.

- Implement dense layers as `A @ W + b`, with each example stored in a row.
- Use ReLU between dense layers, He initialization for hidden weights, and zero biases.
- Start with architecture `784 → 128 → 64 → 10` and Xavier initialization for output weights.
- Return logits and expose stable Softmax separately for probabilities.
- Cache training inputs for backward; inference must preserve those caches.

Implementation: `layers.py`, `activations.py`, and `network.py`.

## Phase 3: Teach the network to correct mistakes

Goal: measure prediction error and determine how each parameter should change.

- Use stable cross-entropy from logits rather than taking logs of underflowed probabilities.
- Backpropagate `(P - Y) / batch_size`, applying the averaging factor once.
- Compute dense gradients `dW = A.T @ dZ`, `db = sum(dZ)`, and `dA = dZ @ W.T`.
- Multiply through ReLU by the positive-input mask.
- Apply SGD only after completing backward for all layers.
- Verify analytical gradients with centered finite differences on a tiny float64 model, avoiding ReLU kinks.

Implementation: `losses.py`, layer/network backward methods, and `optimizers.py`.

## Phase 4: Train, select, and evaluate

Goal: repeat prediction and correction, then measure performance on unseen images.

- Shuffle every training epoch and include the final partial mini-batch.
- Recompute sample-weighted train/validation metrics at epoch end.
- Save the minimum-validation-loss checkpoint and the final model separately.
- Export settings, history, summary, and learning curves.
- Freeze decisions before evaluating the official test split; export its confusion matrix.
- Record measured results and environment information without inventing benchmarks.

Implementation: `trainer.py`, `metrics.py`, and the training/evaluation scripts.

## Suggested study schedule

| Days | Focus | Completion condition |
|---|---|---|
| 1–2 | Data and NumPy arrays | Correct shapes and deterministic split |
| 3–5 | Layers and forward propagation | Finite logits and normalized probabilities |
| 6–8 | Derivatives and SGD | Finite-difference checks and tiny-data learning pass |
| 9–11 | Integration and training | Checkpoints and learning curves produced |
| 12–13 | Validation and final evaluation | Selected configuration and recorded test metrics |
| 14 | Documentation and presentation | Reproducible commands and explainable results |
