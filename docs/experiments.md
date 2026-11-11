# Experiment record

## Protocol

Use only training data to update parameters and validation data to select settings. Select the checkpoint with minimum validation cross-entropy, then freeze all choices before measuring official test performance. Record each experiment in a separate output directory. A single seed is a reproducible example, not an estimate of variation across runs.

## Measured baseline

This full-data run was executed locally on 2026-11-11. These are measured values, not target scores or guarantees.

| Setting | Value |
|---|---|
| Architecture | 784 → 128 → 64 → 10 |
| Hidden activation | ReLU |
| Hidden/output initialization | He normal / Xavier normal |
| Optimizer | SGD, no momentum |
| Learning rate | 0.1 |
| Mini-batch size | 64, including partial batches |
| Epochs | 20 |
| Seed | 42 for split, initialization, and shuffling |
| Training / validation / test examples | 50,000 / 10,000 / 10,000 |
| Input representation | Flattened float32, pixels divided by 255 |
| Environment | Windows, Python 3.14.4, NumPy 2.5.4 |
| Checkpoint selection | Minimum validation loss, epoch 13 |
| Training loss / accuracy at selected epoch | 0.013355 / 99.73% |
| Validation loss / accuracy at selected epoch | 0.083610 / 97.73% |
| Official test loss / accuracy | 0.072777 / 97.90% |

```bash
python scripts/train.py --epochs 20 --batch-size 64 --learning-rate 0.1 --seed 42 --output-dir results/baseline
python scripts/evaluate.py --model results/baseline/best_model.npz --output results/baseline/test_metrics.json --no-download
```

The last epoch reached 97.80% validation accuracy, but epoch 13 had lower validation loss and was selected according to the predefined criterion. The run showed temporary loss spikes during earlier epochs; no settings were changed after inspecting the test score. Floating-point differences between systems can change the selected epoch or metrics.

Local run artifacts (weights, configuration, history, learning curves, and test confusion matrix) remain under `results/baseline/` and are excluded from Git. Regenerate them with the commands above. This record does not claim deployment readiness as a web service or externally audited production validation.

## Fields to record for future runs

For each new experiment, record its identifier and date; exact command; architecture and initializers; optimizer, learning rate, batch size, epochs, and seed; dataset split; Python/NumPy versions; checkpoint selection criterion and epoch; training and validation loss/accuracy; elapsed time; and observations. Only evaluate test data after final selection. If comparing methods statistically, use multiple seeds and report mean and spread using validation results before final testing.
