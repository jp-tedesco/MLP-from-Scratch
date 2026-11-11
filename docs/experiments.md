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
| Test macro precision / recall / F1 | 97.90% / 97.87% / 97.88% |
| Test weighted F1 | 97.90% |
| Test macro / micro ROC AUC (one-vs-rest) | 0.999621 / 0.999617 |
| Correct / incorrect test predictions | 9,790 / 210 |
| Accuracy Wilson 95% interval | 97.60–98.16% (fixed model, binomial assumption) |
| Trainable parameters | 109,386 |

```bash
python scripts/train.py --epochs 20 --batch-size 64 --learning-rate 0.1 --seed 42 --output-dir results/reproduction
python scripts/evaluate.py --model results/reproduction/best_model.npz --output results/reproduction/test_metrics.json --no-download
```

The last epoch reached 97.80% validation accuracy, but epoch 13 had lower validation loss and was selected according to the predefined criterion. The run showed temporary loss spikes during earlier epochs; no settings were changed after inspecting the test score. Floating-point differences between systems can change the selected epoch or metrics.

The recorded run artifacts, including both model checkpoints, are published under `results/baseline/`. The commands above use a separate directory to reproduce the experiment without overwriting the published run. Paths in the published configuration were normalized to repository-relative notation; model parameters and measured results are unchanged. This record does not claim deployment readiness as a web service or externally audited production validation.

## Extended diagnostic report

The original checkpoint was evaluated again to add precision, recall, F1, ROC AUC, and plots. This was additional analysis of the same fixed predictions, not a new independent experiment. No retraining, hyperparameter changes, or checkpoint reselection was performed using these diagnostics. The unchanged test accuracy and loss match the original report.

The report is committed at `docs/baseline_metrics.json`; figures and the epoch history used to draw the learning curves are in `docs/assets/`. The checkpoint's SHA-256 is `6f1bdd538dde0196afb3828915144721ea5a4b6606c9c8a8d72ad2453b5bfe4d`. Figures use the first test example of each class and the first ten misclassifications in dataset order.

ROC uses Softmax probabilities in ten one-vs-rest tasks, grouping equal scores before trapezoidal integration. Macro AUC is the mean of ten class AUCs; micro AUC pools all 100,000 image/class pairs. Classes absent from a general-purpose report receive undefined (`null`) AUC, and the macro AUC is then undefined. This does not occur on the full MNIST test set. Precision, recall, and F1 use zero for zero denominators.

The 95% Wilson interval concerns sample uncertainty for one fixed classifier. It is not an interval across random seeds, and it does not account for test-set reuse, dataset shift, or calibration. The high ROC AUC describes score ranking, not an equivalent multiclass accuracy.

## Fields to record for future runs

For each new experiment, record its identifier and date; exact command; architecture and initializers; optimizer, learning rate, batch size, epochs, and seed; dataset split; Python/NumPy versions; checkpoint selection criterion and epoch; training and validation loss/accuracy; elapsed time; and observations. Only evaluate test data after final selection. If comparing methods statistically, use multiple seeds and report mean and spread using validation results before final testing.
