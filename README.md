# MLP from Scratch: MNIST

A fully vectorized multilayer perceptron built with Python and NumPy. No automatic differentiation, deep learning frameworks, or scikit-learn are used. Matplotlib is used only for visualization.

## Features

- Checksum-verified MNIST downloads, strict IDX parsing, and offline cache support.
- Reproducible stratified training/validation split; the official test set is reserved for final evaluation.
- Dense layers, ReLU, He initialization, stable Softmax and categorical cross-entropy.
- Explicit backpropagation and mini-batch SGD, including the last partial batch.
- Sample-weighted metrics, validation-based checkpoint selection, and learning curves.
- Atomic, pickle-free model checkpoints; finite-difference gradient tests and integration tests.

## Installation

Python 3.10 or newer is required. Run commands from the repository root.

```bash
git clone https://github.com/jp-tedesco/MLP-from-Scratch.git
cd MLP-from-Scratch
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

```bash
python -m pip install -r requirements.txt
```

If PowerShell activation is restricted, use `.\.venv\Scripts\python.exe` instead of `python`. Dependencies use compatible version ranges; each run records Python and NumPy versions. To capture the exact environment, run `python -m pip freeze > results/environment.txt`.

## Training

```bash
python scripts/train.py --epochs 20 --batch-size 64 --learning-rate 0.1 --seed 42 --output-dir results/baseline
```

The initial run downloads approximately 12 MB of gzip files. A default split contains 50,000 training, 10,000 validation, and 10,000 test images. Inputs are flattened to 784 float32 values in `[0, 1]`. Labels remain integer class indices. The training command loads the test partition through the dataset interface but never computes test metrics or uses test data to select parameters.

The output directory must be empty, preventing accidental overwrites. A successful run creates:

| Artifact | Purpose |
|---|---|
| `config.json` | Settings, sample counts, and environment versions |
| `history.csv` | Epoch loss, accuracy, and elapsed seconds |
| `best_model.npz` | Parameters with the lowest validation loss |
| `last_model.npz` | Parameters from the final epoch |
| `summary.json` | Selected epoch and its recorded metrics |
| `learning_curves.png` | Training and validation curves |

The training accuracy and loss are recomputed at the end of each epoch, using the same parameters as the validation metrics. This costs an extra inference pass but makes curves comparable. Accuracy values in exported files are fractions, not percentages.

For a short integration run, use a new directory:

```bash
python scripts/train.py --epochs 2 --train-limit 2000 --output-dir results/smoke
```

Other options include `--hidden-sizes 256 128`, `--validation-size 10000`, `--no-download`, and `--no-plot`. Use `--hidden-sizes` with no values for a linear Softmax classifier. See `python scripts/train.py --help` for details. Subset runs verify the pipeline; they do not represent a full-data benchmark.

## Final evaluation

Freeze architecture and hyperparameters using validation, then evaluate the selected checkpoint:

```bash
python scripts/evaluate.py --model results/baseline/best_model.npz --output results/baseline/test_metrics.json --no-download
```

Evaluation prints test loss, accuracy, and a confusion matrix (rows are true classes, columns are predicted classes). Output files must not already exist. Do not repeatedly tune settings using test scores.

## Architecture and mathematics

Default network: `784 → Dense(128) → ReLU → Dense(64) → ReLU → Dense(10)`.

For a batch of `B` examples, a dense layer computes `Z = A @ W + b`. Weights have shape `(n_in, n_out)` and biases `(1, n_out)`. Hidden-layer weights use He normal initialization, with standard deviation `sqrt(2 / n_in)`; the linear output uses Xavier normal initialization, `sqrt(2 / (n_in + n_out))`. All biases start at zero.

The network returns logits. The loss combines Softmax and cross-entropy from logits using a shifted log-sum-exp calculation. For logits `z` and class `y`, per-example loss is `log(sum(exp(z - max(z)))) - (z[y] - max(z))`. This avoids taking the logarithm of a probability that may have underflowed to zero.

The combined mean-loss derivative is `dZ = (P - Y) / B`, where `Y` is one-hot. Integer labels use indexing to avoid constructing `Y`. For each dense layer:

```text
dW = A_previous.T @ dZ
db = sum(dZ, axis=0, keepdims=True)
dA_previous = dZ @ W.T
```

ReLU multiplies the incoming derivative by `(Z > 0)`, choosing derivative zero at zero. The division by `B` happens once in the loss. All gradients are computed before SGD updates `parameter -= learning_rate * gradient`.

## Python API

```python
import numpy as np
from src.mlp.network import MLP
from src.mlp.losses import CrossEntropyLoss
from src.mlp.optimizers import SGD

model = MLP((784, 128, 64, 10), seed=42)
loss = CrossEntropyLoss()
optimizer = SGD(learning_rate=0.1)
x = np.zeros((2, 784), dtype=np.float32)
y = np.array([0, 1])
logits = model.forward(x, training=True)
value = loss.forward(logits, y)
model.backward(loss.backward())
optimizer.step(model.parameters())
probabilities = model.predict_proba(x)
```

Inference preserves pending training caches. Training caches and gradients are overwritten each batch. Checkpoints store architecture and parameters, not RNG state or trainer state: loading supports inference or a fresh training run, not an exact interrupted-run resume.

## Project structure

```text
src/data/mnist.py       # Download, parsing, normalization, stratified split
src/mlp/activations.py  # ReLU and stable Softmax
src/mlp/layers.py       # Dense forward/backward and parameters
src/mlp/losses.py       # Stable categorical cross-entropy
src/mlp/metrics.py      # Classification accuracy
src/mlp/network.py      # Architecture, prediction, serialization
src/mlp/optimizers.py   # SGD
src/mlp/trainer.py      # Mini-batch training and batched evaluation
scripts/train.py       # Training CLI and artifact export
scripts/evaluate.py    # Test evaluation CLI
tests/                 # Numerical, data, and integration tests
docs/project_plan.md   # Implementation milestones
docs/experiments.md    # Experiment protocol and measured results
```

## Verification and troubleshooting

```bash
python -m unittest discover -s tests -v
```

Tests include finite differences in float64, extreme logits, invalid inputs, synthetic learning, IDX corruption detection, and serialization round trips. Seeds control initialization, splitting, and shuffling. Different NumPy/BLAS versions or thread counts may still produce slightly different floating-point results.

If loss becomes nonfinite, check data and decrease the learning rate. If learning stalls, verify gradients with the test suite and try overfitting a tiny subset. Diverging train/validation curves may indicate overfitting. CPU performance depends on the NumPy BLAS build; limiting BLAS threads can help small matrix multiplications (for example, set `OPENBLAS_NUM_THREADS=1` before starting Python).

The project provides a tested local training and inference pipeline. Deployment as a hosted service, monitoring, and input-image acquisition are outside its scope. External images must match MNIST preprocessing: 28×28 grayscale, dark background, bright foreground, and the same flattening order.

## Dataset reference

MNIST was created by Yann LeCun, Corinna Cortes, and Christopher J. C. Burges. Downloads use the [CVDF MNIST mirror](https://github.com/cvdfoundation/mnist), which documents the original partitions and IDX format, with an OSSCI mirror fallback. Published MD5 hashes detect file corruption; TLS authenticates the download connection.
