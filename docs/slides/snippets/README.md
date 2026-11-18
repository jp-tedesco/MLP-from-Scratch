# Commented implementation excerpts

These excerpts reproduce existing statements from the project, with added
English teaching comments and reduced indentation. They are portions of
functions or methods, not standalone executable scripts. Validation and
surrounding setup remain in the full implementation.

| Excerpt | Original source |
|---|---|
| `preprocess.py` | `src/mnist_mlp/data/mnist.py` |
| `dense.py` | `src/mnist_mlp/layers.py` |
| `predict.py` | `src/mnist_mlp/network.py` |
| `training.py` | `src/mnist_mlp/trainer.py` |

`predict.py` uses the maximum logit. Softmax preserves this ordering, so the
selected class is also the one with the largest Softmax output.
`training.py` is the inner mini-batch loop; the enclosing epoch loop creates
the shuffled `order`, and the loss and optimizer are already initialized.
No numerical implementation or trained parameters were changed.
