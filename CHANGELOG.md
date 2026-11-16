# Changelog

## 0.1.0 - 2026-11-15

### Repository organization

- We moved the numerical implementation into the installable `mnist_mlp` package under `src/`.
- We added `pyproject.toml` as the package metadata and dependency source of truth. `requirements.txt` remains an editable-install convenience file.
- We updated scripts, tests, documentation, and CI to use the installed package.
- We changed the default training output to `results/my-run` because the published `results/baseline` directory is already populated.
- We added contribution guidelines and complete English/Portuguese Beamer presentations.

### Migration

After updating, run `python -m pip install -e .`. Replace imports beginning with `src.mlp` with `mnist_mlp`, and `src.data` with `mnist_mlp.data`. The numerical algorithms and checkpoint format are unchanged. Existing published `.npz` checkpoints remain readable.

### Verification

The local test suite passed all 24 tests after the package move. Evaluation of the original selected checkpoint retained 97.90% test accuracy and 0.0727768813 cross-entropy. No new model was trained for the reorganization. LaTeX sources were checked structurally; compiled-slide rendering remains unverified because no TeX engine was available locally.
