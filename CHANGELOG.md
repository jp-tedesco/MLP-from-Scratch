# Changelog

## Presentation update - 2026-11-16

- We simplified all three presentation parts for first-year statistics students, replacing advanced notation with examples and plain-language explanations while retaining the 30-minute pacing guide.
- Local academic files now live in the ignored `docs/slides/academic-local/` subdirectory; public sources and PDFs remain author-free.

- We moved the canonical presentations to `docs/slides/` and rebuilt the English and Portuguese main sources/PDFs as 24-slide talks with a 30-minute pacing guide.
- Part 1 now covers general theory only; Part 2 introduces MNIST and measured class proportions; Part 3 connects the theory to our implementation and results.
- Public decks contain no group-member names. Separate 27-slide academic variants are local-only and excluded by `.gitignore`.
- We compiled all four versions and checked their page counts, image references, rendered layouts, and the identity of their shared content. No training algorithms or measured model results changed.

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
