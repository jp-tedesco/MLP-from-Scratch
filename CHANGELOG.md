# Changelog

## Presentation update - 2026-11-18

- We added a vector repository QR code and clickable link to each title page.
- Existing implementation slides now include short commented excerpts from the actual NumPy code, with source provenance. The local academic maximum-selection example also shows its implementation.
- Slide counts remain 25 public and 28 academic. Revised narration budgets reserve two minutes for questions and at least 2:25 of margin within the 30-minute slot; dates and author visibility are unchanged.

## Presentation update - 2026-11-16

- The local academic conditional slide now focuses only on choosing the largest output, with 7 as an example. Speaker notes retain the distinction between a conceptual conditional explanation and the actual maximum operation; slide counts and timing remain unchanged.

- We added four editable teaching diagrams within Part 1 and consolidated repeated explanations, yielding 25 public slides and 28 local academic slides. Per-slide narration budgets reserve questions and at least 2:25 of margin inside the 30-minute academic slot. The academic conditional example now distinguishes explicit branches from maximum selection through NumPy.

- We expanded the start of Part 3 into three accessible slides following a real MNIST 7 from pixel brightness through the dense layers to its saved-checkpoint prediction. Example provenance and raw scores accompany the figures; remaining slides and model parameters are unchanged. Public decks now have 27 slides and local academic variants have 30.

- We shortened the preface to the topic of each part and removed author/date fields from the motivation slide. Local academic variants retain their three programming-fundamentals slides at the end of Part 1.

- We removed the dedicated closing slide from both languages and variants, leaving all remaining slides unchanged. Public decks now contain 25 slides; local academic decks contain 28.

- We added a title page before the unchanged motivation slide, followed by a preface, and replaced the final question slide with a closing slide. All intervening slides remain unchanged. Public decks now have 26 slides; local academic variants have 29, with the same dates and author-visibility rules.

- We rebuilt the presentation narrative around understanding connections, activations, predictions, and learning, with the project motivation on the first slide and explicit transition slides for the three parts.
- We removed project code excerpts and filenames from all decks. Academic programming concepts are explained through their roles in the workflow, while the main and academic talks retain a 30-minute pacing plan.

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
