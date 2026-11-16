# Atomic development history

We record real commits instead of rewriting history to simulate work. Earlier milestones already capture sequential implementation:

| Commit | Milestone |
|---|---|
| `a3911de` | Verified MNIST loading and stratified preprocessing |
| `68eac99` | Dense layers, activations, architecture, and checkpoints |
| `33b4574` | Stable cross-entropy and SGD |
| `ec5e513` | Training and evaluation commands |
| `1103d05` | Numerical tests and initial documentation |
| `992a820` | Reproducible trainer and partial-batch tests |
| `1fecaf6` | Validation before training updates |
| `d3b46c3` | ROC/AUC and diagnostic figures |
| `d881ad3` | Motivation, results, comparisons, and license |
| `aee6d56` | Published baseline checkpoints and run artifacts |

For the current update, we use separate commits for packaging, bilingual presentation sources, and contributor documentation. Equivalent commands are:

```bash
git commit -m "refactor(package): adopt installable mnist_mlp src layout"
git commit -m "feat(presentations): add English and Portuguese Beamer decks"
git commit -m "docs: document contribution workflow and package migration"
```

Each commit message describes a concrete change. We verify package behavior before publishing the refactor and clearly record that the local environment lacks a TeX compiler. Use `git log --oneline` to inspect the actual commit IDs for this update. We do not generate fake training runs or replace previously measured results.
