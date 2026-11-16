# Contributing

We welcome focused changes that help explain or improve our NumPy implementation.

## Local setup

Use Python 3.10 or newer, create a virtual environment, and install from the repository root:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

We use the conventional `src/mnist_mlp/` package layout. Import `mnist_mlp`, not `src`. The editable installation makes local source changes available immediately. Training and evaluation commands remain in `scripts/`; execute them from the repository root.

## Changes and review

1. Create a descriptive branch, such as `feat/learning-rate-schedule`.
2. Keep each change focused. Include relevant tests for numerical behavior or regressions.
3. Run the suite and explain the behavior changed, verification performed, and limitations in a pull request.
4. Write code and documentation in English. Use professional neutral wording or collaborative plural ("we implemented"); preserve translated presentation material in its target language.
5. Do not add deep learning frameworks or scikit-learn model dependencies.

## Experiment protocol

We choose settings using training and validation data. Test diagnostics already published in this project must not be described as fresh evidence for new settings. Use a new result directory for each run; record settings, seed, environment, selection criterion, and measured metrics. We version only explicitly published run artifacts, not the downloaded dataset or local environments.

## Atomic commits

Use messages such as `feat(data): validate IDX input`, `fix(training): preserve partial batches`, `test(metrics): cover tied ROC scores`, and `docs: clarify checkpoint selection`. A refactor should preserve predictions unless its purpose explicitly includes a model change. Never rewrite public history merely to simulate earlier implementation stages.

## License

Contributions to the original project code and documentation are provided under the repository's MIT license. Attribute dataset-derived images and external references separately.
