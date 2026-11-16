# Project presentations

We provide two complete, standalone Beamer sources with equivalent coverage:

- `slides_en.tex`: English, 27 frames.
- `slides_pt.tex`: Brazilian Portuguese, 27 frames.
- `assets/`: real MNIST examples and localized plots of the recorded checkpoint.

Both presentations contain exactly three main sections:

1. Neural network fundamentals: our motivation, biological/artificial neurons, weights, biases, ReLU, architecture, forward propagation, and Softmax.
2. Dataset overview: MNIST, flattening, normalization, and the stratified training/validation/test protocol.
3. Learning process and results: cross-entropy, chain rule, dense-layer gradients, SGD, training curves, checkpoint selection, test metrics, confusion matrix, ROC/AUC, errors, and limitations.

The decks use the Madrid Beamer theme in 16:9 format. Portuguese figures are translated; mathematical notation and source-code identifiers are shared. A few hidden speaker notes clarify conventions. We retain the fixed-model measurements and do not imply a new training run. Dataset images are attributed to MNIST's authors; charts are derived from our published checkpoint.

## Compile to PDF

Use a TeX distribution containing Beamer, babel (English and Brazilian Portuguese), lmodern, amsmath, amssymb, booktabs, and graphicx. Run from this directory so relative image paths resolve:

```bash
pdflatex -interaction=nonstopmode -halt-on-error slides_en.tex
pdflatex -interaction=nonstopmode -halt-on-error slides_en.tex
pdflatex -interaction=nonstopmode -halt-on-error slides_pt.tex
pdflatex -interaction=nonstopmode -halt-on-error slides_pt.tex
```

Two passes populate the section overview. The outputs are `slides_en.pdf` and `slides_pt.pdf`. Alternatively, upload this entire directory (including `assets/`) as an Overleaf ZIP project, select one `.tex` file as the main document, use pdfLaTeX, and compile. Switch the main document for the other version. Never upload just the `.tex` files without their images.

We recommend **PDF for email and presentation** because equations and layout remain stable without requiring fonts or LaTeX on the recipient's computer. Use a PDF viewer's full-screen mode. If editable office slides are required, rebuild the content in PowerPoint or Marp and review the equations and layout; PDF-to-PowerPoint conversion may flatten content into images or alter formatting.

Build status: source structure and asset references were checked locally, but no LaTeX engine was available. We therefore do not claim successful compilation or visual verification of compiled slides. Compile with the commands above or Overleaf before presenting.

Beamer usage reference: [Overleaf documentation](https://www.overleaf.com/learn/latex/Beamer).
