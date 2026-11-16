# Thirty-minute project presentations

We publish two equivalent main presentations, each with **24 slides** and no author or group-member names:

| Language | Ready-to-present PDF | Complete editable source |
|---|---|---|
| English | [slides_en.pdf](slides_en.pdf) | [slides_en.tex](slides_en.tex) |
| Portuguese | [slides_pt.pdf](slides_pt.pdf) | [slides_pt.tex](slides_pt.tex) |

## Three-part structure and pacing

| Slides | Part | Suggested time |
|---|---|---|
| 1 | Title and roadmap stated verbally | 0:30 |
| 2-10 | General neural network theory only | 11:15 |
| 11-14 | MNIST, normalization, class distribution, and data split | 4:00 |
| 15-24 | Our implementation, learning curves, evaluation, and limitations | 11:05 |
| Discussion | Audience questions | 3:10 |
| **Total** | | **30:00** |

The intended audience is first-year statistics students and people without a machine learning background. We use short definitions, simple arithmetic, familiar analogies, and questions instead of derivative calculations or dense matrix notation. Terms are explained when introduced. The sources contain hidden `\note{...}` speaker prompts with examples and approximate durations; open the `.tex` file to rehearse them. Rehearsal determines the final pace. We fill the 30-minute slot with explanation and interaction, not additional technical detail.

Part 1 explains learning from examples, the neuron analogy, weights and biases, layers, ReLU, Softmax, prediction errors, and small learning adjustments. Backpropagation and gradient descent are introduced by their roles, without derivatives. It contains no MNIST-specific or project-specific implementation discussion. Part 2 explains images, pixels, class proportions, and the roles of training/validation/test data in plain language. Part 3 focuses on what we built and how to interpret the learning curves, accuracy, confusion matrix, and ROC/AUC, with implementation details reserved for speaker notes or questions.

Class proportions are computed from the actual seed-42 split. Exact counts are recorded in [class_counts.json](class_counts.json). Charts and metrics use the original selected checkpoint; no model was retrained to produce this update. Recorded results and limitations remain in [the experiment record](../experiments.md).

## Compile and present

From `docs/slides/`, run:

```bash
tectonic slides_en.tex
tectonic slides_pt.tex
```

Alternatively, use pdfLaTeX twice for each file to resolve references:

```bash
pdflatex -interaction=nonstopmode -halt-on-error slides_en.tex
pdflatex -interaction=nonstopmode -halt-on-error slides_en.tex
pdflatex -interaction=nonstopmode -halt-on-error slides_pt.tex
pdflatex -interaction=nonstopmode -halt-on-error slides_pt.tex
```

We use Beamer (Madrid theme, 16:9), babel, lmodern, amsmath, amssymb, booktabs, and graphicx. Upload both sources and the entire `assets/` folder to an Overleaf project to compile there. Select the desired `.tex` as the main document. We recommend the compiled **PDF** for email and full-screen presenting; it preserves equations without a TeX installation on the audience's computer.

## Local academic variants

Separate academic variants are stored only under `docs/slides/academic-local/`, which is excluded by `.gitignore`. They sit beside the public slides locally but are **not part of the repository**. They contain 27 slides: the same 24-slide main content, plus author information on the title page and a three-slide programming subsection immediately before Part 2. Public dates remain 16 November 2026; local academic dates remain 17 November 2026.

That subsection explains conditionals, repetition, and functions using short code excerpts and plain-language descriptions. It maps `if`/`elif`/`else`, `for`/`while`, and `def`/return values to our existing modules. Its three 40-second explanations add two minutes; reducing discussion to approximately 1:10 keeps the academic talk near 30 minutes. Apart from the author field, local presentation date, and inserted subsection, the sources are identical to the main versions.

Before publishing, we check that public sources, extracted PDF text, and PDF metadata contain no group-member names. We also check page counts, referenced images, compilation, and rendered layouts. Keep local academic files and any bundles containing them out of commits.

All four versions were compiled with Tectonic 0.17.0. The public PDFs and the local variants were checked for page count and layout. Dataset-derived images are attributed to the MNIST authors; plots use our recorded checkpoint. See [Beamer documentation](https://www.overleaf.com/learn/latex/Beamer) for presentation options.
