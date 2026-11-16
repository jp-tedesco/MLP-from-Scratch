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

The sources contain hidden `\note{...}` speaker prompts with approximate durations. We explain equations and examples verbally rather than reading every bullet. Rehearsal determines the final pace; the plan reserves time for questions rather than assuming exactly one minute per slide.

Part 1 introduces artificial networks, the biological analogy, perceptrons, weights and biases, ReLU, forward propagation, Softmax, cross-entropy, gradient descent, and backpropagation. It contains no MNIST-specific or project-specific implementation discussion. Part 2 introduces the dataset. Only Part 3 connects the theory to our package and recorded experiment.

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

Separate academic variants are stored only under `.local/slides-academic/`, which is excluded by `.gitignore`. They are **not part of the repository**. They contain 27 slides: the same 24-slide main content, plus author information on the title page and a three-slide programming subsection immediately before Part 2.

That subsection maps conditionals, `for`/`while` loops, and `def`/return values to our existing code. It distinguishes best-checkpoint selection from early stopping, which we do not implement. Its three 40-second explanations add two minutes; reducing discussion to approximately 1:10 keeps the academic talk near 30 minutes. Apart from the author field and inserted subsection, the sources are identical to the main versions.

Before publishing, we check that public sources, extracted PDF text, and PDF metadata contain no group-member names. We also check page counts, referenced images, compilation, and rendered layouts. Keep local academic files and any bundles containing them out of commits.

All four versions were compiled with Tectonic 0.17.0. The public PDFs and the local variants were checked for page count and layout. Dataset-derived images are attributed to the MNIST authors; plots use our recorded checkpoint. See [Beamer documentation](https://www.overleaf.com/learn/latex/Beamer) for presentation options.
