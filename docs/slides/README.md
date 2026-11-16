# Thirty-minute project presentations

We publish two equivalent main presentations, each with **27 slides** and no author or group-member names:

| Language | Ready-to-present PDF | Complete editable source |
|---|---|---|
| English | [slides_en.pdf](slides_en.pdf) | [slides_en.tex](slides_en.tex) |
| Portuguese | [slides_pt.pdf](slides_pt.pdf) | [slides_pt.tex](slides_pt.tex) |

## Three-part structure and pacing

| Slides | Part | Suggested time |
|---|---|---|
| 1 | Title page | 0:30 |
| 2 | Project motivation | 0:45 |
| 3 | Preface | 0:30 |
| 4-10 | Part 1: concepts that explain the learning process | 8:45 |
| 11-15 | Part 2: images, brightness values, proportions, and data split | 5:00 |
| 16-27 | Part 3: training, interpretation of results, and limitations | 12:50 |
| After slide 27 | Audience conversation, without a separate closing slide | 1:40 |
| **Total** | | **30:00** |

The intended audience is first-year statistics students and people without a machine learning background. The preface briefly names the topic of each part. Names and dates appear only on the title page of the local academic variants; the motivation slide shows neither. After the title page, the motivation slide explains our purpose: learning the foundations by building the calculations ourselves and connecting them to an observable result. Slides 4, 11, and 16 explicitly introduce each part, with a subtitle explaining its purpose. We organize the content as a story about information, connections, predictions, and improvement. There are no code excerpts, project filenames, derivative calculations, or matrix derivations in any deck. The sources contain hidden `\note{...}` speaker prompts with approximate durations and explanations; open the `.tex` file to rehearse them. Rehearsal determines the final pace. We use audience questions and discussion to make the 30-minute talk engaging rather than adding more technical material.

Part 1 connects the ideas rather than listing technical terms: neurons combine inputs, weights control influence, layers pass results onward, activations provide flexibility, and the output compares possible answers. Learning then follows three roles: measure a loss, trace how connections affect it, and adjust the weights. It contains no MNIST-specific or project-specific implementation discussion. Part 2 introduces the data needed for this cycle. Part 3 begins with a three-slide walkthrough of a real handwritten 7: normalized pixel values, learned combinations in the dense layers, and the actual Softmax output from the saved checkpoint. Example provenance and all ten scores are recorded in [digit_example.json](digit_example.json). We did not retrain or change model parameters. Part 3 then shows what happened when we used it, with guided interpretation of learning curves, accuracy, mistakes, and ROC/AUC. Secondary numerical details remain in the experiment record for questions.

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

Separate academic variants are stored only under `docs/slides/academic-local/`, which is excluded by `.gitignore`. They sit beside the public slides locally but are **not part of the repository**. They contain 30 slides: the same 27-slide main content, plus author information on the title page and a three-slide programming subsection immediately before Part 2. Public dates remain 16 November 2026; local academic dates remain 17 November 2026.

That subsection uses workflow responsibilities, not code or software/module names. Functions organize preparation, prediction, training, and evaluation; loops repeat work across images, layers, and epochs; conditionals check data and decide when a model should be saved. Its three 40-second explanations add two minutes. With 25 seconds for final discussion, the academic pacing plan is approximately 30:45; rehearsal can bring it closer to 30 minutes. Apart from the author field, local presentation date, and inserted subsection, the sources are identical to the main versions.

Before publishing, we check that public sources, extracted PDF text, and PDF metadata contain no group-member names. We also check page counts, referenced images, compilation, and rendered layouts. Keep local academic files and any bundles containing them out of commits.

All four versions were compiled with Tectonic 0.17.0. The public PDFs and the local variants were checked for page count and layout. Dataset-derived images are attributed to the MNIST authors; plots use our recorded checkpoint. See [Beamer documentation](https://www.overleaf.com/learn/latex/Beamer) for presentation options.
