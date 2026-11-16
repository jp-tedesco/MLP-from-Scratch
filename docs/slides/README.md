# Thirty-minute project presentations

We publish two equivalent main presentations, each with **25 slides** and no author or group-member names:

| Language | Ready-to-present PDF | Complete editable source |
|---|---|---|
| English | [slides_en.pdf](slides_en.pdf) | [slides_en.tex](slides_en.tex) |
| Portuguese | [slides_pt.pdf](slides_pt.pdf) | [slides_pt.tex](slides_pt.tex) |

## Three-part structure and pacing

| Public slides | Part | Narration time |
|---|---|---|
| 1-3 | Title page, motivation, and preface | 1:15 |
| 4-9 | Part 1: network, connections, activations, and learning | 7:20 |
| 10-13 | Part 2: images, proportions, and data split | 3:15 |
| 14-25 | Part 3: one digit, training, results, and limitations | 11:45 |
| **Narration** | | **23:35** |
| After slide 25 | Questions | 2:00 |
| Reserved margin | Transitions, pauses, and slower explanations | 4:25 |
| **Maximum slot** | | **30:00** |

The **academic** variants add three 40-second programming slides at the end of Part 1: 25:35 narration, 2:00 for questions, and a 2:25 margin within the same 30-minute maximum. Their part-transition slides are 4, 13, and 17. Public part transitions are 4, 10, and 14. The time budget is recorded per slide in [pacing_en.json](pacing_en.json) and [pacing_pt.json](pacing_pt.json), and matches the hidden speaker notes. Rehearse with a timer; this is a speaking plan, not a guarantee of live duration. Finish the narration by minute 26, limit questions to two minutes, and close by minute 28 to preserve the remaining margin. Do not extend audience activities beyond their allotted slide time.

The intended audience is first-year statistics students and people without a machine learning background. The preface briefly names the topic of each part. Names and dates appear only on the title page of the local academic variants; the motivation slide shows neither. After the title page, the motivation slide explains our purpose: learning the foundations by building the calculations ourselves and connecting them to an observable result. Slides 4, 10, and 14 explicitly introduce each part, with a subtitle explaining its purpose. We organize the content as a story about information, connections, predictions, and improvement. There are no code excerpts, project filenames, derivative calculations, or matrix derivations in any deck. The sources contain hidden `\note{...}` speaker prompts with approximate durations and explanations; open the `.tex` file to rehearse them. Rehearsal determines the final pace. We use audience questions and discussion to make the 30-minute talk engaging rather than adding more technical material.

Part 1 uses four editable TikZ illustrations: a small layered network, a weighted neuron with a bias, the ReLU input/output curve, and an illustrative Softmax comparison. The diagrams live in `diagrams/` and are embedded in existing slides; they add no slides. Illustrative numbers are separate from measured project results. Part 1 connects the ideas rather than listing technical terms: neurons combine inputs, weights control influence, layers pass results onward, activations provide flexibility, and the output compares possible answers. Learning then follows three roles: measure a loss, trace how connections affect it, and adjust the weights. It contains no MNIST-specific or project-specific implementation discussion. Part 2 introduces the data needed for this cycle. Part 3 begins with a three-slide walkthrough of a real handwritten 7: normalized pixel values, learned combinations in the dense layers, and the actual Softmax output from the saved checkpoint. Example provenance and all ten scores are recorded in [digit_example.json](digit_example.json). We did not retrain or change model parameters. Part 3 then shows what happened when we used it, with guided interpretation of learning curves, accuracy, mistakes, and ROC/AUC. Secondary numerical details remain in the experiment record for questions.

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

We use Beamer (Madrid theme, 16:9), babel, lmodern, amsmath, amssymb, booktabs, graphicx, and TikZ. Upload both sources and the complete `assets/` and `diagrams/` folders to an Overleaf project to compile there. Select the desired `.tex` as the main document. We recommend the compiled **PDF** for email and full-screen presenting; it preserves equations without a TeX installation on the audience's computer.

## Local academic variants

Separate academic variants are stored only under `docs/slides/academic-local/`, which is excluded by `.gitignore`. They sit beside the public slides locally but are **not part of the repository**. They contain 28 slides: the same 25-slide main content, plus author information on the title page and a three-slide programming subsection immediately before Part 2. Public dates remain 16 November 2026; local academic dates remain 17 November 2026.

That subsection uses workflow responsibilities, not code or software/module names. Functions organize preparation, prediction, training, and evaluation; loops repeat work across images, layers, and epochs; conditionals check data and decide when a model should be saved. Its three 40-second explanations add two minutes, already included in the academic 25:35 narration budget. The conditionals slide distinguishes actual explicit branches (input validation and saving the best checkpoint) from maximum selection: prediction uses NumPy `argmax` on logits, which has the same ordering as Softmax probabilities. A hand-written maximum search could use conditionals, but that is not how our prediction method is implemented. Apart from the author field, local presentation date, and inserted subsection, the sources are identical to the main versions.

Before publishing, we check that public sources, extracted PDF text, and PDF metadata contain no group-member names. We also check page counts, referenced images, compilation, and rendered layouts. Keep local academic files and any bundles containing them out of commits.

All four versions were compiled with Tectonic 0.17.0. The public PDFs and the local variants were checked for page count and layout. Dataset-derived images are attributed to the MNIST authors; plots use our recorded checkpoint. See [Beamer documentation](https://www.overleaf.com/learn/latex/Beamer) for presentation options.
