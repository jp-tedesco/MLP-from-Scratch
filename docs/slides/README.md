# Four-part project presentations

We publish equivalent English and Portuguese presentations with **26 slides**.
Public sources and PDFs contain no group-member names.

| Language | PDF | Editable source |
|---|---|---|
| English | [slides_en.pdf](slides_en.pdf) | [slides_en.tex](slides_en.tex) |
| Portuguese | [slides_pt.pdf](slides_pt.pdf) | [slides_pt.tex](slides_pt.tex) |

## Structure and 30-minute maximum

| Public slides | Content | Narration |
|---|---|---|
| 1-3 | Title, motivation, and four-part preface | 1:15 |
| 4-9 | Part 1: neural networks linked to commented code | 6:25 |
| 10-12 | Part 2: the MNIST dataset | 1:25 |
| 13-16 | Part 3: preprocessing, labels, and data splitting | 3:45 |
| 17-26 | Part 4: implementation summary, training, and results | 9:30 |
| **Narration** | | **22:20** |
| After slide 26 | Questions | 2:00 |
| Reserved margin | Transitions, pauses, slower explanations | 5:40 |
| **Maximum slot** | | **30:00** |

Local academic variants contain **29 slides**, adding the three-slide function
walkthrough at the end of Part 1. Their narration is **25:25**, with **2:00**
for questions and **2:35** of margin. Part transitions are slides 4, 13, 16,
and 20 in those variants, and 4, 10, 13, and 17 in the public decks.

Per-slide budgets are in [pacing_en.json](pacing_en.json) and
[pacing_pt.json](pacing_pt.json), matching the hidden speaker notes. Rehearse
with a timer. Finish narration by minute 26 and questions by minute 28;
keep the remaining two minutes as margin. A planned duration does not guarantee
live delivery speed. Avoid repeating code explanations after their first use.

## What each part explains

**Part 1** connects the existing diagrams to real implementation excerpts:
weighted input combinations in `Dense.forward`, the ReLU rule in
`ReLU.forward`, and stable score normalization in `softmax`. It distinguishes
Softmax output normalization from pixel preprocessing. No matrix derivations
or derivative calculations are needed.

**Part 2** introduces the original 28-by-28 grayscale images, labels 0 through
9, official dataset sizes, handwriting variation, and class proportions.
Detailed pixel conversion is deliberately left to Part 3.

**Part 3** explains the preprocessing actually used in this project:

- Flatten each image into 784 values in a fixed order.
- Convert image values to `float32`, then divide by 255 to obtain `[0, 1]`.
- Keep labels separate as `int64` class IDs. This experiment uses integer
  targets directly; it does not perform a one-hot conversion.
- Split only the official 60,000-image training pool into 50,000 training and
  10,000 validation images. The split is stratified by digit, proportional to
  class counts, and reproducible with seed 42.
- Keep the official 10,000-image test set separate for final evaluation.

The pixel scale uses a fixed constant, not statistics fitted across the data
splits. Download checksums and IDX checks validate the input files. Shuffling
mini-batch order happens during training and is explained separately in Part 4.
Class counts are in [class_counts.json](class_counts.json).

**Part 4** summarizes the implementation using already prepared inputs:
`MLP.forward` calculates scores, `CrossEntropyLoss.forward` measures the error,
`backward` propagates gradients, `SGD.step` updates parameters, and `predict`
selects the largest-score class. Commented code shows the core calls and the
epoch/mini-batch loops, followed by checkpoint selection, learning curves,
test metrics, a confusion matrix, errors, and the real prediction of a 7.
ROC/AUC is summarized in the final metrics table; full ROC figures remain in
the experiment assets rather than requiring another presentation slide.

## Code excerpts and repository access

Every title page includes a vector QR code and clickable link to
`https://github.com/jp-tedesco/MLP-from-Scratch`. Public dates remain
16 November 2026; local academic dates remain 17 November 2026. Motivation
slides have no author or date fields.

[Commented excerpts](snippets/README.md) reproduce real project statements,
with English teaching comments and selected surrounding lines omitted.
They are excerpts of functions/methods, not standalone scripts. Surrounding
slide explanations are localized. Explain their inputs, purpose, and result
rather than each NumPy argument. The code examples do not alter training code
or model parameters.

The real example and its ten output scores are recorded in
[digit_example.json](digit_example.json). Results use the existing selected
checkpoint; no retraining was performed. Details and limitations remain in
[the experiment record](../experiments.md).

## Compile and present

From `docs/slides/`, run:

```bash
tectonic slides_en.tex
tectonic slides_pt.tex
```

Alternatively, run pdfLaTeX twice per source to resolve references. Upload the
sources plus all of `assets/`, `diagrams/`, and `snippets/` to Overleaf and
select the desired language as the main document. Dependencies are Beamer,
babel, lmodern, amsmath, amssymb, booktabs, graphicx, TikZ, listings, and qrcode.
The existing layout is Madrid, 16:9. We recommend PDF for email and full-screen
presentation. Tectonic 0.17.0 compiles the decks locally.

## Local academic variants

`docs/slides/academic-local/` contains the local-only English and Portuguese
sources/PDFs, supporting assets, excerpts, and timing records. `.gitignore`
excludes the entire directory. These variants retain the title-page authors
and the three extra function/loop/decision slides at the end of Part 1.

Their walkthrough shows score/loss calls, reverse-layer backpropagation,
optimizer updates, and the complete prediction method. A conditional sentence
explains choosing the largest output, while the actual method uses NumPy
`argmax` on logits; Softmax preserves their ordering. No explicit per-class
conditional is claimed.

We check compilation, rendered layouts, timing totals, source references,
code provenance, and QR decoding. Public sources, PDF text, and metadata are
checked for group-member names. Never commit academic variants or bundles
containing them. Dataset images are attributed to the MNIST authors; plots
use the recorded checkpoint.
