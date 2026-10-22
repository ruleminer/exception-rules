# Final supplement

The supplement contains theoretical basics, induction algorithms, exception-rule
discovery results, computational efficiency, and references. Timing comparisons
cover classification, regression, and survival, including a per-dataset analysis
of row and predictor counts. Results are presented in tables and an existing
GBSG2 Kaplan--Meier figure. The revised edition explicitly distinguishes
historical results, current implementation behavior, and outstanding evidence.

## Files to edit

- `latex/main.tex` and `latex/sections/`: authoritative, editable LaTeX sources.
- `../source/supplementary_materials.rst` and `../source/supplementary/`: generated Sphinx edition.
- `data/runtime_by_dataset.csv`: joined dataset dimensions and saved fitting times.

The build synchronises the supplement from LaTeX before creating HTML. Edit
the LaTeX sources; direct edits to generated RST pages are overwritten. The
normal package Sphinx build performs the same synchronisation automatically.
To synchronise sources without building HTML, run:

```console
python docs/supplementary/sync_from_latex.py
python docs/supplementary/sync_from_latex.py --check
```

## Validate and package

From the repository root:

```console
python -m pip install -r docs/supplementary/requirements.txt
python docs/supplementary/build.py
```

The build checks LaTeX inputs, citations, labels, and environments, builds HTML
with Sphinx warnings treated as errors, checks HTML anchors, and creates:

- `docs/build/exception-rules-supplementary-latex.zip`: editable project for Overleaf;
- `docs/build/exception-rules-supplementary.zip`: HTML and supporting assets;
- `docs/build/supplementary/supplementary_materials.html`: local HTML edition.

To validate and package only the LaTeX sources, without Sphinx:

```console
python docs/supplementary/build.py --latex-only
```

This command checks source structure; it does not compile a PDF. Compile
`latex/main.tex` with pdfLaTeX twice or `latexmk -pdf main.tex`. HTML equations
use Sphinx's MathJax renderer, which requires internet access when viewing.
The normal package documentation also includes these RST chapters.

## Reproduce numerical summaries

No model training is required. These helpers use Python's standard library:

```console
python docs/supplementary/summarize_classification_times.py
python docs/supplementary/summarize_survival_times.py
python docs/supplementary/analyze_runtime_by_size.py --write
```

The first two print validated timing summaries. The size-analysis command regenerates the joined
CSV and the dataset-size fragments `runtime_by_size.inc` and `runtime_by_size.tex`;
it overwrites manual changes in those two fragments. Regression aggregate times
are maintained in the LaTeX chapter, and the size analysis uses saved regression
records as well.

The supplement is self-contained, with standard literature citations in the
bibliography. Numerical results describe recorded experiments; implementation
details and limits of inference are stated alongside them. Keep the original
ARFF inputs and experiment records to retain reproducibility.
