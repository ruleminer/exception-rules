# Exception-Rules

This repository contains the datasets, algorithm implementations, and experimental results associated with the article "Discovering Exception Rules via Sequential Covering in
Classification, Regression, and Survival Data" submitted to the ECAI 2025 conference.

## Repository Structure

- `data/` — Datasets used in the experiments.
- `decision-rules/` — Modules related to the generation of decision rules.
- `exception-rules/` — Main package containing algorithm implementations:
  - `classification/` — Algorithms for classification tasks,
  - `regression/` — Algorithms for regression tasks,
  - `survival/` — Algorithms for survival analysis,
  - `measures.py` — Evaluation metrics for models,
  - `tests/` — Unit tests.
- `experiments/` — Scripts and notebooks for reproducing experiments:
  - `plots/` — Visualizations of results,
  - `results/` — Saved experimental results,
  - `rules_from_articles/` — Rules utilized in the article,
  - `example_classification.py`, `example_regression.py`, `example_survival.py` — Example usage scripts.
- `setup.py` — Installation script for the package.
- `README.md` — Repository description.
- `VERSION.txt` — Project version information.

## Requirements

- Python >= 3.8
- Dependencies listed in the `setup.py` file.

## Usage Instructions

1. Install the required libraries:

   ```bash
   pip install -e .
   ```

2. Run the example scripts:

   ```bash
   python experiments/example_classification.py
   python experiments/example_regression.py
   python experiments/example_survival.py
   ```

3. To replicate the experiments presented in the article, use the scripts and Jupyter notebooks (`.ipynb`) located in the `experiments/` folder.

## Additional Information

The code and datasets contained in this repository are intended solely for research and academic purposes in relation to the ECAI 2025 conference submission.
