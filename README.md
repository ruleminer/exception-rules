# Exception-Rules

This repository contains the datasets, algorithm implementations, and experimental results associated with the article "Discovering Exception Rules via Sequential Covering in
Classification, Regression, and Survival Data" submitted to the ECAI 2025 conference.

## Repository Structure

- `data/` — Datasets used in the experiments.
- `decision-rules/` — Library for representing rules.
- `exception-rules/` — Main package containing algorithm implementations:
  - `classification/` — Algorithms for classification tasks,
  - `regression/` — Algorithms for regression tasks,
  - `survival/` — Algorithms for survival analysis,
  - `measures.py` — Evaluation metrics for models,
  - `tests/` — Unit tests.
- `experiments/` — Scripts for reproducing experiments:
  - `plots/` — Visualizations of results,
  - `results/` — Saved experimental results,
  - `rules_from_articles/` — Rules from other articles shown in the article,
  - `example_classification.py`, `example_regression.py`, `example_survival.py` — Example usage scripts.
- `setup.py` — Installation script for the package.
- `README.md` — Repository description.
- `VERSION.txt` — Project version information.

## Requirements

- Python == 3.10
- Dependencies listed in the `requirements.txt` file.

## Usage Instructions

1. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the example scripts:

   ```bash
   python example_classification.py
   python example_regression.py
   python example_survival.py
   ```

3. To replicate the experiments presented in the article, use the scripts located in the `experiments/` folder.

## Additional Information

The code and datasets contained in this repository are intended solely for research and academic purposes in relation to the ECAI 2025 conference submission.
