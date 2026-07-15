# Exception-Rules

This repository contains the datasets, algorithm implementations, and experimental results associated with the article "Discovering Exception Rules via Sequential Covering in
Classification, Regression, and Survival Data".

## Repository Structure

- `exception_rules/` — Main package containing algorithm implementations:
  - `classification/` — Algorithms for classification tasks,
  - `regression/` — Algorithms for regression tasks,
  - `survival/` — Algorithms for survival analysis,
  - `decision-rules/` — Library for representing rules.
  - `measures.py` — Evaluation metrics for models,
- `tests/` — Unit tests.
- `experiments/` — Scripts for research experiments using package
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


## Documentation
 
Full documentation is available [here](https://ruleminer.github.io/exception-rules/)
