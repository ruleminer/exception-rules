"""Recompute survival fitting-time summaries from the saved experiment records.

Run from any directory with Python 3.10+; no third-party dependencies required.
Quartiles use inclusive linear interpolation (the pandas default convention).
"""

import csv
import math
import statistics
from collections import Counter
from pathlib import Path


def summarize():
    root = Path(__file__).resolve().parents[2]
    results = root / "experiments/survival/results/comparision"
    source = results / "exceptions_summary_ALL.csv"
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        records = list(reader)
    with (results / "exceptions_summary.csv").open(encoding="utf-8", newline="") as stream:
        original = list(csv.DictReader(stream, fieldnames=columns))

    models = {
        "survival_tree": "SurvivalTree",
        "algorithm_rulekit": "RuleKit",
        "algorithm_log_rank": "SurvivalRules",
        "algorithm_log_rank_with_exceptions": "SurvivalExceptionRules",
    }
    keys = [(r["dataset"], r["model"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate dataset/model records")
    if len(records) != 100 or len({r["dataset"] for r in records}) != 25:
        raise ValueError("The documented source contains 25 datasets and 100 rows")
    if Counter(r["model"] for r in records) != Counter({key: 25 for key in models}):
        raise ValueError("Incomplete four-method comparison")
    for dataset in {r["dataset"] for r in records}:
        if {r["model"] for r in records if r["dataset"] == dataset} != set(models):
            raise ValueError(f"Incomplete comparison for {dataset}")
    timings = {(r["dataset"], r["model"]): float(r["raw_time"]) for r in records}
    original_timings = {(r["dataset"], r["model"]): float(r["raw_time"]) for r in original}
    if len(original) != len(records) or timings != original_timings:
        raise ValueError("The two source files disagree on fitting times")
    if not all(math.isfinite(t) and t > 0 for t in timings.values()):
        raise ValueError("Missing, nonfinite, or nonpositive fitting time")

    writer = csv.writer(__import__("sys").stdout, lineterminator="\n")
    writer.writerow(["Method", "N", "Mean", "Median", "Q1", "Q3", "Min", "Max"])
    for key, label in models.items():
        values = [float(r["raw_time"]) for r in records if r["model"] == key]
        q1, _, q3 = statistics.quantiles(values, method="inclusive")
        summary = [statistics.mean(values), statistics.median(values), q1, q3, min(values), max(values)]
        writer.writerow([label, len(values), *(f"{value:.3f}" for value in summary)])


if __name__ == "__main__":
    summarize()
