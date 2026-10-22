"""Summarize historical classification fitting times without counting prediction variants twice."""

import csv
import math
import statistics
import sys
from collections import Counter
from pathlib import Path


def summarize():
    root = Path(__file__).resolve().parents[2]
    folder = root / "experiments/classification/results/comparision"
    with (folder / "exceptions_summary_ALL.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        records = list(reader)
    variants = [f"algorithm_cer_with_exceptions_prediction_type_{i}" for i in range(4)]
    models = {
        "decision_tree": "DecisionTreeClassifier",
        "algorithm_rulekit": "RuleKit",
        "algorithm_cer": "ClassificationRules",
        variants[0]: "ClassificationExceptionRules",
    }
    datasets = {r["dataset"] for r in records}
    expected = set(models) | set(variants)
    if len(datasets) != 49 or len(records) != 343:
        raise ValueError("Expected 49 datasets and 343 saved records")
    if Counter(r["model"] for r in records) != Counter({model: 49 for model in expected}):
        raise ValueError("Unexpected method coverage")
    keyed = {(r["dataset"], r["model"]): r for r in records}
    if len(keyed) != len(records):
        raise ValueError("Duplicate dataset/model records")
    for dataset in datasets:
        if {r["model"] for r in records if r["dataset"] == dataset} != expected:
            raise ValueError(f"Missing method for {dataset}")
        for column in ["raw_time", "process_time", "thread_time"]:
            if len({float(keyed[dataset, variant][column]) for variant in variants}) != 1:
                raise ValueError(f"Prediction variants disagree on {column}: {dataset}")
    with (folder / "exceptions_summary.csv").open(encoding="utf-8", newline="") as stream:
        original = list(csv.DictReader(stream, fieldnames=columns))
    times = {(r["dataset"], r["model"]): float(r["raw_time"]) for r in records}
    original_times = {(r["dataset"], r["model"]): float(r["raw_time"]) for r in original}
    if len(original) != len(records) or times != original_times:
        raise ValueError("Source files disagree on timing records")
    if not all(math.isfinite(t) and t > 0 for t in times.values()):
        raise ValueError("Invalid elapsed fitting time")

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["Method", "N", "Mean", "Median", "Q1", "Q3", "Min", "Max"])
    for model, label in models.items():
        values = [times[dataset, model] for dataset in sorted(datasets)]
        q1, _, q3 = statistics.quantiles(values, method="inclusive")
        summary = [statistics.mean(values), statistics.median(values), q1, q3, min(values), max(values)]
        writer.writerow([label, len(values), *(f"{v:.3f}" for v in summary)])


if __name__ == "__main__":
    summarize()
