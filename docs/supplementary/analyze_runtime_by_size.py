"""Join saved fitting times to ARFF dimensions; optionally write supplement tables.

No model training or third-party dependencies. Run with --write to regenerate
the RST/LaTeX fragments and the joined CSV used by the size analysis.
"""

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = {
    "classification": (49, ["decision_tree", "algorithm_rulekit", "algorithm_cer", "algorithm_cer_with_exceptions_prediction_type_0"]),
    "regression": (48, ["DecisionTreeRegressor", "RuleKit", "RegressionRules", "RegressionExceptionRules"]),
    "survival": (25, ["survival_tree", "algorithm_rulekit", "algorithm_log_rank", "algorithm_log_rank_with_exceptions"]),
}
NAMES = {"classification": "Classification", "regression": "Regression", "survival": "Survival analysis"}


def canonical(name):
    # Survival result files use a different numbering from the ARFF directory.
    return re.sub(r"^\d+_", "", Path(name).stem).casefold()


def field_count(line):
    """Count dense ARFF fields while respecting single/double quotes and escapes."""
    count, quote, escaped, start = 1, None, False, True
    for char in line:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif quote:
            if char == quote:
                quote = None
        elif char in "\"'" and start:
            quote, start = char, False
        elif char == ",":
            count, start = count + 1, True
        elif char == "%" and start:
            break
        elif not char.isspace():
            start = False
    if quote:
        raise ValueError("Multiline/unclosed quoted ARFF field")
    return count


def dimensions(path):
    attributes, rows, data = [], 0, False
    with path.open(encoding="utf-8-sig") as stream:
        for number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            if not data:
                match = re.match(r'''@attribute\s+(?:'([^']+)'|"([^"]+)"|(\S+))\s+''', line, re.I)
                if match:
                    attributes.append(next(value for value in match.groups() if value is not None))
                elif re.fullmatch(r"@data\s*(?:%.*)?", line, re.I):
                    data = True
            else:
                if line.startswith("{"):
                    raise ValueError(f"Sparse ARFF requires separate validation: {path}:{number}")
                if field_count(line) != len(attributes):
                    raise ValueError(f"ARFF width mismatch: {path}:{number}")
                rows += 1
    if not data or not rows or not attributes or len(attributes) != len(set(attributes)):
        raise ValueError(f"Invalid/empty ARFF: {path}")
    return rows, attributes


def ranks(values):
    result = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=values.__getitem__)
    i = 0
    while i < len(values):
        end = i + 1
        while end < len(values) and values[ordered[end]] == values[ordered[i]]:
            end += 1
        average = (i + 1 + end) / 2
        for index in ordered[i:end]:
            result[index] = average
        i = end
    return result


def spearman(x, y):
    return statistics.correlation(ranks(x), ranks(y))


def load_records():
    joined = []
    for task, (expected, methods) in TASKS.items():
        folder = ROOT / "experiments/data" / task / "train_test"
        files = defaultdict(list)
        for path in folder.glob("*.arff"):
            files[canonical(path.name)].append(path)
        source = ROOT / "experiments" / task / "results/comparision/exceptions_summary_ALL.csv"
        with source.open(encoding="utf-8", newline="") as stream:
            saved = list(csv.DictReader(stream))
        selected = [r for r in saved if r["model"] in methods]
        keyed = {(r["dataset"], r["model"]): r for r in selected}
        datasets = {r["dataset"] for r in selected}
        if len(datasets) != expected or len(selected) != expected * 4 or len(keyed) != len(selected):
            raise ValueError(f"Incomplete/duplicate saved comparison: {task}")
        for dataset in sorted(datasets):
            paths = files[canonical(dataset)]
            if len(paths) != 1:
                raise ValueError(f"ARFF join is not one-to-one: {task}/{dataset}: {paths}")
            path = paths[0]
            n, attributes = dimensions(path)
            exclude = {"class"} if task != "survival" else {"survival_time", "survival_status", "group"}
            required = {"class"} if task != "survival" else {"survival_time", "survival_status"}
            if not required <= set(attributes):
                raise ValueError(f"Missing outcome column: {path}")
            p = len([a for a in attributes if a not in exclude])
            values = [float(keyed[dataset, model]["raw_time"]) for model in methods]
            if not all(math.isfinite(v) and v > 0 for v in values):
                raise ValueError(f"Invalid fitting times: {task}/{dataset}")
            if task == "classification":
                duplicates = [r for r in saved if r["dataset"] == dataset and r["model"].startswith("algorithm_cer_with_exceptions_prediction_type_")]
                if len(duplicates) != 4 or {float(r["raw_time"]) for r in duplicates} != {values[-1]}:
                    raise ValueError(f"Prediction-variant times differ: {dataset}")
            joined.append(dict(task=task, dataset=dataset, arff=str(path.relative_to(ROOT)).replace("\\", "/"),
                               rows=n, columns=len(attributes), predictors=p,
                               train_rows=n - math.ceil(n * 0.3),
                               tree_seconds=values[0], rulekit_seconds=values[1], rules_seconds=values[2], exception_seconds=values[3]))
    # Independent dimension cross-check against the repository's classification inventory.
    with (ROOT / "experiments/classification/classification_datasets_summary.csv").open(encoding="utf-8") as stream:
        inventory = {r["dataset"]: (int(r["rows"]), int(r["columns"])) for r in csv.DictReader(stream)}
    for row in joined:
        if row["task"] == "classification" and inventory[row["dataset"]] != (row["rows"], row["columns"]):
            raise ValueError(f"Dimension inventory disagrees: {row['dataset']}")
    return joined


def rst_table(title, headers, rows, widths):
    from io import StringIO
    out = StringIO()
    csv.writer(out, lineterminator="\n").writerows(rows)
    header = ", ".join('"' + h + '"' for h in headers)
    return (f".. csv-table:: {title}\n   :header: {header}\n   :widths: {widths}\n\n"
            + "".join("   " + line + "\n" for line in out.getvalue().splitlines()) + "\n")


def escape(text):
    return text.replace("_", r"\_").replace("%", r"\%").replace("{", r"\{").replace("}", r"\}")


def tex_table(title, headers, rows, spec):
    header = " & ".join(headers) + " \\\\\n"
    return ("\\begingroup\n\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{"+spec+"}\n"
            "\\caption{"+title+"}\\\\\n\\toprule\n"+header+"\\midrule\n\\endfirsthead\n\\toprule\n"
            +header+"\\midrule\n\\endhead\n"+"".join(" & ".join(row)+" \\\\\n" for row in rows)
            +"\\bottomrule\n\\end{longtable}\n\\endgroup\n\n")


def write_analysis(joined):
    intro = """Saved wall-clock fitting times are joined to the ARFF datasets in
experiments/data/{classification,regression,survival}/train_test. Dataset names
are matched within each task after removing a leading numerical identifier
(used differently in the survival result files). Every result has exactly one
matching dataset: 49 classification, 48 regression, and 25 survival datasets.
Classification prediction variants are counted as one fit per dataset.

N is the full dataset's number of observations; C counts all declared ARFF
columns, including outcomes. P counts descriptive predictors before encoding.
For classification and regression, P excludes class. For survival, P excludes
survival_time, survival_status, and the optional group column, in accordance
with the induction pipeline. The timing scripts use training partitions from
a 70/30 split, so fitted row counts are N minus ceil(0.3 N). N and C describe
the input datasets, not an encoded tree design matrix. Dimensions are
computed from the current local ARFF files; correspondence to the original
benchmark file versions is not independently established.

Each row reports one saved fitting time per method in seconds. Tree denotes
the task-specific decision tree, Rules the learner without exception search,
and Exceptions the exception-enabled learner. Results are sorted by N, then P.
The joined data, including actual ARFF paths and implied training sizes, are
saved in docs/supplementary/data/runtime_by_dataset.csv. Reproduce the analysis
with python docs/supplementary/analyze_runtime_by_size.py --write.
"""
    rst = ".. _runtime-by-size:\n\nDataset size and induction time\n" + "-" * 31 + "\n\n" + intro + "\n"
    tex = "\\subsection{Dataset size and induction time}\n\\label{runtime-by-size}\n\n" + escape(intro) + "\n"
    corr_rows = []
    for task in TASKS:
        rows = [r for r in joined if r["task"] == task]
        for key, model in [("tree_seconds", "Tree"), ("rulekit_seconds", "RuleKit"), ("rules_seconds", "Rules"), ("exception_seconds", "Exceptions")]:
            corr_rows.append([NAMES[task], model, str(len(rows)),
                              f"{spearman([r['rows'] for r in rows], [r[key] for r in rows]):.3f}",
                              f"{spearman([r['predictors'] for r in rows], [r[key] for r in rows]):.3f}"])
    title = "Spearman rank correlations between saved fitting time and dataset dimensions."
    rst += rst_table(title, ["Task", "Method", "Datasets", "rho(time, N)", "rho(time, P)"], corr_rows, "25, 20, 15, 20, 20")
    tex += tex_table(title, ["Task", "Method", "Datasets", r"$\rho(t,N)$", r"$\rho(t,P)$"], corr_rows, "@{}llrrr@{}")
    exception_correlations = {row[0]: (row[3], row[4]) for row in corr_rows if row[1] == "Exceptions"}
    interpretation = f"""Spearman correlations use average ranks for ties and quantify separate
associations with N and P within each task and method. They do not adjust for
the other dimension, attribute cardinality, rule-search complexity, or censoring.
These are descriptive comparisons across different datasets, not a controlled
scaling experiment or an estimate of algorithmic complexity. The per-dataset
tables below retain the individual observations behind these summaries.

For the exception-enabled learner, the association with row count is strongest
in regression (rho = {exception_correlations['Regression'][0]}), followed by classification ({exception_correlations['Classification'][0]}) and survival
({exception_correlations['Survival analysis'][0]}). The respective associations with predictor count are {exception_correlations['Regression'][1]}, {exception_correlations['Classification'][1]},
and {exception_correlations['Survival analysis'][1]}. These separate, unadjusted correlations do not isolate the effect
of adding observations or attributes to an otherwise unchanged dataset.
"""
    rst += interpretation + "\n"
    tex += escape(interpretation) + "\n"
    for task in TASKS:
        heading = NAMES[task] + " datasets"
        rst += heading + "\n" + "~" * len(heading) + "\n\n"
        tex += "\\subsubsection{" + heading + "}\n"
        rows = sorted([r for r in joined if r["task"] == task], key=lambda r: (r["rows"], r["predictors"], r["dataset"]))
        data = [[re.sub(r"^\d+_", "", Path(r["dataset"]).stem), str(r["rows"]), str(r["columns"]), str(r["predictors"]),
                 *(f"{r[key]:.3f}" for key in ["tree_seconds", "rulekit_seconds", "rules_seconds", "exception_seconds"])] for r in rows]
        title = NAMES[task] + ": dataset dimensions and saved fitting times (seconds)."
        rst += rst_table(title, ["Dataset", "N", "C", "P", "Tree", "RuleKit", "Rules", "Exceptions"], data, "28, 10, 7, 7, 12, 12, 12, 12")
        latex_data = [[r"\nolinkurl{"+row[0]+"}", *row[1:]] for row in data]
        tex += tex_table(title, ["Dataset", "N", "C", "P", "Tree", "RuleKit", "Rules", "Exceptions"], latex_data,
                         r"@{}p{0.25\linewidth}rrrrrrr@{}")
    # Normalise RST title adornments to exact text length.
    lines = rst.splitlines()
    rst = "\n".join(line[0]*len(lines[i-1]) if i and line and len(set(line))==1 and line[0] in "-~" else line for i, line in enumerate(lines)) + "\n"
    (ROOT / "docs/source/supplementary/runtime_by_size.inc").write_text(rst, encoding="utf-8")
    (ROOT / "docs/supplementary/latex/sections/runtime_by_size.tex").write_text(tex, encoding="utf-8")
    target = ROOT / "docs/supplementary/data/runtime_by_dataset.csv"
    target.parent.mkdir(exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(joined[0]))
        writer.writeheader()
        writer.writerows(joined)
    return corr_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    records = load_records()
    print(f"Validated {len(records)} dataset joins and {len(records)*4} fitting times.")
    for task in TASKS:
        subset = [r for r in records if r["task"] == task]
        print(task, "N range", min(r["rows"] for r in subset), max(r["rows"] for r in subset),
              "P range", min(r["predictors"] for r in subset), max(r["predictors"] for r in subset))
    if args.write:
        for result in write_analysis(records):
            print(",".join(result))
