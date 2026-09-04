from types import SimpleNamespace

import matplotlib
import pandas as pd
import pytest

from exception_rules.classification.algorithm import ExceptionRulesClassifier
from exception_rules.decision_rules.conditions import CompoundCondition, NominalCondition


matplotlib.use("Agg")


def _rule(*conditions):
    return SimpleNamespace(
        premise=CompoundCondition(list(conditions)),
        reference_rule=None,
        exception_rule=None,
    )


@pytest.fixture
def fitted_core():
    core = ExceptionRulesClassifier(mincov=1, induction_measure="c2")
    X = pd.DataFrame({
        "age": [20, 30, 40, 50, 60],
        "colour": ["red", "blue", "red", "green", None],
        "city": ["A", "A", "B", "B", "C"],
    })
    core._prepare_training_data(X, pd.Series([0, 0, 1, 1, 1], name="class"))

    cr = _rule(NominalCondition(2, "A"))
    cr.reference_rule = _rule(NominalCondition(1, "red"))
    cr.exception_rule = _rule(NominalCondition(1, "blue"))
    return core, cr


def test_plots_all_numeric_and_nominal_attributes(fitted_core):
    core, cr = fitted_core
    figure, axes = core.plot_covered_examples_distributions(cr, show=False)

    assert len(axes) == 3
    assert axes[0].patches  # numeric histogram
    assert axes[1].patches  # nominal count bars
    assert [axis.get_title() for axis in axes] == ["age", "colour", "city"]
    figure.clear()


def test_can_limit_plot_to_exception_rule_attributes(fitted_core):
    core, cr = fitted_core
    figure, axes = core.plot_covered_examples_distributions(
        cr, all_attributes=False, show=False
    )

    assert len(axes) == 1
    assert axes[0].get_title() == "colour"
    figure.clear()


def test_compact_layout_uses_multiple_columns(fitted_core):
    core, cr = fitted_core
    figure, axes = core.plot_covered_examples_distributions(
        cr, compact=True, show=False
    )

    grid = axes[0].get_subplotspec().get_gridspec()
    assert grid.nrows == 1
    assert grid.ncols == 3
    assert [axis.get_title() for axis in axes] == ["age", "colour", "city"]
    figure.clear()


def test_pca_marks_rule_groups(fitted_core):
    core, cr = fitted_core
    figure, axis = core.plot_rule_pca(cr, show=False)

    assert axis.get_xlabel().startswith("PC1 (")
    assert {text.get_text().split(" ")[0] for text in axis.get_legend().texts} >= {
        "CR", "RR", "ER"
    }
    figure.clear()


def test_exception_separation_uses_lda_axis(fitted_core):
    core, cr = fitted_core
    figure, axis = core.plot_exception_separation(cr, show=False)

    assert axis.get_xlabel() == "LDA separation axis"
    assert {text.get_text().split(" ")[0] for text in axis.get_legend().texts} == {
        "Other", "ER"
    }
    figure.clear()


def test_additional_exception_visualizations(fitted_core):
    core, cr = fitted_core
    calls = [
        lambda: core.plot_exception_neighborhood(cr, all_attributes=True, show=False),
        lambda: core.plot_rule_parallel_coordinates(cr, all_attributes=True, show=False),
        lambda: core.plot_rule_boxplots(cr, all_attributes=True, show=False),
        lambda: core.plot_rule_heatmap(cr, all_attributes=True, show=False),
        lambda: core.plot_rule_coverage_matrix(cr, show=False),
        lambda: core.plot_rule_coverage_intersections(cr, show=False),
    ]

    for call in calls:
        figure, _ = call()
        figure.clear()


def test_boxplots_show_message_for_nominal_only_rule(fitted_core):
    core, cr = fitted_core

    figure, axes = core.plot_rule_boxplots(cr, show=False)

    assert len(axes) == 1
    assert axes[0].axison is False
    assert "only nominal attributes" in " ".join(
        text.get_text() for text in axes[0].texts
    )
    figure.clear()


def test_requires_complete_exception_rule_triple(fitted_core):
    core, cr = fitted_core
    cr.reference_rule = None

    with pytest.raises(ValueError, match="both an RR and an ER"):
        core.plot_covered_examples_distributions(cr, show=False)


def test_print_rules_can_show_only_rules_with_exceptions(fitted_core, capsys):
    core, cr = fitted_core
    rule_without_exception = _rule(NominalCondition(2, "C"))
    core.ruleset = SimpleNamespace(rules=[rule_without_exception, cr])

    core.print_rules(only_with_exceptions=True)

    output = capsys.readouterr().out
    assert "Rule 1:" not in output
    assert "Rule 2:" in output
    assert "Exception Rule 2:" in output
    assert "Reference Rule 2:" in output
