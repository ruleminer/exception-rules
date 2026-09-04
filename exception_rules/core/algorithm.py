"""Common contract and reusable mechanics for rule induction algorithms."""

from abc import ABC, abstractmethod
import copy
from typing import Any

import numpy as np
import pandas as pd

from exception_rules.decision_rules.conditions import ElementaryCondition, NominalCondition
from exception_rules.decision_rules.core.condition import AbstractCondition
from exception_rules.decision_rules.core.rule import AbstractRule
from exception_rules.decision_rules.core.ruleset import AbstractRuleSet


class ExceptionRulesBase(ABC):
    """Shared template for classification, regression and survival learners."""

    def __init__(
        self,
        mincov: int | float,
        max_growing: int | None = None,
        prune: bool = True,
        find_exceptions: bool = False,
        logger: Any = None,
    ) -> None:
        self.mincov = mincov
        self.max_growing = max_growing
        self.prune = prune
        self.find_exceptions = find_exceptions
        self.label_name = None
        self.X_numpy = None
        self.y_numpy = None
        self.conditions_coverage_cache: dict[int, np.ndarray] = {}
        self.logger = logger
        self.if_logging = logger is not None

    def _prepare_training_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        attributes_list: list[list[str]] | None = None,
    ) -> None:
        """Store input in all representations used during induction."""
        self.conditions_coverage_cache = {}
        self.label_name = y.name
        self.X_numpy = X.to_numpy()
        self.y_numpy = y.to_numpy()
        self.X_pandas = X
        self.y_pandas = y
        self.attributes_list = attributes_list
        self.nominal_attributes_indexes = self._get_nominal_indexes(X)
        self.numerical_attributes_indexes = self._get_numerical_indexes(X)
        self.columns_names = X.columns
        self.labels = y.unique()

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        attributes_list: list[list[str]] | None = None,
    ) -> AbstractRuleSet:
        """Fit the domain-specific rule model."""

    @abstractmethod
    def _rule_factory(self, *args: Any, **kwargs: Any) -> AbstractRule:
        """Create an empty rule appropriate for the target type."""

    @abstractmethod
    def _ruleset_factory(self, rules: list[AbstractRule]) -> AbstractRuleSet:
        """Create a ruleset appropriate for the target type."""

    @abstractmethod
    def _calculate_quality(
        self, rule: AbstractRule, X: np.ndarray, y: np.ndarray
    ) -> tuple[float, Any]:
        """Calculate domain-specific rule quality and coverage."""

    def _prune(self, rule: AbstractRule) -> None:
        """Remove conditions while rule quality does not decrease."""
        if len(rule.premise.subconditions) <= 1:
            return
        continue_pruning = True
        while continue_pruning:
            quality_best, _ = self._calculate_quality(rule, self.X_numpy, self.y_numpy)
            condition_to_remove = None
            for condition in rule.premise.subconditions:
                rule_without_condition = copy.deepcopy(rule)
                rule_without_condition.premise.subconditions.remove(condition)
                quality_without_condition, _ = self._calculate_quality(
                    rule_without_condition, self.X_numpy, self.y_numpy
                )
                if quality_without_condition >= quality_best:
                    quality_best = quality_without_condition
                    condition_to_remove = condition
            if condition_to_remove is None:
                continue_pruning = False
            else:
                rule.premise.subconditions.remove(condition_to_remove)
            if len(rule.premise.subconditions) <= 1:
                continue_pruning = False

    def print_rules(self, only_with_exceptions: bool = False) -> None:
        """Print the ruleset in a human-readable format.

        Parameters
        ----------
        only_with_exceptions:
            If true, omit commonsense rules which have no exception rule.
        """
        if not hasattr(self, "ruleset"):
            raise ValueError("No ruleset found. Fit the model first.")
        for i, rule in enumerate(self.ruleset.rules):
            if only_with_exceptions and rule.exception_rule is None:
                continue
            print(f"Rule {i + 1}: {rule}")
            if rule.exception_rule is not None:
                print(f"Exception Rule {i + 1}: {rule.exception_rule}")
                print(f"Reference Rule {i + 1}: {rule.reference_rule}")


    def get_covered_examples(
        self,
        X: np.ndarray | AbstractRule | None = None,
        y: np.ndarray | None = None,
        rule: AbstractRule | None = None,
    ) -> list[np.ndarray]:
        """Return feature and target rows covered by a rule premise.

        When ``X`` and ``y`` are omitted, the training data stored during
        :meth:`fit` are used.  Both the concise ``get_covered_examples(rule)``
        form and the original ``get_covered_examples(X, y, rule)`` form are
        supported.
        """
        if rule is None and X is not None and hasattr(X, "premise") and y is None:
            rule = X
            X = None
        if rule is None:
            raise ValueError("A rule must be provided.")
        if (X is None) != (y is None):
            raise ValueError("X and y must be provided together or both omitted.")
        if X is None:
            if self.X_numpy is None or self.y_numpy is None:
                raise ValueError("No training data found. Fit the model first.")
            X = self.X_numpy
            y = self.y_numpy
        covered_examples_mask = rule.premise.covered_mask(X)
        return [X[covered_examples_mask], y[covered_examples_mask]]

    def plot_covered_examples_distributions(
        self,
        rule: AbstractRule,
        *,
        all_attributes: bool = True,
        bins: int = 10,
        compact: bool = False,
        figsize: tuple[float, float] | None = None,
        show: bool = True,
    ) -> tuple[Any, np.ndarray]:
        """Plot attributes of examples covered by a CR, its RR and its ER.

        Numeric attributes are shown as overlaid histograms with common bin
        edges.  Nominal attributes are shown as grouped count bars.  When
        ``all_attributes`` is false, only attributes used by the exception
        rule are included.

        Parameters
        ----------
        rule:
            A commonsense rule (CR) with ``reference_rule`` (RR) and
            ``exception_rule`` (ER) assigned.
        all_attributes:
            Plot every training attribute.  If false, plot only attributes
            occurring in the ER premise.
        bins:
            Number of bins used for numeric attributes.
        compact:
            Arrange smaller plots in a multi-column grid instead of placing
            every attribute in a separate full-width row.
        figsize:
            Optional matplotlib figure size.  By default it is adjusted to
            the number of attributes and the selected layout.
        show:
            Call ``matplotlib.pyplot.show`` before returning.

        Returns
        -------
        (figure, axes):
            The matplotlib figure and a one-dimensional array of axes, which
            allows callers to further customise or save the visualisation.
        """
        if not hasattr(self, "X_pandas"):
            raise ValueError("No training data found. Fit the model first.")
        if rule.reference_rule is None or rule.exception_rule is None:
            raise ValueError("The selected CR must have both an RR and an ER.")
        if not isinstance(bins, int) or bins < 1:
            raise ValueError("bins must be a positive integer.")

        # Import lazily so using the induction algorithms does not initialise
        # a plotting backend unless a visualisation is explicitly requested.
        import matplotlib.pyplot as plt

        if all_attributes:
            attribute_indexes = list(range(self.X_pandas.shape[1]))
        else:
            attribute_indexes = sorted(rule.exception_rule.premise.attributes)
            if not attribute_indexes:
                raise ValueError("The ER does not contain any attributes to plot.")

        rule_groups = (
            ("CR", rule),
            ("RR", rule.reference_rule),
            ("ER", rule.exception_rule),
        )
        covered = {
            label: self.X_pandas.loc[
                current_rule.premise.covered_mask(self.X_numpy)
            ]
            for label, current_rule in rule_groups
        }

        if compact:
            columns_count = min(3, len(attribute_indexes))
            rows_count = int(np.ceil(len(attribute_indexes) / columns_count))
            if figsize is None:
                figsize = (4.5 * columns_count, 3.2 * rows_count)
        else:
            columns_count = 1
            rows_count = len(attribute_indexes)
            if figsize is None:
                figsize = (10, max(3.5, 3.5 * rows_count))
        figure, axes = plt.subplots(
            rows_count, columns_count, figsize=figsize, squeeze=False
        )
        axes = axes.ravel()
        colors = {"CR": "tab:blue", "RR": "tab:orange", "ER": "tab:red"}

        for axis, attribute_index in zip(axes, attribute_indexes):
            attribute = self.X_pandas.columns[attribute_index]
            if pd.api.types.is_numeric_dtype(self.X_pandas[attribute].dtype):
                values = [
                    frame[attribute].dropna().to_numpy(dtype=float)
                    for frame in covered.values()
                ]
                non_empty = [value for value in values if value.size]
                bin_edges = (
                    np.histogram_bin_edges(np.concatenate(non_empty), bins=bins)
                    if non_empty
                    else bins
                )
                for (label, _), value in zip(rule_groups, values):
                    axis.hist(
                        value,
                        bins=bin_edges,
                        alpha=0.4,
                        label=f"{label} (n={len(value)})",
                        color=colors[label],
                    )
                axis.set_ylabel("Count")
            else:
                category_values = [
                    frame[attribute].astype("string").fillna("<missing>")
                    for frame in covered.values()
                ]
                categories = list(dict.fromkeys(
                    value for series in category_values for value in series.tolist()
                ))
                positions = np.arange(len(categories), dtype=float)
                width = 0.25
                for offset, ((label, _), values) in enumerate(
                    zip(rule_groups, category_values)
                ):
                    counts = values.value_counts().reindex(categories, fill_value=0)
                    axis.bar(
                        positions + (offset - 1) * width,
                        counts.to_numpy(),
                        width=width,
                        label=f"{label} (n={len(values)})",
                        color=colors[label],
                    )
                axis.set_xticks(positions, categories, rotation=45, ha="right")
                axis.set_ylabel("Count")
            axis.set_title(str(attribute))
            axis.legend()

        # A rectangular compact grid can contain unused panels.
        for axis in axes[len(attribute_indexes):]:
            axis.set_visible(False)

        figure.suptitle("Attribute distributions for examples covered by CR, RR and ER")
        figure.tight_layout()
        if show:
            plt.show()
        return figure, axes

    def _rule_projection_data(
        self, rule: AbstractRule, all_attributes: bool
    ) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
        """Prepare mixed-type attributes and exclusive CR/RR/ER labels."""
        if not hasattr(self, "X_pandas"):
            raise ValueError("No training data found. Fit the model first.")
        if rule.reference_rule is None or rule.exception_rule is None:
            raise ValueError("The selected CR must have both an RR and an ER.")

        if all_attributes:
            indexes = list(range(self.X_pandas.shape[1]))
        else:
            indexes = sorted(rule.exception_rule.premise.attributes)
            if not indexes:
                raise ValueError("The ER does not contain any attributes to plot.")
        attributes = [str(self.X_pandas.columns[index]) for index in indexes]

        labels = np.full(len(self.X_pandas), "Uncovered", dtype=object)
        # Assignment is deliberately exclusive.  More specific rules override
        # broader ones, so overlapping points remain readable.
        labels[rule.premise.covered_mask(self.X_numpy)] = "CR"
        labels[rule.reference_rule.premise.covered_mask(self.X_numpy)] = "RR"
        labels[rule.exception_rule.premise.covered_mask(self.X_numpy)] = "ER"
        return self.X_pandas.iloc[:, indexes], labels, attributes

    @staticmethod
    def _encode_projection_attributes(data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        """Impute, scale and one-hot encode numeric and nominal columns."""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        numeric = data.select_dtypes(include=np.number).columns.tolist()
        nominal = [column for column in data.columns if column not in numeric]
        transformers = []
        if numeric:
            transformers.append((
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric,
            ))
        if nominal:
            transformers.append((
                "nominal",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                nominal,
            ))
        transformer = ColumnTransformer(transformers)
        encoded = transformer.fit_transform(data)
        return np.asarray(encoded, dtype=float), transformer.get_feature_names_out().tolist()

    def plot_rule_pca(
        self,
        rule: AbstractRule,
        *,
        all_attributes: bool = False,
        show_uncovered: bool = True,
        figsize: tuple[float, float] = (9, 7),
        show: bool = True,
    ) -> tuple[Any, Any]:
        """Show CR, RR and ER examples in a two-dimensional PCA projection."""
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA

        data, labels, _ = self._rule_projection_data(rule, all_attributes)
        encoded, _ = self._encode_projection_attributes(data)
        if min(encoded.shape) < 2:
            raise ValueError("PCA requires at least two examples and two encoded features.")

        pca = PCA(n_components=2)
        projection = pca.fit_transform(encoded)
        figure, axis = plt.subplots(figsize=figsize)
        styles = {
            "Uncovered": ("lightgray", "o", 24),
            "CR": ("tab:blue", "o", 42),
            "RR": ("tab:orange", "^", 50),
            "ER": ("tab:red", "*", 90),
        }
        masks = {
            "Uncovered": labels == "Uncovered",
            "CR": rule.premise.covered_mask(self.X_numpy),
            "RR": rule.reference_rule.premise.covered_mask(self.X_numpy),
            "ER": rule.exception_rule.premise.covered_mask(self.X_numpy),
        }
        for label, (color, marker, size) in styles.items():
            if label == "Uncovered" and not show_uncovered:
                continue
            mask = masks[label]
            if np.any(mask):
                axis.scatter(
                    projection[mask, 0], projection[mask, 1],
                    c=color, marker=marker, s=size, alpha=0.75,
                    edgecolors="white", linewidths=0.5,
                    label=f"{label} (n={np.count_nonzero(mask)})",
                )
        axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        axis.set_title("PCA projection of CR, RR and ER coverage")
        axis.legend()
        axis.grid(alpha=0.2)
        figure.tight_layout()
        if show:
            plt.show()
        return figure, axis

    def plot_exception_separation(
        self,
        rule: AbstractRule,
        *,
        all_attributes: bool = False,
        bins: int = 15,
        figsize: tuple[float, float] = (9, 5),
        show: bool = True,
    ) -> tuple[Any, Any]:
        """Plot a supervised LDA axis separating ER from other examples."""
        import matplotlib.pyplot as plt
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        if not isinstance(bins, int) or bins < 1:
            raise ValueError("bins must be a positive integer.")
        data, labels, _ = self._rule_projection_data(rule, all_attributes)
        encoded, _ = self._encode_projection_attributes(data)
        is_exception = labels == "ER"
        if np.unique(is_exception).size < 2:
            raise ValueError("Separation requires both ER and non-ER examples.")

        lda = LinearDiscriminantAnalysis(n_components=1)
        scores = lda.fit_transform(encoded, is_exception.astype(int)).ravel()
        figure, axis = plt.subplots(figsize=figsize)
        axis.hist(
            scores[~is_exception], bins=bins, density=True, alpha=0.5,
            color="tab:blue", label=f"Other (n={np.count_nonzero(~is_exception)})",
        )
        axis.hist(
            scores[is_exception], bins=bins, density=True, alpha=0.65,
            color="tab:red", label=f"ER (n={np.count_nonzero(is_exception)})",
        )
        axis.set_xlabel("LDA separation axis")
        axis.set_ylabel("Density")
        axis.set_title("Separation of exception examples")
        axis.legend()
        axis.grid(axis="y", alpha=0.2)
        figure.tight_layout()
        if show:
            plt.show()
        return figure, axis

    def _coverage_masks(self, rule: AbstractRule) -> dict[str, np.ndarray]:
        """Return independent coverage masks for a complete CR/RR/ER triple."""
        if not hasattr(self, "X_numpy") or self.X_numpy is None:
            raise ValueError("No training data found. Fit the model first.")
        if rule.reference_rule is None or rule.exception_rule is None:
            raise ValueError("The selected CR must have both an RR and an ER.")
        return {
            "CR": np.asarray(rule.premise.covered_mask(self.X_numpy), dtype=bool).copy(),
            "RR": np.asarray(
                rule.reference_rule.premise.covered_mask(self.X_numpy), dtype=bool
            ).copy(),
            "ER": np.asarray(
                rule.exception_rule.premise.covered_mask(self.X_numpy), dtype=bool
            ).copy(),
        }

    def plot_exception_neighborhood(
        self, rule: AbstractRule, *, all_attributes: bool = False,
        compact: bool = True, show: bool = True
    ) -> tuple[Any, np.ndarray]:
        """Compare ER with RR-without-ER and CR-without-ER using ECDF/count plots."""
        import matplotlib.pyplot as plt

        data = self._rule_data(rule, all_attributes)
        masks = self._coverage_masks(rule)
        groups = {
            "CR without ER": masks["CR"] & ~masks["ER"],
            "RR without ER": masks["RR"] & ~masks["ER"],
            "ER": masks["ER"],
        }
        count = data.shape[1]
        columns = min(3, count) if compact else 1
        rows = int(np.ceil(count / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 3.2 * rows), squeeze=False)
        axes = axes.ravel()
        colors = ["tab:blue", "tab:orange", "tab:red"]
        for axis, attribute in zip(axes, data.columns):
            if pd.api.types.is_numeric_dtype(data[attribute]):
                for (label, mask), color in zip(groups.items(), colors):
                    values = np.sort(data.loc[mask, attribute].dropna().to_numpy(float))
                    if values.size:
                        axis.step(values, np.arange(1, len(values) + 1) / len(values),
                                  where="post", label=f"{label} (n={len(values)})", color=color)
                axis.set_ylabel("Cumulative proportion")
            else:
                categories = data[attribute].astype("string").fillna("<missing>").unique().tolist()
                positions = np.arange(len(categories))
                for offset, ((label, mask), color) in enumerate(zip(groups.items(), colors)):
                    values = data.loc[mask, attribute].astype("string").fillna("<missing>")
                    proportions = values.value_counts(normalize=True).reindex(categories, fill_value=0)
                    axis.bar(positions + (offset - 1) * .25, proportions, width=.25,
                             label=f"{label} (n={len(values)})", color=color)
                axis.set_xticks(positions, categories, rotation=45, ha="right")
                axis.set_ylabel("Proportion")
            axis.set_title(str(attribute))
            axis.legend(fontsize="small")
        for axis in axes[count:]:
            axis.set_visible(False)
        figure.suptitle("Exception and its direct rule neighbourhood")
        figure.tight_layout()
        if show:
            plt.show()
        return figure, axes

    def _rule_data(self, rule: AbstractRule, all_attributes: bool) -> pd.DataFrame:
        """Select all attributes or only attributes occurring in the ER."""
        data, _, _ = self._rule_projection_data(rule, all_attributes)
        return data

    # Kept separate from projection preparation so plotting helpers have a
    # concise, intention-revealing call site.
    def _rule_data_for_plot(self, rule: AbstractRule, all_attributes: bool) -> pd.DataFrame:
        return self._rule_data(rule, all_attributes)

    def plot_rule_parallel_coordinates(
        self, rule: AbstractRule, *, all_attributes: bool = False,
        max_examples: int = 300, show: bool = True
    ) -> tuple[Any, Any]:
        """Plot standardised mixed-type profiles as parallel coordinates."""
        import matplotlib.pyplot as plt

        data = self._rule_data_for_plot(rule, all_attributes)
        encoded, feature_names = self._encode_projection_attributes(data)
        masks = self._coverage_masks(rule)
        labels = np.full(len(data), "Other", dtype=object)
        labels[masks["CR"]] = "CR"
        labels[masks["RR"]] = "RR"
        labels[masks["ER"]] = "ER"
        selected = np.flatnonzero(labels != "Other")
        if len(selected) > max_examples:
            selected = np.random.default_rng(0).choice(selected, max_examples, replace=False)
        figure, axis = plt.subplots(figsize=(max(9, len(feature_names) * .7), 6))
        colors = {"CR": "tab:blue", "RR": "tab:orange", "ER": "tab:red"}
        for label in ("CR", "RR", "ER"):
            indexes = selected[labels[selected] == label]
            for number, index in enumerate(indexes):
                axis.plot(encoded[index], color=colors[label], alpha=.18,
                          label=label if number == 0 else None)
        axis.set_xticks(np.arange(len(feature_names)), feature_names, rotation=60, ha="right")
        axis.set_ylabel("Transformed value")
        axis.set_title("Parallel profiles of CR, RR and ER examples")
        axis.legend()
        figure.tight_layout()
        if show:
            plt.show()
        return figure, axis

    def plot_rule_boxplots(
        self, rule: AbstractRule, *, all_attributes: bool = False,
        compact: bool = True, show: bool = True
    ) -> tuple[Any, np.ndarray]:
        """Compare numeric CR, RR and ER attributes with boxplots."""
        import matplotlib.pyplot as plt

        data = self._rule_data_for_plot(rule, all_attributes)
        numeric = data.select_dtypes(include=np.number).columns.tolist()
        if not numeric:
            figure, axis = plt.subplots(figsize=(8, 3.5))
            axis.set_axis_off()
            axis.text(
                .5,
                .58,
                "No numerical attributes to display",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color="dimgray",
                transform=axis.transAxes,
            )
            axis.text(
                .5,
                .38,
                "The selected exception rule contains only nominal attributes.\n"
                "Use plot_covered_examples_distributions() or "
                "plot_exception_neighborhood() to inspect their categories.",
                ha="center",
                va="center",
                fontsize=11,
                color="gray",
                linespacing=1.5,
                transform=axis.transAxes,
            )
            figure.tight_layout()
            if show:
                plt.show()
            return figure, np.asarray([axis])
        masks = self._coverage_masks(rule)
        columns = min(3, len(numeric)) if compact else 1
        rows = int(np.ceil(len(numeric) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 3.3 * rows), squeeze=False)
        axes = axes.ravel()
        for axis, attribute in zip(axes, numeric):
            values = [data.loc[masks[label], attribute].dropna().to_numpy(float)
                      for label in ("CR", "RR", "ER")]
            axis.boxplot(values, tick_labels=["CR", "RR", "ER"], patch_artist=True)
            axis.set_title(str(attribute))
            axis.set_ylabel("Value")
        for axis in axes[len(numeric):]:
            axis.set_visible(False)
        figure.suptitle("Numeric attribute distributions")
        figure.tight_layout()
        if show:
            plt.show()
        return figure, axes

    def plot_rule_heatmap(
        self, rule: AbstractRule, *, all_attributes: bool = False,
        max_examples: int = 300, show: bool = True
    ) -> tuple[Any, Any]:
        """Show transformed attribute profiles ordered by CR/RR/ER membership."""
        import matplotlib.pyplot as plt

        data = self._rule_data_for_plot(rule, all_attributes)
        encoded, feature_names = self._encode_projection_attributes(data)
        masks = self._coverage_masks(rule)
        labels = np.full(len(data), "Other", dtype=object)
        labels[masks["CR"]] = "CR"
        labels[masks["RR"]] = "RR"
        labels[masks["ER"]] = "ER"
        selected = np.flatnonzero(labels != "Other")
        order_value = {"CR": 0, "RR": 1, "ER": 2}
        selected = np.array(sorted(selected, key=lambda index: order_value[labels[index]]))
        if len(selected) > max_examples:
            selected = selected[np.linspace(0, len(selected) - 1, max_examples, dtype=int)]
        figure, axis = plt.subplots(figsize=(max(8, len(feature_names) * .55), 7))
        image = axis.imshow(encoded[selected], aspect="auto", cmap="coolwarm")
        axis.set_xticks(np.arange(len(feature_names)), feature_names, rotation=60, ha="right")
        axis.set_ylabel("Examples ordered CR → RR → ER")
        axis.set_title("Attribute-profile heatmap")
        figure.colorbar(image, ax=axis, label="Transformed value")
        figure.tight_layout()
        if show:
            plt.show()
        return figure, axis

    def plot_rule_coverage_matrix(
        self, rule: AbstractRule, *, show: bool = True
    ) -> tuple[Any, Any]:
        """Show the Boolean CR/RR/ER coverage pattern for every example."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        masks = self._coverage_masks(rule)
        matrix = np.column_stack([masks[label] for label in ("CR", "RR", "ER")]).astype(int)
        order = np.argsort(matrix @ np.array([1, 2, 4]))
        figure, axis = plt.subplots(figsize=(5, max(4, min(10, len(matrix) / 100))))
        axis.imshow(matrix[order], aspect="auto", interpolation="nearest",
                    cmap=ListedColormap(["white", "tab:blue"]), vmin=0, vmax=1)
        axis.set_xticks(range(3), ["CR", "RR", "ER"])
        axis.set_ylabel("Examples grouped by coverage pattern")
        axis.set_title("Rule coverage matrix")
        figure.tight_layout()
        if show:
            plt.show()
        return figure, axis

    def plot_rule_coverage_intersections(
        self, rule: AbstractRule, *, show: bool = True
    ) -> tuple[Any, np.ndarray]:
        """Show exact CR/RR/ER intersections in an UpSet-style chart."""
        import matplotlib.pyplot as plt

        masks = self._coverage_masks(rule)
        combinations = []
        for code in range(1, 8):
            membership = tuple(bool(code & (1 << index)) for index in range(3))
            exact = np.ones(len(self.X_numpy), dtype=bool)
            for present, label in zip(membership, ("CR", "RR", "ER")):
                exact &= masks[label] if present else ~masks[label]
            combinations.append((membership, int(np.count_nonzero(exact))))
        combinations.sort(key=lambda item: item[1], reverse=True)

        figure, (bars, matrix_axis) = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True,
            gridspec_kw={"height_ratios": [3, 1.3]}
        )
        positions = np.arange(len(combinations))
        counts = [count for _, count in combinations]
        bars.bar(positions, counts, color="tab:purple")
        bars.set_ylabel("Examples")
        bars.set_title("Exact CR/RR/ER coverage intersections")
        for position, count in zip(positions, counts):
            bars.text(position, count, str(count), ha="center", va="bottom", fontsize="small")
        for position, (membership, _) in zip(positions, combinations):
            active = [index for index, present in enumerate(membership) if present]
            matrix_axis.scatter([position] * 3, range(3), color="lightgray", s=35)
            matrix_axis.scatter([position] * len(active), active, color="black", s=45)
            if len(active) > 1:
                matrix_axis.plot([position, position], [min(active), max(active)], color="black")
        matrix_axis.set_yticks(range(3), ["CR", "RR", "ER"])
        matrix_axis.set_xticks(positions, [str(count) for count in counts])
        matrix_axis.set_xlabel("Intersection size")
        matrix_axis.invert_yaxis()
        figure.tight_layout()
        if show:
            plt.show()
        return figure, np.asarray([bars, matrix_axis])

    def _check_candidate(
        self, new_covered_examples: np.ndarray, uncovered: Any
    ) -> bool:
        """Return whether a candidate satisfies the shared coverage rule."""
        return (len(new_covered_examples) >= self.mincov) or (
            len(uncovered) <= self.mincov
        )

    def _growth_limit_reached(self, rule: AbstractRule) -> bool:
        """Return whether the configured premise-size limit was reached."""
        return self.max_growing is not None and (
            len(rule.premise.subconditions) >= self.max_growing
        )

    def _condition_coverage_mask(
        self, condition: AbstractCondition
    ) -> np.ndarray:
        """Return a condition mask, calculating it once per training run."""
        key = hash(condition)
        if key not in self.conditions_coverage_cache:
            self.conditions_coverage_cache[key] = condition.covered_mask(self.X_numpy)
        return self.conditions_coverage_cache[key]

    def _discard_rule_coverage(
        self,
        uncovered: set[int] | list[int],
        rule: AbstractRule,
    ) -> set[int] | list[int]:
        """Remove all examples classified as covered by a completed rule."""
        positive = np.flatnonzero(
            rule.positive_covered_mask(self.X_numpy, self.y_numpy)
        )
        negative = np.flatnonzero(
            rule.negative_covered_mask(self.X_numpy, self.y_numpy)
        )
        covered = set(positive).union(negative)
        remaining = [index for index in uncovered if index not in covered]
        return set(remaining) if isinstance(uncovered, set) else remaining

    def _grow_rule(
        self,
        rule: AbstractRule,
        X: np.ndarray,
        y: np.ndarray,
        uncovered: Any,
        *,
        truncate_to_best: bool = False,
        refresh_coverage_before_exceptions: bool = False,
    ) -> bool:
        """Run the shared greedy loop used by numeric-target algorithms."""
        qualities: list[float] = []
        iteration = 0
        carry_on = True
        self._log("*******GROWING CR RULE*******")

        while carry_on:
            current_coverage = None
            condition, quality, coverage = self._induce_condition(
                rule, X, y, uncovered
            )
            if condition is None:
                carry_on = False
            else:
                rule.premise.subconditions.append(condition)
                qualities.append(quality)
                if refresh_coverage_before_exceptions:
                    current_coverage = rule.calculate_coverage(X, y)
                if self.find_exceptions:
                    carry_on = not self._search_exceptions(rule, X, y)

            condition_text = (
                condition.to_string(self.columns_names)
                if condition is not None
                else "None"
            )
            self._log(
                f"Iteracja {iteration}: condition_best: {condition_text}, "
                f"quality_best: {round(quality, 3)}, coverage_best: {coverage}"
            )
            if self.if_logging:
                if current_coverage is None:
                    current_coverage = rule.calculate_coverage(X, y)
                self._log(
                    f"Regula po iteracji {iteration}: {rule}, "
                    f"{current_coverage}"
                )

            if self._growth_limit_reached(rule):
                carry_on = False
            iteration += 1

        self._log("*******STOP GROWING CR RULE*******")
        if not rule.premise.subconditions:
            return False
        if truncate_to_best:
            best_index = int(np.argmax(qualities))
            rule.premise.subconditions = rule.premise.subconditions[: best_index + 1]
        return True

    def _grow_reference_rule_common(
        self,
        rule: AbstractRule,
        X: np.ndarray,
        y: np.ndarray,
        uncovered: Any,
        commonsense_covered: Any,
        *,
        truncate_to_best: bool = False,
        refresh_coverage_after_growth: bool = False,
    ) -> bool:
        """Run the shared greedy loop for a reference rule."""
        qualities: list[float] = []
        best_score = 0
        iteration = 0
        carry_on = True
        self._log("*****GROWING RR RULE*****")

        while carry_on:
            condition, quality, coverage, best_score = self._induce_reference_rule(
                rule, X, y, best_score, uncovered, commonsense_covered
            )
            if condition is None:
                carry_on = False
            else:
                rule.premise.subconditions.append(condition)
                qualities.append(quality)

            condition_text = (
                condition.to_string(self.columns_names)
                if condition is not None
                else "None"
            )
            self._log(
                f"Iteracja {iteration}: condition_best: {condition_text}, "
                f"quality_best: {round(quality, 3)}, coverage_best: {coverage}, "
                f"p_value: {round(best_score, 3)}"
            )
            if self.if_logging:
                self._log(
                    f"Regula po iteracji {iteration}: {rule}, "
                    f"{rule.calculate_coverage(X, y)}"
                )

            if self._growth_limit_reached(rule):
                carry_on = False
            iteration += 1

        self.rule_qualities = qualities
        self.rule_covered_negatives = []
        if refresh_coverage_after_growth:
            rule.calculate_coverage(X, y)
        self._log("*****STOP GROWING RR RULE*****")
        if not rule.premise.subconditions:
            return False
        if truncate_to_best:
            best_index = int(np.argmax(qualities))
            rule.premise.subconditions = rule.premise.subconditions[: best_index + 1]
        return True

    def _log(self, message: str) -> None:
        """Write an induction trace when a logger was configured."""
        if self.if_logging:
            self.logger.info(message)

    def _get_possible_conditions(
        self, examples_covered_by_rule: np.ndarray, y: np.ndarray
    ) -> list[AbstractCondition]:
        """Generate nominal equalities and numeric midpoint conditions."""
        conditions: list[AbstractCondition] = []
        for index in self.nominal_attributes_indexes:
            column = examples_covered_by_rule[:, index]
            filtered_column = column[~pd.isnull(column)]
            conditions.extend(
                NominalCondition(column_index=index, value=value)
                for value in np.unique(filtered_column)
            )
        for index in self.numerical_attributes_indexes:
            column = examples_covered_by_rule[:, index].astype(float)
            values = np.sort(np.unique(column[~np.isnan(column)]))
            mid_points = [(left + right) / 2 for left, right in zip(values, values[1:])]
            conditions.extend(
                ElementaryCondition(
                    column_index=index,
                    left_closed=False,
                    right_closed=True,
                    left=float("-inf"),
                    right=point,
                )
                for point in mid_points
            )
            conditions.extend(
                ElementaryCondition(
                    column_index=index,
                    left_closed=True,
                    right_closed=False,
                    left=point,
                    right=float("inf"),
                )
                for point in mid_points
            )
        return conditions

    @staticmethod
    def _get_nominal_indexes(dataframe: pd.DataFrame) -> list[int]:
        """Return indices of features that cannot be treated as numeric."""
        return [
            index
            for index, dtype in enumerate(dataframe.dtypes)
            if not pd.api.types.is_numeric_dtype(dtype)
        ]

    @staticmethod
    def _get_numerical_indexes(dataframe: pd.DataFrame) -> list[int]:
        """Return indices of numeric features, independent of pandas version."""
        return [
            index
            for index, dtype in enumerate(dataframe.dtypes)
            if pd.api.types.is_numeric_dtype(dtype)
        ]
