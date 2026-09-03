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


class BaseRuleInductionAlgorithm(ABC):
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

    def _get_covered_examples(
        self, X: np.ndarray, y: np.ndarray, rule: AbstractRule
    ) -> list[np.ndarray]:
        """Return feature and target rows covered by a rule premise."""
        covered_examples_mask = rule.premise.covered_mask(X)
        return [X[covered_examples_mask], y[covered_examples_mask]]

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
