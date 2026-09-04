"""Separate-and-conquer rule induction for numeric regression targets.

Rules predict the mean target of their covered examples.  During induction,
examples within one standard deviation of a candidate's local mean are treated
as positives for coverage-based quality measures.  Optional reference rules
identify intersections whose target distribution differs significantly from
both source rules.
"""

from exception_rules.decision_rules.core.coverage import Coverage as CoverageClass
from exception_rules.decision_rules.core.ruleset import AbstractRuleSet
from exception_rules.decision_rules.core.rule import AbstractRule
from exception_rules.decision_rules.core.condition import AbstractCondition
from exception_rules.core.algorithm import ExceptionRulesBase
from exception_rules.decision_rules.regression.ruleset import RegressionRuleSet
from exception_rules.decision_rules.regression.rule import RegressionRule, RegressionConclusion
from exception_rules.decision_rules.conditions import CompoundCondition, LogicOperators
from exception_rules.decision_rules.measures import *
from exception_rules.decision_rules.core.coverage import Coverage
import pandas as pd
import numpy as np
from scipy import stats

class ExceptionRulesRegressor(ExceptionRulesBase):
    """Induce regression rules and optional exception-rule triples.

    Parameters
    ----------
    mincov : int
        Minimum number of newly covered examples required for a condition.
    induction_measure : str
        Name of a quality function imported from ``decision_rules.measures``.
        The misspelling is retained for API compatibility.
    max_growing : int or None, default=None
        Maximum conditions per rule; ``None`` imposes no limit.
    prune : bool, default=True
        Compatibility option for pruning.  The current main regression fitting
        loop does not invoke pruning.
    find_exceptions : bool, default=False
        Search for reference rules and attach exception rules during growth.
    logger : logging.Logger or None, default=None
        Optional destination for detailed induction diagnostics.

    Attributes
    ----------
    ruleset : RegressionRuleSet
        Fitted rule set, available after :meth:`fit`.
    conditions_coverage_cache : dict
        Cached Boolean masks keyed by condition hashes.
    """

    def __init__(self, mincov: int = 5, induction_measure: str = "c2", max_growing: int = None, prune: bool = False, find_exceptions:bool = True, logger = None) -> None:
        """Initialize the regressor and its induction configuration."""

        super().__init__(mincov, max_growing, prune, find_exceptions, logger)
        self.measure_function = globals().get(induction_measure)
        

    def fit(self, X: pd.DataFrame, y: pd.Series, attributes_list: list[list[str]] = None) -> AbstractRuleSet:
        """Induce regression rules from a feature table and numeric target.

            Parameters
            ----------
            X : pandas.DataFrame
                Feature table. Object-typed columns are nominal and all other
                columns are treated as numeric.
            y : pandas.Series
                Numeric target aligned row-for-row with ``X``. Its name becomes
                the rule conclusion column name.
            attributes_list : list of list of str or None, default=None
                Optional attribute grouping metadata retained on the model.

            Returns
            -------
            ExceptionRulesRegressor
                Fitted estimator; rules are available as ``ruleset``.

            Notes
            -----
            Rules are grown and their covered examples removed until no
            candidate covers new examples.
        """
        self._prepare_training_data(X, y, attributes_list)
        self.columns_names = X.columns.to_list()

        ruleset = self._ruleset_factory(self._induce_rules())
        ruleset.column_names = self.columns_names
        ruleset.update(X, y, self.measure_function)
        self.ruleset = ruleset
        return self

    def _induce_rules(self) -> list[RegressionRule]:
        """Run separate-and-conquer induction over the training examples."""
        rules: list[RegressionRule] = []
        uncovered = set(range(len(self.y_numpy)))

        while uncovered:
            rule = self._rule_factory(
                self.columns_names,
                self.label_name,
                self.X_numpy,
                self.y_numpy,
            )
            if not self._grow(rule, self.X_numpy, self.y_numpy, uncovered):
                break

            remaining = self._discard_rule_coverage(uncovered, rule)
            if len(remaining) == len(uncovered):
                break
            rules.append(rule)
            uncovered = remaining

        return rules



    def _grow(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray, uncovered:list[int]) -> AbstractRule:
        """Greedily append admissible conditions to a regression rule.

        Returns true if at least one condition was induced.  Exception search,
        when enabled, may terminate growth early.
        """
        return self._grow_rule(
            rule,
            X,
            y,
            uncovered,
            refresh_coverage_before_exceptions=True,
        )
        

            
    def _search_exceptions(self, rule, X, y):
        """Grow a reference rule over examples outside ``rule``'s premise."""
        covered = np.flatnonzero(rule.premise.covered_mask(X))
        uncovered = np.flatnonzero(rule.premise.uncovered_mask(X))
        reference_rule = self._rule_factory(
            self.columns_names,
            self.label_name,
            self.X_numpy,
            self.y_numpy,
        )
        if not self._grow_reference_rule(
            reference_rule, X, y, set(uncovered), covered
        ):
            self._log("***RR NOT FOUND***")
            return False

        self._log("***RR FOUND***")
        self._log(f"Reference rule: {reference_rule}")
        self._update_exception_candidate(X, y, rule, reference_rule)
        return True
    
    def _update_exception_candidate(
        self, X, y, commonsense_rule, reference_rule
    ) -> None:
        """Construct and attach the intersection exception rule.

        The method mutates ``commonsense_rule`` by setting its reference and
        exception attributes, then updates all three rules together.
        """
        self._log("***CHECKING EXCEPTION***")
        exception_rule = self._rule_factory(
            self.columns_names,
            self.label_name,
            self.X_numpy,
            self.y_numpy,
        )
        exception_rule.premise.subconditions.extend(
            commonsense_rule.premise.subconditions
        )
        exception_rule.premise.subconditions.extend(
            reference_rule.premise.subconditions
        )
        exception_rule.calculate_coverage(X, y)

        commonsense_rule.reference_rule = reference_rule
        commonsense_rule.exception_rule = exception_rule
        triple_ruleset = RegressionRuleSet(
            rules=[commonsense_rule, exception_rule, reference_rule]
        )
        triple_ruleset.update(
            self.X_pandas, self.y_pandas, measure=self.measure_function
        )

    def _check_exception_candidate(self, X, y, cr_covered, rr_covered, er_covered) -> bool:
        """Validate distributional separation of a proposed rule triple.

        A candidate is accepted when the non-overlapping commonsense and
        reference groups are not significantly different, while their overlap
        differs from each group at the 0.05 level.
        """
        
        cr_filtered = cr_covered[~np.isin(cr_covered, er_covered)]
        rr_filtered = rr_covered[~np.isin(rr_covered, er_covered)]

        y_cr = y[cr_filtered]
        y_rr = y[rr_filtered]
        y_er = y[er_covered]

        cr_rr_p_value = self.calculate_p_value(y_cr, y_rr)
        er_rr_p_value = self.calculate_p_value(y_er, y_rr)
        er_cr_p_value = self.calculate_p_value(y_er, y_cr)


        return (
            cr_rr_p_value > 0.05
            and er_cr_p_value <= 0.05
            and er_rr_p_value <= 0.05
        )

       

    def _grow_reference_rule(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray,uncovered, cr_covered) -> AbstractRule:
        """Greedily grow a reference rule maximizing exception overlap."""
        return self._grow_reference_rule_common(
            rule,
            X,
            y,
            uncovered,
            cr_covered,
            refresh_coverage_after_growth=True,
        )
        

    def _induce_reference_rule(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray, best_score, uncovered, cr_covered) -> AbstractCondition:
        """Select the next condition for a regression reference rule.

            Candidates must place the intersection mean outside one standard
            deviation of both source means and satisfy pairwise rank tests.

            Returns
            -------
            tuple
                Condition, quality, coverage, and updated overlap score.
        """
        quality_best = float("-inf")
        coverage_best = CoverageClass(0, 0, 0, 0)
        condition_best = None
        rule_mask = rule.premise.covered_mask(X)
        possible_conditions = self._get_possible_conditions(X[rule_mask], y[rule_mask])

        for condition in (
            candidate
            for candidate in possible_conditions
            if candidate not in rule.premise.subconditions
        ):
            candidate_mask = np.logical_and(
                rule_mask, self._condition_coverage_mask(condition)
            )
            rr_covered = np.flatnonzero(candidate_mask)
            rr_uncovered = np.flatnonzero(np.logical_not(candidate_mask))
            er_covered = np.intersect1d(
                cr_covered, rr_covered, assume_unique=True
            )
            score = len(er_covered)
            if score <= best_score or not self._has_exception_distribution(
                y[cr_covered], y[rr_covered], y[er_covered]
            ):
                continue

            new_covered = uncovered.intersection(rr_covered)
            if not self._check_candidate(new_covered, rr_uncovered):
                continue
            if not self._check_exception_candidate(
                X, y, cr_covered, rr_covered, er_covered
            ):
                continue

            quality_best, coverage_best = self._calculate_quality_using_covered(
                X, y, candidate_mask
            )
            condition_best = condition
            best_score = score

        return condition_best, quality_best, coverage_best, best_score

    @staticmethod
    def _has_exception_distribution(y_cr, y_rr, y_er) -> bool:
        """Check whether the intersection mean lies outside both rule bands."""
        er_mean = np.mean(y_er)
        rr_mean, rr_std = ExceptionRulesRegressor._population_mean_std(y_rr)
        cr_mean, cr_std = ExceptionRulesRegressor._population_mean_std(y_cr)
        return (
            cr_mean - cr_std < rr_mean < cr_mean + cr_std
            and not cr_mean - cr_std <= er_mean <= cr_mean + cr_std
            and not rr_mean - rr_std <= er_mean <= rr_mean + rr_std
        )

    @staticmethod
    def _population_mean_std(values) -> tuple[float, float]:
        """Return the population statistics used by regression induction."""
        mean = np.mean(values)
        std = np.sqrt(np.mean(np.square(values)) - mean * mean)
        return mean, std
    
    @staticmethod
    def calculate_p_value(y1, y2):
        """Return the two-sided Mann-Whitney U-test p-value for two samples."""
        stat, p = stats.mannwhitneyu(y1, y2)
        return p
    
    def _induce_condition(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray, uncovered:list[int]) -> AbstractCondition:
        """Select the best condition satisfying minimum coverage.

            Ties favor the candidate covering more locally positive examples.
            Returns ``(condition, quality, coverage)``.
        """
        quality_best = float("-inf")
        coverage_best = CoverageClass(0, 0, 0, 0)
        condition_best = None
        rule_mask = rule.premise.covered_mask(X)
        possible_conditions = self._get_possible_conditions(X[rule_mask], y[rule_mask])

        for condition in (
            candidate
            for candidate in possible_conditions
            if candidate not in rule.premise.subconditions
        ):
            candidate_mask = np.logical_and(
                rule_mask, self._condition_coverage_mask(condition)
            )
            newly_covered = uncovered.intersection(np.flatnonzero(candidate_mask))
            quality, coverage = self._calculate_quality_using_covered(
                X, y, candidate_mask
            )
            is_better = quality > quality_best or (
                quality == quality_best and coverage.p > coverage_best.p
            )
            if is_better and self._check_candidate(newly_covered, uncovered):
                condition_best = condition
                quality_best = quality
                coverage_best = coverage

        return condition_best, quality_best, coverage_best
            
    def _calculate_quality_using_covered(self, X,y, covered_mask):
        """Calculate candidate quality from a precomputed coverage mask.

        Covered targets within one standard deviation of their mean define the
        positive interval; that interval is then evaluated over all targets to
        build the ``p``, ``n``, ``P``, and ``N`` coverage counts.
        """
        covered_y = y[covered_mask]
        y_mean = np.mean(covered_y)
        y_std = np.sqrt((np.sum(np.square(covered_y)) / covered_y.shape[0]) - (y_mean * y_mean))

        low = y_mean - y_std
        high = y_mean + y_std

        positive_mask = (y >= low) & (y <= high)
        covered_positive_mask = ((covered_y >= low) & (covered_y <= high))

        p = np.sum(covered_positive_mask)
        n = covered_y.shape[0] - p
        P = np.sum(positive_mask)
        N = X.shape[0] - P


        coverage = Coverage(p, n, P, N)
        quality: float = self.measure_function(coverage)
        return quality, coverage


    def _rule_factory(self, columns_names, label_name,  X, y) -> RegressionRule:
        """Create an empty regression rule concluding the global target mean."""
        rule = RegressionRule(
            premise=CompoundCondition(subconditions=[],
                                    logic_operator=LogicOperators.CONJUNCTION,),
            conclusion=RegressionConclusion(
                value=np.nan,
                column_name=label_name,
            ),
            column_names=columns_names)
        
        rule.calculate_coverage(X, y) 

        return rule
    
    def _calculate_quality(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray) -> float:
        """Return the configured quality and coverage of a complete rule."""
        coverage = rule.calculate_coverage(X=X, y=y)
        quality = self.measure_function(coverage)

        return quality, coverage
    
    def _ruleset_factory(self, rules: list[RegressionRule]) -> RegressionRuleSet:
        """Wrap induced regression rules in a rule-set object."""
        return RegressionRuleSet(rules=rules)
