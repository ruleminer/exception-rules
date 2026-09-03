"""Separate-and-conquer rule induction for censored survival data.

Each induced rule concludes with a Kaplan-Meier estimator fitted to the
examples covered by its premise.  Candidate conditions are evaluated using a
log-rank comparison between covered and uncovered examples.  Optional
reference rules search for intersections with distinct survival curves.
"""

from exception_rules.decision_rules.core.coverage import Coverage as CoverageClass
from exception_rules.decision_rules.core.ruleset import AbstractRuleSet
from exception_rules.decision_rules.core.rule import AbstractRule
from exception_rules.decision_rules.core.condition import AbstractCondition
from exception_rules.core.algorithm import BaseRuleInductionAlgorithm
from exception_rules.decision_rules.survival.ruleset import SurvivalRuleSet
from exception_rules.decision_rules.survival.rule import SurvivalRule, SurvivalConclusion
from exception_rules.decision_rules.conditions import CompoundCondition, LogicOperators
from exception_rules.decision_rules.survival.kaplan_meier import KaplanMeierEstimator
import pandas as pd
import numpy as np
import copy



class MyRuleSurvival(BaseRuleInductionAlgorithm):
    """Induce survival rules and optional exception-rule triples.

    Parameters
    ----------
    mincov : int
        Minimum number of newly covered examples required for a condition.
    survival_time_attr : str
        Name of the column in ``X`` containing event or censoring times.  This
        column is excluded from generated premise conditions.
    cuts_only_between_classes : bool, default=True
        Compatibility option retained from classification induction.  Numeric
        survival candidates currently use all adjacent distinct midpoints.
    max_growing : int or None, default=None
        Maximum number of conditions per rule; ``None`` imposes no limit.
    prune : bool, default=True
        Whether to remove conditions that do not improve log-rank quality.
    find_exceptions : bool, default=False
        Whether to search for statistically distinct exception intersections.
    delete_cr_n : bool, default=False
        Stored compatibility option used by downstream experimental code.
    logger : logging.Logger or None, default=None
        Optional destination for detailed induction diagnostics.

    Attributes
    ----------
    ruleset : SurvivalRuleSet
        Fitted rule set, available after :meth:`fit`.
    survival_time : numpy.ndarray
        Training times extracted from ``survival_time_attr``.
    survival_status : numpy.ndarray
        Event indicators supplied as ``y``.
    """

    def __init__(self, mincov: int, survival_time_attr: str, cuts_only_between_classes: bool = True, max_growing: int = None, prune: bool = True, find_exceptions:bool = False, delete_cr_n = False, logger = None) -> None:
        """Initialize the survival learner and induction configuration."""
        super().__init__(mincov, max_growing, prune, find_exceptions, logger)
        self.cuts_only_between_classes = cuts_only_between_classes
        self.survival_time_attr = survival_time_attr
        self.delete_cr_n = delete_cr_n

        

    def fit(self, X: pd.DataFrame, y: pd.Series, attributes_list: list[list[str]] = None) -> AbstractRuleSet:
        """Induce rules from survival times and event indicators.

            Parameters
            ----------
            X : pandas.DataFrame
                Feature table containing ``survival_time_attr``. Object-typed
                predictors are nominal; other predictors are numeric.
            y : pandas.Series
                Event-status indicators aligned with ``X``. Coding must follow
                the conventions of ``KaplanMeierEstimator``.
            attributes_list : list of list of str or None, default=None
                Optional attribute grouping metadata retained on the model.

            Returns
            -------
            MyRuleSurvival
                Fitted estimator; rules are available as ``ruleset``.

            Notes
            -----
            The time column remains in the rule matrix but is excluded from
            attributes eligible for premise conditions.
        """
        self._prepare_training_data(X, y, attributes_list)
        self.columns_names = X.columns.to_list()
        self.survival_time = X[self.survival_time_attr].to_numpy()
        self.survival_status = self.y_numpy
        self._exclude_survival_time_from_conditions(X)

        ruleset = self._ruleset_factory(self._induce_rules())
        ruleset.column_names = self.columns_names
        ruleset.update(X, y)
        self.ruleset = ruleset
        return self

    def _exclude_survival_time_from_conditions(self, X: pd.DataFrame) -> None:
        """Prevent the outcome-time column from becoming a premise condition."""
        time_index = X.columns.get_loc(self.survival_time_attr)
        for indexes in (
            self.nominal_attributes_indexes,
            self.numerical_attributes_indexes,
        ):
            if time_index in indexes:
                indexes.remove(time_index)
                break

    def _induce_rules(self) -> list[SurvivalRule]:
        """Run separate-and-conquer induction over survival examples."""
        rules: list[SurvivalRule] = []
        uncovered = list(range(len(self.y_numpy)))

        while uncovered:
            rule = self._rule_factory(self.columns_names, self.label_name)
            if not self._grow(rule, self.X_numpy, self.y_numpy, uncovered):
                break
            if self.prune:
                self._prune(rule)

            remaining = self._discard_rule_coverage(uncovered, rule)
            if len(remaining) == len(uncovered):
                break
            rules.append(self._update_estimator(rule))
            uncovered = remaining

        return rules

    def _update_estimator(self, rule: SurvivalRule) -> SurvivalRule:
        """Fit a Kaplan-Meier conclusion to examples covered by ``rule``."""
        
        covered_examples = rule.premise._calculate_covered_mask(self.X_numpy)
        km = KaplanMeierEstimator()
        km.fit(self.survival_time[covered_examples], self.survival_status[covered_examples])
        rule.conclusion.value = km
        rule.measure, rule.coverage = self._calculate_quality(rule, self.X_numpy, self.y_numpy)
        return rule
    


    def _grow(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray, uncovered:list[int]) -> AbstractRule:
        """Greedily append conditions to a survival rule.

        The final premise is truncated after its highest-quality condition.
        Returns true when at least one condition is induced.
        """
        return self._grow_rule(rule, X, y, uncovered, truncate_to_best=True)
        
    def _search_exceptions(self, rule, X, y):
        """Search outside a commonsense rule for a compatible reference rule."""
        positive_mask = rule.positive_covered_mask(X, y)
        cr_covered = np.flatnonzero(positive_mask)
        cr_uncovered = np.flatnonzero(np.logical_not(positive_mask))
        reference_rule = self._rule_factory(self.columns_names, self.label_name)
        if not self._grow_reference_rule(
            reference_rule, X, y, cr_uncovered, cr_covered
        ):
            self._log("***RR NOT FOUND***")
            return False

        self._log("***RR FOUND***")
        self._log(f"Reference rule: {reference_rule}")
        return self._check_exception_candidate(X, y, rule, reference_rule)

    def _check_exception_candidate(
        self, X, y, commonsense_rule, reference_rule
    ) -> bool:
        """Validate and attach a statistically distinct survival exception."""
        self._log("***CHECKING EXCEPTION***")
        exception_rule = self._rule_factory(self.columns_names, self.label_name)
        exception_rule.premise.subconditions.extend(
            commonsense_rule.premise.subconditions
        )
        exception_rule.premise.subconditions.extend(
            reference_rule.premise.subconditions
        )

        exception_estimator = self._estimator_for_rule(exception_rule, X, y)
        commonsense_estimator = self._estimator_for_rule(commonsense_rule, X, y)
        reference_estimator = self._estimator_for_rule(reference_rule, X, y)
        commonsense_p_value = self._comparison_p_value(
            commonsense_estimator, exception_estimator
        )
        reference_p_value = self._comparison_p_value(
            reference_estimator, exception_estimator
        )
        self._log(f"ER vs CR p_value: {commonsense_p_value}")
        self._log(f"ER vs RR p_value: {reference_p_value}")

        if commonsense_p_value > 0.05 or reference_p_value > 0.05:
            self._log("***ER NOT FOUND***")
            return False

        commonsense_rule.reference_rule = reference_rule
        commonsense_rule.exception_rule = exception_rule
        triple_ruleset = SurvivalRuleSet(
            rules=[commonsense_rule, exception_rule, reference_rule],
            survival_time_attr=self.survival_time_attr,
        )
        triple_ruleset.update(self.X_pandas, self.y_pandas)
        self._log("***ER FOUND***")
        self._log(f"Exception rule: {exception_rule}")
        return True

    def _estimator_for_rule(self, rule, X, y) -> KaplanMeierEstimator:
        """Fit a temporary estimator to the examples covered by ``rule``."""
        covered = np.flatnonzero(rule.positive_covered_mask(X, y))
        return KaplanMeierEstimator().fit(
            self.survival_time[covered],
            self.survival_status[covered],
            update_additional_informations=False,
        )

    @staticmethod
    def _comparison_p_value(first, second) -> float:
        """Return the p-value comparing two Kaplan-Meier estimators."""
        return KaplanMeierEstimator().compare_estimators(first, second)["p_value"]



        
    def _grow_reference_rule(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray,uncovered: list[int], cr_covered) -> AbstractRule:
        """Grow and quality-truncate a survival reference rule."""
        return self._grow_reference_rule_common(
            rule, X, y, uncovered, cr_covered, truncate_to_best=True
        )
        
    def _induce_reference_rule(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray, best_score, uncovered, cr_covered) -> AbstractCondition:
            """Select the next condition for a survival reference rule.

            A candidate must keep the source curves statistically similar while
            their intersection differs from each curve.

            Returns
            -------
            tuple
                Condition, quality, coverage, and updated p-value score.
            """
            
            quality_best = float("-inf")
            coverage_best = CoverageClass(0,0,0,0)
            condition_best = None
            number_of_covered_negatives_best = 0
            examples_covered_by_rule, y_for_examples_covered_by_rule = self._get_covered_examples(X,y,rule)

            scores = []
            negative_numbers = []
            positives_numbers = []

            possible_conditions = self._get_possible_conditions(examples_covered_by_rule, y_for_examples_covered_by_rule)
            possible_conditions_filtered = list(filter(lambda i: i not in rule.premise.subconditions, possible_conditions))
            if len(possible_conditions_filtered) != 0:
                for condition in possible_conditions_filtered:
                    rule_with_condition = copy.deepcopy(rule)
                    rule_with_condition.premise.subconditions.append(condition)

                    rr_covered = np.where(rule_with_condition.positive_covered_mask(X, y) == 1)[0]
                    rr_uncovered = np.where(rule_with_condition.positive_covered_mask(X, y) == 0)[0]

                    new_covered_examples = [i for i in uncovered if i in rr_covered]

                    
                    quality, coverage = self._calculate_quality(rule_with_condition, X, y)


                    cr_estimator  = KaplanMeierEstimator().fit(self.survival_time[cr_covered], self.survival_status[cr_covered], update_additional_informations=False)
                    rr_estimator  = KaplanMeierEstimator().fit(self.survival_time[rr_covered], self.survival_status[rr_covered], update_additional_informations=False)

                    
                    er_covered = [i for i in range(len(self.y_numpy)) if i in cr_covered and i in rr_covered]
                    er_estimator  = KaplanMeierEstimator().fit(self.survival_time[er_covered], self.survival_status[er_covered], update_additional_informations=False)

                    stats_and_pvalue_cr_rr = KaplanMeierEstimator().compare_estimators(
                        cr_estimator, rr_estimator)
                    
                    stats_and_pvalue_cr_er = KaplanMeierEstimator().compare_estimators(
                        cr_estimator, er_estimator)
                    
                    stats_and_pvalue_rr_er = KaplanMeierEstimator().compare_estimators(
                        rr_estimator, er_estimator)


                    score = stats_and_pvalue_cr_rr["p_value"]


                    scores.append(score)

                    if (score > best_score and score > 0.05) and len(er_covered) > 0 and stats_and_pvalue_cr_er["p_value"] <= 0.05 and stats_and_pvalue_rr_er["p_value"] <= 0.05:
                        if self._check_candidate(new_covered_examples, rr_uncovered):
                            condition_best = condition
                            quality_best = quality
                            coverage_best = coverage

                            best_score = score
            

                            
            return condition_best, quality_best, coverage_best, best_score  
    
            
    def _induce_condition(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray, uncovered:list[int]) -> AbstractCondition:
            """Select the best log-rank condition satisfying coverage.

            Returns ``(condition, quality, coverage)`` and favors greater
            covered population when qualities tie.
            """
            quality_best = float("-inf")
            coverage_best = CoverageClass(0,0,0,0)
            condition_best = None
            examples_covered_by_rule, y_for_examples_covered_by_rule = self._get_covered_examples(X,y,rule)

            possible_conditions = self._get_possible_conditions(examples_covered_by_rule, y_for_examples_covered_by_rule)
            possible_conditions_filtered = list(filter(lambda i: i not in rule.premise.subconditions, possible_conditions))
            if len(possible_conditions_filtered) != 0:
                for condition in possible_conditions_filtered:
                    rule_with_condition = copy.deepcopy(rule)
                    rule_with_condition.premise.subconditions.append(condition)

                    covered_examples = np.where(rule_with_condition.positive_covered_mask(self.X_numpy, self.y_numpy) == 1)[0]
                    new_covered_examples = [i for i in uncovered if i in covered_examples]
                    
                    quality, coverage = self._calculate_quality(rule_with_condition, X, y)
                    
                    if (quality > quality_best or ((quality == quality_best) and (coverage.p > coverage_best.p))):
                            if self._check_candidate(new_covered_examples, uncovered):
                                condition_best = condition
                                quality_best = quality
                                coverage_best = coverage

            return condition_best, quality_best, coverage_best     
            
    
    def _rule_factory(self, columns_names, label_name) -> SurvivalRule:
        """Create an empty survival rule with an uninitialized estimator."""
        return SurvivalRule(
            premise=CompoundCondition(subconditions=[],
                                    logic_operator=LogicOperators.CONJUNCTION,),
            conclusion=SurvivalConclusion(
                value=np.nan,
                column_name=label_name,
            ),
            column_names=columns_names,
            survival_time_attr=self.survival_time_attr,)
    
    def _calculate_quality(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray) -> float:
        """Compare covered and uncovered survival curves with a log-rank test.

        Returns a pair containing log-rank quality and a coverage object whose
        positive counts represent covered examples.
        """
        covered_examples_indexes = np.where(rule.premise._calculate_covered_mask(X))[0]
        uncovered_examples_indexes = np.where(rule.premise._calculate_uncovered_mask(X))[0]
        quality = KaplanMeierEstimator.log_rank(self.survival_time, self.survival_status, covered_examples_indexes, uncovered_examples_indexes)
        coverage = CoverageClass(p=len(covered_examples_indexes), n= 0, P=X.shape[0], N=0)
        return quality, coverage

    
    def _ruleset_factory(self, rules: list[SurvivalRule]) -> SurvivalRuleSet:
        """Wrap induced rules in a survival rule-set object."""
        return SurvivalRuleSet(rules=rules, survival_time_attr=self.survival_time_attr)
