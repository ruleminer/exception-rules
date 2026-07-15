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
from exception_rules.decision_rules.regression.ruleset import RegressionRuleSet
from exception_rules.decision_rules.regression.rule import RegressionRule, RegressionConclusion
from exception_rules.decision_rules.conditions import CompoundCondition, LogicOperators, NominalCondition, ElementaryCondition
from exception_rules.decision_rules.measures import *
from exception_rules.decision_rules.core.coverage import Coverage
from typing import List
import pandas as pd
import numpy as np
import copy
import warnings
warnings.filterwarnings('ignore')    

import logging
from scipy import stats

class MyRuleRegressor():
    """Induce regression rules and optional exception-rule triples.

    Parameters
    ----------
    mincov : int
        Minimum number of newly covered examples required for a condition.
    induction_measuer : str
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

    def __init__(self, mincov: int, induction_measuer: str, max_growing: int = None, prune: bool = True, find_exceptions:bool = False, logger = None) -> None:
        """Initialize the regressor and its induction configuration."""

        self.mincov = mincov
        self.measure_function = globals().get(induction_measuer)
        self.max_growing = max_growing
        self.prune = prune
        self.find_exceptions = find_exceptions

        self.label_name = None
        self.X_numpy = None
        self.y_numpy = None

        if logger is not None:
            self.if_logging = True
            self.logger = logger
        else:
            self.if_logging = False
                
                
        self.conditions_coverage_cache: dict[AbstractCondition, np.ndarray] = {}
        

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
            MyRuleRegressor
                Fitted estimator; rules are available as ``ruleset``.

            Notes
            -----
            Rules are grown and their covered examples removed until no
            candidate covers new examples.
            """

            self.label_name = y.name 

            self.X_numpy = X.to_numpy()
            self.y_numpy = y.to_numpy()

            self.X_pandas = X
            self.y_pandas = y

            self.attributes_list = attributes_list
            
            # get indexes of nominal attributes
            self.nominal_attributes_indexes = self._get_nominal_indexes(X)
            self.numerical_attributes_indexes = self._get_numerical_indexes(X)
            self.columns_names = X.columns.to_list()


            rules = []


            carry_on = True

            uncovered = set([i for i in range(len(self.y_numpy))])

            while carry_on:
                rule = self._rule_factory(self.columns_names, self.label_name, self.X_numpy, self.y_numpy)
                carry_on = self._grow(rule, self.X_numpy, self.y_numpy, uncovered)

                if (carry_on):
                    previously_uncovered = len(uncovered)
                    
                    positive_covered_indices = np.where(rule.positive_covered_mask(self.X_numpy, self.y_numpy) == 1)[0]
                    negative_covered_indices = np.where(rule.negative_covered_mask(self.X_numpy, self.y_numpy) == 1)[0]
                    
                    
                    uncovered = set([i for i in uncovered if i not in positive_covered_indices and i not in negative_covered_indices])

                    if (len(uncovered) == previously_uncovered):
                        carry_on = False
                    else:
                        rules.append(rule)



            ruleset = self._ruleset_factory(rules)

            ruleset.column_names = self.columns_names
            ruleset.update(X,y, self.measure_function)
            self.ruleset = ruleset
            return self



    def _grow(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray, uncovered:list[int]) -> AbstractRule:
        """Greedily append admissible conditions to a regression rule.

        Returns true if at least one condition was induced.  Exception search,
        when enabled, may terminate growth early.
        """
        carry_on = True
        rule_qualities = []
        if self.if_logging:
            self.logger.info("*******GROWING CR RULE*******")
        i = 0
        while(carry_on):

            condition_best, quality_best, coverage_best = self._induce_condition(rule, X, y, uncovered)
            
            if condition_best is not None:
                rule.premise.subconditions.append(condition_best) # add the best condition to the rule
                rule_qualities.append(quality_best)
                rule_coverage = rule.calculate_coverage(X,y)
                
                if self.if_logging:
                    self.logger.info(f"Iteracja {i}: condition_best: {condition_best.to_string(self.columns_names)}, quality_best: {round(quality_best,3)}, coverage_best: {str(coverage_best)}")
                    self.logger.info(f"Regula po iteracji {i}: {rule}, {rule_coverage}")

                if self.find_exceptions:
                    carry_on = not self._search_exceptions(rule, X, y)
            else:
                carry_on = False
                rule_coverage = rule.calculate_coverage(X,y)
                if self.if_logging:
                    self.logger.info(f"Iteracja {i}: condition_best: None, quality_best: {round(quality_best,3)}, coverage_best: {str(coverage_best)}")
                    self.logger.info(f"Regula po iteracji {i}: {rule}, {rule_coverage}")


            if (self.max_growing is not None) and (len(rule.premise.subconditions) >= self.max_growing):
                carry_on = False


            i +=1 
        if self.if_logging:
            self.logger.info("*******STOP GROWING CR RULE*******")

        if len(rule.premise.subconditions) > 0:
            return True
        else:
            return False
        

            
    def _search_exceptions(self, rule, X, y):
        """Grow a reference rule over examples outside ``rule``'s premise."""


        cr_covered = np.where(rule.premise._calculate_covered_mask(X) == 1)[0]
        cr_uncovered = np.where(rule.premise._calculate_uncovered_mask(X) == 1)[0]


        reference_rule = self._rule_factory(self.columns_names, self.label_name, self.X_numpy, self.y_numpy)
        # X_tmp = X[cr_uncovered]
        # y_tmp = y[cr_uncovered]

        # uncovered = [i for i in range(len(y_tmp))]

        found_reference_rule = self._grow_reference_rule(reference_rule, X, y, set(cr_uncovered), cr_covered)


        if found_reference_rule:
            if self.if_logging:
                self.logger.info("***RR FOUND***")
                self.logger.info(f"Reference rule: {reference_rule}")
            self._update_exception_candidate(X, y, rule, reference_rule)
            return True
        else:
            if self.if_logging:
                self.logger.info("***RR NOT FOUND***")
            return False
    
    def _update_exception_candidate(self, X, y, comonsense_rules, reference_rule) -> bool:
            """Construct and attach the intersection exception rule.

            The method mutates ``comonsense_rules`` by setting its reference and
            exception attributes, then updates all three rules together.
            """
           
            if self.if_logging:
                self.logger.info("***CHECKING EXCEPTION***")
           
            exception_rule = self._rule_factory(self.columns_names, self.label_name, self.X_numpy, self.y_numpy)
            exception_rule.premise.subconditions.extend(comonsense_rules.premise.subconditions)
            exception_rule.premise.subconditions.extend(reference_rule.premise.subconditions)
 
            exception_rule.calculate_coverage(X, y)
 
            comonsense_rules.reference_rule = reference_rule  
            comonsense_rules.exception_rule = exception_rule
            triple_ruleset = RegressionRuleSet(rules = [comonsense_rules,exception_rule,reference_rule])
            triple_ruleset.update(self.X_pandas,self.y_pandas, measure=self.measure_function)

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


        if cr_rr_p_value > 0.05 and er_cr_p_value <= 0.05 and er_rr_p_value <= 0.05:
            return True
        else:
            return False

       

    def _grow_reference_rule(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray,uncovered, cr_covered) -> AbstractRule:
        """Greedily grow a reference rule maximizing exception overlap."""
        carry_on = True
        rule_qualities = []
        rule_covered_negatives = []
        scores = []
        best_score = 0
        if self.if_logging:
            self.logger.info("*****GROWING RR RULE*****")

        i = 0

        while(carry_on):
            condition_best, quality_best, coverage_best, best_score = self._induce_reference_rule(rule, X, y, best_score, uncovered, cr_covered)
            
            if condition_best is not None:
                rule.premise.subconditions.append(condition_best) # add the best condition to the rule
                rule_qualities.append(quality_best)
                scores.append(best_score)

            else:
                carry_on = False

            if (self.max_growing is not None) and (len(rule.premise.subconditions) >= self.max_growing):
                carry_on = False

            if self.if_logging:
                if condition_best is not None:
                    self.logger.info(f"Iteracja {i}: condition_best: {condition_best.to_string(self.columns_names)}, quality_best: {round(quality_best, 3)}, coverage_best: {str(coverage_best)}, p_value: {round(best_score,3)}")
                else:
                    self.logger.info(f"Iteracja {i}: condition_best: None, quality_best: {round(quality_best, 3)}, coverage_best: {str(coverage_best)}, p_value: {round(best_score,3)}")
                self.logger.info(f"Regula po iteracji {i}: {rule}, {rule.calculate_coverage(X,y)}")
            
            i +=1

        rule_coverage = rule.calculate_coverage(X,y)

        self.rule_qualities = rule_qualities
        self.rule_covered_negatives = rule_covered_negatives

        if self.if_logging:
            self.logger.info("*****STOP GROWING RR RULE*****")
        
        if len(rule.premise.subconditions) > 0:
            return True
        else:
            return False
        

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
            coverage_best = CoverageClass(0,0,0,0)
            condition_best = None
            number_of_covered_negatives_best = 0
            # examples_covered_by_rule, y_for_examples_covered_by_rule = self._get_covered_examples(X,y,rule)

            scores = []
            negative_numbers = []
            positives_numbers = []

            rule_covered_mask: np.ndarray = rule.premise.covered_mask(X)
            examples_covered_by_rule = X[rule_covered_mask]
            y_for_examples_covered_by_rule = y[rule_covered_mask]

            possible_conditions = self._get_possible_conditions(examples_covered_by_rule, y_for_examples_covered_by_rule)
            possible_conditions_filtered = list(filter(lambda i: i not in rule.premise.subconditions, possible_conditions))
            if len(possible_conditions_filtered) != 0:
                for condition in possible_conditions_filtered:

                    candidate = False
                           
                    condition_str = condition.__hash__()
                    if condition_str in self.conditions_coverage_cache:
                        condition_coverage_mask = self.conditions_coverage_cache[condition_str]
                    else:
                        condition_coverage_mask = condition.covered_mask(self.X_numpy)
                        self.conditions_coverage_cache[condition_str] = condition_coverage_mask

                    rule_with_condition_covered_mask = np.logical_and(
                                                                    rule_covered_mask,
                                                                    condition_coverage_mask
                                                                    )
                  
                    # coverage = rule_with_condition.calculate_coverage(X=X, y=y)
                    # quality = self.measure_function(coverage)

                    ### TODO Do zastanowienia czy tu powinny być brane wszystkie pokrywane czy tylko pozytywne

                    rr_covered = np.where(rule_with_condition_covered_mask == 1)[0]
                    rr_uncovered = np.where(rule_with_condition_covered_mask == 0)[0]
                    er_covered = np.intersect1d(cr_covered, rr_covered, assume_unique=True)#[i for i in range(len(self.y_numpy)) if i in cr_covered and i in rr_covered]

                    covered_examples = set(rr_covered)
                    new_covered_examples = uncovered.intersection(covered_examples)


                    y_cr = y[cr_covered]
                    y_rr = y[rr_covered]
                    y_er = y[er_covered]

                    y_er_mean = np.mean(y_er)
                    # y_er_std = np.sqrt((np.sum(np.square(y_er)) / y_er.shape[0]) - (y_er_mean * y_er_mean))
                    
                    y_rr_mean = np.mean(y_rr)
                    y_rr_std = np.sqrt((np.sum(np.square(y_rr)) / y_rr.shape[0]) - (y_rr_mean * y_rr_mean))
                    rr_up = y_rr_mean + y_rr_std
                    rr_down = y_rr_mean - y_rr_std
                    
                    y_cr_mean = np.mean(y_cr)
                    y_cr_std = np.sqrt((np.sum(np.square(y_cr)) / y_cr.shape[0]) - (y_cr_mean * y_cr_mean))
                    cr_up = y_cr_mean + y_cr_std
                    cr_down = y_cr_mean - y_cr_std

                    if y_rr_mean > cr_down and y_rr_mean < cr_up and (y_er_mean > cr_up or y_er_mean < cr_down) and (y_er_mean > rr_up or y_er_mean < rr_down):
                        candidate = True
                    else:
                        candidate = False

                    
                    quality, coverage = self._calculate_quality_using_covered(X, y, rule_with_condition_covered_mask)

                    score = len(er_covered)



                    if (score > best_score) and candidate:
                        if self._check_candidate(new_covered_examples, rr_uncovered) and self._check_exception_candidate(X, y, cr_covered, rr_covered, er_covered):
                            condition_best = condition
                            quality_best = quality
                            coverage_best = coverage


                            best_score = score
    
            return condition_best, quality_best, coverage_best, best_score  
    
    def calculate_p_value(self, y1, y2):
        """Return the two-sided Mann-Whitney U-test p-value for two samples."""
        stat, p = stats.mannwhitneyu(y1, y2)
        return p
    
    def _induce_condition(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray, uncovered:list[int]) -> AbstractCondition:
            """Select the best condition satisfying minimum coverage.

            Ties favor the candidate covering more locally positive examples.
            Returns ``(condition, quality, coverage)``.
            """
            quality_best = float("-inf")
            coverage_best = CoverageClass(0,0,0,0)
            condition_best = None
            # examples_covered_by_rule, y_for_examples_covered_by_rule = self._get_covered_examples(X,y,rule)
            # rule_with_condition = copy.deepcopy(rule)

            rule_covered_mask: np.ndarray = rule.premise.covered_mask(X)
            examples_covered_by_rule = X[rule_covered_mask]
            y_for_examples_covered_by_rule = y[rule_covered_mask]


            possible_conditions = self._get_possible_conditions(examples_covered_by_rule, y_for_examples_covered_by_rule)
            possible_conditions_filtered = list(filter(lambda i: i not in rule.premise.subconditions, possible_conditions))
            if len(possible_conditions_filtered) != 0:
                for condition in possible_conditions_filtered:
                    
                    condition_str = condition.__hash__()
                    if condition_str in self.conditions_coverage_cache:
                        condition_coverage_mask = self.conditions_coverage_cache[condition_str]
                    else:
                        condition_coverage_mask = condition.covered_mask(self.X_numpy)
                        self.conditions_coverage_cache[condition_str] = condition_coverage_mask

                    rule_with_condition_covered_mask = np.logical_and(
                                                                    rule_covered_mask,
                                                                    condition_coverage_mask
                                                                    )

                    
                    # coverage = rule_with_condition.calculate_coverage(X=X, y=y)
                    # quality = self.measure_function(coverage)

                    covered_examples = np.where(rule_with_condition_covered_mask == 1)[0]
                    covered_examples = set(covered_examples)
                    # new_covered_examples = [i for i in uncovered if i in covered_examples]
                    new_covered_examples = uncovered.intersection(covered_examples)
                    
                    quality, coverage = self._calculate_quality_using_covered(X, y, rule_with_condition_covered_mask)

                    #self.logger.info(f"Condition: {str(rule_with_condition)}, p: {coverage.p}, n: {coverage.n}, P: {coverage.P}, N: {coverage.N}, Quality: {quality}, new_covered: {len(new_covered_examples)}")
                    
                    if (quality > quality_best or ((quality == quality_best) and (coverage.p > coverage_best.p))):
                            if self._check_candidate(new_covered_examples, uncovered):
                                condition_best = condition
                                quality_best = quality
                                coverage_best = coverage
                                
                    # rule_with_condition.premise.subconditions.remove(condition)
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


    def _prune(self, rule: AbstractRule):
        """Remove conditions while doing so preserves or improves quality."""
        
        if len(rule.premise.subconditions) == 1:
            return
        
        continue_pruning = True
        while continue_pruning:
            conditions = rule.premise.subconditions
            quality_best, _ = self._calculate_quality(rule, self.X_numpy, self.y_numpy)
            condition_to_remove = None
            for condition in conditions:
                rule_without_condition = copy.deepcopy(rule)
                rule_without_condition.premise.subconditions.remove(condition)

                quality_without_condition, coverage_without_condition = self._calculate_quality(rule_without_condition, self.X_numpy, self.y_numpy)

                if quality_without_condition >= quality_best:
                    quality_best = quality_without_condition
                    condition_to_remove = condition
                
            if condition_to_remove is None:
                continue_pruning = False 
            else:
                rule.premise.subconditions.remove(condition_to_remove)

            if len(rule.premise.subconditions) == 1:
                continue_pruning = False
    
        
    def _check_candidate(self, new_covered_examples: int, uncovered) -> bool:
        """Return whether a candidate meets the minimum-coverage constraint."""
        return  (len(new_covered_examples) >= self.mincov) or (len(uncovered) <= self.mincov)
    
    def _get_covered_examples(self, X: np.ndarray, y: np.ndarray, rule: AbstractRule) -> List[np.ndarray]:
        """Return feature and target subsets covered by ``rule``'s premise."""
        covered_examples_mask = rule.premise.covered_mask(X)
        return [X[covered_examples_mask], y[covered_examples_mask]]
    
    
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

    

    def _get_possible_conditions(self, examples_covered_by_rule: np.ndarray, y: np.ndarray) -> list:
        """Generate nominal equality and numeric midpoint conditions.

        Missing values are ignored.  Each numeric midpoint yields a lower and
        an upper half-line condition.
        """
        conditions = []

        for indx in self.nominal_attributes_indexes:
            # Remove None values
            column = examples_covered_by_rule[:,indx]
            filtered_column = column[~pd.isnull(column)]
            conditions.extend([NominalCondition(column_index=indx, value=val) for val in np.unique(filtered_column)])

        for indx in self.numerical_attributes_indexes:
            # if self.cuts_only_between_classes:
            #     attr_values = examples_covered_by_rule[:,indx].astype(float)
            #     attr_values = np.stack((attr_values, y), axis=1)
            #     attr_values = attr_values[~pd.isnull(attr_values[:, 0])]
            #     sorted_indices = np.argsort(attr_values[:, 0])
            #     sorted_attr_values = attr_values[sorted_indices]
            #     change_indices = [i for i in range(1, len(sorted_attr_values)) if sorted_attr_values[i, 1] != sorted_attr_values[i-1, 1]]
            #     mid_points = np.unique([(sorted_attr_values[indx-1,0] + sorted_attr_values[indx,0]) / 2 for indx in change_indices])
            # else:
            examples_covered_by_rule_for_attr = examples_covered_by_rule[:,indx].astype(float)
            values = np.sort(np.unique(examples_covered_by_rule_for_attr[~np.isnan(examples_covered_by_rule_for_attr)]))
            mid_points = [(x + y) / 2 for x, y in zip(values, values[1:])]

            conditions.extend([ElementaryCondition(column_index=indx, left_closed=False, right_closed=True, left=float('-inf'), right=mid_point) for mid_point in mid_points])
            conditions.extend([ElementaryCondition(column_index=indx, left_closed=True, right_closed=False, left=mid_point, right=float('inf')) for mid_point in mid_points]) 

        return conditions



    def _get_nominal_indexes(self, dataframe: pd.DataFrame) -> list:
        """Return positional indices of object-typed feature columns."""
        dtype_mask = (dataframe.dtypes == 'object')
        nominal_indexes = np.where(dtype_mask)[0]
        return nominal_indexes.tolist()
    
    def _get_numerical_indexes(self, dataframe: pd.DataFrame) -> list:
        """Return positional indices of non-object feature columns."""
        dtype_mask = np.logical_not(dataframe.dtypes == 'object')
        numerical_indexes = np.where(dtype_mask)[0]
        return numerical_indexes.tolist()
