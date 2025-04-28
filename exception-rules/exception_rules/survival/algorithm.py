from decision_rules.core.coverage import Coverage as CoverageClass
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.core.rule import AbstractRule
from decision_rules.core.condition import AbstractCondition
from decision_rules.survival.ruleset import SurvivalRuleSet
from decision_rules.survival.rule import SurvivalRule, SurvivalConclusion
from decision_rules.conditions import CompoundCondition, LogicOperators, NominalCondition, ElementaryCondition
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator
from typing import List
import pandas as pd
import numpy as np
import copy
import warnings
warnings.filterwarnings('ignore')    

import logging


class MyRuleSurvival():

    def __init__(self, mincov: int, survival_time_attr: str, cuts_only_between_classes: bool = True, max_growing: int = None, prune: bool = True, find_exceptions:bool = False, delete_cr_n = False, logger = None) -> None:
        self.cuts_only_between_classes = cuts_only_between_classes
        self.mincov = mincov
        self.survival_time_attr = survival_time_attr
        self.max_growing = max_growing
        self.prune = prune
        self.find_exceptions = find_exceptions
        self.delete_cr_n = delete_cr_n

        self.label_name = None
        self.X_numpy = None
        self.y_numpy = None

        if logger is not None:
            self.if_logging = True
            self.logger = logger
        else:
            self.if_logging = False

        

    def fit(self, X: pd.DataFrame, y: pd.Series, attributes_list: list[list[str]] = None) -> AbstractRuleSet:

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
            self.labels = y.unique()

            self.survival_time = X[self.survival_time_attr].to_numpy()
            self.survival_status = y.to_numpy()

            survival_time_attr_index = X.columns.get_loc(self.survival_time_attr)
            if survival_time_attr_index in self.nominal_attributes_indexes:
                self.nominal_attributes_indexes.remove(survival_time_attr_index)
            elif survival_time_attr_index in self.numerical_attributes_indexes:
                self.numerical_attributes_indexes.remove(survival_time_attr_index)

            rules = []

        

            carry_on = True

            uncovered = [i for i in range(len(self.y_numpy))]



            while carry_on:
                rule = self._rule_factory(self.columns_names, self.label_name)
                carry_on = self._grow(rule, self.X_numpy, self.y_numpy, uncovered)

                if (carry_on):
                    if(self.prune):
                        self._prune(rule)

                    previously_uncovered = len(uncovered)
                    
                    positive_covered_indices = np.where(rule.positive_covered_mask(self.X_numpy, self.y_numpy) == 1)[0]
                    negative_covered_indices = np.where(rule.negative_covered_mask(self.X_numpy, self.y_numpy) == 1)[0]
                    
                    
                    uncovered = [i for i in uncovered if i not in positive_covered_indices and i not in negative_covered_indices]

                    if (len(uncovered) == previously_uncovered):
                        carry_on = False
                    else:
                        rule = self._update_estimator(rule)

                        rules.append(rule)



            ruleset = self._ruleset_factory(rules)

            ruleset.column_names = self.columns_names
            ruleset.update(X,y)
            self.ruleset = ruleset
            return self

    def _update_estimator(self, rule: SurvivalRule) -> SurvivalRule:
        
        covered_examples = rule.premise._calculate_covered_mask(self.X_numpy)
        km = KaplanMeierEstimator()
        km.fit(self.survival_time[covered_examples], self.survival_status[covered_examples])
        rule.conclusion.value = km
        rule.measure, rule.coverage = self._calculate_quality(rule, self.X_numpy, self.y_numpy)
        return rule
    


    def _grow(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray, uncovered:list[int]) -> AbstractRule:
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
                
                if self.if_logging:
                    self.logger.info(f"Iteracja {i}: condition_best: {condition_best.to_string(self.columns_names)}, quality_best: {round(quality_best,3)}, coverage_best: {str(coverage_best)}")
                    self.logger.info(f"Regula po iteracji {i}: {rule}, {rule.calculate_coverage(X,y)}")

                if self.find_exceptions:
                    carry_on = not self._search_exceptions(rule, X, y)
                       
            else:
                carry_on = False
                if self.if_logging:
                    self.logger.info(f"Iteracja {i}: condition_best: None, quality_best: {round(quality_best,3)}, coverage_best: {str(coverage_best)}")
                    self.logger.info(f"Regula po iteracji {i}: {rule}, {rule.calculate_coverage(X,y)}")


            if (self.max_growing is not None) and (len(rule.premise.subconditions) >= self.max_growing):
                carry_on = False


            i +=1 
        if self.if_logging:
            self.logger.info("*******STOP GROWING CR RULE*******")

        if len(rule.premise.subconditions) > 0:
            maks_quality_index = np.argmax(rule_qualities)
            rule.premise.subconditions = rule.premise.subconditions[:maks_quality_index+1]
            return True
        else:
            return False
        
    def _search_exceptions(self, rule, X, y):


        cr_covered = np.where(rule.positive_covered_mask(X, y) == 1)[0]
        cr_uncovered = np.where(rule.positive_covered_mask(X, y) == 0)[0]


        reference_rule = self._rule_factory(self.columns_names, self.label_name)

        # X_tmp = X[cr_uncovered]
        # y_tmp = y[cr_uncovered]

        # uncovered = [i for i in range(len(y_tmp))]

        found_reference_rule = self._grow_reference_rule(reference_rule, X, y, cr_uncovered, cr_covered)


        if found_reference_rule:
            if self.if_logging:
                self.logger.info("***RR FOUND***")
                self.logger.info(f"Reference rule: {reference_rule}")
            found_exception = self._check_exception_candidate(X, y, rule, reference_rule)
            if found_exception:
                return True
            else:
                return False
        else:
            if self.if_logging:
                self.logger.info("***RR NOT FOUND***")
            return False
        
    def _check_exception_candidate(self, X, y, comonsense_rules, reference_rule) -> bool:
            
            if self.if_logging:
                self.logger.info("***CHECKING EXCEPTION***")
            
            exception_rule = self._rule_factory(self.columns_names, self.label_name)
            exception_rule.premise.subconditions.extend(comonsense_rules.premise.subconditions)
            exception_rule.premise.subconditions.extend(reference_rule.premise.subconditions)

            er_covered = np.where(exception_rule.positive_covered_mask(X, y) == 1)[0]
            cr_covered = np.where(comonsense_rules.positive_covered_mask(X, y) == 1)[0]
            rr_covered = np.where(reference_rule.positive_covered_mask(X, y) == 1)[0]



            er_km = KaplanMeierEstimator().fit(self.survival_time[er_covered], self.survival_status[er_covered], update_additional_informations=False)
            cr_km = KaplanMeierEstimator().fit(self.survival_time[cr_covered], self.survival_status[cr_covered], update_additional_informations=False)
            rr_km = KaplanMeierEstimator().fit(self.survival_time[rr_covered], self.survival_status[rr_covered], update_additional_informations=False)


            cr_stats_and_pvalue = KaplanMeierEstimator().compare_estimators(
                        cr_km, er_km)
            
            rr_stats_and_pvalue = KaplanMeierEstimator().compare_estimators(
                    rr_km, er_km)
            
            cr_p_value = cr_stats_and_pvalue["p_value"]
            rr_p_value = rr_stats_and_pvalue["p_value"]
            if self.if_logging:
                self.logger.info(f"ER vs CR p_value: {cr_p_value}")
                self.logger.info(f"ER vs RR p_value: {rr_p_value}")
                                 
            if (cr_stats_and_pvalue["p_value"] <= 0.05) and (rr_stats_and_pvalue["p_value"] <= 0.05):
                comonsense_rules.reference_rule = reference_rule   
                comonsense_rules.exception_rule = exception_rule
                triple_ruleset = SurvivalRuleSet(rules = [comonsense_rules,exception_rule,reference_rule], survival_time_attr=self.survival_time_attr)
                triple_ruleset.update(self.X_pandas,self.y_pandas)
                if self.if_logging:
                    self.logger.info("***ER FOUND***")
                    self.logger.info(f"Exception rule: {exception_rule}")
                return True
            else:
                if self.if_logging:
                    self.logger.info("***ER NOT FOUND***")
                return False



        
    def _grow_reference_rule(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray,uncovered: list[int], cr_covered) -> AbstractRule:
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


        self.rule_qualities = rule_qualities
        self.rule_covered_negatives = rule_covered_negatives

        if self.if_logging:
            self.logger.info("*****STOP GROWING RR RULE*****")
        
        if len(rule.premise.subconditions) > 0:
            maks_quality_index = np.argmax(rule_qualities)
            rule.premise.subconditions = rule.premise.subconditions[:maks_quality_index+1]
            return True
        else:
            return False
        
    def _induce_reference_rule(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray, best_score, uncovered, cr_covered) -> AbstractCondition:
            
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
            
    
    def _prune(self, rule: AbstractRule):
        
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
        return  (len(new_covered_examples) >= self.mincov) or (len(uncovered) <= self.mincov)
    
    def _get_covered_examples(self, X: np.ndarray, y: np.ndarray, rule: AbstractRule) -> List[np.ndarray]:
        covered_examples_mask = rule.premise.covered_mask(X)
        return [X[covered_examples_mask], y[covered_examples_mask]]
    
    
    def _rule_factory(self, columns_names, label_name) -> SurvivalRule:
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
        covered_examples_indexes = np.where(rule.premise._calculate_covered_mask(X))[0]
        uncovered_examples_indexes = np.where(rule.premise._calculate_uncovered_mask(X))[0]
        quality = KaplanMeierEstimator.log_rank(self.survival_time, self.survival_status, covered_examples_indexes, uncovered_examples_indexes)
        coverage = CoverageClass(p=len(covered_examples_indexes), n= 0, P=X.shape[0], N=0)
        return quality, coverage

    
    def _ruleset_factory(self, rules: list[SurvivalRule]) -> SurvivalRuleSet:
        return SurvivalRuleSet(rules=rules, survival_time_attr=self.survival_time_attr)

    

    def _get_possible_conditions(self, examples_covered_by_rule: np.ndarray, y: np.ndarray) -> list:
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
        dtype_mask = (dataframe.dtypes == 'object')
        nominal_indexes = np.where(dtype_mask)[0]
        return nominal_indexes.tolist()
    
    def _get_numerical_indexes(self, dataframe: pd.DataFrame) -> list:
        dtype_mask = np.logical_not(dataframe.dtypes == 'object')
        numerical_indexes = np.where(dtype_mask)[0]
        return numerical_indexes.tolist()
 