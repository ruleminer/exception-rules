from abc import ABC
from abc import abstractmethod
from collections import defaultdict

import time
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.core.rule import AbstractRule
from decision_rules.core.condition import AbstractCondition
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.classification.rule import ClassificationRule, ClassificationConclusion
from decision_rules.conditions import CompoundCondition, LogicOperators, NominalCondition, ElementaryCondition
from decision_rules.measures import *
from typing import List
import pandas as pd
import numpy as np
import copy
from sklearn.datasets import load_iris
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
import cProfile, pstats
import multiprocessing
from scipy.io import arff
import warnings
warnings.filterwarnings('ignore')    

from decision_rules.core.coverage import Coverage
from collections import Counter

import logging


class MyRuleClassifier():

    def __init__(self, mincov: int, induction_measuer: str, cuts_only_between_classes: bool = True, max_growing: int = None, prune: bool = True, find_exceptions:bool = False, threshold:float = 0.8, delete_cr_n:bool = False, logger = None) -> None:
        self.cuts_only_between_classes = cuts_only_between_classes
        self.mincov = mincov
        self.measure_function = globals().get(induction_measuer)
        self.max_growing = max_growing
        self.prune = prune
        self.treshold = threshold
        self.find_exceptions = find_exceptions
        self.delete_cr_n = delete_cr_n
        
        self.label_name = None
        self.X_numpy = None
        self.y_numpy = None

        self.conditions_coverage_cache: dict[AbstractCondition, np.ndarray] = {
        }

        if logger is not None:
            self.if_logging = True
            self.logger = logger
        else:
            self.if_logging = False


        
    def _rule_factory(self, columns_names, label_name, label_value) -> ClassificationRule:
        return ClassificationRule(
            column_names=columns_names,
            premise=CompoundCondition(subconditions=[],
                                    logic_operator=LogicOperators.CONJUNCTION,),
            conclusion=ClassificationConclusion(
                value=label_value,
                column_name=label_name,
            ))
    
    def _ruleset_factory(self, rules: list[ClassificationRule]) -> ClassificationRuleSet:
        return ClassificationRuleSet(rules=rules)
    
    def _prepare_additional_informations(self) -> None:
        unique_values, counts = np.unique(self.y_numpy, return_counts=True)
        self.decision_attribute_distribution = dict(zip(unique_values, counts))

    def fit(self, X: pd.DataFrame, y: pd.Series, attributes_list: list[list[str]] = None) -> AbstractRuleSet:

        self.conditions_coverage_cache: dict[AbstractCondition, np.ndarray] = {
        }
                
        self.label_name = y.name 

        self.X_numpy = X.to_numpy()
        self.y_numpy = y.to_numpy()

        self.X_pandas = X
        self.y_pandas = y

        self.attributes_list = attributes_list
        
        # get indexes of nominal attributes
        self.nominal_attributes_indexes = self._get_nominal_indexes(X)
        self.numerical_attributes_indexes = self._get_numerical_indexes(X)
        self.columns_names = X.columns
        self.labels = y.unique()

        self._prepare_additional_informations()
        rules = []

       

        for label in self.labels:
            carry_on = True

            uncovered_positives = set(np.where(self.y_numpy == label)[0])

            while carry_on:
                rule = self._rule_factory(self.columns_names, self.label_name, label)
                carry_on = self._grow(rule, self.X_numpy, self.y_numpy, uncovered_positives, label)

                if (carry_on):
                    if(self.prune):
                        self._prune(rule)

                    previously_uncovered = len(uncovered_positives)
                    positive_covered_indices = np.where(rule.positive_covered_mask(self.X_numpy, self.y_numpy) == 1)[0]

                    uncovered_positives = set([i for i in uncovered_positives if i not in positive_covered_indices])


                    if (len(uncovered_positives) == previously_uncovered):
                        carry_on = False
                    else:
                        rules.append(rule)




        ruleset = self._ruleset_factory(rules)
        ruleset.update(X, y, self.measure_function)
        self.ruleset = ruleset

        if self.find_exceptions:
            self._evaluate_exceptions(self.ruleset, X, y)

        return self
    
    def _evaluate_exceptions(self, ruleset, X, y):
        
        self.labels = y.unique()
        class_rule_metrics = {label: {'precision': [], 'coverage': [], 'dprecision': []} for label in self.labels}
        self.class_averages = {}

        
        for rule in ruleset.rules:
            rule_coverage = rule.calculate_coverage(self.X_numpy, self.y_numpy)
            prec = precision(rule_coverage)
            cov = coverage(rule_coverage)
            dprec = self.domain_precision(rule_coverage)

            # Dodawanie miar do listy dla danej klasy
            class_rule_metrics[rule.conclusion.value]['precision'].append(prec)
            class_rule_metrics[rule.conclusion.value]['coverage'].append(cov)
            class_rule_metrics[rule.conclusion.value]['dprecision'].append(dprec)

        for label in self.labels:
            precisions = class_rule_metrics[label]['precision']
            coverages = class_rule_metrics[label]['coverage']
            dprecisions = class_rule_metrics[label]['dprecision']

            avg_precision = np.mean(precisions) if precisions else 0
            avg_coverage = np.mean(coverages) if coverages else 0
            avg_dprecision = np.mean(dprecisions) if dprecisions else 0

            self.class_averages[label] = {
                'precision': avg_precision,
                'coverage': avg_coverage,
                'dprecision': avg_dprecision
            }

        for rule in ruleset.rules:
            if len(rule.exception_rules) >= 0:
                for er_rule, rr_rule in zip(rule.exception_rules, rule.reference_rules):
                    self._evaluate_exception(rule, er_rule, rr_rule)
            else:
                self._evaluate_exception(rule, rule.exception_rule, rule.reference_rule)



    def _evaluate_exception(self, cr_rule, er_rule, rr_rule) -> None:    
            

        # Sprawdzamy warunki na Eer
            er_coverage = er_rule.calculate_coverage(self.X_numpy, self.y_numpy)
            rr_coverage = rr_rule.calculate_coverage(self.X_numpy, self.y_numpy)

            er_rule.coverage= er_coverage
            rr_rule.coverage = rr_coverage
            er_label = er_rule.conclusion.value
            rr_label = rr_rule.conclusion.value

            avg_precision_er_class = self.class_averages[er_label]['precision']
            avg_coverage_er_class = self.class_averages[er_label]['coverage']
            avg_dprecision_er_class = self.class_averages[er_label]['dprecision']

            avg_precision_rr_class = self.class_averages[rr_label]['precision']
            avg_coverage_rr_class = self.class_averages[rr_label]['coverage']
            avg_dprecision_rr_class = self.class_averages[rr_label]['dprecision']

            er_precision = precision(er_coverage)
            er_coverage_value = coverage(er_coverage)
            er_dprecision = self.domain_precision(er_coverage)

            rr_precision = precision(rr_coverage)
            rr_coverage_value = coverage(rr_coverage)

            if self.if_logging:
                self.logger.info(f"******")
                self.logger.info(f"CR rule: {cr_rule}")
                self.logger.info(f"RR rule: {rr_rule}")
                self.logger.info(f"er rule: {er_rule}")
                
                
                self.logger.info(f"RR precision: {rr_precision}, RR coverage: {rr_coverage_value}")
                self.logger.info(f"{rr_label}: AVG precision: {avg_precision_rr_class}, AVG coverage: {avg_coverage_rr_class}, AVG domain precision: {avg_dprecision_rr_class}")
                self.logger.info(f"ER precision: {er_precision}, ER coverage: {er_coverage_value}, ER domain precision: {er_dprecision}")
                self.logger.info(f"{er_label}: AVG precision: {avg_precision_er_class}, AVG coverage: {avg_coverage_er_class}, AVG domain precision: {avg_dprecision_er_class}")
                
            # Warunki na Eer
            if (er_coverage_value < avg_coverage_er_class and
                er_precision > avg_precision_er_class and
                er_dprecision > avg_dprecision_er_class and
                rr_coverage_value > avg_coverage_rr_class and
                rr_precision > avg_precision_rr_class):
                er_rule.rule_type = '1'
            elif (er_coverage_value < avg_coverage_er_class and
                er_precision > avg_precision_er_class and
                er_dprecision > avg_dprecision_er_class):
                er_rule.rule_type = '2'
            elif er_precision > avg_precision_er_class:
                er_rule.rule_type = 'AR'
            else:
                # Usuwamy regułę
                er_rule.rule_type = "None"

            if self.if_logging:
                self.logger.info(f"er rule type: {er_rule.rule_type}")
                self.logger.info(f"******")

    def _grow(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray, uncovered_positives:set[int], label: str) -> AbstractRule:
        carry_on = True
        rule_qualities = []
        if self.if_logging:
            self.logger.info("*******GROWING CR RULE*******")
        i = 0
        while(carry_on):
            condition_best, quality_best, coverage_best = self._induce_condition(rule, X, y, uncovered_positives)
            
            if condition_best is not None:
                rule.premise.subconditions.append(condition_best) # add the best condition to the rule
                rule_qualities.append(quality_best)

                if self.if_logging:
                    self.logger.info(f"Iteracja {i}: condition_best: {condition_best.to_string(self.columns_names)}, quality_best: {round(quality_best,3)}, coverage_best: {str(coverage_best)}, precision: {round(precision(coverage_best),3)}")
                    self.logger.info(f"Regula po iteracji {i}: {rule}, {rule.calculate_coverage(X,y)}")
                if coverage_best.n == 0:
                    carry_on = False
                elif (precision(coverage_best) > self.treshold) and self.find_exceptions:
                    carry_on = not self._search_exceptions(rule, X, y, label)
                       
            else:
                carry_on = False
                if self.if_logging:
                    self.logger.info(f"Iteracja {i}: condition_best: None, quality_best: {round(quality_best,3)}, coverage_best: {str(coverage_best)}, precision: {round(precision(coverage_best),3)}")
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


    def _induce_condition(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray, uncovered_positives:set[int]) -> AbstractCondition:
        quality_best = float("-inf")
        coverage_best = Coverage(0,0,0,0)
        condition_best = None

        rule_covered_mask: np.ndarray = rule.premise.covered_mask(X)
        examples_covered_by_rule = X[rule_covered_mask]
        y_for_examples_covered_by_rule = y[rule_covered_mask]

        positive_mask: np.ndarray = rule.conclusion.positives_mask(y)

        # examples_covered_by_rule, y_for_examples_covered_by_rule = self._get_covered_examples(X,y,rule)

        possible_conditions = self._get_possible_conditions(examples_covered_by_rule, y_for_examples_covered_by_rule)
        possible_conditions_filtered = list(filter(lambda i: i not in rule.premise.subconditions, possible_conditions))
        if len(possible_conditions_filtered) != 0:
            for condition in possible_conditions_filtered:
                # rule_with_condition = copy.deepcopy(rule)
                # rule_with_condition.premise.subconditions.append(condition)

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
                rule_with_condition_covered_count: int = np.sum(rule_with_condition_covered_mask)

                rule_with_condition_positive_covered_mask = np.logical_and(
                        rule_with_condition_covered_mask, positive_mask
                    )
                
                covered_positives = np.where(rule_with_condition_positive_covered_mask == 1)[0]

                covered_positives = set(covered_positives)
                new_covered_positives: set[int] = uncovered_positives.intersection(
                    covered_positives)

                
                quality, coverage = self._calculate_quality_using_covered_positives(
                    rule=rule,
                    rule_covered_count=rule_with_condition_covered_count,
                    covered_positives=covered_positives
                )

                if (quality > quality_best or ((quality == quality_best) and (coverage.p > coverage_best.p))):
                        if self._check_candidate(new_covered_positives, rule.conclusion.value):
                            condition_best = condition
                            quality_best = quality
                            coverage_best = coverage

        return condition_best, quality_best, coverage_best   
    
    def _check_candidate(self, new_covered_examples: int, y: str) -> bool:
        return  len(new_covered_examples) >= self.mincov or (self.decision_attribute_distribution[y] <= self.mincov and len(new_covered_examples) > 0)
    

    def _search_exceptions(self, rule, X, y, label):
        rule_covered_positives = np.where(rule.positive_covered_mask(X, y) == 1)[0]
        rule_covered_negatives = np.where(rule.negative_covered_mask(X, y) == 1)[0]
        counter = Counter(y[rule_covered_negatives])
        most_common_label, count = counter.most_common(1)[0]

        y_tmp = y.copy()
        y_tmp[rule_covered_positives] = "not_" + label

        uncovered_positives = set(np.where(y_tmp == label)[0])

        reference_rule = self._rule_factory(self.columns_names, self.label_name, label)
        found_reference_rule = self._grow_reference_rule(reference_rule, X, y_tmp, uncovered_positives, rule_covered_negatives)

        

        if found_reference_rule:
            if self.if_logging:
                self.logger.info("***RR FOUND***")
                self.logger.info(f"Reference rule: {reference_rule}")
            found_exception = self._check_exception_candidate(X, y, rule, reference_rule, most_common_label)
            if found_exception:
                return True
            else:
                
                
                return False
        else:
            if self.if_logging:
                self.logger.info("***RR NOT FOUND***")
            return False
        
    def domain_precision(self,c: Coverage) -> float:  # pylint: disable=missing-function-docstring
        return precision(c) - c.P / (c.P + c.N)
    
    def _grow_reference_rule(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray, uncovered_positives:set[int], negatives_to_cover: list[int]) -> AbstractRule:
        carry_on = True
        rule_qualities = []
        rule_covered_negatives = []
        scores = []
        best_score = 0
        if self.if_logging:
            self.logger.info("*****GROWING RR RULE*****")

        i = 0

        while(carry_on):
            condition_best, quality_best, coverage_best, number_of_covered_negatives_best, best_score = self._induce_reference_rule(rule, X, y, uncovered_positives, negatives_to_cover, best_score)
            
            if condition_best is not None:
                rule.premise.subconditions.append(condition_best) # add the best condition to the rule
                rule_qualities.append(quality_best)
                rule_covered_negatives.append(number_of_covered_negatives_best)
                scores.append(best_score)

                
                if (precision(coverage_best) > self.treshold):
                    carry_on = False

                if coverage_best.n == 0:
                    carry_on = False
            else:
                carry_on = False

            if (self.max_growing is not None) and (len(rule.premise.subconditions) >= self.max_growing):
                carry_on = False

            if self.if_logging:
                if condition_best is not None:
                    self.logger.info(f"Iteracja {i}: condition_best: {condition_best.to_string(self.columns_names)}, quality_best: {round(quality_best, 3)}, coverage_best: {str(coverage_best)}, number_of_covered_negatives_best: {number_of_covered_negatives_best}, best_score: {round(best_score,3)}, precision: {round(precision(coverage_best),3)}")
                else:
                    self.logger.info(f"Iteracja {i}: condition_best: None quality_best: {round(quality_best, 3)}, coverage_best: {str(coverage_best)}, number_of_covered_negatives_best: {number_of_covered_negatives_best}, best_score: {round(best_score,3)}, precision: {round(precision(coverage_best),3)}")
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
        
    def _induce_reference_rule(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray, uncovered_positives:set[int], negatives_to_cover: list[int], best_score) -> AbstractCondition:
            
            quality_best = float("-inf")
            coverage_best = Coverage(0,0,0,0)
            condition_best = None
            number_of_covered_negatives_best = 0
            
            rule_covered_mask: np.ndarray = rule.premise.covered_mask(X)
            examples_covered_by_rule = X[rule_covered_mask]
            y_for_examples_covered_by_rule = y[rule_covered_mask]

            positive_mask: np.ndarray = rule.conclusion.positives_mask(y)
            negative_mask: np.ndarray = rule.conclusion.negatives_mask(y)

            scores = []
            negative_numbers = []
            positives_numbers = []

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
                    rule_with_condition_covered_count: int = np.sum(rule_with_condition_covered_mask)

                    rule_with_condition_positive_covered_mask = np.logical_and(
                            rule_with_condition_covered_mask, positive_mask
                        )
                    
                    covered_positives = np.where(rule_with_condition_positive_covered_mask == 1)[0]

                    covered_positives = set(covered_positives)
                    new_covered_positives: set[int] = uncovered_positives.intersection(
                        covered_positives)

                    rule_with_condition = copy.deepcopy(rule)
                    rule_with_condition.premise.subconditions.append(condition)
                    
                    quality, coverage = self._calculate_quality_using_covered_positives(
                            rule=rule,
                            rule_covered_count=rule_with_condition_covered_count,
                            covered_positives=covered_positives,
                            y=y
                        )

                    rule_with_condition_negative_covered_mask = np.logical_and(
                            rule_with_condition_covered_mask, negative_mask
                        )
                    covered_negatives = np.where(rule_with_condition_negative_covered_mask == 1)[0]

                    negatives_to_cover_covered = [i for i in negatives_to_cover if i in covered_negatives]

                    number_of_covered_negatives = len(negatives_to_cover_covered)
                    number_of_negatives_to_cover = len(negatives_to_cover)

                    # Calculate the combined score
                    if number_of_negatives_to_cover != 0:
                        negatives_score = (1 * number_of_covered_negatives/number_of_negatives_to_cover)
                    else:
                        negatives_score = 0
                        print("negatives_score = 0")

                    if len(uncovered_positives) != 0:
                        positives_score = (1 * len(new_covered_positives)/len(uncovered_positives))
                    else:   
                        positives_score = 0

                    quality_score = (1 * quality)
                    
                    score = quality_score + negatives_score + positives_score

                    scores.append(score)
                    negative_numbers.append(number_of_covered_negatives)
                    positives_numbers.append(len(new_covered_positives))
                    if score > best_score and precision(coverage) > self.treshold:
                        if self._check_candidate(new_covered_positives, rule.conclusion.value):
                            condition_best = condition
                            quality_best = quality
                            coverage_best = coverage
                            number_of_covered_negatives_best = number_of_covered_negatives
                            best_score = score
                            

            return condition_best, quality_best, coverage_best, number_of_covered_negatives_best, best_score              
    
    
    def _check_exception_candidate(self, X, y, comonsense_rule, reference_rule, exception_conclusion) -> bool:
            
            if self.if_logging:
                self.logger.info("***CHECKING EXCEPTION***")
            
            exception_rule = self._rule_factory(self.columns_names, self.label_name, exception_conclusion)
            exception_rule.premise.subconditions.extend(comonsense_rule.premise.subconditions)
            exception_rule.premise.subconditions.extend(reference_rule.premise.subconditions)

            exception_coverage = exception_rule.calculate_coverage(X, y)

            if self.if_logging:
                self.logger.info(f"Exception precision: {precision(exception_coverage)}")


            if (precision(exception_coverage) > self.treshold):
                comonsense_rule.reference_rule = reference_rule   
                comonsense_rule.exception_rule = exception_rule
                comonsense_rule.exception_rules.append(exception_rule)
                comonsense_rule.reference_rules.append(reference_rule)
                triple_ruleset = ClassificationRuleSet(rules = [comonsense_rule,exception_rule,reference_rule])
                triple_ruleset.update(self.X_pandas,self.y_pandas, self.measure_function)
                if self.if_logging:
                    self.logger.info("***ER FOUND***")
                    self.logger.info(f"Exception rule: {exception_rule}")
                return True
            else:
                if self.if_logging:
                    self.logger.info("***ER NOT FOUND***")
                return False




    
    def _calculate_quality(self, rule: AbstractRule, X: np.ndarray, y: np.ndarray) -> float:
        coverage = rule.calculate_coverage(X=X, y=y)
        quality = self.measure_function(coverage)

        return quality, coverage
    
    def _calculate_quality_using_covered_positives(
        self,
        rule: ClassificationRule,
        rule_covered_count: int,
        covered_positives: np.ndarray,
        y = None
    ) -> tuple[Coverage, float]:
        p = len(covered_positives)
        n = rule_covered_count - len(covered_positives)

        if y is not None:
            P = np.sum(y == rule.conclusion.value)
            N = len(y) - P
        else:
            P = self.decision_attribute_distribution[rule.conclusion.value]
            N = sum(self.decision_attribute_distribution.values()) - P
        if n > N:
            print("here")
        coverage = Coverage(p, n, P, N)
        quality: float = self.measure_function(coverage)
        return quality, coverage
    


    def _get_possible_conditions(self, examples_covered_by_rule: np.ndarray, y: np.ndarray) -> list:
        conditions = []

        for indx in self.nominal_attributes_indexes:
            # Remove None values
            column = examples_covered_by_rule[:,indx]
            filtered_column = column[~pd.isnull(column)]
            conditions.extend([NominalCondition(column_index=indx, value=val) for val in np.unique(filtered_column)])

        for indx in self.numerical_attributes_indexes:
            if self.cuts_only_between_classes:
                attr_values = examples_covered_by_rule[:,indx].astype(float)
                attr_values = np.stack((attr_values, y), axis=1)
                attr_values = attr_values[~pd.isnull(attr_values[:, 0])]
                sorted_indices = np.argsort(attr_values[:, 0])
                sorted_attr_values = attr_values[sorted_indices]
                change_indices = [i for i in range(1, len(sorted_attr_values)) if sorted_attr_values[i, 1] != sorted_attr_values[i-1, 1]]
                mid_points = np.unique([(sorted_attr_values[indx-1,0] + sorted_attr_values[indx,0]) / 2 for indx in change_indices])
            else:
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
 
    """
    
    type 0 – klasyfikacja jak w RuleKit  

    type 1 - jeśli reguła, która pokrywa przykład ma exception rule, która też go pokrywa, to ta reguła nie jest brana pod uwagę w głosowaniu (exception rule nic nie wnosi do głosowania, jedynie wyłącza commonsense rule)  

    type 2 - jeśli przykład pokrywany jest przez jakąkolwiek regułę wyjątków to w głosowaniu biorą udział jedynie reguły wyjątków. Jeżeli żadna reguła wyjątków nie pokrywa przykładu to głosują reguły commonsense 

    type 3 – jeżeli przykład jest pokrywany i przez regułę commonsense i przez regułę wyjątku to obydwie reguły biorą udział w głosowaniu (każda głosuje za klasą na którą wskazuje) 
    
    """
    def predict(self, X, type):
        prediction = []
        for i in range(X.shape[0]):
            example = X.iloc[i:i+1].to_numpy()
            result = self._predict_for_example(self.ruleset,example, type)
            prediction.append(result)
        return np.array(prediction)


    def _predict_for_example(self, ruleset, example, type):
        votes = defaultdict(int)
        votes_exception = defaultdict(int)
        rules = ruleset.rules
        if type == "0":
            for rule in rules:
                if rule.premise.covered_mask(example)[0]:
                    votes[rule.conclusion.value] += rule.voting_weight
        elif type == "1":
            for rule in rules:
                if rule.premise.covered_mask(example)[0]:
                    if rule.exception_rule != None:
                        if (not rule.exception_rule.premise.covered_mask(example)[0]) or rule.exception_rule.rule_type == "None":
                            votes[rule.conclusion.value] += rule.voting_weight  
                    else:
                        votes[rule.conclusion.value] += rule.voting_weight 
        elif type == "2":
            for rule in rules:
                if rule.premise.covered_mask(example)[0]:
                    votes[rule.conclusion.value] += rule.voting_weight
                    if rule.exception_rule != None:
                        if rule.exception_rule.premise.covered_mask(example)[0] and rule.exception_rule.rule_type != "None":
                            votes_exception[rule.exception_rule.conclusion.value] += rule.exception_rule.voting_weight 
            if len(votes_exception) > 0:
                votes = votes_exception

        else:
            for rule in rules:
                if rule.premise.covered_mask(example)[0]:
                    votes[rule.conclusion.value] += rule.voting_weight 
                    if rule.exception_rule != None:
                        if rule.exception_rule.premise.covered_mask(example)[0] and rule.exception_rule.rule_type != "None":
                            votes[rule.exception_rule.conclusion.value] += rule.exception_rule.voting_weight 

        if len(votes) > 0:
            prediction = max(votes, key=votes.get)
        else:
            prediction = ruleset.default_conclusion.value


        return prediction
