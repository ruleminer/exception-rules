import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'exception-rules')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'decision-rules')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'rulekit-algorithm')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'scikit-algorithm')))

import numpy as np
import pandas as pd
import time
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import psutil
from scipy.io import arff
from sklearn.model_selection import train_test_split

from exception_rules.decision_rules.measures import *


import warnings
warnings.filterwarnings('ignore')


from exception_rules.survival.algorithm import ExceptionRulesSurvival

import logging

from exception_rules.measures import *
import matplotlib.pyplot as plt
from exception_rules.decision_rules.survival.kaplan_meier import KaplanMeierEstimator

from rulekit_algorithm import RuleKitAlgorithm

from lifelines import CoxPHFitter
from lifelines.utils import restricted_mean_survival_time
from lifelines import KaplanMeierFitter

from survival_tree_algorithm import SurvivalTreeAlgorithm

class Experiment:


    def __init__(self):
        self.logger = logging.getLogger("my_logger")
        self.logger.setLevel(logging.INFO)

    def _get_models(self):

        algorithm_log_rank = ExceptionRulesSurvival(mincov=5, survival_time_attr="survival_time", max_growing = 5, find_exceptions=False, logger = self.logger)
        algorithm_log_rank_with_exceptions = ExceptionRulesSurvival(mincov=5, survival_time_attr="survival_time", max_growing = 5, find_exceptions=True, logger = self.logger)
        algorithm_rulekit = RuleKitAlgorithm(survival_time_attr="survival_time", minsupp_new = 5, max_growing = 5)
        survival_tree = SurvivalTreeAlgorithm(survival_time_col="survival_time", max_depth=5)


        return {

                "algorithm_rulekit": algorithm_rulekit,
                "algorithm_log_rank": algorithm_log_rank,
                "algorithm_log_rank_with_exceptions": algorithm_log_rank_with_exceptions,
                "survival_tree": survival_tree

                }
        # return {
        #         "algorithm_log_rank_with_exceptions": algorithm_log_rank_with_exceptions,
        #         "algorithm_effect_size_with_exceptions": algorithm_effect_size_with_exceptions,
        #         }   
    
    def _get_stats(self, model, model_name):

        rules = model.rules
        stats = dict()

        liczb_wyjatkow = 0
        for rule in rules:
            if rule.exception_rule is not None:
                liczb_wyjatkow += 1
        stats["liczba_wyjatkow"] = liczb_wyjatkow



        conditions_sum = 0
        for rule in rules:
            rule = str(rule)
            preimse, consequence = rule.split("THEN")
            conditions = preimse.split("AND")
            conditions_sum += len(conditions)

        stats["liczba_regul"] = len(rules)
        stats["srednia_dlugosc_reguly"] = conditions_sum / len(rules)
        stats["suma_warunkow"] = conditions_sum

        model_stats = model.calculate_ruleset_stats()
        stats["avg_precision"] = model_stats["avg_precision"]
        stats["avg_coverage"] = model_stats["avg_coverage"]

        return stats



    def _save_rules(self, model, model_name, dataset, X_train, y_train, type=None):
        if type is None:
            path = self.results_path + f"{dataset[:-5]}/{model_name}/"
        else:
            path = self.results_path + f"{dataset[:-5]}/{model_name}_{type}/"
        os.makedirs(path, exist_ok=True)
        if model_name == "survival_tree" or model_name == "survival_forest":
            rules = model.get_rules_text()
            with open(path + "rules.txt", "w+") as file:
                for i, rule in enumerate(rules):
                    file.write(f"\nCR {i}: {str(rule)}\n")
        else:
            rules = model.rules
            with open(path + "rules.txt", "w+") as file:
                for i, rule in enumerate(rules):
                    file.write(f"\nCR {i}: {str(rule)}\n")
                    if rule.exception_rule is not None:
                        file.write(f"RR {i}: {rule.reference_rule}\n")
                        file.write(f"ER {i}: {rule.exception_rule}\n")
                        self._draw_estymator(rule, i, path, X_train, y_train)

    def _draw_estymator(self, rule, i, path, X, y):    
        
        comonsense_rules = rule
        reference_rule = rule.reference_rule
        exception_rule = rule.exception_rule


        survival_time = X["survival_time"].to_numpy()
        
        X = X.to_numpy()
        y = y.to_numpy()
        
        survival_status = y

        er_covered = np.where(exception_rule.positive_covered_mask(X, y) == 1)[0]
        cr_covered = np.where(comonsense_rules.positive_covered_mask(X, y) == 1)[0]
        rr_covered = np.where(reference_rule.positive_covered_mask(X, y) == 1)[0]

        er_uncovered = np.where(exception_rule.positive_covered_mask(X, y) == 0)[0]
        cr_uncovered = np.where(comonsense_rules.positive_covered_mask(X, y) == 0)[0]
        rr_uncovered = np.where(reference_rule.positive_covered_mask(X, y) == 0)[0]

 

        er_km_covered = KaplanMeierEstimator().fit(survival_time[er_covered], survival_status[er_covered], update_additional_informations=False)
        cr_km_covered = KaplanMeierEstimator().fit(survival_time[cr_covered], survival_status[cr_covered], update_additional_informations=False)
        rr_km_covered = KaplanMeierEstimator().fit(survival_time[rr_covered], survival_status[rr_covered], update_additional_informations=False)

        er_km_uncovered = KaplanMeierEstimator().fit(survival_time[er_uncovered], survival_status[er_uncovered], update_additional_informations=False)
        cr_km_uncovered = KaplanMeierEstimator().fit(survival_time[cr_uncovered], survival_status[cr_uncovered], update_additional_informations=False)
        rr_km_uncovered = KaplanMeierEstimator().fit(survival_time[rr_uncovered], survival_status[rr_uncovered], update_additional_informations=False)

        all_example_km = KaplanMeierEstimator().fit(survival_time, survival_status, update_additional_informations=False)


        plt.figure(figsize=(10, 5))
        plt.plot(all_example_km.times, all_example_km.probabilities, label="All examples")
        plt.plot(er_km_covered.times, er_km_covered.probabilities, label="ER")
        plt.plot(cr_km_covered.times, cr_km_covered.probabilities, label="CR")
        plt.plot(rr_km_covered.times, rr_km_covered.probabilities, label="RR")
        plt.legend()

        plt.savefig(path + f"estymatory_{i}.png")

    def evaluate_exceptions(self, rule, X, y, measure = "log_rank"):

        survival_time = X["survival_time"].to_numpy()
        survival_status = y.to_numpy()

        X = X.to_numpy()
        y = y.to_numpy()

        comonsense_rule = rule
        reference_rule = rule.reference_rule
        exception_rule = rule.exception_rule

        cr_covered = np.where(comonsense_rule.positive_covered_mask(X, y) == 1)[0]
        rr_covered = np.where(reference_rule.positive_covered_mask(X, y) == 1)[0]
        er_covered = np.where(exception_rule.positive_covered_mask(X, y) == 1)[0]

        survival_time_cr = survival_time[cr_covered]
        survival_status_cr = survival_status[cr_covered]
        survival_time_rr = survival_time[rr_covered]
        survival_status_rr = survival_status[rr_covered]
        survival_time_er = survival_time[er_covered]
        survival_status_er = survival_status[er_covered]


        df_er = pd.DataFrame({
            "time": survival_time_er,
            "event": survival_status_er,
            "group": 0
        })

        df_cr = pd.DataFrame({
            "time": survival_time_cr,
            "event": survival_status_cr,
            "group": 1
        })
        
        df_rr = pd.DataFrame({
            "time": survival_time_rr,
            "event": survival_status_rr,
            "group": 2
        })

        df_cr_er = pd.concat([df_cr, df_er], ignore_index=True)

        df_rr_er = pd.concat([df_rr, df_er], ignore_index=True)


        df_cr_rr = pd.concat([df_cr, df_rr], ignore_index=True)

        if measure == "log_rank":
            cr_estimator  = KaplanMeierEstimator().fit(survival_time[cr_covered], survival_status[cr_covered], update_additional_informations=False)
            rr_estimator  = KaplanMeierEstimator().fit(survival_time[rr_covered], survival_status[rr_covered], update_additional_informations=False)

    
            er_estimator  = KaplanMeierEstimator().fit(survival_time[er_covered], survival_status[er_covered], update_additional_informations=False)

            stats_and_pvalue_cr_rr = KaplanMeierEstimator().compare_estimators(
                cr_estimator, rr_estimator)
            
            stats_and_pvalue_cr_er = KaplanMeierEstimator().compare_estimators(
                cr_estimator, er_estimator)
            
            stats_and_pvalue_rr_er = KaplanMeierEstimator().compare_estimators(
                rr_estimator, er_estimator)
            
            return stats_and_pvalue_cr_er["p_value"], stats_and_pvalue_rr_er["p_value"], stats_and_pvalue_cr_rr["p_value"]
        
        if measure == "effect_size":
            cph_cr_er = CoxPHFitter()
            cph_cr_er.fit(df_cr_er, duration_col="time", event_col="event")
            hr_cr_er = cph_cr_er.hazard_ratios_["group"]
            es_cr_er = abs(np.log(hr_cr_er))

            cph_rr_er = CoxPHFitter()
            cph_rr_er.fit(df_rr_er, duration_col="time", event_col="event")
            hr_rr_er = cph_rr_er.hazard_ratios_["group"]
            es_rr_er = abs(np.log(hr_rr_er))

            cph_cr_rr = CoxPHFitter()
            cph_cr_rr.fit(df_cr_rr, duration_col="time", event_col="event")
            hr_cr_rr = cph_cr_rr.hazard_ratios_["group"]            
            es_cr_rr = abs(np.log(hr_cr_rr))

            return es_cr_er, es_rr_er, es_cr_rr
        
        if measure == "rmst":
            tau = np.min([np.max(cr_covered), np.max(er_covered), np.max(rr_covered)])

            km_cr = KaplanMeierFitter()
            km_cr.fit(df_cr["time"],df_cr["event"])

            km_er = KaplanMeierFitter()
            km_er.fit(df_er["time"],df_er["event"])

            km_rr = KaplanMeierFitter()
            km_rr.fit(df_rr["time"],df_rr["event"])

            rmst_cr = restricted_mean_survival_time(km_cr, t=tau)
            rmst_er  = restricted_mean_survival_time(km_er, t=tau)
            rmst_rr  = restricted_mean_survival_time(km_rr, t=tau)

            rmst_cr_er = abs(rmst_cr - rmst_er)
            rmst_rr_er = abs(rmst_rr - rmst_er)
            rmst_cr_rr = abs(rmst_cr - rmst_rr)


            return rmst_cr_er, rmst_rr_er, rmst_cr_rr


    def _evaluate_rules(self, ruleset, dataset, X, y):

        rules = ruleset.rules

        evaluation_df = pd.DataFrame()

        rules_dict = dict()
        CR_number = 0
        for rule in rules:
            if rule.exception_rule is not None: 
                
                rules_dict["dataset"] = dataset
                rules_dict["CR_number"] = CR_number

                rules_dict["CR"] = str(rule)
                rules_dict["RR"] = str(rule.reference_rule)
                rules_dict["ER"] = str(rule.exception_rule)


                # rules_dict["GACE"] = calculate_GACE(rule, exception_rule, return_ACE = False)
                # rules_dict["RI"] = calculate_RI(rule, exception_rule, reference_rule)
                # rules_dict["MY_MEASURE"] = calculate_my_measure(rule, exception_rule)

                log_rank_CR_ER, log_rank_RR_ER, log_rank_CR_RR = self.evaluate_exceptions(rule, X, y, measure="log_rank")
                effect_size_CR_ER, effect_size_RR_ER, effect_size_CR_RR = self.evaluate_exceptions(rule, X, y, measure="effect_size")
                rmst_CR_ER, rmst_RR_ER, rmst_CR_RR = self.evaluate_exceptions(rule, X, y, measure="rmst")

                rules_dict["log_rank_p_value_CR_ER"] = log_rank_CR_ER
                rules_dict["log_rank_p_value_RR_ER"] = log_rank_RR_ER       
                rules_dict["log_rank_p_value_CR_RR"] = log_rank_CR_RR
                rules_dict["effect_size_CR_ER"] = effect_size_CR_ER
                rules_dict["effect_size_RR_ER"] = effect_size_RR_ER
                rules_dict["effect_size_CR_RR"] = effect_size_CR_RR
                rules_dict["rmst_CR_ER"] = rmst_CR_ER
                rules_dict["rmst_RR_ER"] = rmst_RR_ER
                rules_dict["rmst_CR_RR"] = rmst_CR_RR


                evaluation_df_tmp = pd.DataFrame(rules_dict, index=[0])

                evaluation_df = pd.concat([evaluation_df, evaluation_df_tmp], axis=0)
              
                CR_number+=1

        if evaluation_df.shape[0] == 0:
            return None
        
        return evaluation_df

    def run_experiments(self, datasets_path: str, results_path: str):
       
        self.results_path = results_path

        results = pd.DataFrame()
        
        self.datasets_path = datasets_path
        datasets = sorted(os.listdir(datasets_path))

        datasets_with_models = list()
        for dataset in datasets:
            models = self._get_models()
            for model_name, model in models.items():
                pair = {"dataset": dataset,
                        "model_name": model_name,
                        "model": model} 
                datasets_with_models.append(pair)

        results = pd.DataFrame()
        measures = pd.DataFrame()
        for dataset_with_model in datasets_with_models:
            experiment_result, evauluation_df = self._run_experiment(datasets_path, dataset_with_model)
            results = pd.concat([results, experiment_result], axis=0)
            if evauluation_df is not None:
                measures = pd.concat([measures, evauluation_df], axis=0)

        results.to_csv(results_path + "exceptions_summary_ALL.csv", index=False)
        measures.to_csv(results_path + "exceptions_details_ALL.csv", index=False)

    def _run_experiment(self,datasets_path:str, dataset_with_model: dict) -> pd.DataFrame:
        dataset = dataset_with_model["dataset"]
        model = dataset_with_model["model"]
        model_name = dataset_with_model["model_name"]

       
        logging_path = self.results_path + dataset[:-5]
        os.makedirs(logging_path, exist_ok=True)
        # Tworzymy i konfigurujemy FileHandler
        file_handler = logging.FileHandler(logging_path +'/log.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        self.logger.addHandler(file_handler)


        self.logger.info(f"**********Starting experiment for {dataset} with {dataset_with_model['model_name']}**********")

        df = pd.DataFrame(arff.loadarff(f"{datasets_path}{dataset}")[0])
        # code to change encoding of the file
        tmp_df = df.select_dtypes([object])
        if len(tmp_df.columns) > 0:
            tmp_df = tmp_df.stack().str.decode("utf-8").unstack()
            for col in tmp_df:
                df[col] = tmp_df[col].replace({'?': None})

        if "group" in df.columns:
            df = df.drop(columns=["group"])

        self.X = df.drop(columns=["survival_status"])
        self.y = df["survival_status"].astype(int).astype(str)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)


        results = dict()

        print(f"{dataset}: {model_name}")
        start_time_process = time.process_time()
        start_time_thread = time.thread_time()
        start_time_raw = time.time()
        model = model.fit(self.X_train, self.y_train)
        end_time_process = time.process_time()
        end_time_thread = time.thread_time()
        end_time_raw = time.time()

        results["dataset"] = dataset
        results["model"] = model_name
        results["thread_time"] = end_time_thread - start_time_thread
        results["process_time"] = end_time_process - start_time_process
        results["raw_time"] = end_time_raw - start_time_raw

        if model_name== "algorithm_rulekit":
            ruleset = model 
        elif model_name == "survival_tree" or model_name == "survival_forest":
            ruleset = model
        else:
            ruleset = model.ruleset


        results_df = self._get_results_for_algorithm(ruleset, model_name, results, self.X_train, self.y_train, self.X_test, self.y_test, dataset)

        evauluation_df = None

        # evauluation_df = self._evaluate_rules(ruleset, dataset, self.X, self.y)

        # if evauluation_df is not None:
        #     evauluation_df.to_csv(
        #         self.results_path + "exceptions_details.csv", index=False, header=False, mode="a"
        #     )


        results_df.to_csv(
            self.results_path + "exceptions_summary.csv", index=False, header=False, mode="a"
        )

        self.logger.info(f"************************Finished experiment for {dataset} with {dataset_with_model['model_name']}************************")
        self.logger.info(f"*************************************************************************************************************************")
        self.logger.info(f"*************************************************************************************************************************")
        self.logger.info(f"*************************************************************************************************************************")
        self.logger.removeHandler(file_handler)
        return results_df, evauluation_df
    


    def _get_results_for_algorithm(self, ruleset, model_name, results, X_train, y_train, X_test, y_test, dataset):

        if model_name != "survival_tree" and model_name != "survival_forest":
            stats = self._get_stats(ruleset, model_name) 
            results["liczba_regul"] = stats["liczba_regul"]
            results["liczba_wyjatkow"] = stats["liczba_wyjatkow"]
            results["srednia_dlugosc_reguly"] = stats["srednia_dlugosc_reguly"]
            results["suma_warunkow"] = stats["suma_warunkow"]
            results["avg_coverage"] = stats["avg_coverage"]

            results["ibs_train"] = ruleset.integrated_bier_score(X_train, y_train)
            results["ibs_test"] = ruleset.integrated_bier_score(X_test, y_test)

            self._save_rules(ruleset, model_name, dataset, X_train, y_train)

        else:
            
           
            stats = ruleset.get_rules_statistics(simplify_onehot=True)
            results["liczba_regul"] = stats["liczba_regul"]
            results["liczba_wyjatkow"] = 0
            results["srednia_dlugosc_reguly"] = stats["srednia_dlugosc_reguly"]
            results["suma_warunkow"] = stats["suma_warunkow"]
            results["avg_coverage"] = 0
            results["ibs_train"] = ruleset.integrated_brier_score(X_train, y_train)
            results["ibs_test"] = ruleset.integrated_brier_score(X_test, y_test)

            self._save_rules(ruleset, model_name, dataset, X_train, y_train)


        results_df = pd.DataFrame(results, index=[0])

        return results_df

if __name__ == "__main__":


    datasets_path = "../../data/survival/train_test/"
    results_path = f"./results/comparision/"

    os.makedirs(results_path, exist_ok=True)

    experiment = Experiment()


    experiment.run_experiments(datasets_path, results_path)
