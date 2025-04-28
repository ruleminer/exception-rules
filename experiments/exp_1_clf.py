import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'exception-rules')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'decision-rules')))

import numpy as np
import pandas as pd
import time
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import psutil
from scipy.io import arff
from sklearn.model_selection import train_test_split

from decision_rules.measures import *


import warnings
warnings.filterwarnings('ignore')

from exception_rules.classification.algorithm import MyRuleClassifier 

import logging

from exception_rules.measures import *

class Experiment:


    def __init__(self):
        self.logger = logging.getLogger("my_logger")
        self.logger.setLevel(logging.INFO)

    def _get_models(self):


        algorithm3 = MyRuleClassifier(mincov=5, induction_measuer="c2", logger=self.logger, find_exceptions=True)

        return {"algorithm3": algorithm3}
    
    
    def _get_stats(self, model, model_name):

        rules = model.rules
        stats = dict()

        stats["liczba_wyjatkow"] = 0
        stats["liczba_wyjatkow_type_1"] = 0
        stats["liczba_wyjatkow_type_2"] = 0
        stats["liczba_wyjatkow_AR"] = 0
        stats["liczba_wyjatkow_None"] = 0
        for rule in rules:
            if len(rule.exception_rules) > 0:
                for exception_rule in rule.exception_rules:
                    stats["liczba_wyjatkow"] = stats.get("liczba_wyjatkow", 0) + 1
                    if exception_rule.rule_type == '1':
                        stats["liczba_wyjatkow_type_1"] = stats.get("liczba_wyjatkow_type_1", 0) + 1
                    elif exception_rule.rule_type == '2':
                        stats["liczba_wyjatkow_type_2"] = stats.get("liczba_wyjatkow_type_2", 0) + 1
                    elif exception_rule.rule_type == 'AR':
                        stats["liczba_wyjatkow_AR"] = stats.get("liczba_wyjatkow_AR", 0) + 1
                    else:
                        stats["liczba_wyjatkow_None"] = stats.get("liczba_wyjatkow_None", 0) + 1



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



    def _save_rules(self, model, model_name, dataset, type=None):
        if type is None:
            path = self.results_path + f"{dataset[:-5]}/{model_name}/"
        else:
            path = self.results_path + f"{dataset[:-5]}/{model_name}_{type}/"
        os.makedirs(path, exist_ok=True)
        if model_name.split("_")[0] == "rulekit":
            print("Brak obsługi RuleKit")
            # factory = RuleKitRuleSetFactory()
            # decision_rules_ruleset = factory.make(model, self.X_train, self.y_train)
            # rules = decision_rules_ruleset.rules
            # with open(path + "rules.txt", "w+") as file:
            #     for rule in rules:
            #         file.write(f"{str(rule)}\n")
        else:
            rules = model.rules
            with open(path + "rules.txt", "w+") as file:
                for rule in rules:
                    file.write(f"\nCR: {str(rule)}\n")
                    if len(rule.exception_rules) > 0:
                        i=0
                        for exception_rule, rr_rule in zip(rule.exception_rules, rule.reference_rules):
                            file.write(f"RR_{i}: {str(rr_rule)}\n")
                            file.write(f"ER_{i}, type {exception_rule.rule_type}: {str(exception_rule)}\n")
                            i+=1
        

    def _evaluate_rules(self, ruleset, dataset):

        rules = ruleset.rules

        evaluation_df = pd.DataFrame()

        rules_dict = dict()
        CR_number = 0
        for rule in rules:
            if len(rule.exception_rules) > 0: 
                ER_number = 0
                for exception_rule, reference_rule in zip(rule.exception_rules, rule.reference_rules):
                    rules_dict["dataset"] = dataset
                    rules_dict["CR_number"] = CR_number
                    rules_dict["ER_number"] = ER_number
                    rules_dict["CR"] = str(rule)
                    rules_dict["RR"] = str(reference_rule)
                    rules_dict["ER"] = str(exception_rule)
                    rules_dict["ER_type"] = exception_rule.rule_type

                    rules_dict["GACE"] = calculate_GACE(rule, exception_rule, return_ACE = False)
                    rules_dict["RI"] = calculate_RI(rule, exception_rule, reference_rule)
                    rules_dict["MY_MEASURE"] = calculate_my_measure(rule, exception_rule)

                    evaluation_df_tmp = pd.DataFrame(rules_dict, index=[0])

                    evaluation_df = pd.concat([evaluation_df, evaluation_df_tmp], axis=0)

                    ER_number+=1
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
        tmp_df = tmp_df.stack().str.decode("utf-8").unstack()
        for col in tmp_df:
            if dataset != "anneal.arff":
                df[col] = tmp_df[col].replace({'?': None})
            else:
                df[col] = tmp_df[col]

        self.X = df.drop(columns=["class"])
        self.y = df["class"]



        results = dict()

        print(f"{dataset}: {model_name}")
        start_time_process = time.process_time()
        start_time_thread = time.thread_time()
        start_time_raw = time.time()
        model = model.fit(self.X, self.y)
        end_time_process = time.process_time()
        end_time_thread = time.thread_time()
        end_time_raw = time.time()

        results["dataset"] = dataset
        results["model"] = model_name
        results["thread_time"] = end_time_thread - start_time_thread
        results["process_time"] = end_time_process - start_time_process
        results["raw_time"] = end_time_raw - start_time_raw


        results_df = self._get_results_for_algorithm(model, model_name, results, dataset)

        evauluation_df = self._evaluate_rules(model.ruleset, dataset)

        if evauluation_df is not None:
            evauluation_df.to_csv(
                self.results_path + "exceptions_details.csv", index=False, header=False, mode="a"
            )


        results_df.to_csv(
            self.results_path + "exceptions_summary.csv", index=False, header=False, mode="a"
        )

        self.logger.info(f"************************Finished experiment for {dataset} with {dataset_with_model['model_name']}************************")
        self.logger.info(f"*************************************************************************************************************************")
        self.logger.info(f"*************************************************************************************************************************")
        self.logger.info(f"*************************************************************************************************************************")
        self.logger.removeHandler(file_handler)
        return results_df, evauluation_df
    


    def _get_results_for_algorithm(self, model, model_name, results, dataset):
        ruleset = model.ruleset
        stats = self._get_stats(ruleset, model_name) 
        results["liczba_regul"] = stats["liczba_regul"]
        results["liczba_wyjatkow"] = stats["liczba_wyjatkow"]
        results["liczba_wyjatkow_type_1"] = stats["liczba_wyjatkow_type_1"]
        results["liczba_wyjatkow_type_2"] = stats["liczba_wyjatkow_type_2"]
        results["liczba_wyjatkow_AR"] = stats["liczba_wyjatkow_AR"]
        results["liczba_wyjatkow_None"] = stats["liczba_wyjatkow_None"]
        results["srednia_dlugosc_reguly"] = stats["srednia_dlugosc_reguly"]
        results["suma_warunkow"] = stats["suma_warunkow"]
        results["avg_precision"] = stats["avg_precision"]
        results["avg_coverage"] = stats["avg_coverage"]

        self._save_rules(ruleset, model_name, dataset)

        results_df = pd.DataFrame(results, index=[0])

        return results_df

if __name__ == "__main__":


    datasets_path = "../data/classification/train_test/"
    results_path = f"./results/exp_1_clf/"

    os.makedirs(results_path, exist_ok=True)

    experiment = Experiment()


    experiment.run_experiments(datasets_path, results_path)
