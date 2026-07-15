import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'exception-rules')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'decision-rules')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'rulekit-algorithm')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', 'scikit-algorithm')))


import numpy as np
import pandas as pd
import time
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score
import psutil
from scipy.io import arff
from sklearn.model_selection import train_test_split

from exception_rules.decision_rules.measures import *


import warnings
warnings.filterwarnings('ignore')


from exception_rules.classification.algorithm_v2 import MyRuleClassifier


import logging

from exception_rules.measures import *
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from classification_tree_algorithm import DecisionTreeClassificationAlgorithm

from rulekit_algorithm_classification import RuleKitAlgorithm

class Experiment:


    def __init__(self):
        self.logger = logging.getLogger("my_logger")
        self.logger.setLevel(logging.INFO)

    def _get_models(self):

      
        algorithm_cer = MyRuleClassifier(mincov=5, induction_measuer="c2", max_growing = 5, find_exceptions=False, logger = self.logger)
        algorithm_cer_with_exceptions = MyRuleClassifier(mincov=5, induction_measuer="c2", max_growing = 5, find_exceptions=True, logger = self.logger)
        algorithm_rulekit = RuleKitAlgorithm(minsupp_new = 5, max_growing = 5)
        decision_tree = DecisionTreeClassificationAlgorithm(max_depth=5)


        return {
                "algorithm_rulekit": algorithm_rulekit,
                "algorithm_cer": algorithm_cer,
                "algorithm_cer_with_exceptions": algorithm_cer_with_exceptions,
                "decision_tree": decision_tree

                }


    
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
        if model_name == "decision_tree":
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
        # measures.to_csv(results_path + "exceptions_details_ALL.csv", index=False)

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
                if dataset != "anneal.arff":
                    df[col] = tmp_df[col].replace({'?': None})
                else:
                    df[col] = tmp_df[col]
        self.X = df.drop(columns=["class"])
        self.y = df["class"]


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



        results_df = self._get_results_for_algorithm(model, model_name, results,self.X_train, self.y_train, self.X_test, self.y_test, dataset)

        evauluation_df = None


        results_df.to_csv(
            self.results_path + "exceptions_summary.csv", index=False, header=False, mode="a"
        )

        self.logger.info(f"************************Finished experiment for {dataset} with {dataset_with_model['model_name']}************************")
        self.logger.info(f"*************************************************************************************************************************")
        self.logger.info(f"*************************************************************************************************************************")
        self.logger.info(f"*************************************************************************************************************************")
        self.logger.removeHandler(file_handler)
        return results_df, evauluation_df
    


    def _get_results_for_algorithm(self, model, model_name, results, X_train, y_train, X_test, y_test, dataset):

        if model_name == "decision_tree" or model_name == "algorithm_rulekit":
            if model_name == "decision_tree":
                stats = model.get_rules_statistics(simplify_onehot=True)
                results["liczba_regul"] = stats["liczba_regul"]
                results["liczba_wyjatkow"] = 0
                results["srednia_dlugosc_reguly"] = stats["srednia_dlugosc_reguly"]
                results["suma_warunkow"] = stats["suma_warunkow"]
                results["avg_precision"] = 0
                results["avg_coverage"] = 0

            elif model_name == "algorithm_rulekit":
                stats = self._get_stats(model, model_name) 
                results["liczba_regul"] = stats["liczba_regul"]
                results["liczba_wyjatkow"] = stats["liczba_wyjatkow"]
                results["srednia_dlugosc_reguly"] = stats["srednia_dlugosc_reguly"]
                results["suma_warunkow"] = stats["suma_warunkow"]
                results["avg_precision"] = stats["avg_precision"]
                results["avg_coverage"] = stats["avg_coverage"]

            train_predictions = model.predict(X_train)
            test_predictions = model.predict(X_test)

            results["balanced_accuracy__train"] = balanced_accuracy_score(y_train, train_predictions)
            results["accuracy__train"] = accuracy_score(y_train, train_predictions)
            results["recall__train"] = recall_score(y_train, train_predictions, average="macro")
            results["precision__train"] = precision_score(y_train, train_predictions, average="macro")
            results["f1_score__train"] = f1_score(y_train, train_predictions, average="macro")

            results["balanced_accuracy__test"] = balanced_accuracy_score(y_test, test_predictions)
            results["accuracy__test"] = accuracy_score(y_test, test_predictions)
            results["recall__test"] = recall_score(y_test, test_predictions, average="macro")
            results["precision__test"] = precision_score(y_test, test_predictions, average="macro")
            results["f1_score__test"] = f1_score(y_test, test_predictions, average="macro")

            self._save_rules(model, model_name, dataset, X_train, y_train)

            results_df = pd.DataFrame(results, index=[0])


        else:
            stats = self._get_stats(model.ruleset, model_name) 
            results["liczba_regul"] = stats["liczba_regul"]
            results["liczba_wyjatkow"] = stats["liczba_wyjatkow"]
            results["srednia_dlugosc_reguly"] = stats["srednia_dlugosc_reguly"]
            results["suma_warunkow"] = stats["suma_warunkow"]
            results["avg_precision"] = stats["avg_precision"]
            results["avg_coverage"] = stats["avg_coverage"]

            self._save_rules(model.ruleset, model_name, dataset, X_train, y_train)

            model_name = results["model"]


            if model_name == "algorithm_cer_with_exceptions":

                for prediction_type in ["0", "1", "2", "3"]:
                    train_predictions = model.predict(X_train, type = prediction_type)
                    test_predictions = model.predict(X_test, type = prediction_type)
                    results["balanced_accuracy__train"] = balanced_accuracy_score(y_train, train_predictions)
                    results["accuracy__train"] = accuracy_score(y_train, train_predictions)
                    results["recall__train"] = recall_score(y_train, train_predictions, average="macro")
                    results["precision__train"] = precision_score(y_train, train_predictions, average="macro")
                    results["f1_score__train"] = f1_score(y_train, train_predictions, average="macro")

                    results["balanced_accuracy__test"] = balanced_accuracy_score(y_test, test_predictions)
                    results["accuracy__test"] = accuracy_score(y_test, test_predictions)
                    results["recall__test"] = recall_score(y_test, test_predictions, average="macro")
                    results["precision__test"] = precision_score(y_test, test_predictions, average="macro")
                    results["f1_score__test"] = f1_score(y_test, test_predictions, average="macro")

                    
                    if prediction_type == "0":
                        results["model"] = model_name + "_prediction_type_" + prediction_type
                        results_df = pd.DataFrame(results, index=[0])
                    else:
                        results["model"] = model_name + "_prediction_type_" + prediction_type
                        temp_df = pd.DataFrame(results, index=[0])
                        results_df = pd.concat([results_df, temp_df], axis=0)
                    
            else:
                    train_predictions = model.ruleset.predict(X_train)
                    test_predictions = model.ruleset.predict(X_test)
                    results["balanced_accuracy__train"] = balanced_accuracy_score(y_train, train_predictions)
                    results["accuracy__train"] = accuracy_score(y_train, train_predictions)
                    results["recall__train"] = recall_score(y_train, train_predictions, average="macro")
                    results["precision__train"] = precision_score(y_train, train_predictions, average="macro")
                    results["f1_score__train"] = f1_score(y_train, train_predictions, average="macro")

                    results["balanced_accuracy__test"] = balanced_accuracy_score(y_test, test_predictions)
                    results["accuracy__test"] = accuracy_score(y_test, test_predictions)
                    results["recall__test"] = recall_score(y_test, test_predictions, average="macro")
                    results["precision__test"] = precision_score(y_test, test_predictions, average="macro")
                    results["f1_score__test"] = f1_score(y_test, test_predictions, average="macro")

                    results_df = pd.DataFrame(results, index=[0])

        

        return results_df

if __name__ == "__main__":


    datasets_path = "../../data_test/classification/train_test/"
    results_path = f"./results/comparision/"

    os.makedirs(results_path, exist_ok=True)

    experiment = Experiment()


    experiment.run_experiments(datasets_path, results_path)
