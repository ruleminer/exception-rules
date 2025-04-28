import json
import os
from typing import Tuple, Optional

import pandas as pd
from decision_rules.classification.ruleset import ClassificationRuleSet
from decision_rules.core.ruleset import AbstractRuleSet
from decision_rules.problem import ProblemTypes
from decision_rules.regression.ruleset import RegressionRuleSet
from decision_rules.serialization import JSONSerializer
from decision_rules.survival.ruleset import SurvivalRuleSet
from scipy.io import arff

def deserialize_ruleset(ruleset: dict, problem_type: ProblemTypes) -> ClassificationRuleSet:
    PROBLEM_TYPE_MAPPING = {
        ProblemTypes.CLASSIFICATION: ClassificationRuleSet,
        ProblemTypes.REGRESSION: RegressionRuleSet,
        ProblemTypes.SURVIVAL: SurvivalRuleSet
    }
    return JSONSerializer.deserialize(
        ruleset,
        PROBLEM_TYPE_MAPPING[problem_type]
    )


def load_ruleset(path: str, problem_type: ProblemTypes) -> AbstractRuleSet:
    ruleset_file_path: str = os.path.join(load_resources_path(), path)
    with open(ruleset_file_path, 'r', encoding='utf-8') as file:
        return deserialize_ruleset(json.load(file), problem_type)

def load_dataset(path: str) -> pd.DataFrame:
    dataset_file_path: str = os.path.join(load_resources_path(), path)
    return pd.DataFrame(arff.loadarff(dataset_file_path)[0])

def load_resources_path() -> str:
    """Return path to resources directory"""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(dir_path, 'resources')
