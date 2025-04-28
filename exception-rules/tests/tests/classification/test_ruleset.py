# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../../../../exception-rules')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../../../../decision-rules')))


from tests.loaders import load_ruleset, load_dataset

from decision_rules.problem import ProblemTypes

from exception_rules.classification import MyRuleClassifier

import warnings
warnings.filterwarnings('ignore')


class TestClassificationPredictionIndicators(unittest.TestCase):


    def test_iris(self):

        df = load_dataset("classification/iris.arff")
        # code to change encoding of the file
        tmp_df = df.select_dtypes([object])
        tmp_df = tmp_df.stack().str.decode("utf-8").unstack()
        for col in tmp_df:
            df[col] = tmp_df[col].replace({'?': None})
            
        X = df.drop(columns=["class"])
        y = df["class"]


        generator = MyRuleClassifier(mincov=5, induction_measuer="c2")


        model = generator.fit(X , y)
        ruleset = model.ruleset

        ruleset_gt = load_ruleset("classification/iris_ruleset.json", ProblemTypes.CLASSIFICATION)

        self.assertEqual(
            ruleset, ruleset_gt,
            'Rulesets should be the same'
        )


    def test_mushroom(self):

        df = load_dataset("classification/mushroom.arff")
        # code to change encoding of the file
        tmp_df = df.select_dtypes([object])
        tmp_df = tmp_df.stack().str.decode("utf-8").unstack()
        for col in tmp_df:
            df[col] = tmp_df[col].replace({'?': None})
            
        X = df.drop(columns=["class"])
        y = df["class"]


        generator = MyRuleClassifier(mincov=5, induction_measuer="c2")


        model = generator.fit(X , y)
        ruleset = model.ruleset

        ruleset_gt = load_ruleset("classification/mushroom_ruleset.json", ProblemTypes.CLASSIFICATION)

        self.assertEqual(
            ruleset, ruleset_gt,
            'Rulesets should be the same'
        )


