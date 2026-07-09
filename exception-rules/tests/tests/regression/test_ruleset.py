# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../../../../exception-rules')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../../../../decision-rules')))


from tests.loaders import load_ruleset, load_dataset

from decision_rules.problem import ProblemTypes

from exception_rules.regression.algorithm import MyRuleRegressor

import warnings
warnings.filterwarnings('ignore')


class TestRegressionPredictionIndicators(unittest.TestCase):


    def test_concrete(self):

        df = load_dataset("regression/concrete.arff")
        # code to change encoding of the file
        tmp_df = df.select_dtypes([object])
        if tmp_df.shape[1] > 0:
            tmp_df = tmp_df.stack().str.decode("utf-8").unstack()
            for col in tmp_df:
                df[col] = tmp_df[col].replace({'?': None})
            
        X = df.drop(columns=["class"])
        y = df["class"]


        generator = MyRuleRegressor(mincov=5, induction_measuer="c2", prune=False, find_exceptions=True, max_growing=5)


        model = generator.fit(X , y)
        ruleset = model.ruleset

        ruleset_gt = load_ruleset("regression/concrete_ruleset.json", ProblemTypes.REGRESSION)

        self.assertEqual(
            ruleset, ruleset_gt,
            'Rulesets should be the same'
        )


    def test_bodyfat(self):

        df = load_dataset("regression/bodyfat.arff")
        # code to change encoding of the file
        tmp_df = df.select_dtypes([object])
        if tmp_df.shape[1] > 0:
            tmp_df = tmp_df.stack().str.decode("utf-8").unstack()
            for col in tmp_df:
                df[col] = tmp_df[col].replace({'?': None})
            
        X = df.drop(columns=["class"])
        y = df["class"]


        generator = MyRuleRegressor(mincov=5, induction_measuer="c2", prune=False, find_exceptions=True, max_growing=5)


        model = generator.fit(X , y)
        ruleset = model.ruleset

        ruleset_gt = load_ruleset("regression/bodyfat_ruleset.json", ProblemTypes.REGRESSION)

        self.assertEqual(
            ruleset, ruleset_gt,
            'Rulesets should be the same'
        )
