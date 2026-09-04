# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../../../../exception-rules')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../../../../decision-rules')))


from tests.loaders import load_ruleset, load_dataset

from exception_rules.decision_rules.problem import ProblemTypes

from exception_rules.survival.algorithm import ExceptionRulesSurvival

import warnings
warnings.filterwarnings('ignore')


class TestSurvivalPredictionIndicators(unittest.TestCase):
    """Compare induced survival rules with stored reference rule sets."""


    def test_gbsg2(self):
        """Induce the expected survival rules for the GBSG2 dataset."""

        df = load_dataset("survival/18_GBSG2.arff")
        # code to change encoding of the file
        tmp_df = df.select_dtypes([object])
        tmp_df = tmp_df.stack().str.decode("utf-8").unstack()
        for col in tmp_df:
            df[col] = tmp_df[col].replace({'?': None})

        if "group" in df.columns:
            df = df.drop(columns=["group"])
            
        X = df.drop(columns=["survival_status"])
        y = df["survival_status"].astype(int).astype(str)


        generator = ExceptionRulesSurvival(mincov=5, survival_time_attr="survival_time", max_growing=5, find_exceptions=True)


        model = generator.fit(X , y)
        ruleset = model.ruleset

        ruleset_gt = load_ruleset("survival/gbsg2_ruleset.json", ProblemTypes.SURVIVAL)

        self.assertEqual(
            ruleset, ruleset_gt,
            'Rulesets should be the same'
        )


    def test_bhs(self):
        """Induce the expected survival rules for the BHS dataset."""

        df = load_dataset("survival/01_BHS.arff")
        # code to change encoding of the file
        tmp_df = df.select_dtypes([object])
        tmp_df = tmp_df.stack().str.decode("utf-8").unstack()
        for col in tmp_df:
            df[col] = tmp_df[col].replace({'?': None})
            
        X = df.drop(columns=["survival_status"])
        y = df["survival_status"].astype(int).astype(str)


        generator = ExceptionRulesSurvival(mincov=5, survival_time_attr="survival_time", max_growing=5, find_exceptions=True)


        model = generator.fit(X , y)
        ruleset = model.ruleset

        ruleset_gt = load_ruleset("survival/bhs_ruleset.json", ProblemTypes.SURVIVAL)

        self.assertEqual(
            ruleset, ruleset_gt,
            'Rulesets should be the same'
        )
