# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import sys
import os
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../../../../exception-rules')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './../../../../decision-rules')))


from tests.loaders import load_ruleset, load_dataset

from decision_rules.problem import ProblemTypes

from exception_rules.survival import MyRuleSurvival

import warnings
warnings.filterwarnings('ignore')

# class TestClassificationPredictionIndicators(unittest.TestCase):


#     def test_bhs(self):

#         df = load_dataset("survival/01_BHS.arff")
#         # code to change encoding of the file
#         tmp_df = df.select_dtypes([object])
#         tmp_df = tmp_df.stack().str.decode("utf-8").unstack()
#         for col in tmp_df:
#             df[col] = tmp_df[col].replace({'?': None})
            
#         X = df.drop(columns=["survival_status"])
#         y = df["survival_status"].astype(int).astype(str)


#         generator = MyRuleSurvival(mincov=5, survival_time_attr="survival_time", max_growing = 5)


#         model = generator.fit(X , y)
#         ruleset = model.ruleset

#         ruleset_gt = load_ruleset("survival/bhs_ruleset.json", ProblemTypes.SURVIVAL)

#         self.assertEqual(
#             ruleset, ruleset_gt,
#             'Rulesets should be the same'
#         )


    