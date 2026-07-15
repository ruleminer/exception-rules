import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'decision-rules')))

from rulekit.params import Measures
from rulekit.regression import RuleRegressor

from exception_rules.decision_rules.ruleset_factories._factories.regression.rulekit_factory import RuleKitRuleSetFactory


class RuleKitAlgorithm:

    def __init__(self, minsupp_new, max_growing) -> None:

        self.minsupp_new = minsupp_new
        self.max_growing = max_growing
        self.ruleset = None


    def fit(self, X, y):

        clf = RuleRegressor(
            induction_measure=Measures.C2,
            pruning_measure=Measures.C2,
            voting_measure=Measures.C2,
            minsupp_new=self.minsupp_new,
            max_growing=self.max_growing
        )


        clf.fit(X, y)

        factory = RuleKitRuleSetFactory()
        self.ruleset = factory.make(clf, X, y)

        return self.ruleset
    
    