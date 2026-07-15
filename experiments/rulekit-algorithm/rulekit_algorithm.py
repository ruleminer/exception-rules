import sys 
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'decision-rules')))


from rulekit.survival import SurvivalRules

from exception_rules.decision_rules.ruleset_factories._factories.survival.rulekit_factory import RuleKitRuleSetFactory


class RuleKitAlgorithm:

    def __init__(self, survival_time_attr, minsupp_new, max_growing) -> None:
        
        self.survival_time_attr = survival_time_attr
        self.minsupp_new = minsupp_new
        self.max_growing = max_growing
        self.ruleset = None


    def fit(self, X, y):

        clf = SurvivalRules(
            survival_time_attr=self.survival_time_attr,
            minsupp_new=self.minsupp_new,
            max_growing=self.max_growing
        )


        clf.fit(X, y)

        factory = RuleKitRuleSetFactory()
        self.ruleset = factory.make(clf, X, y)

        return self.ruleset
    
        