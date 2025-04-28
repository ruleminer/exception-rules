import pandas as pd
from scipy.io import arff
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'exception-rules')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'decision-rules')))

df = pd.DataFrame(arff.loadarff(f"./data/regression/train_test/concrete.arff")[0])
# code to change encoding of the file
tmp_df = df.select_dtypes([object])
if tmp_df.shape[1] > 0:
    tmp_df = tmp_df.stack().str.decode("utf-8").unstack()
    for col in tmp_df:
        df[col] = tmp_df[col].replace({'?': None})
    

X = df.drop(columns=["class"])
y = df["class"]



from exception_rules.regression.algorithm import MyRuleRegressor



generator = MyRuleRegressor(mincov=5, induction_measuer="c2", prune = False, find_exceptions=True, max_growing=5)


model = generator.fit(X, y)
ruleset = model.ruleset

for rule in ruleset.rules:
    print(f"CR: {str(rule)}")
    if rule.exception_rule is not None:
        print(f"RR: {rule.reference_rule}")
        print(f"ER: {rule.exception_rule}")

