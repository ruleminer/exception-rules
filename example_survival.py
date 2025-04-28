import pandas as pd
from scipy.io import arff
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'exception-rules')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'decision-rules')))

df = pd.DataFrame(arff.loadarff(f"./data/survival/train_test/18_GBSG2.arff")[0])
# code to change encoding of the file
tmp_df = df.select_dtypes([object])
tmp_df = tmp_df.stack().str.decode("utf-8").unstack()
for col in tmp_df:
    df[col] = tmp_df[col].replace({'?': None})
    
if "group" in df.columns:
    df = df.drop(columns=["group"])

X = df.drop(columns=["survival_status"])
y = df["survival_status"].astype(int).astype(str)


from exception_rules.survival.algorithm import MyRuleSurvival

generator = MyRuleSurvival(mincov=5, survival_time_attr="survival_time", max_growing = 5, find_exceptions=True)


model = generator.fit(X , y)
ruleset = model.ruleset

for rule in ruleset.rules:
    print(f"CR: {str(rule)}")
    if rule.exception_rule is not None:
        print(f"RR: {rule.reference_rule}")
        print(f"ER: {rule.exception_rule}")

