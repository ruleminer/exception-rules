import pandas as pd
from scipy.io import arff
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'exception-rules')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'decision-rules')))

df = pd.DataFrame(arff.loadarff(f"./data/classification/train_test/mushroom.arff")[0])
# code to change encoding of the file
tmp_df = df.select_dtypes([object])
tmp_df = tmp_df.stack().str.decode("utf-8").unstack()
for col in tmp_df:
    df[col] = tmp_df[col].replace({'?': None})
    
X = df.drop(columns=["class"])
y = df["class"]


from exception_rules.classification.algorithm import MyRuleClassifier

generator = MyRuleClassifier(mincov=5, induction_measuer="c2", find_exceptions=True)


model = generator.fit(X , y)
ruleset = model.ruleset

for rule in ruleset.rules:
    print(f"CR: {str(rule)}")
    if rule.exception_rule is not None:
        print(f"RR: {rule.reference_rule}")
        print(f"ER: {rule.exception_rule}")

bacc_train = balanced_accuracy_score(y, ruleset.predict(X))
print(f"Balanced accuracy train: {bacc_train}")

