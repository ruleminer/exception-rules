import pandas as pd
from scipy.io import arff
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'exception-rules')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'decision-rules')))

import matplotlib.pyplot as plt
import numpy as np
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator

def _draw_estymator(rule, X, y):    
        
    comonsense_rules = rule
    reference_rule = rule.reference_rule
    exception_rule = rule.exception_rule


    survival_time = X["survival_time"].to_numpy()
    
    X = X.to_numpy()
    y = y.to_numpy()
    
    survival_status = y

    er_covered = np.where(exception_rule.positive_covered_mask(X, y) == 1)[0]
    cr_covered = np.where(comonsense_rules.positive_covered_mask(X, y) == 1)[0]
    rr_covered = np.where(reference_rule.positive_covered_mask(X, y) == 1)[0]

    er_uncovered = np.where(exception_rule.positive_covered_mask(X, y) == 0)[0]
    cr_uncovered = np.where(comonsense_rules.positive_covered_mask(X, y) == 0)[0]
    rr_uncovered = np.where(reference_rule.positive_covered_mask(X, y) == 0)[0]



    er_km_covered = KaplanMeierEstimator().fit(survival_time[er_covered], survival_status[er_covered], update_additional_informations=False)
    cr_km_covered = KaplanMeierEstimator().fit(survival_time[cr_covered], survival_status[cr_covered], update_additional_informations=False)
    rr_km_covered = KaplanMeierEstimator().fit(survival_time[rr_covered], survival_status[rr_covered], update_additional_informations=False)

    er_km_uncovered = KaplanMeierEstimator().fit(survival_time[er_uncovered], survival_status[er_uncovered], update_additional_informations=False)
    cr_km_uncovered = KaplanMeierEstimator().fit(survival_time[cr_uncovered], survival_status[cr_uncovered], update_additional_informations=False)
    rr_km_uncovered = KaplanMeierEstimator().fit(survival_time[rr_uncovered], survival_status[rr_uncovered], update_additional_informations=False)

    all_example_km = KaplanMeierEstimator().fit(survival_time, survival_status, update_additional_informations=False)


    plt.figure(figsize=(10, 5))
    # plt.title(f"Estymatory dla trójki")
    plt.plot(all_example_km.times, all_example_km.probabilities, label="All examples")
    plt.plot(er_km_covered.times, er_km_covered.probabilities, label="ER covered")
    plt.plot(cr_km_covered.times, cr_km_covered.probabilities, label="CR covered")
    plt.plot(rr_km_covered.times, rr_km_covered.probabilities, label="RR covered")
    # plt.plot(er_km_uncovered.times, er_km_uncovered.probabilities, label="ER uncovered")
    # plt.plot(cr_km_uncovered.times, cr_km_uncovered.probabilities, label="CR uncovered")
    # plt.plot(rr_km_uncovered.times, rr_km_uncovered.probabilities, label="RR uncovered")
    plt.legend()
    plt.xlim(left=0)

    plt.savefig(f"estymator.png")



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


from exception_rules.survival.algorithm3 import MyRuleSurvival

generator = MyRuleSurvival(mincov=5, survival_time_attr="survival_time", max_growing = 5, find_exceptions=True)


model = generator.fit(X , y)
ruleset = model.ruleset

for rule in ruleset.rules:
    print(f"CR: {str(rule)}")
    if rule.exception_rule is not None:
        print(f"RR: {rule.reference_rule}")
        print(f"ER: {rule.exception_rule}")
        _draw_estymator(rule, X,y)

