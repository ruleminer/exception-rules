import pandas as pd
from scipy.io import arff
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

def draw_histogram(rule, X, y):
    X = X.to_numpy()
    y = y.to_numpy()

    er_covered = np.where(rule.exception_rule.premise._calculate_covered_mask(X) == 1)[0]
    cr_covered = np.where(rule.premise._calculate_covered_mask(X) == 1)[0]
    rr_covered = np.where(rule.reference_rule.premise._calculate_covered_mask(X) == 1)[0]

    y_er = y[er_covered]
    y_cr = y[cr_covered]
    y_rr = y[rr_covered]

    # Liczba binów dobrana na podstawie liczby elementów w y
    bins = int(np.sqrt(len(y)))

    # Tworzenie histogramu
    plt.figure(figsize=(10, 6))
    # plt.hist(y, bins=bins, alpha=0.5, label='y', color='blue', edgecolor='black')
    plt.hist(y_cr, bins=int(np.sqrt(len(y_cr))), alpha=0.5, label='y_cr', color='green', edgecolor='black')
    plt.hist(y_rr, bins=int(np.sqrt(len(y_rr))), alpha=0.5, label='y_rr', color='purple', edgecolor='black')
    plt.hist(y_er, bins=int(np.sqrt(len(y_er))), alpha=1, label='y_er', color='red', edgecolor='black')


    # plot_stats(y, 'blue', 'y')
    plot_stats(y_er, 'red', 'y_er')
    plot_stats(y_cr, 'green', 'y_cr')
    plot_stats(y_rr, 'purple', 'y_rr')

    text_er_cr = mann_whitney_test(y_er, y_cr, 'y_er', 'y_cr')
    text_er_rr = mann_whitney_test(y_er, y_rr, 'y_er', 'y_rr')
    text_cr_rr = mann_whitney_test(y_cr, y_rr, 'y_cr', 'y_rr')

    # plt.figtext(0, 0.06, text_cr_rr, fontsize=10)
    # plt.figtext(0, 0.03, text_er_cr, fontsize=10)
    # plt.figtext(0, 0.00, text_er_rr, fontsize=10)


    plt.legend()
    # plt.xlabel('Wartości')
    # plt.ylabel('Częstość')
    plt.title(text_cr_rr + ", " + text_er_cr + ", " + text_er_rr)
    plt.savefig(f"histogram_concrete.png")

def plot_stats(data, color, label):
    mean = np.mean(data)
    std = np.std(data)
    plt.axvline(mean, color=color, linestyle='dashed', linewidth=2, label=f'{label} mean')
    plt.axvline(mean - std, color=color, linestyle='dotted', linewidth=1, label=f'{label} ± std')
    plt.axvline(mean + std, color=color, linestyle='dotted', linewidth=1)


# Testy statystyczne Manna-Whitneya
def mann_whitney_test(sample1, sample2, label1, label2):
    stat, p_value = mannwhitneyu(sample1, sample2)
    significance = 'statistically insignificant'
    if p_value < 0.05:
        significance = 'statistically significant'
    # if p_value < 0.01:
    #     significance = '**'
    # if p_value < 0.001:
    #     significance = '***'
    return f'{label1} vs {label2}: p_val = {p_value:.5f}'

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


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from exception_rules.regression.algorithm4 import MyRuleRegressor


import logging
logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)
logging_path = "./test/"
os.makedirs(logging_path, exist_ok=True)
# Tworzymy i konfigurujemy FileHandler
file_handler = logging.FileHandler(logging_path +'/log.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger.addHandler(file_handler)


generator = MyRuleRegressor(mincov=5, induction_measuer="c2", logger=logger, prune = False, find_exceptions=True, max_growing=5)


# model = generator.fit(X_train, y_train)
model = generator.fit(X, y)
ruleset = model.ruleset

for rule in ruleset.rules:
    print(f"CR: {str(rule)}")
    if rule.exception_rule is not None:
        print(f"RR: {rule.reference_rule}")
        print(f"ER: {rule.exception_rule}")

        draw_histogram(rule, X,y)




y_pred_test = ruleset.predict(X_test)
y_pred_train = ruleset.predict(X_train)

# from sklearn.metrics import root_mean_squared_error, mean_absolute_error
# print("Train RMSE", root_mean_squared_error(y_train, y_pred_train))
# print("Test RMSE", root_mean_squared_error(y_test, y_pred_test))
# print("Train MAE", mean_absolute_error(y_train, y_pred_train))
# print("Test MAE", mean_absolute_error(y_test, y_pred_test))

