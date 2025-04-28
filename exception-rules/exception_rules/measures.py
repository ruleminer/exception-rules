import numpy as np
from scipy.stats import mannwhitneyu
from decision_rules.survival.kaplan_meier import KaplanMeierEstimator

def calculate_ACE(rule):
    cov = rule.coverage
    if cov.p==0 and cov.n == 0:
        return 0

    X = cov.p/(cov.P + cov.N) * np.log2((cov.p/(cov.p+cov.n))/(cov.P/(cov.P+cov.N)))
    not_X = cov.n/(cov.P + cov.N) * np.log2((cov.n/(cov.p+cov.n))/(cov.N/(cov.P+cov.N)))

    if np.isnan(X):
        X = 0
    if np.isnan(not_X):
        not_X = 0

    return X + not_X

def calculate_GACE(cs_rule, ex_rule, return_ACE = False):
    cs_ACE = calculate_ACE(cs_rule)
    er_ACE = calculate_ACE(ex_rule)

    GACE = np.sqrt(cs_ACE * er_ACE)

    if return_ACE:
        return cs_ACE, er_ACE, GACE
    else: 
        return GACE 
    
def calculate_my_measure(cr_rule, er_rule):
    P = cr_rule.coverage.P
    N = cr_rule.coverage.N

    cr_part = cr_rule.coverage.p/P

    if (er_rule.coverage.p + er_rule.coverage.n) == 0:
        er_part = 0
    else:
        er_part = ((er_rule.coverage.p/(er_rule.coverage.p + er_rule.coverage.n)) + (er_rule.coverage.p/cr_rule.coverage.n))/2

    measure = (cr_part + er_part)/2

    return measure


def calculate_RI(cr_rule, er_rule, rr_rule):
    def safe_division(numerator, denominator):
        return numerator / denominator if denominator != 0 else 0

    def safe_log(value):
        return np.log2(value) if value > 0 else 0

    cr = cr_rule.coverage
    er = er_rule.coverage
    rr = rr_rule.coverage

    PrXIAB = safe_division(er.p, er.p + er.n)
    PrXIA = safe_division(cr.n, cr.p + cr.n)
    PrXIB = safe_division(rr.p, rr.p + rr.n)

    PrnotXIAB = safe_division(er.n, er.p + er.n)
    PrnotXIA = safe_division(cr.p, cr.p + cr.n)
    PrnotXIB = safe_division(rr.n, rr.p + rr.n)
    
    RIc = PrXIAB * safe_log(safe_division(PrXIAB, PrXIA * PrXIB))
    # RIc += PrnotXIAB * safe_log(safe_division(PrnotXIAB, PrnotXIA * PrnotXIB))

    PrABX = safe_division(er.p, er.P + er.N)
    PrAX = safe_division(cr.n, cr.P + cr.N)
    PrBX = safe_division(rr.p, rr.p + rr.N)

    PrABnotX = safe_division(er.n, er.P + er.N)
    PrAnotX = safe_division(cr.p, cr.P + cr.N)
    PrBnotX = safe_division(rr.n, rr.p + rr.N)

    RIs = PrABX * safe_log(safe_division(PrABX, PrAX * PrBX))
    # RIs += PrABnotX * safe_log(safe_division(PrABnotX, PrAnotX * PrBnotX))

    RI = RIc + RIs

    return RI


def calculate_my_measure_reg(cr_rule, rr_rule, er_rule, X, y):

    er_covered = np.where(er_rule.premise._calculate_covered_mask(X) == 1)[0]
    cr_covered = np.where(cr_rule.premise._calculate_covered_mask(X) == 1)[0]
    rr_covered = np.where(rr_rule.premise._calculate_covered_mask(X) == 1)[0]

    y_er = y[er_covered]
    y_cr = y[cr_covered]
    y_rr = y[rr_covered]

    _, er_cr_p_value = mannwhitneyu(y_er, y_cr)
    _, er_rr_p_value = mannwhitneyu(y_er, y_cr)
    _, cr_rr_p_value = mannwhitneyu(y_cr, y_rr)


    measure = 1 - (((cr_rr_p_value - 1) + er_cr_p_value + er_rr_p_value)/3)

    return measure

def calculate_my_measure_srv(cr_rule, rr_rule, er_rule, X, survival_time, survival_status):

    er_covered = np.where(er_rule.premise._calculate_covered_mask(X) == 1)[0]
    cr_covered = np.where(cr_rule.premise._calculate_covered_mask(X) == 1)[0]
    rr_covered = np.where(rr_rule.premise._calculate_covered_mask(X) == 1)[0]

    cr_estimator  = KaplanMeierEstimator().fit(survival_time[cr_covered], survival_status[cr_covered], update_additional_informations=False)
    rr_estimator  = KaplanMeierEstimator().fit(survival_time[rr_covered], survival_status[rr_covered], update_additional_informations=False)
    er_estimator  = KaplanMeierEstimator().fit(survival_time[er_covered], survival_status[er_covered], update_additional_informations=False)

    stats_and_pvalue_cr_rr = KaplanMeierEstimator().compare_estimators(cr_estimator, rr_estimator)
    
    stats_and_pvalue_cr_er = KaplanMeierEstimator().compare_estimators(cr_estimator, er_estimator)
    
    stats_and_pvalue_rr_er = KaplanMeierEstimator().compare_estimators(rr_estimator, er_estimator)


    er_cr_p_value = stats_and_pvalue_cr_er["p_value"]
    er_rr_p_value = stats_and_pvalue_rr_er["p_value"]
    cr_rr_p_value = stats_and_pvalue_cr_rr["p_value"]


    measure = 1 - (((cr_rr_p_value - 1) + er_cr_p_value + er_rr_p_value)/3)

    return measure