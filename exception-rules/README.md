
# ExceptionRules

## Classification  
### Algorithm 1  
Refining negative examples of a CR rule. In the first step, rules are generated “normally.” Then, for each CR that covers negative examples, new rules are generated only on the subset of examples covered by the CR (during generation, they are evaluated only on this subset). The best rule for a class other than the CR is taken and considered a candidate for an ER. It is then checked whether it meets the requirements (precision threshold). RR is not checked. This method does not interfere with the rule or exception generation algorithm. Any algorithm can be used for the double generation process.

### Algorithm 2  
Generating RR candidates for CR rules. In the first step, rules are generated “normally.” Then, for each CR, candidates for RR rules are generated. This generation runs in a loop until all negative examples of the CR are covered, or no further negative examples can be covered. During RR generation, the following score is optimized:

```
# Calculate the combined score
if number_of_negatives_to_cover != 0:
    negatives_score = (1 * number_of_covered_negatives / number_of_negatives_to_cover)
else:
    negatives_score = 0
    print("negatives_score = 0")

if len(uncovered_positives) != 0:
    positives_score = (1 * len(new_covered_positives) / len(uncovered_positives))
else:   
    positives_score = 0

quality_score = (1 * quality)

score = quality_score + negatives_score + positives_score
```

This score considers: the quality of the rule (we want RR to be as good as possible), the number of negative examples from CR that are covered by the rule (we want it to cover as many negatives from the CR rule as possible), and the number of newly covered positive examples (we want the RR rule to have high coverage—these are only the positive examples not covered by the CR. Before generating RR, the class of positive examples covered by CR is changed in the dataset to an artificial other class).  
The algorithm can generate multiple RR candidates. Then, CR is combined with each RR and checked whether it meets the condition to be an ER. We distinguish several types of ER rules and a new kind called Abstaining rules (AR):
* Type 1 ER – ER has Low Cov (lower than average Cov in the class), High Prec (higher than average in the class), High DPrec (greater than DPrec calculated on class averages), and RR has High Cov and High Prec (higher than average in the class)
* Type 2 ER – ER meets Type 1 conditions, but RR does not meet the required conditions
* AR (Abstaining Rules) – rules that are not ER because they don’t meet one of the ER criteria but are useful for modifying classification. AR refers to CR because its purpose is to exclude this rule from classification in special cases.

### Algorithm 3  
Iterative generation of CR and RR. During rule induction, a CR rule is generated (until a set precision threshold is reached), and then an RR rule is generated for this CR. During RR generation, the following score is optimized (as in Algorithm 2):

```
# Calculate the combined score
if number_of_negatives_to_cover != 0:
    negatives_score = (1 * number_of_covered_negatives / number_of_negatives_to_cover)
else:
    negatives_score = 0
    print("negatives_score = 0")

if len(uncovered_positives) != 0:
    positives_score = (1 * len(new_covered_positives) / len(uncovered_positives))
else:   
    positives_score = 0

quality_score = (1 * quality)

score = quality_score + negatives_score + positives_score
```

This score considers: the quality of the rule (we want RR to be as good as possible), the number of negative examples from CR covered by the rule (we want it to cover as many negatives from the CR rule as possible), and the number of newly covered positive examples (we want the RR rule to have high coverage—these are only the positive examples not covered by CR. Before generating RR, the class of positive examples covered by CR is changed in the dataset to an artificial other class).  
Only the best RR rule (with the best score) is generated. Then it is checked whether the combination of CR and RR meets the condition to become an ER (a set precision threshold parameter). If it does, CR rule generation is stopped and we have a trio: CR, RR, and ER. If not, the algorithm returns to CR generation, adds a condition, and again tries to generate an RR. This process continues until CR no longer covers negatives or other stopping conditions are met (cannot add another condition, max_growing reached).

## Survival  
### Algorithm 3  
For survival, only Algorithm 3 has been implemented so far, as generating RR rules during CR generation seems to be the most interesting option, potentially enabling the discovery of various RR rules. Especially since in the Survival case there are no negative examples.  
Algorithm 3 – iterative generation of CR and RR. During rule induction, a CR rule is generated. In the case of Survival, all examples are considered positive, so there is no precision threshold—after each step of CR induction (adding a condition), RR generation is attempted. During RR generation, KM estimators are calculated for CR, RR, and ER = CR + RR, and then compared using a statistical test (log-rank, code taken from RuleKit, function compareEstimators: https://github.com/adaa-polsl/RuleKit/blob/master/adaa.analytics.rules/src/main/java/adaa/analytics/rules/logic/quality/LogRank.java)  
The condition that RR must meet looks like this:
```
if (score > best_score and score > 0.05) and len(er_covered) > 0 and stats_and_pvalue_cr_er["p_value"] <= 0.05 and stats_and_pvalue_rr_er["p_value"] <= 0.05:
```
`score` is the p-value of comparing CR and RR. This condition includes the following components:
* `score > best_score` – we want CR and RR to be as similar as possible
* `score > 0.05` – we want CR and RR to be similar; 0.05 could be made a parameter
* `len(er_covered) > 0` – the combination of CR and RR must cover some common examples; without this condition, ERs often covered nothing
* `stats_and_pvalue_cr_er["p_value"] <= 0.05 and stats_and_pvalue_rr_er["p_value"] <= 0.05` – the estimators of the CR-ER and RR-ER pairs are statistically different

For Survival, I do not rely on coverage thresholds or rule quality but on the comparison of KM estimators. The definition may therefore look like this:  
* CR – covers a set of examples *p* and has a KM_CR estimator for those examples  
* RR – covers a set of examples *p’*, different from CR, but with some overlap (this is ensured by passing examples not covered by CR to the RR induction function and requiring that RR covers some of these examples), and has a KM_RR estimator for those examples  

KM_CR and KM_RR estimators are statistically similar: p-value > 0.05  

ER – the combination of CR and RR, with a KM_ER estimator that is statistically different from both KM_CR and KM_RR. This situation is exceptional because the CR and RR rules produce similar estimators, but their combination yields a statistically different estimator.
