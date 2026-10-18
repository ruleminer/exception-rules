.. Generated from docs/supplementary/latex/sections/experiments.tex; edit the LaTeX source.

Experiments
===========

.. _supplement-experiments:

Evidence and evaluation questions
---------------------------------

.. _experiments:

.. _evidence-and-evaluation-questions:

The experiments document exception-rule discovery: how many rules were
generated and which premises define the discovered exceptions. Discovery was
performed on complete datasets. Computational cost and timing comparisons are collected in
:ref:`the section <supplement-computational-efficiency>`.

The evaluation covers classification, regression, and survival analysis.
The discovery results are consolidated from the earlier studies
[MachaRegression2026]_ [MachaSurvival2026]_;
For each task, dataset-level counts are accompanied by examples of the
discovered rule structures.

.. csv-table:: Overview of the discovery experiments.
   :header: Task,Datasets,Discovery settings
   :widths: 1, 1, 1

   Classification,49,C2; mincov 5 or 0.1; theta 0.6--0.9; growth limit 5
   Regression,48,C2; mincov 5; growth limit 5
   Survival,25,Log-rank; mincov 5; growth limit 5

The experiments use benchmark datasets drawn from public repositories,
with task-specific induction settings.
The minimum new coverage is five for regression and survival. Classification
uses either five observations or 10% of the target class, combined with
precision thresholds of 0.6, 0.7, 0.8, and 0.9. This grid characterises
parameter sensitivity. The five-observation setting follows the RuleKit
configuration adopted in the regression study [MachaRegression2026]_.

C2 is used for classification and regression rule quality
[Wrobel2016]_; survival induction uses log-rank separation. The growth
limit is five conditions per parent rule to constrain premise length.
These settings define the experimental protocol rather than a predictive
hyperparameter optimisation. 

Classification
--------------

Discovery across datasets
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _classification:

.. _discovery-across-datasets:

Classification exception discovery uses complete datasets.
Fractional ``mincov=0.1`` means a fraction of the
current target class, whereas ``mincov=5`` means an absolute count of new
positive observations. The precision threshold applies to the three components
of the exception structure.

 

.. csv-table:: Datasets with at least one exception, out of 49.
   :header: mincov,theta = 0.6,theta = 0.7,theta = 0.8,theta = 0.9
   :widths: 1, 1, 1, 1, 1

   5,13,8,4,2
   0.1,6,4,3,2

 

.. csv-table:: Total exception rules across the 49 datasets.
   :header: mincov,theta = 0.6,theta = 0.7,theta = 0.8,theta = 0.9
   :widths: 1, 1, 1, 1, 1

   5,33,15,6,5
   0.1,13,7,5,3

More restrictive precision requirements yield fewer exceptions in these
experiments. This demonstrates parameter sensitivity; it does not establish
that every retained exception is useful or every rejected one is noise.

Exception counts by dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _classification-counts-by-dataset:

The following table summarises exception counts for the datasets
with at least one discovered exception. The eight
configurations combine two mincov settings (5 and 0.1) with four precision
thresholds (0.6, 0.7, 0.8, and 0.9). Exception counts are summed over these
configurations; the final column counts configurations with a nonzero result.
They are not counts of unique exception premises after deduplication.

.. _tab:classification-counts-by-dataset:

.. csv-table:: Classification exception counts across eight parameter configurations.
   :header: Dataset,Exception rules,Configurations with exceptions
   :widths: 50, 25, 25

   anneal,20,8
   mushroom,18,8
   sonar,8,6
   credit-g,7,2
   vehicle,7,2
   yeast,6,2
   cylinder-bands,5,3
   kdd-synthetic-control,4,2
   diabetes,3,2
   segment,3,1
   ecoli,2,2
   autos,1,1
   bupa-liver-disorders,1,1
   horse-colic,1,1
   sick-euthyroid,1,1

The table contains 87 exception-rule occurrences in 15 of the 49 datasets;
the other 34 datasets have no exceptions in the tested configurations.
Summing its exception counts reproduces the sum of the eight cells in the
total-exception table above. Anneal and Mushroom yield exceptions in all eight
configurations, whereas other datasets are more sensitive to the settings.
These counts include exception rules only; commonsense rules are excluded.

Examples of classification exception rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _classification-rule-examples:

The next table presents four classification exception structures
discovered with mincov = 5 and :math:`\theta=0.8`.  Mushroom classes p and e
denote poisonous and edible, respectively. In the count columns, :math:`p` and :math:`n` are
the covered observations belonging to the rule's target class and other classes;
:math:`P` and :math:`N` are the corresponding totals in the entire dataset. Each rule is
evaluated relative to its own conclusion, so these totals change when the
target class changes.

 

.. _tab:classification-rule-examples:

.. csv-table:: Classification CR--RR--ER examples for mincov = 5 and :math:`\theta=0.8`.
   :header: Example,Type,Rule,:math:`p`,:math:`n`,:math:`P`,:math:`N`
   :widths: 14, 6, 44, 9, 9, 9, 9

   anneal (1),CR,IF surface-quality = {E} THEN class = 3,297,13,684,214
   anneal (1),RR,IF carbon :math:`\geq` 3.50 THEN class = 3,74,2,684,214
   anneal (1),ER,IF surface-quality = {E} AND carbon :math:`\geq` 3.50 THEN class = U,2,0,40,858
   anneal (2),CR,IF carbon :math:`\geq` 3.50 THEN class = 3,74,2,684,214
   anneal (2),RR,IF surface-quality = {E} THEN class = 3,297,13,684,214
   anneal (2),ER,IF carbon :math:`\geq` 3.50 AND surface-quality = {E} THEN class = U,2,0,40,858
   mushroom (1),CR,IF gill-size = {n} THEN class = p,2224,288,3916,4208
   mushroom (1),RR,IF spore-print-color = {h} THEN class = p,1584,48,3916,4208
   mushroom (1),ER,IF gill-size = {n} AND spore-print-color = {h} THEN class = e,48,0,4208,3916
   mushroom (2),CR,IF odor = {n} THEN class = e,3408,120,4208,3916
   mushroom (2),RR,IF stalk-root = {c} THEN class = e,512,44,4208,3916
   mushroom (2),ER,IF odor = {n} AND stalk-root = {c} THEN class = p,8,0,3916,4208

The two Anneal structures describe the same intersection with CR and RR
exchanged; they should not be interpreted as two distinct interactions.
Each parent rule supports class 3, while their conjunction covers two class-U
observations. Mushroom (1) reverses class p to e in an intersection containing
48 edible observations, and Mushroom (2) reverses class e to p in an
intersection containing eight poisonous observations.

All four reported ERs have no covered negatives, but two of them describe
the same subgroup and some have very small support. Perfect precision on the
discovery data does not establish generalisation or replication. In particular,
mincov controls newly covered positives during parent-rule induction and is
not a lower bound of five on the ER subgroup size.

Regression
----------

Discovery across datasets
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _regression:

.. _discovery-across-datasets-1:

Exception rules were discovered in **17 of 48** regression datasets.
The table includes every dataset with a nonzero exception
count. Rules and exceptions are counted separately; an exception count
need not represent an additional independent top-level rule.

 

.. csv-table:: Regression discovery results (datasets with exceptions).
   :header: Dataset,Rules,Exceptions
   :widths: 50, 25, 25

   bodyfat,36,1
   breasttumor,50,1
   cholesterol,44,11
   compressive,61,20
   concrete,11,7
   friedman,68,48
   kidney,13,1
   laser,97,3
   pbc,65,11
   plastic,188,24
   pollution,9,2
   pwlinear,36,1
   sensory,91,3
   stock,67,9
   treasury,79,3
   triazines,20,1
   veteran,16,1

Discovered rule premises
~~~~~~~~~~~~~~~~~~~~~~~~

.. _regression-rule-premises:

Each row defines a triple: CR has premise :math:`A`, RR has premise :math:`B`,
and ER has premise :math:`A\land B`. This table gives one illustrated triple
for each of the 17 datasets with discovered exceptions. The aggregate
counts above may include additional exceptions for the same dataset.

 

.. csv-table:: Examples of regression rule premises.
   :header: No.,Dataset,CR premise :math:`A`,RR premise :math:`B`
   :widths: 5, 19, 38, 38

   1,bodyfat,Density :math:`\leq` 1.04 AND Weight :math:`\leq` 221.12 AND Hip :math:`\leq` 113.95,Hip :math:`\geq` 112.15
   2,breasttumor,Age :math:`\leq` 64.50 AND Age :math:`\geq` 60.50,Age :math:`\leq` 61.50
   3,cholesterol,Thalach :math:`\geq` 150.50 AND Age :math:`\geq` 39.50 AND Thalach :math:`\geq` 156.50,Age :math:`\geq` 66.50
   4,compressive,Age :math:`\geq` 73.00,Superplasticizer :math:`\geq` 8.04
   5,concrete,Cement :math:`\leq` 162.50,Fly ash :math:`\leq` 155.00
   6,friedman,Input1 :math:`\leq` 0.13,Input4 :math:`\leq` 0.05
   7,kidney,Age :math:`\leq` 45.50 AND Age :math:`\geq` 25.00,Age :math:`\leq` 29.00
   8,laser,Input1 :math:`\geq` 44.50 AND Input2 :math:`\leq` 44.50 AND Input3 :math:`\leq` 153.50 AND Input3 :math:`\geq` 8.00 AND Input4 :math:`\leq` 95.50,Input2 :math:`\leq` 9.50
   9,pbc,Z11 :math:`\leq` 17.50,Z2 :math:`\leq` 46.50
   10,plastic,Strength :math:`\leq` 24.35 AND Strength :math:`\geq` 18.15 AND Temperature :math:`\leq` 205.00,Strength :math:`\geq` 23.10
   11,pollution,NONW :math:`\leq` 9.15,JANT :math:`\geq` 39.50
   12,pwlinear,a1 :math:`\leq` 0.00 AND a9 :math:`\leq` 0.50 AND a3 :math:`\geq` -0.50 AND a7 :math:`\leq` 0.50,a2 :math:`\geq` 0.50
   13,sensory,Rows = {2} AND Interval = {2} AND Judges = {4},Trellis = {3}
   14,stock,Company7 :math:`\geq` 80.19,Company8 :math:`\leq` 26.06
   15,treasury,Currency :math:`\geq` 3474.10,CheckableDeposits :math:`\geq` 1148.30
   16,triazines,p4_branch :math:`\geq` 0.40,p4_flex :math:`\leq` 0.16
   17,veteran,Age :math:`\geq` 58.50 AND Karnofsky :math:`\geq` 55.00 AND Karnofsky :math:`\leq` 77.50,Months :math:`\geq` 13.50

 

.. _tab:regression_statistics:

.. csv-table:: Statistical characteristics and coverage of regression rule triples.
   :header: No.,CR Mean,CR Band,CR Cov.,RR Mean,RR Band,RR Cov.,ER Mean,ER Band,ER Cov.,Comparison :math:`p`-values CR/RR,Comparison :math:`p`-values ER/CR,Comparison :math:`p`-values ER/RR
   :widths: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1

   1,29.14,"[25.28, 33.00]",50,32.86,"[26.02, 39.71]",13,43.80,"[40.10, 47.50]",2,0.067249,0.018653,0.025641
   2,24.89,"[15.13, 34.66]",28,24.78,"[14.59, 34.97]",233,14.50,"[9.94, 19.06]",4,0.557321,0.016566,0.020288
   3,249.93,"[194.20, 305.66]",123,259.96,"[186.51, 333.41]",26,340.20,"[226.81, 453.59]",5,0.913350,0.013199,0.013378
   4,46.97,"[34.22, 59.72]",190,43.11,"[25.61, 60.62]",400,60.75,"[49.14, 72.36]",46,0.078053,:math:`4.25\times10^{-15}`,:math:`5.77\times10^{-13}`
   5,31.69,"[26.00, 37.37]",42,33.18,"[25.99, 40.37]",48,25.16,"[20.12, 30.20]",10,0.307769,0.000187,0.000057
   6,9.52,"[5.89, 13.15]",157,8.87,"[4.94, 12.80]",64,4.88,"[2.36, 7.41]",14,0.942553,:math:`2.81\times10^{-6}`,:math:`1.88\times10^{-5}`
   7,0.96,"[0.48, 1.44]",28,1.43,"[0.77, 2.10]",12,2.30,"[2.30, 2.30]",2,0.091689,0.021774,0.036368
   8,115.25,"[74.13, 156.36]",166,113.52,"[32.42, 194.63]",48,203.44,"[168.87, 238.02]",9,0.050321,:math:`2.60\times10^{-6}`,0.000296
   9,2644.38,"[1944.53, 3344.22]",16,1999.96,"[911.88, 3088.05]",156,3487.25,"[2969.43, 4005.07]",4,0.050290,0.004396,0.009139
   10,13.85,"[11.36, 16.34]",80,14.39,"[11.05, 17.74]",1319,11.00,"[9.75, 12.25]",14,0.683115,:math:`1.67\times10^{-6}`,:math:`8.26\times10^{-5}`
   11,905.15,"[855.53, 954.76]",29,934.44,"[852.57, 1016.32]",16,847.88,"[808.62, 887.15]",4,0.053553,0.016336,0.019780
   12,3.79,"[0.98, 6.59]",26,1.20,"[-4.35, 6.75]",55,7.54,"[4.74, 10.33]",6,0.099626,0.000556,0.002520
   13,15.88,"[15.20, 16.55]",16,15.28,"[14.48, 16.09]",144,16.75,"[16.19, 17.31]",4,0.077285,0.007555,0.002585
   14,48.17,"[44.51, 51.84]",137,46.20,"[39.36, 53.04]",691,53.45,"[50.12, 56.78]",33,0.117571,:math:`1.34\times10^{-12}`,:math:`3.35\times10^{-10}`
   15,5.46,"[5.17, 5.76]",250,5.31,"[4.70, 5.92]",35,5.94,"[5.83, 6.05]",4,0.054602,0.002003,0.048704
   16,0.58,"[0.43, 0.73]",37,0.61,"[0.40, 0.82]",67,0.10,"[0.10, 0.10]",2,0.235890,0.020417,0.017455
   17,105.46,"[21.27, 189.65]",35,112.90,"[17.49, 208.32]",21,208.75,"[154.60, 262.90]",4,0.483457,0.005530,0.035257

:ref:`the table <tab:regression_statistics>` summarizes the statistical characteristics of the 17 illustrated regression rule triples, using the same numbering as the preceding premise table. For each triple, it reports the mean target value, the associated variability band, and the number of covered observations for CR, RR, and ER, together with the three pairwise comparison :math:`p`-values. The bands describe within-group variability rather than confidence intervals for the estimated means.

Across all 17 examples, the CR and RR comparisons yield :math:`p`-values above 0.05, whereas both ER comparisons yield values below 0.05. Thus, under the applied acceptance criteria, the parent groups are statistically compatible while their intersection exhibits a significant deviation from both. The direction of this deviation varies across datasets: the ER mean exceeds both parent means in nine examples and falls below both in the remaining eight. This indicates that the discovered exceptions capture both unusually high and unusually low target values, rather than deviations in a single direction.

The coverage statistics further illustrate the localized nature of these patterns. ER coverage ranges from 2 to 46 observations and is consistently smaller than the coverage of either parent rule. Several examples involve very small intersections, particularly bodyfat, kidney, and triazines, each with only two ER observations. Consequently, although the reported comparisons satisfy the discovery criteria, patterns with limited support require particular caution and independent validation. The results should therefore be interpreted as evidence of statistically distinguishable subgroups within the discovery datasets, rather than as confirmation of their generalizability or causal significance.

Survival analysis
-----------------

Discovery across datasets
~~~~~~~~~~~~~~~~~~~~~~~~~

.. _survival-analysis:

.. _discovery-across-datasets-2:

Exception rules were discovered in **18 of 25** survival datasets.
The table below includes all 25 datasets, including those
where no exceptions were found.

 

.. csv-table:: Survival discovery results.
   :header: Dataset,Rules,Exceptions
   :widths: 50, 25, 25

   actg320,4,1
   BHS,4,0
   bmt-ch,2,0
   cancer,11,0
   csb-ch07-halibut,3,1
   csb-ch12-nursing,15,3
   csb-ch15-lung,2,1
   follic,2,1
   GBSG2,2,1
   hd,2,0
   lac_selected,3,2
   lung,2,1
   Melanoma,9,1
   mgus,5,1
   pbc,3,1
   przeszczepy,2,0
   Rossi,4,1
   std,6,3
   TRACE,2,0
   uis,2,1
   unemployed,9,1
   wcgs,3,3
   whas1,3,3
   whas500,3,2
   zinc,3,0

Discovered rule premises
~~~~~~~~~~~~~~~~~~~~~~~~

.. _survival-rule-premises:

Each row defines a triple: CR has premise :math:`A`, RR has premise :math:`B`,
and ER has premise :math:`A\land B`. This table lists 28 discovered triples,
complementing the dataset-level counts above.

 

.. csv-table:: Discovered survival rule premises.
   :header: No.,Dataset,CR premise :math:`A`,RR premise :math:`B`
   :widths: 5, 19, 38, 38

   1,actg320,cd4 :math:`\leq` 18.75,karnof :math:`\leq` 85.00
   2,csb-ch07-halibut,HANDTIME :math:`\leq` 15.50,LOGCAT :math:`\geq` 4.59
   3,csb-ch12-nursing,gender = {0} AND age :math:`\geq` 73.50,hlstat :math:`\leq` 3.50 AND age :math:`\geq` 67.50
   4,csb-ch12-nursing,gender = {0} AND age :math:`\geq` 71.50,hlstat :math:`\leq` 3.50 AND age :math:`\geq` 65.50
   5,csb-ch12-nursing,gender = {0} AND age :math:`\geq` 67.50,hlstat :math:`\leq` 3.50
   6,csb-ch15-lung,Stage4 = {0},Cell_Type :math:`\leq` 1.50
   7,follic,age :math:`\geq` 66.95,hgb :math:`\leq` 117.50 AND age :math:`\geq` 26.05
   8,GBSG2,pnodes :math:`\geq` 4.50,progrec :math:`\leq` 2.50
   9,lac_selected,INHA_a :math:`\leq` 11.33 AND UBC :math:`\leq` 12.51,SLC1A6 :math:`\leq` 9.25 AND CRK :math:`\leq` 8.57
   10,lac_selected,INHA_a :math:`\leq` 11.33 AND FADD :math:`\leq` 10.24 AND ATP2B1 :math:`\leq` 9.92,GCNT1_a :math:`\leq` 7.50
   11,lung,Stage4 = {0},Cell_Type :math:`\leq` 1.50
   12,Melanoma,thick :math:`\leq` 3.38 AND age :math:`\geq` 14.50,age :math:`\leq` 55.50 AND thick :math:`\leq` 12.16
   13,mgus,age :math:`\geq` 62.50,hgb :math:`\leq` 12.85
   14,pbc,bili :math:`\geq` 1.05,albumin :math:`\leq` 3.55
   15,Rossi,age :math:`\leq` 19.50,prio :math:`\geq` 8.50 AND emp6 = {no}
   16,std,age :math:`\geq` 17.50 AND vagina :math:`\leq` 0.50 AND npartner :math:`\leq` 3.50 AND iinfct :math:`\geq` 1.50,race = {W}
   17,std,age :math:`\geq` 17.50 AND vagina :math:`\leq` 0.50 AND age :math:`\leq` 42.00 AND iinfct :math:`\geq` 1.50,race = {W}
   18,std,age :math:`\leq` 17.50,dchexam :math:`\leq` 0.50
   19,uis,FRAC :math:`\leq` 0.94,NDT :math:`\geq` 5.50 AND FRAC :math:`\leq` 1.56
   20,unemployed,age = {middle},gender = {male} AND birth :math:`\leq` 205.50
   21,wcgs,chol0 :math:`\leq` 271.50,dbp0 :math:`\leq` 84.50
   22,wcgs,chol0 :math:`\geq` 264.50,sbp0 :math:`\geq` 147.00
   23,wcgs,age0 :math:`\leq` 47.50,chol0 :math:`\leq` 236.50 AND sbp0 :math:`\leq` 186.00
   24,whas1,SHO :math:`\leq` 0.50,CPK :math:`\geq` 701.50
   25,whas1,AGE :math:`\leq` 69.50,CHF :math:`\leq` 0.50
   26,whas1,AGE :math:`\geq` 66.50,CPK :math:`\geq` 2160.00 AND AGE :math:`\geq` 36.50
   27,whas500,chf :math:`\leq` 0.50,hr :math:`\leq` 71.50
   28,whas500,age :math:`\geq` 64.50,bmi :math:`\leq` 25.40
