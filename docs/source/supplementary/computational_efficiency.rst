.. Generated from docs/supplementary/latex/sections/computational_efficiency.tex; edit the LaTeX source.

Computational efficiency
========================

.. _supplement-computational-efficiency:

This section collects induction times for classification, regression, and
survival analysis. For each task, the comparison includes the exception-enabled
learner, its version without exception search, RuleKit, and a task-specific
decision tree. Classification also includes a parameter-sensitivity study. All durations are
reported in seconds. Measurement scope is specified for each task;
the three task-specific experiments do not form a common-platform speed benchmark.

Classification
--------------

Parameter sensitivity
~~~~~~~~~~~~~~~~~~~~~

The table below summarises median induction times across parameter settings.

.. _execution-times:

 

.. csv-table:: Median classification exception-rule generation times (seconds).
   :header: mincov,theta = 0.6,theta = 0.7,theta = 0.8,theta = 0.9
   :widths: 1, 1, 1, 1, 1

   5,66,86,23,4
   0.1,22,9,3,2

The absolute-coverage setting is slower at every tested threshold. Within a
setting the runtime is not monotonic in theta: with mincov 5, the median rises
from 66 to 86 seconds before falling. The number of candidate refinements and
early stopping affect runtime in addition to dataset dimensions.

Comparison with other induction methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Four methods are compared on the same 49 datasets, using one fitting-time
observation per method and dataset (196 observations). RulesClassifier
and ExceptionRulesClassifier denote the learners without and with
exception search, respectively.

.. _tab:classification-fitting-times:

.. csv-table:: Classification fitting times in seconds, aggregated from saved comparison records (49 datasets per method).
   :header: Method,Mean,Median,Q1,Q3,Min,Max
   :widths: 28, 12, 12, 12, 12, 12, 12

   DecisionTreeClassifier,0.025,0.014,0.008,0.021,0.005,0.170
   RuleKit,0.437,0.291,0.127,0.459,0.013,2.602
   RulesClassifier,5.661,0.766,0.267,2.589,0.010,78.187
   ExceptionRulesClassifier,43.338,3.822,1.201,16.734,0.010,1009.613

The four prediction variants of the exception-enabled model share one fit;
its fitting time is counted once per dataset.

The comparison uses a 70/30 train/test split with ``random_state=42``,
minimum new coverage of five, C2 induction quality, and a five-condition growth
limit; the classification tree has maximum depth five. Elapsed wall-clock
time is measured with ``time.time()`` around the wrapper's ``fit`` call.
For the tree this includes preprocessing; for RuleKit it includes estimator
construction, fitting, and conversion to package rules. Data loading, splitting,
subsequent prediction, and result export are outside the interval. CPU times
are not included in the table.

Means and quantiles weight datasets equally. Q1 and Q3 use inclusive linear
interpolation, as in the survival summary. The median rises from 0.766 seconds
without exception search to 3.822 seconds with it. RuleKit and the classification
tree are faster in these saved runs. These measurements concern
a training-partition experiment, separate from the complete-dataset parameter
study above.

Regression
----------

The table below summarises fitting times for the regression comparison.

.. _training-times:

The timed methods are the regression learner with and without exception
search, RuleKit, and DecisionTreeRegressor. Rule growth and tree depth are
limited to five. The comparison uses the training partition
of a 70/30 split with ``random_state=42``; these fitting times are separate
from the complete-dataset discovery counts above.

 

.. csv-table:: Regression training times in seconds.
   :header: Method,Mean,Median,Q1,Q3,Min,Max
   :widths: 28, 12, 12, 12, 12, 12, 12

   DecisionTreeRegressor,0.009,0.008,0.006,0.010,0.004,0.017
   RuleKit,0.696,0.182,0.062,1.298,0.010,2.951
   RulesRegressor,15.414,1.148,0.366,9.028,0.028,166.118
   ExceptionRulesRegressor,77.055,4.863,1.345,41.804,0.080,975.515

Exception search raises the reported median from 1.148 to 4.863 seconds and
the maximum from 166.118 to 975.515 seconds. The tree and RuleKit are faster.
This reflects the added candidate search and statistical checks, together with
implementation differences; it should not be read as an isolated measurement
of algorithmic complexity. The large difference between mean and median shows
substantial variability across datasets.

Survival analysis
-----------------

.. _execution-times-1:

:ref:`the table <tab:survival-fitting-times>` summarises 100 fitting-time records:
one observation per method and dataset for four methods on 25 datasets.

Models were fitted on the training partition of a 70/30 split
with ``random_state=42``. Rule growth and survival-tree depth are limited
to five. These times are separate from complete-dataset discovery counts.

.. _tab:survival-fitting-times:

.. csv-table:: Survival fitting times in seconds, aggregated from saved experiment records (25 datasets per method).
   :header: Method,Mean,Median,Q1,Q3,Min,Max
   :widths: 28, 12, 12, 12, 12, 12, 12

   SurvivalTree,0.025,0.015,0.012,0.022,0.007,0.171
   RuleKit,0.687,0.445,0.358,0.793,0.181,2.029
   RulesSurvival,69.754,26.366,12.175,53.005,2.517,399.441
   ExceptionRulesSurvival,274.255,97.846,42.313,193.187,4.240,1933.616

The comparison script measures elapsed wall-clock time using ``time.time()``
immediately before and after the model wrapper's ``fit`` call. This includes
work inside that wrapper: preprocessing for the survival tree, and model
construction, fitting, and conversion to package rules for RuleKit. Dataset
loading, the train/test split, subsequent predictive evaluation, and result
export lie outside this interval. The separately saved ``process_time`` and
``thread_time`` CPU measurements are not used in the table.

The summaries weight every dataset equally and use linear-interpolated 25th
and 75th percentiles (the inclusive convention). They describe variation across
datasets rather than repeated-run uncertainty.

Exception search increases the median from 26.366 to 97.846 seconds. The ratio
of the unrounded medians is approximately 3.71; this is a ratio of aggregate
medians, not a median of per-dataset slowdowns. The largest exception-enabled
time is 1933.616 seconds (about 32.2 minutes) on TRACE, compared with 399.441
seconds without exceptions on that dataset. SurvivalTree and RuleKit are
faster in this saved comparison. The difference between means and medians
indicates an uneven distribution of computational costs across datasets.

.. include:: runtime_by_size.inc
