.. Generated from docs/supplementary/latex/sections/runtime_by_size.tex; edit the LaTeX source.

Dataset size and induction time
-------------------------------

.. _runtime-by-size:

The dataset-level analysis combines fitting times with the dimensions of the
49 classification, 48 regression, and 25 survival benchmark datasets.

M is the full dataset's number of rows (observations), while N denotes the
number of descriptive columns before encoding.
The total number of columns, including outcomes, is denoted by
:math:`N_{\mathrm{all}}`.
For classification and regression, N excludes class. For survival, N excludes
survival_time, survival_status, and the optional group column, in accordance
with the induction pipeline. The timing scripts use training partitions from
a 70/30 split, so fitted row counts are M minus :math:`\lceil 0.3M \rceil`.
M and :math:`N_{\mathrm{all}}` describe the input datasets before encoding.

Tree denotes
the task-specific decision tree, Rules the learner without exception search,
and Exceptions the exception-enabled learner. Results are sorted by M, then N.

 

.. csv-table:: Spearman rank correlations between fitting time and dataset dimensions.
   :header: Task,Method,Datasets,":math:`\rho(t,M)`",":math:`\rho(t,N)`"
   :widths: 1, 1, 1, 1, 1

   Classification,Tree,49,0.530,0.869
   Classification,RuleKit,49,0.708,0.584
   Classification,Rules,49,0.683,0.451
   Classification,Exceptions,49,0.530,0.389
   Regression,Tree,48,0.346,0.787
   Regression,RuleKit,48,0.923,0.319
   Regression,Rules,48,0.874,0.325
   Regression,Exceptions,48,0.859,0.336
   Survival analysis,Tree,25,0.207,0.538
   Survival analysis,RuleKit,25,0.625,0.037
   Survival analysis,Rules,25,0.291,0.174
   Survival analysis,Exceptions,25,0.351,0.248

Spearman correlations use average ranks for ties and quantify separate
associations with M and N within each task and method. They do not adjust for
the other dimension, attribute cardinality, rule-search complexity, or censoring.
These are descriptive comparisons across different datasets, not a controlled
scaling experiment or an estimate of algorithmic complexity. The per-dataset
tables below retain the individual observations behind these summaries.

For the exception-enabled learner, the association with row count is strongest
in regression (rho = 0.859), followed by classification (0.530) and survival
(0.351). The respective associations with attributes count are 0.336, 0.389,
and 0.248. 

Classification datasets
~~~~~~~~~~~~~~~~~~~~~~~

 

.. csv-table:: Classification: dataset dimensions and saved fitting times (seconds).
   :header: Dataset,M,N,Tree,RuleKit,Rules,Exceptions
   :widths: 1, 1, 1, 1, 1, 1, 1

   ``contact-lenses``,24,4,0.006,0.013,0.010,0.010
   ``labor``,57,16,0.011,0.194,0.027,0.322
   ``zoo``,101,16,0.013,0.056,0.051,0.051
   ``echocardiogram``,131,11,0.008,0.056,0.095,0.748
   ``hayes-roth``,132,4,0.006,0.038,0.028,0.063
   ``lymph``,148,18,0.014,0.071,0.122,0.340
   ``iris``,150,4,0.005,0.046,0.064,0.827
   ``hepatitis``,155,19,0.014,0.228,0.164,1.283
   ``wine``,178,13,0.007,0.060,0.311,3.264
   ``flag``,194,28,0.017,0.318,0.417,2.359
   ``autos``,205,25,0.017,0.380,1.434,12.123
   ``sonar``,208,60,0.020,0.172,5.358,50.191
   ``glass``,214,9,0.007,0.213,0.941,6.913
   ``audiology``,226,69,0.039,0.519,0.510,1.211
   ``prnn-synth``,250,2,0.005,0.061,0.153,1.214
   ``heart-statlog``,270,13,0.007,0.233,0.698,4.662
   ``breast-cancer``,286,9,0.013,0.094,0.187,0.345
   ``hungarian-heart-disease``,294,13,0.009,0.246,0.435,2.576
   ``cleveland``,303,13,0.010,0.286,1.198,5.822
   ``heart-c``,303,13,0.013,0.242,0.749,3.936
   ``ecoli``,336,7,0.006,0.223,0.708,13.795
   ``primary-tumor``,339,17,0.015,0.149,0.525,0.944
   ``bupa-liver-disorders``,345,6,0.006,0.294,1.236,8.508
   ``ionosphere``,351,34,0.015,0.134,2.589,15.161
   ``horse-colic``,368,22,0.018,0.459,0.863,3.822
   ``auto-mpg``,398,7,0.007,0.443,1.164,11.092
   ``vote``,435,16,0.015,0.078,0.055,0.239
   ``cylinder-bands``,540,35,0.026,0.435,4.908,29.324
   ``kdd-synthetic-control``,600,60,0.030,0.865,40.641,394.847
   ``balance-scale``,625,4,0.006,0.444,0.349,1.061
   ``soybean``,683,35,0.036,0.453,0.692,2.888
   ``credit-a``,690,15,0.017,0.388,2.338,16.734
   ``breast-w``,699,9,0.007,0.382,0.267,3.424
   ``diabetes``,768,8,0.008,0.476,7.397,32.072
   ``vehicle``,846,18,0.012,0.390,15.178,116.268
   ``anneal``,898,38,0.040,2.035,0.842,2.347
   ``tic-tac-toe``,958,9,0.014,0.127,0.518,2.091
   ``mammographic-masses``,961,5,0.008,0.101,0.766,6.005
   ``credit-g``,1000,20,0.025,0.570,7.129,37.547
   ``yeast``,1484,8,0.009,0.291,16.793,93.715
   ``car``,1728,6,0.016,0.326,0.862,1.201
   ``titanic``,2201,3,0.012,0.074,0.165,0.317
   ``segment``,2310,19,0.021,0.758,78.187,1009.613
   ``seismic-bumps``,2584,18,0.025,0.959,50.086,89.303
   ``splice``,3190,60,0.170,2.602,7.116,42.044
   ``kr-vs-kp``,3196,36,0.096,1.020,2.051,4.535
   ``hypothyroid``,3772,29,0.081,1.222,5.486,36.884
   ``sick-euthyroid``,3772,29,0.082,0.936,13.754,47.012
   ``mushroom``,8124,22,0.139,1.266,1.782,2.505

Regression datasets
~~~~~~~~~~~~~~~~~~~

 

.. csv-table:: Regression: dataset dimensions and saved fitting times (seconds).
   :header: Dataset,M,N,Tree,RuleKit,Rules,Exceptions
   :widths: 1, 1, 1, 1, 1, 1, 1

   ``longley``,16,6,0.006,0.010,0.028,0.080
   ``gascons``,27,4,0.004,0.014,0.063,0.223
   ``bolts``,40,7,0.005,0.023,0.081,0.289
   ``diabetes``,43,2,0.004,0.012,0.058,0.199
   ``elusage``,55,2,0.006,0.015,0.061,0.223
   ``pollution``,60,15,0.007,0.056,0.728,3.155
   ``mbagrade``,61,2,0.005,0.016,0.041,0.151
   ``pyrim``,74,27,0.010,0.062,0.149,0.513
   ``kidney``,76,5,0.006,0.057,0.157,0.683
   ``auto93``,93,22,0.015,0.153,0.942,4.302
   ``concrete``,103,9,0.008,0.060,1.035,2.875
   ``cloud``,108,6,0.009,0.032,0.425,1.990
   ``fruitfly``,125,4,0.005,0.066,0.240,1.016
   ``echomonths``,130,9,0.008,0.077,0.937,4.296
   ``veteran``,137,7,0.008,0.062,0.270,1.072
   ``fishcatch``,158,7,0.006,0.069,1.261,6.816
   ``auto-price``,159,15,0.009,0.140,1.292,5.425
   ``servo``,167,4,0.008,0.040,0.072,0.182
   ``triazines``,186,60,0.017,0.191,0.795,3.370
   ``lowbwt``,189,9,0.008,0.065,0.392,1.436
   ``pharynx``,195,11,0.012,0.174,0.607,2.534
   ``pwlinear``,200,10,0.006,0.141,0.288,0.854
   ``machine``,209,6,0.005,0.066,0.644,2.317
   ``cpu``,209,7,0.011,0.066,0.575,2.468
   ``bodyfat``,252,14,0.008,0.684,7.393,40.962
   ``breasttumor``,286,9,0.013,0.343,0.723,3.062
   ``cholesterol``,303,13,0.011,0.246,2.203,10.733
   ``ozone``,330,8,0.006,0.665,3.629,15.894
   ``baseball``,337,16,0.009,0.653,8.992,39.121
   ``dee``,365,6,0.007,0.551,9.139,44.332
   ``auto-mpg``,398,7,0.008,1.815,2.463,11.656
   ``pbc``,418,18,0.012,0.922,14.635,84.217
   ``ele-1``,495,2,0.005,0.124,1.596,7.330
   ``housing``,506,13,0.010,1.638,16.849,92.095
   ``forestFires``,517,12,0.008,1.166,7.325,32.995
   ``meta``,528,21,0.014,0.676,2.074,7.990
   ``sensory``,576,11,0.012,2.624,0.991,2.961
   ``strike``,625,6,0.007,1.289,4.221,19.598
   ``stock``,950,9,0.010,2.578,32.920,159.877
   ``laser``,993,4,0.006,1.340,15.474,72.737
   ``compressive``,1030,8,0.008,1.325,38.464,117.185
   ``mortgage``,1049,15,0.013,1.353,125.302,782.804
   ``treasury``,1049,15,0.015,2.951,166.118,975.515
   ``ele-2``,1056,4,0.005,1.002,1.851,7.481
   ``friedman``,1200,5,0.008,2.776,160.875,621.778
   ``wizmir``,1461,9,0.010,1.437,41.206,190.654
   ``wankara``,1609,9,0.010,2.164,51.580,238.800
   ``plastic``,1650,2,0.007,1.415,12.726,72.409

Survival analysis datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~

 

.. csv-table:: Survival analysis: dataset dimensions and saved fitting times (seconds).
   :header: Dataset,M,N,Tree,RuleKit,Rules,Exceptions
   :widths: 1, 1, 1, 1, 1, 1, 1

   ``BHS``,75,12,0.007,0.354,4.480,17.944
   ``lac_selected``,86,113,0.038,0.793,247.041,598.523
   ``bmt-ch``,187,35,0.022,0.254,27.275,122.337
   ``przeszczepy``,187,37,0.023,0.417,27.498,123.616
   ``Melanoma``,205,7,0.012,0.181,11.760,52.019
   ``cancer``,228,7,0.008,0.267,12.175,57.252
   ``mgus``,241,9,0.013,0.297,21.031,35.971
   ``csb-ch07-halibut``,294,5,0.009,0.445,7.073,22.098
   ``unemployed``,300,6,0.013,0.362,13.215,41.983
   ``pbc``,418,17,0.022,0.461,89.978,177.589
   ``zinc``,431,55,0.171,0.715,26.366,131.657
   ``Rossi``,432,60,0.048,0.709,9.066,42.313
   ``whas1``,481,7,0.010,0.359,23.501,86.742
   ``whas500``,500,13,0.015,0.406,53.005,193.187
   ``follic``,541,4,0.013,0.398,38.965,78.592
   ``uis``,575,13,0.012,0.238,28.388,130.942
   ``GBSG2``,686,8,0.022,0.945,218.451,654.843
   ``hd``,865,6,0.015,0.358,15.035,76.288
   ``std``,877,21,0.019,0.616,19.805,97.846
   ``lung``,1032,7,0.013,1.901,2.517,6.097
   ``csb-ch15-lung``,1032,7,0.012,0.475,2.631,4.240
   ``actg320``,1151,11,0.016,1.370,145.388,1180.621
   ``csb-ch12-nursing``,1601,5,0.014,0.798,39.658,201.104
   ``TRACE``,1878,6,0.046,2.029,399.441,1933.616
   ``wcgs``,3154,10,0.041,2.020,260.117,788.943
