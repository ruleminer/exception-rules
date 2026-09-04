Quick start
===========

This page shows how to install Exception-Rules and run a minimal example.

Exception-Rules is a Python package for discovering exception rules in
classification, regression, and survival analysis problems. The package
implements rule-based algorithms that induce commonsense rules and can
optionally search for exception rules and reference rules.

The basic workflow is similar for all supported problem types:

1. load a dataset,
2. split the data into attributes ``X`` and target ``y``,
3. create an Exception-Rules model,
4. fit the model,
5. inspect the generated ruleset.

Installation
------------

Clone the repository:

.. code-block:: bash

   git clone https://github.com/ruleminer/exception-rules
   cd exception-rules

Install the required dependencies:

.. code-block:: bash

   pip install -r requirements.txt

Install the local packages used by the project:

.. code-block:: bash

   pip install ./decision-rules
   pip install ./exception-rules

The ``exception_rules`` package depends on ``decision_rules`` for rule
representation, rulesets, conditions, and prediction utilities. The
documentation focuses on the ``exception_rules`` package.

Check the installation
----------------------

To verify that the package can be imported, run:

.. code-block:: python

   import exception_rules

If the import finishes without errors, the package is available in the
current Python environment.

Repository examples
-------------------

The repository contains three ready-to-run examples:

.. code-block:: text

   example_classification.py
   example_regression.py
   example_survival.py

They demonstrate the main supported problem types:

- classification,
- regression,
- survival analysis.

You can run them from the repository root:

.. code-block:: bash

   python example_classification.py
   python example_regression.py
   python example_survival.py

Illustrative example
--------------------

The illustrative notebook presents the complete exception-rule workflow and
demonstrates how to inspect CR, RR, and ER coverage using interactive tables,
attribute distributions, PCA, LDA, parallel coordinates, heatmaps, boxplots,
and coverage-intersection plots.

.. toctree::
   :maxdepth: 1

   Open the illustrative example <./quick_start/ilustrative_example.ipynb>

First classification example
----------------------------

The following example uses the ``mushroom`` dataset included in the repository.
It trains a rule classifier and prints commonsense rules together with their
exception and reference rules, if they were found.

.. code-block:: python

   import pandas as pd
   from scipy.io import arff

   from exception_rules.classification.algorithm import MyRuleClassifier

   df = pd.DataFrame(
       arff.loadarff("./data/classification/train_test/mushroom.arff")[0]
   )

   # Decode byte-string columns loaded from the ARFF file.
   object_columns = df.select_dtypes([object])
   object_columns = object_columns.stack().str.decode("utf-8").unstack()

   for column in object_columns:
       df[column] = object_columns[column].replace({"?": None})

   X = df.drop(columns=["class"])
   y = df["class"]

   classifier = MyRuleClassifier(
       mincov=5,
       induction_measuer="c2",
       find_exceptions=True,
   )

   model = classifier.fit(X, y)
   ruleset = model.ruleset

   for rule in ruleset.rules:
       print(f"CR: {rule}")

       if rule.exception_rule is not None:
           print(f"RR: {rule.reference_rule}")
           print(f"ER: {rule.exception_rule}")

The printed rule types are:

``CR``
   Commonsense rule.

``RR``
   Reference rule.

``ER``
   Exception rule.

A fitted model stores the generated rules in the ``ruleset`` attribute.

Prediction
----------

The generated ruleset can also be used for prediction:

.. code-block:: python

   predictions = ruleset.predict(X)

For classification, standard scikit-learn metrics can be used to evaluate
the predictions. For example:

.. code-block:: python

   from sklearn.metrics import balanced_accuracy_score

   score = balanced_accuracy_score(y, predictions)
   print(f"Balanced accuracy: {score}")

Regression and survival
-----------------------

Regression and survival workflows follow the same general pattern.

For regression, use ``MyRuleRegressor``:

.. code-block:: python

   from exception_rules.regression.algorithm import MyRuleRegressor

   regressor = MyRuleRegressor(
       mincov=5,
       induction_measuer="c2",
       prune=False,
       find_exceptions=True,
       max_growing=5,
   )

For survival analysis, use ``MyRuleSurvival`` and provide the name of the
survival time attribute:

.. code-block:: python

   from exception_rules.survival.algorithm import MyRuleSurvival

   survival_model = MyRuleSurvival(
       mincov=5,
       survival_time_attr="survival_time",
       max_growing=5,
       find_exceptions=True,
   )

Complete examples are available in the tutorial section.

Next steps
----------

See the tutorials for complete examples:

- classification tutorial,
- regression tutorial,
- survival tutorial.

See the code documentation for the API reference of the main algorithms.
