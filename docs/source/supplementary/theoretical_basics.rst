.. Generated from docs/supplementary/latex/sections/theoretical_basics.tex; edit the LaTeX source.

Theoretical basics
==================

.. _supplement-theory:

Definition of exception rules
-----------------------------

.. _suzuki-s-formulation-and-the-reference-rule:

The classical formulation attributed to Suzuki and Shimura
[Suzuki1996]_ uses two rules:

.. math::

   \mathrm{CR}: A\to X, \qquad
   \mathrm{ER}: A\land B\to Y, \quad Y\ne X.

The commonsense rule (CR) represents a dominant pattern with high
support and confidence. The exception rule (ER) introduces an
additional condition :math:`B`, identifying a smaller subgroup in which
the outcome differs from the general conclusion. Consequently,
exception rules typically have lower support but high confidence
in the alternative class.

For a classification rule, define

.. math::

   \operatorname{supp}(A\to X)
     =\frac{|C_A|}{n},\qquad
   \operatorname{conf}(A\to X)
     =\frac{|\{i\in C_A:y_i=X\}|}{|C_A|}.

Here support measures the proportion of observations satisfying
the premise. Class coverage, :math:`p/P`, is the proportion of all
observations of class :math:`X` covered by the rule.
Confidence is also called precision.

However, the CR--ER pair alone does not guarantee that the
exception represents a non-trivial dependency. In particular,
if :math:`B` already strongly predicts :math:`Y`, the exceptional
outcome may be explained by :math:`B` independently of :math:`A`.
In such a case, adding :math:`A` does not necessarily reveal
a new relationship.

To address this limitation, Hussain et al. [Hussain2000]_
introduced a **reference rule (RR)**:

.. math::

   \mathrm{RR}: B\to Y.

The reference rule evaluates the relationship between the
additional condition :math:`B` and the exceptional outcome :math:`Y`
without requiring :math:`A`. It provides a baseline against which
the exception rule can be assessed. Ideally, this relationship
has low support and/or low confidence, indicating that :math:`B`
alone does not adequately explain the exceptional outcome.
Conversely, if :math:`B\to Y` already has high confidence, the
exception may simply reflect a strong independent association
between :math:`B` and :math:`Y`, rather than a distinctive effect
of combining :math:`A` and :math:`B`.

A stronger, interaction-oriented interpretation arises when
:math:`B` independently supports the same outcome as the
commonsense rule. This leads to the following formulation,
adopted by the task-specific methods considered here:

.. math::

   \mathrm{CR}: A\to X,\qquad
   \mathrm{RR}: B\to X,\qquad
   \mathrm{ER}: A\land B\to Y,\quad Y\ne X.

In this setting, both :math:`A` and :math:`B` separately favor the
general outcome :math:`X`, whereas their conjunction favors
the alternative outcome :math:`Y`. The reference rule therefore
serves a more specific purpose: it establishes that the
additional condition :math:`B`, considered independently,
supports the general pattern rather than the exception.

This makes the exception particularly informative. Neither
parent premise alone suggests the exceptional outcome,
yet their intersection exhibits a different dominant class.
The resulting reversal cannot be explained by the marginal
association of :math:`B` with :math:`Y` alone and instead indicates
a context-dependent relationship between the premises.

The distinction between these two formulations is important.
The original reference rule :math:`B\to Y` assesses whether the
exceptional outcome is already associated with :math:`B`.
In contrast, the stronger formulation :math:`B\to X` explicitly
requires both parent rules to favor the general outcome,
making the reversal within their intersection the central
pattern of interest.

Classification exception rules
------------------------------

.. _classification-exception-rules:

In classification, :math:`X` and :math:`Y` are class labels. The adopted
structure requires strong CR and RR rules predicting :math:`X` and a strong
intersection predicting a different class :math:`Y`. The precision threshold
:math:`\theta` controls these requirements. In a multiclass problem,
:math:`Y\ne X` is a specific alternative label, rather than a single merged
class containing all other labels. The implementation chooses the most frequent
alternative class in the intersection.

Regression exception rules
--------------------------

.. _regression-exception-rules:

The concept of exception rules can be naturally extended from classification to regression by replacing discrete class labels with continuous target values. While in classification the objective is to identify examples with different class assignments, in regression the goal is to detect subgroups whose expected values differ from the general pattern.

Analogously to the classification setting, regression exception rules consist of three components: a *commonsense rule (CR)*, a *reference rule (RR)*, and an *exception rule (ER)*:

.. math::

   \begin{aligned}
   \text{CR: } & A \rightarrow \mu_{A} \; [l_{A}, u_{A}], \\
   \text{ER: } & A \land B \rightarrow \mu_{AB} \; [l_{AB}, u_{AB}], \\
   \text{RR: } & B \rightarrow \mu_{B} \; [l_{B}, u_{B}].
   \end{aligned}

Each rule is associated with a numerical estimate and a descriptive variability band: the covered-target mean :math:`\pm` one standard deviation. Its role here is an operational search criterion, not an uncertainty bound on the expected outcome.

For a rule covering a set of examples :math:`S`, the predicted value and its variability interval are defined as

.. math::

   \mu_S = \frac{1}{|S|}\sum_{i \in S} y_i, \qquad
   l_S = \mu_S - \sigma_S, \qquad
   u_S = \mu_S + \sigma_S,

where :math:`\sigma_S` denotes the standard deviation of the target variable for the examples in :math:`S`. [Janssen2011]_ [Sikora2012]_

The commonsense rule captures the dominant trend in the data, while the reference rule verifies whether the additional conditions alone influence the target variable. The exception rule identifies a subgroup whose expected value deviates from the general pattern.

A regression exception rule is defined by the following conditions:

* Consistency of the commonsense and reference rules:

  .. math::

     \mu_{A} \in [l_{B}, u_{B}] \quad \text{and} \quad
     \mu_{B} \in [l_{A}, u_{A}].

* Deviation of the exception rule from the commonsense rule:

  .. math::

     \mu_{AB} \notin [l_{A}, u_{A}].

* Deviation of the exception rule from the reference rule:

  .. math::

     \mu_{AB} \notin [l_{B}, u_{B}].

These conditions define a candidate whose parent means are compatible with the specified bands and whose intersection mean lies outside both bands. They do not establish equal parent expectations, statistical significance, causality, or practical importance. The resulting pattern is a hypothesis about context-dependent association.

Candidates are additionally screened using coverage and Mann--Whitney U comparisons [Mann1947]_. Screening on the discovery data is not independent validation and does not control the false-discovery rate of the full search. The precise groups and acceptance thresholds are given in :ref:`the section <algorithm-regression>`.

Survival exception rules
------------------------

.. _survival-exception-rules:

For a covered examples with event times :math:`t_j`, event counts :math:`d_j`, and
numbers at risk :math:`n_j` immediately before each event time, its Kaplan--Meier
conclusion is [MachaSurvival2026]_

.. math::

   \widehat S(t)=\prod_{t_j\leq t}(1-d_j/n_j).

Right-censored observations contribute to the risk set until censoring.
The median is the first time at which :math:`\widehat S(t)\leq0.5`; if that level
is not reached, the median is not estimable within follow-up. Restricted
mean survival time is :math:`\int_0^\tau\widehat S(t)\,dt` for a stated horizon :math:`\tau`.

The concept of exception rules can be naturally extended from classification to survival analysis by replacing discrete class labels with survival distributions. While in classification the goal is to identify subgroups with different class assignments, in survival analysis the objective is to identify subgroups whose time-to-event distributions differ significantly from the general pattern.

Analogously to the classification setting, survival exception rules consist of three components: a *commonsense rule (CR)*, a *reference rule (RR)*, and an *exception rule (ER)*:

:math:`\text{CR: } A \;\rightarrow\; S_X(t)`

:math:`\text{ER: } A \land B \;\rightarrow\; S_Y(t)`

:math:`\text{RR: } B \;\rightarrow\; S_{X'}(t)`

Each rule is associated with an empirical survival function estimated from
the observations satisfying its premise. CR defines the parent subgroup
with estimate :math:`S_X(t)`, RR defines the second parent with estimate :math:`S_{X'}(t)`,
and ER defines their intersection with estimate :math:`S_Y(t)`. The RR comparison
screens for candidates whose parent curves are not separated by the chosen
test; it does not establish that their survival distributions are equivalent.

The operational screening requirement concerns comparisons among three groups -- CR, RR, and ER:

* CR vs RR: no statistically significant difference,

* CR vs ER: statistically significant difference detected,

* RR vs ER: statistically significant difference detected.

These relationships are typically assessed using statistical hypothesis testing procedures, such as the log-rank test. Since the subgroups defined by the rules may overlap (e.g., the ER subgroup is a subset of both CR and RR), these comparisons should be interpreted as approximate measures of differences in survival behavior rather than strictly independent group comparisons.
