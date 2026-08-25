"""Reusable uplift-modeling building blocks: base-learner factories, a
hyperparameter-tuning wrapper, arm-splitting/prediction-collection utilities,
a hand-rolled X-Learner, tree/ensemble/neural baselines, a TARNet uplift
model, a lightweight uplift@k metric, an uplift-score ensembling utility,
and (new in this revision) an X-Learner-specific SHAP decomposition of the
predicted uplift score.

Notebooks compose these pieces; they do not redefine this logic inline (see
03_uplift_models.ipynb). Anything that encodes a *specific notebook's*
experimental design (e.g. which meta-learners to compare) stays in the
notebook; anything reusable across notebooks lives here.

The SHAP-computation logic lives here rather than in ``src/utils.py``
because it is not a generic, architecture-agnostic utility: correctly
attributing a meta-learner's predicted uplift to its input features depends
on that meta-learner's internal structure (an X-Learner's two effect
regressors and propensity blend vs. e.g. an S-Learner's single model with
treatment as an input feature), exactly the same reason
``collect_uplift_predictions`` -- which also has to know each model type's
``.predict`` contract -- already lives here rather than in
``src/evaluation.py`` or ``src/utils.py``. See
``notebooks/05_heterogeneity_shap.ipynb`` Step 2 for the full reasoning
behind the specific approach used for the ``XLearner``.
"""

from __future__ import annotations

import random
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Base-learner factories
# ---------------------------------------------------------------------------


def build_gradient_boosting_base_learner(**params: Any) -> HistGradientBoostingClassifier:
    """Build the primary base learner used throughout the uplift comparison.

    ``HistGradientBoostingClassifier`` is used for every S-/T-/X-Learner
    stage that needs an outcome classifier: it captures nonlinearities and
    interactions among the covariates without hand-engineered interaction
    terms, needs no feature scaling, and is robust to ``history``'s right
    skew without a log-transform (see 01_eda.ipynb).

    Parameters
    ----------
    **params : Any
        Forwarded to ``HistGradientBoostingClassifier``. Typically
        ``max_iter``, ``max_depth``, ``learning_rate``, ``min_samples_leaf``
        (tuned once in 03_uplift_models.ipynb Step 4) plus ``random_state``.

    Returns
    -------
    HistGradientBoostingClassifier
        An unfitted classifier instance.
    """
    return HistGradientBoostingClassifier(**params)


def build_gradient_boosting_effect_regressor(**params: Any) -> HistGradientBoostingRegressor:
    """Build the regressor used for the X-Learner's second-stage effect models.

    The X-Learner's second stage regresses *imputed individual treatment
    effects* (a continuous target), not a class label, so this factory wraps
    ``HistGradientBoostingRegressor`` rather than the classifier version used
    for outcome modeling. Kept as a separate factory (rather than reusing the
    classifier one) so effect-stage and outcome-stage hyperparameters can
    diverge if a future notebook needs that, even though 03_uplift_models.ipynb
    currently reuses the same tuned configuration for both.

    Parameters
    ----------
    **params : Any
        Forwarded to ``HistGradientBoostingRegressor``.

    Returns
    -------
    HistGradientBoostingRegressor
        An unfitted regressor instance.
    """
    return HistGradientBoostingRegressor(**params)


def build_logistic_regression_base_learner(
    numeric_features: Sequence[str],
    C: float = 1.0,
    random_state: int | None = None,
    **params: Any,
) -> Pipeline:
    """Build a scaled logistic regression base learner as an sklearn Pipeline.

    Unlike the boosted-tree base learner, logistic regression needs numeric
    features on comparable scales (``history`` ranges roughly $30-$3,345,
    ``recency`` ranges 1-12; without scaling the former would dominate the
    regularization penalty) and a design matrix free of the multicollinearity
    already checked for in 01_eda.ipynb. Scaling is fit *inside* the pipeline
    so scale parameters are learned fresh from whatever training rows the
    pipeline is fit on, never leaking across arms or folds.

    ``class_weight="balanced"`` is fixed rather than tuned: logistic
    regression has no depth/leaf-size mechanism to naturally attend to a
    minority class the way the boosted trees do, so reweighting the loss is
    the standard, low-cost fix for ``visit``'s moderate imbalance.

    Parameters
    ----------
    numeric_features : sequence of str
        Column names to standardize. All other columns in the frame passed
        to ``.fit``/``.predict`` are passed through unscaled (already-binary
        one-hot/flag columns need no scaling).
    C : float, default=1.0
        Inverse regularization strength, forwarded to ``LogisticRegression``.
    random_state : int or None, default=None
        Forwarded to ``LogisticRegression``.
    **params : Any
        Additional keyword arguments forwarded to ``LogisticRegression``.

    Returns
    -------
    Pipeline
        A two-step pipeline: ``"preprocessor"`` (a ``ColumnTransformer``
        scaling ``numeric_features`` and passing everything else through)
        and ``"classifier"`` (the ``LogisticRegression`` instance).
    """
    preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), list(numeric_features))],
        remainder="passthrough",
    )
    classifier = LogisticRegression(
        C=C, class_weight="balanced", random_state=random_state, max_iter=1000, **params
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def build_decision_tree_base_learner(
    random_state: int | None = None, **params: Any
) -> DecisionTreeClassifier:
    """Build a decision tree base learner, included as a requested baseline.

    This is included because a single decision tree was requested as a
    comparison point, not because it is expected to be well-suited to this
    task: a single tree is high-variance (small changes in the training
    sample can change the whole split structure) and, used as an S-Learner
    base learner, has no uplift-specific splitting criterion -- it optimizes
    outcome-prediction purity, not treatment-effect separation, exactly like
    every other base learner in this comparison. It is not expected to beat
    the tuned ``HistGradientBoostingClassifier`` results already in this
    notebook.

    Defaults are kept conservative given this project's per-arm training
    size (on the order of 25,000 rows) and ``visit``'s roughly 15% positive
    rate: an unconstrained tree at this scale would happily grow leaves
    specific to a handful of rows, which is exactly the shape overfitting to
    a moderately rare outcome takes. ``class_weight="balanced"`` mirrors the
    same fix already used for logistic regression, since a plain decision
    tree has no boosting-style mechanism to attend to the minority class on
    its own.

    See scikit-learn's ``sklearn.tree.DecisionTreeClassifier`` documentation
    for full parameter semantics.

    Parameters
    ----------
    random_state : int or None, default=None
        Forwarded to ``DecisionTreeClassifier``.
    **params : Any
        Additional keyword arguments forwarded to ``DecisionTreeClassifier``,
        overriding the conservative defaults below if supplied.

    Returns
    -------
    DecisionTreeClassifier
        An unfitted classifier instance.
    """
    defaults: dict[str, Any] = {
        "max_depth": 5,
        "min_samples_leaf": 100,
        "class_weight": "balanced",
    }
    defaults.update(params)
    return DecisionTreeClassifier(random_state=random_state, **defaults)


def build_random_forest_base_learner(
    random_state: int | None = None, **params: Any
) -> RandomForestClassifier:
    """Build a random forest base learner, included as a requested baseline.

    Included because it was requested as a comparison point, not because it
    is expected to be the best-suited model class here: a plain random
    forest averages many outcome-prediction trees, but -- like the single
    decision tree above -- has no uplift-specific splitting criterion. A
    dedicated uplift forest (splitting directly on a treatment-effect
    divergence criterion, e.g. as implemented in causal-forest libraries)
    would be the principled version of "a forest for uplift"; a plain
    ``RandomForestClassifier`` used as an S-Learner base learner is not
    that, and isn't expected to beat the existing GBM-based models.

    Defaults are conservative for the same reason as the decision tree
    factory above: a shallow, large-leaved forest given this project's
    per-arm training size and ``visit``'s moderate imbalance.
    ``class_weight="balanced"`` mirrors the same fix used elsewhere in this
    project for the same reason.

    See scikit-learn's ``sklearn.ensemble.RandomForestClassifier``
    documentation for full parameter semantics.

    Parameters
    ----------
    random_state : int or None, default=None
        Forwarded to ``RandomForestClassifier``.
    **params : Any
        Additional keyword arguments forwarded to ``RandomForestClassifier``,
        overriding the conservative defaults below if supplied.

    Returns
    -------
    RandomForestClassifier
        An unfitted classifier instance.
    """
    defaults: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 6,
        "min_samples_leaf": 50,
        "class_weight": "balanced",
        "n_jobs": -1,
    }
    defaults.update(params)
    return RandomForestClassifier(random_state=random_state, **defaults)


def build_mlp_base_learner(
    numeric_features: Sequence[str],
    random_state: int | None = None,
    **params: Any,
) -> Pipeline:
    """Build a scaled MLPClassifier base learner, for the S-Learner comparison.

    This is a generic feed-forward network used as a plug-in S-Learner base
    learner -- it predicts ``visit`` with treatment as an extra input
    feature, exactly like the logistic regression and boosted-tree base
    learners already in this notebook, and inherits the S-Learner's known
    failure mode (a flexible model can lean on the other covariates and
    ignore a weak treatment signal). This is a different thing from the
    TARNet model built alongside it: TARNet is architected specifically for
    treatment-effect estimation (a shared representation feeding two
    treatment-specific heads, trained with a factual-only loss); this MLP is
    a general-purpose classifier with no awareness that one of its inputs is
    a treatment indicator at all. It is included to show that distinction
    concretely, not because it is expected to outperform TARNet.

    Needs the same scaling discipline as logistic regression (an MLP's
    gradient-based optimization also expects comparable input scales), so
    this factory follows the same ``ColumnTransformer`` + ``StandardScaler``
    pattern as ``build_logistic_regression_base_learner``.

    Parameters
    ----------
    numeric_features : sequence of str
        Column names to standardize; all other columns pass through unscaled.
    random_state : int or None, default=None
        Forwarded to ``MLPClassifier``.
    **params : Any
        Additional keyword arguments forwarded to ``MLPClassifier``,
        overriding the conservative defaults below if supplied.

    Returns
    -------
    Pipeline
        A two-step pipeline: ``"preprocessor"`` and ``"classifier"``
        (the ``MLPClassifier`` instance).
    """
    defaults: dict[str, Any] = {
        "hidden_layer_sizes": (64, 32),
        "activation": "relu",
        "alpha": 1e-3,
        "early_stopping": True,
        "n_iter_no_change": 15,
        "max_iter": 300,
    }
    defaults.update(params)
    preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), list(numeric_features))],
        remainder="passthrough",
    )
    classifier = MLPClassifier(random_state=random_state, **defaults)
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------


def tune_base_learner_hyperparameters(
    estimator: BaseEstimator,
    param_distributions: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 15,
    cv: int = 5,
    scoring: str = "average_precision",
    random_state: int | None = None,
) -> BaseEstimator:
    """Lightly tune a base learner's hyperparameters via randomized search.

    Deliberately a thin wrapper around ``RandomizedSearchCV``, not a custom
    search loop: this project's tuning strategy (search against
    outcome-prediction fit, not the uplift metric directly; tune once on
    pooled data, not per arm; random rather than exhaustive search) is
    explained in full in 03_uplift_models.ipynb Step 4. This function only
    implements the mechanical "fit a randomized search, return the winner"
    step.

    Parameters
    ----------
    estimator : BaseEstimator
        An unfitted, sklearn-compatible estimator (or Pipeline).
    param_distributions : dict
        Passed straight to ``RandomizedSearchCV``. Pipeline step parameters
        use the standard ``stepname__param`` syntax.
    X : pd.DataFrame
        Training features.
    y : pd.Series
        Training target.
    n_iter : int, default=15
        Number of parameter settings sampled.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str, default="average_precision"
        Scoring metric, passed to ``RandomizedSearchCV``. Average precision
        (PR-AUC) is preferred over ROC-AUC under class imbalance.
    random_state : int or None, default=None
        Controls both the parameter sampling and (if the estimator itself
        accepts it) reproducibility of the fit.

    Returns
    -------
    BaseEstimator
        The refit best estimator (``RandomizedSearchCV.best_estimator_``).
    """
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_


# ---------------------------------------------------------------------------
# Arm-splitting and prediction-collection utilities
# ---------------------------------------------------------------------------


def prepare_arm_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    treatment_group: str,
    feature_columns: Sequence[str],
    outcome_col: str,
    control_group: str = "No E-Mail",
    treatment_col: str = "segment",
) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.Series]]:
    """Reduce train/val/test to one arm vs control and extract model inputs.

    Composes ``make_binary_treatment`` (arm-vs-control filtering) with
    feature/outcome extraction, so that each arm's train/val/test triples
    are a single function call in the modeling notebook rather than six
    repeated lines per arm.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Model-ready splits as produced by 02_data_prep.ipynb.
    treatment_group : str
        The value of ``treatment_col`` to code as ``treatment == 1``.
    feature_columns : sequence of str
        Column names to extract as ``X``.
    outcome_col : str
        Column name to extract as ``y`` (e.g. ``"visit"`` or ``"conversion"``).
    control_group : str, default="No E-Mail"
        The value of ``treatment_col`` to code as ``treatment == 0``.
    treatment_col : str, default="segment"
        Column holding the original multi-arm assignment.

    Returns
    -------
    dict of str to tuple
        ``{"train": (X, y, treatment), "val": (X, y, treatment),
        "test": (X, y, treatment)}``, one tuple per input split.
    """
    from src.data_prep import make_binary_treatment

    splits = {}
    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        arm_df = make_binary_treatment(
            split_df,
            treatment_group=treatment_group,
            control_group=control_group,
            treatment_col=treatment_col,
        )
        splits[name] = (
            arm_df[list(feature_columns)],
            arm_df[outcome_col],
            arm_df["treatment"],
        )
    return splits


def collect_uplift_predictions(
    models: dict[str, Any], X: pd.DataFrame
) -> dict[str, np.ndarray]:
    """Run ``.predict`` for every fitted uplift model in a dict.

    Every model type used in this project (scikit-uplift's ``SoloModel`` /
    ``TwoModels``, the hand-rolled ``XLearner``, and now
    ``TARNetUpliftModel``) exposes ``.predict(X)`` returning the estimated
    individual uplift score directly (not an outcome probability), so this
    is a one-line loop -- kept as a named helper purely so every notebook
    cell that needs predictions from a whole model dict reads the same way.

    Parameters
    ----------
    models : dict of str to fitted estimator
        Fitted uplift models, keyed by display name.
    X : pd.DataFrame
        Features to predict on.

    Returns
    -------
    dict of str to np.ndarray
        Predicted uplift scores, keyed by the same model names.
    """
    return {name: np.asarray(model.predict(X)) for name, model in models.items()}


# ---------------------------------------------------------------------------
# X-Learner
# ---------------------------------------------------------------------------


class XLearner(BaseEstimator):
    """Hand-rolled X-Learner (Kunzel et al., 2019), since scikit-uplift ships
    S-Learner (``SoloModel``) and T-Learner (``TwoModels``) but no native
    X-Learner.

    Algorithm
    ---------
    1. Fit two outcome models, exactly as a T-Learner would: ``mu_treatment``
       on treated rows, ``mu_control`` on control rows.
    2. Impute individual treatment effects using the *other* arm's outcome
       model: for treated rows, ``D = y - mu_control(X)``; for control rows,
       ``D = mu_treatment(X) - y``.
    3. Regress each set of imputed effects on ``X`` separately:
       ``tau_treatment`` learns the treated-side imputed effects,
       ``tau_control`` learns the control-side imputed effects.
    4. Blend the two effect models with a propensity weight ``g(x)``:
       ``tau(x) = g(x) * tau_control(x) + (1 - g(x)) * tau_treatment(x)``,
       trusting ``tau_control`` more where treated support is thin and vice
       versa.

    Because this project's data comes from a *randomized* experiment, the
    propensity is known by design rather than estimated: by default
    (``propensity=None``), ``g(x)`` is fixed to the empirical fraction of
    treated rows in the training data, which is simpler and lower-variance
    than fitting a covariate-dependent propensity model to recover, with
    added noise, a quantity already known exactly. A constant float or a
    fitted propensity-scoring callable can be supplied instead for
    non-randomized settings.

    Parameters
    ----------
    estimator_outcome_treatment, estimator_outcome_control : BaseEstimator
        Unfitted classifiers for step 1 (must expose ``predict_proba``).
    estimator_effect_treatment, estimator_effect_control : BaseEstimator
        Unfitted regressors for step 3.
    propensity : float, callable, or None, default=None
        If ``None``, uses the empirical treated fraction from the training
        data. If a float, used as a constant propensity for every row. If
        callable, called as ``propensity(X)`` and must return an array of
        per-row propensity scores.
    """

    def __init__(
        self,
        estimator_outcome_treatment: BaseEstimator,
        estimator_outcome_control: BaseEstimator,
        estimator_effect_treatment: BaseEstimator,
        estimator_effect_control: BaseEstimator,
        propensity: float | Any | None = None,
    ) -> None:
        self.estimator_outcome_treatment = estimator_outcome_treatment
        self.estimator_outcome_control = estimator_outcome_control
        self.estimator_effect_treatment = estimator_effect_treatment
        self.estimator_effect_control = estimator_effect_control
        self.propensity = propensity

    def fit(self, X: pd.DataFrame, y: pd.Series, treatment: pd.Series) -> "XLearner":
        """Fit the four-stage X-Learner.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Binary outcome.
        treatment : pd.Series
            Binary treatment indicator (1 = treated, 0 = control).

        Returns
        -------
        XLearner
            self, fitted.
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        t_arr = np.asarray(treatment)

        treated_mask = t_arr == 1
        control_mask = t_arr == 0

        self.estimator_outcome_treatment.fit(X_arr[treated_mask], y_arr[treated_mask])
        self.estimator_outcome_control.fit(X_arr[control_mask], y_arr[control_mask])

        mu_control_on_treated = self.estimator_outcome_control.predict_proba(
            X_arr[treated_mask]
        )[:, 1]
        mu_treatment_on_control = self.estimator_outcome_treatment.predict_proba(
            X_arr[control_mask]
        )[:, 1]

        d_treated = y_arr[treated_mask] - mu_control_on_treated
        d_control = mu_treatment_on_control - y_arr[control_mask]

        self.estimator_effect_treatment.fit(X_arr[treated_mask], d_treated)
        self.estimator_effect_control.fit(X_arr[control_mask], d_control)

        if self.propensity is None:
            self._constant_propensity = float(treated_mask.mean())
        elif isinstance(self.propensity, (int, float)):
            self._constant_propensity = float(self.propensity)
        else:
            self._constant_propensity = None

        return self

    def _propensity_scores(self, X: pd.DataFrame) -> np.ndarray:
        if self._constant_propensity is not None:
            return np.full(len(X), self._constant_propensity)
        return np.asarray(self.propensity(X))

    def predict_propensity(self, X: pd.DataFrame) -> np.ndarray:
        """Return the per-row propensity weight g(x) used by ``predict``.

        Public wrapper around ``_propensity_scores``. ``predict`` only
        needs the blended score, so this was internal until
        ``compute_xlearner_uplift_shap`` needed it too: reproducing
        ``predict``'s output feature-by-feature (rather than treating the
        model as a black box) requires knowing exactly how
        ``tau_control(x)`` and ``tau_treatment(x)`` were weighted for a
        given row, not just the final blended number.

        Parameters
        ----------
        X : pd.DataFrame
            Rows to compute propensity weights for. Only affects the result
            when ``propensity`` was supplied as a callable at construction;
            with the default (``None``) or a fixed float, every row gets
            the same constant weight regardless of ``X``.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Per-row ``g(x)``, matching
            ``tau(x) = g(x) * tau_control(x) + (1 - g(x)) * tau_treatment(x)``
            in ``predict``.
        """
        return self._propensity_scores(np.asarray(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict individual uplift scores.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict on.

        Returns
        -------
        np.ndarray
            Estimated individual treatment effect (uplift) per row.
        """
        X_arr = np.asarray(X)
        tau_treatment = self.estimator_effect_treatment.predict(X_arr)
        tau_control = self.estimator_effect_control.predict(X_arr)
        g = self._propensity_scores(X_arr)
        return g * tau_control + (1 - g) * tau_treatment


# ---------------------------------------------------------------------------
# NEW: TARNet uplift model (PyTorch)
# ---------------------------------------------------------------------------

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when torch is absent
    _TORCH_AVAILABLE = False


class _TARNetModule:
    """Placeholder so the module still imports without torch installed."""


if _TORCH_AVAILABLE:

    class _TARNetModule(nn.Module):  # noqa: F811 - intentional redefinition
        """Shared-trunk, two-head network implementing TARNet's architecture.

        A shared representation ``phi(x)`` feeds two independent linear
        output heads, one per treatment arm. Each head produces a single
        logit for ``P(visit=1)`` under that arm; ``BCEWithLogitsLoss`` is
        applied outside this module (see ``TARNetUpliftModel.fit``) so each
        row's loss only ever touches the head matching its *actually
        observed* treatment arm.

        Parameters
        ----------
        n_features : int
            Number of input (post-preprocessing) features.
        hidden_sizes : sequence of int
            Width of each shared-trunk hidden layer.
        dropout : float
            Dropout probability applied after each hidden layer's activation.
        """

        def __init__(
            self, n_features: int, hidden_sizes: Sequence[int], dropout: float
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            in_dim = n_features
            for hidden_dim in hidden_sizes:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
            self.trunk = nn.Sequential(*layers)
            self.control_head = nn.Linear(in_dim, 1)
            self.treatment_head = nn.Linear(in_dim, 1)

        def forward(self, x: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            """Return (control_logit, treatment_logit) for every row in x."""
            representation = self.trunk(x)
            return (
                self.control_head(representation).squeeze(-1),
                self.treatment_head(representation).squeeze(-1),
            )


class TARNetUpliftModel(BaseEstimator):
    """TARNet: a shared-representation, two-head neural uplift model.

    Architecture from Shalit, Johansson & Sontag (2017), "Estimating
    Individual Treatment Effect: Generalization Bounds and Algorithms"
    (ICML 2017). A shared feed-forward trunk learns a representation
    ``phi(x)`` from the covariates; two independent linear heads sit on top
    of that shared representation, one predicting ``P(visit=1 | do(control))``
    and one predicting ``P(visit=1 | do(treatment))``. The predicted uplift
    for a row is the difference between the two heads' outputs on the same
    shared representation: ``treatment_head(phi(x)) - control_head(phi(x))``.

    Why plain TARNet, not CFRNet or DragonNet
    ------------------------------------------
    Both of TARNet's well-known extensions exist specifically to correct for
    treatment assignment correlating with covariates in *observational*
    data:

    - **CFRNet** adds an integral-probability-metric penalty that pushes the
      treated and control representations to look similar, to compensate for
      systematically different covariate distributions between arms.
    - **DragonNet** adds a third, propensity head and a loss term
      encouraging the shared representation to be just informative enough to
      predict treatment assignment, which stabilizes effect estimates when
      propensity must itself be estimated from data.

    This project is a *confirmed-randomized* experiment: 01_eda.ipynb's SMD
    balance check passed cleanly across every covariate/arm combination, and
    the existing ``XLearner`` already uses the *known* randomization
    probability instead of an estimated propensity for exactly this reason.
    With treatment assignment independent of covariates by design, there is
    no covariate-distribution mismatch between arms for CFRNet's penalty to
    correct, and no unknown propensity for DragonNet's extra head to help
    estimate. Both extensions would add real training complexity (an extra
    loss term and its own weighting hyperparameter, or an extra head and
    loss) to correct a problem this dataset does not have. Plain TARNet
    already captures the piece that *does* matter regardless of
    randomization -- letting the two arms have their own final output layer
    on a shared, jointly-learned representation, rather than forcing one
    global model (S-Learner) or two fully independent ones with no shared
    signal at all (T-Learner) -- which is exactly the middle ground this
    project's S-/T-Learner comparison already motivates wanting.

    Training objective
    -------------------
    Standard TARNet factual-loss training: each row contributes
    ``BCEWithLogitsLoss`` only to the head matching its own observed
    treatment arm, never to the counterfactual head. Each head's loss uses a
    ``pos_weight`` computed from that arm's own training-set class balance
    (``n_negative / n_positive`` within the arm), the direct analogue of
    ``class_weight="balanced"`` used for every other base learner in this
    project.

    Public interface
    -----------------
    ``.fit(X, y, treatment, X_val=None, y_val=None, treatment_val=None)`` and
    ``.predict(X)`` match the ``XLearner`` interface on the three required,
    positional arguments, so this model drops into
    ``collect_uplift_predictions``, ``qini_auc_score``, the decile-table
    diagnostic, and the ensembling step with zero special-casing. The three
    ``_val`` arguments are optional keyword-only-in-practice extras (not part
    of the shared interface) used only for early stopping.

    Parameters
    ----------
    numeric_features : sequence of str
        Column names to standardize before feeding the network; every other
        column passes through unscaled (mirrors
        ``build_logistic_regression_base_learner``'s preprocessing).
    hidden_sizes : sequence of int, default=(64, 32)
        Width of each shared-trunk hidden layer.
    dropout : float, default=0.2
        Dropout probability in the shared trunk.
    lr : float, default=1e-3
        Adam learning rate.
    weight_decay : float, default=1e-4
        Adam L2 weight decay.
    batch_size : int, default=256
        Minibatch size.
    max_epochs : int, default=150
        Upper bound on training epochs; early stopping usually ends training
        sooner.
    patience : int, default=12
        Number of epochs without validation-loss improvement before early
        stopping triggers. Requires validation data to be passed to ``.fit``;
        if none is passed, training runs for the full ``max_epochs`` instead.
    random_state : int, default=42
        Seeds ``torch``, ``numpy``, and the standard library ``random``
        module for reproducibility.

    Notes
    -----
    Hyperparameters here are deliberately light, conservative defaults given
    this project's data scale (tens of thousands of rows per arm before the
    treatment/control split), not the product of a large search -- a wide
    neural-architecture and learning-rate search is not justified at this
    scale, for the same reasoning already applied to the GBM/logistic tuning
    in Step 4.
    """

    def __init__(
        self,
        numeric_features: Sequence[str],
        hidden_sizes: Sequence[int] = (64, 32),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 150,
        patience: int = 12,
        random_state: int = 42,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "TARNetUpliftModel requires PyTorch. Install it with "
                "`pip install torch` (see requirements.txt)."
            )
        self.numeric_features = numeric_features
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state

    def _set_seeds(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

    def _preprocess_fit(self, X: pd.DataFrame) -> np.ndarray:
        self.preprocessor_ = ColumnTransformer(
            transformers=[("scale", StandardScaler(), list(self.numeric_features))],
            remainder="passthrough",
        )
        return self.preprocessor_.fit_transform(X).astype(np.float32)

    def _preprocess_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.preprocessor_.transform(X).astype(np.float32)

    @staticmethod
    def _pos_weight(y_arm: np.ndarray) -> float:
        n_pos = float(y_arm.sum())
        n_neg = float(len(y_arm) - n_pos)
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        treatment_val: pd.Series | None = None,
    ) -> "TARNetUpliftModel":
        """Fit TARNet with factual-only loss and (optional) early stopping.

        Parameters
        ----------
        X : pd.DataFrame
            Training features.
        y : pd.Series
            Binary outcome.
        treatment : pd.Series
            Binary treatment indicator (1 = treated, 0 = control).
        X_val, y_val, treatment_val : optional
            Held-out data used only for early stopping and the training
            diagnostics in 03_uplift_models.ipynb Step 9. If omitted, the
            model trains for the full ``max_epochs`` with no early stopping.
            Passing this project's existing validation split here (rather
            than carving out a further internal validation split from an
            already-limited training set) is a deliberate choice explained
            in the notebook markdown.

        Returns
        -------
        TARNetUpliftModel
            self, fitted. Training history is stored in ``self.history_``.
        """
        self._set_seeds()

        X_train_arr = self._preprocess_fit(X)
        y_arr = np.asarray(y, dtype=np.float32)
        t_arr = np.asarray(treatment, dtype=np.float32)

        self.model_ = _TARNetModule(
            n_features=X_train_arr.shape[1],
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        )
        optimizer = torch.optim.Adam(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        pos_weight_control = self._pos_weight(y_arr[t_arr == 0])
        pos_weight_treatment = self._pos_weight(y_arr[t_arr == 1])
        loss_fn_control = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight_control, dtype=torch.float32)
        )
        loss_fn_treatment = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(pos_weight_treatment, dtype=torch.float32)
        )

        dataset = TensorDataset(
            torch.from_numpy(X_train_arr),
            torch.from_numpy(y_arr),
            torch.from_numpy(t_arr),
        )
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=generator
        )

        has_val = X_val is not None and y_val is not None and treatment_val is not None
        if has_val:
            X_val_arr = self._preprocess_transform(X_val)
            y_val_arr = np.asarray(y_val, dtype=np.float32)
            t_val_arr = np.asarray(treatment_val, dtype=np.float32)
            X_val_tensor = torch.from_numpy(X_val_arr)
            y_val_tensor = torch.from_numpy(y_val_arr)
            t_val_tensor = torch.from_numpy(t_val_arr)

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_avg_precision_control": [],
            "val_avg_precision_treatment": [],
        }
        best_val_loss = float("inf")
        best_state: dict[str, "torch.Tensor"] | None = None
        best_epoch = 0
        epochs_without_improvement = 0

        from sklearn.metrics import average_precision_score

        for epoch in range(self.max_epochs):
            self.model_.train()
            epoch_losses = []
            for x_batch, y_batch, t_batch in loader:
                optimizer.zero_grad()
                control_logit, treatment_logit = self.model_(x_batch)

                control_rows = t_batch == 0
                treatment_rows = t_batch == 1

                batch_loss = torch.tensor(0.0)
                if control_rows.any():
                    batch_loss = batch_loss + loss_fn_control(
                        control_logit[control_rows], y_batch[control_rows]
                    )
                if treatment_rows.any():
                    batch_loss = batch_loss + loss_fn_treatment(
                        treatment_logit[treatment_rows], y_batch[treatment_rows]
                    )

                batch_loss.backward()
                optimizer.step()
                epoch_losses.append(float(batch_loss.detach()))

            history["train_loss"].append(float(np.mean(epoch_losses)))

            if has_val:
                self.model_.eval()
                with torch.no_grad():
                    control_logit_val, treatment_logit_val = self.model_(X_val_tensor)
                    control_rows_val = t_val_tensor == 0
                    treatment_rows_val = t_val_tensor == 1

                    val_loss = torch.tensor(0.0)
                    if control_rows_val.any():
                        val_loss = val_loss + loss_fn_control(
                            control_logit_val[control_rows_val],
                            y_val_tensor[control_rows_val],
                        )
                    if treatment_rows_val.any():
                        val_loss = val_loss + loss_fn_treatment(
                            treatment_logit_val[treatment_rows_val],
                            y_val_tensor[treatment_rows_val],
                        )
                    val_loss_value = float(val_loss)

                    control_probs = torch.sigmoid(control_logit_val).numpy()
                    treatment_probs = torch.sigmoid(treatment_logit_val).numpy()

                history["val_loss"].append(val_loss_value)
                if control_rows_val.any():
                    history["val_avg_precision_control"].append(
                        average_precision_score(
                            y_val_arr[t_val_arr == 0], control_probs[t_val_arr == 0]
                        )
                    )
                else:
                    history["val_avg_precision_control"].append(np.nan)
                if treatment_rows_val.any():
                    history["val_avg_precision_treatment"].append(
                        average_precision_score(
                            y_val_arr[t_val_arr == 1], treatment_probs[t_val_arr == 1]
                        )
                    )
                else:
                    history["val_avg_precision_treatment"].append(np.nan)

                if val_loss_value < best_val_loss:
                    best_val_loss = val_loss_value
                    best_state = {k: v.clone() for k, v in self.model_.state_dict().items()}
                    best_epoch = epoch
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        break

        if has_val and best_state is not None:
            self.model_.load_state_dict(best_state)
            history["best_epoch"] = best_epoch
        else:
            history["best_epoch"] = len(history["train_loss"]) - 1

        self.history_ = history
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict individual uplift scores.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict on.

        Returns
        -------
        np.ndarray
            ``P(visit=1 | do(treatment)) - P(visit=1 | do(control))`` per row,
            both computed from the same shared representation.
        """
        X_arr = self._preprocess_transform(X)
        self.model_.eval()
        with torch.no_grad():
            control_logit, treatment_logit = self.model_(torch.from_numpy(X_arr))
            control_prob = torch.sigmoid(control_logit).numpy()
            treatment_prob = torch.sigmoid(treatment_logit).numpy()
        return treatment_prob - control_prob


# ---------------------------------------------------------------------------
# NEW: lightweight, provisional evaluation helpers for this notebook only
# ---------------------------------------------------------------------------


def compute_uplift_at_k(
    y: pd.Series | np.ndarray,
    uplift_scores: pd.Series | np.ndarray,
    treatment: pd.Series | np.ndarray,
    k: float,
) -> float:
    """Compute observed uplift within the top-``k`` fraction ranked by score.

    This is a lightweight, provisional metric for this notebook's own
    informal model comparison (Step 10), matching the same
    "rank by predicted uplift, compare treated vs. control outcome rate
    within that slice" logic already used by ``uplift_by_decile`` in Step 6.
    It is deliberately not the authoritative evaluation implementation: the
    real version -- with bootstrap confidence intervals, business-relevant
    budget framing (e.g. "if we can only email the top N% of the customer
    base"), and proper Qini-curve machinery -- belongs in
    ``src/evaluation.py`` and gets built fresh for ``04_evaluation.ipynb``,
    on the held-out test set. Reusing this exact function there would
    pre-commit that notebook's evaluation-metric design before the actual
    thinking about metrics happens; it is kept here, separate, on purpose.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Observed binary outcome.
    uplift_scores : array-like of shape (n_samples,)
        Predicted individual uplift scores.
    treatment : array-like of shape (n_samples,)
        Binary treatment indicator (1 = treated, 0 = control).
    k : float
        Fraction of the population to include, in (0, 1]. Rows are ranked by
        descending predicted uplift and the top ``k`` fraction is kept.

    Returns
    -------
    float
        ``mean(y | treatment=1, top-k) - mean(y | treatment=0, top-k)``
        within the top-``k`` slice. Returns ``np.nan`` if the top-``k`` slice
        contains no treated or no control rows (too small a slice, or too
        few rows overall, to compute either arm's rate).
    """
    if not 0 < k <= 1:
        raise ValueError(f"k must be in (0, 1], got {k}.")

    frame = pd.DataFrame(
        {
            "y": np.asarray(y),
            "uplift": np.asarray(uplift_scores),
            "treatment": np.asarray(treatment),
        }
    ).sort_values("uplift", ascending=False)

    n_top = max(1, int(np.ceil(len(frame) * k)))
    top_slice = frame.iloc[:n_top]

    treated = top_slice.loc[top_slice["treatment"] == 1, "y"]
    control = top_slice.loc[top_slice["treatment"] == 0, "y"]

    if len(treated) == 0 or len(control) == 0:
        return float("nan")

    return float(treated.mean() - control.mean())


# ---------------------------------------------------------------------------
# NEW: uplift-score ensembling
# ---------------------------------------------------------------------------


def ensemble_uplift_predictions(
    predictions_by_model: dict[str, np.ndarray],
    model_names: Sequence[str],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Average multiple models' predicted uplift scores into one ensemble score.

    "Ensemble" is ambiguous for uplift models, so the definition used here is
    stated explicitly: given a set of already-fitted models' predicted
    *uplift scores* (not outcome probabilities) on the same rows, take a
    (optionally weighted) average across the selected models. This is
    standard practice for reducing variance in CATE/uplift estimation --
    directly analogous to how a random forest averages many trees' outcome
    predictions -- and is deliberately not
    ``sklearn.ensemble.VotingClassifier`` / ``StackingClassifier``: both of
    those target a single outcome label and have no native concept of a
    continuous, per-individual treatment-effect score to combine. Averaging
    the already-computed uplift scores directly is the natural analogue in
    the uplift setting.

    Parameters
    ----------
    predictions_by_model : dict of str to array-like
        Every fitted model's predicted uplift scores on the same rows,
        typically the output of ``collect_uplift_predictions``. May contain
        more models than are actually ensembled; only the names listed in
        ``model_names`` are used.
    model_names : sequence of str
        Which models (by key into ``predictions_by_model``) to include in
        the ensemble, e.g. the top-K models by validation Qini AUC.
    weights : sequence of float or None, default=None
        Per-model weights, aligned with ``model_names``. If ``None``, every
        selected model is weighted equally (uniform averaging). Weights are
        normalized to sum to 1 internally, so e.g. raw Qini-AUC values can be
        passed directly as a documented "Qini-AUC-proportional" variant.

    Returns
    -------
    np.ndarray
        The (weighted) average predicted uplift score per row.

    Raises
    ------
    ValueError
        If ``model_names`` is empty, if any name is missing from
        ``predictions_by_model``, or if ``weights`` is supplied with a
        different length than ``model_names``.
    """
    if len(model_names) == 0:
        raise ValueError("model_names must contain at least one model.")

    missing = [name for name in model_names if name not in predictions_by_model]
    if missing:
        raise ValueError(f"model_names not found in predictions_by_model: {missing}")

    if weights is None:
        weights_arr = np.full(len(model_names), 1.0 / len(model_names))
    else:
        if len(weights) != len(model_names):
            raise ValueError(
                f"weights (len={len(weights)}) must match model_names "
                f"(len={len(model_names)})."
            )
        weights_arr = np.asarray(weights, dtype=float)
        weights_arr = weights_arr / weights_arr.sum()

    stacked = np.stack(
        [np.asarray(predictions_by_model[name]) for name in model_names], axis=0
    )
    return np.average(stacked, axis=0, weights=weights_arr)


# ---------------------------------------------------------------------------
# NEW: SHAP decomposition of the X-Learner's predicted uplift
# ---------------------------------------------------------------------------

try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when shap is absent
    _SHAP_AVAILABLE = False


def compute_xlearner_uplift_shap(model: XLearner, X: pd.DataFrame) -> Any:
    """Compute an exact SHAP decomposition of an ``XLearner``'s predicted uplift.

    Why this works, and why it is exact (not an approximation)
    -------------------------------------------------------------
    ``XLearner.predict`` computes
    ``tau(x) = g(x) * tau_control(x) + (1 - g(x)) * tau_treatment(x)`` --
    a per-row weighted average of two *independently fitted regressors*,
    each of which already predicts the treatment effect directly (a
    continuous target imputed in ``fit``), not an outcome probability that
    would need a further step to turn into an effect. Both
    ``estimator_effect_control`` and ``estimator_effect_treatment`` are
    tree ensembles (``HistGradientBoostingRegressor``, see
    ``build_gradient_boosting_effect_regressor``), so ``shap.TreeExplainer``
    gives an exact per-row, per-feature additive decomposition of each
    one's own output: no sampling, no approximation, values sum to that
    regressor's prediction. Because ``g(x)`` only reweights two
    already-fully-explained regressors and does not itself depend on how
    credit is split *within* either explanation, the same weighted
    combination applied to the two SHAP explanations reproduces ``tau(x)``
    exactly: ``sum_j(g(x) * phi_control_j(x) + (1-g(x)) * phi_treatment_j(x))
    + (g(x) * base_control + (1-g(x)) * base_treatment) == tau(x)`` for
    every row, up to floating-point precision.

    This does *not* generalize unchanged to this project's other
    meta-learners:

    - **S-Learner** (``sklift.models.SoloModel``): a single model with
      treatment as an input feature has no first-class "effect" output to
      explain directly. Getting its uplift SHAP would need either
      differencing two SHAP explanations computed with the treatment
      feature forced to 1 and to 0, or SHAP *interaction* values between
      the treatment feature and every covariate -- a different, more
      expensive computation than the weighted blend used here.
    - **T-Learner** (``sklift.models.TwoModels``): its two base models each
      predict an *outcome probability* under one arm, not the effect. Their
      raw SHAP explanations attribute ``P(visit) | arm``, not the
      uplift; the per-feature uplift attribution would need to *difference*
      the treated-model and control-model SHAP explanations, feature by
      feature, not weight-and-sum them as done here.

    What this does -- and does not -- explain
    --------------------------------------------
    This decomposes the *model's own predicted uplift score* into
    per-feature contributions: it explains what the fitted model learned to
    associate with a larger or smaller predicted effect, using held-out
    data. It is not, by itself, evidence of a verified causal mechanism --
    a feature with a large SHAP contribution here is a strong driver of
    this model's uplift *estimate*, and that estimate's own reliability is
    still bounded by whatever `04_evaluation.ipynb`'s Qini AUC confidence
    interval says about this model's ranking quality for the arm being
    explained.

    Parameters
    ----------
    model : XLearner
        A fitted ``XLearner`` whose ``estimator_effect_control`` and
        ``estimator_effect_treatment`` are tree-ensemble regressors
        supported by ``shap.TreeExplainer`` (e.g.
        ``HistGradientBoostingRegressor``, this project's only effect
        regressor in use -- see ``build_gradient_boosting_effect_regressor``).
    X : pd.DataFrame
        Rows to explain, using the same feature columns and column order
        the model was fitted on. Must be a ``pd.DataFrame`` (not a bare
        array), so the returned explanation's ``feature_names`` are
        meaningful for plotting.

    Returns
    -------
    shap.Explanation
        ``.values`` has shape ``(n_samples, n_features)``: the exact
        per-feature contribution to the blended uplift score, i.e.
        ``.values.sum(axis=1) + .base_values`` reproduces
        ``model.predict(X)`` up to floating-point precision. ``.data``
        holds the input feature values and ``.feature_names`` the column
        names, so the result can be passed directly to
        ``shap.plots.beeswarm``, ``shap.plots.scatter`` (dependence plots),
        and ``shap.plots.waterfall`` (single-row force plots).

    Raises
    ------
    ImportError
        If the optional ``shap`` dependency is not installed.
    """
    if not _SHAP_AVAILABLE:
        raise ImportError(
            "compute_xlearner_uplift_shap requires the optional 'shap' "
            "dependency, which is not installed. Install it first (e.g. "
            "`%pip install shap` in the notebook)."
        )

    explainer_control = shap.TreeExplainer(model.estimator_effect_control)
    explainer_treatment = shap.TreeExplainer(model.estimator_effect_treatment)

    explanation_control = explainer_control(X)
    explanation_treatment = explainer_treatment(X)

    g = model.predict_propensity(X)
    weight_control = g.reshape(-1, 1)
    weight_treatment = (1.0 - g).reshape(-1, 1)

    blended_values = (
        weight_control * explanation_control.values
        + weight_treatment * explanation_treatment.values
    )
    blended_base_values = g * np.asarray(
        explanation_control.base_values
    ) + (1.0 - g) * np.asarray(explanation_treatment.base_values)

    return shap.Explanation(
        values=blended_values,
        base_values=blended_base_values,
        data=explanation_treatment.data,
        feature_names=list(X.columns),
    )