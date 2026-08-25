"""Precompute the model fits, test-set evaluations, and SHAP decomposition
that ``app/dashboard.py`` needs, and persist them as one artifact bundle.

Why this script exists
-----------------------
The dashboard is a *demo of already-finished analysis*, not a live modeling
tool: the four core meta-learners it displays (S-Learner GBM/Logistic,
T-Learner GBM, X-Learner GBM) are exactly the ones ``04_evaluation.ipynb``
fits and evaluates with a 1000-resample bootstrap CI on Qini AUC and on
uplift@k, for two arms. Re-running that -- hyperparameter search plus
~8 model fits plus ~8x1000 bootstrap resamples plus a SHAP TreeExplainer
pass over the test set -- on every Streamlit rerun (which happens on *every*
widget interaction, not once per session) would make the "responsive demo"
goal impossible. This script does that work once, offline, and the
dashboard only ever deserializes its output.

Run this after ``02_data_prep.ipynb`` has produced
``data/processed/{train,val,test}.parquet`` and ``feature_manifest.json``,
and whenever that processed data changes:

    python app/build_dashboard_artifacts.py

Output: ``data/processed/dashboard_artifacts.joblib``.

A note on scope vs. 04_evaluation.ipynb / 05_heterogeneity_shap.ipynb
-----------------------------------------------------------------------
This script mirrors those notebooks' modeling choices (which four models,
which hyperparameter search, train+val refit, k_values, n_bootstrap=1000,
random_state=42) so the numbers the dashboard shows match the numbers in
the notebooks. The SHAP decomposition itself is not reproduced here: it
imports ``compute_xlearner_uplift_shap`` from ``src.models`` directly, the
same function ``05_heterogeneity_shap.ipynb`` uses, so the dashboard's SHAP
output is identical to the notebook's rather than a parallel derivation of
it.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklift.models import SoloModel, TwoModels

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import compare_models_on_test, evaluate_uplift_model
from src.models import (
    XLearner,
    build_gradient_boosting_base_learner,
    build_gradient_boosting_effect_regressor,
    build_logistic_regression_base_learner,
    collect_uplift_predictions,
    compute_xlearner_uplift_shap,
    prepare_arm_splits,
    tune_base_learner_hyperparameters,
)

ARMS: tuple[str, ...] = ("Mens E-Mail", "Womens E-Mail")
K_VALUES: tuple[float, ...] = (0.10, 0.20, 0.30)
N_BOOTSTRAP = 1000
CI_LEVEL = 0.95
RANDOM_STATE = 42
OUTCOME_COL = "visit"


def _fit_core_models(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    train_treatment: pd.Series,
    gbm_params: dict[str, Any],
    logreg_C: float,
) -> dict[str, Any]:
    """Fit this project's four final meta-learners for one arm.

    Identical in scope to ``fit_core_models`` in ``04_evaluation.ipynb``:
    that function is deliberately notebook-local (per its own docstring --
    "encodes this notebook's own scope decision... not a generic reusable
    primitive"), so it is reproduced here rather than imported, using only
    the actual reusable building blocks from ``src/models.py``.
    """
    models: dict[str, Any] = {}

    models["S-Learner (GBM)"] = SoloModel(
        estimator=build_gradient_boosting_base_learner(**gbm_params, random_state=RANDOM_STATE)
    )
    models["S-Learner (GBM)"].fit(train_X, train_y, train_treatment)

    models["S-Learner (Logistic)"] = SoloModel(
        estimator=build_logistic_regression_base_learner(
            numeric_features=["recency", "history"], C=logreg_C, random_state=RANDOM_STATE
        )
    )
    models["S-Learner (Logistic)"].fit(train_X, train_y, train_treatment)

    models["T-Learner (GBM)"] = TwoModels(
        estimator_trmnt=build_gradient_boosting_base_learner(**gbm_params, random_state=RANDOM_STATE),
        estimator_ctrl=build_gradient_boosting_base_learner(**gbm_params, random_state=RANDOM_STATE),
        method="vanilla",
    )
    models["T-Learner (GBM)"].fit(train_X, train_y, train_treatment)

    models["X-Learner (GBM)"] = XLearner(
        estimator_outcome_treatment=build_gradient_boosting_base_learner(**gbm_params, random_state=RANDOM_STATE),
        estimator_outcome_control=build_gradient_boosting_base_learner(**gbm_params, random_state=RANDOM_STATE),
        estimator_effect_treatment=build_gradient_boosting_effect_regressor(**gbm_params, random_state=RANDOM_STATE),
        estimator_effect_control=build_gradient_boosting_effect_regressor(**gbm_params, random_state=RANDOM_STATE),
        propensity=None,
    )
    models["X-Learner (GBM)"].fit(train_X, train_y, train_treatment)

    return models


def build(processed_dir: Path) -> dict[str, Any]:
    """Fit, evaluate, and explain the project's final models for both arms.

    Parameters
    ----------
    processed_dir : Path
        Directory containing ``train.parquet``, ``val.parquet``,
        ``test.parquet``, ``feature_manifest.json``, as produced by
        ``02_data_prep.ipynb``.

    Returns
    -------
    dict
        The full artifact bundle, saved by ``main`` via ``joblib.dump``.
        See ``app/dashboard.py`` for the consumer-side contract.
    """
    with open(processed_dir / "feature_manifest.json") as f:
        manifest = json.load(f)
    feature_columns = manifest["feature_columns"]

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    total_population = len(train_df) + len(val_df) + len(test_df)

    # Hyperparameters are tuned once, on pooled train_df, and reused for
    # both arms -- exactly mirroring 04_evaluation.ipynb Step 1.
    gbm_param_distributions = {
        "max_iter": [50, 100, 150],
        "max_depth": [3, 5, 7, None],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_samples_leaf": [20, 50, 100],
    }
    tuned_gbm = tune_base_learner_hyperparameters(
        estimator=build_gradient_boosting_base_learner(),
        param_distributions=gbm_param_distributions,
        X=train_df[feature_columns],
        y=train_df[OUTCOME_COL],
        n_iter=15,
        cv=5,
        scoring="average_precision",
        random_state=RANDOM_STATE,
    )
    gbm_params = {
        key: value
        for key, value in tuned_gbm.get_params().items()
        if key in {"max_iter", "max_depth", "learning_rate", "min_samples_leaf"}
    }

    logreg_param_distributions = {"classifier__C": np.logspace(-3, 2, 6)}
    tuned_logreg_pipeline = tune_base_learner_hyperparameters(
        estimator=build_logistic_regression_base_learner(numeric_features=["recency", "history"]),
        param_distributions=logreg_param_distributions,
        X=train_df[feature_columns],
        y=train_df[OUTCOME_COL],
        n_iter=6,
        cv=5,
        scoring="average_precision",
        random_state=RANDOM_STATE,
    )
    logreg_C = float(tuned_logreg_pipeline.named_steps["classifier"].C)

    arms: dict[str, dict[str, Any]] = {}

    for arm in ARMS:
        splits = prepare_arm_splits(
            train_df, val_df, test_df,
            treatment_group=arm,
            feature_columns=feature_columns,
            outcome_col=OUTCOME_COL,
        )
        X_train = pd.concat([splits["train"][0], splits["val"][0]], ignore_index=True)
        y_train = pd.concat([splits["train"][1], splits["val"][1]], ignore_index=True)
        t_train = pd.concat([splits["train"][2], splits["val"][2]], ignore_index=True)
        X_test, y_test, t_test = splits["test"]
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        t_test = t_test.reset_index(drop=True)

        models = _fit_core_models(X_train, y_train, t_train, gbm_params, logreg_C)
        test_predictions = collect_uplift_predictions(models, X_test)

        results = {
            model_name: evaluate_uplift_model(
                y_test, test_predictions[model_name], t_test,
                k_values=K_VALUES, n_bootstrap=N_BOOTSTRAP, ci_level=CI_LEVEL,
                random_state=RANDOM_STATE,
            )
            for model_name in models
        }
        comparison_df = compare_models_on_test(results)
        best_model_name = comparison_df.iloc[0]["model"]
        best_ci_lower = float(comparison_df.iloc[0]["qini_auc_ci_lower"])
        best_ci_upper = float(comparison_df.iloc[0]["qini_auc_ci_upper"])
        best_reliable = best_ci_lower > 0.0

        qini_curves = {name: results[name]["qini_curve"] for name in models}

        shap_bundle: dict[str, Any] | None = None
        best_model_obj = models[best_model_name]
        if best_reliable and isinstance(best_model_obj, XLearner):
            explanation = compute_xlearner_uplift_shap(best_model_obj, X_test)
            reconstructed = explanation.values.sum(axis=1) + explanation.base_values
            max_error = float(np.max(np.abs(reconstructed - test_predictions[best_model_name])))
            assert max_error < 1e-6, (
                f"SHAP decomposition does not reconstruct {best_model_name}'s "
                f"predicted uplift for {arm} (max error {max_error:.2e})."
            )

            mean_abs_shap = np.abs(explanation.values).mean(axis=0)
            importance_df = (
                pd.DataFrame({"feature": explanation.feature_names, "mean_abs_shap": mean_abs_shap})
                .sort_values("mean_abs_shap", ascending=False)
                .reset_index(drop=True)
            )
            shap_bundle = {
                "importance_df": importance_df,
                "shap_values": explanation.values,
                "feature_values": pd.DataFrame(explanation.data, columns=explanation.feature_names),
                "base_value": float(explanation.base_values[0]),
                "reconstruction_max_error": max_error,
            }

        arms[arm] = {
            "comparison_df": comparison_df,
            "qini_curves": qini_curves,
            "best_model": best_model_name,
            "best_model_qini_auc": float(comparison_df.iloc[0]["qini_auc"]),
            "best_model_qini_ci": (best_ci_lower, best_ci_upper),
            "best_model_reliable": best_reliable,
            "test_y": y_test.to_numpy(),
            "test_uplift": test_predictions[best_model_name],
            "test_treatment": t_test.to_numpy(),
            "shap": shap_bundle,
        }

    return {
        "feature_columns": feature_columns,
        "total_population": total_population,
        "gbm_params": gbm_params,
        "logreg_C": logreg_C,
        "arms": arms,
        "k_values": K_VALUES,
        "n_bootstrap": N_BOOTSTRAP,
        "ci_level": CI_LEVEL,
        "random_state": RANDOM_STATE,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    processed_dir = PROJECT_ROOT / "data" / "processed"
    artifact = build(processed_dir)

    output_path = processed_dir / "dashboard_artifacts.joblib"
    joblib.dump(artifact, output_path)

    print(f"Saved dashboard artifacts to {output_path}")
    for arm, arm_data in artifact["arms"].items():
        ci_lower, ci_upper = arm_data["best_model_qini_ci"]
        reliability = "reliable" if arm_data["best_model_reliable"] else "NOT reliable (CI includes zero)"
        print(
            f"  {arm}: best model = {arm_data['best_model']}, "
            f"Qini AUC = {arm_data['best_model_qini_auc']:.4f} "
            f"[{ci_lower:.4f}, {ci_upper:.4f}] -- {reliability}"
        )


if __name__ == "__main__":
    main()