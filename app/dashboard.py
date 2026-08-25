"""Streamlit dashboard for the Hillstrom uplift-modeling project.

Four views, one per tab, chosen to tell this project's actual story rather
than to replicate every notebook plot:

1. **Randomization balance** -- the covariate-balance check from
   ``01_eda.ipynb``, live, so a viewer can confirm for themselves that the
   randomized-experiment assumption this whole project leans on actually
   holds, instead of taking it on faith from a static report.
2. **Model comparison** -- the held-out Qini AUC comparison from
   ``04_evaluation.ipynb``, per arm, with bootstrap confidence intervals
   (not just point estimates -- the notebook's own conclusion for Mens
   E-Mail is that the best model's ranking is statistically
   indistinguishable from random, which only shows up if the CI is on
   screen).
3. **Targeting simulator** -- an interactive version of
   ``compute_targeting_impact`` (``04_evaluation.ipynb`` Step 4): move a
   budget slider, see model-guided vs. random vs. treat-everyone expected
   incremental visits update live.
4. **Uplift drivers (SHAP)** -- the top predicted-uplift drivers from
   ``05_heterogeneity_shap.ipynb``, for whichever arm's best model is
   actually reliable enough to be worth explaining.

Design decisions this file assumes (see the project's own
``app/build_dashboard_artifacts.py`` for the full rationale):

- **Precomputed, not live-refit.** Every model fit, bootstrap CI, and SHAP
  value here is loaded from ``data/processed/dashboard_artifacts.joblib``,
  built once by ``app/build_dashboard_artifacts.py``. Streamlit reruns
  this whole script on every widget interaction; refitting eight
  meta-learners across two arms plus ~8,000 bootstrap resamples plus a SHAP
  pass on every slider nudge would make the app unusable as a demo. Only
  cheap, genuinely interactive quantities (the balance table over raw data,
  ``compute_targeting_impact`` at a user-chosen budget) are computed here,
  and both are wrapped in Streamlit's caching so repeated inputs are free.
- **``st.cache_resource`` for the artifact bundle and the raw dataframe**
  (large, immutable, expensive to load -- fetched/deserialized once per
  server process and reused across every session and rerun).
- **``st.cache_data`` for anything computed from those resources with
  small, hashable inputs** (the balance table; ``compute_targeting_impact``
  at a given arm/k/value combination) -- Streamlit hashes the inputs, not
  the (large) resource itself, and returns a cached copy on repeat calls,
  which is what makes the budget slider feel instant after the first move
  at a given position.

Run with::

    streamlit run app/dashboard.py

from the project root, after ``python app/build_dashboard_artifacts.py``
has produced ``data/processed/dashboard_artifacts.joblib``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep import compute_balance_table, load_hillstrom
from src.evaluation import compute_targeting_impact, plot_qini_curves_comparison

ARTIFACT_PATH = PROJECT_ROOT / "data" / "processed" / "dashboard_artifacts.joblib"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "hillstrom.csv"
BALANCE_COVARIATES = ["recency", "history", "mens", "womens", "zip_code", "newbie", "channel"]
SMD_THRESHOLD = 0.1


# ---------------------------------------------------------------------------
# Cached data / resource loaders
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading precomputed model results...")
def load_artifacts() -> dict[str, Any]:
    """Load the precomputed model-fit, evaluation, and SHAP bundle.

    Returns
    -------
    dict
        The bundle produced by ``app/build_dashboard_artifacts.py``.
        Cached as a resource (not data): it's large, immutable for the life
        of the server process, and expensive to rebuild, so every session
        and every rerun shares one deserialized copy.
    """
    if not ARTIFACT_PATH.exists():
        st.error(
            f"No precomputed artifacts found at `{ARTIFACT_PATH.relative_to(PROJECT_ROOT)}`. "
            "Run `python app/build_dashboard_artifacts.py` from the project root first "
            "(after `02_data_prep.ipynb` has produced `data/processed/*.parquet`)."
        )
        st.stop()
    return joblib.load(ARTIFACT_PATH)


@st.cache_resource(show_spinner="Loading raw Hillstrom data...")
def load_raw_data() -> pd.DataFrame:
    """Load and type-cast the raw Hillstrom dataset for the balance check.

    Returns
    -------
    pd.DataFrame
        Output of ``src.data_prep.load_hillstrom``. Cached as a resource:
        the balance tab is the only consumer, the file is small (~64k rows)
        but downloading/type-casting it on every rerun is still unnecessary
        work with no benefit, since the raw data never changes mid-session.
    """
    return load_hillstrom(str(RAW_DATA_PATH), download_if_missing=True)


@st.cache_data(show_spinner=False)
def cached_balance_table(_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the randomization balance table, cached.

    Parameters
    ----------
    _df : pd.DataFrame
        Raw Hillstrom data. Prefixed with an underscore so Streamlit does
        not try to hash the (large) dataframe itself as a cache key; the
        function has no other arguments, so it is effectively cached once
        per server process, same as the resource it's derived from.

    Returns
    -------
    pd.DataFrame
        Output of ``src.data_prep.compute_balance_table`` over
        ``BALANCE_COVARIATES``, "No E-Mail" as the reference group.
    """
    return compute_balance_table(
        _df,
        covariates=BALANCE_COVARIATES,
        treatment_col="segment",
        reference_group="No E-Mail",
        smd_threshold=SMD_THRESHOLD,
    )


@st.cache_data(show_spinner=False)
def cached_targeting_impact(
    arm: str, k: float, value_per_outcome: float | None, n_bootstrap: int
) -> dict[str, Any]:
    """Compute business-framed targeting impact at a chosen budget, cached.

    Parameters
    ----------
    arm : str
        Treatment arm, e.g. ``"Womens E-Mail"``.
    k : float
        Targeting budget as a fraction of the population, in (0, 1].
    value_per_outcome : float or None
        Optional dollar value per incremental visit; ``None`` to stay in
        incremental-visit-count terms only (see
        ``compute_targeting_impact``'s own docstring on why this is not
        derived internally).
    n_bootstrap : int
        Bootstrap resamples for the model-guided uplift rate's CI. Lower
        than the notebook's canonical 1000 by default in this app (see
        ``main`` below) to keep the slider responsive; this only widens or
        narrows the displayed interval, not the point estimate.

    Returns
    -------
    dict
        Output of ``src.evaluation.compute_targeting_impact``. Cached on
        ``(arm, k, value_per_outcome, n_bootstrap)`` -- the underlying
        test-set arrays come from the cached artifact bundle inside the
        function body, not as arguments, so the cache key stays small and
        repeated slider positions within a session are instant.
    """
    artifact = load_artifacts()
    arm_data = artifact["arms"][arm]
    return compute_targeting_impact(
        y=arm_data["test_y"],
        uplift_scores=arm_data["test_uplift"],
        treatment=arm_data["test_treatment"],
        k=k,
        total_population=artifact["total_population"],
        value_per_incremental_outcome=value_per_outcome,
        n_bootstrap=n_bootstrap,
        ci_level=artifact["ci_level"],
        random_state=artifact["random_state"],
    )


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def render_balance_tab() -> None:
    """Render the randomization balance check tab."""
    st.subheader("Randomization balance check")
    st.markdown(
        "Every downstream uplift estimate in this project assumes treatment assignment "
        "is independent of customer covariates -- true by design in a randomized "
        "experiment, but worth confirming rather than assuming. This is the standardized "
        "mean difference (SMD) between each e-mail arm and the **No E-Mail** reference "
        f"group, for every covariate; a level is flagged if |SMD| exceeds {SMD_THRESHOLD:.1f} "
        "(a common rule-of-thumb threshold, not a formal test)."
    )

    df = load_raw_data()
    balance_table = cached_balance_table(df)
    balance_table = balance_table.assign(smd_abs=balance_table["smd"].abs())

    n_flagged = int(balance_table["flag_imbalanced"].sum())
    if n_flagged == 0:
        st.success(
            "No covariate/group combinations are flagged as imbalanced. "
            "Randomization looks intact -- treatment-arm comparisons on this data are "
            "not confounded by these covariates."
        )
    else:
        st.warning(
            f"{n_flagged} covariate/group combination(s) exceed the |SMD| > "
            f"{SMD_THRESHOLD:.1f} threshold. Review before treating arm comparisons as "
            "unconfounded on these covariates."
        )

    fig = px.bar(
        balance_table.sort_values("smd_abs", ascending=False),
        x="covariate", y="smd_abs", color="group", barmode="group",
        labels={"smd_abs": "|Standardized mean difference|", "covariate": "Covariate"},
        title="Covariate balance vs. No E-Mail, by treatment arm",
    )
    fig.add_hline(
        y=SMD_THRESHOLD, line_dash="dash", line_color="firebrick",
        annotation_text="Imbalance threshold", annotation_position="top right",
    )
    fig.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig, width="stretch")

    with st.expander("Full balance table"):
        st.dataframe(
            balance_table.drop(columns="smd_abs").sort_values("smd", key=abs, ascending=False),
            width="stretch",
        )


def _reliability_caption(arm_data: dict[str, Any]) -> None:
    """Show the best model's name, Qini AUC, CI, and a reliability warning if needed."""
    ci_lower, ci_upper = arm_data["best_model_qini_ci"]
    st.caption(
        f"Best model: **{arm_data['best_model']}** -- Qini AUC "
        f"{arm_data['best_model_qini_auc']:.4f} (95% CI [{ci_lower:.4f}, {ci_upper:.4f}])"
    )
    if not arm_data["best_model_reliable"]:
        st.warning(
            "This model's Qini AUC 95% confidence interval includes zero: its ranking "
            "quality is not statistically distinguishable from random targeting at this "
            "sample size. Figures below are shown for structural completeness, not as "
            "evidence this model should be deployed."
        )


def render_comparison_tab(artifact: dict[str, Any], arm: str) -> None:
    """Render the held-out model comparison tab for the selected arm."""
    st.subheader(f"Model comparison on held-out test data -- {arm}")
    arm_data = artifact["arms"][arm]
    _reliability_caption(arm_data)

    comparison_df = arm_data["comparison_df"]
    display_cols = ["model", "qini_auc", "qini_auc_ci_lower", "qini_auc_ci_upper", "n_samples"]
    st.dataframe(
        comparison_df[display_cols].style.format(
            {"qini_auc": "{:.4f}", "qini_auc_ci_lower": "{:.4f}", "qini_auc_ci_upper": "{:.4f}"}
        ),
        width="stretch",
    )
    st.caption(
        f"Qini AUC and its {artifact['ci_level']:.0%} bootstrap CI "
        f"(n={artifact['n_bootstrap']} resamples), sorted descending. All four models are "
        "the ones carried through to test in `04_evaluation.ipynb`: S-Learner (GBM / "
        "Logistic), T-Learner (GBM), X-Learner (GBM)."
    )

    results_like = {name: {"qini_curve": curve} for name, curve in arm_data["qini_curves"].items()}
    fig = plot_qini_curves_comparison(
        results_like, comparison_df, best_model=arm_data["best_model"],
        title=f"Qini curves -- {arm}",
    )
    st.plotly_chart(fig, width="stretch")


def render_targeting_tab(artifact: dict[str, Any], arm: str) -> None:
    """Render the interactive targeting-budget simulator tab for the selected arm."""
    st.subheader(f"Targeting simulator -- {arm}")
    arm_data = artifact["arms"][arm]
    _reliability_caption(arm_data)
    st.markdown(
        f"Ranks the held-out test set by **{arm_data['best_model']}**'s predicted uplift "
        "and compares three policies at a chosen budget: targeting the top fraction by "
        "predicted uplift, targeting the same number of people at random, and treating "
        "everyone. This scales the test-set uplift rate to "
        f"**{artifact['total_population']:,}** customers (train+val+test combined) -- "
        "the deployment population is assumed to behave like the held-out sample, which "
        "should be revisited if the real target population differs meaningfully."
    )

    col_k, col_value = st.columns([2, 1])
    with col_k:
        k = st.slider(
            "Targeting budget (% of population)", min_value=1, max_value=50, value=20, step=1,
        ) / 100.0
    with col_value:
        add_value = st.checkbox("Add $ value per incremental visit")
        value_per_outcome = (
            st.number_input("$ per incremental visit", min_value=0.0, value=10.0, step=1.0)
            if add_value else None
        )

    impact = cached_targeting_impact(arm, k, value_per_outcome, n_bootstrap=300)

    m1, m2, m3 = st.columns(3)
    m1.metric("Customers targeted", f"{impact['n_targeted']:,}")
    guided = impact["model_guided_uplift_rate"]
    m2.metric(
        "Model-guided incremental visits",
        f"{impact['model_guided_incremental_visits']:.0f}",
        help=(
            f"Uplift rate in the top {k:.0%}: {guided['point_estimate']:.4f} "
            f"(95% CI [{guided['ci_lower']:.4f}, {guided['ci_upper']:.4f}])"
        ),
    )
    m3.metric(
        "vs. random targeting",
        f"{impact['random_targeting_incremental_visits']:.0f}",
        delta=f"{impact['model_guided_incremental_visits'] - impact['random_targeting_incremental_visits']:.0f}",
    )

    policy_df = pd.DataFrame({
        "policy": ["Model-guided", "Random targeting", "Treat everyone"],
        "incremental_visits": [
            impact["model_guided_incremental_visits"],
            impact["random_targeting_incremental_visits"],
            impact["treat_everyone_incremental_visits"],
        ],
    })
    fig = px.bar(
        policy_df, x="policy", y="incremental_visits",
        labels={"incremental_visits": "Expected incremental visits", "policy": ""},
        title=f"Expected incremental visits by targeting policy (budget = {k:.0%})",
    )
    st.plotly_chart(fig, width="stretch")

    if value_per_outcome is not None:
        st.caption(
            f"At ${value_per_outcome:.2f} per incremental visit: model-guided = "
            f"${impact['model_guided_incremental_value']:,.0f}, random = "
            f"${impact['random_targeting_incremental_value']:,.0f}, treat-everyone = "
            f"${impact['treat_everyone_incremental_value']:,.0f}."
        )


def render_shap_tab(artifact: dict[str, Any], arm: str) -> None:
    """Render the SHAP uplift-driver tab for the selected arm, if available."""
    st.subheader(f"Uplift drivers -- {arm}")
    arm_data = artifact["arms"][arm]

    if arm_data["shap"] is None:
        st.info(
            f"No SHAP breakdown for {arm}: its best model "
            f"(**{arm_data['best_model']}**) either isn't reliable enough to explain "
            "(Qini AUC 95% CI includes zero) or isn't an X-Learner, which is the only "
            "model type this dashboard's SHAP decomposition currently supports. "
            "`05_heterogeneity_shap.ipynb` makes the same scope choice for the same reason."
        )
        return

    shap_bundle = arm_data["shap"]
    importance_df = shap_bundle["importance_df"]

    st.caption(
        f"Exact SHAP decomposition of {arm_data['best_model']}'s predicted uplift on the "
        f"held-out test set (reconstruction error {shap_bundle['reconstruction_max_error']:.1e}, "
        "i.e. every feature's contribution plus the base value sums to the model's own "
        "prediction, to floating-point precision)."
    )

    top_n = min(10, len(importance_df))
    fig = px.bar(
        importance_df.head(top_n).sort_values("mean_abs_shap"),
        x="mean_abs_shap", y="feature", orientation="h",
        labels={"mean_abs_shap": "Mean |SHAP value|", "feature": ""},
        title=f"Top {top_n} predicted-uplift drivers",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Dependence: how a driver's value relates to its contribution**")
    top_features = importance_df["feature"].head(5).tolist()
    selected_feature = st.selectbox("Feature", top_features)
    feature_idx = shap_bundle["feature_values"].columns.get_loc(selected_feature)

    dependence_df = pd.DataFrame({
        "feature_value": shap_bundle["feature_values"][selected_feature],
        "shap_value": shap_bundle["shap_values"][:, feature_idx],
    })
    fig2 = go.Figure(go.Scatter(
        x=dependence_df["feature_value"], y=dependence_df["shap_value"],
        mode="markers", marker=dict(size=5, opacity=0.5),
    ))
    fig2.update_layout(
        title=f"SHAP dependence -- {selected_feature}",
        xaxis_title=selected_feature,
        yaxis_title="Contribution to predicted uplift",
    )
    st.plotly_chart(fig2, width="stretch")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Configure the page and dispatch to each tab."""
    st.set_page_config(page_title="Hillstrom Uplift Modeling", layout="wide")
    st.title("Hillstrom Uplift Modeling")
    st.caption(
        "Randomized email-campaign experiment (~64k customers). Two binary comparisons "
        "throughout -- Mens E-Mail vs. No E-Mail, Womens E-Mail vs. No E-Mail -- rather "
        "than one pooled treatment (see `decisions_log.md`)."
    )

    artifact = load_artifacts()

    tab_balance, tab_compare, tab_target, tab_shap = st.tabs(
        ["Randomization balance", "Model comparison", "Targeting simulator", "Uplift drivers"]
    )

    with tab_balance:
        render_balance_tab()

    arm = st.sidebar.radio("E-mail arm", options=list(artifact["arms"].keys()))
    st.sidebar.caption("Applies to Model comparison, Targeting simulator, and Uplift drivers.")

    with tab_compare:
        render_comparison_tab(artifact, arm)
    with tab_target:
        render_targeting_tab(artifact, arm)
    with tab_shap:
        render_shap_tab(artifact, arm)


if __name__ == "__main__":
    main()
