"""Held-out evaluation utilities for uplift models: bootstrap-CI metrics,
Qini curves, cross-model comparison tables, and business-framed targeting
impact.

This module is the only place where test-set metrics are computed for this
project (see 04_evaluation.ipynb). Every function here is deliberately
independent of the lighter, provisional diagnostics used earlier
(`uplift_by_decile` and `compute_uplift_at_k` in 03_uplift_models.ipynb /
src/models.py), which exist only for a first, informal look at
train/validation data and say so explicitly in their own docstrings. The
functions below are the authoritative versions: every point estimate is
paired with a bootstrap confidence interval, because at this project's
per-arm test size (a few thousand rows, further split into treatment and
control), a single point estimate for a metric that depends on a rare
outcome and a predicted ranking is not precise enough on its own to support
a model-selection claim.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklift.metrics import qini_auc_score

ArrayLike = Sequence[Any] | pd.Series | np.ndarray


def _as_array(values: ArrayLike) -> np.ndarray:
    """Convert a Series/list/array-like into a plain numpy array."""
    return np.asarray(values)


def _bootstrap_indices(
    n: int, n_bootstrap: int, random_state: int | None
) -> np.ndarray:
    """Draw ``n_bootstrap`` sets of ``n`` row indices, sampled with replacement.

    A single ``np.random.default_rng`` is seeded once from ``random_state``,
    so repeated calls with the same ``random_state`` reproduce the exact same
    resampling draws. This is what lets 04_evaluation.ipynb compare every
    model within an arm on the same bootstrap draws by fixing
    ``random_state=42`` across every call for that arm, keeping the
    comparison apples-to-apples.

    Parameters
    ----------
    n : int
        Number of rows in the sample being resampled.
    n_bootstrap : int
        Number of bootstrap resamples to draw.
    random_state : int or None
        Seed for the random number generator.

    Returns
    -------
    np.ndarray of shape (n_bootstrap, n)
        Row indices for each bootstrap resample.
    """
    rng = np.random.default_rng(random_state)
    return rng.integers(0, n, size=(n_bootstrap, n))


def _percentile_ci(values: np.ndarray, ci_level: float) -> tuple[float, float]:
    """Compute a percentile bootstrap confidence interval from replicate values.

    NaN replicates (produced when a bootstrap resample happens to contain no
    treated or no control rows within the slice a metric is computed on) are
    dropped before taking percentiles, rather than letting a single
    degenerate resample turn the whole interval into NaN.

    Parameters
    ----------
    values : np.ndarray
        Bootstrap replicate estimates, one per resample. May contain NaN.
    ci_level : float
        Confidence level, e.g. 0.95 for a 95% interval.

    Returns
    -------
    tuple of float
        ``(ci_lower, ci_upper)``. Both are NaN if every replicate was NaN.
    """
    clean = values[~np.isnan(values)]
    if clean.size == 0:
        return float("nan"), float("nan")
    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(clean, 100 * alpha / 2))
    ci_upper = float(np.percentile(clean, 100 * (1 - alpha / 2)))
    return ci_lower, ci_upper


def _observed_uplift_at_k(
    y: np.ndarray, uplift_scores: np.ndarray, treatment: np.ndarray, k: float
) -> float:
    """Observed uplift (treated rate minus control rate) in the top-k fraction.

    Rows are ranked by descending predicted uplift and the top ``k`` fraction
    is kept; the returned value is
    ``mean(y | treatment=1, top-k) - mean(y | treatment=0, top-k)``.

    This reimplements, deliberately independently, the same idea as
    ``compute_uplift_at_k`` in ``src/models.py``. That function's own
    docstring reserves it for 03_uplift_models.ipynb's informal,
    train/validation-only comparisons and explicitly defers the
    authoritative, held-out version to this module -- this is that version.

    Parameters
    ----------
    y : np.ndarray
        Observed binary outcome.
    uplift_scores : np.ndarray
        Predicted individual uplift scores.
    treatment : np.ndarray
        Binary treatment indicator (1 = treated, 0 = control).
    k : float
        Fraction of the population to include, in (0, 1].

    Returns
    -------
    float
        Observed uplift in the top-k slice, or NaN if that slice contains no
        treated or no control rows.
    """
    if not 0 < k <= 1:
        raise ValueError(f"k must be in (0, 1], got {k}.")

    order = np.argsort(-uplift_scores, kind="stable")
    n_top = max(1, int(np.ceil(len(order) * k)))
    top_idx = order[:n_top]

    top_y = y[top_idx]
    top_treatment = treatment[top_idx]

    treated_y = top_y[top_treatment == 1]
    control_y = top_y[top_treatment == 0]

    if treated_y.size == 0 or control_y.size == 0:
        return float("nan")

    return float(treated_y.mean() - control_y.mean())


def qini_auc_with_ci(
    y: ArrayLike,
    uplift_scores: ArrayLike,
    treatment: ArrayLike,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int | None = None,
) -> dict[str, float]:
    """Compute Qini AUC with a percentile bootstrap confidence interval.

    The point estimate is ``sklift.metrics.qini_auc_score`` on the full
    sample (the same function and call signature already used for the
    informal validation-set comparison in 03_uplift_models.ipynb, so the
    point estimate here is directly comparable to that earlier number). The
    confidence interval resamples rows with replacement and recomputes the
    same score on each resample, since at this project's per-arm test size a
    single Qini AUC value is not precise enough on its own to support a
    model-selection claim.

    Parameters
    ----------
    y : array-like
        Observed binary outcome.
    uplift_scores : array-like
        Predicted individual uplift scores.
    treatment : array-like
        Binary treatment indicator (1 = treated, 0 = control).
    n_bootstrap : int, default=1000
        Number of bootstrap resamples.
    ci_level : float, default=0.95
        Confidence level for the interval.
    random_state : int or None, default=None
        Seed for the bootstrap resampling.

    Returns
    -------
    dict
        ``{"point_estimate", "ci_lower", "ci_upper", "n_bootstrap", "ci_level"}``.
    """
    y_arr, uplift_arr, treatment_arr = _as_array(y), _as_array(uplift_scores), _as_array(treatment)
    n = len(y_arr)

    point_estimate = float(qini_auc_score(y_arr, uplift_arr, treatment_arr))

    boot_indices = _bootstrap_indices(n, n_bootstrap, random_state)
    boot_estimates = np.full(n_bootstrap, np.nan)
    for i, idx in enumerate(boot_indices):
        try:
            boot_estimates[i] = qini_auc_score(y_arr[idx], uplift_arr[idx], treatment_arr[idx])
        except (ValueError, ZeroDivisionError):
            continue  # degenerate resample (e.g. no treated or no control rows)

    ci_lower, ci_upper = _percentile_ci(boot_estimates, ci_level)

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstrap": n_bootstrap,
        "ci_level": ci_level,
    }


def uplift_at_k_with_ci(
    y: ArrayLike,
    uplift_scores: ArrayLike,
    treatment: ArrayLike,
    k: float,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int | None = None,
) -> dict[str, float]:
    """Compute observed uplift@k with a percentile bootstrap confidence interval.

    Parameters
    ----------
    y : array-like
        Observed binary outcome.
    uplift_scores : array-like
        Predicted individual uplift scores.
    treatment : array-like
        Binary treatment indicator (1 = treated, 0 = control).
    k : float
        Fraction of the population to include, in (0, 1].
    n_bootstrap : int, default=1000
        Number of bootstrap resamples.
    ci_level : float, default=0.95
        Confidence level for the interval.
    random_state : int or None, default=None
        Seed for the bootstrap resampling.

    Returns
    -------
    dict
        ``{"k", "point_estimate", "ci_lower", "ci_upper", "n_top",
        "n_bootstrap", "ci_level"}``. ``n_top`` is the number of rows in the
        top-k slice of *this* sample (not scaled to any deployment
        population -- see ``compute_targeting_impact`` for that).
    """
    y_arr, uplift_arr, treatment_arr = _as_array(y), _as_array(uplift_scores), _as_array(treatment)
    n = len(y_arr)

    point_estimate = _observed_uplift_at_k(y_arr, uplift_arr, treatment_arr, k)

    boot_indices = _bootstrap_indices(n, n_bootstrap, random_state)
    boot_estimates = np.full(n_bootstrap, np.nan)
    for i, idx in enumerate(boot_indices):
        boot_estimates[i] = _observed_uplift_at_k(y_arr[idx], uplift_arr[idx], treatment_arr[idx], k)

    ci_lower, ci_upper = _percentile_ci(boot_estimates, ci_level)
    n_top = max(1, int(np.ceil(n * k)))

    return {
        "k": k,
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_top": n_top,
        "n_bootstrap": n_bootstrap,
        "ci_level": ci_level,
    }


def compute_qini_curve(
    y: ArrayLike,
    uplift_scores: ArrayLike,
    treatment: ArrayLike,
    n_points: int = 100,
) -> pd.DataFrame:
    """Compute a cumulative-gain Qini curve for plotting.

    Rows are ranked by descending predicted uplift. At each cutoff, the
    cumulative incremental gain is
    ``sum(y | treatment=1, top-cutoff) - sum(y | treatment=0, top-cutoff) *
    (n_treated_top_cutoff / n_control_top_cutoff)`` -- the standard
    Radcliffe (2007) Qini formulation, which corrects for the top slice not
    containing exactly equal numbers of treated and control rows. The
    random-targeting baseline is the straight line from ``(0, 0)`` to
    ``(1, total_gain)``, representing the expected gain from targeting a
    fraction of the population at random rather than by predicted uplift.

    This curve is a visualization aid, not a recomputation of
    ``sklift.metrics.qini_auc_score``'s own (possibly normalized) area --
    that point estimate, from ``qini_auc_with_ci``, is the authoritative
    number. This curve exists to show *where* in the ranking a model is
    strong or weak, which a single area number can hide.

    Parameters
    ----------
    y : array-like
        Observed binary outcome.
    uplift_scores : array-like
        Predicted individual uplift scores.
    treatment : array-like
        Binary treatment indicator (1 = treated, 0 = control).
    n_points : int, default=100
        Number of cutoffs to sample along the ranking for the returned
        curve. Every returned point is an exact cumulative value at a real
        cutoff (no interpolation); if ``n_points`` exceeds the sample size,
        every row is used as a cutoff instead.

    Returns
    -------
    pd.DataFrame
        Columns: ``population_fraction``, ``n_targeted`` (row count at that
        cutoff, in this sample), ``model_gain``, ``random_gain``.
    """
    y_arr, uplift_arr, treatment_arr = _as_array(y), _as_array(uplift_scores), _as_array(treatment)
    n = len(y_arr)

    order = np.argsort(-uplift_arr, kind="stable")
    y_sorted = y_arr[order]
    treatment_sorted = treatment_arr[order]

    is_treated = (treatment_sorted == 1).astype(float)
    is_control = (treatment_sorted == 0).astype(float)

    cum_n1 = np.cumsum(is_treated)
    cum_n0 = np.cumsum(is_control)
    cum_y1 = np.cumsum(y_sorted * is_treated)
    cum_y0 = np.cumsum(y_sorted * is_control)

    ratio = np.divide(
        cum_n1, cum_n0, out=np.zeros_like(cum_n1), where=cum_n0 > 0
    )
    model_gain_full = cum_y1 - cum_y0 * ratio

    cutoff_positions = np.unique(np.linspace(1, n, num=min(n_points, n), dtype=int))
    cutoff_idx = cutoff_positions - 1  # 0-based index into the cumulative arrays

    population_fraction = cutoff_positions / n
    model_gain = model_gain_full[cutoff_idx]
    total_gain = model_gain_full[-1]
    random_gain = population_fraction * total_gain

    return pd.DataFrame({
        "population_fraction": population_fraction,
        "n_targeted": cutoff_positions,
        "model_gain": model_gain,
        "random_gain": random_gain,
    })


def plot_qini_curve(qini_curve: pd.DataFrame, title: str = "Qini curve") -> go.Figure:
    """Plot a Qini curve (model gain vs. random-targeting baseline).

    Parameters
    ----------
    qini_curve : pd.DataFrame
        Output of ``compute_qini_curve``.
    title : str, default="Qini curve"
        Figure title.

    Returns
    -------
    go.Figure
        Plotly figure with two traces: the model's cumulative gain curve and
        the random-targeting baseline (dashed).
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=qini_curve["population_fraction"], y=qini_curve["model_gain"],
        mode="lines", name="Model",
    ))
    fig.add_trace(go.Scatter(
        x=qini_curve["population_fraction"], y=qini_curve["random_gain"],
        mode="lines", name="Random targeting", line={"dash": "dash"},
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Fraction of population targeted (ranked by predicted uplift)",
        yaxis_title="Cumulative incremental outcomes",
    )
    return fig



def plot_qini_curves_comparison(
    results: dict,
    comparison_df,
    best_model: str,
    title: str,
) -> go.Figure:
    """Overlay every model's Qini curve for one arm, highlighting the best model
    and annotating it with its bootstrap 95% CI on Qini AUC.

    Each model's curve comes straight from its own ``evaluate_uplift_model``
    output (``result["qini_curve"]``, i.e. ``compute_qini_curve``'s columns
    ``population_fraction`` / ``model_gain``). The random-targeting baseline
    is plotted once, since every model shares the same ``random_gain`` line
    on a given arm's test set (same y, same n).

    The CI shown is on the scalar Qini AUC (from ``comparison_df``), not a
    per-point curve CI: the bootstrap in ``qini_auc_with_ci`` resamples rows
    and recomputes the AUC each time, not the whole curve, so there is no
    pointwise CI to draw as an envelope around the best model's line without
    re-deriving it from the underlying bootstrap resamples (out of scope
    here). It is shown instead as a shaded horizontal reference band plus an
    annotation, both restricted to the best model's own legend group so it's
    unambiguous which model the interval belongs to.

    Parameters
    ----------
    results : dict
        model_name -> output of ``evaluate_uplift_model`` (must contain
        ``"qini_curve"``, the ``compute_qini_curve`` DataFrame).
    comparison_df : pd.DataFrame
        Output of ``compare_models_on_test``; must contain ``"model"``,
        ``"qini_auc"``, ``"qini_auc_ci_lower"``, ``"qini_auc_ci_upper"``.
    best_model : str
        Name of the model to highlight (bold line) and annotate with its CI.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    for model_name, result in results.items():
        curve = result["qini_curve"]
        is_best = model_name == best_model

        fig.add_trace(go.Scatter(
            x=curve["population_fraction"],
            y=curve["model_gain"],
            mode="lines",
            name=f"{model_name} (best)" if is_best else model_name,
            line=dict(width=3.5 if is_best else 1.5),
            opacity=1.0 if is_best else 0.55,
        ))

    # Random-targeting baseline, plotted once (identical across models on
    # the same arm's test set).
    any_curve = next(iter(results.values()))["qini_curve"]
    fig.add_trace(go.Scatter(
        x=any_curve["population_fraction"],
        y=any_curve["random_gain"],
        mode="lines",
        name="Random targeting",
        line=dict(dash="dash", color="gray"),
    ))

    # Best model's Qini AUC 95% CI, shown as a shaded horizontal band at the
    # curve's final (full-population) gain level plus a text annotation --
    # this is a CI on the scalar AUC, not on the curve shape itself (see
    # docstring).
    best_row = comparison_df.loc[comparison_df["model"] == best_model].iloc[0]
    auc_point = best_row["qini_auc"]
    ci_lower = best_row["qini_auc_ci_lower"]
    ci_upper = best_row["qini_auc_ci_upper"]

    fig.add_hrect(
        y0=ci_lower, y1=ci_upper,
        fillcolor="LightSalmon", opacity=0.15, line_width=0,
        annotation_text=f"{best_model} Qini AUC 95% CI",
        annotation_position="top left",
    )

    fig.add_annotation(
        text=(
            f"{best_model} Qini AUC: {auc_point:.4f} "
            f"(95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])"
        ),
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        align="left",
        font=dict(size=12),
        bgcolor="rgba(255,255,255,0.75)",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Fraction of population targeted (ranked by predicted uplift)",
        yaxis_title="Cumulative incremental outcomes",
        legend_title="Model",
        hovermode="x unified",
    )
    return fig


def evaluate_uplift_model(
    y: ArrayLike,
    uplift_scores: ArrayLike,
    treatment: ArrayLike,
    k_values: tuple[float, ...] = (0.10, 0.20, 0.30),
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int | None = None,
    n_curve_points: int = 100,
) -> dict[str, Any]:
    """Bundle Qini AUC, uplift@k, and a Qini curve for one fitted model.

    One call per model, so 04_evaluation.ipynb doesn't need to separately
    call ``qini_auc_with_ci``, ``uplift_at_k_with_ci`` (once per value in
    ``k_values``), and ``compute_qini_curve`` for every model in a
    comparison.

    Parameters
    ----------
    y : array-like
        Observed binary outcome.
    uplift_scores : array-like
        Predicted individual uplift scores.
    treatment : array-like
        Binary treatment indicator (1 = treated, 0 = control).
    k_values : tuple of float, default=(0.10, 0.20, 0.30)
        Targeting fractions to evaluate uplift@k at.
    n_bootstrap : int, default=1000
        Number of bootstrap resamples, shared across every metric computed
        here.
    ci_level : float, default=0.95
        Confidence level for every bootstrap interval computed here.
    random_state : int or None, default=None
        Seed shared across every bootstrap draw in this call. Passing the
        same ``random_state`` across multiple models being compared keeps
        their intervals built from the same resampling draws.
    n_curve_points : int, default=100
        Forwarded to ``compute_qini_curve``.

    Returns
    -------
    dict
        ``{"qini_auc": <dict from qini_auc_with_ci>,
        "uplift_at_k": {k: <dict from uplift_at_k_with_ci> for k in
        k_values}, "qini_curve": <DataFrame from compute_qini_curve>,
        "n_samples": int}``.
    """
    y_arr, uplift_arr, treatment_arr = _as_array(y), _as_array(uplift_scores), _as_array(treatment)

    qini_result = qini_auc_with_ci(y_arr, uplift_arr, treatment_arr, n_bootstrap, ci_level, random_state)

    uplift_at_k_results = {
        k: uplift_at_k_with_ci(y_arr, uplift_arr, treatment_arr, k, n_bootstrap, ci_level, random_state)
        for k in k_values
    }

    qini_curve = compute_qini_curve(y_arr, uplift_arr, treatment_arr, n_points=n_curve_points)

    return {
        "qini_auc": qini_result,
        "uplift_at_k": uplift_at_k_results,
        "qini_curve": qini_curve,
        "n_samples": len(y_arr),
    }


def compare_models_on_test(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Flatten a dict of ``evaluate_uplift_model`` outputs into one comparison table.

    Parameters
    ----------
    results : dict of str to dict
        Model name -> output of ``evaluate_uplift_model``, e.g. as built by
        a dict comprehension over several fitted models in
        04_evaluation.ipynb.

    Returns
    -------
    pd.DataFrame
        One row per model, columns ``model``, ``qini_auc``,
        ``qini_auc_ci_lower``, ``qini_auc_ci_upper``, then
        ``uplift_at_{k*100:.0f}pct`` / ``_ci_lower`` / ``_ci_upper`` for
        every ``k`` present across the supplied results, then
        ``n_samples``. Sorted by ``qini_auc`` descending, so
        ``.iloc[0]["model"]`` is the model with the highest point-estimate
        Qini AUC.
    """
    all_k_values = sorted({k for result in results.values() for k in result["uplift_at_k"]})

    rows = []
    for model_name, result in results.items():
        row: dict[str, Any] = {
            "model": model_name,
            "qini_auc": result["qini_auc"]["point_estimate"],
            "qini_auc_ci_lower": result["qini_auc"]["ci_lower"],
            "qini_auc_ci_upper": result["qini_auc"]["ci_upper"],
        }
        for k in all_k_values:
            label = f"uplift_at_{int(round(k * 100))}pct"
            k_result = result["uplift_at_k"].get(k)
            if k_result is None:
                row[label] = float("nan")
                row[f"{label}_ci_lower"] = float("nan")
                row[f"{label}_ci_upper"] = float("nan")
            else:
                row[label] = k_result["point_estimate"]
                row[f"{label}_ci_lower"] = k_result["ci_lower"]
                row[f"{label}_ci_upper"] = k_result["ci_upper"]
        row["n_samples"] = result["n_samples"]
        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values("qini_auc", ascending=False)
        .reset_index(drop=True)
    )


def compute_targeting_impact(
    y: ArrayLike,
    uplift_scores: ArrayLike,
    treatment: ArrayLike,
    k: float,
    total_population: int,
    value_per_incremental_outcome: float | None = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Translate a held-out uplift ranking into a business-framed targeting comparison.

    Compares three targeting policies at budget fraction ``k``:

    - **Model-guided**: target the top-``k`` fraction by predicted uplift.
      Its expected incremental-outcome rate is the observed uplift@k on the
      supplied (held-out) sample, extrapolated to ``k * total_population``
      targeted customers.
    - **Random targeting**: target the same number of customers
      (``k * total_population``) chosen at random. Its expected
      incremental-outcome rate is the overall (whole-sample) uplift, i.e.
      the plain treated-minus-control difference in means -- the
      unconditional average treatment effect for this arm.
    - **Treat everyone**: apply the treatment to the entire
      ``total_population`` rather than a subset, using the same overall
      uplift rate.

    Comparing model-guided to random isolates the value of the ranking
    itself; comparing model-guided to treat-everyone isolates the value of
    reaching fewer people at no worse an outcome.

    Parameters
    ----------
    y : array-like
        Observed binary outcome on the evaluation sample.
    uplift_scores : array-like
        Predicted individual uplift scores on the same sample.
    treatment : array-like
        Binary treatment indicator (1 = treated, 0 = control) on the same
        sample.
    k : float
        Targeting budget as a fraction of ``total_population``, in (0, 1].
    total_population : int
        Size of the customer base a real campaign would target. This is
        independent of ``len(y)`` (the evaluation sample size) by design:
        the uplift *rate* is estimated on held-out data and assumed to
        generalize to a (possibly much larger) deployment population, which
        should be supplied by the caller rather than inferred from the
        dataset -- see 04_evaluation.ipynb's explicit note on this
        assumption.
    value_per_incremental_outcome : float or None, default=None
        If supplied, a single dollar (or other currency) value attached to
        each incremental outcome, used to add ``*_incremental_value`` keys
        to the result. If ``None``, the result stays in incremental-outcome
        count terms only. This function does not attempt to derive a dollar
        value internally (e.g. from a conversion-given-visit rate and an
        average order value); that is a deployment-specific assumption the
        caller must supply explicitly.
    n_bootstrap : int, default=1000
        Number of bootstrap resamples for the model-guided uplift rate's
        confidence interval.
    ci_level : float, default=0.95
        Confidence level for that interval.
    random_state : int or None, default=None
        Seed for the bootstrap resampling.

    Returns
    -------
    dict
        ``{"k", "total_population", "n_targeted",
        "model_guided_uplift_rate": {"point_estimate", "ci_lower",
        "ci_upper"}, "overall_uplift_rate",
        "model_guided_incremental_visits", "random_targeting_incremental_visits",
        "treat_everyone_incremental_visits", "value_per_incremental_outcome"}``,
        plus ``"*_incremental_value"`` keys if
        ``value_per_incremental_outcome`` is not ``None``. ("Visits" in the
        key names reflects this project's primary outcome, `visit`; the
        values are equally valid for whatever binary outcome ``y``
        represents.)
    """
    y_arr, uplift_arr, treatment_arr = _as_array(y), _as_array(uplift_scores), _as_array(treatment)

    n_targeted = int(round(k * total_population))

    model_guided = uplift_at_k_with_ci(
        y_arr, uplift_arr, treatment_arr, k, n_bootstrap, ci_level, random_state
    )

    overall_uplift_rate = float(
        y_arr[treatment_arr == 1].mean() - y_arr[treatment_arr == 0].mean()
    )

    model_guided_incremental_visits = model_guided["point_estimate"] * n_targeted
    random_targeting_incremental_visits = overall_uplift_rate * n_targeted
    treat_everyone_incremental_visits = overall_uplift_rate * total_population

    result: dict[str, Any] = {
        "k": k,
        "total_population": total_population,
        "n_targeted": n_targeted,
        "model_guided_uplift_rate": {
            "point_estimate": model_guided["point_estimate"],
            "ci_lower": model_guided["ci_lower"],
            "ci_upper": model_guided["ci_upper"],
        },
        "overall_uplift_rate": overall_uplift_rate,
        "model_guided_incremental_visits": model_guided_incremental_visits,
        "random_targeting_incremental_visits": random_targeting_incremental_visits,
        "treat_everyone_incremental_visits": treat_everyone_incremental_visits,
        "value_per_incremental_outcome": value_per_incremental_outcome,
    }

    if value_per_incremental_outcome is not None:
        result["model_guided_incremental_value"] = (
            model_guided_incremental_visits * value_per_incremental_outcome
        )
        result["random_targeting_incremental_value"] = (
            random_targeting_incremental_visits * value_per_incremental_outcome
        )
        result["treat_everyone_incremental_value"] = (
            treat_everyone_incremental_visits * value_per_incremental_outcome
        )

    return result