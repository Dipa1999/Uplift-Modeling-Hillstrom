"""Data loading, model-ready encoding, and split utilities for the Hillstrom dataset.

This module contains the only place where the raw Hillstrom CSV is read and
typed, where covariate balance across randomized treatment arms is computed,
and where the checked raw data is turned into model-ready, split, and
persisted train/validation/test data. Notebooks should import from here
rather than re-implementing this logic inline, so that every notebook and the
Streamlit app share exactly the same data-loading, balance-checking,
encoding, and splitting behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklift

# ADDED: 'history_segment' to REQUIRED_COLUMNS
REQUIRED_COLUMNS: tuple[str, ...] = (
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
)

# Columns whose dtype should be treated as a fixed-category label rather than
# a numeric quantity, even though some of them are stored as 0/1 integers.
CATEGORICAL_COLUMNS: tuple[str, ...] = ("zip_code", "channel", "segment")
BINARY_COLUMNS: tuple[str, ...] = (
    "mens", "womens", "newbie", "visit", "conversion")


def _make_ordered_history_segment(series: pd.Series) -> pd.Series:
    """Convert raw history_segment strings into an ordered categorical."""
    # The bins naturally sort alphabetically because of their prefixes (e.g., "1) $0 - $100")
    categories = sorted(series.dropna().unique())
    cat_type = pd.CategoricalDtype(categories=categories, ordered=True)
    return series.astype(cat_type)


def _download_hillstrom(destination: Path) -> None:
    """Download the raw Hillstrom dataset and save it as a CSV at ``destination``."""
    from sklift.datasets import fetch_hillstrom

    bunch = fetch_hillstrom(target_col="all", return_X_y_t=False)

    df = bunch.data.copy()
    df["segment"] = bunch.treatment
    for target_column in bunch.target.columns:
        df[target_column] = bunch.target[target_column]

    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)


def load_hillstrom(filepath: str, download_if_missing: bool = True) -> pd.DataFrame:
    """Load and type-cast the raw Hillstrom email marketing dataset."""
    path = Path(filepath)

    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"No file found at '{filepath}' and download_if_missing=False. "
                "Either place the Hillstrom CSV there manually, or call "
                "load_hillstrom with download_if_missing=True."
            )
        _download_hillstrom(path)

    df = pd.read_csv(path)

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Loaded CSV is missing expected Hillstrom columns: "
            f"{sorted(missing_columns)}. Check that the correct file is at "
            f"'{filepath}' and that its schema matches the standard "
            "Hillstrom dataset."
        )

    # Apply Categorical types
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype("category")

    # Apply Binary types
    for col in BINARY_COLUMNS:
        df[col] = df[col].astype("int8")

    # ADDED: Apply Ordered Categorical for history_segment
    df["history_segment"] = _make_ordered_history_segment(
        df["history_segment"])

    # ADDED: Downcast memory footprint as claimed in the EDA notebook
    df["recency"] = pd.to_numeric(df["recency"], downcast="integer")
    for col in ["history", "spend"]:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def _standardized_mean_diff(group_values: pd.Series, reference_values: pd.Series) -> float:
    """Compute the standardized mean difference (SMD) between two samples."""
    group_mean, reference_mean = group_values.mean(), reference_values.mean()
    group_var, reference_var = group_values.var(
        ddof=1), reference_values.var(ddof=1)
    pooled_std = np.sqrt((group_var + reference_var) / 2)

    if pooled_std == 0:
        return 0.0

    return (group_mean - reference_mean) / pooled_std


def compute_balance_table(
    df: pd.DataFrame,
    covariates: list[str],
    treatment_col: str = "segment",
    reference_group: str = "No E-Mail",
    smd_threshold: float = 0.1,
) -> pd.DataFrame:
    """Compute covariate balance across randomized treatment arms via SMD."""
    if reference_group not in df[treatment_col].unique():
        raise ValueError(
            f"reference_group='{reference_group}' not found in "
            f"df['{treatment_col}']. Available groups: "
            f"{sorted(df[treatment_col].unique().astype(str))}."
        )

    treatment_groups = [
        group for group in df[treatment_col].unique() if group != reference_group
    ]
    reference_mask = df[treatment_col] == reference_group

    rows: list[dict[str, object]] = []

    for covariate in covariates:
        column = df[covariate]

        if pd.api.types.is_numeric_dtype(column) and column.nunique() > 2:
            levels = [(covariate, column)]
        else:
            dummies = pd.get_dummies(column.astype(str), prefix=covariate)
            levels = [(level_name, dummies[level_name])
                      for level_name in dummies.columns]

        for level_name, level_series in levels:
            reference_values = level_series[reference_mask]

            for group in treatment_groups:
                group_mask = df[treatment_col] == group
                group_values = level_series[group_mask]

                smd = _standardized_mean_diff(group_values, reference_values)

                rows.append(
                    {
                        "covariate": level_name,
                        "group": group,
                        "reference": reference_group,
                        "smd": smd,
                        "flag_imbalanced": abs(smd) > smd_threshold,
                    }
                )

    return pd.DataFrame(rows)


def check_history_segment_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Check that ``history_segment`` is a clean, non-overlapping binning of ``history``."""
    grouped = df.groupby("history_segment", observed=True)["history"]
    summary = grouped.agg(n="count", history_min="min", history_max="max")

    if isinstance(df["history_segment"].dtype, pd.CategoricalDtype) and df[
        "history_segment"
    ].cat.ordered:
        summary = summary.reindex(df["history_segment"].cat.categories)
        summary.index.name = "history_segment"

    return summary.reset_index()


def check_design_matrix_rank(
    df: pd.DataFrame,
    numeric_covariates: list[str],
    categorical_covariates: list[str],
    drop_first: bool = True,
    include_intercept: bool = True,
) -> dict[str, object]:
    """Check whether a one-hot encoded design matrix is full column rank."""
    design_parts = [df[numeric_covariates].astype(float)]

    for col in categorical_covariates:
        dummies = pd.get_dummies(df[col].astype(
            str), prefix=col, drop_first=drop_first)
        design_parts.append(dummies.astype(float))

    design_matrix = pd.concat(design_parts, axis=1)

    if include_intercept:
        design_matrix.insert(0, "intercept", 1.0)

    rank = int(np.linalg.matrix_rank(design_matrix.to_numpy()))
    n_columns = design_matrix.shape[1]

    return {
        "n_columns": n_columns,
        "rank": rank,
        "full_rank": rank == n_columns,
    }


def compute_vif(
    df: pd.DataFrame,
    numeric_covariates: list[str],
    categorical_covariates: list[str],
) -> pd.DataFrame:
    """Compute the Variance Inflation Factor (VIF) for a set of covariates."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    design_parts = [df[numeric_covariates].astype(float)]

    for col in categorical_covariates:
        dummies = pd.get_dummies(df[col].astype(
            str), prefix=col, drop_first=True)
        design_parts.append(dummies.astype(float))

    design_matrix = pd.concat(design_parts, axis=1)
    design_matrix.insert(0, "intercept", 1.0)

    matrix_values = design_matrix.to_numpy()

    vif_rows = [
        {
            "covariate": column_name,
            "vif": variance_inflation_factor(matrix_values, i),
        }
        for i, column_name in enumerate(design_matrix.columns)
        if column_name != "intercept"
    ]

    return pd.DataFrame(vif_rows).sort_values("vif", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model-ready encoding and train/val/test split utilities.
# Used by notebooks/02_data_prep.ipynb to turn the checked raw data from
# load_hillstrom into persisted, model-ready splits for uplift modeling.
# ---------------------------------------------------------------------------


def build_model_ready_frame(
    df: pd.DataFrame,
    drop_columns: tuple[str, ...] = ("history_segment",),
    categorical_columns: tuple[str, ...] = ("zip_code", "channel"),
    drop_first: bool = True,
) -> pd.DataFrame:
    """Turn checked raw Hillstrom data into a single model-ready encoded frame.

    This applies only transformations that are fixed, deterministic functions
    of each row's own values (dropping a redundant column, one-hot encoding a
    categorical column with pre-known levels) and therefore carry no risk of
    train/validation/test leakage. It does NOT scale or standardize numeric
    columns: those transforms must be fit on training data only, so they
    belong in a per-split (or per-fold) sklearn Pipeline in the modeling
    notebook, not baked once into data saved to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``load_hillstrom`` (or any frame with the same schema).
    drop_columns : tuple of str, default=("history_segment",)
        Columns to drop before encoding. ``history_segment`` is dropped by
        default because ``check_history_segment_consistency`` (see
        01_eda.ipynb) confirmed it is a clean, non-overlapping binning of the
        continuous ``history`` column and therefore carries no information
        ``history`` doesn't already have, for any model class used
        downstream.
    categorical_columns : tuple of str, default=("zip_code", "channel")
        Columns to one-hot encode. Left as a parameter rather than hardcoded
        so callers can encode a different subset if the covariate set
        changes.
    drop_first : bool, default=True
        Passed to ``pd.get_dummies``. Kept ``True`` to match the full-rank
        design matrix confirmed by ``check_design_matrix_rank`` during EDA;
        setting this to ``False`` would reintroduce the dummy variable trap.

    Returns
    -------
    pd.DataFrame
        A new frame (input is not mutated) with ``drop_columns`` removed,
        ``categorical_columns`` one-hot encoded as int8, and every other
        column (including ``segment``, ``visit``, ``conversion``, ``spend``)
        left untouched.
    """
    frame = df.drop(columns=list(drop_columns)).copy()

    dummies = pd.get_dummies(
        frame[list(categorical_columns)], drop_first=drop_first
    ).astype("int8")

    frame = pd.concat(
        [frame.drop(columns=list(categorical_columns)), dummies], axis=1
    )
    return frame


def stratified_train_val_test_split(
    df: pd.DataFrame,
    treatment_col: str = "segment",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/validation/test, preserving treatment-arm proportions.

    Stratifying on ``treatment_col`` (rather than leaving the split
    unstratified, or additionally stratifying on the rare outcome columns) is
    the one property that matters for this being a randomized-experiment
    dataset: it guarantees every split is still a valid, self-contained
    comparison across the three arms, at whatever proportions the full
    dataset has. We deliberately do not also stratify on
    ``visit``/``conversion``: with about 64k rows split only on a
    near-balanced 3-way treatment, the rare-outcome proportions in each split
    are already close to the population proportion by the same
    law-of-large-numbers logic used to interpret the SMD balance check in
    EDA, so adding a compound stratification key would add complexity
    without a measurable benefit.

    Parameters
    ----------
    df : pd.DataFrame
        Model-ready frame, typically the output of ``build_model_ready_frame``.
    treatment_col : str, default="segment"
        Column to stratify on.
    test_size : float, default=0.2
        Fraction of the full dataset held out as the test split.
    val_size : float, default=0.2
        Fraction of the full dataset held out as the validation split.
        Applied as a fraction of the remaining (non-test) data internally, so
        the final proportions are approximately
        ``(1 - test_size - val_size, val_size, test_size)`` of the original
        dataset.
    random_state : int, default=42
        Passed to both internal calls to ``train_test_split`` for
        reproducibility.

    Returns
    -------
    tuple of pd.DataFrame
        ``(train_df, val_df, test_df)``, each with the original index
        preserved (not reset), so rows can always be traced back to the
        source data.
    """
    from sklearn.model_selection import train_test_split

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[treatment_col],
        random_state=random_state,
    )

    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size,
        stratify=train_val_df[treatment_col], random_state=random_state,
    )

    return train_df, val_df, test_df


def make_binary_treatment(
    df: pd.DataFrame,
    treatment_group: str,
    control_group: str = "No E-Mail",
    treatment_col: str = "segment",
) -> pd.DataFrame:
    """Reduce a 3-arm split to a binary treatment/control frame for one arm.

    scikit-uplift's meta-learners (S-/T-/X-Learner) are built around a single
    binary treatment indicator, not a native multi-arm treatment column (see
    decisions_log.md). Rather than collapsing the 3-arm ``segment`` column
    into a single binary column once in the data-prep notebook -- which would
    force a choice of which two arms to compare before that choice is
    actually needed -- this function keeps that decision in the modeling
    notebook: it's called once per analysis (e.g. once for "Mens E-Mail" vs
    "No E-Mail", once for "Womens E-Mail" vs "No E-Mail"), each time
    returning a filtered frame with a fresh binary ``treatment`` column.

    Parameters
    ----------
    df : pd.DataFrame
        A frame containing ``treatment_col`` (e.g. train_df, val_df, or
        test_df as returned by ``stratified_train_val_test_split``).
    treatment_group : str
        The value of ``treatment_col`` to code as ``treatment == 1``.
    control_group : str, default="No E-Mail"
        The value of ``treatment_col`` to code as ``treatment == 0``.
    treatment_col : str, default="segment"
        Column holding the original 3-arm assignment.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` filtered to only ``treatment_group`` and
        ``control_group`` rows, with a new int8 ``treatment`` column added
        (1 = ``treatment_group``, 0 = ``control_group``). ``treatment_col``
        is kept in the output for traceability.

    Raises
    ------
    ValueError
        If ``treatment_group`` or ``control_group`` is not a value present
        in ``df[treatment_col]``.
    """
    valid_groups = set(df[treatment_col].unique())
    for group in (treatment_group, control_group):
        if group not in valid_groups:
            raise ValueError(
                f"'{group}' not found in df['{treatment_col}']. "
                f"Available groups: {sorted(str(g) for g in valid_groups)}."
            )

    mask = df[treatment_col].isin([treatment_group, control_group])
    result = df.loc[mask].copy()
    result["treatment"] = (result[treatment_col]
                            == treatment_group).astype("int8")
    return result


def save_processed_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
    feature_columns: list[str],
    outcome_columns: list[str] | None = None,
    treatment_col: str = "segment",
) -> None:
    """Persist train/val/test splits as parquet, plus a manifest describing them.

    Splits are saved as three separate files (rather than one file with a
    ``split`` indicator column) so that loading e.g. ``train.parquet`` in the
    modeling notebook can never accidentally include validation or test rows
    -- the file boundary itself is the leakage guard, not a filter condition
    that could be forgotten. Parquet (not CSV) is used because it preserves
    dtypes (int8 flags, the one-hot int8 dummies, the ``segment``
    categorical) exactly, so nothing needs to be re-inferred or re-cast on
    load.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Splits as returned by ``stratified_train_val_test_split``.
    output_dir : str
        Directory to write into (created if missing). Expected to be
        ``data/processed/``.
    feature_columns : list of str
        Column names to record as model features in the manifest (not
        enforced on the saved files, which keep every column for flexibility
        -- this is purely so the modeling notebook doesn't have to hardcode
        the list).
    outcome_columns : list of str or None, default=None
        Column names to record as outcomes in the manifest. Defaults to
        ``["visit", "conversion", "spend"]`` if not provided.
    treatment_col : str, default="segment"
        Column name to record as the treatment assignment in the manifest.

    Returns
    -------
    None
        Writes ``train.parquet``, ``val.parquet``, ``test.parquet``, and
        ``feature_manifest.json`` into ``output_dir``.
    """
    if outcome_columns is None:
        outcome_columns = ["visit", "conversion", "spend"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, split_df in splits.items():
        split_df.to_parquet(out_path / f"{name}.parquet", index=False)

    manifest = {
        "feature_columns": list(feature_columns),
        "outcome_columns": list(outcome_columns),
        "treatment_col": treatment_col,
        "split_sizes": {name: int(len(split_df)) for name, split_df in splits.items()},
        "segment_proportions": {
            name: split_df[treatment_col].value_counts(normalize=True).round(4).to_dict()
            for name, split_df in splits.items()
        },
    }
    with open(out_path / "feature_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
