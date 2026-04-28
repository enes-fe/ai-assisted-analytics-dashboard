"""
services/column_profiler.py
─────────────────────────────────────────────────────────────────────────
Statistical profiling of individual DataFrame columns.
Determines type (continuous, discrete, categorical, datetime, boolean)
and computes distributional statistics used by chart_generator and
kpi_engine.
"""

import numpy as np
import pandas as pd
from scipy import stats

from .utils import is_id_column


def get_column_profile(df: pd.DataFrame, col: str) -> dict:
    """
    Returns a profile dict for *col* in *df*.

    Returned keys depend on the detected type:
      - Always: "type", "nunique"
      - Numeric: "mean", "skewness", "outlier_ratio", "min", "max"
      - Categorical: "dominant_flag", "cardinality"
    """
    series = df[col]
    total_count = len(df)

    # ── Type detection ────────────────────────────────────────────────────────
    is_datetime = False
    if series.dtype == "object":
        try:
            if pd.api.types.is_datetime64_any_dtype(series):
                is_datetime = True
            else:
                test_dates = pd.to_datetime(series.dropna().head(50), errors="coerce")
                if len(test_dates) > 0 and test_dates.notna().mean() > 0.5:
                    is_datetime = True
        except Exception:
            pass
    elif pd.api.types.is_datetime64_any_dtype(series):
        is_datetime = True

    is_boolean = series.nunique() <= 2 and (
        series.dropna().isin([0, 1, True, False, "True", "False", "Yes", "No"]).all()
    )

    nunique = series.nunique()

    if is_datetime:
        return {"type": "datetime", "nunique": nunique}
    if is_boolean:
        return {"type": "boolean", "nunique": nunique}

    # ── Numeric ───────────────────────────────────────────────────────────────
    if pd.api.types.is_numeric_dtype(series):
        col_type = "discrete_numeric" if (nunique <= 15 and np.issubdtype(series.dtype, np.integer)) else "continuous_numeric"

        clean_series = series.dropna()
        if clean_series.empty:
            return {"type": "unknown"}

        q1 = clean_series.quantile(0.25)
        q3 = clean_series.quantile(0.75)
        iqr = q3 - q1
        outliers = clean_series[(clean_series < (q1 - 1.5 * iqr)) | (clean_series > (q3 + 1.5 * iqr))]

        return {
            "type": col_type,
            "nunique": nunique,
            "mean": clean_series.mean(),
            "skewness": stats.skew(clean_series),
            "outlier_ratio": len(outliers) / total_count if total_count > 0 else 0,
            "min": clean_series.min(),
            "max": clean_series.max(),
        }

    # ── Categorical ────────────────────────────────────────────────────────────
    value_counts = series.value_counts(normalize=True)
    dominant_flag = (value_counts > 0.8).any() if not value_counts.empty else False

    return {
        "type": "categorical",
        "nunique": nunique,
        "dominant_flag": dominant_flag,
        "cardinality": nunique,
    }


def get_central_cols(df: pd.DataFrame, num_cols: list, top_n: int = 4) -> list:
    """
    Returns the *top_n* most 'central' numeric columns — those with the
    highest average Spearman correlation with all other numeric columns.

    Falls back gracefully if the DataFrame is too small.
    """
    if len(num_cols) <= top_n:
        return num_cols

    try:
        variances = df[num_cols].var()
        active_cols = variances[variances > variances.quantile(0.25)].index.tolist()
        if len(active_cols) < top_n:
            active_cols = num_cols

        sample_size = min(500, len(df))
        sample = df[active_cols].dropna().sample(sample_size, random_state=42)
        if len(sample) < 5:
            return active_cols[:top_n]

        corr = sample.corr(method="spearman").abs()
        np.fill_diagonal(corr.values, 0)

        centrality = corr.mean().sort_values(ascending=False)
        return centrality.head(top_n).index.tolist()
    except Exception:
        return num_cols[:top_n]
