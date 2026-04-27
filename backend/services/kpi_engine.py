"""
services/kpi_engine.py
─────────────────────────────────────────────────────────────────────────
KPI calculation engine.  Selects the most business-relevant numeric
columns, computes aggregates (sum or mean), and calculates a half-period
trend to surface directional insights.
"""

import datetime
import numpy as np
import pandas as pd

from .utils import sanitize_for_json, is_id_column, format_col_name
from .column_profiler import get_central_cols


# ─── Domain heuristics ────────────────────────────────────────────────────────

MEANINGLESS_TOTAL_PATTERNS = [
    "delta", "diff", "duration", "elapsed", "lag", "offset",
    "tenure", "age_days", "day_count", "days_since", "days_to",
    "lead_time", "cycle_time",
]

DOMAIN_PRIORITY_KW = [
    "revenue", "profit", "sales", "income", "margin", "earning",
    "goal", "assist", "score", "match", "game", "gol", "asist", "mac",
    "salary", "cost", "price", "amount", "total", "count", "volume",
    "rating", "satisfaction", "conversion", "churn", "retention",
]


def is_meaningless_total(col_name: str) -> bool:
    col_lower = col_name.lower().replace("_", " ")
    return any(p in col_lower for p in MEANINGLESS_TOTAL_PATTERNS)


def domain_priority(col_name: str) -> int:
    col_lower = col_name.lower()
    return 1 if any(kw in col_lower for kw in DOMAIN_PRIORITY_KW) else 0


# ─── Histogram skip heuristic (used by chart_generator) ──────────────────────

HISTO_SKIP_PATTERNS = [
    "count", "quantity", "qty", "_id", "index", "rank", "order",
    "year", "month", "day", "hour", "minute", "second",
    "delta", "diff", "duration", "elapsed", "lag", "offset", "tenure",
    "zipcode", "zip", "postal", "phone", "latitude", "longitude", "lat", "lon",
]


def should_skip_histogram(col_name: str, profile: dict) -> bool:
    col_lower = col_name.lower().replace("_", " ")
    if any(p in col_lower for p in HISTO_SKIP_PATTERNS):
        return True
    skewness = abs(profile.get("skewness", 0))
    outlier_ratio = profile.get("outlier_ratio", 0)
    if skewness < 0.3 and outlier_ratio < 0.03:
        return True
    return False


# ─── Main function ────────────────────────────────────────────────────────────

def calculate_kpis(df: pd.DataFrame) -> list:
    """
    Selects up to 4 business-relevant numeric columns and computes:
      - Aggregated value (sum for totals, mean for percentages)
      - Half-period trend direction and percentage change
      - Simple directional insight text
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not num_cols:
        return sanitize_for_json([{
            "id": "kpi-count",
            "title": "Total Records",
            "value": f"{len(df):,}",
            "trend": "N/A",
            "trendDirection": "neutral",
            "insight": "Data is primarily categorical.",
        }])

    # ── Detect date column ────────────────────────────────────────────────────
    date_col = None
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_col = col
            break
        try:
            test = pd.to_datetime(df[col].dropna().head(5), errors="coerce")
            if not test.isna().all():
                date_col = col
                break
        except Exception:
            continue

    # ── Filter meaningless columns ────────────────────────────────────────────
    num_cols = [c for c in num_cols if not is_meaningless_total(c) and not is_id_column(c, df[c])]
    if not num_cols:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ── Prioritise domain-relevant columns ────────────────────────────────────
    num_cols_sorted = sorted(num_cols, key=lambda c: domain_priority(c), reverse=True)

    # ── Sort DataFrame by date if available ───────────────────────────────────
    work_df = df.copy()
    if date_col:
        try:
            work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
            work_df = work_df.sort_values(by=date_col)
        except Exception:
            pass

    top_cols = get_central_cols(df, num_cols_sorted, top_n=4)
    kpis: list = []

    for i, col in enumerate(top_cols):
        if col == date_col and pd.api.types.is_datetime64_any_dtype(work_df[col]):
            continue
        if is_meaningless_total(col):
            continue

        series = work_df[col].dropna()
        if series.empty:
            continue

        try:
            col_lower = col.lower()
            is_percentage = any(x in col_lower for x in ["percent", "ratio", "rate", "pct", "%", "accuracy", "completion"])
            if not is_percentage:
                if series.max() <= 100 and series.min() >= 0 and series.mean() <= 100 and "id" not in col_lower and "year" not in col_lower:
                    is_percentage = True

            if is_percentage:
                raw_val = series.mean()
                if isinstance(raw_val, (pd.Timestamp, datetime.datetime)):
                    raise TypeError("Not a numeric value")
                value = float(np.nan_to_num(raw_val))
                formatted_val = f"{value:.1f}%" if value < 100 else f"{value:.0f}%"
                title_prefix = "Avg"
            else:
                raw_val = series.sum()
                if isinstance(raw_val, (pd.Timestamp, datetime.datetime)):
                    raise TypeError("Not a numeric value")
                value = float(np.nan_to_num(raw_val))
                title_prefix = "Total"
                if value >= 1_000_000_000:
                    formatted_val = f"{value/1_000_000_000:.2f}B"
                elif value >= 1_000_000:
                    formatted_val = f"{value/1_000_000:.2f}M"
                elif value >= 1_000:
                    formatted_val = f"{value/1_000:.1f}K"
                else:
                    formatted_val = f"{value:,.0f}" if value % 1 == 0 else f"{value:,.2f}"

            # ── Trend ─────────────────────────────────────────────────────────
            trend_str = "N/A"
            trend_dir = "neutral"
            if len(series) >= 4:
                mid = len(series) // 2
                first_half_mean = series.iloc[:mid].mean()
                second_half_mean = series.iloc[mid:].mean()

                if first_half_mean != 0 and not pd.isna(first_half_mean) and not pd.isna(second_half_mean):
                    pct_change = ((second_half_mean - first_half_mean) / abs(first_half_mean)) * 100
                    trend_str = f"{'+' if pct_change >= 0 else ''}{pct_change:.1f}%"
                    trend_dir = "up" if pct_change > 1 else "down" if pct_change < -1 else "neutral"

            kpis.append({
                "id": f"kpi-{i}",
                "title": f"{title_prefix} {col}",
                "value": formatted_val,
                "rawValue": value,
                "trend": trend_str,
                "trendDirection": trend_dir,
                "insight": (
                    "Values are currently "
                    + ("trending up" if trend_dir == "up" else "trending down" if trend_dir == "down" else "stable")
                    + "."
                ),
            })
        except Exception as e:
            print(f"[KPI] Error calculating KPI for {col}: {e}")
            continue

    if not kpis:
        kpis.append({
            "id": "kpi-count",
            "title": "Total Records",
            "value": f"{len(df):,}",
            "trend": "N/A",
            "trendDirection": "neutral",
            "insight": "Data volume overview.",
        })

    return sanitize_for_json(kpis)
