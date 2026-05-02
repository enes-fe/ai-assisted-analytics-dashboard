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

import re

MEANINGLESS_TOTAL_PATTERNS = [
    "delta", "diff", "duration", "elapsed", "lag", "offset",
    "tenure", "age_days", "day_count", "days_since", "days_to",
    "lead_time", "cycle_time",
]

# Negative prefixes: columns starting with these are likely error/incident metrics, not KPIs
NEGATIVE_PREFIX_PATTERNS = [
    "error", "fault", "fail", "penalty", "violation", "foul", "incident", "warning",
]

DOMAIN_PRIORITY_KW = [
    "revenue", "profit", "sales", "income", "margin", "earning",
    "goal", "assist", "score", "match", "game", "gol", "asist", "mac",
    "salary", "cost", "price", "amount", "total", "volume",
    "rating", "satisfaction", "conversion", "churn", "retention",
]


# --- Shared aggregation keyword rules ------------------------------------------
# Columns whose names match any of these tokens should be aggregated with mean.
MEAN_AGGREGATED_KEYWORDS: frozenset = frozenset([
    'percent', 'percentage', 'ratio', 'rate', 'pct', '%',
    'accuracy', 'completion',
    'rating', 'score', 'xg', 'avg', 'average', 'mean',
    'index', 'satisfaction',
])

# Subset of MEAN_AGGREGATED_KEYWORDS whose values should be displayed as a pct
PERCENT_DISPLAY_KEYWORDS: frozenset = frozenset([
    'percent', 'percentage', 'pct', '%',
])


def _col_matches_keywords(col_lower: str, keywords: frozenset) -> bool:
    return any(kw in col_lower for kw in keywords)


def is_meaningless_total(col_name: str) -> bool:
    col_lower = col_name.lower().replace("_", " ")
    return any(p in col_lower for p in MEANINGLESS_TOTAL_PATTERNS)


def _word_in_col(col_lower: str, keyword: str) -> bool:
    """Return True only if keyword appears as a word-boundary token in col_lower.
    E.g. 'goal' matches 'goals' or 'total_goal' but NOT 'errorLeadToGoal'.
    """
    # Split by common delimiters and check each token
    tokens = re.split(r'[_\s\-]+', col_lower)
    # Also handle camelCase: 'errorLeadToGoal' -> ['error', 'lead', 'to', 'goal']
    camel_tokens: list[str] = []
    for t in tokens:
        camel_tokens.extend(re.findall(r'[a-z]+|[0-9]+', re.sub(r'([A-Z])', r'_\1', t).lower()))
    all_tokens = set(tokens) | set(camel_tokens)
    return any(kw == tok or tok.startswith(kw) for tok in all_tokens for kw in [keyword])


def domain_priority(col_name: str) -> int:
    col_lower = col_name.lower()
    # Negative prefix check: if column starts with an error/fault prefix → deprioritize
    norm = col_lower.replace("_", "").replace(" ", "")
    if any(norm.startswith(p) for p in NEGATIVE_PREFIX_PATTERNS):
        return -1
    return 1 if any(_word_in_col(col_lower, kw) for kw in DOMAIN_PRIORITY_KW) else 0


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

def _kpi_id_for_column(col_name: str, used_ids: set[str]) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(col_name).strip()).strip("-").lower()
    base_id = f"kpi-{slug or 'metric'}"
    if base_id not in used_ids:
        used_ids.add(base_id)
        return base_id

    suffix = 2
    while f"{base_id}-{suffix}" in used_ids:
        suffix += 1
    kpi_id = f"{base_id}-{suffix}"
    used_ids.add(kpi_id)
    return kpi_id


def calculate_kpis(df: pd.DataFrame, selected_columns: list[str] | None = None) -> list:
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
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            test = pd.to_datetime(df[col].dropna().head(5), errors="coerce", format="mixed")
            if not test.isna().all():
                date_col = col
                break
        except Exception:
            continue

    # ── Filter meaningless columns ────────────────────────────────────────────────
    requested_cols = [
        c for c in (selected_columns or [])
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ][:4]

    if requested_cols:
        num_cols = requested_cols
    else:
        num_cols = [
            c for c in num_cols
            if not is_meaningless_total(c)
            and not is_id_column(c, df[c])
            and domain_priority(c) >= 0  # Exclude error/fault/penalty columns
            and df[c].dropna().nunique() > 2  # Exclude binary (0/1) columns
        ]
        if not num_cols:
            # Relax binary filter only as last resort
            num_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if not is_id_column(c, df[c])]

    # ── Prioritise domain-relevant columns ──────────────────────────────────────────
    num_cols_sorted = sorted(num_cols, key=lambda c: domain_priority(c), reverse=True)

    # ── Sort DataFrame by date if available ──────────────────────────────────────────
    work_df = df.copy()
    if date_col:
        try:
            work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
            work_df = work_df.sort_values(by=date_col)
        except Exception:
            pass

    if requested_cols:
        top_cols = requested_cols
    else:
        # Domain-priority columns first; fill remaining slots with centrality selection
        priority_cols = [c for c in num_cols_sorted if domain_priority(c) >= 1][:4]
        if len(priority_cols) < 4:
            remaining = [c for c in num_cols_sorted if c not in priority_cols]
            extra = get_central_cols(df, remaining, top_n=4 - len(priority_cols)) if remaining else []
            top_cols = priority_cols + extra
        else:
            top_cols = priority_cols

    kpis: list = []
    used_kpi_ids: set[str] = set()

    for i, col in enumerate(top_cols):
        if col == date_col and pd.api.types.is_datetime64_any_dtype(work_df[col]):
            continue
        if not requested_cols and is_meaningless_total(col):
            continue

        series = work_df[col].dropna()
        if series.empty:
            continue

        try:
            col_lower = col.lower()
            # Use shared keyword rules: mean for rate/score/rating/etc., sum for additive
            is_mean_agg = _col_matches_keywords(col_lower, MEAN_AGGREGATED_KEYWORDS)
            is_pct_display = _col_matches_keywords(col_lower, PERCENT_DISPLAY_KEYWORDS)

            if is_mean_agg:
                raw_val = series.mean()
                if isinstance(raw_val, (pd.Timestamp, datetime.datetime)):
                    raise TypeError("Not a numeric value")
                value = float(np.nan_to_num(raw_val))
                title_prefix = "Avg"
                if is_pct_display:
                    formatted_val = f"{value:.1f}%" if value < 100 else f"{value:.0f}%"
                else:
                    # Decimal average (e.g. rating = 7.43)
                    formatted_val = f"{value:.2f}"
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
            # Only compute half-period trend when a valid date column is present;
            # cross-sectional datasets (no date) must not show spurious trend arrows.
            trend_str = "N/A"
            trend_dir = "neutral"
            if date_col and len(series) >= 4:
                mid = len(series) // 2
                first_half_mean = series.iloc[:mid].mean()
                second_half_mean = series.iloc[mid:].mean()

                if first_half_mean != 0 and not pd.isna(first_half_mean) and not pd.isna(second_half_mean):
                    pct_change = ((second_half_mean - first_half_mean) / abs(first_half_mean)) * 100
                    trend_str = f"{'+' if pct_change >= 0 else ''}{pct_change:.1f}%"
                    trend_dir = "up" if pct_change > 1 else "down" if pct_change < -1 else "neutral"

            # ── Insight ───────────────────────────────────────────────────────
            unique_count = series.nunique()
            if unique_count <= 2:
                insight = "İkili (binary) veri sütunu; KPI yorumu sınırlı."
            elif is_pct_display:
                insight = f"Ortalama: {value:.1f}% — {'Artış' if trend_dir == 'up' else 'Düşüş' if trend_dir == 'down' else 'Stabil'}."
            elif is_mean_agg:
                insight = f"Ortalama değer: {formatted_val}."
            elif value == 0:
                insight = "Tüm dönemde kayıt sıfır."
            elif trend_dir == "neutral":
                insight = f"Değerler dönem boyunca stabil seyretti ({trend_str})."
            else:
                direction_text = "yükseliş" if trend_dir == "up" else "düşüş"
                insight = f"Dönem içi {direction_text} trendi: {trend_str}."

            kpis.append({
                "id": _kpi_id_for_column(col, used_kpi_ids),
                "title": f"{title_prefix} {format_col_name(col)}",
                "value": formatted_val,
                "rawValue": value,
                "trend": trend_str,
                "trendDirection": trend_dir,
                "insight": insight,
                "column": col,
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
