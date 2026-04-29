from __future__ import annotations

import math
import uuid
from typing import Optional

import pandas as pd

from services.utils import format_col_name

from .schemas import FastSemanticPlan

# Aggregation inference by column name keywords
_SUM_KEYWORDS = [
    "goal", "gol", "assist", "asist", "sales", "revenue", "profit",
    "quantity", "cost", "amount", "delivery", "count", "total",
]
_MEAN_KEYWORDS = [
    "rating", "score", "rate", "percentage", "pct", "ratio", "xg",
    "avg", "average", "mean",
]

_MAX_KPIS = 4


def _infer_agg(col: str) -> str:
    n = col.lower()
    if any(k in n for k in _MEAN_KEYWORDS):
        return "mean"
    return "sum"


def _is_numeric(df: pd.DataFrame, col: Optional[str]) -> bool:
    return bool(col) and col in df.columns and pd.api.types.is_numeric_dtype(df[col])


def _safe_val(v: float | None) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return float(v)


def _fmt(value: float, agg: str) -> str:
    """Format KPI value for display."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    if agg == "mean":
        return f"{value:.2f}"
    return f"{value:,.0f}"


def build_kpis_from_fast_plan(
    df: pd.DataFrame,
    fast_plan: FastSemanticPlan,
) -> list[dict]:
    """Build KPI cards from a FastSemanticPlan using Pandas only.

    primary_metrics are evaluated first; secondary_metrics fill any remaining slots.
    Max 4 KPI cards.
    """
    all_metrics = [
        m for m in (fast_plan.primary_metrics + fast_plan.secondary_metrics)
        if _is_numeric(df, m)
    ]
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_metrics: list[str] = []
    for m in all_metrics:
        if m not in seen:
            seen.add(m)
            unique_metrics.append(m)

    kpis: list[dict] = []
    for metric in unique_metrics[:_MAX_KPIS]:
        agg = _infer_agg(metric)
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if series.empty:
            continue

        raw_value = float(series.sum() if agg == "sum" else series.mean())
        safe = _safe_val(raw_value)
        if safe is None:
            continue

        kpis.append(
            {
                "id": f"ai-kpi-{uuid.uuid4().hex[:8]}",
                "column": metric,
                "title": format_col_name(metric),
                "value": _fmt(safe, agg),
                "rawValue": safe,
                "trend": "AI-selected",
                "trendDirection": "neutral",
                "insight": f"Bu KPI, {format_col_name(metric)} için {'toplam' if agg == 'sum' else 'ortalama'} değeri özetler.",
            }
        )

    return kpis
