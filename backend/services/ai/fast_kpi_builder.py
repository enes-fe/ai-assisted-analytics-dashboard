from __future__ import annotations

import math
import re
from typing import Optional

import pandas as pd

from services.utils import format_col_name
from services.kpi_engine import MEAN_AGGREGATED_KEYWORDS

from .schemas import FastSemanticPlan

_MAX_KPIS = 4


def _infer_agg(col: str) -> str:
    """Return 'mean' if the column name signals a rate/score metric, else 'sum'."""
    n = col.lower()
    if any(kw in n for kw in MEAN_AGGREGATED_KEYWORDS):
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


def _stable_kpi_id(metric: str) -> str:
    """Generate a deterministic, collision-resistant KPI ID from column name."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(metric).strip()).strip("-").lower()
    return f"ai-kpi-{slug or 'metric'}"


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
    used_ids: set[str] = set()

    for metric in unique_metrics[:_MAX_KPIS]:
        agg = _infer_agg(metric)
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if series.empty:
            continue

        raw_value = float(series.sum() if agg == "sum" else series.mean())
        safe = _safe_val(raw_value)
        if safe is None:
            continue

        # Stable deterministic ID (no uuid)
        kpi_id = _stable_kpi_id(metric)
        # Ensure uniqueness within this batch
        if kpi_id in used_ids:
            suffix = 2
            while f"{kpi_id}-{suffix}" in used_ids:
                suffix += 1
            kpi_id = f"{kpi_id}-{suffix}"
        used_ids.add(kpi_id)

        title_prefix = "Avg" if agg == "mean" else "Total"

        kpis.append(
            {
                "id": kpi_id,
                "column": metric,
                "title": f"{title_prefix} {format_col_name(metric)}",
                "value": _fmt(safe, agg),
                "rawValue": safe,
                "trend": "AI-selected",
                "trendDirection": "neutral",
                "insight": f"Bu KPI, {format_col_name(metric)} için {'toplam' if agg == 'sum' else 'ortalama'} değeri özetler.",
            }
        )

    return kpis
