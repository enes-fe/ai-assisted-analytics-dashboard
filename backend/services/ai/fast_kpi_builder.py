from __future__ import annotations

import math
import re
from typing import Optional

import pandas as pd

from services.kpi_engine import MEAN_AGGREGATED_KEYWORDS
from services.utils import format_col_name

from .schemas import FastMetricPlan, FastSemanticPlan

_MAX_KPIS = 4


def _infer_agg(col: str) -> str:
    """Return 'mean' if the column name signals a rate/score metric, else 'sum'."""
    n = col.lower()
    if any(kw in n for kw in MEAN_AGGREGATED_KEYWORDS):
        return "mean"
    return "sum"


def _metric_metadata_map(
    fast_plan: FastSemanticPlan,
    validated_metrics: list[FastMetricPlan] | None = None,
) -> dict[str, FastMetricPlan]:
    metrics = validated_metrics if validated_metrics is not None else fast_plan.metrics
    return {metric.column: metric for metric in metrics}


def _is_rejected_metric(metric_meta: FastMetricPlan | None) -> bool:
    if metric_meta is None:
        return False
    return (
        metric_meta.validation_status == "rejected"
        or metric_meta.role in {"identifier", "dimension"}
    )


def _resolve_agg(metric: str, metric_meta: FastMetricPlan | None) -> str:
    if metric_meta and metric_meta.validation_status in {"accepted", "repaired"}:
        if metric_meta.aggregation in {"mean", "sum", "count", "min", "max"}:
            return metric_meta.aggregation
    return _infer_agg(metric)


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


def _aggregate(series: pd.Series, agg: str) -> float:
    if agg == "mean":
        return float(series.mean())
    if agg == "count":
        return float(series.count())
    if agg == "min":
        return float(series.min())
    if agg == "max":
        return float(series.max())
    return float(series.sum())


def _title_prefix(agg: str) -> str:
    return {
        "mean": "Avg",
        "sum": "Total",
        "count": "Count",
        "min": "Min",
        "max": "Max",
    }.get(agg, "Total")


def _agg_label_tr(agg: str) -> str:
    return {
        "mean": "ortalama",
        "sum": "toplam",
        "count": "sayim",
        "min": "minimum",
        "max": "maksimum",
    }.get(agg, "toplam")


def _stable_kpi_id(metric: str) -> str:
    """Generate a deterministic, collision-resistant KPI ID from column name."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", str(metric).strip()).strip("-").lower()
    return f"ai-kpi-{slug or 'metric'}"


def build_kpis_from_fast_plan(
    df: pd.DataFrame,
    fast_plan: FastSemanticPlan,
    validated_metrics: list[FastMetricPlan] | None = None,
) -> list[dict]:
    """Build KPI cards from a FastSemanticPlan using Pandas only.

    primary_metrics are evaluated first; secondary_metrics fill any remaining slots.
    Max 4 KPI cards.
    """
    metric_meta_by_column = _metric_metadata_map(fast_plan, validated_metrics)
    all_metrics = [
        m for m in (fast_plan.primary_metrics + fast_plan.secondary_metrics)
        if _is_numeric(df, m) and not _is_rejected_metric(metric_meta_by_column.get(m))
    ]
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_metrics: list[str] = []
    for metric in all_metrics:
        if metric not in seen:
            seen.add(metric)
            unique_metrics.append(metric)

    kpis: list[dict] = []
    used_ids: set[str] = set()

    for metric in unique_metrics[:_MAX_KPIS]:
        agg = _resolve_agg(metric, metric_meta_by_column.get(metric))
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if series.empty:
            continue

        safe = _safe_val(_aggregate(series, agg))
        if safe is None:
            continue

        kpi_id = _stable_kpi_id(metric)
        if kpi_id in used_ids:
            suffix = 2
            while f"{kpi_id}-{suffix}" in used_ids:
                suffix += 1
            kpi_id = f"{kpi_id}-{suffix}"
        used_ids.add(kpi_id)

        title_prefix = _title_prefix(agg)
        agg_label = _agg_label_tr(agg)

        kpis.append(
            {
                "id": kpi_id,
                "column": metric,
                "title": f"{title_prefix} {format_col_name(metric)}",
                "value": _fmt(safe, agg),
                "rawValue": safe,
                "trend": "AI-selected",
                "trendDirection": "neutral",
                "insight": f"Bu KPI, {format_col_name(metric)} icin {agg_label} degeri ozetler.",
            }
        )

    return kpis
