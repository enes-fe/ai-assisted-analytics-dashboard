from __future__ import annotations

import re
import uuid
from typing import Optional

import pandas as pd

from services.utils import select_label_column

from .schemas import FastSemanticPlan

# Aggregation inference by column name keywords
_SUM_KEYWORDS = [
    "goal", "gol", "assist", "asist", "sales", "revenue", "profit",
    "quantity", "cost", "amount", "delivery", "count", "total",
]
_MEAN_KEYWORDS = [
    "rating", "score", "rate", "percentage", "pct", "ratio", "xg", "xG",
    "avg", "average", "mean",
]


def _infer_agg(col: str) -> str:
    n = col.lower()
    if any(k in n for k in _MEAN_KEYWORDS):
        return "mean"
    if any(k in n for k in _SUM_KEYWORDS):
        return "sum"
    return "sum"


def _valid_col(df: pd.DataFrame, col: Optional[str]) -> bool:
    return bool(col) and col in df.columns


def _is_numeric(df: pd.DataFrame, col: Optional[str]) -> bool:
    return _valid_col(df, col) and pd.api.types.is_numeric_dtype(df[col])


def _records(data: pd.DataFrame) -> list[dict]:
    safe = data.copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].dt.strftime("%Y-%m-%d")
    records = safe.to_dict(orient="records")
    # Sanitize NaN / Inf
    import math
    cleaned = []
    for row in records:
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean_row[k] = None
            else:
                clean_row[k] = v
        cleaned.append(clean_row)
    return cleaned


def _chart_id() -> str:
    return f"ai-fast-{uuid.uuid4().hex[:8]}"


def build_charts_from_fast_plan(
    df: pd.DataFrame,
    fast_plan: FastSemanticPlan,
    max_charts: int = 5,
) -> list[dict]:
    """Deterministically build charts from a FastSemanticPlan.

    The LLM never touches chartData — all aggregation is done by Pandas here.
    """
    charts: list[dict] = []
    entity = fast_plan.primary_entity
    primary_metrics = [m for m in fast_plan.primary_metrics if _is_numeric(df, m)]
    dimensions = [d for d in fast_plan.dimensions if _valid_col(df, d)]
    time_cols = [t for t in fast_plan.time_columns if _valid_col(df, t)]

    # ── a. Bar charts: first 3 primary_metrics by primary_entity ─────────────
    if entity and _valid_col(df, entity):
        for metric in primary_metrics[:3]:
            chart = _bar_chart(df, metric, entity, fast_plan.detected_domain)
            if chart:
                charts.append(chart)
                if len(charts) >= max_charts:
                    return charts

    # ── b. Scatter: primary_metrics[0] vs primary_metrics[1] ─────────────────
    if len(primary_metrics) >= 2:
        chart = _scatter_chart(df, primary_metrics[0], primary_metrics[1], entity)
        if chart:
            charts.append(chart)
            if len(charts) >= max_charts:
                return charts

    # ── c. Bar: first dimension × first primary_metric ────────────────────────
    if dimensions and primary_metrics:
        dim = dimensions[0]
        metric = primary_metrics[0]
        if not (entity and dim == entity):  # avoid duplicate
            chart = _bar_chart(df, metric, dim, fast_plan.detected_domain)
            if chart:
                charts.append(chart)
                if len(charts) >= max_charts:
                    return charts

    # ── d. Line: first primary_metric over time ───────────────────────────────
    if time_cols and primary_metrics:
        chart = _line_chart(df, primary_metrics[0], time_cols[0])
        if chart:
            charts.append(chart)
            if len(charts) >= max_charts:
                return charts

    return charts


# ── Chart builders ────────────────────────────────────────────────────────────

def _bar_chart(df: pd.DataFrame, metric: str, dimension: str, domain: str) -> Optional[dict]:
    try:
        if not _is_numeric(df, metric) or not _valid_col(df, dimension):
            return None
        agg = _infer_agg(metric)
        grouped = (
            df[[dimension, metric]]
            .dropna()
            .groupby(dimension, dropna=False)[metric]
            .agg(agg)
            .reset_index()
            .sort_values(metric, ascending=False)
            .head(5)
        )
        if grouped.empty:
            return None
        agg_label = "Toplam" if agg == "sum" else "Ortalama"
        return {
            "id": _chart_id(),
            "type": "bar",
            "title": f"{agg_label} {metric} — {dimension}",
            "xAxisKey": dimension,
            "series": [{"key": metric}],
            "chartData": _records(grouped),
            "insight": f"{metric} metriği {dimension} kırılımında {agg_label.lower()} olarak gösterilmektedir.",
            "source": "groq_fast_semantic_plan",
        }
    except Exception:
        return None


def _scatter_chart(df: pd.DataFrame, metric1: str, metric2: str, label_col: Optional[str] = None) -> Optional[dict]:
    try:
        if not (_is_numeric(df, metric1) and _is_numeric(df, metric2)):
            return None
        label_col = select_label_column(df, preferred=label_col, exclude={metric1, metric2})
        selected_cols = [metric1, metric2] + ([label_col] if label_col else [])
        plot_df = df[selected_cols].dropna(subset=[metric1, metric2]).head(300)
        if plot_df.empty:
            return None
        plot_df["__row"] = plot_df.index.astype(int) + 1
        if label_col:
            plot_df["__label"] = plot_df[label_col].astype(str)
        return {
            "id": _chart_id(),
            "type": "scatter",
            "title": f"{metric1} ve {metric2} İlişkisi",
            "xAxisKey": metric1,
            "series": [{"key": metric2}],
            "chartData": _records(plot_df),
            "labelKey": "__label" if label_col else None,
            "labelName": label_col,
            "insight": f"{metric1} ile {metric2} arasındaki ilişki gösteriliyor.",
            "source": "groq_fast_semantic_plan",
        }
    except Exception:
        return None


def _line_chart(df: pd.DataFrame, metric: str, time_col: str) -> Optional[dict]:
    try:
        if not _is_numeric(df, metric) or not _valid_col(df, time_col):
            return None
        temp = df[[time_col, metric]].copy()
        parsed = pd.to_datetime(temp[time_col], errors="coerce")
        if parsed.notna().sum() < 2:
            return None
        temp[time_col] = parsed
        agg = _infer_agg(metric)
        grouped = (
            temp.dropna()
            .groupby(time_col)[metric]
            .agg(agg)
            .reset_index()
            .sort_values(time_col)
            .head(60)
        )
        if grouped.empty:
            return None
        return {
            "id": _chart_id(),
            "type": "line",
            "title": f"{metric} Trendi",
            "xAxisKey": time_col,
            "series": [{"key": metric}],
            "chartData": _records(grouped),
            "insight": f"{metric} metriği zaman içindeki trendi gösteriyor.",
            "source": "groq_fast_semantic_plan",
        }
    except Exception:
        return None
