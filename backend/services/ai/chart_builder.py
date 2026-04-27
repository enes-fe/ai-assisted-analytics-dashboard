from __future__ import annotations

import uuid
from typing import Optional

import pandas as pd

from services.utils import format_col_name, sanitize_for_json

from .schemas import ChartPlan, SemanticDatasetPlan


AGG_FUNCS = {"sum", "mean", "count", "min", "max"}
CHART_TYPES = {"bar", "line", "pie", "scatter", "area", "table"}


def _valid_col(df: pd.DataFrame, col: Optional[str]) -> bool:
    return bool(col) and col in df.columns


def _is_numeric(df: pd.DataFrame, col: Optional[str]) -> bool:
    return _valid_col(df, col) and pd.api.types.is_numeric_dtype(df[col])


def _sort_and_limit(data: pd.DataFrame, value_col: str, sort: str, limit: int) -> pd.DataFrame:
    if sort in {"asc", "desc"} and value_col in data.columns:
        data = data.sort_values(value_col, ascending=(sort == "asc"))
    return data.head(max(1, min(int(limit or 10), 30)))


def _chart_id(plan: ChartPlan) -> str:
    return f"ai-{plan.chart_type}-{uuid.uuid4().hex[:8]}"


def _records(data: pd.DataFrame) -> list[dict]:
    safe = data.copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].dt.strftime("%Y-%m-%d")
    return sanitize_for_json(safe.to_dict(orient="records"))


def build_chart_from_plan(df: pd.DataFrame, plan: ChartPlan) -> Optional[dict]:
    if plan.chart_type not in CHART_TYPES or plan.aggregation not in AGG_FUNCS:
        return None

    chart_type = plan.chart_type
    metric = plan.metric
    second_metric = plan.second_metric
    dimension = plan.dimension

    try:
        if chart_type == "scatter":
            if not (_is_numeric(df, metric) and _is_numeric(df, second_metric)):
                return None
            plot_df = df[[metric, second_metric]].dropna().head(300)
            if plot_df.empty:
                return None
            chart_data = _records(plot_df)
            return {
                "id": _chart_id(plan),
                "type": "scatter",
                "title": plan.title or f"{format_col_name(metric)} vs {format_col_name(second_metric)}",
                "xAxisKey": metric,
                "series": [{"key": second_metric}],
                "chartData": chart_data,
                "insight": plan.reason,
                "source": "ollama_semantic_planner",
            }

        if chart_type == "table":
            cols = [c for c in [dimension, metric, second_metric] if _valid_col(df, c)]
            if not cols:
                cols = list(df.columns[: min(6, len(df.columns))])
            table_df = df[cols].dropna(how="all").head(max(1, min(plan.limit, 30)))
            if table_df.empty:
                return None
            return {
                "id": _chart_id(plan),
                "type": "table",
                "title": plan.title or "Semantic Table",
                "xAxisKey": cols[0],
                "series": [{"key": c} for c in cols[1:]],
                "chartData": _records(table_df),
                "insight": plan.reason,
                "source": "ollama_semantic_planner",
            }

        if not _valid_col(df, dimension):
            return None

        selected_cols = [c for c in [dimension, metric, second_metric] if _valid_col(df, c)]
        temp_df = df[selected_cols].copy()
        if chart_type in {"line", "area"}:
            parsed = pd.to_datetime(temp_df[dimension], errors="coerce")
            if parsed.notna().any():
                temp_df[dimension] = parsed

        if plan.aggregation == "count":
            grouped = temp_df.groupby(dimension, dropna=False).size().reset_index(name="count")
            value_col = "count"
            series_keys = [value_col]
        else:
            if not _is_numeric(df, metric):
                return None
            metric_cols = [metric]
            if _is_numeric(df, second_metric) and second_metric != metric:
                metric_cols.append(second_metric)
            grouped = temp_df.groupby(dimension, dropna=False)[metric_cols].agg(plan.aggregation).reset_index()
            value_col = metric
            series_keys = metric_cols

        if grouped.empty:
            return None
        if chart_type in {"line", "area"}:
            grouped = grouped.sort_values(dimension, ascending=True)
            if plan.limit:
                grouped = grouped.head(max(1, min(plan.limit, 30)))
        else:
            grouped = _sort_and_limit(grouped, value_col, plan.sort, plan.limit)

        chart_data = _records(grouped)
        return {
            "id": _chart_id(plan),
            "type": chart_type,
            "title": plan.title or f"{format_col_name(value_col)} by {format_col_name(dimension)}",
            "xAxisKey": dimension,
            "series": [{"key": key} for key in series_keys],
            "chartData": chart_data,
            "insight": plan.reason,
            "source": "ollama_semantic_planner",
        }
    except Exception:
        return None


def build_charts_from_semantic_plan(df: pd.DataFrame, semantic_plan: SemanticDatasetPlan) -> list[dict]:
    charts: list[dict] = []
    seen: set[tuple] = set()
    plans = sorted(semantic_plan.recommended_charts, key=lambda p: p.priority, reverse=True)

    for plan in plans:
        key = (plan.chart_type, plan.metric, plan.second_metric, plan.dimension, plan.aggregation)
        if key in seen:
            continue
        seen.add(key)
        chart = build_chart_from_plan(df, plan)
        if chart:
            charts.append(chart)

    return charts
