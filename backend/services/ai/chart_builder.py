from __future__ import annotations

import uuid
from typing import Optional

import pandas as pd

from services.utils import format_col_name, sanitize_for_json, is_id_column, select_label_column

from .schemas import ChartPlan, SemanticDatasetPlan


AGG_FUNCS = {"sum", "mean", "count", "min", "max"}
CHART_TYPES = {"bar", "line", "pie", "donut", "scatter", "area", "table"}
PIE_AVERAGE_FALLBACK_REASON = (
    "Pie/donut charts are intended for part-to-whole values. "
    "A bar chart is used for comparing averages."
)


def _valid_col(df: pd.DataFrame, col: Optional[str]) -> bool:
    return bool(col) and col in df.columns


def _is_numeric(df: pd.DataFrame, col: Optional[str]) -> bool:
    return _valid_col(df, col) and pd.api.types.is_numeric_dtype(df[col])


def _is_binary(df: pd.DataFrame, col: Optional[str]) -> bool:
    """Return True if column only has 2 or fewer distinct numeric values (e.g. 0/1 flags)."""
    if not _is_numeric(df, col):
        return False
    return df[col].dropna().nunique() <= 2


def _is_id_dimension(df: pd.DataFrame, col: Optional[str]) -> bool:
    """Return True if column is an identifier (high cardinality or ID-named)."""
    if not _valid_col(df, col):
        return False
    return is_id_column(str(col), df[col])


def _sort_and_limit(data: pd.DataFrame, value_col: str, sort: str, limit: int, max_limit: int = 5) -> pd.DataFrame:
    if sort in {"asc", "desc"} and value_col in data.columns:
        data = data.sort_values(value_col, ascending=(sort == "asc"))
    return data.head(max(1, min(int(limit or max_limit), max_limit)))


def _chart_id(plan: ChartPlan) -> str:
    return f"ai-{plan.chart_type}-{uuid.uuid4().hex[:8]}"


def _chart_title(chart_type: str, value_col: Optional[str], dimension: Optional[str], second_metric: Optional[str] = None) -> str:
    if chart_type == "scatter" and value_col and second_metric:
        return f"{format_col_name(value_col)} ve {format_col_name(second_metric)} İlişkisi"
    if value_col and dimension:
        return f"{format_col_name(value_col)} - {format_col_name(dimension)}"
    if dimension:
        return f"Kayıt Sayısı - {format_col_name(dimension)}"
    return "Semantik Tablo"


def _chart_insight(chart_type: str, value_col: Optional[str], dimension: Optional[str], second_metric: Optional[str] = None) -> str:
    if chart_type == "scatter" and value_col and second_metric:
        return (
            f"Bu dağılım, {format_col_name(value_col)} ve {format_col_name(second_metric)} "
            "arasındaki ilişkiyi görselleştirir; nedensellik göstermez."
        )
    if chart_type in {"pie", "donut"} and value_col and dimension:
        return (
            f"Bu grafik, {format_col_name(dimension)} kategorilerinin "
            f"{format_col_name(value_col)} içindeki payını gösterir."
        )
    if value_col and dimension:
        return (
            f"Bu grafik, {format_col_name(dimension)} kategorilerini "
            f"{format_col_name(value_col)} için karşılaştırır."
        )
    if dimension:
        return f"Bu grafik, {format_col_name(dimension)} kategorilerindeki kayıt sayılarını özetler."
    return "Seçilen alanlar tablo görünümünde listelendi."


def _fallback_message_from_reason(reason: str) -> Optional[str]:
    reason_lower = reason.lower()
    if "not currently supported by the renderer" in reason_lower:
        for chart_type in ["heatmap", "treemap"]:
            if chart_type in reason_lower:
                label = chart_type.capitalize()
                return f"{label} is not currently supported by the renderer. A bar chart is shown as a compatible fallback."
        return "The requested chart is not currently supported by the renderer. A bar chart is shown as a compatible fallback."
    if "part-to-whole" in reason_lower and "comparing averages" in reason_lower:
        return PIE_AVERAGE_FALLBACK_REASON
    if "low-cardinality part-to-whole" in reason_lower:
        return "Pie/donut charts are intended for low-cardinality part-to-whole values. A bar chart is used as a compatible fallback."
    if "fallback" in reason_lower or "not supported" in reason_lower or "desteklen" in reason_lower:
        return "The requested chart type is not currently supported by the renderer. A compatible fallback chart is shown."
    return None


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
    eligibility_fallback_message: Optional[str] = None

    if chart_type in {"pie", "donut"} and _valid_col(df, dimension):
        if pd.api.types.is_numeric_dtype(df[dimension]):
            chart_type = "bar"
            eligibility_fallback_message = (
                "Pie/donut charts are intended for categorical part-to-whole values. "
                "A bar chart is shown as a compatible fallback."
            )
        else:
            category_count = int(df[dimension].dropna().nunique())
            if category_count > 8:
                chart_type = "bar"
                eligibility_fallback_message = (
                    "Pie/donut charts are intended for low-cardinality part-to-whole values. "
                    "A bar chart is used as a compatible fallback."
                )
            elif plan.aggregation not in {"sum", "count"}:
                chart_type = "bar"
                eligibility_fallback_message = PIE_AVERAGE_FALLBACK_REASON

    try:
        if chart_type == "scatter":
            if not (_is_numeric(df, metric) and _is_numeric(df, second_metric)):
                return None
            # Binary columns (0/1 flags) make scatter plots meaningless
            if _is_binary(df, metric) or _is_binary(df, second_metric):
                return None
            label_col = select_label_column(df, preferred=dimension, exclude={metric, second_metric})
            selected_cols = [metric, second_metric] + ([label_col] if label_col else [])
            plot_df = df[selected_cols].dropna(subset=[metric, second_metric]).head(300)
            if plot_df.empty:
                return None
            plot_df["__row"] = plot_df.index.astype(int) + 1
            if label_col:
                plot_df["__label"] = plot_df[label_col].astype(str)
            chart_data = _records(plot_df)
            return {
                "id": _chart_id(plan),
                "type": "scatter",
                "title": _chart_title(chart_type, metric, None, second_metric),
                "xAxisKey": metric,
                "series": [{"key": second_metric}],
                "chartData": chart_data,
                "labelKey": "__label" if label_col else None,
                "labelName": format_col_name(label_col) if label_col else None,
                "insight": _chart_insight(chart_type, metric, None, second_metric),
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
                "title": "Semantik Tablo",
                "xAxisKey": cols[0],
                "series": [{"key": c} for c in cols[1:]],
                "chartData": _records(table_df),
                "insight": _chart_insight(chart_type, None, cols[0]),
                "source": "ollama_semantic_planner",
            }

        if not _valid_col(df, dimension):
            return None
        # Skip charts where dimension is an identifier (e.g. Operation_Id, Player_Id)
        if _is_id_dimension(df, dimension):
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
            grouped = _sort_and_limit(
                grouped,
                value_col,
                plan.sort,
                plan.limit,
                max_limit=8 if chart_type in {"pie", "donut"} else 5,
            )

        chart_data = _records(grouped)
        # Derive title from actual columns, not the LLM plan (avoids title/data mismatch bugs)
        auto_title = f"{format_col_name(value_col)} - {format_col_name(dimension)} Analizi"
        insight = _chart_insight(chart_type, value_col, dimension, second_metric)
        fallback_message = eligibility_fallback_message or _fallback_message_from_reason(plan.reason or "")
        if fallback_message:
            insight = f"{insight} {fallback_message}"
        return {
            "id": _chart_id(plan),
            "type": chart_type,
            "title": auto_title,
            "xAxisKey": dimension,
            "series": [{"key": key} for key in series_keys],
            "chartData": chart_data,
            "insight": insight,
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
