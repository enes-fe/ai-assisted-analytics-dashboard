from __future__ import annotations

import math
import uuid
from typing import Optional

import pandas as pd

from services.utils import format_col_name, is_id_column, select_label_column

from .schemas import FastAnalysisPlan, FastMetricPlan, FastSemanticPlan


# Aggregation inference by column name keywords. This remains the fallback path
# when no validated semantic metric metadata exists.
_SUM_KEYWORDS = [
    "goal", "gol", "assist", "asist", "sales", "revenue", "profit",
    "quantity", "cost", "amount", "delivery", "count", "total",
]
_MEAN_KEYWORDS = [
    "rating", "score", "rate", "percentage", "pct", "ratio", "xg", "xG",
    "avg", "average", "mean",
]

_SUPPORTED_RECOMMENDED_TYPES = {"bar", "line", "scatter", "pie", "donut", "area"}
_UNSUPPORTED_BAR_FALLBACK_TYPES = {"heatmap", "treemap"}
_SAFE_AGGREGATIONS = {"mean", "sum", "count", "min", "max"}
_ADDITIVE_SEMANTIC_TYPES = {"money", "quantity", "count"}
_METRIC_ROLES = {"outcome_metric", "driver_metric", "supporting_metric"}
_PIE_AVERAGE_FALLBACK_REASON = (
    "Pie/donut charts are intended for part-to-whole values. "
    "A bar chart is used for comparing averages."
)


def _infer_agg(col: str) -> str:
    n = col.lower()
    if any(k.lower() in n for k in _MEAN_KEYWORDS):
        return "mean"
    if any(k in n for k in _SUM_KEYWORDS):
        return "sum"
    return "sum"


def _valid_col(df: pd.DataFrame, col: Optional[str]) -> bool:
    return bool(col) and col in df.columns


def _is_numeric(df: pd.DataFrame, col: Optional[str]) -> bool:
    return _valid_col(df, col) and pd.api.types.is_numeric_dtype(df[col])


def _is_datetime_like(df: pd.DataFrame, col: Optional[str]) -> bool:
    if not _valid_col(df, col):
        return False
    if pd.api.types.is_numeric_dtype(df[col]):
        return False
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return True
    parsed = pd.to_datetime(df[col], errors="coerce")
    return parsed.notna().sum() >= 2


def _is_dimension_compatible(df: pd.DataFrame, col: Optional[str]) -> bool:
    if not _valid_col(df, col):
        return False
    series = df[col]
    if is_id_column(str(col), series):
        return False
    if pd.api.types.is_numeric_dtype(series):
        return series.dropna().nunique() <= 20
    return series.dropna().nunique() >= 1


def _is_pie_dimension(df: pd.DataFrame, col: Optional[str]) -> bool:
    return _pie_dimension_issue(df, col) is None


def _pie_dimension_issue(df: pd.DataFrame, col: Optional[str]) -> Optional[str]:
    if not _valid_col(df, col):
        return "pie_dimension_missing"
    if pd.api.types.is_numeric_dtype(df[col]):
        return "pie_dimension_not_categorical"
    if is_id_column(str(col), df[col]):
        return "pie_dimension_id_like"
    category_count = int(df[col].dropna().nunique())
    if category_count < 2:
        return "pie_dimension_too_few_categories"
    if category_count > 8:
        return (
            "Pie/donut charts are intended for low-cardinality part-to-whole values. "
            f"A bar chart is used because {format_col_name(str(col))} has {category_count} categories."
        )
    return None


def _is_metric_id_like(df: pd.DataFrame, col: Optional[str]) -> bool:
    if not _valid_col(df, col):
        return False
    return is_id_column(str(col), df[col])


def _records(data: pd.DataFrame) -> list[dict]:
    safe = data.copy()
    for col in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[col]):
            safe[col] = safe[col].dt.strftime("%Y-%m-%d")
    records = safe.to_dict(orient="records")
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

    The LLM only proposes semantic intent. Pandas builds all chartData here.
    """
    max_charts = max(0, int(max_charts or 0))
    if max_charts <= 0:
        return []

    metric_meta = _metric_metadata_map(fast_plan)
    charts: list[dict] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    skipped_reasons: list[str] = []

    semantic_charts = _charts_from_recommended_analyses(
        df=df,
        fast_plan=fast_plan,
        metric_meta=metric_meta,
        max_charts=max_charts,
        seen=seen,
        skipped_reasons=skipped_reasons,
    )
    charts.extend(semantic_charts)

    if len(charts) >= max_charts:
        return charts[:max_charts]

    fallback_reason = None
    if skipped_reasons and not semantic_charts:
        fallback_reason = "; ".join(_dedupe(skipped_reasons))[:500]
    elif semantic_charts:
        fallback_reason = "semantic_plan_had_fewer_valid_charts"

    heuristic_charts = _heuristic_charts(
        df=df,
        fast_plan=fast_plan,
        metric_meta=metric_meta,
        max_charts=max_charts - len(charts),
        seen=seen,
        fallback_reason=fallback_reason,
    )
    charts.extend(heuristic_charts)
    return charts[:max_charts]


def _charts_from_recommended_analyses(
    df: pd.DataFrame,
    fast_plan: FastSemanticPlan,
    metric_meta: dict[str, FastMetricPlan],
    max_charts: int,
    seen: set[tuple[str, str, str, str, str]],
    skipped_reasons: list[str],
) -> list[dict]:
    charts: list[dict] = []
    seen_requests: set[tuple[str, str, str, str, str]] = set()
    analyses = sorted(
        fast_plan.recommended_analyses or [],
        key=lambda item: item.priority,
        reverse=True,
    )

    for analysis in analyses:
        if len(charts) >= max_charts:
            break

        chart_type = _norm(analysis.type)
        metric = analysis.metric
        second_metric = analysis.second_metric
        dimension = analysis.dimension
        agg = _resolve_agg(metric, metric_meta.get(metric), analysis.aggregation)
        requested_key = _analysis_key(chart_type, metric, second_metric, dimension, agg)
        if requested_key in seen_requests:
            skipped_reasons.append(f"duplicate_recommendation:{chart_type}")
            continue
        seen_requests.add(requested_key)

        if chart_type not in _SUPPORTED_RECOMMENDED_TYPES:
            if chart_type in _UNSUPPORTED_BAR_FALLBACK_TYPES:
                reason = _unsupported_renderer_reason(chart_type)
                chart, reason, output_key = _fallback_bar_from_analysis(
                    df=df,
                    analysis=analysis,
                    metric_meta=metric_meta,
                    aggregation=agg,
                    domain=fast_plan.detected_domain,
                    reason=reason,
                )
            else:
                skipped_reasons.append(f"unsupported_chart_type:{analysis.type}")
                continue
        elif analysis.validation_status == "rejected":
            skipped_reasons.append(analysis.repair_reason or f"rejected_analysis:{chart_type}")
            continue
        else:
            chart, reason, output_key = _build_chart_from_recommended_analysis(
                df=df,
                analysis=analysis,
                chart_type=chart_type,
                metric_meta=metric_meta,
                aggregation=agg,
                domain=fast_plan.detected_domain,
            )

        if not chart:
            skipped_reasons.append(reason or f"invalid_recommendation:{chart_type}")
            continue

        output_key = output_key or requested_key
        if output_key in seen:
            skipped_reasons.append(f"duplicate_visual:{chart.get('type', chart_type)}")
            continue
        seen.add(output_key)
        fallback_reason = reason or (
            analysis.repair_reason if analysis.validation_status == "repaired" else None
        )
        charts.append(_with_metadata(chart, plan_source="semantic_plan", fallback_reason=fallback_reason))

    return charts


def _build_chart_from_recommended_analysis(
    df: pd.DataFrame,
    analysis: FastAnalysisPlan,
    chart_type: str,
    metric_meta: dict[str, FastMetricPlan],
    aggregation: str,
    domain: str,
) -> tuple[Optional[dict], Optional[str], Optional[tuple[str, str, str, str, str]]]:
    metric = analysis.metric
    second_metric = analysis.second_metric
    dimension = analysis.dimension

    if chart_type == "scatter":
        if not _chart_metric_allowed(df, metric, metric_meta.get(metric)):
            return None, "scatter_metric_invalid", None
        if not _chart_metric_allowed(df, second_metric, metric_meta.get(second_metric)):
            return None, "scatter_second_metric_invalid", None
        chart = _scatter_chart(df, metric, second_metric, dimension)
        key = _analysis_key("scatter", metric, second_metric, dimension, None)
        return chart, None if chart else "scatter_empty", key

    if chart_type in {"line", "area"}:
        if not _chart_metric_allowed(df, metric, metric_meta.get(metric)):
            return None, f"{chart_type}_metric_invalid", None
        if not _valid_col(df, dimension):
            return None, f"{chart_type}_dimension_missing", None
        if not (_is_datetime_like(df, dimension) or _is_numeric(df, dimension)):
            return None, f"{chart_type}_dimension_not_ordered", None
        chart = _line_chart(df, metric, dimension, aggregation, chart_type=chart_type)
        key = _analysis_key(chart_type, metric, None, dimension, aggregation)
        return chart, None if chart else f"{chart_type}_empty", key

    if chart_type == "bar":
        if not _chart_metric_allowed(df, metric, metric_meta.get(metric)):
            return None, "bar_metric_invalid", None
        if not _is_dimension_compatible(df, dimension):
            return None, "bar_dimension_invalid", None
        chart = _bar_chart(df, metric, dimension, domain, agg=aggregation)
        key = _analysis_key("bar", metric, None, dimension, aggregation)
        return chart, None if chart else "bar_empty", key

    if chart_type in {"pie", "donut"}:
        count_only = aggregation == "count" and not metric
        if not count_only and not _chart_metric_allowed(df, metric, metric_meta.get(metric)):
            return None, f"{chart_type}_metric_invalid", None

        dimension_issue = _pie_dimension_issue(df, dimension)
        if dimension_issue:
            if _is_human_reason(dimension_issue):
                return _fallback_bar_from_analysis(
                    df=df,
                    analysis=analysis,
                    metric_meta=metric_meta,
                    aggregation=aggregation,
                    domain=domain,
                    reason=dimension_issue,
                )
            return None, dimension_issue, None

        if count_only:
            chart = _count_pie_chart(df, dimension, chart_type=chart_type)
            key = _analysis_key(chart_type, "count", None, dimension, "count")
            return chart, None if chart else f"{chart_type}_empty", key

        if not _is_additive_or_count_metric(metric, metric_meta.get(metric), aggregation):
            reason = _PIE_AVERAGE_FALLBACK_REASON
            return _fallback_bar_from_analysis(
                df=df,
                analysis=analysis,
                metric_meta=metric_meta,
                aggregation=aggregation,
                domain=domain,
                reason=reason,
            )

        chart = _pie_chart(df, metric, dimension, domain, agg=aggregation, chart_type=chart_type)
        key = _analysis_key(chart_type, metric, None, dimension, aggregation)
        return chart, None if chart else f"{chart_type}_empty", key

    return None, f"unsupported_chart_type:{chart_type}", None


def _fallback_bar_from_analysis(
    df: pd.DataFrame,
    analysis: FastAnalysisPlan,
    metric_meta: dict[str, FastMetricPlan],
    aggregation: str,
    domain: str,
    reason: str,
) -> tuple[Optional[dict], Optional[str], Optional[tuple[str, str, str, str, str]]]:
    metric = analysis.metric
    dimension = analysis.dimension
    if aggregation == "count" and not metric:
        if not _is_dimension_compatible(df, dimension):
            return None, "bar_dimension_invalid", None
        chart = _count_bar_chart(df, dimension)
        if chart:
            _append_reason_to_chart(chart, reason)
        return chart, reason, _analysis_key("bar", "count", None, dimension, "count")

    if not _chart_metric_allowed(df, metric, metric_meta.get(metric)):
        return None, "bar_metric_invalid", None
    if not _is_dimension_compatible(df, dimension):
        return None, "bar_dimension_invalid", None

    chart = _bar_chart(df, metric, dimension, domain, agg=aggregation)
    if chart:
        _append_reason_to_chart(chart, reason)
    return chart, reason, _analysis_key("bar", metric, None, dimension, aggregation)


def _heuristic_charts(
    df: pd.DataFrame,
    fast_plan: FastSemanticPlan,
    metric_meta: dict[str, FastMetricPlan],
    max_charts: int,
    seen: set[tuple[str, str, str, str, str]],
    fallback_reason: Optional[str] = None,
) -> list[dict]:
    charts: list[dict] = []
    entity = fast_plan.primary_entity
    primary_metrics = [
        m for m in fast_plan.primary_metrics
        if _chart_metric_allowed(df, m, metric_meta.get(m), allow_unplanned=True)
    ]
    dimensions = [d for d in fast_plan.dimensions if _valid_col(df, d)]
    time_cols = [t for t in fast_plan.time_columns if _valid_col(df, t)]

    def append_chart(
        chart: Optional[dict],
        chart_type: str,
        metric: Optional[str],
        second_metric: Optional[str],
        dimension: Optional[str],
        aggregation: Optional[str],
    ) -> bool:
        if not chart:
            return False
        key = _analysis_key(chart_type, metric, second_metric, dimension, aggregation)
        if key in seen:
            return False
        seen.add(key)
        charts.append(_with_metadata(chart, plan_source="heuristic_fallback", fallback_reason=fallback_reason))
        return len(charts) >= max_charts

    if entity and _valid_col(df, entity):
        for metric in primary_metrics[:3]:
            agg = _infer_agg(metric)
            chart = _bar_chart(df, metric, entity, fast_plan.detected_domain, agg=agg)
            if append_chart(chart, "bar", metric, None, entity, agg):
                return charts

    for dim in dimensions:
        if primary_metrics:
            metric = primary_metrics[0]
            agg = _infer_agg(metric)
            chart = None
            if _is_pie_dimension(df, dim) and _is_additive_or_count_metric(metric, metric_meta.get(metric), agg):
                chart = _pie_chart(df, metric, dim, fast_plan.detected_domain, agg=agg)
            if append_chart(chart, "pie", metric, None, dim, agg):
                return charts
            if chart:
                break

    if len(primary_metrics) >= 2:
        chart = _scatter_chart(df, primary_metrics[0], primary_metrics[1], entity)
        if append_chart(chart, "scatter", primary_metrics[0], primary_metrics[1], entity, None):
            return charts

    if dimensions and primary_metrics:
        dim = dimensions[0]
        metric = primary_metrics[0]
        if not (entity and dim == entity):
            agg = _infer_agg(metric)
            chart = _bar_chart(df, metric, dim, fast_plan.detected_domain, agg=agg)
            if append_chart(chart, "bar", metric, None, dim, agg):
                return charts

    if time_cols and primary_metrics:
        metric = primary_metrics[0]
        agg = _infer_agg(metric)
        chart = _line_chart(df, metric, time_cols[0], agg)
        if append_chart(chart, "line", metric, None, time_cols[0], agg):
            return charts

    return charts


def _metric_metadata_map(fast_plan: FastSemanticPlan) -> dict[str, FastMetricPlan]:
    return {metric.column: metric for metric in fast_plan.metrics}


def _chart_metric_allowed(
    df: pd.DataFrame,
    metric: Optional[str],
    metric_meta: Optional[FastMetricPlan],
    allow_unplanned: bool = False,
) -> bool:
    if not _is_numeric(df, metric):
        return False
    if metric_meta:
        if metric_meta.validation_status == "rejected":
            return False
        if metric_meta.role not in _METRIC_ROLES:
            return False
    elif not allow_unplanned and _is_metric_id_like(df, metric):
        return False
    return True


def _resolve_agg(
    metric: Optional[str],
    metric_meta: Optional[FastMetricPlan],
    requested: Optional[str],
) -> str:
    if metric_meta and metric_meta.validation_status in {"accepted", "repaired"}:
        if metric_meta.aggregation in _SAFE_AGGREGATIONS:
            return metric_meta.aggregation
    if requested in _SAFE_AGGREGATIONS:
        return str(requested)
    return _infer_agg(metric or "")


def _is_additive_or_count_metric(
    metric: Optional[str],
    metric_meta: Optional[FastMetricPlan],
    aggregation: str,
) -> bool:
    if aggregation == "count":
        return True
    if aggregation != "sum":
        return False
    if metric_meta and metric_meta.validation_status in {"accepted", "repaired"}:
        return metric_meta.semantic_type in _ADDITIVE_SEMANTIC_TYPES
    return _infer_agg(metric or "") == "sum"


def _unsupported_renderer_reason(chart_type: str) -> str:
    label = format_col_name(chart_type)
    return f"{label} is not currently supported by the renderer. A bar chart is shown as a compatible fallback."


def _append_reason_to_chart(chart: dict, reason: Optional[str]) -> dict:
    if reason and _is_human_reason(reason):
        insight = str(chart.get("insight") or "").strip()
        if reason in insight:
            return chart
        chart["insight"] = f"{insight} {reason}".strip()
    return chart


def _is_human_reason(reason: Optional[str]) -> bool:
    if not reason:
        return False
    return ":" not in reason and "_" not in reason


def _analysis_key(
    chart_type: Optional[str],
    metric: Optional[str],
    second_metric: Optional[str],
    dimension: Optional[str],
    aggregation: Optional[str],
) -> tuple[str, str, str, str, str]:
    return (
        _norm(chart_type),
        _norm(metric),
        _norm(second_metric),
        _norm(dimension),
        _norm(aggregation),
    )


def _with_metadata(chart: dict, plan_source: str, fallback_reason: Optional[str] = None) -> dict:
    chart["plan_source"] = plan_source
    if fallback_reason:
        chart["fallback_reason"] = fallback_reason
        _append_reason_to_chart(chart, fallback_reason)
    return chart


def _norm(value: Optional[str]) -> str:
    return str(value or "").strip().lower()


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


# Chart builders

def _count_bar_chart(df: pd.DataFrame, dimension: str) -> Optional[dict]:
    try:
        if not _valid_col(df, dimension):
            return None
        grouped = (
            df[[dimension]]
            .dropna()
            .groupby(dimension, dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(5)
        )
        if grouped.empty:
            return None
        return {
            "id": _chart_id(),
            "type": "bar",
            "title": f"Kayit Sayisi - {format_col_name(dimension)}",
            "xAxisKey": dimension,
            "series": [{"key": "count"}],
            "chartData": _records(grouped),
            "insight": f"Bu grafik, {format_col_name(dimension)} kategorilerindeki kayit sayilarini ozetler.",
            "source": "groq_fast_semantic_plan",
        }
    except Exception:
        return None


def _bar_chart(df: pd.DataFrame, metric: str, dimension: str, domain: str, agg: Optional[str] = None) -> Optional[dict]:
    try:
        if not _is_numeric(df, metric) or not _valid_col(df, dimension):
            return None
        agg = agg if agg in _SAFE_AGGREGATIONS else _infer_agg(metric)
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
        agg_label = _agg_label_tr(agg)
        return {
            "id": _chart_id(),
            "type": "bar",
            "title": f"{agg_label} {format_col_name(metric)} - {format_col_name(dimension)}",
            "xAxisKey": dimension,
            "series": [{"key": metric}],
            "chartData": _records(grouped),
            "insight": (
                f"Bu grafik, {format_col_name(dimension)} kategorilerini "
                f"{format_col_name(metric)} icin {agg_label.lower()} degerlerle karsilastirir."
            ),
            "source": "groq_fast_semantic_plan",
        }
    except Exception:
        return None


def _count_pie_chart(df: pd.DataFrame, dimension: str, chart_type: str = "pie") -> Optional[dict]:
    try:
        if not _is_pie_dimension(df, dimension):
            return None
        grouped = (
            df[[dimension]]
            .dropna()
            .groupby(dimension, dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(8)
        )
        if grouped.empty:
            return None
        return {
            "id": _chart_id(),
            "type": chart_type if chart_type in {"pie", "donut"} else "pie",
            "title": f"Kayit Sayisi Dagilimi - {format_col_name(dimension)}",
            "xAxisKey": dimension,
            "series": [{"key": "count"}],
            "chartData": _records(grouped),
            "insight": f"Bu grafik, {format_col_name(dimension)} kategorilerinin kayit payini gosterir.",
            "source": "groq_fast_semantic_plan",
        }
    except Exception:
        return None


def _pie_chart(
    df: pd.DataFrame,
    metric: str,
    dimension: str,
    domain: str,
    agg: Optional[str] = None,
    chart_type: str = "pie",
) -> Optional[dict]:
    try:
        if not _is_numeric(df, metric) or not _valid_col(df, dimension):
            return None
        if pd.api.types.is_numeric_dtype(df[dimension]):
            return None
        category_count = int(df[dimension].dropna().nunique())
        if category_count < 2 or category_count > 8:
            return None

        agg = agg if agg in _SAFE_AGGREGATIONS else _infer_agg(metric)
        grouped = (
            df[[dimension, metric]]
            .dropna()
            .groupby(dimension, dropna=False)[metric]
            .agg(agg)
            .reset_index()
            .sort_values(metric, ascending=False)
            .head(8)
        )
        if grouped.empty:
            return None
        agg_label = _agg_label_tr(agg)
        return {
            "id": _chart_id(),
            "type": chart_type if chart_type in {"pie", "donut"} else "pie",
            "title": f"{agg_label} {format_col_name(metric)} Dagilimi - {format_col_name(dimension)}",
            "xAxisKey": dimension,
            "series": [{"key": metric}],
            "chartData": _records(grouped),
            "insight": (
                f"Bu grafik, {format_col_name(dimension)} kategorilerinin "
                f"{format_col_name(metric)} icindeki payini gosterir."
            ),
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
            "title": f"{format_col_name(metric1)} ve {format_col_name(metric2)} Iliskisi",
            "xAxisKey": metric1,
            "series": [{"key": metric2}],
            "chartData": _records(plot_df),
            "labelKey": "__label" if label_col else None,
            "labelName": label_col,
            "insight": (
                f"Bu dagilim, {format_col_name(metric1)} ve {format_col_name(metric2)} "
                "arasindaki iliskiyi gorsellestirir; nedensellik gostermez."
            ),
            "source": "groq_fast_semantic_plan",
        }
    except Exception:
        return None


def _line_chart(
    df: pd.DataFrame,
    metric: str,
    x_col: str,
    agg: Optional[str] = None,
    chart_type: str = "line",
) -> Optional[dict]:
    try:
        if not _is_numeric(df, metric) or not _valid_col(df, x_col):
            return None
        temp = df[[x_col, metric]].copy()
        if pd.api.types.is_numeric_dtype(temp[x_col]):
            temp[x_col] = pd.to_numeric(temp[x_col], errors="coerce")
        else:
            parsed = pd.to_datetime(temp[x_col], errors="coerce")
            if parsed.notna().sum() >= 2:
                temp[x_col] = parsed
            else:
                return None

        agg = agg if agg in _SAFE_AGGREGATIONS else _infer_agg(metric)
        grouped = (
            temp.dropna()
            .groupby(x_col)[metric]
            .agg(agg)
            .reset_index()
            .sort_values(x_col)
            .head(60)
        )
        if grouped.empty:
            return None
        return {
            "id": _chart_id(),
            "type": chart_type if chart_type in {"line", "area"} else "line",
            "title": f"{format_col_name(metric)} Trendi",
            "xAxisKey": x_col,
            "series": [{"key": metric}],
            "chartData": _records(grouped),
            "insight": f"Bu grafik, {format_col_name(metric)} degerinin sirali eksendeki degisimini ozetler.",
            "source": "groq_fast_semantic_plan",
        }
    except Exception:
        return None


def _agg_label_tr(agg: str) -> str:
    return {
        "mean": "Ortalama",
        "sum": "Toplam",
        "count": "Sayim",
        "min": "Minimum",
        "max": "Maksimum",
    }.get(agg, "Toplam")
