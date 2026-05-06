from __future__ import annotations

from typing import Any

import pandas as pd

from services.utils import is_id_column

from .schemas import FastAnalysisPlan, FastMetricPlan, FastSemanticPlan


METRIC_ROLES = {"outcome_metric", "driver_metric", "supporting_metric"}
ALLOWED_ROLES = METRIC_ROLES | {"identifier", "dimension"}
ALLOWED_SEMANTIC_TYPES = {
    "score",
    "percentage",
    "rate",
    "duration",
    "count",
    "money",
    "quantity",
    "index",
    "raw_numeric",
}
ALLOWED_AGGREGATIONS = {"mean", "sum", "count", "min", "max"}
ALLOWED_DIRECTIONS = {
    "higher_is_better",
    "lower_is_better",
    "context_dependent",
    "neutral",
}
ALLOWED_CONFIDENCE = {"low", "medium", "high"}
ALLOWED_ANALYSIS_TYPES = {"kpi", "bar", "pie", "donut", "scatter", "line", "area", "clustering"}

ROLE_ALIASES = {
    "primary_metric": "outcome_metric",
    "target_metric": "outcome_metric",
    "outcome": "outcome_metric",
    "target": "outcome_metric",
    "performance_metric": "outcome_metric",
    "secondary_metric": "supporting_metric",
    "metric": "supporting_metric",
    "measure": "supporting_metric",
    "supporting": "supporting_metric",
    "factor": "driver_metric",
    "driver": "driver_metric",
    "explanatory_metric": "driver_metric",
    "id": "identifier",
    "identifier_column": "identifier",
    "category": "dimension",
    "categorical": "dimension",
    "group": "dimension",
}

SEMANTIC_TYPE_ALIASES = {
    "percent": "percentage",
    "pct": "percentage",
    "ratio": "rate",
    "average": "score",
    "rating": "score",
    "numeric": "raw_numeric",
    "number": "raw_numeric",
    "currency": "money",
    "amount": "money",
    "time": "duration",
    "hours": "duration",
}

AGGREGATION_ALIASES = {
    "average": "mean",
    "avg": "mean",
    "total": "sum",
    "add": "sum",
    "minimum": "min",
    "maximum": "max",
}

DIRECTION_ALIASES = {
    "positive": "higher_is_better",
    "higher": "higher_is_better",
    "high_is_good": "higher_is_better",
    "negative": "lower_is_better",
    "lower": "lower_is_better",
    "low_is_good": "lower_is_better",
    "contextual": "context_dependent",
    "context": "context_dependent",
}

ANALYSIS_TYPE_ALIASES = {
    "cluster": "clustering",
    "clusters": "clustering",
    "histogram": "bar",
}


def validate_fast_semantic_plan(plan: FastSemanticPlan, df: pd.DataFrame) -> FastSemanticPlan:
    """Validate and annotate AI semantic planning output without calculating values."""
    valid_columns = set(map(str, df.columns.tolist()))
    valid_metrics = [_validate_metric(metric, df, valid_columns) for metric in plan.metrics]
    valid_analyses = [_validate_analysis(analysis, df, valid_columns) for analysis in plan.recommended_analyses]

    primary_metrics = _filter_existing(plan.primary_metrics, valid_columns)
    secondary_metrics = _filter_existing(plan.secondary_metrics, valid_columns)
    dimensions = _filter_existing(plan.dimensions, valid_columns)
    time_columns = _filter_existing(plan.time_columns, valid_columns)
    ignored_columns = _filter_existing(plan.ignored_columns, valid_columns)

    accepted_metric_columns = [
        metric.column
        for metric in valid_metrics
        if metric.validation_status in {"accepted", "repaired"}
        and metric.role in METRIC_ROLES
        and metric.column in valid_columns
    ]
    if not primary_metrics:
        primary_metrics = [
            metric.column
            for metric in valid_metrics
            if metric.validation_status in {"accepted", "repaired"}
            and metric.role == "outcome_metric"
        ]
    if not secondary_metrics:
        secondary_metrics = [
            column for column in accepted_metric_columns if column not in set(primary_metrics)
        ]
    if not dimensions:
        dimensions = [
            metric.column
            for metric in valid_metrics
            if metric.validation_status in {"accepted", "repaired"}
            and metric.role == "dimension"
            and metric.column in valid_columns
        ]

    return FastSemanticPlan(
        detected_domain=plan.detected_domain,
        primary_entity=plan.primary_entity if plan.primary_entity in valid_columns else None,
        primary_metrics=_dedupe(primary_metrics),
        secondary_metrics=_dedupe(secondary_metrics),
        dimensions=_dedupe(dimensions),
        time_columns=_dedupe(time_columns),
        ignored_columns=_dedupe(ignored_columns),
        confidence=plan.confidence,
        metrics=valid_metrics,
        recommended_analyses=valid_analyses,
        validation_summary=_validation_summary(valid_metrics, valid_analyses),
    )


def _validate_metric(metric: FastMetricPlan, df: pd.DataFrame, valid_columns: set[str]) -> FastMetricPlan:
    reasons: list[str] = []
    status = "accepted"

    role, role_changed = _normalize_enum(metric.role, ALLOWED_ROLES, ROLE_ALIASES, "supporting_metric")
    semantic_type, semantic_changed = _normalize_enum(
        metric.semantic_type,
        ALLOWED_SEMANTIC_TYPES,
        SEMANTIC_TYPE_ALIASES,
        "raw_numeric",
    )
    aggregation, aggregation_changed = _normalize_enum(
        metric.aggregation,
        ALLOWED_AGGREGATIONS,
        AGGREGATION_ALIASES,
        "",
    )
    direction, direction_changed = _normalize_enum(
        metric.direction,
        ALLOWED_DIRECTIONS,
        DIRECTION_ALIASES,
        "neutral",
    )
    confidence, confidence_changed = _normalize_confidence(metric.confidence)

    if role_changed:
        reasons.append("role_normalized")
    if semantic_changed:
        reasons.append("semantic_type_normalized")
    if aggregation_changed:
        reasons.append("aggregation_normalized")
    if direction_changed:
        reasons.append("direction_normalized")
    if confidence_changed:
        reasons.append("confidence_normalized")

    if metric.column not in valid_columns:
        return metric.copy(update={
            "validation_status": "rejected",
            "aggregation_source": "backend_inferred",
            "include_as_kpi": False,
            "include_in_clustering": False,
            "repair_reason": "column_not_found",
        })

    series = df[metric.column]
    col_lower = metric.column.lower()
    numeric = pd.api.types.is_numeric_dtype(series)
    id_like = _is_metric_id_like(str(metric.column), series)

    if role in METRIC_ROLES and not numeric:
        return metric.copy(update={
            "role": role,
            "semantic_type": semantic_type,
            "aggregation": aggregation or "count",
            "direction": direction,
            "confidence": confidence,
            "validation_status": "rejected",
            "aggregation_source": "backend_inferred",
            "include_as_kpi": False,
            "include_in_clustering": False,
            "repair_reason": "non_numeric_metric",
        })

    if role in METRIC_ROLES and id_like:
        return metric.copy(update={
            "role": role,
            "semantic_type": semantic_type,
            "aggregation": aggregation or "count",
            "direction": direction,
            "confidence": confidence,
            "validation_status": "rejected",
            "aggregation_source": "backend_inferred",
            "include_as_kpi": False,
            "include_in_clustering": False,
            "repair_reason": "id_like_metric",
        })

    inferred_type = _infer_semantic_type(metric.column, series, semantic_type)
    if inferred_type != semantic_type:
        semantic_type = inferred_type
        reasons.append("semantic_type_inferred")

    inferred_aggregation = _infer_aggregation(metric.column, semantic_type)
    if not aggregation:
        aggregation = inferred_aggregation
        reasons.append("aggregation_inferred")
    elif _should_repair_aggregation(metric.column, semantic_type, aggregation):
        aggregation = inferred_aggregation
        reasons.append(f"aggregation_repaired_to_{aggregation}")

    inferred_direction = _infer_direction(metric.column, semantic_type, direction)
    if inferred_direction != direction:
        direction = inferred_direction
        reasons.append("direction_repaired")

    if role in {"identifier", "dimension"}:
        include_as_kpi = False
        include_in_clustering = False
        if not aggregation:
            aggregation = "count"
    else:
        include_as_kpi = bool(metric.include_as_kpi)
        include_in_clustering = bool(metric.include_in_clustering) and numeric and not id_like

    if reasons:
        status = "repaired"
    aggregation_source = "ai_validated"
    if "aggregation_inferred" in reasons:
        aggregation_source = "backend_inferred"
    elif any(reason.startswith("aggregation_") for reason in reasons):
        aggregation_source = "backend_repaired"

    return metric.copy(update={
        "role": role,
        "semantic_type": semantic_type,
        "aggregation": aggregation,
        "direction": direction,
        "include_as_kpi": include_as_kpi,
        "include_in_clustering": include_in_clustering,
        "confidence": confidence,
        "validation_status": status,
        "aggregation_source": aggregation_source,
        "repair_reason": "; ".join(_dedupe(reasons)) or None,
    })


def _validate_analysis(analysis: FastAnalysisPlan, df: pd.DataFrame, valid_columns: set[str]) -> FastAnalysisPlan:
    reasons: list[str] = []
    analysis_type, type_changed = _normalize_enum(
        analysis.type,
        ALLOWED_ANALYSIS_TYPES,
        ANALYSIS_TYPE_ALIASES,
        "",
    )
    aggregation, aggregation_changed = _normalize_enum(
        analysis.aggregation,
        ALLOWED_AGGREGATIONS,
        AGGREGATION_ALIASES,
        None,
    )
    if type_changed:
        reasons.append("type_normalized")
    if aggregation_changed:
        reasons.append("aggregation_normalized")
    if not analysis_type:
        return analysis.copy(update={
            "validation_status": "rejected",
            "repair_reason": "invalid_analysis_type",
        })

    metric = analysis.metric if analysis.metric in valid_columns else None
    second_metric = analysis.second_metric if analysis.second_metric in valid_columns else None
    dimension = analysis.dimension if analysis.dimension in valid_columns else None
    features = [
        feature
        for feature in analysis.features
        if feature in valid_columns
        and pd.api.types.is_numeric_dtype(df[feature])
        and not _is_metric_id_like(str(feature), df[feature])
    ]

    if analysis.metric and metric is None:
        reasons.append("metric_filtered")
    if analysis.second_metric and second_metric is None:
        reasons.append("second_metric_filtered")
    if analysis.dimension and dimension is None:
        reasons.append("dimension_filtered")
    if len(features) != len(analysis.features):
        reasons.append("features_filtered")

    status = "repaired" if reasons else "accepted"
    return analysis.copy(update={
        "type": analysis_type,
        "metric": metric,
        "second_metric": second_metric,
        "dimension": dimension,
        "aggregation": aggregation,
        "features": features,
        "validation_status": status,
        "repair_reason": "; ".join(_dedupe(reasons)) or None,
    })


def _normalize_enum(value: Any, allowed: set[str], aliases: dict[str, str], default: str | None) -> tuple[str | None, bool]:
    if value is None:
        return default, bool(default)
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in allowed:
        return normalized, False
    if normalized in aliases:
        return aliases[normalized], True
    return default, True


def _normalize_confidence(value: Any) -> tuple[str, bool]:
    if isinstance(value, (float, int)):
        if value >= 0.75:
            return "high", True
        if value >= 0.45:
            return "medium", True
        return "low", True
    normalized = str(value).strip().lower()
    if normalized in ALLOWED_CONFIDENCE:
        return normalized, False
    return "medium", True


def _infer_semantic_type(col: str, series: pd.Series, current: str) -> str:
    name = col.lower()
    compact = name.replace("_", "").replace(" ", "")
    if any(token in compact for token in ["grade", "rating", "score"]):
        return "score"
    if any(token in compact for token in ["percentage", "percent", "pct", "rate", "ratio"]):
        return "percentage" if any(token in compact for token in ["percentage", "percent", "pct"]) else "rate"
    if "attendance" in compact and _numeric_range_within(series, 0, 100) and not _looks_total_or_count(name):
        return "percentage"
    if any(token in compact for token in ["hour", "duration", "minutes", "time", "reactiontime"]):
        return "duration"
    if any(token in compact for token in ["revenue", "sales", "profit", "amount", "cost", "price", "salary"]):
        return "money"
    if any(token in compact for token in ["quantity", "volume", "qty", "count"]):
        return "quantity" if "count" not in compact else "count"
    if "index" in compact:
        return "index"
    return current if current in ALLOWED_SEMANTIC_TYPES else "raw_numeric"


def _infer_aggregation(col: str, semantic_type: str) -> str:
    name = col.lower()
    compact = name.replace("_", "").replace(" ", "")
    if semantic_type in {"score", "percentage", "rate", "duration", "index"}:
        return "mean"
    if "attendance" in compact and not _looks_total_or_count(name):
        return "mean"
    if any(token in compact for token in ["grade", "rating", "score", "reactiontime", "gaminghour", "studyhour", "sleephour"]):
        return "mean"
    if any(token in compact for token in ["revenue", "sales", "profit", "amount", "quantity", "volume", "count", "cost"]):
        return "sum"
    return "mean"


def _should_repair_aggregation(col: str, semantic_type: str, aggregation: str) -> bool:
    if aggregation != "sum":
        return False
    name = col.lower()
    compact = name.replace("_", "").replace(" ", "")
    if semantic_type in {"score", "percentage", "rate", "duration", "index"}:
        return True
    if "attendance" in compact and not _looks_total_or_count(name):
        return True
    if any(token in compact for token in ["grade", "rating", "score", "reactiontime", "gaminghour", "studyhour", "sleephour"]):
        return True
    return False


def _infer_direction(col: str, semantic_type: str, current: str) -> str:
    name = col.lower().replace("_", "").replace(" ", "")
    lower_better_tokens = [
        "cost",
        "debt",
        "risk",
        "error",
        "defect",
        "mortality",
        "readmission",
        "complication",
        "churn",
        "dropout",
        "default",
        "fraud",
        "downtime",
        "waste",
        "delay",
        "reactiontime",
        "recency",
    ]
    if any(token in name for token in lower_better_tokens):
        return "lower_is_better" if current != "context_dependent" else current
    if any(token in name for token in ["grade", "rating", "score", "attendance", "revenue", "sales", "profit"]):
        return "higher_is_better"
    if semantic_type in {"duration", "raw_numeric"}:
        return current if current in ALLOWED_DIRECTIONS else "neutral"
    return current


def _looks_total_or_count(name: str) -> bool:
    compact = name.lower().replace("_", "").replace(" ", "")
    return any(token in compact for token in ["total", "count", "numberof", "num"])


def _is_metric_id_like(col: str, series: pd.Series) -> bool:
    compact = col.lower().replace("_", "").replace(" ", "")
    metric_tokens = [
        "grade",
        "rating",
        "score",
        "attendance",
        "hour",
        "duration",
        "time",
        "revenue",
        "sales",
        "profit",
        "cost",
        "amount",
        "quantity",
        "volume",
    ]
    if any(token in compact for token in metric_tokens):
        return False
    return is_id_column(col, series)


def _numeric_range_within(series: pd.Series, lower: float, upper: float) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return False
    return float(numeric.min()) >= lower and float(numeric.max()) <= upper


def _filter_existing(values: list[str], valid_columns: set[str]) -> list[str]:
    return _dedupe([value for value in values if value in valid_columns])


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _validation_summary(metrics: list[FastMetricPlan], analyses: list[FastAnalysisPlan]) -> dict[str, Any]:
    return {
        "metric_count": len(metrics),
        "accepted_metrics": sum(1 for item in metrics if item.validation_status == "accepted"),
        "repaired_metrics": sum(1 for item in metrics if item.validation_status == "repaired"),
        "rejected_metrics": sum(1 for item in metrics if item.validation_status == "rejected"),
        "analysis_count": len(analyses),
        "repaired_analyses": sum(1 for item in analyses if item.validation_status == "repaired"),
        "rejected_analyses": sum(1 for item in analyses if item.validation_status == "rejected"),
    }
