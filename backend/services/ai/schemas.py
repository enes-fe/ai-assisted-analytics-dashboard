from __future__ import annotations

from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, validator


# Compact fast semantic plan used by the fast dashboard path.

class FastSemanticPlan(BaseModel):
    detected_domain: str
    primary_entity: Optional[str] = None
    primary_metrics: List[str] = Field(default_factory=list)
    secondary_metrics: List[str] = Field(default_factory=list)
    dimensions: List[str] = Field(default_factory=list)
    time_columns: List[str] = Field(default_factory=list)
    ignored_columns: List[str] = Field(default_factory=list)
    confidence: Union[float, str] = Field(default=0.5)
    metrics: list["FastMetricPlan"] = Field(default_factory=list)
    recommended_analyses: list["FastAnalysisPlan"] = Field(default_factory=list)
    validation_summary: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v: Any) -> float:
        """Accept float (0-1) or string labels like 'low'/'medium'/'high'."""
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))
        _map = {"low": 0.25, "medium": 0.5, "high": 0.9}
        return _map.get(str(v).strip().lower(), 0.5)

    @field_validator("recommended_analyses", mode="before")
    @classmethod
    def coerce_analyses(cls, v: Any) -> list:
        """Accept list of strings (e.g. ['kpi', 'bar']) or dicts or objects."""
        if not isinstance(v, list):
            return []
        result = []
        for item in v:
            if isinstance(item, str):
                # Simple string like "kpi" or "bar" → minimal dict
                result.append({"type": item.strip().lower()})
            elif isinstance(item, dict):
                result.append(item)
            else:
                result.append(item)
        return result

    @field_validator("metrics", mode="before")
    @classmethod
    def coerce_metrics(cls, v: Any) -> list:
        """Handle metrics that are dicts or objects; skip invalid items."""
        if not isinstance(v, list):
            return []
        result = []
        for item in v:
            if isinstance(item, dict) and "column" in item:
                result.append(item)
            elif hasattr(item, "column"):
                result.append(item)
            # Skip strings or malformed items
        return result


class FastMetricPlan(BaseModel):
    column: str
    role: str = "supporting_metric"
    semantic_type: str = "raw_numeric"
    aggregation: str = "mean"
    direction: str = "neutral"
    include_as_kpi: bool = False
    include_in_clustering: bool = False
    confidence: str = "medium"
    validation_status: str = "accepted"
    aggregation_source: str = "ai_validated"
    repair_reason: Optional[str] = None

    @field_validator("role", mode="before")
    @classmethod
    def coerce_role(cls, v: Any) -> str:
        allowed = {"outcome_metric", "driver_metric", "supporting_metric", "identifier", "dimension"}
        s = str(v).strip().lower()
        return s if s in allowed else "supporting_metric"

    @field_validator("aggregation", mode="before")
    @classmethod
    def coerce_aggregation(cls, v: Any) -> str:
        allowed = {"mean", "sum", "count", "min", "max"}
        s = str(v).strip().lower()
        return s if s in allowed else "mean"

    @field_validator("direction", mode="before")
    @classmethod
    def coerce_direction(cls, v: Any) -> str:
        allowed = {"higher_is_better", "lower_is_better", "context_dependent", "neutral"}
        s = str(v).strip().lower().replace(" ", "_")
        return s if s in allowed else "neutral"

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_metric_confidence(cls, v: Any) -> str:
        allowed = {"low", "medium", "high"}
        s = str(v).strip().lower()
        return s if s in allowed else "medium"


class FastAnalysisPlan(BaseModel):
    type: str
    metric: Optional[str] = None
    second_metric: Optional[str] = None
    dimension: Optional[str] = None
    aggregation: Optional[str] = None
    features: list[str] = Field(default_factory=list)
    priority: int = Field(default=3, ge=1, le=5)
    reason: str = ""
    validation_status: str = "accepted"
    repair_reason: Optional[str] = None


try:
    FastSemanticPlan.model_rebuild()
except AttributeError:
    FastSemanticPlan.update_forward_refs()


# Legacy verbose schema kept for backward compatibility with /api/analytics/core.

ColumnRole = Literal[
    "primary_entity",
    "primary_metric",
    "secondary_metric",
    "dimension",
    "time",
    "identifier",
    "quality_flag",
    "noise",
]
Aggregation = Literal["sum", "mean", "count", "min", "max", "none"]
ChartAggregation = Literal["sum", "mean", "count", "min", "max"]
ChartType = Literal["bar", "line", "pie", "donut", "scatter", "area", "table"]
SortOrder = Literal["asc", "desc", "none"]


class ColumnCard(BaseModel):
    name: str
    dtype: str
    missing_ratio: float
    unique_count: int
    sample_values: list[object] = Field(default_factory=list)
    numeric_stats: Optional[dict[str, Optional[float]]] = None
    top_values: Optional[list[dict[str, object]]] = None
    possible_semantic_hints: list[str] = Field(default_factory=list)


class ColumnSemantic(BaseModel):
    name: str
    role: ColumnRole
    business_meaning: str
    importance: int = Field(ge=1, le=5)
    preferred_aggregation: Aggregation


class ChartPlan(BaseModel):
    chart_type: ChartType
    title: str
    metric: Optional[str] = None
    second_metric: Optional[str] = None
    dimension: Optional[str] = None
    aggregation: ChartAggregation
    sort: SortOrder = "desc"
    limit: int = Field(default=10, ge=1, le=30)
    reason: str
    priority: int = Field(default=3, ge=1, le=5)


class SemanticDatasetPlan(BaseModel):
    detected_domain: str
    primary_entity: Optional[str] = None
    column_semantics: list[ColumnSemantic] = Field(default_factory=list)
    recommended_charts: list[ChartPlan] = Field(default_factory=list)


class PromptChartIntent(ChartPlan):
    confidence: float = Field(ge=0)
    understood_request: str


class ClusterNameSuggestion(BaseModel):
    cluster_id: int
    name: str
    reason: str = ""


class ClusterNameSuggestions(BaseModel):
    suggestions: list[ClusterNameSuggestion] = Field(default_factory=list)


# Compatibility helpers for Pydantic v1/v2.

def model_dump_compat(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()  # type: ignore[attr-defined]


def model_validate_compat(schema_model: type[BaseModel], data: object) -> BaseModel:
    if hasattr(schema_model, "model_validate"):
        return schema_model.model_validate(data)
    return schema_model.parse_obj(data)  # type: ignore[attr-defined]


def model_json_schema_compat(schema_model: type[BaseModel]) -> dict:
    if hasattr(schema_model, "model_json_schema"):
        return schema_model.model_json_schema()
    return schema_model.schema()  # type: ignore[attr-defined]
