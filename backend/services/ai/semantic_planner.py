from __future__ import annotations

import json
import os
import re

import pandas as pd

from .column_cards import build_column_cards
from .ollama_client import call_ollama_structured
from .schemas import ChartPlan, ColumnSemantic, SemanticDatasetPlan, model_dump_compat


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


SYSTEM_PROMPT = """You are a semantic column selector and dashboard planner.
Prefer domain-important metrics over random numeric columns.
For football/soccer datasets, goals, assists, ratings, xG, shots, minutes, position, team and player name are important.
For sales datasets, revenue, profit, sales, quantity, product, region and date are important.
For HR datasets, salary, attrition, performance, overtime, department, tenure and employee role are important.
For logistics datasets, delivery time, delay, lead time, stock, route, vendor, quantity and cost are important.
Use only existing column names exactly as provided.
Do not invent columns.
Do not calculate metrics.
Classify all provided columns in column_semantics.
Every recommended chart must reference exact column names.
For bar, line, area, and pie charts, set dimension to an exact column name.
For sum, mean, min, and max aggregations, set metric to an exact numeric column name.
For count aggregation, set dimension to an exact column name and metric may be null.
For scatter charts, set metric and second_metric to exact numeric column names.
Return only valid JSON matching the schema."""


def _semantic_issues(plan: SemanticDatasetPlan, df: pd.DataFrame) -> list[str]:
    columns = set(map(str, df.columns.tolist()))
    numeric_columns = {
        str(col) for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    }
    issues: list[str] = []

    semantic_names = {item.name for item in plan.column_semantics}
    missing_semantics = columns - semantic_names
    invalid_semantics = semantic_names - columns
    if missing_semantics:
        issues.append(f"column_semantics missing columns: {sorted(missing_semantics)}")
    if invalid_semantics:
        issues.append(f"column_semantics invented columns: {sorted(invalid_semantics)}")

    for idx, chart in enumerate(plan.recommended_charts):
        prefix = f"recommended_charts[{idx}]"
        if chart.dimension and chart.dimension not in columns:
            issues.append(f"{prefix}.dimension is not an existing column: {chart.dimension}")
        if chart.metric and chart.metric not in columns:
            issues.append(f"{prefix}.metric is not an existing column: {chart.metric}")
        if chart.second_metric and chart.second_metric not in columns:
            issues.append(f"{prefix}.second_metric is not an existing column: {chart.second_metric}")

        if chart.chart_type in {"bar", "line", "area", "pie"} and not chart.dimension:
            issues.append(f"{prefix} needs dimension")
        if chart.chart_type == "scatter":
            if not chart.metric or not chart.second_metric:
                issues.append(f"{prefix} scatter needs metric and second_metric")
            elif chart.metric not in numeric_columns or chart.second_metric not in numeric_columns:
                issues.append(f"{prefix} scatter metrics must be numeric columns")
        elif chart.aggregation != "count":
            if not chart.metric:
                issues.append(f"{prefix} aggregation {chart.aggregation} needs metric")
            elif chart.metric not in numeric_columns:
                issues.append(f"{prefix}.metric must be numeric for {chart.aggregation}: {chart.metric}")

    return issues


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _find_column(df: pd.DataFrame, keywords: list[str], numeric: bool | None = None) -> str | None:
    for col in df.columns:
        normalized = _norm(str(col))
        if not any(_norm(keyword) in normalized for keyword in keywords):
            continue
        if numeric is True and not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if numeric is False and pd.api.types.is_numeric_dtype(df[col]):
            continue
        return str(col)
    return None


def _detect_domain_from_columns(df: pd.DataFrame, current_domain: str | None) -> str:
    names = " ".join(_norm(col) for col in df.columns)
    scores = {
        "football_soccer": ["player", "team", "club", "position", "goal", "gol", "assist", "asist", "rating", "xg", "shot", "minute", "match"],
        "sales": ["sales", "revenue", "profit", "margin", "quantity", "order", "product", "region", "customer"],
        "hr": ["employee", "department", "salary", "attrition", "performance", "overtime", "tenure"],
        "logistics": ["delivery", "delay", "leadtime", "route", "vendor", "shipment", "stock", "inventory", "cost"],
        "finance": ["amount", "price", "cost", "balance", "transaction", "revenue", "profit"],
    }
    best_domain = current_domain or "general_business"
    best_score = 0
    for domain, keywords in scores.items():
        score = sum(1 for keyword in keywords if keyword in names)
        if score > best_score:
            best_domain = domain
            best_score = score
    return best_domain


def _role_for_column(df: pd.DataFrame, col: str, domain: str) -> tuple[str, str, int, str]:
    name = _norm(col)
    is_numeric = pd.api.types.is_numeric_dtype(df[col])

    if any(k in name for k in ["id", "uuid", "code", "key"]):
        return "identifier", "Identifier column", 1, "none"
    if any(k in name for k in ["date", "time", "tarih", "orderdate"]):
        return "time", "Time axis for trend analysis", 4, "none"

    if domain == "football_soccer":
        if any(k in name for k in ["player", "oyuncu"]):
            return "primary_entity", "Player name", 5, "none"
        if any(k in name for k in ["team", "club", "position"]):
            return "dimension", "Football grouping dimension", 4, "none"
        if is_numeric and any(k in name for k in ["goal", "gol", "assist", "asist", "rating", "xg"]):
            agg = "mean" if "rating" in name else "sum"
            return "primary_metric", "Key player performance metric", 5, agg
        if is_numeric and any(k in name for k in ["shot", "minute"]):
            return "secondary_metric", "Supporting player performance metric", 4, "sum"

    if domain == "sales":
        if any(k in name for k in ["product", "region", "customer"]):
            return "dimension", "Sales grouping dimension", 4, "none"
        if is_numeric and any(k in name for k in ["sales", "revenue", "profit", "quantity"]):
            return "primary_metric", "Key sales performance metric", 5, "sum"
        if is_numeric and "margin" in name:
            return "secondary_metric", "Profitability rate metric", 4, "mean"

    if domain == "hr":
        if any(k in name for k in ["employee", "department", "role"]):
            return "dimension", "HR grouping dimension", 4, "none"
        if is_numeric and any(k in name for k in ["salary", "performance", "overtime", "tenure"]):
            return "primary_metric", "Key workforce metric", 4, "mean"

    if domain == "logistics":
        if any(k in name for k in ["route", "vendor", "shipment"]):
            return "dimension", "Logistics grouping dimension", 4, "none"
        if is_numeric and any(k in name for k in ["delivery", "delay", "leadtime", "stock", "quantity", "cost"]):
            return "primary_metric", "Key logistics metric", 4, "sum"

    if is_numeric:
        return "secondary_metric", "Numeric analytical measure", 3, "mean"
    return "dimension", "Categorical grouping dimension", 3, "none"


def _make_chart(
    chart_type: str,
    title: str,
    metric: str | None,
    dimension: str | None,
    aggregation: str,
    reason: str,
    priority: int,
    second_metric: str | None = None,
    sort: str = "desc",
    limit: int = 10,
) -> ChartPlan:
    return ChartPlan(
        chart_type=chart_type,
        title=title,
        metric=metric,
        second_metric=second_metric,
        dimension=dimension,
        aggregation=aggregation,
        sort=sort,
        limit=limit,
        reason=reason,
        priority=priority,
    )


def _local_recommendations(df: pd.DataFrame, domain: str) -> list[ChartPlan]:
    charts: list[ChartPlan] = []
    player = _find_column(df, ["player", "oyuncu"], numeric=False)
    team = _find_column(df, ["team", "club"], numeric=False)
    goals = _find_column(df, ["goals", "goal", "gol"], numeric=True)
    assists = _find_column(df, ["assists", "assist", "asist"], numeric=True)
    rating = _find_column(df, ["rating"], numeric=True)
    sales = _find_column(df, ["sales", "revenue"], numeric=True)
    profit = _find_column(df, ["profit"], numeric=True)
    product = _find_column(df, ["product"], numeric=False)
    region = _find_column(df, ["region"], numeric=False)
    date = _find_column(df, ["date", "tarih"], numeric=None)

    if domain == "football_soccer":
        if player and goals:
            charts.append(_make_chart("bar", f"Top Goals by {player}", goals, player, "sum", "Goals are a primary football performance metric.", 5))
        if player and assists:
            charts.append(_make_chart("bar", f"Top Assists by {player}", assists, player, "sum", "Assists are a primary football performance metric.", 5))
        if goals and assists:
            charts.append(_make_chart("scatter", f"{goals} vs {assists}", goals, None, "mean", "Shows the relationship between scoring and chance creation.", 5, second_metric=assists, sort="none"))
        if team and goals:
            charts.append(_make_chart("bar", f"Total {goals} by {team}", goals, team, "sum", "Team-level scoring distribution.", 4))
        if player and rating:
            charts.append(_make_chart("bar", f"Average {rating} by {player}", rating, player, "mean", "Player rating highlights overall performance.", 4))

    if domain == "sales":
        if region and sales:
            charts.append(_make_chart("bar", f"Total {sales} by {region}", sales, region, "sum", "Regional sales performance.", 5))
        if product and sales:
            charts.append(_make_chart("bar", f"Total {sales} by {product}", sales, product, "sum", "Product sales performance.", 5))
        if profit and sales:
            charts.append(_make_chart("scatter", f"{sales} vs {profit}", sales, None, "mean", "Compares revenue and profitability.", 4, second_metric=profit, sort="none"))
        if date and sales:
            charts.append(_make_chart("line", f"{sales} Trend", sales, date, "sum", "Sales trend over time.", 4, sort="asc", limit=30))
        if region and profit:
            charts.append(_make_chart("bar", f"Total {profit} by {region}", profit, region, "sum", "Regional profit performance.", 4))

    if not charts:
        numeric_cols = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [str(c) for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if categorical_cols and numeric_cols:
            charts.append(_make_chart("bar", f"{numeric_cols[0]} by {categorical_cols[0]}", numeric_cols[0], categorical_cols[0], "mean", "General numeric comparison by category.", 3))
        if len(numeric_cols) >= 2:
            charts.append(_make_chart("scatter", f"{numeric_cols[0]} vs {numeric_cols[1]}", numeric_cols[0], None, "mean", "General relationship between numeric columns.", 3, second_metric=numeric_cols[1], sort="none"))

    return charts[:8]


def _enhance_plan_locally(plan: SemanticDatasetPlan, df: pd.DataFrame) -> SemanticDatasetPlan:
    domain = _detect_domain_from_columns(df, plan.detected_domain)
    existing_semantics = {item.name: item for item in plan.column_semantics if item.name in df.columns}
    local_semantics: list[ColumnSemantic] = []

    for col in map(str, df.columns.tolist()):
        role, meaning, importance, aggregation = _role_for_column(df, col, domain)
        current = existing_semantics.get(col)
        if current and current.role not in {"noise", "dimension"}:
            local_semantics.append(current)
        else:
            local_semantics.append(
                ColumnSemantic(
                    name=col,
                    role=role,
                    business_meaning=meaning,
                    importance=importance,
                    preferred_aggregation=aggregation,
                )
            )

    primary_entity = plan.primary_entity
    if not primary_entity or primary_entity not in df.columns:
        primary_entity = next((item.name for item in local_semantics if item.role == "primary_entity"), None)

    local_charts = _local_recommendations(df, domain)
    existing = {
        (chart.chart_type, chart.metric, chart.second_metric, chart.dimension, chart.aggregation)
        for chart in local_charts
    }
    merged_charts = list(local_charts)
    for chart in plan.recommended_charts:
        key = (chart.chart_type, chart.metric, chart.second_metric, chart.dimension, chart.aggregation)
        if key not in existing:
            merged_charts.append(chart)
            existing.add(key)

    return SemanticDatasetPlan(
        detected_domain=domain,
        primary_entity=primary_entity,
        column_semantics=local_semantics,
        recommended_charts=merged_charts[:8],
    )


def _needs_local_enhancement(plan: SemanticDatasetPlan, df: pd.DataFrame) -> bool:
    if not plan.column_semantics:
        return True
    if _semantic_issues(plan, df):
        return True
    important_roles = {"primary_metric", "primary_entity"}
    if not any(item.role in important_roles and item.importance >= 4 for item in plan.column_semantics):
        return True
    return False


async def generate_semantic_dataset_plan(df: pd.DataFrame) -> SemanticDatasetPlan:
    max_cols = _env_int("AI_MAX_COLUMN_CARDS", 60)
    max_charts = _env_int("AI_MAX_RECOMMENDED_CHARTS", 8)
    cards = build_column_cards(df, max_cols=max_cols)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Dataset has {len(df)} rows and {len(df.columns)} columns. "
                f"Recommend 4 to {max_charts} high-value charts. Column cards:\n"
                f"{json.dumps(cards, ensure_ascii=False)}"
            ),
        },
    ]
    plan = await call_ollama_structured(messages, SemanticDatasetPlan)
    issues = _semantic_issues(plan, df)
    if not issues and not _needs_local_enhancement(plan, df):
        return plan

    repair_messages = messages + [
        {
            "role": "assistant",
            "content": json.dumps(model_dump_compat(plan), ensure_ascii=False),
        },
        {
            "role": "user",
            "content": (
                "The JSON was syntactically valid but semantically invalid. "
                "Fix these issues and return the full JSON plan again. "
                f"Existing columns are: {list(map(str, df.columns.tolist()))}. "
                f"Issues: {issues}"
            ),
        },
    ]
    try:
        repaired = await call_ollama_structured(repair_messages, SemanticDatasetPlan)
    except Exception:
        return _enhance_plan_locally(plan, df)
    if _needs_local_enhancement(repaired, df):
        return _enhance_plan_locally(repaired, df)
    return repaired
