from __future__ import annotations

import json
import os
import re
from typing import Optional

import pandas as pd

from .column_cards import build_column_cards
from .ollama_client import call_ollama_structured
from .schemas import ChartPlan, PromptChartIntent, SemanticDatasetPlan, model_dump_compat


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


SYSTEM_PROMPT = """You convert a user's chart request into one structured chart plan.
Use only existing column names exactly as provided.
Prefer columns classified as primary_metric, primary_entity, dimension, and time.
If the user says gol/asist/goals/assists, prefer goal and assist columns plus player/team columns when available.
The LLM must not calculate metrics or chart data. It only selects columns, chart type, aggregation, sort, and limit.
For bar, line, area, and pie charts, set dimension to an exact column name.
For sum, mean, min, and max aggregations, set metric to an exact numeric column name.
For count aggregation, set dimension to an exact column name and metric may be null.
For scatter charts, set metric and second_metric to exact numeric column names.
Return only valid JSON matching the schema."""


def _plan_issues(plan: ChartPlan, df: pd.DataFrame) -> list[str]:
    columns = set(map(str, df.columns.tolist()))
    numeric_columns = {
        str(col) for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    }
    issues: list[str] = []

    if plan.dimension and plan.dimension not in columns:
        issues.append(f"dimension is not an existing column: {plan.dimension}")
    if plan.metric and plan.metric not in columns:
        issues.append(f"metric is not an existing column: {plan.metric}")
    if plan.second_metric and plan.second_metric not in columns:
        issues.append(f"second_metric is not an existing column: {plan.second_metric}")

    if plan.chart_type in {"bar", "line", "area", "pie"} and not plan.dimension:
        issues.append("dimension is required for grouped charts")
    if plan.chart_type == "scatter":
        if not plan.metric or not plan.second_metric:
            issues.append("scatter needs metric and second_metric")
        elif plan.metric not in numeric_columns or plan.second_metric not in numeric_columns:
            issues.append("scatter metric and second_metric must be numeric columns")
    elif plan.aggregation != "count":
        if not plan.metric:
            issues.append(f"metric is required for {plan.aggregation} aggregation")
        elif plan.metric not in numeric_columns:
            issues.append(f"metric must be numeric for {plan.aggregation}: {plan.metric}")

    return issues


def _to_chart_plan(intent: PromptChartIntent) -> ChartPlan:
    intent_data = model_dump_compat(intent)
    chart_fields = getattr(ChartPlan, "model_fields", getattr(ChartPlan, "__fields__", {}))
    return ChartPlan(**{k: v for k, v in intent_data.items() if k in chart_fields})


def _same_plan_shape(left: ChartPlan, right: ChartPlan) -> bool:
    return (
        left.chart_type == right.chart_type
        and left.metric == right.metric
        and left.second_metric == right.second_metric
        and left.dimension == right.dimension
        and left.aggregation == right.aggregation
    )


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9ığüşöçİĞÜŞÖÇ]+", "", str(value).lower())


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


def _local_prompt_plan(df: pd.DataFrame, prompt: str) -> ChartPlan:
    text = _norm(prompt)
    player = _find_column(df, ["player", "oyuncu"], numeric=False)
    team = _find_column(df, ["team", "takım", "takim", "club"], numeric=False)
    goals = _find_column(df, ["goals", "goal", "gol"], numeric=True)
    assists = _find_column(df, ["assists", "assist", "asist"], numeric=True)
    sales = _find_column(df, ["sales", "revenue", "satis", "satış"], numeric=True)
    profit = _find_column(df, ["profit", "kar"], numeric=True)
    product = _find_column(df, ["product", "ürün", "urun"], numeric=False)
    region = _find_column(df, ["region", "bolge", "bölge"], numeric=False)

    wants_relation = any(k in text for k in ["iliski", "ilişki", "korelasyon", "correlation", "scatter", "vs", "arasındaki", "arasindaki", "aras", "ciz", "çiz"])
    wants_team = any(k in text for k in ["team", "takim", "takım", "tak"])
    wants_player = any(k in text for k in ["player", "oyuncu"])
    wants_assist = any(k in text for k in ["assist", "asist"])
    wants_goal = any(k in text for k in ["goal", "goals", "gol"])
    wants_sales = any(k in text for k in ["sales", "satis", "satış", "revenue"])
    wants_profit = "profit" in text or "kar" in text

    if wants_relation:
        first = goals if wants_goal else sales
        second = assists if wants_assist or first == goals else profit
        if first and second:
            return ChartPlan(
                chart_type="scatter",
                title=f"{first} vs {second}",
                metric=first,
                second_metric=second,
                aggregation="mean",
                sort="none",
                limit=30,
                reason="Prompt asks for a relationship between two metrics.",
                priority=5,
            )

    if wants_goal and goals:
        dimension = team if wants_team else player if (wants_player or player) else team
        second_metric = assists if wants_assist else None
        if dimension:
            return ChartPlan(
                chart_type="bar",
                title=f"{goals}" + (f" and {assists}" if second_metric else "") + f" by {dimension}",
                metric=goals,
                second_metric=second_metric,
                dimension=dimension,
                aggregation="sum",
                sort="desc",
                limit=10,
                reason="Prompt asks for football scoring metrics by entity.",
                priority=5,
            )

    if wants_sales and sales:
        dimension = region or product
        if dimension:
            return ChartPlan(
                chart_type="bar",
                title=f"{sales} by {dimension}",
                metric=sales,
                second_metric=profit if wants_profit and profit else None,
                dimension=dimension,
                aggregation="sum",
                sort="desc",
                limit=10,
                reason="Prompt asks for sales performance by a business dimension.",
                priority=5,
            )

    numeric_cols = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [str(c) for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if cat_cols and numeric_cols:
        return ChartPlan(
            chart_type="bar",
            title=f"{numeric_cols[0]} by {cat_cols[0]}",
            metric=numeric_cols[0],
            dimension=cat_cols[0],
            aggregation="mean",
            sort="desc",
            limit=10,
            reason="Fallback prompt plan using the first usable metric and dimension.",
            priority=3,
        )
    raise ValueError("No usable columns found for prompt chart planning.")


async def generate_prompt_chart_plan(
    df: pd.DataFrame,
    prompt: str,
    semantic_plan: Optional[SemanticDatasetPlan],
) -> ChartPlan:
    cards = build_column_cards(df, max_cols=_env_int("AI_MAX_COLUMN_CARDS", 60))
    semantic_payload = model_dump_compat(semantic_plan) if semantic_plan else None
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "user_prompt": prompt,
                    "columns": list(map(str, df.columns.tolist())),
                    "column_cards": cards,
                    "semantic_plan": semantic_payload,
                },
                ensure_ascii=False,
            ),
        },
    ]
    intent = await call_ollama_structured(messages, PromptChartIntent)
    plan = _to_chart_plan(intent)
    issues = _plan_issues(plan, df)
    local_plan: ChartPlan | None = None
    try:
        local_plan = _local_prompt_plan(df, prompt)
    except Exception:
        local_plan = None
    if not issues:
        if local_plan and local_plan.priority >= 5 and not _same_plan_shape(plan, local_plan):
            return local_plan
        return plan

    repair_messages = messages + [
        {
            "role": "assistant",
            "content": json.dumps(model_dump_compat(intent), ensure_ascii=False),
        },
        {
            "role": "user",
            "content": (
                "The JSON was syntactically valid but semantically invalid. "
                "Fix these issues and return one complete JSON intent again. "
                f"Existing columns are: {list(map(str, df.columns.tolist()))}. "
                f"Issues: {issues}"
            ),
        },
    ]
    repaired_intent = await call_ollama_structured(repair_messages, PromptChartIntent)
    repaired_plan = _to_chart_plan(repaired_intent)
    if _plan_issues(repaired_plan, df):
        if local_plan:
            return local_plan
        return _local_prompt_plan(df, prompt)
    if local_plan and local_plan.priority >= 5 and not _same_plan_shape(repaired_plan, local_plan):
        return local_plan
    return repaired_plan
