from __future__ import annotations

import asyncio
import json
import os

import pandas as pd

from .column_cards import build_column_cards
from .groq_client import call_groq_structured
from .schemas import FastSemanticPlan
from .semantic_validation import validate_fast_semantic_plan


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


FAST_SYSTEM_PROMPT = """\
You are a semantic and analysis-intent planner for an analytics dashboard.
Use ONLY existing column names exactly as provided; do not invent new columns.
Do not calculate any values, percentages, rankings, aggregations, or chartData.
Prefer domain-important columns over random numeric columns.
You may recommend what calculation should be performed, but Pandas backend code performs the calculation.

Domain priorities:
- football/soccer: Player/Team/Position + Goals/Assists/Rating/xG/Shots/Minutes
- sales: Product/Region/Customer/Date + Sales/Revenue/Profit/Quantity
- HR: Department/Employee + Attrition/Salary/Performance/Overtime/Tenure
- logistics: Vendor/Route/Date + Delay/LeadTime/Stock/Quantity/Cost

Populate backward-compatible fields and also provide:
- metrics: semantic plans for important columns with role, semantic_type, aggregation, direction, include_as_kpi, include_in_clustering, confidence.
- recommended_analyses: intended analyses such as kpi, bar, pie, scatter, line, clustering.

Allowed values:
- role: outcome_metric, driver_metric, supporting_metric, identifier, dimension
- semantic_type: score, percentage, rate, duration, count, money, quantity, index, raw_numeric
- aggregation: mean, sum, count, min, max
- direction: higher_is_better, lower_is_better, context_dependent, neutral
- metric confidence: low, medium, high

Return ONLY valid JSON matching the FastSemanticPlan schema.\
"""


async def generate_fast_semantic_plan(df: pd.DataFrame) -> FastSemanticPlan:
    """Call Groq to identify semantic roles and intended analysis plans.

    The LLM only plans semantics and calculation intent; Pandas performs all calculations.
    """
    max_cols = _env_int("AI_MAX_COLUMN_CARDS", 25)
    timeout = _env_int("AI_SEMANTIC_TIMEOUT_SECONDS", 8)

    cards = build_column_cards(df, max_cols=max_cols)
    col_names = list(map(str, df.columns.tolist()))

    messages = [
        {"role": "system", "content": FAST_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "available_columns": col_names,
                    "row_count": len(df),
                    "column_cards": cards,
                    "task": (
                        "Identify detected_domain, primary_entity, primary_metrics, "
                        "secondary_metrics, dimensions, time_columns, ignored_columns, confidence, "
                        "metrics, and recommended_analyses. "
                        "For each metric, recommend aggregation/direction but do not calculate values. "
                        "Use only names from available_columns."
                    ),
                },
                ensure_ascii=False,
            ),
        },
    ]

    plan = await call_groq_structured(
        messages,
        FastSemanticPlan,
        temperature=0,
        timeout_seconds=timeout,
    )

    return validate_fast_semantic_plan(plan, df)


def _validate_plan_columns(plan: FastSemanticPlan, df: pd.DataFrame) -> FastSemanticPlan:
    """Backward-compatible wrapper for older direct imports."""
    return validate_fast_semantic_plan(plan, df)
