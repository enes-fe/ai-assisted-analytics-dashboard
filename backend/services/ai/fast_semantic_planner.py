from __future__ import annotations

import asyncio
import json
import os

import pandas as pd

from .column_cards import build_column_cards
from .groq_client import call_groq_structured
from .schemas import FastSemanticPlan


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


FAST_SYSTEM_PROMPT = """\
You are a semantic column selector for an analytics dashboard.
Use ONLY existing column names exactly as provided — do not invent new columns.
Do not calculate any values or metrics.
Prefer domain-important columns over random numeric columns.

Domain priorities:
- football/soccer: Player/Team/Position + Goals/Assists/Rating/xG/Shots/Minutes
- sales: Product/Region/Customer/Date + Sales/Revenue/Profit/Quantity
- HR: Department/Employee + Attrition/Salary/Performance/Overtime/Tenure
- logistics: Vendor/Route/Date + Delay/LeadTime/Stock/Quantity/Cost

Return ONLY valid JSON matching the FastSemanticPlan schema.\
"""


async def generate_fast_semantic_plan(df: pd.DataFrame) -> FastSemanticPlan:
    """Call Groq to identify semantic roles for dataset columns.

    The LLM only selects column names — Pandas performs all calculations.
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
                        "secondary_metrics, dimensions, time_columns, ignored_columns, confidence. "
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

    # Validate all column names against the actual DataFrame
    plan = _validate_plan_columns(plan, df)
    return plan


def _validate_plan_columns(plan: FastSemanticPlan, df: pd.DataFrame) -> FastSemanticPlan:
    """Remove any LLM-invented column names that don't exist in the DataFrame."""
    valid = set(map(str, df.columns.tolist()))

    def _filter(cols: list[str]) -> list[str]:
        return [c for c in cols if c in valid]

    return FastSemanticPlan(
        detected_domain=plan.detected_domain,
        primary_entity=plan.primary_entity if plan.primary_entity in valid else None,
        primary_metrics=_filter(plan.primary_metrics),
        secondary_metrics=_filter(plan.secondary_metrics),
        dimensions=_filter(plan.dimensions),
        time_columns=_filter(plan.time_columns),
        ignored_columns=_filter(plan.ignored_columns),
        confidence=plan.confidence,
    )
