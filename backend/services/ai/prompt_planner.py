from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Optional

import pandas as pd

from services.utils import select_label_column

from .column_cards import build_column_cards
from .groq_client import call_groq_structured
from .schemas import ChartPlan, FastSemanticPlan, PromptChartIntent, model_dump_compat


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


SYSTEM_PROMPT = """\
You convert a user's chart request into one structured chart plan.
Use ONLY existing column names exactly as provided.
Prefer columns classified as primary_metric, primary_entity, dimension, and time.
If the user says expected goals/xG, prefer the exact expected-goals/xG column over columns that merely contain goal.
If the user says gol/asist/goals/assists, prefer goal and assist columns plus player/team columns when available.
The LLM must NOT calculate metrics or chart data; it only selects columns, chart type, aggregation, sort, and limit.
For bar, line, area, pie, and donut charts, set dimension to an exact column name.
For sum, mean, min, and max aggregations, set metric to an exact numeric column name.
For count aggregation, set dimension to an exact column name and metric may be null.
For scatter charts, set metric and second_metric to exact numeric column names; dimension may be a label column.
Return ONLY valid JSON matching the schema.\
"""


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

    if plan.chart_type in {"bar", "line", "area", "pie", "donut"} and not plan.dimension:
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


PROMPT_CHART_TYPE_ALIASES = {
    "pie": ["pie", "pie chart", "pasta", "pasta grafik"],
    "donut": ["donut", "doughnut", "halka", "halka grafik"],
    "bar": ["bar", "bar chart", "column chart", "cubuk", "çubuk", "sutun", "sütun"],
    "line": ["line", "line chart", "trend", "cizgi", "çizgi"],
    "scatter": ["scatter", "scatter plot", "relationship", "correlation", "dagilim", "dağılım", "iliski", "ilişki"],
    "area": ["area", "area chart", "alan"],
    "heatmap": ["heatmap", "heat map", "isi haritasi", "ısı haritası"],
    "treemap": ["treemap", "tree map", "agac haritasi", "ağaç haritası"],
}

SUPPORTED_OVERRIDE_TYPES = {"pie", "donut", "bar", "line", "scatter", "area"}
PIE_AVERAGE_FALLBACK_REASON = (
    "Pie/donut charts are intended for part-to-whole values. "
    "A bar chart is used for comparing averages."
)


def extract_prompt_chart_type(prompt: str) -> str | None:
    for chart_type, aliases in PROMPT_CHART_TYPE_ALIASES.items():
        if _prompt_has(prompt, aliases):
            return chart_type
    return None


def _replace_plan(plan: ChartPlan, **updates: object) -> ChartPlan:
    data = model_dump_compat(plan)
    data.update(updates)
    return ChartPlan(**data)


def _with_fallback_reason(plan: ChartPlan, reason: str) -> ChartPlan:
    return _replace_plan(plan, reason=f"{reason} Fallback: {plan.reason}")


def _unsupported_renderer_reason(chart_type: str) -> str:
    label = chart_type.strip().capitalize()
    return f"{label} is not currently supported by the renderer. A bar chart is shown as a compatible fallback."


def _bar_fallback_plan(
    plan: ChartPlan,
    reason: str,
    dimension: str | None = None,
    metric: str | None = None,
    aggregation: str | None = None,
) -> ChartPlan:
    fallback = _replace_plan(
        plan,
        chart_type="bar",
        dimension=dimension or plan.dimension,
        metric=metric if metric is not None else plan.metric,
        aggregation=aggregation or plan.aggregation,
        sort="desc",
        limit=min(plan.limit or 5, 8),
    )
    return _with_fallback_reason(fallback, reason)


def _compatible_dimension(df: pd.DataFrame, dimension: str | None) -> bool:
    if not dimension or dimension not in df.columns:
        return False
    if pd.api.types.is_numeric_dtype(df[dimension]):
        return False
    return df[dimension].dropna().nunique() >= 1


def _apply_chart_type_override(df: pd.DataFrame, prompt: str, plan: ChartPlan) -> ChartPlan:
    requested = extract_prompt_chart_type(prompt)
    if not requested:
        return plan

    if requested not in SUPPORTED_OVERRIDE_TYPES:
        return _bar_fallback_plan(plan, _unsupported_renderer_reason(requested))

    dimension = plan.dimension or _dimension_from_prompt(df, prompt)
    metric = plan.metric
    aggregation = plan.aggregation

    if requested in {"pie", "donut"}:
        if not _compatible_dimension(df, dimension):
            return _bar_fallback_plan(
                plan,
                f"Requested {requested} chart needs one categorical dimension.",
                dimension=dimension,
                metric=metric,
                aggregation=aggregation,
            )
        if aggregation != "count" and not (metric and metric in df.columns and pd.api.types.is_numeric_dtype(df[metric])):
            return _bar_fallback_plan(
                plan,
                f"Requested {requested} chart needs one numeric measure.",
                dimension=dimension,
                metric=metric,
                aggregation=aggregation,
            )
        category_count = int(df[dimension].dropna().nunique()) if dimension else 0
        if category_count > 8:
            return _bar_fallback_plan(
                plan,
                (
                    "Pie/donut charts are intended for low-cardinality part-to-whole values. "
                    f"A bar chart is used because {dimension} has {category_count} categories."
                ),
                dimension=dimension,
                metric=metric,
                aggregation=aggregation,
            )
        if aggregation not in {"sum", "count"}:
            return _bar_fallback_plan(
                plan,
                PIE_AVERAGE_FALLBACK_REASON,
                dimension=dimension,
                metric=metric,
                aggregation=aggregation,
            )
        note = f"User requested a {requested} chart and the columns are compatible."
        return _replace_plan(
            plan,
            chart_type=requested,
            dimension=dimension,
            metric=metric,
            aggregation=aggregation,
            sort="desc",
            limit=8,
            reason=note,
        )

    if requested == "scatter":
        first = metric if metric and metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]) else _first_numeric_not(df, set())
        second = plan.second_metric
        if not second or second == first or second not in df.columns or not pd.api.types.is_numeric_dtype(df[second]):
            second = _first_numeric_not(df, {first} if first else set())
        if not first or not second:
            return _with_fallback_reason(
                plan,
                "Requested scatter chart needs two numeric measures.",
            )
        return _replace_plan(
            plan,
            chart_type="scatter",
            metric=first,
            second_metric=second,
            dimension=dimension,
            aggregation="mean",
            sort="none",
            limit=30,
            reason="User requested a scatter chart and two numeric measures were found.",
        )

    if requested in {"bar", "line", "area"}:
        if not dimension:
            return _with_fallback_reason(
                plan,
                f"Requested {requested} chart needs a grouping or time dimension.",
            )
        if aggregation != "count" and not (metric and metric in df.columns and pd.api.types.is_numeric_dtype(df[metric])):
            return _with_fallback_reason(
                plan,
                f"Requested {requested} chart needs a numeric measure.",
            )
        return _replace_plan(
            plan,
            chart_type=requested,
            dimension=dimension,
            metric=metric,
            aggregation=aggregation,
            sort="asc" if requested in {"line", "area"} else "desc",
            limit=30 if requested in {"line", "area"} else min(plan.limit or 5, 8),
            reason=f"User explicitly requested a {requested} chart and the columns are compatible.",
        )

    return plan


def _ascii(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    return "".join(ch for ch in value if not unicodedata.combining(ch))


def _words(value: str) -> list[str]:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value))
    value = _ascii(value).lower()
    return [part for part in re.split(r"[^a-z0-9]+", value) if part]


def _compact(value: str) -> str:
    return "".join(_words(value))


def _prompt_has(prompt: str, aliases: list[str]) -> bool:
    prompt_words = set(_words(prompt))
    prompt_compact = _compact(prompt)
    for alias in aliases:
        alias_words = _words(alias)
        alias_compact = "".join(alias_words)
        if alias_compact and alias_compact in prompt_compact:
            return True
        if alias_words and set(alias_words).issubset(prompt_words):
            return True
    return False


def _find_best_column(
    df: pd.DataFrame,
    prompt: str,
    aliases: list[str],
    numeric: bool | None = None,
) -> str | None:
    prompt_words = set(_words(prompt))
    asks_expected = "expected" in prompt_words or "xg" in prompt_words
    scored: list[tuple[int, str]] = []

    for col in df.columns:
        col_name = str(col)
        if numeric is True and not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if numeric is False and pd.api.types.is_numeric_dtype(df[col]):
            continue

        col_words = set(_words(col_name))
        col_compact = _compact(col_name)
        score = len(prompt_words & col_words) * 8

        for alias in aliases:
            alias_words = set(_words(alias))
            alias_compact = "".join(_words(alias))
            if not alias_compact:
                continue
            if alias_compact == col_compact:
                score += 130
            elif alias_compact in col_compact:
                score += 90 if len(alias_words) > 1 else 45
            if alias_words and alias_words.issubset(col_words):
                score += 110

        if asks_expected and not ({"expected", "xg", "exp"} & col_words or "xg" in col_compact):
            score -= 80
        if "error" in col_words and "error" not in prompt_words:
            score -= 35
        if "lead" in col_words and "lead" not in prompt_words:
            score -= 20
        if "id" in col_words or col_compact.endswith("id"):
            score -= 25

        if score > 0:
            scored.append((score, col_name))

    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _first_numeric_not(df: pd.DataFrame, excluded: set[str]) -> str | None:
    for col in df.columns:
        col_name = str(col)
        if col_name in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].dropna().nunique() > 2:
            return col_name
    return None


def _metric_from_prompt(df: pd.DataFrame, prompt: str) -> tuple[str | None, str]:
    metric_groups = [
        (["expected goals", "expected goal", "xg", "exp goals"], "mean"),
        (["assists", "assist", "asist"], "sum"),
        (["goals", "goal", "gol"], "sum"),
        (["rating", "score", "puan"], "mean"),
        (["sales", "revenue", "satis", "satış"], "sum"),
        (["profit", "kar"], "sum"),
        (["quantity", "qty", "adet"], "sum"),
        (["price", "amount", "cost", "tutar", "maliyet"], "mean"),
    ]
    for aliases, aggregation in metric_groups:
        if _prompt_has(prompt, aliases):
            match = _find_best_column(df, prompt, aliases, numeric=True)
            if match:
                return match, aggregation
    return None, "sum"


def _dimension_from_prompt(df: pd.DataFrame, prompt: str) -> str | None:
    dimension_groups = [
        ["player name", "player", "oyuncu"],
        ["team", "club", "takim", "takım"],
        ["product name", "product", "urun", "ürün"],
        ["customer name", "customer", "musteri", "müşteri"],
        ["region", "bolge", "bölge"],
        ["category", "kategori"],
        ["department", "departman"],
        ["name", "isim", "ad"],
    ]
    for aliases in dimension_groups:
        if _prompt_has(prompt, aliases):
            match = _find_best_column(df, prompt, aliases, numeric=False)
            if match:
                return match
    return select_label_column(df)


def _local_prompt_plan(df: pd.DataFrame, prompt: str) -> ChartPlan:
    text = _compact(prompt)
    prompt_words = set(_words(prompt))
    metric, aggregation = _metric_from_prompt(df, prompt)
    dimension = _dimension_from_prompt(df, prompt)

    wants_relation = (
        _prompt_has(prompt, ["correlation", "korelasyon", "relationship", "iliski", "ilişki", "scatter"])
        or "vs" in prompt_words
        or "versus" in prompt_words
        or "arasindaki" in text
    )

    if wants_relation:
        first = metric
        second = None
        if _prompt_has(prompt, ["assist", "assists", "asist"]):
            second = _find_best_column(df, prompt, ["assists", "assist", "asist"], numeric=True)
        elif _prompt_has(prompt, ["profit", "kar"]):
            second = _find_best_column(df, prompt, ["profit", "kar"], numeric=True)
        elif _prompt_has(prompt, ["goals", "goal", "gol"]):
            second = _find_best_column(df, prompt, ["goals", "goal", "gol"], numeric=True)
        if first and (not second or second == first):
            second = _first_numeric_not(df, {first})
        if first and second and first != second:
            return ChartPlan(
                chart_type="scatter",
                title=f"{first} vs {second}",
                metric=first,
                second_metric=second,
                dimension=dimension,
                aggregation="mean",
                sort="none",
                limit=30,
                reason="Prompt asks for a relationship between two metrics.",
                priority=5,
            )

    if metric and dimension:
        second_metric = None
        if _prompt_has(prompt, ["assist", "assists", "asist"]):
            assist_col = _find_best_column(df, prompt, ["assists", "assist", "asist"], numeric=True)
            if assist_col and assist_col != metric:
                second_metric = assist_col
        return ChartPlan(
            chart_type="bar",
            title=f"{metric}" + (f" and {second_metric}" if second_metric else "") + f" by {dimension}",
            metric=metric,
            second_metric=second_metric,
            dimension=dimension,
            aggregation=aggregation,
            sort="desc",
            limit=5,
            reason="Prompt metric and grouping were matched against existing dataset columns.",
            priority=5,
        )

    numeric_cols = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    label_col = select_label_column(df)
    if label_col and numeric_cols:
        return ChartPlan(
            chart_type="bar",
            title=f"{numeric_cols[0]} by {label_col}",
            metric=numeric_cols[0],
            dimension=label_col,
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
    semantic_plan: Optional[FastSemanticPlan | object],
) -> ChartPlan:
    """Use Groq to convert a natural language prompt into a chart plan.

    Falls back to local keyword-based planner if Groq fails or returns invalid columns.
    """
    cards = build_column_cards(df, max_cols=_env_int("AI_MAX_COLUMN_CARDS", 25))
    semantic_payload: dict | None = None
    if semantic_plan is not None:
        try:
            if hasattr(semantic_plan, "model_dump"):
                semantic_payload = semantic_plan.model_dump()
            elif hasattr(semantic_plan, "dict"):
                semantic_payload = semantic_plan.dict()  # type: ignore[attr-defined]
        except Exception:
            pass

    timeout = _env_int("AI_CHAT_TIMEOUT_SECONDS", 8)
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

    try:
        intent = await call_groq_structured(
            messages,
            PromptChartIntent,
            timeout_seconds=timeout,
        )
        plan = _to_chart_plan(intent)
        issues = _plan_issues(plan, df)
        local_plan: ChartPlan | None = None
        try:
            local_plan = _local_prompt_plan(df, prompt)
        except Exception:
            local_plan = None

        if not issues:
            if local_plan and local_plan.priority >= 5 and not _same_plan_shape(plan, local_plan):
                return _apply_chart_type_override(df, prompt, local_plan)
            return _apply_chart_type_override(df, prompt, plan)

        if local_plan:
            return _apply_chart_type_override(df, prompt, local_plan)
        return _apply_chart_type_override(df, prompt, _local_prompt_plan(df, prompt))

    except Exception:
        # Groq unavailable or failed; use local planner.
        return _apply_chart_type_override(df, prompt, _local_prompt_plan(df, prompt))
