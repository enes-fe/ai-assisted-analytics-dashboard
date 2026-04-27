from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


SEMANTIC_KEYWORDS: dict[str, list[str]] = {
    "football_soccer": [
        "player", "team", "club", "position", "goals", "goal", "gol",
        "assists", "assist", "asist", "rating", "xg", "shots", "minutes",
        "minute", "match",
    ],
    "sales": [
        "sales", "revenue", "profit", "margin", "quantity", "order",
        "product", "region", "customer",
    ],
    "hr": [
        "employee", "department", "salary", "attrition", "performance",
        "overtime", "tenure",
    ],
    "logistics": [
        "delivery", "delay", "lead_time", "lead time", "route", "vendor",
        "shipment", "stock", "inventory", "cost",
    ],
    "finance": [
        "amount", "price", "cost", "balance", "transaction", "revenue",
        "profit",
    ],
}


def _clean_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value) if isinstance(value, float) else int(value)
    return str(value)[:80]


def _hints_for_column(name: str) -> list[str]:
    normalized = re.sub(r"[_\-]+", " ", str(name).lower())
    hints: list[str] = []
    for domain, keywords in SEMANTIC_KEYWORDS.items():
        matches = [kw for kw in keywords if kw in normalized]
        if matches:
            hints.append(f"{domain}: {', '.join(matches[:4])}")
    return hints


def build_column_cards(df: pd.DataFrame, max_cols: int = 60) -> list[dict]:
    """Build compact column summaries for the LLM without exposing full rows."""
    cards: list[dict] = []
    if df.empty:
        return cards

    for col in list(df.columns)[:max_cols]:
        series = df[col]
        non_null = series.dropna()
        card: dict[str, Any] = {
            "name": str(col),
            "dtype": str(series.dtype),
            "missing_ratio": round(float(series.isna().mean()), 4),
            "unique_count": int(series.nunique(dropna=True)),
            "sample_values": [_clean_scalar(v) for v in non_null.head(5).tolist()],
            "possible_semantic_hints": _hints_for_column(str(col)),
        }

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric.empty:
                card["numeric_stats"] = {
                    "min": _clean_scalar(numeric.min()),
                    "max": _clean_scalar(numeric.max()),
                    "mean": _clean_scalar(numeric.mean()),
                    "median": _clean_scalar(numeric.median()),
                }
        else:
            counts = non_null.astype(str).value_counts().head(5)
            card["top_values"] = [
                {"value": str(value)[:80], "count": int(count)}
                for value, count in counts.items()
            ]

        cards.append(card)

    return cards

