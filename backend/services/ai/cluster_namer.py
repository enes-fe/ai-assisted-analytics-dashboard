from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from .groq_client import call_groq_structured
from .schemas import ClusterNameSuggestions, model_dump_compat


GENERIC_NAME_PATTERNS = [
    "segment",
    "balanced",
    "high ",
    "low ",
    "pattern",
    "metric",
]

SYSTEM_PROMPT = """You name exploratory data clusters for an analytics dashboard.

Hard rules:
- Do NOT calculate metrics, counts, percentages, rankings, or chart data.
- Use only the provided deterministic cluster profile.
- Return short, business-readable names, max 5 words each.
- Do not invent unsupported categories or facts.
- Avoid raw column names when a common business phrase is clearer.
- If the evidence is weak, keep the name cautious.
- Return JSON only: {"suggestions":[{"cluster_id":0,"name":"...","reason":"..."}]}.
"""


def _enabled() -> bool:
    value = os.getenv("AI_CLUSTER_NAMING_ENABLED", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _timeout_seconds() -> int:
    try:
        return max(1, int(os.getenv("AI_CLUSTER_NAMING_TIMEOUT_SECONDS", "4")))
    except (TypeError, ValueError):
        return 4


def _is_generic_name(name: str) -> bool:
    normalized = name.strip().lower()
    if not normalized:
        return True
    return any(pattern in normalized for pattern in GENERIC_NAME_PATTERNS)


def _should_try_groq(cluster_result: dict[str, Any]) -> bool:
    if not _enabled() or cluster_result.get("type") != "clustering":
        return False
    if cluster_result.get("detected_domain") in {"sports", "commerce"}:
        return False

    profiles = cluster_result.get("cluster_profiles") or []
    if not profiles:
        return False

    # Prefer the LLM only when deterministic naming is visibly generic.
    return any(_is_generic_name(str(profile.get("cluster_name", ""))) for profile in profiles)


def _compact_feature(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "column": item.get("column"),
        "label": item.get("label"),
        "level": item.get("level"),
        "value": item.get("value"),
        "lower_is_better": bool(item.get("lower_is_better")),
    }


def _compact_category(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": item.get("label"),
        "values": [value.get("value") for value in (item.get("values") or [])[:3]],
    }


def _compact_payload(cluster_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": cluster_result.get("title"),
        "silhouette_quality": cluster_result.get("silhouette_quality"),
        "warnings": cluster_result.get("warnings") or [],
        "clusters": [
            {
                "cluster_id": profile.get("cluster_id"),
                "deterministic_name": profile.get("cluster_name"),
                "size_pct": profile.get("size_pct"),
                "top_features": [
                    _compact_feature(item)
                    for item in (profile.get("feature_rankings") or [])[:4]
                ],
                "frequent_categories": [
                    _compact_category(item)
                    for item in (profile.get("top_categories") or [])[:3]
                ],
            }
            for profile in (cluster_result.get("cluster_profiles") or [])
        ],
    }


def _clean_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(name)).strip(" .:/\\|-")
    cleaned = cleaned[:60].strip()
    return cleaned


def _apply_suggestions(cluster_result: dict[str, Any], suggestions: ClusterNameSuggestions) -> dict[str, Any]:
    valid_cluster_ids = {
        int(profile.get("cluster_id"))
        for profile in cluster_result.get("cluster_profiles", [])
        if profile.get("cluster_id") is not None
    }
    names_by_id: dict[int, str] = {}
    reasons_by_id: dict[int, str] = {}
    used: dict[str, int] = {}

    for suggestion in suggestions.suggestions:
        cluster_id = int(suggestion.cluster_id)
        if cluster_id not in valid_cluster_ids:
            continue
        name = _clean_name(suggestion.name)
        if len(name) < 3:
            continue

        count = used.get(name, 0) + 1
        used[name] = count
        if count > 1:
            name = f"{name} {count}"

        names_by_id[cluster_id] = name
        reasons_by_id[cluster_id] = str(suggestion.reason or "")[:160]

    if not names_by_id:
        return cluster_result

    for profile in cluster_result.get("cluster_profiles", []):
        cluster_id = int(profile.get("cluster_id"))
        if cluster_id in names_by_id:
            profile["deterministic_cluster_name"] = profile.get("cluster_name")
            profile["cluster_name"] = names_by_id[cluster_id]
            profile["cluster_name_source"] = "groq_fallback"
            profile["cluster_name_reason"] = reasons_by_id.get(cluster_id, "")

    chart_data = cluster_result.get("chartData") or []
    for row in chart_data:
        try:
            cluster_id = int(row.get("cluster"))
        except (TypeError, ValueError):
            continue
        if cluster_id in names_by_id:
            row["deterministic_cluster_name"] = row.get("cluster_name")
            row["cluster_name"] = names_by_id[cluster_id]

    cluster_result["cluster_naming"] = {
        "source": "groq_fallback",
        "applied": True,
        "timeout_seconds": _timeout_seconds(),
    }
    return cluster_result


async def apply_groq_cluster_names(cluster_result: dict[str, Any]) -> dict[str, Any]:
    """Optionally replace generic deterministic cluster names with Groq suggestions.

    The input already contains all Pandas-generated metrics and categories.
    On any failure, the original deterministic cluster result is returned.
    """
    if not _should_try_groq(cluster_result):
        cluster_result.setdefault("cluster_naming", {"source": "deterministic", "applied": False})
        return cluster_result

    payload = _compact_payload(cluster_result)
    timeout_seconds = _timeout_seconds()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Suggest better cluster names for this deterministic clustering payload. "
                "Use only the supplied fields; do not calculate anything.\n"
                f"{json.dumps(payload, ensure_ascii=False)}"
            ),
        },
    ]

    try:
        suggestions = await asyncio.wait_for(
            call_groq_structured(
                messages=messages,
                schema_model=ClusterNameSuggestions,
                temperature=0.2,
                timeout_seconds=timeout_seconds,
            ),
            timeout=timeout_seconds + 1,
        )
        typed_suggestions = suggestions
        if not isinstance(typed_suggestions, ClusterNameSuggestions):
            typed_suggestions = ClusterNameSuggestions(**model_dump_compat(suggestions))
        return _apply_suggestions(cluster_result, typed_suggestions)
    except Exception:
        cluster_result["cluster_naming"] = {
            "source": "deterministic",
            "applied": False,
            "fallback_reason": "groq_unavailable_or_timeout",
        }
        return cluster_result
