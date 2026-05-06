"""
clustering_feature_selector.py
──────────────────────────────────────────────────────────────────────────
Backend validation layer for clustering feature selection (Phase D).

Priority:
  1. Features from semantic_plan.recommended_analyses where type == "clustering"
  2. Features from semantic_plan.metrics where include_in_clustering == True
  3. Heuristic variance-based fallback (existing behaviour)

Every candidate feature is validated:
  - column exists in df
  - column is numeric
  - column is not ID-like
  - sufficient non-null values (>= 10 or >= 20% of rows, whichever is larger)
  - sufficient variance (std > 1e-6)
  - missing rate <= 60%

At least 2 features must survive validation; otherwise heuristic fallback runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from services.utils import is_id_column

# ── Thresholds ────────────────────────────────────────────────────────────────
_MIN_NON_NULL = 10
_MIN_NON_NULL_FRAC = 0.20
_MAX_MISSING_RATE = 0.60
_MIN_STD = 1e-6
_MIN_FEATURE_COUNT = 2
_MAX_HEURISTIC_FEATURES = 4


def _validate_feature(col: str, df: pd.DataFrame) -> tuple[bool, str]:
    """Validate a single candidate feature. Returns (ok, reason_if_rejected)."""
    if col not in df.columns:
        return False, "column_not_found"

    series = df[col]

    if not pd.api.types.is_numeric_dtype(series):
        return False, "non_numeric"

    if is_id_column(col, series):
        return False, "id_like"

    n_total = len(series)
    n_non_null = int(series.notna().sum())
    missing_rate = 1.0 - n_non_null / n_total if n_total else 1.0

    min_non_null = max(_MIN_NON_NULL, int(n_total * _MIN_NON_NULL_FRAC))
    if n_non_null < min_non_null:
        return False, f"insufficient_non_null ({n_non_null} < {min_non_null})"

    if missing_rate > _MAX_MISSING_RATE:
        return False, f"high_missing_rate ({missing_rate:.0%})"

    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if float(numeric.std()) < _MIN_STD:
        return False, "low_variance"

    return True, ""


def _extract_semantic_candidates(semantic_plan: Any) -> list[str]:
    """
    Extract candidate feature columns from a validated FastSemanticPlan.
    Priority: clustering recommended_analyses features > include_in_clustering metrics.
    """
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(col: str) -> None:
        if col and col not in seen:
            candidates.append(col)
            seen.add(col)

    # Priority 1: recommended_analyses where type == "clustering"
    for analysis in getattr(semantic_plan, "recommended_analyses", []):
        if getattr(analysis, "type", "") == "clustering":
            for feat in getattr(analysis, "features", []):
                _add(feat)

    # Priority 2: metrics where include_in_clustering == True
    for metric in getattr(semantic_plan, "metrics", []):
        if getattr(metric, "include_in_clustering", False):
            _add(getattr(metric, "column", ""))

    return candidates


def select_clustering_features(
    df: pd.DataFrame,
    semantic_plan: Any | None = None,
) -> dict[str, Any]:
    """
    Select and validate clustering features.

    Returns a dict with:
      feature_source: "semantic_plan" | "heuristic_fallback"
      selected_features: list[str]
      rejected_features: list[dict]   # {column, reason}
      semantic_context: str | None
      fallback_reason: str | None
    """
    rejected: list[dict] = []
    fallback_reason: str | None = None
    semantic_context: str | None = None

    # ── Try semantic plan ──────────────────────────────────────────────────────
    if semantic_plan is not None:
        semantic_context = getattr(semantic_plan, "detected_domain", None)
        candidates = _extract_semantic_candidates(semantic_plan)

        if candidates:
            selected: list[str] = []
            for col in candidates:
                ok, reason = _validate_feature(col, df)
                if ok:
                    selected.append(col)
                else:
                    rejected.append({"column": col, "reason": reason})

            if len(selected) >= _MIN_FEATURE_COUNT:
                return {
                    "feature_source": "semantic_plan",
                    "selected_features": selected,
                    "rejected_features": rejected,
                    "semantic_context": semantic_context,
                    "fallback_reason": None,
                }

            # Not enough valid semantic features → note why and fall through
            fallback_reason = (
                f"only {len(selected)} semantic feature(s) passed validation "
                f"(need >= {_MIN_FEATURE_COUNT})"
            )
        else:
            fallback_reason = "semantic_plan provided no clustering feature candidates"

    elif semantic_plan is None:
        fallback_reason = "no semantic plan provided"

    # ── Heuristic fallback ─────────────────────────────────────────────────────
    num_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns.tolist()
        if df[c].dropna().nunique() > 2 and not is_id_column(str(c), df[c])
    ]
    variances = df[num_cols].apply(pd.to_numeric, errors="coerce").var().sort_values(ascending=False)
    heuristic_selected: list[str] = []
    for col in variances.index:
        ok, reason = _validate_feature(str(col), df)
        if ok:
            heuristic_selected.append(str(col))
        else:
            rejected.append({"column": str(col), "reason": reason})
        if len(heuristic_selected) >= _MAX_HEURISTIC_FEATURES:
            break

    return {
        "feature_source": "heuristic_fallback",
        "selected_features": heuristic_selected,
        "rejected_features": rejected,
        "semantic_context": semantic_context,
        "fallback_reason": fallback_reason,
    }


# ── Title generation ───────────────────────────────────────────────────────────

# Domain tokens that require multi-token confirmation to avoid false positives.
# Single-token matches for "device" should not force "Sensor Clusters", etc.
_DOMAIN_TITLE_MAP: dict[str, tuple[list[str], int, str]] = {
    # (required_tokens, min_match_count, title)
    "customer":    (["customer", "client", "musteri", "churn", "recency", "loyalty"], 1, "Customer Segments"),
    "employee":    (["employee", "staff", "worker", "calisan", "personel", "salary", "tenure", "department"], 2, "Workforce Segments"),
    "sports":      (["player", "athlete", "oyuncu", "goal", "assist", "gol", "minute", "rating", "xg"], 2, "Performance Segments"),
    "health":      (["patient", "diagnosis", "hospital", "clinic", "bmi", "glucose", "mortality"], 2, "Clinical Segments"),
    "finance":     (["credit", "loan", "debt", "default", "fraud", "npl", "income", "balance"], 2, "Financial Risk Segments"),
    "manufacturing": (["machine", "downtime", "defect", "reject", "yield", "production"], 2, "Production Segments"),
    "iot":         (["sensor", "temperature", "humidity", "vibration", "pressure", "voltage"], 2, "Sensor Segments"),
    "logistics":   (["order", "ship", "deliver", "siparis", "kargo", "warehouse", "carrier"], 2, "Logistics Segments"),
    "commerce":    (["product", "item", "sku", "urun", "profit", "revenue", "sales", "price", "retail"], 2, "Product Segments"),
}

_SAFE_GENERIC_TITLES = ["Behavioral Segments", "Feature-Based Segments", "Record Similarity Segments"]


def _safe_generic_title(k: int) -> str:
    return _SAFE_GENERIC_TITLES[0]


def build_clustering_title(
    df: pd.DataFrame,
    selected_features: list[str],
    semantic_context: str | None,
    optimal_k: int,
) -> str:
    """
    Produce a clustering title from semantic context + column signals.
    Falls back to a safe generic title when domain evidence is weak/ambiguous.
    """
    cols_text = " ".join(str(c).lower() for c in list(df.columns) + selected_features)

    best_domain: str | None = None
    best_matches = 0

    for domain, (tokens, min_match, title) in _DOMAIN_TITLE_MAP.items():
        matches = sum(1 for t in tokens if t in cols_text)
        if matches >= min_match and matches > best_matches:
            best_domain = domain
            best_matches = matches

    if best_domain:
        _, _, domain_title = _DOMAIN_TITLE_MAP[best_domain]
        return f"{domain_title} ({optimal_k} Groups)"

    return f"{_safe_generic_title(optimal_k)} ({optimal_k} Groups)"
