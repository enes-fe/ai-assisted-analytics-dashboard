"""
test_semantic_validation.py — Phase D clustering integration tests
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from services.ai.schemas import FastAnalysisPlan, FastMetricPlan, FastSemanticPlan  # noqa: E402
from services.ai.semantic_validation import validate_fast_semantic_plan  # noqa: E402
from services.clustering_feature_selector import (  # noqa: E402
    build_clustering_title,
    select_clustering_features,
)
from services.ml_engine import run_clustering  # noqa: E402


def _by_column(plan: FastSemanticPlan) -> dict[str, FastMetricPlan]:
    return {metric.column: metric for metric in plan.metrics}


# ─── Existing contract test (unchanged) ───────────────────────────────────────

def test_semantic_validation_contract():
    df = pd.DataFrame({
        "student_id": [1, 2, 3, 4, 5, 6],
        "student_name": ["A", "B", "C", "D", "E", "F"],
        "gender": ["F", "M", "F", "M", "F", "M"],
        "grades": [80, 90, 72, 88, 95, 60],
        "attendance": [95, 88, 70, 92, 100, 65],
        "attendance_count": [20, 18, 15, 19, 22, 12],
        "revenue": [1000, 900, 1100, 1200, 1050, 980],
        "cost": [500, 420, 550, 610, 530, 460],
    })
    raw_plan = FastSemanticPlan(
        detected_domain="education",
        primary_metrics=["grades", "made_up"],
        secondary_metrics=["attendance", "revenue"],
        dimensions=["gender"],
        metrics=[
            FastMetricPlan(column="grades", role="outcome_metric", semantic_type="raw_numeric", aggregation="sum", direction="neutral"),
            FastMetricPlan(column="attendance", role="supporting_metric", semantic_type="raw_numeric", aggregation="sum", direction="neutral"),
            FastMetricPlan(column="attendance_count", role="supporting_metric", semantic_type="count", aggregation="sum", direction="neutral"),
            FastMetricPlan(column="revenue", role="supporting_metric", semantic_type="money", aggregation="sum", direction="higher_is_better"),
            FastMetricPlan(column="cost", role="supporting_metric", semantic_type="money", aggregation="sum", direction="higher_is_better"),
            FastMetricPlan(column="student_name", role="outcome_metric", semantic_type="score", aggregation="mean", direction="neutral"),
            FastMetricPlan(column="student_id", role="outcome_metric", semantic_type="count", aggregation="sum", direction="neutral"),
            FastMetricPlan(column="made_up", role="supporting_metric", semantic_type="raw_numeric", aggregation="sum", direction="neutral"),
            FastMetricPlan(column="grades", role="performance_metric", semantic_type="rating", aggregation="average", direction="positive"),
        ],
        recommended_analyses=[
            FastAnalysisPlan(type="cluster", features=["grades", "made_up", "student_id"], priority=4),
        ],
    )

    validated = validate_fast_semantic_plan(raw_plan, df)
    metrics = _by_column(validated)

    assert validated.primary_metrics == ["grades"]
    assert "made_up" not in validated.primary_metrics

    assert metrics["grades"].aggregation == "mean"
    assert metrics["grades"].validation_status == "repaired"
    assert metrics["grades"].aggregation_source == "backend_repaired"

    assert metrics["attendance"].aggregation == "mean"
    assert metrics["attendance"].aggregation_source == "backend_repaired"

    assert metrics["attendance_count"].aggregation == "sum"
    assert metrics["attendance_count"].validation_status in {"accepted", "repaired"}

    assert metrics["revenue"].aggregation == "sum"
    assert metrics["revenue"].validation_status == "accepted"

    assert metrics["cost"].aggregation == "sum"
    assert metrics["cost"].direction == "lower_is_better"
    assert metrics["cost"].validation_status == "repaired"

    assert metrics["student_name"].validation_status == "rejected"
    assert metrics["student_name"].repair_reason == "non_numeric_metric"

    assert metrics["student_id"].validation_status == "rejected"
    assert metrics["student_id"].repair_reason == "id_like_metric"

    assert metrics["made_up"].validation_status == "rejected"
    assert metrics["made_up"].repair_reason == "column_not_found"

    normalized_grades = validated.metrics[-1]
    assert normalized_grades.role == "outcome_metric"
    assert normalized_grades.semantic_type == "score"
    assert normalized_grades.aggregation == "mean"
    assert normalized_grades.direction == "higher_is_better"

    analysis = validated.recommended_analyses[0]
    assert analysis.type == "clustering"
    assert analysis.features == ["grades"]
    assert analysis.validation_status == "repaired"

    assert validated.validation_summary["rejected_metrics"] == 3
    print("  [PASS] semantic_validation_contract")


# ─── Phase D: select_clustering_features ──────────────────────────────────────

def _make_df_30() -> pd.DataFrame:
    """30-row numeric DataFrame for clustering tests."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "revenue": rng.uniform(100, 1000, 30),
        "cost": rng.uniform(50, 500, 30),
        "score": rng.uniform(1, 10, 30),
        "customer_id": list(range(1, 31)),           # ID-like → should be rejected
        "constant": [5.0] * 30,                      # zero variance → should be rejected
    })


def test_semantic_features_used_when_valid():
    df = _make_df_30()
    plan = FastSemanticPlan(
        detected_domain="sales",
        metrics=[
            FastMetricPlan(column="revenue", include_in_clustering=True, role="outcome_metric"),
            FastMetricPlan(column="cost", include_in_clustering=True, role="supporting_metric"),
        ],
        recommended_analyses=[],
    )
    result = select_clustering_features(df, semantic_plan=plan)
    assert result["feature_source"] == "semantic_plan", result
    assert "revenue" in result["selected_features"]
    assert "cost" in result["selected_features"]
    assert result["fallback_reason"] is None
    print("  [PASS] semantic_features_used_when_valid")


def test_invalid_semantic_features_rejected_with_reasons():
    df = _make_df_30()
    plan = FastSemanticPlan(
        detected_domain="sales",
        metrics=[
            FastMetricPlan(column="customer_id", include_in_clustering=True, role="outcome_metric"),
            FastMetricPlan(column="constant", include_in_clustering=True, role="supporting_metric"),
        ],
        recommended_analyses=[],
    )
    result = select_clustering_features(df, semantic_plan=plan)
    # Both fail → must fall back
    assert result["feature_source"] == "heuristic_fallback", result
    rejected_cols = {r["column"] for r in result["rejected_features"]}
    assert "customer_id" in rejected_cols or "constant" in rejected_cols
    print("  [PASS] invalid_semantic_features_rejected_with_reasons")


def test_non_numeric_feature_rejected():
    df = pd.DataFrame({
        "name": ["A"] * 30,
        "revenue": np.random.uniform(100, 1000, 30),
        "cost": np.random.uniform(50, 500, 30),
    })
    plan = FastSemanticPlan(
        detected_domain="general",
        metrics=[
            FastMetricPlan(column="name", include_in_clustering=True, role="dimension"),
            FastMetricPlan(column="revenue", include_in_clustering=True, role="outcome_metric"),
            FastMetricPlan(column="cost", include_in_clustering=True, role="supporting_metric"),
        ],
        recommended_analyses=[],
    )
    result = select_clustering_features(df, semantic_plan=plan)
    assert "name" not in result["selected_features"]
    rejected_cols = {r["column"] for r in result["rejected_features"]}
    assert "name" in rejected_cols
    print("  [PASS] non_numeric_feature_rejected")


def test_id_like_feature_rejected():
    df = _make_df_30()
    plan = FastSemanticPlan(
        detected_domain="general",
        recommended_analyses=[
            FastAnalysisPlan(type="clustering", features=["customer_id", "revenue", "cost"]),
        ],
    )
    result = select_clustering_features(df, semantic_plan=plan)
    assert "customer_id" not in result["selected_features"]
    print("  [PASS] id_like_feature_rejected")


def test_low_variance_feature_rejected():
    df = _make_df_30()
    plan = FastSemanticPlan(
        detected_domain="general",
        recommended_analyses=[
            FastAnalysisPlan(type="clustering", features=["constant", "revenue", "cost"]),
        ],
    )
    result = select_clustering_features(df, semantic_plan=plan)
    assert "constant" not in result["selected_features"]
    rejected_cols = {r["column"] for r in result["rejected_features"]}
    assert "constant" in rejected_cols
    print("  [PASS] low_variance_feature_rejected")


def test_fewer_than_2_semantic_features_triggers_heuristic_fallback():
    df = _make_df_30()
    plan = FastSemanticPlan(
        detected_domain="general",
        recommended_analyses=[
            FastAnalysisPlan(type="clustering", features=["customer_id"]),  # only 1, invalid
        ],
    )
    result = select_clustering_features(df, semantic_plan=plan)
    assert result["feature_source"] == "heuristic_fallback", result
    assert "fallback_reason" in result and result["fallback_reason"]
    print("  [PASS] fewer_than_2_semantic_features_triggers_heuristic_fallback")


def test_feature_source_metadata_is_correct():
    df = _make_df_30()

    # No plan → heuristic
    result_no_plan = select_clustering_features(df, semantic_plan=None)
    assert result_no_plan["feature_source"] == "heuristic_fallback"

    # Valid plan → semantic
    plan = FastSemanticPlan(
        detected_domain="sales",
        recommended_analyses=[
            FastAnalysisPlan(type="clustering", features=["revenue", "cost"]),
        ],
    )
    result_semantic = select_clustering_features(df, semantic_plan=plan)
    assert result_semantic["feature_source"] == "semantic_plan"
    print("  [PASS] feature_source_metadata_is_correct")


# ─── Phase D: build_clustering_title ──────────────────────────────────────────

def test_ambiguous_domain_uses_safe_generic_title():
    """A DataFrame with only 'device_usage' should NOT force 'Sensor Segments'."""
    df = pd.DataFrame({
        "device_usage": np.random.uniform(0, 100, 30),
        "score": np.random.uniform(1, 10, 30),
    })
    title = build_clustering_title(df, ["device_usage", "score"], None, 3)
    # "device" alone should not cross the 2-token threshold for IoT domain
    assert "Sensor" not in title, f"Got wrong domain title: {title}"
    print(f"  [PASS] ambiguous_domain_uses_safe_generic_title (got: '{title}')")


def test_clear_domain_produces_domain_title():
    """Customer + churn columns should yield a Customer Segments title."""
    df = pd.DataFrame({
        "customer_lifetime_value": np.random.uniform(100, 5000, 30),
        "churn_score": np.random.uniform(0, 1, 30),
        "recency": np.random.randint(1, 365, 30),
    })
    title = build_clustering_title(df, ["customer_lifetime_value", "churn_score"], "customer", 2)
    assert "Customer" in title or "Segment" in title, f"Unexpected title: {title}"
    print(f"  [PASS] clear_domain_produces_domain_title (got: '{title}')")


# ─── Phase D: run_clustering metadata ─────────────────────────────────────────

def test_run_clustering_emits_metadata():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "revenue": rng.uniform(100, 1000, 50),
        "cost": rng.uniform(50, 500, 50),
        "score": rng.uniform(1, 10, 50),
    })
    plan = FastSemanticPlan(
        detected_domain="sales",
        recommended_analyses=[
            FastAnalysisPlan(type="clustering", features=["revenue", "cost"]),
        ],
    )
    result = run_clustering(df, semantic_plan=plan)
    assert "error" not in result, result.get("error")
    assert "clustering_metadata" in result
    meta = result["clustering_metadata"]
    assert meta["feature_source"] in {"semantic_plan", "heuristic_fallback"}
    assert isinstance(meta["selected_features"], list)
    assert isinstance(meta["rejected_features"], list)
    print(f"  [PASS] run_clustering_emits_metadata (source={meta['feature_source']})")


def test_run_clustering_no_plan_uses_heuristic():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "a": rng.uniform(0, 10, 30),
        "b": rng.uniform(0, 10, 30),
        "c": rng.uniform(0, 10, 30),
    })
    result = run_clustering(df, semantic_plan=None)
    assert "error" not in result, result.get("error")
    meta = result["clustering_metadata"]
    assert meta["feature_source"] == "heuristic_fallback"
    print("  [PASS] run_clustering_no_plan_uses_heuristic")


# ─── Phase D: Groq naming-only enforcement (structural test) ──────────────────

def test_groq_fallback_does_not_change_calculated_data():
    """
    _apply_suggestions in cluster_namer must only change cluster_name fields,
    never silhouette_score, size, size_pct, or chartData metrics.
    """
    from services.ai.schemas import ClusterNameSuggestion, ClusterNameSuggestions
    from services.ai.cluster_namer import _apply_suggestions

    cluster_result = {
        "type": "clustering",
        "silhouette_score": 0.42,
        "davies_bouldin_score": 1.1,
        "cluster_profiles": [
            {"cluster_id": 0, "cluster_name": "High Segment", "size": 25, "size_pct": 50.0},
            {"cluster_id": 1, "cluster_name": "Low Segment",  "size": 25, "size_pct": 50.0},
        ],
        "chartData": [
            {"cluster": "0", "cluster_name": "High Segment", "revenue": 800},
            {"cluster": "1", "cluster_name": "Low Segment",  "revenue": 200},
        ],
    }

    suggestions = ClusterNameSuggestions(suggestions=[
        ClusterNameSuggestion(cluster_id=0, name="Premium Buyers", reason="high revenue"),
        ClusterNameSuggestion(cluster_id=1, name="Budget Segment", reason="low revenue"),
    ])

    updated = _apply_suggestions(cluster_result, suggestions)

    # Calculated fields must be unchanged
    assert updated["silhouette_score"] == 0.42
    assert updated["davies_bouldin_score"] == 1.1
    assert updated["cluster_profiles"][0]["size"] == 25
    assert updated["cluster_profiles"][0]["size_pct"] == 50.0
    assert updated["chartData"][0]["revenue"] == 800

    # Names should be updated
    assert updated["cluster_profiles"][0]["cluster_name"] == "Premium Buyers"
    assert updated["cluster_profiles"][1]["cluster_name"] == "Budget Segment"
    assert updated["chartData"][0]["cluster_name"] == "Premium Buyers"
    print("  [PASS] groq_fallback_does_not_change_calculated_data")


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running semantic validation + Phase D clustering tests...")
    test_semantic_validation_contract()
    test_semantic_features_used_when_valid()
    test_invalid_semantic_features_rejected_with_reasons()
    test_non_numeric_feature_rejected()
    test_id_like_feature_rejected()
    test_low_variance_feature_rejected()
    test_fewer_than_2_semantic_features_triggers_heuristic_fallback()
    test_feature_source_metadata_is_correct()
    test_ambiguous_domain_uses_safe_generic_title()
    test_clear_domain_produces_domain_title()
    test_run_clustering_emits_metadata()
    test_run_clustering_no_plan_uses_heuristic()
    test_groq_fallback_does_not_change_calculated_data()
    print("\nAll tests passed.")
