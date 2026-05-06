import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from services.ai.fast_kpi_builder import build_kpis_from_fast_plan  # noqa: E402
from services.ai.schemas import FastMetricPlan, FastSemanticPlan  # noqa: E402


def _kpi_by_column(kpis: list[dict]) -> dict[str, dict]:
    return {kpi["column"]: kpi for kpi in kpis}


def _metric(column: str, aggregation: str, status: str = "accepted", role: str = "supporting_metric") -> FastMetricPlan:
    return FastMetricPlan(
        column=column,
        role=role,
        aggregation=aggregation,
        validation_status=status,
        aggregation_source="ai_validated" if status == "accepted" else "backend_repaired",
    )


def test_semantic_mean_kpis():
    df = pd.DataFrame({
        "grades": [80, 90, 70],
        "rating": [4.0, 5.0, 3.0],
        "attendance": [90, 80, 100],
        "score": [10, 20, 30],
    })
    plan = FastSemanticPlan(
        detected_domain="education",
        primary_metrics=["grades", "rating", "attendance", "score"],
        metrics=[
            _metric("grades", "mean"),
            _metric("rating", "mean"),
            _metric("attendance", "mean"),
            _metric("score", "mean"),
        ],
    )

    kpis = _kpi_by_column(build_kpis_from_fast_plan(df, plan))

    assert kpis["grades"]["title"] == "Avg Grades"
    assert kpis["grades"]["rawValue"] == 80
    assert kpis["rating"]["title"] == "Avg Rating"
    assert kpis["attendance"]["title"] == "Avg Attendance"
    assert kpis["score"]["title"] == "Avg Score"


def test_semantic_sum_kpis():
    df = pd.DataFrame({
        "revenue": [100, 200, 300],
        "sales": [10, 20, 30],
        "goals": [1, 2, 3],
        "assists": [3, 2, 1],
        "touches": [30, 40, 50],
    })
    plan = FastSemanticPlan(
        detected_domain="sports_sales",
        primary_metrics=["revenue", "sales", "goals", "assists"],
        metrics=[
            _metric("revenue", "sum"),
            _metric("sales", "sum"),
            _metric("goals", "sum"),
            _metric("assists", "sum"),
            _metric("touches", "sum"),
        ],
    )
    kpis = _kpi_by_column(build_kpis_from_fast_plan(df, plan))

    for column in ["revenue", "sales", "goals", "assists"]:
        assert kpis[column]["title"] == f"Total {column.replace('_', ' ').title()}"

    touches_plan = FastSemanticPlan(
        detected_domain="sports",
        primary_metrics=["touches"],
        metrics=[_metric("touches", "sum")],
    )
    touches_kpi = build_kpis_from_fast_plan(df, touches_plan)[0]
    assert touches_kpi["title"] == "Total Touches"
    assert touches_kpi["rawValue"] == 120


def test_selected_metric_order_and_stable_ids():
    df = pd.DataFrame({
        "attendance": [90, 80],
        "grades": [70, 100],
        "revenue": [200, 300],
    })
    plan = FastSemanticPlan(
        detected_domain="education",
        primary_metrics=["attendance", "grades"],
        secondary_metrics=["revenue", "grades"],
        metrics=[
            _metric("attendance", "mean"),
            _metric("grades", "mean"),
            _metric("revenue", "sum"),
        ],
    )
    kpis = build_kpis_from_fast_plan(df, plan)

    assert [kpi["column"] for kpi in kpis] == ["attendance", "grades", "revenue"]
    assert [kpi["id"] for kpi in kpis] == [
        "ai-kpi-attendance",
        "ai-kpi-grades",
        "ai-kpi-revenue",
    ]


def test_rejected_metrics_are_not_kpis():
    df = pd.DataFrame({
        "student_id": [1, 2, 3],
        "student_name": ["A", "B", "C"],
        "grades": [70, 80, 90],
    })
    plan = FastSemanticPlan(
        detected_domain="education",
        primary_metrics=["student_id", "student_name", "grades"],
        metrics=[
            _metric("student_id", "sum", status="rejected"),
            _metric("student_name", "count", status="rejected"),
            _metric("grades", "mean"),
        ],
    )
    kpis = build_kpis_from_fast_plan(df, plan)

    assert [kpi["column"] for kpi in kpis] == ["grades"]


def test_no_semantic_metadata_falls_back_to_old_inference():
    df = pd.DataFrame({
        "score": [10, 20, 30],
        "sales": [100, 200, 300],
    })
    plan = FastSemanticPlan(
        detected_domain="general",
        primary_metrics=["score", "sales"],
    )
    kpis = _kpi_by_column(build_kpis_from_fast_plan(df, plan))

    assert kpis["score"]["title"] == "Avg Score"
    assert kpis["sales"]["title"] == "Total Sales"


if __name__ == "__main__":
    test_semantic_mean_kpis()
    test_semantic_sum_kpis()
    test_selected_metric_order_and_stable_ids()
    test_rejected_metrics_are_not_kpis()
    test_no_semantic_metadata_falls_back_to_old_inference()
    print("fast KPI builder semantic integration tests passed")
