import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from services.ai.fast_chart_builder import build_charts_from_fast_plan  # noqa: E402
from services.ai.schemas import FastAnalysisPlan, FastMetricPlan, FastSemanticPlan  # noqa: E402
from services.ai.semantic_validation import validate_fast_semantic_plan  # noqa: E402


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["A", "B", "A", "C", "B", "C"],
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "revenue": [100, 200, 150, 300, 250, 350],
            "sales": [10, 20, 15, 30, 25, 35],
            "goals": [1, 2, 1, 3, 2, 4],
            "rating": [7.1, 8.2, 7.8, 8.7, 8.4, 9.1],
            "score": [80, 82, 77, 90, 88, 91],
            "student_id": [101, 102, 103, 104, 105, 106],
            "name": ["Ava", "Ben", "Cara", "Dev", "Eli", "Fay"],
        }
    )


def _validated(plan: FastSemanticPlan, df: pd.DataFrame) -> FastSemanticPlan:
    return validate_fast_semantic_plan(plan, df)


def _revenue_metric() -> FastMetricPlan:
    return FastMetricPlan(
        column="revenue",
        role="outcome_metric",
        semantic_type="money",
        aggregation="sum",
        direction="higher_is_better",
    )


def _goals_metric() -> FastMetricPlan:
    return FastMetricPlan(
        column="goals",
        role="outcome_metric",
        semantic_type="quantity",
        aggregation="sum",
        direction="higher_is_better",
    )


def _rating_metric() -> FastMetricPlan:
    return FastMetricPlan(
        column="rating",
        role="outcome_metric",
        semantic_type="score",
        aggregation="sum",
        direction="higher_is_better",
    )


def test_recommended_bar_analysis_produces_one_valid_chart():
    df = _df()
    plan = _validated(
        FastSemanticPlan(
            detected_domain="commerce",
            metrics=[_revenue_metric()],
            recommended_analyses=[
                FastAnalysisPlan(type="bar", metric="revenue", dimension="category", aggregation="sum", priority=5)
            ],
        ),
        df,
    )

    charts = build_charts_from_fast_plan(df, plan, max_charts=1)

    assert len(charts) == 1
    assert charts[0]["type"] == "bar"
    assert charts[0]["plan_source"] == "semantic_plan"
    assert charts[0]["xAxisKey"] == "category"
    assert charts[0]["series"] == [{"key": "revenue"}]
    assert charts[0]["chartData"]


def test_duplicate_recommended_analyses_produce_one_chart():
    df = _df()
    plan = _validated(
        FastSemanticPlan(
            detected_domain="commerce",
            metrics=[_revenue_metric()],
            recommended_analyses=[
                FastAnalysisPlan(type="bar", metric="revenue", dimension="category", aggregation="sum", priority=5),
                FastAnalysisPlan(type="bar", metric="revenue", dimension="category", aggregation="sum", priority=5),
            ],
        ),
        df,
    )

    charts = build_charts_from_fast_plan(df, plan, max_charts=5)
    semantic_bar_count = sum(1 for chart in charts if chart["type"] == "bar" and chart["plan_source"] == "semantic_plan")

    assert semantic_bar_count == 1
    assert len(charts) == 1


def test_invalid_recommended_columns_are_skipped_without_crash():
    df = _df()
    plan = _validated(
        FastSemanticPlan(
            detected_domain="commerce",
            primary_metrics=["revenue"],
            dimensions=["category"],
            metrics=[_revenue_metric()],
            recommended_analyses=[
                FastAnalysisPlan(type="bar", metric="made_up", dimension="category", aggregation="sum", priority=5)
            ],
        ),
        df,
    )

    charts = build_charts_from_fast_plan(df, plan, max_charts=1)

    assert len(charts) == 1
    assert charts[0]["plan_source"] == "heuristic_fallback"
    assert charts[0]["series"][0]["key"] == "revenue"
    assert "made_up" not in str(charts[0])


def test_heatmap_request_falls_back_to_bar_with_clear_reason():
    df = _df()
    plan = _validated(
        FastSemanticPlan(
            detected_domain="commerce",
            metrics=[_revenue_metric()],
            recommended_analyses=[
                FastAnalysisPlan(type="heatmap", metric="revenue", second_metric="sales", dimension="category", priority=5),
            ],
        ),
        df,
    )

    charts = build_charts_from_fast_plan(df, plan, max_charts=1)

    assert len(charts) == 1
    assert charts[0]["type"] == "bar"
    assert charts[0]["plan_source"] == "semantic_plan"
    assert charts[0]["fallback_reason"] == (
        "Heatmap is not currently supported by the renderer. "
        "A bar chart is shown as a compatible fallback."
    )
    assert charts[0]["fallback_reason"] in charts[0]["insight"]


def test_treemap_request_falls_back_to_supported_chart_with_clear_reason():
    df = _df()
    plan = _validated(
        FastSemanticPlan(
            detected_domain="commerce",
            metrics=[_revenue_metric()],
            recommended_analyses=[
                FastAnalysisPlan(type="treemap", metric="revenue", dimension="category", priority=5),
            ],
        ),
        df,
    )

    charts = build_charts_from_fast_plan(df, plan, max_charts=1)

    assert len(charts) == 1
    assert charts[0]["type"] == "bar"
    assert charts[0]["fallback_reason"] == (
        "Treemap is not currently supported by the renderer. "
        "A bar chart is shown as a compatible fallback."
    )


def test_pie_mean_rating_falls_back_to_bar_with_part_to_whole_reason():
    df = _df()
    plan = _validated(
        FastSemanticPlan(
            detected_domain="sports",
            metrics=[_rating_metric()],
            recommended_analyses=[
                FastAnalysisPlan(type="pie", metric="rating", dimension="category", aggregation="mean", priority=5),
            ],
        ),
        df,
    )

    charts = build_charts_from_fast_plan(df, plan, max_charts=1)

    assert len(charts) == 1
    assert charts[0]["type"] == "bar"
    assert charts[0]["plan_source"] == "semantic_plan"
    assert charts[0]["fallback_reason"] == (
        "Pie/donut charts are intended for part-to-whole values. "
        "A bar chart is used for comparing averages."
    )


def test_pie_total_goals_by_team_remains_pie_for_low_cardinality_additive_metric():
    df = _df()
    plan = _validated(
        FastSemanticPlan(
            detected_domain="sports",
            metrics=[_goals_metric()],
            recommended_analyses=[
                FastAnalysisPlan(type="pie", metric="goals", dimension="category", aggregation="sum", priority=5),
            ],
        ),
        df,
    )

    charts = build_charts_from_fast_plan(df, plan, max_charts=1)

    assert len(charts) == 1
    assert charts[0]["type"] == "pie"
    assert charts[0]["plan_source"] == "semantic_plan"
    assert "fallback_reason" not in charts[0]


def test_high_cardinality_pie_falls_back_to_bar():
    df = pd.DataFrame(
        {
            "team": [f"Team {i}" for i in range(10)],
            "goals": list(range(1, 11)),
        }
    )
    plan = _validated(
        FastSemanticPlan(
            detected_domain="sports",
            metrics=[_goals_metric()],
            recommended_analyses=[
                FastAnalysisPlan(type="pie", metric="goals", dimension="team", aggregation="sum", priority=5),
            ],
        ),
        df,
    )

    charts = build_charts_from_fast_plan(df, plan, max_charts=1)

    assert len(charts) == 1
    assert charts[0]["type"] == "bar"
    assert "low-cardinality part-to-whole" in charts[0]["fallback_reason"]
    assert "10 categories" in charts[0]["fallback_reason"]


def test_no_semantic_recommendations_use_heuristic_fallback():
    df = _df()
    plan = FastSemanticPlan(
        detected_domain="commerce",
        primary_metrics=["revenue"],
        dimensions=["category"],
    )

    charts = build_charts_from_fast_plan(df, plan, max_charts=1)

    assert len(charts) == 1
    assert charts[0]["plan_source"] == "heuristic_fallback"
    assert charts[0]["series"][0]["key"] == "revenue"


if __name__ == "__main__":
    test_recommended_bar_analysis_produces_one_valid_chart()
    test_duplicate_recommended_analyses_produce_one_chart()
    test_invalid_recommended_columns_are_skipped_without_crash()
    test_heatmap_request_falls_back_to_bar_with_clear_reason()
    test_treemap_request_falls_back_to_supported_chart_with_clear_reason()
    test_pie_mean_rating_falls_back_to_bar_with_part_to_whole_reason()
    test_pie_total_goals_by_team_remains_pie_for_low_cardinality_additive_metric()
    test_high_cardinality_pie_falls_back_to_bar()
    test_no_semantic_recommendations_use_heuristic_fallback()
    print("fast chart builder tests passed")
