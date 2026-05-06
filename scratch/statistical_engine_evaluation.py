import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from ml_service import (  # noqa: E402
    generate_heuristic_charts,
    run_clustering,
    run_forecasting,
    run_statistical_tests,
)


def _scenario_result(name, expected, actual, passed, confidence, notes):
    return {
        "scenario": name,
        "expected": expected,
        "actual": actual,
        "passed": passed,
        "confidence": confidence,
        "notes": notes,
    }


def evaluate_statistical_engine():
    rng = np.random.default_rng(42)
    results = []

    normal_df = pd.DataFrame({
        "metric": rng.normal(100, 10, 160),
        "related": np.linspace(10, 90, 160) + rng.normal(0, 3, 160),
    })
    normal_df.loc[::53, "metric"] = np.nan
    normal_stats = run_statistical_tests(normal_df)
    normal_metric = normal_stats["normality"].get("metric", {})
    normal_flag = normal_metric.get("is_normal")
    normal_meta_ok = (
        normal_metric.get("shape") in {"normal", "approximately_normal"}
        and normal_metric.get("sample_size", 0) >= 100
        and normal_metric.get("missing_dropped_count", 0) > 0
    )
    results.append(_scenario_result(
        "Normality and missing-data metadata",
        "Normally distributed metric should be marked normal and report sample/missing diagnostics.",
        {"metric": normal_metric},
        normal_flag is True and normal_meta_ok,
        "high" if normal_flag is True and normal_meta_ok else "medium",
        "Shapiro-Wilk on sampled normal data with injected missing values.",
    ))

    corr_df = pd.DataFrame({
        "x": np.arange(160),
        "y": np.arange(160) * 0.6 + rng.normal(0, 25, 160),
        "noise": rng.normal(0, 1, 160),
    })
    corr_stats = run_statistical_tests(corr_df)
    corr_ids = [item["id"] for item in corr_stats["insights"] if item.get("category") == "Correlation"]
    corr_hit = any("x" in item and "y" in item for item in corr_ids)
    corr_meta_ok = all(
        "q_value" in item and "effect_size" in item and "quality" in item
        for item in corr_stats["insights"]
        if item.get("category") == "Correlation"
    )
    results.append(_scenario_result(
        "Spearman correlation",
        "x/y pair should be surfaced with FDR, effect size, and quality metadata.",
        {"correlation_ids": corr_ids[:3], "metadata_present": corr_meta_ok},
        corr_hit and corr_meta_ok,
        "high" if corr_hit else "medium",
        "Monotonic synthetic relationship with controlled noise.",
    ))

    group_df = pd.DataFrame({
        "segment": ["A"] * 70 + ["B"] * 70 + ["C"] * 70,
        "value": np.concatenate([
            rng.normal(50, 5, 70),
            rng.normal(65, 5, 70),
            rng.normal(82, 5, 70),
        ]),
    })
    group_stats = run_statistical_tests(group_df)
    group_items = [
        item
        for item in group_stats["insights"]
        if item.get("category") in ("Group Variance", "Multivariate Impact")
    ]
    group_hit = any(
        item.get("significant")
        for item in group_items
    )
    group_meta_ok = any(
        "effect_size" in item
        and "quality" in item
        and item.get("posthoc_pairs")
        for item in group_items
    )
    group_charts = generate_heuristic_charts(group_df, stat_tests=group_stats)
    group_chart_types = [chart["type"] for chart in group_charts]
    results.append(_scenario_result(
        "Group difference with post-hoc",
        "Distinct category distributions should be significant, chartable, and include pairwise post-hoc detail.",
        {"group_items": group_items[:2], "metadata_present": group_meta_ok, "chart_types": group_chart_types},
        group_hit and group_meta_ok and ("boxplot" in group_chart_types or "bar" in group_chart_types),
        "high" if group_hit else "medium",
        "Three-category Kruskal-Wallis scenario.",
    ))

    cat_df = pd.DataFrame({
        "channel": ["web"] * 60 + ["store"] * 60,
        "status": ["paid"] * 50 + ["trial"] * 10 + ["paid"] * 10 + ["trial"] * 50,
    })
    cat_stats = run_statistical_tests(cat_df)
    cat_items = [item for item in cat_stats["insights"] if item.get("category") == "Categorical Dependency"]
    cat_hit = any(item.get("significant") for item in cat_items)
    cat_meta_ok = any(
        "q_value" in item
        and "effect_size" in item
        and "quality" in item
        and item.get("method") == "chi_square"
        and "expected_min" in item
        and "sparse_cell_ratio" in item
        for item in cat_items
    )
    cat_charts = generate_heuristic_charts(cat_df, stat_tests=cat_stats)
    stacked_hit = any(chart.get("isStacked") for chart in cat_charts)
    results.append(_scenario_result(
        "Categorical dependency",
        "Channel/status dependency should produce significant Chi-Square/Cramer's V.",
        {"cat_items": cat_items[:2], "metadata_present": cat_meta_ok, "stacked_chart": stacked_hit},
        cat_hit and cat_meta_ok and stacked_hit,
        "medium" if cat_hit else "low",
        "Strong dependency is present, but chart generation is intentionally thresholded.",
    ))

    sparse_df = pd.DataFrame({
        "treatment": ["control"] * 12 + ["variant"] * 12,
        "converted": ["no"] * 11 + ["yes"] + ["no"] * 4 + ["yes"] * 8,
    })
    sparse_stats = run_statistical_tests(sparse_df)
    sparse_items = [
        item
        for item in sparse_stats["insights"]
        if item.get("category") == "Categorical Dependency"
    ]
    fisher_hit = any(item.get("method") == "fisher_exact" and item.get("sparse_warning") for item in sparse_items)
    results.append(_scenario_result(
        "Sparse 2x2 categorical fallback",
        "Sparse 2x2 dependency should use Fisher exact instead of trusting sparse Chi-Square cells.",
        {"sparse_items": sparse_items},
        fisher_hit,
        "high" if fisher_hit else "medium",
        "Small 2x2 contingency table with low expected cell counts.",
    ))

    trend_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=24, freq="ME"),
        "sales": np.linspace(100, 220, 24) + rng.normal(0, 8, 24),
    })
    forecast = run_forecasting(trend_df)
    first_forecast = forecast[0] if isinstance(forecast, list) and forecast else {}
    backtest = first_forecast.get("backtest") or {} if isinstance(first_forecast, dict) else {}
    forecast_ok = (
        isinstance(forecast, list)
        and len(forecast) > 0
        and "forecast" in first_forecast
        and {"mae", "rmse", "mape"}.issubset(backtest.keys())
    )
    results.append(_scenario_result(
        "Forecasting backtest",
        "Clear monthly trend should return forecast chart data and holdout error metrics.",
        {"forecast_count": len(forecast) if isinstance(forecast, list) else 0, "first": first_forecast if forecast_ok else forecast},
        forecast_ok,
        "medium",
        "Forecast confidence depends on model fit and warning flags.",
    ))

    cluster_df = pd.DataFrame({
        "feature_a": np.concatenate([rng.normal(0, 0.5, 60), rng.normal(5, 0.5, 60)]),
        "feature_b": np.concatenate([rng.normal(0, 0.5, 60), rng.normal(5, 0.5, 60)]),
    })
    cluster = run_clustering(cluster_df, selected_cols=["feature_a", "feature_b"])
    cluster_ok = (
        isinstance(cluster, dict)
        and cluster.get("type") == "clustering"
        and cluster.get("optimal_k", 0) >= 2
        and cluster.get("silhouette_score") is not None
        and cluster.get("davies_bouldin_score") is not None
    )
    results.append(_scenario_result(
        "Clustering quality metrics",
        "Separated synthetic groups should produce at least two clusters with quality metrics.",
        {
            "optimal_k": cluster.get("optimal_k") if isinstance(cluster, dict) else None,
            "silhouette_score": cluster.get("silhouette_score") if isinstance(cluster, dict) else None,
            "davies_bouldin_score": cluster.get("davies_bouldin_score") if isinstance(cluster, dict) else None,
        },
        cluster_ok,
        "medium",
        "Elbow method can choose more than two clusters depending on variance.",
    ))

    customer_df = pd.DataFrame({
        "customer_id": [f"C{i:03d}" for i in range(120)],
        "department": np.repeat(["Retail", "Enterprise", "Online"], 40),
        "purchase_amount": np.concatenate([rng.normal(120, 10, 40), rng.normal(850, 60, 40), rng.normal(420, 35, 40)]),
        "recency_days": np.concatenate([rng.normal(90, 8, 40), rng.normal(12, 3, 40), rng.normal(45, 6, 40)]),
        "churn_rate": np.concatenate([rng.normal(0.35, 0.04, 40), rng.normal(0.05, 0.02, 40), rng.normal(0.16, 0.03, 40)]),
    })
    customer_cluster = run_clustering(
        customer_df,
        selected_cols=["purchase_amount", "recency_days", "churn_rate"],
    )
    customer_profiles = customer_cluster.get("cluster_profiles", []) if isinstance(customer_cluster, dict) else []
    customer_names = [profile.get("cluster_name") for profile in customer_profiles]
    category_columns = {
        category.get("column")
        for profile in customer_profiles
        for category in profile.get("top_categories", [])
    }
    customer_cluster_ok = (
        isinstance(customer_cluster, dict)
        and customer_cluster.get("type") == "clustering"
        and customer_cluster.get("title", "").startswith("M\u00fc\u015fteri Segmentasyonu")
        and len(customer_names) == len(set(customer_names))
        and "department" in category_columns
        and any(
            item.get("column") == "churn_rate" and item.get("lower_is_better")
            for profile in customer_profiles
            for item in profile.get("feature_rankings", [])
        )
    )
    results.append(_scenario_result(
        "Clustering domain naming contract",
        "Customer clustering should share domain detection with naming, keep names unique, and surface common categories.",
        {
            "title": customer_cluster.get("title") if isinstance(customer_cluster, dict) else None,
            "cluster_names": customer_names,
            "category_columns": sorted(category_columns),
        },
        customer_cluster_ok,
        "medium",
        "Domain labels remain deterministic; LLM fallback is intentionally not part of this contract.",
    ))

    return {
        "summary": {
            "passed": sum(1 for item in results if item["passed"]),
            "total": len(results),
        },
        "results": results,
    }


if __name__ == "__main__":
    print(json.dumps(evaluate_statistical_engine(), ensure_ascii=False, indent=2))
