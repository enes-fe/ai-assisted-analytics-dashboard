"""
Statistical testing layer for exploratory dashboard insights.

The engine favors statistically honest "signal" language over causal claims.
Each insight carries sample size, missing-row impact, FDR-adjusted q-value
where applicable, effect size, effect label, and a compact quality rating.
"""

import warnings
from collections import defaultdict
from itertools import groupby as itertools_groupby

import numpy as np
import pandas as pd
from scipy import stats

from .utils import sanitize_for_json, is_id_column, format_col_name


_PRIORITY_KW = ["profit", "sales", "revenue", "price", "unit", "cost", "target", "operating"]


def effect_label(value: float) -> str:
    value = abs(value)
    if value >= 0.5:
        return "large"
    if value >= 0.3:
        return "medium"
    if value >= 0.1:
        return "small"
    return "negligible"


def insight_quality(effect_size: float, q_value: float | None, sample_size: int, missing_ratio: float = 0.0) -> str:
    effect = abs(effect_size)
    q = 1.0 if q_value is None else q_value
    missing_penalty = missing_ratio > 0.35
    if sample_size >= 30 and q < 0.01 and effect >= 0.5 and not missing_penalty:
        return "high"
    if sample_size >= 20 and q < 0.05 and effect >= 0.3:
        return "medium" if not missing_penalty else "low"
    if q < 0.1 and effect >= 0.1:
        return "low"
    return "exploratory"


def missing_meta(df: pd.DataFrame, cols: list[str], sample_size: int) -> dict:
    total_rows = len(df)
    missing_dropped_count = max(0, total_rows - sample_size)
    missing_ratio = missing_dropped_count / total_rows if total_rows else 0
    return {
        "sample_size": int(sample_size),
        "missing_dropped_count": int(missing_dropped_count),
        "missing_ratio": float(missing_ratio),
        "analysis_columns": cols,
    }


def apply_fdr_correction(tests: list) -> list:
    """Applies Benjamini-Hochberg FDR correction within one test family."""
    indexed = [
        (idx, float(test["p_value"]))
        for idx, test in enumerate(tests)
        if test.get("p_value") is not None and not pd.isna(test.get("p_value"))
    ]
    m = len(indexed)
    if m == 0:
        return tests

    indexed.sort(key=lambda item: item[1], reverse=True)
    previous_q = 1.0
    for rank_from_end, (idx, p_value) in enumerate(indexed, start=1):
        rank = m - rank_from_end + 1
        q_value = min(previous_q, p_value * m / rank)
        previous_q = q_value
        tests[idx]["q_value"] = float(min(q_value, 1.0))

    for test in tests:
        if "q_value" in test:
            test["significant"] = bool(test["q_value"] < 0.05)
            test["significance_text"] = "anlamlı" if test["significant"] else "anlamlı değil"
    return tests


def classify_distribution(series: pd.Series, shapiro_p: float | None) -> dict:
    clean = series.dropna()
    if len(clean) < 8:
        return {
            "shape": "insufficient_data",
            "is_normal": False,
            "skewness": None,
            "kurtosis": None,
            "outlier_ratio": None,
        }

    skewness = float(stats.skew(clean))
    kurtosis = float(stats.kurtosis(clean, fisher=True))
    q1 = clean.quantile(0.25)
    q3 = clean.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        outlier_ratio = 0.0
    else:
        outlier_ratio = float(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).mean())

    is_normal = bool(shapiro_p is not None and shapiro_p > 0.05 and abs(skewness) < 0.75 and outlier_ratio < 0.05)
    if is_normal:
        shape = "approximately_normal"
    elif outlier_ratio >= 0.08:
        shape = "outlier_sensitive"
    elif abs(kurtosis) >= 2:
        shape = "heavy_tailed"
    elif abs(skewness) >= 0.75:
        shape = "skewed"
    else:
        shape = "non_normal"

    return {
        "shape": shape,
        "is_normal": is_normal,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "outlier_ratio": outlier_ratio,
    }


def human_readable_correlation(col_a: str, col_b: str, r: float, p: float, method: str) -> dict:
    direction = "aynı yönde" if r > 0 else "ters yönde"
    if abs(r) >= 0.7:
        strength_text = "çok güçlü bir istatistiksel sinyal"
    elif abs(r) >= 0.5:
        strength_text = "güçlü bir istatistiksel sinyal"
    elif abs(r) >= 0.3:
        strength_text = "orta düzey bir istatistiksel sinyal"
    else:
        strength_text = "zayıf bir istatistiksel sinyal"

    return {
        "summary": (
            f"{format_col_name(col_a)} ile {format_col_name(col_b)} arasında {strength_text} var; "
            f"değerler genellikle {direction} hareket ediyor. Bu sonuç nedensellik göstermez."
        ),
        "significance_text": "anlamlı" if p < 0.05 else "anlamlı değil",
        "technical": f"{method} r={r:.2f}, p={p:.4f}",
        "score": (1 - p) * abs(r),
    }


def human_readable_group_test(cat_col: str, num_col: str, p: float, test_name: str, group_count: int) -> dict:
    if p < 0.05:
        summary = (
            f"{format_col_name(cat_col)} grupları arasında {format_col_name(num_col)} açısından "
            "istatistiksel olarak anlamlı bir farklılaşma sinyali var."
        )
    else:
        summary = (
            f"{format_col_name(cat_col)} grupları arasında {format_col_name(num_col)} açısından "
            "güçlü bir farklılaşma sinyali görülmüyor."
        )
    return {
        "summary": summary,
        "significance_text": "anlamlı" if p < 0.05 else "anlamlı değil",
        "technical": f"{test_name}, p={p:.4f}, {group_count} grup",
        "score": 1 - p,
    }


def human_readable_chi_square(col_a: str, col_b: str, p: float, cramers_v: float, method: str) -> dict:
    if p < 0.05:
        summary = f"{format_col_name(col_a)} ile {format_col_name(col_b)} arasında kategorik bağımlılık sinyali var."
    else:
        summary = f"{format_col_name(col_a)} ile {format_col_name(col_b)} için güçlü bir bağımlılık sinyali görülmüyor."
    return {
        "summary": summary,
        "significance_text": "anlamlı" if p < 0.05 else "anlamlı değil",
        "technical": f"{method}, p={p:.4f}, Cramér's V={cramers_v:.2f}",
        "score": (1 - p) * (cramers_v + 0.1),
    }


def priority_score(insight: dict) -> float:
    score = insight.get("score", 0)
    technical_lower = insight.get("technical", "").lower()
    for kw in _PRIORITY_KW:
        if kw in technical_lower:
            score += 0.2
    quality_bonus = {"high": 0.3, "medium": 0.2, "low": 0.1}.get(insight.get("quality"), 0)
    return score + quality_bonus


def group_by_category(group_tests: list) -> list:
    grouped: dict = defaultdict(list)
    by_cat_tests: dict = defaultdict(list)
    for test in group_tests:
        cat = test.get("categorical_col")
        if cat:
            by_cat_tests[cat].append(test)
        if test.get("significant"):
            num = test.get("numeric_col")
            if cat and num:
                grouped[cat].append(num)

    final_insights: list = []
    processed_cats: set = set()

    for cat, nums in grouped.items():
        if len(nums) >= 2:
            source_tests = by_cat_tests.get(cat, [])
            best_effect = max((abs(t.get("effect_size", 0)) for t in source_tests), default=0)
            best_q = min((t.get("q_value", 1) for t in source_tests), default=1)
            sample_size = max((t.get("sample_size", 0) for t in source_tests), default=0)
            missing_ratio = max((t.get("missing_ratio", 0) for t in source_tests), default=0)
            cols_text = ", ".join([format_col_name(n) for n in nums])
            final_insights.append({
                "id": f"grouped-{cat}",
                "type": "grouped_finding",
                "category": "Multivariate Impact",
                "categorical_col": cat,
                "affected_cols": nums,
                "summary": (
                    f"{format_col_name(cat)}, birden fazla metrikle ({cols_text}) "
                    "istatistiksel olarak ilişkili görünen önemli bir segment değişkeni."
                ),
                "significance_text": "çoklu metrik sinyali",
                "technical": f"Kombine analiz, {len(nums)} bağımlı değişken",
                "significant": True,
                "effect_size": best_effect,
                "effect_label": effect_label(best_effect),
                "q_value": best_q,
                "sample_size": sample_size,
                "missing_ratio": missing_ratio,
                "quality": insight_quality(best_effect, best_q, sample_size, missing_ratio),
                "score": 1.5,
            })
            processed_cats.add(cat)

    for test in group_tests:
        if test.get("categorical_col") not in processed_cats:
            final_insights.append(test)

    return final_insights


def _rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    if n1 <= 0 or n2 <= 0:
        return 0.0
    return abs(1 - (2 * u_stat) / (n1 * n2))


def _epsilon_squared(h_stat: float, n: int, k: int) -> float:
    if n <= k:
        return 0.0
    return float(max(0, (h_stat - k + 1) / (n - k)))


def _posthoc_pairwise(subset: pd.DataFrame, cat: str, num: str) -> list:
    groups = [(str(name), group[num].dropna().values) for name, group in subset.groupby(cat)]
    pair_tests = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            name_a, values_a = groups[i]
            name_b, values_b = groups[j]
            if len(values_a) < 5 or len(values_b) < 5:
                continue
            try:
                u_stat, p_value = stats.mannwhitneyu(values_a, values_b)
                effect = _rank_biserial_from_u(float(u_stat), len(values_a), len(values_b))
                pair_tests.append({
                    "group_a": name_a,
                    "group_b": name_b,
                    "p_value": float(p_value),
                    "effect_size": float(effect),
                    "effect_label": effect_label(effect),
                    "median_diff": float(np.median(values_a) - np.median(values_b)),
                    "significant": bool(p_value < 0.05),
                })
            except Exception:
                continue
    apply_fdr_correction(pair_tests)
    pair_tests.sort(key=lambda item: (item.get("q_value", 1), -abs(item.get("effect_size", 0))))
    return pair_tests[:3]


def _categorical_test(contingency: pd.DataFrame) -> tuple[str, float, float, dict]:
    chi2, chi_p, _, expected = stats.chi2_contingency(contingency)
    n = contingency.values.sum()
    min_dim = min(contingency.shape)
    cv = np.sqrt(chi2 / (n * (min_dim - 1))) if n > 0 and min_dim > 1 else 0
    sparse_cell_ratio = float((expected < 5).mean()) if expected.size else 0
    expected_min = float(expected.min()) if expected.size else 0
    method = "chi_square"
    method_label = "Chi-Square"
    p_value = float(chi_p)

    if contingency.shape == (2, 2) and sparse_cell_ratio > 0:
        _, fisher_p = stats.fisher_exact(contingency)
        method = "fisher_exact"
        method_label = "Fisher Exact"
        p_value = float(fisher_p)

    diagnostics = {
        "method_label": method_label,
        "expected_min": expected_min,
        "sparse_cell_ratio": sparse_cell_ratio,
        "sparse_warning": bool(sparse_cell_ratio > 0.2),
    }
    return method, p_value, float(cv), diagnostics


def run_statistical_tests(df: pd.DataFrame) -> dict:
    all_insights: list = []
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if not is_id_column(c, df[c])]
    cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if not is_id_column(c, df[c])]

    normality: dict = {}
    for col in num_cols:
        series = df[col].dropna()
        if len(series) < 3:
            continue
        try:
            sample = series.sample(min(500, len(series)), random_state=42)
            _, p = stats.shapiro(sample)
            classification = classify_distribution(series, float(p))
            normality[col] = {
                **classification,
                "p_value": float(p),
                **missing_meta(df, [col], len(series)),
                "quality": "medium" if len(sample) >= 30 else "low",
            }
        except Exception:
            continue

    correlation_tests: list = []
    if len(num_cols) >= 2:
        try:
            sample_df = df[num_cols].dropna()
            if len(sample_df) > 1000:
                sample_df = sample_df.sample(1000, random_state=42)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr_matrix = sample_df.corr(method="spearman")

            pairs = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    r = corr_matrix.loc[c1, c2]
                    if np.isnan(r):
                        continue
                    pairs.append((c1, c2, r))

            pairs.sort(key=lambda x: abs(x[2]), reverse=True)

            for c1, c2, r in pairs[:10]:
                if abs(r) >= 0.95 and (c1.lower() in c2.lower() or c2.lower() in c1.lower()):
                    continue
                try:
                    pair = df[[c1, c2]].dropna()
                    if len(pair) < 10:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _, p = stats.spearmanr(pair[c1], pair[c2])
                    hr = human_readable_correlation(c1, c2, r, p, "Spearman")
                    near_perfect = abs(r) >= 0.95
                    correlation_tests.append({
                        "id": f"corr-{c1}-{c2}",
                        "category": "Near-Perfect Correlation" if near_perfect else "Correlation",
                        "significant": bool(p < 0.05),
                        "r": float(r),
                        "p_value": float(p),
                        "effect_size": float(abs(r)),
                        "effect_label": effect_label(float(r)),
                        "near_perfect": bool(near_perfect),
                        **missing_meta(df, [c1, c2], len(pair)),
                        **hr,
                    })
                except Exception:
                    continue
        except Exception as e:
            print(f"[STAT] Correlation matrix error: {e}")

    apply_fdr_correction(correlation_tests)
    for test in correlation_tests:
        test["quality"] = insight_quality(test["effect_size"], test.get("q_value"), test["sample_size"], test.get("missing_ratio", 0))
        test["technical"] += f", q={test.get('q_value', 1):.4f}, effect={test['effect_size']:.2f} ({test['effect_label']})"

    strong_corrs = [c for c in correlation_tests if abs(c["r"]) >= 0.5]
    if len(strong_corrs) < 3:
        strong_corrs = sorted(correlation_tests, key=lambda x: abs(x["r"]), reverse=True)[:3]
    all_insights.extend(strong_corrs)

    group_tests: list = []
    for cat in cat_cols:
        unique_count = df[cat].nunique()
        if unique_count < 2 or unique_count > 10:
            continue
        for num in num_cols:
            try:
                subset = df[[cat, num]].dropna()
                if subset.empty:
                    continue
                grouped = subset.groupby(cat)[num]
                groups = [group.values for _, group in grouped if len(group) >= 5]
                if len(groups) < 2:
                    continue
                if len(groups) == 2:
                    stat_value, p = stats.mannwhitneyu(groups[0], groups[1])
                    test_name = "Mann-Whitney U"
                    effect = _rank_biserial_from_u(float(stat_value), len(groups[0]), len(groups[1]))
                    posthoc = []
                else:
                    stat_value, p = stats.kruskal(*groups)
                    test_name = "Kruskal-Wallis"
                    effect = _epsilon_squared(float(stat_value), len(subset), len(groups))
                    posthoc = _posthoc_pairwise(subset, cat, num)
                hr = human_readable_group_test(cat, num, p, test_name, len(groups))
                group_tests.append({
                    "id": f"group-{cat}-{num}",
                    "category": "Group Variance",
                    "categorical_col": cat,
                    "numeric_col": num,
                    "significant": bool(p < 0.05),
                    "p_value": float(p),
                    "effect_size": float(effect),
                    "effect_label": effect_label(float(effect)),
                    "posthoc_pairs": posthoc,
                    **missing_meta(df, [cat, num], len(subset)),
                    **hr,
                })
            except Exception:
                continue

    apply_fdr_correction(group_tests)
    for test in group_tests:
        test["quality"] = insight_quality(test["effect_size"], test.get("q_value"), test["sample_size"], test.get("missing_ratio", 0))
        test["technical"] += f", q={test.get('q_value', 1):.4f}, effect={test['effect_size']:.2f} ({test['effect_label']})"

    group_tests_sorted = sorted(group_tests, key=lambda x: (x["categorical_col"], x.get("q_value", 1)))
    filtered_groups: list = []
    for _, group in itertools_groupby(group_tests_sorted, key=lambda x: x["categorical_col"]):
        filtered_groups.extend(list(group)[:2])
    all_insights.extend(group_by_category(filtered_groups))

    chi_tests: list = []
    for i in range(len(cat_cols)):
        for j in range(i + 1, len(cat_cols)):
            c1, c2 = cat_cols[i], cat_cols[j]
            if df[c1].nunique() > 15 or df[c2].nunique() > 15:
                continue
            try:
                subset = df[[c1, c2]].dropna()
                contingency = pd.crosstab(subset[c1], subset[c2])
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                method, p, cv, diagnostics = _categorical_test(contingency)
                if cv >= 0.3:
                    hr = human_readable_chi_square(c1, c2, p, cv, diagnostics["method_label"])
                    chi_tests.append({
                        "id": f"chi-{c1}-{c2}",
                        "category": "Categorical Dependency",
                        "significant": bool(p < 0.05),
                        "p_value": float(p),
                        "cv": float(cv),
                        "effect_size": float(cv),
                        "effect_label": effect_label(float(cv)),
                        "method": method,
                        **diagnostics,
                        **missing_meta(df, [c1, c2], len(subset)),
                        **hr,
                    })
            except Exception:
                continue

    apply_fdr_correction(chi_tests)
    for test in chi_tests:
        sparse_penalty = 0.2 if test.get("sparse_warning") else 0
        missing_ratio = min(1.0, test.get("missing_ratio", 0) + sparse_penalty)
        test["quality"] = insight_quality(test["effect_size"], test.get("q_value"), test["sample_size"], missing_ratio)
        test["technical"] += f", q={test.get('q_value', 1):.4f}, effect={test['effect_size']:.2f} ({test['effect_label']})"
    all_insights.extend(chi_tests[:5])

    all_insights.sort(key=priority_score, reverse=True)
    return sanitize_for_json({"insights": all_insights[:12], "normality": normality})
