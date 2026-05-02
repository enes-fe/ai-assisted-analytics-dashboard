"""
services/chart_generator.py
─────────────────────────────────────────────────────────────────────────
Heuristic chart generation engine.

Analyses a DataFrame and produces a curated deck of chart configs
without any LLM interaction. Default output is intentionally compact.
Advanced statistical enrichments are opt-in. Decision logic:
  1. Histograms for non-trivial numeric distributions
  2. Pie / bar for categorical columns
  3. Correlation table when ≥ 3 numeric columns with strong relationships
  4. Line / area trends for datetime × numeric pairs
  5. Boxplots for statistically significant cat × num group differences
  6. Scatter plots for significant num × num correlations
  7. Stacked bar for significant cat × cat dependencies (Cramér's V)
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats

from .utils import sanitize_for_json, clean_string, is_id_column, is_tautology, format_col_name, select_label_column
from .kpi_engine import should_skip_histogram
from .column_profiler import get_column_profile


def generate_heuristic_charts(
    df: pd.DataFrame,
    stat_tests: dict = None,
    cramers_v_threshold: float = 0.3,
    advanced_stats: bool | None = None,
    max_charts: int = 6,
) -> list:
    """
    Returns a JSON-serialisable list of chart config dicts.

    Parameters
    ----------
    df : pd.DataFrame
        The source dataset.
    stat_tests : dict, optional
        Output of stat_engine.run_statistical_tests().  If None, no
        statistical enrichment is applied (no boxplots, no regression lines).
    cramers_v_threshold : float
        Minimum Cramér's V to generate a stacked-bar chart.
    """
    if advanced_stats is None:
        advanced_stats = bool(stat_tests and stat_tests.get("insights"))
    if stat_tests is None:
        stat_tests = {"insights": [], "normality": {}}

    charts: list = []
    relational_charts: list = []

    cols = df.columns.tolist()

    # ── Clean string columns for display ──────────────────────────────────────
    temp_df = df.copy()
    for col in temp_df.select_dtypes(include=["object"]):
        temp_df[col] = temp_df[col].apply(clean_string)

    profiles = {col: get_column_profile(temp_df, col) for col in cols}

    # ── Drop columns with > 50 % nulls ────────────────────────────────────────
    valid_cols = [c for c in cols if df[c].isna().sum() / len(df) <= 0.5]
    cols = valid_cols
    profiles = {c: profiles[c] for c in valid_cols}

    # ── Column type buckets ───────────────────────────────────────────────────
    num_cols = [
        c for c, p in profiles.items()
        if p["type"] in ("continuous_numeric", "discrete_numeric")
        and not is_id_column(c, temp_df[c])
        and temp_df[c].dropna().nunique() > 2  # Exclude binary (0/1) flag columns
    ]
    cat_cols = [
        c for c, p in profiles.items()
        if p["type"] == "categorical" and not is_id_column(c, temp_df[c]) and p.get("cardinality", 0) <= 15
    ]
    date_cols = [
        c for c, p in profiles.items()
        if p["type"] == "datetime" and not is_id_column(c, temp_df[c])
    ]

    print(f"[CHART] num_cols ({len(num_cols)}): {num_cols[:5]}")
    print(f"[CHART] cat_cols: {cat_cols}")

    # ─────────────────────────────────────────────────────────────────────────
    # 1. HISTOGRAMS (numeric distributions)
    # ─────────────────────────────────────────────────────────────────────────
    for col in num_cols:
        p = profiles[col]
        if should_skip_histogram(col, p):
            continue

        series_clean = df[col].dropna()
        mean_val = float(series_clean.mean()) if not series_clean.empty else 0
        median_val = float(series_clean.median()) if not series_clean.empty else 0

        is_normal = None
        normality_note = None
        if advanced_stats:
            normality_result = stat_tests.get("normality", {}).get(col, {})
            is_normal = normality_result.get("is_normal", False)
            if "is_normal" not in normality_result:
                is_normal = abs(p.get("skewness", 0)) <= 0.5
            normality_note = "Normal dağılım" if is_normal else "Normal dağılım değil; medyan daha temsil edici olabilir."

        insight = "Bu grafik, değerlerin dağılımını özetler."
        if p.get("outlier_ratio", 0) > 0.05:
            insight = f"Aykırı değer oranı yüksek görünüyor ({p['outlier_ratio']:.1%}); dağılım temkinli okunmalıdır."
        elif abs(p.get("skewness", 0)) > 0.5:
            insight = f"Dağılım belirgin şekilde asimetrik görünüyor (skewness={p['skewness']:.2f})."

        try:
            counts, bin_edges = np.histogram(series_clean, bins=15)
            hist_data = [{"range": f"{bin_edges[i]:.1f}", "count": int(counts[i])} for i in range(len(counts))]
            chart = {
                "id": f"hist-{col}",
                "type": "bar",
                "title": f"{format_col_name(col)} - Dağılım Analizi",
                "xAxisKey": "range",
                "series": [{"key": "count"}],
                "insight": insight,
                "chartData": hist_data,
            }
            if advanced_stats:
                chart.update({
                    "isNormal": is_normal,
                    "normalityNote": normality_note,
                    "mean": mean_val,
                    "median": median_val,
                })
            charts.append(chart)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # 2. CATEGORICAL DISTRIBUTION (pie / bar)
    # ─────────────────────────────────────────────────────────────────────────
    for col in cat_cols:
        p = profiles.get(col)
        if p is None or p.get("dominant_flag") or is_id_column(col, temp_df[col]):
            continue
        if p.get("cardinality", 0) > 15:
            continue

        col_low = col.lower()
        counts = temp_df[col].value_counts()
        if counts.empty:
            continue

        if any(kw in col_low for kw in ["score", "rating", "puan"]):
            counts = counts.loc[[idx for idx in counts.index if str(idx).isdigit() or len(str(idx)) <= 3]]
            if counts.empty:
                continue

        pie_count = sum(1 for c in charts if c.get("type") == "pie")
        use_pie = p["cardinality"] <= 5 and pie_count < 2

        if use_pie:
            data = counts.reset_index()
            data.columns = [col, "count"]
            charts.append({
                "id": f"pie-{col}",
                "type": "pie",
                "title": f"Bileşim: {col}",
                "xAxisKey": col,
                "series": [{"key": "count"}],
                "insight": f"Bu grafik, {format_col_name(col)} kategorilerindeki kayıt dağılımını gösterir.",
                "chartData": data.fillna(0).to_dict(orient="records"),
            })
        else:
            actual_n = min(len(counts), 5)
            topN = counts.head(actual_n)
            total_cat_sum = counts.sum()
            top3_sum = topN.head(3).sum()
            top3_pct = (top3_sum / total_cat_sum) * 100 if total_cat_sum > 0 else 0

            data = topN.reset_index()
            data.columns = [col, "count"]

            if top3_pct > 70:
                insight = f"İlk 3 kategori dağılımın %{top3_pct:.1f}'ini oluşturuyor; sonuçlar bu yoğunlaşma dikkate alınarak okunmalıdır."
            elif top3_pct > 40:
                insight = f"İlk 3 kategori toplam dağılımın %{top3_pct:.1f}'ini temsil ediyor."
            else:
                insight = "Kategoriler görece dengeli dağılıyor."

            bar_count = sum(1 for c in charts if c.get("type") == "bar")
            charts.append({
                "id": f"bar-{col}",
                "type": "bar",
                "title": f"Dağılım Analizi: {col}",
                "xAxisKey": col,
                "layout": "vertical" if bar_count % 2 != 0 else "horizontal",
                "series": [{"key": "count"}],
                "insight": insight,
                "chartData": data.fillna(0).to_dict(orient="records"),
            })

    # ─────────────────────────────────────────────────────────────────────────
    # 3. CORRELATION TABLE (num × num, Spearman)
    # ─────────────────────────────────────────────────────────────────────────
    if advanced_stats and len(num_cols) >= 3:
        try:
            corr_mtx = temp_df[num_cols].corr(method="spearman").fillna(0)
            heatmap_data = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    c1, c2 = num_cols[i], num_cols[j]
                    corr_val = round(corr_mtx.loc[c1, c2], 2)
                    heatmap_data.append({
                        "Var A": c1,
                        "Var B": c2,
                        "Correlation": corr_val,
                        "Strength": "Güçlü" if abs(corr_val) >= 0.5 else "Orta" if abs(corr_val) >= 0.3 else "Zayıf",
                    })
            heatmap_data.sort(key=lambda x: abs(x["Correlation"]), reverse=True)
            if sum(1 for d in heatmap_data if abs(d["Correlation"]) >= 0.5) >= 1:
                relational_charts.append({
                    "id": "correlation-matrix",
                    "type": "table",
                    "title": "Değişken İlişki Gücü",
                    "xAxisKey": "Var A",
                    "series": [{"key": "Correlation"}],
                    "insight": "Spearman korelasyon gücüne göre sıralandı. Yüksek mutlak değerler istatistiksel ilişkiye işaret eder, nedensellik göstermez.",
                    "chartData": heatmap_data[:12],
                })
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # 4. TREND LINES (datetime × numeric)
    # ─────────────────────────────────────────────────────────────────────────
    if date_cols and num_cols:
        d_col = date_cols[0]
        for n_col in num_cols[:2]:
            try:
                t_df = temp_df.copy()
                t_df[d_col] = pd.to_datetime(t_df[d_col])
                days = (t_df[d_col].max() - t_df[d_col].min()).days
                resample_rule = "Y" if days > 730 else "ME" if days > 30 else "D"

                line_data = t_df.set_index(d_col)[n_col].resample(resample_rule).mean().reset_index()
                line_data[d_col] = line_data[d_col].dt.strftime("%Y-%m-%d")
                relational_charts.append({
                    "id": f"line-{d_col}-{n_col}",
                    "type": "line",
                    "title": f"{format_col_name(n_col)} - Zaman İçinde Trend",
                    "xAxisKey": d_col,
                    "series": [{"key": n_col}],
                    "insight": f"Bu grafik, {format_col_name(n_col)} değerinin {format_col_name(d_col)} boyunca değişimini özetler.",
                    "chartData": line_data.fillna(0).to_dict(orient="records"),
                })
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # 5. BOXPLOTS / GROUP BARS (categorical × numeric)
    # ─────────────────────────────────────────────────────────────────────────
    if advanced_stats and cat_cols and num_cols:
        try:
            fallback_candidates: list = []
            for cat in cat_cols[:3]:
                for num in num_cols[:4]:
                    significant = False
                    summary = ""

                    for insight_item in stat_tests.get("insights", []):
                        if insight_item.get("category") == "Group Variance":
                            if cat in insight_item.get("categorical_col", "") and num in insight_item.get("numeric_col", ""):
                                if insight_item.get("significant"):
                                    significant = True
                                    summary = insight_item["summary"]
                                break

                    if significant:
                        box_data: list = []
                        groups = df[cat].dropna().unique()
                        if len(groups) > 10:
                            groups = groups[:10]

                        for val in groups:
                            subset = df[df[cat] == val][num].dropna()
                            if len(subset) < 3:
                                continue
                            q1 = float(subset.quantile(0.25))
                            q3 = float(subset.quantile(0.75))
                            box_data.append({
                                "group": str(val),
                                "min": float(subset.min()),
                                "q1": q1,
                                "median": float(subset.median()),
                                "q3": q3,
                                "max": float(subset.max()),
                            })

                        if box_data:
                            relational_charts.append({
                                "id": f"box-{cat}-{num}",
                                "type": "boxplot",
                                "title": f"{format_col_name(num)} Dağılımı - {format_col_name(cat)}",
                                "xAxisKey": "group",
                                "series": [{"key": "median"}],
                                "insight": summary,
                                "chartData": box_data,
                                "effect_size": insight_item.get("effect_size"),
                                "effect_label": insight_item.get("effect_label"),
                                "quality": insight_item.get("quality"),
                            })
                    else:
                        # Limit non-significant cat×num bar charts to avoid chart explosion
                        non_sig_count = sum(1 for c in charts if c.get("id", "").startswith("bar-") and "-" in c["id"])
                        if non_sig_count >= 3:
                            continue
                        try:
                            # Pick the num col with highest variance for this cat — most informative
                            data = df.groupby(cat)[num].mean().nlargest(5).reset_index()
                            data.columns = [cat, num]
                            charts.append({
                                "id": f"bar-{cat}-{num}",
                                "type": "bar",
                                "title": f"Ortalama {format_col_name(num)} - {format_col_name(cat)} Bazında",
                                "xAxisKey": cat,
                                "series": [{"key": num}],
                                "insight": f"Bu grafik, {format_col_name(cat)} kategorileri arasında ortalama {format_col_name(num)} değerlerini karşılaştırır.",
                                "chartData": data.fillna(0).to_dict(orient="records"),
                            })
                        except Exception:
                            pass
        except Exception as e:
            print(f"[CHART] Cat-Num error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. SCATTER PLOTS (num × num correlation)
    # ─────────────────────────────────────────────────────────────────────────
    pair_scores: list = []
    seen_pairs: set = set()

    if len(num_cols) >= 2:
        try:
            sample_for_corr = temp_df[num_cols].dropna()
            if len(sample_for_corr) > 1000:
                sample_for_corr = sample_for_corr.sample(1000, random_state=42)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr_mtx_scatter = sample_for_corr.corr(method="spearman")

            candidate_pairs: list = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    n1, n2 = num_cols[i], num_cols[j]
                    if (n1, n2) in seen_pairs or (n2, n1) in seen_pairs:
                        continue
                    if is_tautology(n1, n2):
                        continue
                    r = corr_mtx_scatter.loc[n1, n2]
                    if np.isnan(r) or abs(r) >= 0.95:
                        continue
                    candidate_pairs.append((n1, n2, r))

            candidate_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            slots_remaining = max(0, 3 - len(pair_scores))

            for n1, n2, r in candidate_pairs[: slots_remaining * 2]:
                if len(pair_scores) >= 3:
                    break
                try:
                    c_df = temp_df[[n1, n2]].dropna()
                    if len(c_df) < 10:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _, p_val = stats.spearmanr(c_df[n1], c_df[n2])
                    pair_scores.append((n1, n2, r, p_val, c_df))
                except Exception:
                    continue
        except Exception as e:
            print(f"[CHART] Scatter matrix error: {e}")

    for n1, n2, corr, p_val, c_df in pair_scores[:3]:
        if abs(corr) >= 0.95:
            continue

        strength = "güçlü" if abs(corr) >= 0.5 and p_val < 0.05 else "orta" if abs(corr) >= 0.3 else "zayıf"

        show_regression = False
        r_squared = None
        coeffs = None

        if advanced_stats:
            for insight_item in stat_tests.get("insights", []):
                if insight_item.get("category") == "Correlation":
                    if n1 in insight_item["id"] and n2 in insight_item["id"]:
                        if insight_item.get("significant") and abs(insight_item.get("r", 0)) >= 0.5:
                            show_regression = True
                            r_squared = insight_item.get("r", 0) ** 2
                            try:
                                m, b = np.polyfit(c_df[n1], c_df[n2], 1)
                                coeffs = {"slope": float(m), "intercept": float(b)}
                            except Exception:
                                pass
                        break

        sample_size = min(1000, len(c_df))
        sample_df = c_df.sample(sample_size, random_state=42).copy()
        label_col = select_label_column(temp_df, exclude={n1, n2})
        sample_df["__row"] = sample_df.index.astype(int) + 1
        if label_col:
            sample_df[label_col] = temp_df.loc[sample_df.index, label_col].astype(str)
            sample_df["__label"] = sample_df[label_col]
        scatter_data = sample_df.fillna(0).to_dict(orient="records")

        relational_charts.append({
            "id": f"scatter-{n1}-{n2}",
            "type": "scatter",
            "title": f"{format_col_name(n1)} ve {format_col_name(n2)} - Korelasyon",
            "xAxisKey": n1,
            "series": [{"key": n2}],
            "insight": (
                f"Bu dağılım, {format_col_name(n1)} ve {format_col_name(n2)} arasındaki ilişkiyi "
                f"görselleştirir; nedensellik göstermez. Korelasyon: {corr:.2f} ({strength})."
            ),
            "chartData": scatter_data,
            "labelKey": "__label" if label_col else None,
            "labelName": label_col,
            "isHexbin": len(c_df) > 10000,
            "showRegressionLine": show_regression,
            "rSquared": r_squared,
            "regressionCoeffs": coeffs,
            **({
                "effect_size": abs(corr),
                "effect_label": "large" if abs(corr) >= 0.5 else "medium" if abs(corr) >= 0.3 else "small",
            } if advanced_stats else {}),
        })

    # ─────────────────────────────────────────────────────────────────────────
    # 7. STACKED BARS (categorical × categorical, Cramér's V)
    # ─────────────────────────────────────────────────────────────────────────
    if advanced_stats and len(cat_cols) >= 2:
        for i in range(len(cat_cols)):
            for j in range(i + 1, len(cat_cols)):
                c1, c2 = cat_cols[i], cat_cols[j]
                if is_tautology(c1, c2):
                    continue
                if df[c1].nunique() > 10 or df[c2].nunique() > 10:
                    continue

                for insight_item in stat_tests.get("insights", []):
                    if insight_item.get("category") == "Categorical Dependency":
                        if c1 in insight_item["id"] and c2 in insight_item["id"]:
                            cv = insight_item.get("cv", 0)
                            if cv == 0:
                                try:
                                    if "Cramér's V=" in insight_item["technical"]:
                                        cv = float(insight_item["technical"].split("Cramér's V=")[1])
                                except Exception:
                                    pass

                            if insight_item.get("significant") and cv >= cramers_v_threshold:
                                crosstab = pd.crosstab(df[c1], df[c2], normalize="index") * 100
                                stacked_data = crosstab.reset_index().to_dict(orient="records")
                                relational_charts.append({
                                    "id": f"stacked-{c1}-{c2}",
                                    "type": "bar",
                                    "title": f"{format_col_name(c1)} ve {format_col_name(c2)} İlişkisi",
                                    "xAxisKey": c1,
                                    "layout": "vertical",
                                    "isStacked": True,
                                    "series": [{"key": str(k)} for k in crosstab.columns],
                                    "insight": insight_item["summary"],
                                    "chartData": stacked_data,
                                    "cramersV": cv,
                                    "effect_size": insight_item.get("effect_size", cv),
                                    "effect_label": insight_item.get("effect_label"),
                                    "quality": insight_item.get("quality"),
                                })
                            break

    print(f"[CHART] relational={len(relational_charts)} single={len(charts)}")

    final_deck = relational_charts + charts[:2]
    return sanitize_for_json(final_deck[: max(1, int(max_charts or 6))])
