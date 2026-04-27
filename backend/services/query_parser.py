"""
services/query_parser.py
─────────────────────────────────────────────────────────────────────────
Natural-language query parser that maps a free-text prompt to a chart
config.  Uses fuzzy column matching and keyword-based chart type
detection — no external LLM required.
"""

import re
import uuid
import numpy as np

from .utils import sanitize_for_json, is_id_column, format_col_name, get_similarity


def simulate_rearchitecting(df, prompt: str) -> list:
    """
    Converts a free-text prompt into one or more chart config dicts.

    Supports chart types: bar, pie, line, scatter, area.
    Uses 'by X' pattern extraction and fuzzy column matching.
    """
    prompt_lower = prompt.lower()

    all_cols = [c for c in df.columns if not is_id_column(c, df[c])]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if not cat_cols:
        cat_cols = [c for c in num_cols if df[c].nunique() < 20]
    if not num_cols:
        num_cols = all_cols

    # ── Chart type detection ──────────────────────────────────────────────────
    chart_type = "bar"
    if any(k in prompt_lower for k in ["pie", "cluster", "dağılım"]):
        chart_type = "pie"
    elif any(k in prompt_lower for k in ["trend", "line", "çizgi"]):
        chart_type = "line"
    elif any(k in prompt_lower for k in ["scatter", "ilişki", "korelasyon"]):
        chart_type = "scatter"
    elif "area" in prompt_lower:
        chart_type = "area"

    # ── Fuzzy column matcher ──────────────────────────────────────────────────
    def fuzzy_match(col_list, text):
        text_tokens = set(re.split(r"\s+|_|-", text.lower()))
        best_col = None
        best_score = -1

        for col in col_list:
            col_lower = col.lower()
            col_tokens = set(re.split(r"\s+|_|-", col_lower))

            exact_name_score = 20 if col_lower in text.lower() else 0
            exact_matches = len(col_tokens & text_tokens)
            similarity = max(
                [get_similarity(ct, tt) for ct in col_tokens for tt in text_tokens] + [0]
            )
            score = exact_name_score + (exact_matches * 5) + similarity

            if is_id_column(col, df[col]) and exact_name_score == 0 and not any("id" in t for t in text_tokens):
                score -= 10

            if score > best_score:
                best_score = score
                best_col = col

        return best_col, best_score

    # ── "by X" pattern extraction ─────────────────────────────────────────────
    by_match = re.search(r"\bby\s+([\w\s]+?)(?:\s+as\b|\s+as\s+pie|\s+as\s+bar|\s+chart|$)", prompt_lower)
    if by_match:
        by_text = by_match.group(1).strip()
        explicit_cat, explicit_cat_score = fuzzy_match(cat_cols, by_text)
        match_cat = explicit_cat if explicit_cat_score > 0 else None
    else:
        match_cat = None

    match_num, _ = fuzzy_match(num_cols, prompt_lower)
    if match_cat is None:
        match_cat, _ = fuzzy_match(cat_cols, prompt_lower)

    if not match_num:
        match_num = num_cols[0] if num_cols else all_cols[0]
    if not match_cat:
        match_cat = cat_cols[0] if cat_cols else all_cols[0]

    print(f"[CHAT] chart_type={chart_type}, match_num={match_num}, match_cat={match_cat}")

    try:
        temp_df = df.copy()
        agg_func = "none"

        if chart_type == "scatter":
            num_cols_other = [c for c in num_cols if c != match_num]
            second_num = num_cols_other[0] if num_cols_other else match_num
            plot_df = temp_df[[match_num, second_num]].dropna().head(100)
            chart_data = sanitize_for_json(plot_df.to_dict(orient="records"))
            xAxisKey = match_num
            series = [{"key": second_num}]
        else:
            # Pie always sums; explicit keywords also force sum
            if chart_type == "pie" or any(w in prompt_lower for w in ["total", "toplam", "sum"]):
                agg_func = "sum"
            else:
                agg_func = "mean"

            print(f"[CHAT] agg_func={agg_func}")
            agg_df = temp_df.groupby(match_cat)[match_num].agg(agg_func).reset_index()

            top_n = 10
            match_top = re.search(r"(top|ilk|en yüksek|en fazla)\s+(\d+)", prompt_lower)
            if match_top:
                top_n = int(match_top.group(2))

            if chart_type == "pie":
                agg_df = agg_df.sort_values(match_num, ascending=False)
            elif any(w in prompt_lower for w in ["en düşük", "bottom", "en az"]):
                agg_df = agg_df.nsmallest(top_n, match_num)
            else:
                agg_df = agg_df.nlargest(top_n, match_num)

            chart_data = sanitize_for_json(agg_df.to_dict(orient="records"))
            xAxisKey = match_cat
            series = [{"key": match_num}]

        agg_label = (
            "toplanarak" if agg_func == "sum"
            else "ortalaması alınarak" if agg_func == "mean"
            else "hesaplanarak"
        )
        agg_title_prefix = (
            "Toplam" if agg_func == "sum"
            else "Ortalama" if agg_func == "mean"
            else ""
        )

        return [{
            "id": f"ai-{uuid.uuid4().hex[:6]}",
            "type": chart_type,
            "title": f"AI Analizi: {agg_title_prefix} {format_col_name(match_num)} — {format_col_name(match_cat)} Bazında".strip(),
            "xAxisKey": xAxisKey,
            "series": series,
            "chartData": chart_data,
            "insight": f"'{prompt}' sorgunuza istinaden veri {agg_label} görselleştirildi.",
        }]

    except Exception as e:
        print(f"[CHAT] Simulation error: {e}")
        return [{
            "id": "error-chart",
            "type": "bar",
            "title": "Veri Uyumsuzluğu",
            "xAxisKey": match_cat,
            "series": [{"key": match_num}],
            "insight": f"İstediğiniz kolonlar ({match_cat}, {match_num}) bulundu ancak matematiksel olarak gruplanamadı.",
            "chartData": [],
        }]
