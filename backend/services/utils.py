"""
services/utils.py
─────────────────────────────────────────────────────────────────────────
Shared, stateless helper functions used across the entire service layer.
Nothing here should import from other services modules to prevent circular
imports.
"""

import string
import numpy as np
import pandas as pd
from difflib import SequenceMatcher


# ─── JSON Sanitization ────────────────────────────────────────────────────────

def sanitize_for_json(obj):
    """Recursively replaces NaN, Inf, -Inf with None for JSON compatibility."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj


# ─── String Utilities ─────────────────────────────────────────────────────────

def clean_string(val):
    """Strips junk / non-printable chars from string values."""
    if not isinstance(val, str):
        return val
    allowed = string.printable + "üÜğĞıİşŞçÇöÖ"
    cleaned = "".join(c for c in val if c in allowed)
    if len(cleaned) > 2 and sum(not c.isalnum() for c in cleaned) / len(cleaned) > 0.5:
        return "N/A"
    return cleaned.strip()[:50]


def normalize_col(name: str) -> str:
    """Removes all non-alphanumeric chars and lowercases the string."""
    return "".join(c.lower() for c in str(name) if c.isalnum())


def format_col_name(name: str) -> str:
    """Formats a column name for display (Title Case, no underscores)."""
    if not name:
        return ""
    return str(name).replace("_", " ").title()


def get_similarity(a: str, b: str) -> float:
    """Calculates string similarity ratio between two strings."""
    return SequenceMatcher(None, str(a).lower().strip(), str(b).lower().strip()).ratio()


# ─── Column Identity ──────────────────────────────────────────────────────────

def is_id_column(col_name: str, series: pd.Series) -> bool:
    """Returns True if the column looks like an identifier (UUID, PK, FK …)."""
    name_lower = col_name.lower().replace("_", "").replace(" ", "")

    id_patterns = ["id", "uuid", "pk", "fk", "key", "code", "ref", "hash"]
    if any(name_lower.endswith(p) or name_lower.startswith(p) for p in id_patterns):
        return True

    unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0

    entity_name_patterns = ["player", "oyuncu"]
    if any(p in name_lower for p in entity_name_patterns):
        return False

    # High-cardinality text fields often behave like identifiers. High-cardinality
    # numeric measures are common in analytics and should remain testable.
    if not pd.api.types.is_numeric_dtype(series) and unique_ratio > 0.95 and len(series) > 50:
        return True

    name_patterns = ["name", "title", "label", "description", "isim", "ad"]
    if any(p in name_lower for p in name_patterns):
        if unique_ratio > 0.8:
            return True

    return False


def select_label_column(
    df: pd.DataFrame,
    preferred: str | None = None,
    exclude: list[str] | set[str] | tuple[str, ...] | None = None,
) -> str | None:
    """Pick a human-readable text column for scatter/cluster point labels."""
    excluded = set(exclude or [])
    entity_keywords = [
        "playername", "player", "oyuncu", "customername", "customer", "musteri",
        "productname", "product", "urun", "team", "club", "employee", "name",
        "isim", "label", "title", "category", "region", "department",
    ]
    obvious_id_tokens = ["id", "uuid", "pk", "fk", "key", "code", "ref", "hash"]

    def _norm(value: str) -> str:
        return str(value).lower().replace("_", "").replace("-", "").replace(" ", "")

    def _usable(col: str) -> bool:
        if col in excluded or col not in df.columns:
            return False
        if pd.api.types.is_numeric_dtype(df[col]):
            return False
        normalized = _norm(col)
        if any(normalized == token or normalized.endswith(token) for token in obvious_id_tokens):
            return False
        return df[col].dropna().astype(str).str.strip().ne("").any()

    if preferred and _usable(preferred):
        return preferred

    scored: list[tuple[int, int, str]] = []
    for col in df.columns:
        col_name = str(col)
        if not _usable(col_name):
            continue
        normalized = _norm(col_name)
        score = 0
        for idx, keyword in enumerate(entity_keywords):
            if keyword in normalized:
                score = max(score, 100 - idx)
        unique_count = int(df[col].nunique(dropna=True))
        scored.append((score, unique_count, col_name))

    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def is_tautology(col_a: str, col_b: str) -> bool:
    """Returns True when one column name is essentially contained in the other."""
    a = col_a.lower().replace("_", "").replace(" ", "")
    b = col_b.lower().replace("_", "").replace(" ", "")
    if len(a) < 3 or len(b) < 3:
        return False
    return a == b or (a in b and len(a) > len(b) * 0.8) or (b in a and len(b) > len(a) * 0.8)


# ─── DataFrame Utilities ──────────────────────────────────────────────────────

def auto_join_dataframes(df_list: list) -> pd.DataFrame:
    """
    Attempts to merge a list of DataFrames by finding common/similar columns.
    Supports exact-match, normalised-match, and fuzzy semantic matching.
    """
    if not df_list:
        raise ValueError("No dataframes provided.")
    if len(df_list) == 1:
        return df_list[0]

    merged_df = df_list[0]

    for i in range(1, len(df_list)):
        current_df = df_list[i]

        merged_map = {normalize_col(c): c for c in merged_df.columns}
        current_map = {normalize_col(c): c for c in current_df.columns}
        common_norms = set(merged_map.keys()) & set(current_map.keys())

        join_keys: list = []
        rename_map: dict = {}

        if common_norms:
            rename_map = {current_map[norm]: merged_map[norm] for norm in common_norms}
            join_keys = [merged_map[norm] for norm in common_norms]
        else:
            best_pairs = []
            for c1 in merged_df.columns:
                n1 = str(c1).lower()
                for c2 in current_df.columns:
                    n2 = str(c2).lower()
                    score = get_similarity(c1, c2)

                    date_keywords = ["date", "dt", "time", "timestamp", "gün", "tarih", "yil", "yıl", "ay"]
                    is_c1_date = any(kw in n1 for kw in date_keywords)
                    is_c2_date = any(kw in n2 for kw in date_keywords)

                    if is_c1_date and is_c2_date:
                        if n2 in n1 or n1 in n2:
                            score = 0.95
                        else:
                            score += 0.3

                    if any(kw in n1 for kw in ["id", "key", "no", "code", "pk"]):
                        score += 0.1

                    if score > 0.75:
                        best_pairs.append((score, c1, c2))

            if best_pairs:
                best_pairs.sort(key=lambda x: x[0], reverse=True)
                seen_m: set = set()
                seen_c: set = set()
                for score, m, c in best_pairs:
                    if m not in seen_m and c not in seen_c:
                        rename_map[c] = m
                        join_keys.append(m)
                        seen_m.add(m)
                        seen_c.add(c)

        if not join_keys:
            m_cols = merged_df.columns.tolist()
            c_cols = current_df.columns.tolist()
            raise ValueError(
                f"Tablolar arasında bağlantı kurulamadı.\n"
                f"Mevcut tablo kolonları: {m_cols[:10]}\n"
                f"Yeni gelen dosya kolonları: {c_cols[:10]}\n"
                f"İpucu: Kolon isimlerinin (örn. Date vs Order Date, ID, UserID) benzer olduğundan emin olun."
            )

        current_df_renamed = current_df.rename(columns=rename_map)
        _validate_join_safety(merged_df, current_df_renamed, join_keys)
        merged_df = pd.merge(merged_df, current_df_renamed, on=join_keys, how="left")

        if len(merged_df) > 50000:
            raise ValueError(
                "Join sonucu 50.000 satır limitini aşıyor. "
                "Daha seçici veya benzersiz join kolonlarıyla tekrar deneyin."
            )

    return merged_df


def _validate_join_safety(left: pd.DataFrame, right: pd.DataFrame, join_keys: list) -> None:
    """Rejects ambiguous many-to-many joins before they multiply row counts."""
    left_dupes = left.duplicated(subset=join_keys, keep=False)
    right_dupes = right.duplicated(subset=join_keys, keep=False)
    if left_dupes.any() and right_dupes.any():
        raise ValueError(
            "Join güvenli değil: iki tabloda da aynı join anahtarları tekrar ediyor. "
            f"Problemli kolonlar: {join_keys}. Many-to-many birleşim veri çoğalmasına yol açabilir."
        )


def process_and_downsample(df: pd.DataFrame, max_points: int = 150):
    """Downsamples a DataFrame to ≤ max_points rows for UI display."""
    if len(df) > max_points:
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

        if date_cols:
            step = len(df) // max_points
            df_sampled = df.iloc[::step].head(max_points)
        else:
            df_sampled = df.sample(n=max_points, random_state=42).sort_index()
    else:
        df_sampled = df.copy()

    for col in df_sampled.columns:
        if pd.api.types.is_numeric_dtype(df_sampled[col]):
            df_sampled[col] = df_sampled[col].fillna(0)
        else:
            df_sampled[col] = df_sampled[col].fillna("N/A")

    return df_sampled.to_dict(orient="records")
