"""
ML / predictive analytics layer:
  - run_forecasting: Holt-Winters with LinearRegression fallback and simple holdout backtest.
  - run_clustering: KMeans with elbow selection plus silhouette/Davies-Bouldin quality diagnostics.
"""

import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .utils import sanitize_for_json, select_label_column, is_id_column, format_col_name
from .forecast_utils import getForecastDirection

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HW_AVAILABLE = True
except ImportError:
    _HW_AVAILABLE = False
    print("[ML] statsmodels not installed; forecasting will use LinearRegression fallback only.")


def _legacy_detect_cluster_domain(df: pd.DataFrame) -> str:
    cols_lower = " ".join(df.columns.str.lower())
    if any(k in cols_lower for k in ["customer", "client", "cust", "musteri"]):
        return "Müşteri Segmentasyonu"
    if any(k in cols_lower for k in ["employee", "staff", "worker", "calisan", "personel", "hire", "tenure", "department"]):
        return "Çalışan Kümeleri"
    if any(k in cols_lower for k in ["product", "item", "sku", "urun", "inventory"]):
        return "Ürün Kümeleri"
    if any(k in cols_lower for k in ["player", "athlete", "oyuncu", "goal", "assist", "gol"]):
        return "Oyuncu Performans Kümeleri"
    if any(k in cols_lower for k in ["order", "ship", "deliver", "siparis", "kargo"]):
        return "Sipariş Kümeleri"
    return "Veri Segmentasyonu"


LOWER_IS_BETTER_PATTERNS = [
    "minutesperscore",
    "minutespergoal",
    "scoringfrequency",
    "delay",
    "error",
    "defect",
    "risk",
    "cost",
    "mortality",
    "readmission",
    "complication",
    "churn",
    "dropout",
    "default",
    "defaultrate",
    "npl",
    "fraud",
    "debt",
    "downtime",
    "rejectrate",
    "reject",
    "waste",
    "loss",
    "failure",
    "complaint",
    "recency",
]

CATEGORY_CANDIDATE_PATTERNS = [
    "product",
    "productname",
    "category",
    "retailer",
    "retailername",
    "region",
    "state",
    "city",
    "teamname",
    "team",
    "position",
    "playerposition",
    "department",
    "segment",
    "diagnosis",
    "industry",
    "country",
    "channel",
    "branch",
    "status",
    "plan",
    "tier",
    "customersegment",
    "employeelevel",
    "jobrole",
    "clinic",
    "hospital",
    "treatment",
    "carrier",
    "warehouse",
    "supplier",
    "device",
    "machine",
    "line",
    "shift",
]


def _norm_name(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


CLUSTER_DOMAIN_TITLES = {
    "customer": "M\u00fc\u015fteri Segmentasyonu",
    "employee": "\u00c7al\u0131\u015fan K\u00fcmeleri",
    "commerce": "\u00dcr\u00fcn K\u00fcmeleri",
    "sports": "Oyuncu Performans K\u00fcmeleri",
    "health": "Sa\u011fl\u0131k Segmentasyonu",
    "finance": "Finansal Risk Segmentasyonu",
    "iot": "Sens\u00f6r K\u00fcmeleri",
    "manufacturing": "\u00dcretim K\u00fcmeleri",
    "logistics": "Sipari\u015f K\u00fcmeleri",
    "general": "Veri Segmentasyonu",
}


def _detect_cluster_domain_key(df: pd.DataFrame, selected_cols: list[str] | None = None) -> str:
    columns = list(df.columns)
    if selected_cols:
        columns.extend(selected_cols)
    cols_lower = " ".join(str(col).lower() for col in columns)

    domain_tokens = [
        ("sports", ["player", "athlete", "oyuncu", "goal", "assist", "gol", "minute", "pass", "position"]),
        ("customer", ["customer", "client", "cust", "musteri", "segment", "churn", "recency", "loyalty"]),
        ("employee", ["employee", "staff", "worker", "calisan", "personel", "hire", "tenure", "department", "salary"]),
        ("health", ["patient", "diagnosis", "treatment", "hospital", "clinic", "bmi", "glucose", "mortality", "readmission"]),
        ("finance", ["credit", "loan", "debt", "default", "fraud", "npl", "income", "balance", "risk_score"]),
        ("manufacturing", ["machine", "downtime", "defect", "reject", "waste", "yield", "production", "quality"]),
        ("iot", ["sensor", "temperature", "humidity", "vibration", "pressure", "voltage", "device"]),
        ("logistics", ["order", "ship", "deliver", "delivery", "siparis", "kargo", "warehouse", "carrier"]),
        ("commerce", ["product", "item", "sku", "urun", "inventory", "profit", "revenue", "sales", "price", "retail", "quantity"]),
    ]
    for domain, tokens in domain_tokens:
        if any(token in cols_lower for token in tokens):
            return domain
    return "general"


def detect_cluster_domain(df: pd.DataFrame) -> str:
    return CLUSTER_DOMAIN_TITLES.get(_detect_cluster_domain_key(df), CLUSTER_DOMAIN_TITLES["general"])


def _metric_domain(selected_cols: list[str], df: pd.DataFrame) -> str:
    return _detect_cluster_domain_key(df, selected_cols)


def _is_lower_better_metric(col: str) -> bool:
    name = _norm_name(col)
    return any(pattern in name for pattern in LOWER_IS_BETTER_PATTERNS)


def _silhouette_quality(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 0.7:
        return "Strong"
    if score >= 0.5:
        return "Good"
    if score >= 0.25:
        return "Moderate"
    return "Weak"


def _feature_rankings(cluster_df: pd.DataFrame, full_df: pd.DataFrame, selected_cols: list[str]) -> list[dict]:
    rankings: list[dict] = []
    for col in selected_cols:
        full_series = pd.to_numeric(full_df[col], errors="coerce").dropna()
        cluster_series = pd.to_numeric(cluster_df[col], errors="coerce").dropna()
        if full_series.empty or cluster_series.empty:
            continue

        q1 = float(full_series.quantile(0.25))
        q3 = float(full_series.quantile(0.75))
        center = float(full_series.mean())
        robust_scale = max(q3 - q1, float(full_series.std() or 0), 1e-9)
        cluster_value = float(cluster_series.mean())
        delta = (cluster_value - center) / robust_scale

        if cluster_value >= q3:
            level = "high"
        elif cluster_value <= q1:
            level = "low"
        else:
            level = "balanced"

        rankings.append({
            "column": col,
            "label": format_col_name(col),
            "value": round(cluster_value, 2),
            "level": level,
            "delta": round(float(delta), 3),
            "lower_is_better": _is_lower_better_metric(col),
        })

    rankings.sort(key=lambda item: abs(item["delta"]), reverse=True)
    return rankings


def _feature_phrase(feature: dict, domain: str) -> str:
    label = feature["label"]
    name = _norm_name(feature["column"])
    level = feature["level"]
    high = level == "high"

    if level == "balanced":
        if domain == "sports" and any(token in name for token in ["goal", "assist", "score", "rating", "xg", "xa"]):
            return "Balanced Contribution"
        if domain == "customer" and any(token in name for token in ["purchase", "spend", "revenue", "value"]):
            return "Balanced Customer Value"
        if domain == "employee" and any(token in name for token in ["salary", "tenure", "performance"]):
            return "Balanced Workforce Profile"
        if domain == "health" and any(token in name for token in ["risk", "mortality", "readmission", "complication"]):
            return "Balanced Clinical Risk"
        if domain == "finance" and any(token in name for token in ["risk", "default", "debt", "fraud"]):
            return "Balanced Financial Risk"
        return f"Balanced {label}"

    if domain == "sports":
        if any(token in name for token in ["minute", "mins", "playingtime"]):
            return "High Minutes" if high else "Limited Minutes"
        if "pass" in name:
            return "High Passing Volume" if high else "Low Passing Volume"
        if any(token in name for token in ["minutesperscore", "minutespergoal", "scoringfrequency"]):
            return "Low Scoring Efficiency" if high else "High Scoring Efficiency"
        if any(token in name for token in ["goal", "assist", "score", "rating", "xg", "xa"]):
            return "High Contribution" if high else "Low Contribution"

    if domain == "commerce":
        if any(token in name for token in ["profit", "margin"]):
            return "Profitable" if high else "Low Profit"
        if any(token in name for token in ["sales", "revenue", "volume", "quantity", "orders"]):
            return "High Volume" if high else "Low Volume"
        if "price" in name:
            return "Premium" if high else "Low Price"
        if "cost" in name:
            return "High Cost" if high else "Low Cost"

    if domain == "customer":
        if any(token in name for token in ["purchase", "spend", "revenue", "sales", "value", "amount"]):
            return "High Customer Value" if high else "Low Customer Value"
        if any(token in name for token in ["recency", "churn", "complaint", "risk"]):
            return "Elevated Retention Risk" if high else "Lower Retention Risk"
        if any(token in name for token in ["frequency", "orders", "visits"]):
            return "Frequent Buyers" if high else "Infrequent Buyers"

    if domain == "employee":
        if "salary" in name or "compensation" in name:
            return "Higher Compensation" if high else "Lower Compensation"
        if "tenure" in name or "experience" in name:
            return "Long Tenure" if high else "Newer Employees"
        if any(token in name for token in ["performance", "rating", "score"]):
            return "High Performance" if high else "Lower Performance"

    if domain == "health":
        if any(token in name for token in ["mortality", "readmission", "complication", "risk"]):
            return "Elevated Clinical Risk" if high else "Lower Clinical Risk"
        if any(token in name for token in ["bmi", "glucose", "cholesterol", "pressure"]):
            return f"High {label}" if high else f"Low {label}"

    if domain == "finance":
        if any(token in name for token in ["default", "npl", "fraud", "risk", "debt"]):
            return "Elevated Financial Risk" if high else "Lower Financial Risk"
        if any(token in name for token in ["income", "balance", "limit", "score"]):
            return f"High {label}" if high else f"Low {label}"

    if domain == "manufacturing":
        if any(token in name for token in ["defect", "reject", "waste", "downtime", "failure"]):
            return "High Production Loss" if high else "Low Production Loss"
        if any(token in name for token in ["yield", "throughput", "output"]):
            return "High Throughput" if high else "Low Throughput"

    if domain == "iot":
        if any(token in name for token in ["temperature", "pressure", "vibration", "humidity", "voltage"]):
            return f"High {label}" if high else f"Low {label}"

    if domain == "logistics":
        if any(token in name for token in ["delivery", "delay", "ship", "transit"]):
            return "Slower Fulfillment" if high else "Faster Fulfillment"
        if any(token in name for token in ["order", "value", "volume", "quantity"]):
            return "High Order Volume" if high else "Low Order Volume"

    return f"High {label}" if level == "high" else f"Low {label}"


def _cluster_name(cluster_id: int, feature_rankings: list[dict], domain: str) -> str:
    if not feature_rankings:
        return f"Balanced Segment {cluster_id + 1}"

    strong = [item for item in feature_rankings if item["level"] != "balanced" and abs(item["delta"]) >= 0.35]
    top = strong[:3]

    if domain == "sports":
        phrases = [_feature_phrase(item, domain) for item in feature_rankings[:4]]
        if "Limited Minutes" in phrases and "High Scoring Efficiency" in phrases:
            return "Limited Minutes / High Scoring Efficiency"
        if "High Minutes" in phrases and "High Passing Volume" in phrases:
            return "High Minutes / High Passing Volume"
        if not top:
            return f"Main Rotation / Balanced Contribution {cluster_id + 1}"

    if domain == "commerce":
        phrases = [_feature_phrase(item, domain) for item in feature_rankings[:4]]
        if "Low Price" in phrases and "Low Profit" in phrases:
            return "Low Price / Low Profit Segment"
        if "High Volume" in phrases and "Profitable" in phrases:
            return "High Volume / Higher Profit Pattern"
        if "Premium" in phrases and any(p in phrases for p in ["Profitable", "High Volume"]):
            return "Premium Price / Higher Metric Pattern"

    if not top:
        balanced_phrase = _feature_phrase(feature_rankings[0], domain)
        return f"{balanced_phrase} Segment {cluster_id + 1}"

    phrases = []
    for item in top:
        phrase = _feature_phrase(item, domain)
        if phrase not in phrases:
            phrases.append(phrase)
    return " / ".join(phrases[:2]) + " Segment"


def _dedupe_cluster_name(name: str, used_names: dict[str, int]) -> str:
    count = used_names.get(name, 0) + 1
    used_names[name] = count
    if count == 1:
        return name
    return f"{name} ({count})"


def _categorical_candidates(df: pd.DataFrame, selected_cols: list[str]) -> list[str]:
    candidates: list[str] = []
    selected = set(selected_cols)
    for col in df.columns:
        if col in selected or pd.api.types.is_numeric_dtype(df[col]):
            continue
        name = _norm_name(col)
        if not any(pattern == name or pattern in name for pattern in CATEGORY_CANDIDATE_PATTERNS):
            continue
        series = df[col].dropna().astype(str).str.strip()
        series = series[series.ne("")]
        unique_count = int(series.nunique())
        if 1 < unique_count <= min(50, max(3, len(series) * 0.6)):
            candidates.append(col)
    return candidates[:4]


def _dominant_categories(df: pd.DataFrame, cluster_df: pd.DataFrame, selected_cols: list[str]) -> list[dict]:
    result: list[dict] = []
    for col in _categorical_candidates(df, selected_cols):
        values = df.loc[cluster_df.index, col].dropna().astype(str).str.strip()
        values = values[values.ne("")]
        if values.empty:
            continue
        counts = values.value_counts().head(3)
        if counts.empty:
            continue
        total = int(values.count())
        result.append({
            "column": col,
            "label": format_col_name(col),
            "values": [
                {
                    "value": str(value),
                    "count": int(count),
                    "pct": round(float(count / total * 100), 1) if total else 0,
                }
                for value, count in counts.items()
            ],
        })
    return result


def _forecast_backtest(y: np.ndarray) -> dict | None:
    if len(y) < 8:
        return None

    holdout = max(2, min(6, len(y) // 5))
    train_y = y[:-holdout]
    test_y = y[-holdout:]
    train_x = np.arange(len(train_y)).reshape(-1, 1)
    test_x = np.arange(len(train_y), len(y)).reshape(-1, 1)

    model = LinearRegression().fit(train_x, train_y)
    pred = model.predict(test_x)
    errors = test_y - pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    nonzero = np.where(np.abs(test_y) < 1e-9, np.nan, test_y)
    mape = float(np.nanmean(np.abs(errors / nonzero)) * 100)

    return {
        "holdout_points": int(holdout),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": None if np.isnan(mape) else round(mape, 3),
    }


def run_forecasting(df: pd.DataFrame, periods: int = 6) -> list | dict:
    results: list = []

    date_col = None
    for col in df.columns:
        try:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                date_col = col
                break
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            sample = df[col].dropna().astype(str).str.strip()
            sample = sample[sample.ne("")].head(25)
            if sample.empty:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                test = pd.to_datetime(sample, errors="coerce")
            valid_dates = int(test.notna().sum())
            if valid_dates >= max(2, min(4, len(sample) // 2)) and test.nunique() >= 2:
                date_col = col
                break
        except Exception:
            continue

    if not date_col:
        return {"error": "Forecast hazırlanamadı: tarih kolonu bulunamadı."}

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return {"error": "Forecast hazırlanamadı: numerik metrik bulunamadı."}

    df_temp = df.copy()
    df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors="coerce")
    df_temp = df_temp.dropna(subset=[date_col]).sort_values(date_col)
    if df_temp.empty:
        return {"error": "Forecast hazırlanamadı: tarih kolonunda yeterli geçerli değer yok."}

    days = (df_temp[date_col].max() - df_temp[date_col].min()).days
    if days > 730:
        resample_rule, freq_label = "YE", "year"
    elif days > 60:
        resample_rule, freq_label = "ME", "month"
    else:
        resample_rule, freq_label = "W", "week"

    freq_map = {"year": "YE", "month": "ME", "week": "W"}

    for target_col in num_cols[:2]:
        try:
            ts = df_temp.set_index(date_col)[target_col].resample(resample_rule).mean().dropna()
            if len(ts) < 4:
                continue

            x = np.arange(len(ts)).reshape(-1, 1)
            y = ts.values
            forecast_warnings: list[str] = []
            fitted = None
            future_pred = None
            trend_dir = "stable"
            r2 = 0.0

            if _HW_AVAILABLE:
                try:
                    hw_model = ExponentialSmoothing(y, seasonal_periods=None, trend="add", seasonal=None)
                    hw_fit = hw_model.fit()
                    future_pred = hw_fit.forecast(periods)
                    fitted = hw_fit.fittedvalues
                    lr = LinearRegression().fit(x, y)
                    r2 = lr.score(x, y)
                    trend_dir = "upward" if lr.coef_[0] > 0 else "downward"
                except Exception:
                    fitted = None

            if fitted is None:
                lr = LinearRegression().fit(x, y)
                r2 = lr.score(x, y)
                fitted = lr.predict(x)
                future_x = np.arange(len(ts), len(ts) + periods).reshape(-1, 1)
                future_pred = lr.predict(future_x)
                trend_dir = "upward" if lr.coef_[0] > 0 else "downward"

            if fitted is None or future_pred is None:
                continue

            backtest = _forecast_backtest(y)
            if backtest and backtest.get("mape") is not None and backtest["mape"] > 30:
                forecast_warnings.append("high_error")
            if r2 < 0.3:
                forecast_warnings.append("low_r2")
            if len(ts) < 8:
                forecast_warnings.append("low_data")

            last_date = ts.index[-1]
            freq_str = freq_map[freq_label]
            future_dates = pd.date_range(last_date, periods=periods + 1, freq=freq_str)[1:]

            historical = [
                {"date": str(ts.index[i].date()), "actual": round(float(y[i]), 2), "fitted": round(float(fitted[i]), 2)}
                for i in range(len(ts))
            ]
            forecast = [
                {"date": str(future_dates[i].date()), "forecast": round(float(future_pred[i]), 2)}
                for i in range(periods)
            ]
            trend_dir = getForecastDirection(forecast)
            caution_note = (
                " Model fit or backtest quality is weak; use this as a directional scenario, not a precise forecast."
                if any(w in forecast_warnings for w in ["low_r2", "high_error", "low_data"])
                else " Review backtest metrics before relying on the forecast."
            )

            results.append({
                "id": f"forecast-{target_col}",
                "type": "forecast",
                "title": f"Tahmin: {target_col}",
                "target_col": target_col,
                "freq_label": freq_label,
                "r2_score": round(r2, 3),
                "data_points": len(ts),
                "warnings": forecast_warnings,
                "backtest": backtest,
                "trend_direction": trend_dir,
                "trend_source": "forecast_series",
                "insight": (
                    f"Forecast signal for {target_col}: R2={r2:.2f}, direction={trend_dir}."
                    f"{caution_note}"
                ),
                "historical": historical,
                "forecast": forecast,
            })
        except Exception:
            continue

    if not results:
        return {"error": "Forecast hazırlanamadı: en az 4 zaman noktası olan numerik seri bulunamadı."}

    return results


def run_clustering(df: pd.DataFrame, selected_cols: list = None, max_k: int = 5) -> dict:
    num_cols_available = [
        c for c in df.select_dtypes(include=[np.number]).columns.tolist()
        if df[c].dropna().nunique() > 2 and not is_id_column(str(c), df[c])
    ]

    if len(num_cols_available) < 2:
        return {"error": "Clustering için en az 2 numerik kolon gerekli."}

    if not selected_cols:
        scaled_candidates = df[num_cols_available].apply(pd.to_numeric, errors="coerce")
        variances = scaled_candidates.var().sort_values(ascending=False)
        selected_cols = variances.head(4).index.tolist()
    else:
        selected_cols = [c for c in selected_cols if c in num_cols_available]
        if len(selected_cols) < 2:
            return {"error": "Lütfen en az 2 geçerli numerik kolon seçin."}

    df_clean = df[selected_cols].dropna()
    if len(df_clean) < 10:
        return {"error": "Yeterli veri yok (min 10 satır)."}

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_clean)

    inertias: list = []
    k_range = range(2, min(max_k + 1, len(df_clean) // 2 + 1))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(x_scaled)
        inertias.append(km.inertia_)

    optimal_k = 2
    if len(inertias) >= 3:
        deltas = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
        second_deriv = [deltas[i] - deltas[i + 1] for i in range(len(deltas) - 1)]
        optimal_k = list(k_range)[second_deriv.index(max(second_deriv)) + 1]

    km_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = km_final.fit_predict(x_scaled)

    silhouette = None
    davies_bouldin = None
    cluster_warnings: list[str] = []
    try:
        if len(set(labels)) > 1 and len(df_clean) > optimal_k:
            silhouette = round(float(silhouette_score(x_scaled, labels)), 3)
            davies_bouldin = round(float(davies_bouldin_score(x_scaled, labels)), 3)
            if silhouette < 0.25:
                cluster_warnings.append("low_silhouette")
            if davies_bouldin > 2:
                cluster_warnings.append("high_overlap")
    except Exception:
        pass

    df_result = df_clean.copy()
    df_result["cluster"] = labels
    domain = _metric_domain(selected_cols, df)
    aggregation_method = "average"
    aggregation_label = "Features shown as: Average values"
    silhouette_quality = _silhouette_quality(silhouette)
    if silhouette is None:
        silhouette_text = "Silhouette Score: Not available."
    else:
        silhouette_text = f"Silhouette Score: {silhouette:.3f} ({silhouette_quality})."
    segmentation_note = (
        "Exploratory segmentation; weak separation means labels should be treated as tentative."
        if "low_silhouette" in cluster_warnings or "high_overlap" in cluster_warnings
        else "Segmentation summary; validate clusters against domain context before using them operationally."
    )

    cluster_profiles: list = []
    cluster_sizes = []
    cluster_names: dict[int, str] = {}
    used_names: dict[str, int] = {}
    for cluster_id in range(optimal_k):
        cluster_df = df_result[df_result["cluster"] == cluster_id]
        cluster_sizes.append(len(cluster_df))
        feature_rankings = _feature_rankings(cluster_df, df, selected_cols)
        name = _cluster_name(cluster_id, feature_rankings, domain)
        cluster_names[cluster_id] = _dedupe_cluster_name(name, used_names)
        profile: dict = {
            "cluster_id": cluster_id,
            "cluster_name": cluster_names[cluster_id],
            "size": len(cluster_df),
            "size_pct": round(len(cluster_df) / len(df_result) * 100, 1),
            "feature_rankings": feature_rankings,
            "top_categories": _dominant_categories(df, cluster_df, selected_cols),
        }
        for col in selected_cols:
            profile[f"{col}_mean"] = round(float(cluster_df[col].mean()), 2)
        cluster_profiles.append(profile)

    if cluster_sizes:
        min_size = min(cluster_sizes)
        max_size = max(cluster_sizes)
        if max_size and min_size / max_size < 0.15:
            cluster_warnings.append("cluster_imbalance")

    try:
        projection = PCA(n_components=2, random_state=42).fit_transform(x_scaled)
    except TypeError:
        projection = PCA(n_components=2).fit_transform(x_scaled)
    axis_x = "PCA Component 1"
    axis_y = "PCA Component 2"

    label_col = select_label_column(df, exclude=selected_cols)
    scatter_data = df_result[selected_cols + ["cluster"]].copy()
    scatter_data[axis_x] = projection[:, 0]
    scatter_data[axis_y] = projection[:, 1]
    scatter_data["cluster"] = scatter_data["cluster"].astype(str)
    scatter_data["cluster_name"] = scatter_data["cluster"].astype(int).map(cluster_names)
    scatter_data["__row"] = scatter_data.index.astype(int) + 1
    if label_col and label_col in df.columns:
        scatter_data["__label"] = df.loc[df_result.index, label_col].astype(str).values

    return sanitize_for_json({
        "id": "clustering-result",
        "type": "clustering",
        "detected_domain": domain,
        "title": f"{detect_cluster_domain(df)} ({optimal_k} Küme)",
        "optimal_k": optimal_k,
        "selected_cols": selected_cols,
        "elbow_data": [{"k": k, "inertia": round(inertias[i], 2)} for i, k in enumerate(k_range)],
        "silhouette_score": silhouette,
        "silhouette_quality": silhouette_quality,
        "silhouette_text": silhouette_text,
        "davies_bouldin_score": davies_bouldin,
        "warnings": cluster_warnings,
        "cluster_profiles": cluster_profiles,
        "aggregation_method": aggregation_method,
        "aggregation_label": aggregation_label,
        "projection_method": "PCA",
        "projection_note": "PCA components are projected dimensions, not raw business metrics.",
        "xAxisLabel": axis_x,
        "yAxisLabel": axis_y,
        "scatter_col_x": axis_x,
        "scatter_col_y": axis_y,
        "labelKey": "__label" if label_col else None,
        "labelName": label_col,
        "metricKeys": selected_cols,
        "insight": (
            f"{segmentation_note} k={optimal_k}; {len(df_result)} records grouped across "
            f"{len(selected_cols)} features. {silhouette_text}"
        ),
        "chartData": scatter_data.fillna(0).to_dict(orient="records"),
    })
