"""
ml_service.py — Thin Facade
─────────────────────────────────────────────────────────────────────────
This file is now a backward-compatible re-export facade.
All business logic lives in backend/services/*.

Import pattern for the rest of the codebase (main.py, etc.) is unchanged:
    import ml_service
    ml_service.calculate_kpis(df)
    ml_service.generate_heuristic_charts(df)
    ...
"""

# ── Shared utilities ─────────────────────────────────────────────────────────
from services.utils import (
    sanitize_for_json,
    clean_string,
    normalize_col,
    format_col_name,
    get_similarity,
    is_id_column,
    is_tautology,
    auto_join_dataframes,
    process_and_downsample,
)

# ── Column profiling ─────────────────────────────────────────────────────────
from services.column_profiler import (
    get_column_profile,
    get_central_cols,
)

# ── KPI engine ───────────────────────────────────────────────────────────────
from services.kpi_engine import (
    calculate_kpis,
    is_meaningless_total,
    domain_priority,
    should_skip_histogram,
    MEANINGLESS_TOTAL_PATTERNS,
    DOMAIN_PRIORITY_KW,
    HISTO_SKIP_PATTERNS,
)

# ── Chart generation ─────────────────────────────────────────────────────────
from services.chart_generator import generate_heuristic_charts

# ── Statistical testing ───────────────────────────────────────────────────────
from services.stat_engine import (
    run_statistical_tests,
    human_readable_correlation,
    human_readable_group_test,
    human_readable_chi_square,
    priority_score,
    group_by_category,
)

# ── ML / predictive ──────────────────────────────────────────────────────────
from services.ml_engine import (
    run_forecasting,
    run_clustering,
    detect_cluster_domain,
)
from services.forecast_utils import getForecastDirection, get_forecast_direction

# ── NLP query parser ─────────────────────────────────────────────────────────
from services.query_parser import simulate_rearchitecting

__all__ = [
    # utils
    "sanitize_for_json", "clean_string", "normalize_col", "format_col_name",
    "get_similarity", "is_id_column", "is_tautology",
    "auto_join_dataframes", "process_and_downsample",
    # column_profiler
    "get_column_profile", "get_central_cols",
    # kpi_engine
    "calculate_kpis", "is_meaningless_total", "domain_priority",
    "should_skip_histogram", "MEANINGLESS_TOTAL_PATTERNS", "DOMAIN_PRIORITY_KW", "HISTO_SKIP_PATTERNS",
    # chart_generator
    "generate_heuristic_charts",
    # stat_engine
    "run_statistical_tests", "human_readable_correlation", "human_readable_group_test",
    "human_readable_chi_square", "priority_score", "group_by_category",
    # ml_engine
    "run_forecasting", "run_clustering", "detect_cluster_domain",
    "getForecastDirection", "get_forecast_direction",
    # query_parser
    "simulate_rearchitecting",
]
