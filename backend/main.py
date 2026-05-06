from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from typing import Any, List, Optional
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import pandas as pd
import io
import math
import os
import time
from collections import OrderedDict
from dotenv import load_dotenv

load_dotenv(override=True)


from database import engine, get_db, Base
from models import DatasetMeta
import ml_service
import storage_manager
from pydantic import BaseModel
from services.ai.chart_builder import build_chart_from_plan, build_charts_from_semantic_plan
from services.ai.prompt_planner import generate_prompt_chart_plan
from services.ai.schemas import SemanticDatasetPlan, FastSemanticPlan, model_dump_compat, model_validate_compat
from services.ai.fast_semantic_planner import generate_fast_semantic_plan
from services.ai.fast_chart_builder import build_charts_from_fast_plan
from services.ai.fast_kpi_builder import build_kpis_from_fast_plan
from services.ai.cluster_namer import apply_groq_cluster_names


class ClusterRequest(BaseModel):
    selected_cols: Optional[List[str]] = None


class KpiColumnsRequest(BaseModel):
    columns: List[str]


MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "5"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
MAX_PAGE_SIZE = int(os.getenv("MAX_PAGE_SIZE", "500"))
CACHE_TTL_SECONDS = int(os.getenv("ANALYTICS_CACHE_TTL_SECONDS", "1800"))
CACHE_MAX_ITEMS = int(os.getenv("ANALYTICS_CACHE_MAX_ITEMS", "20"))
MAX_DEFAULT_MAIN_CHARTS = max(1, min(int(os.getenv("ANALYTICS_MAX_DEFAULT_MAIN_CHARTS", "6")), 6))
MAX_CORE_CHARTS = max(1, min(int(os.getenv("ANALYTICS_MAX_CORE_CHARTS", str(MAX_DEFAULT_MAIN_CHARTS))), 6))


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "*")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _env_enabled(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


def _bounded_chart_count(value: int) -> int:
    return max(1, min(value, MAX_DEFAULT_MAIN_CHARTS))


Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thread pool for CPU-bound analytics (keeps FastAPI event loop free)
_executor = ThreadPoolExecutor(max_workers=4)
# In-memory analytics cache: (dataset_id, advanced) -> (created_at, result)
_analytics_cache: OrderedDict[tuple[int, bool], tuple[float, dict[str, Any]]] = OrderedDict()
# In-memory fast semantic plan cache: dataset_id -> FastSemanticPlan
_fast_plan_cache: dict[int, FastSemanticPlan] = {}

global_dfs = {}


def _error(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _analytics_cache_key(dataset_id: int, advanced: bool) -> tuple[int, bool]:
    return (dataset_id, advanced)


def _get_cached_analytics(dataset_id: int, advanced: bool):
    key = _analytics_cache_key(dataset_id, advanced)
    cached = _analytics_cache.get(key)
    if not cached:
        return None
    created_at, result = cached
    if time.time() - created_at > CACHE_TTL_SECONDS:
        _analytics_cache.pop(key, None)
        return None
    _analytics_cache.move_to_end(key)
    return result


def _set_cached_analytics(dataset_id: int, advanced: bool, result: dict[str, Any]):
    key = _analytics_cache_key(dataset_id, advanced)
    _analytics_cache[key] = (time.time(), result)
    _analytics_cache.move_to_end(key)
    while len(_analytics_cache) > CACHE_MAX_ITEMS:
        _analytics_cache.popitem(last=False)


def _chart_signature(chart: dict[str, Any]) -> tuple:
    series = chart.get("series") or []
    metric = series[0].get("key") if series and isinstance(series[0], dict) else None
    return (chart.get("type"), metric, chart.get("xAxisKey"))


def _combine_charts(ai_charts: list[dict], heuristic_charts: list[dict]) -> list[dict]:
    combined: list[dict] = []
    seen: set[tuple] = set()
    for chart in ai_charts + heuristic_charts:
        sig = _chart_signature(chart)
        if sig in seen:
            continue
        seen.add(sig)
        combined.append(chart)
        if len(combined) >= MAX_CORE_CHARTS:
            break

    if ai_charts and heuristic_charts and not any(c in combined for c in heuristic_charts):
        combined[-1:] = [heuristic_charts[0]]
    return combined


def get_dataset(dataset_id: int):
    """Utility to get dataset from RAM or Disk (Lazy Loading)"""
    if dataset_id in global_dfs:
        return global_dfs[dataset_id]

    # Try loading from disk
    df = storage_manager.load_dataset(dataset_id)
    if df is not None:
        global_dfs[dataset_id] = df
        return df

    return None


@app.post("/api/upload")
async def upload_file(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    if not files:
        raise HTTPException(status_code=400, detail=_error("ERR_NO_FILES", "No files were uploaded."))
    if len(files) > MAX_UPLOAD_FILES:
        raise HTTPException(status_code=400, detail=_error("ERR_MAX_FILES", f"Maximum limit of {MAX_UPLOAD_FILES} files reached."))

    df_list = []

    for file in files:
        filename = file.filename or "unnamed"
        if not filename.lower().endswith(('.csv', '.json', '.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail=_error("ERR_UNSUPPORTED_FORMAT", f"Unsupported file format: {filename}"))

        contents = await file.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            limit_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
            raise HTTPException(status_code=413, detail=_error("ERR_FILE_TOO_LARGE", f"{filename} exceeds the {limit_mb:.0f} MB upload limit."))

        try:
            if filename.lower().endswith('.csv'):
                try:
                    df = pd.read_csv(io.BytesIO(contents), sep=None, engine='python', on_bad_lines='skip')
                except Exception:
                    try:
                        df = pd.read_csv(io.BytesIO(contents), sep=None, engine='python', on_bad_lines='skip', encoding='latin-1')
                    except Exception:
                        df = pd.read_csv(io.BytesIO(contents), sep=None, engine='python', on_bad_lines='skip', encoding='latin-1', quoting=3)
            elif filename.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(io.BytesIO(contents))
            else:
                df = pd.read_json(io.BytesIO(contents))

            if df.empty:
                raise ValueError("Parsed file is empty.")

            for col in df.columns:
                c_str = str(col)
                c_low = c_str.lower()

                if c_low == 'id' or c_low.endswith('_id') or c_low.endswith(' id') or c_str.endswith('ID'):
                    df[col] = df[col].astype(str)
                    continue

                try:
                    if df[col].dtype == 'object':
                        converted = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
                        if converted.notna().sum() > df[col].notna().sum() * 0.6:
                            df[col] = converted
                except Exception:
                    pass

            df_list.append(df)
        except Exception as e:
            raise HTTPException(status_code=400, detail=_error("ERR_PARSE_FILE", f"Error parsing file {filename}: {str(e)}"))

    try:
        final_df = ml_service.auto_join_dataframes(df_list)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=_error("ERR_JOIN_FILES", str(e)))

    import uuid
    table_name = f"dataset_{uuid.uuid4().hex}"
    filename_meta = files[0].filename if len(files) == 1 else f"Multi-Join ({len(files)} files)"

    db_meta = DatasetMeta(
        filename=filename_meta,
        table_name=table_name,
        row_count=len(final_df),
    )
    db.add(db_meta)
    db.commit()
    db.refresh(db_meta)

    # Save to RAM and Disk
    global_dfs[db_meta.id] = final_df
    storage_manager.save_dataset(db_meta.id, final_df)

    # Trigger Cleanup (once)
    storage_manager.cleanup_storage()

    ui_data = ml_service.process_and_downsample(final_df)
    cols = final_df.columns.tolist()

    return {
        "dataset_id": db_meta.id,
        "filename": filename_meta,
        "data": ui_data,
        "columns": cols,
        "row_count": len(final_df)
    }


@app.get("/api/ai/fast-dashboard/{dataset_id}")
async def get_fast_dashboard(dataset_id: int):
    """Primary AI dashboard endpoint using Groq fast semantic planning.

    The LLM only identifies semantic column roles.
    Pandas performs all aggregation, grouping, sorting and KPI calculation.
    """
    df = get_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    timeout_seconds = _env_int("AI_SEMANTIC_TIMEOUT_SECONDS", 8)
    max_charts = _bounded_chart_count(_env_int("AI_MAX_RECOMMENDED_CHARTS", 5))
    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    # Check fast plan cache
    cached_plan = _fast_plan_cache.get(dataset_id)
    cache_hit = cached_plan is not None

    try:
        if cached_plan is None:
            fast_plan = await asyncio.wait_for(
                generate_fast_semantic_plan(df),
                timeout=timeout_seconds,
            )
            _fast_plan_cache[dataset_id] = fast_plan
        else:
            fast_plan = cached_plan

        kpis = build_kpis_from_fast_plan(df, fast_plan)
        charts = build_charts_from_fast_plan(df, fast_plan, max_charts=max_charts)

        plan_dict = (
            fast_plan.model_dump()
            if hasattr(fast_plan, "model_dump")
            else fast_plan.dict()  # type: ignore[attr-defined]
        )

        return {
            "status": "success",
            "kpis": kpis,
            "charts": charts,
            "semantic_plan": plan_dict,
            "dashboard_generation_mode": "groq_fast_semantic",
            "ai_debug": {
                "provider": "groq",
                "model": groq_model,
                "timeout_seconds": timeout_seconds,
                "semantic_plan_cache_hit": cache_hit,
                "kpi_count": len(kpis),
                "chart_count": len(charts),
            },
        }

    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=200,
            content={
                "status": "timeout",
                "kpis": [],
                "charts": [],
                "semantic_plan": None,
                "dashboard_generation_mode": "timeout",
                "message": (
                    "AI semantic analysis timed out. "
                    "Dataset preview is available, but dashboard charts were not generated "
                    "to avoid misleading heuristic results."
                ),
                "ai_debug": {
                    "provider": "groq",
                    "model": groq_model,
                    "timeout_seconds": timeout_seconds,
                    "semantic_plan_cache_hit": False,
                },
            },
        )

    except Exception as e:
        safe_error = str(e)
        # Never expose API key in error messages
        api_key = os.getenv("GROQ_API_KEY", "")
        if api_key and api_key != "your_api_key_here":
            safe_error = safe_error.replace(api_key, "[REDACTED]")

        return JSONResponse(
            status_code=200,
            content={
                "status": "error",
                "kpis": [],
                "charts": [],
                "semantic_plan": None,
                "dashboard_generation_mode": "error",
                "message": safe_error,
                "ai_debug": {
                    "provider": "groq",
                    "model": groq_model,
                    "timeout_seconds": timeout_seconds,
                    "semantic_plan_cache_hit": False,
                },
            },
        )


@app.get("/api/analytics/core/{dataset_id}")
async def get_core_analytics(
    dataset_id: int,
    advanced: bool = Query(False, description="Enable advanced/debug statistical tests."),
):
    """Heuristic analytics — kept for debug and backward compatibility.
    Primary dashboard should use /api/ai/fast-dashboard/{dataset_id}.

    Advanced statistical tests are intentionally opt-in so the default
    dashboard contract stays focused and academically defensible.
    """
    # Serve from cache if available
    cached = _get_cached_analytics(dataset_id, advanced)
    if cached is not None:
        return cached

    df = get_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    def _run_analytics():
        kpis = ml_service.calculate_kpis(df)
        stats = ml_service.run_statistical_tests(df) if advanced else None
        charts = ml_service.generate_heuristic_charts(
            df,
            stat_tests=stats,
            advanced_stats=advanced,
        )
        return {"kpis": kpis, "charts": charts, "statistical_tests": stats}

    try:
        loop = asyncio.get_event_loop()
        base_result = await loop.run_in_executor(_executor, _run_analytics)

        result = {
            "kpis": base_result["kpis"][:5],
            "charts": base_result["charts"][:MAX_CORE_CHARTS],
            "statistical_tests": base_result["statistical_tests"] if advanced else None,
            "advanced_statistics": {
                "enabled": advanced,
                "available": True,
                "enable_with": "/api/analytics/core/{dataset_id}?advanced=true",
                "default_scope": "Advanced statistical tests are hidden from the default dashboard flow.",
            },
            "semantic_plan": None,
            "chart_generation_mode": "heuristic_debug_advanced" if advanced else "heuristic_debug_basic",
            "is_primary_dashboard": False,
            "dashboard_generation_mode": "heuristic_debug_advanced" if advanced else "heuristic_debug_basic",
        }
        _set_cached_analytics(dataset_id, advanced, result)
        return result
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analytical Error: {str(e)}")


@app.post("/api/chat")
async def process_chat(dataset_id: int = Form(...), prompt: str = Form(...)):
    df = get_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if _env_enabled("AI_CHAT_ENABLED", "true"):
        timeout_seconds = _env_int("AI_CHAT_TIMEOUT_SECONDS", 8)

        # Reuse cached fast plan if available
        fast_plan = _fast_plan_cache.get(dataset_id)

        try:
            plan = await asyncio.wait_for(
                generate_prompt_chart_plan(df, prompt, fast_plan),
                timeout=timeout_seconds,
            )
            # Build chartData with Pandas only (chart_builder)
            from services.ai.chart_builder import build_chart_from_plan
            chart = build_chart_from_plan(df, plan)
            if chart is None:
                raise ValueError("AI returned a chart plan that could not be built from this dataset.")

            # Update source label
            chart["source"] = "groq_prompt_chart"
            return {
                "charts": [chart],
                "mode": "groq_prompt_chart",
                "intent": (
                    plan.model_dump()
                    if hasattr(plan, "model_dump")
                    else plan.dict()  # type: ignore[attr-defined]
                ),
            }
        except asyncio.TimeoutError:
            return {
                "charts": [],
                "mode": "ai_prompt_timeout",
                "message": "AI grafik planı zaman aşımına uğradı. Yanlış grafik göstermemek için sonuç eklenmedi.",
            }
        except Exception as ai_error:
            return {
                "charts": [],
                "mode": "ai_prompt_error",
                "message": "Bu komut için güvenilir bir grafik üretilemedi. Lütfen metriği ve kırılımı daha net yazın.",
                "ai_error": str(ai_error),
            }

    return {
        "charts": [],
        "mode": "ai_prompt_disabled",
        "message": "AI prompt analizi kapalı olduğu için grafik üretilmedi.",
    }


@app.get("/api/kpi/columns/{dataset_id}")
async def get_kpi_columns(dataset_id: int):
    """Returns available numeric columns for KPI selection, sorted by domain relevance."""
    df = get_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    from services.kpi_engine import domain_priority, is_meaningless_total
    from services.utils import is_id_column, format_col_name
    import numpy as np
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    result = []
    for col in num_cols:
        if is_id_column(col, df[col]):
            continue
        if is_meaningless_total(col):
            continue
        prio = domain_priority(col)
        result.append({
            "column": col,
            "label": format_col_name(col),
            "priority": prio,
            "nunique": int(df[col].dropna().nunique()),
            "is_binary": int(df[col].dropna().nunique()) <= 2,
        })
    result.sort(key=lambda x: (-x["priority"], x["label"]))
    return {"columns": result}


@app.post("/api/kpi/calculate/{dataset_id}")
async def calculate_custom_kpis(dataset_id: int, req: KpiColumnsRequest):
    """Calculates KPIs for a user-specified list of columns."""
    df = get_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not req.columns:
        raise HTTPException(status_code=400, detail="No columns specified")
    # Filter to valid numeric columns only
    import numpy as np
    valid = [c for c in req.columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not valid:
        raise HTTPException(status_code=400, detail="No valid numeric columns found")
    from services.kpi_engine import calculate_kpis
    filtered_df = df[[c for c in df.columns if c not in df.select_dtypes(include=[np.number]).columns or c in valid]]
    kpis = calculate_kpis(filtered_df, selected_columns=valid)
    return {"kpis": kpis}


@app.get("/api/ai/semantic-plan/{dataset_id}")
async def get_semantic_plan(dataset_id: int):
    """Returns the cached FastSemanticPlan or generates a new one."""
    df = get_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    cached_plan = _fast_plan_cache.get(dataset_id)
    if cached_plan:
        plan_dict = (
            cached_plan.model_dump()
            if hasattr(cached_plan, "model_dump")
            else cached_plan.dict()  # type: ignore[attr-defined]
        )
        return {"semantic_plan": plan_dict, "cache_hit": True}

    if not _env_enabled("AI_SEMANTIC_ENABLED", "true"):
        return {"semantic_plan": {"error": "AI_SEMANTIC_ENABLED is disabled."}}

    try:
        timeout_seconds = _env_int("AI_SEMANTIC_TIMEOUT_SECONDS", 8)
        plan = await asyncio.wait_for(
            generate_fast_semantic_plan(df),
            timeout=timeout_seconds,
        )
        _fast_plan_cache[dataset_id] = plan
        plan_dict = (
            plan.model_dump()
            if hasattr(plan, "model_dump")
            else plan.dict()  # type: ignore[attr-defined]
        )
        return {"semantic_plan": plan_dict, "cache_hit": False}
    except Exception as ai_error:
        return {"semantic_plan": {"error": str(ai_error)}}


@app.get("/api/data/{dataset_id}")
async def get_data(
    dataset_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=MAX_PAGE_SIZE),
    sort_by: Optional[str] = Query(None),
    sort_dir: str = Query("asc"),
):
    df = get_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    sort_dir = sort_dir.lower()
    if sort_dir not in {"asc", "desc"}:
        raise HTTPException(status_code=400, detail="Invalid sort direction")

    if sort_by:
        if sort_by not in df.columns:
            raise HTTPException(status_code=400, detail="Invalid sort column")
        try:
            df = df.sort_values(
                by=sort_by,
                ascending=(sort_dir == "asc"),
                na_position="last",
                kind="mergesort",
            )
        except TypeError:
            temp_key = "__sort_key__"
            while temp_key in df.columns:
                temp_key = f"_{temp_key}"
            df = (
                df.assign(**{temp_key: df[sort_by].astype(str)})
                .sort_values(temp_key, ascending=(sort_dir == "asc"), na_position="last", kind="mergesort")
                .drop(columns=[temp_key])
            )

    total_rows = len(df)
    total_pages = max(1, math.ceil(total_rows / page_size))

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    df_slice = df.iloc[start_idx:end_idx]
    data = ml_service.sanitize_for_json(df_slice.to_dict(orient="records"))

    return {
        "page": page,
        "page_size": page_size,
        "total_rows": total_rows,
        "total_pages": total_pages,
        "sort_by": sort_by,
        "sort_dir": sort_dir,
        "data": data
    }


@app.get("/api/ml/forecast/{dataset_id}")
async def get_forecast(dataset_id: int):
    df = get_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    forecasts = ml_service.run_forecasting(df)

    if isinstance(forecasts, dict) and "error" in forecasts:
        return JSONResponse(status_code=200, content={"charts": [], "error": forecasts["error"]})

    return {"charts": forecasts}


@app.post("/api/ml/cluster/{dataset_id}")
async def get_clustering(dataset_id: int, req: ClusterRequest):
    df = get_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    cluster_result = ml_service.run_clustering(df, selected_cols=req.selected_cols)

    if isinstance(cluster_result, dict) and "error" in cluster_result:
        return JSONResponse(status_code=200, content={"charts": [], "error": cluster_result["error"]})

    cluster_result = await apply_groq_cluster_names(cluster_result)

    return {"charts": [cluster_result]}


@app.get("/api/datasets")
async def list_datasets(db: Session = Depends(get_db)):
    """Returns all metadata for previously uploaded datasets."""
    datasets = db.query(DatasetMeta).order_by(DatasetMeta.created_at.desc()).all()
    return datasets


@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(dataset_id: int, db: Session = Depends(get_db)):
    """Deletes metadata and physical file for a dataset."""
    db_meta = db.query(DatasetMeta).filter(DatasetMeta.id == dataset_id).first()
    if not db_meta:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # 1. Delete from RAM + analytics cache + fast plan cache
    global_dfs.pop(dataset_id, None)
    _analytics_cache.pop(dataset_id, None)
    _fast_plan_cache.pop(dataset_id, None)

    # 2. Delete from Disk
    storage_manager.delete_dataset(dataset_id)

    # 3. Delete from DB
    db.delete(db_meta)
    db.commit()

    return {"success": True}
