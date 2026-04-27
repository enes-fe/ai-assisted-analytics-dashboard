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
from database import engine, get_db, Base
from models import DatasetMeta
import ml_service
import storage_manager
from pydantic import BaseModel
from services.ai.chart_builder import build_chart_from_plan, build_charts_from_semantic_plan
from services.ai.prompt_planner import generate_prompt_chart_plan
from services.ai.schemas import SemanticDatasetPlan, model_dump_compat, model_validate_compat
from services.ai.semantic_planner import generate_semantic_dataset_plan


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


class ClusterRequest(BaseModel):
    selected_cols: Optional[List[str]] = None


MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "5"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
MAX_PAGE_SIZE = int(os.getenv("MAX_PAGE_SIZE", "500"))
CACHE_TTL_SECONDS = int(os.getenv("ANALYTICS_CACHE_TTL_SECONDS", "1800"))
CACHE_MAX_ITEMS = int(os.getenv("ANALYTICS_CACHE_MAX_ITEMS", "20"))
MAX_CORE_CHARTS = int(os.getenv("ANALYTICS_MAX_CORE_CHARTS", "12"))
AI_SEMANTIC_TIMEOUT_SECONDS = float(os.getenv("AI_SEMANTIC_TIMEOUT_SECONDS", "6"))
AI_CHAT_TIMEOUT_SECONDS = float(os.getenv("AI_CHAT_TIMEOUT_SECONDS", "12"))
AI_DEBUG_TIMEOUT_SECONDS = float(os.getenv("AI_DEBUG_TIMEOUT_SECONDS", "30"))


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "*")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _env_enabled(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


async def _with_timeout(coro, timeout_seconds: float, label: str):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"{label} timed out after {timeout_seconds:g}s") from exc

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
# In-memory analytics cache: dataset_id -> (created_at, result)
_analytics_cache: OrderedDict[int, tuple[float, dict[str, Any]]] = OrderedDict()

global_dfs = {}


def _error(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _get_cached_analytics(dataset_id: int):
    cached = _analytics_cache.get(dataset_id)
    if not cached:
        return None
    created_at, result = cached
    if time.time() - created_at > CACHE_TTL_SECONDS:
        _analytics_cache.pop(dataset_id, None)
        return None
    _analytics_cache.move_to_end(dataset_id)
    return result


def _set_cached_analytics(dataset_id: int, result: dict[str, Any]):
    _analytics_cache[dataset_id] = (time.time(), result)
    _analytics_cache.move_to_end(dataset_id)
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


def _semantic_plan_from_cached(cached: Optional[dict[str, Any]]) -> Optional[SemanticDatasetPlan]:
    if not cached:
        return None
    payload = cached.get("semantic_plan")
    if not isinstance(payload, dict) or payload.get("error"):
        return None
    try:
        return model_validate_compat(SemanticDatasetPlan, payload)
    except Exception:
        return None


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

@app.get("/api/analytics/core/{dataset_id}")
async def get_core_analytics(dataset_id: int):
    """Heavy analytics run in thread pool — keeps the event loop responsive."""
    # Serve from cache if available
    cached = _get_cached_analytics(dataset_id)
    if cached is not None:
        return cached

    df = get_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    def _run_analytics():
        kpis   = ml_service.calculate_kpis(df)
        stats  = ml_service.run_statistical_tests(df)
        charts = ml_service.generate_heuristic_charts(df, stat_tests=stats)
        return {"kpis": kpis, "charts": charts, "statistical_tests": stats}

    try:
        loop = asyncio.get_event_loop()
        base_result = await loop.run_in_executor(_executor, _run_analytics)

        semantic_plan = None
        semantic_payload: dict[str, Any] | None = None
        ai_charts: list[dict] = []
        mode = "heuristic_only"

        if _env_enabled("AI_SEMANTIC_ENABLED", "true"):
            try:
                # The LLM only returns a JSON plan; Pandas below builds the data.
                semantic_plan = await _with_timeout(
                    generate_semantic_dataset_plan(df),
                    AI_SEMANTIC_TIMEOUT_SECONDS,
                    "AI semantic planning",
                )
                semantic_payload = model_dump_compat(semantic_plan)
                ai_charts = build_charts_from_semantic_plan(df, semantic_plan)
                if ai_charts:
                    mode = "ollama_semantic_plus_heuristic"
            except Exception as ai_error:
                semantic_payload = {"error": str(ai_error)}

        heuristic_charts = base_result["charts"]
        charts = _combine_charts(ai_charts, heuristic_charts) if ai_charts else heuristic_charts[:MAX_CORE_CHARTS]
        result = {
            "kpis": base_result["kpis"],
            "charts": charts,
            "statistical_tests": base_result["statistical_tests"],
            "semantic_plan": semantic_payload,
            "chart_generation_mode": mode,
        }
        _set_cached_analytics(dataset_id, result)
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
        try:
            cached = _get_cached_analytics(dataset_id)
            semantic_plan = _semantic_plan_from_cached(cached)
            if semantic_plan is None and _env_enabled("AI_SEMANTIC_ENABLED", "true"):
                semantic_plan = await _with_timeout(
                    generate_semantic_dataset_plan(df),
                    min(AI_CHAT_TIMEOUT_SECONDS, AI_DEBUG_TIMEOUT_SECONDS),
                    "AI semantic planning for chat",
                )

            # The LLM selects intent only; chartData is produced by Pandas.
            plan = await _with_timeout(
                generate_prompt_chart_plan(df, prompt, semantic_plan),
                AI_CHAT_TIMEOUT_SECONDS,
                "AI prompt chart planning",
            )
            chart = build_chart_from_plan(df, plan)
            if chart is None:
                raise ValueError("AI returned a chart plan that could not be built from this dataset.")
            return {
                "charts": [chart],
                "mode": "ollama_prompt_chart",
                "intent": model_dump_compat(plan),
            }
        except Exception as ai_error:
            fallback_charts = ml_service.simulate_rearchitecting(df, prompt)
            return {
                "charts": fallback_charts,
                "mode": "fallback_heuristic",
                "ai_error": str(ai_error),
            }

    fallback_charts = ml_service.simulate_rearchitecting(df, prompt)
    return {"charts": fallback_charts, "mode": "fallback_heuristic"}


@app.get("/api/ai/semantic-plan/{dataset_id}")
async def get_semantic_plan(dataset_id: int):
    df = get_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    cached = _get_cached_analytics(dataset_id)
    semantic_plan = cached.get("semantic_plan") if cached else None
    if semantic_plan:
        return {"semantic_plan": semantic_plan}

    if not _env_enabled("AI_SEMANTIC_ENABLED", "true"):
        return {"semantic_plan": {"error": "AI_SEMANTIC_ENABLED is disabled."}}

    try:
        plan = await _with_timeout(
            generate_semantic_dataset_plan(df),
            AI_DEBUG_TIMEOUT_SECONDS,
            "AI semantic planning",
        )
        return {"semantic_plan": model_dump_compat(plan)}
    except Exception as ai_error:
        return {"semantic_plan": {"error": str(ai_error)}}

@app.get("/api/data/{dataset_id}")
async def get_data(
    dataset_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=MAX_PAGE_SIZE),
):
    df = get_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
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
    
    # 1. Delete from RAM + analytics cache
    global_dfs.pop(dataset_id, None)
    _analytics_cache.pop(dataset_id, None)
        
    # 2. Delete from Disk
    storage_manager.delete_dataset(dataset_id)
    
    # 3. Delete from DB
    db.delete(db_meta)
    db.commit()
    
    return {"success": True}
