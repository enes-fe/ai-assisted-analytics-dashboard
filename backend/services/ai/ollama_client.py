from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from pydantic import BaseModel

from .schemas import model_json_schema_compat, model_validate_compat


def _get_timeout(env_var: str, default: int) -> int:
    """Read a timeout value (seconds) from an environment variable."""
    try:
        return max(1, int(os.getenv(env_var, str(default))))
    except (ValueError, TypeError):
        return default


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")

# Context window size — smaller = faster inference. 2048 is sufficient for structured JSON plans.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))


def get_ollama_client():
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError("Ollama Python package is not installed.") from exc
    return ollama.AsyncClient(host=os.getenv("OLLAMA_HOST", OLLAMA_HOST))


def _extract_content(response: Any) -> str:
    if isinstance(response, dict):
        return response.get("message", {}).get("content", "")
    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "") if message is not None else ""


def _loads_json(content: str) -> Any:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start:end + 1])
        raise


async def _chat(messages: list[dict], schema: dict, temperature: float, timeout: float):
    client = get_ollama_client()
    model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
    num_ctx = int(os.getenv("OLLAMA_NUM_CTX", str(OLLAMA_NUM_CTX)))
    return await asyncio.wait_for(
        client.chat(
            model=model,
            messages=messages,
            format=schema,
            options={"temperature": temperature, "num_ctx": num_ctx},
            keep_alive=-1,  # Keep model loaded in memory — prevents cold-start latency
        ),
        timeout=timeout,
    )


async def call_ollama_structured(
    messages: list[dict],
    schema_model: type[BaseModel],
    temperature: float = 0,
    timeout: int | None = None,
    timeout_env: str = "AI_SEMANTIC_TIMEOUT_SECONDS",
    timeout_default: int = 20,
) -> BaseModel:
    """Call Ollama with structured output.

    Timeout priority: explicit ``timeout`` arg > env var ``timeout_env`` > ``timeout_default``.
    """
    resolved_timeout = timeout if timeout is not None else _get_timeout(timeout_env, timeout_default)
    schema = model_json_schema_compat(schema_model)
    try:
        response = await _chat(messages, schema, temperature, resolved_timeout)
        data = _loads_json(_extract_content(response))
        return model_validate_compat(schema_model, data)
    except Exception as first_error:
        raise RuntimeError(
            f"Ollama structured output failed: {first_error}"
        ) from first_error
