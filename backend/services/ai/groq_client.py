from __future__ import annotations

import json
import os
from typing import Any

from pydantic import BaseModel, ValidationError

from .schemas import model_validate_compat


def _get_timeout(env_var: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(env_var, str(default))))
    except (ValueError, TypeError):
        return default


def _get_model() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def _get_api_key() -> str | None:
    return os.getenv("GROQ_API_KEY")


def _get_groq_client():
    """Return an AsyncGroq client. Raises RuntimeError if groq is not installed or API key is missing."""
    try:
        from groq import AsyncGroq  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "groq Python package is not installed. Run: pip install groq"
        ) from exc

    api_key = _get_api_key()
    if not api_key or api_key.strip() == "your_api_key_here":
        raise RuntimeError(
            "GROQ_API_KEY environment variable is not set or is still the placeholder value. "
            "Add your real Groq API key to backend/.env"
        )
    return AsyncGroq(api_key=api_key)


def _loads_json(content: str) -> Any:
    """Parse JSON from LLM response, with fallback bracket extraction."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise


async def call_groq_structured(
    messages: list[dict],
    schema_model: type[BaseModel],
    temperature: float = 0,
    timeout_seconds: int = 8,
) -> BaseModel:
    """Call Groq with JSON mode and validate the response against schema_model.

    On invalid JSON / validation failure, makes one repair attempt.
    Never returns unvalidated LLM output.
    """
    client = _get_groq_client()
    model = _get_model()

    async def _chat(msgs: list[dict]) -> str:
        response = await client.chat.completions.create(
            model=model,
            messages=msgs,  # type: ignore[arg-type]
            temperature=temperature,
            response_format={"type": "json_object"},
            timeout=timeout_seconds,
        )
        return response.choices[0].message.content or ""

    # ── First attempt ────────────────────────────────────────────────────────
    raw_content = await _chat(messages)

    try:
        data = _loads_json(raw_content)
        result = model_validate_compat(schema_model, data)
        # Debug metadata (never exposes key)
        _log_debug(model, timeout_seconds, attempt=1)
        return result
    except (json.JSONDecodeError, ValidationError, Exception) as first_error:
        pass  # Fall through to repair attempt

    # ── Repair attempt ───────────────────────────────────────────────────────
    schema_hint = (
        schema_model.model_json_schema()
        if hasattr(schema_model, "model_json_schema")
        else schema_model.schema()
    )
    repair_messages = messages + [
        {"role": "assistant", "content": raw_content},
        {
            "role": "user",
            "content": (
                "Your previous response was not valid JSON matching the required schema. "
                "Return ONLY valid JSON that exactly matches this schema — no markdown, no explanation:\n"
                f"{json.dumps(schema_hint, ensure_ascii=False)}"
            ),
        },
    ]

    raw_repaired = await _chat(repair_messages)

    try:
        data = _loads_json(raw_repaired)
        result = model_validate_compat(schema_model, data)
        _log_debug(model, timeout_seconds, attempt=2)
        return result
    except (json.JSONDecodeError, ValidationError, Exception) as repair_error:
        raise RuntimeError(
            f"Groq structured output failed after repair attempt. "
            f"Model: {model}. "
            f"Repair error: {repair_error}"
        ) from repair_error


def _log_debug(model: str, timeout_seconds: int, attempt: int) -> None:
    """Print debug metadata without revealing the API key."""
    print(
        f"[groq_client] provider=groq model={model} "
        f"timeout={timeout_seconds}s attempt={attempt} status=ok"
    )
