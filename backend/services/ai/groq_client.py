from __future__ import annotations

import asyncio
import json
import os
import time
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

    Uses a shared deadline for both the first attempt and the optional repair attempt,
    so total time never exceeds timeout_seconds regardless of how many attempts are made.
    Never returns unvalidated LLM output.
    """
    client = _get_groq_client()
    model = _get_model()
    # Reserve a small portion of the budget for the repair attempt (at most 40%)
    first_attempt_timeout = max(3, int(timeout_seconds * 0.7))
    deadline = time.monotonic() + timeout_seconds

    async def _chat(msgs: list[dict], call_timeout: int) -> str:
        response = await client.chat.completions.create(
            model=model,
            messages=msgs,  # type: ignore[arg-type]
            temperature=temperature,
            response_format={"type": "json_object"},
            timeout=call_timeout,
        )
        return response.choices[0].message.content or ""

    # ── First attempt (uses 70% of budget) ───────────────────────────────────
    raw_content = await _chat(messages, first_attempt_timeout)

    try:
        data = _loads_json(raw_content)
        result = model_validate_compat(schema_model, data)
        # Debug metadata (never exposes key)
        _log_debug(model, timeout_seconds, attempt=1)
        return result
    except (json.JSONDecodeError, ValidationError, Exception):
        pass  # Fall through to repair attempt

    # ── Repair attempt (uses remaining budget) ────────────────────────────────
    remaining = deadline - time.monotonic()
    if remaining < 1.0:
        # No time left for a repair attempt
        raise asyncio.TimeoutError(
            f"Groq structured output: first attempt produced invalid JSON and no time "
            f"remains for a repair attempt (budget={timeout_seconds}s)."
        )

    repair_timeout = max(1, int(remaining))
    # For small models, a concrete example works better than an abstract schema.
    # List required top-level fields so the model knows what to fill.
    required_fields = list(
        getattr(schema_model, "model_fields", None) or getattr(schema_model, "__fields__", {}) or {}
    )
    fields_hint = ", ".join(f'"{f}": <value>' for f in required_fields[:12])
    repair_messages = messages + [
        {"role": "assistant", "content": raw_content},
        {
            "role": "user",
            "content": (
                "Your previous response did not match the required JSON format. "
                "Return ONLY a plain JSON object with these top-level keys filled in with real values "
                "(do NOT copy the schema, do NOT add markdown, do NOT add explanation):\n"
                f"{{{fields_hint}}}"
            ),
        },
    ]

    raw_repaired = await _chat(repair_messages, repair_timeout)

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
