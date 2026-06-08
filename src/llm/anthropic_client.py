"""
Anthropic Messages API client.

Public function
---------------
run_anthropic(messages, system, cfg, row_id, subtask, text) -> InferenceResult

Never raises — errors are captured in InferenceResult.error.
API key is read from the ANTHROPIC_API_KEY environment variable.
"""
from __future__ import annotations

import os
from typing import Any

from .base import InferenceResult


def run_anthropic(
    messages: list[dict],
    system: str,
    cfg,              # AnthropicConfig (typed loosely to avoid circular imports)
    row_id: Any,
    subtask: int,
    text: str = "",
) -> InferenceResult:
    """Call Anthropic Messages API and return an InferenceResult."""

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return InferenceResult(
            row_id=row_id,
            subtask=subtask,
            raw_response="",
            text=text,
            model=cfg.model,
            provider="anthropic",
            error="ANTHROPIC_API_KEY environment variable is not set",
        )

    try:
        import anthropic  # imported here so the module loads even without the package

        client = anthropic.Anthropic(api_key=api_key)

        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "system": system,
            "messages": messages,
        }
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.top_k is not None:
            kwargs["top_k"] = cfg.top_k

        response = client.messages.create(**kwargs)
        raw: str = response.content[0].text

        return InferenceResult(
            row_id=row_id,
            subtask=subtask,
            raw_response=raw,
            text=text,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            model=cfg.model,
            provider="anthropic",
        )

    except Exception as exc:  # noqa: BLE001
        return InferenceResult(
            row_id=row_id,
            subtask=subtask,
            raw_response="",
            text=text,
            model=cfg.model,
            provider="anthropic",
            error=f"{type(exc).__name__}: {exc}",
        )
