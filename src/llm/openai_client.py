"""
OpenAI Chat Completions client.

Public function
---------------
run_openai(messages, system, cfg, row_id, subtask, text) -> InferenceResult

Never raises — errors are captured in InferenceResult.error.
API key is read from the OPENAI_API_KEY environment variable.
"""
from __future__ import annotations

import os
from typing import Any

from .base import InferenceResult


def run_openai(
    messages: list[dict],
    system: str,
    cfg,              # OpenAIConfig
    row_id: Any,
    subtask: int,
    text: str = "",
) -> InferenceResult:
    """Call OpenAI Chat Completions API and return an InferenceResult."""

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        return InferenceResult(
            row_id=row_id,
            subtask=subtask,
            raw_response="",
            text=text,
            model=cfg.model,
            provider="openai",
            error="OPENAI_API_KEY environment variable is not set",
        )

    try:
        from openai import OpenAI  # imported here so the module loads without the package

        client = OpenAI(api_key=api_key)

        # OpenAI uses an explicit system message at the front of the list
        all_messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            *messages,
        ]

        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "max_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "messages": all_messages,
        }
        # Pass optional params only when explicitly set (avoid sending null values)
        if cfg.frequency_penalty is not None:
            kwargs["frequency_penalty"] = cfg.frequency_penalty
        if cfg.presence_penalty is not None:
            kwargs["presence_penalty"] = cfg.presence_penalty

        response = client.chat.completions.create(**kwargs)
        raw: str = response.choices[0].message.content or ""
        usage = response.usage

        return InferenceResult(
            row_id=row_id,
            subtask=subtask,
            raw_response=raw,
            text=text,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            model=cfg.model,
            provider="openai",
        )

    except Exception as exc:  # noqa: BLE001
        return InferenceResult(
            row_id=row_id,
            subtask=subtask,
            raw_response="",
            text=text,
            model=cfg.model,
            provider="openai",
            error=f"{type(exc).__name__}: {exc}",
        )
