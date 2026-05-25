"""
Shared contract for all LLM clients.

Every provider function must return an InferenceResult.
The pipeline never imports anthropic or openai directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InferenceResult:
    """Provider-agnostic result returned by every LLM function."""

    row_id: Any               # mirrors dataset id_column value
    subtask: int
    raw_response: str         # full text the model produced

    # Structured extraction of the answer (filled by output.py after the call)
    parsed: dict = field(default_factory=dict)

    # Token usage
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Metadata
    model: str = ""
    provider: str = ""
    text: str = ""            # original headline — carried for output CSVs

    # Non-None means the call failed; pipeline checks this field
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def succeeded(self) -> bool:
        return self.error is None
