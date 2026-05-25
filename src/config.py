"""
Config loader — validates config.yaml using Pydantic v2.

Load order: load_config(path) → AppConfig (fully validated).
CLI overrides are applied by mutating the returned AppConfig before
passing it to the pipeline.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, field_validator, model_validator


# ── Sub-models ────────────────────────────────────────────────────────────────

class AnthropicConfig(BaseModel):
    model: str
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None


class OpenAIConfig(BaseModel):
    model: str
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


class SubtaskPrompts(BaseModel):
    system: str   # file path
    user: str     # file path


class PromptsConfig(BaseModel):
    subtask1: SubtaskPrompts
    subtask2: SubtaskPrompts


class DatasetConfig(BaseModel):
    path: str
    text_column: str = "text"
    id_column: str = "id"
    encoding: str = "utf-8"


class PipelineConfig(BaseModel):
    subtasks: list[int] = [1, 2]
    batch_size: int = 1
    max_rows: Optional[int] = None
    retry_attempts: int = 3
    retry_backoff_seconds: float = 5.0
    skip_on_error: bool = True
    requests_per_minute: Optional[int] = None


class OutputConfig(BaseModel):
    dir: str = "outputs/"
    save_csv: bool = True
    save_jsonl: bool = True
    save_summary: bool = True
    flush_every: int = 50


# ── Root config ───────────────────────────────────────────────────────────────

class AppConfig(BaseModel):
    # Allow mutation so CLI flags can override values after loading.
    model_config = ConfigDict(frozen=False)

    provider: str
    anthropic: AnthropicConfig
    openai: OpenAIConfig
    prompts: PromptsConfig
    dataset: DatasetConfig
    pipeline: PipelineConfig
    output: OutputConfig

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: str) -> str:
        allowed = {"anthropic", "openai"}
        if v not in allowed:
            raise ValueError(f"provider must be one of {allowed!r}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _validate_files_exist(self) -> "AppConfig":
        # --- Prompt files ---
        for subtask_name in ("subtask1", "subtask2"):
            subtask: SubtaskPrompts = getattr(self.prompts, subtask_name)
            for role in ("system", "user"):
                p = Path(getattr(subtask, role))
                if not p.exists():
                    raise FileNotFoundError(
                        f"Prompt file not found: {p}  "
                        f"(looked up from cwd={Path.cwd()})"
                    )

        # --- Dataset file ---
        csv_path = Path(self.dataset.path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Dataset CSV not found: {csv_path}  "
                f"(cwd={Path.cwd()})"
            )

        # --- Column names ---
        with open(csv_path, encoding=self.dataset.encoding, newline="") as fh:
            reader = csv.DictReader(fh)
            headers: list[str] = list(reader.fieldnames or [])
        for col in (self.dataset.text_column, self.dataset.id_column):
            if col not in headers:
                raise ValueError(
                    f"Column {col!r} not found in {csv_path}. "
                    f"Available columns: {headers}"
                )

        return self

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_active_llm_cfg(self) -> Union[AnthropicConfig, OpenAIConfig]:
        """Return the provider-specific model config."""
        if self.provider == "anthropic":
            return self.anthropic
        return self.openai


# ── Loader ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> AppConfig:
    """Parse and validate config.yaml; raise on the first problem found."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return AppConfig(**raw)
