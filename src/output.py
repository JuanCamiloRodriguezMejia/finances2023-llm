"""
Output writer — persists predictions and run metadata.

Artefacts written into outputs/<provider>_<model>_<timestamp>/
  predictions_subtask1.csv   id, text, raw_response, parsed_met, parsed_polarity_met, error
  predictions_subtask2.csv   id, text, raw_response, parsed_polarity_companies, parsed_polarity_consumers, error
  predictions.jsonl          one JSON object per InferenceResult (full)
  run_summary.json           model, provider, timestamp, counts, tokens, config snapshot

Public API
----------
writer = OutputWriter(cfg)
writer.add(result)       # called after every API call
writer.flush()           # write partial CSVs to disk
writer.finalize(cfg)     # close handles, write final artefacts
"""
from __future__ import annotations

import dataclasses
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .llm.base import InferenceResult

if TYPE_CHECKING:
    from .config import AppConfig

logger = logging.getLogger(__name__)

# Regex to find the first {...} block (handles model preamble before JSON)
_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _try_parse_json(text: str) -> dict:
    """
    Best-effort JSON extraction from a raw LLM response.
    Returns an empty dict on failure (never raises).
    """
    stripped = text.strip()

    # 1. Try direct parse
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Strip markdown fences (```json ... ```)
    fenced = re.sub(r"```(?:json)?\s*", "", stripped)
    try:
        return json.loads(fenced.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # 3. Search for first {...} block
    match = _JSON_BLOCK_RE.search(stripped)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    logger.warning("Could not parse JSON from response (first 120 chars): %s", text[:120])
    return {}


def _make_run_dir(cfg: "AppConfig") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = cfg.get_active_llm_cfg().model.replace("/", "-").replace(":", "-")
    run_dir = Path(cfg.output.dir) / f"{cfg.provider}_{model_slug}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class OutputWriter:
    """
    Stateful writer — opened once per pipeline run.
    Thread-safety is not required (batch_size=1 default).
    """

    def __init__(self, cfg: "AppConfig") -> None:
        self._dir = _make_run_dir(cfg)
        self._cfg = cfg
        self._results: list[InferenceResult] = []

        # Open JSONL file immediately so partial results stream out
        if cfg.output.save_jsonl:
            self._jsonl_path = self._dir / "predictions.jsonl"
            self._jsonl_fh = open(self._jsonl_path, "w", encoding="utf-8")
        else:
            self._jsonl_fh = None  # type: ignore[assignment]

        logger.info("Run output directory: %s", self._dir)

    # ── Public interface ──────────────────────────────────────────────────────

    def add(self, result: InferenceResult) -> None:
        """Register a result; parse JSON and stream to JSONL."""
        result.parsed = _try_parse_json(result.raw_response)
        self._results.append(result)

        if self._jsonl_fh is not None:
            self._jsonl_fh.write(json.dumps(dataclasses.asdict(result)) + "\n")

    def flush(self) -> None:
        """Write partial CSVs and flush the JSONL handle."""
        if self._jsonl_fh is not None:
            self._jsonl_fh.flush()
        self._write_csvs()

    def finalize(self, cfg: "AppConfig") -> None:
        """Close all handles and write the final artefacts."""
        if self._jsonl_fh is not None:
            self._jsonl_fh.flush()
            self._jsonl_fh.close()
            self._jsonl_fh = None  # type: ignore[assignment]

        self._write_csvs()

        if cfg.output.save_summary:
            self._write_summary(cfg)

        print(f"\n✓ Outputs written to: {self._dir}")
        logger.info("Finalized outputs in %s", self._dir)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _write_csvs(self) -> None:
        if not self._cfg.output.save_csv:
            return

        for subtask_num in self._cfg.pipeline.subtasks:
            rows = [r for r in self._results if r.subtask == subtask_num]
            if not rows:
                continue

            if subtask_num == 1:
                records = [
                    {
                        "id": r.row_id,
                        "text": r.text,
                        "raw_response": r.raw_response,
                        "parsed_met": r.parsed.get("met"),
                        "parsed_polarity_met": r.parsed.get("polarity_met"),
                        "error": r.error,
                    }
                    for r in rows
                ]
            else:
                records = [
                    {
                        "id": r.row_id,
                        "text": r.text,
                        "raw_response": r.raw_response,
                        "parsed_polarity_companies": r.parsed.get("polarity_companies"),
                        "parsed_polarity_consumers": r.parsed.get("polarity_consumers"),
                        "error": r.error,
                    }
                    for r in rows
                ]

            csv_path = self._dir / f"predictions_subtask{subtask_num}.csv"
            pd.DataFrame(records).to_csv(csv_path, index=False, encoding="utf-8")

    def _write_summary(self, cfg: "AppConfig") -> None:
        total_tokens = sum(r.total_tokens for r in self._results)
        error_results = [r for r in self._results if r.error]

        summary = {
            "provider": cfg.provider,
            "model": cfg.get_active_llm_cfg().model,
            "timestamp": datetime.now().isoformat(),
            "total_calls": len(self._results),
            "error_count": len(error_results),
            "total_prompt_tokens": sum(r.prompt_tokens for r in self._results),
            "total_completion_tokens": sum(r.completion_tokens for r in self._results),
            "total_tokens": total_tokens,
            "subtasks_run": cfg.pipeline.subtasks,
            "errors": [
                {"row_id": r.row_id, "subtask": r.subtask, "error": r.error}
                for r in error_results
            ],
            "config_snapshot": cfg.model_dump(),
        }

        summary_path = self._dir / "run_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.info("Run summary written to %s", summary_path)
