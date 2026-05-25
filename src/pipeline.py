"""
Inference pipeline — orchestrates dataset → prompts → LLM → results.

Flow
----
for row in iter_rows(cfg):
    for subtask in cfg.pipeline.subtasks:
        system, messages = build_messages(subtask, row)
        result = dispatch_llm(...)   ← only place that checks cfg.provider
        collect(result)

Retry logic uses tenacity with exponential back-off.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from tenacity import (
    RetryError,
    retry,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from .dataset import iter_rows
from .llm.anthropic_client import run_anthropic
from .llm.base import InferenceResult
from .llm.openai_client import run_openai
from .prompts import build_messages

if TYPE_CHECKING:
    from .config import AppConfig
    from .output import OutputWriter

logger = logging.getLogger(__name__)


# ── Provider dispatch ─────────────────────────────────────────────────────────

def dispatch_llm(
    messages: list[dict],
    system: str,
    cfg: "AppConfig",
    row_id: Any,
    subtask: int,
    text: str = "",
) -> InferenceResult:
    """
    The **only** place that checks cfg.provider.
    Everything else in the pipeline is provider-blind.
    """
    llm_cfg = cfg.get_active_llm_cfg()
    if cfg.provider == "anthropic":
        return run_anthropic(messages, system, llm_cfg, row_id, subtask, text)
    return run_openai(messages, system, llm_cfg, row_id, subtask, text)


# ── Retry wrapper ─────────────────────────────────────────────────────────────

def _call_with_retry(
    messages: list[dict],
    system: str,
    cfg: "AppConfig",
    row_id: Any,
    subtask: int,
    text: str = "",
) -> InferenceResult:
    """
    Wraps dispatch_llm with tenacity retry on InferenceResult.error != None.
    Uses exponential back-off derived from cfg.pipeline.retry_backoff_seconds.
    """
    attempts = cfg.pipeline.retry_attempts
    base_wait = cfg.pipeline.retry_backoff_seconds

    @retry(
        retry=retry_if_result(lambda r: r.error is not None),
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=base_wait, min=base_wait, max=base_wait * 8),
        reraise=False,
    )
    def _inner() -> InferenceResult:
        result = dispatch_llm(messages, system, cfg, row_id, subtask, text)
        if result.error:
            logger.warning(
                "LLM call failed (row=%s subtask=%d): %s",
                row_id, subtask, result.error,
            )
        return result

    try:
        return _inner()
    except RetryError as exc:
        # All attempts exhausted — return the last InferenceResult (with error set)
        return exc.last_attempt.result()


# ── Optional rate limiter ─────────────────────────────────────────────────────

class _RateLimiter:
    """Simple token-bucket rate limiter (requests per minute)."""

    def __init__(self, rpm: int) -> None:
        self._min_gap = 60.0 / rpm
        self._last = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed < self._min_gap:
            time.sleep(self._min_gap - elapsed)
        self._last = time.monotonic()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(cfg: "AppConfig") -> list[InferenceResult]:
    """
    Orchestrate full inference run.

    Returns all InferenceResult objects (including failed ones).
    Writes partial results to disk every cfg.output.flush_every rows.
    """
    # Import here to avoid circular import (output imports llm.base)
    from .output import OutputWriter

    rows = list(iter_rows(cfg.dataset, cfg.pipeline))
    total_calls = len(rows) * len(cfg.pipeline.subtasks)
    results: list[InferenceResult] = []

    rate_limiter: _RateLimiter | None = (
        _RateLimiter(cfg.pipeline.requests_per_minute)
        if cfg.pipeline.requests_per_minute
        else None
    )

    writer = OutputWriter(cfg)

    logger.info(
        "Pipeline starting: %d rows × %d subtasks = %d total calls",
        len(rows), len(cfg.pipeline.subtasks), total_calls,
    )

    with tqdm(total=total_calls, desc="Inference", unit="call") as pbar:
        for row in rows:
            row_id = row[cfg.dataset.id_column]
            text = row.get(cfg.dataset.text_column, "")

            for subtask in cfg.pipeline.subtasks:
                subtask_prompts = getattr(cfg.prompts, f"subtask{subtask}")
                system, messages = build_messages(
                    subtask_prompts.system,
                    subtask_prompts.user,
                    row,
                )

                if rate_limiter:
                    rate_limiter.wait()

                result = _call_with_retry(
                    messages, system, cfg, row_id, subtask, text
                )

                if result.error and not cfg.pipeline.skip_on_error:
                    writer.finalize(cfg)
                    raise RuntimeError(
                        f"Fatal error on row={row_id} subtask={subtask}: {result.error}"
                    )

                results.append(result)
                writer.add(result)
                pbar.update(1)

                # Flush partial results periodically
                completed_rows = sum(
                    1 for r in results if r.subtask == cfg.pipeline.subtasks[-1]
                )
                if completed_rows > 0 and completed_rows % cfg.output.flush_every == 0:
                    writer.flush()
                    logger.debug("Flushed partial results (%d rows done)", completed_rows)

    writer.finalize(cfg)
    return results
