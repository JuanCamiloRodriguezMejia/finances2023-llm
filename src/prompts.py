"""
Prompt builder — loads .md template files and fills Jinja2 placeholders.

Public API
----------
build_messages(system_path, user_path, row) -> (system_str, messages_list)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined, TemplateError


def _make_env() -> Environment:
    return Environment(
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,
    )


_ENV = _make_env()


def _render(path: str, context: dict[str, Any]) -> str:
    """Read a template file and render it with *context*."""
    source = Path(path).read_text(encoding="utf-8")
    try:
        return _ENV.from_string(source).render(**context)
    except TemplateError as exc:
        raise ValueError(
            f"Failed to render template {path!r}: {exc}. "
            f"Available keys: {list(context.keys())}"
        ) from exc


def build_messages(
    system_path: str,
    user_path: str,
    row: dict[str, Any],
) -> tuple[str, list[dict[str, str]]]:
    """
    Fill Jinja2 templates with row data.

    Parameters
    ----------
    system_path : path to the system prompt .md file
    user_path   : path to the user prompt .md file
    row         : dict of column_name → value from the dataset row

    Returns
    -------
    (system_string, messages_list)
    where messages_list is [{"role": "user", "content": "<rendered user prompt>"}]
    """
    # Coerce all values to str so Jinja2 never complains about None / numeric types
    ctx = {k: ("" if v is None else str(v)) for k, v in row.items()}

    system = _render(system_path, ctx)
    user = _render(user_path, ctx)
    messages: list[dict[str, str]] = [{"role": "user", "content": user}]
    return system, messages
