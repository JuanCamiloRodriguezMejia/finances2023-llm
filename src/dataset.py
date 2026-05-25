"""
Dataset loader for the FinancES 2023 CSV.

Public API
----------
load_df(dataset_cfg)                       -> pd.DataFrame
iter_rows(dataset_cfg, pipeline_cfg)       -> Iterator[dict]
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import pandas as pd

if TYPE_CHECKING:
    from .config import DatasetConfig, PipelineConfig

# Label columns that may contain trailing whitespace (e.g. "neu ")
_LABEL_COLS = ["s. target", "s. others", "s. society"]


def load_df(dataset_cfg: "DatasetConfig") -> pd.DataFrame:
    """
    Load the CSV and normalise label whitespace.
    Returns the full DataFrame — caller applies row cap.
    """
    df = pd.read_csv(
        dataset_cfg.path,
        encoding=dataset_cfg.encoding,
        dtype=str,          # keep everything as str; avoids mixed-type surprises
    )
    # Strip whitespace from label columns (the dataset has one "neu " entry)
    for col in _LABEL_COLS:
        if col in df.columns:
            df[col] = df[col].str.strip()
    return df


def iter_rows(
    dataset_cfg: "DatasetConfig",
    pipeline_cfg: "PipelineConfig",
) -> Iterator[dict]:
    """
    Yield each row as a plain dict, respecting max_rows from pipeline config.
    All values are strings (NaN → empty string) for safe Jinja2 rendering.
    """
    df = load_df(dataset_cfg)

    if pipeline_cfg.max_rows is not None:
        df = df.head(pipeline_cfg.max_rows)

    for _, row in df.iterrows():
        yield {k: ("" if pd.isna(v) else str(v)) for k, v in row.items()}
