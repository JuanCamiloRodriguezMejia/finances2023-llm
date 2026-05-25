# CLAUDE.md — FinancES Dataset LLM Inference Script

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Spanish financial sentiment analysis research project** (Master's thesis / Seminario de Investigación) built around the **FinancES 2023 dataset** — a multi-aspect, aspect-based sentiment analysis (ABSA) benchmark for Spanish financial news.

## Environment Setup

Python 3.13.2 virtual environment is located at `.venv/`.

```powershell
# Activate the virtual environment (PowerShell)
.\.venv\Scripts\Activate.ps1

# Install packages
.\.venv\Scripts\pip.exe install <package>
```

## Dataset: FinancES 2023

**File:** [financial_dataset_quoted.csv](financial_dataset_quoted.csv)  
**Also available as:** [loaded_data.xlsx](loaded_data.xlsx)

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique row identifier |
| `text` | string | Spanish financial news text (tweet or headline) |
| `target` | string | Named entity / aspect being evaluated |
| `s. target` | pos/neg/neu | Sentiment expressed **toward the target** entity |
| `s. others` | pos/neg/neu | Sentiment expressed **toward other** entities in the text |
| `s. society` | pos/neg/neu | Sentiment expressed **toward society** at large |
| `is_tweet` | bool | Whether the text is a tweet (True) or not |
| `is_headline` | bool | Whether the text is a news headline (True) or not |
| `split` | train/eval/test | Official dataset split |

### Statistics

- **Total entries:** 3,829
- **Splits:** train=2,449 · eval=613 · test=767
- **Text types:** 1,563 tweets · 2,266 headlines
- **Language:** Spanish (`es`)
- **Note:** one entry in `s. society` has a trailing space (`"neu "`) — normalize when loading

### Loading the Dataset

```python
import pandas as pd

df = pd.read_csv("financial_dataset_quoted.csv", encoding="utf-8")

# Normalize whitespace in label columns
label_cols = ["s. target", "s. others", "s. society"]
df[label_cols] = df[label_cols].apply(lambda c: c.str.strip())

train = df[df["split"] == "train"]
eval_ = df[df["split"] == "eval"]
test  = df[df["split"] == "test"]
```

## Project Goal

Build a Python pipeline that runs LLM inference (Anthropic Claude or OpenAI GPT) over the
entire **FinancES 2023 / IberLEF** financial sentiment dataset, not just the test split.
The system must be model-agnostic in its output contract, highly configurable, and require
zero code changes to swap providers, models, or prompts.

---

## Repository Layout to Create

```
project/
├── CLAUDE.md                        # this file
├── config.yaml                      # all runtime knobs (model, params, paths, …)
├── prompts/
│   ├── system_subtask1.md           # system prompt for Subtask 1
│   ├── user_subtask1.md             # user-turn template for Subtask 1
│   ├── system_subtask2.md           # system prompt for Subtask 2
│   └── user_subtask2.md             # user-turn template for Subtask 2
├── financial_dataset_quoted.csv     # source data (already present, do NOT modify)
├── src/
│   ├── __init__.py
│   ├── config.py                    # loads & validates config.yaml with pydantic
│   ├── prompts.py                   # loads prompt files, fills Jinja2 placeholders
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py                  # InferenceResult dataclass (shared contract)
│   │   ├── anthropic_client.py      # run_anthropic(messages, cfg) → InferenceResult
│   │   └── openai_client.py         # run_openai(messages, cfg)   → InferenceResult
│   ├── dataset.py                   # loads CSV, yields Row objects
│   ├── pipeline.py                  # orchestrates dataset → prompts → LLM → results
│   └── output.py                    # writes predictions CSV + JSONL + run summary
├── outputs/                         # auto-created; one sub-folder per run
└── requirements.txt
```

---

## Detailed Spec for Each File

### `config.yaml`

Expose **every tunable** under clearly named keys. Nothing is hardcoded in Python.

```yaml
# ── Provider selection ───────────────────────────────────────────────────────
provider: anthropic          # "anthropic" | "openai"

# ── Model identifiers ────────────────────────────────────────────────────────
anthropic:
  model: claude-sonnet-4-20250514
  max_tokens: 512
  temperature: 0.0
  top_p: 1.0
  # top_k: 40              # uncomment to use

openai:
  model: gpt-4.5            # swap freely, e.g. gpt-4o, gpt-4.5
  max_tokens: 512
  temperature: 0.0
  top_p: 1.0
  # frequency_penalty: 0.0
  # presence_penalty: 0.0

# ── Prompt files ─────────────────────────────────────────────────────────────
prompts:
  subtask1:
    system: prompts/system_subtask1.md
    user:   prompts/user_subtask1.md
  subtask2:
    system: prompts/system_subtask2.md
    user:   prompts/user_subtask2.md

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset:
  path: financial_dataset_quoted.csv
  text_column: text             # column holding the headline/post
  id_column: id                 # unique row identifier (adjust to actual header)
  encoding: utf-8

# ── Pipeline behaviour ───────────────────────────────────────────────────────
pipeline:
  subtasks: [1, 2]              # which subtasks to run
  batch_size: 1                 # rows processed concurrently (keep 1 for rate-limit safety)
  max_rows: null                # null = whole dataset; set an int to cap for quick tests
  retry_attempts: 3
  retry_backoff_seconds: 5
  skip_on_error: true           # if false, a single failure aborts the run

# ── Output ───────────────────────────────────────────────────────────────────
output:
  dir: outputs/
  save_csv: true
  save_jsonl: true
  save_summary: true
```

---

### `src/llm/base.py` — Shared Contract

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class InferenceResult:
    """Provider-agnostic result returned by every LLM function."""
    row_id: Any                  # mirrors dataset id_column value
    subtask: int
    raw_response: str            # full text the model produced
    parsed: dict = field(default_factory=dict)   # structured extraction of the answer
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    provider: str = ""
    error: str | None = None     # non-None means the call failed
```

All LLM functions must return `InferenceResult`. The rest of the pipeline never imports
`anthropic` or `openai` directly.

---

### `src/llm/anthropic_client.py`

```python
def run_anthropic(
    messages: list[dict],   # [{"role": "user"|"assistant", "content": str}, …]
    system: str,
    cfg,                    # AnthropicConfig pydantic model
    row_id,
    subtask: int,
) -> InferenceResult:
    """Call Anthropic Messages API; return InferenceResult."""
    ...
```

- Read API key from env var `ANTHROPIC_API_KEY` (never from config).
- Pass all LLM params from `cfg` (model, max_tokens, temperature, top_p, top_k if present).
- On success, populate `raw_response`, `prompt_tokens`, `completion_tokens`, `model`, `provider`.
- On any exception, return an `InferenceResult` with `error` set; do NOT raise.

---

### `src/llm/openai_client.py`

```python
def run_openai(
    messages: list[dict],
    system: str,
    cfg,                    # OpenAIConfig pydantic model
    row_id,
    subtask: int,
) -> InferenceResult:
    """Call OpenAI Chat Completions API; return InferenceResult."""
    ...
```

- Read API key from env var `OPENAI_API_KEY`.
- Mirror the same signature and return contract as `run_anthropic`.
- Pass all optional params only when they are set in `cfg` (avoid sending `null` fields).

---

### `src/config.py`

Use **Pydantic v2** `BaseModel` with `model_validator` / `field_validator`:

- Validate that `text_column` and `id_column` exist in the CSV headers at startup (fail fast).
- Validate that all prompt files referenced in `config.yaml` exist.
- Expose a `get_active_llm_cfg()` helper that returns the provider-specific sub-config.

---

### `src/prompts.py`

- Load `.md` or `.txt` files as raw strings (no special parser needed beyond `pathlib.read_text`).
- Use **Jinja2** to fill `{{ text }}`, `{{ met }}`, etc. placeholders so prompts stay clean.
- Expose `build_messages(system_tpl, user_tpl, row: dict) -> tuple[str, list[dict]]`.

#### Prompt file conventions

- Prompt files are plain Markdown.  
- Available placeholders: `{{ text }}` (the headline), `{{ met }}` (only for Subtask 1 user prompt when MET is needed), plus any column name from the CSV wrapped in `{{ }}`.
- Authors edit only the `.md` files to change prompt strategy; no Python changes needed.

---

### `src/dataset.py`

- Load `financial_dataset_quoted.csv` with **pandas** (`encoding` from config).
- Expose `iter_rows(cfg) -> Iterator[dict]` that yields every row as a plain dict.
- Respect `max_rows` from config.

---

### `src/pipeline.py`

```
for row in iter_rows(cfg):
    for subtask in cfg.pipeline.subtasks:
        system, messages = build_messages(subtask, row)
        result = dispatch_llm(messages, system, cfg, row[id_col], subtask)
        collect(result)
```

- `dispatch_llm` is the **only** place that checks `cfg.provider` and calls either
  `run_anthropic` or `run_openai`.  Everything else is provider-blind.
- Implement retry logic with exponential back-off around `dispatch_llm`.
- Log progress with `tqdm` (one bar per subtask).

---

### `src/output.py`

After the run, write into `outputs/<provider>_<model>_<timestamp>/`:

| File | Contents |
|---|---|
| `predictions_subtask1.csv` | `id, text, raw_response, parsed_met, parsed_polarity_met, error` |
| `predictions_subtask2.csv` | `id, text, raw_response, parsed_polarity_companies, parsed_polarity_consumers, error` |
| `predictions.jsonl` | One JSON object per row containing the full `InferenceResult` |
| `run_summary.json` | model, provider, timestamp, total_rows, errors, total_tokens, config snapshot |

Parsing of `raw_response` into structured fields should be attempted with a simple regex /
JSON extractor; if it fails, leave the field as `null` and log a warning (never crash).

---

## Prompt Files to Create

### `prompts/system_subtask1.md`

Write a clear system prompt that:
- Explains the task: given a Spanish financial headline, (1) identify the Main Economic Target
  (MET — the company, asset, or sector the news is primarily about) and (2) classify the
  sentiment toward that MET as **positive**, **neutral**, or **negative**.
- Instructs the model to respond **only** in valid JSON:
  `{"met": "<entity>", "polarity_met": "<positive|neutral|negative>"}`.
- Forbids any explanation or preamble outside the JSON object.

### `prompts/user_subtask1.md`

```
Headline: {{ text }}
```

### `prompts/system_subtask2.md`

Write a system prompt that:
- Explains the task: given a Spanish financial headline, classify the sentiment it conveys
  toward (1) **other companies** (third-party businesses) and (2) **consumers** (households/society).
- Instructs the model to respond **only** in valid JSON:
  `{"polarity_companies": "<positive|neutral|negative>", "polarity_consumers": "<positive|neutral|negative>"}`.
- Forbids any explanation or preamble outside the JSON object.

### `prompts/user_subtask2.md`

```
Headline: {{ text }}
```

---

## `requirements.txt`

```
anthropic>=0.28
openai>=1.30
pydantic>=2.0
pydantic-settings>=2.0
PyYAML>=6.0
pandas>=2.0
jinja2>=3.1
tqdm>=4.66
tenacity>=8.2        # for retry logic
python-dotenv>=1.0   # load .env for API keys in dev
```

---

## CLI Entry Point

Create `run_inference.py` at the project root:

```bash
python run_inference.py                        # uses config.yaml defaults
python run_inference.py --config config.yaml   # explicit config path
python run_inference.py --provider openai      # override provider at runtime
python run_inference.py --max-rows 50          # quick smoke test
python run_inference.py --subtasks 1           # run only one subtask
```

Use `argparse`. CLI flags override config values (config is the baseline, CLI wins).

---

## Key Design Constraints

1. **No hardcoded values anywhere.** Every string, number, or path lives in `config.yaml`
   or a prompt file.
2. **Single output contract.** `InferenceResult` is the only type that crosses the
   LLM ↔ pipeline boundary. Adding a new provider means adding one new file in `src/llm/`
   and one line in `dispatch_llm`.
3. **Prompt files are the experiment knob.** Researchers iterate by editing `.md` files only.
4. **Fail gracefully.** A bad API response or rate-limit error must not lose prior results.
   Write partial results to disk after every `N` rows (configurable as `output.flush_every`).
5. **Reproducibility.** `run_summary.json` captures the full config snapshot so any run can
   be reproduced exactly.
6. **No modification to the source CSV.** The dataset file is read-only input.

---

## Suggested Implementation Order

1. `config.py` + `config.yaml` — get validation right first.
2. `prompts/` files + `prompts.py` — verify Jinja2 rendering.
3. `dataset.py` — sanity check column names against the real CSV.
4. `llm/base.py` → `llm/anthropic_client.py` → `llm/openai_client.py`.
5. `pipeline.py` — wire it together with a `--max-rows 3` smoke test.
6. `output.py` — final artefacts.
7. `run_inference.py` CLI.

---

## Notes & Suggestions

- **Rate limits**: Anthropic and OpenAI both throttle heavily on free/tier-1 keys.
  Default `batch_size: 1` with back-off is safe. Consider adding `pipeline.requests_per_minute`
  as a config key and implementing a token-bucket limiter in `pipeline.py`.
- **Cost tracking**: Log `prompt_tokens + completion_tokens` per row and aggregate in the
  summary. Add an `--estimate-cost` dry-run mode that prints expected cost before any API call.
- **Determinism**: Set `temperature: 0.0` by default for reproducible benchmark results.
  Document this clearly in the README.
- **Extending to new providers**: To add, e.g., Google Gemini, create `src/llm/gemini_client.py`
  with the same function signature, add a `gemini:` block to `config.yaml`, and register it
  in `dispatch_llm`. No other file changes needed.
- **Gold labels**: If the CSV contains ground-truth columns, `output.py` can optionally compute
  macro-F1 and per-class metrics inline and append them to `run_summary.json`.