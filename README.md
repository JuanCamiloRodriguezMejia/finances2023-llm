# FinancES 2023 — LLM Inference Pipeline

A Python pipeline that runs LLM inference (Anthropic Claude or OpenAI GPT) over the **FinancES 2023 / IberLEF** Spanish financial sentiment dataset. The system is model-agnostic: swapping providers or models requires only a change to `config.yaml`.

---

## Dataset

**FinancES 2023** is a multi-aspect, aspect-based sentiment analysis (ABSA) benchmark for Spanish financial news published as part of IberLEF 2023.

| Split | Rows | Text type |
|-------|------|-----------|
| train | 2 449 | tweets + headlines |
| eval  | 613  | tweets + headlines |
| test  | 767  | tweets + headlines |
| **Total** | **3 829** | — |

Each row contains a headline/tweet in Spanish, a named entity (*target*), and three sentiment labels:

| Column | Values | Description |
|--------|--------|-------------|
| `s. target` | pos / neg / neu | Sentiment toward the main target entity |
| `s. others` | pos / neg / neu | Sentiment toward other companies |
| `s. society` | pos / neg / neu | Sentiment toward consumers / society |

The pipeline maps these to two subtasks:

- **Subtask 1** — identify the Main Economic Target (MET) and classify sentiment toward it (`s. target`).
- **Subtask 2** — classify sentiment toward other companies (`s. others`) and toward society (`s. society`).

---

## Project structure

```
├── run_inference.py          CLI entry point
├── config.yaml               all runtime knobs (model, paths, pipeline, output)
├── requirements.txt
├── .env.example              template for API keys
│
├── prompts/                  plain-Markdown Jinja2 templates — edit to change strategy
│   ├── system_subtask1.md
│   ├── user_subtask1.md
│   ├── system_subtask2.md
│   └── user_subtask2.md
│
├── src/
│   ├── config.py             Pydantic v2 loader — validates paths and CSV columns at startup
│   ├── prompts.py            Jinja2 renderer → (system_str, messages_list)
│   ├── dataset.py            CSV loader, normalises label whitespace, yields row dicts
│   ├── pipeline.py           orchestration + tenacity retry + optional rate limiter
│   ├── output.py             streaming JSONL writer + CSV flush + run_summary.json
│   └── llm/
│       ├── base.py           InferenceResult dataclass — shared contract across providers
│       ├── anthropic_client.py
│       └── openai_client.py
│
├── outputs/                  auto-created; one sub-folder per run (git-ignored)
└── financial_dataset_quoted.csv   source data — read-only, never modified
```

---

## Setup

### 1 — Activate the virtual environment

```powershell
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

### 2 — Install dependencies

```powershell
pip install -r requirements.txt
```

### 3 — Set your API key

Copy `.env.example` to `.env` and fill in the key for the provider you plan to use:

```powershell
copy .env.example .env
```

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...        # only needed if using --provider openai
```

The script loads `.env` automatically on startup via `python-dotenv`. You can also export the variable in your shell instead.

---

## Usage

### Dry-run (no API calls)

Always run this first to check row counts and estimated token usage before spending credits:

```powershell
python run_inference.py --estimate-cost
python run_inference.py --estimate-cost --max-rows 100 --provider openai
```

### Run inference

```powershell
# 3-row smoke test, Subtask 1 only
python run_inference.py --max-rows 3 --subtasks 1

# Full dataset, both subtasks, Anthropic (default)
python run_inference.py

# Full dataset, OpenAI
python run_inference.py --provider openai

# Only Subtask 2, capped at 200 rows
python run_inference.py --subtasks 2 --max-rows 200

# Skip the first 2449 rows (i.e. skip the train split) and run the rest
python run_inference.py --offset 2449

# Run only rows 2449–3061 (eval split: 613 rows)
python run_inference.py --offset 2449 --max-rows 613

# Verbose logging
python run_inference.py --log-level DEBUG
```

> **`--offset` + `--max-rows` together** — offset is applied first, then the row cap.  
> So `--offset 100 --max-rows 50` processes rows 100–149 (0-indexed).

### All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--config PATH` | `config.yaml` | Path to configuration file |
| `--provider {anthropic,openai}` | from config | Override the provider |
| `--offset N` | `0` | Skip the first N rows before processing |
| `--max-rows N` | from config (`null` = all) | Cap rows after the offset |
| `--subtasks {1,2} …` | from config (`[1,2]`) | Which subtasks to run |
| `--estimate-cost` | off | Dry-run: print estimate, no API calls |
| `--log-level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

CLI flags override `config.yaml` values — the config file is the baseline.

---

## Output artefacts

Each run creates a timestamped folder under `outputs/`:

```
outputs/anthropic_claude-sonnet-4-20250514_20250524_143012/
├── predictions_subtask1.csv    id, text, raw_response, parsed_met, parsed_polarity_met, error
├── predictions_subtask2.csv    id, text, raw_response, parsed_polarity_companies, parsed_polarity_consumers, error
├── predictions.jsonl           one JSON object per call (full InferenceResult)
└── run_summary.json            model, timestamp, token counts, error list, config snapshot
```

Partial results are flushed to disk every `output.flush_every` rows (default 50), so a crash mid-run does not lose all progress.

---

## Configuration reference

All knobs live in [`config.yaml`](config.yaml). Key sections:

```yaml
provider: anthropic          # "anthropic" | "openai"

anthropic:
  model: claude-sonnet-4-20250514
  temperature: 0.0           # keep at 0.0 for reproducible benchmarks
  max_tokens: 512

pipeline:
  subtasks: [1, 2]
  max_rows: null             # null = entire dataset
  retry_attempts: 3
  retry_backoff_seconds: 5   # base for exponential back-off
  skip_on_error: true        # false → abort on first failure
  requests_per_minute: null  # null = no throttle; set an int to rate-limit

output:
  flush_every: 50            # write partial CSVs every N completed rows
```

---

## Customising prompts

Prompts are plain Markdown files in `prompts/`. Edit them to change the instruction strategy — no Python changes needed.

Available Jinja2 placeholders inside any prompt file:

| Placeholder | Source |
|-------------|--------|
| `{{ text }}` | The headline/tweet text |
| `{{ target }}` | The gold target entity from the dataset |
| `{{ id }}` | Row ID |
| Any CSV column | e.g. `{{ split }}`, `{{ is_tweet }}` |

The model is instructed to respond with a JSON object only. If the response cannot be parsed, the `parsed_*` columns in the output CSV will be `null` and a warning is logged — the run never crashes on a bad response.

---

## Adding a new provider

1. Create `src/llm/gemini_client.py` (or similar) with the same signature:
   ```python
   def run_gemini(messages, system, cfg, row_id, subtask, text="") -> InferenceResult: ...
   ```
2. Add a `gemini:` block to `config.yaml`.
3. Register it in the `dispatch_llm` function in [`src/pipeline.py`](src/pipeline.py) — one `elif` branch.

No other files need to change.

---

## Evaluation notes

The official FinancES 2023 metric is **macro-averaged F1** computed per sentiment dimension. The gold labels in the CSV are:

| Pipeline output | Gold column |
|-----------------|-------------|
| `parsed_polarity_met` | `s. target` |
| `parsed_polarity_companies` | `s. others` |
| `parsed_polarity_consumers` | `s. society` |

Label values predicted by the model (`positive` / `neutral` / `negative`) differ in case and length from the gold labels (`pos` / `neg` / `neu`) — normalise both sides before computing metrics:

```python
mapping = {"positive": "pos", "negative": "neg", "neutral": "neu"}
df["pred_norm"] = df["parsed_polarity_met"].str.lower().map(mapping)
```

---

## Cost considerations

- **Temperature 0.0** is set by default for reproducibility. Do not change it for benchmark runs.
- At ~210 tokens per call × 2 subtasks × 3 829 rows ≈ **1.6 M tokens** per full run.
- Use `--max-rows 50` to validate prompt quality cheaply before committing to the full dataset.
- `run_summary.json` logs exact token usage for every run so costs can be tracked precisely.
- Rate-limit safe defaults: `batch_size: 1` with exponential back-off. If you have a higher-tier API plan, set `pipeline.requests_per_minute` to speed up the run.
