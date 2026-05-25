# CLAUDE.md

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

## Research Task Context

The task is **multi-dimensional financial ABSA** in Spanish:
- Given a text and its target entity, predict sentiment along **three axes** simultaneously: toward the target, toward others, and toward society.
- Baseline approaches include fine-tuning Spanish/multilingual BERT-family models (e.g., `PlanTL-GOB-ES/roberta-base-bne`, `dccuchile/bert-base-spanish-wwm-cased`, `xlm-roberta-base`).
- Evaluation metric used in FinancES 2023 shared task is **macro-averaged F1** per sentiment dimension.
