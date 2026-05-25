#!/usr/bin/env python3
"""
CLI entry point for LLM inference over the FinancES 2023 dataset.

Usage examples
--------------
python run_inference.py                          # use config.yaml defaults
python run_inference.py --config config.yaml     # explicit config path
python run_inference.py --provider openai        # override provider
python run_inference.py --max-rows 50            # quick smoke test
python run_inference.py --subtasks 1             # run only Subtask 1
python run_inference.py --estimate-cost          # dry-run: show token estimate
python run_inference.py --log-level DEBUG        # verbose output
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Load .env (ANTHROPIC_API_KEY / OPENAI_API_KEY) before anything else
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on real env vars


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LLM inference over the FinancES 2023 financial sentiment dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default=None,
        help="Override the provider set in config.yaml",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of dataset rows (handy for smoke tests)",
    )
    parser.add_argument(
        "--subtasks",
        type=int,
        nargs="+",
        choices=[1, 2],
        default=None,
        metavar="{1,2}",
        help="Which subtasks to run (default: both)",
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Dry-run: print estimated row/call/token counts then exit (no API calls)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _apply_overrides(cfg, args: argparse.Namespace) -> None:
    """Mutate cfg in-place with any CLI flags the user provided."""
    if args.provider:
        cfg.provider = args.provider
    if args.max_rows is not None:
        cfg.pipeline.max_rows = args.max_rows
    if args.subtasks:
        cfg.pipeline.subtasks = args.subtasks


def _print_cost_estimate(cfg) -> None:
    from src.dataset import iter_rows

    rows = list(iter_rows(cfg.dataset, cfg.pipeline))
    n_subtasks = len(cfg.pipeline.subtasks)
    n_calls = len(rows) * n_subtasks
    # Rough estimate: ~150 tokens system prompt + ~30 tokens headline + ~30 tokens response
    avg_tokens = 210
    est_total = n_calls * avg_tokens

    model = cfg.get_active_llm_cfg().model
    print("\n── Cost Estimate (dry-run) ────────────────────────────────")
    print(f"  Config          : {args_global.config}")
    print(f"  Provider        : {cfg.provider}")
    print(f"  Model           : {model}")
    print(f"  Dataset rows    : {len(rows)}")
    print(f"  Subtasks        : {cfg.pipeline.subtasks}")
    print(f"  Total API calls : {n_calls:,}")
    print(f"  Est. tokens     : ~{est_total:,}  (≈{avg_tokens} tokens/call)")
    print("────────────────────────────────────────────────────────────\n")


# Module-level ref for the dry-run printer (set in main)
args_global: argparse.Namespace | None = None


def main() -> None:
    global args_global

    parser = _build_parser()
    args = parser.parse_args()
    args_global = args

    _setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Ensure src/ is importable when run from the project root
    sys.path.insert(0, str(Path(__file__).parent))

    from src.config import load_config

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    _apply_overrides(cfg, args)

    # ── Dry-run mode ─────────────────────────────────────────────────────────
    if args.estimate_cost:
        _print_cost_estimate(cfg)
        return

    # ── Live inference ────────────────────────────────────────────────────────
    logger.info(
        "Starting  provider=%-10s  model=%s  subtasks=%s  max_rows=%s",
        cfg.provider,
        cfg.get_active_llm_cfg().model,
        cfg.pipeline.subtasks,
        cfg.pipeline.max_rows if cfg.pipeline.max_rows is not None else "all",
    )

    from src.pipeline import run_pipeline

    try:
        results = run_pipeline(cfg)
    except RuntimeError as exc:
        logger.error("Pipeline aborted: %s", exc)
        sys.exit(1)

    # ── Summary ───────────────────────────────────────────────────────────────
    errors = [r for r in results if r.error]
    total_tokens = sum(r.total_tokens for r in results)

    logger.info(
        "Done — %d calls  |  %d errors  |  %d tokens used",
        len(results),
        len(errors),
        total_tokens,
    )

    if errors:
        logger.warning("First %d error(s):", min(len(errors), 5))
        for r in errors[:5]:
            logger.warning("  row=%-6s  subtask=%d  %s", r.row_id, r.subtask, r.error)


if __name__ == "__main__":
    main()
