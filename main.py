#!/usr/bin/env python3
"""
LLM Trader — Stock market direction predictor.

Usage:
    python main.py              # Run once now (no email)
    python main.py --email      # Run once now (with email)
    python main.py --schedule   # Start scheduled runner (one process)
    python main.py --supervise  # Start the scheduler under an auto-restart supervisor (production)
    python main.py --dashboard  # Launch the monitoring dashboard (rationale · methods · returns)
"""

import argparse
import sys
from loguru import logger

from src.log_redaction import redaction_filter

# Ensure UTF-8 output on Windows terminals (needed for ▲/▼ arrows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def setup_logging() -> None:
    logger.remove()
    # redaction_filter scrubs API keys/tokens from every record before it reaches
    # a sink, so log files never contain plaintext credentials (e.g. the api_key
    # leaked in upstream httpx exception URLs).
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="INFO",
        filter=redaction_filter,
    )
    logger.add(
        "logs/llm_trader_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        filter=redaction_filter,
    )


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="LLM Trader — AI-powered market analysis")
    parser.add_argument("--email", action="store_true", help="Send email after analysis")
    parser.add_argument("--schedule", action="store_true", help="Run on schedule (market hours)")
    parser.add_argument("--supervise", action="store_true",
                        help="Run the scheduler under an auto-restart supervisor (relaunches it if the process dies)")
    parser.add_argument("--dashboard", action="store_true", help="Launch the monitoring dashboard (Plotly Dash)")
    parser.add_argument("--backfill", action="store_true",
                        help="Backfill OHLCV caches for the whole universe via Massive/Polygon, then exit")
    parser.add_argument("--backfill-30m", action="store_true",
                        help="With --backfill, also warm the 30-min cache (heavier)")
    parser.add_argument("--rescore", action="store_true",
                        help="Rescore stored history with the CURRENT code: replay "
                             "method scores + market conditions, re-walk the "
                             "point-in-time weights, recompute combined_score/"
                             "confidence. Detects what changed and repairs only "
                             "that unless --force")
    parser.add_argument("--force", action="store_true",
                        help="With --rescore, rebuild everything regardless of "
                             "detected changes")
    parser.add_argument("--check", action="store_true",
                        help="With --rescore, report what WOULD be done and exit")
    args = parser.parse_args()

    if args.rescore:
        from src.analysis.refactor import run_refactor
        res = run_refactor(apply=not args.check, force=args.force)
        p = res["plan"]
        print(f"\nchanged           : {p['changed'] or '(none)'}")
        print(f"  regenerate      : {p['regenerate'] or '(none)'}")
        print(f"  mask candidates : {p['mask_candidates'] or '(none)'}")
        print(f"  derived changed : {p['derived_changed'] or '(none)'}")
        if p["unmapped"]:
            print(f"  UNMAPPED (blind): {p['unmapped']}")
        for s in res["steps"]:
            print(f"  {s['step']:<12}{s['status']:<9}{s['detail']}")
        if args.check:
            print("\n(check only — rerun without --check to apply)")
        elif not res["ok"]:
            print("\nA step FAILED — fingerprints were not advanced, so this "
                  "retries rather than silently marking the DB up to date.")
            raise SystemExit(1)
    elif args.backfill:
        from src.data.backfill import backfill
        backfill(with_30m=args.backfill_30m)
    elif args.dashboard:
        from dashboard.app import run as run_dashboard
        run_dashboard()
    elif args.supervise:
        from src.scheduler.supervisor import run_supervised
        run_supervised()
    elif args.schedule:
        from src.scheduler.runner import start_scheduler
        start_scheduler()
    else:
        from src.pipeline import run_pipeline
        run_pipeline(send_email=args.email)


if __name__ == "__main__":
    main()
