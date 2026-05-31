#!/usr/bin/env python3
"""
LLM Trader — Stock market direction predictor.

Usage:
    python main.py              # Run once now (no email)
    python main.py --email      # Run once now (with email)
    python main.py --schedule   # Start scheduled runner
    python main.py --report     # Show realized performance of past calls
    python main.py --report --email   # ...and email the performance report
"""

import argparse
import sys
from loguru import logger


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        level="INFO",
    )
    logger.add(
        "logs/llm_trader_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
    )


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="LLM Trader — AI-powered market analysis")
    parser.add_argument("--email", action="store_true", help="Send email after analysis")
    parser.add_argument("--schedule", action="store_true", help="Run on schedule (market hours)")
    parser.add_argument("--report", action="store_true", help="Show realized performance of past recommendations")
    args = parser.parse_args()

    if args.report:
        from src.performance.scorer import score_matured, build_scorecard
        from src.pipeline import _print_scorecard
        score_matured()
        scorecard = build_scorecard()
        print("\n" + "=" * 60)
        print("  LLM TRADER — PERFORMANCE REPORT")
        print("=" * 60)
        _print_scorecard(scorecard)
        if args.email:
            from src.notifications.email_sender import send_performance_report
            send_performance_report(scorecard)
    elif args.schedule:
        from src.scheduler.runner import start_scheduler
        start_scheduler()
    else:
        from src.pipeline import run_pipeline
        run_pipeline(send_email=args.email)


if __name__ == "__main__":
    main()
