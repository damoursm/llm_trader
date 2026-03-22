"""Main analysis pipeline: fetch → analyse → recommend → notify."""

from loguru import logger
from datetime import datetime, timezone
from config import settings
from src.data.news_fetcher import fetch_all_news
from src.data.market_data import get_snapshots
from src.data.cache import load_news, save_news, load_snapshots, save_snapshots
from src.signals.aggregator import build_signals
from src.analysis.claude_analyst import generate_recommendations
from src.notifications.email_sender import send_recommendations


def run_pipeline(send_email: bool = True) -> None:
    """Execute the full analysis pipeline."""
    start = datetime.now(timezone.utc)
    logger.info(f"Pipeline started at {start.strftime('%Y-%m-%d %H:%M UTC')}")

    tickers = settings.stocks_list
    sectors = settings.sectors_list
    all_tickers = tickers + sectors

    # 1. Fetch news (use cache if available for this hour)
    logger.info("Step 1/4: Fetching news...")
    articles = load_news()
    if articles is None:
        articles = fetch_all_news(tickers, sectors)
        save_news(articles)
    else:
        logger.info("Step 1/4: Using cached news (skip live fetch)")

    # 2. Fetch market data (use cache if available for this hour)
    logger.info("Step 2/4: Fetching market snapshots...")
    snapshots = load_snapshots()
    if snapshots is None:
        snapshots = get_snapshots(all_tickers)
        save_snapshots(snapshots)
    else:
        logger.info("Step 2/4: Using cached snapshots (skip live fetch)")

    if not snapshots:
        logger.warning("No market data retrieved — continuing with news-only signals.")

    # 3. Build signals (news sentiment only)
    logger.info("Step 3/4: Building news-based signals...")
    signals = build_signals(all_tickers, articles)

    # 4. Generate final recommendations via Claude
    logger.info("Step 4/4: Generating recommendations...")
    recommendations = generate_recommendations(signals)

    # Filter to actionable recommendations only (exclude low-confidence HOLDs)
    actionable = [r for r in recommendations if not (r.action == "HOLD" and r.confidence < 0.4)]

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info(
        f"Pipeline complete in {elapsed:.1f}s | "
        f"{len(snapshots)} tickers | {len(articles)} articles | "
        f"{len(actionable)} actionable recommendations"
    )

    # Print to console
    _print_summary(actionable)

    # Send email
    if send_email and actionable:
        send_recommendations(actionable, total_analysed=len(snapshots))


def _print_summary(recommendations) -> None:
    print("\n" + "=" * 60)
    print(f"  RECOMMENDATIONS  ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})")
    print("=" * 60)
    for rec in recommendations:
        bar = "▲" if rec.direction == "BULLISH" else ("▼" if rec.direction == "BEARISH" else "◆")
        print(f"  {bar} {rec.ticker:<6} {rec.action:<5}  conf={rec.confidence:.0%}  {rec.direction}")
        print(f"    {rec.rationale[:100]}")
        print()
    print("=" * 60 + "\n")
