"""Main analysis pipeline: fetch → analyse → recommend → notify."""

from loguru import logger
from datetime import datetime, timezone
from config import settings
from src.data.news_fetcher import fetch_all_news
from src.data.market_data import get_snapshots
from src.data.cache import load_news, save_news, load_snapshots, save_snapshots, load_latest_snapshots
from src.data.trending import get_trending_tickers
from src.signals.aggregator import build_signals
from src.analysis.claude_analyst import generate_recommendations
from src.notifications.email_sender import send_recommendations
from src.performance.tracker import record_new_trades, update_open_trades, log_performance_summary, get_performance_for_email
from src.charts.report import save_html_report


def run_pipeline(send_email: bool = False) -> None:
    """Execute the full analysis pipeline."""
    start = datetime.now(timezone.utc)
    logger.info(f"Pipeline started at {start.strftime('%Y-%m-%d %H:%M UTC')}")

    tickers = settings.stocks_list
    sectors = settings.sectors_list

    # Expand universe with trending/hot tickers discovered from online sources
    logger.info("Discovering trending tickers...")
    all_tickers = get_trending_tickers(tickers, sectors)

    # 1. Fetch news (use cache if available for this hour)
    logger.info("Step 1/4: Fetching news...")
    articles = load_news()
    if articles is None:
        articles = fetch_all_news(tickers, sectors)
        save_news(articles)
    else:
        logger.info("Step 1/4: Using cached news (skip live fetch)")

    # 2. Fetch market data
    logger.info("Step 2/4: Fetching market snapshots...")
    snapshots = load_snapshots()   # current-hour cache (always checked first)
    if snapshots is not None:
        logger.info("Step 2/4: Using cached snapshots (skip live fetch)")
    elif not settings.enable_market_data:
        # Live fetch disabled — fall back to the most recent historical cache
        snapshots = load_latest_snapshots()
        if snapshots:
            logger.info("Step 2/4: ENABLE_MARKET_DATA=false — using latest historical snapshots")
        else:
            logger.warning("Step 2/4: ENABLE_MARKET_DATA=false and no historical cache found — news-only mode")
            snapshots = []
    else:
        snapshots = get_snapshots(all_tickers)
        save_snapshots(snapshots)
        if not snapshots:
            logger.warning("No market data retrieved — continuing with news-only signals.")

    # 3. Build signals (news sentiment only)
    logger.info("Step 3/4: Building news-based signals...")
    signals = build_signals(all_tickers, articles)

    # 4. Generate final recommendations via Claude
    logger.info("Step 4/4: Generating recommendations...")
    recommendations = generate_recommendations(signals)

    # Only surface BUY and SELL as actionable
    actionable = [r for r in recommendations if r.action in ("BUY", "SELL")]

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info(f"Pipeline complete in {elapsed:.1f}s | {len(all_tickers)} tickers | {len(articles)} articles")

    # Log every recommendation
    _log_recommendations(recommendations)

    # Performance tracking
    update_open_trades()          # refresh prices + close expired positions
    record_new_trades(actionable) # open new trades for today's signals
    log_performance_summary()     # write P&L to log
    perf = get_performance_for_email()

    # Generate interactive HTML report (saved to logs/)
    logger.info("Generating charts and HTML report...")
    ts_label = start.strftime("%Y-%m-%d_%H%M")
    report_path = save_html_report(recommendations, performance=perf, label=ts_label)
    if report_path:
        logger.info(f"Report: {report_path.resolve()}")

    # Print actionable summary to console
    _print_summary(actionable)

    # Send email if configured or explicitly requested
    email_configured = bool(settings.smtp_user and settings.email_recipients)
    if (send_email or email_configured) and actionable:
        send_recommendations(
            actionable,
            total_analysed=len(all_tickers),
            performance=perf,
            all_recommendations=recommendations,
        )
    elif not actionable:
        logger.info("No BUY/SELL signals today — no email sent.")


def _log_recommendations(recommendations) -> None:
    """Write every recommendation to the log file."""
    stocks = [r for r in recommendations if r.type == "STOCK"]
    etfs = [r for r in recommendations if r.type == "ETF"]

    logger.info("=" * 60)
    logger.info(f"RECOMMENDATIONS — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    logger.info("=" * 60)

    if stocks:
        logger.info("--- STOCKS ---")
        for r in sorted(stocks, key=lambda x: x.confidence, reverse=True):
            logger.info(f"  {r.ticker:<6} {r.action:<5} {r.direction:<8} conf={r.confidence:.0%}  [{r.time_horizon}]")
            logger.info(f"    {r.rationale}")

    if etfs:
        logger.info("--- ETFs / MARKETS ---")
        for r in sorted(etfs, key=lambda x: x.confidence, reverse=True):
            logger.info(f"  {r.ticker:<6} {r.action:<5} {r.direction:<8} conf={r.confidence:.0%}  [{r.time_horizon}]")
            logger.info(f"    {r.rationale}")

    logger.info("=" * 60)


def _print_summary(recommendations) -> None:
    """Print BUY/SELL signals to the console, grouped by type."""
    if not recommendations:
        print("\n  No BUY/SELL signals today.\n")
        return

    stocks = [r for r in recommendations if r.type == "STOCK"]
    etfs = [r for r in recommendations if r.type == "ETF"]

    print("\n" + "=" * 60)
    print(f"  ACTIONABLE SIGNALS  ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})")
    print("=" * 60)

    if stocks:
        print("\n  STOCKS")
        print("  " + "-" * 40)
        for rec in sorted(stocks, key=lambda x: x.confidence, reverse=True):
            bar = "▲" if rec.direction == "BULLISH" else "▼"
            print(f"  {bar} {rec.ticker:<6} {rec.action:<5}  conf={rec.confidence:.0%}  [{rec.time_horizon}]")
            print(f"    {rec.rationale}")
            print()

    if etfs:
        print("  ETFs / MARKETS")
        print("  " + "-" * 40)
        for rec in sorted(etfs, key=lambda x: x.confidence, reverse=True):
            bar = "▲" if rec.direction == "BULLISH" else "▼"
            print(f"  {bar} {rec.ticker:<6} {rec.action:<5}  conf={rec.confidence:.0%}  [{rec.time_horizon}]")
            print(f"    {rec.rationale}")
            print()

    print("=" * 60 + "\n")
