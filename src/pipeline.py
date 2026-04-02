"""Main analysis pipeline: fetch → analyse → recommend → notify."""

from loguru import logger
from datetime import datetime, timezone
from src.utils import now_et, fmt_et
from config import settings
from src.data.news_fetcher import fetch_all_news
from src.data.market_data import get_snapshots
from src.data.cache import load_news, save_news, load_snapshots, save_snapshots, load_latest_snapshots
from src.data.trending import get_trending_tickers
from src.signals.aggregator import build_signals
from src.analysis.claude_analyst import generate_recommendations
from src.data.insider_trades import fetch_insider_trades, get_tickers_from_smart_money
from src.notifications.email_sender import send_recommendations
from src.performance.tracker import record_new_trades, update_open_trades, log_performance_summary, get_performance_for_email
from src.charts.report import save_html_report


def run_pipeline(send_email: bool = False) -> None:
    """Execute the full analysis pipeline."""
    start = datetime.now(timezone.utc)
    logger.info(f"Pipeline started at {fmt_et(start)}")

    tickers    = settings.stocks_list
    sectors    = settings.sectors_list
    commodities = settings.commodities_list

    # Expand universe with trending/hot tickers discovered from online sources
    logger.info("Discovering trending tickers...")
    all_tickers = get_trending_tickers(tickers, sectors)

    # Pin commodities — always analysed regardless of trending
    new_commodities = [t for t in commodities if t not in all_tickers]
    if new_commodities:
        all_tickers = all_tickers + new_commodities
        logger.info(f"Pinned commodities: {new_commodities}")

    # 1. Fetch news (use cache if available for this hour)
    logger.info("Step 1/5: Fetching news...")
    articles = load_news()
    if articles is None:
        articles = fetch_all_news(tickers, sectors)
        save_news(articles)
    else:
        logger.info("Step 1/4: Using cached news (skip live fetch)")

    # 2. Fetch market data
    logger.info("Step 2/5: Fetching market snapshots...")
    snapshots = load_snapshots()   # current-hour cache (always checked first)
    if snapshots is not None:
        logger.info("Step 2/5: Using cached snapshots (skip live fetch)")
    elif not settings.enable_market_data:
        # Live fetch disabled — fall back to the most recent historical cache
        snapshots = load_latest_snapshots()
        if snapshots:
            logger.info("Step 2/5: ENABLE_MARKET_DATA=false — using latest historical snapshots")
        else:
            logger.warning("Step 2/5: ENABLE_MARKET_DATA=false and no historical cache found — news-only mode")
            snapshots = []
    else:
        snapshots = get_snapshots(all_tickers)
        save_snapshots(snapshots)
        if not snapshots:
            logger.warning("No market data retrieved — continuing with news-only signals.")

    # 3a. Fetch insider / politician trades (House, Senate, EDGAR Form 4)
    smart_money = []
    if settings.enable_insider_trades:
        logger.info("Step 3a: Fetching insider & politician trades...")
        smart_money.extend(fetch_insider_trades(all_tickers))
    else:
        logger.info("Step 3a: Insider trades disabled (ENABLE_INSIDER_TRADES=false)")

    # 3b. Unusual options flow
    if settings.enable_options_flow:
        logger.info("Step 3b: Scanning options chains for unusual sweeps...")
        from src.data.options_flow import fetch_options_flow
        smart_money.extend(fetch_options_flow(all_tickers))
    else:
        logger.info("Step 3b: Options flow disabled (ENABLE_OPTIONS_FLOW=false)")

    # 3c. SEC EDGAR filings (13D/13G activist, Form 144, 13F superinvestors)
    # Discovers tickers from the filings themselves — no predefined list needed
    if settings.enable_sec_filings:
        logger.info("Step 3c: Fetching SEC filings (13D/13G, Form 144, 13F)...")
        from src.data.sec_filings import fetch_sec_filings
        smart_money.extend(fetch_sec_filings())
    else:
        logger.info("Step 3c: SEC filings disabled (ENABLE_SEC_FILINGS=false)")

    # Surface tickers with strong smart money conviction not already in universe
    # Pass empty list (not None) when any strategy is enabled — so the section
    # renders in output/email even on days with no signals found.
    any_smart_money_enabled = (
        settings.enable_insider_trades or
        settings.enable_options_flow or
        settings.enable_sec_filings
    )
    insider_trades = smart_money if (smart_money or any_smart_money_enabled) else None
    if smart_money:
        smart_tickers = get_tickers_from_smart_money(smart_money)
        new_from_smart = [t for t in smart_tickers if t not in all_tickers]
        if new_from_smart:
            logger.info(f"Step 3: Adding {new_from_smart} to universe from smart money signals")
            all_tickers = all_tickers + new_from_smart

    # 4. Build signals (all enabled methods)
    logger.info("Step 4/5: Building signals...")
    signals = build_signals(all_tickers, articles, insider_trades=insider_trades)

    # 5. Generate final recommendations via Claude
    logger.info("Step 5/5: Generating recommendations...")
    recommendations = generate_recommendations(signals, insider_trades=insider_trades)

    # Only surface BUY and SELL as actionable, with minimum confidence guard
    actionable = [r for r in recommendations if r.action in ("BUY", "SELL") and r.confidence >= 0.78]

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
    if settings.enable_charts:
        logger.info("Generating charts and HTML report...")
        ts_label = start.strftime("%Y-%m-%d_%H%M")
        report_path = save_html_report(recommendations, performance=perf, label=ts_label)
        if report_path:
            logger.info(f"Report: {report_path.resolve()}")
    else:
        logger.info("Charts disabled (ENABLE_CHARTS=false) — skipping HTML report")

    # Print actionable summary to console
    _print_summary(actionable, smart_money or [])

    # Send daily report email whenever SMTP is configured or --email flag is set.
    # Always send — even with no BUY/SELL signals the HOLD/WATCH table,
    # smart money section, and performance summary are useful.
    email_configured = bool(settings.smtp_user and settings.email_recipients)
    if send_email or email_configured:
        send_recommendations(
            actionable,
            total_analysed=len(all_tickers),
            performance=perf,
            all_recommendations=recommendations,
            insider_trades=insider_trades,
            signals=signals,
            articles=articles,
        )
    else:
        logger.info("Email not configured — skipping.")


def _log_recommendations(recommendations) -> None:
    """Write every recommendation to the log file."""
    stocks = [r for r in recommendations if r.type == "STOCK"]
    etfs = [r for r in recommendations if r.type == "ETF"]

    logger.info("=" * 60)
    logger.info(f"RECOMMENDATIONS — {fmt_et(now_et())}")
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


def _print_summary(recommendations, smart_money=None) -> None:
    """Print BUY/SELL signals and smart money signals to the console."""
    print("\n" + "=" * 60)
    print(f"  ACTIONABLE SIGNALS  ({fmt_et(now_et())})")
    print("=" * 60)

    if not recommendations:
        print("\n  No BUY/SELL signals today.")
    else:
        stocks = [r for r in recommendations if r.type == "STOCK"]
        etfs   = [r for r in recommendations if r.type == "ETF"]

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

    if smart_money is not None:
        print("  SMART MONEY SIGNALS")
        print("  " + "-" * 40)
        if not smart_money:
            print("  No unusual smart money activity detected.")
        else:
            # Group by ticker, sort bullish first
            from collections import defaultdict
            by_ticker = defaultdict(list)
            for t in smart_money:
                by_ticker[t.ticker].append(t)

            for ticker in sorted(by_ticker):
                trades = sorted(by_ticker[ticker], key=lambda t: not t.is_bullish)
                sigs = ", ".join(
                    f"{'[+]' if t.is_bullish else '[-]'} {t.action_label} ({t.trader_name})"
                    for t in trades[:3]
                )
                print(f"  {ticker:<6}  {sigs}")
        print()

    print("=" * 60 + "\n")
