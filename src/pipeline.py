"""Main analysis pipeline: fetch → analyse → recommend → notify."""

from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from datetime import datetime, timezone

from config import settings
from src.utils import now_et, fmt_et
from src.data.news_fetcher import fetch_all_news
from src.data.market_data import get_snapshots
from src.data.cache import load_news, save_news, load_snapshots, save_snapshots, load_latest_snapshots
from src.data.trending import get_trending_tickers
from src.signals.aggregator import build_signals
from src.analysis.claude_analyst import generate_recommendations
from src.data.insider_trades import fetch_insider_trades, get_tickers_from_smart_money
from src.data.eight_k import fetch_8k_articles
from src.data.google_trends import fetch_google_trends
from src.data.reddit_sentiment import fetch_reddit_sentiment
from src.data.analyst_ratings import fetch_analyst_ratings
from src.data.earnings import fetch_earnings_surprises, fetch_earnings_context
from src.data.short_interest import fetch_short_interest
from src.data.options_flow import fetch_options_flow
from src.data.sec_filings import fetch_sec_filings
from src.data.fred import fetch_macro_context
from src.data.cot import fetch_cot_context
from src.data.ipo_pipeline import fetch_ipo_context
from src.data.vix import fetch_vix_context
from src.data.put_call import fetch_put_call_context
from src.notifications.email_sender import send_recommendations
from src.performance.tracker import record_new_trades, update_open_trades, log_performance_summary, get_performance_for_email
from src.charts.report import save_html_report


# ---------------------------------------------------------------------------
# Fetch helpers (one per data source, called from the thread pool)
# ---------------------------------------------------------------------------

def _safe(label: str, fn, *args, **kwargs):
    """Run fn; log warning and return None on any exception."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.warning(f"[{label}] fetch failed: {e}")
        return None


def _fetch_news(tickers, sectors):
    articles = load_news()
    if articles is None:
        articles = fetch_all_news(tickers, sectors)
        save_news(articles)
    else:
        logger.info("[news] Using cached articles")
    return articles


def _fetch_snapshots(all_tickers):
    snapshots = load_snapshots()
    if snapshots is not None:
        logger.info("[snapshots] Using cache")
        return snapshots
    if not settings.enable_market_data:
        snapshots = load_latest_snapshots()
        if snapshots:
            logger.info("[snapshots] ENABLE_MARKET_DATA=false — using latest historical cache")
            return snapshots
        logger.warning("[snapshots] ENABLE_MARKET_DATA=false and no historical cache — news-only mode")
        return []
    snapshots = get_snapshots(all_tickers)
    save_snapshots(snapshots)
    if not snapshots:
        logger.warning("[snapshots] No market data retrieved — continuing with news-only signals")
    return snapshots


def _run_edgar_tasks(all_tickers):
    """Run all SEC EDGAR fetches sequentially inside one thread.

    Four modules hit EDGAR (eight_k, insider_trades, sec_filings, ipo_pipeline).
    Running them concurrently would combine to ~20+ req/s against the 10 req/s cap.
    Grouping them here lets the EDGAR thread run in parallel with every other source
    while keeping the combined EDGAR rate safe.

    Returns a dict keyed by task name so the caller can unpack results cleanly.
    """
    results = {}

    if settings.enable_8k_filings:
        results["8k"] = _safe("8k", fetch_8k_articles, all_tickers,
                               lookback_days=settings.eight_k_lookback_days)

    if settings.enable_insider_trades:
        results["insider"] = _safe("insider", fetch_insider_trades, all_tickers)

    if settings.enable_sec_filings:
        results["sec"] = _safe("sec", fetch_sec_filings)

    if settings.enable_ipo_pipeline:
        results["ipo"] = _safe("ipo", fetch_ipo_context,
                                lookback_days=settings.ipo_lookback_days)

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(send_email: bool = False) -> None:
    """Execute the full analysis pipeline."""
    start = datetime.now(timezone.utc)
    logger.info(f"Pipeline started at {fmt_et(start)}")

    tickers     = settings.stocks_list
    sectors     = settings.sectors_list
    commodities = settings.commodities_list

    # ── Step 0: Ticker discovery (serial — rest of pipeline depends on it) ──
    logger.info("Step 0: Discovering trending tickers...")
    all_tickers = get_trending_tickers(tickers, sectors)
    new_commodities = [t for t in commodities if t not in all_tickers]
    if new_commodities:
        all_tickers = all_tickers + new_commodities
        logger.info(f"Step 0: Pinned commodities: {new_commodities}")

    # ── Steps 1–3: Parallel data fetch ────────────────────────────────────
    #
    # Two concurrent groups:
    #   A) Non-EDGAR sources — all truly parallel (no shared rate limit)
    #   B) EDGAR sources     — sequential inside one thread to honour the
    #                          10 req/s cap shared by eight_k / insider_trades /
    #                          sec_filings / ipo_pipeline
    #
    # The pool shuts down (wait=True) at the end of the `with` block, so all
    # futures are resolved before we collect results below.
    #
    logger.info("Steps 1–3: Launching parallel data fetch...")

    with ThreadPoolExecutor(max_workers=14, thread_name_prefix="pipeline") as pool:

        # Group A: non-EDGAR — submit all at once, run concurrently
        f_news         = pool.submit(_safe, "news", _fetch_news, tickers, sectors)
        f_snapshots    = pool.submit(_safe, "snapshots", _fetch_snapshots, all_tickers)

        f_trends       = (pool.submit(_safe, "trends", fetch_google_trends, tickers)
                          if settings.enable_google_trends else None)

        f_reddit       = (pool.submit(_safe, "reddit", fetch_reddit_sentiment, tickers,
                                      client_id=settings.reddit_client_id,
                                      client_secret=settings.reddit_client_secret,
                                      user_agent=settings.reddit_user_agent)
                          if settings.enable_reddit_sentiment else None)

        f_analyst      = (pool.submit(_safe, "analyst", fetch_analyst_ratings, tickers,
                                      lookback_days=settings.analyst_ratings_lookback_days)
                          if settings.enable_analyst_ratings else None)

        f_eps          = (pool.submit(_safe, "eps", fetch_earnings_surprises, tickers,
                                      lookback_days=settings.earnings_lookback_days)
                          if settings.enable_earnings else None)

        f_earnings_cal = (pool.submit(_safe, "earnings_cal", fetch_earnings_context, tickers,
                                      upcoming_days=settings.earnings_upcoming_days,
                                      alpha_vantage_key=settings.alpha_vantage_key)
                          if settings.enable_earnings else None)

        f_short        = (pool.submit(_safe, "short", fetch_short_interest, tickers)
                          if settings.enable_short_interest else None)

        f_options      = (pool.submit(_safe, "options", fetch_options_flow, all_tickers)
                          if settings.enable_options_flow else None)

        f_fred         = (pool.submit(_safe, "fred", fetch_macro_context, settings.fred_api_key)
                          if settings.enable_fred else None)

        f_cot          = (pool.submit(_safe, "cot", fetch_cot_context)
                          if settings.enable_cot else None)

        f_vix          = (pool.submit(_safe, "vix", fetch_vix_context)
                          if settings.enable_vix else None)

        f_put_call     = (pool.submit(_safe, "put_call", fetch_put_call_context, tickers)
                          if settings.enable_put_call else None)

        # Group B: EDGAR — one thread, all four sources sequential inside it
        f_edgar = pool.submit(_run_edgar_tasks, all_tickers)

    # ── Collect results ───────────────────────────────────────────────────

    def get(fut):
        return fut.result() if fut is not None else None

    edgar = get(f_edgar) or {}

    # Merge all article sources into a single list
    articles = get(f_news) or []
    _article_chunks = {
        "8k":      edgar.get("8k"),
        "trends":  get(f_trends),
        "reddit":  get(f_reddit),
        "analyst": get(f_analyst),
        "eps":     get(f_eps),
        "short":   get(f_short),
    }
    for label, chunk in _article_chunks.items():
        if chunk:
            articles = articles + chunk
            logger.info(f"  [{label}] +{len(chunk)} article(s)")

    logger.info(f"Steps 1–3: {len(articles)} total articles assembled")

    snapshots = get(f_snapshots) or []

    # Merge smart money signals
    smart_money = []
    for key in ("insider", "sec"):
        chunk = edgar.get(key)
        if chunk:
            smart_money.extend(chunk)
    options_trades = get(f_options)
    if options_trades:
        smart_money.extend(options_trades)

    any_smart_money_enabled = (
        settings.enable_insider_trades or
        settings.enable_options_flow or
        settings.enable_sec_filings
    )
    insider_trades = smart_money if (smart_money or any_smart_money_enabled) else None

    # Surface tickers discovered via smart money signals
    if smart_money:
        smart_tickers = get_tickers_from_smart_money(smart_money)
        new_from_smart = [t for t in smart_tickers if t not in all_tickers]
        if new_from_smart:
            logger.info(f"Adding {new_from_smart} to universe from smart money signals")
            all_tickers = all_tickers + new_from_smart

    macro_context    = get(f_fred)
    cot_context      = get(f_cot)
    ipo_context      = edgar.get("ipo")
    vix_context      = get(f_vix)
    put_call_context = get(f_put_call)
    earnings_context = get(f_earnings_cal)

    # ── Step 4: Build signals ─────────────────────────────────────────────
    logger.info("Step 4: Building signals...")
    signals = build_signals(
        all_tickers,
        articles,
        insider_trades=insider_trades,
        put_call_context=put_call_context,
    )
    signals_by_ticker = {s.ticker: s for s in signals}

    # ── Step 5: Generate recommendations ─────────────────────────────────
    logger.info("Step 5: Generating recommendations...")
    recommendations = generate_recommendations(
        signals,
        insider_trades=insider_trades,
        macro_context=macro_context,
        cot_context=cot_context,
        ipo_context=ipo_context,
        vix_context=vix_context,
        put_call_context=put_call_context,
        earnings_context=earnings_context,
    )

    # Only surface BUY and SELL as actionable, with minimum confidence guard
    actionable = [r for r in recommendations if r.action in ("BUY", "SELL") and r.confidence >= 0.78]

    # Keep only the top 10 recommendations by conviction:
    # BUY/SELL first (sorted by confidence desc), then HOLD/WATCH to fill up to 10.
    _ACTION_RANK = {"BUY": 0, "SELL": 0, "HOLD": 1, "WATCH": 2}
    recommendations = sorted(
        recommendations,
        key=lambda r: (_ACTION_RANK.get(r.action, 3), -r.confidence),
    )[:10]
    actionable = [r for r in recommendations if r.action in ("BUY", "SELL") and r.confidence >= 0.78]

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info(
        f"Pipeline complete in {elapsed:.1f}s | "
        f"{len(all_tickers)} tickers → top {len(recommendations)} shown | "
        f"{len(articles)} articles"
    )

    _log_recommendations(recommendations)

    # Performance tracking
    update_open_trades()
    record_new_trades(actionable)
    log_performance_summary()
    perf = get_performance_for_email()

    if settings.enable_charts:
        logger.info("Generating charts and HTML report...")
        ts_label = start.strftime("%Y-%m-%d_%H%M")
        report_path = save_html_report(recommendations, performance=perf, label=ts_label,
                                       signals_by_ticker=signals_by_ticker)
        if report_path:
            logger.info(f"Report: {report_path.resolve()}")
    else:
        logger.info("Charts disabled (ENABLE_CHARTS=false) — skipping HTML report")

    _print_summary(actionable, smart_money or [])

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
            macro_context=macro_context,
            cot_context=cot_context,
            ipo_context=ipo_context,
            vix_context=vix_context,
            put_call_context=put_call_context,
            earnings_context=earnings_context,
        )
    else:
        logger.info("Email not configured — skipping.")


def _log_recommendations(recommendations) -> None:
    """Write every recommendation to the log file."""
    stocks = [r for r in recommendations if r.type == "STOCK"]
    etfs   = [r for r in recommendations if r.type == "ETF"]

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
            from collections import defaultdict
            by_ticker = defaultdict(list)
            for trade in smart_money:
                by_ticker[trade.ticker].append(trade)

            actionable_set = {r.ticker for r in recommendations}
            top_tickers = sorted(
                by_ticker.keys(),
                key=lambda ticker: (ticker not in actionable_set, ticker),
            )[:settings.smart_money_top_tickers]
            for ticker in top_tickers:
                trades = sorted(by_ticker[ticker], key=lambda x: not x.is_bullish)
                sigs = ", ".join(
                    f"{'[+]' if x.is_bullish else '[-]'} {x.action_label} ({x.trader_name})"
                    for x in trades[:3]
                )
                print(f"  {ticker:<6}  {sigs}")
            if len(by_ticker) > settings.smart_money_top_tickers:
                print(f"  … and {len(by_ticker) - settings.smart_money_top_tickers} more ticker(s) (see email report)")
        print()

    print("=" * 60 + "\n")
