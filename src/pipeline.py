"""Main analysis pipeline: fetch → analyse → recommend → notify."""

from loguru import logger
from datetime import datetime, timezone
from config import settings
from src.data.news_fetcher import fetch_all_news
from src.data.market_data import get_snapshots
from src.signals.aggregator import build_signals
from src.analysis.claude_analyst import generate_recommendations
from src.notifications.email_sender import send_recommendations
from src.performance.scorer import score_matured, build_scorecard
from src.performance.store import init_db, log_recommendations


def run_pipeline(send_email: bool = True) -> None:
    """Execute the full analysis pipeline."""
    start = datetime.now(timezone.utc)
    logger.info(f"Pipeline started at {start.strftime('%Y-%m-%d %H:%M UTC')}")

    # 0. Grade any past recommendations that have now matured (feedback loop).
    try:
        init_db()
        score_matured()
    except Exception as e:
        logger.error(f"Performance scoring failed (continuing): {e}")

    tickers = settings.stocks_list
    sectors = settings.sectors_list
    all_tickers = tickers + sectors

    # 1. Fetch news
    logger.info("Step 1/4: Fetching news...")
    articles = fetch_all_news(tickers, sectors)

    # 2. Fetch market data
    logger.info("Step 2/4: Fetching market snapshots...")
    snapshots = get_snapshots(all_tickers)

    if not snapshots:
        logger.error("No market data retrieved. Aborting pipeline.")
        return

    # 3. Build signals (sentiment + technical per ticker)
    logger.info("Step 3/4: Building signals...")
    signals = build_signals(snapshots, articles)

    # 4. Generate final recommendations via Claude
    logger.info("Step 4/4: Generating recommendations...")
    recommendations = generate_recommendations(signals)

    # Log every recommendation so we can grade it against future returns.
    try:
        price_map = {s.ticker: s.price for s in snapshots}
        signal_map = {s.ticker: s for s in signals}
        log_recommendations(recommendations, price_map, signal_map)
    except Exception as e:
        logger.error(f"Failed to log recommendations (continuing): {e}")

    # Build the realized-performance scorecard (accurate, measured returns).
    scorecard = None
    try:
        scorecard = build_scorecard()
    except Exception as e:
        logger.error(f"Failed to build scorecard (continuing): {e}")

    # Filter to actionable recommendations only (exclude low-confidence HOLDs)
    actionable = [r for r in recommendations if not (r.action == "HOLD" and r.confidence < 0.4)]

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info(
        f"Pipeline complete in {elapsed:.1f}s | "
        f"{len(snapshots)} tickers | {len(articles)} articles | "
        f"{len(actionable)} actionable recommendations"
    )

    # Print to console
    _print_summary(actionable, scorecard)

    # Send email
    if send_email and actionable:
        send_recommendations(actionable, total_analysed=len(snapshots), scorecard=scorecard)


def _print_summary(recommendations, scorecard=None) -> None:
    print("\n" + "=" * 60)
    print(f"  RECOMMENDATIONS  ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})")
    print("=" * 60)
    for rec in recommendations:
        bar = "▲" if rec.direction == "BULLISH" else ("▼" if rec.direction == "BEARISH" else "◆")
        print(f"  {bar} {rec.ticker:<6} {rec.action:<5}  conf={rec.confidence:.0%}  {rec.direction}")
        print(f"    {rec.rationale[:100]}")
        print()
    print("=" * 60)

    if scorecard is not None:
        _print_scorecard(scorecard)
    print()


def _print_scorecard(scorecard) -> None:
    """Print realized performance of past recommendations."""
    print("  PERFORMANCE (realized, directional BUY/SELL calls)")
    print("-" * 60)
    if scorecard.total_graded == 0:
        print("  No recommendations have matured yet — check back after a few")
        print(f"  sessions. ({scorecard.total_logged} logged so far.)")
        print("=" * 60)
        return

    for h in scorecard.horizons:
        if h.graded == 0:
            print(f"  {h.horizon_days}d: no graded calls yet")
            continue
        print(
            f"  {h.horizon_days}d: hit rate {h.hit_rate:.0%} ({h.hits}/{h.graded})  "
            f"avg {h.avg_aligned_return:+.2%}  "
            f"best {h.best:+.2%}  worst {h.worst:+.2%}"
        )

    if scorecard.recent:
        print("\n  Recently graded:")
        for g in scorecard.recent[:8]:
            mark = "✓" if g.hit else "✗"
            print(
                f"    {mark} {g.ticker:<6} {g.action:<5} {g.horizon_days}d  "
                f"{g.aligned_return:+.2%}  ({g.generated_at.strftime('%Y-%m-%d')})"
            )
    print("=" * 60)
