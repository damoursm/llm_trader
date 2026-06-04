"""Main analysis pipeline: fetch → analyse → recommend → notify."""

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from loguru import logger
from datetime import datetime, timezone

from config import settings
from src.utils import now_et, fmt_et
from src.data.news_fetcher import fetch_all_news
from src.data.market_data import get_snapshots
from src.data.cache import load_news, save_news, load_snapshots, save_snapshots, load_latest_snapshots
from src.data.trending import get_trending_tickers
from src.signals.aggregator import build_signals
from src.analysis.claude_analyst import generate_recommendations, get_last_synthesis_meta
from src.data.insider_trades import fetch_insider_trades, get_tickers_from_smart_money
from src.data.eight_k import fetch_8k_articles
from src.data.google_trends import fetch_google_trends
from src.data.reddit_sentiment import fetch_reddit_sentiment
from src.data.analyst_ratings import fetch_analyst_ratings
from src.data.earnings import fetch_earnings_surprises, fetch_earnings_context
from src.data.pead import fetch_pead_context
from src.data.short_interest import fetch_short_interest
from src.data.options_flow import fetch_options_flow
from src.data.sec_filings import fetch_sec_filings
from src.data.fred import fetch_macro_context
from src.data.cot import fetch_cot_context
from src.data.ipo_pipeline import fetch_ipo_context
from src.data.vix import fetch_vix_context
from src.data.credit import fetch_credit_context
from src.data.put_call import fetch_put_call_context
from src.data.tick import fetch_tick_context
from src.data.gamma_exposure import fetch_gex_context
from src.data.breadth import fetch_breadth_context
from src.data.highs_lows import fetch_highs_lows_context
from src.data.mcclellan import fetch_mcclellan_context
from src.data.macro_surprise import fetch_macro_surprise_context
from src.data.fedwatch import fetch_fedwatch_context
from src.data.revision_momentum import fetch_revision_momentum_context
from src.data.earnings_whisper import fetch_whisper_context
from src.data.opex import compute_opex_context
from src.data.seasonality import compute_seasonality_context
from src.data.bond_internals import fetch_bond_internals_context
from src.data.move import fetch_move_context
from src.data.dix import fetch_dix_context
from src.data.global_macro import fetch_global_macro_context
from src.data.sector_rotation import fetch_sector_rotation_context
from src.data.rotation_drivers import fetch_rotation_drivers_context
from src.data.intermarket import fetch_intermarket_context
from src.data.macro_news import fetch_macro_news_context
from src.data.macro_regime import compute_macro_regime
from src.data.market_mode import compute_market_mode
from src.data.business_cycle_rotation import compute_business_cycle_context
from src.data.catalyst_timing import compute_catalyst_context, apply_watch_elevation
from src.data.cluster_watchlist import (
    load_cluster_watchlist, save_cluster_watchlist,
    update_cluster_watchlist, build_cluster_watchlist_context,
)
from src.signals.sector_pairs import find_sector_pairs
from src.signals.cointegration import find_cointegrated_pairs
from src.analysis.sentiment import reset_sentiment_providers, get_sentiment_provider_summary
from src.notifications.email_sender import send_recommendations
from src.performance.tracker import record_new_trades, update_open_trades, close_trades_on_signal_reversal, log_performance_summary, get_performance_for_email, get_open_trade_tickers, monitor_open_positions, reset_price_health, get_price_health, _method_scores_from_signal, _methods_agreeing, _dominant_method
from src.db import repo
from src.performance.hypothetical_tracker import update_hypothetical_trades, get_hypothetical_performance_for_email


# ---------------------------------------------------------------------------
# Fetch helpers (one per data source, called from the thread pool)
# ---------------------------------------------------------------------------

# Per-run record of every data source touched via _safe — the "APIs used" log.
# Thread-safe because _safe runs inside the data-fetch ThreadPoolExecutor.
_SOURCE_LOG: list = []
_SOURCE_LOCK = threading.Lock()


def _reset_source_log() -> None:
    global _SOURCE_LOG
    with _SOURCE_LOCK:
        _SOURCE_LOG = []


def _snapshot_source_log() -> list:
    with _SOURCE_LOCK:
        return list(_SOURCE_LOG)


def _collect_sources() -> list:
    """Per-run data-source health: the timed fetchers' log plus a synthetic
    'Live price feed' entry reflecting the yfinance→Polygon price-fetch outcomes.

    The dashboard banner and the email both read this, so a failing API is
    reported in both places from one source of truth.
    """
    sources = _snapshot_source_log()
    try:
        ph = get_price_health()
        attempts = ph.get("yfinance", 0) + ph.get("polygon", 0)
        failed = ph.get("failed") or []
        if attempts or failed:
            n = len(failed)
            shown = ", ".join(failed[:8]) + ("…" if n > 8 else "")
            sources.append({
                "label": "Live price feed (yfinance→Polygon)",
                "enabled": True,
                "ok": n == 0,
                "error": f"{n} ticker(s) had no price: {shown}" if n else None,
                "duration_s": None,
            })
    except Exception:
        pass
    return sources


def _safe(label: str, fn, *args, **kwargs):
    """Run fn; log warning and return None on any exception.

    Records the per-source outcome (ok / error / duration) into the per-run
    source log so the run's 'APIs used' record can be persisted to DuckDB.
    """
    t0 = time.time()
    ok, err, result = True, None, None
    try:
        result = fn(*args, **kwargs)
    except Exception as e:
        ok, err = False, str(e)
        logger.warning(f"[{label}] fetch failed: {e}")
    finally:
        with _SOURCE_LOCK:
            _SOURCE_LOG.append({
                "label": label,
                "ok": ok,
                "error": err,
                "duration_s": round(time.time() - t0, 3),
            })
    return result


def _persist_run(run_id, start, finished, all_tickers, recommendations, actionable,
                 gate_diag, market_mode_context, macro_regime_context,
                 confidence_threshold, allow_buys, signals_by_ticker) -> None:
    """Write the run, its per-source 'APIs used' record, and every recommendation
    (with method attribution + the LLM provider that synthesised it) to DuckDB."""
    try:
        actionable_ids = {id(r) for r in actionable}
        provider = get_last_synthesis_meta().get("provider")

        rec_rows = []
        for r in recommendations:
            scores = _method_scores_from_signal(r.ticker, r.direction, signals_by_ticker)
            gen_at = r.generated_at
            rec_rows.append({
                "rec_id": hashlib.sha1(f"{run_id}|{r.ticker}".encode("utf-8")).hexdigest()[:16],
                "run_id": run_id,
                "generated_at": gen_at.isoformat() if hasattr(gen_at, "isoformat") else str(gen_at),
                "ticker": r.ticker,
                "type": r.type,
                "direction": r.direction,
                "action": r.action,
                "confidence": r.confidence,
                "time_horizon": r.time_horizon,
                "rationale": r.rationale,
                "actionable": id(r) in actionable_ids,
                "dominant_method": _dominant_method(scores, r.direction),
                "methods_agreeing": _methods_agreeing(scores, r.direction),
                "contributing_scores": scores,
                "llm_provider": provider,
            })

        repo.insert_run({
            "run_id": run_id,
            "started_at": start.isoformat(),
            "finished_at": finished.isoformat(),
            "elapsed_s": (finished - start).total_seconds(),
            "market_mode": getattr(market_mode_context, "mode", None),
            "macro_regime": getattr(macro_regime_context, "regime", None),
            "confidence_threshold": confidence_threshold,
            "allow_buys": allow_buys,
            "universe_size": len(all_tickers),
            "n_recommendations": len(recommendations),
            "n_actionable": len(actionable),
            "llm_synthesis_provider": provider,
            "llm_sentiment_provider": get_sentiment_provider_summary(),
            "gate_diag": gate_diag,
        })
        sources = _collect_sources()
        repo.insert_run_sources(run_id, sources)
        repo.insert_recommendations(rec_rows)
        logger.info(
            f"[db] Persisted run {run_id}: {len(rec_rows)} recommendation(s), "
            f"{len(sources)} source(s) → DuckDB"
        )
    except Exception as e:
        logger.error(f"[db] Failed to persist run metadata (continuing): {e}")


def _assess_llm_health() -> dict:
    """Assess whether the LLM layer functioned this run, for alerting.

    Both per-ticker sentiment scoring and final synthesis run on Claude/DeepSeek.
    When credits are exhausted or keys are invalid the failure is *silent*:
    sentiment returns a neutral 0.0 and synthesis falls through to the rule-based
    last resort. This collapses the two existing provider signals into a single
    'down' verdict so the degradation can be surfaced (CRITICAL log + email
    banner) instead of scrolling past in the logs.

    Returns a dict: {down, synthesis_down, sentiment_down, synthesis_provider,
    sentiment_summary, message}.
    """
    synth_provider = (get_last_synthesis_meta() or {}).get("provider")
    sent_summary   = get_sentiment_provider_summary()   # e.g. "deepseek×40, none×2" | "none×42" | None

    # Synthesis is down only when it fell all the way to the rule-based engine
    # (both Claude and the DeepSeek fallback failed). A DeepSeek-served run is
    # still a working LLM layer.
    synthesis_down = synth_provider == "rule-based"

    # Sentiment is down when calls were attempted but *none* used a real model
    # (every per-ticker call recorded the "none" provider). No attempts at all
    # (sent_summary is None — e.g. no tickers had news) is not a degradation.
    sentiment_attempted = sent_summary is not None
    sentiment_ok        = bool(sent_summary) and ("deepseek" in sent_summary or "anthropic" in sent_summary)
    sentiment_down      = sentiment_attempted and not sentiment_ok

    parts = []
    if synthesis_down:
        parts.append("final synthesis fell through to the rule-based engine (recommendations are NOT LLM-generated)")
    if sentiment_down:
        parts.append(f"per-ticker sentiment scoring failed for every ticker ({sent_summary})")

    return {
        "down":               synthesis_down or sentiment_down,
        "synthesis_down":     synthesis_down,
        "sentiment_down":     sentiment_down,
        "synthesis_provider": synth_provider,
        "sentiment_summary":  sent_summary,
        "message":            "; ".join(parts),
    }


def _assess_broker_health(report: Optional[dict]) -> Optional[dict]:
    """Turn a broker reconcile report into a health verdict for the CRITICAL log
    and the email banner. Returns None when broker_mode=off / no report."""
    if not report or report.get("mode") in (None, "off"):
        return None
    problems = []
    if not report.get("connected"):
        problems.append("broker NOT connected — orders were not placed")
    if report.get("rejects"):
        problems.append(f"{report['rejects']} order(s) rejected")
    if report.get("drift"):
        names = ", ".join(d["ticker"] for d in report["drift"][:6])
        problems.append(f"{len(report['drift'])} position(s) drifted from the ledger ({names})")
    if not report.get("ok") and report.get("errors"):
        problems.append("reconcile error")
    return {
        "down":      bool(problems),
        "mode":      report.get("mode"),
        "connected": report.get("connected"),
        "entries":   report.get("entries_submitted", 0),
        "exits":     report.get("exits_submitted", 0),
        "rejects":   report.get("rejects", 0),
        "drift":     report.get("drift", []),
        "slippage":  report.get("slippage", []),
        "message":   "; ".join(problems),
    }


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
    if not settings.enable_fetch_data:
        snapshots = load_latest_snapshots()
        if snapshots:
            logger.info("[snapshots] ENABLE_FETCH_DATA=false — using latest historical cache")
            return snapshots
        logger.warning("[snapshots] ENABLE_FETCH_DATA=false and no historical cache — news-only mode")
        return []
    snapshots = get_snapshots(all_tickers)
    save_snapshots(snapshots)
    if not snapshots:
        logger.warning("[snapshots] No market data retrieved — continuing with news-only signals")
    return snapshots


def _run_yf_options_tasks(all_tickers):
    """Run options_flow and GEX sequentially inside one thread.

    Both modules scan yfinance options chains.  Running them concurrently
    combines to ~20+ req/s against yfinance's unofficial rate limit, causing
    429 errors on every ticker.  Serialising them here keeps the combined
    options-chain request rate safe while still running in parallel with all
    other data sources.

    Returns a dict keyed by task name so the caller can unpack results cleanly.
    """
    results = {}

    if settings.enable_options_flow:
        results["options"] = _safe("options", fetch_options_flow, all_tickers)

    if settings.enable_gex:
        results["gex"] = _safe("gex", fetch_gex_context, all_tickers)

    return results


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
    run_id = start.strftime("%Y-%m-%d_%H%M%S")
    _reset_source_log()
    reset_sentiment_providers()
    reset_price_health()
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

    # Factor / thematic ETFs — pinned like commodities (Section E): broaden coverage beyond the
    # 11 GICS sectors with style-factor (momentum/quality/value/size/low-vol/growth) and
    # high-interest-theme (semis, software, biotech, defense, clean energy, …) ETFs.
    new_factors = [t for t in settings.factor_list if t not in all_tickers]
    if new_factors:
        all_tickers = all_tickers + new_factors
        logger.info(f"Step 0: Pinned factor/thematic ETFs: {new_factors}")

    # Opportunity screener — proactive setup discovery over a broad liquid universe
    # (unusual volume, 52w breakouts/reversals, relative strength, golden/death cross).
    # Cache-first; injects qualifying setups into the analysis universe.
    screener_context = None
    if settings.enable_opportunity_screener:
        from src.data.screener import run_screener
        screener_context = run_screener()
        if screener_context and screener_context.hits:
            new_from_screen = [h.ticker for h in screener_context.hits if h.ticker not in all_tickers]
            if new_from_screen:
                all_tickers = all_tickers + new_from_screen
                logger.info(
                    f"[screener] Injecting {len(new_from_screen)} setup(s) into universe: {new_from_screen}"
                )

    # Catalyst discovery (Section E) — MARKET-WIDE names with imminent earnings or fresh analyst
    # rating changes (not just the watchlist). Injected at Step 0 so the per-ticker enrichment
    # (analyst ratings, earnings calendar/surprises, full signal stack) picks them up downstream.
    if settings.enable_earnings_discovery:
        from src.data.earnings import discover_earnings_tickers
        _earn_disc = discover_earnings_tickers(
            window_days=settings.earnings_discovery_window_days,
            max_results=settings.earnings_discovery_max,
        )
        _new_earn = [t for t in _earn_disc if t not in all_tickers]
        if _new_earn:
            all_tickers = all_tickers + _new_earn
            logger.info(f"[earnings_disc] Injecting {len(_new_earn)} earnings name(s): {_new_earn}")

    if settings.enable_analyst_discovery:
        from src.data.analyst_ratings import discover_analyst_tickers
        _analyst_disc = discover_analyst_tickers(
            lookback_days=settings.analyst_discovery_lookback_days,
            min_firms=settings.analyst_discovery_min_firms,
            max_results=settings.analyst_discovery_max,
        )
        _new_analyst = [t for t in _analyst_disc if t not in all_tickers]
        if _new_analyst:
            all_tickers = all_tickers + _new_analyst
            logger.info(f"[analyst_disc] Injecting {len(_new_analyst)} analyst name(s): {_new_analyst}")

    # Load cluster watchlist and inject still-active tickers into the universe
    # so they are re-evaluated every day for the full 10-day window even if they
    # have dropped off the trending/discovery list.
    _cluster_raw = load_cluster_watchlist()
    _cluster_ctx_pre = build_cluster_watchlist_context(_cluster_raw)
    if _cluster_ctx_pre.active_tickers:
        new_from_cluster = [t for t in _cluster_ctx_pre.active_tickers if t not in all_tickers]
        if new_from_cluster:
            all_tickers = all_tickers + new_from_cluster
            logger.info(
                f"[cluster_watch] Injecting {new_from_cluster} from cluster watchlist "
                f"({len(_cluster_ctx_pre.active_tickers)} active)"
            )
        else:
            logger.info(
                f"[cluster_watch] {len(_cluster_ctx_pre.active_tickers)} active cluster watch(es) "
                f"already in universe: {_cluster_ctx_pre.active_tickers}"
            )

    # Pin tickers of open trades so their prices are always refreshed,
    # even if they've dropped off the trending/discovery list.
    open_trade_tickers = get_open_trade_tickers()
    new_from_trades = [t for t in open_trade_tickers if t not in all_tickers]
    if new_from_trades:
        all_tickers = all_tickers + new_from_trades
        logger.info(f"[tracker] Pinning open-trade tickers into universe: {new_from_trades}")

    # Pin hypothetical always-open trade tickers so their OHLCV cache stays
    # current — the daily-NAV compound walk needs a fresh close per held day.
    hypothetical_tickers = [t for t, _ in settings.hypothetical_trades_list]
    new_from_hyp = [t for t in hypothetical_tickers if t not in all_tickers]
    if new_from_hyp:
        all_tickers = all_tickers + new_from_hyp
        logger.info(f"[hypothetical] Pinning always-open tickers into universe: {new_from_hyp}")

    # ── Section F: discovery liquidity gate ───────────────────────────────────
    # Drop untradeable microcaps (below the price or 20-day avg dollar-volume floor) from every
    # DISCOVERED name before the universe is processed, so the widened funnel (Sections A–E) doesn't
    # add names the tracker's bid-ask model charges up to 250 bp a side. The pinned/intentional
    # universe is NEVER gated: static watchlist, sector ETFs, commodities, factor/thematic ETFs, and
    # open-trade tickers. gate_budget is a shared cold-fetch allowance reused by the later
    # macro-discovery and cointegration-peer injections.
    gate_budget = {"n": max(0, settings.discovery_gate_max_fetch)}
    from src.data.liquidity import apply_liquidity_gate
    _protected = {s.upper() for s in (list(tickers) + list(sectors) + list(commodities)
                                      + settings.factor_list + list(open_trade_tickers)
                                      + list(hypothetical_tickers))}
    _discovered = [t for t in all_tickers if t.upper() not in _protected]
    _kept = set(apply_liquidity_gate(_discovered, source="step0 discovery", budget=gate_budget))
    _before = len(all_tickers)
    all_tickers = [t for t in all_tickers if t.upper() in _protected or t.upper() in _kept]
    if len(all_tickers) != _before:
        logger.info(f"[liquidity] Step 0 universe gated: {_before} → {len(all_tickers)} tickers")

    # ── Steps 1–3: Parallel data fetch ────────────────────────────────────
    #
    # Three concurrent groups:
    #   A) Non-yfinance-options sources — truly parallel (no shared rate limit)
    #   B) yfinance options sources     — options_flow + GEX run sequentially
    #                                     inside one thread so they never compete
    #                                     for the yfinance options endpoint
    #   C) EDGAR sources                — sequential inside one thread to honour
    #                                     the 10 req/s cap
    #
    # The pool shuts down (wait=True) at the end of the `with` block, so all
    # futures are resolved before we collect results below.
    #
    logger.info("Steps 1–3: Launching parallel data fetch...")

    with ThreadPoolExecutor(max_workers=14, thread_name_prefix="pipeline") as pool:

        # Group A: non-yfinance-options — submit all at once, run concurrently
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

        f_pead         = (pool.submit(_safe, "pead", fetch_pead_context, tickers)
                          if settings.enable_pead else None)

        f_short        = (pool.submit(_safe, "short", fetch_short_interest, tickers)
                          if settings.enable_short_interest else None)

        f_fred         = (pool.submit(_safe, "fred", fetch_macro_context, settings.fred_api_key)
                          if settings.enable_fred else None)

        f_cot          = (pool.submit(_safe, "cot", fetch_cot_context)
                          if settings.enable_cot else None)

        f_vix          = (pool.submit(_safe, "vix", fetch_vix_context)
                          if settings.enable_vix else None)

        f_credit       = (pool.submit(_safe, "credit", fetch_credit_context)
                          if settings.enable_credit else None)

        f_put_call     = (pool.submit(_safe, "put_call", fetch_put_call_context, tickers)
                          if settings.enable_put_call else None)

        f_tick         = (pool.submit(_safe, "tick", fetch_tick_context)
                          if settings.enable_tick else None)

        f_breadth      = (pool.submit(_safe, "breadth", fetch_breadth_context)
                          if settings.enable_breadth else None)

        f_highs_lows   = (pool.submit(_safe, "highs_lows", fetch_highs_lows_context)
                          if settings.enable_highs_lows else None)

        f_mcclellan    = (pool.submit(_safe, "mcclellan", fetch_mcclellan_context)
                          if settings.enable_mcclellan else None)

        f_macro_surprise = (pool.submit(_safe, "macro_surprise", fetch_macro_surprise_context)
                            if settings.enable_macro_surprise else None)

        f_fedwatch       = (pool.submit(_safe, "fedwatch", fetch_fedwatch_context)
                            if settings.enable_fedwatch else None)

        f_revision       = (pool.submit(_safe, "revision", fetch_revision_momentum_context, tickers)
                            if settings.enable_revision_momentum else None)

        f_whisper        = (pool.submit(_safe, "whisper", fetch_whisper_context, tickers)
                            if settings.enable_earnings_whisper else None)

        f_bond_internals = (pool.submit(_safe, "bond_internals", fetch_bond_internals_context)
                            if settings.enable_bond_internals else None)

        f_move           = (pool.submit(_safe, "move", fetch_move_context)
                            if settings.enable_move else None)

        f_dix            = (pool.submit(_safe, "dix", fetch_dix_context)
                            if settings.enable_dix else None)

        f_global_macro   = (pool.submit(_safe, "global_macro", fetch_global_macro_context)
                            if settings.enable_global_macro else None)

        f_sector_rotation = (pool.submit(_safe, "sector_rotation", fetch_sector_rotation_context)
                             if settings.enable_sector_rotation else None)

        f_rotation_drivers = (pool.submit(_safe, "rotation_drivers", fetch_rotation_drivers_context)
                              if settings.enable_rotation_drivers else None)

        f_intermarket    = (pool.submit(_safe, "intermarket", fetch_intermarket_context)
                            if settings.enable_intermarket else None)

        # Group B: yfinance options — options_flow then GEX, sequential in one thread
        # Both modules scan yfinance options chains; running them concurrently causes
        # 429 rate-limit errors on every ticker. One thread keeps the combined rate safe.
        f_yf_options = pool.submit(_run_yf_options_tasks, all_tickers)

        # Group C: EDGAR — one thread, all four sources sequential inside it
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

    # Macro-news scan — derives a geopolitical / oil / tariff / policy regime
    # read from the SAME article flow (no extra fetch cost). Caches hourly to
    # match the news cache TTL. Skipped when the flag is off or article count
    # is too thin to read.
    macro_news_context = None
    if settings.enable_macro_news:
        macro_news_context = _safe("macro_news", fetch_macro_news_context, articles)
        if macro_news_context:
            logger.info(
                f"[macro_news] {macro_news_context.composite_signal}  "
                f"score={macro_news_context.macro_news_score:+.2f}  "
                f"themes={len(macro_news_context.themes)}"
            )

    snapshots = get(f_snapshots) or []

    yf_options = get(f_yf_options) or {}

    # Merge smart money signals
    smart_money = []
    for key in ("insider", "sec"):
        chunk = edgar.get(key)
        if chunk:
            smart_money.extend(chunk)
    options_trades = yf_options.get("options")
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
    credit_context   = get(f_credit)
    put_call_context = get(f_put_call)
    tick_context     = get(f_tick)
    breadth_context       = get(f_breadth)
    highs_lows_context    = get(f_highs_lows)
    mcclellan_context     = get(f_mcclellan)
    macro_surprise_context    = get(f_macro_surprise)
    fedwatch_context          = get(f_fedwatch)
    revision_momentum_context = get(f_revision)
    whisper_context           = get(f_whisper)
    bond_internals_context    = get(f_bond_internals)
    move_context              = get(f_move)
    dix_context               = get(f_dix)
    global_macro_context      = get(f_global_macro)
    sector_rotation_context   = get(f_sector_rotation)
    rotation_drivers_context  = get(f_rotation_drivers)
    intermarket_context       = get(f_intermarket)
    gex_context               = yf_options.get("gex")
    earnings_context = get(f_earnings_cal)
    pead_context     = get(f_pead)

    # OpEx context — pure date math, computed synchronously (no I/O)
    opex_context = compute_opex_context() if settings.enable_opex else None
    if opex_context:
        logger.info(f"[opex] {opex_context.signal}: {opex_context.summary}")

    # Seasonality context — pure date math, computed synchronously (no I/O)
    seasonality_context = compute_seasonality_context() if settings.enable_seasonality else None
    if seasonality_context:
        logger.info(f"[seasonality] {seasonality_context.composite_signal}: {seasonality_context.summary[:120]}")

    # ── Market Mode (TRENDING / NEUTRAL / CHOPPY) ─────────────────────────
    market_mode_context = None
    if settings.enable_market_mode_switching:
        market_mode_context = compute_market_mode(
            vix_context=vix_context,
            breadth_context=breadth_context,
            highs_lows_context=highs_lows_context,
            mcclellan_context=mcclellan_context,
        )

    # ── Business Cycle Rotation ───────────────────────────────────────────
    business_cycle_context = None
    if settings.enable_business_cycle_rotation:
        business_cycle_context = compute_business_cycle_context(
            macro_context=macro_context,
            sector_rotation_context=sector_rotation_context,
        )

    # ── Macro Regime Filter ───────────────────────────────────────────────
    macro_regime_context = None
    if settings.enable_macro_regime_filter:
        macro_regime_context = compute_macro_regime(
            vix_context=vix_context,
            move_context=move_context,
            bond_internals_context=bond_internals_context,
            global_macro_context=global_macro_context,
            macro_context=macro_context,
            breadth_context=breadth_context,
            credit_context=credit_context,
            dix_context=dix_context,
            intermarket_context=intermarket_context,
            macro_news_context=macro_news_context,
        )

    # Macro → discovery loop: pull top holdings of the favored sector/factor ETFs (sector-rotation
    # inflows + business-cycle leaders + DIX factor tilt) into the universe so the macro/regime
    # analysis actually drives single-name selection. Runs here (after the macro contexts exist)
    # and before cointegration/build_signals so the injected names get the full signal stack.
    macro_discovery_context = None
    if settings.enable_macro_discovery:
        from src.data.macro_discovery import run_macro_discovery
        macro_discovery_context = run_macro_discovery(
            sector_rotation_context=sector_rotation_context,
            business_cycle_context=business_cycle_context,
            dix_context=dix_context,
        )
        if macro_discovery_context and macro_discovery_context.tickers:
            new_from_macro = [t for t in macro_discovery_context.tickers if t not in all_tickers]
            new_from_macro = apply_liquidity_gate(new_from_macro, source="macro_discovery", budget=gate_budget)
            if new_from_macro:
                all_tickers = all_tickers + new_from_macro
                logger.info(
                    f"[macro_discovery] Injecting {len(new_from_macro)} favored-sector name(s): {new_from_macro}"
                )

    # Cointegration pairs: statistical-arbitrage spreads (ADF + z-score).
    # Computed before build_signals so its per-ticker directional leans feed the aggregator.
    coint_context = None
    if settings.enable_cointegration:
        logger.info("Step 3.9: Testing cointegration pairs...")
        coint_context = find_cointegrated_pairs(all_tickers)

        # Peer-expansion (Section E): pull the partner leg of a tradeable pair when only one leg
        # is in the universe, so the relationship is tradeable both ways. The partner already
        # carries a cointegration score; injecting it before build_signals gives it a TickerSignal.
        if settings.enable_coint_peer_discovery and coint_context:
            from src.signals.cointegration import get_coint_peer_tickers
            _peers = get_coint_peer_tickers(coint_context, all_tickers, max_peers=settings.coint_peer_max)
            _peers = apply_liquidity_gate(_peers, source="coint_peer", budget=gate_budget)
            if _peers:
                all_tickers = all_tickers + _peers
                logger.info(f"[coint_peer] Injecting {len(_peers)} cointegration peer leg(s): {_peers}")

    # ── Step 4: Build signals ─────────────────────────────────────────────
    logger.info("Step 4: Building signals...")
    signals = build_signals(
        all_tickers,
        articles,
        insider_trades=insider_trades,
        put_call_context=put_call_context,
        gex_context=gex_context,
        market_mode_context=market_mode_context,
        opex_context=opex_context,
        pead_context=pead_context,
        coint_context=coint_context,
    )
    signals_by_ticker = {s.ticker: s for s in signals}

    # Update cluster watchlist: add newly detected clusters, expire stale entries
    _cluster_raw = update_cluster_watchlist(signals_by_ticker, _cluster_raw)
    save_cluster_watchlist(_cluster_raw)
    cluster_watchlist_context = build_cluster_watchlist_context(_cluster_raw)
    if cluster_watchlist_context.entries:
        logger.info(f"[cluster_watch] {cluster_watchlist_context.summary}")

    # Sector pairs: find ETF vs constituent divergences (market-neutral pair trades)
    sector_pairs_context = find_sector_pairs(signals_by_ticker)

    # ── Step 5: Generate recommendations ─────────────────────────────────
    logger.info("Step 5: Generating recommendations...")
    recommendations = generate_recommendations(
        signals,
        insider_trades=insider_trades,
        macro_context=macro_context,
        cot_context=cot_context,
        ipo_context=ipo_context,
        vix_context=vix_context,
        credit_context=credit_context,
        put_call_context=put_call_context,
        tick_context=tick_context,
        breadth_context=breadth_context,
        highs_lows_context=highs_lows_context,
        mcclellan_context=mcclellan_context,
        macro_surprise_context=macro_surprise_context,
        fedwatch_context=fedwatch_context,
        revision_momentum_context=revision_momentum_context,
        whisper_context=whisper_context,
        earnings_context=earnings_context,
        gex_context=gex_context,
        opex_context=opex_context,
        seasonality_context=seasonality_context,
        bond_internals_context=bond_internals_context,
        move_context=move_context,
        dix_context=dix_context,
        global_macro_context=global_macro_context,
        sector_rotation_context=sector_rotation_context,
        rotation_drivers_context=rotation_drivers_context,
        business_cycle_context=business_cycle_context,
        intermarket_context=intermarket_context,
        macro_news_context=macro_news_context,
    )

    # Keep only the top 10 recommendations by conviction:
    # BUY/SELL first (sorted by confidence desc), then HOLD/WATCH to fill up to 10.
    _ACTION_RANK = {"BUY": 0, "SELL": 0, "HOLD": 1, "WATCH": 2}
    recommendations = sorted(
        recommendations,
        key=lambda r: (_ACTION_RANK.get(r.action, 3), -r.confidence),
    )[:10]

    # Macro regime gate — adjust threshold and optionally block BUY entries
    _confidence_threshold = 0.78
    _allow_buys = True
    if macro_regime_context and settings.enable_macro_regime_filter:
        _confidence_threshold = macro_regime_context.confidence_threshold
        _allow_buys           = macro_regime_context.allow_buys
        if not _allow_buys:
            logger.warning(
                f"[macro_regime] Regime={macro_regime_context.regime} — "
                f"all new BUY entries BLOCKED (only SELLs allowed)"
            )
        logger.info(
            f"[macro_regime] Actionable threshold: {_confidence_threshold:.0%} "
            f"(default 78%) | allow_buys={_allow_buys}"
        )

    # Catalyst timing — earnings blackout, OpEx amplifier already applied in build_signals,
    # now apply WATCH elevation (8-K + insider buy) and earnings blackout on actionable
    catalyst_timing_context = None
    if settings.enable_catalyst_timing:
        catalyst_timing_context = compute_catalyst_context(
            earnings_context=earnings_context,
            opex_context=opex_context,
            articles=articles,
            insider_trades=insider_trades,
            signals_by_ticker=signals_by_ticker,
            sectors_list=settings.sectors_list,
            commodities_list=settings.commodities_list,
        )
        # Apply WATCH elevation to the top-10 recommendations
        recommendations = apply_watch_elevation(
            recommendations,
            catalyst_timing_context,
            signals_by_ticker=signals_by_ticker,
            sectors_list=settings.sectors_list,
            commodities_list=settings.commodities_list,
        )

    # ── Actionable filter with per-gate counters ─────────────────────────
    # Each rec is checked against every gate in order; rejections increment
    # the corresponding counter so we can see at end-of-run which constraints
    # are doing the actual filtering vs which never fire.
    earnings_blackout = (catalyst_timing_context.earnings_blackout_tickers
                         if catalyst_timing_context else set())

    gate_diag: dict = {
        "buy_sell_candidates":          0,
        "dropped_below_threshold":      0,
        "dropped_buy_blocked":          0,
        "dropped_earnings_blackout":    0,
        "actionable_survivors":         0,
        "confidence_threshold":         round(_confidence_threshold, 2),
        "allow_buys":                   _allow_buys,
        "regime":                       (macro_regime_context.regime
                                          if macro_regime_context else "NEUTRAL"),
    }

    actionable: List = []
    for r in recommendations:
        if r.action not in ("BUY", "SELL"):
            continue
        gate_diag["buy_sell_candidates"] += 1
        # Gate 1 — regime-tightened confidence threshold
        if r.confidence < _confidence_threshold:
            gate_diag["dropped_below_threshold"] += 1
            continue
        # Gate 2 — BUY block (PANIC / RISK_OFF)
        if r.action == "BUY" and not _allow_buys:
            gate_diag["dropped_buy_blocked"] += 1
            continue
        # Gate 3 — earnings blackout window
        if r.ticker in earnings_blackout:
            gate_diag["dropped_earnings_blackout"] += 1
            continue
        actionable.append(r)
        gate_diag["actionable_survivors"] += 1

    if gate_diag["dropped_earnings_blackout"]:
        logger.warning(
            f"[catalyst] Earnings blackout blocked "
            f"{gate_diag['dropped_earnings_blackout']} actionable(s)"
        )

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info(
        f"Pipeline complete in {elapsed:.1f}s | "
        f"{len(all_tickers)} tickers → top {len(recommendations)} shown | "
        f"{len(articles)} articles"
    )

    _log_recommendations(recommendations)

    # Performance tracking
    update_open_trades()
    # Open-position monitor: signal-decay / regime-flip exits BEFORE the
    # counter-recommendation exit path. Catches the case where a held
    # ticker's thesis has deteriorated but no counter-direction BUY/SELL
    # appeared in today's top-10 (so close_trades_on_signal_reversal would
    # silently keep holding it).
    monitor_open_positions(
        signals_by_ticker=signals_by_ticker,
        macro_regime_context=macro_regime_context,
    )
    close_trades_on_signal_reversal(actionable)   # exit early if signal reversed
    trade_diag = record_new_trades(actionable, signals_by_ticker=signals_by_ticker, run_id=run_id) or {}

    # Broker shadow execution (paper-first): once the internal ledger is final for
    # this tick, reconcile a real broker (IBKR paper) against it — submit entries
    # for new opens, close exits, report slippage/drift. No-op when broker_mode=off;
    # the reconciler is exception-safe and never breaks the run.
    broker_report = None
    if settings.broker_mode and settings.broker_mode != "off":
        from src.broker.reconcile import sync as _broker_sync
        try:
            broker_report = _broker_sync()
        except Exception as e:
            logger.warning(f"[broker] sync raised unexpectedly (internal sim unaffected): {e}")

    # Merge per-gate counters from the actionable filter + the trade-entry path
    # into one diagnostic blob so the user can see at a glance which constraints
    # are doing the work and which are passive backstops.
    gate_diag.update({
        "trade_considered":             int(trade_diag.get("considered", 0)),
        "trade_skipped_already_open":   int(trade_diag.get("skipped_already_open", 0)),
        "trade_skipped_correlation_cap": int(trade_diag.get("skipped_correlation_cap", 0)),
        "trade_skipped_no_price":       int(trade_diag.get("skipped_no_price", 0)),
        "trade_haircut_applied":        int(trade_diag.get("haircut_applied", 0)),
        "trade_opened":                 int(trade_diag.get("opened", 0)),
    })
    logger.info(
        "[gates] "
        f"regime={gate_diag['regime']} threshold={gate_diag['confidence_threshold']:.0%} "
        f"allow_buys={gate_diag['allow_buys']} | "
        f"actionable filter: {gate_diag['buy_sell_candidates']} BUY/SELL → "
        f"survived={gate_diag['actionable_survivors']} "
        f"(rejected by threshold={gate_diag['dropped_below_threshold']}, "
        f"BUY-block={gate_diag['dropped_buy_blocked']}, "
        f"earnings-blackout={gate_diag['dropped_earnings_blackout']}) | "
        f"trade entry: {gate_diag['trade_considered']} considered → "
        f"opened={gate_diag['trade_opened']} "
        f"(already_open={gate_diag['trade_skipped_already_open']}, "
        f"corr_cap={gate_diag['trade_skipped_correlation_cap']}, "
        f"no_price={gate_diag['trade_skipped_no_price']}, "
        f"corr_haircut_applied={gate_diag['trade_haircut_applied']})"
    )
    log_performance_summary()
    perf = get_performance_for_email()

    # Hypothetical always-open book — fully isolated from the real-trade
    # ledger above. Update marks then snapshot for the email section.
    update_hypothetical_trades()
    hypothetical_perf = get_hypothetical_performance_for_email()

    # The per-run static HTML report has been retired in favour of the live
    # Plotly Dash dashboard (python main.py --dashboard), backed by DuckDB.
    logger.info("Monitor performance + rationale in the dashboard: python main.py --dashboard")

    # Persist run + 'APIs used' + every recommendation (with attribution) to DuckDB.
    _persist_run(
        run_id, start, datetime.now(timezone.utc), all_tickers, recommendations, actionable,
        gate_diag, market_mode_context, macro_regime_context,
        _confidence_threshold, _allow_buys, signals_by_ticker,
    )

    # Surface a silent LLM-layer outage (credits exhausted / bad key) loudly:
    # a CRITICAL log line here, plus an email banner + subject tag below.
    llm_health = _assess_llm_health()
    if llm_health["down"]:
        logger.critical(
            f"[llm] LLM LAYER DEGRADED — {llm_health['message']}. "
            f"Top up / check Anthropic + DeepSeek API credits and keys."
        )

    # Surface broker/execution problems (not connected, rejects, position drift)
    # the same way — CRITICAL log here, email banner + subject tag below.
    broker_health = _assess_broker_health(broker_report)
    if broker_health and broker_health["down"]:
        logger.critical(
            f"[broker] EXECUTION ISSUE ({broker_health['mode']}) — {broker_health['message']}."
        )

    _print_summary(actionable, smart_money or [])

    email_configured = bool(settings.smtp_user and settings.email_recipients)
    if send_email or email_configured:
        send_recommendations(
            actionable,
            total_analysed=len(all_tickers),
            performance=perf,
            hypothetical_performance=hypothetical_perf,
            all_recommendations=recommendations,
            insider_trades=insider_trades,
            signals=signals,
            articles=articles,
            macro_context=macro_context,
            cot_context=cot_context,
            ipo_context=ipo_context,
            vix_context=vix_context,
            credit_context=credit_context,
            put_call_context=put_call_context,
            tick_context=tick_context,
            breadth_context=breadth_context,
            highs_lows_context=highs_lows_context,
            mcclellan_context=mcclellan_context,
            macro_surprise_context=macro_surprise_context,
            fedwatch_context=fedwatch_context,
            revision_momentum_context=revision_momentum_context,
            whisper_context=whisper_context,
            earnings_context=earnings_context,
            pead_context=pead_context,
            gex_context=gex_context,
            opex_context=opex_context,
            seasonality_context=seasonality_context,
            bond_internals_context=bond_internals_context,
            move_context=move_context,
            dix_context=dix_context,
            global_macro_context=global_macro_context,
            macro_regime_context=macro_regime_context,
            market_mode_context=market_mode_context,
            catalyst_timing_context=catalyst_timing_context,
            cluster_watchlist_context=cluster_watchlist_context,
            screener_context=screener_context,
            macro_discovery_context=macro_discovery_context,
            sector_pairs_context=sector_pairs_context,
            cointegration_context=coint_context,
            sector_rotation_context=sector_rotation_context,
            rotation_drivers_context=rotation_drivers_context,
            business_cycle_context=business_cycle_context,
            intermarket_context=intermarket_context,
            macro_news_context=macro_news_context,
            gate_diag=gate_diag,
            source_health=[s for s in _collect_sources() if not s.get("ok")],
            llm_health=llm_health,
            broker_health=broker_health,
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
