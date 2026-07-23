"""Main analysis pipeline: fetch → analyse → recommend → notify."""

import hashlib
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from loguru import logger
from datetime import datetime, timezone

from config import settings
from src.utils import now_et, fmt_et
from src.data.news_fetcher import fetch_all_news, fetch_cached_news, fetch_rss_news, fetch_google_news, _dedupe_by_url
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
from src.data.ticker_events import fetch_ticker_events
from src.data.earnings import fetch_earnings_surprises, fetch_earnings_context
from src.data.pead import fetch_pead_context
from src.data.fundamentals import fetch_fundamentals_context
from src.data.corporate_actions import fetch_corporate_actions_context
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
from src.data.provider_news import fetch_polygon_news, fetch_finnhub_news, fetch_alpha_vantage_news
from src.data.stocktwits import fetch_stocktwits_sentiment
from src.data.quiver import (
    fetch_congress_trades, fetch_gov_contracts, fetch_lobbying, fetch_offexchange,
)
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
from src.analysis.sentiment import reset_sentiment_providers, get_sentiment_provider_summary, get_dominant_sentiment_model
from src.analysis.data_quality import EXPECTED_SPARSE_SOURCES, KNOWN_DEAD_SOURCES, is_context_populated
from src.notifications.email_sender import send_recommendations
from src.performance.market_calendar import current_session
from src.performance.tracker import record_new_trades, update_open_trades, close_trades_on_signal_reversal, log_performance_summary, get_performance_for_email, get_open_trade_tickers, get_open_position_summaries, get_open_trades, monitor_open_positions, calibrate_sim_costs, reset_price_health, get_price_health, _method_scores_from_signal, _methods_agreeing, _dominant_method, _provider_of_synth_model, _confidence_floor, _LLM_ENGINES, RULE_FILL_MODEL as _RULE_FILL_MODEL
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
        attempts = ph.get("ibkr", 0) + ph.get("yfinance", 0) + ph.get("polygon", 0)
        failed = ph.get("failed") or []
        if attempts or failed:
            n = len(failed)
            shown = ", ".join(failed[:8]) + ("…" if n > 8 else "")
            sources.append({
                "label": "Live price feed (IBKR→yfinance→Polygon)",
                "enabled": True,
                "ok": n == 0,
                "error": f"{n} ticker(s) had no price: {shown}" if n else None,
                "duration_s": None,
            })
    except Exception:
        pass

    # FX sizing rate — a silent fallback to the static CAD→USD constant (or an
    # assumed 1.0) mis-sizes every converted order. Only reported when the broker
    # actually performed a non-USD conversion this run.
    try:
        from src.broker.fx import get_fx_health
        fxh = get_fx_health()
        fx_attempts = fxh["live"] + fxh["fallback"] + fxh["assumed_one"]
        if fx_attempts:
            bad = fxh["fallback"] + fxh["assumed_one"]
            sources.append({
                "label": "FX rate (sizing CAD→USD)",
                "enabled": True,
                "ok": bad == 0,
                "error": (f"live quote unavailable — {fxh['fallback']} fallback-rate / "
                          f"{fxh['assumed_one']} assumed-1.0 conversion(s)") if bad else None,
                "duration_s": None,
            })
    except Exception:
        pass

    # Correlation sizing — pairs that fail to compute are silently dropped, so a
    # high failure rate quietly weakens the concentration haircut.
    try:
        from src.performance.correlation import get_correlation_health
        ch = get_correlation_health()
        if ch["attempted"]:
            fail_pct = ch["failed"] / ch["attempted"]
            over = fail_pct > float(settings.correlation_health_max_fail_pct)
            sources.append({
                "label": "Correlation sizing",
                "enabled": True,
                "ok": not over,
                "error": (f"{ch['failed']}/{ch['attempted']} pair(s) ({fail_pct:.0%}) "
                          "failed to compute — haircut weakened") if over else None,
                "duration_s": None,
            })
    except Exception:
        pass

    # Macro-regime input coverage — too few surviving macro feeds means the
    # composite is untrustworthy (and now forced to fail-safe CAUTION).
    try:
        from src.data.macro_regime import get_regime_coverage
        rc = get_regime_coverage()
        if rc["total"]:
            ok = rc["available"] >= int(settings.macro_regime_min_inputs)
            sources.append({
                "label": "Macro regime inputs",
                "enabled": True,
                "ok": ok,
                "error": (f"only {rc['available']}/{rc['total']} macro inputs available "
                          "— regime forced to CAUTION (fail-safe)") if not ok else None,
                "duration_s": None,
            })
    except Exception:
        pass

    # Market-mode input coverage — low-stakes (NEUTRAL default applies no tilt),
    # so flagged unhealthy only when ALL structural inputs are dark.
    try:
        from src.data.market_mode import get_mode_coverage
        mc = get_mode_coverage()
        if mc["total"]:
            sources.append({
                "label": "Market mode inputs",
                "enabled": True,
                "ok": mc["available"] >= 1,
                "error": (f"0/{mc['total']} market-structure inputs available "
                          "— mode defaulted to NEUTRAL blindly") if mc["available"] < 1 else None,
                "duration_s": None,
            })
    except Exception:
        pass
    return sources


def _result_size(result) -> Optional[int]:
    """Item count of a fetcher result, for emptiness detection.

    0 for ``None`` or an empty container (the 'returned nothing' case); the
    length for a populated list/dict/str/etc.; ``None`` for a non-sized object
    (a present Pydantic context model has no length but is NOT empty — it ran and
    produced a structured result)."""
    if result is None:
        return 0
    if isinstance(result, (list, tuple, set, frozenset, dict, str, bytes)):
        return len(result)
    return None


def _safe(label: str, fn, *args, **kwargs):
    """Run fn; log warning and return None on any exception.

    Records the per-source outcome — ok / error / duration AND emptiness
    (``n_items`` + ``empty``) — into the per-run source log so the run's 'APIs
    used' record persists it to DuckDB. "Ran OK but returned nothing" is a
    distinct, first-class state (not silently ``ok``): it is logged — WARNING for
    a source expected to always return data, INFO for an event-driven sparse
    feed (see ``EXPECTED_SPARSE_SOURCES``) — and surfaced in the Data Quality
    dashboard tab so a silently-dark source gets investigated.
    """
    t0 = time.time()
    ok, err, result = True, None, None
    try:
        result = fn(*args, **kwargs)
    except Exception as e:
        # str(e) is '' for a bare TimeoutError etc. — fall back to the class name
        # so the source-health email shows a reason, not a blank (same blank-error
        # class fixed in ibkr._exc_text).
        ok, err = False, (str(e) or type(e).__name__)
        logger.warning(f"[{label}] fetch failed: {err}")
    finally:
        size = _result_size(result) if ok else None
        empty = bool(ok and size == 0)
        # A present-but-hollow / partially-filled context object (key fields
        # missing) counts as "returned nothing useful" even though it's not None.
        hollow = False
        if ok and not empty and is_context_populated(label, result) is False:
            empty, hollow = True, True
        with _SOURCE_LOCK:
            _SOURCE_LOG.append({
                "label": label,
                "ok": ok,
                "error": err,
                "duration_s": round(time.time() - t0, 3),
                "n_items": size,
                "empty": empty,
            })
        if empty:
            if label in KNOWN_DEAD_SOURCES:
                logger.debug(f"[{label}] returned no data (known-dead source, "
                             "no free replacement)")
            elif label in EXPECTED_SPARSE_SOURCES:
                logger.info(f"[{label}] returned no data this run "
                            "(event-driven source — may be normal)")
            elif hollow:
                logger.warning(f"[{label}] returned a HOLLOW/partial context — "
                               "present but key fields missing; investigate.")
            else:
                logger.warning(f"[{label}] returned NOTHING — ran OK but empty. "
                               "An always-on feed is dark; investigate.")
    return result


def _persist_run(run_id, start, finished, all_tickers, recommendations, actionable,
                 gate_diag, market_mode_context, macro_regime_context,
                 confidence_threshold, allow_buys, signals_by_ticker,
                 broker_report=None, snapshots=None,
                 synthesis_meta=None, sentiment_summary=None,
                 universe_sources=None) -> None:
    """Write the run, its per-source 'APIs used' record, every recommendation
    (with method attribution + the LLM provider that synthesised it), the
    broker reconcile report (per-order slippage/commission rows), and the FULL
    per-ticker signal cross-section (the learning panel) to DuckDB.

    ``synthesis_meta`` / ``sentiment_summary`` MUST be the values captured right
    after the MAIN synthesis: the opener-pinned hold-reviews call
    ``generate_recommendations`` again before this persist runs and clobber the
    process-global ``_LAST_SYNTHESIS_META`` / sentiment tallies, so re-reading
    them here recorded whatever engine the LAST review used, not the engine that
    produced the run's recommendations (observed 2026-07-01: a flash-thinking
    run persisted as plain flash — the dashboard's per-LLM eval keyed on it)."""
    try:
        actionable_ids = {id(r) for r in actionable}
        meta = synthesis_meta if synthesis_meta is not None else get_last_synthesis_meta()
        provider = meta.get("provider")
        # Per-rec attribution stores the EXACT engine id (legacy rows held only
        # the provider string) so the per-LLM evaluation never has to guess
        # which Claude model "anthropic" meant at the time.
        rec_llm = meta.get("model") or ("rule-based (no LLM)" if provider == "rule-based" else provider)

        rec_rows = []
        for r in recommendations:
            scores = _method_scores_from_signal(r.ticker, r.direction, signals_by_ticker)
            gen_at = r.generated_at
            _hsig = (signals_by_ticker or {}).get(r.ticker)
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
                # A back-filled ticker is attributed to the rule-based engine,
                # not to whichever model synthesised the rest of this run — it
                # was never asked about this ticker (tracker.RULE_FILL_MODEL).
                "llm_provider": _RULE_FILL_MODEL if getattr(r, "rule_filled", False) else rec_llm,
                "target_horizon": getattr(_hsig, "target_horizon", "") if _hsig else "",
                "horizon_net_edge_pct": float(getattr(_hsig, "horizon_net_edge_pct", 0.0) or 0.0) if _hsig else 0.0,
                "shadow_target_horizon": getattr(_hsig, "shadow_target_horizon", "") if _hsig else "",
                "shadow_direction": getattr(_hsig, "shadow_direction", "") if _hsig else "",
                "shadow_horizon_net_edge_pct": float(getattr(_hsig, "shadow_horizon_net_edge_pct", 0.0) or 0.0) if _hsig else 0.0,
                "expected_move_pct": float(getattr(_hsig, "expected_move_pct", 0.0) or 0.0) if _hsig else 0.0,
                "market_aligned": getattr(_hsig, "market_aligned", "") if _hsig else "",
                "upside_score": float(getattr(_hsig, "upside_score", 0.0) or 0.0) if _hsig else 0.0,
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
            "llm_sentiment_provider": (sentiment_summary if sentiment_summary is not None
                                       else get_sentiment_provider_summary()),
            "gate_diag": gate_diag,
        })
        sources = _collect_sources()
        repo.insert_run_sources(run_id, sources)
        repo.insert_recommendations(rec_rows)
        n_orders = 0
        if broker_report and broker_report.get("mode") not in (None, "off"):
            repo.insert_broker_report(run_id, broker_report)
            n_orders = len(broker_report.get("orders") or [])

        # Signals panel: persist EVERY ticker's full method-score vector — not
        # just the top-10 recommendations — so the cross-section accumulates
        # into a forward-testing dataset (news/options inputs can't be
        # reconstructed historically; a run not persisted is data lost).
        # Analyse with `python -m src.analysis.signal_panel`.
        price_by_ticker = {s.ticker: s.price for s in (snapshots or []) if getattr(s, "price", None)}
        sig_rows = []
        for tk, s in (signals_by_ticker or {}).items():
            # Base 19-method scores drive agreement / dominance (the trade-
            # attribution set); the multi-timeframe components are persisted
            # alongside for the panel's IC table but excluded from those counts.
            base_scores = _method_scores_from_signal(tk, s.direction, signals_by_ticker)
            tf_scores = getattr(s, "timeframe_scores", None) or {}
            fund_scores = getattr(s, "fundamental_scores", None) or {}
            # Agreement-quality pseudo-methods (2026-07-19): the cross-family net
            # vote + the raw-tape structure score ride the panel's scores JSON
            # (and the long-format sim reshape below) so their forward IC is
            # monitored like any method — WITHOUT joining the attribution set
            # (base_scores), the combine, or the exit consensus. Zero scores are
            # dropped by the reshape's no-view guard automatically.
            agree_scores = {
                "fam_net": float(getattr(s, "family_net_score", 0.0) or 0.0),
                "tape":    float(getattr(s, "tape_confirmation_score", 0.0) or 0.0),
            }
            all_scores = {**base_scores, **tf_scores, **fund_scores, **agree_scores}
            sig_rows.append({
                "ticker": tk,
                "type": str(getattr(s, "type", "STOCK") or "STOCK"),
                "direction": str(getattr(s.direction, "value", s.direction)),
                "combined_score": float(getattr(s, "combined_score", 0.0)),
                "confidence": float(s.confidence),
                "n_methods_agreeing": len(_methods_agreeing(base_scores, s.direction)),
                "dominant_method": _dominant_method(base_scores, s.direction),
                "price": price_by_ticker.get(tk),
                "universe_source": (universe_sources or {}).get(str(tk).upper()),
                # Confidence-formula components (2026-07-21) — verbatim factors from
                # the SAME confidence = raw × coherence × movement × volume × family ×
                # tape chain, so src/analysis/confidence_components.py can isolate
                # each one's forward-return contribution. See models.TickerSignal.
                "raw_confidence": float(getattr(s, "raw_confidence", 0.0)),
                "coherence_factor": float(getattr(s, "coherence_factor", 1.0)),
                "movement_factor": float(getattr(s, "movement_factor", 1.0)),
                "volume_factor": float(getattr(s, "volume_factor", 1.0)),
                "family_conf_factor": float(getattr(s, "family_conf_factor", 1.0)),
                "tape_conf_factor": float(getattr(s, "tape_conf_factor", 1.0)),
                "scores": all_scores,
            })
        if sig_rows:
            from src.utils import ET
            generated_at = start.isoformat()
            signal_date = start.astimezone(ET).date().isoformat()
            repo.insert_signals(
                run_id,
                generated_at=generated_at,
                signal_date=signal_date,
                rows=sig_rows,
            )
            # Long-format reshape: one simulated single-method trade per
            # (ticker, method) with a non-zero score — the unbiased dataset for
            # "would this method alone have called the direction right" over
            # EVERY scored ticker (vs the gate-selected ledger). combined_score
            # rides along as the synthesized-baseline "method".
            sim_rows = []
            for sr in sig_rows:
                px = sr.get("price")
                scores = dict(sr.get("scores") or {})
                scores["combined_score"] = sr.get("combined_score")
                for method, score in scores.items():
                    if score is None or abs(float(score)) < 1e-9:
                        continue  # exactly-zero / no-view methods are not a trade
                    sim_rows.append({
                        "ticker": sr["ticker"],
                        "method": method,
                        "score": float(score),
                        "direction": "BUY" if float(score) > 0 else "SELL",
                        "entry_price": px,
                    })
            if sim_rows:
                repo.insert_simulated_trades(
                    run_id, generated_at=generated_at, signal_date=signal_date,
                    rows=sim_rows,
                )

        logger.info(
            f"[db] Persisted run {run_id}: {len(rec_rows)} recommendation(s), "
            f"{len(sources)} source(s), {n_orders} broker order event(s), "
            f"{len(sig_rows)} signal row(s) → DuckDB"
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
    sentiment_ok        = bool(sent_summary) and any(
        e in sent_summary for e in _LLM_ENGINES)
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


# Drift actions that are self-resolving off-RTH convergence, not a failure —
# see _assess_broker_health. Kept as a set so reconcile.py and here can't drift
# apart on the label spelling.
_PENDING_DRIFT_ACTIONS = {"flatten_pending_open", "flatten_pending_fill"}


def _assess_broker_health(report: Optional[dict]) -> Optional[dict]:
    """Turn a broker reconcile report into a health verdict for the CRITICAL log
    and the email banner. Returns None when broker_mode=off / no report."""
    if not report or report.get("mode") in (None, "off"):
        return None
    problems = []
    if not report.get("connected"):
        problems.append("broker NOT connected — orders were not placed")
    if report.get("broker_timeouts"):
        # Not real IBKR rejects — the gateway did not respond (alive-but-wedged:
        # socket open, API dead). Actionable wording so the operator restarts the
        # gateway / checks IBC instead of hunting for a bad order.
        problems.append(
            f"broker NOT RESPONDING — {report['broker_timeouts']} request(s) timed out "
            f"(IB Gateway likely wedged; restart it / check IBC auto-restart)"
        )
    if report.get("rejects"):
        problems.append(f"{report['rejects']} order(s) rejected")
    if report.get("drift"):
        # Off-RTH positions still converging to the ledger are PENDING, NOT a
        # failure — exclude them so they don't force a CRITICAL/email every
        # off-RTH tick (they stay visible in the drift list + broker order log +
        # dashboard). Two flavours: the overnight venue rejects the flatten
        # outright ('flatten_pending_open', self-resolves at the pre-market
        # open), or it's accepted and just working a thin off-RTH book
        # ('flatten_pending_fill', e.g. PRCH 2026-07-10 — 6 pre-market ticks
        # before a clean fill). RTH drift always stays hard.
        hard = [d for d in report["drift"] if d.get("action") not in _PENDING_DRIFT_ACTIONS]
        if hard:
            names = ", ".join(d["ticker"] for d in hard[:6])
            flattened = report.get("drift_flattened", 0)
            suffix = (f"; auto-flatten submitted for {flattened}" if flattened
                      else "; report-only" if all(
                          d.get("action") in (None, "report") for d in hard)
                      else "; auto-flatten FAILED — check broker order log")
            problems.append(
                f"{len(hard)} position(s) drifted from the ledger ({names}){suffix}"
            )
    if not report.get("ok") and report.get("errors"):
        problems.append("reconcile error")
    return {
        "down":           bool(problems),
        "mode":           report.get("mode"),
        "connected":      report.get("connected"),
        "entries":        report.get("entries_submitted", 0),
        "exits":          report.get("exits_submitted", 0),
        "fills_repaired": report.get("fills_repaired", 0),
        "rejects":        report.get("rejects", 0),
        "broker_timeouts": report.get("broker_timeouts", 0),   # gateway-unresponsive submits
        # Benign: overnight-venue refusals resubmitted at the pre-market open
        # (visible here, never a problem — see reconcile._tally_submit_failure).
        "overnight_deferred": report.get("overnight_deferred", 0),
        # Reliability observability: transient submit retries this tick and
        # stale resting orders cancelled + re-anchored. Neither is a failure
        # by itself (the mechanism worked) — they appear in the healthy line
        # so a flaky Gateway is visible before it becomes rejects.
        "retries":        report.get("retries", 0),
        "stale_cancels":  report.get("stale_cancels", 0),
        "entry_cancels_on_close": report.get("entry_cancels_on_close", 0),
        "drift_flattened": report.get("drift_flattened", 0),
        "drift":          report.get("drift", []),
        # Benign: off-RTH positions still converging to the ledger — surfaced
        # (not a problem, so no alert).
        "drift_pending":  [d["ticker"] for d in report.get("drift", [])
                           if d.get("action") in _PENDING_DRIFT_ACTIONS],
        "slippage":       report.get("slippage", []),
        "message":        "; ".join(problems),
        # The EXACT per-order failure reasons (e.g. "exit NET: Cancelled: Error
        # 10329 … directly routed to OVERNIGHT … Precautionary Settings") — order-
        # preserving dedupe. The concise `message` above is the count summary; this
        # is what the operator actually needs to diagnose, surfaced in the email
        # banner + the CRITICAL log so no one has to open the broker order log.
        "errors":         list(dict.fromkeys(report.get("errors", []))),
    }


def _assess_price_provenance(run_id, snapshots) -> Optional[dict]:
    """Flag trades opened THIS run whose recorded entry_price diverges from the
    run's snapshot price beyond a session-appropriate band.

    The standing, automatic version of the one-off fill-vs-snapshot audit run on
    the 2026-06-15 stale-price bug (CRDO entered at Friday's stale close while the
    snapshot — and the live pre-market print — was ~4.5% higher). ``entry_price``
    comes from ``tracker._fetch_price`` (a SEPARATE live fetch from the analysis
    snapshot), so a feed/staleness bug shows up as a large divergence here even
    though both *should* agree. RTH uses a tight band; off-hours allows more drift
    between the snapshot and the entry fetch on a thin tape.

    Returns ``{down, n_checked, flagged:[...], message}`` or None when the check is
    disabled / there is nothing to compare.
    """
    if not settings.enable_price_provenance_check or not run_id:
        return None
    # Only compare against LIVE snapshot prices — a "prev_close" fallback anchor
    # (grouped-daily fill when no live quote existed) vs a live fill would diverge
    # legitimately and falsely trip the band.
    price_by_ticker = {s.ticker: float(s.price) for s in (snapshots or [])
                       if getattr(s, "price", None)
                       and getattr(s, "price_source", "live") == "live"}
    if not price_by_ticker:
        return None
    bands = {
        "rth":       settings.price_provenance_band_rth_bps,
        "extended":  settings.price_provenance_band_extended_bps,
        "overnight": settings.price_provenance_band_overnight_bps,
    }
    try:
        trades = repo.load_trades()
    except Exception as e:
        logger.debug(f"[price] provenance check could not read trades: {e}")
        return None

    flagged, n_checked = [], 0
    for t in trades:
        if t.get("run_id") != run_id:
            continue
        entry = t.get("entry_price")
        snap = price_by_ticker.get(t.get("ticker"))
        if not entry or not snap or float(snap) <= 0:
            continue
        n_checked += 1
        session = t.get("entry_session") or "rth"
        band = bands.get(session, bands["rth"])
        bps = abs(float(entry) - float(snap)) / float(snap) * 1e4
        if bps > band:
            flagged.append({
                "ticker":         t.get("ticker"),
                "entry_price":    round(float(entry), 4),
                "snapshot_price": round(float(snap), 4),
                "bps":            round(bps, 1),
                "session":        session,
                "band":           band,
            })
    if not n_checked:
        return None
    message = ""
    if flagged:
        names = ", ".join(f"{f['ticker']} {f['bps']:.0f}bp" for f in flagged[:6])
        message = (f"{len(flagged)} of {n_checked} new trade(s) entered at a price "
                   f"far from the run snapshot ({names}) — possible stale-price/feed bug")
    return {"down": bool(flagged), "n_checked": n_checked, "flagged": flagged, "message": message}


def _fetch_news(tickers, sectors):
    # Cache-worthy bundle (per-ticker yfinance + NewsAPI): hourly cache so the
    # rate-limited / quota-bound feeds aren't re-hammered every 30-min tick.
    cached = load_news()
    if cached is None:
        cached = fetch_cached_news(tickers, sectors)
        save_news(cached)
        logger.info(f"[news] Fetched cache-worthy bundle ({len(cached)} articles, hourly cache)")
    else:
        logger.info(f"[news] Using cached per-ticker + NewsAPI bundle ({len(cached)} articles)")
    # Fast-lane: RSS + press-release wires fetched FRESH every tick (never cached),
    # mirroring the 8-K fast path — so a catalyst breaking between the hourly
    # refreshes is seen at the NEXT 30-min tick instead of up to ~an hour later.
    fresh_rss = fetch_rss_news()
    articles = _dedupe_by_url(cached + fresh_rss)
    logger.info(
        f"[news] {len(articles)} articles ({len(cached)} cached + "
        f"{len(fresh_rss)} live RSS/wires, deduped)"
    )
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


class _ReviewMap(dict):
    """``{ticker: Recommendation}`` with an ``engines`` side-channel
    (ticker → provider that actually judged it) so callers that only need the
    reviews keep treating it as a plain dict, while ``_persist_trade_reviews``
    can record which engine produced each row (pinned vs fallback)."""
    engines: dict = {}


def _hold_review_groups(open_trades, run_sent=None, run_synth=None):
    """Group held positions by their OPENING (sentiment, synthesis) engine combo.

    LLM-opened positions are pinned to their stamped engines (Fix #2). Legacy /
    rule-based-opened positions have nothing to pin: when this run's engines are
    KNOWN (``run_sent``/``run_synth`` supplied — the sequential path), they are
    merged into that combo so the exit is still LLM-judged rather than handed to
    the poor aggregator-decay backstop (30%-win historically); when they are NOT
    yet known (the overlap branch starts before Step 5 resolves them), they are
    returned separately so the branch can review them once the engines arrive.
    Returns ``(pinned_groups, legacy_tickers)``.
    """
    from collections import defaultdict

    groups: dict = defaultdict(list)
    legacy: list = []
    run_is_llm = run_sent in _LLM_ENGINES and run_synth in _LLM_ENGINES
    # Group each LLM-opened position by the (sentiment, synthesis) engines that
    # OPENED it, so the hold-review re-judges with the same engines (Fix #2). The
    # legacy Qwen-primary override collapsed every opener into one ("qwen","qwen")
    # review pass (all LLM calls on Qwen); under llm_primary_provider=="deepseek"
    # (2026-07-13) it's off, restoring the per-opener same-engine invariant that the
    # 50/50 DeepSeek/Qwen synthesis split needs. Re-arms only if provider→"qwen".
    qwen_primary = settings.llm_primary_provider == "qwen"
    for t in open_trades:
        se = _provider_of_synth_model(t.get("llm_sentiment_model"))
        sy = _provider_of_synth_model(t.get("llm_synthesis_model"))
        if se in _LLM_ENGINES and sy in _LLM_ENGINES:
            key = ("qwen", "qwen") if qwen_primary else (se, sy)
            groups[key].append(t["ticker"])               # opener-pinned (Fix #2)
        elif run_is_llm:
            key = ("qwen", "qwen") if qwen_primary else (run_sent, run_synth)
            groups[key].append(t["ticker"])
        elif run_sent is None and run_synth is None:
            legacy.append(t["ticker"])                    # engines resolve later (branch)
    return dict(groups), legacy


# Cross-engine order tried when an opener-PINNED hold-review engine can't answer.
# DeepSeek leads: it is the cheap, funded workhorse the rest of the system falls
# back to (synthesis `_synthesis_attempts_for`, sentiment `_sentiment_engine_order`).
_HOLD_REVIEW_FALLBACK_ORDER = ("deepseek", "qwen", "anthropic")


def hold_review_fallbacks(pinned: str) -> list:
    """Engines to try, in order, when ``pinned`` produced no review this tick.

    EVERY other provider is tried, not just one. Until 2026-07-22 this was a
    single fixed alternate (``"anthropic" if pinned == "deepseek" else "deepseek"``),
    so a DeepSeek pin dead-ended on an Anthropic account that was out of credits
    and never reached Qwen — leaving the position with no exit gate for the tick.
    """
    return [e for e in _HOLD_REVIEW_FALLBACK_ORDER if e != pinned]


def _run_hold_reviews(groups, legacy_tickers, sectors, build_kwargs, session,
                      synth_kwargs_wait, run_engines_wait):
    """The pinned hold-review machinery: fresh refetch + per-combo re-judgment.

    For every held position, produce TODAY's recommendation using the SAME
    synthesis + sentiment engines that opened it, on FRESHLY refetched news +
    prices — so ``monitor_open_positions`` compares entry-vs-now confidence
    apples-to-apples (identical engines, temperature=0 ⇒ low volatility).

    Two blocking waits decouple this from the main pipeline so it can run
    CONCURRENTLY with Steps 4–5 (the overlap branch) or inline (sequential path,
    where both waits return immediately):
      • ``synth_kwargs_wait()`` → the synthesis context kwargs. Each combo's
        ``build_signals`` starts IMMEDIATELY (overlapping the main Step-4 scoring
        pass); only the pinned SYNTHESIS blocks here, because the review prompt
        must carry the identical context blocks as the main pass — including the
        Step-4.5 catalyst timing context, which needs the main signals.
      • ``run_engines_wait()`` → this run's (sentiment, synthesis) engines,
        consulted only for legacy positions with no opener stamps (resolved
        after Step 5).
    Either wait returning None aborts that stage fail-soft — no review means the
    position simply holds this tick, same as a failed news refetch.

    Engine fallback (``hold_review_engine_fallback``): a combo whose pinned
    synthesis engine produces nothing (e.g. Anthropic credits exhausted — observed
    2026-06-26→07-01, which left every Claude-opened position ungoverned for days)
    is re-judged once by the OTHER provider on the same freshly-built signals.
    Cross-engine confidence is not apples-to-apples, but an available judge beats
    an absent one. The returned mapping carries ``.engines`` (ticker → provider
    that actually judged it) so ``_persist_trade_reviews`` records fallback
    reviews honestly.
    """
    if not groups and not legacy_tickers:
        return {}
    held = sorted({tk for tks in groups.values() for tk in tks} | set(legacy_tickers))
    # FRESH market data (bypassing the hourly caches), every tick, for the held set.
    try:
        fresh_articles = fetch_all_news(held, sectors)
    except Exception as e:
        logger.warning(f"[hold_review] fresh news fetch failed ({e}) — reviews skipped this tick")
        return {}
    try:
        fresh_snaps = get_snapshots(held)
    except Exception as e:
        logger.warning(f"[hold_review] fresh snapshot fetch failed ({e}) — proceeding without it")
        fresh_snaps = []

    review_engines: dict = {}   # ticker → provider that actually judged it this tick

    def _review(item):
        (se, sy), tickers = item
        want = set(tickers)

        out: dict = {}
        engine_used = sy
        try:
            # Signal build first — needs only build_kwargs, so in the overlap
            # branch it runs while the main Step 4 is still scoring.
            sub = build_signals(tickers, fresh_articles, snapshots=fresh_snaps,
                                session=session, force_sentiment_engine=se, **build_kwargs)
        except Exception as e:
            logger.warning(f"[hold_review] combo (sent={se}, synth={sy}) signal build failed: {e}")
            return {}
        synth_kwargs = synth_kwargs_wait()
        if synth_kwargs is None:
            logger.warning(
                f"[hold_review] synthesis context never arrived — combo "
                f"(sent={se}, synth={sy}) skipped (positions hold this tick)")
            return {}

        def _attempt(engine):
            recs = generate_recommendations(sub, session=session, force_engine=engine, **synth_kwargs)
            # force_engine returns [] on failure, so any non-empty result is from `engine`.
            return {r.ticker: r for r in (recs or []) if r.ticker in want}

        try:
            out = _attempt(sy)
        except Exception as e:
            logger.warning(f"[hold_review] combo (sent={se}, synth={sy}) failed: {e}")
        # Pinned engine unavailable → EVERY other provider re-judges in turn
        # rather than leaving the position with no exit gate this tick.
        # 2026-07-22: this used to try exactly ONE alternate ("anthropic" for a
        # deepseek pin, else "deepseek"), so a deepseek pin dead-ended on a broke
        # Anthropic account and never reached Qwen. DeepSeek leads the order — it
        # is the cheap, funded workhorse the rest of the system falls back to.
        if not out and settings.hold_review_engine_fallback:
            for other in hold_review_fallbacks(sy):
                try:
                    out = _attempt(other)
                except Exception as e:
                    logger.warning(f"[hold_review] fallback engine {other} failed too: {e}")
                    out = {}
                if out:
                    engine_used = other
                    logger.warning(
                        f"[hold_review] pinned engine {sy} produced no review — "
                        f"{len(out)} position(s) re-judged by {other} (fallback)"
                    )
                    break
        for tk in out:
            review_engines[tk] = engine_used
        return out

    reviews: dict = {}
    items = list(groups.items())
    if len(items) == 1:
        reviews.update(_review(items[0]))
    elif items:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(items)) as ex:
            for part in ex.map(_review, items):
                reviews.update(part)

    # Legacy / rule-based-opened positions (no opener stamps): reviewed with THIS
    # run's engines once they are known — immediately on the sequential path,
    # after Step 5 on the overlap branch. A non-LLM run leaves them unreviewed
    # (the aggregator backstop still governs them), exactly as before.
    if legacy_tickers:
        engines = run_engines_wait()
        if engines and all(e in _LLM_ENGINES for e in engines):
            reviews.update(_review((tuple(engines), sorted(set(legacy_tickers)))))
        elif engines is None:
            logger.warning(
                "[hold_review] run engines never arrived — legacy positions not reviewed this tick")

    logger.info(
        f"[hold_review] pinned: {len(reviews)}/{len(held)} held position(s) re-judged "
        f"by their opening engines across {len(items) + (1 if legacy_tickers else 0)} engine combo(s)"
    )
    out_map = _ReviewMap(reviews)
    out_map.engines = dict(review_engines)
    return out_map


def _build_hold_reviews(open_trades, run_sent, run_synth, full_recs, sectors,
                        build_kwargs, synth_kwargs, session):
    """Fix #2 — opener-pinned, fresh-data hold-review (one entry per held position).

    Sequential entry point (the overlap branch is ``_HoldReviewBranch``): groups
    positions with this run's engines already known, then runs the pinned
    machinery inline — both waits resolve immediately. Returns
    ``{ticker: Recommendation}``.

    Off (``enable_pinned_hold_review`` false): the cheap fallback — reuse THIS
    run's recs, but only for positions whose opening engines BOTH match this
    run's engines (no extra LLM calls, no refetch).
    """
    groups, _legacy = _hold_review_groups(open_trades, run_sent, run_synth)
    if not groups:
        return {}

    # Cheap fallback: reuse this run's recs for positions whose engines BOTH match.
    if not settings.enable_pinned_hold_review:
        by_ticker = {r.ticker: r for r in full_recs}
        return {
            tk: by_ticker[tk]
            for (se, sy), tks in groups.items() if se == run_sent and sy == run_synth
            for tk in tks if tk in by_ticker
        }

    return _run_hold_reviews(
        groups, [], sectors, build_kwargs, session,
        synth_kwargs_wait=lambda: synth_kwargs,
        run_engines_wait=lambda: (run_sent, run_synth),
    )


class _HoldReviewBranch:
    """Runs the opener-pinned hold-review CONCURRENTLY with pipeline Steps 4–5.

    The review needs no output of the main synthesis (it fetches its own fresh
    news/snapshots and re-judges with the OPENING engines), yet it used to run
    strictly after Step 5 — ~2.5 min of serial tick→order latency. Constructed
    right after ``build_kwargs`` exists (pre-Step 4, after ``update_open_trades``
    has refreshed marks + OHLCV on the main thread): the news refetch and each
    combo's ``build_signals`` overlap the main Step-4 scoring pass, and each
    combo's pinned synthesis blocks until ``supply_synth_kwargs`` (called once
    the synthesis context is complete, just before Step 5) so the review prompt
    carries the identical context blocks — including catalyst timing — as the
    main pass. Legacy (unstamped) positions additionally wait for
    ``supply_run_engines`` (right after Step 5). ``result()`` joins the branch
    before the monitor consumes the reviews.

    Provenance safety: forced-engine synthesis/sentiment calls do NOT touch the
    process-global last-synthesis meta or the sentiment provider tallies (see
    claude_analyst/_set_synthesis_meta and sentiment._record_sentiment_provider),
    so running concurrently with the main pass cannot mis-stamp the run.

    If the pipeline dies before supplying a future, the waits time out and the
    branch returns fail-soft (no reviews → positions hold this tick), matching
    the semantics of a failed news refetch.
    """

    _WAIT_TIMEOUT_S = 1200.0

    def __init__(self, open_trades, sectors, build_kwargs, session):
        from concurrent.futures import Future, ThreadPoolExecutor
        self._synth_kwargs: "Future" = Future()
        self._run_engines: "Future" = Future()
        groups, legacy = _hold_review_groups(open_trades)
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hold-review")
        self._fut = self._ex.submit(
            _run_hold_reviews, groups, legacy, sectors, build_kwargs, session,
            self._waiter(self._synth_kwargs), self._waiter(self._run_engines),
        )

    def _waiter(self, fut):
        timeout = self._WAIT_TIMEOUT_S

        def wait():
            try:
                return fut.result(timeout=timeout)
            except Exception:
                return None
        return wait

    def supply_synth_kwargs(self, synth_kwargs) -> None:
        if not self._synth_kwargs.done():
            self._synth_kwargs.set_result(dict(synth_kwargs))

    def supply_run_engines(self, run_sent, run_synth) -> None:
        if not self._run_engines.done():
            self._run_engines.set_result((run_sent, run_synth))

    def result(self):
        try:
            return self._fut.result()
        finally:
            self._ex.shutdown(wait=False)


def _persist_trade_reviews(run_id, hold_reviews, open_trades):
    """Append this tick's opener-pinned reviews to the ``trade_reviews`` table —
    the confidence-over-time trajectory the dashboard plots per ticker. Each row
    pairs the review's fresh confidence/action with the position's entry
    confidence + current price/return so deterioration can be read against the
    entry baseline, the close threshold, and the eventual direction change.
    Exception-safe: a DB hiccup never breaks the run."""
    if not hold_reviews:
        return
    by_ticker = {t["ticker"]: t for t in open_trades}
    review_engines = getattr(hold_reviews, "engines", {}) or {}
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for ticker, rec in hold_reviews.items():
        t = by_ticker.get(ticker) or {}
        entry_conf = t.get("confidence")
        # Normally the reviewing engine IS the opener (pinned), so the opener's
        # exact model id is recorded. A fallback review (pinned engine down) is
        # stamped with the provider that actually judged it, marked as such.
        synth_stamp = t.get("llm_synthesis_model")
        opener_prov = _provider_of_synth_model(synth_stamp)
        reviewer = review_engines.get(ticker)
        if reviewer and opener_prov in _LLM_ENGINES and reviewer != opener_prov:
            synth_stamp = f"{reviewer} (fallback)"
        rows.append({
            "run_id": run_id,
            "reviewed_at": now,
            "ticker": ticker,
            "position_id": t.get("recommendation_id"),   # NOTE: recommendation_id, NOT trade_id
            "entry_datetime": t.get("entry_datetime"),
            "confidence": getattr(rec, "confidence", None),
            "action": getattr(rec, "action", None),
            "direction": getattr(rec, "direction", None),
            "conf_floor": _confidence_floor(entry_conf),
            "entry_confidence": entry_conf,
            "entry_action": t.get("action"),
            "price": t.get("current_price"),
            "return_pct": t.get("return_pct"),
            "synthesis_model": synth_stamp,
            "sentiment_model": t.get("llm_sentiment_model"),
        })
    try:
        repo.insert_trade_reviews(rows)
        logger.info(f"[hold_review] persisted {len(rows)} review observation(s) to trade_reviews")
    except Exception as e:
        logger.warning(f"[hold_review] persisting trade_reviews failed: {e}")


def _persist_exit_signals(run_id, hold_reviews, open_trades, signals_by_ticker, macro_regime_context):
    """Append this tick's per-held-position exit-method scores to the ``exit_signals``
    panel — the unbiased dataset that powers the dashboard's Exit Performance IC
    table. One row per (open position, non-zero exit method): the synthesized
    ``llm_review`` that actually decides plus the macro/horizon/aggregator overlays
    and the entry signal methods re-scored as exit signals, each as a signed
    hold-conviction (see ``analysis.exit_methods``). Exception-safe: a DB hiccup
    never breaks the run."""
    if not open_trades:
        return
    from src.analysis.exit_methods import build_exit_scores
    from src.utils import ET
    now = datetime.now(timezone.utc).isoformat()
    signal_date = datetime.now(ET).date().isoformat()
    rows = []
    for t in open_trades:
        hr = (hold_reviews or {}).get(t["ticker"])
        scores = build_exit_scores(t, hr, signals_by_ticker, macro_regime_context)
        for method, score in scores.items():
            rows.append({
                "reviewed_at": now,
                "signal_date": signal_date,
                "ticker": t["ticker"],
                "position_id": t.get("recommendation_id"),
                "entry_direction": t.get("direction"),
                "method": method,
                "score": score,
                "price": t.get("current_price"),
            })
    try:
        repo.insert_exit_signals(run_id, rows)
        logger.info(f"[exit_panel] persisted {len(rows)} exit-method score(s) across "
                    f"{len(open_trades)} held position(s) to exit_signals")
    except Exception as e:
        logger.warning(f"[exit_panel] persisting exit_signals failed: {e}")


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


def _email_decision(*, observe_only: bool, send_email: bool,
                    email_if_configured: bool, email_configured: bool,
                    health_problem: bool) -> str:
    """Resolve the per-tick email gate to 'observe' | 'suppress' | 'send' | 'skip'.

    A detected problem (``health_problem``) forces 'send' even on a non-email slot
    — the always-alert-on-a-problem guarantee — and callers only set health_problem
    when email is configured, so the forced send always has working SMTP. Pure, so
    the gating is unit-tested directly (tests/test_email_on_problem.py)."""
    if observe_only:
        return "observe"
    if not send_email and not email_if_configured and not health_problem:
        return "suppress"
    if send_email or email_configured or health_problem:
        return "send"
    return "skip"


def _is_tradeable(ticker: str, budget: dict) -> bool:
    """Tradeable-liquidity gate (Gate 4 of the actionable filter): True iff
    ``ticker`` clears the higher TRADE price + 20-day dollar-volume floor
    (``trade_min_price`` / ``trade_min_dollar_volume``). False → OBSERVE-ONLY: the
    name is still scored + persisted to the signals panel (so penny / thin-volume
    performance keeps accruing) but never opens a trade. Fail-closed (``is_liquid``
    returns False when liquidity can't be verified). Gate off → always tradeable."""
    if not getattr(settings, "enable_trade_liquidity_gate", False):
        return True
    from src.data.liquidity import is_liquid
    return is_liquid(ticker, budget, settings.trade_min_price,
                     settings.trade_min_dollar_volume)


def _recent_runup_pct(ticker: str):
    """Trailing close-to-close return (%) over the last
    ``overextension_lookback_bars`` COMPLETED daily bars, from the same cached
    OHLCV the scorers read (forming bar already dropped during RTH). ``None``
    when it can't be computed (no cache, short history, bad closes) — the
    overextension gate FAILS OPEN on None."""
    try:
        from src.data.cache import load_ohlcv
        bars = load_ohlcv(ticker)
        if bars is None or bars.empty or "Close" not in bars.columns:
            return None
        closes = bars["Close"].tolist()
        n = max(1, int(settings.overextension_lookback_bars))
        if len(closes) < n + 1:
            return None
        last, prev = float(closes[-1]), float(closes[-(n + 1)])
        if not (last > 0 and prev > 0) or last != last or prev != prev:
            return None
        return (last - prev) / prev * 100.0
    except Exception:
        return None


def _is_overextended(ticker: str) -> bool:
    """Overextension / anti-chase gate (Gate 5 of the actionable filter,
    2026-07-22): True when the ticker already ran more than
    ``overextension_runup_pct`` over the trailing lookback window — the cohort
    the BUY-vs-SELL forensics measured at a 32.5% hit rate / −4.1% median 5d
    excess (chasing into short-term reversal; see settings for the full
    evidence). Applied to BUYs ONLY at the call site — SELLs on extended or
    crashed names are measured edge and must never be gated here. Fail-open:
    unknown run-up → not overextended. Gate off → never blocks."""
    if not getattr(settings, "enable_overextension_gate", False):
        return False
    runup = _recent_runup_pct(ticker)
    return runup is not None and runup > settings.overextension_runup_pct


def _passes_agreement_gate(direction: str, sig) -> bool:
    """Agreement floor (Gate 1b of the actionable filter, 2026-07-20): True unless
    ``sig`` exists, ``direction`` (the recommendation's own call) matches the
    aggregator's own ``sig.direction``, AND ``sig.sources_agreeing`` is below
    ``min_sources_agreeing_gate`` — mechanically enforcing the CLAUDE.md-documented
    "a single strong signal source never produces a BUY/SELL" invariant, which
    previously had NO code-level check (only a prompt instruction).

    ``sig.sources_agreeing`` is the aggregator's own count of ACTIVE weighted
    methods (post win-rate-filter, effective/inversion-corrected sign) agreeing
    with ``sig.direction`` — only meaningful for THIS call when the LLM's own
    direction matches it (the ~96% common "echo" case, 2026-07-12 agreement
    study). A genuine LLM override (``direction != sig.direction``) or a missing
    signal has no correctly-attributable count, so it PASSES this gate unchecked
    (same as before this gate existed) — gate off (``enable_agreement_gate``)
    always passes too."""
    if not settings.enable_agreement_gate or sig is None or direction != sig.direction:
        return True
    return sig.sources_agreeing >= settings.min_sources_agreeing_gate


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(send_email: bool = False, observe_only: bool = False,
                 email_if_configured: bool = True) -> None:
    """Execute the full analysis pipeline.

    ``observe_only`` (extended-hours observation ticks): the full analysis runs
    and is persisted to DuckDB (runs + recommendations + signals — the
    session-tagged evidence base), but every ledger/broker mutation is skipped
    — no trade opens/closes/marks, no position monitoring, no hypothetical
    updates, no orders — and no email is assembled or sent.

    ``email_if_configured``: legacy convenience for MANUAL runs — when True
    (default), a configured SMTP sends the report even without
    ``send_email``. The scheduler passes False so its per-slot ``send_email``
    decision is authoritative: extended trading ticks run the full ledger
    path but must never re-send the daily report.
    """
    start = datetime.now(timezone.utc)
    run_id = start.strftime("%Y-%m-%d_%H%M%S")
    _reset_source_log()
    reset_sentiment_providers()
    reset_price_health()
    # Per-run health counters for the silent-default sites surfaced via
    # _collect_sources (FX sizing rate, correlation-pair compute, regime/mode
    # input coverage) — reset so each run's source-health reflects only this tick.
    try:
        from src.broker.fx import reset_fx_health
        from src.performance.correlation import reset_correlation_health
        from src.data.macro_regime import reset_regime_coverage
        from src.data.market_mode import reset_mode_coverage
        reset_fx_health(); reset_correlation_health()
        reset_regime_coverage(); reset_mode_coverage()
    except Exception:
        pass
    # Session classification, fixed once per run so every consumer (weight
    # overlay, ext_gap scorer, threshold bump, analyst prompt, gate diag)
    # sees the same answer even when the run straddles a session boundary.
    run_session = current_session()
    logger.info(f"Pipeline started at {fmt_et(start)} (session: {run_session})")
    if run_session != "rth":
        logger.info(
            f"[extended] {run_session}-session run — extended signal profile active: "
            "stale-options down-weight, news up-weight, ext_gap scorer, "
            f"+{settings.extended_confidence_bump:.0%} actionable threshold"
        )

    tickers     = settings.stocks_list
    sectors     = settings.sectors_list
    commodities = settings.commodities_list

    # ── Step 0: Ticker discovery (serial — rest of pipeline depends on it) ──
    logger.info("Step 0: Discovering trending tickers...")
    all_tickers = get_trending_tickers(tickers, sectors)

    # Universe PROVENANCE (2026-07-03): which discovery source first surfaced
    # each ticker this run — stamped onto every signals row and every new trade
    # so per-source hit rates become measurable (the prerequisite for an
    # adaptive discovery budget). First source wins; the pinned watchlist /
    # sector / commodity sets take precedence over the trending overlap.
    universe_source: dict = {}

    def _mark_source(tks, label):
        for _t in tks or []:
            universe_source.setdefault(str(_t).upper(), label)

    _mark_source(tickers, "watchlist")
    _mark_source(sectors, "sector_etf")
    _mark_source(commodities, "commodity")
    _mark_source(all_tickers, "trending")
    new_commodities = [t for t in commodities if t not in all_tickers]
    if new_commodities:
        all_tickers = all_tickers + new_commodities
        _mark_source(new_commodities, "commodity")
        logger.info(f"Step 0: Pinned commodities: {new_commodities}")

    # Factor / thematic ETFs — pinned like commodities (Section E): broaden coverage beyond the
    # 11 GICS sectors with style-factor (momentum/quality/value/size/low-vol/growth) and
    # high-interest-theme (semis, software, biotech, defense, clean energy, …) ETFs.
    new_factors = [t for t in settings.factor_list if t not in all_tickers]
    if new_factors:
        all_tickers = all_tickers + new_factors
        _mark_source(new_factors, "factor_etf")
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
                _mark_source(new_from_screen, "screener")
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
            _mark_source(_new_earn, "earnings_discovery")
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
            _mark_source(_new_analyst, "analyst_discovery")
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
            _mark_source(new_from_cluster, "insider_cluster")
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
        _mark_source(new_from_trades, "open_position_pin")
        logger.info(f"[tracker] Pinning open-trade tickers into universe: {new_from_trades}")

    # Pin hypothetical always-open trade tickers so their OHLCV cache stays
    # current — the daily-NAV compound walk needs a fresh close per held day.
    hypothetical_tickers = [t for t, _ in settings.hypothetical_trades_list]
    new_from_hyp = [t for t in hypothetical_tickers if t not in all_tickers]
    if new_from_hyp:
        all_tickers = all_tickers + new_from_hyp
        _mark_source(new_from_hyp, "hypothetical_pin")
        logger.info(f"[hypothetical] Pinning always-open tickers into universe: {new_from_hyp}")

    # Related-company peer discovery (Massive) — widen with peers of the watchlist +
    # held names; added to the discovered set so it is liquidity-gated below (never raw).
    if settings.enable_related_discovery:
        try:
            from src.data.related_companies import discover_related_tickers
            _related = discover_related_tickers(list(tickers) + list(open_trade_tickers),
                                                max_results=settings.related_discovery_max)
            _new_rel = [t for t in _related if t not in all_tickers]
            if _new_rel:
                all_tickers = all_tickers + _new_rel
                _mark_source(_new_rel, "related_peer")
                logger.info(f"[related] Peer-discovery added {len(_new_rel)} name(s): {_new_rel[:12]}")
        except Exception as e:
            logger.warning(f"[related] peer discovery failed: {e}")

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

    # Drop junk tickers ("N/A" etc.) that bypassed the gate via a protected path
    # (e.g. a malformed open-trade / hypothetical row) — they otherwise reach
    # yfinance and spam "'Response' object is not subscriptable". The gate itself
    # now also validates, so this covers only the never-gated pinned set.
    from src.data.market_data import sanitize_tickers
    _pre_sane = len(all_tickers)
    all_tickers = sanitize_tickers(all_tickers)
    if len(all_tickers) != _pre_sane:
        logger.warning(f"[universe] Dropped {_pre_sane - len(all_tickers)} invalid ticker(s) from the universe")

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

    # Per-ticker signal fetchers must cover the DISCOVERED universe, not the
    # static watchlist. `stocks_list` is empty in this deployment (the universe
    # is built entirely by Step-0 discovery), so the news / PEAD / put-call / EPS
    # / analyst / short / revision / whisper / trends / reddit fetchers — all
    # wired to `tickers` below — were fetching for an EMPTY list and scoring 0
    # across the board (the root cause behind the dead news/PEAD/put-call
    # signals). The trending seed and the liquidity-gate protected set above
    # already ran on the original static list, so reassigning here only affects
    # those fetchers. Full all_tickers coverage is the deliberate cost tradeoff.
    tickers = all_tickers

    with ThreadPoolExecutor(max_workers=14, thread_name_prefix="pipeline") as pool:

        # Group A: non-yfinance-options — submit all at once, run concurrently
        f_news         = pool.submit(_safe, "news", _fetch_news, tickers, sectors)
        f_snapshots    = pool.submit(_safe, "snapshots", _fetch_snapshots, all_tickers)

        # Provider news feeds (flag-gated). Polygon carries per-article sentiment
        # insights (feeds the provider-sentiment hybrid → LLM-skip); Finnhub adds
        # real-time news coverage (no sentiment on the free tier).
        f_polygon_news = (pool.submit(_safe, "polygon_news", fetch_polygon_news, all_tickers)
                          if settings.enable_polygon_news else None)
        f_finnhub_news = (pool.submit(_safe, "finnhub_news", fetch_finnhub_news, tickers)
                          if (settings.enable_finnhub_news and settings.finnhub_api_key) else None)

        # Per-ticker Google News RSS (free, no key) — fresh every tick. Widens
        # coverage to Reuters/Bloomberg/Barron's/FT and closes the Business Wire gap.
        f_google_news = (pool.submit(_safe, "google_news", fetch_google_news, tickers)
                         if settings.enable_google_news else None)

        # Alpha Vantage pre-scored news (LLM-skip hybrid; one batched call, hourly
        # cached) and StockTwits crowd sentiment (LLM-scored, like Reddit). Both
        # key/token-gated and OFF by default — see settings.
        f_av_news = (pool.submit(_safe, "av_news", fetch_alpha_vantage_news, all_tickers)
                     if (settings.enable_alpha_vantage_news and settings.alpha_vantage_key) else None)
        f_stocktwits = (pool.submit(_safe, "stocktwits", fetch_stocktwits_sentiment, tickers)
                        if (settings.enable_stocktwits and settings.stocktwits_access_token) else None)

        # Quiver Quantitative alt-data (key-gated). Congress → smart_money (revives
        # the congressional feed); gov-contracts / lobbying / dark-pool → synthetic
        # NewsArticles scored by the sentiment pipeline.
        _quiver_on = bool(settings.quiver_api_key)
        f_quiver_congress = (pool.submit(_safe, "quiver_congress", fetch_congress_trades)
                             if (_quiver_on and settings.enable_quiver_congress) else None)
        f_quiver_contracts = (pool.submit(_safe, "quiver_contracts", fetch_gov_contracts, all_tickers)
                              if (_quiver_on and settings.enable_quiver_gov_contracts) else None)
        f_quiver_lobbying = (pool.submit(_safe, "quiver_lobbying", fetch_lobbying, all_tickers)
                             if (_quiver_on and settings.enable_quiver_lobbying) else None)
        f_quiver_offexchange = (pool.submit(_safe, "quiver_offexchange", fetch_offexchange, all_tickers)
                                if (_quiver_on and settings.enable_quiver_offexchange) else None)

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

        # Ticker events (renames/delistings) — protective, only the held + watchlist
        # names (one call each), not the whole universe.
        f_ticker_events = (pool.submit(_safe, "ticker_events", fetch_ticker_events,
                                       list(settings.stocks_list) + list(open_trade_tickers))
                           if settings.enable_ticker_events else None)

        f_eps          = (pool.submit(_safe, "eps", fetch_earnings_surprises, tickers,
                                      lookback_days=settings.earnings_lookback_days)
                          if settings.enable_earnings else None)

        f_earnings_cal = (pool.submit(_safe, "earnings_cal", fetch_earnings_context, tickers,
                                      upcoming_days=settings.earnings_upcoming_days,
                                      alpha_vantage_key=settings.alpha_vantage_key)
                          if settings.enable_earnings else None)

        f_pead         = (pool.submit(_safe, "pead", fetch_pead_context, tickers)
                          if settings.enable_pead else None)

        f_fundamentals = (pool.submit(_safe, "fundamentals", fetch_fundamentals_context, tickers)
                          if settings.enable_fundamentals else None)

        f_corp_actions = (pool.submit(_safe, "corp_actions", fetch_corporate_actions_context, tickers)
                          if settings.enable_corporate_actions else None)

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
        "8k":           edgar.get("8k"),
        "trends":       get(f_trends),
        "reddit":       get(f_reddit),
        "analyst":      get(f_analyst),
        "ticker_events": get(f_ticker_events),
        "eps":          get(f_eps),
        "short":        get(f_short),
        "polygon_news": get(f_polygon_news),
        "finnhub_news": get(f_finnhub_news),
        "google_news":  get(f_google_news),
        "av_news":      get(f_av_news),
        "stocktwits":   get(f_stocktwits),
        "quiver_contracts": get(f_quiver_contracts),
        "quiver_lobbying":  get(f_quiver_lobbying),
        "quiver_darkpool":  get(f_quiver_offexchange),
    }
    for label, chunk in _article_chunks.items():
        if chunk:
            articles = articles + chunk
            logger.info(f"  [{label}] +{len(chunk)} article(s)")

    # Dedup across ALL merged sources (Google News especially overlaps the direct
    # feeds) — first occurrence wins, preserving source ordering.
    articles = _dedupe_by_url(articles)
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
    # Quiver congressional trades — revives the politician smart-money signal that
    # died when the Stock Watcher S3 went 403 (same InsiderTrade shape).
    quiver_congress = get(f_quiver_congress)
    if quiver_congress:
        smart_money.extend(quiver_congress)

    any_smart_money_enabled = (
        settings.enable_insider_trades or
        settings.enable_options_flow or
        settings.enable_sec_filings or
        bool(settings.quiver_api_key and settings.enable_quiver_congress)
    )
    insider_trades = smart_money if (smart_money or any_smart_money_enabled) else None

    # Surface tickers discovered via smart money signals — GATED like every other
    # discovery source (macro/peer below). 13D/G, 13F, Form 144 and options flow
    # can surface untradeable OTC microcaps with no OHLCV (e.g. FGRS / "Figure Tech
    # Blockchain"); without the gate they leaked straight into technical analysis
    # ("empty data" / "not enough history" warnings) and the bid-ask cost model.
    # Fail-closed: a name with no verifiable liquidity is dropped (re-appears once
    # its OHLCV cache is warm).
    if smart_money:
        smart_tickers = get_tickers_from_smart_money(smart_money)
        new_from_smart = [t for t in smart_tickers if t not in all_tickers]
        new_from_smart = apply_liquidity_gate(new_from_smart, source="smart_money", budget=gate_budget)
        if new_from_smart:
            logger.info(f"Adding {new_from_smart} to universe from smart money signals")
            all_tickers = all_tickers + new_from_smart
            _mark_source(new_from_smart, "smart_money")

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
    fundamentals_context = get(f_fundamentals)
    corporate_actions_context = get(f_corp_actions)

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
                _mark_source(new_from_macro, "macro_discovery")
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
                _mark_source(_peers, "coint_peer")
                logger.info(f"[coint_peer] Injecting {len(_peers)} cointegration peer leg(s): {_peers}")

    # ── Snapshot top-up for post-fetch discoveries ─────────────────────────
    # The bulk snapshot fetch (Step 2) ran on the universe as of Step 1, but the
    # expansions above (smart money, macro discovery, cointegration peers) can
    # more than double it — those names were scored with NO snapshot price, so
    # 2/3 of the signals panel had a NULL price anchor (observed 2026-07-01:
    # 106 snapshotted vs 313 scored). Fetch the missing ones now so every scored
    # ticker carries the same price context (ext_gap, prompt summaries,
    # signals.price, provenance) as the original set, and merge the result back
    # into the hourly snapshot cache so the next intra-hour tick reuses it.
    if settings.enable_fetch_data:
        _snapped = {s.ticker for s in snapshots}
        _missing_snap = [t for t in all_tickers if t not in _snapped]
        if _missing_snap:
            try:
                _extra_snaps = get_snapshots(_missing_snap)
            except Exception as e:
                logger.warning(f"[snapshots] top-up fetch failed ({e}) — "
                               f"{len(_missing_snap)} discovered ticker(s) stay price-less")
                _extra_snaps = []
            if _extra_snaps:
                snapshots = list(snapshots) + list(_extra_snaps)
                logger.info(
                    f"[snapshots] top-up: +{len(_extra_snaps)}/{len(_missing_snap)} "
                    f"discovered ticker(s) snapshotted (universe total {len(snapshots)})"
                )
                try:
                    save_snapshots(snapshots)
                except Exception as e:
                    logger.debug(f"[snapshots] top-up cache save failed: {e}")

    # ── Step 4: Build signals ─────────────────────────────────────────────
    logger.info("Step 4: Building signals...")
    # Context kwargs shared by the main signal build AND the every-tick
    # opener-pinned hold-review (fix #2) — the review reuses them verbatim so it
    # runs the identical algorithm, only with fresh news/prices + a pinned
    # sentiment engine. (tickers / articles / snapshots are supplied fresh there.)
    # Massive fundamental factor scores (value/quality/growth/short-squeeze) per
    # ticker — computed BEFORE build_signals so they can be folded into combined_score
    # as an additive overlay (fundamental_factor_weight), then reused for the panel.
    from src.data.fundamentals import factor_scores as _ffactor
    _fund_by_tk = ({fs.ticker: fs for fs in fundamentals_context.signals}
                   if fundamentals_context and getattr(fundamentals_context, "signals", None) else {})
    _fund_factors = {tk: _ffactor(fs) for tk, fs in _fund_by_tk.items()}

    # Broker advisor (IBKR short-borrow) — fetched from the live broker before
    # build_signals so the broker_advisor method can score it. Gated + fail-soft
    # (returns {} when off / no gateway), capped per tick (each ticker costs a
    # market-data request). None ⇒ method inactive. The reconciler reuses this same
    # broker connection later in the tick.
    borrow_context = None
    if settings.enable_broker_advisor and settings.broker_mode and settings.broker_mode != "off":
        from src.signals.broker_advisor import fetch_borrow_context
        borrow_context = fetch_borrow_context(
            list(all_tickers)[:settings.broker_advisor_max_tickers]) or None

    build_kwargs = dict(
        insider_trades=insider_trades,
        put_call_context=put_call_context,
        gex_context=gex_context,
        market_mode_context=market_mode_context,
        opex_context=opex_context,
        pead_context=pead_context,
        coint_context=coint_context,
        corp_factors=(corporate_actions_context.factor_scores if corporate_actions_context else None),
        fundamental_factors=_fund_factors,
        borrow_context=borrow_context,
    )

    # ── Early ledger refresh + overlapped hold-review branch (2026-07-08) ────
    # These used to run AFTER Step 5, back-to-back with the ~2-min hold-review
    # behind them — ~2.5 min of serial tick→order latency. Everything they need
    # exists HERE, so: refresh the ledger marks on the MAIN thread (the
    # IBKR-first price path keeps its thread/event-loop affinity, and Step 4.6's
    # open-position summaries then read FRESH marks instead of last tick's),
    # then start the pinned hold-review branch whose signal build overlaps
    # Step 4 and whose pinned synthesis overlaps Step 5. Trade-off accepted:
    # position marks are stamped at Step-4 start (~5 min earlier than before) —
    # well inside the 30-min tick resolution. Observation ticks skip all of it.
    hold_review_branch = None
    _open_trades_now: List = []
    if not observe_only:
        # Calibrate the sim cost to real IBKR fills BEFORE any cost-dependent
        # step this tick (normalize / M2M / horizon hurdle / new entries / NAV),
        # so every figure the run produces and persists shares one cost basis.
        calibrate_sim_costs()
        update_open_trades()   # also force-refreshes OHLCV for open tickers (fresh prices)
        _open_trades_now = get_open_trades()
        if (_open_trades_now and settings.enable_pinned_hold_review
                and settings.enable_hold_review_overlap):
            hold_review_branch = _HoldReviewBranch(
                _open_trades_now, sectors, build_kwargs, run_session)

    signals = build_signals(
        all_tickers,
        articles,
        snapshots=snapshots,
        session=run_session,
        **build_kwargs,
    )
    signals_by_ticker = {s.ticker: s for s in signals}

    # Attach diagnostic FACTOR scores to each signal for the dashboard's Signal-IC
    # table (IC + Sim win% + Sim ret%): Massive fundamentals (value/quality/growth/
    # short-squeeze, panel-only) + corporate-action directional factors (f_split/
    # f_dividend, which ALSO nudge combined_score via the aggregator overlay above).
    _corp_fs = getattr(corporate_actions_context, "factor_scores", None) or {}
    if _fund_factors or _corp_fs:
        for _tk, _sig in signals_by_ticker.items():
            merged: dict = {}
            _ff = _fund_factors.get(_tk) or _fund_factors.get(_tk.upper())
            if _ff:
                merged.update(_ff)
            _cf = _corp_fs.get(_tk) or _corp_fs.get(_tk.upper())
            if _cf:
                merged.update(_cf)
            if merged:
                _sig.fundamental_scores = merged

    # ── Step 4.4: Horizon synthesis (term-structure of edge) ─────────────
    # Weight every method's LIVE score by its MEASURED per-horizon IC (sign-aware,
    # from the simulated_trades panel), pick the cost-aware holding horizon, and
    # stamp it on each signal — so the LLM reasons WITH it (confirm or shorten,
    # never lengthen) and the matched exit can close a position once its edge
    # window passes. Pure IC weighting (no static blend) by design.
    if settings.enable_horizon_synthesis:
        try:
            from src.signals import edge_curve
            ic_matrix = edge_curve.get_ic_matrix()
            dmatrix = (edge_curve.get_directional_ic_matrix()
                       if settings.enable_directional_shadow else {})
            if ic_matrix or dmatrix:
                n_tradeable = n_shadow_flip = 0
                # The regime layer owns the market direction (alpha/beta split);
                # selection then amplifies the biggest expected move in that direction.
                _mkt_dir = edge_curve.market_direction_from_regime(
                    getattr(macro_regime_context, "regime", ""))
                for _tk, _sig in signals_by_ticker.items():
                    _base = _method_scores_from_signal(_tk, _sig.direction, signals_by_ticker)
                    _scores = {**_base,
                               **(getattr(_sig, "timeframe_scores", None) or {}),
                               **(getattr(_sig, "fundamental_scores", None) or {}),
                               "combined_score": float(getattr(_sig, "combined_score", 0.0))}
                    # Live (pooled, gross) curve — drives target_horizon + matched exit.
                    if ic_matrix:
                        _curve = edge_curve.compute_edge_curve(_scores, ic_matrix)
                        _sel = edge_curve.select_horizon(_curve)
                        _sig.target_horizon = _sel["target_horizon"]
                        _sig.horizon_label = _sel["horizon_label"]
                        _sig.horizon_conviction = _sel["conviction"]
                        _sig.horizon_net_edge_pct = _sel["net_edge_pct"]
                        _sig.horizon_tradeable = _sel["tradeable"]
                        _sig.horizon_curve = {h: c["net"] for h, c in _curve.items()}
                        n_tradeable += int(_sel["tradeable"])
                        # Expected favourable move (magnitude) + market-aligned upside
                        # rank key — selection should prefer the biggest expected
                        # mover in the regime's direction.
                        _sig.expected_move_pct = _sel.get("expected_move_pct", 0.0)
                        if settings.enable_expected_move_ranking:
                            _align = edge_curve.market_alignment(_sig.direction, _mkt_dir)
                            _sig.market_aligned = _align
                            _sig.upside_score = edge_curve.upside_score(
                                _sel["conviction"], _sel.get("expected_move_pct", 0.0), _align)
                    # Shadow (direction-aware, market-neutral) curve — persisted +
                    # displayed only; does NOT touch entries/exits.
                    if dmatrix:
                        _dcurve = edge_curve.compute_directional_edge_curve(_scores, dmatrix)
                        _dsel = edge_curve.select_horizon(_dcurve)
                        _sig.shadow_target_horizon = _dsel["target_horizon"]
                        _sig.shadow_horizon_label = _dsel["horizon_label"]
                        _sig.shadow_direction = _dsel["direction"]
                        _sig.shadow_horizon_net_edge_pct = _dsel["net_edge_pct"]
                        _sig.shadow_horizon_tradeable = _dsel["tradeable"]
                        _sig.shadow_horizon_curve = {h: c["net"] for h, c in _dcurve.items()}
                        _live_dir = str(getattr(_sig.direction, "value", _sig.direction)).upper()
                        if _dsel["direction"] in ("BULLISH", "BEARISH") and _live_dir != _dsel["direction"]:
                            n_shadow_flip += 1
                logger.info(
                    f"[horizon] edge curve stamped on {len(signals_by_ticker)} ticker(s); "
                    f"{n_tradeable} cost-clearing"
                    + (f"; shadow (dir/mkt-neutral) {len(dmatrix)} method(s), "
                       f"{n_shadow_flip} direction disagreement(s)" if dmatrix else "")
                    + f" (IC matrix: {len(ic_matrix)} method(s))")
            else:
                logger.info("[horizon] IC matrices empty — horizon synthesis idle (panel still thin)")
        except Exception as e:
            logger.warning(f"[horizon] synthesis skipped ({type(e).__name__}: {e})")

    # Update cluster watchlist: add newly detected clusters, expire stale entries
    _cluster_raw = update_cluster_watchlist(signals_by_ticker, _cluster_raw)
    save_cluster_watchlist(_cluster_raw)
    cluster_watchlist_context = build_cluster_watchlist_context(_cluster_raw)
    if cluster_watchlist_context.entries:
        logger.info(f"[cluster_watch] {cluster_watchlist_context.summary}")

    # Sector pairs: find ETF vs constituent divergences (market-neutral pair trades)
    sector_pairs_context = find_sector_pairs(signals_by_ticker)

    # ── Step 4.5: Catalyst timing (computed BEFORE synthesis) ────────────
    # Earnings blackout, OpEx amplifier state, and 8-K+insider setups are
    # known facts at this point — the LLM should reason WITH them (don't
    # spend a BUY on a blackout ticker; weigh a catalyst setup) instead of
    # being silently vetoed afterwards. The hard enforcement stays below:
    # WATCH elevation on the ranked top-10 and the earnings-blackout gate on
    # the actionable set are mechanical guarantees, not LLM suggestions.
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
            pead_context=pead_context,
        )

    # ── Step 4.6: held-positions prompt A/B (coin flip per run) ──────────
    # 50/50 experiment: half the runs tell the LLM which tickers the system
    # currently holds (plus a zero-endowment-bias review instruction), half
    # leave it blind (the pre-experiment behavior). The flip is stamped on
    # every trade CLOSED this run (exit_hold_prompt) so the dashboard's
    # method-evaluation table accumulates an exit-outcome comparison.
    open_position_summaries = get_open_position_summaries()
    hold_prompt_active = bool(open_position_summaries) and (
        random.random() < float(settings.open_positions_prompt_share)
    )
    if open_position_summaries:
        logger.info(
            f"[hold_prompt] {'ON' if hold_prompt_active else 'OFF'} this run "
            f"(share={settings.open_positions_prompt_share:g}, "
            f"{len(open_position_summaries)} open position(s))"
        )

    # ── Step 5: Generate recommendations ─────────────────────────────────
    logger.info("Step 5: Generating recommendations...")
    # Context kwargs shared by the main synthesis AND the every-tick opener-pinned
    # hold-review (fix #2) — the review reuses them with force_engine + its own
    # (fresh-data) signals, and WITHOUT open_positions (fresh-candidate framing,
    # apples-to-apples with how the position was judged at entry).
    synth_kwargs = dict(
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
        catalyst_timing_context=catalyst_timing_context,
        fundamentals_context=fundamentals_context,
        corporate_actions_context=corporate_actions_context,
    )
    if hold_review_branch is not None:
        # Unblock the branch's pinned synthesis calls: the context kwargs are
        # complete (incl. the Step-4.5 catalyst timing block), identical to what
        # the main synthesis below receives — so the reviews run concurrently
        # with Step 5 on the same context the run's entries are judged on.
        # NOTE blind_synthesis is deliberately NOT in synth_kwargs: reviews are
        # never blinded (stable exit governance — only ENTRY judgment is A/B'd).
        hold_review_branch.supply_synth_kwargs(synth_kwargs)
    # Blind-synthesis A/B (2026-07-12): a per-run coin flip hides the aggregator's
    # verdict from the MAIN synthesis so the LLM's independent judgment becomes
    # measurable (the agreement eval found 96% of sighted calls just echo the
    # aggregator). Stamped into gate_diag + entry_blind_synthesis on new trades →
    # the dashboard's "Entry eval · blind-synthesis ON/OFF" rows.
    blind_synthesis = random.random() < float(settings.blind_synthesis_share)
    logger.info(f"[blind_synth] {'BLIND' if blind_synthesis else 'SIGHTED'} this run "
                f"(share={settings.blind_synthesis_share:g})")
    recommendations = generate_recommendations(
        signals,
        open_positions=open_position_summaries if hold_prompt_active else None,
        session=run_session,
        blind_synthesis=blind_synthesis,
        **synth_kwargs,
    )

    # ── Fix #2: capture this run's engines for the opener-pinned hold-review ──
    # Captured here BEFORE the top-10 truncation: this run's synthesis +
    # sentiment engines (for the legacy-position group + the cheap fallback
    # path) and the full pre-truncation recs (every held ticker is pinned into
    # the universe at Step 0, so each has a rec). The globals read here are
    # safe against the concurrently-running review branch: forced-engine calls
    # no longer touch the last-synthesis meta or the sentiment tallies, so both
    # reflect the MAIN pass only.
    _synth_meta = get_last_synthesis_meta()
    run_synthesis_provider = _synth_meta.get("provider")
    _sent_model = get_dominant_sentiment_model()
    _sent_summary = get_sentiment_provider_summary()
    run_sentiment_provider = _provider_of_synth_model(_sent_model)
    _full_recs = list(recommendations)
    if hold_review_branch is not None:
        # Unblock the branch's legacy-position group (positions with no opener
        # stamps are reviewed with THIS run's engines, now known).
        hold_review_branch.supply_run_engines(run_sentiment_provider, run_synthesis_provider)

    # Keep only the top 10 recommendations by conviction:
    # BUY/SELL first (sorted by confidence desc), then HOLD/WATCH to fill up to 10.
    _ACTION_RANK = {"BUY": 0, "SELL": 0, "HOLD": 1, "WATCH": 2}
    recommendations = sorted(
        recommendations,
        key=lambda r: (_ACTION_RANK.get(r.action, 3), -r.confidence),
    )[:10]

    # Macro regime gate — adjust threshold and optionally block BUY entries.
    # Baseline 0.85 (2026-07-21 user directive): the fallback when the regime
    # filter is off / has no context; the filter otherwise REPLACES it with the
    # regime-specific threshold (NEUTRAL is also 0.85 — see macro_regime._REGIME_THRESHOLD).
    _confidence_threshold = 0.85
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
            f"(default 85%) | allow_buys={_allow_buys}"
        )

    # Engine-relative translation: the regime threshold is an ABSOLUTE
    # confidence anchor, but confidence scales are engine-specific — translate
    # it into THIS run's engine scale by matching gate selectivity over recent
    # recommendations (shrunk + clamped; fail-soft to the static value).
    threshold_meta = None
    try:
        from src.analysis.threshold_calibration import engine_relative_threshold
        _confidence_threshold, threshold_meta = engine_relative_threshold(
            _confidence_threshold, _synth_meta.get("model"))
        if threshold_meta.get("applied"):
            logger.info(
                f"[threshold] engine-relative gate: static {threshold_meta['static']:.0%} "
                f"→ {_confidence_threshold:.0%} for {threshold_meta['engine']} "
                f"(selectivity {threshold_meta.get('selectivity', 0):.0%}, "
                f"n={threshold_meta['n_engine']})"
            )
    except Exception as e:
        logger.debug(f"[threshold] engine-relative translation skipped: {e}")

    # Off-RTH gate — thin books and wide spreads demand more conviction, so the
    # regime threshold is bumped further for any run outside RTH. Overnight
    # (the thinnest book of the day) carries its own, larger bump. Observation
    # ticks only shape the persisted `actionable` flag; trade mode inherits it.
    _session_bump = (settings.overnight_confidence_bump if run_session == "overnight"
                     else settings.extended_confidence_bump)
    if run_session != "rth" and _session_bump > 0:
        _confidence_threshold = min(0.95, _confidence_threshold + _session_bump)
        logger.info(
            f"[extended] {run_session}-session threshold bump: "
            f"+{_session_bump:.0%} → {_confidence_threshold:.0%}"
        )

    # Catalyst timing enforcement — the context was computed BEFORE synthesis
    # (Step 4.5) and the LLM already saw it; here the mechanical guarantees
    # run regardless of what the LLM did with the information: WATCH
    # elevation on the ranked top-10, earnings blackout on actionable below.
    if catalyst_timing_context is not None:
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
        "dropped_low_combined_score":   0,
        "dropped_low_agreement":        0,
        "dropped_buy_blocked":          0,
        "dropped_earnings_blackout":    0,
        "dropped_untradeable":          0,
        "dropped_overextended":         0,
        "actionable_survivors":         0,
        "confidence_threshold":         round(_confidence_threshold, 2),
        "threshold_calibration":        threshold_meta,   # engine-relative gate audit
        "allow_buys":                   _allow_buys,
        "session":                      run_session,
        # This run's discovery-funnel shape: how many tickers each source
        # contributed (first-source-wins). Trades/signals carry the per-ticker
        # stamp; this is the run-level summary.
        "universe_sources":             {s: sum(1 for v in universe_source.values() if v == s)
                                         for s in sorted(set(universe_source.values()))},
        "regime":                       (macro_regime_context.regime
                                          if macro_regime_context else "NEUTRAL"),
        # Regime/mode INPUT COVERAGE — persisted per run so the dashboard can show
        # whether a verdict came from full or degraded feeds (a 1/10-input regime
        # reads identically to an 8/10 one without this). Mirrors price_provenance.
        "regime_coverage":              {"available": getattr(macro_regime_context, "inputs_available", 0),
                                          "total":     getattr(macro_regime_context, "inputs_total", 0)},
        "mode_coverage":                {"available": getattr(market_mode_context, "inputs_available", 0),
                                          "total":     getattr(market_mode_context, "inputs_total", 0)},
        "hold_prompt_active":           hold_prompt_active,
        "hold_prompt_n_positions":      len(open_position_summaries),
        # Blind-synthesis A/B arm this run (entry-side prompt experiment).
        "blind_synthesis":              blind_synthesis,
    }

    # Fresh cold-fetch allowance for the trade gate (actionable tickers are almost
    # always already cached from scoring, so this rarely fetches).
    _trade_gate_budget = {"n": max(0, int(settings.discovery_gate_max_fetch))}
    # Per-ticker gate outcome, persisted in gate_diag so the dashboard's
    # decision-funnel evaluation (tracker.compute_stage_eval) attributes each
    # drop to its exact gate — the run-level counters alone can't split gate 3
    # vs 4 when both fire in one run. Recs are deduped per ticker upstream, so
    # one outcome per ticker per run.
    _gate_outcomes: dict = {}
    actionable: List = []
    for r in recommendations:
        if r.action not in ("BUY", "SELL"):
            continue
        gate_diag["buy_sell_candidates"] += 1
        # Gate 1 — regime-tightened confidence threshold
        if r.confidence < _confidence_threshold:
            gate_diag["dropped_below_threshold"] += 1
            _gate_outcomes[r.ticker] = "below_threshold"
            continue
        # Gate 1b — agreement floor: mechanically enforces the CLAUDE.md-documented
        # "a single strong signal source never produces a BUY/SELL" invariant
        # (previously only a prompt instruction — see _passes_agreement_gate).
        if not _passes_agreement_gate(r.direction, signals_by_ticker.get(r.ticker)
                                      if signals_by_ticker else None):
            gate_diag["dropped_low_agreement"] += 1
            _gate_outcomes[r.ticker] = "low_agreement"
            continue
        # Gate 2 — BUY block (PANIC / RISK_OFF)
        if r.action == "BUY" and not _allow_buys:
            gate_diag["dropped_buy_blocked"] += 1
            _gate_outcomes[r.ticker] = "buy_blocked"
            continue
        # Gate 3 — earnings blackout window
        if r.ticker in earnings_blackout:
            gate_diag["dropped_earnings_blackout"] += 1
            _gate_outcomes[r.ticker] = "earnings_blackout"
            continue
        # Gate 4 — tradeable liquidity floor: penny / thin names (< trade_min_price
        # or < trade_min_dollar_volume 20d ADV) are OBSERVE-ONLY — still scored +
        # persisted to the signals panel (penny-stock performance keeps accruing)
        # but never actionable (no sim trade / broker order). Fail-closed via
        # is_liquid; discovery admits them at the LOWER observation floor.
        if not _is_tradeable(r.ticker, _trade_gate_budget):
            gate_diag["dropped_untradeable"] += 1
            _gate_outcomes[r.ticker] = "untradeable"
            continue
        # Gate 5 — overextension (anti-chase, BUY-only): a BUY whose ticker
        # already ran > overextension_runup_pct over the trailing 5 completed
        # bars is deferred — the 2026-07-22 forensics measured that cohort at a
        # 32.5% 5d hit rate (median −4.1% vs SPY): the combined score peaks
        # right after the run-up and short-term reversal eats the entry. The
        # name re-qualifies at any later tick it has cooled below the bar.
        # SELLs pass untouched (fading spikes / riding crashes is measured edge).
        if r.action == "BUY" and _is_overextended(r.ticker):
            gate_diag["dropped_overextended"] += 1
            _gate_outcomes[r.ticker] = "overextended"
            continue
        actionable.append(r)
        gate_diag["actionable_survivors"] += 1
        _gate_outcomes[r.ticker] = "pass"
    gate_diag["gate_outcomes"] = _gate_outcomes

    if gate_diag["dropped_overextended"]:
        logger.info(
            f"[overextension] Gate 5 deferred {gate_diag['dropped_overextended']} "
            f"BUY(s) that ran > {settings.overextension_runup_pct:.0f}% in the last "
            f"{settings.overextension_lookback_bars} sessions (anti-chase)"
        )

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

    # Performance tracking. On observation ticks (extended sessions) the whole
    # mutation block is skipped: the evidence is what _persist_run writes
    # (runs + recommendations + signals); the ledger, hypothetical book, and
    # broker are never touched, and no email payload is assembled.
    gate_diag["observe_only"] = observe_only
    broker_report = None
    if observe_only:
        logger.info(
            "[pipeline] OBSERVATION tick (extended session) — skipping ledger "
            "updates, position monitoring, trade entries, broker sync, "
            "hypothetical marks, and email assembly"
        )
        trade_diag = {}
        perf, hypothetical_perf = {}, {}
    else:
        # Sim-cost calibration + ledger mark refresh ran BEFORE Step 4 (the
        # overlap block above), and the opener-pinned hold-review (fix #2) has
        # been running concurrently with Steps 4–5 since then — join it here,
        # before the monitor consumes it. When the overlap is off (or pinned
        # reviews are disabled → the cheap reuse path), build them inline
        # exactly as before.
        if hold_review_branch is not None:
            hold_reviews = hold_review_branch.result()
        else:
            hold_reviews = _build_hold_reviews(
                _open_trades_now, run_sentiment_provider, run_synthesis_provider,
                _full_recs, sectors, build_kwargs, synth_kwargs, run_session,
            )
        # Persist the trajectory (confidence/action vs entry + price) BEFORE the
        # monitor may close on it, so the review that triggered a close is recorded.
        _persist_trade_reviews(run_id, hold_reviews, _open_trades_now)
        # Decompose the exit decision into per-method hold-conviction scores and
        # persist them (exit_signals panel) for the Exit Performance IC table.
        _persist_exit_signals(run_id, hold_reviews, _open_trades_now,
                              signals_by_ticker, macro_regime_context)
        # Open-position monitor: opener-pinned LLM exits (+ macro-regime, + the
        # aggregator backstop for legacy trades) BEFORE the counter-recommendation
        # exit path.
        monitor_open_positions(
            signals_by_ticker=signals_by_ticker,
            macro_regime_context=macro_regime_context,
            hold_prompt_active=hold_prompt_active,
            hold_reviews=hold_reviews,
            run_synthesis_provider=run_synthesis_provider,
        )
        close_trades_on_signal_reversal(               # legacy/rule-based only (LLM-opened owned by monitor)
            actionable, hold_prompt_active=hold_prompt_active)
        # Stamp each new trade with the exact LLM engines in use this run (final-call
        # synthesis model + run-dominant sentiment scorer) for per-LLM attribution.
        # (_synth_meta was captured right after synthesis, above.)
        _synth_model = _synth_meta.get("model") or (
            "rule-based (no LLM)" if _synth_meta.get("provider") == "rule-based" else None
        )
        trade_diag = record_new_trades(
            actionable, signals_by_ticker=signals_by_ticker, run_id=run_id,
            llm_synthesis_model=_synth_model,
            llm_sentiment_model=_sent_model,   # snapshotted before the hold-reviews
            universe_sources=universe_source,
            blind_synthesis=blind_synthesis,   # A/B arm stamp (entry_blind_synthesis)
        ) or {}

        # Broker shadow execution (paper-first): once the internal ledger is final for
        # this tick, reconcile a real broker (IBKR paper) against it — submit entries
        # for new opens, close exits, report slippage/drift. No-op when broker_mode=off;
        # the reconciler is exception-safe and never breaks the run.
        if settings.broker_mode and settings.broker_mode != "off":
            from src.broker.reconcile import sync as _broker_sync
            # Wall-clock watchdog: if the reconcile hangs past the cap (a stuck
            # broker call RequestTimeout somehow didn't catch), force-exit so the
            # task manager restarts a fresh process — a hung tick blocks the whole
            # poll loop, and nothing else recovers a frozen (non-exited) process.
            _wd = None
            _wd_cap = float(getattr(settings, "broker_sync_watchdog_seconds", 0) or 0)
            if _wd_cap > 0:
                import os
                import threading

                def _broker_watchdog_kill():
                    try:
                        logger.critical(
                            f"[broker] reconcile exceeded {_wd_cap:.0f}s wall-clock — a broker "
                            "call is stuck; force-exiting so the scheduler restarts (this was the "
                            "2026-07-06 6-hour freeze). Internal sim/ledger already persisted.")
                    except Exception:
                        pass
                    os._exit(1)
                _wd = threading.Timer(_wd_cap, _broker_watchdog_kill)
                _wd.daemon = True
                _wd.start()
            try:
                # The tick's actionable set (ticker → BUY/SELL) drives the
                # price-aware next-tick resubmit for previously-unfilled entries
                # (reconcile._resubmit_decision): still-wanted → chase; decayed →
                # only resubmit at an equal-or-better price than the decision.
                _actionable_by_ticker = {
                    r.ticker: r.action for r in (actionable or [])
                    if getattr(r, "ticker", None) and getattr(r, "action", None)
                }
                broker_report = _broker_sync(
                    run_id=run_id, actionable_by_ticker=_actionable_by_ticker)
            except Exception as e:
                logger.warning(f"[broker] sync raised unexpectedly (internal sim unaffected): {e}")
            finally:
                if _wd is not None:
                    _wd.cancel()

    # Merge per-gate counters from the actionable filter + the trade-entry path
    # into one diagnostic blob so the user can see at a glance which constraints
    # are doing the work and which are passive backstops.
    gate_diag.update({
        "trade_considered":             int(trade_diag.get("considered", 0)),
        "trade_skipped_already_open":   int(trade_diag.get("skipped_already_open", 0)),
        "trade_skipped_reentry_cooldown": int(trade_diag.get("skipped_reentry_cooldown", 0)),
        "trade_skipped_correlation_cap": int(trade_diag.get("skipped_correlation_cap", 0)),
        "trade_skipped_no_price":       int(trade_diag.get("skipped_no_price", 0)),
        "trade_haircut_applied":        int(trade_diag.get("haircut_applied", 0)),
        "trade_breadth_tier_applied":   int(trade_diag.get("breadth_tier_applied", 0)),
        "trade_edge_blend_applied":     int(trade_diag.get("edge_blend_applied", 0)),
        "trade_opened":                 int(trade_diag.get("opened", 0)),
    })
    logger.info(
        "[gates] "
        f"regime={gate_diag['regime']} threshold={gate_diag['confidence_threshold']:.0%} "
        f"allow_buys={gate_diag['allow_buys']} | "
        f"actionable filter: {gate_diag['buy_sell_candidates']} BUY/SELL → "
        f"survived={gate_diag['actionable_survivors']} "
        f"(rejected by threshold={gate_diag['dropped_below_threshold']}, "
        f"low-agreement={gate_diag['dropped_low_agreement']}, "
        f"BUY-block={gate_diag['dropped_buy_blocked']}, "
        f"earnings-blackout={gate_diag['dropped_earnings_blackout']}, "
        f"untradeable={gate_diag['dropped_untradeable']}, "
        f"overextended={gate_diag['dropped_overextended']}) | "
        f"trade entry: {gate_diag['trade_considered']} considered → "
        f"opened={gate_diag['trade_opened']} "
        f"(already_open={gate_diag['trade_skipped_already_open']}, "
        f"reentry_cooldown={gate_diag['trade_skipped_reentry_cooldown']}, "
        f"corr_cap={gate_diag['trade_skipped_correlation_cap']}, "
        f"no_price={gate_diag['trade_skipped_no_price']}, "
        f"corr_haircut_applied={gate_diag['trade_haircut_applied']}, "
        f"breadth_tier={gate_diag['trade_breadth_tier_applied']}, "
        f"edge_blend={gate_diag['trade_edge_blend_applied']})"
    )
    if not observe_only:
        log_performance_summary()
        perf = get_performance_for_email()

        # Hypothetical always-open book — fully isolated from the real-trade
        # ledger above. Update marks then snapshot for the email section.
        update_hypothetical_trades()
        hypothetical_perf = get_hypothetical_performance_for_email()

    # The per-run static HTML report has been retired in favour of the live
    # Plotly Dash dashboard (python main.py --dashboard), backed by DuckDB.
    logger.info("Monitor performance + rationale in the dashboard: python main.py --dashboard")

    # Price-provenance guard: compare every new trade's recorded entry price to
    # the run's snapshot (the standing version of the one-off stale-price audit).
    # Stash in gate_diag so it persists for the dashboard, and reuse the verdict
    # for the email banner + subject tag below.
    price_health = _assess_price_provenance(run_id, snapshots)
    if price_health:
        gate_diag["price_provenance"] = price_health
        if price_health["down"]:
            logger.critical(f"[price] PRICE PROVENANCE — {price_health['message']}.")

    # Calibration snapshot: the exact self-calibrated parameter values this run
    # traded with (real-fill cost, session spread multipliers, cost hurdle,
    # breadth ramp, …) — stashed in gate_diag (the no-schema-change channel,
    # like the provenance verdict) so every run's calibrations are durable and
    # the dashboard's Data Quality tab can show current-vs-prior-vs-evidence.
    try:
        from src.performance.calibration import get_calibrations
        cals = get_calibrations()
        if cals:
            gate_diag["calibrations"] = cals
    except Exception as e:
        logger.debug(f"[calibration] snapshot skipped: {e}")

    # Persist run + 'APIs used' + every recommendation (with attribution) +
    # the broker reconcile report (per-order slippage/commissions) + the full
    # per-ticker signal cross-section (the learning panel) to DuckDB.
    _persist_run(
        run_id, start, datetime.now(timezone.utc), all_tickers, recommendations, actionable,
        gate_diag, market_mode_context, macro_regime_context,
        _confidence_threshold, _allow_buys, signals_by_ticker,
        broker_report=broker_report, snapshots=snapshots,
        synthesis_meta=_synth_meta, sentiment_summary=_sent_summary,
        universe_sources=universe_source,
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
        # Append the exact per-order reasons so the log line is self-contained
        # (the concise message is only the count summary).
        _berrs = broker_health.get("errors") or []
        _bdetail = ("  ::  " + "  |  ".join(_berrs[:8])
                    + (f"  (+{len(_berrs) - 8} more)" if len(_berrs) > 8 else "")) if _berrs else ""
        logger.critical(
            f"[broker] EXECUTION ISSUE ({broker_health['mode']}) — {broker_health['message']}.{_bdetail}"
        )

    _print_summary(actionable, smart_money or [])

    email_configured = bool(settings.smtp_user and settings.email_recipients)
    # A detected PROBLEM this run (broker/execution issue, LLM-layer outage, or a
    # price-provenance flag) forces the report email even on a non-email slot, so an
    # issue surfacing between the scheduled slots reaches the user at the next tick
    # rather than hours later (settings.email_on_problem). Gated on email being
    # configured so the send branch below always has working SMTP.
    health_problem = bool(
        email_configured
        and getattr(settings, "email_on_problem", True)
        and (
            (price_health and price_health.get("down"))
            or (llm_health and llm_health.get("down"))
            or (broker_health and broker_health.get("down"))
        )
    )
    forced_by_problem = health_problem and not (send_email or email_if_configured)
    _decision = _email_decision(
        observe_only=observe_only, send_email=send_email,
        email_if_configured=email_if_configured,
        email_configured=email_configured, health_problem=health_problem,
    )
    if _decision == "observe":
        logger.info("[pipeline] Observation tick — email skipped by design.")
    elif _decision == "suppress":
        logger.info(
            "[pipeline] Email suppressed for this tick (per-slot scheduler "
            "decision: with scheduler_email_every_tick off, only the 16:00 "
            "closing slot emails)."
        )
    elif _decision == "send":
        if forced_by_problem:
            logger.warning(
                "[pipeline] Forcing OUT-OF-SCHEDULE email — a problem was detected "
                "this tick (broker / LLM / price health). See the banner + 🔔 "
                "subject tag for details."
            )
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
            fundamentals_context=fundamentals_context,
            corporate_actions_context=corporate_actions_context,
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
            price_health=price_health,
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
