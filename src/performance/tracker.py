"""Track algorithm performance by recording BUY/SELL entries and calculating returns.

Trades are stored in cache/trades.json. Each daily run:
  1. Opens new trades for today's BUY/SELL recommendations.
  2. Fetches current prices for all open trades and marks unrealised P&L.
  3. Closes positions whose thesis has deteriorated (signal decay / regime /
     reversal). There is no fixed time cap — a position is held as long as its
     rationale holds.
  4. Logs a full performance summary.
"""

import hashlib
import json
import yfinance as yf
from statistics import median
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from src.models import Recommendation
from src.utils import ET
from src.performance.spread import (_dynamic_half_spread, _one_side_cost, _pct_return,
                                     fmt_price, set_real_cost_override)
from src.db import repo
from config import settings

TRADES_FILE = Path("cache/trades.json")

# Live-price-fetch health for the current run — surfaced to the dashboard + email
# as a "Live price feed" data source so a broken feed is visible. yfinance is
# tried first, Polygon second; a ticker that fails both lands in ``failed``.
_PRICE_HEALTH: dict = {"ibkr": 0, "yfinance": 0, "polygon": 0, "failed": []}


def reset_price_health() -> None:
    """Clear the live-price health counters (call once at the start of a run)."""
    _PRICE_HEALTH["ibkr"] = 0
    _PRICE_HEALTH["yfinance"] = 0
    _PRICE_HEALTH["polygon"] = 0
    _PRICE_HEALTH["failed"] = []


def get_price_health() -> dict:
    """Snapshot of this run's live-price fetch outcomes."""
    return {
        "ibkr": _PRICE_HEALTH["ibkr"],
        "yfinance": _PRICE_HEALTH["yfinance"],
        "polygon": _PRICE_HEALTH["polygon"],
        "failed": list(_PRICE_HEALTH["failed"]),
    }

# ── Position sizing ───────────────────────────────────────────────────────────
# Confidence-scaled Kelly-inspired tiers: allocate more capital to higher conviction.
# The hard per-sector cap was removed in favour of the correlation-aware
# haircut (see src/performance/correlation.py), which handles cross-sector
# factor concentration that the GICS-bucket cap missed (e.g. NVDA + AVGO + SMH
# all loading semis even though SMH lives in the ETF bucket). The sector_key
# field is still computed on every new trade as a diagnostic — useful for
# slicing the performance breakdown by sector — but no longer gates entry.
SIZE_TIER_HALF  = 0.85   # below this → 1.0× (baseline)
SIZE_TIER_FULL  = 0.92   # 0.85–0.92 → 1.5×; above → 2.0×



# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

# A ledger row must carry these to be a trade at all. Anything else is
# corruption (e.g. the 2026-06-10 incident: a stale test wrote its fixture row
# into the production table, and every consumer that sorted on entry_date or
# formatted entry_price crashed the pipeline tick).
_REQUIRED_TRADE_FIELDS = ("ticker", "entry_date", "entry_price", "status")


def _sanitize_trades(trades: List[dict]) -> List[dict]:
    """Drop rows that aren't structurally trades (missing a core field).

    This is the single integrity boundary: every consumer downstream of
    ``_load_trades`` may rely on the core fields existing, instead of
    defending each sort/format site individually. Excluded rows are warned
    about loudly; because every save is a full-replace of the table, the next
    ledger write naturally purges them from the DB.
    """
    def _ok(t: dict) -> bool:
        return all(t.get(f) is not None for f in _REQUIRED_TRADE_FIELDS)

    valid = [t for t in trades if _ok(t)]
    if len(valid) != len(trades):
        bad = [t.get("ticker", "?") for t in trades if not _ok(t)]
        logger.warning(
            f"[tracker] {len(trades) - len(valid)} malformed ledger row(s) excluded "
            f"(missing one of {_REQUIRED_TRADE_FIELDS}): {bad} — purged from the DB "
            "on the next ledger write"
        )
    return valid


def _load_trades() -> List[dict]:
    """Load the trade ledger from DuckDB (the single source of truth).

    Rows are passed through ``_sanitize_trades`` so corrupted entries can
    never reach (and crash) downstream consumers.

    One-time safety net: if the DB holds no valid trades but the legacy
    ``cache/trades.json`` still exists, seed the DB from it so the JSON→DuckDB
    cutover never loses history (even if the pipeline runs before
    ``python -m src.db.migrate``).
    """
    # FAIL-CLOSED on DB errors: a transient lock/IO failure must ABORT the caller
    # (connection.connect already retried with backoff), not masquerade as an
    # empty ledger — returning [] here put every load→mutate→save cycle one step
    # from a full wipe (observed lock race 2026-07-01 14:49; wipe class
    # 2026-06-11). repo.save_trades' shrink guard is the second line of defense.
    trades = _sanitize_trades(repo.load_trades())
    if not trades and TRADES_FILE.exists():
        try:
            legacy = _sanitize_trades(json.loads(TRADES_FILE.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning(f"[tracker] legacy {TRADES_FILE} unreadable ({e}) — starting from the empty DB")
            return trades
        if legacy:
            repo.save_trades(legacy)
            logger.info(f"[tracker] Seeded DuckDB from {TRADES_FILE} ({len(legacy)} trades)")
            return legacy
    return trades


def _save_trades(trades: List[dict]) -> None:
    """Persist the full trade ledger to DuckDB (full-replace of the trades table)."""
    repo.save_trades(trades)


def _fetch_price(ticker: str) -> Optional[float]:
    """Fetch the latest live last-trade price for *ticker*, **session-aware**.

    yfinance ``fast_info.last_price`` reports the REGULAR session only — during
    pre/after-market it silently returns the stale RTH close (at the 04:00
    pre-market tick that is the *prior trading day's* close, stale by a whole
    weekend). It must therefore NOT be the primary source off-hours, or an
    extended-session entry/mark is booked at a stale price (observed 2026-06-15:
    CRDO entered at Friday's 250.81 close while the live pre-market print — and
    the analysis snapshot — was ~262). The source order mirrors the snapshot
    path in ``market_data.get_snapshots``:

    * **RTH** — yfinance ``fast_info.last_price`` first (fast, no API key), then
      Polygon ``lastTrade`` as the fallback (unchanged behaviour).
    * **Extended / overnight** — yfinance 1-min ``prepost`` bars first (the
      genuine extended print, available on the free data plan), then Polygon's
      single-ticker ``lastTrade`` (a secondary — it 403s without the snapshot
      entitlement that this plan lacks), and only then stale ``fast_info`` as a
      last resort so we still return *something* rather than ``None``.

    The success source and any total failure are recorded in ``_PRICE_HEALTH``
    so a broken feed surfaces in the dashboard and email.
    """
    if not settings.enable_fetch_data:
        return None

    def _ibkr() -> Optional[float]:
        # Preferred when enabled: IBKR's real-time feed — the SAME data that
        # fills the orders — over the existing broker connection (reused; no
        # reconnect per call). Requires an IBKR broker_mode; silently yields to
        # the other sources when off, disconnected, or no quote is returned.
        if not settings.enable_ibkr_price_feed:
            return None
        try:
            from src.broker import get_broker
            broker = get_broker()
            if broker is None:
                return None
            if not broker.is_connected():
                broker.connect()
            px = broker.get_market_price(ticker)
            if px and float(px) > 0:
                _PRICE_HEALTH["ibkr"] = _PRICE_HEALTH.get("ibkr", 0) + 1
                return float(px)
        except Exception as e:
            logger.debug(f"[tracker] IBKR price failed for {ticker}: {e}")
        return None

    def _yf_fast() -> Optional[float]:
        try:
            px = float(yf.Ticker(ticker).fast_info.last_price)
            if px > 0:
                _PRICE_HEALTH["yfinance"] += 1
                return px
        except Exception as e:
            logger.debug(f"[tracker] yfinance price failed for {ticker}: {e}")
        return None

    def _polygon() -> Optional[float]:
        try:
            from src.data import polygon_client
            px = polygon_client.get_last_price(ticker)
            if px and float(px) > 0:
                _PRICE_HEALTH["polygon"] += 1
                return float(px)
        except Exception as e:
            logger.debug(f"[tracker] Polygon price failed for {ticker}: {e}")
        return None

    def _yf_prepost() -> Optional[float]:
        # 1-minute prepost bars — the genuine extended print fast_info hides.
        try:
            from src.data.market_data import _extended_last_price
            px = _extended_last_price(yf.Ticker(ticker))
            if px and px > 0:
                _PRICE_HEALTH["yfinance"] += 1
                return float(px)
        except Exception as e:
            logger.debug(f"[tracker] yfinance prepost price failed for {ticker}: {e}")
        return None

    from src.performance.market_calendar import current_session
    # IBKR real-time first (when enabled) in BOTH sessions — its feed includes
    # extended prints and matches the fill venue; then the existing chain.
    if current_session() != "rth":
        px = _ibkr() or _yf_prepost() or _polygon() or _yf_fast()
    else:
        px = _ibkr() or _yf_fast() or _polygon()

    if px is not None and px > 0:
        return px
    _PRICE_HEALTH["failed"].append(ticker)
    logger.warning(f"[tracker] Could not fetch price for {ticker} — all sources failed")
    return None


def _now_iso() -> str:
    """Return the current wall-clock instant as a UTC ISO 8601 string.

    Stored as ``decision_datetime`` on every entry/exit so the audit trail
    captures the exact moment the pipeline made its call — even if the
    actual fill would have happened later (see ``_execution_iso``).
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _execution_iso() -> str:
    """Return the next realistic execution instant (ISO 8601 UTC).

    RTH: now (a fill can happen immediately). Extended sessions when
    ``extended_hours_mode="trade"``: also now — extended fills are real fills
    (the broker leg submits marketable LMT + outsideRth orders, and the cost
    model charges the wider extended spread on that leg). Everything else
    (overnight, weekends/holidays, or extended hours while NOT in trade mode)
    snaps forward to the next NYSE regular-session open, so the recorded
    execution time always corresponds to a moment the system could actually
    have traded.
    """
    from src.performance.market_calendar import current_session, effective_execution_iso
    if ((settings.extended_hours_mode or "").lower() == "trade"
            and current_session() == "extended"):
        return _now_iso()
    return effective_execution_iso()


def _reference_close(ticker: str) -> Optional[dict]:
    """Snapshot the most recent **finalized** daily OHLCV close for *ticker*.

    Returned as ``{"date": YYYY-MM-DD, "close": float}`` (or ``None`` when
    the ticker has no cached history).  Stored on every new trade entry and
    on every exit so the daily-NAV walk can recover the **exact split /
    dividend adjustment factor** later: if the cache is retroactively
    rescaled by a corporate action, ``close_series[ref_date] / ref_close``
    yields the multiplier that brings the recorded entry/exit price back
    onto the current cache scale, eliminating the phantom-jump bug that
    naked adjusted-close walks suffer through a split.

    Critically this must skip the **still-forming current-session bar**: an
    intraday entry/exit would otherwise record today's *in-progress* close
    (the latest print, not the finalized daily close), which later finalizes
    to a different value. The split-adjustment then divides the finalized
    close by that intraday snapshot and reads the gap as a phantom corporate
    action (observed 2026-06-15: INTC's same-day intraday exit booked a 1.04×
    "split", inflating its daily-NAV compound to +5.1% vs a true +1.5%). We
    therefore return the last close strictly BEFORE today's session — always a
    finalized bar — mirroring the completed-bars-only principle the indicators
    use (``market_data._drop_forming_bar``).
    """
    try:
        from src.data.cache import load_ohlcv
        from src.performance.daily_nav import _session_date
        df = load_ohlcv(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        today = date.today()
        # Walk newest→oldest to the last finalized session (strictly before
        # today), skipping the forming current-day bar and any bad close.
        for ts, close in zip(reversed(df.index.tolist()), reversed(df["Close"].tolist())):
            d = _session_date(ts)
            if d is None or d >= today:
                continue
            c = float(close)
            if c > 0:
                return {"date": d.isoformat(), "close": c}
        return None
    except Exception as e:
        logger.debug(f"[tracker] _reference_close({ticker}) failed: {e}")
        return None


def _trading_days_held(entry_date: str) -> int:
    """Count NYSE sessions in (entry_date, today] — exclusive of entry, inclusive of today.

    Delegates to ``market_calendar`` so US market holidays (MLK Day, Good
    Friday, Memorial Day, Juneteenth, Independence Day, Labor Day,
    Thanksgiving, Christmas, …) are excluded.  The previous implementation
    counted any weekday, so a trade held across a holiday auto-closed one
    market day too early.
    """
    try:
        from src.performance.market_calendar import market_days_between
        return market_days_between(date.fromisoformat(entry_date), date.today())
    except Exception:
        return 0


#   _dynamic_half_spread, _pct_return, and fmt_price live in
#   src/performance/spread.py — both tracker.py and daily_nav.py import them
#   from there to keep the dependency tree acyclic.


# ── Out-of-sample train/holdout split (deterministic, NOT walk-forward) ──────
#
# Each closed trade is permanently assigned to "train" or "holdout" via a
# deterministic hash of (seed, ticker, entry_date). The split is computed on
# the fly (no persistence) so legacy trades without a stored bucket get a
# stable assignment too, and so changing oos_split_seed reshuffles cleanly.
# This is a fixed partition, NOT a walk-forward — there is no rolling window
# and no time-ordered re-training. Adaptive weights use "train" only; the
# email surfaces "holdout" stats alongside so overfit is visible.

def _trade_split(trade: dict) -> str:
    """Return ``"train"`` or ``"holdout"`` for *trade*.

    Hash key = ``settings.oos_split_seed + "|" + ticker + "|" + entry_date``.
    A SHA-256 of the key, taken modulo 100, partitions trades into buckets:
    ``hash_pct < oos_holdout_pct`` → holdout, else train.

    Returns ``"train"`` unconditionally when the feature is disabled or the
    trade has no ``entry_date``/``ticker`` to hash.
    """
    if not settings.enable_oos_validation:
        return "train"
    holdout_pct = max(0, min(50, int(settings.oos_holdout_pct or 0)))
    if holdout_pct <= 0:
        return "train"
    ticker = (trade.get("ticker") or "").upper()
    entry  = trade.get("entry_date") or ""
    if not ticker or not entry:
        return "train"
    key = f"{settings.oos_split_seed}|{ticker}|{entry}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "holdout" if bucket < holdout_pct else "train"


def _filter_by_split(trades: List[dict], split: Optional[str]) -> List[dict]:
    """Return *trades* filtered to a given split, or unchanged for ``None``."""
    if split is None:
        return trades
    if split not in ("train", "holdout"):
        return trades
    return [t for t in trades if _trade_split(t) == split]


# ── Method attribution ────────────────────────────────────────────────────────
_ALL_METHODS = ("news", "sent_velocity", "tech", "massive", "insider", "put_call", "max_pain", "oi_skew", "vwap", "pattern", "momentum", "sector_momentum", "market_momentum", "money_flow", "trend_strength", "pead", "iv_rank", "iv_expr", "coint", "cross_sectional", "ext_gap", "broker_advisor", "f_value", "f_quality", "f_growth", "f_short_squeeze", "f_split", "f_dividend")
_METHOD_AGREE_THRESHOLD = 0.0    # any non-zero method score counts as a view (was 0.10)

# Category groupings: how methods map to higher-level signal families
METHOD_CATEGORIES: Dict[str, List[str]] = {
    "Sentiment":   ["news", "sent_velocity"],
    "Technical":   ["tech", "vwap", "pattern", "momentum", "sector_momentum", "money_flow", "trend_strength", "iv_rank", "ext_gap"],
    "Smart Money": ["insider"],
    "Options":     ["put_call", "max_pain", "oi_skew", "iv_expr"],
    "Fundamental": ["pead"],
    "Relative":    ["cross_sectional", "coint"],
    "Broker":      ["broker_advisor"],
}

# Human-readable method labels for reports
METHOD_LABELS: Dict[str, str] = {
    "news":     "News Sentiment",
    "sent_velocity": "Sentiment Velocity (Δ)",
    "tech":     "Technical Analysis",
    "massive":  "Massive Tech (RSI+MACD)",
    "insider":  "Smart Money / Insider",
    "put_call": "Put/Call Ratio",
    "max_pain": "Max Pain (GEX)",
    "oi_skew":  "OI Skew",
    "vwap":     "VWAP Distance",
    "pattern":  "Pattern Recognition",
    "momentum":   "Price Momentum",
    "sector_momentum": "Sector-Relative Momentum",
    "market_momentum": "Market-Relative Momentum (vs SPY)",
    "money_flow": "Money Flow (MFI+CMF+OBV)",
    "trend_strength": "Trend Strength (ADX/DMI+Donchian)",
    "pead":       "Post-Earnings Drift (PEAD)",
    "iv_rank":    "IV Rank + Directional",
    "iv_expr":    "IV Expression (Options Chain)",
    "coint":      "Cointegration Pairs",
    "cross_sectional": "Cross-Sectional Ranking",
    "ext_gap":    "Extended-Session Gap",
    "broker_advisor": "Broker Advisor (IBKR short-borrow)",
}

# Multi-timeframe technical labels (panel-only) — derived from the base daily
# labels so the 30-min / weekly variants stay in sync (e.g. "VWAP Distance ·
# 30m"). Mirrors schema.SIGNAL_TIMEFRAME_COLUMNS / multi_timeframe.TECHNICAL_METHODS.
METHOD_LABELS.update({
    f"{m}_{suffix}": f"{METHOD_LABELS[m]} · {label}"
    for m in ("tech", "vwap", "momentum", "money_flow",
              "trend_strength", "iv_rank", "pattern", "sector_momentum")
    for suffix, label in (("30m", "30m"), ("1w", "Weekly"))
})

# Massive fundamentals factor diagnostics (panel-only IC columns).
METHOD_LABELS.update({
    "f_value":         "Value (Massive: cheap P/E·P/B)",
    "f_quality":       "Quality (Massive: ROE·margin)",
    "f_growth":        "Growth (Massive: rev YoY)",
    "f_short_squeeze": "Short Squeeze (Massive: short%·days-to-cover)",
    "f_split":         "Split direction (Massive: forward+ / reverse−)",
    "f_dividend":      "Dividend change (Massive: increase+ / cut−)",
})


def _method_scores_from_signal(ticker: str, direction: str, signals_by_ticker: Optional[dict]) -> dict:
    """Extract per-method scores from the TickerSignal for a given ticker."""
    empty = {m: 0.0 for m in _ALL_METHODS}
    if signals_by_ticker is None:
        return empty
    sig = signals_by_ticker.get(ticker)
    if sig is None:
        return empty
    return {
        "news":      sig.sentiment_score,
        "sent_velocity": getattr(sig, "sentiment_velocity_score", 0.0),
        "tech":      sig.technical_score,
        "massive":   getattr(sig, "massive_score", 0.0),
        "insider":   sig.insider_score,
        "put_call":  sig.put_call_score,
        "max_pain":  sig.max_pain_score,
        "oi_skew":   sig.oi_skew_score,
        "vwap":      sig.vwap_score,
        "pattern":    getattr(sig, "pattern_score", 0.0),
        "momentum":   getattr(sig, "momentum_score", 0.0),
        "sector_momentum": getattr(sig, "sector_momentum_score", 0.0),
        "market_momentum": getattr(sig, "market_momentum_score", 0.0),
        "money_flow": getattr(sig, "money_flow_score", 0.0),
        "trend_strength": getattr(sig, "trend_strength_score", 0.0),
        "pead":       getattr(sig, "pead_score", 0.0),
        "iv_rank":    getattr(sig, "iv_rank_score", 0.0),
        "iv_expr":    getattr(sig, "iv_expr_score", 0.0),
        "coint":      getattr(sig, "coint_score", 0.0),
        "cross_sectional": getattr(sig, "cross_sectional_score", 0.0),
        "ext_gap":    getattr(sig, "ext_gap_score", 0.0),
        "broker_advisor": getattr(sig, "broker_advisor_score", 0.0),
        # Fundamental + corp-action factors live on the signal's fundamental_scores dict.
        **{m: float((getattr(sig, "fundamental_scores", None) or {}).get(m, 0.0))
           for m in ("f_value", "f_quality", "f_growth", "f_short_squeeze", "f_split", "f_dividend")},
    }


def _methods_agreeing(scores: dict, direction: str) -> List[str]:
    """Return list of method names whose score agrees with the trade direction."""
    sign = 1 if direction == "BULLISH" else -1
    return [m for m, v in scores.items() if v * sign > _METHOD_AGREE_THRESHOLD]


def _dominant_method(scores: dict, direction: str) -> str:
    """Return the method with the strongest agreement with the trade direction."""
    sign = 1 if direction == "BULLISH" else -1
    agreed = {m: v * sign for m, v in scores.items() if v * sign > _METHOD_AGREE_THRESHOLD}
    return max(agreed, key=agreed.get) if agreed else "none"


def _compute_nav_compound(trades: list) -> Optional[float]:
    """Capital-weighted, time-weighted compound return over *trades* (in %).

    Thin wrapper around ``daily_nav.compute_compound_return``: the engine
    walks every trade through its real OHLCV closes (no interpolation,
    no synthetic uniform-daily decomposition) and compounds the per-date
    capital-weighted portfolio returns only on dates where at least one
    position produced an event — weekends and market holidays drop out
    naturally because no close exists for them.  100% deterministic given
    ``(trades.json, cache/ohlcv/*.json)``.

    Accepts trade dicts directly (the previous (entry_str, exit_str, ret_pct,
    weight) tuple form was lossy — it threw away ticker/action/prices needed
    to read real daily marks).  Returns ``None`` when no daily-return event
    can be produced.
    """
    from src.performance.daily_nav import compute_compound_return

    return compute_compound_return(trades or [])


def _compute_segment_stats(trades: List[dict]) -> Optional[dict]:
    """Compute all standard performance metrics for a slice of trades.

    Compound return is computed from the actual daily price walk (no
    interpolation) via ``_compute_nav_compound``, which accepts trade dicts
    directly so the OHLCV cache can be consulted per ticker.
    """
    if not trades:
        return None
    returns = [t["return_pct"] for t in trades]
    multipliers = [t.get("position_size_multiplier", 1.0) for t in trades]
    total_mul = sum(multipliers)
    weighted_avg = (
        sum(r * m for r, m in zip(returns, multipliers)) / total_mul
        if total_mul else 0.0
    )
    wins = [r for r in returns if r > 0]
    return {
        "trades":          len(trades),
        "win_rate":        round(len(wins) / len(returns) * 100, 1),
        "compound_return": _compute_nav_compound(trades) or 0.0,
        "avg_return":      round(sum(returns) / len(returns), 2),
        "median_return":   round(median(returns), 2),
        "wtd_avg_return":  round(weighted_avg, 2),
        "best":            round(max(returns), 2),
        "worst":           round(min(returns), 2),
    }


# ── Per-LLM performance attribution ──────────────────────────────────────────
# New trades are stamped with the exact engine ids at entry (record_new_trades);
# legacy trades fall back to their run's recorded provider, resolved to a model
# id through the mappings below (and frozen onto the trade once by
# scripts/backfill_trade_llms.py, so a later ANALYST_MODEL change can't
# relabel history).

def _llm_run_map() -> Dict[str, dict]:
    """run_id → recorded LLM providers, from the runs table ({} when unavailable)."""
    try:
        df = repo.fetch_df(
            "SELECT run_id, llm_synthesis_provider, llm_sentiment_provider FROM runs"
        )
    except Exception as e:
        logger.debug(f"[tracker] llm run map unavailable: {e}")
        return {}
    return {
        str(r["run_id"]): {
            "synthesis_provider": r["llm_synthesis_provider"],
            "sentiment_summary":  r["llm_sentiment_provider"],
        }
        for _, r in df.iterrows()
    }


def _synthesis_model_for_provider(provider: Optional[str]) -> Optional[str]:
    """Provider string (runs table) → exact model id, mirroring claude_analyst routing."""
    p = (provider or "").lower()
    if p == "anthropic":
        return settings.analyst_model
    if p == "deepseek":
        return "deepseek-v4-flash"
    if p == "rule-based":
        return "rule-based (no LLM)"
    return None


def _sentiment_model_for_summary(summary: Optional[str]) -> Optional[str]:
    """'deepseek×40, anthropic×2' → exact model id of the majority provider."""
    if not summary:
        return None
    from src.analysis.sentiment import SENTIMENT_PROVIDER_MODELS
    counts: Dict[str, int] = {}
    for tok in str(summary).split(","):
        name, sep, cnt = tok.strip().partition("×")
        key = name.strip().lower()
        try:
            counts[key] = int(cnt) if sep else 0
        except ValueError:
            counts[key] = 0
    counted = {k: v for k, v in counts.items() if k in SENTIMENT_PROVIDER_MODELS}
    if not counted:
        return None
    top = max(counted.items(), key=lambda kv: kv[1])[0]
    return SENTIMENT_PROVIDER_MODELS[top]


def _llm_models_for_trade(trade: dict, run_map: Dict[str, dict]) -> tuple:
    """(synthesis_model, sentiment_model) for a trade — stamped fields first,
    then the run-record fallback for legacy trades."""
    synth = trade.get("llm_synthesis_model")
    sent  = trade.get("llm_sentiment_model")
    if synth and sent:
        return synth, sent
    run = run_map.get(str(trade.get("run_id") or "")) or {}
    if not synth:
        synth = _synthesis_model_for_provider(run.get("synthesis_provider"))
    if not sent:
        sent = _sentiment_model_for_summary(run.get("sentiment_summary"))
    return synth, sent


def _synthesis_model_for_rec(value: Optional[str]) -> Optional[str]:
    """``recommendations.llm_provider`` → exact model id.

    Legacy rows stored the provider string ('anthropic' / 'deepseek' /
    'rule-based'); newer rows persist the exact model id directly.
    """
    if not value:
        return None
    v = str(value).strip()
    return _synthesis_model_for_provider(v) or v


def _match_direction(t: dict, direction: Optional[str]) -> bool:
    """Whether trade/call ``t`` matches a LONG/SHORT filter (None/'' = both).
    Long = BUY / BULLISH entry; Short = SELL / BEARISH."""
    if not direction:
        return True
    tag = str(t.get("action") or t.get("direction") or "").upper()
    if direction == "long":
        return tag in ("BUY", "BULLISH")
    if direction == "short":
        return tag in ("SELL", "BEARISH")
    return True


def _build_pseudo_trades(calls: List[dict], session: Optional[str] = None,
                         bars_memo: Optional[dict] = None,
                         direction: Optional[str] = None) -> List[dict]:
    """Mark a list of directional *calls* to forward returns as pseudo-trades.

    Shared scoring core for the unbiased recommendation/signal streams (used by
    both ``_compute_llm_perf`` and ``compute_macro_eval``). The real ledger only
    holds gate-surviving trades, so to judge a decision layer on its FULL output
    we treat every directional call as a pseudo-trade: anchored at the snapshot
    price recorded when the call was made, marked at the latest cached close, and
    charged the same entry-leg cost model as a real trade (so a brand-new call
    starts slightly negative by the round-trip cost, exactly like a real position).

    Each call dict needs: ``ticker``, ``type``, ``action`` ('BUY'|'SELL'),
    ``entry_date``, ``entry_datetime``, ``snap_price`` (None → that day's close).
    ``llm_synthesis_model`` / ``llm_sentiment_model`` ride through when present.
    Calls are dropped when the ticker has no cached bars, no usable entry price,
    or no forward bar at/after the entry date yet. Calls must already be deduped
    by the caller (the dedup key differs per stream). Pass a shared ``bars_memo``
    to reuse OHLCV loads across several streams (e.g. the macro layers).
    """
    from src.data.cache import load_ohlcv

    if bars_memo is None:
        bars_memo = {}

    def _bars(ticker: str):
        if ticker not in bars_memo:
            try:
                b = load_ohlcv(ticker)
                bars_memo[ticker] = None if (b is None or b.empty or "Close" not in b.columns) else b
            except Exception:
                bars_memo[ticker] = None
        return bars_memo[ticker]

    out: List[dict] = []
    for c in calls:
        row_session = _trade_session(c)
        if not _session_matches(c, session):
            continue
        if not _match_direction(c, direction):
            continue
        bars = _bars(c["ticker"])
        if bars is None:
            continue
        # Entry anchor: snapshot price recorded with the call, else a close the
        # caller could have KNOWN at call time — the same-day close only for a
        # call generated after the RTH close, the PRIOR day's close otherwise
        # (anchoring an intraday call at that day's 16:00 close is look-ahead:
        # it erases the post-call drift from the engine's measured return).
        try:
            entry_price = float(c.get("snap_price"))
        except (TypeError, ValueError):
            entry_price = None
        if entry_price is not None and not entry_price > 0:   # also catches NaN
            entry_price = None
        if entry_price is None:
            closes = [(d.date().isoformat(), float(cl))
                      for d, cl in zip(bars.index, bars["Close"])]
            if _generated_after_rth_close(c.get("entry_datetime")):
                day = [cl for d, cl in closes if d == c["entry_date"]]
                entry_price = day[-1] if day and day[-1] > 0 else None
            if entry_price is None:
                prior = [cl for d, cl in closes if d < c["entry_date"]]
                entry_price = prior[-1] if prior and prior[-1] > 0 else None
        if entry_price is None:
            continue
        # End anchor: latest cached close (skip when no bar exists at/after entry yet).
        last_close = float(bars["Close"].iloc[-1])
        last_date = bars.index[-1].date().isoformat()
        if last_close <= 0 or last_date < c["entry_date"]:
            continue
        ctype = c.get("type") or "STOCK"
        t = {
            "ticker": c["ticker"],
            "type": ctype,
            "action": c["action"],
            "direction": c["action"],       # BUY/SELL — same convention as the hypothetical book
            "status": "OPEN",
            "entry_date": c["entry_date"],
            "entry_datetime": c["entry_datetime"],
            "entry_price": entry_price,
            # Entry leg bears that session's spread (extended calls pay the wider
            # book); the exit anchor is an RTH cached close, so no exit session.
            "entry_session": row_session,
            "current_price": last_close,
            "current_price_datetime": last_date,
            "exit_date": None,
            "exit_price": None,
            "return_pct": round(_pct_return(c["action"], entry_price, last_close, ctype,
                                            entry_session=row_session), 3),
            "position_size_multiplier": 1.0,
        }
        for k in ("llm_synthesis_model", "llm_sentiment_model"):
            if c.get(k):
                t[k] = c[k]
        out.append(t)
    return out


def _compute_llm_perf(window_days: Optional[int] = None,
                      session: Optional[str] = None,
                      direction: Optional[str] = None) -> dict:
    """Per-LLM engine stats over EVERY recommended trade, executed or not.

    The real ledger only holds recommendations that survived the actionable +
    sizing gates — far too few (and selection-biased) to compare engines. Each
    engine is therefore scored on its full recommendation stream: every
    BUY/SELL it produced (actionable or not), deduped to the engine's LAST
    call per (ticker, day) — mirroring the signal panel's last-run-per-day
    rule so a signal persisting across intraday runs isn't multi-counted.
    Each call becomes a pseudo-trade anchored at the snapshot price recorded
    at recommendation time (signals table; legacy recs fall back to the
    rec-day cached close) and marked at the latest cached close, through the
    same entry/exit cost model as real trades — so a brand-new call starts
    slightly negative by the round-trip cost, exactly like a real position.

    Returns ``{"synthesis": {model: stats}, "sentiment": {model: stats}}`` —
    synthesis = the final BUY/SELL caller, sentiment = the run-dominant
    per-ticker news scorer (attributed via the run record).
    """
    try:
        df = repo.fetch_df(
            "SELECT r.run_id, r.generated_at, r.ticker, r.type, r.action, "
            "       r.llm_provider, s.price AS snap_price "
            "FROM recommendations r "
            "LEFT JOIN signals s ON s.run_id = r.run_id AND s.ticker = r.ticker "
            "WHERE r.action IN ('BUY', 'SELL') "
            "ORDER BY r.generated_at"
        )
    except Exception as e:
        logger.debug(f"[tracker] llm_perf recommendations query failed: {e}")
        return {}
    if df.empty:
        return {}

    run_map = _llm_run_map()
    cutoff = (date.today() - timedelta(days=window_days)).isoformat() if window_days is not None else None

    # Dedup: the engine's last BUY/SELL call per (ticker, day) wins (rows are
    # ordered by generated_at, so later calls overwrite earlier ones).
    deduped: Dict[tuple, dict] = {}
    for rec in df.itertuples(index=False):
        gen = str(rec.generated_at or "")
        entry_date = gen[:10]
        if not entry_date or (cutoff and entry_date < cutoff):
            continue
        run = run_map.get(str(rec.run_id or "")) or {}
        synth = (_synthesis_model_for_rec(rec.llm_provider)
                 or _synthesis_model_for_provider(run.get("synthesis_provider")))
        sent = _sentiment_model_for_summary(run.get("sentiment_summary"))
        if not synth and not sent:
            continue
        deduped[(entry_date, rec.ticker, synth)] = {
            "ticker": rec.ticker,
            "type": rec.type or "STOCK",
            "action": rec.action,
            "entry_date": entry_date,
            "entry_datetime": gen,
            "snap_price": rec.snap_price,
            "llm_synthesis_model": synth,
            "llm_sentiment_model": sent,
        }

    pseudo = _build_pseudo_trades(list(deduped.values()), session, direction=direction)

    groups: Dict[str, Dict[str, List[dict]]] = {"synthesis": {}, "sentiment": {}}
    for t in pseudo:
        if t.get("llm_synthesis_model"):
            groups["synthesis"].setdefault(t["llm_synthesis_model"], []).append(t)
        if t.get("llm_sentiment_model"):
            groups["sentiment"].setdefault(t["llm_sentiment_model"], []).append(t)
    return {
        role: {m: s for m, s in ((m, _compute_segment_stats(ts)) for m, ts in by_model.items()) if s}
        for role, by_model in groups.items()
    }


# ── Macro / aggregated-layer evaluation ──────────────────────────────────────
# Compares the high-level DECISION LAYERS — the LLM synthesis final call, the
# mechanical aggregator's combined signal, and each method bundle (category) — on
# ONE unbiased, apples-to-apples basis: every directional call each layer made
# (not just the few that survived the trading gates), scored as a pseudo-trade
# over forward returns through the same cost model as real trades. This is the
# "whose confidence is more reliable" panel: synthesis vs aggregator vs bundles.
#
# A bundle's direction per (ticker, run) is the sign of the SUM of its member
# method scores; it "has a view" only when |sum| clears this floor (filters the
# all-but-zero noise so a near-flat family isn't counted as a directional call).
_BUNDLE_VIEW_FLOOR = 0.10

_MACRO_HEADLINE_ORDER = ("synthesis", "aggregator")


def compute_macro_eval(window_days: Optional[int] = None,
                       session: Optional[str] = None,
                       direction: Optional[str] = None) -> List[dict]:
    """Performance rows for the dashboard's Macro Evaluation table — the
    aggregated decision layers, each scored on its OWN full stream of directional
    calls via the same pseudo-trade model as ``_compute_llm_perf``.

    Returns a list of ``{label, group, trades, win_rate, avg_return,
    compound_return, wtd_avg_return, best, worst}`` rows (empty streams omitted),
    ordered LLM synthesis → aggregator → bundles (by win rate). Streams, all
    deduped to the LAST call per (ticker, day):
      • LLM Synthesis — every BUY/SELL recommendation (any engine, actionable or
        not), anchored at the recommendation-time snapshot (``signals.price``).
      • Aggregator — every ``signals`` row whose stored direction is BULLISH/BEARISH.
      • Bundle · <category> — direction = sign(Σ member method scores) from the
        ``signals`` per-method columns, when |Σ| ≥ ``_BUNDLE_VIEW_FLOOR``.
    """
    cutoff = (date.today() - timedelta(days=window_days)).isoformat() if window_days is not None else None
    bars_memo: Dict[str, object] = {}     # shared across every stream's NAV anchors
    rows: List[dict] = []

    def _emit(label: str, group: str, calls: List[dict]) -> None:
        st = _compute_segment_stats(_build_pseudo_trades(calls, session, bars_memo, direction))
        if st:
            rows.append({"label": label, "group": group, **st})

    # ── 1. LLM Synthesis (final call) — all engines combined ──────────────────
    try:
        rdf = repo.fetch_df(
            "SELECT r.run_id, r.generated_at, r.ticker, r.type, r.action, "
            "       s.price AS snap_price "
            "FROM recommendations r "
            "LEFT JOIN signals s ON s.run_id = r.run_id AND s.ticker = r.ticker "
            "WHERE r.action IN ('BUY', 'SELL') ORDER BY r.generated_at"
        )
    except Exception as e:
        logger.debug(f"[tracker] macro_eval recommendations query failed: {e}")
        rdf = None
    if rdf is not None and not rdf.empty:
        syn_ded: Dict[tuple, dict] = {}
        for rec in rdf.itertuples(index=False):
            gen = str(rec.generated_at or "")
            d = gen[:10]
            if not d or (cutoff and d < cutoff):
                continue
            syn_ded[(d, rec.ticker)] = {
                "ticker": rec.ticker, "type": rec.type or "STOCK", "action": rec.action,
                "entry_date": d, "entry_datetime": gen, "snap_price": rec.snap_price,
            }
        _emit("LLM Synthesis (final call)", "synthesis", list(syn_ded.values()))

    # ── 2 + 3. Aggregator + method bundles — from the signals cross-section ────
    try:
        method_cols = ", ".join(_ALL_METHODS)
        sdf = repo.fetch_df(
            "SELECT generated_at, ticker, type, direction, price, "
            f"{method_cols} FROM signals ORDER BY generated_at"
        )
    except Exception as e:
        logger.debug(f"[tracker] macro_eval signals query failed: {e}")
        sdf = None
    if sdf is not None and not sdf.empty:
        agg_ded: Dict[tuple, dict] = {}
        bundle_ded: Dict[str, Dict[tuple, dict]] = {c: {} for c in METHOD_CATEGORIES}
        for r in sdf.itertuples(index=False):
            gen = str(r.generated_at or "")
            d = gen[:10]
            if not d or (cutoff and d < cutoff):
                continue
            base = {"ticker": r.ticker, "type": (r.type or "STOCK"),
                    "entry_date": d, "entry_datetime": gen, "snap_price": r.price}
            direction = str(r.direction or "").upper()
            if direction in ("BULLISH", "BEARISH"):
                agg_ded[(d, r.ticker)] = {**base, "action": "BUY" if direction == "BULLISH" else "SELL"}
            for cat, members in METHOD_CATEGORIES.items():
                total = 0.0
                for m in members:
                    v = getattr(r, m, 0.0)
                    if v is not None and v == v:     # skip None / NaN
                        total += float(v)
                if abs(total) >= _BUNDLE_VIEW_FLOOR:
                    bundle_ded[cat][(d, r.ticker)] = {**base, "action": "BUY" if total > 0 else "SELL"}
        _emit("Aggregator (combined signal)", "aggregator", list(agg_ded.values()))
        for cat in METHOD_CATEGORIES:
            _emit(f"Bundle · {cat}", "bundle", list(bundle_ded[cat].values()))

    # Headline layers first (synthesis, then aggregator), bundles after by win rate.
    head = sorted((r for r in rows if r["group"] in _MACRO_HEADLINE_ORDER),
                  key=lambda x: _MACRO_HEADLINE_ORDER.index(x["group"]))
    bundles = sorted((r for r in rows if r["group"] == "bundle"),
                     key=lambda x: -(x.get("win_rate") or 0))
    return head + bundles


_ASSET_TYPE_LABELS: Dict[str, str] = {
    "STOCK":     "Stocks only",
    "ETF":       "ETFs only",
    "COMMODITY": "Commodities only",
}


_SESSION_LABELS: Dict[str, str] = {
    "rth":       "Regular hours (RTH) only",
    "extended":  "Extended hours only",
    "overnight": "Overnight only",
}


def _compute_performance_table(trades: List[dict]) -> List[dict]:
    """Build unified breakdown rows: total → asset → direction → session → methods.

    Accepts both closed and open trades; open trades use their current M2M return_pct
    (maintained by update_open_trades() each run), equivalent to a hypothetical exit.
    Session rows (entry-session split) appear only once at least one trade was
    entered outside RTH — before that they would just duplicate "All Trades".
    """
    rows: List[dict] = []

    # Total row
    stats = _compute_segment_stats(trades)
    if stats:
        rows.append({"label": "All Trades", "group": "total", **stats})

    # Asset-type rows
    for type_key, label in _ASSET_TYPE_LABELS.items():
        subset = [t for t in trades if t.get("type") == type_key]
        stats = _compute_segment_stats(subset)
        if stats:
            rows.append({"label": label, "group": "asset", **stats})

    # Direction rows (long vs short)
    longs  = [t for t in trades if t.get("action") == "BUY"]
    shorts = [t for t in trades if t.get("action") == "SELL"]
    for subset, label in ((longs, "Longs only (BUY)"), (shorts, "Shorts only (SELL)")):
        stats = _compute_segment_stats(subset)
        if stats:
            rows.append({"label": label, "group": "direction", **stats})

    # Session rows (entry session: RTH / extended / overnight)
    by_session: Dict[str, List[dict]] = {}
    for t in trades:
        by_session.setdefault(_trade_session(t), []).append(t)
    if set(by_session) - {"rth"}:
        for sess_key, label in _SESSION_LABELS.items():
            stats = _compute_segment_stats(by_session.get(sess_key, []))
            if stats:
                rows.append({"label": label, "group": "session", **stats})

    # Signal-method rows (attribution-enabled trades only)
    attributed = [t for t in trades if t.get("methods_agreeing")]
    for method in _ALL_METHODS:
        subset = [t for t in attributed if method in t.get("methods_agreeing", [])]
        if len(subset) < 2:
            continue
        stats = _compute_segment_stats(subset)
        if stats:
            rows.append({"label": METHOD_LABELS[method], "group": "method", **stats})

    return rows


def _compute_method_stats(closed_trades: List[dict]) -> dict:
    """
    For each method, compute win rate and avg return across all closed trades
    where that method agreed with the trade direction at entry.

    Returns a dict keyed by method name → {trades, win_rate, avg_return, weighted_avg_return}.
    Only includes methods with ≥ 3 attributed trades (too few → noise).
    """
    MIN_TRADES = 3
    buckets: Dict[str, List[tuple]] = {m: [] for m in _ALL_METHODS}

    for t in closed_trades:
        method_scores = t.get("method_scores", {})
        methods_agreed = t.get("methods_agreeing", [])
        ret = t.get("return_pct", 0.0)
        mul = t.get("position_size_multiplier", 1.0)
        for m in methods_agreed:
            if m in buckets:
                buckets[m].append((ret, mul))

    result = {}
    for method, entries in buckets.items():
        if len(entries) < MIN_TRADES:
            continue
        returns = [r for r, _ in entries]
        muls    = [m for _, m in entries]
        total_mul = sum(muls)
        wins = [r for r in returns if r > 0]
        weighted_avg = sum(r * m for r, m in zip(returns, muls)) / total_mul if total_mul else 0.0
        result[method] = {
            "trades":              len(entries),
            "win_rate":            round(len(wins) / len(returns) * 100, 1),
            "avg_return":          round(sum(returns) / len(returns), 2),
            "weighted_avg_return": round(weighted_avg, 2),
        }

    return result


def _compute_category_stats(closed_trades: List[dict]) -> dict:
    """
    Roll per-method attribution up into category groups.
    A trade contributes to a category if ANY method in that category agreed.
    Returns dict keyed by category name → {trades, win_rate, avg_return, weighted_avg_return}.
    """
    MIN_TRADES = 2
    buckets: Dict[str, List[tuple]] = {cat: [] for cat in METHOD_CATEGORIES}

    for t in closed_trades:
        methods_agreed = set(t.get("methods_agreeing", []))
        if not methods_agreed:
            continue
        ret = t.get("return_pct", 0.0)
        mul = t.get("position_size_multiplier", 1.0)
        seen_cats: set = set()
        for cat, members in METHOD_CATEGORIES.items():
            if any(m in methods_agreed for m in members) and cat not in seen_cats:
                buckets[cat].append((ret, mul))
                seen_cats.add(cat)

    result = {}
    for cat, entries in buckets.items():
        if len(entries) < MIN_TRADES:
            continue
        returns  = [r for r, _ in entries]
        muls     = [m for _, m in entries]
        total_m  = sum(muls)
        wins     = [r for r in returns if r > 0]
        w_avg    = sum(r * m for r, m in zip(returns, muls)) / total_m if total_m else 0.0
        result[cat] = {
            "trades":              len(entries),
            "win_rate":            round(len(wins) / len(returns) * 100, 1),
            "avg_return":          round(sum(returns) / len(returns), 2),
            "weighted_avg_return": round(w_avg, 2),
        }

    return result


def _compute_convergence_stats(closed_trades: List[dict]) -> dict:
    """
    Group closed trades by how many methods agreed at entry.
    Directly tests whether the convergence multiplier (1.25×/0.60×) is justified.
    Returns dict keyed by convergence label → {trades, win_rate, avg_return}.
    """
    def _bucket(n: int) -> str:
        if n >= 4: return "4+ methods"
        if n == 3: return "3 methods"
        if n == 2: return "2 methods"
        return "1 method"

    buckets: Dict[str, List[float]] = {}

    for t in closed_trades:
        agreed = t.get("methods_agreeing", [])
        if not agreed:   # no attribution data (legacy trade)
            continue
        label = _bucket(len(agreed))
        buckets.setdefault(label, []).append(t.get("return_pct", 0.0))

    order = ["4+ methods", "3 methods", "2 methods", "1 method"]
    result = {}
    for label in order:
        returns = buckets.get(label, [])
        if len(returns) < 2:
            continue
        wins = [r for r in returns if r > 0]
        result[label] = {
            "trades":     len(returns),
            "win_rate":   round(len(wins) / len(returns) * 100, 1),
            "avg_return": round(sum(returns) / len(returns), 2),
        }

    return result


def _compute_dominant_stats(closed_trades: List[dict]) -> dict:
    """
    Group closed trades by the method that had the strongest agreement at entry.
    Shows which single signal type best predicts profitable outcomes.
    Returns dict keyed by method name → {trades, win_rate, avg_return}.
    """
    MIN_TRADES = 2
    buckets: Dict[str, List[float]] = {}

    for t in closed_trades:
        dom = t.get("dominant_method", "none")
        if dom == "none":
            continue
        buckets.setdefault(dom, []).append(t.get("return_pct", 0.0))

    result = {}
    for method, returns in buckets.items():
        if len(returns) < MIN_TRADES:
            continue
        wins = [r for r in returns if r > 0]
        result[method] = {
            "trades":     len(returns),
            "win_rate":   round(len(wins) / len(returns) * 100, 1),
            "avg_return": round(sum(returns) / len(returns), 2),
        }

    return result


def _compute_confidence_ranked(closed_trades: List[dict]) -> List[dict]:
    """Return closed trades sorted by confidence descending with rolling cumulative stats.

    Each row adds a cumulative_avg and cumulative_win_rate that reflect the
    portfolio return if you had *only* taken the top-N most confident signals.
    """
    sorted_trades = sorted(
        closed_trades,
        key=lambda t: t.get("confidence", 0.0),
        reverse=True,
    )
    rows = []
    running_sum = 0.0
    running_wins = 0
    for i, t in enumerate(sorted_trades, 1):
        ret = t.get("return_pct", 0.0)
        running_sum += ret
        if ret > 0:
            running_wins += 1
        rows.append({
            "rank":                 i,
            "ticker":               t["ticker"],
            "action":               t["action"],
            "confidence":           t.get("confidence", 0.0),
            "return_pct":           ret,
            "entry_date":           t.get("entry_date", ""),
            "exit_date":            t.get("exit_date", ""),
            "cumulative_avg":       round(running_sum / i, 2),
            "cumulative_win_rate":  round(running_wins / i * 100, 1),
        })
    return rows


# ---------------------------------------------------------------------------
# Position sizing helpers
# ---------------------------------------------------------------------------

def _position_multiplier(confidence: float) -> float:
    """Map confidence to a position-size multiplier — continuous in [1.0, 2.0].

    Replaces the previous three-tier step function (1.0× / 1.5× / 2.0×)
    with a piecewise-linear interpolation that pins the same anchor points
    so the headline endpoints (the lowest actionable conf and the highest
    realistic conf) still produce 1.0× and 2.0×, but a confidence of 0.86
    no longer jumps a full 0.5× over 0.85.

        conf ≤ 0.78  → 1.0× (baseline; anything below would have been
                            filtered out by the actionable gate anyway)
        conf = 0.85  → 1.50× (former Mid-tier midpoint)
        conf = 0.92  → 1.85× (former High-tier entry)
        conf ≥ 0.95  → 2.00× (cap)

    Linear interpolation inside each of those three segments. The result
    is rounded to 2 decimals for storage stability and so the multiplier
    composes cleanly with the correlation haircut downstream.
    """
    if confidence <= 0.78:
        return 1.00
    if confidence >= 0.95:
        return 2.00
    if confidence <= SIZE_TIER_HALF:        # 0.78 → 0.85 ramps 1.00 → 1.50
        t = (confidence - 0.78) / (SIZE_TIER_HALF - 0.78)
        return round(1.00 + t * 0.50, 2)
    if confidence <= SIZE_TIER_FULL:        # 0.85 → 0.92 ramps 1.50 → 1.85
        t = (confidence - SIZE_TIER_HALF) / (SIZE_TIER_FULL - SIZE_TIER_HALF)
        return round(1.50 + t * 0.35, 2)
    # 0.92 → 0.95 ramps 1.85 → 2.00
    t = (confidence - SIZE_TIER_FULL) / (0.95 - SIZE_TIER_FULL)
    return round(1.85 + t * 0.15, 2)


def _sector_key(rec: Recommendation) -> str:
    """Return a sector grouping key, stored on every new trade as a passive
    diagnostic (useful for sector-level performance slicing). The per-sector
    cap that previously consumed this value was removed in favour of the
    correlation-aware haircut, which catches cross-sector factor concentration
    (e.g. NVDA + AVGO + SMH) the GICS bucket cap missed.

    - Sector ETFs (XLK, XLF …):  the ETF ticker itself (each ETF = its own sector)
    - Commodities (GLD, SLV …):  "COMMODITY"
    - Stocks:  look up in aggregator._SECTOR_MAP; fall back to "STOCK/<ticker>".
    """
    ticker = rec.ticker.upper()
    if rec.type == "COMMODITY" or ticker in [
        t.upper() for t in settings.commodities_list
    ]:
        return "COMMODITY"
    if rec.type == "ETF" or ticker in [
        t.upper() for t in settings.sectors_list
    ]:
        return ticker   # each sector ETF is its own bucket
    # Stocks — import lazily to avoid circular dependency
    try:
        from src.signals.aggregator import _SECTOR_MAP
        mapped = _SECTOR_MAP.get(ticker)
        if mapped:
            return mapped
    except Exception:
        pass
    return f"STOCK/{ticker}"   # unique key → each unknown stock is uncapped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record_new_trades(
    recommendations: List[Recommendation],
    signals_by_ticker: Optional[dict] = None,
    run_id: Optional[str] = None,
    llm_synthesis_model: Optional[str] = None,
    llm_sentiment_model: Optional[str] = None,
) -> dict:
    """Open a new trade for each BUY/SELL recommendation not already open today.

    Sizing chain:
      1. Confidence tier → 1.0× / 1.5× / 2.0× base multiplier.
      2. Correlation haircut against same-direction open peers (continuous
         scale-down based on realized pairwise correlations; portfolio-wide
         hard skip when Σ|ρ|·size exceeds correlation_portfolio_cap).

    The legacy per-sector cap was removed because the correlation haircut
    handles cross-sector factor concentration better (the GICS bucket cap
    missed combos like NVDA + AVGO + SMH that share factor exposure across
    sector boundaries). sector_key is still computed and stored on each
    trade as a passive diagnostic for sector-level performance slicing.

    Per-method signal scores are stored for later attribution analysis.
    """
    trades = _load_trades()
    today = date.today().isoformat()

    # Prevent stacking: skip any ticker that already has an OPEN position.
    # Tickers opened later in THIS loop are added too, so a recommendations
    # list that repeats a ticker (duplicate LLM output) opens it exactly once.
    already_open = {t["ticker"] for t in trades if t["status"] == "OPEN"}

    # Running per-direction list of OPEN positions for correlation sizing.
    # Includes both pre-existing opens AND any trades opened earlier in this
    # same loop iteration, so the second candidate in a batch sees the first
    # as a same-direction peer.
    opens_by_direction: Dict[str, List[dict]] = {"BUY": [], "SELL": []}
    for t in trades:
        if t.get("status") == "OPEN" and t.get("action") in ("BUY", "SELL"):
            opens_by_direction[t["action"]].append(t)

    from src.performance.correlation import correlation_haircut

    # Per-gate counters — surfaced by the pipeline as the run's diagnostic
    # so the user can see which sizing constraints are actually firing.
    diag = {
        "considered":           0,   # BUY/SELL recs received
        "skipped_already_open": 0,
        "skipped_reentry_cooldown": 0,
        "skipped_correlation_cap": 0,
        "skipped_no_price":     0,
        "haircut_applied":      0,   # informational — size reduced, not skipped
        "extended_haircut_applied": 0,  # entries sized down for an off-RTH fill
        "deferred_intraday_timing": 0,
        "opened":               0,
    }

    # Re-entry cooldown: the most recent exit instant per (ticker, action) among
    # CLOSED trades. A rule-based exit (horizon_expired / llm_confidence_loss)
    # firing while the entry side still likes the name produced close→reopen
    # churn within the SAME tick (observed HUM 2026-06-29: closed 11:38:30,
    # reopened 11:38:31 — a pure round-trip cost). Same direction only — an
    # opposite-direction entry is a genuine flip and is never blocked.
    cooldown_h = float(settings.reentry_cooldown_hours or 0.0)
    recent_exits: Dict[tuple, datetime] = {}
    if cooldown_h > 0:
        for t in trades:
            if t.get("status") != "CLOSED":
                continue
            raw = t.get("exit_datetime") or t.get("exit_date")
            if not raw:
                continue
            try:
                _dt = datetime.fromisoformat(str(raw))
            except (ValueError, TypeError):
                continue
            if _dt.tzinfo is None:
                _dt = _dt.replace(tzinfo=ET)
            key = (t.get("ticker"), t.get("action"))
            if key not in recent_exits or _dt > recent_exits[key]:
                recent_exits[key] = _dt

    new_count = 0
    for rec in recommendations:
        if rec.action not in ("BUY", "SELL"):
            continue
        diag["considered"] += 1
        if rec.ticker in already_open:
            diag["skipped_already_open"] += 1
            continue
        if cooldown_h > 0:
            prev_exit = recent_exits.get((rec.ticker, rec.action))
            if prev_exit is not None:
                age_h = (datetime.now(ET) - prev_exit.astimezone(ET)).total_seconds() / 3600.0
                if 0 <= age_h < cooldown_h:
                    diag["skipped_reentry_cooldown"] += 1
                    logger.info(
                        f"[tracker] Skipping {rec.ticker} {rec.action} — same-direction "
                        f"position closed {age_h:.1f}h ago (< {cooldown_h:g}h re-entry cooldown)"
                    )
                    continue

        # ── Step 1: confidence tier (base size) ───────────────────────────
        conf_multiplier = _position_multiplier(rec.confidence)
        sector = _sector_key(rec)  # stored on the trade as a passive diagnostic

        # ── Step 2: correlation haircut against same-direction open peers ─
        same_dir_peers = opens_by_direction.get(rec.action, [])
        corr_multiplier, corr_diag = correlation_haircut(rec.ticker, same_dir_peers)
        if corr_diag.get("portfolio_cap_hit"):
            diag["skipped_correlation_cap"] += 1
            logger.info(
                f"[tracker] Skipping {rec.ticker} — correlated-exposure cap hit "
                f"(Σ|ρ|·size={corr_diag['weighted_exposure']:.2f} > "
                f"{settings.correlation_portfolio_cap:.2f}); peers={list(corr_diag['per_ticker'].keys())}"
            )
            continue
        multiplier = round(conf_multiplier * corr_multiplier, 3)
        if corr_multiplier < 0.999 and corr_diag.get("mean_corr") is not None:
            diag["haircut_applied"] += 1
            logger.info(
                f"[tracker] {rec.ticker}: correlation haircut "
                f"{conf_multiplier:.2f}× × {corr_multiplier:.2f}× = {multiplier:.2f}×  "
                f"(mean ρ={corr_diag['mean_corr']:+.2f} across {corr_diag['n_pairs']} "
                f"{rec.action} peer{'s' if corr_diag['n_pairs'] != 1 else ''})"
            )

        # Capture the exact fetch instant (decision_datetime) AND the
        # realistic fill instant (entry_datetime) — identical during RTH and
        # extended-trade ticks, snapped to the next session open otherwise.
        decision_at = _now_iso()
        executed_at = _execution_iso()

        # ── Step 3: extended-session sizing haircut ───────────────────────
        # Entries filled outside RTH are sized down: the book is thin, the
        # modeled spread is 4× wider, and the pre-prod goal is to accumulate
        # evidence at reduced weight until paper fills prove the edge.
        entry_session = _session_of_iso(executed_at)
        ext_mult = float(settings.extended_size_multiplier)
        if entry_session != "rth" and ext_mult < 0.999:
            multiplier = round(multiplier * ext_mult, 3)
            diag["extended_haircut_applied"] += 1
            logger.info(
                f"[tracker] {rec.ticker}: {entry_session}-session entry — "
                f"size ×{ext_mult:g} → {multiplier:.2f}×"
            )

        price = _fetch_price(rec.ticker)
        if price is None:
            diag["skipped_no_price"] += 1
            logger.warning(f"[tracker] Skipping {rec.ticker} — could not fetch entry price")
            continue
        if price <= 0:
            diag["skipped_no_price"] += 1
            logger.warning(f"[tracker] Skipping {rec.ticker} — entry price is {fmt_price(price)} (warrant/delisted/negative?)")
            continue

        # ── Intraday timing gate ──────────────────────────────────────────
        # Hybrid model: the daily stack decided the DIRECTION; here a 30-min
        # momentum read only times the ENTRY. Defer (don't open this tick) when
        # intraday momentum is strongly against the trade; the next 30-min tick
        # re-evaluates, so the position simply waits for a less hostile entry.
        if settings.enable_intraday_timing:
            from src.signals.intraday_timing import compute_intraday_timing, opposes_entry
            timing = compute_intraday_timing(rec.ticker)
            if opposes_entry(rec.action, timing, settings.intraday_timing_defer_threshold):
                diag["deferred_intraday_timing"] += 1
                logger.info(
                    f"[tracker] Deferring {rec.action} {rec.ticker} — intraday "
                    f"{timing['classification']} (30-min score {timing['score']:+.2f}) "
                    f"opposes entry; will re-check next tick"
                )
                continue

        # Capture per-method scores for attribution analysis
        mscores  = _method_scores_from_signal(rec.ticker, rec.direction, signals_by_ticker)
        agreed   = _methods_agreeing(mscores, rec.direction)
        dominant = _dominant_method(mscores, rec.direction)

        # Snapshot the entry-time signal state so monitor_open_positions can
        # later compare today's signal against it for decay detection.
        sig_at_entry = None
        pattern_at_entry: Optional[str] = None
        pattern_score_at_entry: Optional[float] = None
        if signals_by_ticker is not None:
            sig = signals_by_ticker.get(rec.ticker)
            if sig is not None:
                sig_at_entry = {
                    "combined_score":  round(float(getattr(sig, "combined_score", 0.0)), 4),
                    "confidence":      round(float(sig.confidence), 4),
                    "direction":       sig.direction,
                    "methods_agreeing": list(agreed),
                }
                # Capture the chart pattern present at entry (if any) and the
                # score it produced — feeds the live pattern registry on close
                # so subsequent runs can blend live-trade outcomes into the
                # per-ticker synthetic prior.
                pn = getattr(sig, "pattern_name", "")
                if pn:
                    pattern_at_entry = pn
                    pattern_score_at_entry = round(float(getattr(sig, "pattern_score", 0.0)), 3)

        # Reference close at trade time — frozen here so any future split or
        # special dividend that rescales the cache can be detected and
        # un-applied by the daily-NAV engine.  None for tickers without
        # cached history; the walk falls back to a no-adjustment factor.
        ref = _reference_close(rec.ticker)

        # NOTE: prices are stored at full float64 precision (no round(price,4))
        # so sub-penny stocks/warrants retain their genuine decimals — the old
        # 4-decimal rounding lost up to ~5% of accuracy on names like TALKW.
        # Display formatting is handled by fmt_price() at render time.
        # Horizon synthesis: cap the LLM's time_horizon to the mechanical target
        # (it may confirm or SHORTEN, never lengthen) and stamp the effective
        # holding horizon for the matched exit + the dashboard.
        _sig = (signals_by_ticker or {}).get(rec.ticker)
        eff_horizon, horizon_label, horizon_net = "", "", 0.0
        if (_sig is not None and settings.enable_horizon_synthesis
                and getattr(_sig, "target_horizon", "")):
            from src.signals.edge_curve import cap_horizon, HORIZON_BUCKET
            eff_horizon = cap_horizon(_sig.target_horizon, rec.time_horizon)
            horizon_label = HORIZON_BUCKET.get(eff_horizon, "")
            horizon_net = float(getattr(_sig, "horizon_net_edge_pct", 0.0) or 0.0)

        new_trade = {
            "ticker": rec.ticker,
            "run_id": run_id,
            "recommendation_id": (
                hashlib.sha1(f"{run_id}|{rec.ticker}".encode("utf-8")).hexdigest()[:16]
                if run_id else None
            ),
            "type": rec.type,
            "action": rec.action,
            "direction": rec.direction,
            "confidence": rec.confidence,
            "position_size_multiplier": multiplier,
            "confidence_size_multiplier": conf_multiplier,
            "correlation_size_multiplier": corr_multiplier,
            "correlation_mean_corr":      corr_diag.get("mean_corr"),
            "correlation_max_corr":       corr_diag.get("max_corr"),
            "correlation_weighted_exposure": corr_diag.get("weighted_exposure", 0.0),
            "correlation_peers":          corr_diag.get("per_ticker", {}),
            "sector_key": sector,
            "entry_date": today,
            "entry_datetime": executed_at,
            "entry_session": entry_session,
            "extended_size_multiplier": ext_mult if entry_session != "rth" else 1.0,
            "decision_datetime": decision_at,
            "entry_price": float(price),
            "entry_ref_close": ref["close"] if ref else None,
            "entry_ref_close_date": ref["date"] if ref else None,
            "rationale": rec.rationale,
            # Horizon synthesis — the LLM bucket, the mechanical/capped holding
            # horizon (drives the matched exit's time-stop), and its net edge.
            "time_horizon":        rec.time_horizon,
            "target_horizon":      eff_horizon,
            "horizon_label":       horizon_label,
            "horizon_net_edge_pct": horizon_net,
            "current_price": float(price),
            "current_price_datetime": decision_at,
            "return_pct": 0.0,
            "weighted_return_pct": 0.0,
            "days_held": 0,
            "exit_date": None,
            "exit_datetime": None,
            "exit_decision_datetime": None,
            "exit_price": None,
            "exit_ref_close": None,
            "exit_ref_close_date": None,
            "exit_reason": None,
            "status": "OPEN",
            # Method attribution fields
            "method_scores":    mscores,
            "methods_agreeing": agreed,
            "dominant_method":  dominant,
            # Exact LLM engines in use at entry (synthesis = final call,
            # sentiment = run-dominant per-ticker scorer) — per-LLM attribution.
            "llm_synthesis_model": llm_synthesis_model,
            "llm_sentiment_model": llm_sentiment_model,
            # Open-position monitoring fields
            "signal_at_entry":  sig_at_entry,
            "max_favorable_excursion":  0.0,
            "mfe_date":                 today,
            "max_adverse_excursion":    0.0,
            "mae_date":                 today,
            # Pattern-registry fields (live outcome feedback loop)
            "pattern_at_entry":       pattern_at_entry,
            "pattern_score_at_entry": pattern_score_at_entry,
        }
        trades.append(new_trade)
        already_open.add(rec.ticker)
        # Register the just-opened trade as a same-direction peer so the next
        # candidate in this batch sees it for correlation purposes.
        opens_by_direction.setdefault(rec.action, []).append(new_trade)
        new_count += 1
        diag["opened"] += 1
        corr_log = ""
        if corr_diag.get("mean_corr") is not None:
            corr_log = (
                f", mean ρ={corr_diag['mean_corr']:+.2f}/{corr_diag['n_pairs']} peer"
                f"{'s' if corr_diag['n_pairs'] != 1 else ''}"
            )
        logger.info(
            f"[tracker] Opened {rec.action} {rec.ticker} @ {price:.2f} "
            f"(decision={decision_at}, exec={executed_at}) | size={multiplier}× "
            f"(conf={rec.confidence:.0%}{corr_log})"
        )

    _save_trades(trades)
    logger.info(f"[tracker] {new_count} new trade(s) recorded")
    return diag


def get_open_trades() -> List[dict]:
    """Every OPEN trade dict (full records, incl. ``llm_synthesis_model`` /
    ``llm_sentiment_model``). Used by the pipeline's opener-pinned hold-review to
    group positions by their opening engines."""
    return [t for t in _load_trades() if t.get("status") == "OPEN"]


def get_open_position_summaries() -> List[dict]:
    """Compact snapshot of every OPEN trade for the held-positions prompt
    block: ticker, direction, entry anchor, and the latest mark from the
    previous tick (``update_open_trades`` runs after synthesis, so the mark
    is ≤1 tick old — fine for a review prompt, labelled 'last mark')."""
    out = []
    for t in _load_trades():
        if t.get("status") != "OPEN":
            continue
        out.append({
            "ticker": t["ticker"],
            "action": t.get("action"),
            "entry_date": t.get("entry_date"),
            "entry_price": t.get("entry_price"),
            "current_price": t.get("current_price") or t.get("entry_price"),
            "return_pct": t.get("return_pct") or 0.0,
            "days_held": t.get("days_held") or 0,
        })
    return out


def compute_hold_prompt_eval(trades: Optional[List[dict]] = None) -> dict:
    """A/B outcome comparison for the held-positions prompt experiment.

    Every trade CLOSED on a run carries that run's coin flip
    (``exit_hold_prompt`` True/False, stamped by the two close paths). The
    prompt can only influence exits directly through the synthesis →
    actionable → signal-reversal path, but ALL closes are stamped so the
    comparison also captures second-order effects (a reversal close today
    preempts a decay close tomorrow). Returns ``{"on": stats, "off": stats}``
    with the same row shape as the dashboard's LLM-engine rows; segments
    with no data are None. Pre-experiment closes (no stamp) are excluded.
    """
    trades = trades if trades is not None else _load_trades()

    def _seg(flag: bool) -> Optional[dict]:
        seg = [t for t in trades
               if t.get("status") == "CLOSED" and t.get("exit_hold_prompt") is flag]
        if not seg:
            return None
        rets = [float(t.get("return_pct") or 0.0) for t in seg]
        wins = sum(1 for r in rets if r > 0)
        return {
            "trades": len(seg),
            "win_rate": round(100.0 * wins / len(seg), 1),
            "avg_return": round(sum(rets) / len(rets), 2),
        }

    return {"on": _seg(True), "off": _seg(False)}


def compute_exit_reason_perf() -> List[dict]:
    """Per exit-reason performance over CLOSED trades — one row per ``exit_reason``
    with the standard segment stats (trades / win_rate / avg / median / compound /
    best / worst via ``_compute_segment_stats``). Groups by the reason stamped when
    the position closed (``llm_signal_flipped``, ``llm_confidence_loss``,
    ``horizon_expired``, ``macro_regime_exit``, ``signal_flipped``, ``signal_decay``,
    ``confidence_loss``, ``intraday_reversal`` …). This is the realized outcome of
    each exit RULE — the day-one companion to the forward-looking exit-method IC
    table (both live in the Exit Performance tab). Sorted by trade count desc."""
    closed = [t for t in _load_trades() if t.get("status") == "CLOSED"]
    by_reason: Dict[str, List[dict]] = {}
    for t in closed:
        by_reason.setdefault(t.get("exit_reason") or "(unspecified)", []).append(t)
    rows: List[dict] = []
    for reason, seg in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        stats = _compute_segment_stats(seg)
        if stats:
            rows.append({"exit_reason": reason, **stats})
    return rows


def close_trades_on_signal_reversal(actionable_recs: List["Recommendation"],
                                    hold_prompt_active: Optional[bool] = None) -> int:
    """Close open trades whose direction has reversed in today's actionable recommendations.

    An open BUY is closed when an actionable SELL exists for the same ticker (and vice versa).
    Uses current_price already refreshed by update_open_trades() — no second fetch needed.
    The closed trade is immediately saved; record_new_trades() will then open the new leg.
    ``hold_prompt_active`` is this run's held-positions-prompt coin flip, stamped on each
    close as ``exit_hold_prompt`` for the A/B evaluation.

    Fix #2 — when ``enable_llm_hold_review`` is on this path SKIPS LLM-opened
    positions entirely: they are owned by the opener-pinned hold-review in
    ``monitor_open_positions`` (which runs first and detects flips via the opening
    engine every tick). This path serves only legacy / rule-based-opened trades.
    """
    trades = _load_trades()
    today = date.today().isoformat()

    # ticker → action for all actionable BUY/SELL recs
    rev_signal: Dict[str, str] = {
        r.ticker: r.action
        for r in actionable_recs
        if r.action in ("BUY", "SELL")
    }

    closed = 0
    for trade in trades:
        if trade["status"] != "OPEN":
            continue
        new_action = rev_signal.get(trade["ticker"])
        if new_action is None:
            continue
        if not (
            (trade["action"] == "BUY"  and new_action == "SELL") or
            (trade["action"] == "SELL" and new_action == "BUY")
        ):
            continue
        # Fix #2 — LLM-opened positions are owned by the opener-pinned
        # hold-review in monitor_open_positions (which runs first and detects
        # flips via the opening engine every tick). This path, driven by THIS
        # run's non-pinned actionable set, must not close them — it serves only
        # legacy / rule-based-opened trades (no LLM engine).
        open_provider = _provider_of_synth_model(trade.get("llm_synthesis_model"))
        if settings.enable_llm_hold_review and open_provider in ("anthropic", "deepseek"):
            continue

        # Reuse the most recent intraday mark when available; otherwise
        # capture a fresh sample now and timestamp it.
        if trade.get("current_price"):
            exit_price       = trade["current_price"]
            exit_decision_at = trade.get("current_price_datetime") or _now_iso()
        else:
            exit_decision_at = _now_iso()
            exit_price       = _fetch_price(trade["ticker"])
        exit_executed_at = _execution_iso()  # realistic fill time (now in RTH/extended-trade)
        if not exit_price or exit_price <= 0:
            logger.warning(f"[tracker] Cannot close {trade['ticker']} on reversal — no/non-positive price")
            continue

        exit_session = _session_of_iso(exit_executed_at)
        ret = _pct_return(trade["action"], trade["entry_price"], exit_price, trade.get("type", "STOCK"),
                          entry_session=trade.get("entry_session"), exit_session=exit_session)
        mul = trade.get("position_size_multiplier", 1.0)
        exit_ref = _reference_close(trade["ticker"])
        trade["status"]                 = "CLOSED"
        trade["exit_date"]              = today
        trade["exit_datetime"]          = exit_executed_at
        trade["exit_session"]           = exit_session
        trade["exit_decision_datetime"] = exit_decision_at
        trade["exit_price"]             = float(exit_price)
        trade["exit_ref_close"]         = exit_ref["close"] if exit_ref else None
        trade["exit_ref_close_date"]    = exit_ref["date"] if exit_ref else None
        trade["return_pct"]             = round(ret, 3)
        trade["weighted_return_pct"]    = round(ret * mul, 3)
        trade["days_held"]              = _trading_days_held(trade["entry_date"])
        trade["exit_reason"]            = "signal_reversal"
        trade["exit_hold_prompt"]       = hold_prompt_active
        closed += 1
        logger.info(
            f"[tracker] Signal reversal → closed {trade['action']} {trade['ticker']} "
            f"@ {fmt_price(exit_price)} (decision={exit_decision_at}, exec={exit_executed_at})  "
            f"return={ret:+.2f}%  (new signal: {new_action})"
        )

    if closed:
        _save_trades(trades)
    return closed


# ---------------------------------------------------------------------------
# Open-position monitoring — signal decay + regime exits
# ---------------------------------------------------------------------------

def _provider_of_synth_model(model: Optional[str]) -> Optional[str]:
    """Inverse of ``_synthesis_model_for_provider``: a stored synthesis model id
    → its engine ('anthropic' | 'deepseek' | 'rule-based'). ``None`` for a
    blank/unknown value (treated as "no LLM engine" → aggregator backstop)."""
    m = (model or "").strip().lower()
    if not m:
        return None
    if "deepseek" in m:
        return "deepseek"
    if "claude" in m or "anthropic" in m:
        return "anthropic"
    if "rule" in m:
        return "rule-based"
    return None


def _confidence_floor(entry_conf) -> float:
    """The opener-review confidence a held position must clear to stay open:
    ``max(absolute backstop, relative × LLM ENTRY confidence)``. Shared by the
    exit gate (``_evaluate_decay``) and the persisted review trajectory so the
    plotted close-threshold line always matches what actually fires."""
    absf = float(settings.signal_decay_confidence_floor)
    if entry_conf is None:
        return absf
    return max(absf, float(settings.signal_decay_confidence_floor_relative) * float(entry_conf))


def _held_hours(trade: dict) -> Optional[float]:
    """Wall-clock hours a trade has been open (entry → now, ET). None if unknown."""
    raw = trade.get("entry_datetime") or trade.get("entry_date")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)
        return max(0.0, (datetime.now(ET) - dt.astimezone(ET)).total_seconds() / 3600.0)
    except (ValueError, TypeError):
        return None


def _horizon_expiry(trade: dict) -> Optional[dict]:
    """Matched-exit state once a position outlives its target-horizon window, else
    ``None`` (still inside the window, horizon synthesis / matched exit disabled, or
    no target horizon). Returns ``{"floor", "ramp"}``.

    CONTINUOUS, not a cliff: the required re-confirmation ``floor`` ramps from the
    normal base floor (``ramp`` 0, AT the window — so nothing changes the instant a
    position crosses its horizon) up to ``base × horizon_expiry_floor_mult``
    (``ramp`` 1), reaching full strength ``horizon_expiry_ramp_windows`` windows
    past expiry. A same-direction position whose conviction drops below the ramped
    floor closes; a still-conviction winner is NOT cut. Short-horizon trades tighten
    fast (the window is small), long-horizon trades stay patient — the ramp is in
    units of the position's OWN horizon."""
    if not (settings.enable_horizon_synthesis and settings.enable_horizon_matched_exit):
        return None
    # Positions without a target_horizon (opened before horizon synthesis, or a
    # run where it failed) fall back to horizon_default_window so they still have
    # a time-stop — without it a persistent-neutral loser rides forever (the
    # HOLD/WATCH-never-closes rule has no counterweight). Empty setting = legacy
    # behavior (no fallback, no time-stop).
    target_h = trade.get("target_horizon") or (settings.horizon_default_window or "").strip()
    if not target_h:
        return None
    from src.signals.edge_curve import horizon_hours
    window = horizon_hours(target_h)
    if not window:
        return None
    held = _held_hours(trade)
    if held is None or held < window:
        return None
    overage = held / window - 1.0                              # 0 at the window, grows after
    ramp_windows = max(1e-9, float(settings.horizon_expiry_ramp_windows))
    ramp = min(1.0, max(0.0, overage / ramp_windows))
    base = _confidence_floor(trade.get("confidence"))
    floor = min(0.99, base * (1.0 + (float(settings.horizon_expiry_floor_mult) - 1.0) * ramp))
    return {"floor": floor, "ramp": ramp}


def _horizon_expired_floor(trade: dict) -> Optional[float]:
    """The (ramped) matched-exit confidence floor, or ``None`` when the position is
    within its window / the matched exit is off. Thin accessor over
    ``_horizon_expiry`` (kept for callers that only need the floor value)."""
    st = _horizon_expiry(trade)
    return st["floor"] if st else None


def _evaluate_decay(
    trade: dict,
    today_signal,
    macro_regime_context,
    hold_review=None,
) -> Optional[str]:
    """Return an ``exit_reason`` when an open trade should close, else ``None``.

    Fix #2 — the gate that HOLDS/CLOSES a position uses the SAME rationale that
    OPENED it. A position opened by an LLM engine (Claude / DeepSeek) is governed
    by ``hold_review`` — its OWN opening engine's fresh re-judgment of the ticker.
    The caller (pipeline ``_build_hold_reviews``) guarantees every entry in
    ``hold_reviews`` is produced by that position's opening synthesis + sentiment
    engines (apples-to-apples), so this function just trusts a present review.
    The aggregator's combined_score / confidence (a different decision-maker,
    often near zero on LLM-conviction names — it was killing those trades one
    tick after entry) is no longer consulted whenever a review is present. As of
    the #4 hardening, the pipeline builds a review for rule-based / legacy-opened
    trades too (with THIS run's engine, since there is no opener to pin), so the
    aggregator backstop now fires ONLY as a true last resort — a rule-based /
    legacy trade with no LLM review available this tick (e.g. the LLM is wholly
    unavailable).

    When a review is present (LLM-opened → opener-pinned; rule-based → this
    run's engine), in priority order:
      1. ``macro_regime_exit``   — long while macro is PANIC/RISK_OFF (top-down
         risk overlay, engine-independent — always applies).
      2. ``llm_signal_flipped``  — the engine now calls the opposite direction.
      3. ``llm_confidence_loss`` — same-direction conviction fell below the
         entry-relative floor (max(absolute, relative × LLM ENTRY confidence));
         HOLD/WATCH counts as zero directional conviction (but does NOT close).
    No review this tick (engine failed to produce one) → hold.
    """
    action = trade.get("action")
    if action not in ("BUY", "SELL"):
        return None

    # 1. Macro regime exit (only blocks long positions, matching entry-side logic)
    if (settings.signal_decay_regime_exit
            and macro_regime_context is not None
            and action == "BUY"
            and getattr(macro_regime_context, "regime", "") in ("PANIC", "RISK_OFF")):
        return "macro_regime_exit"

    # 2. LLM-driven hold/close — a fresh LLM re-judgment governs the exit.
    #    For LLM-opened trades that review is the OPENER-PINNED one (Fix #2 — the
    #    same engine that opened it). For rule-based / legacy-opened trades the
    #    pipeline now builds a review with THIS run's engine, so their exit is
    #    LLM-judged too instead of being handed to the aggregator-decay backstop
    #    (30%-win historically). A present review governs for ALL trades.
    open_provider = _provider_of_synth_model(trade.get("llm_synthesis_model"))
    opener_is_llm = open_provider in ("anthropic", "deepseek")
    if settings.enable_llm_hold_review and hold_review is not None:
        rev_action = getattr(hold_review, "action", None)
        conv = float(getattr(hold_review, "confidence", 0.0) or 0.0)
        # Flip — the engine now actively calls the OTHER way.
        if (action == "BUY" and rev_action == "SELL") or (action == "SELL" and rev_action == "BUY"):
            return "llm_signal_flipped"
        # Horizon-matched time-stop (CONTINUOUS): once the position outlives the
        # horizon its measured edge supports, the required re-confirmation floor
        # RAMPS up with age (see _horizon_expiry) — no cliff at the boundary. A
        # same-direction re-affirmation whose conviction has faded below that ramped
        # floor closes. A neutral HOLD/WATCH is NOT cut at the boundary (that
        # contradicted the never-close-on-neutral rule below and drove premature
        # exits); it is flushed only once FULLY past the window (ramp saturated) and
        # only when horizon_expiry_flush_neutral is on. A flip already closed above.
        exp = _horizon_expiry(trade)
        if exp is not None:
            if rev_action == action:
                return "horizon_expired" if conv < exp["floor"] else None
            if settings.horizon_expiry_flush_neutral and exp["ramp"] >= 1.0:
                return "horizon_expired"        # persistent neutral, well past its window
            return None
        # Same-direction re-affirmation whose conviction collapsed below the
        # entry-relative floor. A neutral HOLD/WATCH is NOT a close: the engine
        # isn't calling the other way, and on hold-prompt-OFF runs it judges a
        # held ticker as a fresh candidate (commonly HOLD for want of a NEW
        # catalyst) — treating that as an exit would just reintroduce premature
        # closes through another door. Only an explicit reversal or a genuine
        # same-direction conviction collapse closes.
        if rev_action == action:
            if conv < _confidence_floor(trade.get("confidence")):  # entry conf = the LLM number that gated the open
                return "llm_confidence_loss"
        return None
    if settings.enable_llm_hold_review and opener_is_llm:
        # LLM-opened but NO review produced this tick (opener engine failed) →
        # HOLD. Never fall through to the aggregator backstop, which was killing
        # LLM-conviction names one tick after entry (the Fix #2 invariant).
        return None

    # 3. Aggregator backstop — rule-based / legacy-opened trades with NO LLM
    #    review available this tick (true last resort, e.g. the LLM is entirely
    #    unavailable). Original four-trigger behaviour.
    if not settings.enable_signal_decay_exits:
        return None
    if today_signal is None:
        return None

    today_combined = float(getattr(today_signal, "combined_score", 0.0))
    today_confidence = float(getattr(today_signal, "confidence", 0.0))
    direction_sign = 1 if action == "BUY" else -1
    today_oriented = today_combined * direction_sign

    # 3a. Signal flipped — today's combined crossed against the trade
    if today_oriented < settings.signal_decay_flip_threshold:
        return "signal_flipped"

    # 3b. Signal decay — needs entry baseline
    entry = trade.get("signal_at_entry") or {}
    entry_combined = entry.get("combined_score")
    if entry_combined is not None:
        entry_oriented = float(entry_combined) * direction_sign
        if entry_oriented - today_oriented > settings.signal_decay_drop_threshold:
            return "signal_decay"

    # 3c. Confidence loss — entry-relative floor with absolute backstop.
    entry_conf_raw = (entry or {}).get("confidence")
    relative_factor = float(settings.signal_decay_confidence_floor_relative)
    absolute_floor  = float(settings.signal_decay_confidence_floor)
    if entry_conf_raw is None:
        effective_floor = absolute_floor
    else:
        effective_floor = max(absolute_floor, relative_factor * float(entry_conf_raw))
    if today_confidence < effective_floor:
        return "confidence_loss"

    return None


def monitor_open_positions(
    signals_by_ticker: Optional[dict] = None,
    macro_regime_context=None,
    hold_prompt_active: Optional[bool] = None,
    hold_reviews: Optional[dict] = None,
    run_synthesis_provider: Optional[str] = None,
) -> int:
    """For every open trade, re-judge it and close on deterioration. Returns the
    number of trades closed.

    Fix #2 — an LLM-opened position is reviewed by its OWN opening engine via
    ``hold_reviews`` (ticker → today's opener-pinned synthesis ``Recommendation``,
    built fresh every tick by the pipeline). ``signals_by_ticker`` (the aggregator
    cross-section) is now only the backstop for legacy / rule-based-opened trades —
    see ``_evaluate_decay``. ``run_synthesis_provider`` is retained for logging.

    Exit reasons it can set:
      - macro_regime_exit    (PANIC/RISK_OFF while long — always applies)
      - llm_signal_flipped   (opening engine now calls the opposite direction)
      - llm_confidence_loss  (opening engine's conviction fell below entry floor)
      - signal_flipped / signal_decay / confidence_loss  (aggregator backstop,
        legacy / rule-based-opened trades only)
      - intraday_reversal    (opt-in 30-min hard reversal)

    Designed to run AFTER ``update_open_trades`` (so ``current_price`` is fresh)
    and BEFORE ``close_trades_on_signal_reversal``.
    """
    if not (settings.enable_llm_hold_review or settings.enable_signal_decay_exits
            or settings.enable_intraday_exit or settings.signal_decay_regime_exit):
        return 0

    trades = _load_trades()
    today = date.today().isoformat()
    closed_count = 0
    decision_at = _now_iso()
    executed_at = _execution_iso()

    for trade in trades:
        if trade.get("status") != "OPEN":
            continue

        today_signal = (signals_by_ticker or {}).get(trade["ticker"])
        hold_review = (hold_reviews or {}).get(trade["ticker"])
        reason = _evaluate_decay(trade, today_signal, macro_regime_context, hold_review)
        # Opt-in intraday exit: close when the 30-min trend has reversed hard
        # against the position (Hybrid model — intraday only times the exit).
        if reason is None and settings.enable_intraday_exit:
            from src.signals.intraday_timing import compute_intraday_timing, reverses_position
            timing = compute_intraday_timing(trade["ticker"])
            if reverses_position(trade["action"], timing, settings.intraday_exit_threshold):
                reason = "intraday_reversal"
                logger.info(
                    f"[monitor] intraday reversal on {trade['action']} {trade['ticker']} "
                    f"(30-min score {timing['score']:+.2f}) → closing"
                )
        if reason is None:
            continue

        # Use the live mark already on the trade (refreshed by update_open_trades).
        exit_price = trade.get("current_price")
        if not exit_price or exit_price <= 0:
            # Try a fresh fetch as a last resort.
            exit_price = _fetch_price(trade["ticker"])
            if not exit_price or exit_price <= 0:
                logger.warning(
                    f"[monitor] Cannot close {trade['ticker']} ({reason}) — no usable price"
                )
                continue

        mul = trade.get("position_size_multiplier", 1.0)
        exit_session = _session_of_iso(executed_at)
        ret = _pct_return(trade["action"], trade["entry_price"], exit_price, trade.get("type", "STOCK"),
                          entry_session=trade.get("entry_session"), exit_session=exit_session)
        exit_ref = _reference_close(trade["ticker"])

        trade["status"]                 = "CLOSED"
        trade["exit_date"]              = today
        trade["exit_datetime"]          = executed_at
        trade["exit_session"]           = exit_session
        trade["exit_decision_datetime"] = trade.get("current_price_datetime") or decision_at
        trade["exit_price"]             = float(exit_price)
        trade["exit_ref_close"]         = exit_ref["close"] if exit_ref else None
        trade["exit_ref_close_date"]    = exit_ref["date"] if exit_ref else None
        trade["return_pct"]             = round(ret, 3)
        trade["weighted_return_pct"]    = round(ret * mul, 3)
        trade["days_held"]              = _trading_days_held(trade["entry_date"])
        trade["exit_reason"]            = reason
        trade["exit_hold_prompt"]       = hold_prompt_active
        closed_count += 1

        # Logging — branch by which decision-maker fired so the line is legible.
        regime = getattr(macro_regime_context, "regime", "") if macro_regime_context else ""
        if reason in ("llm_signal_flipped", "llm_confidence_loss"):
            # LLM exit: entry conf is the LLM number on the trade; today is the
            # opening engine's fresh re-judgment of the ticker (hold_review).
            entry_conf = trade.get("confidence")
            rev_action = getattr(hold_review, "action", None)
            rev_conf = getattr(hold_review, "confidence", None)
            floor = _confidence_floor(entry_conf)
            open_prov = _provider_of_synth_model(trade.get("llm_synthesis_model"))
            logger.info(
                f"[monitor] {reason} → closed {trade['action']} {trade['ticker']} "
                f"@ {fmt_price(exit_price)}  return={ret:+.2f}%  "
                f"opener_engine={open_prov} (run={run_synthesis_provider})  "
                f"entry_conf={entry_conf}  today_review={rev_action}@{rev_conf}  "
                f"conf_floor={floor:.2f}  regime={regime}"
            )
        else:
            # Aggregator backstop / regime: entry vs today combined + conf floor.
            entry = trade.get("signal_at_entry") or {}
            ec = entry.get("combined_score")
            ef_conf = entry.get("confidence")
            tc = float(getattr(today_signal, "combined_score", 0.0)) if today_signal else None
            cf = float(getattr(today_signal, "confidence", 0.0)) if today_signal else None
            if ef_conf is None:
                eff_floor = float(settings.signal_decay_confidence_floor)
                floor_kind = "abs"
            else:
                rel_floor = float(settings.signal_decay_confidence_floor_relative) * float(ef_conf)
                abs_floor = float(settings.signal_decay_confidence_floor)
                eff_floor = max(abs_floor, rel_floor)
                floor_kind = "rel" if rel_floor >= abs_floor else "abs"
            logger.info(
                f"[monitor] {reason} → closed {trade['action']} {trade['ticker']} "
                f"@ {fmt_price(exit_price)}  return={ret:+.2f}%  "
                f"entry_combined={ec}  today_combined={tc}  "
                f"entry_conf={ef_conf}  today_conf={cf}  "
                f"conf_floor={eff_floor:.2f}({floor_kind})  regime={regime}"
            )

    if closed_count:
        _save_trades(trades)
    return closed_count


def _normalize_closed_returns(trades: List[dict]) -> int:
    """Re-derive ``return_pct`` for every closed trade from its stored
    entry/exit prices using the current spread model.

    Idempotent and deterministic — given the same prices and spread model,
    the output never changes. Run on every pipeline tick so summary stats
    (avg/best/worst/win_rate over stored ``return_pct``) always reconcile
    with the per-trade compound produced by the daily-NAV engine.

    Returns the number of trades whose ``return_pct`` changed.
    """
    changed = 0
    for t in trades:
        if t.get("status") != "CLOSED":
            continue
        e = t.get("entry_price")
        x = t.get("exit_price")
        if e is None or x is None:
            continue
        ret = round(_pct_return(t.get("action", "BUY"), float(e), float(x), t.get("type", "STOCK"),
                                entry_session=t.get("entry_session"),
                                exit_session=t.get("exit_session")), 3)
        if abs(ret - t.get("return_pct", 0.0)) > 1e-3:
            t["return_pct"] = ret
            mul = t.get("position_size_multiplier", 1.0)
            t["weighted_return_pct"] = round(ret * mul, 3)
            changed += 1
    return changed


def _refresh_open_trade_ohlcv(trades: List[dict]) -> None:
    """Force-refresh the OHLCV cache for every open-trade ticker.

    The daily-NAV engine walks one mark per trading day held.  When the
    OHLCV cache for a ticker is stale (last bar < yesterday), days inside
    the holding period get lumped onto the next available mark — still
    deterministic, but less granular.  Refreshing here guarantees we have a
    real close for every day the position was actually held.
    """
    if not settings.enable_fetch_data:
        return
    tickers = sorted({t["ticker"] for t in trades if t.get("status") == "OPEN"})
    if not tickers:
        return
    try:
        from src.data.market_data import get_history
        from src.data.cache import load_ohlcv
    except Exception as e:
        logger.warning(f"[tracker] OHLCV refresh skipped — import failed: {e}")
        return

    yesterday = date.today() - timedelta(days=1)
    refreshed = 0
    for tk in tickers:
        cached = load_ohlcv(tk)
        needs = cached is None or cached.empty or cached.index[-1].date() < yesterday
        if not needs:
            continue
        try:
            get_history(tk, period="3mo", force_refresh=True)
            refreshed += 1
        except Exception as e:
            logger.debug(f"[tracker] OHLCV refresh failed for {tk}: {e}")
    if refreshed:
        logger.info(f"[tracker] Refreshed OHLCV cache for {refreshed} open-trade ticker(s)")


def update_open_trades() -> None:
    """Refresh current prices and unrealised P&L for all open trades.

    Positions are never closed here on a time basis — exits are thesis-driven
    (``monitor_open_positions`` for signal decay / regime, and
    ``close_trades_on_signal_reversal`` for an opposite-direction signal).
    """
    trades = _load_trades()
    today = date.today().isoformat()
    updated = 0
    # The live M2M is "what if you closed right now" — so the hypothetical
    # exit leg bears the CURRENT session's spread (an extended-hours mark
    # charges the wider extended book it would actually cross).
    from src.performance.market_calendar import current_session
    mark_session = current_session()

    # Keep OHLCV current for every open-trade ticker so the daily-NAV walk
    # has a real close for each day the position was held (no synthetic
    # lumping over a stale-cache gap).
    _refresh_open_trade_ohlcv(trades)

    # Refresh stored return_pct on closed trades so summary stats always
    # match the current spread model and the daily-NAV engine's compound.
    normalized = _normalize_closed_returns(trades)
    if normalized:
        logger.info(f"[tracker] Normalized return_pct on {normalized} closed trade(s) to current spread model")

    for trade in trades:
        if trade["status"] != "OPEN":
            continue

        # Timestamp every live mark.  decision_at = actual fetch instant
        # (audit truth); executed_at = next realistic fill time (used as
        # exit_datetime if this refresh triggers auto-close).
        decision_at = _now_iso()
        price = _fetch_price(trade["ticker"])
        if price is None:
            continue
        if price <= 0:
            logger.warning(f"[tracker] Skipping mark for {trade['ticker']} — non-positive price {fmt_price(price)}")
            continue

        days = _trading_days_held(trade["entry_date"])
        ret = _pct_return(trade["action"], trade["entry_price"], price, trade.get("type", "STOCK"),
                          entry_session=trade.get("entry_session"), exit_session=mark_session)

        mul = trade.get("position_size_multiplier", 1.0)
        trade["current_price"] = float(price)
        trade["current_price_datetime"] = decision_at
        trade["return_pct"] = round(ret, 3)
        trade["weighted_return_pct"] = round(ret * mul, 3)
        trade["days_held"] = days
        updated += 1

        # ── MFE / MAE update (pure observability) ────────────────────────
        # Legacy trades may lack these fields — initialise to current return
        # so the high-water/low-water marks start from where we are now
        # and only ratchet in the favourable / unfavourable direction from
        # this tick forward. New trades come in at 0.0 so this is a no-op
        # on the entry tick.
        prev_mfe = trade.get("max_favorable_excursion")
        prev_mae = trade.get("max_adverse_excursion")
        if prev_mfe is None:
            prev_mfe = ret
            trade["mfe_date"] = today
        if prev_mae is None:
            prev_mae = ret
            trade["mae_date"] = today
        if ret > prev_mfe:
            trade["max_favorable_excursion"] = round(ret, 3)
            trade["mfe_date"] = today
        else:
            trade["max_favorable_excursion"] = round(prev_mfe, 3)
        if ret < prev_mae:
            trade["max_adverse_excursion"] = round(ret, 3)
            trade["mae_date"] = today
        else:
            trade["max_adverse_excursion"] = round(prev_mae, 3)

        # No time-based exit — a position is held until its thesis deteriorates
        # (monitor_open_positions) or an opposite signal appears
        # (close_trades_on_signal_reversal). days_held above is kept for display.

    _save_trades(trades)
    logger.info(f"[tracker] Updated {updated} open trade(s)")

    # Feed the live pattern registry with any newly-closed trades that had a
    # pattern_at_entry. Idempotent — already-registered trades are skipped.
    try:
        from src.signals.pattern_registry import record_batch as _pattern_record_batch
        _pattern_record_batch(trades)
    except Exception as e:
        logger.debug(f"[tracker] pattern_registry update skipped: {e}")


def log_performance_summary() -> None:
    """Log a full performance breakdown to the log file."""
    trades = _load_trades()   # sanitized — malformed rows can't reach this point
    if not trades:
        logger.info("[tracker] No trades recorded yet.")
        return

    open_trades = [t for t in trades if t["status"] == "OPEN"]
    closed_trades = [t for t in trades if t["status"] == "CLOSED"]

    logger.info("=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)

    # Open positions
    if open_trades:
        logger.info(f"  Open positions ({len(open_trades)})")
        for t in sorted(open_trades, key=lambda x: x["entry_date"], reverse=True):
            mul = t.get("position_size_multiplier", 1.0)
            logger.info(
                f"    {t['action']:<4} {t['ticker']:<6} | "
                f"date={t['entry_date']}  entry={fmt_price(t['entry_price'])}  now={fmt_price(t['current_price'])}  "
                f"P&L={t['return_pct']:+.2f}%  size={mul}×  ({t['days_held']}d held)"
            )

    # Closed positions
    if closed_trades:
        returns = [t["return_pct"] for t in closed_trades]
        multipliers = [t.get("position_size_multiplier", 1.0) for t in closed_trades]
        wins = [r for r in returns if r > 0]
        win_rate = len(wins) / len(returns) * 100
        avg_return = sum(returns) / len(returns)
        total_mul = sum(multipliers)
        weighted_avg = sum(r * m for r, m in zip(returns, multipliers)) / total_mul if total_mul else avg_return
        best = max(returns)
        worst = min(returns)

        logger.info(f"  Closed positions ({len(closed_trades)})")
        for t in sorted(closed_trades, key=lambda x: x["exit_date"] or "", reverse=True):
            mul = t.get("position_size_multiplier", 1.0)
            logger.info(
                f"    {t['action']:<4} {t['ticker']:<6} | "
                f"date={t['entry_date']}  entry={fmt_price(t['entry_price'])}  exit={fmt_price(t['exit_price'])}  "
                f"return={t['return_pct']:+.2f}%  size={mul}×  ({t['days_held']}d)"
            )

        logger.info(f"  --- Stats ({len(closed_trades)} closed trades) ---")
        logger.info(f"    Win rate:         {win_rate:.1f}%")
        logger.info(f"    Avg return:       {avg_return:+.2f}%")
        logger.info(f"    Weighted avg ret: {weighted_avg:+.2f}%  (size-adjusted)")
        logger.info(f"    Best trade:       {best:+.2f}%")
        logger.info(f"    Worst trade:      {worst:+.2f}%")

        # Per-method attribution
        attributed = [t for t in closed_trades if t.get("methods_agreeing")]
        method_stats = _compute_method_stats(attributed)
        if method_stats:
            logger.info("  --- Method Attribution ---")
            for method, ms in sorted(method_stats.items(), key=lambda x: -x[1]["win_rate"]):
                logger.info(
                    f"    {method:<10} trades={ms['trades']:>2}  "
                    f"win={ms['win_rate']:>5.1f}%  avg={ms['avg_return']:>+6.2f}%"
                )

        cat_stats = _compute_category_stats(attributed)
        if cat_stats:
            logger.info("  --- Category Attribution ---")
            for cat, cs in sorted(cat_stats.items(), key=lambda x: -x[1]["win_rate"]):
                logger.info(
                    f"    {cat:<15} trades={cs['trades']:>2}  "
                    f"win={cs['win_rate']:>5.1f}%  avg={cs['avg_return']:>+6.2f}%"
                )

        conv_stats = _compute_convergence_stats(attributed)
        if conv_stats:
            logger.info("  --- Convergence ---")
            for label, cs in conv_stats.items():
                logger.info(
                    f"    {label:<15} trades={cs['trades']:>2}  "
                    f"win={cs['win_rate']:>5.1f}%  avg={cs['avg_return']:>+6.2f}%"
                )

    logger.info("=" * 60)


def get_open_trade_tickers() -> List[str]:
    """Return tickers of all currently open trades so the pipeline always fetches their prices."""
    return [t["ticker"] for t in _load_trades() if t["status"] == "OPEN"]


def _build_trades_svg(closed_trades: List[dict]) -> str:
    """
    Return an inline SVG string visualising closed trades over time.

    Top panel  — equity curve: compound cumulative return as a filled line chart.
    Bottom panel — per-trade return bars coloured green (win) / red (loss).
    No external dependencies; safe to embed directly inside HTML email.
    """
    import math

    trades  = sorted(closed_trades, key=lambda t: t.get("exit_date") or "")
    n       = len(trades)
    returns = [t["return_pct"]           for t in trades]
    tickers = [t["ticker"]               for t in trades]
    dates   = [(t.get("exit_date") or "")[-5:] for t in trades]   # "MM-DD"
    actions = [t.get("action", "BUY")    for t in trades]

    # Compound cumulative return at each exit
    cum: List[float] = []
    compound = 1.0
    for r in returns:
        compound *= (1 + r / 100)
        cum.append(round((compound - 1) * 100, 2))

    # ── Layout ────────────────────────────────────────────────────────────────
    W               = 820          # total SVG width
    PL, PR          = 58, 18       # left / right padding (room for y-axis labels)
    PT              = 28           # top padding
    CHART_W         = W - PL - PR

    TOP_H           = 160          # equity curve panel
    SEP             = 22           # gap between panels
    BOT_H           = 80           # per-trade bars panel
    BOT_Y           = PT + TOP_H + SEP
    XLABEL_H        = 46           # space for date + ticker labels
    H               = BOT_Y + BOT_H + XLABEL_H

    # ── Coordinate helpers ────────────────────────────────────────────────────
    def xp(i: int) -> float:
        return PL + (i * CHART_W / (n - 1) if n > 1 else CHART_W / 2)

    # Top panel y-scale (equity curve)
    all_y   = [0.0] + cum
    ylo, yhi = min(all_y), max(all_y)
    y_span  = (yhi - ylo) or 1.0
    ypad    = y_span * 0.15

    def yp_top(v: float) -> float:
        lo = ylo - ypad
        hi = yhi + ypad
        return PT + TOP_H * (1.0 - (v - lo) / (hi - lo))

    zero_y = yp_top(0.0)

    # Bottom panel y-scale (bars centred at zero)
    ret_max = max(abs(r) for r in returns) or 1.0
    bar_scale = (BOT_H / 2) / ret_max * 0.88
    center_y  = BOT_Y + BOT_H / 2
    bar_w     = max(4.0, min(24.0, CHART_W / n * 0.55))

    # ── Y-axis ticks for top panel ────────────────────────────────────────────
    raw_step  = y_span / 4
    magnitude = 10 ** math.floor(math.log10(abs(raw_step) or 1))
    nice_step = round(raw_step / magnitude) * magnitude or magnitude
    tick_start = math.ceil((ylo - ypad) / nice_step) * nice_step
    ticks: List[float] = []
    v = tick_start
    while v <= yhi + ypad + nice_step * 0.5:
        ticks.append(round(v, 6))
        v += nice_step

    # ── SVG assembly ─────────────────────────────────────────────────────────
    parts: List[str] = []
    a = parts.append

    a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
      f'viewBox="0 0 {W} {H}" '
      f'style="background:#0f172a;border-radius:8px;display:block;max-width:100%;margin:14px 0;">')

    # Gradient for equity-curve fill
    a('<defs>'
      '<linearGradient id="ecg" x1="0" y1="0" x2="0" y2="1">'
      '<stop offset="0%" stop-color="#60a5fa" stop-opacity="0.30"/>'
      '<stop offset="100%" stop-color="#60a5fa" stop-opacity="0.02"/>'
      '</linearGradient>'
      '</defs>')

    # Panel labels
    a(f'<text x="{PL}" y="{PT-8}" fill="#64748b" font-size="10" font-family="Arial,sans-serif">'
      'Cumulative Return (%)</text>')
    a(f'<text x="{PL}" y="{BOT_Y-7}" fill="#64748b" font-size="10" font-family="Arial,sans-serif">'
      'Per-trade Return (%)</text>')

    # ── Top panel grid ────────────────────────────────────────────────────────
    for tv in ticks:
        ty = yp_top(tv)
        if PT - 1 <= ty <= PT + TOP_H + 1:
            is_zero = abs(tv) < nice_step * 0.05
            col  = "#475569" if is_zero else "#1e293b"
            dash = "" if is_zero else ' stroke-dasharray="3,3"'
            a(f'<line x1="{PL}" y1="{ty:.1f}" x2="{PL+CHART_W}" y2="{ty:.1f}" '
              f'stroke="{col}" stroke-width="1"{dash}/>')
            a(f'<text x="{PL-4}" y="{ty+4:.1f}" fill="#64748b" font-size="9" '
              f'font-family="Arial,sans-serif" text-anchor="end">{tv:+.0f}%</text>')

    # ── Bottom panel zero line ────────────────────────────────────────────────
    a(f'<line x1="{PL}" y1="{center_y:.1f}" x2="{PL+CHART_W}" y2="{center_y:.1f}" '
      'stroke="#334155" stroke-width="1"/>')

    # ── Equity-curve fill area ────────────────────────────────────────────────
    pts_top = [(xp(i), yp_top(cum[i])) for i in range(n)]
    path_d  = f"M {xp(0):.1f} {yp_top(0.0):.1f} " + \
              " ".join(f"L {x:.1f} {y:.1f}" for x, y in pts_top)
    fill_d  = path_d + f" L {xp(n-1):.1f} {zero_y:.1f} L {xp(0):.1f} {zero_y:.1f} Z"
    a(f'<path d="{fill_d}" fill="url(#ecg)"/>')

    # ── Equity-curve line ─────────────────────────────────────────────────────
    a(f'<path d="{path_d}" fill="none" stroke="#60a5fa" stroke-width="2.5" stroke-linejoin="round"/>')

    # Zero dashed line (equity panel)
    a(f'<line x1="{PL}" y1="{zero_y:.1f}" x2="{PL+CHART_W}" y2="{zero_y:.1f}" '
      'stroke="#475569" stroke-width="1.2" stroke-dasharray="5,3"/>')

    # ── Dots on equity curve ──────────────────────────────────────────────────
    for i, (px, py) in enumerate(pts_top):
        dot_col = "#4ade80" if cum[i] >= 0 else "#f87171"
        a(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" '
          f'fill="{dot_col}" stroke="#0f172a" stroke-width="1.2">'
          f'<title>{tickers[i]} exit {dates[i]}: {cum[i]:+.1f}% cumulative</title>'
          f'</circle>')

    # End-value label
    ex, ey  = pts_top[-1]
    ec      = "#4ade80" if cum[-1] >= 0 else "#f87171"
    label_x = min(ex + 5, PL + CHART_W - 35)
    a(f'<text x="{label_x:.1f}" y="{ey - 5:.1f}" fill="{ec}" font-size="10" '
      f'font-weight="bold" font-family="Arial,sans-serif">{cum[-1]:+.1f}%</text>')

    # ── Per-trade bars ────────────────────────────────────────────────────────
    for i, (r, ticker, action) in enumerate(zip(returns, tickers, actions)):
        bc    = "#22c55e" if r >= 0 else "#ef4444"
        bh    = abs(r) * bar_scale
        bx    = xp(i) - bar_w / 2
        by    = center_y - bh if r >= 0 else center_y
        a(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{max(bh, 1.5):.1f}" '
          f'fill="{bc}" opacity="0.9" rx="1">'
          f'<title>{ticker} {r:+.1f}%</title></rect>')

    # ── X-axis labels (skip if too many) ─────────────────────────────────────
    step = max(1, math.ceil(n / 18))
    for i in range(0, n, step):
        lx = xp(i)
        a(f'<text x="{lx:.1f}" y="{BOT_Y+BOT_H+14}" fill="#64748b" font-size="9" '
          f'font-family="Arial,sans-serif" text-anchor="middle">{dates[i]}</text>')
        a(f'<text x="{lx:.1f}" y="{BOT_Y+BOT_H+27}" fill="#94a3b8" font-size="9" '
          f'font-family="Arial,sans-serif" text-anchor="middle">{tickers[i]}</text>')

    a('</svg>')
    return '\n'.join(parts)


def _build_timeline_svg(all_trades: list) -> str:
    """Gantt-style timeline of all trades (closed + open).

    One row per trade, sorted chronologically.  Bars span entry_date → exit_date
    (or today for open positions).  Color: green=profit, red=loss, blue=open.
    Left margin: ticker + action label.  Right of bar: return % annotation.
    Hover tooltip: entry/exit dates, prices, and return.
    """
    if not all_trades:
        return ""

    today = date.today()
    MAX_ROWS = 40

    rows = []
    for t in all_trades:
        entry_str = t.get("entry_date", "")
        if not entry_str:
            continue
        try:
            entry_dt = date.fromisoformat(entry_str)
        except Exception:
            continue
        exit_str = t.get("exit_date", "") or ""
        try:
            exit_dt = date.fromisoformat(exit_str) if exit_str else today
        except Exception:
            exit_dt = today
        rows.append({
            "ticker":    t.get("ticker", "?"),
            "action":    t.get("action", "BUY"),
            "entry_dt":  entry_dt,
            "exit_dt":   exit_dt,
            "entry_str": entry_str,
            "exit_str":  exit_str,
            "status":    t.get("status", "CLOSED"),
            "ret":       t.get("return_pct", 0.0),
            "entry_px":  t.get("entry_price") or 0.0,
            "exit_px":   t.get("exit_price") or t.get("current_price") or 0.0,
        })

    if not rows:
        return ""

    rows.sort(key=lambda r: (r["entry_str"], r["ticker"]))
    truncated = len(rows) > MAX_ROWS
    if truncated:
        rows = rows[-MAX_ROWS:]

    min_dt = min(r["entry_dt"] for r in rows)
    max_dt = max(r["exit_dt"] for r in rows)
    # +1 so a same-day entry/exit gets a non-zero bar width
    span_days = max(1, (max_dt - min_dt).days + 1)

    W       = 820
    LABEL_W = 100
    RET_W   = 60
    CHART_W = W - LABEL_W - RET_W
    ROW_H   = 20
    ROW_GAP = 3
    PT      = 34   # top padding for axis date labels
    PB      = 24   # bottom padding for legend
    N       = len(rows)
    H       = PT + N * (ROW_H + ROW_GAP) + PB

    def x_dt(d: date) -> float:
        return LABEL_W + (d - min_dt).days / span_days * CHART_W

    parts: List[str] = []
    a = parts.append

    a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
      f'viewBox="0 0 {W} {H}" '
      f'style="background:#0f172a;border-radius:8px;display:block;max-width:100%;margin:14px 0;">')

    suffix = "  (latest 40)" if truncated else ""
    a(f'<text x="{LABEL_W}" y="15" fill="#64748b" font-size="10" '
      f'font-family="Arial,sans-serif">Trade Timeline{suffix}</text>')

    # X-axis date ticks + vertical grid lines
    chart_bot = PT + N * (ROW_H + ROW_GAP)
    n_ticks = min(8, max(2, span_days // 5 + 1))
    for i in range(n_ticks):
        frac = i / (n_ticks - 1) if n_ticks > 1 else 0.5
        tick_day = round(frac * (span_days - 1))
        d = min_dt + timedelta(days=tick_day)
        tx = x_dt(d)
        a(f'<line x1="{tx:.1f}" y1="{PT - 4}" x2="{tx:.1f}" y2="{chart_bot}" '
          f'stroke="#1e293b" stroke-width="1"/>')
        a(f'<text x="{tx:.1f}" y="{PT - 7}" fill="#475569" font-size="9" '
          f'font-family="Arial,sans-serif" text-anchor="middle">{d.strftime("%m/%d")}</text>')

    # Today dashed marker
    if min_dt <= today <= max_dt + timedelta(days=1):
        tx = x_dt(today)
        a(f'<line x1="{tx:.1f}" y1="{PT - 4}" x2="{tx:.1f}" y2="{chart_bot}" '
          f'stroke="#334155" stroke-width="1" stroke-dasharray="4,3"/>')

    # Trade rows
    for i, row in enumerate(rows):
        y  = PT + i * (ROW_H + ROW_GAP)
        cy = y + ROW_H // 2 + 4  # text baseline centred in row

        if i % 2 == 1:
            a(f'<rect x="0" y="{y}" width="{W}" height="{ROW_H}" fill="#111827" opacity="0.4"/>')

        # Ticker label (right-aligned to x=62)
        a(f'<text x="{LABEL_W - 38}" y="{cy}" fill="#cbd5e1" font-size="10" '
          f'font-family="Arial,sans-serif" text-anchor="end" font-weight="600">'
          f'{row["ticker"]}</text>')
        # Action label (right-aligned to x=97)
        act_col = "#60a5fa" if row["action"] == "BUY" else "#fb923c"
        a(f'<text x="{LABEL_W - 3}" y="{cy}" fill="{act_col}" font-size="8" '
          f'font-family="Arial,sans-serif" text-anchor="end">{row["action"]}</text>')

        # Bar coordinates — extend by one full day-width so exit day is included
        bx = x_dt(row["entry_dt"])
        ex = min(x_dt(row["exit_dt"]) + CHART_W / span_days, LABEL_W + CHART_W)
        bw = max(2.0, ex - bx)
        bar_y = y + 4
        bar_h = ROW_H - 8

        if row["status"] == "OPEN":
            fill, stroke = "#1e3a5f", "#60a5fa"
        elif row["ret"] > 0:
            fill, stroke = "#14532d", "#22c55e"
        else:
            fill, stroke = "#7f1d1d", "#ef4444"

        # Tooltip
        ep_s = f"${fmt_price(row['entry_px'])}" if row["entry_px"] else "n/a"
        xp_s = f"${fmt_price(row['exit_px'])}"  if row["exit_px"]  else "n/a"
        exit_label = row["exit_str"] if row["exit_str"] else "open"
        tip = (f"{row['ticker']} {row['action']} | "
               f"{row['entry_str']} → {exit_label} | "
               f"Entry {ep_s}  Exit {xp_s} | "
               f"{row['ret']:+.2f}%")
        if row["status"] == "OPEN":
            tip += " [OPEN]"

        a(f'<rect x="{bx:.1f}" y="{bar_y}" width="{bw:.1f}" height="{bar_h}" '
          f'fill="{fill}" stroke="{stroke}" stroke-width="0.8" rx="2">'
          f'<title>{tip}</title></rect>')

        # Entry price inside bar (only when bar is wide enough to fit the label)
        if bw > 52 and row["entry_px"]:
            a(f'<text x="{bx + 4:.1f}" y="{cy}" fill="{stroke}" font-size="8" '
              f'font-family="Arial,sans-serif" opacity="0.9">'
              f'${fmt_price(row["entry_px"])}</text>')

        # Return annotation to the right of the bar
        ann_x = min(ex + 4, W - RET_W + 4)
        ret_col = "#4ade80" if row["ret"] > 0 else "#f87171" if row["ret"] < 0 else "#94a3b8"
        sfx = "*" if row["status"] == "OPEN" else ""
        a(f'<text x="{ann_x:.1f}" y="{cy}" fill="{ret_col}" font-size="9" '
          f'font-family="Arial,sans-serif" font-weight="700">{row["ret"]:+.1f}%{sfx}</text>')

    # Legend
    leg_y = H - 8
    for off, col, lbl in [(0, "#22c55e", "Profit"), (58, "#ef4444", "Loss"), (110, "#60a5fa", "Open*")]:
        lx = LABEL_W + off
        a(f'<rect x="{lx}" y="{leg_y - 8}" width="10" height="8" fill="{col}" rx="2" opacity="0.7"/>')
        a(f'<text x="{lx + 13}" y="{leg_y}" fill="#64748b" font-size="9" '
          f'font-family="Arial,sans-serif">{lbl}</text>')

    a('</svg>')
    return '\n'.join(parts)


def compute_portfolio_metrics(closed_trades: list, open_trades: list) -> dict:
    """True time-weighted compound portfolio return across all trades (closed + open).

    Uses ``daily_nav.compute_compound_return``: every per-day return is derived
    from the real OHLCV close for that ticker on that day, capital-weighted
    across whatever positions are active. No geometric interpolation, no
    proxy assumptions about intermediate prices — same trades.json + same
    OHLCV cache → identical numbers every run.

    Closed trades: walk uses entry_price → daily closes → exit_price (each
        bracket-adjusted by the dynamic bid-ask spread on entry/exit only).
    Open trades:   walk uses entry_price → daily closes → today's
        current_price (also spread-adjusted, treating the live mark as a
        hypothetical close-out so the per-trade compound matches the trade's
        stored ``return_pct``).

    Time windows include only trades ENTERED within the last N calendar days.
    Inception includes every trade ever recorded.

    Returns dict: {
        compound_inception — compound over all trades ever,
        return_1w          — compound for trades entered in last 7 days,
        return_2w          — compound for trades entered in last 14 days,
        return_1m          — compound for trades entered in last 30 days,
    }  Values are percentages.  None when no trades exist for that window.
    """
    today = date.today()
    all_trades = [t for t in (closed_trades + open_trades) if t.get("entry_date")]
    if not all_trades:
        return {}

    def _window(days: int) -> Optional[float]:
        cutoff = str(today - timedelta(days=days))
        return _compute_nav_compound([t for t in all_trades if t["entry_date"] >= cutoff])

    return {
        "compound_inception": _compute_nav_compound(all_trades),
        "return_1w":          _window(7),
        "return_2w":          _window(14),
        "return_1m":          _window(30),
    }


def _flip_trade(trade: dict) -> dict:
    """Return a shallow-copied trade with action/direction inverted.

    Used to model the "what if this method had signalled the opposite
    direction" hypothetical. Prices, dates, and ticker are preserved so the
    daily NAV engine walks the real OHLCV closes — the only thing that
    changes is whether each day's price move is counted as long-favourable or
    short-favourable. The trade's ``return_pct`` is also re-derived from
    ``_pct_return`` for the flipped action so summary stats line up with the
    flipped compound.
    """
    flipped = dict(trade)
    action = trade.get("action", "BUY")
    flipped["action"]    = "SELL" if action == "BUY" else "BUY"
    flipped["direction"] = "BEARISH" if trade.get("direction") == "BULLISH" else "BULLISH"

    entry_px   = trade.get("entry_price")
    asset_type = trade.get("type", "STOCK")
    e_sess     = trade.get("entry_session")
    if entry_px is not None:
        if flipped["status"] == "CLOSED" and trade.get("exit_price") is not None:
            flipped["return_pct"] = round(
                _pct_return(flipped["action"], float(entry_px), float(trade["exit_price"]), asset_type,
                            entry_session=e_sess, exit_session=trade.get("exit_session")),
                3,
            )
        else:
            cp = trade.get("current_price") or entry_px
            flipped["return_pct"] = round(
                _pct_return(flipped["action"], float(entry_px), float(cp), asset_type,
                            entry_session=e_sess),
                3,
            )
    return flipped


def _hypothetical_trades_for_method(method: str, closed: List[dict]) -> List[dict]:
    """Build the per-method hypothetical trade list (real or flipped).

    For each closed trade with stored method_scores: if the method's view
    matches the actual action, include the trade verbatim; if it disagrees,
    include the flipped trade. Trades where the method has no view
    (|score| < threshold) are skipped entirely.
    """
    out: List[dict] = []
    for trade in closed:
        score = trade.get("method_scores", {}).get(method, 0.0)
        # Skip methods with no view: either exactly zero (degenerate "no opinion")
        # or below the configured floor. The exact-zero guard is required because
        # _METHOD_AGREE_THRESHOLD = 0 would otherwise pass score == 0.0 through
        # to the BUY/SELL solo classifier and mis-tag it as SELL.
        if score == 0.0 or abs(score) < _METHOD_AGREE_THRESHOLD:
            continue
        actual = trade.get("action", "BUY")
        solo   = "BUY" if score > 0 else "SELL"
        out.append(trade if solo == actual else _flip_trade(trade))
    return out


def _solo_stats(trades: list) -> dict:
    """Summary stats from a list of (real or flipped) trade dicts.

    Compound return walks the actual OHLCV closes via the daily-NAV engine;
    every other stat is computed from the trades' ``return_pct`` so it
    reconciles exactly with the compound on a single-trade slice.
    """
    if not trades:
        return {}
    hyp = [t["return_pct"] for t in trades]
    wins = [r for r in hyp if r > 0]
    return {
        "trades":          len(trades),
        "win_rate":        round(len(wins) / len(hyp) * 100, 1),
        "avg_return":      round(sum(hyp) / len(hyp), 2),
        "compound_return": _compute_nav_compound(trades) or 0.0,
        "best":            round(max(hyp), 2),
        "worst":           round(min(hyp), 2),
    }


_EVAL_BANDS = [
    ("Low (0.10–0.35)",    0.10, 0.35),
    ("Medium (0.35–0.65)", 0.35, 0.65),
    ("High (0.65+)",       0.65, 1.01),
]


def _eval_stats(entries: list) -> dict:
    """Directional accuracy stats from (trade_dict, abs_score) pairs.

    The trade dict is either the real trade (when the method agreed with the
    actual action) or a flipped copy (when it disagreed) — see
    ``_hypothetical_trades_for_method``. Compound returns are walked through
    the actual OHLCV cache via the daily-NAV engine, never interpolated.
    """
    if not entries:
        return {}
    correct = [t["return_pct"] for t, _ in entries if t["return_pct"] > 0]
    wrong   = [t["return_pct"] for t, _ in entries if t["return_pct"] <= 0]
    conviction_bands = []
    for label, lo, hi in _EVAL_BANDS:
        band = [t for t, s in entries if lo <= s < hi]
        if not band:
            conviction_bands.append({"label": label, "trades": 0, "accuracy": 0.0, "avg_return": 0.0, "compound_return": 0.0})
            continue
        band_correct = [t["return_pct"] for t in band if t["return_pct"] > 0]
        conviction_bands.append({
            "label":           label,
            "trades":          len(band),
            "accuracy":        round(len(band_correct) / len(band) * 100, 1),
            "avg_return":      round(sum(t["return_pct"] for t in band) / len(band), 2),
            "compound_return": _compute_nav_compound(band) or 0.0,
        })
    return {
        "trades":               len(entries),
        "directional_accuracy": round(len(correct) / len(entries) * 100, 1),
        "avg_return_correct":   round(sum(correct) / len(correct), 2) if correct else 0.0,
        "avg_return_wrong":     round(sum(wrong)   / len(wrong),   2) if wrong   else 0.0,
        "compound_return":      _compute_nav_compound([t for t, _ in entries]) or 0.0,
        "conviction_bands":     conviction_bands,
    }


def compute_solo_method_performance(split: Optional[str] = None, window_days: Optional[int] = None,
                                    session: Optional[str] = None,
                                    direction: Optional[str] = None) -> dict:
    """Simulate performance for each signal method used in isolation, split by direction.

    For each closed trade with stored method_scores, each method is asked:
    "What direction would you alone have signalled?"

      score > 0  → solo BUY;  score < 0  → solo SELL;  |score| < threshold → skip

    Hypothetical return:
      same direction as actual trade  → actual return_pct
      opposite direction              → −actual return_pct

    When ``split`` is ``"train"`` or ``"holdout"``, only trades assigned to
    that bucket via the deterministic ``_trade_split`` hash are included.
    Pass ``split=None`` for the in-sample union (legacy behaviour).

    Returns dict: method_name → {
        "overall": {trades, win_rate, avg_return, compound_return, best, worst},
        "buys":    {same fields, only BUY-signal trades}   — {} if none,
        "sells":   {same fields, only SELL-signal trades}  — {} if none,
    }
    Only methods with ≥ 1 qualifying trade are returned.
    """
    MIN_TRADES = 1
    trades = _load_trades()
    closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("method_scores")]
    closed = _filter_by_split(closed, split)
    if window_days is not None:
        _cutoff = (date.today() - timedelta(days=window_days)).isoformat()
        closed = [t for t in closed if (t.get("entry_date") or "") >= _cutoff]
    if session:
        closed = [t for t in closed if _session_matches(t, session)]
    if direction:
        closed = [t for t in closed if _match_direction(t, direction)]
    if not closed:
        return {}

    results: dict = {}
    for method in _ALL_METHODS:
        hyp = _hypothetical_trades_for_method(method, closed)
        if len(hyp) < MIN_TRADES:
            continue
        buys  = [t for t in hyp if t.get("action") == "BUY"]
        sells = [t for t in hyp if t.get("action") == "SELL"]
        results[method] = {
            "overall": _solo_stats(hyp),
            "buys":    _solo_stats(buys),
            "sells":   _solo_stats(sells),
        }

    return results


def compute_method_eval_stats(split: Optional[str] = None) -> dict:
    """Per-method directional accuracy and conviction calibration, split by direction.

    For each signal method, across all closed trades with stored method_scores:
      - Directional accuracy: % of times the method's direction led to a positive hyp return
      - avg_return when correct vs wrong
      - Conviction bands (Low/Medium/High |score|): trades, accuracy, avg_return

    Results are split into "overall", "buys" (score > 0), and "sells" (score < 0).

    When ``split`` is ``"train"`` or ``"holdout"``, only trades assigned to
    that bucket via the deterministic ``_trade_split`` hash are included.

    Returns dict: method_name → {
        "overall": {trades, directional_accuracy, avg_return_correct, avg_return_wrong,
                    conviction_bands: [{"label","trades","accuracy","avg_return"}, ...]},
        "buys":    {same structure, only BUY-signal entries}  — {} if none,
        "sells":   {same structure, only SELL-signal entries} — {} if none,
    }
    Only methods with ≥ 1 qualifying trade are returned (others shown as no-data in email).
    """
    MIN_TRADES = 1
    trades = _load_trades()
    closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("method_scores")]
    closed = _filter_by_split(closed, split)
    if not closed:
        return {}

    results: dict = {}
    for method in _ALL_METHODS:
        all_entries: List[tuple] = []
        for trade in closed:
            score = trade["method_scores"].get(method, 0.0)
            # Same exact-zero guard as _hypothetical_trades_for_method —
            # required when _METHOD_AGREE_THRESHOLD = 0 so a literal zero
            # (no view) doesn't get mis-tagged as a SELL signal below.
            if score == 0.0 or abs(score) < _METHOD_AGREE_THRESHOLD:
                continue
            actual_action = trade.get("action", "BUY")
            solo_action   = "BUY" if score > 0 else "SELL"
            hyp_trade     = trade if solo_action == actual_action else _flip_trade(trade)
            all_entries.append((hyp_trade, abs(score)))

        if len(all_entries) < MIN_TRADES:
            continue

        buy_entries  = [(t, s) for t, s in all_entries if t.get("action") == "BUY"]
        sell_entries = [(t, s) for t, s in all_entries if t.get("action") == "SELL"]

        results[method] = {
            "overall": _eval_stats(all_entries),
            "buys":    _eval_stats(buy_entries),
            "sells":   _eval_stats(sell_entries),
        }

    return results


def compute_oos_comparison() -> dict:
    """Side-by-side per-method comparison: train (in-sample) vs holdout (OOS).

    Returns a payload the email template can render directly:

        {
          "enabled":      True/False,
          "holdout_pct":  int,
          "split_counts": {"train": int, "holdout": int},
          "rows":         [
            {
              "method":         str,
              "train_trades":   int,
              "train_acc":      float,   # %  — directional accuracy on train
              "holdout_trades": int,
              "holdout_acc":    float,   # %  — directional accuracy on holdout
              "delta":          float,   # holdout − train (pp); large negative = overfit
            }, …
          ],
        }

    Honest evaluation rule: a method whose holdout accuracy is materially
    below its train accuracy is overfit to the sample — its weight should be
    discounted (the adaptive-weight machinery already trains on the train
    slice, so this table is the report-card on whether that decision was
    right).

    Returns ``{"enabled": False}`` when the feature is off so the email
    template can hide the section.
    """
    if not settings.enable_oos_validation:
        return {"enabled": False}

    trades = _load_trades()
    closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("method_scores")]
    if not closed:
        return {
            "enabled":      True,
            "holdout_pct":  max(0, min(50, int(settings.oos_holdout_pct or 0))),
            "split_counts": {"train": 0, "holdout": 0},
            "rows":         [],
        }

    train_eval   = compute_method_eval_stats(split="train")
    holdout_eval = compute_method_eval_stats(split="holdout")

    rows: List[dict] = []
    for method in _ALL_METHODS:
        tr = train_eval.get(method, {}).get("overall", {})
        ho = holdout_eval.get(method, {}).get("overall", {})
        tr_n = int(tr.get("trades", 0) or 0)
        ho_n = int(ho.get("trades", 0) or 0)
        if tr_n == 0 and ho_n == 0:
            continue
        tr_acc = float(tr.get("directional_accuracy", 0.0) or 0.0)
        ho_acc = float(ho.get("directional_accuracy", 0.0) or 0.0)
        # delta is meaningful only when both sides have any trades
        delta = round(ho_acc - tr_acc, 1) if (tr_n > 0 and ho_n > 0) else None
        rows.append({
            "method":         method,
            "train_trades":   tr_n,
            "train_acc":      round(tr_acc, 1),
            "holdout_trades": ho_n,
            "holdout_acc":    round(ho_acc, 1),
            "delta":          delta,
        })

    # Sort by holdout accuracy desc (methods passing OOS surface first), then
    # by train trades to break ties consistently.
    rows.sort(key=lambda r: (-(r["holdout_acc"] if r["holdout_trades"] else -1),
                             -r["train_trades"]))

    train_n   = sum(1 for t in closed if _trade_split(t) == "train")
    holdout_n = sum(1 for t in closed if _trade_split(t) == "holdout")

    return {
        "enabled":      True,
        "holdout_pct":  max(0, min(50, int(settings.oos_holdout_pct or 0))),
        "split_counts": {"train": train_n, "holdout": holdout_n},
        "rows":         rows,
    }


def _session_of_iso(raw) -> str:
    """US-market session of an ISO timestamp: ``rth | extended | overnight``.

    RTH 09:30–16:00 · extended 04:00–09:30 + 16:00–20:00 · overnight 20:00–04:00.
    Date-only/missing values (no time component) default to 'rth' — matches
    every legacy record, which could only have traded in the regular session.
    """
    if not raw or ("T" not in str(raw) and ":" not in str(raw)):
        return "rth"
    try:
        dt = datetime.fromisoformat(str(raw))
        dt = dt.astimezone(ET) if dt.tzinfo is not None else dt.replace(tzinfo=ET)
        mins = dt.hour * 60 + dt.minute
    except Exception:
        return "rth"
    if 9 * 60 + 30 <= mins < 16 * 60:
        return "rth"
    if (4 * 60 <= mins < 9 * 60 + 30) or (16 * 60 <= mins < 20 * 60):
        return "extended"
    return "overnight"


def _trade_session(trade: dict) -> str:
    """Session a trade was ENTERED in, from its ET entry timestamp (the
    stored ``entry_session`` stamp and this derivation always agree — both
    come from the same executed-at instant)."""
    return _session_of_iso(trade.get("entry_datetime"))


def _session_of_iso_fine(raw) -> str:
    """Finer session split: ``rth | premarket | afterhours | overnight``.

    Splits the coarse ``extended`` session into its two halves — pre-market
    (04:00–09:30 ET) and after-hours (16:00–20:00 ET). Used ONLY for the
    dashboard's session filter/display; the stored ``entry_session`` /
    ``exit_session`` stamps stay coarse because the cost/spread/size
    multipliers key on ``extended``. Date-only/missing values default to
    'rth' (matches every legacy record)."""
    if not raw or ("T" not in str(raw) and ":" not in str(raw)):
        return "rth"
    try:
        dt = datetime.fromisoformat(str(raw))
        dt = dt.astimezone(ET) if dt.tzinfo is not None else dt.replace(tzinfo=ET)
        mins = dt.hour * 60 + dt.minute
    except Exception:
        return "rth"
    if 9 * 60 + 30 <= mins < 16 * 60:
        return "rth"
    if 4 * 60 <= mins < 9 * 60 + 30:
        return "premarket"
    if 16 * 60 <= mins < 20 * 60:
        return "afterhours"
    return "overnight"


def _trade_session_fine(trade: dict) -> str:
    """Fine session (rth|premarket|afterhours|overnight) a trade was ENTERED in."""
    return _session_of_iso_fine(trade.get("entry_datetime"))


def _generated_after_rth_close(raw) -> bool:
    """True when an ISO timestamp falls at/after 16:00 ET on its own date — i.e.
    that day's RTH close was already printed when the call was made, so the
    same-day close is a legitimate (non-look-ahead) anchor for it. Unparseable
    or date-only values return False (conservative → prior-day close)."""
    if not raw or ("T" not in str(raw) and ":" not in str(raw)):
        return False
    try:
        dt = datetime.fromisoformat(str(raw))
        dt = dt.astimezone(ET) if dt.tzinfo is not None else dt.replace(tzinfo=ET)
    except Exception:
        return False
    return (dt.hour * 60 + dt.minute) >= 16 * 60


def _session_matches(trade: dict, session: Optional[str]) -> bool:
    """Session-filter predicate understanding BOTH the coarse stamp values
    (``rth|extended|overnight``) and the finer dashboard values
    (``premarket|afterhours``). ``None`` → always matches (no filter)."""
    if not session:
        return True
    if session in ("premarket", "afterhours"):
        return _trade_session_fine(trade) == session
    return _trade_session(trade) == session


def calibrate_sim_costs(trades: Optional[List[dict]] = None) -> Optional[float]:
    """Install the real-fill cost calibration (see settings.sim_use_real_fill_costs).

    Measures the average all-in one-way cost from actual **LMT** IBKR fills
    (``repo.fetch_filled_lmt_legs`` — MKT fills are excluded, since LMT is what
    the system uses going forward) and sets it as the flat per-leg cost the
    whole sim then charges (``spread._one_side_cost``) — so the simulated
    returns reflect what LMT execution really costs, not the modeled estimate.
    Clears the override (→ model) when disabled or when fewer than
    ``sim_real_fill_costs_min_legs`` LMT legs exist. Returns the applied
    fraction (None = model). Idempotent; call before any cost-dependent step
    (record/normalize/M2M/NAV) so every figure in a run uses one cost basis.
    The ``trades`` arg is accepted for call-site symmetry but unused — the cost
    basis comes from the broker fill log, not the ledger.
    """
    if not settings.sim_use_real_fill_costs:
        set_real_cost_override(None)
        return None
    from src.performance.broker_view import real_one_way_cost_fraction
    legs = repo.fetch_filled_lmt_legs()
    frac = real_one_way_cost_fraction(legs, settings.sim_real_fill_costs_min_legs)
    set_real_cost_override(frac)
    if frac is not None:
        logger.info(
            f"[costs] sim calibrated to real LMT fills ({len(legs)} legs): "
            f"{frac * 100:.3f}% one-way per leg"
        )
    return frac


def _displayed_one_way_cost_pct(trades: List[dict], real_frac: Optional[float]) -> Optional[float]:
    """The per-leg one-way cost % shown in the dashboard's simulated Returns
    tile: the calibrated real cost when active, else the modeled half-spread +
    commission averaged over these trades' legs (entry for all, exit for
    closed)."""
    if real_frac is not None:
        return round(real_frac * 100.0, 4)
    costs: List[float] = []
    for t in trades:
        atype = t.get("type", "STOCK")
        ep = t.get("entry_price")
        if ep and ep > 0:
            costs.append(_one_side_cost(float(ep), atype, t.get("entry_session")) * 100.0)
        if t.get("status") == "CLOSED":
            xp = t.get("exit_price")
            if xp and xp > 0:
                costs.append(_one_side_cost(float(xp), atype, t.get("exit_session")) * 100.0)
    return round(sum(costs) / len(costs), 4) if costs else None


def get_performance_for_email(window_days: Optional[int] = None,
                              session: Optional[str] = None,
                              direction: Optional[str] = None) -> dict:
    """Return structured performance data for inclusion in the email report.

    When ``window_days`` is set, only trades ENTERED within the last N calendar
    days are included (1w = 7, 1m = 30); ``None`` = inception (every trade ever).
    ``session`` additionally restricts to trades entered in that US-market
    session — the coarse ``rth | extended | overnight`` or the finer
    ``premarket | afterhours`` (the two halves of ``extended``); None = all
    sessions. The dashboard's window + session toggles pass these so its
    metrics/plots recompute to match.
    """
    trades = _load_trades()
    # Calibrate the sim cost to real IBKR fills (no-op when disabled / too few
    # fills) BEFORE the NAV walk below, and from the FULL ledger (not the
    # windowed/session slice) so the cost basis is stable across toggles.
    real_cost_frac = calibrate_sim_costs(trades)
    if window_days is not None:
        _cutoff = (date.today() - timedelta(days=window_days)).isoformat()
        trades = [t for t in trades if (t.get("entry_date") or "") >= _cutoff]
    if session:
        trades = [t for t in trades if _session_matches(t, session)]
    if direction:
        trades = [t for t in trades if _match_direction(t, direction)]
    open_trades   = [t for t in trades if t["status"] == "OPEN"]
    closed_trades = [t for t in trades if t["status"] == "CLOSED"]
    # Open trades carry a current M2M return_pct (updated each run by update_open_trades()).
    # Treat them as hypothetical exits so every live position is reflected in the stats.
    all_trades = closed_trades + open_trades

    # Inception date — earliest entry across ALL trades (open + closed)
    all_dates = [t["entry_date"] for t in trades if t.get("entry_date")]
    first_trade_date = min(all_dates) if all_dates else None
    inception_days: Optional[int] = None
    if first_trade_date:
        try:
            inception_days = (date.today() - date.fromisoformat(first_trade_date)).days
        except Exception:
            pass

    # Base stats always populated when any trades exist
    stats: dict = {}
    if trades:
        stats = {
            "total_closed":     len(closed_trades),
            "total_open":       len(open_trades),
            "total_all":        len(trades),
            "first_trade_date": first_trade_date,
            "inception_days":   inception_days,
        }

    # Headline metrics over ALL trades (closed final P&L + open M2M return_pct).
    if all_trades:
        returns = [t["return_pct"] for t in all_trades]
        multipliers = [t.get("position_size_multiplier", 1.0) for t in all_trades]
        total_mul = sum(multipliers)
        weighted_avg = (
            sum(r * m for r, m in zip(returns, multipliers)) / total_mul
            if total_mul else sum(returns) / len(returns)
        )
        # Compound return: trades sorted by entry date, sequentially re-invested.
        # Will be overwritten below by portfolio_metrics["compound_inception"] which
        # uses the more accurate daily-batch model, but compute it as a fallback.
        sorted_returns = [
            t["return_pct"]
            for t in sorted(all_trades, key=lambda x: x.get("entry_date", ""))
        ]
        compound = 1.0
        for r in sorted_returns:
            compound *= (1 + r / 100)

        stats.update({
            "win_rate":            round(len([r for r in returns if r > 0]) / len(returns) * 100, 1),
            "avg_return":          round(sum(returns) / len(returns), 2),
            "median_return":       round(median(returns), 2),
            "weighted_avg_return": round(weighted_avg, 2),
            "compound_return":     round((compound - 1) * 100, 2),
            "best":                round(max(returns), 2),
            "worst":               round(min(returns), 2),
        })

    attributed = [t for t in all_trades if t.get("methods_agreeing")]
    confidence_ranked    = _compute_confidence_ranked(closed_trades)          # realized P&L only
    performance_table    = _compute_performance_table(all_trades) if all_trades else []
    trades_svg           = _build_trades_svg(closed_trades) if len(closed_trades) >= 2 else ""
    timeline_svg         = _build_timeline_svg(all_trades) if all_trades else ""
    solo_method_perf     = compute_solo_method_performance(window_days=window_days, session=session, direction=direction)
    llm_perf             = _compute_llm_perf(window_days=window_days, session=session, direction=direction)
    macro_eval           = compute_macro_eval(window_days=window_days, session=session, direction=direction)
    method_eval_stats    = compute_method_eval_stats()
    oos_comparison       = compute_oos_comparison()
    portfolio_metrics    = compute_portfolio_metrics(closed_trades, open_trades)

    # Keep stats["compound_return"] in sync with the authoritative portfolio_metrics value
    # (both use _compute_nav_compound but portfolio_metrics also includes live M2M for open trades).
    if portfolio_metrics and "compound_inception" in portfolio_metrics:
        stats["compound_return"] = portfolio_metrics["compound_inception"]

    # Methods ordered by overall win rate descending; no-data methods go to the bottom.
    method_order_by_winrate = sorted(
        list(_ALL_METHODS),
        key=lambda m: solo_method_perf.get(m, {}).get("overall", {}).get("win_rate", -1),
        reverse=True,
    )

    return {
        "open_trades":              sorted(open_trades,   key=lambda x: x["entry_date"],    reverse=True),
        "closed_trades":            sorted(closed_trades, key=lambda x: x["exit_date"] or "", reverse=True),
        "stats":                    stats,
        "portfolio_metrics":        portfolio_metrics,          # session-based compound + time windows
        "confidence_ranked":        confidence_ranked,
        "performance_table":        performance_table,
        "attributed_count":         len(attributed),
        "trades_svg":               trades_svg,
        "timeline_svg":             timeline_svg,
        "solo_method_perf":         solo_method_perf,          # hypothetical per-method solo simulation
        "llm_perf":                 llm_perf,                  # actual trades grouped by LLM engine (synthesis / sentiment)
        "macro_eval":               macro_eval,                # aggregated decision layers (synthesis vs aggregator vs bundles)
        "sim_one_way_cost_pct":     _displayed_one_way_cost_pct(all_trades, real_cost_frac),  # per-leg 1-way cost shown in Returns
        "sim_cost_is_real":         real_cost_frac is not None,   # True = calibrated to real IBKR fills, False = modeled
        "hold_prompt_eval":         compute_hold_prompt_eval(trades),  # held-positions prompt A/B (exit outcomes ON vs OFF)
        "method_eval_stats":        method_eval_stats,         # per-method accuracy + conviction calibration
        "oos_comparison":           oos_comparison,            # train vs holdout accuracy (honest OOS)
        "method_labels":            METHOD_LABELS,
        "method_order_by_winrate":  method_order_by_winrate,   # methods ranked by solo win rate, no-data last
    }
