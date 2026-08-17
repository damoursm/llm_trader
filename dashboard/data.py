"""Read-only data access for the dashboard.

The dashboard must never take the DuckDB write lock, so the repo is put in
read-only mode and every read is wrapped in a short retry (the pipeline holds
the write lock only momentarily during its end-of-run persistence). The heavy
performance computation is cached briefly so tab switches stay responsive.
"""

from __future__ import annotations

import time
from typing import Optional

import pandas as pd
from loguru import logger

from src.db import repo

# The dashboard is a read-only consumer of the database.
repo.set_read_only(True)

_RETRY_ATTEMPTS = 6
_RETRY_BASE_DELAY = 0.4  # exponential backoff: 0.4, 0.8, 1.6, 3.2, 5.0, … (~11s total)
_RETRY_MAX_DELAY = 5.0


def _retry(fn, what: str = ""):
    last = None
    delay = _RETRY_BASE_DELAY
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return fn()
        except FileNotFoundError:
            raise
        except Exception as e:  # most likely a transient DuckDB lock during a run
            last = e
            if attempt < _RETRY_ATTEMPTS - 1:
                time.sleep(delay)
                delay = min(delay * 2, _RETRY_MAX_DELAY)
    logger.warning(f"[dashboard] read failed ({what}): {last}")
    raise last


# Horizons the simulated ENTRY/EXIT tables report. The three intraday ones
# (30m/3h/6h) are deliberately absent: computing them forces a read of the whole
# 30-min OHLCV cache (~2,700 files / ~190 MB), which measured ~28 s of the warm
# sweep across the three panels — the most expensive columns on the page by a
# wide margin. Dropping them here is a REPORTING choice only; the live horizon
# synthesis (`edge_curve`) still sees the full curve, which matters because `6h`
# is the most-used `target_horizon` in the ledger. Add a label back and it
# reappears in every simulated table (`app._SIM_HORIZONS` is derived from this).
PANEL_HORIZONS = ("1d", "3d", "1w", "2w", "1m")

_REC_COLS = (
    "generated_at, ticker, type, direction, action, confidence, time_horizon, "
    "target_horizon, horizon_net_edge_pct, "
    "shadow_target_horizon, shadow_direction, shadow_horizon_net_edge_pct, "
    "expected_move_pct, market_aligned, upside_score, "
    "actionable, dominant_method, llm_provider, rationale"
)


# The columns the run list actually consumes (dropdown + models-used table).
# ``SELECT *`` here dragged the gate_diag JSON blob of every run (1,200+) into
# every page build; the full row is still available per run via run_row().
_RUN_LIST_COLS = ("run_id, started_at, market_mode, macro_regime, "
                  "llm_synthesis_provider, llm_sentiment_provider")


def runs_df() -> pd.DataFrame:
    return _retry(lambda: repo.fetch_df(
        f"SELECT {_RUN_LIST_COLS} FROM runs ORDER BY started_at DESC"), "runs")


def latest_run_id() -> Optional[str]:
    df = _retry(lambda: repo.fetch_df(
        "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"), "latest_run")
    return None if df.empty else str(df.iloc[0]["run_id"])


def latest_run_info() -> Optional[dict]:
    """One cheap row for the header chip: when the last pipeline run happened and
    under which regime/mode. None when there are no runs (or on a read error —
    the header must never take the page down)."""
    try:
        df = _retry(lambda: repo.fetch_df(
            f"SELECT {_RUN_LIST_COLS} FROM runs ORDER BY started_at DESC LIMIT 1"),
            "latest_run_info")
    except Exception:
        return None
    return None if df.empty else df.iloc[0].to_dict()


def run_row(run_id: str):
    df = _retry(lambda: repo.fetch_df("SELECT * FROM runs WHERE run_id = ?", [run_id]), "run_row")
    return None if df.empty else df.iloc[0]


def recommendations_df(run_id: Optional[str] = None) -> pd.DataFrame:
    if run_id:
        return _retry(lambda: repo.fetch_df(
            f"SELECT {_REC_COLS} FROM recommendations WHERE run_id = ? "
            f"ORDER BY actionable DESC, confidence DESC", [run_id]), "recs")
    return _retry(lambda: repo.fetch_df(
        f"SELECT {_REC_COLS} FROM recommendations ORDER BY generated_at DESC LIMIT 200"), "recs")


def run_sources_df(run_id: str) -> pd.DataFrame:
    return _retry(lambda: repo.fetch_df(
        "SELECT source_label, ok, duration_s, error FROM run_sources "
        "WHERE run_id = ? ORDER BY ok, source_label", [run_id]), "sources")


def latest_run_failures() -> list:
    """Failed data sources (ok = false) for the most recent run — powers the
    dashboard health banner. Returns ``[]`` when there are none or no runs."""
    rid = latest_run_id()
    if not rid:
        return []
    df = run_sources_df(rid)
    if df.empty or "ok" not in df.columns:
        return []
    failed = df[~df["ok"].astype(bool)]
    return failed.to_dict("records")


# The heavy accessors (NAV walks, per-method solo/IC/shadow OHLCV joins) are memoised
# and, crucially, invalidated by DATA VERSION rather than a short timer: the underlying
# data only changes when a new pipeline run persists, so we key each cache entry on the
# latest run_id and recompute ONLY when that changes. Between runs every tab switch /
# revisit is instant instead of re-triggering the joins every 60 s. _PERF_TTL is just a
# safety cap (recompute at least this often even if version detection ever misses).
_perf_cache: dict = {}          # key -> {"ts": float, "data": Any, "ver": str|None}
_PERF_TTL = 1800.0
_DATA_VER_TTL = 15.0
_data_ver: dict = {"ts": 0.0, "val": None}


def _data_version() -> Optional[str]:
    """The latest run_id — the cache's data version. Cheap (LIMIT 1) and itself
    re-checked at most every _DATA_VER_TTL s. Returns the last-known value on a
    transient read error so a momentary write-lock never forces a recompute storm."""
    now = time.time()
    if (now - _data_ver["ts"]) < _DATA_VER_TTL:
        return _data_ver["val"]
    try:
        df = repo.fetch_df("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1")
        val = None if df is None or df.empty else str(df.iloc[0]["run_id"])
    except Exception:
        val = _data_ver["val"]
    _data_ver.update(ts=now, val=val)
    return val


def _cached(key, producer, force: bool = False):
    """Version-aware memo: serve the cached value until a NEW pipeline run lands
    (data version changed) or the safety TTL lapses; otherwise recompute via
    ``producer``. Shared by every heavy accessor.

    **Serving STALE while a warm is in flight is the load-bearing part** (added
    2026-08-14). Recomputing here costs ~400 s of pandas on the REQUEST thread and
    holds the GIL, which is precisely what made the dashboard unreachable — a
    static 4 KB file timed out from localhost. Moving the sweep to a child process
    only helps if the parent stops recomputing too; otherwise the first visitor
    after every run simply re-does the whole sweep in the web process. So while
    the background warmer is rebuilding, the previous snapshot is served: a few
    minutes stale beats unreachable, and the swap is atomic per accessor.

    The ``_WARM_STALE_GRACE`` cap keeps a permanently broken warmer from serving
    ancient numbers forever — past it, correctness wins and we pay the recompute.

    **A KNOWN data version makes the entry fresh, full stop** (2026-08-15,
    corrected 2026-08-16). The version (latest ``run_id``) is the real freshness
    signal: same version ⇒ same database ⇒ the cached value is not merely
    tolerable, it is exactly what a recompute would produce. So a version match
    refreshes the entry's timestamp and serves it — there is nothing to redo.

    The first attempt at this let the TTL lapse and asked the warmer to re-sweep
    instead. That inverted the quiet-day case it was written for: over a weekend
    the version never changes, so every ~30 min of TTL plus one page visit
    launched another ~530 s child sweep that recomputed byte-identical numbers
    (observed 5×, one run_id, ~965 CPU-seconds). Recomputing known-identical
    data is waste whichever process pays for it.

    The TTL therefore only bites when the version is UNKNOWN (``None`` — the
    run_id query failed), which is the case it was always meant to cover: no
    freshness signal, so fall back to time.

    Order matters: the version check comes FIRST, and a version MISMATCH must
    never fall through to the TTL — a new run has to invalidate immediately, or
    fresh numbers would sit behind a 30-minute timer.
    """
    now = time.time()
    ver = _data_version()
    entry = _perf_cache.get(key)
    if not force and entry is not None:
        if ver is not None and entry.get("ver") == ver:
            entry["ts"] = now        # same DB ⇒ a recompute changes nothing
            return entry["data"]
        age = now - entry["ts"]
        # A new run landed (or the version is unreadable): serve the previous
        # snapshot only while the warmer is actually rebuilding it.
        if _warm_state.get("in_flight") and age < _WARM_STALE_GRACE:
            return entry["data"]
        if ver is None and age < _PERF_TTL:
            return entry["data"]
    data = producer()
    _perf_cache[key] = {"ts": now, "data": data, "ver": ver}
    return data


def performance(window_days: Optional[int] = None, session: Optional[str] = None,
                direction: Optional[str] = None, asset_type: Optional[str] = None,
                force: bool = False) -> dict:
    """Windowed + session + direction + type-filtered performance bundle.
    ``window_days`` = 7 / 30 (None = inception); ``session`` = rth / extended /
    overnight (None = all); ``direction`` = long / short (None = both);
    ``asset_type`` = stock / etf / commodity (None = all)."""
    from src.performance.tracker import get_performance_for_email
    key = ("all" if window_days is None else int(window_days), session or "all",
           direction or "all", asset_type or "all")
    return _cached(key, lambda: _retry(
        lambda: get_performance_for_email(window_days=window_days, session=session,
                                          direction=direction, asset_type=asset_type),
        "performance"), force=force)


def filled_lmt_legs() -> list:
    """Real LMT fills (ENTRY/EXIT, no drift flattens) from broker_orders —
    the basis for the IBKR one-way cost tile and the sim-cost calibration.
    Read-only + retry, like every dashboard accessor."""
    return _retry(lambda: repo.fetch_filled_lmt_legs(), "lmt_legs")


def trade_reviews_df(ticker: str) -> pd.DataFrame:
    """The opener-pinned hold-review trajectory for one ticker (fix #2), ordered
    by time. Empty DataFrame when there's no history yet or the table predates
    this feature (a read-only dashboard can't create it — the next pipeline run
    will)."""
    def _q():
        return repo.fetch_df(
            "SELECT * FROM trade_reviews WHERE ticker = ? ORDER BY reviewed_at", [ticker])
    try:
        return _retry(_q, "trade_reviews")
    except Exception:
        return pd.DataFrame()


def trades_for_ticker(ticker: str) -> list:
    """All ledger trade dicts for one ticker (entry/exit markers for the review
    timeline). Read-only via repo.load_trades()."""
    try:
        trades = _retry(lambda: repo.load_trades(), "trades_for_ticker")
    except Exception:
        return []
    return [t for t in trades if t.get("ticker") == ticker]


def broker_trades(force: bool = False) -> list:
    """The IBKR-fills projection of the ledger (real executions, real
    commissions — see ``src.performance.broker_view``), cached briefly.
    Reads through ``repo.load_trades()`` so the read-only mode set above
    applies; never touches the tracker's write paths."""
    from src.performance.broker_view import build_broker_trades
    return _cached("broker_trades",
                   lambda: build_broker_trades(_retry(lambda: repo.load_trades(), "broker_trades")),
                   force=force)


def broker_account_equity_usd() -> Optional[float]:
    """Latest IBKR account NAV (NetLiquidation) in USD from the most recent
    ``broker_reconciles`` row, or None when there's no reconcile / no equity yet.
    A non-USD account (e.g. CAD) is converted via live FX (fail-soft). Used by the
    IBKR Returns view for the account-relative return %."""
    def _q():
        df = repo.fetch_df(
            "SELECT account_equity, account_currency FROM broker_reconciles "
            "WHERE account_equity IS NOT NULL AND account_equity > 0 "
            "ORDER BY created_at DESC LIMIT 1")
        if df is None or df.empty:
            return None
        eq = float(df.iloc[0]["account_equity"])
        ccy = str(df.iloc[0]["account_currency"] or "USD").upper()
        if ccy == "USD":
            return round(eq, 2)
        from src.broker.fx import usd_per_unit
        return round(eq * usd_per_unit(ccy), 2)
    try:
        return _retry(_q, "broker_account_equity")
    except Exception:
        return None


def broker_account_pnl() -> Optional[dict]:
    """Latest IBKR account P&L snapshot (``reqPnL``) from ``broker_reconciles``,
    converted to USD: ``{"daily", "unrealized", "realized"}``. Ground truth (all
    fees / FX / dividends), account-level — ``unrealized`` is the current open P&L;
    ``daily``/``realized`` are TODAY's. None when no reconcile has captured P&L yet."""
    def _q():
        df = repo.fetch_df(
            "SELECT pnl_daily, pnl_unrealized, pnl_realized, account_currency "
            "FROM broker_reconciles WHERE (pnl_daily IS NOT NULL OR pnl_unrealized "
            "IS NOT NULL OR pnl_realized IS NOT NULL) ORDER BY created_at DESC LIMIT 1")
        if df is None or df.empty:
            return None
        row = df.iloc[0]
        ccy = str(row["account_currency"] or "USD").upper()
        rate = 1.0
        if ccy != "USD":
            from src.broker.fx import usd_per_unit
            rate = usd_per_unit(ccy)

        def conv(v):
            return round(float(v) * rate, 2) if v is not None and pd.notna(v) else None

        out = {"daily": conv(row["pnl_daily"]), "unrealized": conv(row["pnl_unrealized"]),
               "realized": conv(row["pnl_realized"])}
        return out if any(v is not None for v in out.values()) else None
    try:
        return _retry(_q, "broker_account_pnl")
    except Exception:
        return None


# ── Diagnostics accessors (IC · calibration · exit quality · execution) ──────

def signal_ic(days: Optional[int] = None, horizons=(1, 5, 10), min_n: int = 10) -> dict:
    """Per-method information-coefficient tables over the persisted signals panel
    joined with forward returns. Cached (the OHLCV join is heavy — done ONCE; the
    three compute_ic passes over it are cheap). Returns ``{panel_rows, tickers,
    ic, ic_buy, ic_sell}`` — ``ic`` over every call, ``ic_buy``/``ic_sell``
    restricted to each method's bullish / bearish calls (2026-07-22: the BUY-vs-
    SELL forensics found method skill is heavily side-dependent, so each side is
    evaluated on its own)."""
    from src.analysis.signal_panel import build_panel, compute_ic

    def _q():
        panel = build_panel(horizons=horizons, days=days)
        empty = panel is None or panel.empty
        return {
            "panel_rows": 0 if empty else int(len(panel)),
            "tickers":    0 if empty else int(panel["ticker"].nunique()),
            "ic":      pd.DataFrame() if empty else compute_ic(panel, horizons=horizons, min_n=min_n),
            "ic_buy":  pd.DataFrame() if empty else compute_ic(panel, horizons=horizons,
                                                               min_n=min_n, side="buy"),
            "ic_sell": pd.DataFrame() if empty else compute_ic(panel, horizons=horizons,
                                                               min_n=min_n, side="sell"),
        }
    key = ("signal_ic", days, tuple(horizons), int(min_n))
    return _cached(key, lambda: _retry(_q, "signal_ic"))


def ticker_perf(days: Optional[int] = None, horizons=(1, 5, 10),
                source: Optional[str] = None, min_days: int = 1) -> pd.DataFrame:
    """Per-TICKER simulated performance + decision funnel over the signals panel.

    Gate-independent by construction: every scored ticker-day counts, so a name
    the gates never let through is still measured. Cached — shares the memoised
    panel with the other analyses, so this is a groupby, not a rebuild."""
    from src.analysis.ticker_performance import compute_ticker_perf
    return _cached(("ticker_perf", days, tuple(horizons), source or "all", int(min_days)),
                   lambda: _retry(lambda: compute_ticker_perf(
                       days=days, horizons=horizons, source=source,
                       min_days=min_days), "ticker_perf"))


def market_relative_skill():
    """``({method: {...}}, baseline_pct)`` — market-relative method skill and the
    MEASURED bar it must beat. Cached; empty dict + fallback baseline on error so
    the table degrades to absolute-only rather than showing wrong numbers."""
    from src.analysis.market_relative import market_relative_skill as _mrs, market_relative_baseline
    def _q():
        return (_mrs(None) or {}), float(market_relative_baseline())
    try:
        return _cached(("market_relative_skill",), lambda: _retry(_q, "market_relative_skill"))
    except Exception:
        return {}, 48.6


def arm_eval(days: Optional[int] = None, horizons=("pv", 1, 5, 10)) -> dict:
    """Synthesis prompt-arm bake-off over the ``arm_recommendations`` panel.

    Every arm's call on every ticker each tick (one live, the rest shadow), so
    the arms are comparable PER TICKER-DAY rather than over whichever runs their
    coin came up on. Cached — the forward-return join reads the OHLCV cache once
    per ticker. Returns ``{summary, pairs, horizons, calls, shadow}``; empty
    until shadow arms have run."""
    from src.analysis.arm_eval import evaluate
    return _cached(("arm_eval", days, tuple(horizons)),
                   lambda: _retry(lambda: evaluate(days=days, horizons=horizons),
                                  "arm_eval"))


def simulated_method_perf(days: Optional[int] = None, min_n: int = 10,
                          session: Optional[str] = None,
                          direction: Optional[str] = None) -> pd.DataFrame:
    """Per-method directional win rate + mean gross return at the pivot basis +
    ``PANEL_HORIZONS`` (1d/3d/1w/2w/1m) over the ``simulated_trades`` table (every
    scored ticker treated as a solo single-method trade). Cached (the OHLCV join
    is heavy). ``session`` filters by the session the signal was GENERATED in;
    ``direction`` by the side of the method's own call (positive score = its long
    call). Empty until forward returns exist."""
    from src.analysis.simulated_trades import compute_method_perf
    return _cached(("sim_method_perf", days, int(min_n), session or "all", direction or "all"),
                   lambda: _retry(lambda: compute_method_perf(days=days, min_n=min_n,
                                                              session=session, direction=direction,
                                                              horizons=PANEL_HORIZONS),
                                  "simulated_method_perf"))


def exit_method_perf(days: Optional[int] = None, min_n: int = 10,
                     session: Optional[str] = None,
                     direction: Optional[str] = None) -> pd.DataFrame:
    """Per-EXIT-method win rate / IC / IC-std / ICIR / signed return at the pivot
    basis + ``PANEL_HORIZONS`` over the ``exit_signals`` panel (every held position
    re-scored each tick), plus the synthesized ``llm_review`` row from
    ``trade_reviews``. The exit-side counterpart to ``simulated_method_perf``.
    ``session`` filters by the session the REVIEW happened in; ``direction`` by
    the held position's side. Cached (the OHLCV join is heavy). Empty until
    forward returns exist."""
    from src.analysis.exit_panel import compute_exit_method_perf
    return _cached(("exit_method_perf", days, int(min_n), session or "all", direction or "all"),
                   lambda: _retry(lambda: compute_exit_method_perf(days=days, min_n=min_n,
                                                                   session=session, direction=direction,
                                                                   horizons=PANEL_HORIZONS),
                                  "exit_method_perf"))


def shadow_exit_method_perf(days: Optional[int] = None, min_n: int = 10,
                            session: Optional[str] = None,
                            direction: Optional[str] = None) -> pd.DataFrame:
    """Simulated exit-method performance over ALL scored tickers — every ticker in
    the signals panel treated as a hypothetical position held in its aggregate
    direction. The large-sample, selection-bias-free counterpart to
    ``exit_method_perf``; covers the position-independent methods (aggregator +
    the signal-methods-as-exits). ``horizon`` / ``llm_review`` are held-only and
    not present here. ``session`` filters by signal-generation session;
    ``direction`` by the hypothetical position's side. Cached (the OHLCV join is
    heavy)."""
    from src.analysis.exit_panel import compute_shadow_exit_method_perf
    return _cached(("shadow_exit_method_perf", days, int(min_n), session or "all", direction or "all"),
                   lambda: _retry(lambda: compute_shadow_exit_method_perf(days=days, min_n=min_n,
                                                                          session=session, direction=direction,
                                                                          horizons=PANEL_HORIZONS),
                                  "shadow_exit_method_perf"))


def horizon_edge_curve(days: Optional[int] = 90, conf_min: float = 0.78) -> dict:
    """The realized edge-decay curve of combined_score by holding horizon (the
    ground truth behind the edge-decay time-stop) + its calibration
    (``edge_days`` window / ``strength``). Cached (the OHLCV forward-return join is
    heavy); run-based. ``{curve: DataFrame, cal: dict}``, empty until forward
    returns exist."""
    from src.analysis.horizon_edge import compute_horizon_edge_curve, calibrate_edge_horizon

    def _q():
        return {"curve": compute_horizon_edge_curve(days=days, conf_min=conf_min),
                "cal": calibrate_edge_horizon()}
    return _cached(("horizon_edge", days, conf_min), lambda: _retry(_q, "horizon_edge"))


def exit_reason_breakdown(session: Optional[str] = None,
                          direction: Optional[str] = None) -> list:
    """Per-exit-reason realized performance over CLOSED trades (trades / win_rate /
    avg / median / compound / best / worst) — the realized outcome of each exit
    RULE. ``session`` filters by the session the trade EXITED in (the rule's
    firing moment); ``direction`` by the position's side. Cached; not windowed."""
    from src.performance.tracker import compute_exit_reason_perf
    return _cached(("exit_reason_breakdown", session or "all", direction or "all"),
                   lambda: _retry(lambda: compute_exit_reason_perf(session=session,
                                                                   direction=direction),
                                  "exit_reason_breakdown"))


def exit_forward(session: Optional[str] = None,
                 direction: Optional[str] = None) -> dict:
    """Post-exit forward-return report over CLOSED trades — what each exited
    position would have earned held 1/3/5/10 more sessions, per trade and per
    exit rule (analysis/exit_forward.py). Same session (exit session) /
    direction filter semantics as ``exit_reason_breakdown``; not windowed.
    Cached (walks the OHLCV cache per closed trade — this was the single
    heaviest uncached call on the page: warming it did nothing while every
    page load re-paid the OHLCV parse on the request thread)."""
    from src.analysis.exit_forward import compute_exit_forward_report
    return _cached(("exit_forward", session or "all", direction or "all"),
                   lambda: _retry(lambda: compute_exit_forward_report(session=session,
                                                                      direction=direction),
                                  "exit_forward"))


def monte_carlo_methods() -> dict:
    """Method luck-vs-skill Monte Carlo (bootstrap CIs + permutation p-values on
    each method's gross solo win rate — the exact number the win-rate filter
    selects on) + the filter's selection-bias null (how many methods pure chance
    would keep at these sample sizes). Deterministic (fixed seed); cached — the
    resampling over the ledger is ~instant but the trade extraction reads the DB."""
    from src.analysis.monte_carlo import compute_method_overfit_report
    return _cached("mc_methods", lambda: _retry(compute_method_overfit_report, "mc_methods"))


def monte_carlo_exits(session: Optional[str] = None,
                      direction: Optional[str] = None) -> dict:
    """Exit-timing-vs-random Monte Carlo per exit rule (does the rule time exits
    better than uniform-random exits over each trade's feasible window?). Same
    session (exit session) / direction filter semantics as ``exit_forward``;
    cached (loads OHLCV close series per closed trade)."""
    from src.analysis.monte_carlo import compute_exit_timing_report
    return _cached(("mc_exits", session or "all", direction or "all"),
                   lambda: _retry(lambda: compute_exit_timing_report(session=session,
                                                                     direction=direction),
                                  "mc_exits"))


def confidence_components_entry(days: Optional[int] = None, min_n: int = 10) -> dict:
    """Confidence-formula component isolation (raw vs raw×each factor) over the
    unbiased signals panel — entry-side, every scored ticker in its own
    combined_score direction. Cached; forward-collected from 2026-07-21 (the
    persisted factor columns are NULL on older rows, so ``has_factors`` is
    False until fresh rows accrue)."""
    from src.analysis.confidence_components import compute_entry_component_report
    return _cached(("conf_components_entry", days or "all"),
                   lambda: _retry(lambda: compute_entry_component_report(days=days, min_n=min_n),
                                  "conf_components_entry"))


def confidence_components_exit(days: Optional[int] = None, min_n: int = 10,
                               session: Optional[str] = None,
                               direction: Optional[str] = None) -> dict:
    """Confidence-formula component isolation — exit-side: signals-panel rows
    that fall inside an already-open position's holding window, oriented by
    the trade's own direction. Same session (re-read generated-in) / direction
    (held side) filter semantics as the other Exit-Performance blocks."""
    from src.analysis.confidence_components import compute_exit_component_report
    return _cached(("conf_components_exit", days or "all", session or "all", direction or "all"),
                   lambda: _retry(lambda: compute_exit_component_report(
                       days=days, min_n=min_n, session=session, direction=direction),
                       "conf_components_exit"))


def confidence_calibration(window_days: Optional[int] = None, session: Optional[str] = None,
                           direction: Optional[str] = None) -> dict:
    """Confidence-calibration report (buckets + slope) over the windowed/session/
    direction perf bundle's closed + open trades — so it tracks the tab's toggles
    and reuses the cached perf computation. Cached so the warmer covers it."""
    from src.analysis.confidence_calibration import compute_calibration

    def _q():
        perf = performance(window_days=window_days, session=session, direction=direction)
        trades = (perf.get("closed_trades") or []) + (perf.get("open_trades") or [])
        return compute_calibration(trades)
    return _cached(("confidence_calibration", window_days or "all", session or "all",
                    direction or "all"), lambda: _retry(_q, "confidence_calibration"))


def exit_quality(window_days: Optional[int] = None, session: Optional[str] = None,
                 direction: Optional[str] = None) -> dict:
    """MFE/MAE exit-quality report over the windowed/session/direction closed trades
    (the sim ledger carries the excursion fields). Cached so the warmer covers it."""
    from src.analysis.exit_quality import compute_exit_quality

    def _q():
        perf = performance(window_days=window_days, session=session, direction=direction)
        return compute_exit_quality(perf.get("closed_trades") or [])
    return _cached(("exit_quality", window_days or "all", session or "all",
                    direction or "all"), lambda: _retry(_q, "exit_quality"))


def broker_forensics() -> dict:
    """Slippage / fill-rate / drift / reject forensics over the broker tables
    (all runs — not windowed). Cached so the warmer covers it."""
    from src.analysis.broker_forensics import (
        compute_forensics, load_broker_orders, load_broker_reconciles)
    return _cached("broker_forensics",
                   lambda: _retry(lambda: compute_forensics(load_broker_orders(),
                                                            load_broker_reconciles()),
                                  "broker_forensics"))


def tracking_error() -> dict:
    """Sim-vs-broker tracking-error report over every trade with a matching
    broker fill. Cached (walks OHLCV per matched trade) so the warmer covers it."""
    from src.analysis.tracking_error import compute_tracking_error

    def _q():
        return compute_tracking_error(repo.load_trades())
    return _cached("tracking_error", lambda: _retry(_q, "tracking_error"))


def source_reliability(days: int = 14) -> list:
    """Per-source success rate + latency over the last N days (from run_sources)
    — surfaces chronically-flaky or slow data sources. Cached + retry."""
    from src.analysis.data_quality import compute_source_reliability, load_source_rows
    return _cached(("source_reliability", int(days)),
                   lambda: _retry(lambda: compute_source_reliability(load_source_rows(days)), "source_reliability"))


def method_coverage(days: int = 14) -> dict:
    """Per-method data coverage (% of tickers with a real, non-zero score) + a
    recent-vs-prior delta to flag feeds that went dark. From the signals panel.
    Cached + retry."""
    from src.analysis.data_quality import compute_method_coverage, load_signal_rows
    return _cached(("method_coverage", int(days)),
                   lambda: _retry(lambda: compute_method_coverage(load_signal_rows(days)), "method_coverage"))


def policy_comparison(days: Optional[int] = 90, horizon: int = 5) -> pd.DataFrame:
    """Counterfactual sizing-policy comparison over the signals panel (offline
    policy evaluation). Cached (the OHLCV forward-return join is heavy);
    run-based, so it ignores the window/session toggles. Empty until the panel
    has forward-return history."""
    from src.analysis.policy_eval import compare_policies

    def _q():
        try:                                   # match the ledger's calibrated cost basis
            from src.performance.tracker import calibrate_sim_costs
            calibrate_sim_costs()
        except Exception:
            pass
        return compare_policies(days=days, horizon=horizon)
    return _cached(("policy_comparison", days, int(horizon)), lambda: _retry(_q, "policy_comparison"))


def exit_policy_comparison(days: Optional[int] = 90, horizon: int = 5) -> pd.DataFrame:
    """Counterfactual CLOSE-rule comparison over the exit_signals panel (offline
    exit policy evaluation — does exit-breadth/aggregator beat the current
    LLM-scalar close?). Cached (the OHLCV forward-return join is heavy);
    run-based. Empty until the exit panel has forward-return history."""
    from src.analysis.exit_policy_eval import compare_exit_policies
    return _cached(("exit_policy_comparison", days, int(horizon)),
                   lambda: _retry(lambda: compare_exit_policies(days=days, horizon=horizon),
                                  "exit_policy_comparison"))


def source_performance(days: Optional[int] = None, horizons=("pv", 1, 5, 10),
                       min_n: int = 10) -> pd.DataFrame:
    """Per-discovery-source forward-return performance over the signals panel
    (funnel share + mean forward return + up-share win % + combined_score IC per
    horizon) — the unbiased evidence for an adaptive discovery budget. Cached
    (the OHLCV forward-return join is heavy); run-based, so it ignores the window/
    session toggles. Empty until the panel has forward-return history."""
    from src.analysis.source_performance import load_source_performance
    return _cached(("source_performance", days, tuple(horizons), int(min_n)),
                   lambda: _retry(lambda: load_source_performance(
                       horizons=horizons, days=days, min_n=min_n), "source_performance"))


def predictability(days: Optional[int] = None, horizons=("pv", 1, 5, 10), min_n: int = 30,
                   n_buckets: int = 3) -> dict:
    """Predictability-feature IC panel — does ``combined_score`` predict forward
    returns better inside high-trend / moderate-vol / high-breadth buckets of the
    signals panel? Returns ``{"buckets": df, "edges": df}`` (bucketed conditional
    IC + the best-minus-worst-bucket edge summary). Cached (the OHLCV feature +
    forward-return join is heavy); run-based, uses the WHOLE panel (features are
    OHLCV-derived, not stamp-dependent) so it has signal immediately."""
    from src.analysis.predictability import load_predictability

    def _q():
        buckets, edges = load_predictability(horizons=horizons, days=days,
                                             min_n=min_n, n_buckets=n_buckets)
        return {"buckets": buckets, "edges": edges}
    return _cached(("predictability", days, tuple(horizons), int(min_n), int(n_buckets)),
                   lambda: _retry(_q, "predictability"))


def price_volume_perf(force: bool = False) -> dict:
    """Return & score bucketed by stock PRICE and DOLLAR VOLUME — do penny /
    thin names diverge from pricier / liquid ones? ``{"trades": {...}, "signals":
    {...}}``: realized ledger returns by band + combined_score / 5-day forward
    return by band over the unbiased signals panel. Cached (the OHLCV join is
    heavy); run-based so it refreshes when a new run lands."""
    from src.analysis.price_volume_perf import (trade_return_by_price_volume,
                                                 score_by_price_volume)

    def _q():
        trades = repo.load_trades()
        return {"trades": trade_return_by_price_volume(trades),
                "signals": score_by_price_volume()}
    return _cached(("price_volume_perf",), lambda: _retry(_q, "price_volume_perf"), force=force)


def source_trade_perf() -> list:
    """Realized per-source trade outcomes from the ledger (trades / win_rate /
    avg / median / best / worst by ``universe_source``) — what actually traded
    per discovery source. Cached + retry; small-n and gate-selection-biased, the
    realized counterpart to ``source_performance``."""
    from src.analysis.source_performance import compute_source_trade_perf
    return _cached("source_trade_perf",
                   lambda: compute_source_trade_perf(_retry(lambda: repo.load_trades(),
                                                            "source_trade_perf")))


def dark_sources(days: int = 14) -> list:
    """Historically-populated feeds whose recent successful fetches are ALL
    empty (the 0%→100% went-dark Δ — e.g. quiver_congress 2026-06-29). Powers
    the amber feed-darkness banner. Cached + retry."""
    from src.analysis.data_quality import compute_dark_sources, load_source_rows
    return _cached(("dark_sources", int(days)),
                   lambda: _retry(lambda: compute_dark_sources(load_source_rows(days)), "dark_sources"))


def latest_gate_diag() -> dict:
    """gate_diag JSON of the most recent run (carries the price-provenance
    verdict for the banner + Execution tab). ``{}`` when unavailable."""
    rid = latest_run_id()
    if not rid:
        return {}
    df = _retry(lambda: repo.fetch_df("SELECT gate_diag FROM runs WHERE run_id = ?", [rid]),
                "gate_diag")
    if df.empty or not df.iloc[0]["gate_diag"]:
        return {}
    import json
    try:
        return json.loads(df.iloc[0]["gate_diag"])
    except Exception:
        return {}


# ── background pre-warm ─────────────────────────────────────────────────────
#
# The version cache makes a WARM page load ~0.3s, but every new pipeline run
# invalidates it, so whoever opens the dashboard next pays the full cold
# rebuild (~60s: NAV walks over every decision stream, the 5M-row simulated
# trades panel, the predictability/price-volume sweeps). Since the pipeline
# ticks every ~30 min, that "next person" is almost always the user.
#
# Nothing about that work needs a browser waiting on it. This warms the same
# accessors on a daemon thread as soon as a new run lands, so the cache is
# already populated by the time anyone looks. Purely an optimisation: it only
# ever fills the cache the request path would have filled itself, so a failure
# here costs a slow page load, never a wrong one.

_WARM_POLL_SECONDS = 20.0
_WARM_SUBPROCESS_TIMEOUT = 3600.0   # a sweep is ~400s alone, ~1600s against a busy tick
_WARM_STALE_GRACE = 3 * 3600.0      # how long _cached may serve a stale snapshot
_WARM_SNAPSHOT = "cache/dashboard_warm.pkl"   # survives a restart (see _load_snapshot)
_warm_thread = None
_warm_state: dict = {"ver": None, "running": False, "in_flight": False,
                     "last_ok": 0.0, "last_took": 0.0, "last_entries": 0}


def method_decile_curves(days: Optional[int] = None) -> dict:
    """Per-method DECILE curves on the pivot basis (2026-08-13 rank directive's
    dashboard companion): for every method column with data, each panel row's
    score is ranked WITHIN ITS DAY among that method's non-zero scores, bucketed
    into deciles, and each decile reports n / win% / mean winsorized oriented
    pivot return. Long-oriented: the outcome is the SIGNED pivot move (not
    direction-adjusted), so an upward-sloping curve = higher score -> more
    upside — the exact consumption the rank basis feeds the combine.

    Returns ``{"methods": {m: {"ret": [...10], "win": [...10], "n": [...10]}},
    "meta": {...}}``. One computation covers every method; the dropdown slices.
    """
    def _q():
        import numpy as np
        from src.analysis.signal_panel import build_panel
        from src.db.schema import SIGNAL_BASE_METHOD_COLUMNS
        panel = build_panel(horizons=(5,), days=days)
        if panel is None or panel.empty or "fwd_ret_pivot" not in panel.columns:
            return {"methods": {}, "meta": {}}
        fp = panel.copy()
        fp["fwd_ret_pivot"] = pd.to_numeric(fp["fwd_ret_pivot"], errors="coerce")
        fp = fp[fp["fwd_ret_pivot"].notna()]
        if fp.empty:
            return {"methods": {}, "meta": {}}
        fp["day"] = fp["signal_date"].astype(str).str[:10]
        lo, hi = np.percentile(fp["fwd_ret_pivot"], [1, 99])
        fp["retw"] = fp["fwd_ret_pivot"].clip(lo, hi)
        out: dict = {}
        for m in SIGNAL_BASE_METHOD_COLUMNS:
            if m not in fp.columns:
                continue
            sc = pd.to_numeric(fp[m], errors="coerce")
            sub = fp[sc.notna() & (sc != 0)].copy()
            if len(sub) < 200 or sub["day"].nunique() < 5:
                continue
            sub["score"] = pd.to_numeric(sub[m], errors="coerce")
            pct = sub.groupby("day")["score"].rank(pct=True)
            b = np.clip((pct * 10).astype(int), 0, 9)
            g = sub.groupby(b).agg(
                n=("retw", "size"), ret=("retw", "mean"),
                win=("retw", lambda s: 100.0 * (s > 0).mean()))
            g = g.reindex(range(10))
            out[m] = {
                "ret": [None if v != v else round(float(v), 3) for v in g["ret"]],
                "win": [None if v != v else round(float(v), 1) for v in g["win"]],
                "n": [0 if v != v else int(v) for v in g["n"]],
            }
        meta = {"rows": int(len(fp)), "days": int(fp["day"].nunique()),
                "d0": str(fp["day"].min()), "d1": str(fp["day"].max()),
                "winsor": [round(float(lo), 2), round(float(hi), 2)]}
        return {"methods": out, "meta": meta}
    return _cached(("method_decile_curves", days or "all"),
                   lambda: _retry(_q, "method_decile_curves"))


def _warm_targets():
    """The accessors a cold page load would otherwise force.

    Ordered heaviest-first (measured cold, 2026-07-24: simulated_method_perf
    ~30s, performance ~19s — it also drives compute_macro_eval/compute_stage_eval
    internally — price_volume_perf ~18s, predictability ~17s, signal_ic ~6s), so
    the biggest win lands earliest if a new run interrupts the sweep. Resolved
    by NAME so a renamed/removed accessor degrades to "not warmed" instead of
    raising on import."""
    names = ("simulated_method_perf", "performance", "price_volume_perf",
             "predictability", "signal_ic", "source_performance",
             "exit_method_perf", "shadow_exit_method_perf", "exit_forward",
             "confidence_components_entry", "confidence_components_exit",
             "confidence_calibration", "monte_carlo_methods", "monte_carlo_exits",
             "policy_comparison", "exit_policy_comparison", "horizon_edge_curve",
             "exit_quality", "broker_forensics", "tracking_error",
             "exit_reason_breakdown",
             "source_reliability", "method_coverage", "broker_trades",
             # 2026-07-25: both share the memoised panel, so they are cheap —
             # but an unwarmed accessor still costs the FIRST visitor after
             # every run, which is nearly every visit at a 30-min tick.
             "ticker_perf", "arm_eval", "market_relative_skill",
             "method_decile_curves")
    return [(n, globals().get(n)) for n in names]


def warm_caches(reason: str = "") -> float:
    """Populate every heavy cache for the CURRENT data version. Returns seconds
    spent. Safe to call from any thread; individual failures are logged and
    skipped."""
    started = time.time()
    ok = 0
    for label, fn in _warm_targets():
        if fn is None:
            continue
        try:
            fn()
            ok += 1
        except Exception as e:
            logger.debug(f"[dashboard] warm {label} failed: {e}")
    took = time.time() - started
    logger.info(f"[dashboard] cache warm{f' ({reason})' if reason else ''} — "
                f"{ok} accessor(s) in {took:.1f}s; page loads served from cache")
    return took


def _repo_root() -> str:
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _merge_snapshot(blob: dict, source: str) -> int:
    """Merge a warm snapshot into ``_perf_cache``. Returns entries merged.

    Each value is unpickled individually so one bad entry costs that accessor
    alone. Unpickling is the only GIL-held work the parent does here, and it is
    seconds against the ~400 s the compute would have cost."""
    import pickle
    merged = 0
    for key, raw in (blob.get("cache") or {}).items():
        try:
            _perf_cache[key] = pickle.loads(raw)
            merged += 1
        except Exception as e:
            logger.debug(f"[dashboard] warm merge skipped {key!r} ({source}): {e}")
    return merged


def _warm_in_subprocess(ver: str) -> None:
    """Run the sweep in a CHILD process and merge the result.

    The whole point is that the heavy pandas work happens under a DIFFERENT GIL,
    so the web server keeps answering while it runs. On any failure we log and
    return WITHOUT falling back to an in-process sweep — that fallback would
    reintroduce exactly the outage this exists to prevent. A failed warm costs
    stale numbers (``_cached`` keeps serving the last snapshot), never a hang.
    """
    import os
    import pickle
    import subprocess
    import sys
    import tempfile

    started = time.time()
    fd, path = tempfile.mkstemp(prefix="dash_warm_", suffix=".pkl")
    os.close(fd)
    _warm_state["in_flight"] = True
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "dashboard.warm_worker", path],
            cwd=_repo_root(), capture_output=True, timeout=_WARM_SUBPROCESS_TIMEOUT,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or b"").decode(errors="replace").strip().splitlines()[-3:]
            logger.warning(f"[dashboard] warm subprocess failed (rc={proc.returncode}); "
                           f"serving the previous snapshot. {' | '.join(tail)}")
            return
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        merged = _merge_snapshot(blob, "subprocess")
        took = time.time() - started
        _warm_state.update(last_ok=time.time(), last_took=took, last_entries=merged)
        logger.info(f"[dashboard] cache warm (run {ver}) — {merged} entries in {took:.1f}s "
                    f"in a CHILD process; the server stayed responsive throughout")
        if blob.get("dropped"):
            logger.debug(f"[dashboard] warm entries not transferable: {blob['dropped']}")
        _save_snapshot(path)
    except subprocess.TimeoutExpired:
        logger.warning(f"[dashboard] warm subprocess exceeded "
                       f"{_WARM_SUBPROCESS_TIMEOUT:.0f}s; serving the previous snapshot")
    except Exception as e:
        logger.warning(f"[dashboard] warm subprocess error: {e}; serving the previous snapshot")
    finally:
        _warm_state["in_flight"] = False
        try:
            os.unlink(path)
        except OSError:
            pass


def _save_snapshot(src_path: str) -> None:
    """Keep the newest warm result on disk so a RESTART starts warm.

    Without this, the first page load after every restart recomputes the whole
    sweep on the request thread — the ~400 s hang that reads to a phone as
    "Loading…" forever and is indistinguishable from a broken dashboard."""
    import os
    import shutil
    try:
        dest = os.path.join(_repo_root(), _WARM_SNAPSHOT)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(src_path, dest + ".tmp")
        os.replace(dest + ".tmp", dest)             # atomic: never a torn snapshot
    except Exception as e:
        logger.debug(f"[dashboard] could not persist warm snapshot: {e}")


def _load_snapshot() -> None:
    """Populate the cache from the last persisted sweep, if any (best-effort).

    A snapshot written for the CURRENT run also adopts its version as the
    warmer's starting point, so a restart does not immediately re-sweep data it
    just restored. Without that, ``_warm_state["ver"]`` began at None and every
    restart paid a full ~530 s child sweep to recompute the snapshot it had
    loaded seconds earlier — invisible except as fan noise (observed
    2026-08-15: restore at 15:06, redundant sweep finished 15:12). A snapshot
    from an OLDER run is left alone: there the sweep is real work.
    """
    import os
    import pickle
    path = os.path.join(_repo_root(), _WARM_SNAPSHOT)
    if not os.path.exists(path):
        return
    try:
        with open(path, "rb") as fh:
            blob = pickle.load(fh)
        merged = _merge_snapshot(blob, "snapshot")
        age = (time.time() - os.path.getmtime(path)) / 60.0
        current = _data_version()
        fresh = merged > 0 and current is not None and blob.get("ver") == current
        if fresh:
            _warm_state["ver"] = current
        logger.info(f"[dashboard] restored {merged} cache entries from the last warm "
                    f"snapshot ({age:.0f} min old) — "
                    + ("already current (run {}), so no re-sweep is needed".format(current)
                       if fresh else
                       "first page load is fast, and the numbers refresh as soon as "
                       "the background warm finishes"))
    except Exception as e:
        logger.debug(f"[dashboard] warm snapshot unreadable: {e}")


def _warm_loop() -> None:
    while True:
        try:
            ver = _data_version()
            if ver is not None and ver != _warm_state["ver"]:
                _warm_state["ver"] = ver
                _warm_in_subprocess(ver)
        except Exception as e:                      # never let the thread die
            logger.debug(f"[dashboard] warm loop error: {e}")
        time.sleep(_WARM_POLL_SECONDS)


def start_cache_warmer() -> None:
    """Start the background warmer (idempotent). Called once from ``app.run``."""
    global _warm_thread
    if _warm_state["running"]:
        return
    import threading
    _load_snapshot()                                # start warm, not cold
    _warm_state["running"] = True
    _warm_thread = threading.Thread(target=_warm_loop, name="dash-cache-warmer", daemon=True)
    _warm_thread.start()
    logger.info(f"[dashboard] background cache warmer started "
                f"(polls every {_WARM_POLL_SECONDS:.0f}s; warms in a CHILD process on each "
                f"new pipeline run, so the sweep never holds this process's GIL)")
