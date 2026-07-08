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


_REC_COLS = (
    "generated_at, ticker, type, direction, action, confidence, time_horizon, "
    "target_horizon, horizon_net_edge_pct, "
    "shadow_target_horizon, shadow_direction, shadow_horizon_net_edge_pct, "
    "expected_move_pct, market_aligned, upside_score, "
    "actionable, dominant_method, llm_provider, rationale"
)


def runs_df() -> pd.DataFrame:
    return _retry(lambda: repo.fetch_df("SELECT * FROM runs ORDER BY started_at DESC"), "runs")


def latest_run_id() -> Optional[str]:
    df = runs_df()
    return None if df.empty else str(df.iloc[0]["run_id"])


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
    ``producer``. Shared by every heavy accessor."""
    now = time.time()
    ver = _data_version()
    entry = _perf_cache.get(key)
    if not force and entry is not None and entry.get("ver") == ver and (now - entry["ts"]) < _PERF_TTL:
        return entry["data"]
    data = producer()
    _perf_cache[key] = {"ts": now, "data": data, "ver": ver}
    return data


def performance(window_days: Optional[int] = None, session: Optional[str] = None,
                direction: Optional[str] = None, force: bool = False) -> dict:
    """Windowed + session + direction-filtered performance bundle. ``window_days`` =
    7 / 30 (None = inception); ``session`` = rth / extended / overnight (None = all);
    ``direction`` = long / short (None = both)."""
    from src.performance.tracker import get_performance_for_email
    key = ("all" if window_days is None else int(window_days), session or "all", direction or "all")
    return _cached(key, lambda: _retry(
        lambda: get_performance_for_email(window_days=window_days, session=session, direction=direction),
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
    """Per-method information-coefficient table over the persisted signals panel
    joined with forward returns. Cached (the OHLCV join is heavy). Returns
    ``{panel_rows, tickers, ic}`` where ``ic`` is a DataFrame (empty until the
    panel has enough forward-return history)."""
    from src.analysis.signal_panel import build_panel, compute_ic

    def _q():
        panel = build_panel(horizons=horizons, days=days)
        ic = (compute_ic(panel, horizons=horizons, min_n=min_n)
              if panel is not None and not panel.empty else pd.DataFrame())
        return {
            "panel_rows": 0 if panel is None or panel.empty else int(len(panel)),
            "tickers":    0 if panel is None or panel.empty else int(panel["ticker"].nunique()),
            "ic":         ic,
        }
    key = ("signal_ic", days, tuple(horizons), int(min_n))
    return _cached(key, lambda: _retry(_q, "signal_ic"))


def simulated_method_perf(days: Optional[int] = None, min_n: int = 10,
                          session: Optional[str] = None,
                          direction: Optional[str] = None) -> pd.DataFrame:
    """Per-method directional win rate + mean gross return at 30m/3h/6h/1d/3d/1w/2w/1m
    over the ``simulated_trades`` table (every scored ticker treated as a solo
    single-method trade). Cached (the OHLCV join is heavy). ``session`` filters
    by the session the signal was GENERATED in; ``direction`` by the side of the
    method's own call (positive score = its long call). Empty until forward
    returns exist."""
    from src.analysis.simulated_trades import compute_method_perf
    return _cached(("sim_method_perf", days, int(min_n), session or "all", direction or "all"),
                   lambda: _retry(lambda: compute_method_perf(days=days, min_n=min_n,
                                                              session=session, direction=direction),
                                  "simulated_method_perf"))


def exit_method_perf(days: Optional[int] = None, min_n: int = 10,
                     session: Optional[str] = None,
                     direction: Optional[str] = None) -> pd.DataFrame:
    """Per-EXIT-method win rate / IC / IC-std / ICIR / signed return at
    30m/3h/6h/1d/3d/1w/2w/1m over the ``exit_signals`` panel (every held position
    re-scored each tick), plus the synthesized ``llm_review`` row from
    ``trade_reviews``. The exit-side counterpart to ``simulated_method_perf``.
    ``session`` filters by the session the REVIEW happened in; ``direction`` by
    the held position's side. Cached (the OHLCV join is heavy). Empty until
    forward returns exist."""
    from src.analysis.exit_panel import compute_exit_method_perf
    return _cached(("exit_method_perf", days, int(min_n), session or "all", direction or "all"),
                   lambda: _retry(lambda: compute_exit_method_perf(days=days, min_n=min_n,
                                                                   session=session, direction=direction),
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
                                                                          session=session, direction=direction),
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
    firing moment); ``direction`` by the position's side. Cheap; not windowed."""
    from src.performance.tracker import compute_exit_reason_perf
    return _retry(lambda: compute_exit_reason_perf(session=session, direction=direction),
                  "exit_reason_breakdown")


def confidence_calibration(window_days: Optional[int] = None, session: Optional[str] = None,
                           direction: Optional[str] = None) -> dict:
    """Confidence-calibration report (buckets + slope) over the windowed/session/
    direction perf bundle's closed + open trades — so it tracks the tab's toggles
    and reuses the cached perf computation."""
    perf = performance(window_days=window_days, session=session, direction=direction)
    from src.analysis.confidence_calibration import compute_calibration
    trades = (perf.get("closed_trades") or []) + (perf.get("open_trades") or [])
    return compute_calibration(trades)


def exit_quality(window_days: Optional[int] = None, session: Optional[str] = None,
                 direction: Optional[str] = None) -> dict:
    """MFE/MAE exit-quality report over the windowed/session/direction closed trades
    (the sim ledger carries the excursion fields)."""
    perf = performance(window_days=window_days, session=session, direction=direction)
    from src.analysis.exit_quality import compute_exit_quality
    return compute_exit_quality(perf.get("closed_trades") or [])


def broker_forensics() -> dict:
    """Slippage / fill-rate / drift / reject forensics over the broker tables
    (all runs — not windowed)."""
    from src.analysis.broker_forensics import (
        compute_forensics, load_broker_orders, load_broker_reconciles)
    return _retry(lambda: compute_forensics(load_broker_orders(), load_broker_reconciles()),
                  "broker_forensics")


def tracking_error() -> dict:
    """Sim-vs-broker tracking-error report over every trade with a matching
    broker fill."""
    from src.analysis.tracking_error import compute_tracking_error
    trades = _retry(lambda: repo.load_trades(), "tracking_error")
    return compute_tracking_error(trades)


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


def source_performance(days: Optional[int] = None, horizons=(1, 5, 10),
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


def predictability(days: Optional[int] = None, horizons=(1, 5, 10), min_n: int = 30,
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
