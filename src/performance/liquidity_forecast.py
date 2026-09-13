"""Expected-liquidity forecast — will THIS ticker's order price drift?

Predicts, per ticker and BEFORE any order exists, the expected one-way
execution deviation (|fill − decision price|, in bp) — the quantity the
2026-08-30 drift study measured as the book's one reliable execution leak
(72% of fills adverse; average loss pinned at the session LMT cap). A wide
expected spread means a high chance the fill drifts from the decision price
(or is capped/killed), so the forecast is the ex-ante drift-risk signal.

Design (validated 2026-08-30 against 122 tickers with >= 3 real LMT fills):

* **Measured quoted spread FIRST** — the live NBBO (point-in-time, the
  canonical basis) or the EOD BID_ASK sweep (``spread_sweep.py`` →
  ``cache/ibkr_spread.json``, a day-average converted down onto the
  point-in-time basis). Either goes through the **refitted POWER LAW**
  ``|deviation| ≈ a × spread^b`` (priors a 2.295 / b 0.647, registry rows
  ``liquidity_map_a`` / ``liquidity_map_b``), fitted per LEG against the book
  at that leg's own submit instant and validated out-of-sample on four folds.
  ``b < 1`` because we do not pay the full touch on a wide book — the session
  LMT cap refuses it (60.6 bp quoted → 20.4 bp paid on 32+ bp books).
* **Structural estimator from cached daily OHLCV (fallback)** — Corwin–Schultz
  (2012) two-day high/low spread, gap-adjusted, negatives floored per pair
  (Spearman +0.374 vs realized per-ticker median |slippage_bps|, the best
  single predictor measured — beats ADV −0.285 and ATR +0.328); Abdi–Ranaldo
  (2017) close-vs-midrange as the fallback when CS is unavailable (corr with
  CS +0.73, but half the coverage). CS UNDERSTATES genuinely wide names
  (LX: 43.3 bp measured vs 19.7 forecast), which is why IBKR leads.
* **Level calibration** — CS/AR estimate the spread embedded in daily RANGES,
  which overstates what a marketable LMT actually pays (and our realized
  sample is fills-only, truncated by the session cap): measured ratio
  realized/struct-half = **0.27**. The factor is re-measured from the broker's
  own filled legs (median-of-medians over tickers with >= 3 fills), shrunk
  toward the 0.27 prior, clamped — the standard calibration idiom. Ranking is
  the estimator's job; the LEVEL comes from our own fills.
* **Per-ticker realized blend** — a ticker with its own fill history pulls the
  forecast toward its measured median |slippage| (w = n/(n+6)).
* **Class fallback** — no usable OHLCV estimator → the (price band × ADV band)
  realized-median table below; no class info either → None (no forecast is
  honest; inventing one is not).

Risk classes (RTH basis, thresholds vs the 20 bp RTH LMT cap; validated
monotone: realized median 8.5 / 13.0 / 14.0 bp and cap-breach share
19% / 34% / 38% across low / medium / high):

    low    expected < 10 bp   — comfortably inside the cap
    medium 10–20 bp           — near the cap; drift likely, mostly capped
    high   > 20 bp            — cap-breach territory: fills only when price
                                comes to us; high drift-or-kill chance

Runs in the pipeline right after the data fetch (``prime_liquidity_forecast``),
persists per run to ``signals.exp_halfspread_bps`` and stamps new trades, so
the forecast-vs-realized join stays point-in-time. PANEL-FIRST: nothing gates
or sizes on it yet — it accrues evidence first (house convention).

Everything is cache-only (never spends an API call), memoised per (ticker,
ET session date), and fail-soft: any failure returns None and the pipeline
proceeds untouched. ``enable_liquidity_forecast`` turns the whole mechanism
off. Tests: ``tests/test_liquidity_forecast.py``.
"""

from __future__ import annotations

import threading
import time
from datetime import date
from typing import Dict, Optional, Sequence

import numpy as np

from loguru import logger  # project configures loguru sinks only

from config.settings import settings

# ── module constants (not settings: thresholds that weaken a defence are not
#    knobs, and these are pinned to the measured drift-study scale) ──────────
DRIFT_RISK_LOW_BPS = 10.0     # below: "low"
DRIFT_RISK_HIGH_BPS = 20.0    # above: "high" (== the RTH LMT cap)
_LEVEL_PRIOR = 0.27           # realized/struct-half ratio measured 2026-08-30
_LEVEL_PRIOR_N = 30           # tickers of evidence at which measurement = prior
_LEVEL_CLAMP = (0.10, 1.00)
_BLEND_K = 6                  # per-ticker fills at which realized weight = 0.5
_CLAMP_BPS = (0.5, 400.0)
_MIN_BARS = 25                # fewer daily bars than this → no structural read
_REALIZED_TTL = 3600.0        # seconds between broker-fill re-reads
_REALIZED_MIN_FILLS = 3       # per-ticker fills before its own evidence counts

# Entries older than the max age are ignored (the nightly sweep refreshes the
# pool on rotation; a name that fell out of the pool decays back to CS).
_IBKR_SPREAD_MAX_AGE_DAYS = 10

# ── the spread → expected-deviation mapping (REFIT 2026-09-01) ──────────────
# POWER LAW  |deviation| ≈ a × spread^b  on the POINT-IN-TIME basis.
#
# Refitted on 937 real filled legs, each joined to the consolidated NBBO at its
# OWN submit instant, and validated out-of-sample on four held-out folds
# (contiguous halves both directions + odd/even days both directions). The
# power form won the pre-registered selection on BOTH criteria — best worst
# fold (6.84 bp median abs error vs linear 7.09, spread-only 8.24, and the
# retired additive `10 + 0.30×spread` 8.94) and best mean (5.03 vs 5.57 /
# 5.19 / 8.41) — with stabler parameters across folds (b 0.53–0.69, a
# 2.05–3.30) than the linear alternative (slope 0.38–0.65, floor 2.2–6.0).
#
# **b < 1 is mechanistic, not curve-fitting**: we do NOT pay the full touch on
# a wide book, because the session LMT cap refuses it and the order fills only
# when the market comes to us (measured on 32+ bp books: 60.6 bp quoted,
# 20.4 bp actually paid). The retired additive form got this exactly backwards
# at both ends — it over-charged tight names ~7× (10.2 predicted vs 1.5
# measured) and under-charged the widest (28 vs 38).
_MAP_A_PRIOR, _MAP_A_PRIOR_N = 2.295, 30
_MAP_A_CLAMP = (0.8, 6.0)
_MAP_B_PRIOR, _MAP_B_PRIOR_N = 0.647, 30
_MAP_B_CLAMP = (0.35, 1.0)
_MAP_MIN_LEGS = 40          # below this the priors simply hold
_MAP_MAX_QUOTE_AGE_S = 300  # a stale book never describes what we crossed

# LIVE NBBO layer (2026-08-31): the Polygon batch snapshot's lastQuote gives
# the whole universe's CURRENT half-spread each tick for free. It outranks the
# EOD-measured store because the point of the forecast is drift risk NOW — a
# name quoting 349 bp pre-market against a 12 bp RTH median is exactly what
# the store cannot see.
_LIVE_SPREAD_TTL = 5400.0     # seconds a tick's live map stays authoritative

# The two measured layers are DIFFERENT QUANTITIES: the EOD sweep is a
# whole-RTH TIME-AVERAGE (spreads are U-shaped intraday, widest at the open),
# the live NBBO is a point-in-time book. Measured 2026-08-31 over 70 names, the
# sweep runs ~1.88× the point-in-time book.
#
# **The canonical basis is POINT-IN-TIME** (2026-09-01): that is what the
# mapping is now fitted on, and it is the book an order actually crosses. So
# the conversion runs the other way than it did — a live NBBO passes through
# UNCHANGED and the day-average SWEEP is divided down onto the point-in-time
# basis before the mapping. Self-calibrates from the overlap (names carrying
# both a live book and a sweep entry today).
_SWEEP_TO_PIT_PRIOR, _SWEEP_TO_PIT_PRIOR_N = 1.0 / 1.88, 20
_SWEEP_TO_PIT_CLAMP = (0.25, 1.0)

# Cold-start fallback: realized median |slippage_bps| by liquidity class,
# measured 2026-08-30 over the broker's filled legs (price band × ADV band,
# same edges as spread.price_band/adv_band). The (>= $100, < $20M) cell was
# unobserved; 18 bp interpolates its neighbours conservatively.
_CLASS_FALLBACK_BPS = {
    (0, 0): 25.0, (0, 1): 13.0,
    (1, 0): 18.0, (1, 1): 12.0,
    (2, 0): 18.0, (2, 1): 12.0,
}

# ── state ───────────────────────────────────────────────────────────────────
_FORECAST_MEMO: dict = {"day": None, "map": {}}   # ticker → forecast dict | None
_REALIZED_CACHE: dict = {"ts": 0.0, "by_ticker": {}, "level": None, "level_n": 0}
_LIVE_SPREADS: dict = {"ts": 0.0, "map": {}}      # ticker → live half-spread bp
_PRIME_RESULTS: dict = {"ts": 0.0, "map": {}}     # what the last prime computed
_LOCK = threading.Lock()


def reset_cache() -> None:
    """Drop all memoised state (tests / forced refresh)."""
    _FORECAST_MEMO.update(day=None, map={})
    _REALIZED_CACHE.update(ts=0.0, by_ticker={}, level=None, level_n=0,
                           ibkr_map=None)
    _IBKR_MAP_CACHE.update(mtime=None, map={})
    _LIVE_SPREADS.update(ts=0.0, map={}, scale=None)
    _PRIME_RESULTS.update(ts=0.0, map={})


def set_live_spreads(half_bps_by_ticker: Dict[str, float]) -> int:
    """Install this tick's LIVE observed half-spreads (bp) — the pipeline calls
    it right after the snapshot fetch with the batch NBBO. Replaces the prior
    tick's map wholesale; authoritative for ``_LIVE_SPREAD_TTL`` seconds, after
    which the forecast decays to the EOD-measured store, then CS. Returns the
    number of usable entries installed."""
    if not getattr(settings, "enable_liquidity_forecast", True):
        return 0
    clean = {}
    for tk, v in (half_bps_by_ticker or {}).items():
        try:
            if tk and v is not None and float(v) >= 0:
                clean[str(tk).upper()] = float(v)
        except (TypeError, ValueError):
            continue
    _LIVE_SPREADS.update(ts=time.time(), map=clean, scale=None)
    return len(clean)


def _live_spread(ticker: str) -> Optional[float]:
    """The tick's live half-spread for one ticker, or None past the TTL."""
    if (time.time() - _LIVE_SPREADS["ts"]) > _LIVE_SPREAD_TTL:
        return None
    return _LIVE_SPREADS["map"].get(str(ticker or "").upper())


def _sweep_to_pit_scale() -> float:
    """Factor putting a day-average sweep value onto the POINT-IN-TIME basis
    the mapping is fitted on. Measured from today's overlap (median
    nbbo/sweep ratio), shrunk toward the 1/1.88 prior and clamped; the prior
    holds exactly when nothing overlaps."""
    cached = _LIVE_SPREADS.get("scale")
    if cached is not None:
        return cached
    from src.performance.calibration import report_calibration, shrink
    live = _LIVE_SPREADS["map"]
    sweep = _ibkr_spread_map()
    ratios = [v / sweep[tk] for tk, v in live.items()
              if v and v > 0.05 and sweep.get(tk) and sweep[tk] > 0.05]
    val = shrink(_SWEEP_TO_PIT_PRIOR, _SWEEP_TO_PIT_PRIOR_N,
                 float(np.median(ratios)) if ratios else None, len(ratios))
    val = min(max(val, _SWEEP_TO_PIT_CLAMP[0]), _SWEEP_TO_PIT_CLAMP[1])
    _LIVE_SPREADS["scale"] = val
    try:
        report_calibration("liquidity_sweep_to_pit", value=val,
                           prior=_SWEEP_TO_PIT_PRIOR, n_evidence=len(ratios),
                           unit="ratio",
                           note="day-average sweep -> point-in-time basis")
    except Exception:
        pass
    return val


# ── structural estimators (fractions, FULL spread) ──────────────────────────

def _log(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(x)


def corwin_schultz_spread(high: Sequence[float], low: Sequence[float]) -> Optional[float]:
    """Corwin–Schultz (2012) full bid-ask spread as a fraction, from daily
    high/low ranges: consecutive-day pairs, overnight-gap adjusted, negative
    pair estimates floored to 0 (the paper's convention), then averaged.
    None when fewer than 10 valid pairs exist."""
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    if len(h) < 20 or len(h) != len(l):
        return None
    ok = np.isfinite(h) & np.isfinite(l) & (l > 0) & (h >= l)
    h = np.where(ok, h, np.nan)
    l = np.where(ok, l, np.nan)
    h1, l1 = h[:-1], l[:-1]
    h2, l2 = h[1:].copy(), l[1:].copy()
    # Overnight-gap adjustment: shift day-2's range back onto day-1's close
    # range so a gap is not read as spread.
    up = l2 > h1
    dn = h2 < l1
    adj = np.where(up, l2 - h1, np.where(dn, h2 - l1, 0.0))
    h2 = h2 - adj
    l2 = l2 - adj
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        beta = _log(h1 / l1) ** 2 + _log(h2 / l2) ** 2
        gamma = _log(np.maximum(h1, h2) / np.minimum(l1, l2)) ** 2
        k = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    s = s[np.isfinite(s)]
    if len(s) < 10:
        return None
    return float(np.mean(np.maximum(s, 0.0)))


def abdi_ranaldo_spread(close: Sequence[float], high: Sequence[float],
                        low: Sequence[float]) -> Optional[float]:
    """Abdi–Ranaldo (2017) close-vs-high/low-midpoint full-spread fraction
    (pooled form). None when undefined or on fewer than 10 valid pairs."""
    c = np.asarray(close, dtype=float)
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    if len(c) < 20 or len(c) != len(h) or len(c) != len(l):
        return None
    ok = np.isfinite(c) & np.isfinite(h) & np.isfinite(l) & (c > 0) & (l > 0) & (h >= l)
    lc = _log(np.where(ok, c, np.nan))
    m = (_log(np.where(ok, h, np.nan)) + _log(np.where(ok, l, np.nan))) / 2.0
    x = (lc[:-1] - m[:-1]) * (lc[:-1] - m[1:])
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return None
    return float(np.sqrt(max(0.0, 4.0 * float(np.mean(x)))))


# ── realized evidence (broker fills) + level calibration ────────────────────

def _refresh_realized() -> None:
    """Re-read the broker's filled legs: per-ticker median |slippage_bps| and
    the level factor's raw measurement. TTL-cached; {} / prior on any failure.
    Runs in the pipeline (writer) process — repo.fetch_df follows the process
    role, so this never opens a second write handle."""
    now = time.time()
    if (now - _REALIZED_CACHE["ts"]) < _REALIZED_TTL:
        return
    with _LOCK:
        if (now - _REALIZED_CACHE["ts"]) < _REALIZED_TTL:
            return
        by_ticker: Dict[str, tuple] = {}
        try:
            import pandas as pd
            from src.db import repo
            df = repo.fetch_df(
                "SELECT ticker, ABS(slippage_bps) AS a FROM broker_orders "
                "WHERE fill_price IS NOT NULL AND fill_price > 0 "
                "AND slippage_bps IS NOT NULL")
            if df is not None and not df.empty:
                g = df.groupby("ticker")["a"].agg(["median", "size"])
                by_ticker = {str(t).upper(): (float(r["median"]), int(r["size"]))
                             for t, r in g.iterrows()}
        except Exception as e:
            logger.debug(f"[liquidity] realized-fill read unavailable: {e}")
        _REALIZED_CACHE.update(ts=now, by_ticker=by_ticker)
        _REALIZED_CACHE.update(level=None, level_n=0, ibkr_map=None)  # lazily recomputed


def _level_factor() -> float:
    """Rescale factor mapping the structural HALF-spread onto the realized
    |slippage| scale. Measured as median(realized_i / struct_half_i) over
    tickers with >= _REALIZED_MIN_FILLS fills, shrunk toward the documented
    prior, clamped. Prior holds exactly when no fills exist (cold start)."""
    _refresh_realized()
    cached = _REALIZED_CACHE.get("level")
    if cached is not None:
        return cached
    ratios = []
    for tk, (med, n) in _REALIZED_CACHE["by_ticker"].items():
        if n < _REALIZED_MIN_FILLS or med <= 0:
            continue
        s = _structural(tk)
        # CS/AR rows ONLY: this factor rescales the RANGE-BASED estimators.
        # An IBKR-measured row would put realized/true-spread ratios (~20x for
        # megacaps, all drift) into a multiplicative calibration built for a
        # different quantity — those pairs belong to _ibkr_mapping instead.
        if s and s.get("struct_half_bps") and s.get("estimator") in ("cs", "ar"):
            ratios.append(med / s["struct_half_bps"])
    from src.performance.calibration import report_calibration, shrink
    n_obs = len(ratios)
    observed = float(np.median(ratios)) if ratios else None
    val = shrink(_LEVEL_PRIOR, _LEVEL_PRIOR_N, observed, n_obs)
    val = min(max(val, _LEVEL_CLAMP[0]), _LEVEL_CLAMP[1])
    _REALIZED_CACHE.update(level=val, level_n=n_obs)
    try:
        report_calibration(
            "liquidity_level_factor", value=val, prior=_LEVEL_PRIOR,
            n_evidence=n_obs, unit="ratio",
            note="struct half-spread → realized |slippage| rescale "
                 "(liquidity_forecast)")
    except Exception:
        pass
    return val


# ── IBKR measured-spread layer ──────────────────────────────────────────────

_IBKR_MAP_CACHE: dict = {"mtime": None, "map": {}}


def _ibkr_spread_map() -> Dict[str, float]:
    """Fresh entries of the EOD sweep's store: {ticker: measured half-spread
    bp}. Cached against the store file's mtime; {} when absent/stale/disabled —
    the CS fallback then carries every name, exactly the pre-sweep behaviour."""
    try:
        from src.performance.spread_sweep import SPREAD_STORE_PATH, load_spread_store
        mtime = SPREAD_STORE_PATH.stat().st_mtime if SPREAD_STORE_PATH.exists() else None
        if _IBKR_MAP_CACHE["mtime"] == mtime:
            return _IBKR_MAP_CACHE["map"]
        out: Dict[str, float] = {}
        if mtime is not None:
            today = date.today()
            for tk, e in load_spread_store().items():
                try:
                    age = (today - date.fromisoformat(str(e.get("fetched_at", ""))[:10])).days
                    if 0 <= age <= _IBKR_SPREAD_MAX_AGE_DAYS and float(e["half_bps"]) >= 0:
                        out[str(tk).upper()] = float(e["half_bps"])
                except Exception:
                    continue
        _IBKR_MAP_CACHE.update(mtime=mtime, map=out)
        return out
    except Exception as e:
        logger.debug(f"[liquidity] ibkr spread store unavailable: {e}")
        return _IBKR_MAP_CACHE["map"]


def _theil_sen(x: np.ndarray, y: np.ndarray, max_pairs: int = 200_000,
               seed: int = 7) -> tuple:
    """Median-of-pairwise-slopes fit with a median intercept — the robust
    estimator this calibration needs, because realized deviation carries a fat
    favourable tail that drags least squares."""
    n = len(x)
    rng = np.random.default_rng(seed)
    if n * (n - 1) // 2 <= max_pairs:
        i, j = np.triu_indices(n, 1)
    else:
        i = rng.integers(0, n, max_pairs)
        j = rng.integers(0, n, max_pairs)
        ok = i != j
        i, j = i[ok], j[ok]
    dx = x[j] - x[i]
    ok = np.abs(dx) > 1e-9
    if not ok.any():
        return float("nan"), float("nan")
    m = float(np.median((y[j][ok] - y[i][ok]) / dx[ok]))
    return float(np.median(y - m * x)), m


def _fit_pairs() -> list:
    """(point-in-time half-spread bp, |realized deviation| bp) per FILLED LEG.

    Two provenances, deliberately pooled here and only here: the live capture
    (`broker_orders.bid_at_submit`) and books RECOVERED after the fact for
    older orders (`broker_order_quotes`). The calibration only needs the true
    book at submit, which both are; anything auditing what the ORDER PATH had
    available must read `broker_orders` alone (see the schema comment)."""
    try:
        from src.db import repo
        df = repo.fetch_df(
            "SELECT o.slippage_bps AS slip, "
            "       COALESCE(o.bid_at_submit, q.bid) AS bid, "
            "       COALESCE(o.ask_at_submit, q.ask) AS ask, "
            "       COALESCE(q.quote_age_s, 0.0) AS age "
            "FROM broker_orders o "
            "LEFT JOIN broker_order_quotes q "
            "  ON q.ticker = o.ticker AND q.submitted_at = o.submitted_at "
            "WHERE o.fill_price > 0 AND o.slippage_bps IS NOT NULL")
        if df is None or df.empty:
            return []
        out = []
        for r in df.itertuples():
            b, a, s = r.bid, r.ask, r.slip
            if b is None or a is None or b != b or a != a or s != s:
                continue
            if not (0 < float(b) <= float(a)) or abs(float(r.age)) > _MAP_MAX_QUOTE_AGE_S:
                continue
            mid = (float(a) + float(b)) / 2.0
            if mid <= 0:
                continue
            out.append(((float(a) - float(b)) / 2.0 / mid * 1e4, abs(float(s))))
        return out
    except Exception as e:
        logger.debug(f"[liquidity] mapping fit pairs unavailable: {e}")
        return []


def _spread_mapping() -> tuple:
    """``(a, b)`` of the power law ``|deviation| ≈ a × spread^b`` on the
    point-in-time basis. Fitted in LOG space (a power law is linear there) by
    Theil-Sen over per-leg pairs, shrunk toward the validated priors and
    clamped; the priors hold exactly below ``_MAP_MIN_LEGS``."""
    _refresh_realized()
    cached = _REALIZED_CACHE.get("ibkr_map")
    if cached is not None:
        return cached
    from src.performance.calibration import report_calibration, shrink
    pairs = [(s, y) for s, y in _fit_pairs() if s > 0.05 and y > 0.05]
    a_obs = b_obs = None
    n = len(pairs)
    if n >= _MAP_MIN_LEGS:
        lx = np.log(np.array([p[0] for p in pairs], dtype=float))
        ly = np.log(np.array([p[1] for p in pairs], dtype=float))
        li, lm = _theil_sen(lx, ly)
        if li == li and lm == lm:
            a_obs, b_obs = float(np.exp(li)), float(lm)
    a = shrink(_MAP_A_PRIOR, _MAP_A_PRIOR_N, a_obs, n if a_obs is not None else 0)
    b = shrink(_MAP_B_PRIOR, _MAP_B_PRIOR_N, b_obs, n if b_obs is not None else 0)
    a = min(max(a, _MAP_A_CLAMP[0]), _MAP_A_CLAMP[1])
    b = min(max(b, _MAP_B_CLAMP[0]), _MAP_B_CLAMP[1])
    _REALIZED_CACHE["ibkr_map"] = (a, b)
    try:
        report_calibration("liquidity_map_a", value=a, prior=_MAP_A_PRIOR,
                           n_evidence=n, unit="bp",
                           note="power-law scale: |dev| = a x spread^b")
        report_calibration("liquidity_map_b", value=b, prior=_MAP_B_PRIOR,
                           n_evidence=n, unit="exponent",
                           note="power-law exponent (<1 = we don't pay the full touch)")
    except Exception:
        pass
    return a, b


def expected_deviation_bps(spread_pit_bps: float) -> float:
    """Apply the mapping to a POINT-IN-TIME half-spread (bp)."""
    a, b = _spread_mapping()
    return float(a * max(float(spread_pit_bps), 0.0) ** b)


# ── per-ticker structural read (memoised per day + store version) ───────────

def _memo_epoch() -> tuple:
    """Memo key: the ET day plus the sweep store's mtime, so a nightly sweep
    landing mid-session invalidates the day's cached structural reads instead
    of waiting for midnight."""
    try:
        from src.performance.spread_sweep import SPREAD_STORE_PATH
        mt = SPREAD_STORE_PATH.stat().st_mtime if SPREAD_STORE_PATH.exists() else 0.0
    except Exception:
        mt = 0.0
    return (date.today().isoformat(), mt)


def _structural(ticker: str) -> Optional[dict]:
    """One ticker's structural read: the IBKR MEASURED half-spread when the
    sweep has a fresh entry, else CS (AR fallback) from cache-only OHLCV, plus
    price/ADV for the class fallback. Memoised per (ticker, day, store
    version). None = nothing knowable."""
    tk = str(ticker or "").upper()
    if not tk:
        return None
    epoch = _memo_epoch()
    if _FORECAST_MEMO["day"] != epoch:
        _FORECAST_MEMO.update(day=epoch, map={})
    memo = _FORECAST_MEMO["map"]
    if tk in memo:
        return memo[tk]
    measured = _ibkr_spread_map().get(tk)
    if measured is not None:
        out = {"struct_half_bps": measured, "estimator": "ibkr",
               "price": None, "adv": None}
        memo[tk] = out
        return out
    out: Optional[dict] = None
    try:
        import pandas as pd
        from src.data.cache import load_ohlcv
        df = load_ohlcv(tk)
        if df is not None and not df.empty and len(df) >= _MIN_BARS:
            window = int(getattr(settings, "liquidity_forecast_window", 63))
            df = df.tail(window + 1)
            c = pd.to_numeric(df.get("Close"), errors="coerce").to_numpy(dtype=float)
            h = pd.to_numeric(df.get("High"), errors="coerce").to_numpy(dtype=float)
            l = pd.to_numeric(df.get("Low"), errors="coerce").to_numpy(dtype=float)
            v = pd.to_numeric(df.get("Volume"), errors="coerce").to_numpy(dtype=float)
            cs = corwin_schultz_spread(h, l)
            ar = abdi_ranaldo_spread(c, h, l)
            spread_frac = cs if cs is not None else ar
            with np.errstate(invalid="ignore"):
                adv = float(np.nanmedian((c * v)[-20:]))
                px = float(c[-1]) if np.isfinite(c[-1]) else None
            out = {
                "struct_half_bps": (spread_frac * 1e4 / 2.0
                                    if spread_frac is not None else None),
                "estimator": ("cs" if cs is not None
                              else "ar" if ar is not None else None),
                "price": px,
                "adv": adv if adv == adv else None,
            }
    except Exception as e:
        logger.debug(f"[liquidity] structural read failed for {tk}: {e}")
        out = None
    memo[tk] = out
    return out


# ── public API ──────────────────────────────────────────────────────────────

def forecast_for(ticker: str) -> Optional[dict]:
    """Full forecast for one ticker::

        {"exp_halfspread_bps": float, "risk": "low|medium|high",
         "source": "blend|struct|class", "n_fills": int}

    None when the flag is off or nothing at all is knowable (no OHLCV, no
    liquidity class, no fills). RTH basis — apply the session multiplier via
    ``expected_halfspread_bps(ticker, session=...)`` when needed."""
    if not getattr(settings, "enable_liquidity_forecast", True):
        return None
    # LIVE NBBO first: the current book outranks any typical-spread estimate —
    # the same quantity as the IBKR-measured layer, so the same additive
    # mapping applies. Falls to the EOD store / CS past the tick TTL.
    live = _live_spread(ticker)
    if live is not None:
        # Already the canonical POINT-IN-TIME basis the mapping is fitted on —
        # passes through unconverted (see _sweep_to_pit_scale).
        s: Optional[dict] = {"struct_half_bps": live, "estimator": "nbbo",
                             "price": None, "adv": None}
    else:
        s = _structural(ticker)
    _refresh_realized()
    med_n = _REALIZED_CACHE["by_ticker"].get(str(ticker or "").upper())
    realized, n_fills = (med_n if med_n else (None, 0))

    base = None
    source = None
    if s and s.get("struct_half_bps") is not None:
        est = s.get("estimator")
        if est in ("ibkr", "nbbo"):
            # A MEASURED quoted half-spread → the refitted power law. The live
            # NBBO is already point-in-time; the EOD sweep is a day-average and
            # is converted down onto that basis first.
            spread = s["struct_half_bps"]
            if est == "ibkr":
                spread *= _sweep_to_pit_scale()
            base = expected_deviation_bps(spread)
        else:
            base = s["struct_half_bps"] * _level_factor()
        source = "struct"
    elif s is not None:
        # OHLCV exists but the estimators were degenerate → class fallback.
        from src.performance.spread import adv_band, price_band
        pb, ab = price_band(s.get("price")), adv_band(s.get("adv"))
        if pb is not None and ab is not None:
            base = _CLASS_FALLBACK_BPS.get((pb, ab))
            source = "class"
    if base is None and realized is None:
        return None
    if base is None:
        base, source = realized, "realized"
    elif (realized is not None and n_fills >= 1
          and (s or {}).get("estimator") not in ("nbbo", "ibkr")):
        # Blend toward the ticker's own realized history ONLY where the
        # structural estimate is crude (CS/AR/class table). For a MEASURED
        # spread it measurably HURTS (2026-09-01, validated on all four
        # held-out folds: mean median-abs-error 5.71 blended vs 5.03
        # mapping-only, and worse on every individual fold): the fitted
        # mapping already conditions on TODAY's book, so blending drags the
        # estimate back toward the ticker's unconditional historical average
        # — stale information the spread has already superseded.
        w = n_fills / (n_fills + _BLEND_K)
        base = w * realized + (1.0 - w) * base
        source = "blend"
    bps = min(max(float(base), _CLAMP_BPS[0]), _CLAMP_BPS[1])
    risk = ("low" if bps < DRIFT_RISK_LOW_BPS
            else "medium" if bps <= DRIFT_RISK_HIGH_BPS else "high")
    return {"exp_halfspread_bps": round(bps, 2), "risk": risk,
            "source": source, "n_fills": int(n_fills),
            "estimator": (s or {}).get("estimator")}


def quoted_halfspread_bps(ticker: str, session: Optional[str] = None) -> Optional[dict]:
    """The MEASURED point-in-time QUOTED half-spread in bp, on the RTH basis.

    ``{"bps": float, "raw_bps": float, "estimator": "nbbo"|"ibkr"}`` or None.

    This is the raw ``(ask-bid)/2/mid`` book — NOT ``exp_halfspread_bps``, which
    is that spread pushed through the fitted power law into an expected
    execution DEVIATION. A caller whose thresholds were calibrated on quoted
    books must read THIS one: a 20 bp quoted book maps to only ~16 bp of
    deviation, so the same number means two different things.

    ONLY the tick's LIVE NBBO. Everything else is refused, for two different
    reasons, and both refusals are load-bearing:

      * Corwin-Schultz / Abdi-Ranaldo are volatility PROXIES rather than
        quotes; measured 2026-09-02 they separated no trade outcomes at any
        threshold while the real book did (contrast +0.09 pp, p 0.91).
      * The nightly IBKR sweep IS a measured book, but its LEVEL is biased on
        this basis even after the point-in-time conversion. Compared like for
        like against 769 tickers' RTH NBBO medians (2026-09-02): rank
        correlation is high (Spearman 0.913) but the sweep runs at **0.78x**
        the real book, which would put 53.7% of names under 4 bp where the true
        share is 41.0%, and only 2.2% over 20 bp where the truth is 4.4% — it
        DISAGREES about which side of the 4/5 bp line a name sits on for 15-17%
        of names. A level-based curve fed a level-biased input over-boosts and
        under-cuts silently, so the sweep is deliberately not offered here. It
        remains the right input for ``forecast_for``, whose mapping is fitted on
        it.

    The live map is primed universe-wide each tick and stays authoritative for
    90 minutes, so in an RTH tick this is available for ~all of the book (99.9%
    of panel rows since 2026-09-01); off-hours, where Polygon refuses a quote
    past 120 s, it correctly returns None and the caller applies no tilt.

    ``session`` divides out that session's own widening, so a threshold means
    the same thing at 03:00 as at 15:00. The session's extra cost is already
    charged by the session size haircut and the session-aware cost model;
    reading an overnight book at face value here would charge it twice.
    """
    if not getattr(settings, "enable_liquidity_forecast", True):
        return None
    live = _live_spread(ticker)
    if live is None or not float(live) > 0:
        return None
    raw, est = float(live), "nbbo"
    mult = 1.0
    if session:
        try:
            from src.performance.spread import _session_spread_multiplier
            mult = float(_session_spread_multiplier(session)) or 1.0
        except Exception:
            mult = 1.0
    return {"bps": round(raw / mult, 3), "raw_bps": round(raw, 3), "estimator": est}


def expected_halfspread_bps(ticker: str, session: Optional[str] = None,
                            ) -> Optional[float]:
    """Expected one-way execution deviation in bp; ``session`` applies the
    same extended/overnight multiplier the cost model uses. None = no view."""
    f = forecast_for(ticker)
    if f is None:
        return None
    mult = 1.0
    if session:
        try:
            from src.performance.spread import _session_spread_multiplier
            mult = float(_session_spread_multiplier(session))
        except Exception:
            mult = 1.0
    return round(f["exp_halfspread_bps"] * mult, 2)


def drift_risk(ticker: str) -> Optional[str]:
    """``low | medium | high`` — chance the fill drifts from the decision
    price (RTH basis; thresholds are the validated 10/20 bp constants)."""
    f = forecast_for(ticker)
    return f["risk"] if f else None


def cached_bps(ticker: str) -> Optional[float]:
    """Persistence-path read: exactly what the last ``prime_liquidity_forecast``
    computed for this ticker — never a fresh computation — so the panel records
    what the run actually saw, and a disabled/failed/absent prime persists NULL
    rather than a value no decision could have used. Stale primes (older than
    the live TTL) also read NULL."""
    if not getattr(settings, "enable_liquidity_forecast", True):
        return None
    if (time.time() - _PRIME_RESULTS["ts"]) > _LIVE_SPREAD_TTL:
        return None
    return _PRIME_RESULTS["map"].get(str(ticker or "").upper())


def prime_liquidity_forecast(tickers: Sequence[str]) -> dict:
    """Batch-compute the forecast for every ticker (pipeline: right after the
    data fetch). Cache-only and fail-soft per ticker. Returns a summary dict
    for the run log."""
    if not getattr(settings, "enable_liquidity_forecast", True):
        return {}
    t0 = time.time()
    n = n_ok = n_struct = n_live = n_high = 0
    vals = []
    results: Dict[str, float] = {}
    for tk in dict.fromkeys(str(t).upper() for t in (tickers or []) if t):
        n += 1
        try:
            f = forecast_for(tk)
        except Exception:
            f = None
        if f:
            n_ok += 1
            vals.append(f["exp_halfspread_bps"])
            results[tk] = f["exp_halfspread_bps"]
            n_struct += f["source"] in ("struct", "blend")
            n_live += f.get("estimator") == "nbbo"
            n_high += f["risk"] == "high"
    _PRIME_RESULTS.update(ts=time.time(), map=results)
    return {
        "n": n, "forecast": n_ok, "structural": n_struct, "live_nbbo": n_live,
        "high_risk": n_high,
        "median_bps": round(float(np.median(vals)), 1) if vals else None,
        "level_factor": round(_level_factor(), 3),
        "elapsed_s": round(time.time() - t0, 1),
    }
