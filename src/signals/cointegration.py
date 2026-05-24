"""
Cointegration Pairs — statistical-arbitrage market-neutral alpha.

Goes beyond ``sector_pairs`` (which keys off opposing directional *signals*) by
testing whether two price series are *cointegrated*: i.e. whether a linear
combination of them is stationary (mean-reverting), even though each series
individually wanders (is non-stationary / has a unit root). When two
economically-linked securities are cointegrated, the spread between them is a
mean-reverting process you can trade: short the rich leg, long the cheap leg,
and collect the reversion — with market beta hedged out by the hedge ratio.

Engle-Granger two-step
──────────────────────
  1. OLS the (log) prices of A on B:   log(A) = α + β·log(B) + ε
     β is the hedge ratio; ε is the spread (residual).
  2. ADF-test the residual spread for a unit root. Rejecting the null (stat
     below the critical value) ⇒ the spread is stationary ⇒ A and B are
     cointegrated.

The ADF test is implemented natively in numpy (no statsmodels dependency):
the Augmented Dickey-Fuller regression with a constant and ``maxlag`` lagged
differences,

    Δs_t = c + γ·s_{t-1} + Σ_{i=1..p} δ_i·Δs_{t-i} + u_t

and the test statistic is the t-ratio on γ̂. We compare it against the
Engle-Granger / MacKinnon critical values for a residual-based test (which are
more demanding than plain-ADF criticals because the residual is *estimated*).

Trade signal
────────────
  spread_zscore = (spread_today − mean(spread)) / std(spread)
    z ≥ +entry  → spread rich → SHORT A / LONG B
    z ≤ −entry  → spread cheap → LONG A / SHORT B
    |z| < exit  → near fair value, no edge

Half-life (Ornstein-Uhlenbeck) of the spread filters out pairs that revert too
slowly to be tradeable: Δs_t = λ·s_{t-1} + c + u_t ⇒ half_life = −ln(2)/λ.

Per-ticker directional lean
───────────────────────────
Although the natural output is *pairs*, each tradeable pair also implies a
single-name directional view (the long leg is cheap → bullish nudge; the short
leg is rich → bearish nudge). ``ticker_scores`` aggregates these across all
pairs a ticker belongs to so the aggregator can fold cointegration into the
per-ticker combined score like any other method.

Data: cache-first via ``load_ohlcv``; falls back to ``get_history(period="1y")``
only when the cache is too short. Works with ``ENABLE_FETCH_DATA=false`` when
the OHLCV chart caches are populated.
"""

from __future__ import annotations

from datetime import date
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from config import settings
from src.models import CointPair, CointPairsContext
from src.data.cache import load_ohlcv
from src.data.market_data import get_history


_MIN_OVERLAP   = 50      # minimum common observations to attempt a test
_PREFERRED_HIST = 180    # below this many cached bars, fetch 1y for a powerful test
_LOOKBACK      = 252     # cap the test window to the most recent N obs
_ADF_MAXLAG    = 1       # lagged differences in the ADF regression
_MAX_HALF_LIFE = 60      # skip pairs whose spread reverts slower than this (days)
_MAX_PAIRS_OUT = 12      # cap reported tradeable pairs (email bloat guard)

# Engle-Granger residual-based ADF critical values (constant, N=2 variables,
# MacKinnon 2010 asymptotic). More demanding than plain-ADF because the spread
# is an estimated residual, not an observed series.
_EG_CRIT = {0.01: -3.90, 0.05: -3.34, 0.10: -3.04}

# Curated, economically-linked candidate pairs spanning sectors and asset classes.
# These have a plausible structural reason to be cointegrated (substitutes,
# same industry, dual-listings, commodity trackers). Only pairs whose BOTH legs
# have sufficient cached/fetchable history are actually tested.
_CANDIDATE_PAIRS: List[Tuple[str, str]] = [
    # Precious-metal trackers (near-perfect cointegration)
    ("GLD", "IAU"), ("GLD", "SLV"), ("GDX", "GLD"), ("SLV", "GDX"),
    # Index / large-cap ETFs
    ("SPY", "IVV"), ("SPY", "QQQ"), ("QQQ", "XLK"), ("SPY", "XLK"),
    # Semiconductors
    ("AMD", "NVDA"), ("INTC", "AMD"), ("AVGO", "QCOM"), ("NVDA", "AVGO"),
    # Mega-cap tech
    ("MSFT", "AAPL"), ("GOOGL", "META"), ("AAPL", "MSFT"),
    # Payments
    ("V", "MA"),
    # Beverages
    ("KO", "PEP"),
    # Big banks
    ("JPM", "BAC"), ("GS", "MS"), ("WFC", "C"), ("BAC", "C"),
    # Energy
    ("XOM", "CVX"), ("COP", "EOG"),
    # Retail
    ("HD", "LOW"), ("WMT", "TGT"),
    # Autos
    ("F", "GM"),
    # Telecom
    ("T", "VZ"),
    # Streaming / media
    ("NFLX", "DIS"),
    # Airlines
    ("DAL", "UAL"),
]


# ── Native ADF test ───────────────────────────────────────────────────────────

def _adf_tstat(series: np.ndarray, max_lag: int = _ADF_MAXLAG) -> Optional[float]:
    """Augmented Dickey-Fuller t-statistic (constant, ``max_lag`` lags).

    Returns the t-ratio on the lagged-level coefficient γ in
        Δs_t = c + γ·s_{t-1} + Σ δ_i·Δs_{t-i} + u_t
    A more negative value is stronger evidence the series is stationary
    (mean-reverting). Returns ``None`` when the sample is too short.
    """
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < max_lag + 12:
        return None

    dy = np.diff(y)               # Δy_t, length n-1
    y_lag = y[:-1]                # y_{t-1}, length n-1, aligned with dy

    nobs = dy.size - max_lag
    if nobs < 10:
        return None

    # Dependent variable: Δy_t for t = max_lag .. end
    Y = dy[max_lag:]
    # Regressors: constant, y_{t-1}, and lagged differences Δy_{t-1..max_lag}
    cols = [np.ones(nobs), y_lag[max_lag:]]
    for i in range(1, max_lag + 1):
        cols.append(dy[max_lag - i: dy.size - i])
    X = np.column_stack(cols)

    # OLS via least squares
    try:
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return None

    resid = Y - X @ beta
    dof = nobs - X.shape[1]
    if dof <= 0:
        return None
    sigma2 = float(resid @ resid) / dof
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    var_beta = sigma2 * np.diag(xtx_inv)
    gamma_idx = 1  # coefficient on y_{t-1}
    se_gamma = float(np.sqrt(var_beta[gamma_idx])) if var_beta[gamma_idx] > 0 else 0.0
    if se_gamma == 0.0:
        return None
    return float(beta[gamma_idx]) / se_gamma


def _adf_pvalue(tstat: float) -> float:
    """Approximate p-value for an Engle-Granger ADF stat via log-linear
    interpolation across the MacKinnon critical-value anchors. Coarse but
    monotone — adequate for ranking/display, not for formal inference."""
    # (tstat, pvalue) anchors from the residual-based critical values plus
    # softer tail points for interpolation.
    anchors = [(-4.50, 0.001), (-3.90, 0.01), (-3.34, 0.05),
               (-3.04, 0.10), (-2.50, 0.30), (-2.00, 0.60), (-1.00, 0.90)]
    if tstat <= anchors[0][0]:
        return anchors[0][1]
    if tstat >= anchors[-1][0]:
        return anchors[-1][1]
    for (t_hi, p_hi), (t_lo, p_lo) in zip(anchors, anchors[1:]):
        # anchors are ordered most-negative → least-negative
        if t_hi <= tstat <= t_lo:
            # linear interpolation in t
            frac = (tstat - t_hi) / (t_lo - t_hi) if t_lo != t_hi else 0.0
            return round(p_hi + frac * (p_lo - p_hi), 4)
    return 0.5


# ── OU half-life ────────────────────────────────────────────────────────────

def _half_life(spread: np.ndarray) -> float:
    """Ornstein-Uhlenbeck mean-reversion half-life in observations.

    Regress Δs_t on s_{t-1}: Δs_t = c + λ·s_{t-1} + u_t. half_life = −ln(2)/λ.
    Returns ``inf`` when λ ≥ 0 (no mean reversion)."""
    s = np.asarray(spread, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 10:
        return float("inf")
    ds = np.diff(s)
    s_lag = s[:-1]
    X = np.column_stack([np.ones(s_lag.size), s_lag])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, ds, rcond=None)
    except np.linalg.LinAlgError:
        return float("inf")
    lam = float(beta[1])
    if lam >= 0:
        return float("inf")
    return float(-np.log(2.0) / lam)


# ── Data access ───────────────────────────────────────────────────────────────

def _log_close(ticker: str) -> Optional[pd.Series]:
    """Return a date-indexed log-close Series for *ticker*.

    Cache-first, but cointegration needs a long window to have statistical
    power, so when the cache is shorter than ``_PREFERRED_HIST`` *and* fetching
    is enabled we pull a full year (force_refresh bypasses the TTL short-circuit
    that would otherwise return the short cache). With fetching disabled we use
    whatever is cached and just run an under-powered test rather than failing.
    """
    df = load_ohlcv(ticker)
    need_more = df is None or len(df) < _PREFERRED_HIST
    if need_more and settings.enable_fetch_data:
        try:
            df = get_history(ticker, period="1y", force_refresh=True)
        except Exception as exc:
            logger.debug(f"[coint] {ticker}: history fetch failed — {exc}")
    if df is None or df.empty or "Close" not in df.columns:
        return None
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    close = close[close > 0]
    if len(close) < _MIN_OVERLAP:
        return None
    return np.log(close)


# ── Pair test ─────────────────────────────────────────────────────────────────

def _test_pair(
    a: str, b: str,
    series_cache: Dict[str, Optional[pd.Series]],
) -> Optional[CointPair]:
    """Run the Engle-Granger test on (a, b). Returns a CointPair or None."""
    la = series_cache.get(a)
    lb = series_cache.get(b)
    if la is None or lb is None:
        return None

    joined = pd.concat([la, lb], axis=1, join="inner").dropna()
    if len(joined) < _MIN_OVERLAP:
        return None
    joined = joined.tail(_LOOKBACK)
    y = joined.iloc[:, 0].to_numpy()
    x = joined.iloc[:, 1].to_numpy()
    n = len(joined)

    # ── OLS hedge ratio: y = α + β·x ─────────────────────────────────────────
    X = np.column_stack([np.ones(n), x])
    try:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    alpha, beta = float(coef[0]), float(coef[1])
    if not np.isfinite(beta) or abs(beta) < 1e-6:
        return None

    spread = y - (alpha + beta * x)
    spread_std = float(np.std(spread, ddof=1))
    if spread_std < 1e-9:
        return None
    spread_mean = float(np.mean(spread))

    # ── ADF on the spread ────────────────────────────────────────────────────
    adf_stat = _adf_tstat(spread)
    if adf_stat is None:
        return None
    crit5 = _EG_CRIT[0.05]
    level = float(settings.cointegration_pvalue)
    crit = _EG_CRIT.get(level, crit5)
    is_coint = adf_stat <= crit
    pvalue = _adf_pvalue(adf_stat)

    # ── Half-life filter ─────────────────────────────────────────────────────
    hl = _half_life(spread)

    # ── Current z-score ──────────────────────────────────────────────────────
    z = float((spread[-1] - spread_mean) / spread_std)
    corr = float(np.corrcoef(y, x)[0, 1])

    entry = float(settings.cointegration_entry_z)
    exit_z = float(settings.cointegration_exit_z)

    # Direction: high z = A rich relative to B → short A / long B
    if z >= entry:
        long_leg, short_leg, signal = b, a, "ENTRY"
    elif z <= -entry:
        long_leg, short_leg, signal = a, b, "ENTRY"
    elif abs(z) >= exit_z:
        # leans toward an entry but not yet stretched enough
        if z > 0:
            long_leg, short_leg = b, a
        else:
            long_leg, short_leg = a, b
        signal = "MONITOR"
    else:
        long_leg, short_leg, signal = a, b, "NEUTRAL"

    if is_coint and abs(z) >= entry and np.isfinite(hl) and hl <= _MAX_HALF_LIFE:
        signal = "ENTRY"
    elif is_coint and abs(z) >= entry:
        signal = "STRETCHED"  # cointegrated + stretched but slow reversion

    rationale = (
        f"{a}/{b}: β={beta:.2f}, ADF={adf_stat:.2f} (crit {crit:.2f}, "
        f"{'cointegrated' if is_coint else 'not cointegrated'}), "
        f"half-life={hl:.0f}d, z={z:+.2f}. "
        + (
            f"Spread {'rich' if z > 0 else 'cheap'} → LONG {long_leg} / SHORT {short_leg}; "
            f"bet on mean reversion."
            if signal in ("ENTRY", "STRETCHED")
            else "Spread near fair value — monitor."
        )
    )

    return CointPair(
        ticker_a=a, ticker_b=b,
        hedge_ratio=round(beta, 4),
        adf_stat=round(adf_stat, 3),
        adf_pvalue=round(pvalue, 4),
        adf_crit_5pct=round(crit, 3),
        is_cointegrated=bool(is_coint),
        half_life_days=round(hl, 1) if np.isfinite(hl) else 999.0,
        correlation=round(corr, 3),
        spread_mean=round(spread_mean, 5),
        spread_std=round(spread_std, 5),
        spread_zscore=round(z, 3),
        long_leg=long_leg, short_leg=short_leg,
        signal=signal,
        lookback_days=n,
        rationale=rationale,
    )


def _candidate_pairs(tickers: List[str]) -> List[Tuple[str, str]]:
    """Curated pairs + same-sector combinations restricted to the universe."""
    universe = {t.upper() for t in tickers}
    pairs: List[Tuple[str, str]] = []
    seen: set = set()

    def _add(a: str, b: str):
        a, b = a.upper(), b.upper()
        if a == b:
            return
        key = tuple(sorted((a, b)))
        if key in seen:
            return
        seen.add(key)
        pairs.append((a, b))

    # Curated pairs always tested (data availability checked later)
    for a, b in _CANDIDATE_PAIRS:
        _add(a, b)

    # Same-sector combinations from the aggregator's sector map, restricted to
    # tickers actually in today's universe (keeps the count bounded).
    try:
        from src.signals.aggregator import _SECTOR_MAP
        by_sector: Dict[str, List[str]] = {}
        for stock, etf in _SECTOR_MAP.items():
            if stock.upper() in universe:
                by_sector.setdefault(etf, []).append(stock.upper())
        for etf, members in by_sector.items():
            for a, b in combinations(sorted(set(members)), 2):
                _add(a, b)
    except Exception:
        pass

    return pairs


def find_cointegrated_pairs(tickers: List[str]) -> CointPairsContext:
    """Test candidate pairs for cointegration; return tradeable pairs + per-ticker scores."""
    candidates = _candidate_pairs(tickers)

    # Pre-load each unique ticker's log-close once.
    unique = sorted({t for pair in candidates for t in pair})
    series_cache: Dict[str, Optional[pd.Series]] = {}
    for t in unique:
        series_cache[t] = _log_close(t)

    tested = 0
    results: List[CointPair] = []
    for a, b in candidates:
        if series_cache.get(a) is None or series_cache.get(b) is None:
            continue
        tested += 1
        cp = _test_pair(a, b, series_cache)
        if cp is not None:
            results.append(cp)

    cointegrated = [p for p in results if p.is_cointegrated]

    # Tradeable = cointegrated AND at an actionable z-extreme (ENTRY/STRETCHED).
    tradeable = [
        p for p in cointegrated
        if p.signal in ("ENTRY", "STRETCHED")
    ]
    tradeable.sort(key=lambda p: abs(p.spread_zscore), reverse=True)
    tradeable = tradeable[:_MAX_PAIRS_OUT]

    # ── Per-ticker directional lean ──────────────────────────────────────────
    # Each tradeable pair nudges its long leg bullish and short leg bearish,
    # scaled by how far the spread is stretched beyond the entry threshold.
    entry = float(settings.cointegration_entry_z)
    contribs: Dict[str, List[float]] = {}
    for p in tradeable:
        # magnitude grows past the entry threshold, saturating via tanh
        mag = float(np.tanh((abs(p.spread_zscore) - entry) / 1.5 + 0.3))
        mag = max(0.0, min(1.0, mag))
        contribs.setdefault(p.long_leg, []).append(+mag)
        contribs.setdefault(p.short_leg, []).append(-mag)
    ticker_scores = {
        t: round(float(np.clip(np.mean(v), -1.0, 1.0)), 3)
        for t, v in contribs.items() if v
    }

    if tradeable:
        entries = [
            f"L {p.long_leg}/S {p.short_leg} (z={p.spread_zscore:+.1f})"
            for p in tradeable[:5]
        ]
        summary = (
            f"{len(tradeable)} tradeable / {len(cointegrated)} cointegrated "
            f"of {tested} tested: " + " | ".join(entries)
        )
    else:
        summary = (
            f"No tradeable cointegration setups "
            f"({len(cointegrated)} cointegrated of {tested} tested; none at a z-extreme)."
        )

    logger.info(f"[coint] {summary}")
    for p in tradeable:
        logger.info(
            f"[coint] {p.signal}: LONG {p.long_leg} / SHORT {p.short_leg} "
            f"| β={p.hedge_ratio:.2f} ADF={p.adf_stat:.2f} z={p.spread_zscore:+.2f} "
            f"HL={p.half_life_days:.0f}d"
        )

    return CointPairsContext(
        pairs=tradeable,
        candidates_tested=tested,
        cointegrated_count=len(cointegrated),
        ticker_scores=ticker_scores,
        report_date=date.today(),
        summary=summary,
    )
