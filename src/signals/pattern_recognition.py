"""
Pattern Recognition signal — historical price pattern analysis.

Detects 8 classical chart patterns and derives a [-1, +1] signal score
based on each pattern's historical win rate for the specific ticker.

First run (cold cache):
  - Fetches 2 years of OHLCV data via get_history
  - Scans the full history with a sliding 40-bar window (step=5)
  - For each detected pattern records the 5d / 10d forward return
  - Saves a per-ticker pattern library → cache/patterns/<TICKER>.json (TTL 7 days)

Subsequent runs (warm cache):
  - Loads the cached library instantly (no network call)
  - Detects the current pattern from the most recent 60 bars
  - Converts historical win rate to a directional score

Score formula
─────────────
  win_rate: fraction of historical occurrences where the pattern correctly
            predicted its inherent direction (bullish or bearish).
  edge     = (win_rate - 0.5) × 2          ∈ [-1, +1]
  score    = clip(edge × inherent_direction, -1, +1)

  win_rate 0.50 → edge 0.00 → score 0.00  (pattern has no historical edge)
  win_rate 0.75 → edge 0.50 → score ±0.50 (moderate historical accuracy)
  win_rate 0.90 → edge 0.80 → score ±0.80 (strong historical accuracy)

When the pattern has < 3 historical occurrences a weak prior (±0.25) is used.
"""

import json
from datetime import date
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.data.market_data import get_history
from src.data.cache import load_ohlcv


# ── Cache ─────────────────────────────────────────────────────────────────────
_CACHE_DIR      = Path("cache/patterns")
_CACHE_TTL_DAYS = 7       # refresh library weekly
_HISTORY_PERIOD = "2y"    # long fetch for library build
_DETECT_WINDOW  = 60      # most recent bars used for current-pattern detection
_PATTERN_WINDOW = 40      # bars in each sliding-window scan
_SCAN_STEP      = 5       # step between windows (overlapping is fine)
_MIN_HISTORY    = 50      # minimum bars to build meaningful statistics
_MIN_PATTERN_N  = 3       # minimum historical occurrences to trust win rate
_WEAK_PRIOR     = 0.25    # fallback score magnitude when history is sparse


# ── Pattern direction table ───────────────────────────────────────────────────
# +1: bullish pattern — a "win" is a positive 10d forward return
# -1: bearish pattern — a "win" is a negative 10d forward return
_PATTERN_DIR: dict = {
    "double_bottom":       +1,
    "inv_head_shoulders":  +1,
    "ascending_triangle":  +1,
    "bull_flag":           +1,
    "double_top":          -1,
    "head_shoulders":      -1,
    "descending_triangle": -1,
    "bear_flag":           -1,
}


# ── Peak / trough detection ───────────────────────────────────────────────────

def _local_extrema(prices: np.ndarray, hw: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays of local-maxima and local-minima indices."""
    n = len(prices)
    peaks, troughs = [], []
    for i in range(hw, n - hw):
        window = prices[i - hw : i + hw + 1]
        if prices[i] == window.max():
            peaks.append(i)
        if prices[i] == window.min():
            troughs.append(i)
    return np.array(peaks, dtype=int), np.array(troughs, dtype=int)


# ── Individual pattern detectors ─────────────────────────────────────────────

def _double_bottom(prices: np.ndarray, troughs: np.ndarray, tol: float = 0.04) -> bool:
    if len(troughs) < 2:
        return False
    t1, t2 = int(troughs[-2]), int(troughs[-1])
    if t2 - t1 < 8:
        return False
    l1, l2 = prices[t1], prices[t2]
    return abs(l1 - l2) / max(l1, l2) <= tol


def _double_top(prices: np.ndarray, peaks: np.ndarray, tol: float = 0.04) -> bool:
    if len(peaks) < 2:
        return False
    p1, p2 = int(peaks[-2]), int(peaks[-1])
    if p2 - p1 < 8:
        return False
    h1, h2 = prices[p1], prices[p2]
    return abs(h1 - h2) / max(h1, h2) <= tol


def _head_shoulders(prices: np.ndarray, peaks: np.ndarray, tol: float = 0.05) -> bool:
    if len(peaks) < 3:
        return False
    ls, hd, rs = int(peaks[-3]), int(peaks[-2]), int(peaks[-1])
    lh, hh, rh = prices[ls], prices[hd], prices[rs]
    return hh > lh and hh > rh and abs(lh - rh) / max(lh, rh) <= tol


def _inv_head_shoulders(prices: np.ndarray, troughs: np.ndarray, tol: float = 0.05) -> bool:
    if len(troughs) < 3:
        return False
    ls, hd, rs = int(troughs[-3]), int(troughs[-2]), int(troughs[-1])
    ll, hl, rl = prices[ls], prices[hd], prices[rs]
    return hl < ll and hl < rl and abs(ll - rl) / max(ll, rl) <= tol


def _ascending_triangle(
    prices: np.ndarray, peaks: np.ndarray, troughs: np.ndarray, tol: float = 0.025
) -> bool:
    if len(peaks) < 2 or len(troughs) < 2:
        return False
    ph1, ph2 = prices[int(peaks[-2])],   prices[int(peaks[-1])]
    tl1, tl2 = prices[int(troughs[-2])], prices[int(troughs[-1])]
    return abs(ph1 - ph2) / max(ph1, ph2) <= tol and tl2 > tl1 * (1 + tol / 2)


def _descending_triangle(
    prices: np.ndarray, peaks: np.ndarray, troughs: np.ndarray, tol: float = 0.025
) -> bool:
    if len(peaks) < 2 or len(troughs) < 2:
        return False
    ph1, ph2 = prices[int(peaks[-2])],   prices[int(peaks[-1])]
    tl1, tl2 = prices[int(troughs[-2])], prices[int(troughs[-1])]
    return abs(tl1 - tl2) / max(tl1, tl2) <= tol and ph2 < ph1 * (1 - tol / 2)


def _bull_flag(prices: np.ndarray, min_pole: float = 0.04, max_flag: float = 0.04) -> bool:
    n = len(prices)
    if n < 20:
        return False
    half  = n // 2
    pole  = prices[:half]
    flag  = prices[half:]
    up    = (pole[-1] - pole[0]) / pole[0]
    rng   = (flag.max() - flag.min()) / flag.min() if flag.min() > 0 else 99.0
    return up > min_pole and rng < max_flag


def _bear_flag(prices: np.ndarray, min_pole: float = 0.04, max_flag: float = 0.04) -> bool:
    n = len(prices)
    if n < 20:
        return False
    half  = n // 2
    pole  = prices[:half]
    flag  = prices[half:]
    down  = (pole[0] - pole[-1]) / pole[0]
    rng   = (flag.max() - flag.min()) / flag.min() if flag.min() > 0 else 99.0
    return down > min_pole and rng < max_flag


# ── Multi-pattern classifier ──────────────────────────────────────────────────

def _detect_pattern(prices: np.ndarray) -> Optional[str]:
    """
    Classify the dominant chart pattern in the price window.
    3-point patterns take priority over 2-point ones; flags checked last.
    Returns None if no pattern is detected.
    """
    if len(prices) < 20:
        return None
    peaks, troughs = _local_extrema(prices)
    if _head_shoulders(prices, peaks):
        return "head_shoulders"
    if _inv_head_shoulders(prices, troughs):
        return "inv_head_shoulders"
    if _double_top(prices, peaks):
        return "double_top"
    if _double_bottom(prices, troughs):
        return "double_bottom"
    if _ascending_triangle(prices, peaks, troughs):
        return "ascending_triangle"
    if _descending_triangle(prices, peaks, troughs):
        return "descending_triangle"
    if _bull_flag(prices):
        return "bull_flag"
    if _bear_flag(prices):
        return "bear_flag"
    return None


# ── Pattern library cache ─────────────────────────────────────────────────────

def _load_library(ticker: str) -> Optional[dict]:
    path = _CACHE_DIR / f"{ticker.upper()}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        last = date.fromisoformat(data.get("last_updated", "2000-01-01"))
        if (date.today() - last).days > _CACHE_TTL_DAYS:
            return None
        return data
    except Exception:
        return None


def _save_library(ticker: str, library: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        (_CACHE_DIR / f"{ticker.upper()}.json").write_text(json.dumps(library))
    except Exception as e:
        logger.warning(f"[pattern] Failed to save library for {ticker}: {e}")


def _build_library(ticker: str, df: pd.DataFrame) -> dict:
    """
    Scan the full price history with a sliding window to compute per-pattern
    win rates and average returns for this specific ticker.
    """
    closes = df["Close"].values.astype(float)
    n = len(closes)

    stats: dict = {p: {"count": 0, "wins": 0, "ret5": 0.0, "ret10": 0.0}
                   for p in _PATTERN_DIR}

    for i in range(0, n - _PATTERN_WINDOW - 10, _SCAN_STEP):
        window = closes[i : i + _PATTERN_WINDOW]
        pattern = _detect_pattern(window)
        if pattern is None:
            continue

        entry = closes[i + _PATTERN_WINDOW]
        if entry <= 0:
            continue

        i5  = i + _PATTERN_WINDOW + 5
        i10 = i + _PATTERN_WINDOW + 10
        if i10 >= n:
            continue

        r5  = (closes[i5]  - entry) / entry
        r10 = (closes[i10] - entry) / entry

        s = stats[pattern]
        s["count"] += 1
        s["ret5"]  += r5
        s["ret10"] += r10
        # Win: pattern correctly predicted its inherent direction using 10d return
        if _PATTERN_DIR[pattern] > 0:
            s["wins"] += 1 if r10 > 0 else 0
        else:
            s["wins"] += 1 if r10 < 0 else 0

    patterns_out: dict = {}
    for pat, s in stats.items():
        if s["count"] >= _MIN_PATTERN_N:
            patterns_out[pat] = {
                "count":    s["count"],
                "win_rate": round(s["wins"] / s["count"], 3),
                "avg_r5":   round(s["ret5"]  / s["count"], 4),
                "avg_r10":  round(s["ret10"] / s["count"], 4),
            }

    return {
        "ticker":       ticker.upper(),
        "last_updated": date.today().isoformat(),
        "history_bars": n,
        "patterns":     patterns_out,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def compute_pattern_score(ticker: str) -> Tuple[float, str]:
    """
    Detect the current chart pattern and return (score, pattern_name).

    score ∈ [-1.0, +1.0]  positive = bullish setup, negative = bearish setup.
    Returns (0.0, "") when no pattern is detected or data is insufficient.

    Cold path (first call / expired cache):
      Fetches 2 years of OHLCV data, builds pattern library, saves to cache.
    Warm path (cache hit):
      Loads library instantly, detects pattern from OHLCV chart cache.
    """
    library = _load_library(ticker)

    if library is None:
        df = get_history(ticker, period=_HISTORY_PERIOD)
        if df.empty or len(df) < _MIN_HISTORY:
            logger.debug(f"[pattern] {ticker}: insufficient history ({len(df)} bars) — skipping")
            return 0.0, ""
        logger.info(f"[pattern] Building pattern library for {ticker} ({len(df)} bars)…")
        library = _build_library(ticker, df)
        _save_library(ticker, library)
        recent_prices = df["Close"].values[-_DETECT_WINDOW:].astype(float)
    else:
        recent_df = load_ohlcv(ticker)
        if recent_df is None or len(recent_df) < 20:
            recent_df = get_history(ticker, period="3mo")
        if recent_df is None or recent_df.empty or len(recent_df) < 20:
            return 0.0, ""
        recent_prices = recent_df["Close"].values[-_DETECT_WINDOW:].astype(float)

    current = _detect_pattern(recent_prices)
    if current is None:
        return 0.0, ""

    inherent = _PATTERN_DIR[current]
    pstats   = library.get("patterns", {}).get(current)

    # ── Synthetic per-ticker prior ──────────────────────────────────────────
    if pstats is None or pstats["count"] < _MIN_PATTERN_N:
        syn_win_rate = 0.5 + (_WEAK_PRIOR / 2.0) * inherent   # implies a weak prior
        syn_n = 0   # treated as no real evidence
        syn_source = "weak_prior"
    else:
        syn_win_rate = float(pstats["win_rate"])
        syn_n        = int(pstats["count"])
        syn_source   = "synthetic"

    # ── Live registry overlay (Bayesian shrinkage) ─────────────────────────
    blended_wr, blend_info = _blend_with_live_registry(
        pattern_name=current,
        ticker=ticker,
        syn_win_rate=syn_win_rate,
        syn_n=syn_n,
    )

    edge  = (blended_wr - 0.5) * 2.0
    score = round(max(-1.0, min(1.0, edge * inherent)), 3)

    logger.debug(
        f"[pattern] {ticker}: {current}  blended_wr={blended_wr:.0%}  "
        f"edge={edge:+.2f}  score={score:+.3f}  "
        f"(syn={syn_source} n={syn_n} wr={syn_win_rate:.0%}  {blend_info})"
    )
    return score, current


# ── Live-registry blend ───────────────────────────────────────────────────────

def _blend_with_live_registry(
    pattern_name: str,
    ticker: str,
    syn_win_rate: float,
    syn_n: int,
) -> Tuple[float, str]:
    """Blend the synthetic per-ticker prior with the live global win rate.

    Returns ``(blended_win_rate, info_str)``. ``info_str`` is a short tag
    summarising what was blended in, for the debug log.

    Blending logic
    ──────────────
    Let live_n be the number of *real trades* the system has taken when this
    pattern was active at entry (across all tickers), live_wr their win rate.

      w_live = live_n / (live_n + PRIOR_N)
      blended_wr = w_live × live_wr + (1 − w_live) × syn_win_rate

    PRIOR_N controls how much live evidence is needed before it dominates the
    synthetic prior. With PRIOR_N = 10:
      live_n = 0   → blended = synthetic                   (no evidence yet)
      live_n = 5   → blended = 0.33 × live + 0.67 × syn   (live has some pull)
      live_n = 20  → blended = 0.67 × live + 0.33 × syn   (live is dominant)
      live_n = 50  → blended = 0.83 × live + 0.17 × syn   (live wins)

    If a per-(ticker, pattern) live row exists with n ≥ ``_MIN_TP_LIVE_N``,
    use it instead of the global aggregate — that's a stronger signal for this
    specific ticker and overrides the cross-ticker average.
    """
    from config import settings as _settings
    if not getattr(_settings, "enable_pattern_registry", True):
        return syn_win_rate, "registry_disabled"

    try:
        from src.signals.pattern_registry import pattern_stats, ticker_pattern_stats
    except Exception:
        return syn_win_rate, "registry_unavailable"

    # Prefer per-(ticker, pattern) when meaningfully populated.
    # NOTE: we use *pattern_accuracy* (did the price move in the pattern's
    # inherent direction?), NOT *win_rate* (did the system's trade win?).
    # The synthetic library measures pattern accuracy too, so the blend stays
    # apples-to-apples. The two diverge when the system overrides the pattern
    # direction via its other 9 signals (e.g., shorts a bullish pattern); in
    # that case win_rate is irrelevant to the pattern's own predictive value.
    tp = ticker_pattern_stats(ticker, pattern_name)
    if tp is not None and int(tp.get("n", 0)) >= _MIN_TP_LIVE_N:
        live_acc = tp.get("pattern_accuracy")
        if live_acc is None:
            live_acc = tp.get("win_rate") or 0.5   # schema-v1 fallback
        live_acc = float(live_acc)
        live_n   = int(tp.get("n", 0))
        prior_n  = max(1, getattr(_settings, "pattern_registry_prior_n", 10))
        w_live   = live_n / (live_n + prior_n)
        blended  = w_live * live_acc + (1 - w_live) * syn_win_rate
        return blended, f"live(per-ticker)_acc={live_acc:.0%} n={live_n} w={w_live:.2f}"

    glb = pattern_stats(pattern_name)
    if glb is None:
        return syn_win_rate, "no_live_data"
    live_acc = glb.get("pattern_accuracy")
    if live_acc is None:
        live_acc = glb.get("win_rate") or 0.5   # schema-v1 fallback
    live_acc = float(live_acc)
    live_n   = int(glb.get("n_trades", 0))
    prior_n  = max(1, getattr(_settings, "pattern_registry_prior_n", 10))
    w_live   = live_n / (live_n + prior_n)
    blended  = w_live * live_acc + (1 - w_live) * syn_win_rate
    return blended, f"live(global)_acc={live_acc:.0%} n={live_n} w={w_live:.2f}"


_MIN_TP_LIVE_N = 3   # need at least this many per-(ticker,pattern) trades to override global