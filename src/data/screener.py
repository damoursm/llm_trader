"""Opportunity Screener — proactive setup discovery over a broad liquid universe.

Discovery elsewhere is reactive (what's *trending* / in the news). This screener is
proactive: it scans a broad, liquid universe for names that are technically **set up**
right now — surfacing them into the analysis universe even when they aren't trending.

It reuses the existing OHLCV cache and computes four well-worn, complementary screens:

  • Unusual volume   — today's volume vs the trailing 20-day average (often *precedes* news).
  • 52-week range    — new 52-week high (breakout) / near-high (breakout-watch) / new low (reversal).
  • Relative strength — trailing return vs SPY over ``screen_rs_lookback_days`` (leaders/laggards).
  • Golden/death cross — a fresh 50/200-day SMA crossover.

A liquidity gate (min price + min 20-day average dollar volume) keeps illiquid microcaps
out, so widening the funnel doesn't lower quality. Candidates are the curated liquid
universe below ∪ everything already in the OHLCV cache. Data is loaded **cache-first**
(works offline once warm); a bounded per-run warm-up fetch (``screen_max_fetch_per_run``)
primes the curated universe over a few runs. Names that fire ≥1 screen are returned for
injection into the universe, where they then receive the full signal stack — including
Trend Strength, whose Donchian breakout often confirms the screener's 52-week breakout.

Fail-graceful: any error returns an empty context; the screener never blocks the run.
"""

from datetime import date
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from config import settings
from src.models import ScreenHit, ScreenerContext
from src.data.cache import load_ohlcv, OHLCV_DIR
from src.data.market_data import get_history

_BENCHMARK = "SPY"
_MIN_BARS  = 60     # minimum bars to evaluate any screen

# Curated broad, liquid universe — large/mega-caps across every sector + major,
# sector and factor/thematic ETFs. Liquid by construction; the liquidity gate
# still applies. Union'd with whatever is already in the OHLCV cache.
_SCREEN_UNIVERSE: tuple = (
    # Technology
    "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "ORCL", "CRM", "ADBE", "CSCO", "INTC",
    "QCOM", "TXN", "AMAT", "MU", "SMCI", "ARM", "PLTR", "NOW", "PANW", "SNPS",
    "CDNS", "KLAC", "LRCX", "ANET", "DELL", "HPQ", "IBM",
    # Communication services
    "GOOGL", "META", "NFLX", "DIS", "TMUS", "T", "VZ", "CMCSA",
    # Consumer discretionary
    "AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "LOW", "BKNG", "TJX", "ABNB",
    "GM", "F", "CMG",
    # Consumer staples
    "WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "CL", "MDLZ",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "V", "MA", "AXP", "SCHW",
    "SPGI", "COF", "PYPL",
    # Health care
    "UNH", "LLY", "JNJ", "MRK", "ABBV", "PFE", "TMO", "ABT", "DHR", "AMGN",
    "ISRG", "GILD", "VRTX", "BMY", "CVS",
    # Industrials
    "CAT", "BA", "HON", "GE", "UPS", "RTX", "LMT", "DE", "UNP", "GD", "MMM",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "OXY",
    # Materials
    "LIN", "FCX", "NEM", "APD", "NUE",
    # Utilities
    "NEE", "DUK", "SO", "D",
    # Real estate
    "PLD", "AMT", "EQIX", "O",
    # Foreign / semis mega
    "TSM", "ASML",
    # Broad / style ETFs
    "QQQ", "IWM", "DIA", "MDY", "VTV", "VUG", "MTUM", "QUAL", "USMV", "VLUE",
    # Sector / thematic ETFs
    "SOXX", "SMH", "XBI", "ARKK", "KRE", "ITB", "XHB", "IBB", "XOP", "JETS",
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_pool() -> List[str]:
    """Curated liquid universe ∪ cached OHLCV tickers, cleaned to equity/ETF symbols."""
    pool = set(_SCREEN_UNIVERSE)
    try:
        if OHLCV_DIR.exists():
            for p in OHLCV_DIR.glob("*.json"):
                pool.add(p.stem.upper())
    except Exception as e:
        logger.debug(f"[screener] cache scan failed: {e}")

    out = []
    for t in pool:
        t = (t or "").strip().upper()
        if not t or t == _BENCHMARK:
            continue
        # Skip indices / futures / crypto (^VIX, GC=F, BTC-USD)
        if any(c in t for c in ("^", "=", "-")):
            continue
        if 1 <= len(t) <= 5:
            out.append(t)
    return sorted(set(out))


def _load_for_screen(ticker: str, budget: Dict[str, int]) -> Optional[pd.DataFrame]:
    """Cache-first OHLCV load; a bounded warm-up fetch primes the cache when allowed."""
    df = load_ohlcv(ticker)
    if df is not None and not df.empty and len(df) >= _MIN_BARS:
        return df
    if settings.enable_fetch_data and budget["n"] > 0:
        budget["n"] -= 1
        try:
            fetched = get_history(ticker, period="15mo")
            if fetched is not None and not fetched.empty:
                return fetched
        except Exception as e:
            logger.debug(f"[screener] fetch failed for {ticker}: {e}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker screen
# ─────────────────────────────────────────────────────────────────────────────

def _screen_one(ticker: str, df: pd.DataFrame, spy_ret: Optional[float]) -> Optional[ScreenHit]:
    n = len(df)
    if n < _MIN_BARS:
        return None

    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    vol   = df["Volume"].astype(float)
    last_close = float(close.iloc[-1])

    # ── Liquidity gate ───────────────────────────────────────────────────────
    if last_close < settings.screen_min_price:
        return None
    avg20_vol = float(vol.iloc[-20:].mean())
    if avg20_vol * last_close < settings.screen_min_dollar_volume:
        return None

    screens: List[str] = []
    notes:   List[str] = []
    net = 0
    vol_ratio = 0.0
    rs_excess = 0.0

    # ── Unusual volume (vs prior 20-day average, excluding today) ─────────────
    if n >= 22:
        prior20 = float(vol.iloc[-21:-1].mean())
        if prior20 > 0:
            vol_ratio = float(vol.iloc[-1]) / prior20
            if vol_ratio >= settings.screen_volume_ratio:
                prev = float(close.iloc[-2])
                day_dir = 1 if (prev and last_close >= prev) else -1
                screens.append("UNUSUAL_VOLUME")
                net += day_dir
                notes.append(f"{vol_ratio:.1f}× volume")

    # ── 52-week breakout / near-high / new-low reversal ──────────────────────
    if n >= _MIN_BARS:
        win = min(252, n - 1)
        prior_high = float(high.iloc[-(win + 1):-1].max())
        prior_low  = float(low.iloc[-(win + 1):-1].min())
        near = settings.screen_near_high_pct / 100.0
        if prior_high > 0 and last_close > prior_high:
            screens.append("NEW_52W_HIGH"); net += 1; notes.append("new 52w high")
        elif prior_high > 0 and last_close >= prior_high * (1 - near):
            screens.append("NEAR_52W_HIGH"); net += 1
            notes.append(f"≤{settings.screen_near_high_pct:.0f}% from 52w high")
        if prior_low > 0 and last_close < prior_low:
            screens.append("NEW_52W_LOW"); net -= 1; notes.append("new 52w low (reversal watch)")

    # ── Relative strength vs SPY ─────────────────────────────────────────────
    lb = settings.screen_rs_lookback_days
    if spy_ret is not None and n >= lb + 1:
        base = float(close.iloc[-(lb + 1)])
        if base > 0:
            tkr_ret = (last_close / base - 1) * 100.0
            rs_excess = tkr_ret - spy_ret
            if rs_excess >= settings.screen_rs_threshold_pct:
                screens.append("STRONG_RS"); net += 1; notes.append(f"+{rs_excess:.0f}pp vs SPY")
            elif rs_excess <= -settings.screen_rs_threshold_pct:
                screens.append("WEAK_RS"); net -= 1; notes.append(f"{rs_excess:.0f}pp vs SPY")

    # ── Golden / death cross (50 / 200 SMA) ──────────────────────────────────
    k = max(1, settings.screen_cross_lookback)
    if n >= 200 + k:
        sma50  = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        now_above  = float(sma50.iloc[-1])      > float(sma200.iloc[-1])
        then_above = float(sma50.iloc[-1 - k])  > float(sma200.iloc[-1 - k])
        if now_above and not then_above:
            screens.append("GOLDEN_CROSS"); net += 1; notes.append("golden cross")
        elif then_above and not now_above:
            screens.append("DEATH_CROSS"); net -= 1; notes.append("death cross")

    if not screens:
        return None

    direction = "BULLISH" if net > 0 else ("BEARISH" if net < 0 else "BULLISH")
    return ScreenHit(
        ticker=ticker,
        direction=direction,
        screens=screens,
        last_price=round(last_close, 2),
        vol_ratio=round(vol_ratio, 2),
        rs_excess_pct=round(rs_excess, 1),
        note="; ".join(notes),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_screener() -> ScreenerContext:
    """Screen the broad liquid universe for technical setups. Fail-graceful."""
    today = date.today()
    if not settings.enable_opportunity_screener:
        return ScreenerContext(hits=[], report_date=today, summary="Opportunity screener disabled.")

    try:
        budget = {"n": max(0, settings.screen_max_fetch_per_run)}

        # SPY benchmark return (for relative strength)
        spy_ret: Optional[float] = None
        spy_df = _load_for_screen(_BENCHMARK, budget)
        lb = settings.screen_rs_lookback_days
        if spy_df is not None and not spy_df.empty and len(spy_df) >= lb + 1:
            c = spy_df["Close"].astype(float)
            base = float(c.iloc[-(lb + 1)])
            if base > 0:
                spy_ret = (float(c.iloc[-1]) / base - 1) * 100.0

        pool = _candidate_pool()
        hits: List[ScreenHit] = []
        evaluated = 0
        required = {"High", "Low", "Close", "Volume"}
        for tk in pool:
            df = _load_for_screen(tk, budget)
            if df is None or df.empty or not required.issubset(df.columns):
                continue
            evaluated += 1
            try:
                hit = _screen_one(tk, df, spy_ret)
            except Exception as e:
                logger.debug(f"[screener] {tk}: screen error — {e}")
                hit = None
            if hit:
                hits.append(hit)

        # Rank: most screens first, then strongest RS, then biggest volume surge
        hits.sort(key=lambda h: (len(h.screens), abs(h.rs_excess_pct), h.vol_ratio), reverse=True)
        top = hits[: settings.screen_max_results]
        fetched = max(0, settings.screen_max_fetch_per_run) - budget["n"]

        n_bull = sum(1 for h in top if h.direction == "BULLISH")
        n_bear = len(top) - n_bull
        if top:
            lead = ", ".join(f"{h.ticker}({'+' if h.direction == 'BULLISH' else '-'}{len(h.screens)})" for h in top[:8])
            summary = (
                f"{len(top)} setup(s) from {evaluated} liquid names "
                f"({n_bull} bullish / {n_bear} bearish): {lead}"
            )
        else:
            summary = f"No technical setups passed the screens ({evaluated} liquid names evaluated)."

        logger.info(
            f"[screener] {len(top)} setup(s) / {evaluated} evaluated / {fetched} fetched | {summary}"
        )
        return ScreenerContext(
            hits=top, universe_size=evaluated, fetched=fetched, report_date=today, summary=summary,
        )
    except Exception as e:
        logger.warning(f"[screener] failed: {e}")
        return ScreenerContext(hits=[], report_date=today, summary=f"Screener error: {e}")
