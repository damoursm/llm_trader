"""
Detect unusual options sweep activity using yfinance options chain data.

For each ticker, scans near-term expirations (≤ 60 days) and flags contracts where:
  - Volume >= 2× open interest   (sweep-like institutional flow)
  - Contract is out-of-the-money by >= 1%
  - Notional premium >= $25,000  (volume × last price × 100)

Returns bullish InsiderTrade objects for call sweeps, bearish for put sweeps.
No API key required — uses yfinance.
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import List, Optional

from loguru import logger

from src.models import InsiderTrade
from src.data.insider_trades import _notional_to_amount_range


def _safe_int(val) -> int:
    """Convert a value to int, returning 0 for None/NaN/invalid."""
    if val is None:
        return 0
    try:
        f = float(val)
        return 0 if math.isnan(f) or math.isinf(f) else int(f)
    except (TypeError, ValueError):
        return 0


def _safe_float(val) -> float:
    """Convert a value to float, returning 0.0 for None/NaN/invalid."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return 0.0


_VOL_OI_MIN_RATIO = 2.0    # volume must be >= N× open interest (2× is the standard sweep signal)
_OTM_PCT          = 0.01   # must be >= 1% out of the money
_MIN_NOTIONAL     = 25_000 # minimum $25k notional
_MAX_DTE          = 60     # max days to expiry (captures the most active sweep window)
_MAX_SWEEPS_PER_DIRECTION = 3   # cap per ticker to avoid flooding


def _parse_expiry(s: str) -> Optional[date]:
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _scan_chain(ticker: str, yt, expiry: str, current_price: float, today: date) -> List[InsiderTrade]:
    """Scan one expiry's calls and puts for sweep activity."""
    results: List[InsiderTrade] = []
    try:
        chain = yt.option_chain(expiry)
    except Exception:
        return results

    for df, opt_type, tx_type in [
        (chain.calls, "call", "unusual_call"),
        (chain.puts,  "put",  "unusual_put"),
    ]:
        if df is None or df.empty:
            continue

        direction_count = 0
        for _, row in df.iterrows():
            if direction_count >= _MAX_SWEEPS_PER_DIRECTION:
                break

            volume = _safe_int(row.get("volume"))
            oi     = _safe_int(row.get("openInterest"))
            strike = _safe_float(row.get("strike"))
            last   = _safe_float(row.get("lastPrice"))

            if volume < 10 or last <= 0 or strike <= 0:
                continue

            vol_oi_ratio = volume / max(oi, 1)
            notional     = volume * last * 100

            # OTM check
            if opt_type == "call":
                is_otm = strike >= current_price * (1 + _OTM_PCT)
            else:
                is_otm = strike <= current_price * (1 - _OTM_PCT)

            if vol_oi_ratio < _VOL_OI_MIN_RATIO or not is_otm or notional < _MIN_NOTIONAL:
                continue

            results.append(InsiderTrade(
                ticker=ticker,
                trader_name=f"Options Sweep — {opt_type.upper()}",
                trader_type="options_flow",
                role=f"Strike ${strike:.2f} | Exp {expiry}",
                transaction_type=tx_type,
                amount_range=_notional_to_amount_range(notional),
                transaction_date=today,
                disclosure_date=today,
                notes=(
                    f"Vol {volume:,} / OI {oi:,} ({vol_oi_ratio:.1f}×) | "
                    f"Last ${last:.2f} | Notional ${notional:,.0f}"
                ),
            ))
            direction_count += 1

    return results


def fetch_options_flow(tickers: List[str]) -> List[InsiderTrade]:
    """
    Scan options chains for unusual sweep activity across all tickers.
    Returns InsiderTrade objects with transaction_type 'unusual_call' or 'unusual_put'.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("[options] yfinance not installed — skipping options flow")
        return []

    today   = date.today()
    results: List[InsiderTrade] = []

    for ticker in tickers:
        try:
            yt = yf.Ticker(ticker)
            expirations = yt.options
            if not expirations:
                continue

            current_price: float = 0.0
            try:
                current_price = float(yt.fast_info.last_price or 0)
            except Exception:
                pass
            if current_price <= 0:
                continue

            near_exps = [
                e for e in expirations
                if (exp_d := _parse_expiry(e)) and 0 <= (exp_d - today).days <= _MAX_DTE
            ]
            for expiry in near_exps[:4]:   # at most 4 expirations per ticker
                results.extend(_scan_chain(ticker, yt, expiry, current_price, today))

        except Exception as e:
            logger.warning(f"[options] {ticker} scan failed: {e}")

    call_sweeps = sum(1 for r in results if r.transaction_type == "unusual_call")
    put_sweeps  = sum(1 for r in results if r.transaction_type == "unusual_put")
    logger.info(
        f"[options] {len(tickers)} tickers scanned | "
        f"{call_sweeps} call sweep(s), {put_sweeps} put sweep(s)"
    )
    return results
