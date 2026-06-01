"""Always-open hypothetical trades — isolated reference book.

A separate category of trades configured in settings (``HYPOTHETICAL_TRADES``)
where the listed tickers are ALWAYS held in an open position (BUY or SELL).
Useful as a baseline for comparing the signal-driven book against a passive
hold (long or short) on the same names.

Stored in ``cache/hypothetical_trades.json`` — completely isolated from
``cache/trades.json``. ``get_hypothetical_performance_for_email()`` never
touches the real-trade ledger, and the real-trade engine never reads this
file. They are fully independent by design.

Each entry is one always-open position per ticker:
  - Created on first run (entry_date / entry_price = today's mark).
  - Marked-to-market on every subsequent run (current_price, return_pct).
  - Stays open indefinitely — no auto-close, no signal-decay exits.
  - If a ticker is removed from the config, its row is left in the file as
    history but is no longer updated or surfaced in the email.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

import yfinance as yf

from config import settings
from src.db import repo
from src.performance.daily_nav import compute_compound_return
from src.performance.spread import _pct_return, fmt_price


HYPOTHETICAL_FILE = Path("cache/hypothetical_trades.json")


# ---------------------------------------------------------------------------
# Asset-type classification (mirrors tracker._sector_key intent)
# ---------------------------------------------------------------------------

def _classify_asset_type(ticker: str) -> str:
    """Best-effort classification used by the spread model.

    ETF and commodity tiers in ``_dynamic_half_spread`` are tighter than the
    stock tiers, so misclassifying matters for the M2M return numbers.
    """
    tk = ticker.upper()
    if tk in {s.upper() for s in settings.commodities_list}:
        return "COMMODITY"
    if tk in {s.upper() for s in settings.sectors_list}:
        return "ETF"
    if tk in {s.upper() for s in settings.factor_list}:
        return "ETF"
    return "STOCK"


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

_LEGACY_DIRECTION = {"LONG": "BUY", "SHORT": "SELL"}


def _normalize_loaded(trades: List[dict]) -> List[dict]:
    """Migrate legacy LONG/SHORT direction values to BUY/SELL on read.

    Old entries persisted before the BUY/SELL rename stored ``direction``
    as ``LONG``/``SHORT``. The next ``_save`` call will rewrite them in the
    new vocabulary, but we normalise here so in-memory logic only ever sees
    one form.
    """
    for t in trades:
        d = t.get("direction")
        if d in _LEGACY_DIRECTION:
            t["direction"] = _LEGACY_DIRECTION[d]
        # Keep action in sync when missing or contradictory.
        if not t.get("action") and t.get("direction") in ("BUY", "SELL"):
            t["action"] = t["direction"]
    return trades


def _load() -> List[dict]:
    """Load the hypothetical book from DuckDB (source of truth).

    One-time safety net: seed from the legacy ``cache/hypothetical_trades.json``
    if the DB table is empty (mirrors ``tracker._load_trades``).
    """
    try:
        trades = repo.load_hypothetical()
        if not trades and HYPOTHETICAL_FILE.exists():
            legacy = json.loads(HYPOTHETICAL_FILE.read_text(encoding="utf-8"))
            if legacy:
                repo.save_hypothetical(legacy)
                logger.info(f"[hypothetical] Seeded DuckDB from {HYPOTHETICAL_FILE} ({len(legacy)} trades)")
                return _normalize_loaded(legacy)
        return _normalize_loaded(trades)
    except Exception as e:
        logger.warning(f"[hypothetical] Could not load: {e}")
        return []


def _save(trades: List[dict]) -> None:
    """Persist hypothetical trades to DuckDB (full-replace of the table)."""
    repo.save_hypothetical(trades)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _fetch_price(ticker: str) -> Optional[float]:
    if not settings.enable_fetch_data:
        return None
    try:
        info = yf.Ticker(ticker).fast_info
        return float(info.last_price)
    except Exception as e:
        logger.warning(f"[hypothetical] Could not fetch price for {ticker}: {e}")
        return None


def _reference_close(ticker: str) -> Optional[dict]:
    try:
        from src.data.cache import load_ohlcv
        df = load_ohlcv(ticker)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        last_close = float(df["Close"].iloc[-1])
        if last_close <= 0:
            return None
        last_date = df.index[-1].date().isoformat()
        return {"date": last_date, "close": last_close}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------

def update_hypothetical_trades() -> None:
    """Maintain the always-open hypothetical book.

    For every configured ``(ticker, action)``:
      - Open a new entry on first run (entry_price = today's mark).
      - Refresh current_price + return_pct on subsequent runs.
      - Re-anchor at today's price if the user flipped BUY ↔ SELL in config.

    Tickers no longer in the config are left in the file as history but are
    not refreshed.
    """
    pairs = settings.hypothetical_trades_list
    if not pairs:
        return

    trades = _load()
    today_iso = date.today().isoformat()
    by_ticker: Dict[str, dict] = {t["ticker"].upper(): t for t in trades}
    opened = 0
    updated = 0

    for ticker, action in pairs:
        asset_type = _classify_asset_type(ticker)
        existing = by_ticker.get(ticker)

        decision_at = _now_iso()
        price = _fetch_price(ticker)

        # ── First-time open ───────────────────────────────────────────────
        if existing is None:
            if price is None:
                logger.warning(
                    f"[hypothetical] Skipping {ticker} — could not fetch entry price"
                )
                continue
            if price <= 0:
                logger.warning(
                    f"[hypothetical] Skipping {ticker} — entry price is {fmt_price(price)}"
                )
                continue
            ref = _reference_close(ticker)
            new_entry = {
                "ticker": ticker,
                "direction": action,         # BUY/SELL — stored as direction for back-compat
                "action": action,            # BUY/SELL — re-used by daily_nav
                "type": asset_type,
                "status": "OPEN",
                "entry_date": today_iso,
                "entry_datetime": decision_at,
                "entry_price": float(price),
                "entry_ref_close": ref["close"] if ref else None,
                "entry_ref_close_date": ref["date"] if ref else None,
                "current_price": float(price),
                "current_price_datetime": decision_at,
                "return_pct": 0.0,
                "position_size_multiplier": 1.0,
            }
            trades.append(new_entry)
            by_ticker[ticker] = new_entry
            opened += 1
            logger.info(
                f"[hypothetical] Opened {action} {ticker} @ {fmt_price(price)}"
            )
            continue

        # ── User flipped BUY ↔ SELL in config: re-anchor at today's price.
        if existing.get("action") != action:
            logger.info(
                f"[hypothetical] {ticker} direction changed "
                f"{existing.get('action')} → {action}; re-anchoring entry"
            )
            if price is None or price <= 0:
                logger.warning(
                    f"[hypothetical] {ticker} direction flip skipped — no usable price"
                )
                continue
            ref = _reference_close(ticker)
            existing["direction"] = action
            existing["action"] = action
            existing["entry_date"] = today_iso
            existing["entry_datetime"] = decision_at
            existing["entry_price"] = float(price)
            existing["entry_ref_close"] = ref["close"] if ref else None
            existing["entry_ref_close_date"] = ref["date"] if ref else None
            existing["current_price"] = float(price)
            existing["current_price_datetime"] = decision_at
            existing["return_pct"] = 0.0
            opened += 1
            continue

        # Refresh asset type in case settings changed.
        existing["type"] = asset_type
        # Keep ``direction`` aligned with ``action`` so legacy LONG/SHORT
        # entries normalised on load get rewritten to BUY/SELL on next save.
        existing["direction"] = action
        if price is None:
            continue
        if price <= 0:
            logger.warning(
                f"[hypothetical] Skipping mark for {ticker} — non-positive price"
            )
            continue

        entry_price = float(existing["entry_price"])
        ret = _pct_return(action, entry_price, float(price), asset_type)
        existing["current_price"] = float(price)
        existing["current_price_datetime"] = decision_at
        existing["return_pct"] = round(ret, 3)
        updated += 1

    _save(trades)
    if opened or updated:
        logger.info(
            f"[hypothetical] Maintained {len(pairs)} always-open trade(s): "
            f"{opened} opened, {updated} refreshed"
        )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def _active_trades() -> List[dict]:
    """Return only the trades for tickers still in the current config."""
    pairs = settings.hypothetical_trades_list
    if not pairs:
        return []
    keep = {t for t, _ in pairs}
    return [t for t in _load() if t.get("ticker", "").upper() in keep]


def _segment_stats(trades: List[dict]) -> Optional[dict]:
    """Equal-weighted summary stats over the supplied trade slice."""
    if not trades:
        return None
    returns = [float(t.get("return_pct", 0.0)) for t in trades]
    wins = [r for r in returns if r > 0]
    return {
        "trades":          len(trades),
        "win_rate":        round(len(wins) / len(returns) * 100, 1) if returns else 0.0,
        "compound_return": compute_compound_return(trades) or 0.0,
        "avg_return":      round(sum(returns) / len(returns), 2),
        "best":            round(max(returns), 2),
        "worst":           round(min(returns), 2),
    }


def _trading_days_since(entry_iso: str) -> int:
    try:
        from src.performance.market_calendar import market_days_between
        return market_days_between(date.fromisoformat(entry_iso), date.today())
    except Exception:
        return 0


def get_hypothetical_performance_for_email() -> dict:
    """Render structured data for the hypothetical-trades email section.

    Returns ``{}`` when the feature is disabled. When enabled:
      {
        "trades":          [trade dict, …]  — augmented with days_held,
        "total":           {trades, win_rate, compound_return, avg, best, worst},
        "buys":            {…} or None,
        "sells":           {…} or None,
        "config":          [(ticker, action), …],  # action is BUY or SELL
      }
    """
    if not settings.enable_hypothetical_trades:
        return {}

    active = _active_trades()
    if not active:
        return {
            "trades":  [],
            "total":   None,
            "buys":    None,
            "sells":   None,
            "config":  settings.hypothetical_trades_list,
        }

    # Pretty-print augmentation
    rows: List[dict] = []
    for t in sorted(active, key=lambda x: x.get("ticker", "")):
        row = dict(t)
        row["days_held"] = _trading_days_since(row.get("entry_date", ""))
        rows.append(row)

    buys  = [t for t in active if t.get("action") == "BUY"]
    sells = [t for t in active if t.get("action") == "SELL"]

    return {
        "trades": rows,
        "total":  _segment_stats(active),
        "buys":   _segment_stats(buys),
        "sells":  _segment_stats(sells),
        "config": settings.hypothetical_trades_list,
    }