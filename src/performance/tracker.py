"""Track algorithm performance by recording BUY/SELL entries and calculating returns.

Trades are stored in cache/trades.json. Each daily run:
  1. Opens new trades for today's BUY/SELL recommendations.
  2. Fetches current prices for all open trades and marks unrealised P&L.
  3. Auto-closes trades that have been held for HOLDING_DAYS trading days.
  4. Logs a full performance summary.
"""

import json
import yfinance as yf
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from src.models import Recommendation
from config import settings

TRADES_FILE = Path("cache/trades.json")
HOLDING_DAYS = 5   # auto-close after this many calendar days
SPREAD_PCT   = 0.10  # round-trip bid-ask spread in % (5 bps each way)
               # Applied as a penalty: half at entry (you buy at the ask / short at the bid),
               # half at exit (you sell at the bid / cover at the ask).


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_trades() -> List[dict]:
    if not TRADES_FILE.exists():
        return []
    try:
        return json.loads(TRADES_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"[tracker] Could not load trades: {e}")
        return []


def _save_trades(trades: List[dict]) -> None:
    TRADES_FILE.parent.mkdir(exist_ok=True)
    TRADES_FILE.write_text(json.dumps(trades, indent=2, default=str), encoding="utf-8")


def _fetch_price(ticker: str) -> Optional[float]:
    if not settings.enable_market_data:
        return None
    try:
        info = yf.Ticker(ticker).fast_info
        return float(info.last_price)
    except Exception as e:
        logger.warning(f"[tracker] Could not fetch price for {ticker}: {e}")
        return None


def _days_held(entry_date: str) -> int:
    try:
        return (date.today() - date.fromisoformat(entry_date)).days
    except Exception:
        return 0


def _pct_return(action: str, entry: float, current: float) -> float:
    """Positive = profitable regardless of direction.

    Applies a round-trip bid-ask spread penalty (SPREAD_PCT).
    BUY:  you paid the ask at entry  (+half spread),
          you receive the bid at exit (-half spread).
    SELL: you shorted at the bid at entry  (-half spread),
          you covered at the ask at exit   (+half spread).
    Net effect is identical for both directions: raw return minus SPREAD_PCT.
    """
    half = SPREAD_PCT / 2 / 100          # fractional half-spread
    if action == "BUY":
        effective_entry = entry  * (1 + half)
        effective_exit  = current * (1 - half)
        return (effective_exit - effective_entry) / effective_entry * 100
    else:  # SELL = short position
        effective_entry = entry  * (1 - half)
        effective_exit  = current * (1 + half)
        return (effective_entry - effective_exit) / effective_entry * 100


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record_new_trades(recommendations: List[Recommendation]) -> None:
    """Open a new trade for each BUY/SELL recommendation not already open today."""
    trades = _load_trades()
    today = date.today().isoformat()

    # Index open trades to avoid duplicates on the same day
    open_today = {t["ticker"] for t in trades if t["entry_date"] == today and t["status"] == "OPEN"}

    new_count = 0
    for rec in recommendations:
        if rec.action not in ("BUY", "SELL"):
            continue
        if rec.ticker in open_today:
            continue

        price = _fetch_price(rec.ticker)
        if price is None:
            logger.warning(f"[tracker] Skipping {rec.ticker} — could not fetch entry price")
            continue

        trades.append({
            "ticker": rec.ticker,
            "type": rec.type,
            "action": rec.action,
            "direction": rec.direction,
            "confidence": rec.confidence,
            "entry_date": today,
            "entry_price": round(price, 4),
            "rationale": rec.rationale,
            "current_price": round(price, 4),
            "return_pct": 0.0,
            "days_held": 0,
            "exit_date": None,
            "exit_price": None,
            "status": "OPEN",
        })
        new_count += 1
        logger.info(f"[tracker] Opened {rec.action} {rec.ticker} @ {price:.2f}")

    _save_trades(trades)
    logger.info(f"[tracker] {new_count} new trade(s) recorded")


def update_open_trades() -> None:
    """Refresh current prices and P&L for all open trades; close expired ones."""
    trades = _load_trades()
    today = date.today().isoformat()
    updated = 0
    closed = 0

    for trade in trades:
        if trade["status"] != "OPEN":
            continue

        price = _fetch_price(trade["ticker"])
        if price is None:
            continue

        days = _days_held(trade["entry_date"])
        ret = _pct_return(trade["action"], trade["entry_price"], price)

        trade["current_price"] = round(price, 4)
        trade["return_pct"] = round(ret, 3)
        trade["days_held"] = days
        updated += 1

        # Auto-close after HOLDING_DAYS
        if days >= HOLDING_DAYS:
            trade["status"] = "CLOSED"
            trade["exit_date"] = today
            trade["exit_price"] = round(price, 4)
            closed += 1
            logger.info(
                f"[tracker] Closed {trade['action']} {trade['ticker']} | "
                f"entry={trade['entry_price']:.2f} exit={price:.2f} "
                f"return={ret:+.2f}% over {days}d"
            )

    _save_trades(trades)
    logger.info(f"[tracker] Updated {updated} open trade(s), closed {closed}")


def log_performance_summary() -> None:
    """Log a full performance breakdown to the log file."""
    trades = _load_trades()
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
            logger.info(
                f"    {t['action']:<4} {t['ticker']:<6} | "
                f"entry={t['entry_price']:.2f}  now={t['current_price']:.2f}  "
                f"P&L={t['return_pct']:+.2f}%  ({t['days_held']}d held)"
            )

    # Closed positions
    if closed_trades:
        returns = [t["return_pct"] for t in closed_trades]
        wins = [r for r in returns if r > 0]
        win_rate = len(wins) / len(returns) * 100
        avg_return = sum(returns) / len(returns)
        best = max(returns)
        worst = min(returns)

        logger.info(f"  Closed positions ({len(closed_trades)})")
        for t in sorted(closed_trades, key=lambda x: x["exit_date"] or "", reverse=True):
            logger.info(
                f"    {t['action']:<4} {t['ticker']:<6} | "
                f"entry={t['entry_price']:.2f}  exit={t['exit_price']:.2f}  "
                f"return={t['return_pct']:+.2f}%  ({t['days_held']}d)"
            )

        logger.info(f"  --- Stats ({len(closed_trades)} closed trades) ---")
        logger.info(f"    Win rate:     {win_rate:.1f}%")
        logger.info(f"    Avg return:   {avg_return:+.2f}%")
        logger.info(f"    Best trade:   {best:+.2f}%")
        logger.info(f"    Worst trade:  {worst:+.2f}%")

    logger.info("=" * 60)


def get_performance_for_email() -> dict:
    """Return structured performance data for inclusion in the email report."""
    trades = _load_trades()
    open_trades = [t for t in trades if t["status"] == "OPEN"]
    closed_trades = [t for t in trades if t["status"] == "CLOSED"]

    stats = {}
    if closed_trades:
        returns = [t["return_pct"] for t in closed_trades]
        stats = {
            "total_closed": len(closed_trades),
            "win_rate": round(len([r for r in returns if r > 0]) / len(returns) * 100, 1),
            "avg_return": round(sum(returns) / len(returns), 2),
            "best": round(max(returns), 2),
            "worst": round(min(returns), 2),
        }

    return {
        "open_trades": sorted(open_trades, key=lambda x: x["entry_date"], reverse=True),
        "closed_trades": sorted(closed_trades, key=lambda x: x["exit_date"] or "", reverse=True)[:10],
        "stats": stats,
    }
