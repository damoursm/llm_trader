"""Live pattern outcome registry — closes the loop on pattern recognition.

The per-ticker pattern library in ``pattern_recognition.py`` is a *synthetic*
backtest: it scans each ticker's own price history with a sliding window and
records "if this pattern formed, did the 10-day forward return move the
expected direction?". That's a useful prior but it doesn't know whether the
*system* actually capitalised on the pattern when it appeared — the system
filters patterns through 9 other signals, a macro regime gate, an earnings
blackout, a sector cap, and a confidence threshold before opening a trade.

This module records what happened to the *real* trades the system took when
each pattern was active at entry, so the pattern score can adapt over time
toward "this pattern works for *this* system" rather than the synthetic
ground-truth alone.

Storage
───────
A single JSON file ``cache/pattern_registry.json`` with this shape::

    {
      "schema_version": 1,
      "last_updated":   "2026-05-18T10:00:00+00:00",
      "patterns": {
        "double_bottom": {
          "n_trades": 5, "n_wins": 4, "win_rate": 0.80, "avg_return": 8.3,
          "buys":  {"n": 5, "wins": 4, "win_rate": 0.80, "avg_return": 8.3},
          "sells": {"n": 0, "wins": 0, "win_rate": null, "avg_return": null},
          "trades": [
            {"ticker": "MSFT", "entry": "2026-04-15", "exit": "2026-04-22",
             "action": "BUY", "ret_pct": 5.2, "won": true,
             "exit_reason": "holding_period"},
            ...
          ]
        },
        ...
      },
      "by_ticker_pattern": {
        "MSFT|double_bottom": {"n": 2, "wins": 2, "win_rate": 1.0, "avg_return": 6.4}
      }
    }

Usage
─────
- ``record_outcome(trade)``: append a closed trade to the registry; idempotent
  per (ticker, entry_date, exit_date) so re-running update_open_trades doesn't
  double-count.
- ``pattern_stats(pattern_name)``: global aggregates for the pattern.
- ``ticker_pattern_stats(ticker, pattern_name)``: per-(ticker, pattern) aggregates.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict

from loguru import logger


REGISTRY_PATH = Path("cache/pattern_registry.json")
SCHEMA_VERSION = 2   # bumped: added pattern_accuracy alongside trade win_rate

_EMPTY_DIR_BUCKET = {
    "n":            0,
    "wins":         0,           # trade outcomes (return_pct > 0)
    "win_rate":     None,        # trade outcome rate
    "pattern_correct": 0,        # times the pattern's inherent direction proved right
    "pattern_accuracy": None,    # pattern_correct / n
    "avg_return":   None,
}


def _pattern_was_correct(pattern_name: str, action: str, ret_pct: float) -> bool:
    """Did the price move in the pattern's inherent direction?

    For a bullish pattern (+1): correct when price went up (BUY won OR SELL lost).
    For a bearish pattern (-1): correct when price went down (SELL won OR BUY lost).

    Equivalent: ``(action_sign * outcome_sign) == inherent_direction``.
    """
    from src.signals.pattern_recognition import _PATTERN_DIR  # lazy to avoid cycle
    inherent = _PATTERN_DIR.get(pattern_name, 0)
    if inherent == 0:
        return False
    action_sign  = +1 if action == "BUY" else -1
    outcome_sign = +1 if ret_pct > 0 else -1
    return (action_sign * outcome_sign) == inherent


# ── Persistence ──────────────────────────────────────────────────────────────

def load_registry() -> dict:
    """Return the registry from disk, or an empty skeleton when missing/corrupt."""
    if not REGISTRY_PATH.exists():
        return _empty_registry()
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            logger.warning(
                f"[pattern_registry] schema mismatch ({data.get('schema_version')} != "
                f"{SCHEMA_VERSION}); reinitialising"
            )
            return _empty_registry()
        # Ensure required sub-keys exist
        data.setdefault("patterns", {})
        data.setdefault("by_ticker_pattern", {})
        return data
    except Exception as e:
        logger.warning(f"[pattern_registry] load failed ({e}) — starting fresh")
        return _empty_registry()


def save_registry(reg: dict) -> None:
    """Persist the registry, creating cache/ if needed."""
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    reg["last_updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"[pattern_registry] save failed: {e}")


def _empty_registry() -> dict:
    return {
        "schema_version":     SCHEMA_VERSION,
        "last_updated":       datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "patterns":           {},
        "by_ticker_pattern":  {},
    }


# ── Aggregation helpers ──────────────────────────────────────────────────────

def _update_bucket(bucket: dict, ret_pct: float, won: bool, pattern_correct: bool) -> dict:
    """Update an aggregate bucket in place.

    Tracks both ``wins`` (system trade outcome) and ``pattern_correct`` (the
    pattern's directional prediction was right). These can disagree when the
    system trades against the pattern's inherent direction (contrarian override
    by the other 9 signals).
    """
    n_prev   = int(bucket.get("n", 0))
    wins_prev = int(bucket.get("wins", 0))
    pc_prev  = int(bucket.get("pattern_correct", 0))
    avg_prev  = float(bucket.get("avg_return") or 0.0)
    n_new = n_prev + 1
    wins_new = wins_prev + (1 if won else 0)
    pc_new = pc_prev + (1 if pattern_correct else 0)
    avg_new = avg_prev + (ret_pct - avg_prev) / n_new
    bucket["n"]                = n_new
    bucket["wins"]             = wins_new
    bucket["win_rate"]         = round(wins_new / n_new, 4)
    bucket["pattern_correct"]  = pc_new
    bucket["pattern_accuracy"] = round(pc_new / n_new, 4)
    bucket["avg_return"]       = round(avg_new, 4)
    return bucket


def _trade_key(trade: dict) -> str:
    """Stable per-trade identity for idempotent registration."""
    return f"{trade.get('ticker','?')}|{trade.get('entry_date','?')}|{trade.get('exit_date','?')}"


# ── Public API ───────────────────────────────────────────────────────────────

def record_outcome(trade: dict, reg: Optional[dict] = None) -> bool:
    """Register the outcome of a single closed trade against its entry pattern.

    Returns True when the registry was modified (new outcome recorded), False
    when the trade was skipped (no pattern_at_entry, already registered, or
    open). Caller is responsible for persisting via ``save_registry`` after a
    batch of updates.

    A trade is considered "won" when ``return_pct > 0`` — the spread-adjusted
    figure stored by the tracker. Strictly positive matches the existing
    win_rate convention in the rest of the system.
    """
    if reg is None:
        # If called solo, load-modify-save in one go
        reg = load_registry()
        own_save = True
    else:
        own_save = False

    if trade.get("status") != "CLOSED":
        return False
    pattern = trade.get("pattern_at_entry")
    if not pattern:
        return False
    ret = trade.get("return_pct")
    if ret is None:
        return False

    ticker = (trade.get("ticker") or "").upper()
    action = trade.get("action", "BUY")
    key = _trade_key(trade)

    # Idempotency: skip if this trade is already in the global pattern's trades list
    p_entry = reg["patterns"].setdefault(pattern, {
        "n_trades":          0,
        "n_wins":            0,
        "win_rate":          None,
        "n_pattern_correct": 0,
        "pattern_accuracy":  None,
        "avg_return":        None,
        "buys":              dict(_EMPTY_DIR_BUCKET),
        "sells":             dict(_EMPTY_DIR_BUCKET),
        "trades":            [],
    })
    if any(_trade_key(t) == key for t in p_entry.get("trades", [])):
        return False

    won = float(ret) > 0.0
    pattern_correct = _pattern_was_correct(pattern, action, float(ret))

    # Global pattern aggregates
    n_prev    = int(p_entry.get("n_trades", 0))
    wins_prev = int(p_entry.get("n_wins", 0))
    pc_prev   = int(p_entry.get("n_pattern_correct", 0))
    avg_prev  = float(p_entry.get("avg_return") or 0.0)
    n_new     = n_prev + 1
    wins_new  = wins_prev + (1 if won else 0)
    pc_new    = pc_prev + (1 if pattern_correct else 0)
    avg_new   = avg_prev + (float(ret) - avg_prev) / n_new
    p_entry["n_trades"]          = n_new
    p_entry["n_wins"]            = wins_new
    p_entry["win_rate"]          = round(wins_new / n_new, 4)
    p_entry["n_pattern_correct"] = pc_new
    p_entry["pattern_accuracy"]  = round(pc_new / n_new, 4)
    p_entry["avg_return"]        = round(avg_new, 4)

    # Per-direction aggregates
    dir_key = "buys" if action == "BUY" else "sells"
    p_entry[dir_key] = _update_bucket(
        p_entry.get(dir_key) or dict(_EMPTY_DIR_BUCKET),
        float(ret), won, pattern_correct,
    )

    # Per-(ticker, pattern) aggregates
    tp_key = f"{ticker}|{pattern}"
    tp = reg["by_ticker_pattern"].get(tp_key) or dict(_EMPTY_DIR_BUCKET)
    reg["by_ticker_pattern"][tp_key] = _update_bucket(tp, float(ret), won, pattern_correct)

    # Append the per-trade detail (capped — keep the most recent 200 per pattern)
    p_entry["trades"].append({
        "ticker":           ticker,
        "entry":            trade.get("entry_date"),
        "exit":             trade.get("exit_date"),
        "action":           action,
        "ret_pct":          round(float(ret), 3),
        "won":              won,
        "pattern_correct":  pattern_correct,
        "exit_reason":      trade.get("exit_reason"),
        "score_at_entry":   trade.get("pattern_score_at_entry"),
    })
    if len(p_entry["trades"]) > 200:
        p_entry["trades"] = p_entry["trades"][-200:]

    if own_save:
        save_registry(reg)
    return True


def record_batch(trades: list) -> int:
    """Register every closed-with-pattern trade in ``trades`` that isn't already
    in the registry. Returns the count of new outcomes recorded.

    Idempotent: re-running on the same ledger is a no-op.
    """
    reg = load_registry()
    added = 0
    for t in trades:
        if record_outcome(t, reg=reg):
            added += 1
    if added:
        save_registry(reg)
        logger.info(f"[pattern_registry] registered {added} new pattern outcome(s)")
    return added


def pattern_stats(pattern_name: str) -> Optional[dict]:
    """Global aggregates for ``pattern_name``, or None when no entry exists."""
    if not pattern_name:
        return None
    reg = load_registry()
    p = reg.get("patterns", {}).get(pattern_name)
    if not p or p.get("n_trades", 0) <= 0:
        return None
    return {
        "n_trades":          int(p["n_trades"]),
        "n_wins":            int(p["n_wins"]),
        "win_rate":          p.get("win_rate"),
        "n_pattern_correct": int(p.get("n_pattern_correct", 0)),
        "pattern_accuracy":  p.get("pattern_accuracy"),
        "avg_return":        p.get("avg_return"),
        "buys":              p.get("buys")  or dict(_EMPTY_DIR_BUCKET),
        "sells":             p.get("sells") or dict(_EMPTY_DIR_BUCKET),
    }


def ticker_pattern_stats(ticker: str, pattern_name: str) -> Optional[dict]:
    """Per-(ticker, pattern) aggregates, or None when no entry exists."""
    if not pattern_name or not ticker:
        return None
    reg = load_registry()
    tp = reg.get("by_ticker_pattern", {}).get(f"{ticker.upper()}|{pattern_name}")
    if not tp or tp.get("n", 0) <= 0:
        return None
    return dict(tp)


def all_pattern_stats() -> Dict[str, dict]:
    """Return ``{pattern_name: stats}`` for every pattern in the registry."""
    reg = load_registry()
    return {
        p: {
            "n_trades":          int(v.get("n_trades", 0)),
            "n_wins":            int(v.get("n_wins", 0)),
            "win_rate":          v.get("win_rate"),
            "n_pattern_correct": int(v.get("n_pattern_correct", 0)),
            "pattern_accuracy":  v.get("pattern_accuracy"),
            "avg_return":        v.get("avg_return"),
            "buys":              v.get("buys")  or dict(_EMPTY_DIR_BUCKET),
            "sells":             v.get("sells") or dict(_EMPTY_DIR_BUCKET),
        }
        for p, v in reg.get("patterns", {}).items()
        if int(v.get("n_trades", 0)) > 0
    }
