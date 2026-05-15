"""Track algorithm performance by recording BUY/SELL entries and calculating returns.

Trades are stored in cache/trades.json. Each daily run:
  1. Opens new trades for today's BUY/SELL recommendations.
  2. Fetches current prices for all open trades and marks unrealised P&L.
  3. Auto-closes trades that have been held for HOLDING_DAYS trading days.
  4. Logs a full performance summary.
"""

import json
import yfinance as yf
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from src.models import Recommendation
from config import settings

TRADES_FILE = Path("cache/trades.json")
HOLDING_DAYS = 5   # auto-close after this many trading days (Mon–Fri, weekends excluded)

# ── Position sizing ───────────────────────────────────────────────────────────
# Confidence-scaled Kelly-inspired tiers: allocate more capital to higher conviction.
SIZE_TIER_HALF  = 0.85   # below this → 1.0× (baseline)
SIZE_TIER_FULL  = 0.92   # 0.85–0.92 → 1.5×; above → 2.0×
SECTOR_CAP      = 3.0    # max total size units per sector in open positions



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
    """Fetch the latest available price.

    NOTE: the pipeline runs at 8:00 AM ET — 90 min before market open.
    At that time fast_info.last_price returns the previous close or a thin
    pre-market print, NOT the actual next-open tradeable price.  Performance
    numbers will therefore understate slippage on gap-open news catalysts.
    """
    if not settings.enable_fetch_data:
        return None
    try:
        info = yf.Ticker(ticker).fast_info
        return float(info.last_price)
    except Exception as e:
        logger.warning(f"[tracker] Could not fetch price for {ticker}: {e}")
        return None


def _now_iso() -> str:
    """Return the current wall-clock instant as a UTC ISO 8601 string.

    Stored on every entry/exit so the price used for the trade is anchored to
    the exact moment it was fetched — no ambiguity about whether the value
    is the prior close, a pre-market print, or an intraday mark.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _trading_days_held(entry_date: str) -> int:
    """Count weekdays (Mon–Fri) between entry_date (exclusive) and today (inclusive)."""
    try:
        start = date.fromisoformat(entry_date)
        end = date.today()
        count = 0
        current = start + timedelta(days=1)
        while current <= end:
            if current.weekday() < 5:  # 0–4 = Mon–Fri
                count += 1
            current += timedelta(days=1)
        return count
    except Exception:
        return 0


def fmt_price(p) -> str:
    """Format a price with enough decimal places to show meaningful digits.

    Handles sub-penny stocks/warrants (e.g. 0.003 → '$0.0030') without
    rounding to '$0.00'.
    """
    if p is None:
        return "N/A"
    try:
        p = float(p)
    except (TypeError, ValueError):
        return str(p)
    if p >= 1.0:
        return f"{p:.2f}"
    if p >= 0.01:
        return f"{p:.4f}"
    return f"{p:.6f}"


def _dynamic_half_spread(price: float, asset_type: str = "STOCK") -> float:
    """One-way bid-ask half-spread as a fraction (not %).

    Realistic tiers (conservative-but-not-pessimistic estimates):
      ETF       → 1 bp  (sector SPDR ETFs, highly liquid)
      COMMODITY → price ≥ $100: 1.5 bp (GLD/GDX); else 3 bp (SLV/CPER)
      STOCK     → price-tiered:
                   ≥ $50        →   2 bp   (large-cap, tight market)
                   $10–$50      →   4 bp   (mid-cap)
                   $1–$10       →  12.5 bp (small-cap / penny approach)
                   $0.10–$1     →  37.5 bp (micro-cap)
                   $0.01–$0.10  → 100 bp   (penny stock)
                   < $0.01      → 250 bp   (sub-penny / warrant)
    1 bp = 0.0001 fractional.
    """
    if asset_type == "ETF":
        return 0.0001
    if asset_type == "COMMODITY":
        return 0.00015 if price >= 100 else 0.0003
    # STOCK — price-tiered
    if price >= 50:
        return 0.0002
    if price >= 10:
        return 0.0004
    if price >= 1:
        return 0.00125
    if price >= 0.10:
        return 0.00375
    if price >= 0.01:
        return 0.0100
    return 0.0250


def _pct_return(action: str, entry: float, current: float, asset_type: str = "STOCK") -> float:
    """Positive = profitable regardless of direction.

    Applies a realistic dynamic bid-ask spread based on asset type and price tier.
    Entry and exit spreads are computed separately from their respective prices.
    BUY:  paid the ask at entry (+half), receive bid at exit (-half).
    SELL: shorted at the bid at entry (-half), covered at the ask at exit (+half).
    """
    entry_half = _dynamic_half_spread(entry, asset_type)
    exit_half  = _dynamic_half_spread(current, asset_type)
    if action == "BUY":
        effective_entry = entry   * (1 + entry_half)
        effective_exit  = current * (1 - exit_half)
        return (effective_exit - effective_entry) / effective_entry * 100
    else:  # SELL = short position
        effective_entry = entry   * (1 - entry_half)
        effective_exit  = current * (1 + exit_half)
        return (effective_entry - effective_exit) / effective_entry * 100


# ── Method attribution ────────────────────────────────────────────────────────
_ALL_METHODS = ("news", "tech", "insider", "put_call", "max_pain", "oi_skew", "vwap", "pattern", "momentum", "money_flow")
_METHOD_AGREE_THRESHOLD = 0.10   # minimum |score| to count a method as "having a view"

# Category groupings: how methods map to higher-level signal families
METHOD_CATEGORIES: Dict[str, List[str]] = {
    "Sentiment":   ["news"],
    "Technical":   ["tech", "vwap", "pattern", "momentum", "money_flow"],
    "Smart Money": ["insider"],
    "Options":     ["put_call", "max_pain", "oi_skew"],
}

# Human-readable method labels for reports
METHOD_LABELS: Dict[str, str] = {
    "news":     "News Sentiment",
    "tech":     "Technical Analysis",
    "insider":  "Smart Money / Insider",
    "put_call": "Put/Call Ratio",
    "max_pain": "Max Pain (GEX)",
    "oi_skew":  "OI Skew",
    "vwap":     "VWAP Distance",
    "pattern":  "Pattern Recognition",
    "momentum":   "Price Momentum",
    "money_flow": "Money Flow (MFI+CMF+OBV)",
}


def _method_scores_from_signal(ticker: str, direction: str, signals_by_ticker: Optional[dict]) -> dict:
    """Extract per-method scores from the TickerSignal for a given ticker."""
    empty = {m: 0.0 for m in _ALL_METHODS}
    if signals_by_ticker is None:
        return empty
    sig = signals_by_ticker.get(ticker)
    if sig is None:
        return empty
    return {
        "news":      sig.sentiment_score,
        "tech":      sig.technical_score,
        "insider":   sig.insider_score,
        "put_call":  sig.put_call_score,
        "max_pain":  sig.max_pain_score,
        "oi_skew":   sig.oi_skew_score,
        "vwap":      sig.vwap_score,
        "pattern":    getattr(sig, "pattern_score", 0.0),
        "momentum":   getattr(sig, "momentum_score", 0.0),
        "money_flow": getattr(sig, "money_flow_score", 0.0),
    }


def _methods_agreeing(scores: dict, direction: str) -> List[str]:
    """Return list of method names whose score agrees with the trade direction."""
    sign = 1 if direction == "BULLISH" else -1
    return [m for m, v in scores.items() if v * sign > _METHOD_AGREE_THRESHOLD]


def _dominant_method(scores: dict, direction: str) -> str:
    """Return the method with the strongest agreement with the trade direction."""
    sign = 1 if direction == "BULLISH" else -1
    agreed = {m: v * sign for m, v in scores.items() if v * sign > _METHOD_AGREE_THRESHOLD}
    return max(agreed, key=agreed.get) if agreed else "none"


def _compute_nav_compound(trades: list) -> Optional[float]:
    """Capital-weighted, time-weighted compound return over *trades* (in %).

    Backed by ``daily_nav.compute_compound_return``: every daily return is
    derived from the actual cached OHLCV close for that day, not from a
    geometric split of the trade's total return. 100% deterministic — same
    input trades always produce the same output.

    Accepts trade dicts directly (the previous (entry_str, exit_str, ret_pct,
    weight) tuple form was lossy — it threw away ticker/action/prices needed
    to read real daily marks).  Returns ``None`` when no daily-return event
    can be produced.
    """
    from src.performance.daily_nav import compute_compound_return

    return compute_compound_return(trades or [])


def _compute_segment_stats(trades: List[dict]) -> Optional[dict]:
    """Compute all standard performance metrics for a slice of trades.

    Compound return is computed from the actual daily price walk (no
    interpolation) via ``_compute_nav_compound``, which accepts trade dicts
    directly so the OHLCV cache can be consulted per ticker.
    """
    if not trades:
        return None
    returns = [t["return_pct"] for t in trades]
    multipliers = [t.get("position_size_multiplier", 1.0) for t in trades]
    total_mul = sum(multipliers)
    weighted_avg = (
        sum(r * m for r, m in zip(returns, multipliers)) / total_mul
        if total_mul else 0.0
    )
    wins = [r for r in returns if r > 0]
    return {
        "trades":          len(trades),
        "win_rate":        round(len(wins) / len(returns) * 100, 1),
        "compound_return": _compute_nav_compound(trades) or 0.0,
        "avg_return":      round(sum(returns) / len(returns), 2),
        "wtd_avg_return":  round(weighted_avg, 2),
        "best":            round(max(returns), 2),
        "worst":           round(min(returns), 2),
    }


_ASSET_TYPE_LABELS: Dict[str, str] = {
    "STOCK":     "Stocks only",
    "ETF":       "ETFs only",
    "COMMODITY": "Commodities only",
}


def _compute_performance_table(trades: List[dict]) -> List[dict]:
    """Build unified breakdown rows: total → asset types → signal methods.

    Accepts both closed and open trades; open trades use their current M2M return_pct
    (maintained by update_open_trades() each run), equivalent to a hypothetical exit.
    """
    rows: List[dict] = []

    # Total row
    stats = _compute_segment_stats(trades)
    if stats:
        rows.append({"label": "All Trades", "group": "total", **stats})

    # Asset-type rows
    for type_key, label in _ASSET_TYPE_LABELS.items():
        subset = [t for t in trades if t.get("type") == type_key]
        stats = _compute_segment_stats(subset)
        if stats:
            rows.append({"label": label, "group": "asset", **stats})

    # Direction rows (long vs short)
    longs  = [t for t in trades if t.get("action") == "BUY"]
    shorts = [t for t in trades if t.get("action") == "SELL"]
    for subset, label in ((longs, "Longs only (BUY)"), (shorts, "Shorts only (SELL)")):
        stats = _compute_segment_stats(subset)
        if stats:
            rows.append({"label": label, "group": "direction", **stats})

    # Signal-method rows (attribution-enabled trades only)
    attributed = [t for t in trades if t.get("methods_agreeing")]
    for method in _ALL_METHODS:
        subset = [t for t in attributed if method in t.get("methods_agreeing", [])]
        if len(subset) < 2:
            continue
        stats = _compute_segment_stats(subset)
        if stats:
            rows.append({"label": METHOD_LABELS[method], "group": "method", **stats})

    return rows


def _compute_method_stats(closed_trades: List[dict]) -> dict:
    """
    For each method, compute win rate and avg return across all closed trades
    where that method agreed with the trade direction at entry.

    Returns a dict keyed by method name → {trades, win_rate, avg_return, weighted_avg_return}.
    Only includes methods with ≥ 3 attributed trades (too few → noise).
    """
    MIN_TRADES = 3
    buckets: Dict[str, List[tuple]] = {m: [] for m in _ALL_METHODS}

    for t in closed_trades:
        method_scores = t.get("method_scores", {})
        methods_agreed = t.get("methods_agreeing", [])
        ret = t.get("return_pct", 0.0)
        mul = t.get("position_size_multiplier", 1.0)
        for m in methods_agreed:
            if m in buckets:
                buckets[m].append((ret, mul))

    result = {}
    for method, entries in buckets.items():
        if len(entries) < MIN_TRADES:
            continue
        returns = [r for r, _ in entries]
        muls    = [m for _, m in entries]
        total_mul = sum(muls)
        wins = [r for r in returns if r > 0]
        weighted_avg = sum(r * m for r, m in zip(returns, muls)) / total_mul if total_mul else 0.0
        result[method] = {
            "trades":              len(entries),
            "win_rate":            round(len(wins) / len(returns) * 100, 1),
            "avg_return":          round(sum(returns) / len(returns), 2),
            "weighted_avg_return": round(weighted_avg, 2),
        }

    return result


def _compute_category_stats(closed_trades: List[dict]) -> dict:
    """
    Roll per-method attribution up into category groups.
    A trade contributes to a category if ANY method in that category agreed.
    Returns dict keyed by category name → {trades, win_rate, avg_return, weighted_avg_return}.
    """
    MIN_TRADES = 2
    buckets: Dict[str, List[tuple]] = {cat: [] for cat in METHOD_CATEGORIES}

    for t in closed_trades:
        methods_agreed = set(t.get("methods_agreeing", []))
        if not methods_agreed:
            continue
        ret = t.get("return_pct", 0.0)
        mul = t.get("position_size_multiplier", 1.0)
        seen_cats: set = set()
        for cat, members in METHOD_CATEGORIES.items():
            if any(m in methods_agreed for m in members) and cat not in seen_cats:
                buckets[cat].append((ret, mul))
                seen_cats.add(cat)

    result = {}
    for cat, entries in buckets.items():
        if len(entries) < MIN_TRADES:
            continue
        returns  = [r for r, _ in entries]
        muls     = [m for _, m in entries]
        total_m  = sum(muls)
        wins     = [r for r in returns if r > 0]
        w_avg    = sum(r * m for r, m in zip(returns, muls)) / total_m if total_m else 0.0
        result[cat] = {
            "trades":              len(entries),
            "win_rate":            round(len(wins) / len(returns) * 100, 1),
            "avg_return":          round(sum(returns) / len(returns), 2),
            "weighted_avg_return": round(w_avg, 2),
        }

    return result


def _compute_convergence_stats(closed_trades: List[dict]) -> dict:
    """
    Group closed trades by how many methods agreed at entry.
    Directly tests whether the convergence multiplier (1.25×/0.60×) is justified.
    Returns dict keyed by convergence label → {trades, win_rate, avg_return}.
    """
    def _bucket(n: int) -> str:
        if n >= 4: return "4+ methods"
        if n == 3: return "3 methods"
        if n == 2: return "2 methods"
        return "1 method"

    buckets: Dict[str, List[float]] = {}

    for t in closed_trades:
        agreed = t.get("methods_agreeing", [])
        if not agreed:   # no attribution data (legacy trade)
            continue
        label = _bucket(len(agreed))
        buckets.setdefault(label, []).append(t.get("return_pct", 0.0))

    order = ["4+ methods", "3 methods", "2 methods", "1 method"]
    result = {}
    for label in order:
        returns = buckets.get(label, [])
        if len(returns) < 2:
            continue
        wins = [r for r in returns if r > 0]
        result[label] = {
            "trades":     len(returns),
            "win_rate":   round(len(wins) / len(returns) * 100, 1),
            "avg_return": round(sum(returns) / len(returns), 2),
        }

    return result


def _compute_dominant_stats(closed_trades: List[dict]) -> dict:
    """
    Group closed trades by the method that had the strongest agreement at entry.
    Shows which single signal type best predicts profitable outcomes.
    Returns dict keyed by method name → {trades, win_rate, avg_return}.
    """
    MIN_TRADES = 2
    buckets: Dict[str, List[float]] = {}

    for t in closed_trades:
        dom = t.get("dominant_method", "none")
        if dom == "none":
            continue
        buckets.setdefault(dom, []).append(t.get("return_pct", 0.0))

    result = {}
    for method, returns in buckets.items():
        if len(returns) < MIN_TRADES:
            continue
        wins = [r for r in returns if r > 0]
        result[method] = {
            "trades":     len(returns),
            "win_rate":   round(len(wins) / len(returns) * 100, 1),
            "avg_return": round(sum(returns) / len(returns), 2),
        }

    return result


def _compute_confidence_ranked(closed_trades: List[dict]) -> List[dict]:
    """Return closed trades sorted by confidence descending with rolling cumulative stats.

    Each row adds a cumulative_avg and cumulative_win_rate that reflect the
    portfolio return if you had *only* taken the top-N most confident signals.
    """
    sorted_trades = sorted(
        closed_trades,
        key=lambda t: t.get("confidence", 0.0),
        reverse=True,
    )
    rows = []
    running_sum = 0.0
    running_wins = 0
    for i, t in enumerate(sorted_trades, 1):
        ret = t.get("return_pct", 0.0)
        running_sum += ret
        if ret > 0:
            running_wins += 1
        rows.append({
            "rank":                 i,
            "ticker":               t["ticker"],
            "action":               t["action"],
            "confidence":           t.get("confidence", 0.0),
            "return_pct":           ret,
            "entry_date":           t.get("entry_date", ""),
            "exit_date":            t.get("exit_date", ""),
            "cumulative_avg":       round(running_sum / i, 2),
            "cumulative_win_rate":  round(running_wins / i * 100, 1),
        })
    return rows


# ---------------------------------------------------------------------------
# Position sizing helpers
# ---------------------------------------------------------------------------

def _position_multiplier(confidence: float) -> float:
    """Map confidence to a position-size multiplier using three tiers.

    Tier    Confidence range   Multiplier
    ───     ────────────────   ──────────
    Base    0.78 – 0.85        1.0×
    Mid     0.85 – 0.92        1.5×
    High    > 0.92             2.0×
    """
    if confidence > SIZE_TIER_FULL:
        return 2.0
    if confidence > SIZE_TIER_HALF:
        return 1.5
    return 1.0


def _sector_key(rec: Recommendation) -> str:
    """Return a sector grouping key for per-sector cap enforcement.

    - Sector ETFs (XLK, XLF …):  the ETF ticker itself (each ETF = its own sector)
    - Commodities (GLD, SLV …):  "COMMODITY"
    - Stocks:  look up in aggregator._SECTOR_MAP; fall back to "STOCK/<ticker>"
               so unknown stocks each count independently (no cross-stock cap).
    """
    ticker = rec.ticker.upper()
    if rec.type == "COMMODITY" or ticker in [
        t.upper() for t in settings.commodities_list
    ]:
        return "COMMODITY"
    if rec.type == "ETF" or ticker in [
        t.upper() for t in settings.sectors_list
    ]:
        return ticker   # each sector ETF is its own bucket
    # Stocks — import lazily to avoid circular dependency
    try:
        from src.signals.aggregator import _SECTOR_MAP
        mapped = _SECTOR_MAP.get(ticker)
        if mapped:
            return mapped
    except Exception:
        pass
    return f"STOCK/{ticker}"   # unique key → each unknown stock is uncapped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def record_new_trades(
    recommendations: List[Recommendation],
    signals_by_ticker: Optional[dict] = None,
) -> None:
    """Open a new trade for each BUY/SELL recommendation not already open today.

    Position size is confidence-scaled (1×/1.5×/2×) with a per-sector cap of
    SECTOR_CAP units across all currently open positions.
    Per-method signal scores are stored for later attribution analysis.
    """
    trades = _load_trades()
    today = date.today().isoformat()

    # Prevent stacking: skip any ticker that already has an OPEN position.
    already_open = {t["ticker"] for t in trades if t["status"] == "OPEN"}

    # Current sector exposure from open positions (sum of size multipliers per sector)
    sector_exposure: Dict[str, float] = {}
    for t in trades:
        if t["status"] == "OPEN":
            key = t.get("sector_key", f"STOCK/{t['ticker']}")
            sector_exposure[key] = sector_exposure.get(key, 0.0) + t.get("position_size_multiplier", 1.0)

    new_count = 0
    for rec in recommendations:
        if rec.action not in ("BUY", "SELL"):
            continue
        if rec.ticker in already_open:
            continue

        # Compute base multiplier from confidence tier
        multiplier = _position_multiplier(rec.confidence)
        sector = _sector_key(rec)

        # Enforce sector cap
        current_exposure = sector_exposure.get(sector, 0.0)
        if current_exposure >= SECTOR_CAP:
            logger.info(
                f"[tracker] Skipping {rec.ticker} — sector '{sector}' at cap "
                f"({current_exposure:.1f}× / {SECTOR_CAP}×)"
            )
            continue
        if current_exposure + multiplier > SECTOR_CAP:
            multiplier = round(SECTOR_CAP - current_exposure, 2)
            logger.info(
                f"[tracker] {rec.ticker}: size reduced to {multiplier}× "
                f"(sector '{sector}' cap)"
            )

        # Capture the exact instant the price is observed.  Stored alongside
        # the price so the audit trail is self-explanatory: which value was
        # used, and when it was sampled.
        fetched_at = _now_iso()
        price = _fetch_price(rec.ticker)
        if price is None:
            logger.warning(f"[tracker] Skipping {rec.ticker} — could not fetch entry price")
            continue
        if price <= 0:
            logger.warning(f"[tracker] Skipping {rec.ticker} — entry price is ${price:.4f} (warrant/delisted?)")
            continue

        # Capture per-method scores for attribution analysis
        mscores  = _method_scores_from_signal(rec.ticker, rec.direction, signals_by_ticker)
        agreed   = _methods_agreeing(mscores, rec.direction)
        dominant = _dominant_method(mscores, rec.direction)

        trades.append({
            "ticker": rec.ticker,
            "type": rec.type,
            "action": rec.action,
            "direction": rec.direction,
            "confidence": rec.confidence,
            "position_size_multiplier": multiplier,
            "sector_key": sector,
            "entry_date": today,
            "entry_datetime": fetched_at,
            "entry_price": round(price, 4),
            "rationale": rec.rationale,
            "current_price": round(price, 4),
            "current_price_datetime": fetched_at,
            "return_pct": 0.0,
            "weighted_return_pct": 0.0,
            "days_held": 0,
            "exit_date": None,
            "exit_datetime": None,
            "exit_price": None,
            "status": "OPEN",
            # Method attribution fields
            "method_scores":    mscores,
            "methods_agreeing": agreed,
            "dominant_method":  dominant,
        })
        sector_exposure[sector] = current_exposure + multiplier
        new_count += 1
        logger.info(
            f"[tracker] Opened {rec.action} {rec.ticker} @ {price:.2f} "
            f"({fetched_at}) | size={multiplier}× (conf={rec.confidence:.0%})"
        )

    _save_trades(trades)
    logger.info(f"[tracker] {new_count} new trade(s) recorded")


def close_trades_on_signal_reversal(actionable_recs: List["Recommendation"]) -> int:
    """Close open trades whose direction has reversed in today's actionable recommendations.

    An open BUY is closed when an actionable SELL exists for the same ticker (and vice versa).
    Uses current_price already refreshed by update_open_trades() — no second fetch needed.
    The closed trade is immediately saved; record_new_trades() will then open the new leg.
    """
    trades = _load_trades()
    today = date.today().isoformat()

    # ticker → action for all actionable BUY/SELL recs
    rev_signal: Dict[str, str] = {
        r.ticker: r.action
        for r in actionable_recs
        if r.action in ("BUY", "SELL")
    }

    closed = 0
    for trade in trades:
        if trade["status"] != "OPEN":
            continue
        new_action = rev_signal.get(trade["ticker"])
        if new_action is None:
            continue
        if not (
            (trade["action"] == "BUY"  and new_action == "SELL") or
            (trade["action"] == "SELL" and new_action == "BUY")
        ):
            continue

        # Reuse the most recent intraday mark when available; otherwise
        # capture a fresh sample now and timestamp it.
        if trade.get("current_price"):
            exit_price = trade["current_price"]
            exit_at    = trade.get("current_price_datetime") or _now_iso()
        else:
            exit_at    = _now_iso()
            exit_price = _fetch_price(trade["ticker"])
        if not exit_price:
            logger.warning(f"[tracker] Cannot close {trade['ticker']} on reversal — no price")
            continue

        ret = _pct_return(trade["action"], trade["entry_price"], exit_price, trade.get("type", "STOCK"))
        mul = trade.get("position_size_multiplier", 1.0)
        trade["status"]             = "CLOSED"
        trade["exit_date"]          = today
        trade["exit_datetime"]      = exit_at
        trade["exit_price"]         = round(exit_price, 4)
        trade["return_pct"]         = round(ret, 3)
        trade["weighted_return_pct"]= round(ret * mul, 3)
        trade["days_held"]          = _trading_days_held(trade["entry_date"])
        closed += 1
        logger.info(
            f"[tracker] Signal reversal → closed {trade['action']} {trade['ticker']} "
            f"@ {fmt_price(exit_price)} ({exit_at})  return={ret:+.2f}%  (new signal: {new_action})"
        )

    if closed:
        _save_trades(trades)
    return closed


def _normalize_closed_returns(trades: List[dict]) -> int:
    """Re-derive ``return_pct`` for every closed trade from its stored
    entry/exit prices using the current spread model.

    Idempotent and deterministic — given the same prices and spread model,
    the output never changes. Run on every pipeline tick so summary stats
    (avg/best/worst/win_rate over stored ``return_pct``) always reconcile
    with the per-trade compound produced by the daily-NAV engine.

    Returns the number of trades whose ``return_pct`` changed.
    """
    changed = 0
    for t in trades:
        if t.get("status") != "CLOSED":
            continue
        e = t.get("entry_price")
        x = t.get("exit_price")
        if e is None or x is None:
            continue
        ret = round(_pct_return(t.get("action", "BUY"), float(e), float(x), t.get("type", "STOCK")), 3)
        if abs(ret - t.get("return_pct", 0.0)) > 1e-3:
            t["return_pct"] = ret
            mul = t.get("position_size_multiplier", 1.0)
            t["weighted_return_pct"] = round(ret * mul, 3)
            changed += 1
    return changed


def _refresh_open_trade_ohlcv(trades: List[dict]) -> None:
    """Force-refresh the OHLCV cache for every open-trade ticker.

    The daily-NAV engine walks one mark per trading day held.  When the
    OHLCV cache for a ticker is stale (last bar < yesterday), days inside
    the holding period get lumped onto the next available mark — still
    deterministic, but less granular.  Refreshing here guarantees we have a
    real close for every day the position was actually held.
    """
    if not settings.enable_fetch_data:
        return
    tickers = sorted({t["ticker"] for t in trades if t.get("status") == "OPEN"})
    if not tickers:
        return
    try:
        from src.data.market_data import get_history
        from src.data.cache import load_ohlcv
    except Exception as e:
        logger.warning(f"[tracker] OHLCV refresh skipped — import failed: {e}")
        return

    yesterday = date.today() - timedelta(days=1)
    refreshed = 0
    for tk in tickers:
        cached = load_ohlcv(tk)
        needs = cached is None or cached.empty or cached.index[-1].date() < yesterday
        if not needs:
            continue
        try:
            get_history(tk, period="3mo", force_refresh=True)
            refreshed += 1
        except Exception as e:
            logger.debug(f"[tracker] OHLCV refresh failed for {tk}: {e}")
    if refreshed:
        logger.info(f"[tracker] Refreshed OHLCV cache for {refreshed} open-trade ticker(s)")


def update_open_trades() -> None:
    """Refresh current prices and P&L for all open trades; close expired ones."""
    trades = _load_trades()
    today = date.today().isoformat()
    updated = 0
    closed = 0

    # Keep OHLCV current for every open-trade ticker so the daily-NAV walk
    # has a real close for each day the position was held (no synthetic
    # lumping over a stale-cache gap).
    _refresh_open_trade_ohlcv(trades)

    # Refresh stored return_pct on closed trades so summary stats always
    # match the current spread model and the daily-NAV engine's compound.
    normalized = _normalize_closed_returns(trades)
    if normalized:
        logger.info(f"[tracker] Normalized return_pct on {normalized} closed trade(s) to current spread model")

    for trade in trades:
        if trade["status"] != "OPEN":
            continue

        # Timestamp every live mark so the audit trail records WHEN the
        # current_price was observed, not just its value.
        fetched_at = _now_iso()
        price = _fetch_price(trade["ticker"])
        if price is None:
            continue

        days = _trading_days_held(trade["entry_date"])
        ret = _pct_return(trade["action"], trade["entry_price"], price, trade.get("type", "STOCK"))

        mul = trade.get("position_size_multiplier", 1.0)
        trade["current_price"] = round(price, 4)
        trade["current_price_datetime"] = fetched_at
        trade["return_pct"] = round(ret, 3)
        trade["weighted_return_pct"] = round(ret * mul, 3)
        trade["days_held"] = days
        updated += 1

        # Auto-close after HOLDING_DAYS
        if days >= HOLDING_DAYS:
            trade["status"] = "CLOSED"
            trade["exit_date"] = today
            trade["exit_datetime"] = fetched_at
            trade["exit_price"] = round(price, 4)
            closed += 1
            logger.info(
                f"[tracker] Closed {trade['action']} {trade['ticker']} | "
                f"entry={fmt_price(trade['entry_price'])} exit={fmt_price(price)} ({fetched_at}) "
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
            mul = t.get("position_size_multiplier", 1.0)
            logger.info(
                f"    {t['action']:<4} {t['ticker']:<6} | "
                f"date={t['entry_date']}  entry={fmt_price(t['entry_price'])}  now={fmt_price(t['current_price'])}  "
                f"P&L={t['return_pct']:+.2f}%  size={mul}×  ({t['days_held']}d held)"
            )

    # Closed positions
    if closed_trades:
        returns = [t["return_pct"] for t in closed_trades]
        multipliers = [t.get("position_size_multiplier", 1.0) for t in closed_trades]
        wins = [r for r in returns if r > 0]
        win_rate = len(wins) / len(returns) * 100
        avg_return = sum(returns) / len(returns)
        total_mul = sum(multipliers)
        weighted_avg = sum(r * m for r, m in zip(returns, multipliers)) / total_mul if total_mul else avg_return
        best = max(returns)
        worst = min(returns)

        logger.info(f"  Closed positions ({len(closed_trades)})")
        for t in sorted(closed_trades, key=lambda x: x["exit_date"] or "", reverse=True):
            mul = t.get("position_size_multiplier", 1.0)
            logger.info(
                f"    {t['action']:<4} {t['ticker']:<6} | "
                f"date={t['entry_date']}  entry={fmt_price(t['entry_price'])}  exit={fmt_price(t['exit_price'])}  "
                f"return={t['return_pct']:+.2f}%  size={mul}×  ({t['days_held']}d)"
            )

        logger.info(f"  --- Stats ({len(closed_trades)} closed trades) ---")
        logger.info(f"    Win rate:         {win_rate:.1f}%")
        logger.info(f"    Avg return:       {avg_return:+.2f}%")
        logger.info(f"    Weighted avg ret: {weighted_avg:+.2f}%  (size-adjusted)")
        logger.info(f"    Best trade:       {best:+.2f}%")
        logger.info(f"    Worst trade:      {worst:+.2f}%")

        # Per-method attribution
        attributed = [t for t in closed_trades if t.get("methods_agreeing")]
        method_stats = _compute_method_stats(attributed)
        if method_stats:
            logger.info("  --- Method Attribution ---")
            for method, ms in sorted(method_stats.items(), key=lambda x: -x[1]["win_rate"]):
                logger.info(
                    f"    {method:<10} trades={ms['trades']:>2}  "
                    f"win={ms['win_rate']:>5.1f}%  avg={ms['avg_return']:>+6.2f}%"
                )

        cat_stats = _compute_category_stats(attributed)
        if cat_stats:
            logger.info("  --- Category Attribution ---")
            for cat, cs in sorted(cat_stats.items(), key=lambda x: -x[1]["win_rate"]):
                logger.info(
                    f"    {cat:<15} trades={cs['trades']:>2}  "
                    f"win={cs['win_rate']:>5.1f}%  avg={cs['avg_return']:>+6.2f}%"
                )

        conv_stats = _compute_convergence_stats(attributed)
        if conv_stats:
            logger.info("  --- Convergence ---")
            for label, cs in conv_stats.items():
                logger.info(
                    f"    {label:<15} trades={cs['trades']:>2}  "
                    f"win={cs['win_rate']:>5.1f}%  avg={cs['avg_return']:>+6.2f}%"
                )

    logger.info("=" * 60)


def get_open_trade_tickers() -> List[str]:
    """Return tickers of all currently open trades so the pipeline always fetches their prices."""
    return [t["ticker"] for t in _load_trades() if t["status"] == "OPEN"]


def _build_trades_svg(closed_trades: List[dict]) -> str:
    """
    Return an inline SVG string visualising closed trades over time.

    Top panel  — equity curve: compound cumulative return as a filled line chart.
    Bottom panel — per-trade return bars coloured green (win) / red (loss).
    No external dependencies; safe to embed directly inside HTML email.
    """
    import math

    trades  = sorted(closed_trades, key=lambda t: t.get("exit_date") or "")
    n       = len(trades)
    returns = [t["return_pct"]           for t in trades]
    tickers = [t["ticker"]               for t in trades]
    dates   = [(t.get("exit_date") or "")[-5:] for t in trades]   # "MM-DD"
    actions = [t.get("action", "BUY")    for t in trades]

    # Compound cumulative return at each exit
    cum: List[float] = []
    compound = 1.0
    for r in returns:
        compound *= (1 + r / 100)
        cum.append(round((compound - 1) * 100, 2))

    # ── Layout ────────────────────────────────────────────────────────────────
    W               = 820          # total SVG width
    PL, PR          = 58, 18       # left / right padding (room for y-axis labels)
    PT              = 28           # top padding
    CHART_W         = W - PL - PR

    TOP_H           = 160          # equity curve panel
    SEP             = 22           # gap between panels
    BOT_H           = 80           # per-trade bars panel
    BOT_Y           = PT + TOP_H + SEP
    XLABEL_H        = 46           # space for date + ticker labels
    H               = BOT_Y + BOT_H + XLABEL_H

    # ── Coordinate helpers ────────────────────────────────────────────────────
    def xp(i: int) -> float:
        return PL + (i * CHART_W / (n - 1) if n > 1 else CHART_W / 2)

    # Top panel y-scale (equity curve)
    all_y   = [0.0] + cum
    ylo, yhi = min(all_y), max(all_y)
    y_span  = (yhi - ylo) or 1.0
    ypad    = y_span * 0.15

    def yp_top(v: float) -> float:
        lo = ylo - ypad
        hi = yhi + ypad
        return PT + TOP_H * (1.0 - (v - lo) / (hi - lo))

    zero_y = yp_top(0.0)

    # Bottom panel y-scale (bars centred at zero)
    ret_max = max(abs(r) for r in returns) or 1.0
    bar_scale = (BOT_H / 2) / ret_max * 0.88
    center_y  = BOT_Y + BOT_H / 2
    bar_w     = max(4.0, min(24.0, CHART_W / n * 0.55))

    # ── Y-axis ticks for top panel ────────────────────────────────────────────
    raw_step  = y_span / 4
    magnitude = 10 ** math.floor(math.log10(abs(raw_step) or 1))
    nice_step = round(raw_step / magnitude) * magnitude or magnitude
    tick_start = math.ceil((ylo - ypad) / nice_step) * nice_step
    ticks: List[float] = []
    v = tick_start
    while v <= yhi + ypad + nice_step * 0.5:
        ticks.append(round(v, 6))
        v += nice_step

    # ── SVG assembly ─────────────────────────────────────────────────────────
    parts: List[str] = []
    a = parts.append

    a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
      f'viewBox="0 0 {W} {H}" '
      f'style="background:#0f172a;border-radius:8px;display:block;max-width:100%;margin:14px 0;">')

    # Gradient for equity-curve fill
    a('<defs>'
      '<linearGradient id="ecg" x1="0" y1="0" x2="0" y2="1">'
      '<stop offset="0%" stop-color="#60a5fa" stop-opacity="0.30"/>'
      '<stop offset="100%" stop-color="#60a5fa" stop-opacity="0.02"/>'
      '</linearGradient>'
      '</defs>')

    # Panel labels
    a(f'<text x="{PL}" y="{PT-8}" fill="#64748b" font-size="10" font-family="Arial,sans-serif">'
      'Cumulative Return (%)</text>')
    a(f'<text x="{PL}" y="{BOT_Y-7}" fill="#64748b" font-size="10" font-family="Arial,sans-serif">'
      'Per-trade Return (%)</text>')

    # ── Top panel grid ────────────────────────────────────────────────────────
    for tv in ticks:
        ty = yp_top(tv)
        if PT - 1 <= ty <= PT + TOP_H + 1:
            is_zero = abs(tv) < nice_step * 0.05
            col  = "#475569" if is_zero else "#1e293b"
            dash = "" if is_zero else ' stroke-dasharray="3,3"'
            a(f'<line x1="{PL}" y1="{ty:.1f}" x2="{PL+CHART_W}" y2="{ty:.1f}" '
              f'stroke="{col}" stroke-width="1"{dash}/>')
            a(f'<text x="{PL-4}" y="{ty+4:.1f}" fill="#64748b" font-size="9" '
              f'font-family="Arial,sans-serif" text-anchor="end">{tv:+.0f}%</text>')

    # ── Bottom panel zero line ────────────────────────────────────────────────
    a(f'<line x1="{PL}" y1="{center_y:.1f}" x2="{PL+CHART_W}" y2="{center_y:.1f}" '
      'stroke="#334155" stroke-width="1"/>')

    # ── Equity-curve fill area ────────────────────────────────────────────────
    pts_top = [(xp(i), yp_top(cum[i])) for i in range(n)]
    path_d  = f"M {xp(0):.1f} {yp_top(0.0):.1f} " + \
              " ".join(f"L {x:.1f} {y:.1f}" for x, y in pts_top)
    fill_d  = path_d + f" L {xp(n-1):.1f} {zero_y:.1f} L {xp(0):.1f} {zero_y:.1f} Z"
    a(f'<path d="{fill_d}" fill="url(#ecg)"/>')

    # ── Equity-curve line ─────────────────────────────────────────────────────
    a(f'<path d="{path_d}" fill="none" stroke="#60a5fa" stroke-width="2.5" stroke-linejoin="round"/>')

    # Zero dashed line (equity panel)
    a(f'<line x1="{PL}" y1="{zero_y:.1f}" x2="{PL+CHART_W}" y2="{zero_y:.1f}" '
      'stroke="#475569" stroke-width="1.2" stroke-dasharray="5,3"/>')

    # ── Dots on equity curve ──────────────────────────────────────────────────
    for i, (px, py) in enumerate(pts_top):
        dot_col = "#4ade80" if cum[i] >= 0 else "#f87171"
        a(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.5" '
          f'fill="{dot_col}" stroke="#0f172a" stroke-width="1.2">'
          f'<title>{tickers[i]} exit {dates[i]}: {cum[i]:+.1f}% cumulative</title>'
          f'</circle>')

    # End-value label
    ex, ey  = pts_top[-1]
    ec      = "#4ade80" if cum[-1] >= 0 else "#f87171"
    label_x = min(ex + 5, PL + CHART_W - 35)
    a(f'<text x="{label_x:.1f}" y="{ey - 5:.1f}" fill="{ec}" font-size="10" '
      f'font-weight="bold" font-family="Arial,sans-serif">{cum[-1]:+.1f}%</text>')

    # ── Per-trade bars ────────────────────────────────────────────────────────
    for i, (r, ticker, action) in enumerate(zip(returns, tickers, actions)):
        bc    = "#22c55e" if r >= 0 else "#ef4444"
        bh    = abs(r) * bar_scale
        bx    = xp(i) - bar_w / 2
        by    = center_y - bh if r >= 0 else center_y
        a(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{max(bh, 1.5):.1f}" '
          f'fill="{bc}" opacity="0.9" rx="1">'
          f'<title>{ticker} {r:+.1f}%</title></rect>')

    # ── X-axis labels (skip if too many) ─────────────────────────────────────
    step = max(1, math.ceil(n / 18))
    for i in range(0, n, step):
        lx = xp(i)
        a(f'<text x="{lx:.1f}" y="{BOT_Y+BOT_H+14}" fill="#64748b" font-size="9" '
          f'font-family="Arial,sans-serif" text-anchor="middle">{dates[i]}</text>')
        a(f'<text x="{lx:.1f}" y="{BOT_Y+BOT_H+27}" fill="#94a3b8" font-size="9" '
          f'font-family="Arial,sans-serif" text-anchor="middle">{tickers[i]}</text>')

    a('</svg>')
    return '\n'.join(parts)


def _build_timeline_svg(all_trades: list) -> str:
    """Gantt-style timeline of all trades (closed + open).

    One row per trade, sorted chronologically.  Bars span entry_date → exit_date
    (or today for open positions).  Color: green=profit, red=loss, blue=open.
    Left margin: ticker + action label.  Right of bar: return % annotation.
    Hover tooltip: entry/exit dates, prices, and return.
    """
    if not all_trades:
        return ""

    today = date.today()
    MAX_ROWS = 40

    rows = []
    for t in all_trades:
        entry_str = t.get("entry_date", "")
        if not entry_str:
            continue
        try:
            entry_dt = date.fromisoformat(entry_str)
        except Exception:
            continue
        exit_str = t.get("exit_date", "") or ""
        try:
            exit_dt = date.fromisoformat(exit_str) if exit_str else today
        except Exception:
            exit_dt = today
        rows.append({
            "ticker":    t.get("ticker", "?"),
            "action":    t.get("action", "BUY"),
            "entry_dt":  entry_dt,
            "exit_dt":   exit_dt,
            "entry_str": entry_str,
            "exit_str":  exit_str,
            "status":    t.get("status", "CLOSED"),
            "ret":       t.get("return_pct", 0.0),
            "entry_px":  t.get("entry_price") or 0.0,
            "exit_px":   t.get("exit_price") or t.get("current_price") or 0.0,
        })

    if not rows:
        return ""

    rows.sort(key=lambda r: (r["entry_str"], r["ticker"]))
    truncated = len(rows) > MAX_ROWS
    if truncated:
        rows = rows[-MAX_ROWS:]

    min_dt = min(r["entry_dt"] for r in rows)
    max_dt = max(r["exit_dt"] for r in rows)
    # +1 so a same-day entry/exit gets a non-zero bar width
    span_days = max(1, (max_dt - min_dt).days + 1)

    W       = 820
    LABEL_W = 100
    RET_W   = 60
    CHART_W = W - LABEL_W - RET_W
    ROW_H   = 20
    ROW_GAP = 3
    PT      = 34   # top padding for axis date labels
    PB      = 24   # bottom padding for legend
    N       = len(rows)
    H       = PT + N * (ROW_H + ROW_GAP) + PB

    def x_dt(d: date) -> float:
        return LABEL_W + (d - min_dt).days / span_days * CHART_W

    parts: List[str] = []
    a = parts.append

    a(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
      f'viewBox="0 0 {W} {H}" '
      f'style="background:#0f172a;border-radius:8px;display:block;max-width:100%;margin:14px 0;">')

    suffix = "  (latest 40)" if truncated else ""
    a(f'<text x="{LABEL_W}" y="15" fill="#64748b" font-size="10" '
      f'font-family="Arial,sans-serif">Trade Timeline{suffix}</text>')

    # X-axis date ticks + vertical grid lines
    chart_bot = PT + N * (ROW_H + ROW_GAP)
    n_ticks = min(8, max(2, span_days // 5 + 1))
    for i in range(n_ticks):
        frac = i / (n_ticks - 1) if n_ticks > 1 else 0.5
        tick_day = round(frac * (span_days - 1))
        d = min_dt + timedelta(days=tick_day)
        tx = x_dt(d)
        a(f'<line x1="{tx:.1f}" y1="{PT - 4}" x2="{tx:.1f}" y2="{chart_bot}" '
          f'stroke="#1e293b" stroke-width="1"/>')
        a(f'<text x="{tx:.1f}" y="{PT - 7}" fill="#475569" font-size="9" '
          f'font-family="Arial,sans-serif" text-anchor="middle">{d.strftime("%m/%d")}</text>')

    # Today dashed marker
    if min_dt <= today <= max_dt + timedelta(days=1):
        tx = x_dt(today)
        a(f'<line x1="{tx:.1f}" y1="{PT - 4}" x2="{tx:.1f}" y2="{chart_bot}" '
          f'stroke="#334155" stroke-width="1" stroke-dasharray="4,3"/>')

    # Trade rows
    for i, row in enumerate(rows):
        y  = PT + i * (ROW_H + ROW_GAP)
        cy = y + ROW_H // 2 + 4  # text baseline centred in row

        if i % 2 == 1:
            a(f'<rect x="0" y="{y}" width="{W}" height="{ROW_H}" fill="#111827" opacity="0.4"/>')

        # Ticker label (right-aligned to x=62)
        a(f'<text x="{LABEL_W - 38}" y="{cy}" fill="#cbd5e1" font-size="10" '
          f'font-family="Arial,sans-serif" text-anchor="end" font-weight="600">'
          f'{row["ticker"]}</text>')
        # Action label (right-aligned to x=97)
        act_col = "#60a5fa" if row["action"] == "BUY" else "#fb923c"
        a(f'<text x="{LABEL_W - 3}" y="{cy}" fill="{act_col}" font-size="8" '
          f'font-family="Arial,sans-serif" text-anchor="end">{row["action"]}</text>')

        # Bar coordinates — extend by one full day-width so exit day is included
        bx = x_dt(row["entry_dt"])
        ex = min(x_dt(row["exit_dt"]) + CHART_W / span_days, LABEL_W + CHART_W)
        bw = max(2.0, ex - bx)
        bar_y = y + 4
        bar_h = ROW_H - 8

        if row["status"] == "OPEN":
            fill, stroke = "#1e3a5f", "#60a5fa"
        elif row["ret"] > 0:
            fill, stroke = "#14532d", "#22c55e"
        else:
            fill, stroke = "#7f1d1d", "#ef4444"

        # Tooltip
        ep_s = f"${fmt_price(row['entry_px'])}" if row["entry_px"] else "n/a"
        xp_s = f"${fmt_price(row['exit_px'])}"  if row["exit_px"]  else "n/a"
        exit_label = row["exit_str"] if row["exit_str"] else "open"
        tip = (f"{row['ticker']} {row['action']} | "
               f"{row['entry_str']} → {exit_label} | "
               f"Entry {ep_s}  Exit {xp_s} | "
               f"{row['ret']:+.2f}%")
        if row["status"] == "OPEN":
            tip += " [OPEN]"

        a(f'<rect x="{bx:.1f}" y="{bar_y}" width="{bw:.1f}" height="{bar_h}" '
          f'fill="{fill}" stroke="{stroke}" stroke-width="0.8" rx="2">'
          f'<title>{tip}</title></rect>')

        # Entry price inside bar (only when bar is wide enough to fit the label)
        if bw > 52 and row["entry_px"]:
            a(f'<text x="{bx + 4:.1f}" y="{cy}" fill="{stroke}" font-size="8" '
              f'font-family="Arial,sans-serif" opacity="0.9">'
              f'${fmt_price(row["entry_px"])}</text>')

        # Return annotation to the right of the bar
        ann_x = min(ex + 4, W - RET_W + 4)
        ret_col = "#4ade80" if row["ret"] > 0 else "#f87171" if row["ret"] < 0 else "#94a3b8"
        sfx = "*" if row["status"] == "OPEN" else ""
        a(f'<text x="{ann_x:.1f}" y="{cy}" fill="{ret_col}" font-size="9" '
          f'font-family="Arial,sans-serif" font-weight="700">{row["ret"]:+.1f}%{sfx}</text>')

    # Legend
    leg_y = H - 8
    for off, col, lbl in [(0, "#22c55e", "Profit"), (58, "#ef4444", "Loss"), (110, "#60a5fa", "Open*")]:
        lx = LABEL_W + off
        a(f'<rect x="{lx}" y="{leg_y - 8}" width="10" height="8" fill="{col}" rx="2" opacity="0.7"/>')
        a(f'<text x="{lx + 13}" y="{leg_y}" fill="#64748b" font-size="9" '
          f'font-family="Arial,sans-serif">{lbl}</text>')

    a('</svg>')
    return '\n'.join(parts)


def compute_portfolio_metrics(closed_trades: list, open_trades: list) -> dict:
    """True time-weighted compound portfolio return across all trades (closed + open).

    Uses ``daily_nav.compute_compound_return``: every per-day return is derived
    from the real OHLCV close for that ticker on that day, capital-weighted
    across whatever positions are active. No geometric interpolation, no
    proxy assumptions about intermediate prices — same trades.json + same
    OHLCV cache → identical numbers every run.

    Closed trades: walk uses entry_price → daily closes → exit_price (each
        bracket-adjusted by the dynamic bid-ask spread on entry/exit only).
    Open trades:   walk uses entry_price → daily closes → today's
        current_price (also spread-adjusted, treating the live mark as a
        hypothetical close-out so the per-trade compound matches the trade's
        stored ``return_pct``).

    Time windows include only trades ENTERED within the last N calendar days.
    Inception includes every trade ever recorded.

    Returns dict: {
        compound_inception — compound over all trades ever,
        return_1w          — compound for trades entered in last 7 days,
        return_2w          — compound for trades entered in last 14 days,
        return_1m          — compound for trades entered in last 30 days,
    }  Values are percentages.  None when no trades exist for that window.
    """
    today = date.today()
    all_trades = [t for t in (closed_trades + open_trades) if t.get("entry_date")]
    if not all_trades:
        return {}

    def _window(days: int) -> Optional[float]:
        cutoff = str(today - timedelta(days=days))
        return _compute_nav_compound([t for t in all_trades if t["entry_date"] >= cutoff])

    return {
        "compound_inception": _compute_nav_compound(all_trades),
        "return_1w":          _window(7),
        "return_2w":          _window(14),
        "return_1m":          _window(30),
    }


def _flip_trade(trade: dict) -> dict:
    """Return a shallow-copied trade with action/direction inverted.

    Used to model the "what if this method had signalled the opposite
    direction" hypothetical. Prices, dates, and ticker are preserved so the
    daily NAV engine walks the real OHLCV closes — the only thing that
    changes is whether each day's price move is counted as long-favourable or
    short-favourable. The trade's ``return_pct`` is also re-derived from
    ``_pct_return`` for the flipped action so summary stats line up with the
    flipped compound.
    """
    flipped = dict(trade)
    action = trade.get("action", "BUY")
    flipped["action"]    = "SELL" if action == "BUY" else "BUY"
    flipped["direction"] = "BEARISH" if trade.get("direction") == "BULLISH" else "BULLISH"

    entry_px   = trade.get("entry_price")
    asset_type = trade.get("type", "STOCK")
    if entry_px is not None:
        if flipped["status"] == "CLOSED" and trade.get("exit_price") is not None:
            flipped["return_pct"] = round(
                _pct_return(flipped["action"], float(entry_px), float(trade["exit_price"]), asset_type),
                3,
            )
        else:
            cp = trade.get("current_price") or entry_px
            flipped["return_pct"] = round(
                _pct_return(flipped["action"], float(entry_px), float(cp), asset_type),
                3,
            )
    return flipped


def _hypothetical_trades_for_method(method: str, closed: List[dict]) -> List[dict]:
    """Build the per-method hypothetical trade list (real or flipped).

    For each closed trade with stored method_scores: if the method's view
    matches the actual action, include the trade verbatim; if it disagrees,
    include the flipped trade. Trades where the method has no view
    (|score| < threshold) are skipped entirely.
    """
    out: List[dict] = []
    for trade in closed:
        score = trade.get("method_scores", {}).get(method, 0.0)
        if abs(score) < _METHOD_AGREE_THRESHOLD:
            continue
        actual = trade.get("action", "BUY")
        solo   = "BUY" if score > 0 else "SELL"
        out.append(trade if solo == actual else _flip_trade(trade))
    return out


def _solo_stats(trades: list) -> dict:
    """Summary stats from a list of (real or flipped) trade dicts.

    Compound return walks the actual OHLCV closes via the daily-NAV engine;
    every other stat is computed from the trades' ``return_pct`` so it
    reconciles exactly with the compound on a single-trade slice.
    """
    if not trades:
        return {}
    hyp = [t["return_pct"] for t in trades]
    wins = [r for r in hyp if r > 0]
    return {
        "trades":          len(trades),
        "win_rate":        round(len(wins) / len(hyp) * 100, 1),
        "avg_return":      round(sum(hyp) / len(hyp), 2),
        "compound_return": _compute_nav_compound(trades) or 0.0,
        "best":            round(max(hyp), 2),
        "worst":           round(min(hyp), 2),
    }


_EVAL_BANDS = [
    ("Low (0.10–0.35)",    0.10, 0.35),
    ("Medium (0.35–0.65)", 0.35, 0.65),
    ("High (0.65+)",       0.65, 1.01),
]


def _eval_stats(entries: list) -> dict:
    """Directional accuracy stats from (trade_dict, abs_score) pairs.

    The trade dict is either the real trade (when the method agreed with the
    actual action) or a flipped copy (when it disagreed) — see
    ``_hypothetical_trades_for_method``. Compound returns are walked through
    the actual OHLCV cache via the daily-NAV engine, never interpolated.
    """
    if not entries:
        return {}
    correct = [t["return_pct"] for t, _ in entries if t["return_pct"] > 0]
    wrong   = [t["return_pct"] for t, _ in entries if t["return_pct"] <= 0]
    conviction_bands = []
    for label, lo, hi in _EVAL_BANDS:
        band = [t for t, s in entries if lo <= s < hi]
        if not band:
            conviction_bands.append({"label": label, "trades": 0, "accuracy": 0.0, "avg_return": 0.0, "compound_return": 0.0})
            continue
        band_correct = [t["return_pct"] for t in band if t["return_pct"] > 0]
        conviction_bands.append({
            "label":           label,
            "trades":          len(band),
            "accuracy":        round(len(band_correct) / len(band) * 100, 1),
            "avg_return":      round(sum(t["return_pct"] for t in band) / len(band), 2),
            "compound_return": _compute_nav_compound(band) or 0.0,
        })
    return {
        "trades":               len(entries),
        "directional_accuracy": round(len(correct) / len(entries) * 100, 1),
        "avg_return_correct":   round(sum(correct) / len(correct), 2) if correct else 0.0,
        "avg_return_wrong":     round(sum(wrong)   / len(wrong),   2) if wrong   else 0.0,
        "compound_return":      _compute_nav_compound([t for t, _ in entries]) or 0.0,
        "conviction_bands":     conviction_bands,
    }


def compute_solo_method_performance() -> dict:
    """Simulate performance for each signal method used in isolation, split by direction.

    For each closed trade with stored method_scores, each method is asked:
    "What direction would you alone have signalled?"

      score > 0  → solo BUY;  score < 0  → solo SELL;  |score| < threshold → skip

    Hypothetical return:
      same direction as actual trade  → actual return_pct
      opposite direction              → −actual return_pct

    Returns dict: method_name → {
        "overall": {trades, win_rate, avg_return, compound_return, best, worst},
        "buys":    {same fields, only BUY-signal trades}   — {} if none,
        "sells":   {same fields, only SELL-signal trades}  — {} if none,
    }
    Only methods with ≥ 1 qualifying trade are returned.
    """
    MIN_TRADES = 1
    trades = _load_trades()
    closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("method_scores")]
    if not closed:
        return {}

    results: dict = {}
    for method in _ALL_METHODS:
        hyp = _hypothetical_trades_for_method(method, closed)
        if len(hyp) < MIN_TRADES:
            continue
        buys  = [t for t in hyp if t.get("action") == "BUY"]
        sells = [t for t in hyp if t.get("action") == "SELL"]
        results[method] = {
            "overall": _solo_stats(hyp),
            "buys":    _solo_stats(buys),
            "sells":   _solo_stats(sells),
        }

    return results


def compute_method_eval_stats() -> dict:
    """Per-method directional accuracy and conviction calibration, split by direction.

    For each signal method, across all closed trades with stored method_scores:
      - Directional accuracy: % of times the method's direction led to a positive hyp return
      - avg_return when correct vs wrong
      - Conviction bands (Low/Medium/High |score|): trades, accuracy, avg_return

    Results are split into "overall", "buys" (score > 0), and "sells" (score < 0).

    Returns dict: method_name → {
        "overall": {trades, directional_accuracy, avg_return_correct, avg_return_wrong,
                    conviction_bands: [{"label","trades","accuracy","avg_return"}, ...]},
        "buys":    {same structure, only BUY-signal entries}  — {} if none,
        "sells":   {same structure, only SELL-signal entries} — {} if none,
    }
    Only methods with ≥ 1 qualifying trade are returned (others shown as no-data in email).
    """
    MIN_TRADES = 1
    trades = _load_trades()
    closed = [t for t in trades if t.get("status") == "CLOSED" and t.get("method_scores")]
    if not closed:
        return {}

    results: dict = {}
    for method in _ALL_METHODS:
        all_entries: List[tuple] = []
        for trade in closed:
            score = trade["method_scores"].get(method, 0.0)
            if abs(score) < _METHOD_AGREE_THRESHOLD:
                continue
            actual_action = trade.get("action", "BUY")
            solo_action   = "BUY" if score > 0 else "SELL"
            hyp_trade     = trade if solo_action == actual_action else _flip_trade(trade)
            all_entries.append((hyp_trade, abs(score)))

        if len(all_entries) < MIN_TRADES:
            continue

        buy_entries  = [(t, s) for t, s in all_entries if t.get("action") == "BUY"]
        sell_entries = [(t, s) for t, s in all_entries if t.get("action") == "SELL"]

        results[method] = {
            "overall": _eval_stats(all_entries),
            "buys":    _eval_stats(buy_entries),
            "sells":   _eval_stats(sell_entries),
        }

    return results


def get_performance_for_email() -> dict:
    """Return structured performance data for inclusion in the email report."""
    trades = _load_trades()
    open_trades   = [t for t in trades if t["status"] == "OPEN"]
    closed_trades = [t for t in trades if t["status"] == "CLOSED"]
    # Open trades carry a current M2M return_pct (updated each run by update_open_trades()).
    # Treat them as hypothetical exits so every live position is reflected in the stats.
    all_trades = closed_trades + open_trades

    # Inception date — earliest entry across ALL trades (open + closed)
    all_dates = [t["entry_date"] for t in trades if t.get("entry_date")]
    first_trade_date = min(all_dates) if all_dates else None
    inception_days: Optional[int] = None
    if first_trade_date:
        try:
            inception_days = (date.today() - date.fromisoformat(first_trade_date)).days
        except Exception:
            pass

    # Base stats always populated when any trades exist
    stats: dict = {}
    if trades:
        stats = {
            "total_closed":     len(closed_trades),
            "total_open":       len(open_trades),
            "total_all":        len(trades),
            "first_trade_date": first_trade_date,
            "inception_days":   inception_days,
        }

    # Headline metrics over ALL trades (closed final P&L + open M2M return_pct).
    if all_trades:
        returns = [t["return_pct"] for t in all_trades]
        multipliers = [t.get("position_size_multiplier", 1.0) for t in all_trades]
        total_mul = sum(multipliers)
        weighted_avg = (
            sum(r * m for r, m in zip(returns, multipliers)) / total_mul
            if total_mul else sum(returns) / len(returns)
        )
        # Compound return: trades sorted by entry date, sequentially re-invested.
        # Will be overwritten below by portfolio_metrics["compound_inception"] which
        # uses the more accurate daily-batch model, but compute it as a fallback.
        sorted_returns = [
            t["return_pct"]
            for t in sorted(all_trades, key=lambda x: x.get("entry_date", ""))
        ]
        compound = 1.0
        for r in sorted_returns:
            compound *= (1 + r / 100)

        stats.update({
            "win_rate":            round(len([r for r in returns if r > 0]) / len(returns) * 100, 1),
            "avg_return":          round(sum(returns) / len(returns), 2),
            "weighted_avg_return": round(weighted_avg, 2),
            "compound_return":     round((compound - 1) * 100, 2),
            "best":                round(max(returns), 2),
            "worst":               round(min(returns), 2),
        })

    attributed = [t for t in all_trades if t.get("methods_agreeing")]
    confidence_ranked    = _compute_confidence_ranked(closed_trades)          # realized P&L only
    performance_table    = _compute_performance_table(all_trades) if all_trades else []
    trades_svg           = _build_trades_svg(closed_trades) if len(closed_trades) >= 2 else ""
    timeline_svg         = _build_timeline_svg(all_trades) if all_trades else ""
    solo_method_perf     = compute_solo_method_performance()
    method_eval_stats    = compute_method_eval_stats()
    portfolio_metrics    = compute_portfolio_metrics(closed_trades, open_trades)

    # Keep stats["compound_return"] in sync with the authoritative portfolio_metrics value
    # (both use _compute_nav_compound but portfolio_metrics also includes live M2M for open trades).
    if portfolio_metrics and "compound_inception" in portfolio_metrics:
        stats["compound_return"] = portfolio_metrics["compound_inception"]

    # Methods ordered by overall win rate descending; no-data methods go to the bottom.
    method_order_by_winrate = sorted(
        list(_ALL_METHODS),
        key=lambda m: solo_method_perf.get(m, {}).get("overall", {}).get("win_rate", -1),
        reverse=True,
    )

    return {
        "open_trades":              sorted(open_trades,   key=lambda x: x["entry_date"],    reverse=True),
        "closed_trades":            sorted(closed_trades, key=lambda x: x["exit_date"] or "", reverse=True),
        "stats":                    stats,
        "portfolio_metrics":        portfolio_metrics,          # session-based compound + time windows
        "confidence_ranked":        confidence_ranked,
        "performance_table":        performance_table,
        "attributed_count":         len(attributed),
        "trades_svg":               trades_svg,
        "timeline_svg":             timeline_svg,
        "solo_method_perf":         solo_method_perf,          # hypothetical per-method solo simulation
        "method_eval_stats":        method_eval_stats,         # per-method accuracy + conviction calibration
        "method_labels":            METHOD_LABELS,
        "method_order_by_winrate":  method_order_by_winrate,   # methods ranked by solo win rate, no-data last
    }
