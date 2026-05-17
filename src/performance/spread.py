"""Bid-ask spread model and return-calculation primitives.

Extracted from ``tracker.py`` so the daily-NAV engine (``daily_nav.py``) and
the trade-tracker can both depend on it without round-trip imports.  Earlier
the engine pulled ``_dynamic_half_spread`` and ``_pct_return`` out of
``tracker`` lazily inside helper functions to avoid the import cycle — this
module breaks that cycle for good.

Dependency tree after this extraction:
    spread.py    (no internal deps)
        ↑
    daily_nav.py (depends on spread, cache)
        ↑
    tracker.py   (depends on spread, daily_nav, market_calendar, ...)

The half-spread numbers were tightened in May 2026 to better match published
effective-spread statistics:
  * SEC Rule 605 reports for top-tier names show effective spreads of ~1–3 bp
    full / ~0.5–1.5 bp half — the OLD 2 bp half-spread on large-caps was
    realistic for NBBO mid but ignored the price impact a typical $25–50k
    paper order would experience walking through the book.
  * Small-cap (Russell-2000-bottom-quartile) effective spreads run 50–100 bp
    full / 25–50 bp half — the OLD 12.5 bp half-spread underestimated by ~2×.
  * Penny / sub-penny names routinely trade with 5–20% spreads in OTC venues
    (FINRA TRACE data) — the OLD 250 bp on sub-penny captured only a fraction.

These tiers are still conservative for an institutional-size simulation but
realistic-or-slightly-tighter for the recommendation-sized paper positions
this system tracks.  If you need a different profile (e.g., zero-friction
for backtesting an alpha hypothesis), wrap or replace this function rather
than editing the numbers — the engine and tracker both go through it.
"""

from __future__ import annotations


def _dynamic_half_spread(price: float, asset_type: str = "STOCK") -> float:
    """One-way bid-ask half-spread as a fraction (NOT percent).

    Tiers (May 2026 calibration):

    ETF
      ≥ $100  → 1.5 bp  (SPY/QQQ-class index ETFs)
      <  $100 → 2.5 bp  (sector / style ETFs: XLK, XLF, IWM, …)

    Commodity
      ≥ $100  → 2.5 bp  (GLD, GDX)
      <  $100 → 5 bp    (SLV, CPER, GDXJ)

    Stock (price-tiered — proxy for liquidity)
      ≥ $100        → 3 bp    (mega-cap: AAPL, MSFT, NVDA)
      $50–$100      → 4 bp    (large-cap S&P 500 core)
      $10–$50       → 8 bp    (mid-cap)
      $1–$10        → 25 bp   (small-cap, lower Russell 2000)
      $0.10–$1      → 75 bp   (micro-cap)
      $0.01–$0.10   → 250 bp  (penny)
      < $0.01       → 500 bp  (sub-penny / warrant)

    Non-positive prices return 0 — the model is undefined there; callers
    refuse to open or mark such positions.
    """
    if price is None or price <= 0:
        return 0.0
    if asset_type == "ETF":
        return 0.00015 if price >= 100 else 0.00025
    if asset_type == "COMMODITY":
        return 0.00025 if price >= 100 else 0.0005
    # STOCK — price-tiered (1 bp = 0.0001)
    if price >= 100:
        return 0.0003
    if price >= 50:
        return 0.0004
    if price >= 10:
        return 0.0008
    if price >= 1:
        return 0.0025
    if price >= 0.10:
        return 0.0075
    if price >= 0.01:
        return 0.0250
    return 0.0500


def _pct_return(action: str, entry: float, current: float, asset_type: str = "STOCK") -> float:
    """Percent return, sign-aware, with the realistic round-trip half-spread.

    BUY  : paid the ask at entry (+half), receive the bid at exit (−half).
    SELL : shorted at the bid at entry (−half), covered at the ask (+half).
    Both half-spreads are evaluated against their own price (entry and
    current independently), so a position that crosses a price tier
    naturally picks up the wider/narrower spread on each leg.

    Returns ``0.0`` for non-positive prices — the round-trip is undefined
    there.  Callers refuse to trade at such prices; this guard exists to
    keep stats clean for any record that slipped through historically.
    """
    if entry is None or current is None or entry <= 0 or current <= 0:
        return 0.0
    entry_half = _dynamic_half_spread(entry, asset_type)
    exit_half  = _dynamic_half_spread(current, asset_type)
    if action == "BUY":
        effective_entry = entry   * (1 + entry_half)
        effective_exit  = current * (1 - exit_half)
        return (effective_exit - effective_entry) / effective_entry * 100
    # SELL = short
    effective_entry = entry   * (1 - entry_half)
    effective_exit  = current * (1 + exit_half)
    return (effective_entry - effective_exit) / effective_entry * 100


def fmt_price(p) -> str:
    """Format a price with enough decimal places to show meaningful digits.

    Handles sub-penny stocks/warrants (e.g. 0.003 → '$0.0030') without
    rounding to '$0.00'.  Used by the email template and log lines.
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
