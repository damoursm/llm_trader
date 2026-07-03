"""Bid-ask spread + commission model and return-calculation primitives.

Extracted from ``tracker.py`` so the daily-NAV engine (``daily_nav.py``) and
the trade-tracker can both depend on it without round-trip imports.  Earlier
the engine pulled ``_dynamic_half_spread`` and ``_pct_return`` out of
``tracker`` lazily inside helper functions to avoid the import cycle — this
module breaks that cycle for good.

Dependency tree after this extraction:
    spread.py    (depends only on config.settings — no src-internal deps)
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

from typing import Optional

from config.settings import settings


# Real-fill cost calibration (see settings.sim_use_real_fill_costs). When the
# pipeline / performance layer measures the average all-in one-way cost from
# actual IBKR fills, it sets this module-global; _one_side_cost then returns it
# flat for EVERY leg instead of the modeled half-spread + commission, so the
# simulation charges what execution really costs. None = use the model (the
# default until enough real fills accumulate, and always in tests — reset
# per-test by conftest). It is a single process-global, recomputed
# deterministically from the DB's broker fills each run/perf call.
_REAL_COST_OVERRIDE: Optional[float] = None
# Per-SESSION calibrated one-way cost fractions ({"rth": f, "extended": f,
# "overnight": f}) — installed alongside the flat override by
# tracker.calibrate_sim_costs once real fills support a per-session split
# (rth measured directly; off-RTH = rth × a shrunk session multiplier so the
# documented ×4/×10 priors hold until that session's own fills accrue). When
# absent, the flat blended override applies to every leg as before.
_REAL_COST_SESSION: Optional[dict] = None


def set_real_cost_override(fraction: Optional[float],
                           by_session: Optional[dict] = None) -> None:
    """Install (or clear with None) the real-fill one-way cost fraction that
    _one_side_cost returns for every leg, optionally with a per-session split.
    Clamped ≥ 0 — a net-favorable fill streak must never make the sim pay you
    to trade."""
    global _REAL_COST_OVERRIDE, _REAL_COST_SESSION
    _REAL_COST_OVERRIDE = None if fraction is None else max(0.0, float(fraction))
    if fraction is None or not by_session:
        _REAL_COST_SESSION = None
    else:
        _REAL_COST_SESSION = {str(k): max(0.0, float(v))
                              for k, v in by_session.items() if v is not None}


def get_real_cost_override() -> Optional[float]:
    return _REAL_COST_OVERRIDE


def get_real_cost_session_overrides() -> Optional[dict]:
    return dict(_REAL_COST_SESSION) if _REAL_COST_SESSION else None


def effective_cost_hurdle_pct() -> float:
    """The round-trip cost hurdle a horizon's net edge must clear — DERIVED
    from the calibrated real one-way cost when available:

        hurdle% = 2 × one-way% × cost_hurdle_safety

    so horizon selection self-tightens/loosens as measured execution costs
    drift, instead of judging edges against the frozen
    ``settings.horizon_cost_hurdle_pct`` (which stays as the fallback when no
    real-fill calibration exists, and when ``cost_hurdle_use_calibrated`` is
    off). Clamped to a sane [0.05, 2.0]% band — the hurdle is a decision
    threshold, and a degenerate calibration must not zero it out or make every
    horizon untradeable."""
    static = float(settings.horizon_cost_hurdle_pct)
    if not settings.cost_hurdle_use_calibrated or _REAL_COST_OVERRIDE is None:
        return static
    derived = 2.0 * _REAL_COST_OVERRIDE * 100.0 * float(settings.cost_hurdle_safety)
    return min(2.0, max(0.05, derived))


def _session_spread_multiplier(session) -> float:
    """Half-spread multiplier for the trading session a price was struck in.

    RTH (or unspecified — every pre-extended-hours record) → 1.0, so all
    historical numbers are bit-identical. Extended/overnight multipliers are
    deliberate over-estimates to be calibrated against IBKR paper fills later,
    same plan as ``commission_buffer``. Commission is session-independent —
    only the spread term widens off-hours.
    """
    if session == "extended":
        return float(settings.spread_extended_multiplier)
    if session == "overnight":
        return float(settings.spread_overnight_multiplier)
    return 1.0


def _dynamic_half_spread(price: float, asset_type: str = "STOCK", session=None) -> float:
    """One-way bid-ask half-spread as a fraction (NOT percent).

    ``session`` ("rth" | "extended" | "overnight" | None=rth) scales the tier
    via ``_session_spread_multiplier`` — books are thinner outside RTH.

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
    mult = _session_spread_multiplier(session)
    if asset_type == "ETF":
        return (0.00015 if price >= 100 else 0.00025) * mult
    if asset_type == "COMMODITY":
        return (0.00025 if price >= 100 else 0.0005) * mult
    # STOCK — price-tiered (1 bp = 0.0001)
    if price >= 100:
        base = 0.0003
    elif price >= 50:
        base = 0.0004
    elif price >= 10:
        base = 0.0008
    elif price >= 1:
        base = 0.0025
    elif price >= 0.10:
        base = 0.0075
    elif price >= 0.01:
        base = 0.0250
    else:
        base = 0.0500
    return base * mult


def _commission_fraction(price: float) -> float:
    """One-side commission as a fraction of traded notional — a deliberate
    CEILING, not a best estimate, so reported results err conservative.

    Converts IBKR's per-share commission schedule into percentage terms using
    the assumed position notional ``settings.commission_notional_usd`` (a
    deliberate constant — NOT live FX or live equity — so the return math
    stays 100% deterministic; see the setting's docstring).

        shares = notional / price
        ibkr_fixed  : max($1.00, $0.005  × shares), capped at 1% of trade value
        ibkr_tiered : max($0.35, $0.0035 × shares), capped at 1% of trade value
        none        : 0 (spread-only, legacy behavior)
        … then × settings.commission_buffer (default 1.5)

    The buffer is applied AFTER the min/cap schedule math — including after
    the 1%-of-value cap — as the intended fee ceiling: it covers the costs the
    published schedule excludes (SEC transaction fee + FINRA TAF on sells,
    exchange/clearing fees under tiered pricing, odd venue surcharges) and
    schedule drift. Actual commissions captured from broker fills
    (``broker_orders.commission`` in DuckDB) are the ground truth to calibrate
    the buffer against once paper data accumulates.

    The minimum-commission floor dominates at this system's order sizes, so
    cheap (high-share-count) names converge to the per-share rate while
    expensive names pay the flat minimum as a larger fraction of a smaller
    share count's notional. Non-positive prices return 0 — the model is
    undefined there, matching ``_dynamic_half_spread``.
    """
    model = (settings.commission_model or "none").lower()
    if model == "none" or price is None or price <= 0:
        return 0.0
    notional = float(settings.commission_notional_usd)
    if notional <= 0:
        return 0.0
    shares = notional / price
    if model == "ibkr_tiered":
        fee = max(0.35, 0.0035 * shares)
    else:  # "ibkr_fixed" (default for any unrecognized value — the pricier plan)
        fee = max(1.00, 0.005 * shares)
    fee = min(fee, 0.01 * notional)   # IBKR caps commission at 1% of trade value
    buffer = float(settings.commission_buffer)
    if buffer > 0:                    # ≤0 would silently zero out fees — treat as off
        fee *= buffer
    return fee / notional


def _one_side_cost(price: float, asset_type: str = "STOCK", session=None) -> float:
    """Total one-way transaction cost as a fraction: half-spread + commission.

    The single cost figure both return engines apply to entry/exit prices —
    ``_pct_return`` here and the daily-NAV walk's anchor marks in
    ``daily_nav.py`` — so the per-trade buy-and-hold return and the
    path-faithful daily compound charge identical costs. ``session`` widens
    the spread term outside RTH (commission is session-independent).

    When a real-fill calibration is installed (``set_real_cost_override`` —
    the measured average all-in one-way cost from actual IBKR fills), it is
    returned for every leg INSTEAD of the model, so the simulation charges
    what execution actually costs. When the calibration carries a per-SESSION
    split (rth measured; extended/overnight = rth × shrunk multipliers toward
    the ×4/×10 priors until those sessions' own fills accrue), the leg's own
    session picks its cost — the flat blend is the fallback. EXCEPT for legs
    priced below ``sim_real_fill_min_price``: the fills the calibration is
    measured from are liquid names, and charging that cost to a sub-$1
    instrument grossly understates its spread (a $0.054 warrant with a
    ~35%-wide book was being charged 8 bp — ARQQW, 2026-07-01). Those legs
    keep the modeled price-tiered cost.
    """
    if _REAL_COST_OVERRIDE is not None:
        min_px = float(settings.sim_real_fill_min_price)
        if price is not None and price >= min_px:
            if _REAL_COST_SESSION:
                return _REAL_COST_SESSION.get(session or "rth",
                                              _REAL_COST_SESSION.get("rth", _REAL_COST_OVERRIDE))
            return _REAL_COST_OVERRIDE
    return _dynamic_half_spread(price, asset_type, session) + _commission_fraction(price)


def _pct_return(action: str, entry: float, current: float, asset_type: str = "STOCK",
                entry_session=None, exit_session=None) -> float:
    """Percent return, sign-aware, with round-trip half-spread + commission.

    BUY  : paid the ask at entry (+cost), receive the bid at exit (−cost).
    SELL : shorted at the bid at entry (−cost), covered at the ask (+cost).
    Each leg's cost (half-spread + commission, see ``_one_side_cost``) is
    evaluated against its own price (entry and current independently), so a
    position that crosses a price tier naturally picks up the wider/narrower
    spread on each leg. Per-leg sessions widen the spread for a leg struck
    outside RTH (None = rth — every pre-extended-hours record).

    Returns ``0.0`` for non-positive prices — the round-trip is undefined
    there.  Callers refuse to trade at such prices; this guard exists to
    keep stats clean for any record that slipped through historically.
    """
    if entry is None or current is None or entry <= 0 or current <= 0:
        return 0.0
    entry_cost = _one_side_cost(entry, asset_type, entry_session)
    exit_cost  = _one_side_cost(current, asset_type, exit_session)
    if action == "BUY":
        effective_entry = entry   * (1 + entry_cost)
        effective_exit  = current * (1 - exit_cost)
        return (effective_exit - effective_entry) / effective_entry * 100
    # SELL = short
    effective_entry = entry   * (1 - entry_cost)
    effective_exit  = current * (1 + exit_cost)
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


def fmt_price_full(p) -> str:
    """Show the price at its full stored precision (up to 6 decimals).

    Different from ``fmt_price`` which truncates large prices to 2 decimals
    (typical penny ticks). Used in trade tables where the user wants to see
    the EXACT stored entry/exit value, including any sub-penny precision the
    underlying float carries (some data providers return prices like
    106.31999...). Trailing zeros are stripped but at least 2 decimals are
    always retained so the cents are obvious.
    """
    if p is None:
        return "N/A"
    try:
        p = float(p)
    except (TypeError, ValueError):
        return str(p)
    # 6 decimals captures any sub-penny precision; strip trailing zeros so
    # 106.32 doesn't render as 106.320000.
    s = f"{p:.6f}".rstrip("0")
    if s.endswith("."):
        s += "00"
    elif "." in s and len(s.split(".")[1]) == 1:
        s += "0"
    return s
