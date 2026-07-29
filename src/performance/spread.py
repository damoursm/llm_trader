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


# ── per-trade cost attribution (2026-07-23) ─────────────────────────────────
# The flat/session override above charges EVERY sim leg one calibrated number.
# Per-trade attribution refines that WITHOUT breaking ledger purity (every sim
# trade still assumes it fills): for a leg the broker actually filled, charge
# that leg's OWN realized cost; for an unfilled leg, charge the average cost of
# the trades that DID fill in the same tick (run); if that sample is too thin,
# the average for the leg's time-of-day period (rth / premarket / afterhours /
# overnight); and finally the modeled/global cost. Installed each tick by
# tracker.calibrate_sim_costs from the SAME DB fills as the flat override, so
# the daily-NAV compound stays deterministic (a per-run average can still drift
# as later fills land — identical to how the flat override already behaves).
_TICK_COST_FRAC: dict = {}       # run_id → mean one-way cost FRACTION of that run's filled legs
_SESSION_COST_FRAC: dict = {}    # fine bucket (rth|premarket|afterhours|overnight) → mean fraction
_LEG_REF_RUN: dict = {}          # client_ref → run_id (maps a leg to the tick it was decided in)


def set_cost_attribution(tick_costs: Optional[dict], session_costs: Optional[dict],
                         ref_to_run: Optional[dict]) -> None:
    """Install the per-trade cost lookups (or clear them all with None). Values
    are one-way FRACTIONS (e.g. 0.0018), already sanity-banded and min-sampled
    by the builder in tracker."""
    global _TICK_COST_FRAC, _SESSION_COST_FRAC, _LEG_REF_RUN
    _TICK_COST_FRAC = {str(k): max(0.0, float(v)) for k, v in (tick_costs or {}).items()}
    _SESSION_COST_FRAC = {str(k): max(0.0, float(v)) for k, v in (session_costs or {}).items()}
    _LEG_REF_RUN = {str(k): str(v) for k, v in (ref_to_run or {}).items()}


def session_bucket_fine(raw) -> str:
    """Time-of-day period of an ISO timestamp: ``rth | premarket | afterhours
    | overnight`` (ET). The single source of truth for the fine session split —
    tracker._session_of_iso_fine delegates here. Date-only/missing → 'rth'
    (every legacy record could only have traded in the regular session)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    if not raw or ("T" not in str(raw) and ":" not in str(raw)):
        return "rth"
    try:
        dt = datetime.fromisoformat(str(raw))
        et = ZoneInfo("America/New_York")
        dt = dt.astimezone(et) if dt.tzinfo is not None else dt.replace(tzinfo=et)
        mins = dt.hour * 60 + dt.minute
    except Exception:
        return "rth"
    if 9 * 60 + 30 <= mins < 16 * 60:
        return "rth"
    if 4 * 60 <= mins < 9 * 60 + 30:
        return "premarket"
    if 16 * 60 <= mins < 20 * 60:
        return "afterhours"
    return "overnight"


def _coarse_of_fine(fine: str) -> str:
    """Fine bucket → the coarse session the spread multipliers key on."""
    if fine in ("premarket", "afterhours"):
        return "extended"
    return fine if fine in ("rth", "overnight") else "rth"


def real_leg_cost_frac(side: str, filled_qty, model_price, fill_price,
                       commission) -> Optional[float]:
    """One-way cost FRACTION of a single REAL filled leg — commission as a
    fraction of notional plus the cost-normalized execution-vs-decision
    slippage (a BUY is adverse filling above model, a SELL below). None when
    the leg is unusable OR when the measured cost is an implausible outlier
    (|cost| beyond ``sim_real_fill_cost_sanity_pct`` — a stale decision price,
    not real execution; the caller then falls through to an estimate). Mirrors
    broker_view.leg_one_way_cost_pct, inlined here to keep spread import-free."""
    try:
        fq = int(filled_qty or 0)
        fp = float(fill_price) if fill_price is not None else 0.0
    except (TypeError, ValueError):
        return None
    if fq <= 0 or fp <= 0:
        return None
    comm = 0.0
    try:
        comm = float(commission or 0.0)
    except (TypeError, ValueError):
        comm = 0.0
    comm_pct = comm / (fq * fp) * 100.0
    slip = 0.0
    try:
        mp = float(model_price) if model_price is not None else 0.0
        if mp > 0:
            is_buy = str(side or "").upper() == "BUY"
            slip = ((fp - mp) if is_buy else (mp - fp)) / mp * 100.0
    except (TypeError, ValueError):
        slip = 0.0
    pct = comm_pct + slip
    band = abs(float(getattr(settings, "sim_real_fill_cost_sanity_pct", 2.0) or 0.0))
    if band > 0 and abs(pct) > band:
        return None
    return max(0.0, pct) / 100.0


def resolve_leg_cost(*, side: str, price: float, asset_type: str, session_fine: str,
                     ref, filled_qty, model_price, fill_price, commission,
                     coarse_session: Optional[str] = None, run=None) -> float:
    """The per-trade one-way cost FRACTION for one leg, by the hierarchy:
      1. the leg's OWN realized cost, if it filled at the broker;
      2. else the average of the trades that filled in the SAME tick (run);
      3. else the average for the leg's time-of-day period;
      4. else the modeled / global-override cost.
    Attribution is OFF (→ straight to the modeled/global cost, i.e. today's
    behaviour) when the master switch is off or no lookups are installed.
    Sub-``sim_real_fill_min_price`` legs always use the model — the fills the
    averages are built from are liquid names, so charging that to a penny stock
    understates its true spread (same guard as ``_one_side_cost``).

    ``coarse_session`` is the leg's STORED session stamp; it drives the modeled
    tier-4 fallback (the source of truth for the spread multiplier) so a trade
    whose ``entry_session``/``exit_session`` is set still gets the right modeled
    spread even when its timestamp — the fine-bucket source — is absent. Falls
    back to the fine bucket's coarse mapping when the stamp is missing."""
    coarse = coarse_session or _coarse_of_fine(session_fine)
    if not settings.sim_per_trade_cost_attribution:
        return _one_side_cost(price, asset_type, coarse)
    try:
        above_min = price is not None and float(price) >= float(settings.sim_real_fill_min_price)
    except (TypeError, ValueError):
        above_min = False
    if above_min:
        real = real_leg_cost_frac(side, filled_qty, model_price, fill_price, commission)
        if real is not None:
            return real
        # Tick average: the run the leg was DECIDED in. The entry leg passes it
        # directly (trade.run_id); otherwise fall back to the ref→run map (only
        # populated for FILLED legs — so this second path only helps a filled
        # leg whose own cost was banded out as an outlier).
        rk = str(run) if run is not None else (_LEG_REF_RUN.get(str(ref)) if ref is not None else None)
        if rk is not None and rk in _TICK_COST_FRAC:
            return _TICK_COST_FRAC[rk]
        if session_fine in _SESSION_COST_FRAC:
            return _SESSION_COST_FRAC[session_fine]
    return _one_side_cost(price, asset_type, coarse)


# ── short borrow / carry (2026-07-25) ───────────────────────────────────────
#
# The one genuinely direction-ASYMMETRIC cost. A short pays a daily stock-loan
# fee (and, on hard-to-borrow names, a locate cost) that a long simply does not;
# every other cost in this module — half-spread, commission, session multiplier
# — is symmetric. Until now nothing charged it anywhere in the spread, tracker
# or NAV modules, so every short's simulated return read BETTER than reality by
# roughly the borrow rate times the holding period.
#
# It is a HOLDING cost, not a per-leg one, so it does not go through
# ``_one_side_cost``: it accrues per day held and is subtracted from the return.

def borrow_annual_pct(trade: Optional[dict] = None) -> float:
    """Annual borrow rate (%) to charge a short, preferring the REAL rate.

    ``trade["borrow_fee_pct"]`` is used when present — the broker's own
    stock-loan rate for that name (``Broker.get_short_borrow`` → ``fee_pct``),
    stamped at entry. Falls back to ``settings.short_borrow_annual_pct``, a
    blended assumption for a universe that mixes large caps (typically well
    under 1%) with small caps and hard-to-borrow names (which can run far
    higher). Same prefer-measured-over-modelled idiom as the real-fill costs.
    """
    if trade is not None:
        raw = trade.get("borrow_fee_pct")
        if raw is not None:
            try:
                v = float(raw)
                if v >= 0:
                    return v
            except (TypeError, ValueError):
                pass
    try:
        return max(0.0, float(settings.short_borrow_annual_pct))
    except (TypeError, ValueError):
        return 0.0


def borrow_cost_fraction(action: str, days_held: Optional[float],
                         trade: Optional[dict] = None) -> float:
    """Borrow carry as a RETURN-REDUCING fraction for one position.

    Zero for longs, for non-positive holding periods, and when the feature is
    off. Uses a 365-day year: borrow accrues on calendar days (you pay over a
    weekend), unlike the trading-day conventions elsewhere in the ledger.
    """
    if not getattr(settings, "enable_short_borrow_cost", False):
        return 0.0
    if str(action or "").upper() != "SELL":
        return 0.0
    try:
        d = float(days_held or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if d <= 0:
        return 0.0
    return borrow_annual_pct(trade) / 100.0 * (d / 365.0)


def resolve_trade_leg_cost(trade: dict, which: str, price: float, session_fine: str) -> float:
    """One-way cost FRACTION for a real trade's entry or exit leg. Extracts the
    broker leg fields from the trade UNIFORMLY so ``_pct_return`` (tracker) and
    the daily-NAV anchors (daily_nav) resolve the SAME cost for the same leg —
    the invariant that keeps the two return engines charging identical costs.
    ``price`` is the price being priced (entry_price; exit_price for a closed
    leg; the live mark for an open-trade M2M — which carries no exit fill, so it
    correctly falls to the tick/session/model estimate)."""
    action = str(trade.get("action") or "BUY").upper()
    asset_type = trade.get("type", "STOCK")
    if which == "entry":
        side = "BUY" if action == "BUY" else "SELL"
        return resolve_leg_cost(
            side=side, price=price, asset_type=asset_type, session_fine=session_fine,
            coarse_session=trade.get("entry_session"), run=trade.get("run_id"),
            ref=trade.get("broker_client_ref") or trade.get("recommendation_id"),
            filled_qty=trade.get("broker_fill_qty"), model_price=trade.get("entry_price"),
            fill_price=trade.get("broker_fill_price"), commission=trade.get("broker_commission"))
    # exit leg — the closing side is the opposite of the entry
    side = "SELL" if action == "BUY" else "BUY"
    return resolve_leg_cost(
        side=side, price=price, asset_type=asset_type, session_fine=session_fine,
        coarse_session=trade.get("exit_session"),
        ref=trade.get("broker_exit_client_ref"),
        filled_qty=trade.get("broker_exit_fill_qty"), model_price=trade.get("exit_price"),
        fill_price=trade.get("broker_exit_fill_price"), commission=trade.get("broker_exit_commission"))


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
                entry_session=None, exit_session=None,
                entry_cost: Optional[float] = None,
                exit_cost: Optional[float] = None,
                borrow_cost: Optional[float] = None) -> float:
    """Percent return, sign-aware, with round-trip half-spread + commission.

    BUY  : paid the ask at entry (+cost), receive the bid at exit (−cost).
    SELL : shorted at the bid at entry (−cost), covered at the ask (+cost).
    Each leg's cost (half-spread + commission, see ``_one_side_cost``) is
    evaluated against its own price (entry and current independently), so a
    position that crosses a price tier naturally picks up the wider/narrower
    spread on each leg. Per-leg sessions widen the spread for a leg struck
    outside RTH (None = rth — every pre-extended-hours record).

    ``entry_cost`` / ``exit_cost`` (fractions) override the modeled leg cost
    when supplied — the per-trade attribution path (see ``resolve_leg_cost``)
    passes each leg's REALIZED-or-estimated cost so a real trade is charged
    what its own execution cost, not a portfolio-flat number. None → the
    modeled/global cost, i.e. the original behaviour.

    ``borrow_cost`` (a fraction) is the SHORT-only carry for the holding
    period — see ``borrow_cost_fraction``. Ignored for longs, which never pay
    it. None = no borrow charged (the pre-2026-07-25 behaviour).

    Returns ``0.0`` for non-positive prices — the round-trip is undefined
    there.  Callers refuse to trade at such prices; this guard exists to
    keep stats clean for any record that slipped through historically.
    """
    if entry is None or current is None or entry <= 0 or current <= 0:
        return 0.0
    entry_cost = entry_cost if entry_cost is not None else _one_side_cost(entry, asset_type, entry_session)
    exit_cost  = exit_cost  if exit_cost  is not None else _one_side_cost(current, asset_type, exit_session)
    if action == "BUY":
        effective_entry = entry   * (1 + entry_cost)
        effective_exit  = current * (1 - exit_cost)
        return (effective_exit - effective_entry) / effective_entry * 100
    # SELL = short. Borrow carry (a SHORT-only holding cost — see
    # borrow_cost_fraction) reduces the realised return; it is not a price
    # adjustment, so it is subtracted after the round trip.
    effective_entry = entry   * (1 - entry_cost)
    effective_exit  = current * (1 + exit_cost)
    gross = (effective_entry - effective_exit) / effective_entry * 100
    return gross - (borrow_cost or 0.0) * 100.0


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
