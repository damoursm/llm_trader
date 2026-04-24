"""Shared data models used across the pipeline."""

from datetime import date, datetime
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class NewsArticle(BaseModel):
    title: str
    summary: str
    url: str
    source: str
    published_at: datetime


class TickerSnapshot(BaseModel):
    ticker: str
    price: float
    pct_change_1d: float
    pct_change_5d: float
    volume: int
    market_cap: Optional[float] = None


_BULLISH_TX_TYPES: frozenset = frozenset({
    "unusual_call", "13d_activist_stake", "13g_passive_stake",
    "13f_new_position", "13f_increase",
})
_BEARISH_TX_TYPES: frozenset = frozenset({
    "unusual_put", "planned_sale_144", "13f_exit", "13f_decrease",
})
_ACTION_LABELS: dict = {
    "unusual_call":       "UNUSUAL CALL SWEEP",
    "unusual_put":        "UNUSUAL PUT SWEEP",
    "13d_activist_stake": "ACTIVIST STAKE (13D)",
    "13g_passive_stake":  "PASSIVE STAKE (13G)",
    "planned_sale_144":   "PLANNED SALE (Form 144)",
    "13f_new_position":   "NEW POSITION (13F)",
    "13f_increase":       "INCREASED (13F)",
    "13f_exit":           "EXITED (13F)",
    "13f_decrease":       "DECREASED (13F)",
    "form_4_filing":      "FORM 4 FILING",
    "exchange":           "EXCHANGED",
    "purchase":           "BOUGHT",
    "sale":               "SOLD",
    "sale_partial":       "SOLD (partial)",
}


class InsiderTrade(BaseModel):
    ticker: str
    trader_name: str
    trader_type: str           # "politician" | "corporate_insider" | "options_flow" | "institutional"
    role: str                  # "Representative", "Senator", "CEO", "CFO", etc.
    transaction_type: str      # "purchase" | "sale" | "unusual_call" | "13d_activist_stake" | etc.
    amount_range: str          # e.g. "$15,001 - $50,000"
    transaction_date: date
    disclosure_date: date
    notes: str = ""

    @property
    def is_bullish(self) -> bool:
        tx = self.transaction_type
        return "purchase" in tx or tx in _BULLISH_TX_TYPES

    @property
    def action_label(self) -> str:
        return _ACTION_LABELS.get(
            self.transaction_type,
            "BOUGHT" if "purchase" in self.transaction_type else "SOLD",
        )


Direction = Literal["BULLISH", "BEARISH", "NEUTRAL"]


class MacroContext(BaseModel):
    """Macro regime snapshot derived from FRED indicators."""
    # Yield curve
    yield_spread_10y2y: Optional[float] = None   # 10Y - 2Y spread in %
    yield_curve_signal: str = "UNKNOWN"           # INVERTED | FLAT | NORMAL | STEEP

    # Monetary policy
    fed_funds_rate: Optional[float] = None        # Effective Fed Funds Rate %

    # Inflation
    cpi_yoy: Optional[float] = None              # CPI year-over-year %
    inflation_signal: str = "UNKNOWN"            # HIGH | ELEVATED | MODERATE | LOW

    # Labor market
    unemployment_rate: Optional[float] = None
    unemployment_trend: str = "STABLE"           # RISING | STABLE | FALLING

    # Credit markets
    hy_spread: Optional[float] = None            # HY OAS spread %
    ig_spread: Optional[float] = None            # IG OAS spread %
    credit_signal: str = "UNKNOWN"               # STRESSED | ELEVATED | NORMAL | TIGHT

    # Liquidity
    m2_growth_yoy: Optional[float] = None        # M2 year-over-year %

    # Overall
    regime: str = "UNKNOWN"                      # EXPANSION | SLOWDOWN | LATE_CYCLE | RECESSION
    summary: str = ""                            # Human-readable 2-3 sentence summary


class COTSignal(BaseModel):
    """Positioning signal for a single futures contract from the CFTC COT report."""
    contract: str                   # friendly name, e.g. "Gold"
    tickers: List[str]              # related ETF/futures tickers, e.g. ["GLD", "IAU", "GDX"]
    report_date: date
    net_speculator_pct: float       # (longs - shorts) / OI × 100
    net_change_wow: float           # week-over-week change in net_speculator_pct
    percentile_52w: float           # 0-100: where current positioning sits in 52-week range
    signal: str                     # EXTREME_LONG | BULLISH_TREND | NEUTRAL | BEARISH_TREND | EXTREME_SHORT
    direction: str                  # BULLISH | BEARISH | NEUTRAL (contrarian applied at extremes)
    summary: str


class COTContext(BaseModel):
    """All COT signals for the current week."""
    signals: List[COTSignal]
    report_date: date
    summary: str                    # Human-readable overview of notable extremes/trends


class IPOFiling(BaseModel):
    """A single S-1 or S-11 registration statement from SEC EDGAR."""
    company: str
    filing_date: date
    form_type: str          # "S-1" | "S-1/A" | "S-11" | "S-11/A"
    sector: str             # inferred sector
    is_amendment: bool


class IPOContext(BaseModel):
    """IPO pipeline intelligence derived from recent S-1/S-11 filings."""
    filings: List[IPOFiling]         # initial registrations (S-1, S-11)
    amendments: List[IPOFiling]      # amendments (S-1/A, S-11/A) — pipeline advancing
    sector_counts: Dict[str, int]    # sector → count of initial filings
    hot_sectors: List[str]           # top sectors by filing count
    total_new: int
    total_amendments: int
    lookback_days: int
    report_date: date
    summary: str


class TICKContext(BaseModel):
    """NYSE TICK index breadth / short-term exhaustion signal."""
    tick_high:  Optional[float] = None   # session maximum TICK reading
    tick_low:   Optional[float] = None   # session minimum TICK reading
    tick_close: Optional[float] = None   # session closing TICK reading

    session_date: date
    signal: str = "UNKNOWN"             # EXTREME_BULLS | EXTREME_BEARS | WHIPSAW | NEUTRAL | UNKNOWN
    direction: str = "NEUTRAL"          # contrarian: EXTREME_BULLS → BEARISH, EXTREME_BEARS → BULLISH

    # Pattern over recent sessions
    extreme_high_count: int = 0         # sessions with TICK > +1000 in 5-day lookback
    extreme_low_count:  int = 0         # sessions with TICK < −1000 in 5-day lookback

    report_date: date
    summary: str


class CreditContext(BaseModel):
    """Credit market leading indicator — HYG vs SPY divergence."""
    hyg_price:     Optional[float] = None   # HYG last close
    spy_price:     Optional[float] = None   # SPY last close
    hyg_return_1d: Optional[float] = None   # HYG 1-day return %
    hyg_return_5d: Optional[float] = None   # HYG 5-day return %
    spy_return_1d: Optional[float] = None   # SPY 1-day return %
    spy_return_5d: Optional[float] = None   # SPY 5-day return %
    divergence_5d: Optional[float] = None   # hyg_5d − spy_5d; negative = HYG lagging equities
    signal:    str = "UNKNOWN"              # CREDIT_STRESS | CREDIT_CAUTION | NEUTRAL | CREDIT_STRONG | CREDIT_SURGE
    direction: str = "NEUTRAL"             # BEARISH | NEUTRAL | BULLISH
    report_date: date
    summary: str


class VIXContext(BaseModel):
    """VIX volatility regime and term structure."""
    # Spot levels
    vix:   Optional[float] = None   # 30-day S&P 500 implied vol (^VIX)
    vxn:   Optional[float] = None   # 30-day Nasdaq implied vol (^VXN)
    vvix:  Optional[float] = None   # vol-of-vol / VIX of VIX (^VVIX)
    vix9d: Optional[float] = None   # 9-day VIX (^VIX9D)
    vix3m: Optional[float] = None   # 3-month VIX (^VIX3M)
    vix6m: Optional[float] = None   # 6-month mid-term VIX (^VXMT)

    # Term structure
    term_structure: str = "UNKNOWN"    # CONTANGO | FLAT | BACKWARDATION
    slope_1m_3m: Optional[float] = None  # VIX3M − VIX (positive = contango)

    # Regime
    vix_signal: str = "UNKNOWN"        # PANIC | EXTREME_FEAR | HIGH | ELEVATED | NORMAL | LOW | COMPLACENCY
    vix_direction: str = "NEUTRAL"     # contrarian: EXTREME_FEAR → BULLISH bias

    report_date: date
    summary: str


class PutCallSignal(BaseModel):
    """Put/call ratio signal for a single ticker."""
    ticker: str
    put_volume: int
    call_volume: int
    put_call_ratio: float
    signal: str    # EXTREME_PUTS | PUTS_HEAVY | BALANCED | CALLS_HEAVY | EXTREME_CALLS
    direction: str # BEARISH | BEARISH | NEUTRAL | BULLISH | BULLISH
    summary: str


class PutCallContext(BaseModel):
    """Market-wide and per-ticker put/call ratio sentiment."""
    # Market-wide (CBOE equity P/C ratio) — contrarian indicator
    market_pc_ratio: Optional[float] = None
    market_signal: str = "UNKNOWN"    # EXTREME_GREED | GREED | NEUTRAL | FEAR | EXTREME_FEAR
    market_direction: str = "NEUTRAL" # contrarian: FEAR → BULLISH bias

    # Per-ticker extremes (only tickers with non-BALANCED readings)
    ticker_signals: List["PutCallSignal"] = []

    report_date: date
    summary: str


class EarningsEvent(BaseModel):
    """An upcoming earnings report for a single ticker."""
    ticker: str
    earnings_date: date
    days_until: int
    estimated_eps: Optional[float] = None
    is_confirmed: bool = False


class EarningsContext(BaseModel):
    """Upcoming earnings calendar for the watchlist."""
    upcoming: List[EarningsEvent]   # sorted by date ascending
    report_date: date
    summary: str


class GEXSignal(BaseModel):
    """Gamma Exposure (GEX) signal for a single ticker."""
    ticker: str
    spot_price: float
    net_gex_bn: float            # net GEX in $B  (positive = stabilising, negative = amplifying)
    gex_normalized: float        # net / |total|  ∈ [-1, +1]
    gex_signal: str              # PINNED | NEUTRAL | AMPLIFIED
    gamma_flip: Optional[float]  # price below which moves accelerate; None if not found
    max_pain: Optional[float]    # strike minimising total options $ loss at nearest expiry
    expected_move_pct: float     # OI-weighted straddle / spot as %  (market-implied ±1σ range)
    max_pain_bias: str           # BULLISH (price < max_pain) | BEARISH (price > max_pain) | NEUTRAL
    oi_skew: float = 0.0         # OI-weighted directional lean ∈ [-1,+1]; +ve = call-heavy (bullish)
    dominant_expiry: str         # nearest expiry with meaningful OI
    report_date: date
    summary: str


class GEXContext(BaseModel):
    """GEX context for all covered tickers."""
    signals: List[GEXSignal]
    report_date: date
    summary: str                 # e.g. "SPY: PINNED; QQQ: AMPLIFIED; IWM: NEUTRAL"


class McClellanContext(BaseModel):
    """NYSE McClellan Oscillator and Summation Index — A/D breadth momentum."""
    # Core values
    oscillator: float                       # 19d EMA − 39d EMA of net advances (daily A−D)
    oscillator_5d_ago: Optional[float] = None
    summation: float                        # running cumulative total of oscillator
    summation_5d_ago: Optional[float] = None
    ema19: float                            # 19-day EMA of net advances (short / "10% trend")
    ema39: float                            # 39-day EMA of net advances (long  / "5% trend")

    # Classification
    osc_signal: str    # OVERBOUGHT | BULLISH_MOMENTUM | NEUTRAL | BEARISH_MOMENTUM | OVERSOLD
    sum_signal: str    # EXTENDED_BULL | BULL_TREND | NEUTRAL | BEAR_TREND | EXTENDED_BEAR
    direction: str     # BULLISH | BEARISH | NEUTRAL

    # Crossover flags (last 3 sessions)
    is_bullish_cross: bool = False   # oscillator crossed above 0
    is_bearish_cross: bool = False   # oscillator crossed below 0

    report_date: date
    summary: str


class BreadthContext(BaseModel):
    """S&P 500 market breadth — % of sector ETFs above their 200-day SMA."""
    pct_above_200d: float                      # 0–100: % of covered sector ETFs above 200d SMA
    pct_above_200d_5d_ago: Optional[float] = None  # same reading 5 trading days ago
    etf_count: int                             # total ETFs checked
    etfs_above: int                            # how many are above their 200d SMA
    signal: str                                # BREADTH_COLLAPSE | BREADTH_WEAK | BREADTH_MIXED | BREADTH_HEALTHY | BREADTH_EXTENDED
    direction: str                             # BULLISH | BEARISH | NEUTRAL
    is_breadth_thrust: bool = False            # rising ≥8pp from sub-35% reading = confirmed thrust
    spy_above_200d: Optional[bool] = None      # whether SPY itself is above its 200d SMA
    spy_200d_distance_pct: Optional[float] = None  # % SPY is above/below its 200d SMA
    report_date: date
    summary: str


class HighsLowsContext(BaseModel):
    """52-week new highs vs. new lows — market breadth divergence signal."""
    # Current counts
    highs_count: int            # tickers near 52-week high (within 5% threshold)
    lows_count: int             # tickers near 52-week low  (within 5% threshold)
    neutral_count: int          # tickers in the middle
    total_count: int            # total tickers with valid data

    # Percentages and spread
    pct_near_highs: float       # highs_count / total_count × 100
    pct_near_lows: float        # lows_count  / total_count × 100
    hl_spread: float            # pct_near_highs − pct_near_lows  ∈ [−100, +100]

    # Historical spreads (for trend detection)
    hl_spread_5d_ago:  Optional[float] = None
    hl_spread_10d_ago: Optional[float] = None

    # SPY reference (index context for divergence)
    spy_pct_from_52w_high: Optional[float] = None   # ≤ 0  (negative = below high)
    spy_pct_from_52w_low:  Optional[float] = None   # ≥ 0  (positive = above low)

    # Classification
    signal: str     # STRONG_HIGHS | HIGHS_DOMINATE | BALANCED | LOWS_DOMINATE | STRONG_LOWS
    direction: str  # BULLISH | BEARISH | NEUTRAL

    # Divergence flags (precede index reversals by 1–2 weeks)
    is_bearish_divergence: bool = False  # SPY near 52w high, HL spread declining
    is_bullish_divergence: bool = False  # SPY near 52w low,  HL spread rising

    report_date: date
    summary: str


class FedWatchContext(BaseModel):
    """Market-implied Fed rate expectations derived from T-bill spreads and FOMC calendar."""
    # Current FOMC target range
    ff_upper: float            # upper bound of FF target (DFEDTARU)
    ff_lower: float            # lower bound (DFEDTARL)
    ff_midpoint: float         # midpoint = (upper + lower) / 2

    # Market-implied rates from T-bills (proxy for expected average FF rate)
    tbill_3m: Optional[float] = None   # DTB3  — 90-day forward expectation
    tbill_6m: Optional[float] = None   # DTB6  — 180-day forward expectation
    tbill_12m: Optional[float] = None  # DTB1YR — 365-day forward expectation

    # Rate-change expectations (positive = cuts expected, negative = hikes)
    implied_cuts_3m_bp: float = 0.0    # basis points of cuts priced into 3m T-bill
    implied_cuts_6m_bp: float = 0.0    # ... 6m T-bill
    implied_cuts_12m_bp: float = 0.0   # ... 12m T-bill

    # Per-meeting probabilities for the NEXT FOMC meeting
    next_meeting: Optional[date] = None
    days_to_next_meeting: Optional[int] = None
    p_cut_next: float = 0.0    # P(≥25bp cut at next meeting)
    p_hold_next: float = 1.0   # P(no change at next meeting)
    p_hike_next: float = 0.0   # P(≥25bp hike at next meeting)

    # Week-over-week trend (5 trading days)
    tbill_3m_5d_ago: Optional[float] = None
    rate_trend: str = "NEUTRAL"   # DOVISH_SHIFT | NEUTRAL | HAWKISH_SHIFT

    # Classification
    signal: str     # STRONGLY_DOVISH | DOVISH | MILDLY_DOVISH | NEUTRAL | MILDLY_HAWKISH | HAWKISH | STRONGLY_HAWKISH
    direction: str  # BULLISH | NEUTRAL | BEARISH

    report_date: date
    summary: str


class MacroSurpriseIndicator(BaseModel):
    """Per-indicator surprise result from the CESI-style computation."""
    series_id: str        # FRED series ID, e.g. "PAYEMS"
    name: str             # friendly name, e.g. "Nonfarm Payrolls"
    unit: str             # display unit, e.g. "k jobs"
    actual: float         # most recent reading (or MoM change)
    expected: float       # trailing 3-period average
    surprise: float       # actual − expected
    z_score: float        # sign-adjusted z-score ∈ [-3, +3]
    signal: str           # BEAT | IN_LINE | MISS
    release_date: str     # date of most recent non-missing observation


class MacroSurpriseContext(BaseModel):
    """Composite economic surprise score across 6 FRED indicators."""
    score: float          # weighted composite ∈ [-1, +1]
    signal: str           # STRONG_BEAT | MILD_BEAT | NEUTRAL | MILD_MISS | STRONG_MISS
    direction: str        # BULLISH | NEUTRAL | BEARISH
    indicators: List[MacroSurpriseIndicator]
    beats: int
    misses: int
    in_line: int
    report_date: date
    summary: str


class WhisperSignal(BaseModel):
    """Per-ticker earnings whisper proxy signal."""
    ticker: str

    # Upcoming earnings (if within the look-ahead window)
    earnings_date: Optional[date] = None
    days_until_earnings: Optional[int] = None

    # Historical beat/miss record (last 4–8 quarters of yfinance earnings_dates)
    quarters_analyzed: int = 0
    beat_count:  int = 0
    miss_count:  int = 0
    beat_rate_pct: float = 0.0           # 0–100
    avg_eps_surprise_pct: float = 0.0    # mean Surprise(%) over history; positive = company beats

    # Implied whisper (consensus × (1 + avg_beat_pct/100))
    current_eps_estimate: Optional[float] = None   # current quarter sell-side consensus
    implied_whisper:      Optional[float] = None   # historical-beat-adjusted expectation
    whisper_gap_pct:      Optional[float] = None   # ≈ avg_eps_surprise_pct; positive = market expects above consensus

    # Consensus revision trend (from eps_trend — how estimate has moved 7/30d)
    eps_trend_current: Optional[float] = None
    eps_trend_7d:      Optional[float] = None
    eps_trend_30d:     Optional[float] = None
    eps_trend_direction: str = "STABLE"            # REVISING_UP | STABLE | REVISING_DOWN

    # Net analyst revisions (from eps_revisions)
    revisions_up_30d:   int = 0
    revisions_down_30d: int = 0

    # Classification
    signal:    str = "NEUTRAL"   # BEAT_LIKELY | BEAT_POSSIBLE | NEUTRAL | MISS_POSSIBLE | MISS_LIKELY
    direction: str = "NEUTRAL"   # BULLISH | NEUTRAL | BEARISH
    summary:   str = ""


class WhisperContext(BaseModel):
    """Aggregate earnings whisper proxy context across the watchlist."""
    signals: List["WhisperSignal"]
    n_beat_likely:   int = 0
    n_beat_possible: int = 0
    n_miss_possible: int = 0
    n_miss_likely:   int = 0
    avg_beat_rate_pct: float = 0.0   # mean historical beat rate across all tickers
    report_date: date
    summary: str


class TickerRevisionData(BaseModel):
    """Per-ticker analyst estimate revision momentum (recent 0-30d vs prior 31-60d window)."""
    ticker: str
    recent_upgrades:   int = 0     # upgrades in last 30 days
    recent_downgrades: int = 0
    recent_pt_raises:  int = 0     # PT raises on maintained coverage (last 30d)
    recent_pt_cuts:    int = 0
    prior_upgrades:    int = 0     # upgrades in 31-60 days ago
    prior_downgrades:  int = 0
    prior_pt_raises:   int = 0
    prior_pt_cuts:     int = 0
    momentum_score:  float = 0.0   # [-1, +1]: positive = accelerating positive revisions
    direction: str = "STABLE"      # IMPROVING | STABLE | DETERIORATING
    avg_pt_current: Optional[float] = None   # mean PT across recent window (with PT data)
    avg_pt_prior:   Optional[float] = None   # mean PT across prior window
    pt_change_pct:  Optional[float] = None   # % change in avg PT between windows
    n_firms: int = 0               # unique firms covering in the full 60d window


class RevisionMomentumContext(BaseModel):
    """Estimate revision momentum — analyst consensus trend across the watchlist."""
    tickers:     List[TickerRevisionData]
    breadth_score: float          # mean momentum ∈ [-1, +1]
    signal: str                   # STRONG_IMPROVING | IMPROVING | NEUTRAL | DETERIORATING | STRONG_DETERIORATING
    direction: str                # BULLISH | NEUTRAL | BEARISH
    top_improving:     List[str]  # tickers with strongest positive momentum
    top_deteriorating: List[str]  # tickers with strongest negative momentum
    report_date: date
    summary: str


class SeasonalEffect(BaseModel):
    """A single active calendar/seasonal effect."""
    name: str             # e.g. "Month-End Rebalancing"
    direction: str        # BULLISH | BEARISH | NEUTRAL
    assets_affected: str  # e.g. "equities broadly; especially if month had equity losses"
    description: str      # human-readable explanation of the effect


class SeasonalityContext(BaseModel):
    """Seasonal calendar context — pure date math, no API calls."""
    today: date
    month: int
    month_name: str
    quarter: int

    # Monthly baseline bias
    monthly_bias: str          # BULLISH | NEUTRAL | BEARISH
    monthly_signal: str        # e.g. APRIL_STRONG | SEPTEMBER_WEAK | SELL_IN_MAY
    monthly_description: str

    # Calendar window flags
    in_month_end_window: bool      # last 3 calendar days of month
    in_month_start_window: bool    # first 3 calendar days of month
    in_quarter_end_window: bool    # last 5 calendar days of quarter-end month
    in_quarter_start_window: bool  # first 5 calendar days of quarter-start month
    in_january_effect: bool        # January 1–15 small-cap rebound window
    is_fiscal_year_end: bool       # June or December quarter-end (more intense window dressing)

    # Active effects and composite signal
    active_effects: List["SeasonalEffect"]
    composite_signal: str      # STRONG_TAILWIND | TAILWIND | NEUTRAL | HEADWIND | STRONG_HEADWIND
    composite_direction: str   # BULLISH | NEUTRAL | BEARISH
    summary: str


class BondInternalsContext(BaseModel):
    """Bond market internals — macro regime signals from Treasury and credit ETFs (yfinance)."""
    # Raw Treasury yields (%)
    yield_10y:  Optional[float] = None     # ^TNX — 10-year Treasury yield
    yield_3m:   Optional[float] = None     # ^IRX — 13-week T-bill yield
    yield_5y:   Optional[float] = None     # ^FVX — 5-year Treasury yield
    yield_30y:  Optional[float] = None     # ^TYX — 30-year Treasury yield

    # Yield curve spreads (percentage points)
    spread_10y_3m:  Optional[float] = None   # 10Y − 3M: premier recession predictor
    spread_10y_5y:  Optional[float] = None   # 10Y − 5Y: mid-curve steepness
    spread_30y_10y: Optional[float] = None   # 30Y − 10Y: long-end term premium
    curve_signal: str = "UNKNOWN"            # DEEPLY_INVERTED | INVERTED | FLAT | NORMAL | STEEP

    # TLT (20+ year Treasury ETF) momentum — proxy for long-rate direction
    tlt_return_5d:  Optional[float] = None   # 1-week return %
    tlt_return_20d: Optional[float] = None   # 4-week return %
    tlt_return_40d: Optional[float] = None   # 8-week return %
    tlt_signal: str = "UNKNOWN"              # RALLYING_STRONG | RALLYING | FLAT | FALLING | FALLING_STRONG

    # Duration positioning: TLT vs IEF (long-end vs intermediate, 5-day spread)
    tlt_ief_spread_5d: Optional[float] = None   # TLT 5d return − IEF 5d return
    tlt_ief_signal: str = "UNKNOWN"              # LONG_END_PRESSURE | FLAT | LONG_END_RALLY

    # Inflation expectations: TIP vs IEF (5-day spread)
    tip_ief_spread_5d: Optional[float] = None    # TIP 5d return − IEF 5d return
    real_yield_signal: str = "UNKNOWN"            # REAL_RATES_RISING | NEUTRAL | REAL_RATES_FALLING

    # IG credit risk premium: LQD vs TLT (5-day spread)
    lqd_tlt_spread_5d: Optional[float] = None    # LQD 5d return − TLT 5d return
    ig_credit_signal: str = "UNKNOWN"             # IG_STRESS | IG_CAUTION | NEUTRAL | IG_STRONG

    # Bond-equity divergence: TLT/IEF vs SPY (5-day return spread)
    spy_return_5d:   Optional[float] = None   # SPY 5-day return %
    spy_return_20d:  Optional[float] = None   # SPY 20-day return %
    tlt_spy_div_5d:  Optional[float] = None   # TLT 5d − SPY 5d (positive = bonds leading)
    ief_spy_div_5d:  Optional[float] = None   # IEF 5d − SPY 5d (intermediate-bond confirmation)
    # EQUITY_CATCHUP_LIKELY | EQUITY_CATCHUP_POSSIBLE | NEUTRAL |
    # EQUITY_SELLOFF_RISK | SYNCHRONIZED_RISK_OFF | SYNCHRONIZED_RISK_ON
    bond_equity_signal: str = "NEUTRAL"
    bond_equity_direction: str = "NEUTRAL"    # BULLISH | NEUTRAL | BEARISH

    # Composite regime
    regime: str = "UNKNOWN"     # RISK_OFF | DEFENSIVE | NEUTRAL | CONSTRUCTIVE | RISK_ON | REFLATIONARY
    direction: str = "NEUTRAL"  # BULLISH | NEUTRAL | BEARISH (for equities)

    report_date: date
    summary: str


class GlobalMacroContext(BaseModel):
    """Global macro cross-asset regime — DXY strength and Copper/Gold ratio.

    DXY (US Dollar Index): strong dollar is a headwind for EM equities, commodities,
    and multinationals. DX-Y.NYB via yfinance.

    Copper/Gold ratio (HG=F / GC=F): "Dr. Copper" vs safe-haven gold.
    Rising ratio = industrial demand > safety demand = risk-on expansion.
    Declining ratio = safety demand > industrial demand = risk-off contraction.
    """
    # DXY — US Dollar Index
    dxy:            Optional[float] = None   # current level (~100 = neutral baseline)
    dxy_return_5d:  Optional[float] = None   # 5-day % return
    dxy_return_20d: Optional[float] = None   # 20-day % return
    # STRONG_BULL | BULL | NEUTRAL | BEAR | STRONG_BEAR (for the USD itself)
    dxy_signal:    str = "UNKNOWN"
    # for equities: strong DXY = BEARISH; weak DXY = BULLISH
    dxy_direction: str = "NEUTRAL"

    # Copper / Gold ratio
    copper_price:          Optional[float] = None   # HG=F last close (cents/lb)
    gold_price:            Optional[float] = None   # GC=F last close ($/oz)
    copper_gold_ratio:     Optional[float] = None   # copper / gold
    copper_gold_ratio_5d_ago:  Optional[float] = None
    copper_gold_ratio_20d_ago: Optional[float] = None
    copper_gold_change_5d:  Optional[float] = None  # % change in ratio over 5d
    copper_gold_change_20d: Optional[float] = None  # % change in ratio over 20d
    # RISK_ON_SURGE | RISK_ON | NEUTRAL | RISK_OFF | RISK_OFF_CRASH
    copper_gold_signal:    str = "UNKNOWN"
    copper_gold_direction: str = "NEUTRAL"   # BULLISH | NEUTRAL | BEARISH

    # Oil (WTI crude — CL=F)
    oil_price:     Optional[float] = None   # $/barrel
    oil_return_5d: Optional[float] = None   # 5-day % return
    oil_return_20d: Optional[float] = None  # 20-day % return

    # Oil/Bond divergence — CL=F vs TLT 5-day co-movement
    # Normally oil and bonds are inversely correlated (oil up = inflation → bonds down).
    # When both move the same direction, the usual macro logic is being overridden:
    #   Both up   → POLICY_PIVOT_SIGNAL  (market pricing Fed cut despite oil = unusual, BULLISH equities)
    #   Both down → DEFLATION_SHOCK      (demand destruction, BEARISH)
    #   Oil up, bonds down → STAGFLATION_RISK  (worst combo for equities, BEARISH)
    #   Oil down, bonds up → GROWTH_FEAR_RISK_OFF (classic risk-off, BEARISH for cyclicals)
    tlt_return_5d_ob: Optional[float] = None  # TLT 5d return used for this calc
    # POLICY_PIVOT_SIGNAL | GROWTH_FEAR_RISK_OFF | STAGFLATION_RISK | DEFLATION_SHOCK | NEUTRAL
    oil_bond_signal:    str = "NEUTRAL"
    oil_bond_direction: str = "NEUTRAL"   # BULLISH | BEARISH | NEUTRAL

    # Composite
    composite_signal:    str = "UNKNOWN"  # RISK_ON | CONSTRUCTIVE | NEUTRAL | DEFENSIVE | RISK_OFF
    composite_direction: str = "NEUTRAL"  # BULLISH | NEUTRAL | BEARISH

    report_date: date
    summary: str


class MOVEContext(BaseModel):
    """ICE BofA MOVE Index — Treasury market implied volatility (bond market VIX).

    Spikes in MOVE precede equity dislocations by 1–5 days: rising bond vol
    compresses risk appetite, triggers de-leveraging, and widens credit spreads
    before equity markets fully reprice the risk.
    """
    move: Optional[float] = None          # current MOVE level
    move_5d_ago: Optional[float] = None   # level 5 trading days ago
    move_20d_avg: Optional[float] = None  # 20-day rolling average

    # Spike detection
    spike_5d: Optional[float] = None      # absolute change over 5 trading days
    is_spiking: bool = False              # True when |spike_5d| > 20pt

    # Classification
    signal: str = "UNKNOWN"    # CALM | LOW | NORMAL | ELEVATED | HIGH | EXTREME | PANIC
    direction: str = "NEUTRAL" # BEARISH | NEUTRAL | BULLISH (spikes = bearish for equities)

    # Cross-asset context
    move_vix_ratio: Optional[float] = None  # MOVE / VIX — >8 signals bond fear >> equity fear

    source: str = "^MOVE"      # ticker used (^MOVE primary, VXTLT fallback)
    report_date: date
    summary: str


class OpExContext(BaseModel):
    """Options expiration calendar context — pure date math, no API calls."""
    today: date
    next_opex: date                  # 3rd Friday of current or next month
    prev_opex: date                  # 3rd Friday of the previous expiry cycle
    days_to_opex: int                # calendar days until next_opex
    days_since_prev_opex: int        # calendar days since prev_opex
    opex_week_monday: date           # Monday of the OpEx week
    in_opex_week: bool               # True if today is Mon–Fri of OpEx week
    is_triple_witching: bool         # True if next_opex is in Mar/Jun/Sep/Dec
    in_post_opex_window: bool        # True if 1–5 calendar days after prev_opex
    signal: str                      # OPEX_DAY | OPEX_IMMINENT | TRIPLE_WITCHING_WEEK | OPEX_WEEK | POST_OPEX | NEUTRAL
    summary: str


class TickerSignal(BaseModel):
    ticker: str
    direction: Direction
    confidence: float           # 0.0 – 1.0
    sentiment_score: float      # -1.0 to +1.0  (news + all article-based sources)
    technical_score: float      # -1.0 to +1.0
    insider_score: float = 0.0  # -1.0 to +1.0  (smart money: insider trades, options flow, SEC)
    put_call_score: float = 0.0 # -1.0 to +1.0  (per-ticker options put/call sentiment)
    vwap_score: float = 0.0     # -1.0 to +1.0  (mean-reversion: above VWAP→bearish, below→bullish)
    vwap_distance_pct: float = 0.0  # raw (price - VWAP) / VWAP × 100 (positive = above VWAP)
    rationale: str
    insider_summary: str = ""   # human-readable insider/politician trade context
    sources_agreeing: int = 0   # how many enabled signal layers agree with the direction
    # GEX fields — populated when enable_gex=true
    gex_signal: str = ""           # PINNED | AMPLIFIED | NEUTRAL | ""
    gamma_flip: Optional[float] = None
    max_pain_bias: str = ""        # BULLISH | BEARISH | NEUTRAL | ""
    max_pain_score: float = 0.0    # [-1, +1] max-pain gravity score (expiry-weighted)
    oi_skew_score: float = 0.0     # [-1, +1] OI-weighted directional lean (from GEX options chain)
    expected_move_pct: float = 0.0
    # Insider cluster fields — populated when ≥3 different insiders buy within 5 days
    insider_cluster_detected: bool = False
    insider_cluster_size: int = 0


class Recommendation(BaseModel):
    ticker: str
    type: str = "STOCK"        # "STOCK" or "ETF"
    direction: Direction
    confidence: float
    action: str                # "BUY", "SELL", "HOLD", "WATCH"
    time_horizon: str = "N/A"  # "SWING", "SHORT-TERM", "POSITION", "N/A"
    rationale: str
    generated_at: datetime
