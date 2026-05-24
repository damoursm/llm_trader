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


class PEADSignal(BaseModel):
    """Per-ticker Post-Earnings Announcement Drift signal."""
    ticker: str
    report_date: date            # date of the latest earnings report
    days_since_report: int       # 0 = today; signal decays linearly to 0 at decay_window
    actual_eps: Optional[float] = None
    estimated_eps: Optional[float] = None
    surprise_pct: float          # (actual - estimate) / |estimate| * 100
    sue_normalized: float        # tanh(surprise_pct / surprise_scale_pct), ∈ [-1, +1]
    time_decay: float            # max(0, 1 - days_since_report / decay_window), ∈ [0, 1]
    pead_score: float            # sue_normalized * time_decay, ∈ [-1, +1]
    direction: str               # BULLISH | BEARISH | NEUTRAL
    summary: str


class PEADContext(BaseModel):
    """Post-Earnings Announcement Drift signals across the universe.

    PEAD is one of the most-replicated cross-sectional anomalies in academic
    finance: stocks that beat (miss) earnings tend to continue drifting in
    the surprise direction for ~60 days as the market under-reacts to the
    information at announcement. Score = standardized unexpected earnings ×
    time-decay so the signal fades to zero as the post-announcement window
    closes.
    """
    signals: List[PEADSignal]
    report_date: date
    decay_window_days: int
    surprise_scale_pct: float
    top_drift_bullish: List[str] = []   # tickers with strongest positive scores
    top_drift_bearish: List[str] = []   # tickers with strongest negative scores
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


class DIXContext(BaseModel):
    """Dark Pool Index (DIX) + market-wide Gamma Exposure (GEX) — SqueezeMetrics feed.

    DIX measures the dollar- and volume-weighted short-volume across off-exchange
    (dark pool) venues — a proxy for *hidden institutional accumulation*. Because
    large buyers route through dark pools to avoid moving the lit market, a high DIX
    means strong concealed buying pressure and is historically bullish for forward
    S&P returns (it leads price by ~1–4 weeks).

    The market-wide GEX here is distinct from the per-ticker dealer gamma computed in
    gamma_exposure.py: it is the whole-index dealer gamma. High/positive GEX = vol
    suppression (pinning, mean-reversion); low/negative GEX = vol expansion (moves get
    amplified). The classic SqueezeMetrics bull setup is **high DIX + low GEX**:
    hidden buying with room to run.
    """
    # Dark Pool Index
    dix: Optional[float] = None              # latest DIX as a 0–1 fraction
    dix_pct: Optional[float] = None          # DIX × 100 for display
    dix_5d_avg: Optional[float] = None
    dix_20d_avg: Optional[float] = None
    dix_percentile_1y: Optional[float] = None  # 0–100 percentile of latest DIX in trailing ~252 obs
    dix_trend: str = "FLAT"                   # RISING | FLAT | FALLING (5-day change)

    # Market-wide Gamma Exposure (whole-index, from the same feed)
    gex: Optional[float] = None               # latest GEX ($ notional)
    gex_percentile_1y: Optional[float] = None # 0–100 percentile
    gex_regime: str = "UNKNOWN"               # VOL_SUPPRESSION | NEUTRAL | VOL_EXPANSION

    spx_price: Optional[float] = None         # SPX close from the same feed
    obs_count: int = 0                        # rows in the percentile window

    signal: str = "UNKNOWN"     # STRONG_ACCUMULATION | ACCUMULATION | NEUTRAL | DISTRIBUTION | STRONG_DISTRIBUTION
    direction: str = "NEUTRAL"  # BULLISH | NEUTRAL | BEARISH (equity implication)
    source: str = "SqueezeMetrics DIX.csv"
    report_date: date
    summary: str


class MarketModeContext(BaseModel):
    """Market mode classification for dynamic signal weight switching."""
    mode: str                    # TRENDING | NEUTRAL | CHOPPY
    composite_score: float       # weighted composite; >+0.5 = trending, <-0.5 = choppy
    weight_profile: Dict[str, float]  # raw unnormalised weights for the aggregator
    evidence: str = ""           # pipe-separated per-source contributions
    weight_summary: str = ""     # human-readable weight delta vs. NEUTRAL baseline
    summary: str = ""


class MacroRegimeContext(BaseModel):
    """Composite macro regime derived from VIX, MOVE, bond internals, global macro, FRED, breadth, and credit."""
    regime: str                     # PANIC | RISK_OFF | CAUTION | NEUTRAL | RISK_ON
    composite_score: float          # weighted composite ∈ [-3, +1]; negative = risk-off
    confidence_threshold: float     # adjusted min confidence for actionable BUY/SELL signals
    allow_buys: bool                # False during PANIC or RISK_OFF — new longs blocked
    has_panic_signal: bool = False  # at least one source (VIX or MOVE) fired PANIC
    evidence: str = ""              # pipe-separated per-source contributions
    summary: str = ""              # human-readable full summary


class SectorPair(BaseModel):
    """A market-neutral pair trade: one leg long + one leg short within the same sector.

    Formed when a sector ETF and one of its constituents disagree on direction.
    Long the BULLISH leg, short the BEARISH leg — removes broad market beta,
    isolates the idiosyncratic signal.
    """
    stock: str                  # individual stock ticker
    etf: str                    # sector ETF ticker (e.g. XLK)
    long_leg: str               # ticker to go long
    short_leg: str              # ticker to go short
    stock_direction: str        # BULLISH | BEARISH
    etf_direction: str          # BULLISH | BEARISH (opposite of stock_direction)
    stock_confidence: float
    etf_confidence: float
    pair_score: float           # composite conviction = (stock_conf + etf_conf) / 2
    setup_type: str             # "ETF_BULL_STOCK_BEAR" | "ETF_BEAR_STOCK_BULL"
    rationale: str


class SectorPairsContext(BaseModel):
    """All sector-pair relative-value opportunities detected in the current run."""
    pairs: List[SectorPair]
    summary: str


class CointPair(BaseModel):
    """A statistically cointegrated pair trade (Engle-Granger ADF + z-score).

    Unlike SectorPair (which keys off opposing directional *signals*), this is a
    pure statistical-arbitrage relationship: two price series whose linear
    combination (spread = log(A) − β·log(B)) is mean-reverting (stationary).
    When the spread deviates far from its mean (|z| ≥ entry), long the cheap leg
    and short the expensive leg, betting the spread reverts. Market-neutral by
    construction — the hedge ratio β removes shared market beta.
    """
    ticker_a: str               # dependent leg (Y) in the cointegrating regression
    ticker_b: str               # independent leg (X)
    hedge_ratio: float          # β: log(A) ≈ α + β·log(B); units of B per unit of A
    adf_stat: float             # ADF t-statistic on the spread (more negative = more stationary)
    adf_pvalue: float           # approximate p-value of the ADF stat
    adf_crit_5pct: float        # 5% critical value used for the cointegration decision
    is_cointegrated: bool       # adf_stat <= critical value at the configured level
    half_life_days: float       # OU mean-reversion half-life (lower = faster reversion)
    correlation: float          # log-price correlation (sanity check)
    spread_mean: float          # mean of the spread over the lookback
    spread_std: float           # std of the spread over the lookback
    spread_zscore: float        # current (spread − mean) / std
    long_leg: str               # ticker to go long (the relatively cheap leg)
    short_leg: str              # ticker to go short (the relatively expensive leg)
    signal: str                 # ENTRY | STRETCHED | MONITOR | NEUTRAL
    lookback_days: int          # number of overlapping observations used
    rationale: str


class CointPairsContext(BaseModel):
    """Cointegration-based statistical-arbitrage pairs across the universe.

    Cointegration pairs trading is a classic market-neutral alpha strategy:
    test economically-linked candidate pairs for a stationary (mean-reverting)
    spread via the Engle-Granger two-step (OLS hedge ratio → ADF test on the
    residual), then trade z-score extremes of the spread. Distinct from the
    direction-based sector_pairs overlay.
    """
    pairs: List[CointPair]                  # tradeable pairs (cointegrated, sorted by |z|)
    candidates_tested: int                  # how many candidate pairs were evaluated
    cointegrated_count: int                 # how many passed the ADF test
    ticker_scores: Dict[str, float] = {}    # per-ticker net directional lean ∈ [-1, +1]
    report_date: date
    summary: str


class SectorRotationEntry(BaseModel):
    """Per-sector money-flow signal derived from relative price momentum and volume."""
    etf: str                             # e.g. "XLK"
    name: str                            # e.g. "Technology"
    return_5d:    Optional[float] = None # absolute 5-day return %
    return_21d:   Optional[float] = None # absolute 21-day return %
    return_63d:   Optional[float] = None # absolute 63-day return %
    relative_5d:  Optional[float] = None # excess return vs SPY, 5d (%)
    relative_21d: Optional[float] = None # excess return vs SPY, 21d (%)
    relative_63d: Optional[float] = None # excess return vs SPY, 63d (%)
    volume_ratio: Optional[float] = None # 5d avg vol / 20d avg vol (>1.15 = elevated)
    rotation_score: float = 0.0          # composite flow score ∈ [-1, +1]
    flow_signal: str = "NEUTRAL"         # STRONG_INFLOW|INFLOW|NEUTRAL|OUTFLOW|STRONG_OUTFLOW
    direction: str = "NEUTRAL"           # BULLISH | NEUTRAL | BEARISH


class SectorRotationContext(BaseModel):
    """Cross-sector money flow rotation — 'Ebb and Flow' mechanism."""
    sectors: List[SectorRotationEntry]   # all 11 SPDR sectors, sorted by score desc
    top_inflow:  List[str]               # top ETF tickers attracting capital
    top_outflow: List[str]               # top ETF tickers losing capital
    rotation_regime: str                 # RISK_ON | NEUTRAL | RISK_OFF
    rotation_direction: str              # BULLISH | NEUTRAL | BEARISH
    cyclical_avg: float = 0.0            # avg score across cyclical sectors
    defensive_avg: float = 0.0          # avg score across defensive sectors
    cyc_def_spread: float = 0.0          # cyclical_avg − defensive_avg
    rotation_pairs: List[str]            # e.g. ["XLK → XLP (growth → defensive)"]
    report_date: date
    summary: str


class RotationDriversContext(BaseModel):
    """Federal Reserve rate-cycle phase — cross-asset rotation signal."""
    report_date: date

    # Fed Funds Rate trajectory (from FRED DFF, daily)
    fed_rate_current:  Optional[float] = None   # most recent DFF %
    fed_rate_3m_ago:   Optional[float] = None   # ~63 trading days ago
    fed_rate_6m_ago:   Optional[float] = None   # ~126 trading days ago
    fed_rate_12m_ago:  Optional[float] = None   # ~252 trading days ago
    rate_change_3m_bp: Optional[float] = None   # change in basis points, 3m
    rate_change_12m_bp: Optional[float] = None  # change in basis points, 12m
    rate_trajectory: str = "UNKNOWN"            # ACTIVE_HIKING | PAUSING | ACTIVE_CUTTING | EASING_PAUSE | STABLE

    # CPI inflation trend (from FRED CPIAUCSL, monthly)
    cpi_yoy_current: Optional[float] = None     # most recent CPI YoY %
    cpi_yoy_6m_ago:  Optional[float] = None     # CPI YoY 6 months ago
    inflation_trend: str = "UNKNOWN"            # ACCELERATING | RISING | ELEVATED_STABLE | STABLE | MODERATING | DECLINING | LOW_STABLE

    # Real rate (Fed Funds − CPI YoY)
    real_rate: Optional[float] = None           # %
    real_rate_regime: str = "UNKNOWN"           # HIGHLY_RESTRICTIVE | RESTRICTIVE | NEUTRAL | ACCOMMODATIVE

    # Cycle phase synthesis
    cycle_phase: str = "NEUTRAL"                # EARLY_TIGHTENING | PEAK_TIGHTENING | TIGHTENING_PAUSE | PIVOT_IMMINENT | EASING_CYCLE | NEUTRAL
    cycle_direction: str = "NEUTRAL"            # BULLISH | BEARISH | NEUTRAL (for equities)

    # Asset rotation
    favoured_assets: List[str] = []             # e.g. ["TLT", "XLRE", "XLU"]
    avoid_assets: List[str] = []                # e.g. ["XLK", "QQQ"]

    summary: str = ""


class SectorCycleBias(BaseModel):
    """Per-sector expected leadership for the current business cycle phase."""
    etf: str                      # e.g. "XLK"
    name: str                     # e.g. "Technology"
    cycle_score: float = 0.0      # ∈ [-1, +1]; +1 = strongest historical outperformance in this phase
    cycle_signal: str = "NEUTRAL" # STRONG_LEADER|LEADER|NEUTRAL|LAGGARD|STRONG_LAGGARD


class BusinessCycleContext(BaseModel):
    """Business cycle phase → sector rotation biases (Fidelity-style historical model)."""
    cycle_phase: str = "UNKNOWN"        # EARLY_EXPANSION|MID_EXPANSION|LATE_EXPANSION|LATE_CYCLE|CONTRACTION|UNKNOWN
    cycle_direction: str = "NEUTRAL"    # BULLISH|NEUTRAL|BEARISH for broad risk assets
    evidence: str = ""                  # how the phase was derived from FRED inputs
    sector_biases: List[SectorCycleBias] = []
    top_cycle_leaders: List[str] = []   # ETF tickers with score > 0.4
    weak_cycle_sectors: List[str] = []  # ETF tickers with score < -0.4
    convergence_notes: str = ""         # agreement/disagreement with Ebb-and-Flow real-time flows
    report_date: date
    summary: str = ""


class ClusterWatchEntry(BaseModel):
    """A single ticker under active insider-cluster surveillance (within 10-day window)."""
    ticker: str
    detected_at: date
    cluster_size: int                   # number of distinct insiders in the triggering cluster
    insider_summary: str = ""           # human-readable names/roles from TickerSignal
    days_elapsed: int = 0               # calendar days since detection
    days_remaining: int = 10            # days left in the 10-day watch window


class ClusterWatchlistContext(BaseModel):
    """Cross-run persistent watchlist of insider cluster signals (up to 10 days)."""
    entries: List[ClusterWatchEntry]
    active_tickers: List[str]           # tickers still within the watch window
    summary: str


class CatalystSetup(BaseModel):
    """A ticker with both a recent 8-K filing and an insider buy — highest-conviction pre-signal setup."""
    ticker: str
    has_8k: bool = False
    has_insider_buy: bool = False
    has_vol_spike: bool = False
    catalyst_reason: str = ""


class CatalystTimingContext(BaseModel):
    """Event-driven catalyst timing signals: earnings blackout, OpEx amplifier, and 8-K+insider WATCH elevation."""
    earnings_blackout_tickers: List[str]
    earnings_blackout_details: Dict[str, int]   # ticker → days_until_earnings
    opex_max_pain_weight: float                 # 0.12 base, 0.20 opex week, 0.28 triple witching
    opex_boost_active: bool
    opex_is_triple_witching: bool
    opex_signal: str                            # from OpExContext.signal
    catalyst_setups: List[CatalystSetup]
    watch_elevation_tickers: List[str]
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
    combined_score: float = 0.0 # -1.0 to +1.0 weighted sum of method scores (pre-confidence factors).
                                # Stored so monitor_open_positions can compare today's signal
                                # against signal_at_entry to detect thesis decay.
    sentiment_score: float      # -1.0 to +1.0  (news + all article-based sources; the LEVEL)
    # Sentiment velocity (Δsentiment, not level) — populated when enable_sentiment_velocity=true.
    # = recent-window news tone − prior-window news tone; the rate of change leads short-horizon moves.
    sentiment_velocity_score: float = 0.0  # -1.0 to +1.0  (tanh-normalised Δ tone)
    sentiment_recent: float = 0.0          # mean lexical tone of the recent window
    sentiment_prior: float = 0.0           # mean lexical tone of the prior window
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
    # Insider persistence fields — populated when the SAME insider buys on multiple
    # separate days within the lookback window (depth of conviction, vs. cluster's breadth).
    insider_persistence_detected: bool = False
    insider_persistence_count: int = 0       # max distinct buy days by a single insider
    insider_persistence_buyer: str = ""       # name of the most-persistent repeat buyer
    # Pattern recognition fields — populated when enable_pattern_recognition=true
    pattern_score: float = 0.0   # [-1, +1] historical win-rate based pattern signal
    pattern_name: str = ""        # detected pattern (e.g. "double_bottom", "head_shoulders")
    # Price momentum fields — populated when enable_price_momentum=true
    momentum_score: float = 0.0  # [-1, +1] Perceived Value: normalised multi-period price trend
    momentum_1m_pct: float = 0.0 # raw 1-month return %
    momentum_3m_pct: float = 0.0 # raw 3-month return %
    # Money flow fields — populated when enable_money_flow=true
    money_flow_score: float = 0.0  # [-1, +1] accumulation/distribution composite (MFI+CMF+OBV)
    mfi_value: float = 50.0        # raw MFI reading 0–100 (< 20 = accumulation, > 80 = distribution)
    cmf_value: float = 0.0         # raw CMF reading [-1, +1] (positive = accumulation)
    # Trend strength — populated when enable_trend_strength=true.
    # ADX/DMI directional movement × strength + Donchian 20-day breakout.
    trend_strength_score: float = 0.0   # [-1, +1] confirmed uptrend (+) / downtrend (−); ~0 = chop
    adx_value: float = 0.0              # raw ADX (trend strength: <20 chop, >25 trend, >40 strong)
    trend_strength_label: str = "NO_DATA"  # NO_TREND|UPTREND|STRONG_UPTREND|BREAKOUT_UP|…|DOWN…
    # PEAD (Post-Earnings Announcement Drift) — populated when enable_pead=true
    pead_score: float = 0.0          # [-1, +1] SUE × time-decay
    pead_surprise_pct: float = 0.0   # most recent EPS surprise %
    pead_days_since_report: int = 0  # 0 = today; signal fades to 0 at decay_window
    # IV Rank + Directional — populated when enable_iv_rank=true.
    # iv_rank_score uses RV (realized vol) percentile as proxy for IV Rank, combined
    # with directional 5-day return / ATR to derive regime-aware contrarian / trend bias.
    iv_rank_score: float = 0.0       # [-1, +1] regime-aware directional bias
    iv_rank: float = 50.0            # 0–100 percentile of current 21d RV in 252d distribution
    iv_rank_ret_5d_pct: float = 0.0  # raw 5-day return %
    iv_rank_label: str = "NEUTRAL"   # CAPITULATION_BUY|FADE_EXTREME|CALM_UPTREND|CALM_DOWNTREND|TREND_FOLLOWING|EVENT_CAUTION|NEUTRAL
    # IV Expression — populated when enable_iv_expr=true.
    # Uses true market-implied vol from the options chain (gex_context.expected_move_pct)
    # ranked against the ticker's own trailing GEX-cache history; combines with oi_skew
    # to derive a stock-vs-options expression directional bias.
    iv_expr_score: float = 0.0          # [-1, +1] expression bias from real options chain
    iv_expr_rank: float = 50.0          # 0–100 percentile of current market-implied IV
    iv_expr_oi_skew: float = 0.0        # raw OI-weighted directional skew used as input
    iv_expr_label: str = "NEUTRAL"      # FADE_PREMIUM|EXPENSIVE_NEUTRAL|CHEAP_DIRECTIONAL_LONG|CHEAP_DIRECTIONAL_SHORT|CHEAP_COMPLACENT|MID_IV_DIRECTIONAL|NEUTRAL|NO_OPTIONS_DATA
    # Cross-sectional ranking — populated when enable_cross_sectional=true.
    # Mean of per-method universe z-scores (capped at zcap) divided by zcap,
    # clipped to [-1, +1]. Positive = ticker stands out vs universe; negative =
    # lags. Composes additively into combined_score with cross_sectional_weight.
    cross_sectional_score: float = 0.0
    # Cointegration pairs — populated when enable_cointegration=true.
    # Net directional lean derived from the ticker's role (cheap/long vs expensive/short)
    # across all cointegrated pairs it belongs to, weighted by spread z-score extremity.
    coint_score: float = 0.0


class Recommendation(BaseModel):
    ticker: str
    type: str = "STOCK"        # "STOCK" or "ETF"
    direction: Direction
    confidence: float
    action: str                # "BUY", "SELL", "HOLD", "WATCH"
    time_horizon: str = "N/A"  # "SWING", "SHORT-TERM", "POSITION", "N/A"
    rationale: str
    generated_at: datetime
