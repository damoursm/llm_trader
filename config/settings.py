from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List


class Settings(BaseSettings):
    # Claude API
    anthropic_api_key: str

    # Model selection
    # Options: "claude-haiku-4-5-20251001" (fast/cheap), "claude-opus-4-6" (highest quality)
    analyst_model: str = "claude-haiku-4-5-20251001"

    # DeepSeek API (used for low-reasoning tasks)
    deepseek_api_key: str = ""

    # News sources
    newsapi_key: str = ""
    alpha_vantage_key: str = ""

    # Email (all optional — only needed when running with --email)
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_recipients: str = ""  # comma-separated

    # Watchlist
    stock_watchlist: str = "AAPL,MSFT,NVDA,TSLA,AMZN,META,GOOGL"
    sector_etfs: str = "XLK,XLF,XLE,XLV,XLY,XLP,XLI,XLB,XLU,XLRE,XLC"

    # Commodities — always included in every run regardless of trending
    # GLD=Gold, SLV=Silver, IAU=Gold(alt), GDX=Gold Miners, PPLT=Platinum, PALL=Palladium, CPER=Copper
    commodity_etfs: str = "GLD,SLV,IAU,GDX,PPLT,PALL,CPER"

    # Feature flags
    enable_fetch_data: bool = True        # set to false to skip all yfinance fetching entirely
    enable_charts: bool = False           # set to true to build Plotly charts and HTML report

    # Analysis method flags (at least one should be true)
    enable_news_sentiment: bool = True    # method 1: LLM sentiment from news/RSS
    enable_technical_analysis: bool = True  # method 2: RSI, MACD, SMA, Bollinger Bands
    enable_insider_trades: bool = True    # method 3: politician + corporate insider trades

    # Insider trades config
    insider_lookback_days: int = 90      # how far back to look for trades
    smart_money_top_tickers: int = 5     # max number of ticker groups shown in smart money section
    # Comma-separated names to prioritise (empty = include all politicians)
    tracked_politicians: str = (
        "Nancy Pelosi,Paul Pelosi,Austin Scott,Michael McCaul,"
        "Dan Crenshaw,Tommy Tuberville,Shelley Moore Capito,"
        "Josh Gottheimer,Ro Khanna,Brian Higgins"
    )

    # Unusual options flow (strategy 4)
    enable_options_flow: bool = True    # scan yfinance options chains for unusual sweeps

    # SEC EDGAR filings (strategies 5, 6, 7)
    enable_sec_filings: bool = True     # 13D/13G activist stakes, Form 144, 13F
    sec_filings_lookback_days: int = 30  # lookback window for 13D/13G and Form 144
    # Comma-separated institution names for 13F superinvestor tracking.
    # The pipeline resolves CIKs dynamically from EDGAR — no manual lookup needed.
    tracked_institutions: str = (
        "Berkshire Hathaway,Pershing Square Capital Management,"
        "Appaloosa Management,Baupost Group"
    )

    # FRED (Federal Reserve of St. Louis) — macro regime context
    # Free API key: https://fred.stlouisfed.org/docs/api/api_key.html
    fred_api_key: str = ""
    enable_fred: bool = True    # fetch yield curve, CPI, unemployment, credit spreads, M2

    # Earnings whisper vs. consensus gap — infers the implied "whisper number" from:
    # historical beat rate, avg EPS surprise magnitude, and consensus revision trend.
    # Uses yfinance earnings_dates + eps_trend + eps_revisions (free, no key required).
    enable_earnings_whisper: bool = True

    # Analyst estimate revision momentum — compares PT/rating changes over two 30-day windows
    # to detect whether analyst consensus is accelerating (improving) or decelerating (deteriorating).
    # Requires enable_analyst_ratings=True for yfinance data; cached daily.
    enable_revision_momentum: bool = True

    # CESI-style Macro Surprise Index — compare recent FRED releases to trailing 3-period averages
    # Consistent beats → cyclical tailwind; consistent misses → defensive bias. Requires FRED_API_KEY.
    enable_macro_surprise: bool = True

    # Market-implied Fed rate expectations — T-bill spreads proxy for CME FedWatch.
    # Derives P(cut/hold/hike) at next FOMC meeting + 12m cumulative cuts in bp. Requires FRED_API_KEY.
    enable_fedwatch: bool = True

    # CFTC Commitment of Traders — weekly futures positioning (no API key required)
    enable_cot: bool = True     # download COT data from CFTC; cached by ISO week

    # SEC 8-K material event filings — faster than RSS feeds (no API key required)
    enable_8k_filings: bool = True
    eight_k_lookback_days: int = 5   # fetch 8-Ks filed in the last N days

    # SEC S-1/S-11 IPO pipeline — sector-level institutional demand signal (no API key required)
    enable_ipo_pipeline: bool = True
    ipo_lookback_days: int = 30      # S-1 filings accumulate over weeks; 30 days gives a full picture

    # Analyst upgrades/downgrades/price-target changes — yfinance (free, no key required)
    enable_analyst_ratings: bool = True
    analyst_ratings_lookback_days: int = 30

    # Put/Call ratio — market sentiment + per-ticker directional bias
    # Market-wide: CBOE equity P/C CSV (free, no key); per-ticker: yfinance options volume
    enable_put_call: bool = True

    # Credit market leading indicator — HYG vs SPY divergence (yfinance, no key required)
    # High-yield bonds lead equities by 1-3 days; divergence warns of coming equity moves.
    enable_credit: bool = True

    # VIX & term structure — CBOE volatility indices via yfinance (no key required)
    # ^VIX, ^VXN, ^VVIX, ^VIX9D, ^VIX3M, ^VXMT
    enable_vix: bool = True

    # NYSE TICK index — breadth exhaustion / reversal signal (^TICK via yfinance, no key required)
    # Extreme readings (>+1000 or <-1000) are contrarian reversal signals.
    enable_tick: bool = True

    # VWAP distance — rolling 20-day volume-weighted average price vs current price.
    # Mean-reversion signal: large deviations attract institutional order flow back toward VWAP.
    enable_vwap: bool = True

    # Earnings calendar + EPS surprises
    # Upcoming dates: yfinance (free) + Alpha Vantage EARNINGS_CALENDAR (free with key)
    # EPS beat/miss: yfinance earnings_dates (free)
    enable_earnings: bool = True
    earnings_lookback_days: int = 90   # how far back to look for recent EPS surprises
    earnings_upcoming_days: int = 14   # how many days ahead to include in calendar

    # Short interest — FINRA Reg SHO daily short volume + yfinance (no API key required)
    # Squeeze setups (high SI + low days-to-cover), bearish positioning, short covering signals
    enable_short_interest: bool = True

    # Gamma Exposure (GEX) — options market structure: dealer positioning, gamma flip,
    # max pain, and expected move derived from yfinance options chains (no key required).
    # Covers SPY/QQQ/IWM always + any watchlist ticker with OI ≥ 1000 contracts.
    enable_gex: bool = True

    # McClellan Oscillator & Summation Index — NYSE A/D breadth momentum (^NYAD via yfinance, no key required)
    # Oscillator = EMA19 − EMA39 of daily net advances; Summation = running total; zero crosses = swing timing
    enable_mcclellan: bool = True

    # New 52-week highs vs. lows — HL Spread = %near_highs − %near_lows over sector ETFs + watchlist (yfinance, no key)
    # Divergence: SPY near 52w high + HL spread declining → bearish; SPY near 52w low + HL spread rising → bullish
    enable_highs_lows: bool = True

    # Market breadth — % of S&P 500 sector ETFs above their 200-day SMA (yfinance, no key required)
    # < 30% = broadly oversold; rising from < 30% = confirmed breadth thrust (strong multi-month bullish signal)
    enable_breadth: bool = True

    # Google Trends — search interest spike/drop as retail attention proxy (no API key required)
    enable_google_trends: bool = True   # uses pytrends (unofficial API); cached daily

    # Reddit social sentiment — r/wallstreetbets, r/stocks, r/investing
    # Free Reddit API credentials: https://www.reddit.com/prefs/apps (create "script" app)
    enable_reddit_sentiment: bool = True
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "llm_trader/1.0 (stock analysis bot)"

    # OpEx calendar — options expiration week effects (pure date math, no API calls)
    # OpEx week (3rd Friday) → max pain pinning; Triple Witching (Mar/Jun/Sep/Dec) → strongest
    enable_opex: bool = True

    # Seasonality calendar — end-of-month rebalancing, quarter-end window dressing,
    # January effect (small-cap rebound), and monthly historical biases (pure date math, no API calls)
    enable_seasonality: bool = True

    # Bond market internals — 1–8 week macro regime signals via yfinance (no API key required)
    # Treasury curve (10Y-3M), TLT/IEF/TIP/LQD price momentum, real yield, IG credit
    enable_bond_internals: bool = True

    # MOVE Index — ICE BofA Treasury implied volatility (bond market VIX) via yfinance (no key required)
    # ^MOVE primary ticker; VXTLT fallback. Spikes precede equity dislocations by 1–5 days.
    enable_move: bool = True

    # Global macro cross-asset regime — DXY strength (DX-Y.NYB) + Copper/Gold ratio (HG=F / GC=F)
    # DXY: strong dollar = headwind for EM, commodities, multinationals.
    # Copper/Gold: rising ratio = risk-on expansion; declining = risk-off contraction.
    enable_global_macro: bool = True

    # Scheduling — daily pre-market run (Mon-Fri, US/Eastern)
    schedule_daily: str = "0 8 * * 1-5"

    @property
    def tracked_politicians_list(self) -> List[str]:
        return [p.strip() for p in self.tracked_politicians.split(",") if p.strip()]

    @property
    def tracked_institutions_list(self) -> List[str]:
        return [i.strip() for i in self.tracked_institutions.split(",") if i.strip()]

    @property
    def recipients_list(self) -> List[str]:
        return [r.strip() for r in self.email_recipients.split(",") if r.strip()]

    @property
    def stocks_list(self) -> List[str]:
        return [s.strip() for s in self.stock_watchlist.split(",") if s.strip()]

    @property
    def sectors_list(self) -> List[str]:
        return [s.strip() for s in self.sector_etfs.split(",") if s.strip()]

    @property
    def commodities_list(self) -> List[str]:
        return [s.strip() for s in self.commodity_etfs.split(",") if s.strip()]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
