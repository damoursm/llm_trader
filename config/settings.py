from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List


class Settings(BaseSettings):
    # Claude API
    anthropic_api_key: str

    # Model selection
    # Options: "claude-haiku-4-5-20251001" (fast/cheap), "claude-opus-4-6" (highest quality)
    analyst_model: str = "claude-haiku-4-5-20251001"

    # Polygon.io market data (primary source for equity/ETF price + OHLCV)
    # Free API key: https://polygon.io — no credit card, works globally.
    # If absent, the pipeline falls back to yfinance for all market data.
    polygon_api_key: str = ""

    # DeepSeek API (used for low-reasoning tasks)
    deepseek_api_key: str = ""

    # News sources
    newsapi_key: str = ""
    alpha_vantage_key: str = ""
    # Financial Modeling Prep — market-wide analyst upgrades/downgrades feed (Section E catalyst
    # discovery). Free key: https://site.financialmodelingprep.com/developer/docs . Empty → the
    # market-wide analyst discovery source is skipped (yfinance analyst data is per-ticker only).
    fmp_api_key: str = ""

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
    enable_fetch_data: bool = True        # set to false to skip all live data fetching (Polygon + yfinance)
    enable_charts: bool = False           # set to true to build Plotly charts and HTML report

    # Analysis method flags (at least one should be true)
    enable_news_sentiment: bool = True    # method 1: LLM sentiment from news/RSS

    # Sentiment velocity (Δsentiment, not level) — the rate of change of news tone leads
    # short-horizon (1–5 day) moves better than the absolute level. Deterministic lexical
    # polarity per article, bucketed by published_at into a recent vs prior window;
    # velocity = recent_tone − prior_tone. No extra LLM/API cost (reuses stored timestamps).
    enable_sentiment_velocity: bool = True
    sentiment_velocity_recent_hours: int = 24   # "recent" window: articles ≤ N hours old
    sentiment_velocity_prior_hours: int = 96    # "prior" window: from recent_hours to N hours old

    enable_technical_analysis: bool = True  # method 2: RSI, MACD, SMA, Bollinger Bands
    enable_insider_trades: bool = True    # method 3: politician + corporate insider trades

    # Insider trades config
    insider_lookback_days: int = 90      # how far back to look for trades
    smart_money_top_tickers: int = 5     # max number of ticker groups shown in smart money section

    # Insider buying persistence — amplify insider_score when the SAME insider buys the same
    # ticker on multiple SEPARATE days within the lookback window. Repeated accumulation by one
    # name (depth of conviction) is a stronger tell than a one-off purchase; distinct from the
    # cluster amplifier (which measures breadth — many DIFFERENT insiders buying at once).
    # Amplifier scales 1.0 + 0.25×(distinct_buys − 1), capped at 1.75× (mirrors the cluster ceiling).
    enable_insider_persistence: bool = True
    insider_persistence_min_buys: int = 2   # min distinct buy days by one insider to qualify
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
    # Comma-separated institution names for 13F superinvestor tracking (broader = stronger
    # consensus when multiple filers buy the same name). The pipeline resolves CIKs dynamically
    # from EDGAR — no manual lookup needed. Each adds EDGAR calls, so keep the list reasonable.
    tracked_institutions: str = (
        "Berkshire Hathaway,Pershing Square Capital Management,"
        "Appaloosa Management,Baupost Group,Scion Asset Management,"
        "Greenlight Capital,Third Point,Icahn Capital,"
        "Tiger Global Management,Duquesne Family Office"
    )

    # Market-wide Form 4 open-market-buy scan — surfaces insider ACCUMULATION everywhere (not
    # just the watchlist). Parses recent Form 4 XML for transaction code "P" (open-market
    # purchase); these become corporate_insider "purchase" records that feed the insider CLUSTER
    # + PERSISTENCE detectors and discovery. Open-market buys are only ~1-2% of all Form 4s, so
    # the scan parses a bounded number of the most recent filings (cached daily). enable_sec_filings.
    enable_form4_scan: bool = True
    form4_scan_lookback_days: int = 3
    form4_scan_max_filings: int = 150     # recent Form 4 filings parsed per run (cached daily)

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

    # Ticker discovery — extra sources that widen the analysis universe (Step 0, fail-graceful).
    # WSB cashtag discovery: most-mentioned valid tickers across r/wallstreetbets, r/stocks,
    # r/investing hot/rising posts via Reddit's public JSON (no key) — validated against the SEC
    # ticker universe. NewsAPI/headline discovery is open-vocabulary via the same SEC map.
    # StockTwits trending is implemented but its public API is Cloudflare-gated (403 without a
    # browser/token), so it's OFF by default — enable only if you have working access.
    enable_stocktwits_discovery: bool = False
    enable_wsb_discovery: bool = True
    wsb_discovery_min_mentions: int = 3   # min distinct hot/rising posts mentioning a ticker

    # Opportunity Screener — PROACTIVE setup discovery over a broad liquid universe.
    # Shifts discovery from "what's trending" to "what's technically set up". Screens a
    # curated liquid universe (+ everything already in the OHLCV cache) cache-first for:
    # unusual volume, 52-week breakouts / new-low reversals, relative strength vs SPY, and
    # golden/death crosses — then injects qualifying names into the analysis universe. Reuses
    # the OHLCV cache (works offline once warm); a bounded warm-up fetch primes the cache.
    enable_opportunity_screener: bool = True
    screen_volume_ratio: float = 2.0           # today vol / 20d avg ≥ this → unusual volume
    screen_rs_lookback_days: int = 63          # ~3 months for relative strength vs SPY
    screen_rs_threshold_pct: float = 10.0      # excess return vs SPY (pp) → strong/weak RS
    screen_cross_lookback: int = 5             # bars within which a 50/200 SMA cross is "fresh"
    screen_near_high_pct: float = 2.0          # within this % of the 52-week high → breakout-watch
    screen_min_price: float = 5.0               # liquidity gate: minimum last close ($)
    screen_min_dollar_volume: float = 20_000_000  # liquidity gate: min 20d avg $ volume (filters thin pumps)
    screen_max_fetch_per_run: int = 30         # cap cold OHLCV fetches per run (cache warms over time)
    screen_max_results: int = 20               # max setups the screener injects into the universe

    # Macro → Discovery loop — closes the gap between the macro/regime modules and stock selection.
    # The sector-rotation (top inflows), business-cycle (phase leaders), and DIX (regime → factor
    # tilt) modules identify FAVORED sector/factor ETFs; this auto-pulls their top holdings as
    # candidates so the analysis universe is biased toward where macro money is flowing. Holdings
    # come from yfinance funds_data (cached daily) with a static SPDR fallback.
    enable_macro_discovery: bool = True
    macro_discovery_top_sectors: int = 3       # favored ETFs to pull from each source (rotation / cycle)
    macro_discovery_holdings_per_etf: int = 8  # top N holdings pulled per favored ETF
    macro_discovery_max: int = 25              # max constituent names injected into the universe

    # ── Section E: Catalyst & relationship expansion ──────────────────────────────
    # Widen the universe along three axes the static watchlist misses: (1) market-wide CATALYST
    # discovery — names with imminent earnings or fresh analyst rating changes ANYWHERE in the
    # market (not just the watchlist); (2) RELATIONSHIP discovery — pull the partner leg of a
    # cointegrated pair when one leg is already in the universe; (3) a richer factor/thematic ETF
    # universe. All fail-graceful and capped; discovered names are injected at Step 0 so the
    # existing per-ticker enrichment (analyst ratings, earnings, signal stack) picks them up.

    # Earnings-calendar discovery — inject names reporting within the window (market-wide via the
    # Alpha Vantage EARNINGS_CALENDAR feed; yfinance has no market-wide calendar). Needs
    # alpha_vantage_key; skipped without it.
    enable_earnings_discovery: bool = True
    earnings_discovery_window_days: int = 7    # report within N days to be injected
    earnings_discovery_max: int = 15           # cap names injected per run

    # Analyst-ratings discovery — inject names with fresh upgrades/downgrades anywhere in the market
    # (Financial Modeling Prep upgrades-downgrades RSS feed). Needs fmp_api_key; skipped without it.
    enable_analyst_discovery: bool = True
    analyst_discovery_lookback_days: int = 3   # rating change within N days
    analyst_discovery_min_firms: int = 1       # min distinct firms acting on a name to inject
    analyst_discovery_max: int = 15            # cap names injected per run

    # Cointegration peer-expansion — when a tradeable cointegrated pair has one leg in the universe
    # and the partner outside it, pull the partner in so the relationship is tradeable both ways.
    enable_coint_peer_discovery: bool = True
    coint_peer_max: int = 10                   # cap partner legs injected per run

    # Factor / thematic ETF universe — pinned ETFs (like commodities) broadening coverage beyond the
    # 11 GICS sectors: style factors (momentum/quality/value/size/low-vol/growth) + high-interest
    # themes (semis, software, biotech, defense, clean energy, homebuilders, airlines, regional banks).
    enable_factor_etfs: bool = True
    factor_etfs: str = "MTUM,QUAL,VLUE,SIZE,USMV,IWF,IWD"
    thematic_etfs: str = "SMH,IGV,XBI,ITA,TAN,LIT,IBB,XHB,JETS,KRE"

    # ── Section F: Discovery liquidity gate ───────────────────────────────────────
    # A uniform quality floor on EVERY discovered candidate (trending/open-vocab, screener,
    # macro→discovery, market-wide earnings/analyst catalysts, cointegration peers) so widening the
    # funnel (Sections A–E) doesn't inject untradeable microcaps — the names the tracker's bid-ask
    # model charges up to 250 bp a side. NEVER gates the pinned universe (watchlist, sector ETFs,
    # commodities, factor/thematic ETFs) or open-trade tickers. Cache-first with a bounded warm-up
    # fetch; a name whose liquidity can't be verified is dropped (fail-closed).
    enable_discovery_liquidity_gate: bool = True
    discovery_min_price: float = 5.0                 # minimum last close ($)
    discovery_min_dollar_volume: float = 20_000_000  # minimum 20-day average dollar volume ($)
    discovery_gate_max_fetch: int = 25               # cap cold OHLCV fetches per run for the gate

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

    # Dark Pool Index (DIX) + market-wide GEX — SqueezeMetrics free CSV (no API key required).
    # DIX = dollar/volume-weighted off-exchange short volume → proxy for hidden institutional
    # accumulation; high DIX is historically bullish for forward S&P returns (leads ~1–4 weeks).
    # Market-wide GEX gauges whole-index dealer gamma (low/negative = vol expansion). High DIX +
    # low GEX = classic "hidden buying with room to run". Macro-context overlay (not per-ticker);
    # feeds the Claude prompt and the Macro Regime Filter. Cached daily.
    enable_dix: bool = True

    # Global macro cross-asset regime — DXY strength (DX-Y.NYB) + Copper/Gold ratio (HG=F / GC=F)
    # DXY: strong dollar = headwind for EM, commodities, multinationals.
    # Copper/Gold: rising ratio = risk-on expansion; declining = risk-off contraction.
    enable_global_macro: bool = True

    # Pattern recognition — detects 8 classical chart patterns and scores them by historical win rate.
    # On first run per ticker: fetches 2y of OHLCV data and builds a pattern success library
    # (cache/patterns/<TICKER>.json, TTL 7 days). Subsequent runs are instant (warm-cache lookup).
    enable_pattern_recognition: bool = True

    # Pattern-registry feedback loop — record the outcome of every real
    # BUY/SELL the system takes when a chart pattern is active at entry, then
    # blend that live win rate into the synthetic per-ticker prior used by
    # compute_pattern_score. Bayesian shrinkage with ``pattern_registry_prior_n``
    # virtual trials on the synthetic side: live evidence only dominates once
    # ``live_n >> prior_n``. Stored at ``cache/pattern_registry.json``.
    enable_pattern_registry: bool = True
    pattern_registry_prior_n: int = 10

    # Market Mode Switching — dynamically adjusts signal weights based on TRENDING/CHOPPY regime.
    # TRENDING (low VIX, healthy breadth): up-weights tech/news, down-weights vwap/put_call.
    # CHOPPY   (high VIX, mixed breadth):  up-weights vwap/put_call, down-weights tech.
    # NEUTRAL:  uses baseline _BASE_WEIGHTS unchanged.
    enable_market_mode_switching: bool = True

    # Macro Regime Filter — top-down overlay that gates BUY entries and adjusts thresholds.
    # Reads VIX, MOVE, bond internals, global macro, FRED, breadth, and credit to produce:
    #   PANIC     → threshold 0.88, BUY entries blocked
    #   RISK_OFF  → threshold 0.82, BUY entries blocked
    #   CAUTION   → threshold 0.80
    #   NEUTRAL   → threshold 0.78 (baseline, unchanged)
    #   RISK_ON   → threshold 0.72
    enable_macro_regime_filter: bool = True

    # Sector Rotation / "Ebb and Flow" — cross-sector money flow via relative momentum + volume
    # Ranks all 11 SPDR sector ETFs by excess return vs SPY (1w/1m/3m) adjusted for volume.
    # Identifies top inflow/outflow sectors, rotation regime (RISK_ON/NEUTRAL/RISK_OFF),
    # and explicit rotation pairs (e.g., "XLK → XLP"). Cached daily, yfinance/Polygon OHLCV.
    enable_sector_rotation: bool = True

    # Rotation Drivers — rate-cycle phase from actual DFF trajectory (3m/12m) + CPI trend.
    # Maps Fed hiking/pausing/cutting cycles to cross-asset rotation implications:
    # favoured/avoided asset classes per phase (EARLY_TIGHTENING → EASING_CYCLE).
    # Requires FRED_API_KEY. Cached daily.
    enable_rotation_drivers: bool = True

    # Price Momentum (Perceived Value) — multi-period price trend normalised against own history.
    # Captures the self-reinforcing dynamic: rising perceived value attracts more capital →
    # trend continues. Scores 1m and 3m returns vs trailing 252-day return distribution.
    # Uses OHLCV chart cache first (works with ENABLE_FETCH_DATA=false); falls back to yfinance.
    enable_price_momentum: bool = True

    # Market-Relative Momentum — diagnostic ticker − SPY residual.
    # Answers "is this name lagging the broad market?" independently of its
    # sector. NOT added to the weighted aggregator combo (would double-count
    # beta against the sector-relative score, since market_rel = sector_rel +
    # (sector − market)). The email/prompt surface it side-by-side with the
    # sector-relative reading so divergences are visible:
    #   sector_rel positive, market_rel negative → best-of-a-bad-sector
    #   sector_rel ≈ 0,      market_rel negative → "beta drag" (riding weak sector)
    #   sector_rel negative, market_rel positive → stock-specific weakness in a
    #                                              strong sector
    enable_market_relative_momentum: bool = True

    # Sector-Relative Momentum — beta-stripped alpha factor (ticker − sector ETF).
    # The classic absolute momentum signal mixes idiosyncratic alpha with sector
    # beta — if the whole sector is ripping, every constituent looks like it has
    # momentum even though none is genuinely outperforming. This module subtracts
    # the benchmark return to leave only the residual: did NVDA *beat* tech, or
    # just ride the wave? Resolved benchmark: sector ETF for stocks (via the
    # aggregator map + yfinance lookup, SPY fallback), SPY for ETFs, no benchmark
    # for commodities. Uses cached OHLCV; falls back to a one-shot fetch when
    # the benchmark's series is missing.
    enable_sector_relative_momentum: bool = True

    # Money Flow Indicators — accumulation/distribution composite (MFI + CMF + OBV slope).
    # MFI (14-period): volume-weighted RSI; < 20 = accumulation, > 80 = distribution.
    # CMF (20-period): Chaikin Money Flow; positive = institutional buying.
    # OBV slope z-score: sustained volume trend direction.
    # Uses OHLCV chart cache first (works with ENABLE_FETCH_DATA=false); falls back to yfinance.
    enable_money_flow: bool = True

    # Trend Strength — ADX/DMI directional movement (Welles Wilder) + Donchian channel breakout
    # (the "Turtle" system). Measures trend QUALITY/strength + breakout confirmation — a dimension
    # not captured by momentum (return size) or RSI/Bollinger (overbought/oversold). ADX<20 = chop
    # (signal dampened); strong +DI/-DI separation with high ADX or a 20-day breakout = confirmed
    # trend. Uses OHLCV chart cache first (works with ENABLE_FETCH_DATA=false); falls back to yfinance.
    enable_trend_strength: bool = True
    trend_adx_period: int = 14        # Wilder ADX/DMI smoothing period
    trend_donchian_period: int = 20   # Donchian breakout channel lookback (Turtle = 20)

    # Post-Earnings Announcement Drift (PEAD) — one of the most-replicated cross-sectional
    # anomalies in academic finance. Stocks that beat (miss) EPS estimates tend to continue
    # drifting in the surprise direction for ~60 days as the market under-reacts. Score is
    # tanh(surprise_pct / surprise_scale_pct) × max(0, 1 − days_since_report / decay_window).
    # Uses yfinance earnings_dates (same source as enable_earnings); cached daily.
    enable_pead: bool = True
    pead_decay_window_days: int = 60       # days for the linear time-decay to reach zero
    pead_surprise_scale_pct: float = 25.0  # tanh saturation point (±25% surprise -> ±0.76)

    # IV Rank + Directional — volatility-regime-aware directional bias.
    # Uses 21-day realized vol percentile (vs trailing 252-day distribution) as a proxy
    # for IV Rank, combined with 5-day return / ATR to switch between contrarian (high IR)
    # and trend-confirming (low IR) directional scoring. Robust to regime shifts because
    # both inputs are self-normalised against each ticker's own vol footprint.
    # Uses OHLCV chart cache first (works with ENABLE_FETCH_DATA=false); falls back to yfinance.
    enable_iv_rank: bool = True

    # IV Expression — stock-vs-options expression decision from the real options chain.
    # Pulls live market-implied vol (expected_move_pct from GEX context) and ranks it
    # against the ticker's own trailing IV history (reconstructed from prior gex_*.json
    # caches). Combines with options-market oi_skew to derive expression bias:
    # cheap options + strong skew → CHEAP_DIRECTIONAL (confirm);
    # expensive options + strong skew → FADE_PREMIUM (contrarian).
    # No new data fetching — reuses the already-fetched GEX context.
    enable_iv_expr: bool = True

    # Cointegration Pairs — statistical-arbitrage market-neutral alpha (beyond sector_pairs).
    # Engle-Granger two-step: OLS hedge ratio on log prices → native (numpy) ADF test on
    # the residual spread. Cointegrated pairs whose spread z-score is stretched past the
    # entry band become market-neutral LONG-cheap / SHORT-rich trades. Also derives a
    # per-ticker directional lean fed into the aggregator. Cache-first OHLCV (works with
    # ENABLE_FETCH_DATA=false). No statsmodels dependency.
    enable_cointegration: bool = True
    cointegration_entry_z: float = 2.0     # |z| at/above which a pair is an actionable ENTRY
    cointegration_exit_z: float = 0.5      # |z| below which a pair is fair-value / no edge
    cointegration_pvalue: float = 0.05     # ADF significance level (0.01 | 0.05 | 0.10)

    # Cross-sectional ranking — measures how each ticker's per-method scores deviate from
    # the universe mean on each method, then averages the (capped) z-scores into a single
    # "stand-out" score per ticker. Composes additively into combined_score so the absolute
    # aggregation stays intact while the cross-sectional view adds a relative-value dimension
    # (robust to bull/bear regimes where absolute scores all skew one way).
    enable_cross_sectional: bool = True
    cross_sectional_weight: float = 0.20   # how strongly cs_score adjusts combined_score
    cross_sectional_zcap: float = 2.5      # cap individual z-scores to this magnitude

    # Business Cycle Rotation — Fidelity-style structural economic cycle phase → sector biases.
    # Derives EARLY_EXPANSION|MID_EXPANSION|LATE_EXPANSION|LATE_CYCLE|CONTRACTION from the
    # already-fetched FRED macro context (regime, yield curve, inflation, unemployment).
    # Pure synthesis module: no new API calls, no cache, instant computation.
    enable_business_cycle_rotation: bool = True

    # Catalyst Timing — three event-driven guards and amplifiers:
    #   1. Earnings Blackout: block BUY/SELL for tickers within 2 days of earnings (IV crush/gap risk)
    #   2. OpEx Max-Pain Amplifier: boost max_pain weight during OpEx week (0.20) / Triple Witching (0.28)
    #   3. 8-K + Insider Buy: auto-elevate to WATCH when both signals coincide for the same ticker
    enable_catalyst_timing: bool = True

    # ── Open-position monitoring: signal-decay exits + MFE/MAE tracking ──────
    # For every open trade, every pipeline tick re-evaluates the per-ticker
    # signal against today's data and exits early when the thesis has
    # materially deteriorated — even when no counter-direction recommendation
    # appears in the day's top-10 (which is the gap close_trades_on_signal_reversal
    # leaves open). Exit triggers, in order of severity:
    #
    #   1. signal_flipped     — today's combined score crossed against the trade.
    #                           For a BUY this means today's combined < flip_threshold
    #                           (e.g., -0.10). For a SELL, > -flip_threshold.
    #   2. signal_decay       — signal weakened by more than drop_threshold from entry.
    #                           Sized in oriented combined-score space so a BUY at
    #                           +0.65 dropping to +0.10 = 0.55 decay > 0.40 threshold.
    #   3. confidence_loss    — today's confidence dropped below confidence_floor.
    #   4. macro_regime_exit  — macro regime flipped to PANIC/RISK_OFF while long;
    #                           mirrors the existing entry-side block.
    #
    # MFE (max favorable excursion) and MAE (max adverse excursion) are tracked
    # passively on every tick — pure observability, doesn't trigger anything.
    enable_signal_decay_exits: bool = True
    signal_decay_flip_threshold: float = -0.10   # oriented combined < this -> flipped
    signal_decay_drop_threshold: float = 0.40    # oriented (entry - today) > this -> decayed
    signal_decay_confidence_floor: float = 0.60  # today's confidence < this -> exit
    signal_decay_regime_exit: bool = True        # exit longs in PANIC/RISK_OFF

    # ── Adaptive signal weighting ────────────────────────────────────────────
    # Multiply each method's static weight in the aggregator by a per-method
    # multiplier derived from its rolling solo win rate (data from
    # ``tracker.compute_solo_method_performance``). Methods that have been
    # right historically get up-weighted; methods that have underperformed
    # 50% get down-weighted. Bayesian shrinkage with ``prior_n`` virtual
    # trials at 50% smooths small-sample noise so 3-for-3 doesn't immediately
    # blow up to 2× weight.
    #
    # Formula:
    #   shrunk_wr = (wins + 0.5 × prior_n) / (n + prior_n)         in [0, 1]
    #   raw_mult  = shrunk_wr / 0.5                                 → 1.0 at 50% WR
    #   final     = clip(raw_mult, min_multiplier, max_multiplier)
    #
    # The multiplier is applied on top of whichever weight_profile is active
    # (base, market-mode override, or opex amplifier). So adaptivity composes
    # with regime-aware weighting rather than overriding it.
    enable_adaptive_weights: bool = True
    adaptive_weight_prior_n: int = 10              # virtual trades at 50% WR (Bayesian prior)
    adaptive_weight_min_multiplier: float = 0.5    # floor: a bad method keeps at least half its baseline
    adaptive_weight_max_multiplier: float = 2.0    # cap:   a great method gets at most double

    # ── Macro News Regime (geopolitics / oil / tariffs / policy) ────────────
    # Scans the day's news flow for macro-level themes (active wars and
    # geopolitical escalation, trade / tariff actions, oil/energy shocks,
    # central-bank surprises, fiscal/policy events, black swans) and produces
    # a composite regime read with sector implications. The composite feeds
    # the Macro Regime Filter so a CRISIS-grade headline actually tightens
    # the BUY threshold (and blocks longs during PANIC), and is passed to the
    # Claude analyst so geopolitical narrative gets folded into BUY/SELL/HOLD.
    #
    # Uses the news articles already fetched in step 1 — no extra feed cost.
    # Classification: a single DeepSeek call when ``DEEPSEEK_API_KEY`` is set,
    # otherwise a deterministic keyword-density heuristic (lower fidelity but
    # still useful). Cached hourly matching the news cache.
    enable_macro_news: bool = True
    macro_news_max_articles: int = 60   # cap for LLM payload size
    macro_news_min_articles: int = 3    # below this count, skip (not enough signal)

    # ── Intermarket Divergence (broad index ETFs vs SPY) ─────────────────────
    # Cross-market regime detector — tracks whether IWM (small-caps), RSP (equal-
    # weight S&P), QQQ (NASDAQ-100), DIA (Dow), MDY (mid-caps), EFA (developed
    # ex-US), EEM (emerging markets), IWF (growth), and IWD (value) are leading
    # or lagging SPY over 1m / 3m windows.
    #
    # Named regime labels surface canonical intermarket tells:
    #   NARROW_LEADERSHIP   — IWM + RSP both lag → mega-cap dependence, classic
    #                         late-cycle / distribution warning
    #   BROAD_PARTICIPATION — IWM + RSP both lead → healthy rally
    #   GROWTH_ROTATION     — QQQ + IWF lead vs IWD
    #   VALUE_ROTATION      — IWD leads vs IWF + QQQ
    #   US_EXCEPTIONALISM   — EFA + EEM both lag → dollar strength regime
    #   INTERNATIONAL_STRENGTH — EFA or EEM leading → softer dollar / global growth
    #
    # The composite intermarket_health ∈ [-1, +1] feeds the Macro Regime Filter
    # alongside VIX/MOVE/credit/breadth, so narrow leadership tightens the BUY
    # confidence threshold and broad participation relaxes it. Also passed to
    # the Claude analyst as macro context so it can synthesise the regime read
    # into the BUY/SELL/HOLD/WATCH decision.
    enable_intermarket: bool = True

    # ── Correlation-aware position sizing ────────────────────────────────────
    # The existing per-sector cap (tracker.SECTOR_CAP) catches GICS-sector
    # concentration but misses cross-sector factor concentration: three
    # high-beta semis (NVDA + AVGO + SMH) load the same factor even though
    # SMH is an ETF in a different sector bucket; three "high-beta growth"
    # names rated independently move together when the growth factor rolls.
    #
    # At trade-open time we compute realized pairwise correlations from the
    # last N trading days of OHLCV closes between the candidate and every
    # OPEN same-direction trade. Same-direction matters: a long-X / short-Y
    # pair is a hedge, not concentration — so opposite-direction positions
    # do NOT count toward the haircut.
    #
    # Multiplier scales linearly from 1.0× at mean_corr ≤ low_threshold to
    # min_multiplier at mean_corr ≥ high_threshold. A separate portfolio cap
    # caps the SUM of pairwise-correlation-weighted exposure across all
    # open same-direction positions — when adding the candidate would push
    # that sum over the cap, the trade is skipped entirely.
    enable_correlation_sizing: bool = True
    correlation_lookback_days: int = 60
    correlation_low_threshold: float = 0.30   # ρ̄ ≤ this → no haircut (1.0×)
    correlation_high_threshold: float = 0.80  # ρ̄ ≥ this → full haircut to min_multiplier
    correlation_min_multiplier: float = 0.25  # deepest soft haircut applied to the candidate
    correlation_portfolio_cap: float = 2.5    # Σ(size·ρ) hard skip threshold per direction
    correlation_min_overlap_days: int = 20    # minimum overlapping bars to trust a pair ρ

    # ── Out-of-sample validation (deterministic hash split) ──────────────────
    # Every closed trade is permanently assigned to "train" or "holdout" via a
    # deterministic hash of (seed, ticker, entry_date). Adaptive weights and
    # any other fit-on-history machinery use train ONLY — the holdout slice is
    # reserved for honest evaluation so a method that looks great on the data
    # it was tuned against can be seen failing on data it never touched.
    #
    # NOTE: this is a fixed train/holdout PARTITION, not a walk-forward (no
    # rolling window, no time-ordered re-training). The split is stable across
    # runs so a given trade is always evaluated the same way.
    #
    # Increase oos_split_seed to ANY new string to reshuffle the split (e.g.
    # after a strategy revamp where you want fresh evaluation). Setting
    # oos_holdout_pct=0 disables the holdout (degenerate: all data is train).
    enable_oos_validation: bool = True
    oos_holdout_pct: int = 30        # % of trades reserved as holdout (clamped 0..50)
    oos_split_seed: str = "llm_trader_v1"

    # ── Hypothetical always-open trades ──────────────────────────────────────
    # A separate, isolated category of trades where the listed tickers are ALWAYS
    # in an open position (BUY or SELL) — used as a baseline / reference book
    # alongside the real signal-driven trades. Each entry is "TICKER:BUY" or
    # "TICKER:SELL" (default BUY if no direction given; legacy LONG/SHORT are
    # accepted and normalised to BUY/SELL). Stored in
    # cache/hypothetical_trades.json — completely separate from cache/trades.json
    # so they NEVER contaminate real-trade performance metrics.
    enable_hypothetical_trades: bool = True
    hypothetical_trades: str = "GLD:BUY,SLV:BUY,GDX:BUY,NVDA:BUY"

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

    @property
    def hypothetical_trades_list(self) -> List[tuple]:
        """Parsed [(ticker, action)] list for the always-open hypothetical book.

        Action is 'BUY' or 'SELL' (default BUY when unspecified). Legacy
        'LONG'/'SHORT' values are accepted and normalised to 'BUY'/'SELL'
        for convenience. Empty list when the feature is disabled.
        """
        if not self.enable_hypothetical_trades:
            return []
        _aliases = {"LONG": "BUY", "SHORT": "SELL", "BUY": "BUY", "SELL": "SELL"}
        pairs: List[tuple] = []
        seen: set = set()
        for spec in self.hypothetical_trades.split(","):
            spec = spec.strip()
            if not spec:
                continue
            if ":" in spec:
                tk, action = spec.split(":", 1)
                tk = tk.strip().upper()
                action = _aliases.get(action.strip().upper(), "BUY")
            else:
                tk = spec.upper()
                action = "BUY"
            if tk and tk not in seen:
                seen.add(tk)
                pairs.append((tk, action))
        return pairs

    @property
    def factor_list(self) -> List[str]:
        """Merged, de-duplicated factor + thematic ETF universe (empty when disabled)."""
        if not self.enable_factor_etfs:
            return []
        out: List[str] = []
        seen: set = set()
        for s in f"{self.factor_etfs},{self.thematic_etfs}".split(","):
            sym = s.strip().upper()
            if sym and sym not in seen:
                seen.add(sym)
                out.append(sym)
        return out

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
