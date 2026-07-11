from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List

# Absolute path to <project root>/.env so settings load no matter the current
# working directory (e.g. when launched via Windows Task Scheduler from System32,
# or `python <abs path>/main.py` from your home dir). config/settings.py →
# parent (config/) → parent (project root) → .env
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    # Claude API
    anthropic_api_key: str

    # Model selection
    # Options: "claude-haiku-4-5-20251001" (fast/cheap), "claude-opus-4-6" (highest quality)
    analyst_model: str = "claude-haiku-4-5-20251001"

    # LLM A/B split — probability that a run picks the Anthropic engine as its
    # PRIMARY (the other provider stays as fallback). Applied independently to
    # synthesis (Claude analyst vs DeepSeek) and sentiment (Haiku vs DeepSeek),
    # re-flipped once per run. 0.5 = even split for side-by-side evaluation in
    # the dashboard's per-LLM rows; 1.0 = always Anthropic-first (legacy
    # behavior); 0.0 = always DeepSeek-first.
    llm_ab_anthropic_share: float = 0.5

    # SENTIMENT engine policy. When False (default), per-ticker news sentiment is
    # scored by DeepSeek ONLY — Claude/Haiku is never called for sentiment (not even
    # as fallback or via a hold-review engine pin); Claude is reserved for SYNTHESIS.
    # Set True to restore the Haiku ⇄ DeepSeek A/B on sentiment (governed by
    # llm_ab_anthropic_share). Synthesis routing is unaffected either way.
    enable_claude_sentiment: bool = False

    # SYNTHESIS-only N-way model bake-off. When set to a comma-separated list of
    # model ids, the per-run synthesis engine is picked UNIFORMLY from this pool
    # (equal split) instead of the binary llm_ab_anthropic_share flip — so 3+
    # models (e.g. Haiku, Opus 4.8, DeepSeek) accumulate comparable samples and
    # each shows as its own row in the dashboard's per-LLM evaluation. The chosen
    # model's provider is inferred (deepseek* → DeepSeek, else Anthropic); the
    # OTHER provider is the error fallback. Empty = legacy binary behavior above.
    # DeepSeek arms may carry a "-thinking" suffix (logical id → API model + reasoning
    # mode, decoded by claude_analyst._deepseek_spec). Sentiment is unaffected (stays
    # Haiku ⇄ DeepSeek flash non-thinking via llm_ab_anthropic_share). Example:
    #   LLM_AB_SYNTHESIS_MODELS=claude-haiku-4-5-20251001,claude-opus-4-8,deepseek-v4-flash-thinking,deepseek-v4-pro-thinking
    llm_ab_synthesis_models: str = ""

    # Anthropic prompt caching (cache_control) on the SYNTHESIS prompt. The large
    # persona + macro-context prefix is identical across the main call and the
    # opener-pinned hold-review calls within a tick, so caching it (5-min ephemeral)
    # makes the 2nd+ same-model call within a tick read the prefix at ~10% cost
    # instead of full price. Output is unchanged (same prompt, just billed cheaper).
    # A cache WRITE costs 1.25× the prefix tokens, a READ 0.1×, so this is a net win
    # only when prefixes are reused — measure via the "[claude] prompt cache:" log
    # and flip off if your model mix yields few reads. Sentiment is NOT cached (its
    # prefix is below Haiku's 2048-token cache minimum). Default on.
    enable_prompt_caching: bool = True

    # Held-positions prompt A/B — probability that a run includes the
    # <open_positions_context> block (the system's current holdings + a
    # zero-endowment-bias review instruction) in the synthesis prompt.
    # Re-flipped once per run; the flip is stamped on every trade CLOSED that
    # run (exit_hold_prompt) so the dashboard's method-evaluation table can
    # compare exit outcomes prompt-ON vs prompt-OFF over time. 0.5 = even
    # split for the experiment; 1.0 = always include; 0.0 = never (the LLM
    # stays blind to holdings, the pre-experiment behavior).
    open_positions_prompt_share: float = 0.5

    # Massive / Polygon.io market data — primary source for equity/ETF price + all
    # OHLCV timeframes (daily bulk via grouped-daily; 30-min intraday via aggregates,
    # REAL-TIME on the Stocks Advanced plan). Key: https://polygon.io (works globally,
    # no credit card for the free tier). If absent, the pipeline falls back to yfinance.
    polygon_api_key: str = ""

    # Company fundamentals (TTM valuation / profitability / leverage ratios) from the
    # Massive/Polygon financials & ratios endpoint — requires the Stocks Advanced plan
    # (or the ratios add-on). Fed into the LLM synthesis prompt as a quality/valuation
    # overlay. Key-gated + fail-graceful: inert (no block) without the entitlement.
    enable_fundamentals: bool = True
    # Per-ticker fundamentals ENRICHMENT (Massive short-interest + short-volume +
    # income-statement margin/YoY-growth) on top of the batched ratios — ~3 extra
    # calls/ticker, so capped to the first N (watchlist + early discovery). 0 = off.
    fundamentals_enrich_max_tickers: int = 50

    # Ticker-events watch (Massive corporate ticker-events: symbol/name changes,
    # delistings) — surfaces recent events on held + watchlist names as material
    # NewsArticles so a rename/delisting can't silently strand a position.
    enable_ticker_events: bool = True
    ticker_events_lookback_days: int = 45

    # Corporate actions (dividends + splits) from Massive/Polygon — upcoming ex-dividend
    # dates + recent/upcoming splits, fed to synthesis as a WHEN/mechanics overlay (§29).
    enable_corporate_actions: bool = True
    corp_actions_div_lookahead_days: int = 14   # surface ex-dividends within this many days
    corp_actions_split_window_days: int = 30    # surface splits within ± this many days
    corp_actions_div_max_tickers: int = 60      # per-ticker dividend-history fetches for the increase/cut factor (0 = off)
    # Additive (NOT normalised-pool) overlay weight for the directional corporate-action
    # factors (f_split + f_dividend) on combined_score — event-driven so it never dampens
    # the ~95% of tickers with no action. Placeholder; tune once IC data accrues.
    corp_action_factor_weight: float = 0.10
    # Additive-overlay weight for the 4 Massive fundamental factors (value/quality/
    # growth/short-squeeze), folded into combined_score 2026-06-24. Applied OUTSIDE
    # the normalised pool (like corp_action_factor_weight) so the capped/sparse
    # fundamentals nudge the combine without dampening non-enriched tickers. Small +
    # tunable; they remain forward-IC-validated in the dashboard Signal-IC table.
    fundamental_factor_weight: float = 0.08

    # Related-company peer discovery (Massive related-companies graph) — widens the
    # universe with peers of the watchlist + held names (liquidity-gated in Step 0).
    enable_related_discovery: bool = True
    related_discovery_max: int = 25

    # Massive/Polygon server-side technical indicators (RSI + MACD) scored as the
    # `massive` method, run ALONGSIDE our own `tech` for head-to-head dashboard
    # comparison. 2 API calls/ticker, so capped per tick. Set the weight via
    # aggregator `_BASE_WEIGHTS["massive"]`; off → the method scores 0 (no effect).
    enable_massive_tech: bool = True
    massive_tech_max_tickers: int = 0    # 0 = every ticker. Now a WEIGHTED member of
    # combined_score (promoted 2026-06-24), so it must score every ticker — a positive
    # cap would leave capped-out tickers with massive=0 while still reserving its weight
    # in the normalised pool, dampening their combined_score. Keep 0 unless reverting.

    # DeepSeek API (used for low-reasoning tasks)
    deepseek_api_key: str = ""

    # News sources
    newsapi_key: str = ""
    alpha_vantage_key: str = ""
    # Finnhub — real-time company news (free tier). Empty key → the source is
    # skipped. Free company-news has no per-article sentiment, so it adds news
    # COVERAGE; the provider-sentiment LLM-skip is driven by Polygon insights.
    finnhub_api_key: str = ""
    enable_finnhub_news: bool = False
    # Polygon/Massive news + per-article sentiment "insights" (each article carries
    # {ticker, sentiment, reasoning}). ON by default now that we're on the Advanced
    # plan — real-time Benzinga-sourced coverage; feeds the provider-sentiment hybrid
    # below so the LLM scorer can be skipped for these articles.
    enable_polygon_news: bool = True
    # Provider-sentiment hybrid: when an article carries a provider sentiment
    # (Polygon insights), derive the per-ticker news score from those instead of
    # calling the DeepSeek/Haiku scorer — a latency + cost win. Falls back to the
    # LLM when too few provider-scored articles exist. ON by default (Advanced plan).
    enable_provider_sentiment: bool = True
    provider_sentiment_min_articles: int = 2   # min provider-scored relevant articles to skip the LLM
    provider_sentiment_magnitude: float = 0.6  # |score| a positive/negative label maps to ([-1,1] scale)

    # Quiver Quantitative — alternative data (Hobbyist tier). Empty key → every
    # Quiver source is skipped. Congress trades revive the smart-money congressional
    # feed (dead since the Stock Watcher S3 went 403); gov-contracts / lobbying /
    # off-exchange (dark-pool) are rendered as synthetic NewsArticles and scored by
    # the sentiment pipeline (same pattern as Trends/Reddit/short-interest).
    quiver_api_key: str = ""
    enable_quiver_congress: bool = True        # → smart_money (List[InsiderTrade])
    enable_quiver_gov_contracts: bool = True   # → NewsArticle (federal contract awards = revenue catalyst)
    enable_quiver_lobbying: bool = True        # → NewsArticle (lobbying spend = regulatory-attention context)
    enable_quiver_offexchange: bool = True     # → NewsArticle (per-ticker dark-pool accumulation/distribution)
    quiver_lookback_days: int = 30             # window for congress / contracts / lobbying events
    quiver_offexchange_max_tickers: int = 60   # cap the per-ticker dark-pool loop (Hobbyist rate limits)

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
    # Per-ticker news via yfinance Ticker.news — gives EVERY symbol real,
    # ticker-tagged articles (not just the ~30 mega-caps the keyword aliases
    # cover), which is what actually feeds the news + sentiment-velocity scores.
    # Fetched once per hour (cached with the news pool). Kill switch if Yahoo
    # rate-limits.
    enable_ticker_news: bool = True

    # Per-ticker Google News RSS — free, no key, near-real-time, and far broader
    # than the 5 fixed market feeds (surfaces Reuters/Bloomberg/Barron's/FT AND
    # Business Wire, the one wire our direct feeds miss). Fetched fresh every tick
    # (reactivity fast-lane). google_news_max_tickers caps the per-tick request
    # burst; google_news_business_wire adds the per-ticker site:businesswire.com query.
    enable_google_news: bool = True
    google_news_max_tickers: int = 50
    google_news_business_wire: bool = True

    # FDA / MedWatch regulatory catalyst RSS (free, no key) — drug approvals/CRLs +
    # device recalls, on the fresh-every-tick fast lane. High signal for drug/device
    # names; market-wide (mapped via keyword aliases like the other RSS feeds).
    enable_fda_news: bool = True

    # Alpha Vantage NEWS_SENTIMENT — pre-scored per-ticker news that feeds the
    # LLM-skip hybrid (like Polygon insights). ONE batched call, hourly-cached.
    # OFF by default: the free tier is ~25 req/DAY, shared with AV discovery +
    # earnings — only enable on a paid tier (or if you don't use AV elsewhere).
    enable_alpha_vantage_news: bool = False
    alpha_vantage_news_max_tickers: int = 50

    # StockTwits crowd sentiment — one synthetic chatter-summary article per ticker
    # (LLM-scored, like Reddit). The public endpoint now 403s without auth, so this
    # needs a (free) StockTwits API token and is OFF by default.
    enable_stocktwits: bool = False
    stocktwits_access_token: str = ""
    stocktwits_max_tickers: int = 30

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
    earnings_upcoming_days: int = 30   # how many days ahead to include in calendar

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
    enable_google_trends: bool = False  # OFF 2026-06-25: pytrends' unofficial API is chronically 429'd by Google (low-value retail-attention proxy, no clean fix). Cached daily when on.

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
    # Loosened 2026-07-05 to WIDEN the net toward penny / lower-volume names so the
    # predictability panel can measure whether they are easier or harder to
    # predict (bucket features `price` + `dollar_vol`). The $1 floor keeps out
    # sub-$1 OTC junk (awful spreads / data). NOTE: dollar-volume is the main
    # universe-SIZE lever (most tickers are gated by it, not price) — if ticks get
    # slow or costly, raise it. 2026-07-08: these are now the OBSERVATION floor —
    # kept LOW so penny / thin-volume names enter the universe and keep accruing
    # performance data (the signals panel + the price/volume + predictability
    # panels); a SEPARATE, HIGHER TRADE floor (trade_min_* below) decides what can
    # actually trade, so sub-threshold names are scored + tracked but observe-only.
    discovery_min_price: float = 1.0                 # observation floor — min last close ($)
    discovery_min_dollar_volume: float = 1_000_000   # observation floor — min 20d avg $ volume ($)
    # TRADE liquidity floor (separate from + higher than the discovery/observation
    # floor above): a BUY/SELL for a ticker below EITHER of these is OBSERVE-ONLY —
    # still scored + persisted to the signals panel (so penny-stock performance
    # keeps accruing) but NEVER actionable (no sim trade, no broker order). Applied
    # in the pipeline actionable filter (Gate 4); fail-closed via is_liquid (a name
    # whose liquidity can't be verified is not traded). Flag false = disable.
    enable_trade_liquidity_gate: bool = True
    trade_min_price: float = 5.0                      # min last close ($) to be tradeable
    trade_min_dollar_volume: float = 5_000_000        # min 20d avg $ volume to be tradeable
    discovery_gate_max_fetch: int = 25               # cap cold OHLCV fetches per run for the gate
    # Drop exotic security TYPES from discovery (preferred series, warrants, units,
    # rights, OTC foreign ordinaries) — redundant with a primary listing and/or not
    # on the US consolidated tape (can't be priced deterministically). The pinned
    # watchlist bypasses the gate, so an explicitly-chosen preferred is still honored.
    enable_security_type_filter: bool = True

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
    # Minimum number of macro inputs (of ~10: VIX, MOVE, bond, global, FRED, breadth,
    # credit, DIX, intermarket, macro_news) that must be available for the composite
    # regime to be trusted. Below this, the regime is forced to at least CAUTION rather
    # than allowed to fail-OPEN to a permissive NEUTRAL when the feeds go dark.
    macro_regime_min_inputs: int = 3

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

    # ── Multi-timeframe technical signals (30-min / daily / weekly) ───────────────
    # Every OHLCV-based method (tech, vwap, momentum, money_flow, trend_strength,
    # iv_rank, pattern, sector_momentum) is also computed on a faster 30-min candle
    # and a slower weekly candle. The three timeframe scores are BLENDED (weights
    # below, renormalised over whichever timeframes are available) into the live
    # combined_score, and every per-(method, timeframe) score is persisted to the
    # signals panel for the dashboard's 4-category Information-Coefficient table.
    # Master OFF ⇒ exactly the legacy daily-only behaviour.
    enable_multi_timeframe_signals: bool = True
    enable_intraday_30m: bool = True       # fetch + score the 30-min candle (Massive/Polygon → yfinance)
    enable_weekly_signals: bool = True     # resample the daily cache → weekly candle (free, no fetch)
    intraday_30m_lookback_days: int = 120  # 30-min history depth per Massive/Polygon fetch
    # Strategy blend weights across timeframes (renormalised at runtime over the
    # timeframes that actually produced a score for a given ticker). Daily-dominant
    # by default; set tf_blend_1d=1.0 to revert to daily-only without flipping the flag.
    tf_blend_30m: float = 0.20
    tf_blend_1d: float = 0.60
    tf_blend_1w: float = 0.20

    # ── Horizon synthesis (term-structure of edge) ─────────────────────────
    # Per-ticker edge curve: each method's LIVE score weighted by its MEASURED
    # per-horizon information coefficient (from the simulated_trades panel) — pure
    # IC, no static blend, SIGN-AWARE (a negative-IC method is flipped, a no-skill
    # one drops out). Evaluated at every simulated-trade horizon
    # (30m/3h/6h/1d/3d/1w/2w/1m), then COST-AWARE selection picks the holding
    # horizon whose net-of-cost expected gross return is highest. The LLM may
    # CONFIRM or SHORTEN it (never lengthen — enforced mechanically at trade time);
    # the matched exit raises the hold-review floor once a position outlives its
    # horizon window. Judge nothing until the IC panel is thick.
    enable_horizon_synthesis: bool = True
    horizon_ic_days: int = 120            # lookback window for the per-horizon IC matrix
    horizon_ic_min_n: int = 15            # min joint obs before a (method,horizon) IC is used
    horizon_min_conviction: float = 0.05  # min |edge(h)| for a horizon to be a trade candidate
    horizon_cost_hurdle_pct: float = 0.40  # round-trip cost the net edge must clear (~27-41bp)
    horizon_ic_cache_seconds: int = 1800  # reuse the heavy IC matrix across ticks (30 min)
    # Matched exit: once a position is held ≥ its target-horizon duration, multiply
    # the opener-pinned hold-review confidence floor by this so it must still be
    # STRONGLY confirmed to survive past its edge window (short-horizon trades exit
    # sooner; a still-conviction winner is not cut). Own flag so it can be disabled
    # independently of the horizon assignment + dashboard.
    enable_horizon_matched_exit: bool = True
    horizon_expiry_floor_mult: float = 1.5
    # The matched-exit floor is CONTINUOUS, not a cliff: past the horizon the
    # required re-confirmation floor ramps from the normal base floor (at the
    # window) up to base × horizon_expiry_floor_mult, reaching full strength
    # horizon_expiry_ramp_windows windows past expiry (1.0 → full effect at 2× the
    # horizon window; short-horizon trades still tighten fast, long-horizon trades
    # stay patient). A same-direction position whose conviction drops below the
    # ramped floor closes (horizon_expired). Set ramp_windows→0 for the old cliff.
    horizon_expiry_ramp_windows: float = 1.0
    # A neutral HOLD/WATCH re-judgment past the horizon does NOT close at the
    # boundary (that was a premature-cut bug — it contradicted the same
    # never-close-on-neutral rule the within-window gate enforces). Instead a
    # persistent neutral position is flushed only once FULLY past the window
    # (ramp saturated) — and only when this is on. False = the matched exit is
    # purely a ramped conviction bar and never force-closes a neutral hold.
    horizon_expiry_flush_neutral: bool = True
    # Fallback horizon label for positions with NO target_horizon (opened before
    # horizon synthesis existed, or a run where it failed). Without it those
    # positions have no time-stop at all: a persistent-neutral loser rides
    # forever (observed 2026-07-01: the June-17 cohort at −20% with HOLD-0.09
    # reviews). Must be a HORIZON_HOURS label ("1w" → flush at 2× = 14 days with
    # the default ramp). Empty string disables the fallback (legacy behavior).
    horizon_default_window: str = "1w"
    # Re-entry cooldown: skip a new entry when a SAME ticker + SAME direction
    # trade was closed within this many hours. Stops the close→reopen churn where
    # a rule-based exit (horizon_expired / llm_confidence_loss) fires and the same
    # tick's entry pass immediately reopens the position (observed 2026-06-29: HUM
    # closed 11:38:30, reopened 11:38:31 — a pure round-trip cost). 0 disables.
    # Opposite-direction entries (a genuine flip) are never blocked.
    reentry_cooldown_hours: float = 4.0
    # When the opener-pinned hold-review engine is unavailable (e.g. Anthropic
    # credits exhausted — observed 2026-06-26→07-01: Claude-opened positions went
    # unreviewed for days, leaving them with NO exit gate), re-judge the position
    # with the OTHER provider instead of holding blind. The review row records the
    # actual reviewing engine, so provenance stays honest. False = strict pinning
    # (no review ⇒ hold, the original Fix #2 behavior).
    hold_review_engine_fallback: bool = True

    # ── Direction-aware, market-neutral edge curve (SHADOW MODE) ───────────
    # A second edge curve that weights each method by its DIRECTION-CONDITIONAL,
    # MARKET-RELATIVE skill: a method's bullish calls and bearish calls are scored
    # separately (some methods are reliable one way only), on returns net of SPY's
    # same-horizon move (so market drift can't masquerade as directional skill).
    # Per-side skill = 2·(market-relative hit rate − 0.5), shrunk toward the
    # method's both-sides skill by sample size. SHADOW ONLY — it is computed,
    # persisted, and shown on the dashboard next to the live (pooled) horizon, but
    # does NOT drive entries or exits, so it can be validated against outcomes
    # before promotion.
    enable_directional_shadow: bool = True
    horizon_dir_shrink_prior_n: int = 30   # virtual both-sides obs the per-side skill shrinks toward
    horizon_market_benchmark: str = "SPY"  # market leg subtracted to neutralise drift

    # ── Expected-move / market-aligned upside ranking ──────────────────────
    # Selection should favour the names with the biggest EXPECTED FAVOURABLE MOVE
    # (probability × magnitude) in the MARKET's direction: when the regime is
    # risk-on, the highest-upside longs; when risk-off, the biggest-downside shorts
    # (beta as a deliberate tailwind, decided by the regime layer — not the
    # drift-contaminated stock skill). expected_move = the edge curve's gross
    # expected favourable return at the target horizon; upside = conviction ×
    # expected_move × an alignment factor. SOFT: counter-market candidates are
    # haircut, not banned. Fed to the synthesis prompt + persisted; does not change
    # the actionable gate mechanics.
    enable_expected_move_ranking: bool = True
    horizon_counter_market_mult: float = 0.5   # upside haircut for a position fighting the regime
    # 0 = attempt every ticker (full coverage). A positive cap throttles the fetch;
    # only needed on the yfinance fallback path (≤60d history, per-IP 429s) — the
    # Massive/Polygon Advanced plan is unlimited-rate, so leave at 0 when configured.
    intraday_30m_max_tickers: int = 0
    # Re-fetch the 30-min OHLCV cache when its newest bar is older than this (minutes).
    intraday_30m_ttl_minutes: int = 25
    # Per-ticker scoring concurrency. The build_signals loop is I/O-bound (DeepSeek
    # sentiment ~7s/ticker + Massive/OHLCV reads), so a bounded thread pool collapses
    # the serial sum to ~max wall-time with IDENTICAL scores. 1 = sequential (legacy).
    # DeepSeek (sentiment = deepseek-v4-flash) caps at 2500 CONCURRENT, not RPM, and
    # throttles by latency, not 429s — so 32 still has huge headroom. Raised 16→32
    # (2026-07-08): Step 4 wall time scales ≈ tickers ÷ workers (measured 229s at
    # 433 tickers / 16 workers) — the latency-profile work. Watch for DeepSeek /
    # yfinance 429s in the logs; revert per-deploy via the env var if they appear.
    signal_scoring_max_workers: int = 32

    # ── Tick→order latency levers (2026-07-08 latency profile) ──────────────
    # Run the opener-pinned hold-review CONCURRENTLY with Steps 4–5 instead of
    # strictly after Step 5 (the review needs no main-synthesis output — it
    # refetches its own news/prices and re-judges with the OPENING engines; its
    # pinned synthesis still waits for the completed synthesis context, so the
    # review prompt is identical to the sequential path). Saved ~2.5 min/tick
    # measured. False = legacy sequential review (same code path, inline).
    # NOTE: the ledger-mark refresh (calibrate_sim_costs + update_open_trades)
    # runs before Step 4 in BOTH modes now — marks are stamped ~5 min earlier.
    enable_hold_review_overlap: bool = True
    # Cache the raw per-ticker sentiment LLM verdict keyed by (ticker, engine,
    # exact article set). A new article changes the key → fresh score, so news
    # reactivity is unchanged; only re-scoring an IDENTICAL digest is skipped
    # (the next 30-min tick, and the hold-review re-scoring held names minutes
    # after the main pass). TTL bounds staleness of the digest's age labels.
    enable_sentiment_cache: bool = True
    sentiment_cache_ttl_minutes: int = 180
    # Bounded concurrency for the per-ticker yfinance options-chain scan (the
    # fetch pool's slowest source, ~64s median). Kept LOW deliberately: the scan
    # shares yfinance's unofficial per-IP rate limit with the GEX pass (which
    # stays sequential after it) — history shows ~20+ req/s combined causes 429s
    # on every ticker. 1 = sequential (legacy).
    options_flow_max_workers: int = 2
    # Bounded concurrency for the discovery liquidity gate's COLD OHLCV warm-up
    # fetches (Polygon-first, so per-IP rate limits are not a concern; yfinance
    # is only the fallback). The sequential loop was the 7-minute stall on the
    # first tick after midnight (~200 uncached smart-money names). 1 = legacy.
    liquidity_gate_fetch_workers: int = 8

    # ── Classic cross-sectional anomalies (2026-07-08, panel-first) ─────────
    # Three literature-proven OHLCV-only methods (signals/classic_anomalies.py):
    # 52-week-high proximity (George-Hwang 2004 continuation), 12-1 skip-month
    # momentum (Jegadeesh-Titman 1993), and short-term reversal (Lehmann 1990,
    # 1-week, sign-flipped, liquid names only). Scored on every ticker and
    # IC-tracked in the signals panel at ZERO combine weight — promotion into
    # combined_score is a later, evidence-gated code change, not a flag flip.
    enable_high_52w: bool = True
    enable_momentum_12_1: bool = True
    enable_st_reversal: bool = True
    # Liquidity floor for the reversal signal (20-day average dollar volume) —
    # deliberately far ABOVE the $5M trade floor: below institutional size the
    # measured "reversal" is mostly bid-ask bounce, not a real snapback.
    st_reversal_min_dollar_volume: float = 50_000_000

    # ── Tier-2 panel-first methods (2026-07-08, weight 0 — same contract) ───
    # TTM Squeeze: BB(20,2σ) coiling inside Keltner(20,1.5×ATR); the release
    # fires in the direction of Carter's momentum oscillator (ttm_squeeze.py).
    enable_ttm_squeeze: bool = True
    # IV term-structure slope: front vs back ATM IV captured for free during
    # the GEX chain fetch (needs enable_gex for coverage; sparse ⇒ no view).
    # Backwardation = near-term event/stress premium → bearish tilt.
    enable_iv_term_structure: bool = True
    # Anchored VWAP from the 52-week high/low anchor days — POSITIONING read
    # (above the anchor = support = bullish), deliberately the opposite
    # convention of the mean-reversion rolling `vwap` method.
    enable_anchored_vwap: bool = True

    # ── Tier-3 panel-first methods (2026-07-08, weight 0 — same contract) ───
    # Residual momentum (Blitz-Huij-Martens 2011): 12-1 momentum on the
    # residual of a true-beta regression vs SPY — unlike sector/market
    # momentum's implicit beta of 1, a high-beta name in an up-market gets no
    # free momentum credit (residual_momentum.py).
    enable_residual_momentum: bool = True
    # Volume profile: 60-day volume-at-price histogram → POC + 70% value area;
    # acceptance outside value = trend score, inside value = small POC gravity
    # (volume_profile.py).
    enable_volume_profile: bool = True

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
    # Fix #2 — symmetric entry/exit rationale. When True (default), an LLM-opened
    # position is HELD/CLOSED by its OWN opening engine's fresh re-judgment each
    # run (the same synthesis call that gates entry), and ONLY on a tick that
    # engine is running — a Claude position is never closed by a DeepSeek run, and
    # vice versa. The aggregator's combined_score/confidence (a different
    # decision-maker, near zero on LLM-conviction names) is NO LONGER consulted
    # for LLM-opened trades; it survives only as the backstop for legacy /
    # rule-based-opened trades (gated by enable_signal_decay_exits below). Set
    # False to revert entirely to the aggregator-driven monitor.
    enable_llm_hold_review: bool = True
    # Every-tick, opener-pinned hold-review. When True (default), each open
    # LLM-opened position is re-evaluated on EVERY trading tick by the SAME
    # synthesis AND sentiment engines that opened it — fresh news + prices are
    # refetched, the ticker's signal is re-aggregated with the pinned sentiment
    # engine, and the pinned synthesis engine re-judges it — so entry-vs-now is a
    # same-engine, apples-to-apples comparison (temp=0 ⇒ low volatility). When
    # False, falls back to reviewing a position only on ticks whose A/B engine
    # happens to match the opener (the cheaper, partial behaviour). Costs extra
    # LLM calls per tick (one synthesis + pinned sentiment per engine combo held).
    enable_pinned_hold_review: bool = True
    enable_signal_decay_exits: bool = True
    signal_decay_flip_threshold: float = -0.10   # oriented combined < this -> flipped
    signal_decay_drop_threshold: float = 0.40    # oriented (entry - today) > this -> decayed
    # ── Confidence-loss floor — entry-relative with absolute backstop ──
    # Effective floor = max(signal_decay_confidence_floor,
    #                       signal_decay_confidence_floor_relative × entry_confidence)
    # where entry_confidence is the aggregator-confidence captured at trade entry
    # (signal_at_entry.confidence, NOT Claude's adjusted confidence — apples-to-apples
    # with today's aggregator confidence).
    #
    # Why entry-relative: a fixed 0.60 floor was firing on routine day-to-day signal
    # variance (78% entries decaying to 55% next day is normal noise), causing every
    # trade to exit on confidence_loss before the thesis got room to develop. Tying
    # the floor to entry conviction means a high-conviction trade gets a tighter
    # floor and must really collapse to trigger exit, while a borderline entry gets
    # a looser floor that tolerates routine variance. The absolute backstop catches
    # genuine conviction collapse.
    #
    # 2026-07-11 recalibration (post-exit forward-return monitor, exit_forward.py):
    # at 0.65 the relative leg bound at ~0.52-0.65 and fired on reviews in the
    # 0.50-0.62 band — closes that kept running +2.5% @1d / +7.2% @5d / +12.6% @10d
    # in the position's direction (n=19). Sweeping candidate floors over those
    # closes: 0.55 was the clean split — what it keeps ran +9.2% @5d, what it still
    # fires (conviction ≤~0.45-0.50) genuinely bled (−6.2% @5d, ADBE −12% @10d
    # avoided). The all-reviews panel agrees: NO confidence level separates
    # hold-profitable from bleed (exit_floor_calibration boundary = None at 1d AND
    # 5d over 162/67 review-days; the 0.5-0.6 bucket runs +6.8% @5d) — review-
    # confidence LEVEL is ~uninformative, so the floor should only catch collapse,
    # not lukewarmness. Freed trades stay governed by the flip / trailing-stop /
    # mechanical / horizon-ramp / macro exits (horizon_expired measures GOOD:
    # −1.4% @5d). Re-measure via `python -m src.analysis.exit_forward` as closes
    # accrue; the absolute stays calibration-adaptive (exit_floor_calibration).
    signal_decay_confidence_floor: float = 0.45          # absolute hard backstop
    signal_decay_confidence_floor_relative: float = 0.55  # factor × entry_confidence (2026-07-11: 0.65→0.55, measured)
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

    # ── IC-informed adaptive weights (panel-driven; ON, but CONFIDENCE-GATED) ──
    # A better-founded sibling of the win-rate layer above. Instead of solo win rate
    # from the gate-selected (thin, biased) trade ledger, it tilts each method's weight
    # by its RELIABILITY-ADJUSTED information coefficient — ICIR = mean(daily IC)/
    # std(daily IC) — measured over the UNBIASED `signals` panel (every scored ticker,
    # not just the trades the gates let through). By default the IC is MARKET-NEUTRAL
    # (shadow basis — ticker return minus SPY, pooled both directions), so a method is
    # weighted by its regime-robust SELECTION skill (alpha), NOT by how much market beta
    # it happened to ride in the lookback window (which may not repeat if the regime
    # turns); the market-direction call is owned separately by the regime/mode layers.
    # Set ic_weight_basis="edge" to weight by absolute-return IC instead. By design:
    #   • CONFIDENCE GATE: a method is reweighted ONLY when its mean-daily-IC t-stat
    #     clears the bar — t = |ICIR|·sqrt(n_days) ≥ ic_weight_min_t (t≥2 ≈ 95%). A
    #     method whose IC is not yet statistically distinguishable from zero keeps its
    #     base weight untouched. So on a thin panel NOTHING clears the gate and this is
    #     a pure NO-OP; methods earn a tilt only once the data proves their IC is real.
    #     (This is what makes it safe to run live now — it respects "judge nothing on a
    #     thin panel" automatically, then reweights method-by-method as each matures.)
    #   • Positive-only among the confident: a confident method's boost = clip(ICIR /
    #     median(confident positive ICIR), min, max) — the typical confident method is
    #     unchanged (1.0×), better ones boosted, weaker-but-still-positive ones trimmed.
    #     A confidently NON-positive method is floored to ic_weight_min_multiplier (it
    #     is anti-predictive — minimise it until the inversion switch flips its sign).
    # Applied as a multiplier on the active weight profile, exactly like the win-rate
    # layer (the two STACK if both on; set enable_adaptive_weights=false to use IC alone).
    enable_ic_weights: bool = True
    ic_weight_basis: str = "shadow"        # "shadow" = market-neutral IC (alpha, regime-robust) | "edge" = absolute-return IC
    ic_weight_horizon_days: int = 5        # EDGE-basis forward horizon in sessions (≈ a typical hold)
    ic_weight_shadow_horizon: str = "1w"   # SHADOW-basis horizon LABEL (30m|3h|6h|1d|3d|1w|2w|1m); "1w" = 5 sessions = the 5d edge horizon
    ic_weight_min_days: int = 5            # min distinct signal-days before an ICIR is even computed
    ic_weight_min_per_day: int = 5         # min cross-section per day for that day's IC to count toward ICIR
    ic_weight_min_t: float = 2.0           # CONFIDENCE GATE: reweight only if |ICIR|·sqrt(n_days) ≥ this
    ic_weight_min_multiplier: float = 0.25 # floor for a confidently anti-predictive method
    ic_weight_max_multiplier: float = 3.0  # cap on a strongly-predictive method's boost
    ic_weight_cache_seconds: int = 1800    # reuse the heavy panel IC across ticks / hold-review calls
    # Method INVERSION (manual, evidence-driven) — comma-separated method names whose
    # RAW score is reliably anti-predictive NET OF BETA across horizons (confirm via
    # `python -m src.analysis.scorecard` or `simulated_trades --directional`: a side
    # whose market-relative ICIR is confidently negative and STAYS negative across
    # horizons). Such a method contributes with a FLIPPED sign in combined_score — a
    # reliably-backwards signal, corrected, is a reliably-right one — and is lifted
    # off the IC anti-predictive floor to base weight. The signals panel keeps the RAW
    # score, so the inversion stays re-validatable (raw IC still shows backwards ⇒
    # keep inverting; raw IC flips positive ⇒ remove it). Empty = none. Set via
    # INVERTED_METHODS in .env — a reversible ops call, not a baked-in regime bet.
    inverted_methods: str = ""

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
    # Replaces the legacy hard per-sector cap, which caught only GICS-sector
    # concentration and missed cross-sector factor exposure: three high-beta
    # semis (NVDA + AVGO + SMH) load the same factor even though SMH is an
    # ETF in a different sector bucket; three "high-beta growth" names rated
    # independently move together when the growth factor rolls.
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
    correlation_health_max_fail_pct: float = 0.5  # >this share of intended pairs failing to compute → flag the sizing-correlation feed unhealthy (Data Quality)

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

    # Intraday scheduling — the runner ticks every 30 min and only acts inside
    # the regular session window below (ET, Mon-Fri). Combined with live prices
    # and completed-only daily bars, there is no dependency on an unclosed bar.
    intraday_session_start: str = "09:30"
    intraday_session_end: str = "16:00"

    # Extended-hours operation:
    #   "off"     — scheduler ticks RTH only (legacy behavior).
    #   "observe" — Phase 0: extended ticks run the FULL pipeline (signals,
    #               recommendations, DB persistence — the session-tagged
    #               evidence base) but NO ledger or broker mutation and no
    #               email.
    #   "trade"   — Phase 1 (default): extended ticks are FULL trading ticks —
    #               ledger entries/exits/marks and broker paper orders happen
    #               off-hours too, with session-aware costs (×4 spread),
    #               sizing (extended_size_multiplier), a stricter actionable
    #               gate (extended_confidence_bump), and LMT+outsideRth broker
    #               submissions. The daily email still fires only on the
    #               16:00 RTH closing tick.
    extended_hours_mode: str = "trade"        # "off" | "observe" | "trade"
    # Extended observation windows (ET, comma-separated HH:MM-HH:MM, optional
    # "@MM" per-window cadence override). Default covers the FULL extended day
    # 04:00–20:00: the liquid shoulders (07:00–09:30 pre-market ramp,
    # 16:00–17:30 earnings-reaction window) tick every extended_tick_minutes;
    # the thin dead zones (04:00–07:00, 18:00–19:00) tick hourly — spreads
    # there are widest and the evidence value per LLM dollar lowest, and the
    # hourly cadence matches the news cache TTL so each tick sees fresh news.
    # The LAST slot is 19:50, not 20:00: the pipeline needs ~4 min from tick to
    # order submission AND the every-tick engine-pinned hold-review adds more on
    # the critical path (fresh news/price refetch + per-engine re-synthesis), so
    # the final slot leaves a ~10-min pre-close buffer — a 20:00 (or even 19:55)
    # slot's orders would reach IBKR after the extended session closed and could
    # never fill same-day. 19:50 leaves the orders a live book before the close.
    extended_windows: str = "04:00-07:00@60,07:00-09:30,16:00-17:30,18:00-19:00@60,19:50-19:50"
    extended_tick_minutes: int = 30
    # Bid-ask half-spread multipliers outside RTH (commission is session-
    # independent — only the spread term widens). Rough placeholders to be
    # calibrated against IBKR paper fills later, same plan as commission_buffer.
    spread_extended_multiplier: float = 4.0
    spread_overnight_multiplier: float = 10.0

    # ── Overnight session (20:00 ET → 04:00 ET; IBKR overnight venue) ───────
    # Same three-state rollout as extended_hours_mode, for the OVERNIGHT
    # session. "trade": overnight slots are FULL trading ticks — the ledger
    # enters/exits/marks at the (heavily penalised: ×10 spread,
    # overnight_size_multiplier, overnight_confidence_bump) overnight terms and
    # the broker leg routes to IBKR's overnight venue (broker_overnight_routing).
    # The venue trades Sunday night → Thursday night, 20:00–03:50 ET
    # (market_calendar.is_overnight_session_open — Friday/Saturday/holiday-eve
    # nights have NO session and are never ticked or booked).
    overnight_hours_mode: str = "trade"       # "off" | "observe" | "trade"
    # Overnight tick slots (same "HH:MM-HH:MM[@MM]" grammar as extended_windows;
    # windows may NOT cross midnight — list the evening and morning halves
    # separately). Hourly: overnight books are the thinnest of the day and the
    # hourly cadence matches the news-cache TTL. The 20:30 first slot lets the
    # 20:00 close of after-hours settle; the 03:30 single slot is the last tick
    # whose orders can still work the book before the venue's 03:50 close
    # (mirroring the 19:50-not-20:00 rule) — 04:00 is already the first
    # extended slot.
    overnight_windows: str = "20:30-23:30@60,01:00-03:00@60,03:30-03:30"
    # Off-RTH entry sizing: overnight entries are sized harder down than
    # extended ones (×0.25 vs ×0.5) — the modeled spread is 10× RTH and no
    # fill evidence exists yet; accumulate evidence at low weight first.
    overnight_size_multiplier: float = 0.25
    # Actionable-threshold bump for OVERNIGHT runs (replaces, not stacks with,
    # extended_confidence_bump): the thinnest books demand the most conviction.
    overnight_confidence_bump: float = 0.10

    # ── Price-provenance health check ───────────────────────────────────────
    # Per-run guard against the stale-price class (the 2026-06-15 CRDO bug: an
    # entry booked at Friday's stale close while the live pre-market print — and
    # the analysis snapshot — was ~4.5% higher). After trades open, every leg's
    # recorded entry_price is compared to the run's snapshot price for that
    # ticker; a divergence beyond the session-appropriate band is flagged
    # (CRITICAL log + email/dashboard banner). RTH is tight (the snapshot and the
    # entry fetch are near-simultaneous); off-hours allows more drift between the
    # snapshot and the entry fetch on a thin, fast-moving tape.
    enable_price_provenance_check: bool = True
    price_provenance_band_rth_bps: float = 100.0
    price_provenance_band_extended_bps: float = 350.0
    price_provenance_band_overnight_bps: float = 600.0

    # ── Extended-session signal profile ─────────────────────────────────────
    # Outside RTH the information landscape changes: options chains (put/call,
    # max pain, OI skew, IV expression) are FROZEN at the last regular-session
    # close, while news flow and the extended-session price action are the
    # live, tradeable information. These knobs adapt the run accordingly.
    #
    # Confidence bump: added to the macro-regime actionable threshold for runs
    # executing outside RTH (extended AND overnight) — thin books + wide
    # spreads demand more conviction before a signal counts as actionable.
    # In "observe" mode it only shapes the persisted `actionable` flag; in
    # "trade" mode it directly gates which extended signals become positions.
    extended_confidence_bump: float = 0.06

    # ── Aggregator-agreement entry gate (combined_score) ───────────────────
    # A trade is accepted ONLY when BOTH decision-makers endorse it: the synthesis
    # (the LLM produced a BUY/SELL above the confidence threshold) AND the
    # aggregator (its weighted combined_score points the SAME way with at least
    # this magnitude). Prevents the LLM from opening a position the underlying
    # weighted methods don't support. Empirically accepted trades cluster at
    # |combined_score| 0.20–0.46 in their direction (none have ever opposed), so the
    # default is a floor that catches future weak/contradicted calls without
    # blocking well-formed ones — raise it to demand stronger method agreement.
    enable_combined_score_gate: bool = True
    min_combined_score_for_entry: float = 0.15
    # Position-size multiplier applied ON TOP of the confidence tier and the
    # correlation haircut for trades ENTERED outside RTH. Extended books are
    # thin and the modeled spread 4× wider, so pre-prod sizes off-hours
    # entries at half weight until paper fills prove the edge. 1.0 = off.
    extended_size_multiplier: float = 0.5

    # ── Evidence-based conviction sizing (2026-07-02 ledger study, n=44) ────
    # Two findings from the attributed ledger drive these knobs:
    # (1) LLM entry CONFIDENCE carries almost no return information (Spearman
    #     +0.10 with outcome; calibration slope ~0; the ≥0.92 bucket actually
    #     UNDERPERFORMED 0.85–0.92). The old ramp paid up to 2.0× for it. The
    #     ramp's span above 1.0× is therefore compressed:
    #       multiplier = 1.0 + (legacy_ramp − 1.0) × confidence_size_span
    #     1.0 restores the legacy 1.0→2.0× ramp; 0.0 = confidence-blind sizing.
    confidence_size_span: float = 0.5
    # (2) Agreement BREADTH — how many methods agreed with the direction at
    #     entry — was the strongest entry-time discriminator (Spearman +0.48;
    #     realized-only win rates 46% above the median split vs 23% below,
    #     Laplace-smoothed gap d≈0.20 over 26 closed trades; survives
    #     excluding the 2026-06-25 winner cohort). The convergence multiplier
    #     saturates at 2 agreeing methods, so breadth was previously unused
    #     above that. Sizing tilt, CONTINUOUS + SELF-CALIBRATING (2026-07-03):
    #       frac  = n_agreeing / len(attribution set)   ← survives method-set
    #               growth (the set already grew 19→28 once; absolute
    #               thresholds would have silently broken). NOT normalized by
    #               "methods that voted" — that flips the signal negative
    #               (the ledger shows information RICHNESS is the edge).
    #       ramp  = clamp((frac − center) / half_width, −1, +1)
    #       mult  = 1 + breadth_size_span × edge × ramp
    #     center/half_width are re-estimated each tick from the ledger's own
    #     recent breadth distribution (median / IQR over the last
    #     breadth_adaptive_window attributed trades once ≥ min_trades exist;
    #     the measured priors below until then), so the tilt ranks entries
    #     against the CURRENT book — new methods or regime shifts recenter it
    #     automatically. `edge` throttles the whole tilt by REALIZED evidence,
    #     Bayesian-shrunk: d_post = (prior_n·d_prior + n·d_obs)/(prior_n+n),
    #     edge = clamp(d_post / edge_ref, 0, 1) — grows toward full span as
    #     closed trades keep confirming the effect, decays to NEUTRAL (never
    #     auto-inverts) if it stops holding. Priors measured 2026-07-02.
    breadth_sizing_enabled: bool = True
    breadth_size_span: float = 0.2          # max ± size tilt at full evidence + saturation
    breadth_center_prior: float = 0.46      # ledger median frac (measured)
    breadth_halfwidth_floor: float = 0.09   # min ramp half-width (measured IQR)
    breadth_adaptive_min_trades: int = 10   # attributed trades before center/width adapt
    breadth_adaptive_window: int = 200      # recent attributed trades used to calibrate
    breadth_edge_prior: float = 0.20        # prior hi−lo realized win-rate gap (measured)
    breadth_edge_prior_n: int = 30          # pseudo-trades behind the prior (shrinkage)
    breadth_edge_ref: float = 0.25          # gap that counts as FULL evidence (edge=1)

    # Master switch for the session-dependent SIGNAL profile (default OFF so
    # scores are comparable across the trading day — the requirement behind the
    # fix #2 confidence trajectory). When OFF: one fixed weight profile is used
    # in every session (no stale-options/fresh-news overlay), `ext_gap` is part
    # of the active method set in ALL sessions (so the method set + weight
    # normalisation are identical — it still reads 0 in RTH by design, since the
    # daily technical stack already captures the gap), and the synthesis prompt
    # carries NO extended-session context block. When ON: restores the original
    # session-adaptive behaviour (overlay below + ext_gap off-hours-only +
    # the prompt's SESSION CONTEXT block). NOTE: the actionable-threshold bump
    # `extended_confidence_bump` is a GATE, not a score, so it is independent —
    # set it to 0.0 if you also want session-invariant actionability.
    enable_extended_signal_profile: bool = False
    # Aggregator weight overlay for extended/overnight runs (only applied when
    # enable_extended_signal_profile=True): options-derived methods (put_call,
    # max_pain, oi_skew, iv_expr) are scaled DOWN (stale since the RTH close —
    # yesterday's positioning, not live confirmation); news + sentiment-velocity
    # + the extended gap are scaled UP (the live extended-hours edge is overnight
    # news repricing).
    extended_stale_options_weight_mult: float = 0.5
    extended_news_weight_mult: float = 1.25

    # Extended-session gap momentum (ext_gap) — per-ticker scorer active ONLY
    # outside RTH (returns 0.0 = "no view" during the regular session, where
    # the open gap is already captured by technicals). Measures the live
    # extended print vs the last COMPLETED daily close, normalised by the
    # ticker's own ATR: pre-market gap-and-go / after-hours earnings reaction.
    # Score = tanh(gap_atr / scale) with a deadband so micro-moves don't read
    # as signal: |gap| < deadband × ATR → 0.0.
    enable_extended_gap: bool = True
    extended_gap_deadband_atr: float = 0.25   # min |gap| in ATR units to register a view
    extended_gap_scale_atr: float = 1.5       # tanh saturation: 1.5 ATR gap → ~0.76 score

    # Scheduler resilience on laptops / Modern-Standby machines. This box exposes
    # only S0 "Connected Standby" (no S1/S2/S3): when the display sleeps Windows
    # *suspends* this process, so cron fires come due while frozen and APScheduler's
    # default 1-second misfire grace silently drops them (the runner logged
    # "Scheduler started" but never ticked). Two guards:
    #   • keep_awake       — issue a Windows ES_SYSTEM_REQUIRED power request so the
    #     OS will not idle into standby while the scheduler runs (no-op off-Windows).
    #   • misfire_grace_sec — if the process IS suspended across a fire time, run the
    #     tick on resume (coalesced) instead of dropping it. Kept under the 30-min
    #     cadence so a late tick never duplicates the next scheduled one.
    scheduler_keep_awake: bool = True
    # How often the poll-loop runner re-reads the wall clock to decide whether a
    # 30-min slot is due. Short so that after a Modern-Standby suspension it
    # re-evaluates within this many seconds of resuming (instead of relying on a
    # precomputed long sleep, which standby skews).
    scheduler_poll_seconds: int = 30
    # How late a 30-min slot may still run after its boundary. If the machine was
    # suspended past this, the slot is skipped (logged) rather than run stale.
    scheduler_misfire_grace_sec: int = 1500
    # Flip this on to email on EVERY in-window tick (every 30 min) — handy for
    # confirming the scheduler fires. Turn back off once verified to avoid 13/day.
    # Takes precedence over scheduler_email_times below.
    scheduler_email_every_tick: bool = False
    # Comma-separated ET slot times (HH:MM) at which the daily report is emailed.
    # Non-empty (default) → the report sends ONLY at these slots; each MUST be an
    # actual tick slot (the RTH 30-min grid or an extended_windows boundary) to fire
    # — the scheduler logs a warning at startup for any configured time that isn't a
    # slot. Empty "" → legacy behaviour (only the 16:00 close). Overridden by
    # scheduler_email_every_tick. Default = 4 AM (pre-market open), 9:30 (RTH open),
    # 4 PM (RTH close), 7:50 PM (last after-hours tick).
    scheduler_email_times: str = "04:00,09:30,16:00,19:50"
    # Always send the report email — even on a non-email slot — when this run
    # detected a PROBLEM: a broker/execution issue (disconnect, rejects, drift),
    # an LLM-layer outage (credits/keys), or a price-provenance flag. So an issue
    # that surfaces between the scheduled email slots reaches you at the next tick
    # (~30 min) instead of waiting for 09:30/16:00/19:50. The forced email carries
    # the usual 🔔 banner + subject tag. Independent of scheduler_email_every_tick.
    email_on_problem: bool = True

    # Intraday timing overlay (Hybrid: daily trend decides direction; a 30-min
    # momentum read only gates entry/exit *timing*).
    enable_intraday_timing: bool = True
    intraday_timing_defer_threshold: float = 0.50   # |opposing 30-min momentum| above this defers an entry
    enable_intraday_exit: bool = False               # close a position on a strong intraday reversal against it
    intraday_exit_threshold: float = 0.60

    # ── Database (DuckDB — single source of truth for trades / recs / run meta) ──
    # One embedded file. The daily pipeline is the sole writer; the dashboard
    # reads it read-only. Created on first run (parent dir auto-made).
    db_path: str = "data/llm_trader.db"

    # ── Dashboard (Plotly Dash monitoring app: rationale · methods · returns) ──
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8050

    # ── Broker / live execution (paper-first; OFF by default → no broker calls) ──
    # Pre-production: drive a real broker's PAPER account in parallel with the
    # internal NAV sim ("shadow & reconcile"), then flip to live with a port swap.
    # IBKR is the only API broker offering a Canadian resident both paper and live
    # for US securities (CIRO blocks API only for Canadian-listed names, which this
    # system never trades — the universe is 100% US stocks/ETFs). See src/broker/.
    #   off        — no broker calls; internal simulation only (default; unchanged behavior)
    #   dry_run    — log the orders that WOULD be placed (sizing + idempotency); submit nothing
    #   ibkr_paper — submit to IB Gateway PAPER account
    #   ibkr_live  — submit to IB Gateway LIVE account  [gated: only after paper validation]
    broker_mode: str = "off"
    # ── Broker advisor (IBKR account / short-borrow-aware method group) ──
    # First method in a broker-aware group: scores short-borrow state (hard/expensive
    # to short → bullish squeeze tilt → fades a SELL). The only method aware of
    # IBKR-unique data; decision-only (never trades). Needs a live IBKR connection,
    # so it is OFF unless broker_mode != off AND this flag is on. Each scored ticker
    # costs one market-data request, so the per-tick fetch is capped.
    enable_broker_advisor: bool = False
    broker_advisor_max_score: float = 0.6          # cap on the squeeze tilt (a single tell can't dominate)
    broker_advisor_expensive_fee_pct: float = 10.0 # borrow fee %/yr mapping to a strong tilt (tanh scale)
    broker_advisor_hard_shares: float = 200000.0   # shortable shares at/below which a name is "hard to borrow"
    broker_advisor_max_tickers: int = 60           # cap on per-tick borrow fetches (held names + universe slice)
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4002              # IB Gateway: 4002 paper / 4001 live  (TWS: 7497 / 7496)
    ibkr_client_id: int = 11           # any stable int unique to this API connection
    ibkr_account: str = ""             # optional: pin a specific IBKR account id (else the sole/first)
    ibkr_connect_timeout: int = 15     # seconds to wait for the Gateway socket
    # Bound every ib_async API REQUEST (reqExecutions / reqPositions / placeOrder /
    # reqPnL / reqMktData) so a stuck request can't freeze the whole scheduler —
    # the 2026-07-06 hang: the overnight reconcile blocked on reqExecutions after
    # connecting, froze for 6+ h, missed every tick + the 04:00 email. ib_async's
    # IB.RequestTimeout raises after this many seconds instead of waiting forever;
    # the reconcile's fail-soft try/excepts then continue the tick. 0 = the old
    # freeze-forever behaviour. See src/broker/ibkr.py.
    broker_request_timeout_seconds: float = 45.0
    # SHORT timeout for the real-time quote snapshot in get_market_price only. A
    # live quote arrives in ~1-3s in RTH (reqTickers returns as soon as the snapshot
    # completes, so this never delays a successful fetch); pre-market/thin data never
    # arrives, so without a tighter bound each priced ticker would burn the full
    # broker_request_timeout_seconds (45s) and the reconcile — which prices every
    # open/drift position one-by-one — could march into the sync watchdog and force a
    # pre-market respawn loop. Fail fast to None instead (caller falls back). 0 = use
    # the global request timeout.
    broker_price_timeout_seconds: float = 8.0
    # Hard wall-clock backstop on the ENTIRE broker reconcile: if it somehow still
    # exceeds this (a hang RequestTimeout doesn't cover), force-exit the process so
    # the task manager restarts it (a fresh tick > a permanent freeze). Set well
    # above a legitimate slow reconcile; 0 disables. See pipeline.py.
    broker_sync_watchdog_seconds: int = 600
    # Prefer IBKR's real-time last/mark price (free Cboe One + IEX feed via the
    # broker connection) over yfinance in tracker._fetch_price — the same data
    # that fills the orders, so the mark matches the execution venue. Requires an
    # IBKR broker_mode (ibkr_paper/ibkr_live); falls back to yfinance/Polygon when
    # off, disconnected, or the quote is missing. OFF by default for A/B.
    enable_ibkr_price_feed: bool = False
    # Position sizing. Two modes (broker_sizing_mode), each × the 1.0/1.5/2.0×
    # confidence tier already on every trade:
    #   "notional"   — fixed base order size: broker_base_notional in broker_base_notional_ccy.
    #                  US securities are USD-priced, so a non-USD base is converted to a USD share
    #                  budget via live FX (src/broker/fx.py); broker_fx_fallback_cad_usd is used
    #                  only if the live quote is unavailable.
    #   "equity_pct" — broker_base_position_pct of account equity.
    # shares = floor(budget_in_USD × size_multiplier / price).
    broker_sizing_mode: str = "notional"          # "notional" | "equity_pct"
    broker_base_notional: float = 1000.0          # base order size per 1.0× position (notional mode)
    broker_base_notional_ccy: str = "CAD"         # currency of broker_base_notional (matches the account)
    broker_fx_fallback_cad_usd: float = 0.73      # CAD→USD, used only if the live FX quote is unavailable
    broker_base_position_pct: float = 0.05        # equity_pct mode: 5% of equity per 1.0× position
    # IBKR's API can't place FRACTIONAL equity orders (error 10243 — GUI-only), so a small
    # base is rounded to whole shares. "nearest" rounds half-up (a name still places as long
    # as it's ≥ half a share, i.e. priced up to ~2× the budget); "floor" sticks to the budget
    # (skips any name priced above one position's worth). "nearest" avoids needless skips.
    broker_share_rounding: str = "nearest"        # "nearest" | "floor"
    broker_max_positions: int = 20                # hard cap on concurrent broker positions
    broker_max_gross_exposure_pct: float = 1.0    # cap on Σ|notional_usd| / equity_usd (1.0 = no leverage)
    # Order type for broker submissions:
    #   "LMT" (default) — marketable limit at model price ± the session cap in the
    #           adverse direction (BUY above / SELL below). Bounds the worst
    #           acceptable fill; on liquid names it executes as fast as MKT.
    #           The settle pass removes LMT's historical downside: an order
    #           whose cap is missed is re-anchored at a fresh quote within the
    #           tick and KILLED if still unfilled — it never rests to fill
    #           late at a stale price.
    #   "MKT" — market order: fills immediately at whatever the book offers,
    #           with NO price bound (observed 2026-06-12: −107 bp ZUMZ and
    #           −91 bp ATGL fills on thin RTH books). Off-RTH ticks force LMT
    #           regardless (IBKR rejects MKT outside regular hours).
    broker_order_type: str = "LMT"                # "LMT" | "MKT"
    broker_limit_cap_bps: float = 20.0            # LMT only: max adverse distance from model price (RTH)
    # Off-RTH the book is thin and the REAL spread is ~4× wider (the sim's own
    # cost model charges 4× extended / 10× overnight half-spreads) — a 20 bp
    # cap sits INSIDE the extended spread, so every off-RTH order rests
    # unfilled, is cancelled next tick and chases the market on stale data.
    # A wider extended cap keeps off-RTH orders genuinely marketable: fill in
    # the decision tick at the current (wide) spread rather than later at a
    # drifted price.
    broker_limit_cap_bps_extended: float = 80.0   # LMT cap outside regular hours
    # Overnight LMT cap: the overnight book is thinner still (sim models it at
    # ×10 RTH spread) — a cap inside the spread can never fill. Placeholder to
    # calibrate against real overnight paper fills, like the extended cap.
    broker_limit_cap_bps_overnight: float = 150.0
    # Route overnight-session orders to IBKR's overnight venue (contract
    # exchange "OVERNIGHT" instead of SMART; LMT-only, TIF DAY, ~10k eligible
    # US stocks/ETFs). Fail-soft: an ineligible symbol / unentitled account
    # rejects, lands as SUBMIT_FAILED, and the tick-scoped lifecycle retries or
    # kills it — never breaking the run. False = legacy behavior (off-venue
    # orders rest until the 04:00 pre-market open, where the settle pass kills
    # them — i.e. no overnight broker fills).
    broker_overnight_routing: bool = True
    broker_paper_equity: float = 100000.0         # USD equity used for the exposure cap in dry_run

    # ── Order-submission reliability (acceptance check + bounded retry) ──
    # Every submission's broker answer is verified; TRANSIENT failures
    # (connection drop, timeout, pacing/rate limit) retry in-tick a few
    # times. The window is deliberately SHORT — a delayed fill must stay
    # anchored to THIS tick's model price — and every retry goes out as a
    # marketable LMT capped at broker_limit_cap_bps from that model price,
    # so the worst acceptable fill never drifts past the cap no matter when
    # the retry lands. Hard rejects (insufficient funds, permissions,
    # invalid contract) never retry — a retry can't fix them. Before each
    # retry the broker is checked for an order already carrying this
    # client_ref: an attempt that errored AFTER transmission may have
    # reached the broker, and resubmitting blind would double the position.
    broker_submit_retries: int = 2        # transient retries per order (0 = off)
    broker_retry_wait_seconds: int = 5    # pause before each retry (reconnect window)
    # Connect retries at sync start: IB Gateway has a daily re-login window —
    # without a retry, one badly-timed tick loses its whole order cycle
    # (observed 2026-06-11 15:41 ET: "not connected", 4 positions waited a tick).
    broker_connect_retries: int = 2           # extra connect attempts at sync start
    broker_connect_retry_wait_seconds: int = 10
    # Automatic in-tick reconnect (2026-07-11): when the Gateway session DROPS
    # (gateway restart, network blip, daily re-login), every IBKRBroker call
    # revives it via a throttled implicit connect (_ensure_connected) instead of
    # failing soft until the next tick's sync. After a failed dial, implicit
    # revives pause this long so a down/wedged gateway can't charge each price
    # fetch / sync step the full ibkr_connect_timeout; explicit connect() calls
    # (the sync-start retry loop, _submit_with_retry) bypass the pause, and any
    # successful dial clears it. 0 = no throttle (dial on every touchpoint).
    broker_reconnect_cooldown_seconds: float = 60.0
    # ── Settle pass: fill fast or kill ───────────────────────────────────
    # After this tick's orders are submitted, actively watch them for up to
    # this many seconds: fills are recorded the moment they land; a zero-fill
    # order is cancelled and re-anchored at a fresh capped quote EARLY and
    # REPEATEDLY (see broker_settle_reanchor_every) so a mispriced order is
    # re-priced within seconds and keeps chasing the spread in bounded steps;
    # anything still unfilled at the deadline is CANCELLED — nothing ever rests
    # across ticks, so an order either executes within this window of its
    # decision or it does not exist (the next tick re-decides from fresh data
    # and prices). Partial fills are left working. 0 = legacy (rest until next
    # tick). Lowered 60→30 (2026-07-07) to bound the added tick time while the
    # tighter poll + repeated re-anchor keep fills fast.
    broker_settle_seconds: int = 30
    # Fill-poll cadence inside the settle window (seconds). Smaller = fills are
    # detected + orders re-anchored sooner (more reliable), at more broker
    # round-trips. 3s catches a fast fill in ~one poll; well inside IBKR pacing.
    broker_settle_poll_seconds: int = 3
    # Re-anchor a still-unfilled order every N settle polls (not just once at the
    # halfway mark): with a 3s poll this re-prices at ~6s, ~12s, ~18s … so an
    # order that missed on a moving book gets several fresh-quote retries within
    # the budget instead of one. The final poll never re-anchors (it observes).
    broker_settle_reanchor_every: int = 2
    # ── Order lifetime: tick-scoped (default) or age-based ──────────────
    # Tick-scoped (True): an order lives exactly one tick. Any order still
    # unfilled at the next sync is cancelled and re-decided from THIS tick's
    # data — entries resubmit only if the trade survived this tick's signal
    # pass (monitor_open_positions runs first), re-anchored at the current
    # mark; exits always resubmit re-anchored. No order ever works the book
    # on a previous tick's price. False: legacy age rule below.
    broker_tick_scoped_orders: bool = True
    # Age fallback (used when tick-scoped is False; also an upper bound when
    # True): an ACCEPTED order resting unfilled this many minutes is
    # cancelled and resubmitted re-anchored at the current mark. Partial
    # fills are left working. 0 = never (no age rule).
    broker_unfilled_cancel_minutes: int = 90
    # Price-aware next-tick resubmit for a previously-unfilled ENTRY. Instead of
    # blindly chasing the current price on the next tick, decide per the fresh
    # signal + the price vs the original decision: (a) still actionable same
    # direction → resubmit; (b) decayed to non-actionable BUT the price is
    # as-good-or-better than the decision (≤ entry for a long / ≥ entry for a
    # short) → resubmit at the better price; (c) decayed AND the price drifted
    # adverse → HOLD (don't chase — the tick after re-evaluates); (d) signal
    # flipped to the opposite side → don't resubmit. EXITS are unaffected — they
    # always resubmit to get flat. Needs the tick's actionable set threaded into
    # reconcile.sync(); with none supplied it falls back to the legacy chase.
    broker_price_aware_resubmit: bool = True

    # ── Drift auto-reconciliation ────────────────────────────────────────
    # The ledger is the source of truth; a broker position the ledger cannot
    # explain (no OPEN trade, no close in flight) is an ORPHAN. Drift is
    # prevented at the source where possible (working entry orders are
    # cancelled the moment their trade closes; exits flatten what the broker
    # ACTUALLY holds), and whatever still slips through (ledger restores,
    # manual TWS trades, fill races) is handled per this setting:
    #   "flatten" (default) — submit a price-capped marketable LMT at a live
    #                         quote to close the orphan; re-anchored each tick
    #                         while it rests, so the broker converges to the
    #                         ledger in bounded ≤cap steps.
    #   "report"            — legacy behavior: surface it, touch nothing.
    # SAFETY: in ibkr_live mode "flatten" is refused and downgraded to
    # "report" — auto-selling unexpected REAL-money positions (e.g. something
    # bought manually in TWS) is a deliberate human decision, not a default.
    broker_drift_action: str = "flatten"   # "flatten" | "report"

    # ── Simulated trading costs (commission term in the sim's return math) ──
    # The bid-ask half-spread model (src/performance/spread.py) covers spread cost;
    # this adds per-order commission so small positions don't overstate their edge
    # (at ~$730 USD/position, IBKR Fixed's $1 minimum is ~27 bp round trip — larger
    # than the modeled spread on a liquid large cap). Applied symmetrically in
    # _pct_return AND the daily-NAV walk's entry/exit anchors, so both engines agree.
    #
    # CONSERVATIVE BY DEFAULT: the model deliberately errs toward OVERSTATING fees
    # so reported performance understates the edge rather than flattering it —
    # the pricier all-in plan (ibkr_fixed) is assumed, and commission_buffer adds
    # headroom for everything the published schedule excludes. Actual commissions
    # captured from fills (broker_orders.commission in DuckDB) are the ground
    # truth for calibrating the model down later.
    #   "ibkr_fixed"  — max($1.00, $0.005/share), capped at 1% of trade value
    #                   (default — the more expensive plan, exchange fees included)
    #   "ibkr_tiered" — max($0.35, $0.0035/share), capped at 1% of trade value
    #                   (commission only — venue/clearing fees are NOT in the base
    #                   rate; rely on commission_buffer to cover them)
    #   "none"        — spread-only (legacy behavior)
    commission_model: str = "ibkr_fixed"          # "ibkr_fixed" | "ibkr_tiered" | "none"
    # Fee ceiling multiplier applied AFTER the min/cap schedule math. Covers the
    # pass-throughs the schedule omits — SEC transaction fee + FINRA TAF on sells,
    # exchange/clearing fees (tiered), odd venue surcharges — plus schedule drift.
    # 1.5 ≈ $1.50 min/side → ~41 bp round trip at the $730 base notional (actual
    # paper fills run ~27 bp — the gap is the intended safety margin). Set 1.0 to
    # model the published schedule exactly.
    commission_buffer: float = 1.5
    # Assumed USD notional per 1.0× position, used to convert the per-order minimum
    # into percentage terms. Keep roughly in sync with broker_base_notional × FX
    # (1000 CAD × 0.73 ≈ 730 USD). A deliberate constant (not live FX) so the
    # return math stays 100% deterministic — and conservative: larger (1.5–2.0×)
    # positions experience a SMALLER min-commission floor in % terms, so pricing
    # every trade at the 1.0× base notional is the worst case.
    commission_notional_usd: float = 730.0
    # ── Calibrate the sim cost to REAL IBKR fills ────────────────────────
    # When on, the simulated per-leg cost (half-spread + commission in
    # _one_side_cost) is REPLACED by the measured average all-in one-way cost
    # from actual broker fills (real commission + execution cost vs the
    # decision price, the same number the dashboard's IBKR "Avg 1-way cost"
    # shows), applied flat to every entry/exit leg. The sim then charges what
    # execution actually costs instead of the modeled estimate. Falls back to
    # the model until at least sim_real_fill_costs_min_legs filled legs exist
    # (a flat average over 2-3 fills would be noise) and is clamped ≥ 0 (a
    # net-favorable fill streak never pays the sim to trade). Still fully
    # deterministic: the calibration is a pure function of the broker fills in
    # the same DuckDB. Set False to keep the modeled cost.
    sim_use_real_fill_costs: bool = True
    sim_real_fill_costs_min_legs: int = 10
    # ── Per-session cost calibration (2026-07-03) ────────────────────────
    # The flat real-fill override blends all sessions into one number, which
    # under-charges extended/overnight legs and slightly over-charges RTH.
    # When enough fills exist the override carries a per-session split: RTH
    # measured directly (needs ≥ session_cost_min_legs RTH legs); extended /
    # overnight = RTH × a session multiplier Bayesian-shrunk from that
    # session's own fills toward the documented ×4/×10 priors
    # (spread_extended_multiplier / spread_overnight_multiplier), floored at
    # 1.0 (off-hours never cheaper than RTH) and capped at 2× the prior. So
    # the punitive ×10 overnight assumption relaxes toward MEASURED overnight
    # costs as venue fills accrue — automatically, with the conservative
    # prior in force until then. Deterministic (a pure function of the
    # broker_orders fills in the same DuckDB).
    session_spread_calibration_enabled: bool = True
    session_cost_min_legs: int = 5        # min legs before a session's own mean counts
    session_spread_prior_n: int = 20      # pseudo-legs behind the ×4/×10 priors
    # ── Derived horizon cost hurdle (2026-07-03) ─────────────────────────
    # horizon_cost_hurdle_pct froze the round-trip hurdle at 0.40% while the
    # system MEASURES its real cost (~0.16% round trip from LMT fills). When
    # a real-fill calibration is active the hurdle derives instead:
    #   hurdle% = 2 × calibrated one-way% × cost_hurdle_safety
    # (clamped to [0.05, 2.0]%), so horizon selection tracks actual execution
    # costs. The static setting remains the no-calibration fallback.
    cost_hurdle_use_calibrated: bool = True
    cost_hurdle_safety: float = 1.5
    # ── Engine-relative actionable threshold (2026-07-03) ────────────────
    # Confidence distributions are ENGINE-specific (DeepSeek hands out 1.00s,
    # Claude tops out lower), so absolute thresholds silently change meaning
    # when the A/B flip or ANALYST_MODEL changes engines. When enabled, each
    # static threshold (regime ladder included) is translated into the run
    # engine's own confidence scale by matching the gate's SELECTIVITY —
    # effective = Q_engine(F_global(static)) over the last
    # threshold_calibration_days of BUY/SELL recommendations — then shrunk
    # toward the static anchor while the engine's history is thin and clamped
    # to ±threshold_max_shift (the gate drifts with evidence, never jumps).
    threshold_engine_relative_enabled: bool = True
    threshold_calibration_days: int = 30
    threshold_min_global_recs: int = 300   # min all-engine sample before translating
    threshold_engine_prior_n: int = 150    # pseudo-recs behind the static anchor
    threshold_max_shift: float = 0.08      # max drift from the static threshold
    # ── Calibrated exit-confidence floor (2026-07-03) ────────────────────
    # signal_decay_confidence_floor froze the absolute close threshold at 0.45
    # while trade_reviews records, every tick, exactly the evidence that
    # decides it: re-affirmation confidence vs the position's oriented forward
    # return. The calibrated floor = the lowest confidence where holding is
    # still profitable (below-floor reviews lose money, above-floor make it),
    # Bayesian-shrunk toward the static prior and clamped to the band below.
    # The static setting remains the prior/fallback; the relative floor
    # (× entry confidence) stays static by design.
    exit_floor_calibration_enabled: bool = True
    exit_floor_calibration_days: int = 60
    exit_floor_prior_n: int = 150          # pseudo-reviews behind the static prior
    exit_floor_min_side: int = 25          # min reviews on each side of a candidate
    exit_floor_min: float = 0.30
    exit_floor_max: float = 0.70
    # ── Exit-conviction consensus (2026-07-03) — the entry-breadth analog ──
    # The exit decision used a single LLM-scalar hold-review and ignored the
    # 27-method exit-conviction panel. This nudges the same-direction close
    # floor by the breadth-of-raw-methods EXIT CONSENSUS (mean of the signal
    # methods' oriented exit scores; LLM review + aggregator excluded so Fix #2
    # trigger-happiness isn't reintroduced): consensus says exit → floor raised
    # (close more readily); says hold → floor lowered (hold more readily).
    # EVIDENCE-THROTTLED + BOUNDED: span_eff ramps with the closed-trade sample
    # from a small prior toward a cap, so a confident LLM hold is never
    # overridden (only borderline convictions get tipped) and with no data it's
    # a gentle nudge. UNVALIDATED by design — to be confirmed/denied over the
    # coming weeks by the offline exit_policy_eval harness; set enabled=False to
    # revert to the pure LLM-scalar close. See analysis/exit_conviction.py.
    enable_exit_conviction: bool = True
    exit_conviction_span_prior: float = 0.03   # floor nudge with ~no evidence (gentle)
    exit_conviction_span_max: float = 0.10     # bounded cap at full evidence
    exit_conviction_prior_n: int = 40          # closed trades for the ramp half-life
    # ── Edge-decay time-stop (2026-07-06) ─────────────────────────────────────
    # The realized edge of combined_score decays with holding horizon (measured
    # tick-by-tick over the signals panel — analysis/horizon_edge.py). On the
    # traded subset the edge peaks ~1-2d and turns negative by ~5d. This layer
    # measures the edge-positive WINDOW and, evidence-throttled, raises the
    # confidence-loss close floor once a position is held past it — an
    # edge-decay time-stop that fires around the realized decay point rather than
    # only the entry target_horizon. Inert on today's thin long-horizon sample;
    # firms up as the decay confirms. Also emitted as the `edge_decay` exit signal
    # so its OWN exit-IC is measured before it earns real weight.
    enable_edge_decay_exit: bool = True
    edge_decay_conf_min: float = 0.78          # confidence subset the edge curve is measured on (the traded population)
    edge_decay_cal_days: int = 90              # panel window for the edge curve
    edge_decay_min_n: int = 20                 # min obs per horizon before it counts
    edge_decay_prior_obs: int = 250            # obs prior shrinking the evidence strength (gentle now)
    edge_decay_floor_cap: float = 0.08         # max confidence-loss-floor raise from the edge-decay stop
    edge_decay_cal_ttl_seconds: int = 21600    # calibration cache TTL (6h; changes ~daily)
    # ── Exit fixes (2026-07-08) — the scorecard flagged a −58% MFE give-back (no
    # profit-taking: winners round-trip) and that `llm_signal_flipped` closes
    # positions already ~5.6% DOWN (the LLM flip fires too LATE), while the mechanical
    # exit signals carry POSITIVE exit-IC (money_flow +0.21, max_pain +0.34). Two
    # monitor-level exits address both, leaning on the mechanical side — they only
    # fire when the LLM exit logic (_evaluate_decay) did NOT already close.
    #  (1) Trailing profit-capture: once a position's MFE (cost-adjusted peak return)
    #      arms past trailing_arm_pct, close it if it has given back ≥
    #      trailing_give_back_frac of that peak (exit_reason "trailing_stop") — locks a
    #      consistent fraction of every winner instead of round-tripping it.
    enable_trailing_exit: bool = True
    trailing_arm_pct: float = 3.0              # arm the trail once MFE ≥ this (cost-adj %)
    trailing_give_back_frac: float = 0.5       # close after giving back this fraction of the peak MFE
    #  (2) Mechanical-consensus exit: close when the raw SIGNAL methods' exit
    #      consensus (`exit_method_consensus`: money_flow / max_pain / …; − = exit,
    #      LLM + aggregator + time-overlays EXCLUDED) is confidently "exit"
    #      (≤ −mechanical_exit_threshold), independent of the LLM review (exit_reason
    #      "mechanical_exit") — times the exit off positive-exit-IC signals rather
    #      than waiting for the late LLM flip.
    enable_mechanical_exit: bool = True
    mechanical_exit_threshold: float = 0.35    # consensus ≤ −this → mechanical exit

    # ── End-of-day maintenance (2026-07-04) — scalability for the weeks ahead ──
    # Once per market day, at/after eod_maintenance_time ET (robust to missed
    # slots: fires on the first poll past the trigger), the scheduler (a) WARMS
    # the forward-return OHLCV cache for every learning-panel ticker — the fuel
    # for the IC panels / policy evals / calibrations, which otherwise starve on
    # a stale cache (observed 2026-07-03) — and (b) runs table RETENTION.
    enable_eod_maintenance: bool = True
    eod_maintenance_time: str = "16:20"        # ET; after the 16:00 close tick settles
    eod_cache_warm_days: int = 120             # panel lookback to warm
    eod_cache_warm_max_tickers: int = 0        # 0 = all panel tickers
    # Retention: simulated_trades is a derived long-format reshape of `signals`
    # (~25×/row) growing ~130k rows/day. Keep a recent RAW window (the entry-
    # event detector needs its intraday sequence), collapse older data to the
    # deduped last-per-(day,ticker,method) the analysis reads (behavior-neutral
    # over the old window), and hard-delete beyond the keep window. exit_signals
    # is age-pruned only; `signals`/`trade_reviews` (source/primary) are left to
    # a generous prune. Set enable_sim_retention=false to keep everything.
    enable_sim_retention: bool = True
    sim_retention_raw_days: int = 14           # keep full intraday resolution this recent
                                               # (tune down to ~7 for a tighter bound)
    sim_retention_keep_days: int = 150         # hard-delete simulated_trades beyond this
    exit_signals_keep_days: int = 150          # hard-delete exit_signals beyond this
    # ── Unified expected-edge sizing (2026-07-03) ────────────────────────
    # The learned successor to the hand-shaped conviction tiers: a ridge model
    # of REALIZED returns on standardized entry features (breadth · confidence
    # · combined_score · news · momentum · off-RTH, direction-oriented) whose
    # say over the final size grows with closed-trade evidence —
    #   final = tier chain × ((1−w)·prior + w·edge_mult)/prior,
    #   w = n_closed/(n_closed + edge_prior_n), 0 below edge_min_closed —
    # so today it nudges (~15% weight) and takes over only as the ledger
    # earns it. Never a gate, never flips direction; bounded by
    # edge_size_span and a hard ratio clamp. See performance/edge_sizing.py.
    edge_sizing_enabled: bool = True
    edge_min_closed: int = 20              # realized closes before the model has ANY say
    edge_prior_n: int = 150                # closes for a 50/50 split with the tier prior
    edge_size_span: float = 0.25           # max ± tilt the model alone can express
    edge_ridge_lambda: float = 1.0         # ridge strength (× n, standardized features)
    # ── Predictability sizing (Tier 1) ────────────────────────────────────────
    # A per-stock "is this name's direction forecastable at a swing horizon"
    # tilt, sized up for clean-trend names and down for chop. The score blends
    # Kaufman trend efficiency + ADX (the Tier-0 predictability panel found both
    # separate a ~60%-hit clean-trend cohort from a ~50% coin-flip at 5 days).
    # SELF-CALIBRATING on the same evidence-throttled idiom as breadth sizing:
    #   mult = 1 + span_eff × clamp((score − center)/half_width, −1, +1)
    #   span_eff = predictability_size_span × clamp(d_post/edge_ref, 0, 1)
    #   d_post = (prior_n·d_prior + n_days·d_obs)/(prior_n + n_days)
    # where d_obs is the measured 5-day directional-hit gap (high vs low
    # predictability) over the UNBIASED signals panel and n_days = signal-days of
    # evidence — so it starts ~inert (heavily shrunk toward the small prior on
    # ~2 weeks of one regime) and STRENGTHENS as the panel thickens over weeks/
    # months, or fades to neutral if the edge doesn't hold. Never a gate (every
    # name still trades — we keep accumulating outcomes), never inverts the sign.
    # See performance/predictability_sizing.py.
    enable_predictability_sizing: bool = True
    predictability_size_span: float = 0.12      # max ± size tilt at full evidence
    predictability_er_weight: float = 0.5       # Kaufman efficiency-ratio weight in the score
    predictability_adx_weight: float = 0.5      # ADX weight in the score
    predictability_adx_cap: float = 40.0        # ADX value that maps to a full 1.0 score component
    predictability_er_window: int = 20          # efficiency-ratio lookback (sessions)
    predictability_adx_period: int = 14         # Wilder ADX period
    predictability_horizon: int = 5             # swing horizon the edge is measured at (sessions)
    predictability_halfwidth_floor: float = 0.10   # min ramp half-width (guards a degenerate spread)
    predictability_edge_prior: float = 0.02     # documented prior hit-gap (heavily shrunk early)
    predictability_edge_prior_n: int = 40       # signal-DAYS for the prior to fade (≈2 months)
    predictability_edge_ref: float = 0.10       # hit-gap that counts as FULL strength (edge=1)
    predictability_cal_days: int = 60           # panel window used to calibrate (recent regime)
    predictability_cal_min_rows: int = 60       # min panel rows before the edge is measured
    predictability_cal_ttl_seconds: int = 21600  # calibration cache TTL (6h; changes ~daily)
    # ── Trend-predictability METHODS (Kaufman/ADX × trend context) ─────────────
    # The signed Kaufman efficiency ratio + ADX·DMI, expressed as four methods by
    # trend CONTEXT — kaufman_long / kaufman_short = uptrend / downtrend context,
    # same for adx_* — each active only in its context. They fold into
    # combined_score (entry AND exit scores) as an additive overlay OUTSIDE the
    # normalised weight pool (sparse/one-sided context, so pooling would dampen
    # non-trending names — same idiom as the fundamental/corp-action f_* factors);
    # tracked per-method in the signals panel + IC table + trade attribution.
    #
    # Each method's raw trend signal is multiplied by a LEARNED orientation ∈
    # [−1,+1]: +1 predicts CONTINUATION (with the trend), −1 predicts REVERSAL
    # (against it), magnitude = confidence. The orientation is measured per method
    # from the signals panel — how often that trend context has CONTINUED vs
    # reversed at the swing horizon — shrunk toward a CONTINUATION prior (+1) by
    # signal-days, so each method starts as continuation and flips toward reversal
    # only as the forward returns confirm it (e.g. it would learn "clean downtrends
    # bounce" and predict the bounce). See signals/trend_predictability.py.
    enable_trend_predictability_methods: bool = True
    trend_method_weight: float = 0.10        # additive-overlay weight on the 4 oriented trend scores
    trend_orientation_prior_n: int = 25      # signal-days of the +1 continuation prior (shrinkage)
    trend_orientation_cal_days: int = 60     # panel window used to calibrate (recent regime)
    trend_orientation_cal_min_rows: int = 40  # min active rows before a method's orientation is measured
    trend_orientation_cal_ttl_seconds: int = 21600  # orientation cache TTL (6h; changes ~daily)
    # The flat real-fill override is measured from LIQUID LMT fills; applying it
    # to instruments far outside that basis grossly understates their cost (a
    # $0.05 warrant with a ~35%-wide book was being charged 8 bp — observed
    # ARQQW 2026-07-01). Legs priced below this floor keep the modeled
    # price-tiered half-spread + commission instead of the flat override.
    sim_real_fill_min_price: float = 1.0
    # ── Scheduler outage alerting ────────────────────────────────────────
    # Email an alert when the poll loop discovers it slept through tick slots
    # (machine suspended — observed 2026-06-30: a full trading day dark with 24
    # open positions and no notification). Uses the normal SMTP settings; off
    # when email is unconfigured.
    scheduler_alert_email: bool = True
    # After a missed slot, run ONE catch-up tick immediately when a trading
    # session (RTH or extended) is still live — managing positions late beats
    # not at all. The catch-up never emails the report.
    scheduler_catchup_tick: bool = True

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
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
