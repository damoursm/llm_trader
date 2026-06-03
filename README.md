# LLM Trader

An AI-powered stock analysis system that aggregates dozens of free data sources — news, technicals, insider trades, options flow, SEC filings, macro indicators, breadth signals, and alternative data — weights them with a configurable signal aggregator, and feeds the combined picture to Claude for final BUY/SELL/HOLD/WATCH recommendations with explicit time horizons.

---

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│  0.  Ticker Discovery     — trending tickers beyond the static watchlist │
│  1.  News Fetch           — RSS feeds + NewsAPI (last 24 h)              │
│  1b. SEC 8-K Filings      — material events: earnings, M&A, restatements │
│  1c. Google Trends        — search interest spikes (retail attention)    │
│  1d. Reddit Sentiment     — WSB / r/stocks / r/investing mention counts  │
│  1e. Analyst Ratings      — upgrades, downgrades, price-target changes   │
│  1f. EPS Surprises        — recent beat/miss history from yfinance       │
│  1g. Short Interest       — FINRA Reg SHO short volume + squeeze signals │
│  2.  Market Data          — price snapshots via yfinance                 │
│  3a. Insider Trades       — politician disclosures + EDGAR Form 4        │
│  3b. Options Flow         — unusual call/put sweep detection             │
│  3c. SEC Filings          — 13D/13G activist, Form 144, 13F positions    │
│  3d. FRED Macro           — yield curve, CPI, credit spreads, M2        │
│  3e. CFTC COT             — weekly futures speculator positioning         │
│  3f. IPO Pipeline         — S-1/S-11 sector demand signal                │
│  3g. VIX & Term Structure — ^VIX/^VXN/^VVIX/^VIX3M/^VXMT regime        │
│  3h. Put/Call Ratio       — CBOE equity P/C + per-ticker bias            │
│  3i. Earnings Calendar    — upcoming reports + IV-warning detection      │
│  3j. Credit Market        — HYG vs SPY divergence (leading indicator)    │
│  3k. Market Breadth       — % of sector ETFs above 200d SMA             │
│  3l. McClellan Oscillator — NYSE A/D breadth momentum + zero-cross       │
│  3m. 52-Week Highs/Lows   — HL spread divergence (1-2 week lead)        │
│  3n. Macro Surprise Index — CESI-style: FRED actuals vs trailing avg     │
│  3o. Fed Rate Expectations— T-bill spread proxy for CME FedWatch        │
│  3p. Revision Momentum    — analyst PT/rating trend: 30d vs 31-60d      │
│  3q. Earnings Whisper     — implied whisper = consensus × (1+avg_beat%)  │
│  3r. Insider Cluster      — ≥3 different insiders buying within 5 days  │
│  3A. Pattern Recognition  — 8 classical chart patterns + per-ticker      │
│                              historical win rates (2y library, 7d cache) │
│  3B. Sector Rotation      — "Ebb and Flow": relative momentum vs SPY     │
│                              across 11 GICS sectors; cyclical/defensive  │
│  3C. Rotation Drivers     — rate-cycle phase from DFF + CPI trajectory;  │
│                              favoured/avoid assets per HIKING→EASING cycle│
│  3D. Business Cycle Rotation — Fidelity-style economic phase → sector    │
│                              leadership biases (no new API calls)         │
│  3E. Price Momentum       — perceived-value trend: 1m/2m returns vs own  │
│                              252-bar history; volume-confirmed tanh score  │
│  3F. Money Flow           — MFI + CMF + OBV slope composite:             │
│                              accumulation vs distribution signal           │
│  3G. PEAD                 — Post-Earnings Announcement Drift: SUE × time-│
│                              decay; 40-year-replicated drift anomaly        │
│  3H. Cross-Sectional Rank — per-method universe z-scores averaged;        │
│                              relative-value overlay robust to regime bias  │
│  3I. IV Rank + Directional — 21d RV percentile (IV-Rank proxy) × 5d ret/  │
│                              ATR; contrarian at high IR, trend-follow at low│
│  3J. IV Expression        — real options-chain IV × OI skew; cheap+skew → │
│                              confirm, expensive+skew → fade premium         │
│  3K. Cointegration Pairs  — Engle-Granger ADF on log-price spread;       │
│                             market-neutral stat-arb mean-reversion       │
│  3L. Sentiment Velocity   — Δ news tone (recent − prior); rate-of-       │
│                             change leads 1–5 day moves over the level    │
│  3M. Trend Strength       — ADX/DMI trend quality + Donchian 20-day      │
│                             breakout — confirmed-trend direction         │
│  4.  Signal Aggregation   — weighted combination with coherence scoring  │
│  5.  Recommendations      — Claude: BUY / SELL / HOLD / WATCH           │
│  6.  Performance Tracking — paper trades, P&L, method attribution        │
│  7.  Charts + Email       — HTML report + inline-chart email             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

### Step 0 — Ticker Discovery (`src/data/trending.py`)

Builds a **dynamic ticker universe** by merging several sources before any news is fetched:

| Source | Method |
|---|---|
| Yahoo Finance trending | Public JSON endpoint — top 20 tickers currently trending |
| Alpha Vantage movers | `TOP_GAINERS_LOSERS` — top 5 gainers, losers, most-traded |
| NewsAPI business headlines | **Open-vocabulary** — mines `$cashtags`/capitalized tokens from top US business headlines and resolves them against the full SEC ticker universe; returns tickers in ≥2 articles |
| Alpha Vantage news feed | `NEWS_SENTIMENT` — tickers in the most news items today |
| WSB cashtag discovery | Reddit **public JSON** (`hot`/`rising` for r/wallstreetbets, r/stocks, r/investing — no key); most-mentioned valid tickers (≥`WSB_DISCOVERY_MIN_MENTIONS`, default 3) |
| StockTwits trending | `trending/symbols.json` — social-momentum symbols. **Off by default**: the public API is Cloudflare-gated (403 without a browser/token) |

**Open-vocabulary resolution (`src/data/ticker_extract.py`):** instead of a hardcoded allowlist, headline/social text is scanned for `$cashtags` and capitalized tokens and validated against the ~10k-symbol SEC reference map (`company_tickers.json`, loaded once by `eight_k.py`). `$cashtags` are accepted on validity alone; bare ALL-CAPS tokens must also clear a stopword set (common English words + finance/WSB acronyms that collide with real tickers — ALL, ARE, ON, CASH, OPEN, CEO, GDP, …). This lets discovery surface **any** listed name (e.g., RKLB, ASTS, NBIS, QBTS), not just mega-caps.

Up to 30 discovered tickers are appended after the static `STOCK_WATCHLIST` and `SECTOR_ETFS`. **Pinned commodities** (`COMMODITY_ETFS`) are always included regardless of trending. All discovery sources are fail-graceful (a source erroring out never blocks the run). Flags: `ENABLE_WSB_DISCOVERY` (default on), `ENABLE_STOCKTWITS_DISCOVERY` (default off).

---

### Step 0b — Opportunity Screener (`src/data/screener.py`)

When `ENABLE_OPPORTUNITY_SCREENER=true`, runs a **proactive** technical scan that shifts discovery from "what's trending" to "what's *set up*". It screens a broad, liquid universe — a curated large/mega-cap + ETF list **∪ every ticker already in the OHLCV cache** — **cache-first** (works offline once warm; a bounded `SCREEN_MAX_FETCH_PER_RUN` warm-up fetch primes the cache over a few runs).

**Four screens** (a name surfaces if it fires ≥1):

| Screen | Logic |
|---|---|
| Unusual volume | today's volume ÷ trailing 20-day avg ≥ `SCREEN_VOLUME_RATIO` (2.0) — often *precedes* news |
| 52-week range | new 52-week high (breakout) / within `SCREEN_NEAR_HIGH_PCT` of it / new 52-week low (reversal watch) |
| Relative strength | trailing `SCREEN_RS_LOOKBACK_DAYS` (63) return minus SPY's ≥ `SCREEN_RS_THRESHOLD_PCT` (±10pp) |
| Golden / death cross | 50-day SMA crossing the 200-day within the last `SCREEN_CROSS_LOOKBACK` (5) bars |

A **liquidity gate** (`SCREEN_MIN_PRICE` $5 + `SCREEN_MIN_DOLLAR_VOLUME` $20M avg dollar volume) keeps illiquid micro-pumps out, so widening the funnel doesn't lower quality. The top `SCREEN_MAX_RESULTS` (20) setups — ranked by screen confluence, then relative strength, then volume surge — are **injected into the analysis universe**, where each then receives the full signal stack. This pairs directly with **Trend Strength**: a screener 52-week/Donchian breakout is both the discovery trigger *and* a scoring signal. Surfaced in the email's **Opportunity Screener** section (ticker · bias · setups). Fail-graceful; disable with `ENABLE_OPPORTUNITY_SCREENER=false`.

---

### Step 0c — Macro → Discovery Loop (`src/data/macro_discovery.py`)

When `ENABLE_MACRO_DISCOVERY=true`, closes the loop between the macro/regime modules and stock selection. The pipeline already identifies *favored regimes/sectors* — but discovery used to ignore them. This auto-pulls the **top holdings of the favored sector/factor ETFs** into the analysis universe, biasing it toward where macro money is flowing. (Executes after the parallel fetch, once the macro contexts exist, and before cointegration/`build_signals` so injected names receive the full signal stack.)

**Favored ETFs** are drawn from all three macro modules:

| Source | Favored ETFs |
|---|---|
| Sector rotation | `top_inflow` SPDR sectors (real-time cross-sector money flow) |
| Business cycle | `top_cycle_leaders` (structural phase leadership) |
| DIX regime | factor tilt — bullish (accumulation) → `MTUM` (momentum); bearish (distribution) → `USMV` (low-vol/defensive) |

Top `MACRO_DISCOVERY_HOLDINGS_PER_ETF` (8) holdings of each favored ETF are fetched from **yfinance `funds_data`** (cached daily in `cache/etf_holdings_*.json`, with a static SPDR fallback), deduplicated, capped at `MACRO_DISCOVERY_MAX` (25), and injected into the universe — each then scored by the full signal stack. Surfaced in the email's **Macro → Discovery** section (favored ETF · why · holdings pulled). Fail-graceful; disable with `ENABLE_MACRO_DISCOVERY=false`.

---

### Step 0d — Catalyst & Relationship Expansion (Section E)

Four independent, fail-graceful sources that widen the universe along axes the static watchlist misses. Catalyst and factor names are injected at **Step 0**; cointegration peers at **Step 3.9**. Every injected name then receives the full per-ticker signal stack.

**Market-wide earnings discovery** (`earnings.discover_earnings_tickers`) — an imminent earnings date is a binary catalyst worth scoring even for off-watchlist names. Reads the **whole-market** Alpha Vantage `EARNINGS_CALENDAR` feed (yfinance has no market-wide calendar), injects names reporting within `EARNINGS_DISCOVERY_WINDOW_DAYS` (7), validated against the SEC ticker universe, capped at `EARNINGS_DISCOVERY_MAX` (15), cached daily. **Requires `ALPHA_VANTAGE_KEY`** — skipped without it. `ENABLE_EARNINGS_DISCOVERY`.

**Market-wide analyst discovery** (`analyst_ratings.discover_analyst_tickers`) — yfinance `upgrades_downgrades` is per-ticker only, so market-wide discovery uses the **Financial Modeling Prep** `upgrades-downgrades-rss-feed` (all symbols, newest-first, paginated). Injects names with an upgrade/downgrade from ≥ `ANALYST_DISCOVERY_MIN_FIRMS` (1) distinct firms within `ANALYST_DISCOVERY_LOOKBACK_DAYS` (3), SEC-validated, ranked by firm count then recency, capped at `ANALYST_DISCOVERY_MAX` (15), cached daily. **Requires a new `FMP_API_KEY`** ([free tier](https://site.financialmodelingprep.com/developer/docs)) — skipped without it. `ENABLE_ANALYST_DISCOVERY`.

Both catalyst sources inject at Step 0, so the existing per-ticker enrichment — analyst ratings (1e), EPS surprises (1f), earnings calendar (3i) — and the full signal stack pick the new names up automatically; no separate scoring path is needed.

**Cointegration peer-expansion** (`cointegration.get_coint_peer_tickers`) — a tradeable cointegrated pair implies a directional view on *both* legs, but if only one leg is in the universe the relationship is half-expressed. After `find_cointegrated_pairs` runs, any tradeable (ENTRY/STRETCHED) pair with **exactly one** leg in the universe has its partner injected (capped at `COINT_PEER_MAX`, 10) before `build_signals`. The partner already carries a cointegration score, so once in the universe it gets a full `TickerSignal` and can become a recommendation. `ENABLE_COINT_PEER_DISCOVERY`.

**Richer factor/thematic ETF universe** — `FACTOR_ETFS` (style factors: `MTUM`, `QUAL`, `VLUE`, `SIZE`, `USMV`, `IWF`, `IWD`) and `THEMATIC_ETFS` (`SMH`, `IGV`, `XBI`, `ITA`, `TAN`, `LIT`, `IBB`, `XHB`, `JETS`, `KRE`) are pinned every run like commodities, broadening coverage beyond the 11 GICS sectors. `ENABLE_FACTOR_ETFS`.

---

### Step 0e — Discovery Liquidity Gate (Section F)

Steps 0–0d widen the discovery funnel; this applies one **uniform quality floor** to every *discovered* candidate before it enters the analysis universe, so breadth doesn't translate into untradeable microcaps — the names the tracker's price-tiered bid-ask model (`_dynamic_half_spread`) charges up to **250 bp a side**, which silently erodes realised P&L.

`liquidity.apply_liquidity_gate()` keeps a candidate only when its **last close ≥ `DISCOVERY_MIN_PRICE`** ($5) **and** its **20-day average dollar volume ≥ `DISCOVERY_MIN_DOLLAR_VOLUME`** ($20M). Data is loaded **cache-first** with a bounded warm-up fetch (`DISCOVERY_GATE_MAX_FETCH`, a single shared budget per run); a name whose liquidity cannot be verified is **dropped (fail-closed)** — the base watchlist is never affected, and a genuinely liquid name re-appears next run once its OHLCV cache is warm.

**Never gated** (pinned/intentional): the static watchlist, sector ETFs, commodities, factor/thematic ETFs, and open-trade tickers. The gate runs as one pass at the end of Step 0 (covering trending, market-wide earnings/analyst catalysts, and cluster-watch injections) and again at the macro→discovery (Step 3.85) and cointegration-peer (Step 3.9) injection points. The opportunity screener (Step 0b) applies the same price/dollar-volume floor internally. Disable with `ENABLE_DISCOVERY_LIQUIDITY_GATE=false`.

---

### Step 1 — News Fetch (`src/data/news_fetcher.py`)

Pulls articles from two layers and deduplicates by URL, filtered to the last 24 hours. Cached hourly.

**Layer A — RSS feeds (no key required)**

| Feed | Coverage |
|---|---|
| Reuters Business | General markets and macro |
| CNBC Markets | US equity and sector news |
| MarketWatch | Real-time headlines |
| Seeking Alpha | Individual stock analysis |
| Yahoo Finance | Broad market news |
| WSJ Markets | Premium financial coverage |

**Layer B — NewsAPI targeted queries (requires `NEWSAPI_KEY`)**

Two targeted queries: the first 10 watchlist tickers joined with `OR`, and sector names for ETF context.

---

### Step 1b — SEC 8-K Material Event Filings (`src/data/eight_k.py`)

Fetches recent 8-K filings for every ticker directly from SEC EDGAR's submissions API. **Always fetched fresh** — not cached hourly, so same-day filings are never missed. No API key required.

**Why 8-Ks beat RSS:** Companies must file within 4 business days. EDGAR receives the filing before most financial outlets publish coverage.

**Material items tracked:**

| Item | Event |
|---|---|
| 1.01 | Entry into Material Definitive Agreement |
| 2.01 | Completion of Acquisition or Disposition |
| 2.02 | Results of Operations (Earnings Release) |
| 2.05 | Restructuring / Layoffs |
| 2.06 | Material Impairment |
| 3.02 | Unregistered Securities Sales (Dilution) |
| 4.02 | Non-Reliance on Prior Financials (Restatement) |
| 5.01 | Change in Control |
| 5.02 | Departure/Appointment of Principal Officers |
| 1.03 | Bankruptcy or Receivership |
| 1.05 | Material Cybersecurity Incident |
| 3.01 | Notice of Delisting |

8-K filings are converted to `NewsArticle` objects and scored by the same DeepSeek LLM as RSS articles — no separate pipeline stage needed.

---

### Step 1c — Google Trends (`src/data/google_trends.py`)

When `ENABLE_GOOGLE_TRENDS=true`, fetches relative search interest for each watchlist ticker via the `pytrends` unofficial API. Cached daily. No API key required.

**Signal logic:** Compares the latest weekly interest score to the 4-week average. A spike (current ≥ 130% of average) signals rising retail attention — often precedes breakout moves or short squeezes. A sharp drop signals fading interest. Articles describe the spike/drop intensity and are scored by the DeepSeek sentiment pipeline alongside news.

---

### Step 1d — Reddit Sentiment (`src/data/reddit_sentiment.py`)

When `ENABLE_REDDIT_SENTIMENT=true`, scans three subreddits — **r/wallstreetbets**, **r/stocks**, and **r/investing** — via the Reddit API. Cached hourly. Requires free Reddit API credentials.

**What is measured:** For each watchlist ticker, counts posts and comments mentioning the ticker in the last 24 hours, computes upvote-weighted sentiment (post score as a weight proxy), and classifies the combined signal as BULLISH, BEARISH, or NEUTRAL. Results are surfaced as `NewsArticle` objects so they flow through the same DeepSeek scoring pipeline as news.

**Why it matters:** r/wallstreetbets in particular can generate self-fulfilling retail squeezes. A significant spike in WSB mention count + positive sentiment often precedes the initial leg of a retail-driven move.

---

### Step 1e — Analyst Ratings (`src/data/analyst_ratings.py`)

When `ENABLE_ANALYST_RATINGS=true`, fetches recent upgrades, downgrades, initiations, and price-target changes from yfinance for each ticker. Cached daily. No API key required.

Each rating change is converted to a `NewsArticle` with a structured summary (e.g., "Goldman Sachs upgrades NVDA from Neutral to Buy, raises PT from $500 to $650") and scored by DeepSeek. Upgrades contribute positively; downgrades and PT cuts contribute negatively to sentiment scores.

---

### Step 1f — EPS Surprises (`src/data/earnings.py`)

When `ENABLE_EARNINGS=true`, fetches recent earnings beat/miss data from yfinance `earnings_dates`. Configurable lookback (default: 90 days). Cached daily.

Beat/miss records are surfaced as `NewsArticle` objects. A beat of >10% is a strong positive catalyst; a miss of >10% is a strong negative one. These are combined with news articles before DeepSeek scoring, so an earnings beat in the recent past raises the ticker's sentiment score even without current news coverage.

---

### Step 1g — Short Interest (`src/data/short_interest.py`)

When `ENABLE_SHORT_INTEREST=true`, combines two free sources to detect squeeze setups, bearish positioning builds, and short-covering signals. Cached daily.

| Source | Data |
|---|---|
| FINRA Reg SHO daily short volume | Published daily by FINRA; measures what fraction of each day's volume is short sales |
| yfinance `short_interest` / `info` | Outstanding short interest, shares float, days-to-cover ratio |

**Signal categories surfaced as articles:**

- **Squeeze setup** — short interest >20% of float AND days-to-cover <3: heavily shorted but low covering time → explosive upside risk on any positive catalyst
- **Bearish positioning** — short interest >15% of float: institutions are paying significant borrow cost to maintain the short → strong directional conviction
- **Short covering** — recent short interest declining significantly: shorts are closing → removes structural selling pressure, often precedes price recovery

---

### Step 2 — Market Data (`src/data/market_data.py`)

Price snapshots for every ticker via yfinance, with four-level cache fallback. When a 429 rate-limit is detected: exponential backoff (60s → 120s → 240s), stops after 3 consecutive failures.

**Intraday operation (no unclosed-bar look-ahead):** the pipeline runs every 30 min during market hours on **live** prices, and the daily OHLCV history used by the indicators is **completed-bars-only** — the still-forming current-session bar is dropped until the 16:00 ET close (`market_data._drop_forming_bar`). So no indicator or return calculation ever reads an unclosed daily bar; the live price is used for fills and mark-to-market, the completed daily bars for the multi-day signals (Hybrid model).

---

### Step 3a — Insider & Politician Trades (`src/data/insider_trades.py`)

When `ENABLE_INSIDER_TRADES=true`, fetches from three sources filtered to the last `INSIDER_LOOKBACK_DAYS` (default: 90 days):

| Source | Data |
|---|---|
| House Stock Watcher | US House representatives' stock disclosures |
| Senate Stock Watcher | US Senate members' stock disclosures |

Politician trades are filtered to `TRACKED_POLITICIANS`; newly discovered tickers are added to the universe automatically (congressional buys feed discovery on ≥1 occurrence when the politician list is curated, ≥2 when it's open). Corporate-insider **Form 4 buys are now captured market-wide in Step 3c** — the old watchlist-capped (first 20 tickers, non-directional) Form 4 scan here was removed.

---

### Step 3b — Unusual Options Flow (`src/data/options_flow.py`)

When `ENABLE_OPTIONS_FLOW=true`, scans near-term options chains (≤60 days) via yfinance. A contract is flagged when: volume/OI ≥ 2×, ≥1% OTM, and notional premium ≥ $25,000. Call sweeps are bullish; put sweeps are bearish. No API key required.

---

### Step 3c — SEC EDGAR Filings (`src/data/sec_filings.py`)

When `ENABLE_SEC_FILINGS=true`, three strategies from the public EDGAR API. No key required.

**Strategy 1 — SC 13D/13G (Activist & Institutional Stakes)**
Filings disclosing >5% ownership within `SEC_FILINGS_LOOKBACK_DAYS` (default: 30 days). 13D = activist (forces change), 13G = passive institutional (large accumulation).

**Strategy 2 — Form 144 (Planned Insider Sales)**
Pre-sale disclosures by officers and directors before selling restricted shares. Bearish signal — insider planning to distribute.

**Strategy 3 — Form 13F-HR (Superinvestor Quarterly Holdings)**
For each institution in `TRACKED_INSTITUTIONS` (broadened to ~10 well-known superinvestors — Buffett, Ackman, Burry, Einhorn, Loeb, Icahn, Tiger Global, Druckenmiller, … — so **consensus** emerges when multiple filers buy the same name), diffs the two most recent 13F filings quarter-over-quarter to detect new positions, exits, and significant size changes (>10%). CIKs are resolved dynamically from EDGAR. Carries a ~45-day reporting lag.

**Strategy 4 — Market-Wide Form 4 Open-Market Buys (`fetch_form4_open_market_buys`)**
A **market-wide** scan (not watchlist-capped) of recent Form 4 filings, parsing each filing's XML for **transaction code "P"** — an *open-market purchase*, the high-signal insider action (vs. grants/option exercises/sales). Emits real `corporate_insider` **purchase** records that feed the **insider cluster + persistence** detectors (Steps 3r/3r2) and universe discovery — surfacing insider accumulation *everywhere*, not just the watchlist. Open-market buys are only ~1–2% of all Form 4s, so it parses a bounded `FORM4_SCAN_MAX_FILINGS` (default 150) most-recent filings via paginated EDGAR full-text search, cached daily in `cache/form4_buys_*.json`. Enable/disable with `ENABLE_FORM4_SCAN`.

---

### Step 3d — FRED Macro Context (`src/data/fred.py`)

When `ENABLE_FRED=true`, fetches macro regime indicators from the St. Louis Fed API. Free key required.

| Series | Indicator |
|---|---|
| `T10Y2Y` | 10Y-2Y Treasury yield spread |
| `DFF` | Effective Federal Funds Rate |
| `CPIAUCSL` | CPI (YoY computed) |
| `UNRATE` | Unemployment rate + trend |
| `BAMLH0A0HYM2` | High-yield OAS credit spread |
| `BAMLC0A0CM` | Investment-grade OAS credit spread |
| `M2SL` | M2 money supply (YoY computed) |

**Derived macro regime:**

| Regime | Conditions |
|---|---|
| `RECESSION` | Inverted curve + rising unemployment |
| `LATE_CYCLE` | Inverted/flat curve + stressed/elevated credit |
| `SLOWDOWN` | Normal curve + elevated credit or rising unemployment |
| `EXPANSION` | Normal/steep curve + normal/tight credit + stable employment |

The macro regime is injected into the Claude prompt as a `<macro_context>` block. Claude uses it to calibrate conviction on all recommendations — e.g., RECESSION → avoid POSITION-horizon longs; EXPANSION → macro tailwind raises BUY conviction.

---

### Step 3e — CFTC Commitment of Traders (`src/data/cot.py`)

When `ENABLE_COT=true`, downloads the weekly CFTC COT report. No key required. Cached by ISO week.

**Contracts tracked:**

| Contract | Related tickers |
|---|---|
| Gold | GLD, IAU, GDX |
| Silver | SLV |
| Crude Oil | USO |
| Natural Gas | UNG |
| Copper | CPER |
| Platinum | PPLT |
| Palladium | PALL |
| S&P 500 E-mini | SPY |
| Nasdaq 100 E-mini | QQQ |

Net speculator position = (longs − shorts) / open interest × 100, ranked within the 52-week range to produce a percentile:

| Percentile | Signal | Direction applied |
|---|---|---|
| ≥ 80th | `EXTREME_LONG` | **BEARISH** (contrarian — crowd is overcrowded) |
| 60–79th | `BULLISH_TREND` | BULLISH (momentum) |
| 40–59th | `NEUTRAL` | NEUTRAL |
| 20–39th | `BEARISH_TREND` | BEARISH (momentum) |
| ≤ 20th | `EXTREME_SHORT` | **BULLISH** (contrarian — max-short coiled for squeeze) |

---

### Step 3f — IPO Pipeline (`src/data/ipo_pipeline.py`)

When `ENABLE_IPO_PIPELINE=true`, fetches recent S-1 and S-11 registration statements from EDGAR. Cached daily. No key required.

A cluster of S-1 filings in a sector is a revealed-preference signal that institutional capital is flowing there 4–12 weeks before it shows up in ETF flows. Claude uses this as a confirming layer — if news is already bullish on XLK and Technology has the most S-1s, that convergence raises conviction.

---

### Step 3g — VIX & Term Structure (`src/data/vix.py`)

When `ENABLE_VIX=true`, fetches six CBOE volatility indices via yfinance. Cached daily. No key required.

| Ticker | Description |
|---|---|
| `^VIX` | 30-day S&P 500 implied volatility |
| `^VXN` | 30-day Nasdaq implied volatility |
| `^VVIX` | Vol-of-vol (VIX of VIX) |
| `^VIX9D` | 9-day short-term VIX |
| `^VIX3M` | 3-month VIX |
| `^VXMT` | 6-month mid-term VIX |

**VIX regime:**

| Level | Signal | Contrarian bias |
|---|---|---|
| > 45 | `PANIC` | Very strong contrarian BUY |
| 35–45 | `EXTREME_FEAR` | Strong contrarian BUY |
| 25–35 | `HIGH` | Elevated risk; require stronger convergence |
| 20–25 | `ELEVATED` | Slight headwind; no override |
| 15–20 | `NORMAL` | Standard regime |
| 12–15 | `LOW` | Mild complacency; reduce aggressive BUYs |
| < 12 | `COMPLACENCY` | Crowd not hedging; contrarian BEARISH risk |

**Term structure (VIX3M − VIX):**
- `BACKWARDATION` (VIX > VIX3M): near-term panic spike, often marks a short-term bottom. When combined with EXTREME_FEAR, this is one of the strongest contrarian BUY signals available.
- `CONTANGO`: normal regime; future uncertainty priced above current.

**VVIX > 120** signals VIX itself is oscillating wildly — extreme uncertainty; reduce confidence on all calls. **VXN − VIX > 5pt** signals disproportionate tech-sector fear — tech names with positive signals may have an oversold bounce setup.

---

### Step 3h — Put/Call Ratio (`src/data/put_call.py`)

When `ENABLE_PUT_CALL=true`, fetches the CBOE equity put/call ratio plus per-ticker options volume from yfinance. Cached daily. No key required.

**Market-wide P/C ratio (contrarian):**

| Level | Signal | Interpretation |
|---|---|---|
| < 0.60 | `EXTREME_GREED` | Too many calls → crowd over-bullish → contrarian BEARISH |
| 0.60–0.80 | `GREED` | Mild complacency |
| 0.80–1.00 | `NEUTRAL` | Balanced |
| 1.00–1.20 | `FEAR` | Elevated hedging → mild contrarian BULLISH |
| > 1.20 | `EXTREME_FEAR` | Panic hedging → strong contrarian BULLISH |

**Per-ticker directional signal:** For each watchlist ticker, put volume and call volume are compared. EXTREME_PUTS / PUTS_HEAVY = institutional bearish positioning; CALLS_HEAVY / EXTREME_CALLS = institutional bullish positioning. Per-ticker P/C is directional (not contrarian) — it follows the positioning of informed participants.

The per-ticker put/call score feeds directly into the signal aggregator as a fourth method (`put_call` weight = 15%) alongside news, technical, and insider.

---

### Step 3i — Earnings Calendar (`src/data/earnings.py`)

When `ENABLE_EARNINGS=true`, fetches upcoming earnings dates for every watchlist ticker from yfinance. Configurable lookahead (default: 14 days). Cached daily.

Claude uses this to apply earnings-event caution: imminent reporters (≤3 days) get capped at SWING time horizon; this-week reporters (4–7 days) get confidence capped at 0.85. Pre-earnings IV expansion risk is flagged in the rationale.

---

### Step 3j — Credit Market (`src/data/credit.py`)

When `ENABLE_CREDIT=true`, fetches HYG (high-yield bond ETF) and SPY (S&P 500) 5-day returns from yfinance and computes their divergence. Cached daily. No key required.

**Why it matters:** High-yield bonds lead equities by 1–3 days. When credit spreads widen (HYG underperforms SPY), institutional credit desks are repricing default risk before equity markets catch up.

| Divergence (HYG − SPY, 5d) | Signal | Direction |
|---|---|---|
| < −3.0% | `CREDIT_STRESS` | BEARISH — equity weakness likely in 1-3 days |
| < −1.5% | `CREDIT_CAUTION` | BEARISH — mild warning |
| −1.5% to +1.5% | `NEUTRAL` | — |
| > +1.5% | `CREDIT_STRONG` | BULLISH — credit leading equities higher |
| > +3.0% | `CREDIT_SURGE` | BULLISH — strong risk-on confirmation |

---

### Step 3k — Market Breadth (`src/data/breadth.py`)

When `ENABLE_BREADTH=true`, checks what fraction of the 11 S&P 500 sector ETFs are trading above their 200-day SMA. Cached daily. No key required.

| Reading | Signal | Interpretation |
|---|---|---|
| ≥ 85% | `BREADTH_EXTENDED` | Contrarian BEARISH — market over-extended |
| 70–84% | `BREADTH_HEALTHY` | BULLISH — broad participation confirms trend |
| 50–69% | `BREADTH_MIXED` | NEUTRAL — stock-picking environment |
| 30–49% | `BREADTH_WEAK` | BEARISH — more sectors below 200d than above |
| < 30% | `BREADTH_COLLAPSE` | BEARISH; rising 8+ pp from this level = **breadth thrust** |

**Breadth thrust:** A rise of ≥8pp from a sub-35% reading is one of the highest-conviction multi-month bullish setups historically — it signals forced selling has exhausted and institutional money is returning.

---

### Step 3l — McClellan Oscillator (`src/data/mcclellan.py`)

When `ENABLE_MCCLELLAN=true`, fetches the NYSE Advance/Decline series (`^NYAD`) from yfinance and computes the McClellan Oscillator and Summation Index. Cached daily. No key required.

**Oscillator = EMA19 − EMA39 of daily net advances** (advances minus declines):

| Reading | Signal |
|---|---|
| > +100 | `OVERBOUGHT` — contrarian BEARISH; breadth exhaustion near |
| +50 to +100 | `BULLISH_MOMENTUM` — breadth accelerating |
| −50 to +50 | `NEUTRAL` |
| −50 to −100 | `BEARISH_MOMENTUM` — breadth decelerating |
| < −100 | `OVERSOLD` — contrarian BULLISH; coiling for reversal |

**Zero-line crossings** (oscillator crossing above/below zero) are the highest-reliability swing-timing signals from the McClellan. A bullish cross = EMA19 crossed above EMA39 = momentum shifting positive.

**Summation Index** (running cumulative total): SI > 0 = bull trend; SI < 0 = bear trend; |SI| > 500 = trend overstretched.

---

### Step 3m — New 52-Week Highs/Lows (`src/data/highs_lows.py`)

When `ENABLE_HIGHS_LOWS=true`, checks how many tickers across the sector ETFs and watchlist are within 5% of their 52-week high or low. Cached daily. No key required.

**HL Spread = %near_highs − %near_lows** (range: −100 to +100):

| Spread | Signal |
|---|---|
| ≥ +50 | `STRONG_HIGHS` — BULLISH |
| +20 to +50 | `HIGHS_DOMINATE` — BULLISH |
| −20 to +20 | `BALANCED` — NEUTRAL |
| −50 to −20 | `LOWS_DOMINATE` — BEARISH |
| ≤ −50 | `STRONG_LOWS` — BEARISH |

**Divergence signals (lead reversals by 1–2 weeks):**
- **Bearish divergence:** SPY near 52-week high but HL spread declining → rally led by fewer names → distribution
- **Bullish divergence:** SPY near 52-week low but HL spread rising → lows contracting → capitulation exhaustion

---

### Step 3n — Macro Surprise Index (`src/data/macro_surprise.py`)

When `ENABLE_MACRO_SURPRISE=true`, computes a CESI-style (Citi Economic Surprise Index) composite from six FRED indicators. Requires FRED API key. Cached daily.

For each indicator, compares the most recent reading to the trailing 3-period average. The sign-adjusted z-score represents whether the economy is beating or missing recent trend expectations.

| Indicator | FRED Series | Unit | Sign flip? |
|---|---|---|---|
| Nonfarm Payrolls | `PAYEMS` | k jobs | No |
| CPI MoM | `CPIAUCSL` | % | Yes (higher = negative) |
| Retail Sales MoM | `RSAFS` | % | No |
| Industrial Production | `INDPRO` | index | No |
| ISM Manufacturing PMI | `MANEMP` | index | No |
| Unemployment Rate | `UNRATE` | % | Yes (higher = negative) |

**Composite score (weighted average of z-scores, clipped to [−1, +1]):**

| Score | Signal | Interpretation |
|---|---|---|
| > +0.40 | `STRONG_BEAT` | Economy accelerating above trend → cyclical tailwind |
| +0.15 to +0.40 | `MILD_BEAT` | Modest positive momentum |
| ±0.15 | `NEUTRAL` | In line with trend |
| −0.40 to −0.15 | `MILD_MISS` | Modest negative momentum |
| < −0.40 | `STRONG_MISS` | Economy decelerating → defensive bias |

---

### Step 3o — Fed Rate Expectations (`src/data/fedwatch.py`)

When `ENABLE_FEDWATCH=true`, derives market-implied rate-change expectations from T-bill spreads. Requires FRED API key. Cached daily.

**Method:** The spread between the current Fed Funds target midpoint and 3m/6m/12m T-bill rates measures what the bond market expects the average Fed Funds rate to be over each horizon. Positive spread = cuts priced in; negative spread = hikes priced in.

Per-meeting probabilities for the next FOMC meeting are estimated from the 3m T-bill spread using a logistic mapping calibrated to historical FedWatch data.

| 12m implied cuts | Signal |
|---|---|
| ≥ +75bp | `STRONGLY_DOVISH` — 3+ cuts priced in; major equity tailwind |
| ≥ +25bp | `DOVISH` — 1-3 cuts priced in |
| ≥ +8bp | `MILDLY_DOVISH` |
| ±8bp | `NEUTRAL` |
| ≤ −8bp | `MILDLY_HAWKISH` |
| ≤ −25bp | `HAWKISH` — hikes priced in; headwind for growth/tech |
| ≤ −75bp | `STRONGLY_HAWKISH` — major tightening |

**Week-over-week T-bill trend** detects DOVISH_SHIFT or HAWKISH_SHIFT, which are impulse signals for rate-sensitive sectors (tech, REITs, financials).

---

### Step 3p — Estimate Revision Momentum (`src/data/revision_momentum.py`)

When `ENABLE_REVISION_MOMENTUM=true`, compares analyst activity across two 30-day windows: recent (0-30d) vs prior (31-60d). Uses yfinance `upgrades_downgrades`. Cached daily. No key required.

**Momentum score formula:**
```
bull_recent = recent_upgrades + recent_pt_raises
bear_recent = recent_downgrades + recent_pt_cuts
bull_prior  = prior_upgrades + prior_pt_raises
bear_prior  = prior_downgrades + prior_pt_cuts

momentum = (bull_recent − bear_recent − (bull_prior − bear_prior)) / max(1, total_activity) × 3
```
Clipped to [−1, +1]. Score ≥ +0.25 = IMPROVING; ≤ −0.25 = DETERIORATING.

**Why it matters:** The trend in analyst revisions over 30 days matters more than any single upgrade. A stream of rising price targets = earnings momentum factor. Accelerating downgrades = deteriorating earnings visibility. Claude uses this to apply a mild ±0.03 confidence adjustment on tickers in active revision cycles.

---

### Step 3q — Earnings Whisper (`src/data/earnings_whisper.py`)

When `ENABLE_EARNINGS_WHISPER=true`, derives an implied "whisper number" from three free yfinance sources. Cached daily. No key required.

**The core insight:** The market prices the whisper, not the consensus. A stock that beats the printed consensus by $0.02 but misses the whisper by $0.05 will sell off.

**Proxy construction:**
1. **Historical beat rate** (`earnings_dates`): how often has the company beaten consensus in the last 4–8 quarters?
2. **Average EPS surprise %**: how large are the typical beats?
3. **Consensus revision trend** (`eps_trend`): is the estimate being revised up or down over 7d/30d?
4. **Net analyst revisions** (`eps_revisions`): how many analysts raised vs cut estimates in the last 30 days?

**Implied whisper = consensus × (1 + avg_historical_surprise_pct / 100)**

Companies with a consistent beat pattern have the beat "baked in" to market expectations — so the implied whisper is the true bar.

**Signal classification:**

| Signal | Conditions |
|---|---|
| `BEAT_LIKELY` | Beat rate ≥75% + avg_surprise ≥3% + estimate not being revised down |
| `BEAT_POSSIBLE` | Beat rate ≥60% OR avg_surprise ≥1.5% OR consensus REVISING_UP |
| `NEUTRAL` | Mixed signals or insufficient data |
| `MISS_POSSIBLE` | Beat rate <45% OR avg_surprise ≤−1.0% OR consensus REVISING_DOWN |
| `MISS_LIKELY` | Beat rate <30% OR avg_surprise ≤−3.0% |

---

### Step 3s — OpEx Calendar (`src/data/opex.py`)

When `ENABLE_OPEX=true`, computes options expiration week context from pure date arithmetic. **Zero API calls, zero network I/O** — always runs synchronously after the parallel data fetch completes.

**What it detects:**

| Signal | Condition | Max pain gravity |
|---|---|---|
| `OPEX_DAY` | Today is the 3rd Friday | PEAK — strongest pin force |
| `OPEX_IMMINENT` | OpEx is tomorrow | Near-peak — strong pinning pressure |
| `TRIPLE_WITCHING_WEEK` | OpEx week, quarterly (Mar/Jun/Sep/Dec) | ELEVATED — stock + index options + futures all expiring |
| `OPEX_WEEK` | Mon–Fri of the 3rd-Friday week | ELEVATED vs baseline |
| `POST_OPEX` | 1–5 calendar days after OpEx | RELEASED — pin unwinding, directional moves more reliable |
| `NEUTRAL` | >5 days from any OpEx | BASELINE — normal weighting |

**Triple Witching** occurs in March, June, September, and December when stock options, stock index futures, and stock index options all expire simultaneously. This produces significantly higher volume, larger intraday moves, and stronger pinning toward max pain than a standard monthly expiry.

**How it affects recommendations:** The OpEx context is a *timing and magnitude modifier* — it tells Claude how much to trust the `max_pain_score` signal that day, not which direction to trade:
- OPEX_WEEK / OPEX_IMMINENT → upgrade max_pain_score weight by +0.03–0.05
- POST_OPEX → discount max_pain_score (new OI cycle just starting, max pain not yet meaningful)
- NEUTRAL → standard weighting

---

### Step 3u — Bond Market Internals (`src/data/bond_internals.py`)

When `ENABLE_BOND_INTERNALS=true`, fetches 9 Treasury and credit ETF tickers via yfinance and computes five independent macro regime signals over 1–8 week horizons. No API key required. Cached daily.

**Six signals:**

| Signal | Source | What it measures |
|---|---|---|
| Yield curve shape | `^TNX` (10Y) − `^IRX` (3M) | 10Y-3M spread — best recession predictor; differs from FRED's 2Y-10Y |
| TLT momentum | `TLT` price | 1/4/8-week returns — proxy for direction and velocity of long-rate changes |
| Duration positioning | `TLT` − `IEF` 5-day spread | Long-end vs intermediate: bear steepening (inflation/fiscal) vs bull flattening |
| Real yield / inflation | `TIP` − `IEF` 5-day spread | TIPS outperforming IEF = inflation expectations rising = real rates falling |
| IG credit premium | `LQD` − `TLT` 5-day spread | Corporate spreads vs risk-free: IG_STRESS leads equity weakness by 1–5 days |
| Bond-equity divergence | `TLT` − `SPY` 5-day spread | When bonds rally hard while equities hold, equities typically catch up within 1–2 weeks |

**Composite regime:**

| Regime | Meaning for equities |
|---|---|
| `RISK_ON` | Bond market broadly constructive; mild confidence boost on BUY calls |
| `CONSTRUCTIVE` | Mild tailwind |
| `NEUTRAL` | No bond market override |
| `DEFENSIVE` | Some headwinds; mild haircut on POSITION-horizon longs |
| `RISK_OFF` | Multiple warning signals; avoid aggressive new longs |
| `REFLATIONARY` | TLT falling + inflation rising; favour commodities/cyclicals over growth tech |

**Bond-equity divergence signals:**

| Signal | Condition | Equity implication |
|---|---|---|
| `EQUITY_CATCHUP_LIKELY` | TLT 5d ≥ +2.5%, SPY 5d flat (−1.5% to +1.5%) | Bond market pricing rate cuts ahead of equities — catch-up rally expected within 1–2 weeks |
| `EQUITY_CATCHUP_POSSIBLE` | TLT 5d ≥ +1.5%, SPY 5d flat | Milder version — modest bullish lean |
| `SYNCHRONIZED_RISK_ON` | Both TLT and SPY rallying (≥+2%) | Unusual; occurs at dovish pivots — broadly constructive |
| `NEUTRAL` | No significant divergence | No actionable signal |
| `EQUITY_SELLOFF_RISK` | TLT 5d ≤ −2.0%, SPY 5d flat | Rising rate headwind not yet priced into stocks — weakness likely within 1–2 weeks |
| `SYNCHRONIZED_RISK_OFF` | TLT ≤ −1.5% and SPY ≤ −2.0% | Broad de-risking underway |

**Why this is additive to FRED:** FRED provides current *levels* (e.g., "IG spread is 1.2%"). Bond internals provides 1–8 week *price momentum and direction* (e.g., "IG spreads widened 20bp this week" from LQD/TLT relative performance). It also adds the 10Y-3M spread specifically, which is a better recession predictor than the 2Y-10Y spread FRED provides. The bond-equity divergence adds a cross-asset leading indicator: the bond market tends to price macro regime shifts 1–2 weeks before equities react. Claude instruction #25 applies all signals as a medium-term overlay.

---

### Step 3v — MOVE Index (`src/data/move.py`)

When `ENABLE_MOVE=true`, fetches the ICE BofA MOVE Index (`^MOVE`) via yfinance and computes Treasury market implied volatility signals. Falls back to `VXTLT` (CBOE 30-day Treasury ETF Volatility Index) if `^MOVE` is unavailable. No API key required. Cached daily.

**What the MOVE Index measures:**

The MOVE Index is to Treasury bonds what VIX is to S&P 500 equities. It is constructed from 1-month OTM options on 2Y, 5Y, 10Y, and 30Y Treasuries. When MOVE spikes, the bond market is pricing in significant rate uncertainty — which typically compresses risk appetite and precedes equity dislocations by **1–5 days**.

**Signal thresholds:**

| Level | Signal | Equity implication |
|---|---|---|
| < 60 | `CALM` | Unusually quiet; no signal |
| 60–80 | `LOW` | Below-average vol; mild constructive backdrop |
| 80–100 | `NORMAL` | Typical regime; no override |
| 100–120 | `ELEVATED` | Above-average; watch for equity spillover |
| 120–150 | `HIGH` | Significant stress; BEARISH for equities |
| 150–200 | `EXTREME` | Major disruption; strong BEARISH warning |
| > 200 | `PANIC` | GFC-level bond market crisis |

**Additional signals:**

- **5-day spike detection:** A jump of >20pt in 5 trading days triggers `is_spiking=True` — early warning regardless of absolute level.
- **MOVE/VIX ratio:** Normally 4–7×. When ratio exceeds 8×, bond market is pricing significantly more stress than equity market — historically resolves by equities selling off.

**Claude prompt overlay (instruction 11b):** MOVE is treated as a broad-market / regime overlay. Elevated MOVE applies a −0.05 confidence haircut on POSITION-horizon BUY calls. Spikes trigger additional caution on rate-sensitive sectors (XLK, XLRE, high-P/E growth). MOVE/VIX divergence flags that equity complacency may be temporary.

---

### Step 3v2 — Dark Pool Index (DIX) + market-wide GEX (`src/data/dix.py`)

When `ENABLE_DIX=true`, fetches the free SqueezeMetrics `DIX.csv` (`date, price, dix, gex`) via httpx — no API key required. Cache-first (`cache/dix_*.json`, daily); any feed/parse failure returns `None` and the run continues unaffected.

**What DIX measures:**

The Dark Pool Index is the dollar- and volume-weighted short volume across **off-exchange (dark pool) venues**, expressed as a fraction (~0.38–0.48). Large institutions route through dark pools to accumulate *without moving the lit market*, so a **high DIX is a proxy for hidden institutional buying** — and it is a **leading** indicator, preceding S&P price by roughly **1–4 weeks**. This is the "institutional accumulation off-exchange" signal.

**Signal classification** (percentile of current DIX vs its own trailing year; absolute-level fallback when <1y of history):

| DIX percentile | Signal | Equity implication |
|---|---|---|
| ≥ 75th | `STRONG_ACCUMULATION` | Strong hidden buying — BULLISH |
| ≥ 58th | `ACCUMULATION` | Above-average hidden buying — mild BULLISH |
| 42–58th | `NEUTRAL` | No edge |
| ≤ 42nd | `DISTRIBUTION` | Below-average hidden buying — mild BEARISH |
| ≤ 25th | `STRONG_DISTRIBUTION` | Little hidden support — BEARISH |

A 5-day `RISING`/`FALLING`/`FLAT` trend (±0.5pp threshold) confirms or warns on the level.

**Market-wide GEX (from the same feed):** the *whole-index* dealer gamma, distinct from the per-ticker dealer gamma in `gamma_exposure.py`. Low/negative GEX = `VOL_EXPANSION` (dealers amplify moves; trends and gaps run further); high GEX = `VOL_SUPPRESSION` (dealers pin price; expect a grind/mean-reversion). The classic SqueezeMetrics bull setup is **high DIX + low GEX** — "hidden buying with room to run".

**Integration (two paths, so it genuinely steers recommendations):**
- **Claude prompt** (block + instruction 11c): a market-wide overlay (never per-ticker). `STRONG_ACCUMULATION` → +0.03–0.05 confidence nudge on already-supported BULLISH BUYs; `STRONG_DISTRIBUTION` → −0.03–0.05 haircut on fresh longs. High DIX during a VIX spike is flagged as a powerful contrarian bullish tell.
- **Macro Regime Filter** (`macro_regime.py`): contributes to the composite regime score (`_DIX_SCORES`, weight 1.0 — same tier as breadth/FRED), so strong accumulation/distribution shifts the actionable confidence threshold and the BUY gate alongside VIX/MOVE/credit.

DIX is rendered in the email report (section 21b) as a gauge card (DIX %, trailing-year percentile, 5d trend, market GEX regime) with the high-DIX+low-GEX / low-DIX+low-GEX combo callouts. Disable with `ENABLE_DIX=false`.

---

### Step 3w — Global Macro Cross-Asset Regime (`src/data/global_macro.py`)

When `ENABLE_GLOBAL_MACRO=true`, fetches DXY, Copper/Gold, WTI crude oil, and TLT via yfinance and computes three independent cross-asset macro regime signals. No API key required. Cached daily.

**DXY (US Dollar Index — `DX-Y.NYB`):**

A rising dollar tightens global financial conditions: it raises the cost of USD-denominated debt for EM borrowers, depresses commodity prices (priced in USD), and compresses overseas earnings of US multinationals.

| Signal | 5-day return | Equity implication |
|---|---|---|
| `STRONG_BULL` | > +1.5% | Significant headwind for EM equities (EEM, VWO), commodities (GLD, CPER), multinationals |
| `BULL` | +0.5–1.5% | Mild headwind |
| `NEUTRAL` | ±0.5% | No signal |
| `BEAR` | −0.5 to −1.5% | Mild tailwind for commodities and EM |
| `STRONG_BEAR` | < −1.5% | Strong tailwind for commodities, EM ETFs |

**Copper/Gold ratio (`HG=F` / `GC=F`) — Dr. Copper barometer:**

Copper prices reflect global industrial demand; gold prices reflect safe-haven demand. The ratio isolates the growth vs. fear dimension without the inflation component. Signal uses 20-day % change to reduce noise.

| Signal | 20-day % change | Equity implication |
|---|---|---|
| `RISK_ON_SURGE` | > +5% | Dr. Copper strongly bullish; favour cyclicals (XLI, XLB), EM |
| `RISK_ON` | +2–5% | Mild growth-positive lean |
| `NEUTRAL` | ±2% | No directional signal |
| `RISK_OFF` | −2 to −5% | Mild contraction; cautious on cyclicals |
| `RISK_OFF_CRASH` | < −5% | Recession risk pricing; avoid cyclical longs; favour defensives/gold |

**Oil/Bonds divergence (`CL=F` vs `TLT` — 5-day co-movement):**

Oil and Treasury bonds are normally inversely correlated (oil up = inflation → bond yields rise → TLT falls). When they move in the same direction, the normal macro framework is suspended.

| Signal | Condition | Equity implication |
|---|---|---|
| `POLICY_PIVOT_SIGNAL` | Oil > +2.5% AND TLT > +1.5% (5d) | Both rallying — unusual; market pricing Fed cut despite oil; **BULLISH** for equities short-term |
| `STAGFLATION_RISK` | Oil > +2.5% AND TLT < −1.5% (5d) | Rising costs + tightening rates; **worst regime** for equities; −0.07 on all POSITION longs |
| `GROWTH_FEAR_RISK_OFF` | Oil < −2.5% AND TLT > +1.5% (5d) | Demand destruction + flight to safety; **BEARISH** cyclicals; favour gold/defensives |
| `DEFLATION_SHOCK` | Oil < −2.5% AND TLT < −1.5% (5d) | Both selling off — broad de-risking; avoid all new longs |
| `NEUTRAL` | Otherwise | No divergence signal |

**Composite regime:** DXY + Copper/Gold directions combine into `RISK_ON` / `CONSTRUCTIVE` / `NEUTRAL` / `DEFENSIVE` / `RISK_OFF`. Oil/Bond divergence is a separate, independent signal reported alongside.

**Claude prompt overlay (instruction 26):** DXY is treated as a sector-level modifier. STRONG_BULL DXY applies −0.05 haircut on EM ETFs, commodity names, and multinationals. RISK_OFF_CRASH copper/gold applies −0.08 haircut on cyclical BUY calls. STAGFLATION_RISK oil/bond applies −0.07 on all new POSITION longs. POLICY_PIVOT_SIGNAL applies +0.03 boost on rate-sensitive names. Convergence of DXY STRONG_BULL + Cu/Au RISK_OFF + STAGFLATION_RISK + elevated MOVE = maximum bearish confidence.

---

### Step 3y — Market Mode Switching (`src/data/market_mode.py`)

When `ENABLE_MARKET_MODE_SWITCHING=true` (default), classifies the market as **TRENDING**, **NEUTRAL**, or **CHOPPY** and dynamically adjusts the signal weight profile passed to `build_signals()`. **Zero API calls, zero network I/O** — pure computation from already-fetched contexts. Runs synchronously before Step 4.

**Why fixed weights underperform:**

Momentum strategies (technical analysis, news catalysts) work in low-volatility, directional markets. Mean-reversion strategies (VWAP deviation, put/call contrarian) work in high-volatility, range-bound markets. Blending both with fixed weights gives mediocre results in either regime.

**Scoring inputs:**

| Source | Weight | Trending signal | Choppy signal |
|---|---|---|---|
| VIX | 2.0× | LOW/COMPLACENCY → +1.5/+2.0 | HIGH/EXTREME_FEAR → −1.5/−2.0 |
| Market breadth | 1.5× | BREADTH_HEALTHY/EXTENDED → +1.0/+1.5 | BREADTH_WEAK/COLLAPSE → −1.0/−2.0 |
| 52w Highs/Lows | 1.0× | HIGHS_DOMINATE/STRONG_HIGHS → +0.75/+1.5 | LOWS_DOMINATE/STRONG_LOWS → −0.75/−1.5 |
| McClellan | 1.0× | BULLISH_MOMENTUM/OVERBOUGHT → +0.75/+1.5 | BEARISH_MOMENTUM/OVERSOLD → −0.75/−1.5 |

Normalised composite → **TRENDING** (> +0.5) | **NEUTRAL** (±0.5) | **CHOPPY** (< −0.5)

**Weight profiles (raw unnormalised, `_normalised_weights()` divides by total):**

| Method | TRENDING | NEUTRAL (baseline) | CHOPPY |
|---|---|---|---|
| `news` | 0.40 | 0.40 | 0.30 |
| `tech` | **0.45** ↑ | 0.30 | **0.15** ↓ |
| `insider` | 0.30 | 0.30 | 0.30 |
| `put_call` | **0.08** ↓ | 0.15 | **0.28** ↑ |
| `max_pain` | 0.12 | 0.12 | 0.12 |
| `oi_skew` | 0.15 | 0.15 | 0.15 |
| `vwap` | **0.04** ↓ | 0.12 | **0.28** ↑ |

The weight profile is passed to `build_signals()` as a `weight_profile` override on `_normalised_weights()`. NEUTRAL uses the baseline `_BASE_WEIGHTS` unchanged.

---

### Step 3x — Macro Regime Filter (`src/data/macro_regime.py`)

When `ENABLE_MACRO_REGIME_FILTER=true` (default), computes a composite top-down regime from all available macro context objects. **Zero API calls, zero network I/O** — pure computation from already-fetched contexts. Runs synchronously between Step 3 and Step 4.

**What it does:**

Reads VIX, MOVE, bond internals, global macro, FRED, breadth, and credit signals, weights them, and classifies the market into one of five regimes. Each regime gates signal entry and adjusts the minimum confidence threshold required for a signal to be marked actionable.

**Input sources and weights:**

| Source | Weight | Signal fields used |
|---|---|---|
| VIX | 2.0× | `vix_signal` (PANIC → COMPLACENCY) |
| MOVE | 2.0× | `signal` (PANIC → CALM) |
| Bond internals | 1.5× | `regime` (RISK_OFF → REFLATIONARY) |
| Global macro | 1.0× | `composite_signal` (RISK_OFF → RISK_ON) |
| FRED | 1.0× | `regime` (RECESSION → EXPANSION) |
| Market breadth | 1.0× | `signal` (BREADTH_COLLAPSE → BREADTH_EXTENDED) |
| Credit | 0.5× | `signal` (CREDIT_STRESS → CREDIT_SURGE) |

**Regime thresholds:**

| Regime | Composite score | Confidence threshold | BUY entries |
|---|---|---|---|
| `PANIC` | ≤ −1.5 (or any PANIC source) | **88%** | **BLOCKED** |
| `RISK_OFF` | −1.5 to −0.8 | **82%** | **BLOCKED** |
| `CAUTION` | −0.8 to −0.3 | 80% | Allowed |
| `NEUTRAL` | −0.3 to +0.3 | 78% (baseline) | Allowed |
| `RISK_ON` | > +0.3 | **72%** | Allowed |

**Effect on the pipeline:**

The regime is computed after Step 3 (all macro data collected) and before the actionable signal filter. It replaces the hardcoded `0.78` threshold in `pipeline.py` with a dynamic value, and optionally blocks all new BUY entries during PANIC and RISK_OFF regimes. SELL entries are always allowed (shorting into a downturn is valid). The regime and its evidence are reported in the email report (section 4) and logged at INFO level.

**When not all sources are available** (e.g. `ENABLE_FETCH_DATA=false`), only the available contexts contribute to the weighted composite. If only FRED is available, the regime reflects FRED's expansion/recession signal at full weight.

---

### Step 3z — Catalyst Timing (`src/data/catalyst_timing.py`)

When `ENABLE_CATALYST_TIMING=true` (default), applies three event-driven guards and amplifiers. **Zero additional API calls** — all inputs come from already-fetched contexts. Runs after Step 5 (recommendations ranked) and before the actionable filter.

**Mechanism 1 — Earnings Blackout**

Any ticker with an earnings report within 2 calendar days is removed from the actionable BUY/SELL set. IV crush, gap risk, and binary outcomes make directional trades around earnings highly unreliable. The blackout covers both sides: no new longs or shorts into a catalyst that erases the edge.

**Mechanism 2 — OpEx Max-Pain Amplifier**

During OpEx week (3rd Friday of each month), the `max_pain` signal weight in the aggregator is boosted from its 0.12 baseline. Triple Witching months (Mar/Jun/Sep/Dec) get the strongest boost:

| Condition | max_pain weight | Boost |
|---|---|---|
| Normal (non-OpEx week) | 0.12 | baseline |
| OpEx week | **0.20** | +67% |
| Triple Witching week | **0.28** | +133% |

This is applied as a weight_profile override in `build_signals()` before normalisation, so it combines cleanly with the market-mode weight profile.

**Mechanism 3 — 8-K + Insider Buy → WATCH Elevation**

The combination of a freshly filed SEC 8-K (material catalyst) and an insider purchase by a corporate officer or politician is among the highest-predictive pre-signal setups. When both are present for the same ticker:
- If the ticker is already in the top-10 as HOLD → upgraded to WATCH in-place
- If not yet in the top-10 → injected as a new WATCH recommendation

Volume confirmation (via TickerSignal `technical_score`/`vwap_score` > 0.10) is noted when available but is not required to trigger the elevation.

---

### Step 3t — Seasonality Calendar (`src/data/seasonality.py`)

When `ENABLE_SEASONALITY=true`, computes seasonal calendar context from pure date arithmetic. **Zero API calls, zero network I/O** — runs synchronously after the parallel fetch completes, immediately after OpEx.

**Four documented patterns with measurable statistical edge:**

| Pattern | Description |
|---|---|
| End-of-month rebalancing | Last 3 / first 3 calendar days: pension funds and 401(k) plans rebalance to target allocations — systematic equity bid |
| Quarter-end window dressing | Last 5 calendar days of Mar/Jun/Sep/Dec: fund managers buy YTD winners to improve quarterly statements |
| January effect | Days 1–15: tax-loss harvesting selling has passed; small-caps (IWM) historically outperform SPY in the first two weeks |
| Monthly historical biases | Based on ~95 years of S&P 500 data: April strongest (+2–3% avg), September weakest (−1%), Sell in May effect (May–Oct underperforms Nov–Apr by ~6–7pp annualised) |

**Calendar window flags computed:**

| Flag | Condition |
|---|---|
| `in_month_end_window` | Last 3 calendar days of any month |
| `in_month_start_window` | First 3 calendar days of any month |
| `in_quarter_end_window` | Last 5 calendar days of Mar/Jun/Sep/Dec |
| `in_quarter_start_window` | First 5 calendar days of Jan/Apr/Jul/Oct |
| `in_january_effect` | January 1–15 |
| `is_fiscal_year_end` | June or December quarter-end (more intense window dressing) |

**Composite signal** combines monthly bias score (±1) with count of active bullish/bearish calendar effects:

| Signal | Meaning |
|---|---|
| `STRONG_TAILWIND` | Total score ≥ 2: monthly bias + multiple active bullish windows |
| `TAILWIND` | Total score = 1: net seasonal advantage |
| `NEUTRAL` | Total score = 0: no seasonal edge |
| `HEADWIND` | Total score = −1: net seasonal disadvantage |
| `STRONG_HEADWIND` | Total score ≤ −2: monthly bias + multiple active bearish windows |

**How it affects recommendations:** Seasonality is a *weak secondary overlay* — it shifts probability but never overrides strong company-level catalysts. Claude instruction #24 uses it as a tie-breaker and notes seasonal headwinds/tailwinds explicitly in rationale when applicable.

---

### Step 3r — Insider Cluster Detection (`src/signals/aggregator.py`)

Computed as part of signal aggregation (no separate data fetch step). No extra data required.

**Definition:** A cluster is detected when ≥3 **different** corporate insiders or politicians independently purchase the same stock within any 5-day rolling window.

**Why clusters are more predictive than single trades:**
- A single insider buy may reflect a scheduled 10b5-1 plan, personal diversification, or option exercise.
- When 3+ different senior executives, directors, and/or politicians buy simultaneously without coordination, it is genuine independent conviction from multiple people with full business visibility.
- Historically, insider clusters precede significant positive re-ratings (earnings beats, M&A announcements, guidance raises).

**Implementation:** `_detect_insider_cluster()` scans all purchase transactions for the ticker, sorts by `transaction_date`, and finds the maximum number of distinct `trader_name` values within any 5-day anchor window. Options flow and institutional (13F) signals are excluded from cluster detection — only direct purchases count.

**Amplifier:** When a cluster is detected AND the baseline `insider_score` is positive, the score is multiplied by **1.75×** (capped at +1.0). The `TickerSignal` stores `insider_cluster_detected=True` and `insider_cluster_size=N`. Claude receives an explicit `*** INSIDER CLUSTER ***` flag in the signals block and instruction #22 explaining how to weight it.

**Cross-run persistence (`src/data/cluster_watchlist.py`):** The same-day score amplifier captures the signal on detection day, but insider clusters historically precede price movement by 5–20 days. To exploit the full lead window:

- When a cluster is first detected, the ticker is written to `cache/cluster_watchlist.json` with the detection date and cluster metadata.
- On every subsequent run within the **10-day watch window**, the ticker is injected into the analysis universe at Step 0 — so it is always re-evaluated by the full signal stack even if it has dropped off the trending/discovery list.
- After 10 days the entry expires and is removed.
- Active watchlist entries are shown in the email report (section 5z) with a progress bar showing days elapsed vs. remaining.

The watchlist survives pipeline restarts (JSON file on disk) and adds no network I/O — it is pure computation from already-built `TickerSignal` objects.

---

### Step 3r2 — Insider Buying Persistence (`src/signals/aggregator.py`)

Computed as part of signal aggregation (no separate data fetch step) — the depth counterpart to the cluster's breadth.

**Definition:** Persistence is detected when the **same** insider (by name) purchases the same stock on **multiple separate days** within the lookback window. The cluster signal measures *how many different people* bought at once; persistence measures *how repeatedly one person* keeps buying.

**Why repeated buying by one name beats a one-off:**
- A single insider buy can be a scheduled 10b5-1 plan, diversification, or an option exercise — noisy.
- The same insider buying again and again is escalating personal conviction: they keep putting more capital at risk as the thesis develops. Repeat buyers tend to be early and right, acting before the catalyst is public.

**Implementation:** `_detect_insider_persistence()` filters the ticker's corporate-insider and politician purchases (same filter as the cluster detector — options flow, 13F, and bare Form 4 filings are excluded), groups them by case-insensitive `trader_name`, and counts the number of **distinct transaction dates** per name (so a single multi-line filing on one day counts once, not as repetition). Persistence fires when the most-active buyer's distinct-day count ≥ `insider_persistence_min_buys` (default 2).

**Amplifier:** When persistence is detected AND the baseline `insider_score` is positive, the score is multiplied by `_persistence_factor(count) = min(1.75, 1.0 + 0.25 × (count − 1))` — so 2 days → 1.25×, 3 → 1.50×, 4+ → 1.75× (capped, applied after the cluster amplifier and clamped to +1.0). The `TickerSignal` stores `insider_persistence_detected=True`, `insider_persistence_count=N`, and `insider_persistence_buyer="<name>"`. Claude receives an explicit `*** INSIDER PERSISTENCE ***` flag in the signals block and instruction #22b. The email Smart Money card shows a teal `PERSISTENT • <name> ×N` badge.

Because persistence amplifies `insider_score`, it is automatically reflected in performance attribution under the existing **Smart Money / insider** method — no separate attribution key. Persistence + cluster on the same ticker is the highest-conviction insider configuration available. Disable with `ENABLE_INSIDER_PERSISTENCE=false`.

---

### Step 3A — Pattern Recognition (`src/signals/pattern_recognition.py`)

Detects 8 classical chart patterns in recent price action and converts each detection into a `[-1, +1]` signal score driven by the **ticker's own historical win rate** for that pattern type.

**Patterns detected:**

| Pattern | Inherent direction | Type |
|---|---|---|
| Double Bottom | Bullish | Reversal |
| Inverse Head & Shoulders | Bullish | Reversal |
| Ascending Triangle | Bullish | Continuation |
| Bull Flag | Bullish | Continuation |
| Double Top | Bearish | Reversal |
| Head & Shoulders | Bearish | Reversal |
| Descending Triangle | Bearish | Continuation |
| Bear Flag | Bearish | Continuation |

**Two-phase design:**

1. **Cold path (first run per ticker):** fetches 2 years of OHLCV data, scans the full history with a 40-bar sliding window (step=5), records the 5d/10d forward return for each pattern detected, and computes per-pattern win rates. Library cached for 7 days in `cache/patterns/<TICKER>.json`.
2. **Warm path (subsequent runs):** loads the cached library instantly, detects the current pattern from the last 60 price bars.

**Scoring formula:**
```
win_rate  = fraction of historical occurrences where the pattern correctly predicted its direction
edge      = (win_rate − 0.5) × 2         ∈ [−1, +1]
score     = clip(edge × inherent_direction, −1, +1)
```

A win rate of 0.75 → edge = +0.50 → score ±0.50 (moderate historical edge).  
When fewer than 3 historical occurrences exist, a weak prior (±0.25) is used instead.

**Key insight:** the score overrides the pattern's theoretical direction with actual per-ticker history. If double tops for a specific stock have historically been followed by rallies (e.g., fake-out breakdowns that reverse), the score turns bullish rather than blindly applying the bearish template.

**Base weight:** 0.18 in the aggregator (between news/insider at 0.30–0.40 and VWAP/max_pain at 0.12).

---

### Step 3B — Sector Rotation / "Ebb and Flow" (`src/data/sector_rotation.py`)

When `ENABLE_SECTOR_ROTATION=true`, computes per-sector money-flow rotation scores across the 11 SPDR sector ETFs relative to SPY. **Money acts like water** — when it floods into one sector it is usually exiting another. Cached daily. No API key required.

**Score construction:**

1. **Relative return** per period: `sector_ret − SPY_ret` for 1-week, 1-month, and 3-month windows
2. **Weighted composite:** `0.5 × rel_5d + 0.3 × rel_21d + 0.2 × rel_63d`
3. **Cross-sectional z-score:** normalises across all 11 sector peers so scores are relative, not absolute
4. **Volume modifier:** `vol_mod = min(0.25, (ratio − 1.0) × 0.6)` when 5d/20d volume ratio > 1.15 — amplifies confirmed accumulation/distribution
5. **Final score:** `clip(z / 2.0 + vol_mod, −1, +1)`

**Flow signals:**

| Score | Signal | Equity implication |
|---|---|---|
| ≥ +0.5 | `STRONG_INFLOW` | Capital actively flooding in — BULLISH |
| ≥ +0.2 | `INFLOW` | Meaningful relative inflow — BULLISH |
| −0.2 to +0.2 | `NEUTRAL` | No directional flow conviction |
| ≤ −0.2 | `OUTFLOW` | Money leaving — BEARISH |
| ≤ −0.5 | `STRONG_OUTFLOW` | Significant capital exodus — BEARISH |

**Rotation regime (cyclical vs defensive balance):**

| Regime | Condition | Meaning |
|---|---|---|
| `RISK_ON` | cyclical avg − defensive avg > +0.2 | Growth / cyclical rotation dominant |
| `NEUTRAL` | spread ±0.2 | Mixed or no clear bias |
| `RISK_OFF` | spread < −0.2 | Defensive rotation dominant |

**Cyclical sectors:** XLK, XLF, XLY, XLI, XLB, XLC, XLE  
**Defensive sectors:** XLV, XLP, XLU, XLRE

**Rotation pairs:** explicit cross-product of top outflow → top inflow sectors (e.g. `XLK → XLP  (Technology → Consumer Staples)`).

**Claude prompt overlay (instruction 27b):** Stocks in STRONG_INFLOW sectors get a mild confidence boost; stocks in STRONG_OUTFLOW sectors require stronger signals before a BUY is issued ("water already leaving this bucket"). The rotation regime is applied as a sector-level overlay: RISK_OFF rotation → favour defensive names; RISK_ON → raise conviction on cyclical BUY calls.

---

### Step 3C — Rotation Drivers (`src/data/rotation_drivers.py`)

When `ENABLE_ROTATION_DRIVERS=true`, synthesises the Federal Reserve rate trajectory and CPI inflation trend into a named rate-cycle phase and maps it to cross-asset rotation implications. Requires `FRED_API_KEY`. Cached daily.

**What's distinct from FedWatch + FRED:** FedWatch gives forward-looking market-implied expectations; FRED gives current levels. Rotation Drivers gives the **actual backward-looking trajectory** — where has the Fed been over 3/6/12 months — and combines it with the inflation trend to name the current point in the cycle.

**Rate trajectory (FRED DFF, 270 daily observations ≈ 13 months):**

| Label | Condition |
|---|---|
| `ACTIVE_HIKING` | DFF +>25bp over 12m AND +>10bp over 3m |
| `PAUSING` | DFF +>25bp over 12m AND flat last 3m (±15bp) |
| `ACTIVE_CUTTING` | DFF −>25bp over 12m AND −>10bp over 3m |
| `EASING_PAUSE` | DFF −>25bp over 12m AND flat last 3m |
| `STABLE` | no clear directional trend |

**Inflation trend (FRED CPIAUCSL, CPI YoY now vs 6 months ago):**

| Label | Condition |
|---|---|
| `ACCELERATING` | CPI YoY rose >1.0pp over 6m |
| `RISING` | CPI YoY rose 0.4–1.0pp |
| `ELEVATED_STABLE` | CPI >4% and roughly flat |
| `STABLE` | near-target, flat |
| `MODERATING` | CPI YoY fell 0.5–1.5pp |
| `DECLINING` | CPI YoY fell >1.5pp |
| `LOW_STABLE` | CPI <2.5%, flat |

**Cycle phases and equity implications:**

| Phase | Conditions | Equity direction |
|---|---|---|
| `EARLY_TIGHTENING` | ACTIVE_HIKING + real rate not yet restrictive | BEARISH |
| `PEAK_TIGHTENING` | ACTIVE_HIKING + real rate RESTRICTIVE, or PAUSING + elevated inflation | BEARISH |
| `TIGHTENING_PAUSE` | PAUSING + CPI moderating | NEUTRAL |
| `PIVOT_IMMINENT` | PAUSING/STABLE + declining CPI + FedWatch ≥25bp cuts priced | BULLISH |
| `EASING_CYCLE` | ACTIVE_CUTTING | BULLISH |
| `NEUTRAL` | STABLE trajectory, no inflation driver | NEUTRAL |

**Asset rotation per phase:**

| Phase | Favoured | Avoid |
|---|---|---|
| `EARLY_TIGHTENING` | XLE, XLF, XLV | XLRE, XLU, TLT, XLK |
| `PEAK_TIGHTENING` | XLE, GLD, SLV, XLF | XLRE, XLU, TLT, QQQ, XLK |
| `TIGHTENING_PAUSE` | XLV, XLP, GLD | XLRE, XLU |
| `PIVOT_IMMINENT` | TLT, XLRE, XLU, GLD, IEF | XLE, TBF |
| `EASING_CYCLE` | XLK, XLY, XLC, QQQ, XLRE | XLP, XLU |
| `NEUTRAL` | — | — |

**Claude prompt overlay (instruction 27c):** EARLY/PEAK_TIGHTENING → −0.04 on POSITION-horizon longs in rate-sensitive names. PIVOT_IMMINENT → +0.03 on rate-sensitive BUYs (TLT, XLRE, XLU). EASING_CYCLE → +0.03 on growth/cyclical BUYs. Rate-cycle phase is never allowed to override a strong near-term individual catalyst.

---

### Step 3D — Business Cycle Rotation (`src/data/business_cycle_rotation.py`)

When `ENABLE_BUSINESS_CYCLE_ROTATION=true`, derives the current structural economic cycle phase from the already-fetched FRED macro context — no new API calls, no cache, instant computation (same pattern as `compute_market_mode()`).

**What's distinct from the other rotation layers:**

| Layer | What it answers | Signal type |
|---|---|---|
| Sector Rotation (3B) | Where is money **flowing right now**? | Reactive, real-time momentum |
| Rotation Drivers (3C) | What is the **Fed doing**? | Monetary cycle (rate/CPI trajectory) |
| Business Cycle Rotation (3D) | Where are we in the **economic cycle**? | Structural, historically repeating |

**Phase classification (derived from FRED macro context):**

| Phase | Key conditions | Equity direction |
|---|---|---|
| `EARLY_EXPANSION` | EXPANSION regime, inflation LOW/MODERATE, unemployment FALLING | BULLISH |
| `MID_EXPANSION` | EXPANSION regime, inflation MODERATE, unemployment STABLE | BULLISH |
| `LATE_EXPANSION` | EXPANSION with ELEVATED inflation, or SLOWDOWN without inverted curve | NEUTRAL |
| `LATE_CYCLE` | LATE_CYCLE/SLOWDOWN regime, or yield curve INVERTED | BEARISH |
| `CONTRACTION` | RECESSION regime | BEARISH |
| `UNKNOWN` | Insufficient FRED data | NEUTRAL |

**Sector leadership by phase (Fidelity-style historical model, score ∈ [−1, +1]):**

| Sector | EARLY | MID | LATE | LATE_CYCLE | CONTRACTION |
|---|---|---|---|---|---|
| XLF Financials | +0.80 | +0.35 | 0.00 | −0.35 | −0.50 |
| XLRE Real Estate | +0.70 | +0.10 | −0.30 | −0.45 | −0.70 |
| XLY Consumer Disc. | +0.65 | +0.50 | −0.20 | −0.55 | −0.65 |
| XLK Technology | +0.50 | +0.80 | −0.15 | −0.45 | −0.50 |
| XLI Industrials | +0.30 | +0.65 | +0.45 | −0.20 | −0.55 |
| XLE Energy | +0.10 | +0.20 | +0.80 | +0.25 | −0.30 |
| XLB Materials | +0.20 | +0.25 | +0.65 | +0.10 | −0.45 |
| XLV Healthcare | 0.00 | +0.05 | +0.30 | +0.80 | +0.65 |
| XLP Consumer Staples | −0.20 | −0.25 | +0.10 | +0.70 | +0.75 |
| XLU Utilities | −0.30 | −0.35 | +0.20 | +0.60 | +0.60 |

**Convergence check:** after computing sector biases, the module compares cycle leaders (score ≥ 0.40) against real-time Ebb-and-Flow top inflows and laggards against outflows. Agreement is surfaced as "Confirming" in the prompt; disagreement is flagged as "Divergence."

**Claude prompt overlay (instruction 27d):** cycle leaders get a +0.03 confidence boost; STRONG_LAGGARD sectors get a −0.03 haircut on POSITION-horizon BUY calls. Maximum conviction is applied when business-cycle, Ebb-and-Flow, and rate-cycle (rotation drivers) all point the same direction. Contradicting layers reduce conviction by 0.03.

---

### Step 3L — Sentiment Velocity (`src/signals/sentiment_velocity.py`)

When `ENABLE_SENTIMENT_VELOCITY=true`, computes the **rate of change** of news tone for each ticker — Δsentiment, *not* the level. **Why velocity beats level for short horizons:** the static sentiment level is largely priced in, but the *change* in tone is the new information. A stock improving from very negative toward neutral often rallies even while its level is still mildly negative (the second derivative turned up); a stock fading from very positive often sells off even while still net-positive. The change leads 1–5 day price moves better than the level.

**Zero extra cost:** this reuses the article timestamps already stored on every `NewsArticle` (`published_at`) and a deterministic financial **lexical polarity** scorer — no additional LLM or API calls. (The LLM sentiment in Step 1 provides the *level*; this provides the *velocity*.)

**Score derivation:**

1. Filter to the ticker's relevant articles (same keyword filter as news sentiment).
2. Score each article's tone ∈ [−1, +1] from a positive/negative financial keyword balance: `(pos − neg) / (pos + neg)`.
3. Bucket by recency: **recent** window = articles ≤ `SENTIMENT_VELOCITY_RECENT_HOURS` old (default 24h); **prior** window = from there to `SENTIMENT_VELOCITY_PRIOR_HOURS` (default 96h).
4. `velocity = mean_tone(recent) − mean_tone(prior)` ∈ [−2, +2].
5. `score = tanh(velocity / 0.6) × confidence`, clamped to [−1, +1], where `confidence` damps thin windows (log-scaled by article count in each window).
6. Returns 0 when either window is empty (no change can be measured).

**Aggregator integration:** folded in as method `sent_velocity` with base weight **0.12**, stored on `TickerSignal.sentiment_velocity_score` (plus `sentiment_recent` / `sentiment_prior`). Tracked in performance attribution under the **Sentiment** category — so the system measures whether velocity adds alpha over the level. Claude sees a per-ticker `Sentiment VELOCITY` line and instruction **1b** (treat it as a short-horizon timing overlay: velocity agreeing with the level is the strongest news configuration; velocity opposing the level warns the level is about to mean-revert).

**Email:** a per-ticker "Sentiment Velocity" row in the signal breakdown and section 34a **Sentiment Velocity Leaderboard** (tone accelerating up vs deteriorating). Disable with `ENABLE_SENTIMENT_VELOCITY=false`.

---

### Step 3M — Trend Strength (`src/signals/trend_strength.py`)

When `ENABLE_TREND_STRENGTH=true`, computes a trend-**quality** signal for each ticker by combining two of the most empirically durable technical systems. It answers a question the rest of the stack doesn't: *is price in a strong, confirmed directional trend, and which way?* — distinct from price momentum (which measures the *size* of the return) and from RSI/Bollinger (which measure overbought/oversold). Uses the OHLCV chart cache first (works with `ENABLE_FETCH_DATA=false`); minimum 50 bars.

**1. ADX / DMI (Welles Wilder, 1978).** `+DI` and `-DI` measure upward vs downward directional movement (Wilder-smoothed); **ADX** measures trend strength independent of direction:

| ADX | Meaning |
|---|---|
| < 20 | No trend (chop) — directional signals unreliable; score is dampened toward 0 |
| 25–40 | Established trend |
| > 40 | Very strong trend |

Direction = `(+DI − -DI) / (+DI + -DI)` ∈ [−1, +1], scaled by `clip((ADX − 15)/30, 0, 1)`.

**2. Donchian channel breakout (the "Turtle" system).** A close above the prior 20-day high is a long breakout (+1); below the prior 20-day low, a short breakout (−1); between the bands, a mild lean toward the nearer band.

**Composite score** ∈ [−1, +1]: `score = clip(0.60 × adx_dir + 0.40 × donchian, -1, +1)`. Positive = confirmed uptrend; negative = confirmed downtrend; near zero = chop (dampened by design rather than guessing). A `trend_strength_label` (`STRONG_UPTREND`, `BREAKOUT_UP`, `NO_TREND`, …) is stored for display.

**Aggregator integration:** method `trend_strength`, base weight **0.15**, on `TickerSignal.trend_strength_score` (+ `adx_value`, `trend_strength_label`). Tracked in performance attribution under **Technical**. Claude receives a per-ticker `TrendStrength` line and instruction **16b** (treat it as a trend-following confirmation/gate: lean into strong aligned trends and 20-day breakouts; in ADX < 20 chop, prefer mean-reversion setups). Email: per-ticker row + a **Trend Strength Leaderboard** (strongest confirmed up/down trends). Tunable via `TREND_ADX_PERIOD` (14) and `TREND_DONCHIAN_PERIOD` (20). Disable with `ENABLE_TREND_STRENGTH=false`.

---

### Step 3E — Price Momentum / Perceived Value (`src/signals/price_momentum.py`)

When `ENABLE_PRICE_MOMENTUM=true`, computes a normalised multi-period price momentum score for each ticker. **Academic basis:** Jegadeesh & Titman (1993) showed that stocks outperforming over 3–12 months continue to outperform over the next 3–12 months — one of the most replicated factors in finance. The underlying mechanism is perceived value: as prices rise, investors perceive higher intrinsic value, attracting further capital and reinforcing the trend.

**Score derivation:**

```
1. Compute 1m (21-bar) and 2m (42-bar) raw returns from OHLCV history.
2. Normalise: z_1m = mom_1m / σ_1m  (trailing 252-bar std of 21-day returns)
              z_2m = mom_2m / (σ_1m × √2)  (approximate 2m std from 1m σ)
3. z_composite = 0.6 × z_1m + 0.4 × z_2m   (recent momentum weighted more)
4. score = tanh(z_composite / 1.5) + vol_adj  (clamped to [-1, +1])
```

| z_composite | Raw tanh score | Interpretation |
|---|---|---|
| 0.0 | 0.00 | In line with own history |
| +1.0 | +0.62 | 1σ above baseline — clear uptrend |
| +2.0 | +0.90 | 2σ above — market is chasing this name |
| −1.0 | −0.62 | Momentum selling territory |

**Volume confirmation adjustment (±0.10 max):**

| Condition | Adjustment | Interpretation |
|---|---|---|
| Uptrend (z > 0.5) + rising volume (ratio > 1.3×) | +0.10 | Institutional participation confirmed |
| Downtrend (z < −0.5) + rising volume (ratio > 1.3×) | −0.10 | Distribution confirmed |
| Any trend + thin volume (ratio < 0.6×) | −0.05 × sign(z) | "Thin air" move — less conviction |

**Cache strategy:** Prefers the incremental OHLCV chart cache (`cache/ohlcv/<TICKER>.json`). Falls back to a live `yfinance` fetch (`get_history(ticker, period="18mo")`) on cold cache. Works with `ENABLE_FETCH_DATA=false` when chart caches are populated. Minimum 50 bars required.

**Weight in aggregator:** 0.18 (base). Tracked in performance attribution under the **Technical** method category alongside RSI/MACD/BB.

**Email section 5x — Price Momentum Leaderboard:** The top 5 most-chased tickers (highest positive score) and bottom 5 most-sold (lowest negative score) are shown as colour-coded bar charts with 1m/3m return annotations.

**Why separate from VWAP:** VWAP distance measures **short-term mean-reversion** (where price is vs. 20-day average volume-weighted cost basis). Price Momentum measures **medium-term trend persistence** (whether the 1–2 month move is large relative to the ticker's own historical volatility). The two signals often conflict — when they do, the aggregator balances both.

**`.env` flag:**
```env
ENABLE_PRICE_MOMENTUM=true
```

---

### Step 4a — Sector Pairs / Relative Value (`src/signals/sector_pairs.py`)

After signal aggregation, `find_sector_pairs()` scans `_SECTOR_MAP` for divergences between sector ETFs and their constituent stocks. When the ETF and the stock disagree on direction, a market-neutral pair trade removes sector beta and isolates idiosyncratic alpha.

**Two setup types:**

| Setup | ETF | Stock | Trade |
|---|---|---|---|
| `ETF_BULL_STOCK_BEAR` | BULLISH | BEARISH | Long ETF / Short Stock |
| `ETF_BEAR_STOCK_BULL` | BEARISH | BULLISH | Long Stock / Short ETF |

**Entry criteria:** both legs must have a non-NEUTRAL direction and confidence ≥ 35%. The `pair_score` = average of both confidences — pairs are sorted by score descending and capped at 10 per run.

**Why it works:** when XLK is BULLISH but INTC is BEARISH, the sector tailwind is not lifting INTC. That idiosyncratic weakness is the alpha. Going Long XLK / Short INTC captures both the sector momentum and the relative underperformance, while the XLK long hedges away broad-tech beta from the short.

**Note:** pairs are reported in the email but are **not** fed into the trade ledger — pair-trade P&L accounting (two legs, correlated moves) requires different tracking than single-leg signals.

---

### Step 3F — Money Flow Indicators (`src/signals/money_flow.py`)

When `ENABLE_MONEY_FLOW=true`, computes a composite accumulation/distribution score from three independent volume-based indicators.

**Why money flow matters:** Price alone does not reveal conviction. When institutional investors accumulate positions, they do so at increasing volume — pushing typical price × volume (money flow) higher. Conversely, distribution occurs at declining price with elevated selling volume. Three indicators capture different facets of this dynamic:

| Indicator | Period | Interpretation |
|---|---|---|
| MFI (Money Flow Index) | 14-period | Volume-weighted RSI. < 20 = accumulation zone (bullish); > 80 = distribution (bearish). Contrarian at extremes. |
| CMF (Chaikin Money Flow) | 20-period | Close location × volume, summed. Positive = buyers in control; negative = sellers in control. Directional. |
| OBV slope z-score | 21-bar regression | Cumulative volume trend normalised. Rising slope = sustained buying; falling = distribution. |

**Score derivation:**

```
mfi_score  = tanh((50 − MFI) / 20)     # contrarian: low MFI → bullish
cmf_score  = tanh(CMF / 0.15)           # directional: + = accumulation
obv_score  = tanh(obv_z / 1.0)          # trend: rising OBV → bullish

composite  = 0.40 × mfi_score + 0.40 × cmf_score + 0.20 × obv_score
score      = tanh(composite / 0.6)  clamped to [−1, +1]
```

**Divergence signal:** When price rises but `MoneyFlow_score` falls (distribution), the move may be weak — a classic pump-without-conviction setup. When price falls but score is positive (accumulation), a reversal may be building.

**Cache strategy:** Uses the OHLCV chart cache (`cache/ohlcv/<TICKER>.json`) first — works with `ENABLE_FETCH_DATA=false` when caches are populated. Falls back to `get_history(18mo)` on cold cache. Minimum 30 bars required.

**Weight in aggregator:** 0.15 (base). Tracked in performance attribution under the **Technical** category.

**Email section 5z — Money Flow Leaderboard:** Top 5 strongest accumulation tickers and bottom 5 strongest distribution tickers, with MFI and CMF annotations.

**`.env` flag:**
```env
ENABLE_MONEY_FLOW=true
```

---

### Step 3I — IV Rank + Directional (`src/signals/iv_rank.py`)

A **regime-aware directional bias** built from each ticker's own volatility footprint. True historical implied volatility per ticker isn't stored, so realized volatility is used as a proxy — RV and IV are tightly correlated, and the **rank** of current vol within a ticker's own trailing distribution carries the same regime information as IV Rank.

**IV-Rank proxy:**
```
RV_21d       = stdev(log_returns_21d) × √252
IV_Rank ≈ percentile_rank(RV_21d, trailing_252_RV_21d)  ∈ [0, 100]
```

**Directional input:** 5-day return ÷ ATR%. This is "how many average daily ranges did this stock cover this week" — self-normalised against the ticker's own volatility so the same threshold works across the universe.

**Scoring switches by IR band:**

| Band | Logic | Score |
|---|---|---|
| **IR ≥ 70** + 5d move strongly negative (z_ret ≤ −1) | Capitulation regime — vol mean-reverts, fear bottoms | `+0.55` `CAPITULATION_BUY` |
| **IR ≥ 70** + 5d move strongly positive (z_ret ≥ +1) | Euphoric chase — expensive options + crowded long | `-0.55` `FADE_EXTREME` |
| **IR ≥ 70** + mild move | Event being priced in — small directional caution | `-0.20 × sign(z_ret)` `EVENT_CAUTION` |
| **IR ≤ 30** + positive trend | Cheap options + steady uptrend — room for vol expansion | `+0.40 × tanh(z_ret/1.5)` `CALM_UPTREND` |
| **IR ≤ 30** + negative trend | Complacency in decline — downtrend not yet feared | `-0.40 × tanh(|z_ret|/1.5)` `CALM_DOWNTREND` |
| **Mid IR (30 < IR < 70)** | Mild trend-following bias only | `0.30 × tanh(z_ret/1.5)` `TREND_FOLLOWING` |

**Robustness to regime shifts:** Both inputs are self-normalised — IR ranks against the ticker's own RV history; z_ret normalises the return by the ticker's own ATR. The same thresholds work across high-vol names (NVDA, TSLA) and low-vol names (SPY, XLP) without retuning.

**Cache strategy:** Uses the OHLCV chart cache first (works with `ENABLE_FETCH_DATA=false`). Falls back to `get_history(18mo)` on cold cache. Minimum 65 bars required.

**Weight in aggregator:** 0.13 (base). Tracked in performance attribution under the **Technical** category.

**Email per-ticker mrow + Leaderboard:** Score + IR + 5-day return + regime label per actionable ticker; leaderboard shows top 5 bullish / bottom 5 bearish biases with IR and regime annotations.

**`.env` flag:**
```env
ENABLE_IV_RANK=true
```

---

### Step 3J — IV Expression (`src/signals/iv_expr.py`)

A **stock-vs-options expression decision** built from the **real options chain** — distinct from `iv_rank`, which uses a realized-vol proxy. This method reads true market-implied volatility straight from the options chain that's already fetched for GEX, then decides whether a directional thesis should be expressed in stock, in long options, or faded entirely.

**Inputs (all reused from the day's GEX context — zero new fetches):**

| Input | Meaning |
|---|---|
| `expected_move_pct` | Market-implied ±1σ move (ATM straddle ÷ spot) — the live IV reading |
| `oi_skew` | OI-weighted directional positioning ∈ [−1, +1]; +1 = call OI piled above spot |
| `gex_signal` | Dealer gamma regime: PINNED / AMPLIFIED / NEUTRAL |

**Real IV Rank (no synthetic proxy):** `expected_move_pct` is volatile in absolute terms (2% on SPY is sleepy; 19% on a single name is an event), so it's *ranked* against the ticker's own recent history — reconstructed by reading prior `cache/gex_*.json` files on disk (60-day window). With ≥6 historical readings the current IV is ranked by percentile; with fewer, a universe-median-relative fallback keeps the signal alive on cold-cache days.

**Score logic (expression decision):**

| Regime | Logic | Score |
|---|---|---|
| **IV rank ≥ 75** + strong skew (\|skew\| ≥ 0.50) | Rich premium + directional crowd → vol mean-reverts | `−0.55 × sign(skew)` `FADE_PREMIUM` |
| **IV rank ≥ 75** + weak skew | Event being priced, no conviction | `−0.20 × sign(skew)` `EXPENSIVE_NEUTRAL` |
| **IV rank ≤ 25** + strong skew | Cheap options + decisive positioning → high-conviction expression | `+0.55 × sign(skew)` `CHEAP_DIRECTIONAL_LONG/SHORT` |
| **IV rank ≤ 25** + weak skew | Cheap but unconvinced | `+0.20 × sign(skew)` `CHEAP_COMPLACENT` |
| **Mid IV (25–75)** | Options market's directional view, un-amplified | `0.30 × oi_skew` `MID_IV_DIRECTIONAL` |

**AMPLIFIED dealer-gamma adjustment:** when dealers are short gamma (moves accelerate), add `+0.10 × sign(score)` to push further in the favourable direction. PINNED gets no boost (dealers are suppressing vol).

**Why this differs from `iv_rank`:** `iv_rank` answers "what's the directional thesis given the vol regime?" using realized vol. `iv_expr` answers "given the options market's *own* positioning and *true* implied vol, should I express this in stock or options, and is the premium worth paying?" — it can act as a contrarian counterweight when the options market is over-pricing a crowded directional move (e.g. XLK with rich IV + heavy put skew → `FADE_PREMIUM` long counterweight).

**Cache strategy:** No cache of its own — reads the live GEX context for current values and the on-disk GEX caches for history. **Requires `ENABLE_GEX=true`** (`gex_context` must be present); tickers without a GEX entry return `NO_OPTIONS_DATA` and contribute nothing.

**Weight in aggregator:** 0.12 (base). Tracked in performance attribution under the **Options** category.

**Email per-ticker mrow + Leaderboard:** Score + IV-rank + OI-skew + expression label per ticker; leaderboard shows top 5 cheap-directional / bottom 5 fade-premium biases.

**`.env` flag:**
```env
ENABLE_IV_EXPR=true
```

---

### Step 3K — Cointegration Pairs (`src/signals/cointegration.py`)

A **statistical-arbitrage, market-neutral** overlay that goes beyond Sector Pairs (which keys off opposing directional *signals*). It tests whether two economically-linked price series are *cointegrated* — i.e. a linear combination of them is stationary (mean-reverting) even though each series individually wanders (has a unit root). When a cointegrated spread stretches far from its mean, Long the cheap leg / Short the rich leg and bet on reversion; the hedge ratio removes the shared market beta.

**Engle-Granger two-step (native numpy — no `statsmodels` dependency):**

1. OLS the log prices: `log(A) = α + β·log(B) + ε`. β is the hedge ratio; ε is the spread (residual).
2. ADF-test the residual spread for a unit root. Rejecting the null (stat below the critical value) ⇒ the spread is stationary ⇒ A and B are cointegrated. Critical values are the Engle-Granger residual-based MacKinnon values — more demanding than plain ADF because the spread is an *estimated* residual: **1% −3.90, 5% −3.34, 10% −3.04**.

**Candidate pairs:** a curated list of economically-linked securities (gold trackers GLD/IAU/SLV/GDX, index ETFs SPY/IVV/QQQ/XLK, semis AMD/NVDA/INTC/AVGO, mega-cap tech, payments V/MA, banks JPM/BAC/GS/MS, energy XOM/CVX, retail HD/LOW, autos F/GM, telecom T/VZ, …) plus same-sector combinations drawn from the aggregator's `_SECTOR_MAP`, restricted to today's universe. Only pairs whose **both** legs have enough cached/fetchable history are tested.

**Trade signal (spread z-score = (spread − mean) ÷ std):**

| z-score | Meaning | Action |
|---|---|---|
| `z ≥ +entry` | spread rich (A expensive vs B) | SHORT A / LONG B |
| `z ≤ −entry` | spread cheap | LONG A / SHORT B |
| `\|z\| < exit` | near fair value | no edge |

Defaults: `entry = 2.0σ`, `exit = 0.5σ`. An **Ornstein-Uhlenbeck half-life** filter (regress Δs on s₋₁ ⇒ half-life = −ln2/λ) drops pairs that revert too slowly to trade (> 60 days). A pair is `ENTRY` when cointegrated **and** `\|z\| ≥ entry` **and** the half-life is fast; `STRETCHED` if cointegrated and stretched but slow-reverting; `MONITOR`/`NEUTRAL` otherwise.

**Per-ticker directional lean:** although the natural output is *pairs*, each tradeable pair also implies a single-name view — its cheap (long) leg earns a bullish nudge, its rich (short) leg a bearish one, scaled by how far the spread is stretched past the entry threshold (tanh-saturated). These are averaged across every pair a ticker belongs to into `ticker_scores ∈ [−1, +1]`, which the aggregator folds into the per-ticker `combined_score` like any other method.

**Cache strategy:** No cache of its own — cache-first via `load_ohlcv()` (works with `ENABLE_FETCH_DATA=false`); falls back to `get_history(period="1y", force_refresh=True)` only when the cache is shorter than 180 bars and fetching is enabled (cointegration needs a long window for statistical power).

**Weight in aggregator:** 0.12 (base). Tracked in performance attribution under the **Relative** category (alongside Cross-Sectional Ranking).

**Email section 35b + per-ticker mrow:** the Cointegration Pairs section lists each tradeable pair (β, ADF vs critical value, half-life, z-score, LONG/SHORT legs, rationale); each affected ticker also gets a row in its signal breakdown. Distinct from Sector Pairs — this is a pure statistical relationship, not a directional-signal divergence.

**`.env` flags:**
```env
ENABLE_COINTEGRATION=true
COINTEGRATION_ENTRY_Z=2.0      # |z| at/above which a pair is an actionable ENTRY
COINTEGRATION_EXIT_Z=0.5       # |z| below which a pair is fair-value / no edge
COINTEGRATION_PVALUE=0.05      # ADF significance level (0.01 | 0.05 | 0.10)
```

---

### Step 4 — Signal Aggregation (`src/signals/aggregator.py`)

Combines up to sixteen signal methods with dynamically normalized weights:

| Method | Base weight | Source |
|---|---|---|
| News sentiment | 40% | DeepSeek V3 LLM scoring of all article-type sources |
| Sentiment velocity | 12% | Δ news tone (recent − prior window) — rate of change, not level |
| Technical analysis | 30% | RSI, MACD, SMA20/50, Bollinger Bands |
| Smart money / insider | 30% | Insider trades + options flow + SEC filings |
| Put/call ratio | 15% | Per-ticker CBOE options volume |
| Max pain gravity | 12% | GEX options chain, expiry-decay weighted |
| OI-weighted skew | 15% | GEX call/put OI directional lean |
| VWAP distance | 12% | Price vs. rolling 20-day VWAP (mean-reversion) |
| Pattern recognition | 18% | Historical win rate for current chart pattern |
| Price momentum | 18% | 1m/2m normalised returns vs own 252-bar history |
| Money flow | 15% | MFI + CMF + OBV slope composite accumulation/distribution |
| Trend strength | 15% | ADX/DMI directional movement + Donchian 20-day breakout |
| PEAD | 15% | SUE × time-decay drift from earnings surprises |
| IV Rank + Directional | 13% | 21d RV percentile × 5d return/ATR, regime-aware switching |
| IV Expression | 12% | Real options-chain IV percentile × OI skew, stock-vs-options decision |
| Cointegration | 12% | Engle-Granger ADF + spread z-score; per-ticker lean from stat-arb pairs |

Weights are re-normalized at runtime based on which methods are enabled — they always sum to 100%.

**Confidence formula:**
```
raw_confidence     = min(1.0, |combined_score| / 0.5)
coherence_factor   = 0.45 + agreement_ratio × 0.90   ∈ [0.45, 1.35]
movement_factor    = f(ATR%, BB-width%, GEX_signal)   ∈ [0.70, 1.30]
volume_factor      = f(vol_ratio, |combined|, coherence)  ∈ [0.90, 1.15]

confidence = raw_confidence × coherence_factor × movement_factor × volume_factor
```

`coherence_factor` is a continuous measure of how strongly methods agree, magnitude-weighted — a weak outlier pointing opposite costs less than a strong one.

**Interaction adjustments (additive, capped ±0.15):**
1. **Insider accumulation at technical support** — insiders bullish while price technically weak: contrarian value-accumulation setup (+0.10)
2. **Options extreme aligned with direction** — extreme put/call skew confirming combined signal (+0.07)
3. **News catalyst confirmed by volume** — high sentiment magnitude + elevated volume (1.5×) (+0.06)

**Second pass — sector alignment:** Individual stocks are cross-referenced against their sector ETF. Alignment → 1.10× confidence boost; contradiction → 0.75× penalty.

**Actionable threshold:** Only `BUY` and `SELL` with `confidence ≥ 0.78` AND `sources_agreeing ≥ 2` are considered actionable. A single strong source never produces a BUY/SELL regardless of score magnitude.

---

### Step 5 — Final Recommendations (`src/analysis/claude_analyst.py`)

All ticker signals plus every macro/breadth/volatility context block are passed in a single structured prompt to the configured **analyst model** (default: `claude-haiku-4-5-20251001`, configurable via `ANALYST_MODEL`).

**Context blocks injected into the prompt:**

| Block | Source |
|---|---|
| `<macro_context>` | FRED macro regime |
| `<macro_surprise_context>` | CESI-style economic surprise score |
| `<fedwatch_context>` | Market-implied Fed rate path |
| `<revision_momentum_context>` | Analyst estimate revision trend |
| `<cot_context>` | CFTC speculator positioning table |
| `<ipo_pipeline>` | SEC S-1/S-11 sector filing counts |
| `<vix_context>` | Volatility regime + term structure |
| `<credit_context>` | HYG vs SPY divergence |
| `<put_call_context>` | Market-wide + per-ticker P/C ratio |
| `<tick_context>` | NYSE TICK breadth exhaustion |
| `<breadth_context>` | % of sector ETFs above 200d SMA |
| `<highs_lows_context>` | 52-week HL spread + divergence |
| `<mcclellan_context>` | A/D breadth oscillator + summation |
| `<whisper_context>` | Implied whisper vs consensus |
| `<earnings_calendar>` | Upcoming earnings dates |
| `<gex_context>` | Gamma exposure + max pain |

Claude acts as an elite portfolio manager with 22 numbered decision rules covering: conviction thresholds, smart money weighting, macro overlays, cluster handling, volatility regimes, breadth conditions, earnings event caution, and more. When no ticker clears the bar, it outputs HOLD/WATCH for all.

**Automatic fallback chain:** If the Claude API call fails for any reason (credits exhausted, authentication error, rate limit, server error, or connection failure), `generate_recommendations()` automatically re-sends the identical prompt to **DeepSeek V3** (`deepseek-chat`) via the OpenAI-compatible streaming API. If DeepSeek also fails, a rule-based converter produces conservative HOLD/WATCH/BUY/SELL from the raw signal scores. The active analyst model is logged at INFO level.

---

### Step 6 — Performance Tracking (`src/performance/tracker.py` + `src/performance/daily_nav.py`)

Every actionable signal is recorded in the **DuckDB** trade ledger (`data/llm_trader.db`; the legacy `cache/trades.json` is now import-only — see [Database & Dashboard](#database--dashboard)). Each trade carries:

| Field | Purpose |
|---|---|
| `ticker`, `type`, `action`, `direction`, `confidence` | Identity of the position. |
| `entry_date`, `entry_datetime`, `entry_price` | The date plus the **exact UTC ISO 8601 instant** the price was sampled, plus the price itself — paired together so the audit trail says "we bought X at price P at time T". |
| `current_price`, `current_price_datetime` | Live M2M mark + timestamp; refreshed each pipeline tick. |
| `exit_date`, `exit_datetime`, `exit_price` | Same datetime/price pairing when the trade closes (auto-close or signal reversal). |
| `return_pct` | Spread-adjusted buy-and-hold percent return (see formula below). |
| `position_size_multiplier`, `sector_key` | Confidence-tier sizing and the bucket used for the 3× per-sector cap. |
| `method_scores`, `methods_agreeing`, `dominant_method` | Per-method attribution captured at entry. |
| `status` | `OPEN` or `CLOSED`. |

Lifecycle:

1. **Open** (`record_new_trades`) — entry price fetched **live** at recommendation time and stamped together with `entry_datetime`; position-size multiplier set from confidence tier; correlation haircut applied. The **intraday timing gate** (`enable_intraday_timing`, default on) defers an entry whose 30-min momentum is strongly against it — the next 30-min tick re-checks, so the position waits for a less hostile entry.
2. **Refresh / mark** (`update_open_trades`) — every tick re-fetches the live price and updates `current_price`/`current_price_datetime`/`return_pct`/`weighted_return_pct`/`days_held` for every open trade. **There is no time cap** — a position is held as long as its thesis holds (`days_held` is observability only).
3. **Thesis-decay close** (`monitor_open_positions`) — closes a position when its rationale deteriorates, checked in priority order: `macro_regime_exit` (holding a long while macro = PANIC/RISK_OFF), `signal_flipped` (today's oriented combined score crosses against the trade), `signal_decay` (entry strength minus today's strength exceeds the drop threshold), `confidence_loss` (today's aggregator confidence below `max(absolute_floor, relative_factor × entry_confidence)`). Toggle with `enable_signal_decay_exits` (default on). With `enable_intraday_exit` (opt-in) it also closes on a hard 30-min reversal against the position (`intraday_reversal`).
4. **Reversal close** (`close_trades_on_signal_reversal`) — if today's actionable signal flips the direction of an open position, it closes with `exit_datetime = current_price_datetime` (re-uses the most recent live mark, no extra fetch) and the new leg is opened immediately after.

Before either refresh runs, `update_open_trades` does two preparatory passes that make every downstream metric deterministic and current:

- `_refresh_open_trade_ohlcv()` — for every open-trade ticker whose OHLCV cache is older than yesterday, calls `market_data.get_history(force_refresh=True)` (bypassing the normal 3-day TTL) so the daily-NAV walk has a real close for every day the position was held.
- `_normalize_closed_returns()` — idempotently re-derives `return_pct` for every closed trade from its stored entry/exit prices using the current spread model. Without this, legacy closed trades carry whatever spread model was in effect at close time and the summary stats drift; with it, the data file is always consistent with the current code.

---

#### Per-trade return — `_pct_return()` (buy-and-hold, spread-aware)

`return_pct` is the percent change from effective entry to effective exit, **with the bid-ask half-spread applied on both legs**:

```
half_in  = _dynamic_half_spread(entry_price,            asset_type)
half_out = _dynamic_half_spread(exit_or_current_price,  asset_type)

BUY  : eff_entry = entry × (1 + half_in)        # paid the ask
       eff_exit  = exit  × (1 − half_out)       # received the bid
       return_pct = (eff_exit − eff_entry) / eff_entry × 100

SELL : eff_entry = entry × (1 − half_in)        # received the bid
       eff_exit  = exit  × (1 + half_out)       # paid the ask to cover
       return_pct = (eff_entry − eff_exit) / eff_entry × 100
```

**Half-spread by asset type and price tier (`_dynamic_half_spread`):**

| Asset type | Price tier | Half-spread |
|---|---|---|
| ETF (any price) | — | 1 bp |
| Commodity (≥$100) | GLD, GDX-style | 1.5 bp |
| Commodity (<$100) | SLV, CPER-style | 3 bp |
| Stock ≥$50 | Large-cap | 2 bp |
| Stock $10–$50 | Mid-cap | 4 bp |
| Stock $1–$10 | Small-cap | 12.5 bp |
| Stock $0.10–$1 | Micro-cap | 37.5 bp |
| Stock $0.01–$0.10 | Penny | 100 bp |
| Stock < $0.01 | Sub-penny / warrant | 250 bp |

Entry and exit half-spreads are evaluated independently from their respective prices, so a trade can enter at one tier and exit at another (e.g. a penny stock crossing $1).

For **open trades**, `return_pct` is the live M2M using `current_price` in place of `exit_price` — same formula, same spread treatment. So open positions count exactly like closed positions in every win-rate, average, best/worst calculation. Refreshed each pipeline tick by `update_open_trades`.

For a brand-new trade entered and immediately marked at the same price, `return_pct` is **`−2 × half_spread`** (small negative) — the round-trip transaction cost. This means **win rate already accounts for spread**: a "win" requires the price move to cover the full round-trip cost; the threshold stays `> 0` because the spread is baked into `return_pct` before the comparison.

---

#### Price formatting — `fmt_price()`

Prices throughout the email and logs are formatted with `2`, `4`, or `6` decimal places depending on magnitude (`≥$1` → `$12.34`, `$0.01–$1` → `$0.0312`, `<$0.01` → `$0.003142`). Prevents sub-penny stocks and warrants from displaying as `$0.00`.

---

#### Confidence-scaled position sizing

Each trade is assigned a `position_size_multiplier` based on the signal's confidence level:

| Confidence | Multiplier | Interpretation |
|---|---|---|
| 0.78 – 0.85 | **1.0×** | Baseline — meets the actionable threshold |
| 0.85 – 0.92 | **1.5×** | Mid-conviction — worth committing more capital |
| > 0.92 | **2.0×** | High-conviction — maximum allocation |

The multiplier is the **weight** used everywhere capital-weighting applies: `wtd_avg_return`, the per-day portfolio return inside the daily-NAV engine, and the size-adjusted method statistics.

**Per-sector cap:** the sum of multipliers across all *open* positions in the same sector cannot exceed **3.0×**. If a new trade would push the sector over the cap, its multiplier is reduced to fit (or the trade is skipped if the sector is already at capacity). Sector groupings:
- Sector ETFs (XLK, XLF …): each ETF is its own bucket
- Commodities (GLD, SLV, GDX …): grouped together as "COMMODITY"
- Stocks: looked up in `_SECTOR_MAP`; unknown stocks each count independently

---

#### Daily-NAV compound — `daily_nav.compute_compound_return()`

This is the engine behind every "compound" number you see in the email — the portfolio inception value, the 1w/2w/1m time windows, and the per-segment compound in the performance breakdown table. It walks the **real daily closes** from `cache/ohlcv/<TICKER>.json` for every trade, then aggregates across positions.

**Per-trade walk** (`_build_marks` + `_daily_returns_for_trade`):

1. **Day 0 anchor** — `(entry_date, effective_entry)` where `effective_entry = entry × (1 ± half_spread)` matching `_pct_return`.
2. **Intermediate marks** — every cached OHLCV close strictly between `entry_date` and the end date, at its raw close (no spread — these days are MTM marks, not actual trades).
3. **End anchor** — `(exit_date, effective_exit)` for closed trades, or `(today, effective_exit_on_current_price)` for open trades. The open-trade end mark applies the spread because it represents "what you'd get closing right now" — same convention as `return_pct`.

Then for every adjacent pair `(mark_{d−1}, mark_d)`:

```
r_d = sign × (mark_d − mark_{d−1}) / mark_{d−1}
sign = +1  for BUY (long)
sign = −1  for SELL (short)
```

The walk is **path-faithful**: each day's return depends only on two adjacent observed prices. For longs the compound of these daily returns telescopes back to the buy-and-hold return. For shorts it doesn't — the daily-compounded short suffers volatility decay just like a daily-rebalanced 1× inverse ETF would, so its compound legitimately diverges from the trade's `return_pct` (a volatile short with 0% round-trip can have a negative daily compound; this is the "more accurate" model of what the position actually did under daily marking).

**Portfolio aggregation** (`compute_compound_return`):

1. Collect every `(date, daily_return, weight)` triple from every trade in the slice (`weight = position_size_multiplier`).
2. Group by date; on each date the portfolio's day return is the capital-weighted average:
   ```
   day_return = Σ(r · w over active positions) / Σ(w over active positions)
   ```
3. Compound sequentially across dates: `compound = ∏(1 + day_return) − 1`, expressed as a percent.

**Determinism guarantee:** same DuckDB trades (`data/llm_trader.db`) + same `cache/ohlcv/*.json` produces bit-identical output. No network call, no random seed, no time-of-day dependency. Verified by running `get_performance_for_email()` twice and comparing the full JSON payload.

Time-window variants just filter the trade list before aggregating:

| Key | Trades included |
|---|---|
| `compound_inception` | Every trade ever recorded (closed + open). |
| `return_1w`  | Trades with `entry_date >= today − 7 calendar days`. |
| `return_2w`  | Trades with `entry_date >= today − 14 calendar days`. |
| `return_1m`  | Trades with `entry_date >= today − 30 calendar days`. |

The email **Portfolio Performance** card displays these as tile widgets alongside the inception compound. `daily_pnl_breakdown(trade)` returns the per-day `(date, mark, return_pct)` series for one trade — used for the audit log so a human can verify every mark came from a real OHLCV close.

---

#### Summary statistics — `_compute_segment_stats()`

Each row of the email's Performance Breakdown table comes from this function. For any slice of trades (All / Stocks / ETFs / Commodities / Longs / Shorts / per-method):

| Metric | Formula | Source |
|---|---|---|
| `trades` | `len(trades)` | — |
| `win_rate` | `100 × count(t.return_pct > 0) / len(trades)` | Stored `return_pct` (spread-adjusted). |
| `compound_return` | `daily_nav.compute_compound_return(trades)` | Path-faithful daily walk. |
| `avg_return` | `mean(t.return_pct)` | Stored `return_pct`. |
| `wtd_avg_return` | `Σ(t.return_pct · t.position_size_multiplier) / Σ(t.position_size_multiplier)` | Capital-weighted. |
| `best` / `worst` | `max(t.return_pct)` / `min(t.return_pct)` | Per-trade extremes. |

Open trades are included at their live M2M `return_pct`, so every metric reflects the full live portfolio. The compound column uses the daily walk; the other columns use the per-trade `return_pct`. These two views can disagree for shorts (path-faithful vs buy-and-hold) and the table shows both side by side on purpose.

`stats["compound_return"]` in `get_performance_for_email()` is overwritten with `portfolio_metrics["compound_inception"]` so the headline number, the "All Trades" performance row, and the Portfolio Performance tile all show the exact same daily-walked compound.

**Method Attribution Analytics**

Every *new* trade stores ten raw method scores at entry time:

| Field | Signal |
|---|---|
| `news` | News sentiment (DeepSeek) |
| `tech` | Technical analysis (RSI/MACD/SMA/BB) |
| `insider` | Smart money (Form 4 + options flow + 13F) |
| `put_call` | Put/Call ratio |
| `max_pain` | Options max pain / GEX |
| `oi_skew` | Open interest skew |
| `vwap` | VWAP distance |
| `pattern` | Chart pattern recognition |
| `momentum` | Price Momentum (Perceived Value) |
| `money_flow` | Money Flow (MFI + CMF + OBV) |
| `pead` | Post-Earnings Announcement Drift (SUE × time-decay) |
| `iv_rank` | IV Rank + Directional (RV percentile × 5d return/ATR) |
| `iv_expr` | IV Expression (real options-chain IV percentile × OI skew) |
| `coint` | Cointegration Pairs (Engle-Granger ADF + spread z-score) |
| `cross_sectional` | Cross-Sectional Ranking (avg per-method z-score vs universe) |

Plus `methods_agreeing` (the subset with `|score| > 0.10` in the trade direction) and `dominant_method` (highest absolute score). After sufficient attributed trades accumulate, the email section **Signal Method Attribution** shows four analytics tables:

1. **Individual Methods** — win rate, avg return, size-adjusted weighted avg return; sorted descending by win rate; color-coded green ≥55%, amber 45–55%, red <45%.
2. **Method Categories** — rolls individual methods up into six groups: Sentiment (news), Technical (tech, vwap, pattern, momentum, money_flow, iv_rank), Smart Money (insider), Options (put_call, max_pain, oi_skew, iv_expr), Fundamental (pead), Relative (cross_sectional).
3. **Signal Convergence** — performance grouped by how many methods agreed (1 / 2 / 3 / 4+); validates whether the coherence multiplier in the aggregator is actually improving outcomes.
4. **Lead Signal** — which single method had the highest score in the trade direction; shows which signal type tends to lead profitable setups.

**Solo Method Performance (email section 4b)**

A separate simulated backtest answers: *"what would the return have been if you had only followed this one signal?"*

`compute_solo_method_performance()` iterates every closed trade with stored method_scores and, for each signal method, applies `_hypothetical_trades_for_method`:

- `|score| < 0.10` → method has no view → trade skipped.
- `score > 0` → solo signal is BUY; `score < 0` → solo signal is SELL.
- Same direction as actual trade → the trade is included **verbatim** — same dates, same prices, same `return_pct`, same daily walk over the real OHLCV closes.
- Opposite direction → `_flip_trade(t)` returns a same-ticker/same-dates dict with `action` and `direction` inverted and `return_pct` re-derived from the stored entry/exit prices via `_pct_return()` for the *flipped* action. The daily-NAV engine then walks the real OHLCV closes in the opposite direction (path-faithful), giving the genuine compound a daily-marked short-of-this-long position would have produced. No "just negate the stored number" tricks.

The result is a per-method table (email section 4b) showing trades, win rate, compound return (daily-walked, via `daily_nav`), avg return, best, and worst for the solo-signal scenario. This differs from the attribution breakdown (which counts only trades where the method agreed with the aggregated direction) by also accounting for trades the method would have gone against — giving a true standalone predictive-power picture.

**Method Evaluation (email section 4c)**

`compute_method_eval_stats()` answers two distinct questions beyond section 4b's P&L simulation:
1. **Directional accuracy** — what fraction of the time did this method's direction lead to a profitable hypothetical outcome?
2. **Conviction calibration** — does higher |score| predict better accuracy?

For each method the function produces `directional_accuracy`, `avg_return_correct`, `avg_return_wrong`, and `conviction_bands`:

| Band | |score| range | Interpretation |
|---|---|---|---|
| Low | 0.10 – 0.35 | Weak signal |
| Medium | 0.35 – 0.65 | Moderate conviction |
| High | 0.65+ | Strong conviction |

A well-calibrated signal shows rising accuracy Low→Medium→High. Flat or declining accuracy across bands indicates the method adds directional noise in isolation. Section 4c renders as per-method cards with overall stats and a compact conviction-band table.

Legacy trades (recorded before this feature) have no `methods_agreeing` field and are excluded from attribution stats. The email section shows a graceful "no attribution data yet" placeholder until enough attributed trades close.

---

### Step 7 — Charts & Email (`src/charts/`, `src/notifications/email_sender.py`)

**HTML report** (`logs/report_YYYY-MM-DD_HHMM.html`) — self-contained, browser-ready:

| Section | Content |
|---|---|
| Signal overview | Horizontal bar chart of all tickers |
| BUY/SELL cards | Full signal breakdown per actionable ticker |
| Monitor list | Compact HOLD/WATCH table |
| Macro dashboard | FRED regime, COT, VIX, credit, breadth, McClellan |
| Estimate revision | Analyst consensus trend per ticker |
| Earnings whisper | Implied whisper vs. consensus per ticker |
| Smart money | Insider/politician trades with cluster badges |
| Portfolio | 1w/2w/1m + since-inception dollar-weighted return tiles, P&L curve, open/closed trades with `fmt_price()` precision |

**Email** — charts embedded as inline base64 PNG (no attachments). Degrades gracefully to text-only if `kaleido` is not installed. Signal method performance tables are sorted by solo win rate (best signal first). Prices in trade tables use `fmt_price()` for sub-penny precision.

---

## Model Routing

| Task | Model | Fallback |
|---|---|---|
| Per-ticker sentiment scoring | DeepSeek V3 (`deepseek-chat`) | Claude Haiku 4.5 |
| Technical analysis scoring | Computed locally (RSI, MACD, SMA, BB) | — |
| Final synthesis / BUY/SELL/HOLD/WATCH | Configurable via `ANALYST_MODEL` (default: `claude-haiku-4-5-20251001`) | DeepSeek V3 (`deepseek-chat`) → rule-based fallback |

To use Sonnet for higher quality: set `ANALYST_MODEL=claude-sonnet-4-6` in `.env`.

**DeepSeek V3 analyst fallback:** When the configured Claude model raises any API error — credits exhausted (400/402), bad key (401), permission denied (403), rate limit (429), server error (5xx), or connection failure — `generate_recommendations()` automatically retries the identical prompt through `deepseek-chat` (DeepSeek V3) via the OpenAI-compatible API. Requires `DEEPSEEK_API_KEY` in `.env`. If DeepSeek also fails, the pipeline falls back to a simple rule-based converter (`_fallback_recommendations()`). The source model is logged at INFO level so you can see which analyst ran.

---

## Database & Dashboard

### DuckDB — single source of truth (`src/db/`)

Trades, recommendations, and run history live in a single embedded **DuckDB** file (`data/llm_trader.db`), created automatically on first run. It replaces the legacy `cache/trades.json` / `cache/hypothetical_trades.json` ledgers, which are now **import-only**.

| Table | Contents |
|---|---|
| `runs` | One row per pipeline invocation — market mode, macro regime, confidence threshold, universe size, LLM providers, timing |
| `run_sources` | Per-source "APIs used" record for each run (ok / error / duration) |
| `recommendations` | Every recommendation with rationale, attribution (`dominant_method`, `methods_agreeing`), and the actionable flag |
| `trades` | The real signal-driven trade ledger — full dict in a JSON `data` column + projected scalar columns |
| `hypothetical_trades` | The always-open paper book |

**Concurrency:** DuckDB allows a single read-write handle *or* many read-only handles across processes. The pipeline is the **sole writer** and holds the write lock only momentarily (open → write → close); it persists run metadata, sources, and recommendations at the end of every run, wrapped so a database hiccup never aborts a run. The dashboard connects **read-only**.

**Migration:** import an existing JSON ledger once with `python -m src.db.migrate` (idempotent; pass `--force` to overwrite). As a safety net the tracker also self-seeds the database from `cache/trades.json` when the DB is empty, so the cutover never loses history.

### Monitoring dashboard (`dashboard/`)

A read-only [Plotly Dash](https://dash.plotly.com/) app for inspecting the database:

```bash
python main.py --dashboard      # http://127.0.0.1:8050 by default
```

Three tabs:

| Tab | Shows |
|---|---|
| Recommendations & Rationale | Per-run data sources used (✓/✗) and the run's recommendations with rationale |
| Method Performance | Solo win-rate per signal method (bar chart + table), plus an *LLM models used* table — the exact synthesis & sentiment models that ran (including DeepSeek / rule-based fallbacks) |
| Returns | KPI tiles (compound, win rate, best/worst), equity curve, and open/closed trades |

Each tab's content is embedded directly in the tab, so switching is instant and handled client-side; the page is rebuilt with fresh data on each load (reload to refresh — the selected tab is remembered). Tables are sortable (click a column header, shift-click for multi-sort) and filterable (per-column search box), with human-readable headers and Eastern-time timestamps. Hover any column header, metric tile, or section heading for a plain-English explanation. It is served by **waitress** — a production-grade, multi-threaded, cross-platform WSGI server (the right choice on Windows, where gunicorn doesn't run) — wrapped in an auto-restart supervisor loop so it stays alive for always-on use (it falls back to the Dash dev server only if waitress isn't installed). All database access is read-only with exponential-backoff retry around the brief daily write-lock window, and the heavy performance computation is cached for 60 seconds. Host and port are configurable via `DASHBOARD_HOST` / `DASHBOARD_PORT`.

---

## Caching Strategy

| Cache | Key | TTL | Location |
|---|---|---|---|
| News + 8-K | `YYYY-MM-DD_HH` | 1 hour | `cache/news_*.json` |
| Snapshots | `YYYY-MM-DD_HH` | 1 hour | `cache/snapshots_*.json` |
| Reddit sentiment | `YYYY-MM-DD_HH` | 1 hour | `cache/reddit_*.json` |
| Google Trends | `YYYY-MM-DD` | 1 day | `cache/trends_*.json` |
| IPO pipeline | `YYYY-MM-DD` | 1 day | `cache/ipo_*.json` |
| VIX & term structure | `YYYY-MM-DD` | 1 day | `cache/vix_*.json` |
| Credit market (HYG/SPY) | `YYYY-MM-DD` | 1 day | `cache/credit_*.json` |
| Put/call ratio | `YYYY-MM-DD` | 1 day | `cache/put_call_*.json` |
| Analyst ratings | `YYYY-MM-DD` | 1 day | `cache/analyst_ratings_*.json` |
| Earnings surprises | `YYYY-MM-DD` | 1 day | `cache/earnings_surprises_*.json` |
| Earnings calendar | `YYYY-MM-DD` | 1 day | `cache/earnings_calendar_*.json` |
| Short interest | `YYYY-MM-DD` | 1 day | `cache/short_interest_*.json` |
| Market breadth | `YYYY-MM-DD` | 1 day | `cache/breadth_*.json` |
| New 52w highs/lows | `YYYY-MM-DD` | 1 day | `cache/highs_lows_*.json` |
| McClellan Oscillator | `YYYY-MM-DD` | 1 day | `cache/mcclellan_*.json` |
| Macro Surprise Index | `YYYY-MM-DD` | 1 day | `cache/macro_surprise_*.json` |
| Fed Rate Expectations | `YYYY-MM-DD` | 1 day | `cache/fedwatch_*.json` |
| Revision Momentum | `YYYY-MM-DD` | 1 day | `cache/revision_momentum_*.json` |
| Earnings Whisper | `YYYY-MM-DD` | 1 day | `cache/whisper_*.json` |
| OHLCV (charts) | per ticker | incremental | `cache/ohlcv/*.json` |
| COT positioning | ISO week | 1 week | `cache/cot_YYYY_WW.json` |
| Sector rotation | `YYYY-MM-DD` | 1 day | `cache/sector_rotation_*.json` |
| Rotation Drivers | `YYYY-MM-DD` | 1 day | `cache/rotation_drivers_*.json` |
| Business Cycle Rotation | — | none (instant synthesis) | no cache |
| Price Momentum | — | via OHLCV cache | `cache/ohlcv/<TICKER>.json` |
| Money Flow | — | via OHLCV cache | `cache/ohlcv/<TICKER>.json` |
| IV Rank + Directional | — | via OHLCV cache | `cache/ohlcv/<TICKER>.json` |
| IV Expression | — | reuses GEX caches | `cache/gex_*.json` |
| Cointegration | — | via OHLCV cache | `cache/ohlcv/<TICKER>.json` |
| Trades · hypotheticals · runs · recs | — | permanent | **DuckDB** `data/llm_trader.db` |

---

## Prerequisites

- Python 3.11+
- [Anthropic API key](https://console.anthropic.com/) — analyst model (synthesis) + Haiku (sentiment fallback)
- [DeepSeek API key](https://platform.deepseek.com/) — V3 for per-ticker sentiment scoring

Optional (extend coverage):
- [NewsAPI key](https://newsapi.org/) — targeted ticker/sector queries + trending detection
- [Alpha Vantage key](https://www.alphavantage.co/) — top movers + news-active ticker discovery
- [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) — macro regime + surprise index + FedWatch (free, no credit card)
- [Reddit API credentials](https://www.reddit.com/prefs/apps) — create a "script" app; free
- CFTC COT, SEC EDGAR (8-K, S-1, 13F), CBOE P/C, FINRA Reg SHO — all free, no key required

---

## Setup

```bash
pip install -r requirements.txt
```

Configure `.env`:

```env
# Required
ANTHROPIC_API_KEY=your_key
DEEPSEEK_API_KEY=your_key

# Model selection (default: Haiku; use Sonnet for higher quality)
ANALYST_MODEL=claude-haiku-4-5-20251001

# Recommended
NEWSAPI_KEY=your_key
ALPHA_VANTAGE_KEY=your_key
FRED_API_KEY=your_key      # https://fred.stlouisfed.org/docs/api/api_key.html

# Email (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your@gmail.com
SMTP_PASSWORD=your_app_password
EMAIL_RECIPIENTS=you@example.com

# Watchlist
STOCK_WATCHLIST=AAPL,MSFT,NVDA,TSLA,AMZN,META,GOOGL
SECTOR_ETFS=XLK,XLF,XLE,XLV,XLY,XLP,XLI,XLB,XLU,XLRE,XLC
COMMODITY_ETFS=GLD,SLV,IAU,GDX,PPLT,PALL,CPER

# Feature flags (all default to true)
ENABLE_FETCH_DATA=true
ENABLE_CHARTS=false
ENABLE_NEWS_SENTIMENT=true
ENABLE_TECHNICAL_ANALYSIS=true
ENABLE_INSIDER_TRADES=true
ENABLE_OPTIONS_FLOW=true
ENABLE_SEC_FILINGS=true
ENABLE_FRED=true
ENABLE_COT=true
ENABLE_8K_FILINGS=true
ENABLE_IPO_PIPELINE=true
ENABLE_ANALYST_RATINGS=true
ENABLE_VIX=true
ENABLE_PUT_CALL=true
ENABLE_EARNINGS=true
ENABLE_CREDIT=true
ENABLE_BREADTH=true
ENABLE_MCCLELLAN=true
ENABLE_HIGHS_LOWS=true
ENABLE_MACRO_SURPRISE=true
ENABLE_FEDWATCH=true
ENABLE_REVISION_MOMENTUM=true
ENABLE_EARNINGS_WHISPER=true
ENABLE_SHORT_INTEREST=true
ENABLE_GEX=true
ENABLE_TICK=true
ENABLE_VWAP=true
ENABLE_GOOGLE_TRENDS=true
ENABLE_REDDIT_SENTIMENT=true
ENABLE_SECTOR_ROTATION=true
ENABLE_ROTATION_DRIVERS=true
ENABLE_BUSINESS_CYCLE_ROTATION=true

# Reddit (required for reddit sentiment)
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret

# Insider / SEC config
INSIDER_LOOKBACK_DAYS=90
TRACKED_POLITICIANS=Nancy Pelosi,Paul Pelosi,Austin Scott,Tommy Tuberville,...
SEC_FILINGS_LOOKBACK_DAYS=30
TRACKED_INSTITUTIONS=Berkshire Hathaway,Pershing Square Capital Management,...

# Scheduler — intraday runner ticks every 30 min inside the session window (ET, Mon-Fri)
INTRADAY_SESSION_START=09:30
INTRADAY_SESSION_END=16:00
```

---

## Running

```bash
python main.py             # Run once, console output only
python main.py --email     # Run once and send email report
python main.py --schedule  # Start APScheduler (every 30 min, 9:30-16:00 ET, Mon-Fri; emails at close)
python main.py --dashboard # Launch the read-only monitoring dashboard (Plotly Dash via waitress)
```

---

## Project Structure

```
llm_trader/
├── main.py
├── requirements.txt
├── .env
├── cache/
├── logs/
├── data/                            # DuckDB database (single source of truth)
├── config/
│   └── settings.py
├── dashboard/                       # Read-only Plotly Dash monitoring app (served via waitress)
│   ├── app.py                       # 3 tabs: rationale · method performance · returns
│   ├── data.py                      # Read-only DuckDB access + retry + 60s perf cache
│   └── figures.py                   # Plotly figures (win-rate bars, equity curve)
└── src/
    ├── pipeline.py
    ├── models.py                     # All Pydantic models
    ├── utils.py
    ├── data/
    │   ├── trending.py               # Dynamic ticker discovery
    │   ├── news_fetcher.py           # RSS + NewsAPI
    │   ├── eight_k.py                # SEC 8-K material events
    │   ├── google_trends.py          # pytrends search interest spikes
    │   ├── reddit_sentiment.py       # WSB / r/stocks / r/investing
    │   ├── analyst_ratings.py        # Upgrades / downgrades / PT changes
    │   ├── earnings.py               # EPS surprises + upcoming calendar
    │   ├── short_interest.py         # FINRA Reg SHO + yfinance short data
    │   ├── market_data.py            # yfinance snapshots + rate-limit backoff
    │   ├── insider_trades.py         # House/Senate watchers + EDGAR Form 4
    │   ├── options_flow.py           # Unusual call/put sweep detection
    │   ├── sec_filings.py            # 13D/13G activist, Form 144, 13F
    │   ├── fred.py                   # FRED macro regime
    │   ├── cot.py                    # CFTC COT futures positioning
    │   ├── ipo_pipeline.py           # SEC S-1/S-11 sector demand signal
    │   ├── vix.py                    # VIX term structure + regime
    │   ├── put_call.py               # CBOE equity P/C ratio
    │   ├── credit.py                 # HYG vs SPY divergence
    │   ├── breadth.py                # % sector ETFs above 200d SMA
    │   ├── mcclellan.py              # NYSE A/D McClellan Oscillator
    │   ├── highs_lows.py             # 52-week highs/lows HL spread
    │   ├── macro_surprise.py         # CESI-style economic surprise index
    │   ├── fedwatch.py               # T-bill spread Fed rate expectations
    │   ├── revision_momentum.py      # Analyst PT/rating revision trend
    │   ├── earnings_whisper.py       # Implied whisper vs consensus
    │   ├── gamma_exposure.py         # GEX: dealer gamma positioning
    │   ├── tick.py                   # NYSE TICK index breadth exhaustion
    │   ├── sector_rotation.py        # "Ebb and Flow" per-sector money flow
    │   ├── rotation_drivers.py       # Rate-cycle phase: DFF+CPI → EASING_CYCLE|PIVOT_IMMINENT…
    │   ├── business_cycle_rotation.py # Fidelity-style economic phase → sector leadership biases
    │   └── cache.py                  # Hourly cache + incremental OHLCV
    ├── analysis/
    │   ├── sentiment.py              # DeepSeek V3 / Haiku sentiment scoring
    │   ├── technical.py              # RSI, MACD, SMA, Bollinger Bands
    │   └── claude_analyst.py         # Final recommendations (22 decision rules)
    ├── signals/
    │   ├── aggregator.py             # Weighted combination + coherence + cluster
    │   └── vwap.py                   # Rolling 20-day VWAP distance score
    ├── performance/
    │   ├── tracker.py                # Paper trades, P&L, auto-close
    │   └── daily_nav.py              # Path-faithful daily-compound NAV engine
    ├── db/
    │   ├── connection.py             # Short-lived DuckDB connections (read-write / read-only)
    │   ├── schema.py                 # Idempotent table DDL (runs, recs, trades, …)
    │   ├── repo.py                   # Read/write API (trades, runs, recommendations)
    │   └── migrate.py                # One-time JSON → DuckDB import
    ├── scheduler/
    │   └── runner.py                 # APScheduler intraday automation (every 30 min, 9:30-16:00 ET)
    ├── charts/
    │   ├── builder.py                # Plotly figures
    │   └── report.py                 # Self-contained HTML report
    └── notifications/
        └── email_sender.py           # HTML email with inline charts
```

---

## Disclaimer

This tool is for informational and educational purposes only. It does not constitute financial advice. Always conduct your own research before making investment decisions.
