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
│  4.  Signal Aggregation   — weighted combination with coherence scoring  │
│  5.  Recommendations      — Claude: BUY / SELL / HOLD / WATCH           │
│  6.  Performance Tracking — paper trades, P&L, auto-close at 5 days     │
│  7.  Charts + Email       — HTML report + inline-chart email             │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

### Step 0 — Ticker Discovery (`src/data/trending.py`)

Builds a **dynamic ticker universe** by merging four sources before any news is fetched:

| Source | Method |
|---|---|
| Yahoo Finance trending | Public JSON endpoint — top 20 tickers currently trending |
| Alpha Vantage movers | `TOP_GAINERS_LOSERS` — top 5 gainers, losers, most-traded |
| NewsAPI business headlines | Scans top US business headlines; returns tickers in ≥2 articles |
| Alpha Vantage news feed | `NEWS_SENTIMENT` — tickers in the most news items today |

Up to 30 discovered tickers are appended after the static `STOCK_WATCHLIST` and `SECTOR_ETFS`. **Pinned commodities** (`COMMODITY_ETFS`) are always included in every run regardless of trending.

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

---

### Step 3a — Insider & Politician Trades (`src/data/insider_trades.py`)

When `ENABLE_INSIDER_TRADES=true`, fetches from three sources filtered to the last `INSIDER_LOOKBACK_DAYS` (default: 90 days):

| Source | Data |
|---|---|
| House Stock Watcher | US House representatives' stock disclosures |
| Senate Stock Watcher | US Senate members' stock disclosures |
| SEC EDGAR Form 4 | Corporate insider filings (officers, directors, >10% holders) |

Politician trades are filtered to `TRACKED_POLITICIANS`. Newly discovered tickers are added to the analysis universe automatically.

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
For each institution in `TRACKED_INSTITUTIONS`, diffs the two most recent 13F filings quarter-over-quarter to detect new positions, exits, and significant size changes (>10%). CIKs are resolved dynamically from EDGAR — no manual lookup needed. Carries a ~45-day reporting lag.

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

---

### Step 4 — Signal Aggregation (`src/signals/aggregator.py`)

Combines up to seven signal methods with dynamically normalized weights:

| Method | Base weight | Source |
|---|---|---|
| News sentiment | 40% | DeepSeek V3 LLM scoring of all article-type sources |
| Technical analysis | 30% | RSI, MACD, SMA20/50, Bollinger Bands |
| Smart money / insider | 30% | Insider trades + options flow + SEC filings |
| Put/call ratio | 15% | Per-ticker CBOE options volume |
| Max pain gravity | 12% | GEX options chain, expiry-decay weighted |
| OI-weighted skew | 15% | GEX call/put OI directional lean |
| VWAP distance | 12% | Price vs. rolling 20-day VWAP (mean-reversion) |

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

---

### Step 6 — Performance Tracking (`src/performance/tracker.py`)

Every actionable signal is recorded in `cache/trades.json`:

1. **Open** — entry price fetched at recommendation time
2. **Update** — each daily run refreshes current price and unrealised P&L
3. **Auto-close** — positions are automatically closed after **5 calendar days**

P&L is sign-aware: BUY profits when price rises; SELL (short) profits when price falls. Metrics: win rate, average return, best/worst trade.

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
| Portfolio | Win rate, P&L curve, open/closed trades |

**Email** — charts embedded as inline base64 PNG (no attachments). Degrades gracefully to text-only if `kaleido` is not installed.

---

## Model Routing

| Task | Model | Fallback |
|---|---|---|
| Per-ticker sentiment scoring | DeepSeek V3 (`deepseek-chat`) | Claude Haiku |
| Technical analysis scoring | Computed locally (RSI, MACD, SMA, BB) | — |
| Final synthesis / BUY/SELL/HOLD/WATCH | Configurable via `ANALYST_MODEL` (default: `claude-haiku-4-5-20251001`) | Rule-based fallback |

To use Sonnet for higher quality: set `ANALYST_MODEL=claude-sonnet-4-6` in `.env`.

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
| Trades ledger | — | permanent | `cache/trades.json` |

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

# Reddit (required for reddit sentiment)
REDDIT_CLIENT_ID=your_id
REDDIT_CLIENT_SECRET=your_secret

# Insider / SEC config
INSIDER_LOOKBACK_DAYS=90
TRACKED_POLITICIANS=Nancy Pelosi,Paul Pelosi,Austin Scott,Tommy Tuberville,...
SEC_FILINGS_LOOKBACK_DAYS=30
TRACKED_INSTITUTIONS=Berkshire Hathaway,Pershing Square Capital Management,...

# Scheduler (cron, US/Eastern)
SCHEDULE_DAILY=0 8 * * 1-5
```

---

## Running

```bash
python main.py             # Run once, console output only
python main.py --email     # Run once and send email report
python main.py --schedule  # Start APScheduler (8:00 AM ET, Mon-Fri)
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
├── config/
│   └── settings.py
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
    │   └── cache.py                  # Hourly cache + incremental OHLCV
    ├── analysis/
    │   ├── sentiment.py              # DeepSeek V3 / Haiku sentiment scoring
    │   ├── technical.py              # RSI, MACD, SMA, Bollinger Bands
    │   └── claude_analyst.py         # Final recommendations (22 decision rules)
    ├── signals/
    │   ├── aggregator.py             # Weighted combination + coherence + cluster
    │   └── vwap.py                   # Rolling 20-day VWAP distance score
    ├── performance/
    │   └── tracker.py                # Paper trades, P&L, auto-close
    ├── scheduler/
    │   └── runner.py                 # APScheduler daily automation
    ├── charts/
    │   ├── builder.py                # Plotly figures
    │   └── report.py                 # Self-contained HTML report
    └── notifications/
        └── email_sender.py           # HTML email with inline charts
```

---

## Disclaimer

This tool is for informational and educational purposes only. It does not constitute financial advice. Always conduct your own research before making investment decisions.
