# LLM Trader

An AI-powered stock analysis tool that discovers trending securities, aggregates real-time news, scores sentiment with LLMs, incorporates insider trades, politician disclosures, unusual options flow, and SEC EDGAR smart money signals, then generates high-conviction BUY/SELL signals with explicit time horizons.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  0. Ticker Discovery   — expand universe with hot/trending tickers  │
│  1. News Fetch         — RSS feeds + NewsAPI (last 24 h)            │
│  1b. SEC 8-K Filings   — material events (earnings, M&A, changes)  │
│  2. Market Data        — price snapshots via yfinance               │
│  3a. Insider Trades    — politician disclosures + EDGAR Form 4      │
│  3b. Options Flow      — unusual sweep detection via yfinance       │
│  3c. SEC Filings       — 13D/13G activist, Form 144, 13F positions  │
│  3d. FRED Macro        — yield curve, CPI, credit spreads, M2       │
│  3e. CFTC COT          — weekly speculator positioning in futures    │
│  3f. IPO Pipeline      — S-1/S-11 sector demand signal              │
│  4. Signal Aggregation — weighted combination of all active methods │
│  5. Recommendations    — Claude: BUY / SELL / HOLD / WATCH          │
│  6. Performance Track  — open trades, mark P&L, auto-close at 5 d  │
│  7. Notify             — console summary + HTML email with trades   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Algorithm

### Step 0 — Ticker Discovery (`src/data/trending.py`)

Before fetching any news the pipeline builds a **dynamic ticker universe** by merging four sources:

| Source | Method |
|---|---|
| **Yahoo Finance trending** | Public JSON endpoint — top 20 tickers currently trending on YF |
| **Alpha Vantage movers** | `TOP_GAINERS_LOSERS` API — top 5 gainers, losers, most-traded |
| **NewsAPI business headlines** | Scans top US business headlines; returns known tickers mentioned in ≥ 2 articles |
| **Alpha Vantage news feed** | `NEWS_SENTIMENT` endpoint — tickers appearing in the most news items today |

Up to **30 discovered tickers** are appended after your static `STOCK_WATCHLIST` and `SECTOR_ETFS`, giving a combined universe that is always fresh and market-relevant.

**Pinned commodities** (`COMMODITY_ETFS`) are always included in every run regardless of trending, after the dynamic discovery step.

> The static watchlist is always analysed first and is never diluted by trending additions.

---

### Step 1 — News Fetch (`src/data/news_fetcher.py`)

The fetcher pulls articles from **two layers**:

**Layer A — Public RSS feeds (no API key required)**

| Feed | Coverage |
|---|---|
| Reuters Business | General markets and macro |
| CNBC Markets | US equity and sector news |
| MarketWatch | Real-time headlines |
| Seeking Alpha | Individual stock analysis |
| Yahoo Finance | Broad market news |
| WSJ Markets | Premium financial coverage |

**Layer B — NewsAPI targeted queries (requires `NEWSAPI_KEY`)**

Two targeted queries are sent:
1. The first 10 watchlist tickers joined with `OR`
2. Sector names (e.g. "Technology OR Energy OR Financials") for ETF context

All articles are deduplicated by URL and filtered to the **last 24 hours**. Results are cached keyed by `YYYY-MM-DD_HH` — re-runs within the same hour reuse cached data instantly.

---

### Step 2 — Market Data (`src/data/market_data.py`)

Price snapshots for every ticker are fetched via **yfinance** with the following priority:

1. **Current-hour cache** — if a snapshot file for this hour exists, use it (no network call)
2. **Live fetch** — if `ENABLE_MARKET_DATA=true` and no current cache, fetch all tickers and save
3. **Historical cache fallback** — if `ENABLE_MARKET_DATA=false`, load the most recent snapshot file from any previous run
4. **News-only mode** — if no cache exists at all and market data is disabled, the pipeline continues without price context

**Rate-limit handling** — Yahoo Finance enforces a per-IP quota. When a 429 is detected:
- Exponential backoff per ticker: 60 s → 120 s → 240 s (capped at 600 s), with a live countdown log
- After 3 consecutive failures the fetch loop stops early, returning however many snapshots were collected
- Non-rate-limit errors (invalid ticker, delisted) skip that ticker immediately with no retry

---

### Step 1b — SEC 8-K Material Event Filings (`src/data/eight_k.py`)

When `ENABLE_8K_FILINGS=true`, the pipeline fetches recent 8-K filings for every ticker in the analysis universe directly from SEC EDGAR's submissions API. No API key required.

**Why 8-Ks beat RSS feeds:** Companies must file within 4 business days of the triggering event. EDGAR receives the filing before most financial news outlets publish their coverage, giving a structural time advantage on catalysts.

**Material items tracked:**

| Item | Event |
|---|---|
| 1.01 | Entry into Material Definitive Agreement |
| 1.02 | Termination of Material Agreement |
| 1.03 | Bankruptcy or Receivership |
| 1.05 | Material Cybersecurity Incident |
| 2.01 | Completion of Acquisition or Disposition |
| 2.02 | Results of Operations — **Earnings Release** |
| 2.05 | Costs Associated with Exit/Disposal — Restructuring/Layoffs |
| 2.06 | Material Impairment |
| 3.01 | Notice of Delisting |
| 3.02 | Unregistered Sales of Securities (Dilution) |
| 4.01 | Change in Certifying Accountant |
| 4.02 | Non-Reliance on Previously Issued Financial Statements — **Restatement** |
| 5.01 | Change in Control of Registrant |
| 5.02 | Departure or Appointment of Principal Officers/Directors |

Exhibit-only items (9.01) and pure compliance items (5.03, 5.08, 1.04) are filtered out.

**Integration:** 8-K filings are converted to `NewsArticle` objects and injected directly into the news feed **before** sentiment scoring. The DeepSeek/Haiku LLM then scores them alongside RSS and NewsAPI articles. No separate pipeline stage or prompt change is needed — a restatement 8-K naturally scores -0.8, an acquisition completion scores +0.4, and earnings releases get scored on content context.

**Always fetched fresh** — unlike RSS news (cached hourly), 8-K articles are fetched on every run so same-day filings are never missed even on pipeline re-runs.

Tickers with no CIK in the SEC master list (some ETFs, indices) are skipped silently.

---

### Step 3a — Insider & Politician Trades (`src/data/insider_trades.py`)

When `ENABLE_INSIDER_TRADES=true`, the pipeline fetches recent buying and selling activity from three sources:

| Source | Data |
|---|---|
| **House Stock Watcher** | US House representatives' stock disclosures |
| **Senate Stock Watcher** | US Senate members' stock disclosures |
| **SEC EDGAR Form 4** | Corporate insider filings (officers, directors, >10% holders) |

Trades are filtered to the last `INSIDER_LOOKBACK_DAYS` (default: 90 days). Politician trades are further filtered to a configurable list of `TRACKED_POLITICIANS` — names considered to have informational edge.

**Ticker expansion** — if a trade involves a ticker not already in the universe, it is added automatically before signal building.

Each trade record includes: ticker, trader name, role, transaction type (purchase/sale), amount range, and date.

---

### Step 3b — Unusual Options Flow (`src/data/options_flow.py`)

When `ENABLE_OPTIONS_FLOW=true`, the pipeline scans near-term options chains via yfinance for **institutional sweep activity**. No API key required.

For each ticker, expirations ≤ 60 days out are scanned. A contract is flagged as a sweep when:

| Filter | Threshold |
|---|---|
| Volume / Open Interest ratio | ≥ 2× (sweep-like institutional flow) |
| Out-of-the-money | ≥ 1% |
| Notional premium | ≥ $25,000 |

- **Call sweeps** → bullish signal
- **Put sweeps** → bearish signal

Each sweep is emitted as an `InsiderTrade` object with `trader_type="options_flow"`, including strike, expiry, vol/OI ratio, and notional size.

---

### Step 3c — SEC EDGAR Filings (`src/data/sec_filings.py`)

When `ENABLE_SEC_FILINGS=true`, the pipeline pulls three categories of smart money signals directly from the SEC's public EDGAR system. No API key required.

**Strategy 1 — SC 13D / SC 13G (Activist & Institutional Stakes)**

Fetches all recent filings disclosing >5% ownership in a company within `SEC_FILINGS_LOOKBACK_DAYS` (default: 30 days). 13D filers are classified as "Activist Investor", 13G as "Passive Institutional". Tickers are discovered from the filings themselves via the SEC company tickers index — not limited to any predefined watchlist.

**Strategy 2 — Form 144 (Planned Insider Sales)**

Fetches all recent Form 144 filings — the pre-sale disclosure filed by officers and directors before selling restricted shares. These are bearish signals indicating planned insider distribution.

**Strategy 3 — Form 13F-HR (Superinvestor Quarterly Holdings)**

For each institution in `TRACKED_INSTITUTIONS`, the pipeline:
1. Resolves the SEC CIK dynamically from EDGAR (no manual lookup needed)
2. Downloads the two most recent 13F-HR filings
3. Diffs holdings quarter-over-quarter to detect meaningful changes (>10% position size change)
4. Emits signals for: new positions, exits, significant increases, and significant decreases

Tickers are discovered from the 13F holdings themselves — adds stocks to the analysis universe beyond the static watchlist.

---

### Step 3d — FRED Macro Context (`src/data/fred.py`)

When `ENABLE_FRED=true`, the pipeline fetches macro regime indicators from the St. Louis Fed FRED API. Requires a free API key (no credit card). All data is public.

**Series fetched:**

| Series | Indicator | Frequency |
|---|---|---|
| `T10Y2Y` | 10Y-2Y Treasury yield spread | Daily |
| `DFF` | Effective Federal Funds Rate | Daily |
| `CPIAUCSL` | CPI (YoY computed) | Monthly |
| `UNRATE` | Unemployment rate + trend | Monthly |
| `BAMLH0A0HYM2` | HY OAS credit spread | Daily |
| `BAMLC0A0CM` | IG OAS credit spread | Daily |
| `M2SL` | M2 money supply (YoY computed) | Monthly |

**Derived regime signals:**

| Signal | Labels |
|---|---|
| Yield curve | `INVERTED` (<-0.25%) / `FLAT` / `NORMAL` / `STEEP` |
| Inflation | `HIGH` (>5%) / `ELEVATED` (>3%) / `MODERATE` / `LOW` |
| Credit | `STRESSED` (HY >6%) / `ELEVATED` / `NORMAL` / `TIGHT` |
| Unemployment trend | `RISING` / `STABLE` / `FALLING` |

**Overall macro regime** is derived from the combination of yield curve, credit conditions, and unemployment trend:

| Regime | Conditions |
|---|---|
| `RECESSION` | Inverted curve + rising unemployment |
| `LATE_CYCLE` | Inverted/flat curve + stressed/elevated credit |
| `SLOWDOWN` | Normal curve + elevated credit or rising unemployment |
| `EXPANSION` | Normal/steep curve + normal/tight credit + stable employment |

**How it affects recommendations:** The macro context is injected directly into the Claude prompt as a labeled `<macro_context>` block. Claude applies the regime overlay to every BUY/SELL call:
- `RECESSION`: favors shorts, avoids POSITION-horizon longs
- `LATE_CYCLE`: prefers recession-resistant names and SWING horizons
- `SLOWDOWN`: tilts bearish on cyclicals, constructive on defensives and gold
- `EXPANSION`: macro tailwind raises conviction on longs

The macro dashboard also appears as a dedicated card in the HTML email with color-coded indicators for each FRED series.

---

### Step 3e — CFTC Commitment of Traders (`src/data/cot.py`)

When `ENABLE_COT=true`, the pipeline downloads the weekly CFTC COT report. No API key required — data is public and updated every Friday.

**Two reports downloaded:**

| Report | Coverage | Speculator proxy |
|---|---|---|
| **Disaggregated Futures Only** | Physical commodities | Managed Money (hedge funds) |
| **Traders in Financial Futures (TFF)** | S&P 500, Nasdaq futures | Leveraged Money (hedge funds) |

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
| S&P 500 | SPY |
| Nasdaq 100 | QQQ |

**Signal logic:**

Net speculator position = (longs − shorts) / open interest × 100. This is ranked within the last 52 weekly observations to produce a percentile:

| Percentile | Signal | Applied direction |
|---|---|---|
| ≥ 80th | `EXTREME_LONG` | **BEARISH** (contrarian — specs crowded, reversal risk high) |
| 60–79th | `BULLISH_TREND` | BULLISH (momentum) |
| 40–59th | `NEUTRAL` | NEUTRAL |
| 20–39th | `BEARISH_TREND` | BEARISH (momentum) |
| ≤ 20th | `EXTREME_SHORT` | **BULLISH** (contrarian — specs max short, coiled for squeeze) |

Week-over-week change in net position is also tracked as a trend confirmation.

**Results are cached by ISO week** — re-runs within the same week skip the download entirely. The current + previous year's files are fetched to ensure a full 52-week history even early in the calendar year.

**How it affects recommendations:** COT data is injected into the Claude prompt as a `<cot_context>` table. Claude uses it as a medium-term overlay: extreme readings cap conviction on positions in the crowded direction, while confirmed momentum builds. COT alone is never sufficient for a BUY/SELL — it must converge with at least one other signal layer.

---

### Step 3f — IPO Pipeline (`src/data/ipo_pipeline.py`)

When `ENABLE_IPO_PIPELINE=true`, the pipeline fetches recent S-1 and S-11 registration statements from EDGAR. No API key required. Results are cached daily.

**What S-1/S-11 filings signal:** Institutional underwriters only launch the IPO window when their real-money clients (hedge funds, pension funds, asset managers) have committed to buy. A cluster of S-1 filings in a sector is a revealed-preference signal that institutional capital is actively flowing there — 4 to 12 weeks ahead of when it shows up in sector ETF flows.

**Forms tracked:**

| Form | Meaning |
|---|---|
| S-1 | Initial IPO registration |
| S-1/A | Amendment — company is actively progressing toward listing |
| S-11 | REIT / real estate trust IPO |
| S-11/A | REIT amendment |

**Sector classification:** Two-tier — SIC code (from EDGAR) when available, company name keyword matching as fallback. S-11/S-11/A filings are always classified as Real Estate.

**What goes to Claude:**
- Sector breakdown table (filing counts per sector)
- Recent initial registrations (last 10)
- Amendment count (pipeline maturity signal)
- Overall IPO market temperature: `highly active` (≥20 new) / `moderately active` (≥10) / `modest` / `quiet`

Claude uses this as a **confirming layer** — if news is already bullish on XLK and Technology has the most S-1s, that convergence raises conviction. A cold IPO market tempers aggressive growth-sector BUYs even when news is positive.

**Email:** Sector bar chart with highlighted top-3 sectors + recent filings table.

---

### Step 4 — Signal Aggregation (`src/signals/aggregator.py`)

Signals are built by combining **up to three methods**, with weights that adjust automatically based on which are enabled:

| Active Methods | News | Technical | Insider |
|---|---|---|---|
| All three | 50% | 30% | 20% |
| News + Technical | 60% | 40% | — |
| News + Insider | 70% | — | 30% |
| Technical + Insider | — | 60% | 40% |
| Single method | 100% | 100% | 100% |

**Method 1 — News sentiment** — the most recent 20 relevant articles are sent to DeepSeek V3 (Haiku fallback) for a float score in `[−1.0, +1.0]`.

**Method 2 — Technical analysis** — RSI, MACD, SMA 20/50, Bollinger Bands computed from OHLCV cache; returns a `[−1.0, +1.0]` score.

**Method 3 — Insider trades** — net buying vs. selling weighted by dollar-amount tier; normalised to `[−1.0, +1.0]`. Incorporates all smart money sources (insider trades, options flow, SEC filings).

**Combined score → signal:**
```
combined ≥ +0.15   →  BULLISH
combined ≤ −0.15   →  BEARISH
otherwise          →  NEUTRAL

confidence = min(1.0, |combined| / 0.5)
```

A combined score of ±0.50 maps to 100% confidence.

---

### Step 5 — Final Recommendations (`src/analysis/claude_analyst.py`)

All ticker signals — plus the raw smart money context — are passed in a single prompt to the configured **analyst model** (default: `claude-haiku-4-5-20251001`, configurable via `ANALYST_MODEL`), which acts as a high-conviction portfolio manager.

**What the model is asked to do:**
- Find the **3–5 best opportunities** across the full universe — both longs and shorts
- **Must produce at least one BUY and one SELL** per day if the news supports it
- Classify every signal by **time horizon**: `SWING` (2–10 d) / `SHORT-TERM` (1–4 wk) / `POSITION` (1–3 mo)
- Apply **conviction discipline**: confidence ≥ 0.78 required for BUY/SELL; 0.50–0.77 → HOLD; < 0.50 → WATCH
- Short-selling rules: only SELL when catalyst is unambiguous, uncontested, and market is not in capitulation

**Output per ticker:**
- `action`: BUY / SELL / HOLD / WATCH
- `direction`: BULLISH / BEARISH / NEUTRAL
- `confidence`: 0.0 – 1.0
- `time_horizon`: SWING / SHORT-TERM / POSITION / N/A
- `rationale`: catalyst → price mechanism → time horizon + risk

Only `BUY` and `SELL` actions with confidence ≥ 78% are considered **actionable** and flow to the performance tracker and email.

---

### Step 6 — Performance Tracking (`src/performance/tracker.py`)

Every actionable signal is recorded in `cache/trades.json` as a paper trade:

1. **Open** — entry price fetched at recommendation time via yfinance
2. **Update** — each daily run refreshes current price and calculates unrealised P&L
3. **Auto-close** — positions are automatically closed after **5 calendar days**

P&L direction is sign-aware:
- `BUY` → profit when price rises
- `SELL` (short) → profit when price falls

**Metrics tracked:** win rate, average return, best/worst trade — logged every run and included in the email report.

---

### Step 7 — Charts & Notification (`src/charts/`, `src/notifications/email_sender.py`)

**Interactive HTML report** — saved to `logs/report_YYYY-MM-DD_HHMM.html` every run (no server required, open in any browser):

| Chart | Content |
|---|---|
| **Signals overview** | Horizontal bar chart of all tickers — BUY bars right (green), SELL bars left (red), 75% conviction threshold line |
| **Stock chart** (per BUY/SELL ticker) | 4-panel: Candlestick + SMA 20/50 + EMA 9 + Bollinger Bands / Volume / RSI 14 (30/70 lines) / MACD histogram + signal |
| **Equity curve** | Cumulative P&L area chart + per-trade return bars, with win rate and average return in the title |

**Email** — charts are embedded as **inline base64 PNG images** (no attachments, no external links). The email includes:
- BUY/SELL signal cards with rationale and chart
- **Smart Money section** — for any ticker with a BUY/SELL signal, shows all smart money signals (insider trades, options sweeps, SEC filings) with trader name, role, signal type, amount, and date
- Performance summary (win rate, avg return, best/worst trade)

Email is sent only if `SMTP_USER` and `EMAIL_RECIPIENTS` are configured. Degrades gracefully to text-only if `kaleido` is not installed. The smart money section only appears when data is available.

**OHLCV cache** (`cache/ohlcv/<TICKER>.json`) — chart data is cached per ticker and updated incrementally:

| State | Behaviour |
|---|---|
| No cache yet | Fetches full 3-month history, saves to disk |
| Cache exists, up to date | Returns cache immediately — no network call |
| Cache exists, missing recent days | Fetches only the missing date range, appends and saves |
| `ENABLE_MARKET_DATA=false` | Reads cache only — yfinance is never called |
| yfinance errors | Falls back to whatever is already in cache |

---

## Model Routing Summary

| Task | Model | Fallback |
|---|---|---|
| Per-ticker sentiment scoring | DeepSeek V3 (`deepseek-chat`) | Claude Haiku (`claude-haiku-4-5-20251001`) |
| Technical analysis scoring | Computed locally (RSI, MACD, SMA, BB) | — |
| Final synthesis / recommendations | Configurable via `ANALYST_MODEL` (default: `claude-haiku-4-5-20251001`) | Rule-based fallback |

---

## Local Cache

```
cache/
├── news_YYYY-MM-DD_HH.json          # articles fetched that hour
├── snapshots_YYYY-MM-DD_HH.json     # price snapshots fetched that hour
├── ohlcv/
│   ├── AAPL.json                    # incremental OHLCV history per ticker
│   ├── GLD.json
│   └── …
└── trades.json                      # paper trade ledger (open + closed)
```

| Cache | Key | TTL |
|---|---|---|
| News | `YYYY-MM-DD_HH` | 1 hour |
| Snapshots | `YYYY-MM-DD_HH` | 1 hour |
| OHLCV (charts) | per ticker | incremental — only missing days fetched |
| Trades | — | permanent ledger |

---

## Prerequisites

- Python 3.11+
- [Anthropic API key](https://console.anthropic.com/) — analyst model (synthesis) + Haiku (sentiment fallback)
- [DeepSeek API key](https://platform.deepseek.com/) — V3 for per-ticker sentiment scoring

Optional (extend coverage):
- [NewsAPI key](https://newsapi.org/) — targeted ticker/sector queries + trending detection
- [Alpha Vantage key](https://www.alphavantage.co/) — top movers + news-active ticker discovery
- [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) — macro regime context (yield curve, CPI, credit spreads, M2) — free, no credit card
- **CFTC COT** — no key required; weekly futures positioning data fetched directly from `cftc.gov`
- **SEC 8-K** — no key required; material event filings fetched directly from EDGAR
- **SEC S-1/S-11** — no key required; IPO pipeline fetched directly from EDGAR

---

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure `.env`**

```env
# Required
ANTHROPIC_API_KEY=your_anthropic_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key

# Model selection (default: claude-haiku-4-5-20251001, use claude-sonnet-4-6 for higher quality)
ANALYST_MODEL=claude-haiku-4-5-20251001

# Recommended — extends news coverage and trending discovery
NEWSAPI_KEY=your_newsapi_key
ALPHA_VANTAGE_KEY=your_alpha_vantage_key

# FRED (Federal Reserve macro context — free key, no credit card)
FRED_API_KEY=your_fred_api_key            # https://fred.stlouisfed.org/docs/api/api_key.html

# Email reports (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password     # Gmail: use an App Password
EMAIL_RECIPIENTS=you@example.com,partner@example.com

# Watchlist — base tickers always analysed
STOCK_WATCHLIST=AAPL
SECTOR_ETFS=XLK,XLF,XLE,XLV,XLY,XLP,XLI,XLB,XLU,XLRE,XLC

# Commodities — always included in every run (pinned, never dropped by trending)
COMMODITY_ETFS=GLD,SLV,IAU,GDX,PPLT,PALL,CPER

# Feature flags
ENABLE_MARKET_DATA=true           # false = news-only mode (uses historical cache for charts)
ENABLE_CHARTS=false               # true = generate Plotly charts and HTML report
ENABLE_NEWS_SENTIMENT=true        # method 1: LLM sentiment from news/RSS
ENABLE_TECHNICAL_ANALYSIS=true    # method 2: RSI, MACD, SMA, Bollinger Bands
ENABLE_INSIDER_TRADES=true        # method 3: politician + corporate insider trades (House/Senate/Form 4)
ENABLE_OPTIONS_FLOW=true          # method 4: unusual options sweep detection (yfinance, no key needed)
ENABLE_SEC_FILINGS=true           # method 5: SEC EDGAR 13D/13G activist, Form 144, 13F superinvestors
ENABLE_FRED=true                  # macro regime overlay: yield curve, CPI, credit spreads, M2 (requires FRED_API_KEY)
ENABLE_COT=true                   # CFTC COT futures positioning (no API key; cached weekly)
ENABLE_8K_FILINGS=true            # SEC 8-K material events (no API key; always fresh)
EIGHT_K_LOOKBACK_DAYS=5           # fetch 8-Ks filed in the last N days
ENABLE_IPO_PIPELINE=true          # SEC S-1/S-11 sector demand signal (no API key; cached daily)
IPO_LOOKBACK_DAYS=30              # S-1 filings accumulate over weeks; 30 days gives full picture

# Insider trades config (optional)
INSIDER_LOOKBACK_DAYS=90
TRACKED_POLITICIANS=Nancy Pelosi,Paul Pelosi,Austin Scott,Tommy Tuberville,...

# SEC EDGAR filings config (optional)
SEC_FILINGS_LOOKBACK_DAYS=30      # lookback window for 13D/13G and Form 144 filings
TRACKED_INSTITUTIONS=Berkshire Hathaway,Pershing Square Capital Management,...

# Scheduler (cron syntax, US/Eastern)
SCHEDULE_DAILY=0 8 * * 1-5
```

---

## Running

```bash
# Run once
python main.py

# Start daily scheduler (8:00 AM ET, Mon–Fri)
python main.py --schedule
```

---

## Output Example

```
============================================================
  ACTIONABLE SIGNALS  (2026-04-09 08:22 EDT)
============================================================

  STOCKS
  ----------------------------------------
  ▼ USAR   SELL   conf=90%  [SWING]
    Near-maximum combined confidence at 98% with strongly negative
    news sentiment (-0.70). Multiple converging signals: risk-off macro,
    Iran conflict escalation, strategist 'grind lower' warnings.
    Key risk: sudden geopolitical de-escalation or short squeeze.

  ETFs / MARKETS
  ----------------------------------------
  ▲ XLE    BUY    conf=95%  [SWING]
    Maximum combined confidence at 100% driven by Iran conflict →
    direct oil supply shock risk premium. Goldman Sachs supply-shock
    analysis reinforces the fundamental case.
    Key risk: rapid diplomatic resolution.

  SMART MONEY SIGNALS
  ----------------------------------------
  GLD     [+] Unusual CALL Sweep (Options Sweep — CALL)
  XLE     [+] 13D Activist Stake (Elliott Management), [+] Unusual CALL Sweep
  NVDA    [-] Planned Sale (Officer/Director Form 144)
============================================================
```

---

## Project Structure

```
llm_trader/
├── main.py                     # Entry point (--schedule flag)
├── requirements.txt
├── .env                        # API keys and config (not committed)
├── cache/                      # News + snapshot cache + trades.json
├── logs/                       # Rotating daily logs (7-day retention)
├── config/
│   └── settings.py             # .env loader
└── src/
    ├── pipeline.py             # Main orchestration (all 7 steps)
    ├── models.py               # Pydantic models: NewsArticle, TickerSignal, Recommendation, InsiderTrade
    ├── utils.py                # Shared utilities (Eastern timezone helpers)
    ├── data/
    │   ├── trending.py         # Dynamic ticker discovery (Yahoo, AV, NewsAPI)
    │   ├── news_fetcher.py     # RSS + NewsAPI aggregator
    │   ├── market_data.py      # yfinance snapshots + rate-limit backoff
    │   ├── insider_trades.py   # House/Senate watchers + EDGAR Form 4 filings
    │   ├── options_flow.py     # Unusual options sweep detection (yfinance, no key)
    │   ├── sec_filings.py      # SEC EDGAR: 13D/13G activist, Form 144, 13F superinvestors
    │   ├── fred.py             # FRED macro regime: yield curve, CPI, credit spreads, M2
    │   ├── cot.py              # CFTC COT: weekly speculator positioning (9 contracts, cached by week)
    │   ├── eight_k.py          # SEC 8-K: material event filings → NewsArticle objects
    │   ├── ipo_pipeline.py     # SEC S-1/S-11: IPO pipeline sector intelligence (cached daily)
    │   └── cache.py            # Hourly cache + incremental OHLCV cache
    ├── analysis/
    │   ├── sentiment.py        # DeepSeek V3 / Haiku sentiment scoring
    │   ├── technical.py        # RSI, MACD, SMA, Bollinger Bands scoring
    │   └── claude_analyst.py   # Analyst model final recommendations
    ├── signals/
    │   └── aggregator.py       # Weighted combination of news + technical + insider
    ├── performance/
    │   └── tracker.py          # Paper trade recording, P&L, auto-close
    ├── scheduler/
    │   └── runner.py           # APScheduler daily automation
    ├── charts/
    │   ├── builder.py          # Plotly figures: stock chart, signals overview, equity curve
    │   └── report.py           # Self-contained interactive HTML report → logs/
    └── notifications/
        └── email_sender.py     # HTML email with inline charts + smart money section
```

---

## Disclaimer

This tool is for informational and educational purposes only. It does not constitute financial advice. Always conduct your own research before making investment decisions.