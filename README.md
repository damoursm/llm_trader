# LLM Trader

An AI-powered stock analysis tool that discovers trending securities, aggregates real-time news, scores sentiment with LLMs, incorporates insider & politician trades, and generates high-conviction BUY/SELL signals with explicit time horizons.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  0. Ticker Discovery   — expand universe with hot/trending tickers  │
│  1. News Fetch         — RSS feeds + NewsAPI (last 24 h)            │
│  2. Market Data        — price snapshots via yfinance               │
│  3. Insider Trades     — politician disclosures + EDGAR Form 4      │
│  4. Signal Aggregation — weighted combination of all active methods │
│  5. Recommendations    — Claude Sonnet: BUY / SELL / HOLD / WATCH   │
│  6. Performance Track  — open trades, mark P&L, auto-close at 5 d   │
│  7. Notify             — console summary + HTML email with trades    │
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

### Step 3 — Insider & Politician Trades (`src/data/insider_trades.py`)

When `ENABLE_INSIDER_TRADES=true`, the pipeline fetches recent buying and selling activity from three sources:

| Source | Data |
|---|---|
| **House Stock Watcher** | US House representatives' stock disclosures |
| **Senate Stock Watcher** | US Senate members' stock disclosures |
| **SEC EDGAR Form 4** | Corporate insider filings (officers, directors, >10% holders) |

Trades are filtered to the last `INSIDER_LOOKBACK_DAYS` (default: 90 days). Politician trades are further filtered to a configurable list of `TRACKED_POLITICIANS` — names considered to have informational edge.

**Ticker expansion** — if a politician trade involves a ticker not already in the universe, it is added automatically before signal building.

Each trade record includes: ticker, trader name, role, transaction type (purchase/sale), amount range, and date.

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

**Method 3 — Insider trades** — net buying vs. selling weighted by dollar-amount tier; normalised to `[−1.0, +1.0]`.

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

All ticker signals — plus the raw insider trade context — are passed in a single prompt to **Claude Sonnet** (`claude-sonnet-4-6`), which acts as a high-conviction portfolio manager.

**What Claude is asked to do:**
- Find the **3–5 best opportunities** across the full universe — both longs and shorts
- **Must produce at least one BUY and one SELL** per day if the news supports it
- Classify every signal by **time horizon**: `SWING` (2–10 d) / `SHORT-TERM` (1–4 wk) / `POSITION` (1–3 mo)
- Apply **conviction discipline**: confidence ≥ 0.75 required for BUY/SELL; 0.50–0.74 → HOLD; < 0.50 → WATCH
- Short-selling rules: only SELL when catalyst is unambiguous, uncontested, and market is not in capitulation

**Output per ticker:**
- `action`: BUY / SELL / HOLD / WATCH
- `direction`: BULLISH / BEARISH / NEUTRAL
- `confidence`: 0.0 – 1.0
- `time_horizon`: SWING / SHORT-TERM / POSITION / N/A
- `rationale`: catalyst → price mechanism → time horizon + risk

Only `BUY` and `SELL` actions are considered **actionable** and flow to the performance tracker and email.

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
- **Insider & Politician Trades section** — for any ticker with a BUY/SELL signal, shows recent insider/politician trades (trader name, role, buy/sell, amount range, date)
- Performance summary (win rate, avg return, best/worst trade)

Email is sent only if `SMTP_USER` and `EMAIL_RECIPIENTS` are configured. Degrades gracefully to text-only if `kaleido` is not installed. The insider trades section only appears when data is available.

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
| Per-ticker sentiment scoring | DeepSeek V3 (`deepseek-chat`) | Claude Haiku (`claude-haiku-4-5`) |
| Technical analysis scoring | Computed locally (RSI, MACD, SMA, BB) | — |
| Final synthesis / recommendations | Claude Sonnet (`claude-sonnet-4-6`) | Rule-based fallback |

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
- [Anthropic API key](https://console.anthropic.com/) — Claude Sonnet (synthesis) + Haiku (sentiment fallback)
- [DeepSeek API key](https://platform.deepseek.com/) — V3 for per-ticker sentiment scoring

Optional (extend coverage):
- [NewsAPI key](https://newsapi.org/) — targeted ticker/sector queries + trending detection
- [Alpha Vantage key](https://www.alphavantage.co/) — top movers + news-active ticker discovery

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

# Recommended — extends news coverage and trending discovery
NEWSAPI_KEY=your_newsapi_key
ALPHA_VANTAGE_KEY=your_alpha_vantage_key

# Email reports (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password     # Gmail: use an App Password
EMAIL_RECIPIENTS=you@example.com,partner@example.com

# Watchlist — base tickers always analysed
STOCK_WATCHLIST=AAPL
SECTOR_ETFS=XLK,XLF,XLE,XLV,XLY,XLP,XLI,XLB,XLU,XLRE,XLC,GLD,SLV

# Feature flags
ENABLE_MARKET_DATA=true           # false = news-only mode (uses historical cache for charts)
ENABLE_CHARTS=false               # true = generate Plotly charts and HTML report
ENABLE_NEWS_SENTIMENT=true        # method 1: LLM sentiment from news/RSS
ENABLE_TECHNICAL_ANALYSIS=true    # method 2: RSI, MACD, SMA, Bollinger Bands
ENABLE_INSIDER_TRADES=true        # method 3: politician + corporate insider trades

# Insider trades config (optional)
INSIDER_LOOKBACK_DAYS=90
TRACKED_POLITICIANS=Nancy Pelosi,Paul Pelosi,Austin Scott,Tommy Tuberville,...

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
  ACTIONABLE SIGNALS  (2026-03-29 16:22 UTC)
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

  ▼ ES=F   SELL   conf=88%  [SWING]
    Near-maximum combined confidence (98%). BofA strategist 'grind lower'
    + Wall Street consensus + U.S. ground troops in Iran = acute risk-off
    pressure on S&P 500 futures.
    Key risk: central bank intervention.
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
    ├── data/
    │   ├── trending.py         # Dynamic ticker discovery (Yahoo, AV, NewsAPI)
    │   ├── news_fetcher.py     # RSS + NewsAPI aggregator
    │   ├── market_data.py      # yfinance snapshots + rate-limit backoff
    │   ├── insider_trades.py   # House/Senate watchers + EDGAR Form 4 filings
    │   └── cache.py            # Hourly cache + incremental OHLCV cache
    ├── analysis/
    │   ├── sentiment.py        # DeepSeek V3 / Haiku sentiment scoring
    │   ├── technical.py        # RSI, MACD, SMA, Bollinger Bands scoring
    │   └── claude_analyst.py   # Claude Sonnet final recommendations
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
        └── email_sender.py     # HTML email with inline charts + insider trades section
```

---

## Disclaimer

This tool is for informational and educational purposes only. It does not constitute financial advice. Always conduct your own research before making investment decisions.
