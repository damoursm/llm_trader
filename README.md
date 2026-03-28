# LLM Trader

An AI-powered stock analysis tool that discovers trending securities, aggregates real-time news, scores sentiment with LLMs, and generates high-conviction BUY/SELL signals with explicit time horizons.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  0. Ticker Discovery   — expand universe with hot/trending tickers  │
│  1. News Fetch         — RSS feeds + NewsAPI (last 24 h)            │
│  2. Market Data        — price snapshots via yfinance               │
│  3. Sentiment Scoring  — DeepSeek V3 per ticker (Haiku fallback)    │
│  4. Signal Aggregation — score → direction + confidence             │
│  5. Recommendations    — Claude Sonnet: BUY / SELL / HOLD / WATCH   │
│  6. Performance Track  — open trades, mark P&L, auto-close at 5 d   │
│  7. Notify             — console summary + optional HTML email       │
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

Real-time price snapshots for every ticker in the universe are fetched via **yfinance**. The same hourly cache applies — no redundant network calls during development or repeated runs.

---

### Step 3 — Sentiment Scoring (`src/analysis/sentiment.py`)

For each ticker, the most recent **20 relevant articles** are selected (filtered by ticker name and known aliases), then sent to an LLM for scoring.

**Prompt task:** "Analyse these headlines and give a float score −1.0 (very bearish) to +1.0 (very bullish) and a one-sentence rationale citing the specific catalyst."

**Model routing:**

| Model | Role | Fallback |
|---|---|---|
| **DeepSeek V3** (`deepseek-chat`) | Primary — fast and cheap per-ticker scoring | |
| **Claude Haiku** (`claude-haiku-4-5`) | Fallback if DeepSeek is unavailable or errors | |

Each call returns:
- `score`: float in `[−1.0, +1.0]`
- `rationale`: 1–3 sentences explaining the catalyst

Articles with no keyword match fall back to the full article corpus to avoid missing low-coverage tickers.

---

### Step 4 — Signal Aggregation (`src/signals/aggregator.py`)

Sentiment scores are converted into structured `TickerSignal` objects:

```
score ≥ +0.15   →  BULLISH
score ≤ −0.15   →  BEARISH
otherwise       →  NEUTRAL

confidence = min(1.0, |score| / 0.5)
```

A score of ±0.50 maps to 100% confidence. This linear scaling means the model must produce a clear, decisive score to trigger a high-confidence signal.

---

### Step 5 — Final Recommendations (`src/analysis/claude_analyst.py`)

All ticker signals are passed in a single prompt to **Claude Sonnet** (`claude-sonnet-4-6`), which acts as a high-conviction portfolio manager.

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

### Step 7 — Notification (`src/notifications/email_sender.py`)

Actionable signals (BUY/SELL only) are sent as an HTML email with:
- Color-coded action badges (green = BUY, red = SELL)
- Direction arrow, confidence %, time horizon
- Rationale citing the specific news catalyst
- Performance table showing open positions and historical stats

Email is sent only if `SMTP_USER` and `EMAIL_RECIPIENTS` are configured.

---

## Model Routing Summary

| Task | Model | Fallback |
|---|---|---|
| Per-ticker sentiment scoring | DeepSeek V3 (`deepseek-chat`) | Claude Haiku (`claude-haiku-4-5`) |
| Final synthesis / recommendations | Claude Sonnet (`claude-sonnet-4-6`) | Rule-based fallback |

---

## Local Cache

Fetched data is cached in `cache/` keyed by `YYYY-MM-DD_HH`:

- **Same hour** → reuses cached news and snapshots (fast re-runs)
- **New hour** → fetches fresh live data
- **Paper trades** → persisted in `cache/trades.json`

```python
from src.data.cache import load_news, load_snapshots, list_cached_keys

print(list_cached_keys())                    # ['2026-03-22_08', ...]
articles  = load_news("2026-03-22_08")
snapshots = load_snapshots("2026-03-22_08")
```

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
STOCK_WATCHLIST=NRGU,AGQ,SHNY,FNGU,QQQ
SECTOR_ETFS=XLK,XLF,XLE,XLV,XLY,XLP,XLI,XLB,XLU,XLRE,XLC,GLD,SLV

# Scheduler (cron syntax, US/Eastern)
SCHEDULE_DAILY=0 8 * * 1-5
```

---

## Running

```bash
# Run once — console output only
python main.py

# Run once and send email report
python main.py --email

# Start daily scheduler (8:00 AM ET, Mon–Fri)
python main.py --schedule
```

---

## Output Example

```
============================================================
  ACTIONABLE SIGNALS  (2026-03-26 13:00 UTC)
============================================================

  STOCKS
  ----------------------------------------
  ▲ NVDA   BUY    conf=82%  [SWING]
    Nvidia dominated headlines after announcing a major inference
    partnership with three hyperscalers; supply-chain fears eased
    by management guidance. Expected catalyst to fully price in
    over 3–7 days; risk is a broader tech selloff.

  ▼ INTC   SELL   conf=76%  [SHORT-TERM]
    Intel announced a delayed node transition and lowered Q2
    guidance, with no clear recovery catalyst cited. Short thesis
    holds for 2–4 weeks; risk is an activist or buyout rumour.

  ETFs / MARKETS
  ----------------------------------------
  ▲ GLD    BUY    conf=78%  [POSITION]
    Safe-haven demand surging amid escalating geopolitical tensions;
    central bank buying at record pace reported this week. Multi-week
    to multi-month thesis; risk is a rapid de-escalation event.
============================================================
```

---

## Project Structure

```
llm_trader/
├── main.py                     # Entry point (--email, --schedule flags)
├── requirements.txt
├── .env                        # API keys and config (not committed)
├── cache/                      # News + snapshot cache + trades.json
├── logs/                       # Rotating daily logs (7-day retention)
├── config/
│   └── settings.py             # .env loader
└── src/
    ├── pipeline.py             # Main orchestration (all 7 steps)
    ├── models.py               # Pydantic models: NewsArticle, TickerSignal, Recommendation
    ├── data/
    │   ├── trending.py         # Dynamic ticker discovery (Yahoo, AV, NewsAPI)
    │   ├── news_fetcher.py     # RSS + NewsAPI aggregator
    │   ├── market_data.py      # yfinance price snapshots
    │   └── cache.py            # Hourly file-based cache
    ├── analysis/
    │   ├── sentiment.py        # DeepSeek V3 / Haiku sentiment scoring
    │   └── claude_analyst.py   # Claude Sonnet final recommendations
    ├── signals/
    │   └── aggregator.py       # Score → TickerSignal (direction + confidence)
    ├── performance/
    │   └── tracker.py          # Paper trade recording, P&L, auto-close
    ├── scheduler/
    │   └── runner.py           # APScheduler daily automation
    └── notifications/
        └── email_sender.py     # Jinja2 HTML email via SMTP
```

---

## Disclaimer

This tool is for informational and educational purposes only. It does not constitute financial advice. Always conduct your own research before making investment decisions.