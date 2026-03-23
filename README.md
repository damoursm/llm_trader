# LLM Trader

An AI-powered stock analysis tool that aggregates real-time news, scores sentiment with LLMs, and generates actionable trading signals (BUY/SELL/HOLD/WATCH).

## How It Works

1. **Fetch News** — Aggregates articles from RSS feeds (Reuters, CNBC, MarketWatch, Yahoo Finance, etc.) and optionally NewsAPI. News window covers the last 24 hours.
2. **Fetch Market Data** — Gets real-time prices and historical data via yfinance (cached locally).
3. **Sentiment Analysis** — Each ticker's recent news is scored by **DeepSeek V3** (fast, low-cost). Falls back to Claude Haiku if DeepSeek is unavailable.
4. **Signal Aggregation** — News sentiment score is the sole signal driver (no technical analysis).
5. **Final Recommendations** — **Claude Sonnet** synthesizes all signals into BUY/SELL/HOLD/WATCH actions with rationale citing specific news catalysts.
6. **Email Delivery** — Sends HTML-formatted reports (optional).

## Model Routing

| Task | Model | Fallback |
|---|---|---|
| Per-ticker sentiment scoring | DeepSeek V3 (`deepseek-chat`) | Claude Haiku |
| Final synthesis / recommendations | Claude Sonnet (`claude-sonnet-4-6`) | Rule-based fallback |

## Local Data Cache

Fetched news and market snapshots are cached in `cache/` keyed by date and hour:

- **Same hour** → reuses cached data (no network calls, fast iteration during development)
- **New hour** → fetches fresh live data and saves a new snapshot
- **Historical replay** → old files accumulate as a local archive

```python
from src.data.cache import load_news, load_snapshots, list_cached_keys

print(list_cached_keys())                    # ['2026-03-22_08', ...]
articles  = load_news("2026-03-22_08")
snapshots = load_snapshots("2026-03-22_08")
```

## Prerequisites

- Python 3.11+
- [Anthropic API key](https://console.anthropic.com/) (Claude Sonnet for synthesis)
- [DeepSeek API key](https://platform.deepseek.com/) (V3 for sentiment scoring)

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure environment variables**

Create a `.env` file at the project root:

```env
# Required
ANTHROPIC_API_KEY=your_anthropic_api_key_here
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# Optional - additional news coverage
NEWSAPI_KEY=your_newsapi_key_here

# Optional - email reports
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password_here
EMAIL_RECIPIENTS=recipient1@example.com,recipient2@example.com

# Customize your watchlist (defaults shown)
STOCK_WATCHLIST=AAPL,MSFT,NVDA,TSLA,AMZN,META,GOOGL
SECTOR_ETFS=XLK,XLF,XLE,XLV,XLY,XLP,XLI,XLB,XLU,XLRE,XLC

# Scheduling (cron, US/Eastern timezone)
SCHEDULE_DAILY=0 8 * * 1-5
```

> **Gmail users:** Use an [App Password](https://support.google.com/accounts/answer/185833) rather than your account password.

## Running

**Run once (console output)**

```bash
python main.py
```

**Run once and send email report**

```bash
python main.py --email
```

**Start the daily scheduler (runs at 8:00 AM ET, Mon–Fri)**

```bash
python main.py --schedule
```

## Output

Each run prints a summary to the console and (optionally) sends an HTML email with:

- Ticker, recommended action (BUY/SELL/HOLD/WATCH), and confidence score
- News-driven rationale citing specific catalysts
- Color-coded action badges

## Logs

Logs are written to `logs/llm_trader_YYYY-MM-DD.log` (DEBUG level) and to the console (INFO level). Log files rotate daily and are retained for 7 days.

## Project Structure

```
llm_trader/
├── main.py                     # Entry point
├── requirements.txt
├── .env                        # API keys and configuration (not committed)
├── cache/                      # Local data cache (not committed)
├── logs/                       # Rotating log files (not committed)
├── config/
│   └── settings.py             # Environment variable loader
└── src/
    ├── pipeline.py             # Main orchestration
    ├── models.py               # Data models
    ├── data/
    │   ├── market_data.py      # yfinance integration
    │   ├── news_fetcher.py     # RSS + NewsAPI aggregator
    │   └── cache.py            # File-based data cache
    ├── analysis/
    │   ├── sentiment.py        # DeepSeek V3 sentiment scoring
    │   └── claude_analyst.py   # Claude Sonnet final recommendations
    ├── signals/
    │   └── aggregator.py       # News-driven signal builder
    ├── scheduler/
    │   └── runner.py           # APScheduler daily automation
    └── notifications/
        └── email_sender.py     # SMTP email delivery
```

## Disclaimer

This tool is for informational purposes only and does not constitute financial advice. Always do your own research before making investment decisions.
