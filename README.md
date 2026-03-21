# LLM Trader

An AI-powered stock analysis tool that combines real-time market data, multi-source news aggregation, and Claude LLM to generate actionable trading signals.

## How It Works

1. **Fetch News** — Aggregates articles from RSS feeds (Reuters, CNBC, MarketWatch, Yahoo Finance, etc.) and optionally NewsAPI
2. **Fetch Market Data** — Gets real-time prices and historical data via yfinance
3. **Sentiment Analysis** — Claude analyzes news articles for each ticker
4. **Technical Analysis** — Computes RSI, MACD, SMA crossovers, and Bollinger Bands
5. **Signal Aggregation** — Combines sentiment (60%) and technical (40%) scores
6. **Final Recommendations** — Claude synthesizes signals into BUY/SELL/HOLD/WATCH actions
7. **Email Delivery** — Sends HTML-formatted reports (optional)

## Prerequisites

- Python 3.7+
- An [Anthropic API key](https://console.anthropic.com/)

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Configure environment variables**

```bash
cp .env.example .env
```

Then edit `.env` and fill in your values:

```env
# Required
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional - for additional news sources
NEWSAPI_KEY=your_newsapi_key_here
ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here

# Optional - for email reports
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password_here
EMAIL_RECIPIENTS=recipient1@example.com,recipient2@example.com

# Customize your watchlist (defaults shown)
STOCK_WATCHLIST=AAPL,MSFT,NVDA,TSLA,AMZN,META,GOOGL
SECTOR_ETFS=XLK,XLF,XLE,XLV,XLY,XLP,XLI,XLB,XLU,XLRE,XLC

# Scheduling (cron expressions, US/Eastern timezone)
SCHEDULE_PREMARKET=0 8 * * 1-5
SCHEDULE_MIDDAY=0 12 * * 1-5
SCHEDULE_CLOSE=30 16 * * 1-5
```

> **Gmail users:** Use an [App Password](https://support.google.com/accounts/answer/185833) rather than your account password.

## Running

**Run analysis once (console output only)**

```bash
python main.py
```

**Run analysis once and send email report**

```bash
python main.py --email
```

**Start the scheduler (runs automatically at 8 AM, 12 PM, and 4:30 PM ET on weekdays)**

```bash
python main.py --schedule
```

## Output

Each run prints a summary to the console and (optionally) sends an HTML email with:

- Ticker, recommended action (BUY/SELL/HOLD/WATCH), and confidence score
- Rationale combining sentiment and technical signals
- Color-coded action badges

## Logs

Logs are written to `logs/llm_trader_YYYY-MM-DD.log` (DEBUG level) and to the console (INFO level). Log files rotate daily and are retained for 7 days.

## Project Structure

```
llm_trader/
├── main.py                     # Entry point
├── requirements.txt
├── .env.example                # Configuration template
├── config/
│   └── settings.py             # Environment variable loader
└── src/
    ├── pipeline.py             # Main orchestration
    ├── models.py               # Data models
    ├── data/
    │   ├── market_data.py      # yfinance integration
    │   └── news_fetcher.py     # RSS + NewsAPI aggregator
    ├── analysis/
    │   ├── sentiment.py        # LLM-based sentiment scoring
    │   ├── technical.py        # Technical indicator scoring
    │   └── claude_analyst.py   # Final LLM recommendations
    ├── signals/
    │   └── aggregator.py       # Combines sentiment + technical
    ├── scheduler/
    │   └── runner.py           # APScheduler automation
    └── notifications/
        └── email_sender.py     # SMTP email delivery
```

## Disclaimer

This tool is for informational purposes only and does not constitute financial advice. Always do your own research before making investment decisions.
