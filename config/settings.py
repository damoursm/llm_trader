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
    enable_market_data: bool = True       # set to false to skip yfinance fetching entirely
    enable_charts: bool = False           # set to true to build Plotly charts and HTML report

    # Analysis method flags (at least one should be true)
    enable_news_sentiment: bool = True    # method 1: LLM sentiment from news/RSS
    enable_technical_analysis: bool = True  # method 2: RSI, MACD, SMA, Bollinger Bands
    enable_insider_trades: bool = True    # method 3: politician + corporate insider trades

    # Insider trades config
    insider_lookback_days: int = 90      # how far back to look for trades
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


settings = Settings()
