"""Open-vocabulary ticker extraction from free text (headlines, social posts).

Resolves `$CASHTAGS` and bare capitalized tokens against the **full SEC ticker
universe** (~10k symbols from company_tickers.json, loaded once per process by
``eight_k``) instead of a small hardcoded allowlist — so discovery can surface
any listed name, not just a curated set of mega-caps.

Precision controls (false positives are costly — a junk ticker pollutes the run):
  • Candidates must be REAL tickers (present in the SEC reference set).
  • `$CASHTAGS` are explicit and high-precision → accepted on validity alone.
  • Bare ALL-CAPS tokens must additionally NOT be a common English word or a
    finance/market acronym (the `_STOPWORDS` set below) — many such words are
    also real tickers (ALL, ARE, ON, IT, CASH, OPEN, REAL…) and would otherwise
    flood every headline.
  • Callers apply a minimum mention count across distinct items as a final filter.

This module imports only from ``eight_k`` (which owns the SEC map), so it can be
shared by both ``trending`` and ``reddit_sentiment`` without a circular import.
"""

import re
from typing import Set

from src.data.eight_k import get_valid_tickers

_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5})\b")
_TOKEN_RE   = re.compile(r"\b([A-Z]{2,5})\b")

# Common English words + finance/market/WSB acronyms that collide with real
# tickers. Bare ALL-CAPS tokens matching these are rejected (cashtags bypass it).
_STOPWORDS: frozenset = frozenset({
    # short function words / pronouns
    "AM", "PM", "ET", "PT", "US", "USA", "UK", "EU", "ON", "IN", "AT", "BE", "DO",
    "IF", "OF", "OR", "TO", "UP", "VS", "BY", "AS", "IT", "IS", "SO", "GO", "NO",
    "WE", "HE", "ME", "MY", "AN", "OK", "RE", "ID",
    # common words
    "ALL", "AND", "ANY", "ARE", "BUT", "CAN", "DID", "FOR", "GET", "GOT", "HAD",
    "HAS", "HER", "HIM", "HIS", "HOW", "ITS", "LET", "MAY", "NEW", "NOT", "NOW",
    "OFF", "OLD", "ONE", "OUR", "OUT", "OWN", "PER", "PUT", "SAY", "SEE", "SHE",
    "THE", "TOO", "TOP", "TWO", "USE", "VIA", "WAS", "WAY", "WHO", "WHY", "WIN",
    "YES", "YET", "YOU", "BIG", "DAY", "END", "FAR", "FEW", "LOT", "LOW", "MAN",
    "RUN", "SET", "SIX", "TEN", "WAR", "BAD", "BUY", "EAT",
    "THAT", "THIS", "THAN", "THEN", "THEY", "THEM", "WHEN", "WHAT", "WILL",
    "WITH", "FROM", "HAVE", "BEEN", "OVER", "INTO", "ALSO", "JUST", "LIKE",
    "MORE", "MOST", "SOME", "SUCH", "VERY", "YEAR", "WEEK", "ONLY", "GOOD",
    "BEST", "DEAL", "RISK", "NEWS", "DATA", "PLAY", "REAL", "OPEN", "CASH",
    "GOLD", "BANK", "FUND", "LOVE", "HUGE", "HIGH", "RICH", "DEBT", "RATE",
    "SOON", "LATE", "HARD", "EASY", "FREE", "FULL", "LONG", "NEXT", "LAST",
    "HELP", "CARE", "HOME", "LIFE", "WORK", "TIME", "TEAM", "PLAN", "CALL",
    # finance / market / WSB acronyms
    "CEO", "CFO", "COO", "CTO", "IPO", "ETF", "GDP", "CPI", "PPI", "FED", "SEC",
    "FDA", "DOJ", "FBI", "IRS", "NYC", "API", "AI", "EV", "EPS", "ROE", "ROI",
    "PEG", "YOY", "QOQ", "ATH", "ATL", "DD", "IV", "EOD", "EOW", "AH", "USD",
    "EUR", "GBP", "JPY", "CAD", "IMF", "WTO", "OPEC", "LLC", "INC", "LTD", "PLC",
    "ESG", "OTC", "NYSE", "SPAC", "WSB", "YOLO", "FOMO", "HODL", "OTM", "ITM",
    "CTB", "RSI", "MACD", "SMA", "EMA", "VWAP", "ADX", "ATR", "PE", "FY", "Q1",
    "Q2", "Q3", "Q4", "SELL", "HOLD", "PUTS", "MOON", "BULL", "BEAR", "PUMP",
    "DUMP", "LOSS", "GAIN", "TECH", "OIL", "DOW", "SPX", "ER", "PR", "UI", "UX",
})

# Process-cached valid-ticker set (built once from the SEC map).
_VALID: frozenset = frozenset()


def _valid() -> frozenset:
    global _VALID
    if not _VALID:
        _VALID = get_valid_tickers()
    return _VALID


def extract_candidate_tickers(text: str) -> Set[str]:
    """Return the set of valid tickers referenced in ``text``.

    `$CASHTAGS` are accepted on SEC-validity alone; bare ALL-CAPS tokens must also
    clear the stopword filter. Returns a *set* (deduplicated within this text), so
    a caller counting across many texts measures how many distinct items mention
    each ticker. Empty set when the SEC map is unavailable or no candidate matches.
    """
    if not text:
        return set()
    valid = _valid()
    if not valid:
        return set()

    found: Set[str] = set()
    for sym in _CASHTAG_RE.findall(text):
        s = sym.upper()
        if s in valid:
            found.add(s)
    for sym in _TOKEN_RE.findall(text):
        s = sym.upper()
        if s in valid and s not in _STOPWORDS:
            found.add(s)
    return found
