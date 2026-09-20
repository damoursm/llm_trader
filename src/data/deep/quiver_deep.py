"""Quiver Quantitative historical endpoints available on the Hobbyist key
(measured 2026-09-19): congress trading 2014→, lobbying 1999→, government
contracts 2022→, per-ticker off-exchange (dark-pool) share 2021→. WSB mentions
and corporate-flight data are paid tiers and are not requested.

Point-in-time keys: congress ``ReportDate`` (the disclosure, weeks after the
trade — never ``TransactionDate``), lobbying / contracts ``Date``, DPI ``Date``.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from src.data.deep import RateLimiter, http_get

_BASE = "https://api.quiverquant.com/beta"
_LIMITER = RateLimiter(0.3)
ENDPOINTS = {
    "quiver_congress": "/historical/congresstrading/{tk}",
    "quiver_lobbying": "/historical/lobbying/{tk}",
    "quiver_contracts": "/historical/govcontractsall/{tk}",
    "quiver_dpi": "/historical/offexchange/{tk}",
}


def _headers() -> dict:
    from config.settings import settings
    return {"Authorization": f"Bearer {settings.quiver_api_key}", "Accept": "application/json",
            "User-Agent": "llm-trader/1.0"}


def fetch(family: str, ticker: str) -> Optional[pd.DataFrame]:
    path = ENDPOINTS[family].format(tk=ticker.upper())
    r = http_get(_BASE + path, headers=_headers(), timeout=60, limiter=_LIMITER)
    if r is None:
        raise RuntimeError("no response")
    if r.status_code in (403, 404):
        return pd.DataFrame()
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    try:
        j = r.json()
    except ValueError:
        return pd.DataFrame()
    if not isinstance(j, list) or not j:
        return pd.DataFrame()
    df = pd.DataFrame(j)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    if "Ticker" in df.columns:
        df = df.drop(columns=["Ticker"])
    df.insert(0, "ticker", ticker.upper())
    for c in df.columns:
        if c != "ticker" and df[c].dtype == object:
            df[c] = df[c].map(lambda v: v if (v is None or isinstance(v, (str, int, float, bool))) else str(v))
    return df


def make_fetcher(family: str):
    return lambda tk: fetch(family, tk)
