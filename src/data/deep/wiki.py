"""Wikipedia attention — daily page views per company article since 2015-07,
no key, point-in-time by construction (day D's count is final on D+1).

Ticker → article comes from Wikidata: items with a stock-exchange claim (P414)
qualified by a ticker symbol (P249) on NYSE / Nasdaq / NYSE American / Cboe
and an English Wikipedia sitelink (~3,750 rows, measured 2026-09-19). Small
caps mostly have no article or a handful of views a day — usable for the
liquid half of the universe, and the feature builder should treat a thin
series as absent rather than as zero attention.
"""
from __future__ import annotations

import csv
import io
from datetime import date, timedelta
from typing import Dict, Optional
from urllib.parse import quote, unquote

import pandas as pd
from loguru import logger

from src.data.deep import RateLimiter, family_dir, http_get, read_parquet, write_parquet

_UA = {"User-Agent": "llm-trader/1.0 (research bot, single-threaded, ~1 req/s; maintainer: repo owner)"}
_SPARQL = """SELECT ?item ?ticker ?exchangeLabel ?article WHERE {
  ?item p:P414 ?st . ?st ps:P414 ?exchange ; pq:P249 ?ticker .
  VALUES ?exchange { wd:Q13677 wd:Q82059 wd:Q2632892 wd:Q1207951 }
  ?article schema:about ?item ; schema:isPartOf <https://en.wikipedia.org/> .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}"""
_PV = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/"
       "all-access/user/{title}/daily/{start}/{end}")
# Measured 2026-09-19: four concurrent workers drew HTTP 429 with Retry-After 43 s
# on 10 of 12 calls, so this family runs ONE worker at ~1 request/second.
_LIMITER = RateLimiter(1.0)
START = "20150701"


def title_from_url(url: str) -> str:
    return unquote(str(url).rsplit("/wiki/", 1)[-1])


def fetch_mapping() -> pd.DataFrame:
    r = http_get("https://query.wikidata.org/sparql", params={"query": _SPARQL},
                 headers={**_UA, "Accept": "text/csv"}, timeout=300)
    if r is None or r.status_code != 200:
        raise RuntimeError(f"wikidata HTTP {getattr(r, 'status_code', None)}")
    rows = list(csv.DictReader(io.StringIO(r.text)))
    df = pd.DataFrame(rows)
    if not len(df):
        return df
    df = df.rename(columns={"exchangeLabel": "exchange"})
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    df["title"] = df["article"].map(title_from_url)
    df = df.drop_duplicates(["ticker"]).reset_index(drop=True)
    return df[["ticker", "exchange", "item", "article", "title"]]


def mapping(refresh: bool = False) -> Dict[str, str]:
    p = family_dir("wiki") / "wikidata_tickers.parquet"
    if p.exists() and not refresh:
        df = read_parquet(p)
    else:
        df = fetch_mapping()
        if len(df):
            write_parquet(df, p)
            logger.info(f"[deep.wiki] wikidata mapping: {len(df)} tickers -> {p}")
    return dict(zip(df["ticker"], df["title"])) if len(df) else {}


def pageviews(title: str, start: str = START, end: Optional[str] = None) -> pd.DataFrame:
    end = end or (date.today() - timedelta(days=1)).strftime("%Y%m%d")
    url = _PV.format(title=quote(title.replace(" ", "_"), safe=""), start=start, end=end)
    r = http_get(url, headers=_UA, timeout=60, limiter=_LIMITER)
    if r is None:
        raise RuntimeError("no response")
    if r.status_code == 404:
        return pd.DataFrame()
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    items = (r.json() or {}).get("items") or []
    if not items:
        return pd.DataFrame()
    df = pd.DataFrame({"date": [i["timestamp"][:8] for i in items],
                       "views": [int(i.get("views") or 0) for i in items]})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d").dt.date.astype(str)
    return df


def pageviews_for_ticker(ticker: str) -> pd.DataFrame:
    title = mapping().get(ticker.upper())
    if not title:
        return pd.DataFrame()
    df = pageviews(title)
    if len(df):
        df.insert(0, "ticker", ticker.upper())
        df.insert(1, "title", title)
    return df
