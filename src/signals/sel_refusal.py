"""IBKR'S REFUSAL OF AN OPENING SHORT, PREDICTED BY NAME (from 2026-10-23; PREREG190 — the user: "make sure we can
correctly predict refused short", then "Yes please add the two productions steps").

The frozen model M2c (``sel_short_dir``/refusal/model.json, installed from PREREG190's ``recal190_b.json``): a logistic
regression on ``micro`` (on IBKR's U.S. Microcap list at the session's open, `src.data.ibkr_microcaps`), ``fpi`` (a
20-F, 20-F/A, 40-F or 6-K filed in the 400 days before — a foreign private issuer; the deep store's ``sec_filings``),
``age`` (ln(years since Polygon's ``list_date`` + 0.1)) and ``mcap`` (log10 of Polygon's market cap in the deep
``ticker_details``; else its shares x the pick's price), each feature missing taking the model's stand-in (``fill``);
an exchange-traded product is never refused (0). PREREG190: on two batches of the short arms' candidates asked after it
was frozen it ranked IBKR's refusals with an AUC of 0.95 and 0.98, and its recalibrated level predicted the second batch
out of sample (3.3 expected, 3 refused); IBKR refused 5 of 203 such stocks (2.5%, 95% 0.8-5.7%).

JOURNAL-ONLY: the live calibration scores each funded pick's IBKR what-if answer against it (`live_calibration`'s
``refusal_model``); production's entry step keeps asking IBKR, whose answer decides. The ticker details are the deep
store's latest snapshot (a past pick is scored on today's market cap).
"""
from __future__ import annotations

import json
import math
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

from config import settings

ETP_TYPES = {"ETF", "ETN", "ETV", "ETS", "FUND"}
FPI_FORMS = ("20-F", "20-F/A", "40-F", "6-K")
FPI_DAYS = 400
DEEP = Path("cache") / "ml" / "deep"


def model_path() -> Path:
    return Path(settings.sel_short_dir) / "refusal" / "model.json"


@lru_cache(maxsize=4)
def _model_at(path: str, mtime: float) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        logger.warning(f"[refusal] model unreadable ({e})")
        return None


def model() -> Optional[dict]:
    """The installed model (None when it is missing)."""
    p = model_path()
    try:
        return _model_at(str(p), p.stat().st_mtime)
    except OSError:
        return None


@lru_cache(maxsize=4096)
def _details(ticker: str) -> Dict:
    p = DEEP / "ticker_details" / "parts" / f"{ticker}.parquet"
    if not p.exists():
        return {}
    try:
        import duckdb
        r = duckdb.connect().execute(
            "select type, cik, list_date, market_cap, share_class_shares_outstanding sh, weighted_shares_outstanding wsh "
            f"from read_parquet('{p.as_posix()}')").fetchall()
    except Exception as e:                                     # noqa: BLE001
        logger.warning(f"[refusal] {ticker}'s details unreadable ({e})")
        return {}
    if not r:
        return {}
    t, cik, ld, mc, sh, wsh = r[-1]
    return {"type": t, "cik": cik, "list_date": ld, "market_cap": mc, "shares": sh or wsh}


def _fpi(ticker: str, cik: Optional[str], d: date) -> Optional[int]:
    p = DEEP / "sec_filings.parquet"
    if not p.exists():
        return None
    lo, hi = (d - timedelta(days=FPI_DAYS)).isoformat(), d.isoformat()
    forms = ",".join(f"'{f}'" for f in FPI_FORMS)
    try:
        import duckdb
        n = duckdb.connect().execute(
            f"select count(*) from read_parquet('{p.as_posix()}') where form in ({forms}) and filing_date > ? and "
            "filing_date < ? and (ticker = ? or cik = ?)", [lo, hi, ticker, str(cik or "")]).fetchone()[0]
    except Exception as e:                                     # noqa: BLE001
        logger.warning(f"[refusal] {ticker}'s filings unreadable ({e})")
        return None
    return int(n > 0)


def features(ticker: str, d: date, price: Optional[float] = None) -> Dict[str, Optional[float]]:
    """The model's inputs for ``ticker`` at session ``d`` (None = unknown), and ``kind`` (stock / etf)."""
    from src.data import ibkr_microcaps
    tk = str(ticker or "").strip().upper()
    det = _details(tk)
    out: Dict[str, Optional[float]] = {"kind": "etf" if det.get("type") in ETP_TYPES else "stock"}
    m = ibkr_microcaps.on_list(tk, d)
    out["micro"] = None if m is None else float(m)
    f = _fpi(tk, det.get("cik"), d)
    out["fpi"] = None if f is None else float(f)
    try:
        yrs = (d - date.fromisoformat(str(det["list_date"])[:10])).days / 365.25 if det.get("list_date") else None
    except ValueError:
        yrs = None
    out["age"] = math.log(max(yrs, 0.0) + 0.1) if yrs is not None else None
    cap = det.get("market_cap")
    if not (cap and cap > 0) and det.get("shares") and price:
        cap = float(det["shares"]) * float(price)
    out["mcap"] = math.log10(float(cap)) if cap and cap > 0 else None
    return out


def probabilities(tickers, d: date, prices: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """`probability` for many names at once (the nightly probe ranks ~2,000): the details in one read of the deep store's
    parts, the filers in one query, the list once. {ticker: p}; empty without the model."""
    mdl = model()
    if not mdl:
        return {}
    from src.data import ibkr_microcaps
    tks = sorted({str(t).strip().upper() for t in tickers if t})
    if not tks:
        return {}
    import duckdb
    con = duckdb.connect()
    con.register("want", __import__("pandas").DataFrame({"ticker": tks}))
    det = {}
    try:
        rows = con.execute(
            "select d.ticker, d.type, d.cik, d.list_date, d.market_cap, coalesce(d.share_class_shares_outstanding, "
            f"d.weighted_shares_outstanding) sh from read_parquet('{(DEEP / 'ticker_details' / 'parts' / '*.parquet').as_posix()}', "
            "union_by_name=true) d join want w on d.ticker = w.ticker").fetchall()
        det = {r[0]: {"type": r[1], "cik": r[2], "list_date": r[3], "market_cap": r[4], "shares": r[5]} for r in rows}
    except Exception as e:                                     # noqa: BLE001
        logger.warning(f"[refusal] bulk details unreadable ({e})")
    fpi_t, fpi_c = set(), set()
    p = DEEP / "sec_filings.parquet"
    if p.exists():
        lo, hi = (d - timedelta(days=FPI_DAYS)).isoformat(), d.isoformat()
        forms = ",".join(f"'{f}'" for f in FPI_FORMS)
        try:
            for t, c in con.execute(f"select distinct upper(ticker), cik from read_parquet('{p.as_posix()}') where form in "
                                    f"({forms}) and filing_date > ? and filing_date < ?", [lo, hi]).fetchall():
                fpi_t.add(str(t))
                fpi_c.add(str(c))
        except Exception as e:                                 # noqa: BLE001
            logger.warning(f"[refusal] bulk filings unreadable ({e})")
    listed = ibkr_microcaps.symbols_before(d)
    m, fill = mdl["model"], mdl["fill"]
    out: Dict[str, float] = {}
    for tk in tks:
        dt = det.get(tk, {})
        if dt.get("type") in ETP_TYPES:
            out[tk] = 0.0
            continue
        f = {"micro": None if listed is None else float(tk in listed[1] or tk.replace("-", " ") in listed[1]),
             "fpi": float(tk in fpi_t or (dt.get("cik") is not None and str(dt.get("cik")) in fpi_c))}
        try:
            yrs = (d - date.fromisoformat(str(dt["list_date"])[:10])).days / 365.25 if dt.get("list_date") else None
        except ValueError:
            yrs = None
        f["age"] = math.log(max(yrs, 0.0) + 0.1) if yrs is not None else None
        cap = dt.get("market_cap")
        px = (prices or {}).get(tk)
        if not (cap and cap > 0) and dt.get("shares") and px:
            cap = float(dt["shares"]) * float(px)
        f["mcap"] = math.log10(float(cap)) if cap and cap > 0 else None
        z = 0.0
        for name, mu, sd, w in zip(m["features"], m["mean"], m["sd"], m["w"][1:]):
            x = f.get(name)
            x = float(fill[name]) if x is None else float(x)
            z += w * (x - float(mu)) / (float(sd) or 1.0)
        out[tk] = 1.0 / (1.0 + math.exp(-(float(m["w"][0]) + z)))
    return out


def probability(ticker: str, d: date, price: Optional[float] = None) -> Optional[float]:
    """The model's probability that IBKR refuses an opening short in ``ticker`` at session ``d`` (0 for an
    exchange-traded product); None without the model."""
    mdl = model()
    if not mdl:
        return None
    f = features(ticker, d, price)
    if f["kind"] == "etf":
        return 0.0
    m, fill = mdl["model"], mdl["fill"]
    z = 0.0
    for name, mu, sd, w in zip(m["features"], m["mean"], m["sd"], m["w"][1:]):
        x = f.get(name)
        x = float(fill[name]) if x is None else float(x)
        z += w * (x - float(mu)) / (float(sd) or 1.0)
    return 1.0 / (1.0 + math.exp(-(float(m["w"][0]) + z)))
