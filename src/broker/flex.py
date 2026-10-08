"""IBKR's Flex Web Service — the account's own statement data (user 2026-10-06: "We should use IBKR data as much as we
can ... we want the real exact borrow fees we would have in live trading a real account").

The Activity Flex Query ``ibkr_flex_query_id`` (Client Portal -> Performance & Reports -> Flex Queries; section
"Borrow Fees Details", XML, period "Last 365 Calendar Days"), read with the Flex Web Service token
``ibkr_flex_token``. Each row is one short position on one value date: symbol, conid, quantity, price (IBKR: 102% of
the prior day's settlement price, rounded up to the dollar), value, borrowFeeRate, borrowFee. ``fetch_borrow_fees``
stores every row in ``broker_borrow_fees`` (each statement's period replaced whole) and keeps the raw XML under
``data/ibkr_flex/`` (never delete). The ledger's borrow schedule (``src/performance/borrow_fees.py``) takes these
charges over its formula. The section is read by attribute, not by element name: any element carrying ``borrowFee``.

Two calls: SendRequest (token + query id) answers a reference code; GetStatement (token + that code) answers the
statement — or "generation in progress" (code 1019), retried.

    python -m src.broker.flex --borrow-fees
"""
from __future__ import annotations

import json
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
from loguru import logger

from config.settings import settings

SEND_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/SendRequest"
GET_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService/GetStatement"
RAW_DIR = Path("data/ibkr_flex")
IN_PROGRESS = {"1019"}                      # "Statement generation in progress. Please try again shortly."
BUSY = {"1018"}                             # "Too many requests have been made from this token": wait, ask again
PENDING = "pending.json"                    # a statement IBKR was still generating when the last run gave up
_HEADERS = {"User-Agent": "llm_trader/1.0"}


class FlexError(RuntimeError):
    def __init__(self, code: Optional[str], message: Optional[str]):
        super().__init__(f"Flex error {code}: {message}")
        self.code, self.message = code, message


def _get(url: str, params: dict, timeout: float = 60.0) -> str:
    r = httpx.get(url, params=params, timeout=timeout, headers=_HEADERS, follow_redirects=True)
    r.raise_for_status()
    return r.text


def request_statement(token: str, query_id: str) -> Tuple[str, str]:
    """SendRequest -> (reference code, the GetStatement URL IBKR names)."""
    root = ET.fromstring(_get(SEND_URL, {"t": token, "q": query_id, "v": "3"}))
    if (root.findtext("Status") or "").strip() != "Success":
        raise FlexError(root.findtext("ErrorCode"), root.findtext("ErrorMessage"))
    return (root.findtext("ReferenceCode") or "").strip(), (root.findtext("Url") or GET_URL).strip()


def _get_retrying(url: str, params: dict, attempts: int = 4, pause: float = 3.0) -> str:
    """A GET that rides out this machine's intermittent DNS failures (2026-10-06: the router answers with a 3-second
    lifetime and some lookups of ``gdcdyn.interactivebrokers.com`` fail, the next succeeds)."""
    for i in range(attempts):
        try:
            return _get(url, params)
        except httpx.ConnectError:
            if i == attempts - 1:
                raise
            time.sleep(pause)
    raise AssertionError("unreachable")


def get_statement(token: str, ref: str, url: str = GET_URL, tries: int = 60, wait: float = 30.0,
                  first_wait: float = 10.0) -> str:
    """GetStatement, retried while IBKR is still generating the statement (a year of one account took over 45
    minutes on 2026-10-06). A host IBKR names that stays unreachable falls back to the SendRequest host."""
    time.sleep(first_wait)
    for _ in range(max(1, tries)):
        try:
            text = _get_retrying(url, {"t": token, "q": ref, "v": "3"})
        except httpx.ConnectError as e:
            if url == GET_URL:
                raise
            logger.info(f"[flex] {httpx.URL(url).host} unreachable ({e}) — GetStatement on {httpx.URL(GET_URL).host}")
            url = GET_URL
            text = _get_retrying(url, {"t": token, "q": ref, "v": "3"})
        root = ET.fromstring(text)
        if root.tag != "FlexStatementResponse":
            return text
        code = (root.findtext("ErrorCode") or "").strip()
        if code not in IN_PROGRESS | BUSY:
            raise FlexError(code, root.findtext("ErrorMessage"))
        time.sleep(wait)
    raise FlexError("timeout", f"statement {ref} still generating after {tries} tries")


def _num(x) -> Optional[float]:
    if x is None or str(x).strip() == "":
        return None
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None


def _day(x) -> Optional[date]:
    """IBKR writes dates as yyyyMMdd by default; yyyy-MM-dd and MM/dd/yyyy are query options."""
    s = str(x or "").strip().split(";")[0].split(" ")[0]
    for fmt in ("%Y%m%d", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def parse_borrow_fees(text: str) -> Tuple[List[dict], List[dict]]:
    """(rows, statement periods) from a Flex statement: every element with a ``borrowFee`` attribute is one charge."""
    from src.broker.ibkr import from_ib_symbol
    root = ET.fromstring(text)
    fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows: List[dict] = []
    periods: List[dict] = []
    for st in root.iter("FlexStatement"):
        acct = st.get("accountId") or ""
        periods.append({"account": acct, "start": _day(st.get("fromDate")), "end": _day(st.get("toDate"))})
        for el in st.iter():
            a = el.attrib
            if "borrowFee" not in a:
                continue
            sym = (a.get("symbol") or "").strip()
            vd = _day(a.get("valueDate") or a.get("date") or a.get("reportDate"))
            if not sym or vd is None:
                continue
            conid = _num(a.get("conid"))
            rows.append({"account": a.get("accountId") or acct, "ticker": from_ib_symbol(sym), "symbol": sym,
                         "conid": int(conid) if conid is not None else None, "value_date": vd,
                         "quantity": _num(a.get("quantity")), "price": _num(a.get("price")),
                         "value": _num(a.get("value")), "fee_rate": _num(a.get("borrowFeeRate")),
                         "fee": _num(a.get("borrowFee")), "currency": a.get("currency"),
                         "fx_to_base": _num(a.get("fxRateToBase")), "description": a.get("description"),
                         "fetched_at": fetched})
    return rows, periods


def fetch_borrow_fees(raw_dir: Optional[Path] = None) -> Dict[str, object]:
    """Fetch the Flex statement, archive it raw, store its borrow fees. Skipped without a token and query id."""
    token = str(getattr(settings, "ibkr_flex_token", "") or "").strip()
    qid = str(getattr(settings, "ibkr_flex_query_id", "") or "").strip()
    if not token or not qid:
        return {"skipped": "no Flex token / query id"}
    raw_dir = Path(raw_dir or RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    pend = raw_dir / PENDING
    text = None
    if pend.exists():                       # the last run's statement first: IBKR says never re-initiate
        try:
            p = json.loads(pend.read_text(encoding="utf-8"))
            if p.get("query") == qid and time.time() - float(p.get("at", 0)) < 2 * 86400:
                text = get_statement(token, p["ref"], p.get("url") or GET_URL, tries=1, wait=0.0, first_wait=0.0)
        except FlexError:
            text = None
        except Exception as e:                                       # noqa: BLE001
            logger.debug(f"[flex] pending statement unreadable: {e}")
        pend.unlink(missing_ok=True)
    if text is None:
        ref, url = request_statement(token, qid)
        try:
            text = get_statement(token, ref, url)
        except FlexError as e:
            if e.code == "timeout":
                pend.write_text(json.dumps({"ref": ref, "url": url, "query": qid, "at": time.time()}), encoding="utf-8")
                logger.warning(f"[flex] statement {ref} still generating — collected first on the next run")
            raise
    (raw_dir / f"{datetime.now():%Y-%m-%d_%H%M%S}_q{qid}.xml").write_text(text, encoding="utf-8")
    rows, periods = parse_borrow_fees(text)
    from src.db import repo
    stored = 0
    for p in periods:
        if p["start"] is None or p["end"] is None:
            continue
        mine = [r for r in rows if (r["account"] or p["account"]) == p["account"]]
        stored += repo.save_broker_borrow_fees(mine, p["account"], p["start"], p["end"])
    from src.performance import borrow_fees
    borrow_fees.reset()
    out = {"rows": len(rows), "stored": stored, "periods": [{k: str(v) for k, v in p.items()} for p in periods],
           "names": len({r["ticker"] for r in rows})}
    logger.info(f"[flex] IBKR borrow fees: {out}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="IBKR Flex Web Service")
    ap.add_argument("--borrow-fees", action="store_true", help="fetch and store the Borrow Fees Details")
    if ap.parse_args().borrow_fees:
        print(json.dumps(fetch_borrow_fees(), indent=1, default=str))
