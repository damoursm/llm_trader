"""US trading halts in force NOW (2026-10-07, user: "protecting ourselves against trading halts. Fast-moving stocks
get paused for 5-10 minutes, and news or regulatory halts can last days. We can't buy back during a halt").

NYSE's current-halt list covers every US listing (NYSE, NYSE American, NYSE Arca, Nasdaq, Cboe BZX): a row without a
resume date is a security halted now — an LULD pause while it lasts, a news / regulatory halt, a suspension. Read
at most once a minute per process; fail-soft: None = unknown (nothing is blocked on an unknown).

The same API's history (`/api/trade-halts/historical/download`, one call per year, kept in `cache/ml/halts/`) is
what PREREG22b measured the vol arm's halt exposure on.
"""
from __future__ import annotations

import io
import time
from typing import Dict, Optional

from loguru import logger

URL = "https://www.nyse.com/api/trade-halts/current/download"
_HEADERS = {"Accept": "*/*", "Referer": "https://www.nyse.com/trade-halt"}   # without them: 406 / an empty body
_CACHE: dict = {"at": 0.0, "rows": None, "warned": 0.0}


def _symbol(ticker: str) -> str:
    """The list's spelling: class shares with a space (``BRK B``)."""
    return str(ticker).strip().upper().replace("-", " ").replace(".", " ")


def current(max_age_seconds: float = 60.0) -> Optional[Dict[str, dict]]:
    """``{symbol: {since, reason, exchange}}`` for every security halted now, or None when the list
    cannot be read."""
    if time.time() - float(_CACHE["at"]) < max_age_seconds:
        return _CACHE["rows"]
    rows: Optional[Dict[str, dict]] = None
    try:
        import httpx
        import pandas as pd
        r = httpx.get(URL, headers=_HEADERS, timeout=15.0, follow_redirects=True)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text.lstrip("﻿")), dtype=str).fillna("")
        rows = {}
        for _, x in df.iterrows():
            if str(x.get("Resume Date", "")).strip():
                continue
            rows[str(x["Symbol"]).strip().upper()] = {
                "since": f"{x.get('Halt Date', '')} {x.get('Halt Time', '')}".strip(),
                "reason": str(x.get("Reason", "")).strip(), "exchange": str(x.get("Exchange", "")).strip()}
    except Exception as e:                                       # noqa: BLE001
        if time.time() - float(_CACHE["warned"]) > 600:
            logger.warning(f"[halts] current-halt list unreadable ({e}) — halts are not checked this tick")
            _CACHE["warned"] = time.time()
        rows = None
    _CACHE.update(at=time.time(), rows=rows)
    return rows


def halted(ticker: str) -> Optional[dict]:
    """The halt in force on ``ticker`` now, or None (not halted, or the list is unreadable)."""
    rows = current()
    if not rows:
        return None
    return rows.get(_symbol(ticker)) or rows.get(str(ticker).strip().upper())


def reset() -> None:
    _CACHE.update(at=0.0, rows=None, warned=0.0)
