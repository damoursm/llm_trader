"""SEC requests for the data feeds (`efts`, `plans`): the deep store's headers, at most one request every
``feeds_sec_pace_seconds``, and none inside production's own SEC lanes (``feeds_sec_quiet_et``: the nightly
refresh's from 23:45 and the pre-open run's from 08:30 ET share the SEC's 10 requests a second)."""
from __future__ import annotations

import time
from datetime import datetime, time as dtime
from typing import Callable

from config import settings

_last = [0.0]


def quiet(now: datetime) -> bool:
    """True inside one of ``feeds_sec_quiet_et``'s ET windows ("HH:MM-HH:MM,..."; a window may wrap midnight)."""
    from src.utils import ET
    t = now.astimezone(ET).time()
    for part in str(settings.feeds_sec_quiet_et or "").split(","):
        part = part.strip()
        if not part:
            continue
        a, b = (dtime.fromisoformat(x.strip()) for x in part.split("-"))
        if (a <= t < b) if a <= b else (t >= a or t < b):
            return True
    return False


def default_get() -> Callable:
    """A GET with the deep store's SEC headers, paced across every call in this process."""
    import httpx
    from src.data.deep import SEC_HEADERS
    client = httpx.Client(timeout=180.0, headers=SEC_HEADERS, follow_redirects=True)

    def get(url, params=None):
        wait = _last[0] + float(settings.feeds_sec_pace_seconds) - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()
        return client.get(url, params=params)
    return get
