"""yfinance's download serialized for the whole process (2026-10-12, `src/data/yf_guard.py`): yfinance 1.2.0 keeps
every download's results in one module-wide dict, so two at once wipe each other's (the 01:00 tick hung in one)."""
from __future__ import annotations

import threading
import time

import yfinance as yf

from src.data import yf_guard


def test_market_data_installs_the_guard():
    import src.data.market_data  # noqa: F401
    assert getattr(yf.download, "_serialized", False)


def test_two_downloads_never_overlap(monkeypatch):
    inside, overlap = [0], [False]

    def fake(*a, **k):
        inside[0] += 1
        overlap[0] = overlap[0] or inside[0] > 1
        time.sleep(0.05)
        inside[0] -= 1
        return a
    monkeypatch.setattr(yf, "download", fake)
    assert yf_guard.install() and yf.download is not fake and yf.download.__wrapped__ is fake
    ts = [threading.Thread(target=yf.download, args=(f"T{i}",)) for i in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(10)
    assert not overlap[0]
    assert yf.download("X", period="1d") == ("X",)
    assert yf_guard.install() and yf.download.__wrapped__ is fake          # once
