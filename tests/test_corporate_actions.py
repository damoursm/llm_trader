"""Corporate actions (Massive/Polygon dividends + splits) + synthesis injection."""

from datetime import date, timedelta

import pytest

import src.analysis.claude_analyst as ca
import src.data.corporate_actions as cac
import src.data.polygon_client as pc
from src.models import CorporateActionsContext, DividendEvent, TickerSignal

_TODAY = date.today()


# ── polygon_client calendars ───────────────────────────────────────────────────

def test_get_dividends_calendar(monkeypatch):
    monkeypatch.setattr(pc.settings, "polygon_api_key", "x")
    captured = {}

    def fake_paginated(path, params, max_pages=10):
        captured["path"] = path
        captured["params"] = params
        return [{"ticker": "KO", "ex_dividend_date": "2026-07-01", "cash_amount": 0.5}]

    monkeypatch.setattr(pc, "_get_paginated", fake_paginated)
    out = pc.get_dividends_calendar("2026-06-22", "2026-07-06")
    assert captured["path"] == "/v3/reference/dividends"
    assert captured["params"]["ex_dividend_date.gte"] == "2026-06-22"
    assert out[0]["ticker"] == "KO"


def test_get_splits_calendar_no_key(monkeypatch):
    monkeypatch.setattr(pc.settings, "polygon_api_key", "")
    assert pc.get_splits_calendar("2026-06-01", "2026-07-01") == []


# ── fetch_corporate_actions_context ────────────────────────────────────────────

def _isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(cac, "_cache_path", lambda: tmp_path / "ca.json")
    monkeypatch.setattr(cac.settings, "enable_corporate_actions", True)
    monkeypatch.setattr(cac.settings, "enable_fetch_data", True)
    monkeypatch.setattr(cac.polygon_client, "is_available", lambda: True)


def test_fetch_filters_to_universe_and_builds(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    exd = (_TODAY + timedelta(days=5)).isoformat()
    exe = (_TODAY + timedelta(days=10)).isoformat()
    monkeypatch.setattr(cac.polygon_client, "get_dividends_calendar",
        lambda s, e: [
            {"ticker": "KO", "ex_dividend_date": exd, "cash_amount": 0.5, "frequency": 4, "pay_date": exd},
            {"ticker": "ZZZZ", "ex_dividend_date": exd, "cash_amount": 1.0},   # outside universe
        ])
    monkeypatch.setattr(cac.polygon_client, "get_splits_calendar",
        lambda s, e: [{"ticker": "NVDA", "execution_date": exe, "split_from": 1, "split_to": 10}])

    ctx = cac.fetch_corporate_actions_context(["KO", "NVDA", "AAPL"])
    assert ctx is not None
    assert [d.ticker for d in ctx.dividends] == ["KO"]          # ZZZZ filtered out
    assert ctx.dividends[0].days_until_ex == 5
    assert ctx.splits[0].ticker == "NVDA" and ctx.splits[0].ratio == "10:1"
    assert (tmp_path / "ca.json").exists()


def test_fetch_disabled_returns_none(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(cac.settings, "enable_corporate_actions", False)
    assert cac.fetch_corporate_actions_context(["KO"]) is None


def test_fetch_empty_returns_none(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(cac.polygon_client, "get_dividends_calendar", lambda s, e: [])
    monkeypatch.setattr(cac.polygon_client, "get_splits_calendar", lambda s, e: [])
    assert cac.fetch_corporate_actions_context(["KO"]) is None


# ── prompt injection ───────────────────────────────────────────────────────────

def test_corp_actions_block_injected(monkeypatch):
    captured = {}
    monkeypatch.setattr(ca, "_call_claude_analyst",
                        lambda prompt, model=None: captured.update(prompt=prompt) or "[]")
    sig = TickerSignal(ticker="KO", direction="BULLISH", confidence=0.8,
                       sentiment_score=0.0, technical_score=0.0, rationale="t")
    ctx = CorporateActionsContext(
        dividends=[DividendEvent(ticker="KO", ex_dividend_date=_TODAY + timedelta(days=5),
                                 cash_amount=0.5, frequency=4, days_until_ex=5)],
        splits=[], report_date=_TODAY,
    )
    try:
        ca.generate_recommendations([sig], corporate_actions_context=ctx, force_engine="anthropic")
    except Exception:
        pass
    p = captured.get("prompt", "")
    assert "<corporate_actions_context>" in p and "KO: ex-div in 5d" in p
