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
    # defaults: no declarations / no per-ticker history (tests opt in) — never
    # hits the network
    monkeypatch.setattr(cac.polygon_client, "get_recent_dividend_declarations",
                        lambda gte, lte, max_pages=4: [])
    monkeypatch.setattr(cac.polygon_client, "get_dividend_history", lambda tk, limit=8: [])


def test_fetch_filters_to_universe_and_builds(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    exd = (_TODAY + timedelta(days=5)).isoformat()
    exe = (_TODAY + timedelta(days=10)).isoformat()
    monkeypatch.setattr(cac.polygon_client, "get_dividends_calendar",
        lambda s, e, order="asc", max_pages=10: [
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
    monkeypatch.setattr(cac.polygon_client, "get_dividends_calendar", lambda s, e, order="asc", max_pages=10: [])
    monkeypatch.setattr(cac.polygon_client, "get_splits_calendar", lambda s, e: [])
    assert cac.fetch_corporate_actions_context(["KO"]) is None


# ── directional factors (f_split, f_dividend) ─────────────────────────────────

def test_split_factor_direction():
    from src.models import SplitEvent
    fwd = [SplitEvent(ticker="X", execution_date=_TODAY, split_from=1, split_to=4, days_until=0)]
    rev = [SplitEvent(ticker="Y", execution_date=_TODAY, split_from=10, split_to=1, days_until=0)]
    assert cac._split_factor(fwd, 30) > 0       # forward → bullish
    assert cac._split_factor(rev, 30) < 0       # reverse → bearish (distress)
    assert cac._split_factor([], 30) is None


def test_dividend_change_prior_selection():
    """Latest regular vs the prior COMPARABLE regular — specials and frequency
    mixes (the old factor's fake-deep-cut artifacts) are skipped."""
    rows = [{"cash_amount": 0.30, "frequency": 4, "dividend_type": "CD"},
            {"cash_amount": 2.00, "frequency": 1, "dividend_type": "SD"},   # special — skipped
            {"cash_amount": 0.10, "frequency": 12, "dividend_type": "CD"},  # freq mix — skipped
            {"cash_amount": 0.25, "frequency": 4, "dividend_type": "CD"}]
    assert cac._dividend_change(rows) == pytest.approx(0.30 / 0.25 - 1)
    # one-row history is NOT an initiation signal any more (measured −0.61%)
    assert cac._dividend_change([{"cash_amount": 0.5, "frequency": 4, "dividend_type": "CD"}]) is None
    assert cac._dividend_change([]) is None
    # no comparable prior at all → abstain
    assert cac._dividend_change(
        [{"cash_amount": 0.5, "frequency": 4, "dividend_type": "CD"},
         {"cash_amount": 2.0, "frequency": 1, "dividend_type": "SD"}]) is None


def test_dividend_change_score_bands_and_decay():
    """Event-windowed, raise-only, with the measured abstain bands."""
    w = 10
    assert cac._dividend_change_score(0.25, 0, w) == pytest.approx(0.762, abs=1e-3)   # fresh huge raise
    assert cac._dividend_change_score(0.05, 0, w) == pytest.approx(0.197, abs=1e-3)   # small raise, mild
    assert cac._dividend_change_score(0.0, 0, w) is None                              # flat abstains
    # ALL cuts abstain — the big-cut drift did not replicate across year-halves.
    assert cac._dividend_change_score(-0.05, 0, w) is None
    assert cac._dividend_change_score(-0.50, 0, w) is None
    assert cac._dividend_change_score(0.25, w + 1, w) is None                         # past the window
    fresh = cac._dividend_change_score(0.25, 0, w)
    stale = cac._dividend_change_score(0.25, 8, w)
    assert 0 < stale < fresh                                                          # decay
    assert cac._dividend_change_score(None, 0, w) is None
    assert cac._dividend_change_score(0.25, None, w) is None


def test_context_carries_factor_scores(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    # KO declared a raise 2 days ago (0.53 vs prior comparable 0.48 → +10.4%);
    # ZZZZ declared too but is outside the universe.
    decl = (_TODAY - timedelta(days=2)).isoformat()
    monkeypatch.setattr(cac.polygon_client, "get_recent_dividend_declarations",
        lambda gte, lte, max_pages=4: [
            {"ticker": "KO", "declaration_date": decl, "cash_amount": 0.53,
             "frequency": 4, "dividend_type": "CD"},
            {"ticker": "ZZZZ", "declaration_date": decl, "cash_amount": 1.0,
             "frequency": 4, "dividend_type": "CD"}])
    monkeypatch.setattr(cac.polygon_client, "get_dividend_history",
        lambda tk, limit=8: ([{"cash_amount": 0.53, "frequency": 4, "dividend_type": "CD"},
                              {"cash_amount": 0.48, "frequency": 4, "dividend_type": "CD"}]
                             if tk == "KO" else []))
    monkeypatch.setattr(cac.polygon_client, "get_splits_calendar",
        lambda s, e: [{"ticker": "NVDA", "execution_date": _TODAY.isoformat(), "split_from": 1, "split_to": 10}])
    ctx = cac.fetch_corporate_actions_context(["KO", "NVDA"])
    assert ctx is not None
    assert ctx.factor_scores["KO"]["f_dividend"] > 0.2    # fresh declared raise
    assert "ZZZZ" not in ctx.factor_scores                # outside universe
    assert ctx.factor_scores["NVDA"]["f_split"] > 0       # forward 1→10 split


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
