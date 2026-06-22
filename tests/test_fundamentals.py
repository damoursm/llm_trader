"""Fundamentals source (Massive/Polygon financials & ratios).

Covers: the ratios batch parser, the FundamentalsContext builder (units + summary
formatting + flag/empty gating), and that the context is injected into the LLM
synthesis prompt as a <fundamentals_context> block.
"""

from datetime import date

import pytest

import src.analysis.claude_analyst as ca
import src.data.fundamentals as fund
import src.data.polygon_client as pc
from src.models import FundamentalsContext, FundamentalsSignal, TickerSignal

# A realistic ratios row (fractions for roe / dividend_yield, like the live API).
_AAPL = {
    "ticker": "AAPL", "date": "2026-06-18", "price": 298.0,
    "price_to_earnings": 35.71, "price_to_book": 41.1, "price_to_sales": 9.7,
    "ev_to_ebitda": 27.6, "return_on_equity": 1.151, "return_on_assets": 0.33,
    "debt_to_equity": 0.8, "dividend_yield": 0.0036, "current": 1.07,
    "free_cash_flow": 1.29e11, "market_cap": 4.377e12, "enterprise_value": 4.4e12,
}
_KO = {"ticker": "KO", "price_to_earnings": 24.9, "return_on_equity": 0.4074,
       "dividend_yield": 0.0321, "debt_to_equity": 1.3, "market_cap": 3.42e11}


# ── get_ratios_batch ───────────────────────────────────────────────────────────

def test_get_ratios_batch_parses_and_keys_by_ticker(monkeypatch):
    monkeypatch.setattr(pc.settings, "polygon_api_key", "x")
    captured = {}

    def fake_get(path, params=None):
        captured["path"] = path
        captured["any_of"] = (params or {}).get("ticker.any_of")
        assert "sort" not in (params or {})        # sort=date 400s — must be omitted
        return {"results": [_AAPL, _KO]}

    monkeypatch.setattr(pc, "_get", fake_get)
    out = pc.get_ratios_batch(["AAPL", "KO"])
    assert captured["path"] == "/stocks/financials/v1/ratios"
    assert captured["any_of"] == "AAPL,KO"
    assert set(out) == {"AAPL", "KO"}
    assert out["AAPL"]["price_to_earnings"] == 35.71


def test_get_ratios_batch_chunks(monkeypatch):
    monkeypatch.setattr(pc.settings, "polygon_api_key", "x")
    calls = []

    def fake_get(path, params=None):
        syms = (params or {})["ticker.any_of"].split(",")
        calls.append(syms)
        return {"results": [{"ticker": s, "price_to_earnings": 10.0} for s in syms]}

    monkeypatch.setattr(pc, "_get", fake_get)
    out = pc.get_ratios_batch(["AAPL", "KO", "MSFT"], chunk=2)
    assert calls == [["AAPL", "KO"], ["MSFT"]]
    assert set(out) == {"AAPL", "KO", "MSFT"}


def test_get_ratios_batch_no_key(monkeypatch):
    monkeypatch.setattr(pc.settings, "polygon_api_key", "")
    assert pc.get_ratios_batch(["AAPL"]) == {}


# ── fetch_fundamentals_context ─────────────────────────────────────────────────

def _isolate(monkeypatch, tmp_path):
    """Force a cache miss + a writable cache dir for the fetch tests."""
    monkeypatch.setattr(fund, "_cache_path", lambda: tmp_path / "fund.json")
    monkeypatch.setattr(fund.settings, "enable_fundamentals", True)
    monkeypatch.setattr(fund.settings, "enable_fetch_data", True)
    monkeypatch.setattr(fund.polygon_client, "is_available", lambda: True)


def test_fetch_context_builds_signals_and_units(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(fund.polygon_client, "get_ratios_batch",
                        lambda tks: {"AAPL": _AAPL, "KO": _KO})
    ctx = fund.fetch_fundamentals_context(["AAPL", "KO"])
    assert ctx is not None and len(ctx.signals) == 2
    aapl = next(s for s in ctx.signals if s.ticker == "AAPL")
    assert aapl.pe == 35.71 and aapl.ev_ebitda == 27.6
    # roe / dividend_yield are fractions → rendered as % in the summary
    assert "ROE 115%" in aapl.summary
    ko = next(s for s in ctx.signals if s.ticker == "KO")
    assert "yield 3.2%" in ko.summary
    assert (tmp_path / "fund.json").exists()           # cached for the day


def test_fetch_context_disabled_returns_none(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(fund.settings, "enable_fundamentals", False)
    monkeypatch.setattr(fund.polygon_client, "get_ratios_batch",
                        lambda tks: pytest.fail("should not fetch when disabled"))
    assert fund.fetch_fundamentals_context(["AAPL"]) is None


def test_fetch_context_empty_returns_none(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(fund.polygon_client, "get_ratios_batch", lambda tks: {})
    assert fund.fetch_fundamentals_context(["AAPL"]) is None


# ── prompt injection ───────────────────────────────────────────────────────────

def test_fundamentals_block_injected_into_prompt(monkeypatch):
    captured = {}

    def fake_call(prompt, model=None):
        captured["prompt"] = prompt
        return "[]"                                   # parse result irrelevant here

    monkeypatch.setattr(ca, "_call_claude_analyst", fake_call)

    sig = TickerSignal(ticker="AAPL", direction="BULLISH", confidence=0.8,
                       sentiment_score=0.0, technical_score=0.0, rationale="test")
    ctx = FundamentalsContext(
        signals=[FundamentalsSignal(ticker="AAPL", pe=35.7,
                                    summary="AAPL: P/E 35.7, ROE 115%")],
        report_date=date.today(),
    )
    try:
        ca.generate_recommendations([sig], fundamentals_context=ctx,
                                    force_engine="anthropic")
    except Exception:
        pass

    prompt = captured.get("prompt", "")
    assert "<fundamentals_context>" in prompt
    assert "AAPL: P/E 35.7, ROE 115%" in prompt


def test_no_fundamentals_block_when_context_absent(monkeypatch):
    captured = {}
    monkeypatch.setattr(ca, "_call_claude_analyst",
                        lambda prompt, model=None: captured.update(prompt=prompt) or "[]")
    sig = TickerSignal(ticker="AAPL", direction="BULLISH", confidence=0.8,
                       sentiment_score=0.0, technical_score=0.0, rationale="test")
    try:
        ca.generate_recommendations([sig], fundamentals_context=None,
                                    force_engine="anthropic")
    except Exception:
        pass
    assert "<fundamentals_context>" not in captured.get("prompt", "")
