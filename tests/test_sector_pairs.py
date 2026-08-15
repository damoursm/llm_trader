"""Sector divergence pairs (`src/signals/sector_pairs.py`).

Builds market-neutral long/short pairs from a stock disagreeing with its sector
ETF. It is a REPORTING surface (the email's Sector Pairs block) — it opens no
trades — so the risk is not P&L but a mislabelled leg: an inverted long/short
assignment reads as a coherent recommendation and is impossible to spot from the
summary line. Hence the leg-orientation tests below are the core of this file.

The other invariant worth pinning is that aligned pairs are SKIPPED. Alignment
is already handled by the aggregator's sector-alignment confidence factor, so
emitting a pair there would double-count the same observation in two places.
"""

from __future__ import annotations

import pytest

from src.models import TickerSignal
from src.signals import sector_pairs as sp


def _sig(ticker: str, direction: str, confidence: float) -> TickerSignal:
    return TickerSignal(ticker=ticker, direction=direction, confidence=confidence,
                        sentiment_score=0.0, technical_score=0.0, rationale="t")


def _book(**kw) -> dict:
    """{ticker: TickerSignal} from ticker=(direction, confidence) kwargs."""
    return {t: _sig(t, d, c) for t, (d, c) in kw.items()}


# A stock/ETF pair that really is in the aggregator's map, so the scan reaches it.
_STOCK, _ETF = "AAPL", "XLK"


def test_the_fixture_pair_is_really_in_the_sector_map():
    """The scan iterates `aggregator._SECTOR_MAP`; if this pairing were removed
    every test below would pass vacuously by finding nothing."""
    from src.signals.aggregator import _SECTOR_MAP
    assert _SECTOR_MAP.get(_STOCK) == _ETF


# ── leg orientation ─────────────────────────────────────────────────────────

def test_bullish_etf_bearish_stock_goes_long_the_etf():
    """Sector tailwind the stock isn't catching → own the sector, short the
    laggard."""
    ctx = sp.find_sector_pairs(_book(**{_ETF: ("BULLISH", 0.8), _STOCK: ("BEARISH", 0.6)}))
    assert len(ctx.pairs) == 1
    p = ctx.pairs[0]
    assert p.setup_type == "ETF_BULL_STOCK_BEAR"
    assert (p.long_leg, p.short_leg) == (_ETF, _STOCK)
    assert p.etf_direction == "BULLISH" and p.stock_direction == "BEARISH"


def test_bearish_etf_bullish_stock_goes_long_the_stock():
    """The mirror: idiosyncratic strength against a sector headwind → own the
    stock, short the sector as the hedge."""
    ctx = sp.find_sector_pairs(_book(**{_ETF: ("BEARISH", 0.7), _STOCK: ("BULLISH", 0.9)}))
    assert len(ctx.pairs) == 1
    p = ctx.pairs[0]
    assert p.setup_type == "ETF_BEAR_STOCK_BULL"
    assert (p.long_leg, p.short_leg) == (_STOCK, _ETF)


def test_the_long_leg_is_always_the_bullish_one():
    """Stated as a property rather than per-case, because an inverted assignment
    still produces a well-formed, plausible-looking pair."""
    for etf_dir, stock_dir in (("BULLISH", "BEARISH"), ("BEARISH", "BULLISH")):
        ctx = sp.find_sector_pairs(
            _book(**{_ETF: (etf_dir, 0.8), _STOCK: (stock_dir, 0.8)}))
        p = ctx.pairs[0]
        bullish_leg = _ETF if etf_dir == "BULLISH" else _STOCK
        bearish_leg = _STOCK if etf_dir == "BULLISH" else _ETF
        assert p.long_leg == bullish_leg and p.short_leg == bearish_leg
        assert p.long_leg != p.short_leg


def test_rationale_names_both_legs():
    """The email renders this string verbatim; a rationale that doesn't mention
    the legs it is arguing for is the readable half of a mislabelling bug."""
    ctx = sp.find_sector_pairs(_book(**{_ETF: ("BULLISH", 0.8), _STOCK: ("BEARISH", 0.6)}))
    r = ctx.pairs[0].rationale
    assert _ETF in r and _STOCK in r and "Market-neutral" in r


# ── what must NOT produce a pair ────────────────────────────────────────────

def test_aligned_legs_produce_no_pair():
    """Both bullish (or both bearish) is not a divergence — and the aggregator's
    sector-alignment confidence factor already prices it."""
    for d in ("BULLISH", "BEARISH"):
        ctx = sp.find_sector_pairs(_book(**{_ETF: (d, 0.9), _STOCK: (d, 0.9)}))
        assert ctx.pairs == []
        assert "aligned" in ctx.summary


@pytest.mark.parametrize("etf_dir,stock_dir", [
    ("NEUTRAL", "BEARISH"), ("BULLISH", "NEUTRAL"), ("NEUTRAL", "NEUTRAL")])
def test_a_neutral_leg_produces_no_pair(etf_dir, stock_dir):
    """A pair needs a view on BOTH legs — 'no opinion' is not the opposite of
    bullish."""
    ctx = sp.find_sector_pairs(_book(**{_ETF: (etf_dir, 0.9), _STOCK: (stock_dir, 0.9)}))
    assert ctx.pairs == []


def test_a_missing_leg_produces_no_pair():
    """Only one side scored this run (the other wasn't in the universe)."""
    assert sp.find_sector_pairs(_book(**{_ETF: ("BULLISH", 0.9)})).pairs == []
    assert sp.find_sector_pairs(_book(**{_STOCK: ("BEARISH", 0.9)})).pairs == []
    assert sp.find_sector_pairs({}).pairs == []


def test_both_legs_must_clear_the_confidence_floor():
    lo = sp.MIN_LEG_CONFIDENCE - 0.01
    hi = sp.MIN_LEG_CONFIDENCE + 0.30
    assert sp.find_sector_pairs(
        _book(**{_ETF: ("BULLISH", lo), _STOCK: ("BEARISH", hi)})).pairs == []
    assert sp.find_sector_pairs(
        _book(**{_ETF: ("BULLISH", hi), _STOCK: ("BEARISH", lo)})).pairs == []
    # Exactly at the floor passes (the gate is `<`, not `<=`).
    assert sp.find_sector_pairs(_book(**{
        _ETF: ("BULLISH", sp.MIN_LEG_CONFIDENCE),
        _STOCK: ("BEARISH", sp.MIN_LEG_CONFIDENCE)})).pairs != []


# ── ranking and output bounds ───────────────────────────────────────────────

def test_pair_score_is_the_mean_of_the_two_leg_confidences():
    ctx = sp.find_sector_pairs(_book(**{_ETF: ("BULLISH", 0.9), _STOCK: ("BEARISH", 0.5)}))
    assert ctx.pairs[0].pair_score == pytest.approx(0.7)


def test_pairs_are_sorted_by_conviction_and_capped():
    """Ten pairs is the email's budget; the ones that survive must be the
    highest-conviction ones, not the first N in map order."""
    from src.signals.aggregator import _SECTOR_MAP
    book = {}
    # Give every mapped stock a bearish view and every ETF a bullish one, with
    # ascending stock confidence so the ranking has something to order.
    for i, (stock, etf) in enumerate(_SECTOR_MAP.items()):
        book[etf] = _sig(etf, "BULLISH", 0.90)
        book[stock] = _sig(stock, "BEARISH", round(0.40 + 0.01 * i, 3))
    ctx = sp.find_sector_pairs(book)
    assert len(ctx.pairs) == sp.MAX_PAIRS
    scores = [p.pair_score for p in ctx.pairs]
    assert scores == sorted(scores, reverse=True)
    # The cap keeps the TOP scores, so the weakest survivor beats the strongest
    # name that was cut.
    assert min(scores) > 0.40 + 0.01 * (len(_SECTOR_MAP) - sp.MAX_PAIRS - 1) / 2


def test_summary_reports_the_pairs_it_found():
    ctx = sp.find_sector_pairs(_book(**{_ETF: ("BULLISH", 0.8), _STOCK: ("BEARISH", 0.6)}))
    assert f"L {_ETF}/S {_STOCK}" in ctx.summary
    assert ctx.summary.startswith("1 pair(s)")


def test_each_stock_etf_combination_appears_once():
    """`_SECTOR_MAP` maps several stocks onto the same ETF, so the dedupe key is
    the (stock, etf) pair — an ETF appearing in several pairs is correct, the
    SAME pair twice is not."""
    from src.signals.aggregator import _SECTOR_MAP
    book = {}
    for stock, etf in _SECTOR_MAP.items():
        book[etf] = _sig(etf, "BULLISH", 0.9)
        book[stock] = _sig(stock, "BEARISH", 0.9)
    ctx = sp.find_sector_pairs(book)
    keys = [(p.stock, p.etf) for p in ctx.pairs]
    assert len(keys) == len(set(keys))
