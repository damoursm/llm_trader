"""Tests for src.performance.spread — half-spread tiers and _pct_return math."""

import math

import pytest

from src.performance.spread import _dynamic_half_spread, _pct_return, fmt_price


# ── _dynamic_half_spread tiers ────────────────────────────────────────────

@pytest.mark.parametrize("price,asset,expected_bp", [
    (500.0, "ETF",       1.5),  # SPY/QQQ-class
    (80.0,  "ETF",       2.5),  # sector ETFs
    (440.0, "STOCK",     3.0),  # mega-cap
    (80.0,  "STOCK",     4.0),  # large-cap
    (25.0,  "STOCK",     8.0),  # mid-cap
    (5.0,   "STOCK",    25.0),  # small-cap
    (0.50,  "STOCK",    75.0),  # micro-cap
    (0.05,  "STOCK",   250.0),  # penny
    (0.003, "STOCK",   500.0),  # sub-penny
    (430.0, "COMMODITY", 2.5),  # GLD-class
    (40.0,  "COMMODITY", 5.0),  # SLV-class
])
def test_half_spread_tier(price, asset, expected_bp):
    assert _dynamic_half_spread(price, asset) * 10000 == pytest.approx(expected_bp, rel=1e-9)


@pytest.mark.parametrize("price", [0, -1.0, None])
def test_half_spread_non_positive_returns_zero(price):
    """Non-positive prices yield zero spread — the model is undefined there."""
    assert _dynamic_half_spread(price) == 0.0


# ── _pct_return: sign convention and spread accounting ───────────────────

def test_pct_return_buy_profitable():
    # Long, $100 → $110.  Both prices sit in the **mega-cap** tier (≥$100 →
    # 3 bp half), not large-cap.  Hand math:
    #   eff_entry = 100 × 1.0003 = 100.03
    #   eff_exit  = 110 × 0.9997 = 109.967
    #   return    = (109.967 − 100.03) / 100.03 × 100 ≈ +9.934 %
    r = _pct_return("BUY", 100.0, 110.0, "STOCK")
    assert r == pytest.approx(9.934, abs=1e-3)


def test_pct_return_sell_profitable_when_price_falls():
    # Short, $100 → $90.  Entry $100 = mega-cap tier (3 bp), exit $90 = large-cap
    # tier (4 bp). Hand math:
    #   eff_entry = 100 × 0.9997 = 99.97
    #   eff_exit  =  90 × 1.0004 = 90.036
    #   return    = (99.97 − 90.036) / 99.97 × 100 ≈ +9.937 %
    r = _pct_return("SELL", 100.0, 90.0, "STOCK")
    assert r == pytest.approx(9.937, abs=1e-2)


def test_pct_return_buy_loses_to_round_trip_spread_when_flat():
    """Open and immediately close at the same price → loss = round-trip spread cost."""
    r = _pct_return("BUY", 100.0, 100.0, "STOCK")
    # Round trip on $100 mega-cap (3 bp each side) ≈ -6 bp
    assert r < 0
    assert r == pytest.approx(-0.06, abs=1e-2)


def test_pct_return_sell_loses_to_round_trip_spread_when_flat():
    """Short and immediately cover at the same price → same loss as a long."""
    r = _pct_return("SELL", 100.0, 100.0, "STOCK")
    assert r < 0
    assert r == pytest.approx(-0.06, abs=1e-2)


@pytest.mark.parametrize("entry,exit_p", [(-1.0, 50.0), (50.0, -1.0), (0.0, 50.0), (50.0, 0.0)])
def test_pct_return_non_positive_price_guard(entry, exit_p):
    """Non-positive prices return 0.0 instead of NaN/inf."""
    assert _pct_return("BUY", entry, exit_p, "STOCK") == 0.0


def test_pct_return_uses_correct_tier_at_each_leg():
    """Entry and exit spreads come from their OWN price, not a frozen tier."""
    # Penny → small-cap crossing.
    # entry tier ($0.50 micro-cap): 75 bp half
    # exit  tier ($5  small-cap):   25 bp half
    r = _pct_return("BUY", 0.50, 5.0, "STOCK")
    # eff_entry = 0.50 × 1.0075 = 0.50375
    # eff_exit  = 5.00 × 0.9975 = 4.9875
    # return    = (4.9875 − 0.50375) / 0.50375 × 100 ≈ +890 %
    assert r == pytest.approx(890.07, abs=1e-1)


# ── fmt_price formatting ──────────────────────────────────────────────────

@pytest.mark.parametrize("p,expected", [
    (12.34567,    "12.35"),
    (0.0312,      "0.0312"),
    (0.003142,    "0.003142"),
    (1.0,         "1.00"),
    (0.01,        "0.0100"),
    (0.00001,     "0.000010"),
])
def test_fmt_price_tiers(p, expected):
    assert fmt_price(p) == expected


def test_fmt_price_handles_none():
    assert fmt_price(None) == "N/A"
