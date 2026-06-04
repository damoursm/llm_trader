"""Tests for src.broker.sizing — share sizing and risk caps."""

import pytest

from config.settings import settings
from src.broker.sizing import shares_for, within_caps


@pytest.mark.parametrize("equity,price,mult,base,expected", [
    (100_000, 50.0, 1.0, 0.05, 100),     # 5% × $100k / $50
    (100_000, 50.0, 1.5, 0.05, 150),     # confidence tier scales notional
    (100_000, 50.0, 2.0, 0.05, 200),
    (100_000, 33.33, 1.0, 0.05, 150),    # floor(5000/33.33)
    (100_000, 0.0, 1.0, 0.05, 0),        # bad price → 0 (never divide-by-zero)
    (0.0, 50.0, 1.0, 0.05, 0),           # no equity → 0
    (100_000, 50.0, 0.0, 0.05, 0),       # zero multiplier → 0
    (100_000, 50.0, None, 0.05, 100),    # None multiplier treated as 1.0
])
def test_shares_for(equity, price, mult, base, expected):
    assert shares_for(equity, price, mult, base) == expected


def test_shares_for_defaults_to_settings_base(monkeypatch):
    monkeypatch.setattr(settings, "broker_base_position_pct", 0.10)
    assert shares_for(100_000, 50.0, 1.0) == 200      # 10% × 100k / 50


def test_within_caps_max_positions(monkeypatch):
    monkeypatch.setattr(settings, "broker_max_positions", 3)
    assert within_caps(2, 0.0, 100_000)[0] is True
    allowed, reason = within_caps(3, 0.0, 100_000)
    assert allowed is False and "max_positions" in reason


def test_within_caps_gross_exposure(monkeypatch):
    monkeypatch.setattr(settings, "broker_max_positions", 100)
    monkeypatch.setattr(settings, "broker_max_gross_exposure_pct", 1.0)
    assert within_caps(0, 90_000, 100_000)[0] is True
    allowed, reason = within_caps(0, 110_000, 100_000)
    assert allowed is False and "exposure" in reason
