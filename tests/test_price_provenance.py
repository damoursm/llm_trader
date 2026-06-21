"""Tests for the standing price-provenance health check (item #8).

Generalizes the one-off CRDO fill-vs-snapshot audit into a per-run guard:
``pipeline._assess_price_provenance`` flags any trade opened this run whose
recorded ``entry_price`` diverges from the run's snapshot price beyond a
session-appropriate band.
"""

import types

import pytest

import src.pipeline as pipeline
from config.settings import settings


def _snap(ticker, price):
    return types.SimpleNamespace(ticker=ticker, price=price)


def _trade(ticker, entry, run_id="R1", session="rth"):
    return {"ticker": ticker, "entry_price": entry, "run_id": run_id, "entry_session": session}


def _patch_trades(monkeypatch, trades):
    monkeypatch.setattr("src.db.repo.load_trades", lambda: trades)


def test_within_band_not_down(monkeypatch):
    _patch_trades(monkeypatch, [_trade("AAA", 100.2)])     # 20 bp from snapshot
    v = pipeline._assess_price_provenance("R1", [_snap("AAA", 100.0)])
    assert v["down"] is False
    assert v["n_checked"] == 1
    assert v["flagged"] == []


def test_stale_price_flagged(monkeypatch):
    # The CRDO case: entered at the stale 250.81 close while the snapshot was 262.
    _patch_trades(monkeypatch, [_trade("CRDO", 250.81)])
    v = pipeline._assess_price_provenance("R1", [_snap("CRDO", 262.0)])
    assert v["down"] is True
    assert v["flagged"][0]["ticker"] == "CRDO"
    assert v["flagged"][0]["bps"] > 400
    assert "CRDO" in v["message"]


def test_session_band_selection(monkeypatch):
    """A ~300 bp gap is flagged under the tight RTH band but tolerated extended."""
    _patch_trades(monkeypatch, [_trade("X", 103.0, session="rth")])
    assert pipeline._assess_price_provenance("R1", [_snap("X", 100.0)])["down"] is True

    _patch_trades(monkeypatch, [_trade("X", 103.0, session="extended")])
    assert pipeline._assess_price_provenance("R1", [_snap("X", 100.0)])["down"] is False


def test_only_this_run_considered(monkeypatch):
    trades = [_trade("AAA", 999.0, run_id="OTHER"), _trade("BBB", 100.1, run_id="R1")]
    _patch_trades(monkeypatch, trades)
    v = pipeline._assess_price_provenance("R1", [_snap("AAA", 100.0), _snap("BBB", 100.0)])
    assert v["n_checked"] == 1          # the OTHER-run trade is ignored
    assert v["down"] is False


def test_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(settings, "enable_price_provenance_check", False)
    _patch_trades(monkeypatch, [_trade("AAA", 999.0)])
    assert pipeline._assess_price_provenance("R1", [_snap("AAA", 100.0)]) is None


def test_prev_close_anchor_is_skipped(monkeypatch):
    # A grouped-daily 'prev_close' fallback anchor must NOT be compared against a
    # live fill — that gap is legitimate and would falsely trip the band.
    _patch_trades(monkeypatch, [_trade("CRDO", 250.81)])
    snap = types.SimpleNamespace(ticker="CRDO", price=262.0, price_source="prev_close")
    assert pipeline._assess_price_provenance("R1", [snap]) is None   # no live anchors → nothing to check


def test_no_snapshots_returns_none(monkeypatch):
    _patch_trades(monkeypatch, [_trade("AAA", 100.0)])
    assert pipeline._assess_price_provenance("R1", []) is None
