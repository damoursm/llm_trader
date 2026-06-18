"""Tests for src.analysis.data_quality (source reliability + method coverage)."""

from types import SimpleNamespace

import pandas as pd

from src.analysis.data_quality import (
    compute_source_reliability, compute_method_coverage,
    source_status, is_unexpected_empty, is_context_populated,
)


def test_source_reliability_rates_order_and_last_error():
    df = pd.DataFrame([
        {"source_label": "good",  "ok": True,  "error": None,           "duration_s": 1.0, "started_at": "2026-06-15T10:00"},
        {"source_label": "good",  "ok": True,  "error": None,           "duration_s": 1.2, "started_at": "2026-06-16T10:00"},
        {"source_label": "flaky", "ok": True,  "error": None,           "duration_s": 2.0, "started_at": "2026-06-15T10:00"},
        {"source_label": "flaky", "ok": False, "error": "429 rate limit","duration_s": 5.0, "started_at": "2026-06-16T10:00"},
        {"source_label": "dead",  "ok": False, "error": "403 Forbidden", "duration_s": 0.5, "started_at": "2026-06-16T10:00"},
    ])
    rel = compute_source_reliability(df)
    by = {r["source"]: r for r in rel}
    assert by["good"]["success_rate"] == 100.0
    assert by["flaky"]["success_rate"] == 50.0
    assert by["dead"]["success_rate"] == 0.0
    assert by["flaky"]["last_error"] == "429 rate limit"   # most recent failing row
    assert rel[0]["source"] == "dead"     # worst-first
    assert rel[-1]["source"] == "good"


def test_source_reliability_empty():
    assert compute_source_reliability(pd.DataFrame()) == []


def test_method_coverage_and_drop_detection():
    # 4 tickers/day over 3 days. news is dense on the first 2 days then goes dark
    # on the most recent day (a feed dropping); tech is always dense; pead always
    # sparse (0 by design).
    rows = []
    for d, news in [("2026-06-14", 0.5), ("2026-06-15", 0.5), ("2026-06-16", 0.0)]:
        for _ in range(4):
            rows.append({"signal_date": d, "news": news, "tech": 0.3, "pead": 0.0})
    cov = compute_method_coverage(pd.DataFrame(rows), methods=["news", "tech", "pead"], recent_days=1)
    pm = {r["method"]: r for r in cov["per_method"]}

    assert pm["tech"]["coverage_pct"] == 100.0           # dense
    assert pm["pead"]["coverage_pct"] == 0.0             # sparse by design
    assert pm["news"]["coverage_pct"] == round(100.0 * 8 / 12, 1)  # 2 of 3 days scored
    assert pm["news"]["delta"] == -100.0                 # recent day went dark
    assert cov["per_method"][0]["method"] == "news"      # biggest drop surfaces first
    assert cov["n_rows"] == 12
    assert cov["days"] == ["2026-06-14", "2026-06-15", "2026-06-16"]


def test_method_coverage_empty():
    cov = compute_method_coverage(pd.DataFrame(), methods=["news"])
    assert cov["n_rows"] == 0 and cov["per_method"] == []


def test_source_status_and_unexpected_helpers():
    assert source_status(False, False) == "error"
    assert source_status(True, True) == "empty"
    assert source_status(True, False) == "ok"
    # known-dead feeds report 'dead', not 'empty' (and self-heal to 'ok' if data returns)
    assert source_status(True, True, "insider") == "dead"
    assert source_status(True, True, "tick") == "dead"
    assert source_status(True, False, "insider") == "ok"
    # vix is an always-on feed → empty is unexpected; 8k is event-driven → not;
    # a dead feed reports 'dead' so it is never flagged unexpected.
    assert is_unexpected_empty("vix", "empty") is True
    assert is_unexpected_empty("8k", "empty") is False
    assert is_unexpected_empty("vix", "ok") is False
    assert is_unexpected_empty("insider", source_status(True, True, "insider")) is False


def test_context_populated_detects_hollow_and_partial():
    # present level → populated
    assert is_context_populated("vix", SimpleNamespace(vix=18.4)) is True
    # hollow: None, and 0 (never a legit VIX level) → not populated
    assert is_context_populated("vix", SimpleNamespace(vix=None)) is False
    assert is_context_populated("vix", SimpleNamespace(vix=0)) is False
    # multi-component PARTIAL: DXY ok but copper/gold + oil missing → not populated
    assert is_context_populated(
        "global_macro",
        SimpleNamespace(dxy=99.0, copper_gold_ratio=None, oil_price=None)) is False
    assert is_context_populated(
        "global_macro",
        SimpleNamespace(dxy=99.0, copper_gold_ratio=0.21, oil_price=78.0)) is True
    # list payloads: empty → not populated, non-empty → populated
    assert is_context_populated("sector_rotation", SimpleNamespace(sectors=[])) is False
    assert is_context_populated("sector_rotation", SimpleNamespace(sectors=["XLK"])) is True
    # a None / list result is not this check's job → None (no opinion)
    assert is_context_populated("vix", None) is None
    assert is_context_populated("news", ["article"]) is None
    # unknown source → None (no registered check)
    assert is_context_populated("nope", SimpleNamespace(x=1)) is None
    # an object exposing its own is_populated() wins
    assert is_context_populated("vix", SimpleNamespace(is_populated=lambda: False)) is False


def test_dead_set_is_tick_and_insider():
    # ^TICK + congressional are dead-no-free-fix; mcclellan was RESTORED via the
    # Polygon RANA breadth source, so it is an always-on source again (an empty
    # mcclellan now reads 'empty ⚠', i.e. a real regression worth investigating).
    from src.analysis.data_quality import KNOWN_DEAD_SOURCES
    assert KNOWN_DEAD_SOURCES == frozenset({"tick", "insider"})
    assert source_status(True, True, "tick") == "dead"
    assert source_status(True, True, "mcclellan") == "empty"


def test_source_reliability_dead_source_not_flagged():
    # insider's only feeds (congressional) are 403-dead → it returns nothing every
    # run. It must show status 'dead' (visible), NOT 'empty ⚠' (false alarm) nor
    # silently 'ok'.
    df = pd.DataFrame([
        {"source_label": "insider", "ok": True, "error": None, "duration_s": 1.2,
         "started_at": "2026-06-16T10:00", "n_items": 0, "empty": True},
    ])
    rel = compute_source_reliability(df)
    r = rel[0]
    assert r["last_status"] == "dead"
    assert r["known_dead"] is True
    assert r["unexpected_empty"] is False


def test_source_reliability_emptiness_is_first_class():
    df = pd.DataFrame([
        # always-on feed that returned data, then went dark on its latest run
        {"source_label": "vix", "ok": True, "error": None, "duration_s": 1.0,
         "started_at": "2026-06-15T10:00", "n_items": 5, "empty": False},
        {"source_label": "vix", "ok": True, "error": None, "duration_s": 1.0,
         "started_at": "2026-06-16T10:00", "n_items": 0, "empty": True},
        # event-driven feed empty on its latest run — expected, NOT flagged
        {"source_label": "8k", "ok": True, "error": None, "duration_s": 1.0,
         "started_at": "2026-06-16T10:00", "n_items": 0, "empty": True},
        # healthy feed
        {"source_label": "news", "ok": True, "error": None, "duration_s": 1.0,
         "started_at": "2026-06-16T10:00", "n_items": 50, "empty": False},
    ])
    rel = compute_source_reliability(df)
    by = {r["source"]: r for r in rel}

    assert by["vix"]["last_status"] == "empty"
    assert by["vix"]["unexpected_empty"] is True       # always-on feed dark → investigate
    assert by["vix"]["empty_rate"] == 50.0             # 1 of 2 successful runs empty
    assert by["8k"]["last_status"] == "empty"
    assert by["8k"]["expected_sparse"] is True
    assert by["8k"]["unexpected_empty"] is False        # event-driven → not flagged
    assert by["news"]["last_status"] == "ok"
    assert by["news"]["unexpected_empty"] is False
    # an unexpected-empty always-on feed sorts ahead of healthy + expected-empty
    assert rel[0]["source"] == "vix"


def test_source_reliability_legacy_rows_without_empty_column():
    # A DB not yet carrying the n_items/empty columns must still work (no crash,
    # empty_rate None, status from ok only).
    df = pd.DataFrame([
        {"source_label": "vix", "ok": True, "error": None, "duration_s": 1.0,
         "started_at": "2026-06-16T10:00"},
    ])
    rel = compute_source_reliability(df)
    assert rel[0]["empty_rate"] is None
    assert rel[0]["last_status"] == "ok"
    assert rel[0]["unexpected_empty"] is False
