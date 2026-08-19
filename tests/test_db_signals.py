"""Tests for the signals panel: schema↔tracker drift guard, insert_signals
round-trip, forward-return join, and the IC computation."""

from datetime import date

import pandas as pd
import pytest

from config.settings import settings
from src.db.schema import (
    SIGNAL_BASE_METHOD_COLUMNS,
    SIGNAL_METHOD_COLUMNS,
    SIGNAL_TIMEFRAME_COLUMNS,
)


# ── drift guard: BASE schema columns must mirror tracker._ALL_METHODS ──────

def test_signal_base_columns_match_tracker():
    from src.performance.tracker import _ALL_METHODS
    assert tuple(SIGNAL_BASE_METHOD_COLUMNS) == tuple(_ALL_METHODS), (
        "schema.SIGNAL_BASE_METHOD_COLUMNS must mirror tracker._ALL_METHODS — "
        "the trade-attribution set. When adding a base method, add its column "
        "here (and ALTER TABLE signals ADD COLUMN <m> DOUBLE on existing DBs)."
    )


def test_method_categories_cover_every_attributed_method():
    """Every _ALL_METHODS entry must appear in exactly one METHOD_CATEGORIES bucket.

    2026-07-20 post-mortem: `massive` and `market_momentum` (promoted into the
    weighted combine 2026-06-24) plus all 6 fundamental/corp-action factors and
    all 4 trend-predictability methods were silently missing from
    METHOD_CATEGORIES — a method not in ANY category contributes to NO bundle in
    tracker.compute_macro_eval's "method bundle" rollup and NO category in
    tracker._compute_category_stats, with no error or "(uncategorized)" fallback
    to signal the gap. This guard makes that class of drift a test failure
    instead of a silent dashboard/email undercount."""
    from src.performance.tracker import _ALL_METHODS, METHOD_CATEGORIES
    categorized = set()
    for members in METHOD_CATEGORIES.values():
        categorized.update(members)
    missing = [m for m in _ALL_METHODS if m not in categorized]
    assert not missing, (
        f"{missing} in _ALL_METHODS but missing from every METHOD_CATEGORIES "
        "bucket — add each to whichever category it belongs to."
    )
    dupes = [m for m in categorized
            if sum(m in members for members in METHOD_CATEGORIES.values()) > 1]
    assert not dupes, f"{dupes} appear in MORE THAN ONE METHOD_CATEGORIES bucket"


def test_fundamentals_are_trade_attributed():
    # The 6 fundamental/corp-action factors are now in _ALL_METHODS and read from the
    # signal's fundamental_scores dict by _method_scores_from_signal → they show up in
    # the solo/eval Method-Performance tables (not just the Signal-IC table).
    from src.performance.tracker import _method_scores_from_signal, _ALL_METHODS
    from src.models import TickerSignal
    sig = TickerSignal(ticker="AAPL", direction="BULLISH", confidence=0.8,
                       sentiment_score=0.0, technical_score=0.0, rationale="test")
    sig.fundamental_scores = {"f_value": 0.6, "f_short_squeeze": -0.4}
    scores = _method_scores_from_signal("AAPL", "BULLISH", {"AAPL": sig})
    assert set(scores) == set(_ALL_METHODS)        # every attributed method present
    assert scores["f_value"] == 0.6
    assert scores["f_short_squeeze"] == -0.4
    assert scores["f_growth"] == 0.0               # absent factor → 0.0 (no view)
    for m in ("f_value", "f_quality", "f_growth", "f_short_squeeze", "f_split", "f_dividend"):
        assert m in scores


def test_every_getattr_default_names_a_real_TickerSignal_field():
    """A `getattr(sig, "<name>", <default>)` that misspells the field returns
    the DEFAULT forever — the column persists as 0.0 / 1.0 / "STOCK" on every
    row, which reads as "this method never fires" or "everything is a stock"
    rather than as a bug. Nothing warns, and `test_fundamentals_are_trade_
    attributed` above (set(scores) == set(_ALL_METHODS)) cannot see it: the KEY
    is still there, only its value is a fiction.

    Checked on the AST across the two places that read a TickerSignal by
    getattr — the trade-attribution extractor and the signals-panel row builder.

    KNOWN VIOLATION, pinned as an equality so it self-clears: `pipeline` reads
    `getattr(s, "type", "STOCK")` for the panel's asset-class column, and
    TickerSignal has no `type` field — so `signals.type` is the constant
    "STOCK" on every row ever written (the LLM-supplied type lives on
    `recommendations`, not on the signal). Nothing reads the column today. Fix
    the source and delete the entry here; do not add new ones."""
    import ast
    import inspect

    from src import pipeline
    from src.models import TickerSignal
    from src.performance import tracker

    KNOWN_FABRICATED = {"type"}
    fields = set(TickerSignal.model_fields)

    def _sig_getattr_names(fn_or_mod, var: str) -> set:
        tree = ast.parse(inspect.getsource(fn_or_mod).strip())
        return {n.args[1].value for n in ast.walk(tree)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "getattr" and len(n.args) >= 2
                and isinstance(n.args[0], ast.Name) and n.args[0].id == var
                and isinstance(n.args[1], ast.Constant)
                and isinstance(n.args[1].value, str)}

    extractor = _sig_getattr_names(tracker._method_scores_from_signal, "sig")
    assert extractor, "extractor no longer reads the signal by getattr — retarget this guard"
    assert not (extractor - fields), (
        f"{sorted(extractor - fields)} read off TickerSignal by getattr but are "
        f"not fields — every attributed trade would store the default instead")

    # The panel-row builder shares the loop variable name with unrelated
    # snapshot reads, so restrict to the names it actually persists as columns.
    from src.db.schema import (SIGNAL_ABS_SHADOW_COLUMNS,
                               SIGNAL_CONFIDENCE_COMPONENT_COLUMNS,
                               SIGNAL_NEWS_ATTENTION_COLUMNS,
                               SIGNAL_NEWS_EVENT_COLUMNS)
    persisted = (set(SIGNAL_CONFIDENCE_COMPONENT_COLUMNS)
                 | set(SIGNAL_NEWS_ATTENTION_COLUMNS)
                 | {c for c, _t in SIGNAL_NEWS_EVENT_COLUMNS}
                 | set(SIGNAL_ABS_SHADOW_COLUMNS)
                 | {"type", "combined_score", "combined_buy_score",
                    "combined_sell_score", "combine_source"})
    row_names = _sig_getattr_names(pipeline, "s") & persisted
    assert row_names, "panel-row builder no longer reads the signal — retarget this guard"
    assert (row_names - fields) == KNOWN_FABRICATED, (
        f"panel-row getattr defaults that name no TickerSignal field changed: "
        f"{sorted(row_names - fields)} (known: {sorted(KNOWN_FABRICATED)})")


def test_signal_timeframe_columns_convention():
    """The panel-only multi-timeframe columns follow ``{method}_{tf}`` for a
    known technical method × non-daily timeframe, and compose with the base
    set without overlap."""
    from src.signals.multi_timeframe import TECHNICAL_METHODS, NON_DAILY_TIMEFRAMES
    valid = {f"{m}_{tf}" for m in TECHNICAL_METHODS for tf in NON_DAILY_TIMEFRAMES}
    from src.db.schema import SIGNAL_FUNDAMENTAL_COLUMNS
    assert set(SIGNAL_TIMEFRAME_COLUMNS) == valid
    # Panel columns = base (trade-attributed) + timeframe diagnostics. The fundamental
    # factors are now PART OF the base set (solo-attributed, 2026-06-24);
    # SIGNAL_FUNDAMENTAL_COLUMNS is a categorisation SUBSET of BASE (the IC table's
    # "Fundamentals" grouping), no longer a separately-appended panel-only group.
    assert tuple(SIGNAL_METHOD_COLUMNS) == (tuple(SIGNAL_BASE_METHOD_COLUMNS)
                                            + tuple(SIGNAL_TIMEFRAME_COLUMNS))
    assert len(set(SIGNAL_METHOD_COLUMNS)) == len(SIGNAL_METHOD_COLUMNS)   # no dupes
    assert not (set(SIGNAL_BASE_METHOD_COLUMNS) & set(SIGNAL_TIMEFRAME_COLUMNS))
    assert set(SIGNAL_FUNDAMENTAL_COLUMNS) <= set(SIGNAL_BASE_METHOD_COLUMNS)


# ── insert_signals round-trip (temporary DuckDB file) ─────────────────────

@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))


def _row(ticker="AAPL", news=0.5):
    scores = {m: 0.0 for m in SIGNAL_METHOD_COLUMNS}
    scores["news"] = news
    return {"ticker": ticker, "type": "STOCK", "direction": "BULLISH",
            "combined_score": 0.4, "confidence": 0.8, "n_methods_agreeing": 3,
            "dominant_method": "news", "price": 100.0, "scores": scores}


def test_insert_signals_roundtrip(tmp_db):
    from src.db import repo
    repo.insert_signals("run-1", "2026-06-09T14:00:00+00:00", "2026-06-09",
                        [_row("AAPL", 0.5), _row("MSFT", -0.3)])
    df = repo.fetch_df("SELECT * FROM signals ORDER BY ticker", read_only=False)
    assert len(df) == 2
    assert df.iloc[0]["ticker"] == "AAPL"
    assert df.iloc[0]["news"] == pytest.approx(0.5)        # projected method column
    assert df.iloc[1]["news"] == pytest.approx(-0.3)
    assert df.iloc[0]["signal_date"] == "2026-06-09"
    assert df.iloc[0]["dominant_method"] == "news"
    assert '"news": 0.5' in df.iloc[0]["scores"]           # full dict kept as JSON


def test_insert_signals_timeframe_columns(tmp_db):
    from src.db import repo
    row = _row("AAPL")
    row["scores"]["tech_30m"] = 0.42
    row["scores"]["sector_momentum_1w"] = -0.31
    repo.insert_signals("run-1", "2026-06-09T14:00:00+00:00", "2026-06-09", [row])
    df = repo.fetch_df("SELECT * FROM signals", read_only=False)
    assert df.iloc[0]["tech_30m"] == pytest.approx(0.42)         # projected tf column
    assert df.iloc[0]["sector_momentum_1w"] == pytest.approx(-0.31)


def test_insert_signals_idempotent_per_run(tmp_db):
    from src.db import repo
    at = "2026-06-09T14:00:00+00:00"
    repo.insert_signals("run-1", at, "2026-06-09", [_row("AAPL")])
    repo.insert_signals("run-1", at, "2026-06-09", [_row("AAPL")])   # replaced
    repo.insert_signals("run-2", at, "2026-06-09", [_row("AAPL")])   # appended
    df = repo.fetch_df("SELECT count(*) AS n FROM signals", read_only=False)
    assert int(df.iloc[0]["n"]) == 2


# ── build_panel: forward-return join + per-day dedupe ─────────────────────

def test_build_panel_forward_returns(monkeypatch):
    import src.analysis.signal_panel as sp
    closes = {date(2026, 6, 1): 100.0, date(2026, 6, 2): 110.0,
              date(2026, 6, 3): 99.0, date(2026, 6, 4): 132.0}
    monkeypatch.setattr(sp, "_close_series", lambda tk: closes)
    sig = pd.DataFrame([
        {"generated_at": "t1", "signal_date": "2026-06-01", "ticker": "STK", "news": 0.5},
        {"generated_at": "t2", "signal_date": "2026-06-02", "ticker": "STK", "news": -0.2},
    ])
    panel = sp.build_panel(horizons=(1, 2), signals_df=sig)
    r1 = panel[panel.signal_date == "2026-06-01"].iloc[0]
    assert r1["fwd_ret_1d"] == pytest.approx(10.0)      # 100 → 110
    assert r1["fwd_ret_2d"] == pytest.approx(-1.0)      # 100 → 99
    r2 = panel[panel.signal_date == "2026-06-02"].iloc[0]
    assert r2["fwd_ret_1d"] == pytest.approx(-10.0)     # 110 → 99
    assert r2["fwd_ret_2d"] == pytest.approx(20.0)      # 110 → 132


def test_build_panel_fwd_nan_when_history_too_short(monkeypatch):
    import src.analysis.signal_panel as sp
    monkeypatch.setattr(sp, "_close_series",
                        lambda tk: {date(2026, 6, 1): 100.0})
    sig = pd.DataFrame([{"generated_at": "t1", "signal_date": "2026-06-01",
                         "ticker": "STK", "news": 0.5}])
    panel = sp.build_panel(horizons=(1,), signals_df=sig)
    assert pd.isna(panel.iloc[0]["fwd_ret_1d"])


def test_build_panel_dedupes_to_last_run_per_day(monkeypatch):
    # `tech` (no scorer epoch) rather than `news`: this test pins DEDUPE, and an
    # epoch-registered method's pre-epoch rows are correctly masked to NaN,
    # which would make the assertion test the mask instead (bitten 2026-08-14
    # when `news` gained an epoch).
    import src.analysis.signal_panel as sp
    monkeypatch.setattr(sp, "_close_series", lambda tk: {})
    sig = pd.DataFrame([
        {"generated_at": "2026-06-01T14:00:00", "signal_date": "2026-06-01",
         "ticker": "STK", "tech": 0.1},
        {"generated_at": "2026-06-01T20:00:00", "signal_date": "2026-06-01",
         "ticker": "STK", "tech": 0.9},     # later run wins
    ])
    panel = sp.build_panel(horizons=(1,), signals_df=sig, dedupe="last")
    assert len(panel) == 1
    assert panel.iloc[0]["tech"] == pytest.approx(0.9)
    # dedupe="all" keeps both
    assert len(sp.build_panel(horizons=(1,), signals_df=sig, dedupe="all")) == 2


# ── compute_ic ─────────────────────────────────────────────────────────────

def test_compute_ic_perfect_and_inverse_ranking():
    import src.analysis.signal_panel as sp
    n = 30
    panel = pd.DataFrame({
        "signal_date": ["2026-06-01"] * n,
        "ticker": [f"T{i}" for i in range(n)],
        "tech": [(i + 1) / n for i in range(n)],            # same ranking as fwd
        "vwap": [-(i + 1) / n for i in range(n)],           # inverse ranking
        "news": [0.0] * n,                                  # no view → excluded
        "fwd_ret_1d": [float(i + 1) for i in range(n)],
    })
    ic = sp.compute_ic(panel, horizons=(1,), min_n=10)
    tech = ic[ic.method == "tech"].iloc[0]
    assert tech["ic_1d"] == pytest.approx(1.0)
    assert tech["hit_1d"] == pytest.approx(100.0)
    vwap = ic[ic.method == "vwap"].iloc[0]
    assert vwap["ic_1d"] == pytest.approx(-1.0)
    assert vwap["hit_1d"] == pytest.approx(0.0)
    news = ic[ic.method == "news"].iloc[0]
    assert news["views"] == 0
    assert pd.isna(news["ic_1d"])                           # zero scores excluded


def test_compute_ic_min_n_gate():
    import src.analysis.signal_panel as sp
    panel = pd.DataFrame({
        "signal_date": ["2026-06-01"] * 5,
        "ticker": [f"T{i}" for i in range(5)],
        "tech": [0.1, 0.2, 0.3, 0.4, 0.5],
        "fwd_ret_1d": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    ic = sp.compute_ic(panel, horizons=(1,), min_n=20)
    tech = ic[ic.method == "tech"].iloc[0]
    assert tech["n_1d"] == 5
    assert pd.isna(tech["ic_1d"])                           # below min_n → unreported
