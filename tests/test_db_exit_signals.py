"""exit_signals panel: insert round-trip, per-run idempotency, column constant
guard, and that the table is created by ensure_schema."""

import pytest

from config.settings import settings
from src.db.repo import _EXIT_SIGNAL_COLS


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "test.db"))


def _row(ticker="AAA", method="aggregator", score=0.5, direction="BULLISH", pos="p1"):
    return {"reviewed_at": "2026-06-09T14:00:00+00:00", "signal_date": "2026-06-09",
            "ticker": ticker, "position_id": pos, "entry_direction": direction,
            "method": method, "score": score, "price": 100.0}


def test_exit_signal_columns_constant():
    # Guard the insert column list against drift from the schema table definition.
    assert _EXIT_SIGNAL_COLS == [
        "run_id", "reviewed_at", "signal_date", "ticker", "position_id",
        "entry_direction", "method", "score", "price",
    ]


def test_exit_signals_table_created_by_schema(tmp_db):
    from src.db.connection import connect
    with connect() as conn:                       # write connect → ensure_schema runs
        cols = {r[1] for r in conn.execute("PRAGMA table_info('exit_signals')").fetchall()}
    assert {"run_id", "position_id", "entry_direction", "method", "score"} <= cols


def test_insert_exit_signals_roundtrip(tmp_db):
    from src.db import repo
    repo.insert_exit_signals("run-1", [_row("AAA", "aggregator", 0.5),
                                       _row("AAA", "llm_review", 0.8)])
    df = repo.fetch_df("SELECT * FROM exit_signals ORDER BY method", read_only=False)
    assert len(df) == 2
    assert set(df["method"]) == {"aggregator", "llm_review"}
    agg = df[df.method == "aggregator"].iloc[0]
    assert agg["score"] == pytest.approx(0.5)
    assert agg["entry_direction"] == "BULLISH"
    assert agg["position_id"] == "p1"
    assert agg["signal_date"] == "2026-06-09"


def test_insert_exit_signals_idempotent_per_run(tmp_db):
    from src.db import repo
    repo.insert_exit_signals("run-1", [_row()])
    repo.insert_exit_signals("run-1", [_row()])   # replaced (DELETE+INSERT per run)
    repo.insert_exit_signals("run-2", [_row()])   # appended
    df = repo.fetch_df("SELECT count(*) AS n FROM exit_signals", read_only=False)
    assert int(df.iloc[0]["n"]) == 2


def test_insert_exit_signals_empty_is_noop(tmp_db):
    from src.db import repo
    repo.insert_exit_signals("run-1", [])          # must not raise


def test_persist_exit_signals_hook_end_to_end(tmp_db):
    """The pipeline hook decomposes a held position into exit-method rows and
    persists them — synthesized llm_review + macro/aggregator overlays + the
    signal methods, each position-oriented."""
    from types import SimpleNamespace as NS
    from src import pipeline
    from src.models import TickerSignal

    sig = TickerSignal(ticker="AAA", direction="BULLISH", confidence=0.8,
                       sentiment_score=0.5, technical_score=0.0, rationale="t")
    sig.combined_score = 0.4
    open_trades = [{"ticker": "AAA", "action": "BUY", "direction": "BULLISH",
                    "recommendation_id": "pos-1", "current_price": 100.0}]
    hold_reviews = {"AAA": NS(action="BUY", confidence=0.75, direction="BULLISH")}
    pipeline._persist_exit_signals("run-1", hold_reviews, open_trades,
                                   {"AAA": sig}, NS(regime="PANIC"))

    from src.db import repo
    df = repo.fetch_df("SELECT * FROM exit_signals", read_only=False)
    by_method = {r["method"]: r for _, r in df.iterrows()}
    assert {"llm_review", "aggregator", "macro_regime", "news"} <= set(by_method)
    assert by_method["llm_review"]["score"] == pytest.approx(0.75)     # reaffirmed BUY
    assert by_method["aggregator"]["score"] == pytest.approx(0.4)      # combined × +1
    assert by_method["macro_regime"]["score"] == pytest.approx(-1.0)   # long in PANIC → exit
    assert by_method["news"]["score"] == pytest.approx(0.5)            # bullish news = hold a long
    assert by_method["llm_review"]["position_id"] == "pos-1"
    assert set(df["entry_direction"]) == {"BULLISH"}
