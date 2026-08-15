"""Operational tooling — charts, scorecard, migrations, the broker smoke test.

These are the modules a human runs by hand, usually while something is already
wrong. None of them make trading decisions, so the risk is not signal quality:
it is that a tool DESTROYS something, or that it dies on the degraded input it
exists to inspect.

So the contract here is narrow and blunt:

* **the destructive ones are opt-in.** `migrate` must refuse to overwrite a
  populated ledger without `--force`, `backfill_rule_fill_provider.run()` must
  default to a dry run, and the broker smoke test must not place an order
  without `--order`. Each of these is one flag away from an irreversible action
  taken by someone debugging at speed;
* **the reporting ones degrade.** A chart builder that raises on a ticker with
  no history takes the whole email down; a scorecard that raises on an empty
  ledger is useless on exactly the day you first want it.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.models import Recommendation


# ── charts ──────────────────────────────────────────────────────────────────

def _rec(ticker="AAPL", action="BUY") -> Recommendation:
    return Recommendation(ticker=ticker, type="STOCK", direction="BULLISH",
                          action=action, confidence=0.8, rationale="r",
                          generated_at=datetime.now(timezone.utc))


def _ohlcv(n=90) -> pd.DataFrame:
    idx = pd.date_range("2026-05-01", periods=n, freq="D")
    close = pd.Series([100.0 + i * 0.3 for i in range(n)], index=idx)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                         "Close": close, "Volume": [1e6] * n}, index=idx)


def test_a_ticker_with_no_history_yields_no_chart_rather_than_raising(monkeypatch):
    """Charts are built per recommendation inside the email path; one bad
    ticker must not cost the whole report."""
    from src.charts import builder
    monkeypatch.setattr(builder, "_fetch_ohlcv", lambda *a, **k: None)
    assert builder.build_stock_chart("ZZZZ", _rec("ZZZZ")) is None


def test_a_chart_is_built_from_usable_history(monkeypatch):
    from src.charts import builder
    monkeypatch.setattr(builder, "_fetch_ohlcv", lambda *a, **k: _ohlcv())
    fig = builder.build_stock_chart("AAPL", _rec())
    assert fig is not None and hasattr(fig, "to_dict")


def test_indicators_do_not_shorten_the_frame(monkeypatch):
    """The overlays are added as columns; dropping the warm-up rows here would
    silently truncate every chart's left edge."""
    from src.charts import builder
    df = _ohlcv()
    out = builder._add_indicators(df)
    assert len(out) == len(df)
    assert set(df.columns) <= set(out.columns)


def test_an_empty_equity_curve_is_none_not_an_empty_figure():
    from src.charts import builder
    assert builder.build_equity_curve([]) is None


def test_the_signals_overview_handles_an_empty_universe():
    from src.charts import builder
    out = builder.build_signals_overview([])
    assert out is None or hasattr(out, "to_dict")


def test_every_action_has_a_chart_colour():
    """A missing key is a KeyError mid-render, i.e. no email — and the actions
    are a closed set the recommendation model already enforces."""
    from src.charts.builder import ACTION_COLORS
    for action in ("BUY", "SELL", "HOLD", "WATCH"):
        assert action in ACTION_COLORS


def test_png_conversion_is_fail_soft(monkeypatch):
    """kaleido is deliberately NOT a production dependency (ENABLE_CHARTS is
    off); the converter has to return None rather than explode when it is
    missing."""
    from src.charts import builder
    assert builder.fig_to_png_b64(None) is None


# ── scorecard ───────────────────────────────────────────────────────────────

def test_the_scorecard_renders_on_an_empty_database(empty_db):
    """The first thing a new deployment runs, when there is nothing to score.
    Every section has to degrade independently — one empty table must not cost
    the other sections' output."""
    from src.analysis.scorecard import build_scorecard
    out = build_scorecard(days=7)
    assert isinstance(out, str) and out.strip()
    for section in ("ENTRY", "EXIT", "TRADES"):
        assert section in out, f"the {section} section vanished on an empty DB"


def test_the_scorecard_reports_the_window_it_covers(empty_db):
    from src.analysis.scorecard import build_scorecard
    assert "7" in build_scorecard(days=7)


def test_closed_filter_keeps_only_closed_trades():
    from src.analysis.scorecard import _closed
    trades = [{"status": "CLOSED", "ticker": "A"}, {"status": "OPEN", "ticker": "B"},
              {"ticker": "C"}]
    assert [t["ticker"] for t in _closed(trades)] == ["A"]


def test_number_formatting_passes_non_floats_through_untouched():
    """Every row is built from possibly-absent metrics, so the formatter has to
    accept them. It rounds floats and passes everything else through verbatim —
    including None, which renders as the literal 'None' rather than raising."""
    from src.analysis.scorecard import _fmt
    assert _fmt(1.23456) == 1.235
    assert _fmt(1.23456, nd=1) == 1.2
    for v in (None, "n/a", 7):
        assert _fmt(v) is v


# ── migrate: must not overwrite a populated ledger ─────────────────────────

def test_migrate_refuses_to_overwrite_a_populated_ledger(monkeypatch, tmp_path):
    """The legacy JSON import is a ONE-TIME operation. Running it against a live
    DuckDB ledger without `--force` must be a no-op — `save_trades` is a
    full-replace, so a stray run would swap months of history for whatever is
    in an old cache file."""
    from src.db import migrate as mig
    saved = {}
    monkeypatch.setattr(mig, "_count", lambda table: 400)
    monkeypatch.setattr(mig.repo, "save_trades",
                        lambda t: saved.setdefault("trades", t))
    monkeypatch.setattr(mig.repo, "save_hypothetical",
                        lambda t: saved.setdefault("hyp", t))
    monkeypatch.setattr(mig, "_read_json", lambda p: [{"ticker": "JUNK"}])
    mig.migrate()
    assert saved == {}, "migrate overwrote a populated ledger without --force"


def test_migrate_imports_into_an_empty_ledger(monkeypatch):
    from src.db import migrate as mig
    saved = {}
    monkeypatch.setattr(mig, "_count", lambda table: 0)
    monkeypatch.setattr(mig.repo, "save_trades", lambda t: saved.setdefault("trades", t))
    monkeypatch.setattr(mig.repo, "save_hypothetical", lambda t: saved.setdefault("hyp", t))
    monkeypatch.setattr(mig, "_read_json", lambda p: [{"ticker": "AAPL"}])
    mig.migrate()
    assert saved["trades"] == [{"ticker": "AAPL"}]


def test_migrate_force_overwrites_deliberately(monkeypatch):
    from src.db import migrate as mig
    saved = {}
    monkeypatch.setattr(mig, "_count", lambda table: 400)
    monkeypatch.setattr(mig.repo, "save_trades", lambda t: saved.setdefault("trades", t))
    monkeypatch.setattr(mig.repo, "save_hypothetical", lambda t: saved.setdefault("hyp", t))
    monkeypatch.setattr(mig, "_read_json", lambda p: [{"ticker": "AAPL"}])
    mig.migrate(force=True)
    assert "trades" in saved


def test_migrate_reads_a_missing_json_as_empty(tmp_path, monkeypatch):
    from src.db import migrate as mig
    assert mig._read_json(tmp_path / "absent.json") == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert mig._read_json(bad) == []


# ── the rule-fill backfill: dry run by default ─────────────────────────────

def test_the_backfill_is_a_dry_run_unless_applied(tmp_db_with_recs, monkeypatch):
    """A one-shot UPDATE over the recommendations table. The default must
    report, not write."""
    from src.db import backfill_rule_fill_provider as bf
    before = tmp_db_with_recs()
    n = bf.run(apply=False)
    assert n >= 0
    assert tmp_db_with_recs() == before, "dry run modified the table"


def test_the_backfill_predicate_excludes_already_stamped_rows():
    """Idempotency: a second `--apply` must not re-stamp (or count) rows that
    already carry the rule-fill label."""
    from src.db.backfill_rule_fill_provider import _where, _RULE_LABELS
    where = _where()
    for label in _RULE_LABELS:
        assert label in where
    assert "NOT IN" in where


# ── the broker smoke test: no trading without --order ──────────────────────

def test_the_smoketest_places_no_order_by_default(monkeypatch):
    """It is run against a LIVE-capable gateway while debugging. Placing an
    order on the default invocation is the one thing it must never do."""
    import src.broker.smoketest as st

    submitted = []

    class _FakeBroker:
        host, port, client_id = "127.0.0.1", 4002, 11

        def connect(self):
            return True

        def get_account(self):
            return {"NetLiquidation": 100000.0}

        def get_positions(self):
            return []

        def submit_order(self, req):
            submitted.append(req)
            raise AssertionError("smoketest submitted an order without --order")

        def disconnect(self):
            return None

    monkeypatch.setattr(st, "IBKRBroker", _FakeBroker)
    monkeypatch.setattr(sys, "argv", ["smoketest"])
    st.main()
    assert submitted == []


def test_the_smoketest_stops_cleanly_when_it_cannot_connect(monkeypatch):
    import src.broker.smoketest as st

    class _Dead:
        host, port, client_id = "127.0.0.1", 4002, 11

        def connect(self):
            return False

        def get_account(self):
            raise AssertionError("queried a broker it never connected to")

        def disconnect(self):
            return None

    monkeypatch.setattr(st, "IBKRBroker", _Dead)
    monkeypatch.setattr(sys, "argv", ["smoketest"])
    st.main()


def test_the_live_round_trip_is_refused_while_the_market_is_closed(monkeypatch):
    """A market order placed after hours sits pending and fills at the open —
    an unattended position opened by a diagnostic."""
    import src.broker.smoketest as st

    submitted = []

    class _FakeBroker:
        host, port, client_id = "127.0.0.1", 4002, 11

        def connect(self):
            return True

        def get_account(self):
            return {}

        def get_positions(self):
            return []

        def submit_order(self, req):
            submitted.append(req)
            raise AssertionError("placed a market order with the market closed")

        def disconnect(self):
            return None

    monkeypatch.setattr(st, "IBKRBroker", _FakeBroker)
    monkeypatch.setattr(st, "_market_open", lambda: False)
    monkeypatch.setattr(sys, "argv", ["smoketest", "--order"])
    st.main()
    assert submitted == []


# ── fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_db():
    """Create the schema on the throwaway DB without inserting anything — the
    read paths open read-only and refuse a file that does not exist yet."""
    from src.db.connection import connect
    with connect(read_only=False) as conn:
        conn.execute("SELECT 1")
    return True


@pytest.fixture
def tmp_db_with_recs():
    """Seed the throwaway DB with one legacy rule-based fill and return a
    snapshot callable."""
    from src.db.connection import connect

    with connect(read_only=False) as conn:
        conn.execute(
            "INSERT INTO recommendations (rec_id, run_id, generated_at, ticker, "
            "action, confidence, rationale, llm_provider) VALUES "
            "('rec-1', 'r1', ?, 'AAPL', 'BUY', 1.0, "
            "'rule-based fill: aggregator', 'claude-opus')",
            [datetime.now(timezone.utc).isoformat()])

    def _snapshot():
        with connect(read_only=False) as conn:
            return conn.execute(
                "SELECT ticker, llm_provider FROM recommendations ORDER BY ticker"
            ).fetchall()

    return _snapshot
