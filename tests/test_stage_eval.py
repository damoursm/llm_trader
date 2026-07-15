"""Decision-funnel stage evaluation (tracker.compute_stage_eval, 2026-07-11).

Every deduped BUY/SELL recommendation is walked through the SAME four gates the
pipeline applied — reconstructed from each run's persisted context — and each
stage's survivors + each gate's drops are scored as pseudo-trades:

  • Gate 1 from runs.confidence_threshold vs recommendations.confidence,
  • Gate 2 from runs.allow_buys (SELLs always pass),
  • the final actionable set from recommendations.actionable (exact),
  • Gate 3-vs-4 attribution of the remaining drops from the per-ticker
    gate_diag.gate_outcomes stamp when present, else the run's drop counters.

All fakes — no DB, no OHLCV files, no network.
"""

import json
from datetime import date, timedelta

import pandas as pd
import pytest

import src.performance.daily_nav as dn
import src.performance.tracker as tr

D1, D2 = "2026-07-01", "2026-07-02"


def _bars(days_closes):
    idx = pd.to_datetime([d for d, _ in days_closes])
    return pd.DataFrame({"Close": [c for _, c in days_closes]}, index=idx)


@pytest.fixture
def fake_env(monkeypatch):
    """Fake runs/recommendations/signals frames + OHLCV so classification and
    stream membership are fully controlled."""
    # Every ticker: flat 100 → 110 closes so BUY wins, SELL loses (direction
    # legible in stats; exact returns are the cost model's business, not ours).
    closes = [(D1, 100.0), (D2, 105.0), ("2026-07-03", 110.0)]
    bars = {t: _bars(closes) for t in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "AGG")}

    import src.data.cache as cache_mod
    monkeypatch.setattr(cache_mod, "load_ohlcv", lambda tk, **kw: bars.get(tk))
    monkeypatch.setattr(dn, "_load_close_series",
                        lambda tk: {date.fromisoformat(d): c for d, c in closes})

    # Run r1 (D1): legacy — no gate_outcomes stamp; counters say untradeable=1,
    # blackout=0 → count-attribution must label CCC's drop Gate 4.
    # Run r2 (D2): stamped — gate_outcomes overrides everything, INCLUDING the
    # (deliberately lying) counters, so stamp precedence is proven.
    runs = pd.DataFrame([
        {"run_id": "r1", "confidence_threshold": 0.78, "allow_buys": True,
         "gate_diag": json.dumps({"dropped_earnings_blackout": 0, "dropped_untradeable": 1})},
        {"run_id": "r2", "confidence_threshold": 0.80, "allow_buys": False,
         "gate_diag": json.dumps({"dropped_earnings_blackout": 0, "dropped_untradeable": 5,
                                  "gate_outcomes": {"FFF": "earnings_blackout"}})},
    ])

    recs = pd.DataFrame([
        # r1 (D1): AAA passes all; BBB below threshold; CCC dropped → counters say Gate 4.
        {"run_id": "r1", "generated_at": f"{D1}T14:00:00+00:00", "ticker": "AAA",
         "type": "STOCK", "action": "BUY", "confidence": 0.90, "actionable": True, "snap_price": 100.0},
        {"run_id": "r1", "generated_at": f"{D1}T14:00:00+00:00", "ticker": "BBB",
         "type": "STOCK", "action": "BUY", "confidence": 0.50, "actionable": False, "snap_price": 100.0},
        {"run_id": "r1", "generated_at": f"{D1}T14:00:00+00:00", "ticker": "CCC",
         "type": "STOCK", "action": "BUY", "confidence": 0.85, "actionable": False, "snap_price": 100.0},
        # r2 (D2): DDD is a BUY under allow_buys=False → Gate 2; EEE is a SELL
        # (never BUY-blocked) and actionable; FFF stamped earnings_blackout.
        {"run_id": "r2", "generated_at": f"{D2}T14:00:00+00:00", "ticker": "DDD",
         "type": "STOCK", "action": "BUY", "confidence": 0.90, "actionable": False, "snap_price": 100.0},
        {"run_id": "r2", "generated_at": f"{D2}T14:00:00+00:00", "ticker": "EEE",
         "type": "STOCK", "action": "SELL", "confidence": 0.95, "actionable": True, "snap_price": 100.0},
        {"run_id": "r2", "generated_at": f"{D2}T14:00:00+00:00", "ticker": "FFF",
         "type": "STOCK", "action": "SELL", "confidence": 0.85, "actionable": False, "snap_price": 100.0},
    ])

    signals = pd.DataFrame([
        {"generated_at": f"{D1}T14:00:00+00:00", "ticker": "AGG", "type": "STOCK",
         "direction": "BULLISH", "price": 100.0},
    ])

    def fake_fetch_df(sql, params=None):
        s = " ".join(sql.split())
        if "FROM recommendations" in s:
            return recs.copy()
        if "FROM runs" in s:
            return runs.copy()
        if "FROM signals" in s:
            return signals.copy()
        return pd.DataFrame()

    monkeypatch.setattr(tr.repo, "fetch_df", fake_fetch_df)
    return {"runs": runs, "recs": recs}


def _by_label(rows):
    return {r["label"]: r for r in rows}

def _one(rows, needle):
    hits = [r for r in rows if needle in r["label"]]
    assert len(hits) == 1, f"expected exactly one row containing {needle!r}"
    return hits[0]


def test_funnel_counts_and_order(fake_env):
    rows = tr.compute_stage_eval()
    labels = [r["label"] for r in rows]

    # All ten rows present, funnel order, each gate's drops right after its survivors.
    assert labels[0] == "Aggregator (combined signal)"
    assert labels[1] == "LLM Synthesis (all BUY/SELL)"
    assert [l for l in labels[2:] if l.startswith("→")] == [
        "→ past Gate 1 · regime confidence threshold",
        "→ past Gate 2 · PANIC/RISK_OFF BUY-block",
        "→ past Gate 3 · earnings blackout",
        "→ past Gate 4 · liquidity floor = ACTIONABLE",
    ]
    assert len([l for l in labels if l.startswith("✂")]) == 4

    by = _by_label(rows)
    assert by["Aggregator (combined signal)"]["trades"] == 1            # AGG
    assert by["LLM Synthesis (all BUY/SELL)"]["trades"] == 6
    # BBB out at gate 1; DDD at gate 2; FFF at gate 3; CCC at gate 4.
    assert by["→ past Gate 1 · regime confidence threshold"]["trades"] == 5
    assert by["→ past Gate 2 · PANIC/RISK_OFF BUY-block"]["trades"] == 4
    assert by["→ past Gate 3 · earnings blackout"]["trades"] == 3
    assert by["→ past Gate 4 · liquidity floor = ACTIONABLE"]["trades"] == 2   # AAA + EEE
    assert _one(rows, "Gate 1 drops")["trades"] == 1
    assert _one(rows, "Gate 2 drops")["trades"] == 1
    assert _one(rows, "Gate 3 drops")["trades"] == 1
    assert _one(rows, "Gate 4 drops")["trades"] == 1


def test_gate2_only_blocks_buys(fake_env):
    """EEE (SELL, r2) must survive Gate 2 even though r2 has allow_buys=False."""
    rows = tr.compute_stage_eval()
    # The gate-2 survivor set is 4 = {AAA, CCC, EEE, FFF}; if SELLs were blocked
    # it would be 3. The only gate-2 drop is the BUY (DDD).
    assert _by_label(rows)["→ past Gate 2 · PANIC/RISK_OFF BUY-block"]["trades"] == 4
    assert _one(rows, "Gate 2 drops")["trades"] == 1


def test_stamp_overrides_lying_counters(fake_env):
    """r2's counters claim 5 untradeable drops / 0 blackout, but FFF's per-ticker
    gate_outcomes stamp says earnings_blackout — the stamp must win."""
    rows = tr.compute_stage_eval()
    assert _one(rows, "Gate 3 drops")["trades"] == 1     # FFF, via stamp
    # Gate 4 drop is CCC alone (r1 count attribution), NOT FFF.
    assert _one(rows, "Gate 4 drops")["trades"] == 1


def test_legacy_count_attribution_labels_gate4(fake_env):
    """r1 has no stamp; its counters (ut=1, eb=0) must attribute CCC to Gate 4 —
    so the gate-3 survivor count stays 3 (CCC still alive at that stage)."""
    rows = tr.compute_stage_eval()
    assert _by_label(rows)["→ past Gate 3 · earnings blackout"]["trades"] == 3


def test_empty_stage_reports_zero_row(fake_env, monkeypatch):
    """A gate that never fired must still render (n=0), not vanish."""
    # Make every rec actionable → all four drop rows are zero rows.
    recs = fake_env["recs"].copy()
    recs["actionable"] = True
    recs["confidence"] = 0.99
    runs = fake_env["runs"].copy()
    runs["allow_buys"] = True
    runs["gate_diag"] = json.dumps({"dropped_earnings_blackout": 0, "dropped_untradeable": 0})

    def fetch(sql, params=None):
        s = " ".join(sql.split())
        if "FROM recommendations" in s:
            return recs.copy()
        if "FROM runs" in s:
            return runs.copy()
        return pd.DataFrame()

    monkeypatch.setattr(tr.repo, "fetch_df", fetch)
    rows = tr.compute_stage_eval()
    for k in (1, 2, 3, 4):
        drop = _one(rows, f"Gate {k} drops")
        assert drop["trades"] == 0
        assert drop["win_rate"] is None
    assert _by_label(rows)["→ past Gate 4 · liquidity floor = ACTIONABLE"]["trades"] == 6


def test_window_cutoff_filters_old_calls(fake_env):
    """window_days must drop calls whose entry_date predates the cutoff."""
    days_back_d2 = (date.today() - date.fromisoformat(D2)).days
    rows = tr.compute_stage_eval(window_days=days_back_d2)   # keeps D2, drops D1
    assert _by_label(rows)["LLM Synthesis (all BUY/SELL)"]["trades"] == 3   # r2 only


def test_dedupe_keeps_last_call_per_ticker_day(fake_env, monkeypatch):
    """Two same-day calls on one ticker collapse to the LAST one (its run's
    context decides the gates), mirroring compute_macro_eval's dedupe."""
    recs = fake_env["recs"].copy()
    dup = {"run_id": "r1", "generated_at": f"{D1}T18:00:00+00:00", "ticker": "AAA",
           "type": "STOCK", "action": "BUY", "confidence": 0.10, "actionable": False,
           "snap_price": 100.0}
    recs = pd.concat([recs, pd.DataFrame([dup])], ignore_index=True)
    runs = fake_env["runs"]

    def fetch(sql, params=None):
        s = " ".join(sql.split())
        if "FROM recommendations" in s:
            return recs.copy()
        if "FROM runs" in s:
            return runs.copy()
        return pd.DataFrame()

    monkeypatch.setattr(tr.repo, "fetch_df", fetch)
    rows = tr.compute_stage_eval()
    by = _by_label(rows)
    assert by["LLM Synthesis (all BUY/SELL)"]["trades"] == 6           # still 6 deduped
    # AAA's LAST call (conf 0.10) is now a gate-1 drop: BBB + AAA = 2.
    assert _one(rows, "Gate 1 drops")["trades"] == 2
    assert by["→ past Gate 4 · liquidity floor = ACTIONABLE"]["trades"] == 1   # EEE only


def test_stats_shape_matches_macro_eval_rows(fake_env):
    """Rows must carry the segment-stats keys the dashboard table reads."""
    rows = tr.compute_stage_eval()
    for r in rows:
        for k in ("label", "group", "trades", "win_rate", "avg_return"):
            assert k in r
        assert r["group"] in ("stage", "dropped")
