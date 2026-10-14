"""THE RECURRING OVERFITTING AUDIT (src/backtest/audit.py; PREREG106 item 4): PREREG103's evaluation rerun from saved
outputs — the registered labels, set D labelled only once it holds enough trades, a label that moved down since the
previous audit (or PREREG103, audit zero) as a note for the scorer line, never the research harness. Every test builds
its saved outputs in a temporary directory; the parity test reads PREREG103's own (skipped where they are absent)."""
from __future__ import annotations

import ast
import json
import pickle
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest import audit as AU
from src.backtest import trials as TR

ET = "America/New_York"
NOW = "2026-10-20 17:00"                 # ET: set D = 2026-09-25 .. 2026-10-20
START = {"A0": "2019-01-02", "A": "2022-01-03", "B": "2025-01-02", "D": "2026-09-25"}


def _bdays(s: str, n: int):
    days = pd.bdate_range(START[s], periods=n)
    if s == "D":
        days = pd.bdate_range(START[s], "2026-10-20")
        days = [days[i % len(days)] for i in range(n)]
    return [d.date() for d in days]


def _trades(spec, seed=0) -> pd.DataFrame:
    """The cards' trade table: spec = [(book, set, n, mean net %, hold days)]."""
    rng = np.random.default_rng(seed)
    recs = []
    for book, s, n, mu, hold in spec:
        for i, d in enumerate(_bdays(s, n)):
            ent = pd.Timestamp(f"{d} 10:00", tz=ET)
            net = float(rng.normal(mu, 2.0))
            recs.append({"book": book, "ticker": f"{book}{s}{i}", "entry": ent, "exit": ent + pd.Timedelta(days=hold),
                         "days": float(hold), "net": net, "gross": net + 0.2, "pnl": net * 10.0, "decision": d})
    return pd.DataFrame(recs)


def _card(root: Path, name: str, df: pd.DataFrame) -> Path:
    p = root / name
    p.mkdir(parents=True, exist_ok=True)
    df.to_pickle(p / "trades_base.pkl")
    return p


GOOD = 3.0       # % a trade over 5 days: return per day ~ +0.6%, far from 0 with 40 trades
BAD = -1.0


def _world(tmp_path, d_trades: int = 0, prev_rows=None, baseline_rows=None):
    """Saved outputs: the A/B and A0 builds (and a D build), three account draws a set, PREREG103's file, a seeded
    ledger, a live calibration record, and optionally an earlier audit."""
    rel = tmp_path / "rel"
    ab = [(b, s, 40, GOOD, 5.0) for b in ("vol", "etf", "fall", "dip") for s in ("A", "B")] + [("crash", "A", 6, 5.0, 4.0)]
    _card(rel, "2026-10-13", _trades(ab, 1))
    a0 = [("vol", "A0", 40, GOOD, 5.0), ("etf", "A0", 40, BAD, 5.0), ("fall", "A0", 40, GOOD, 5.0),
          ("dip", "A0", 40, GOOD, 5.0)]
    _card(rel, "2026-10-14_A0", _trades(a0, 2))
    if d_trades:
        _card(rel, "2026-10-20_D", _trades([("vol", "D", d_trades, GOOD, 5.0)], 3))
    draws = tmp_path / "draws"
    draws.mkdir()
    rng = np.random.default_rng(4)
    for s in ("A0", "A", "B"):
        for sd in (0, 1, 2):
            pos = []
            for i, d in enumerate(_bdays(s, 30)):
                e = pd.Timestamp(f"{d} 10:00", tz=ET).tz_convert("UTC")
                r = float(rng.normal(0.02, 0.03))
                pos.append({"strategy": "vol", "tkn": f"X{i}", "entry_ns": e.value,
                            "exit_ns": (e + pd.Timedelta(days=4)).value, "ret": r, "pnl": r * 500.0})
            with open(draws / f"{s}_LIVE_S2_s{sd}.pkl", "wb") as fh:
                pickle.dump({"pos": pos, "growth": 0.3, "maxdd": 15.0, "calls": [], "rpd_all": 0.4}, fh)
    seeds = tmp_path / "account101.json"
    seeds.write_text(json.dumps({"seeds": [0, 1, 2]}), encoding="utf-8")
    base = tmp_path / "global103.json"
    rows = baseline_rows if baseline_rows is not None else [
        {"model": "dip", "oos": "LONG", "label": "ROBUST", "decay": 0.93, "oos_lo95": 0.13, "oos_hi95": 0.2},
        {"model": "etf", "oos": "A0", "label": "ROBUST", "decay": 0.9}]
    base.write_text(json.dumps({"built_at": "2026-10-14T11:37:17-04:00", "table3": {"rows": rows},
                                "trials": {"counts": {m: {"low": 10, "high": 20} for m in
                                                      ("vol", "etf", "fall", "dip", "crash", "combination")}}}),
                    encoding="utf-8")
    led = tmp_path / "ledger.jsonl"
    TR.seed_from_prereg103(base, led)
    lc = tmp_path / "live_cal"
    lc.mkdir()
    (lc / "2026-10-19.json").write_text(json.dumps({"since": {"from": "2026-10-09", "to": "2026-10-19", "sessions": 7,
                                                              "arms": {"vol": {"n": 2, "open_n": 1, "return_per_day": 0.4,
                                                                               "verdict": "too few trades"}}}}),
                                        encoding="utf-8")
    audits = tmp_path / "audit"
    audits.mkdir()
    if prev_rows is not None:
        (audits / "2026-10-19.json").write_text(json.dumps({"day": "2026-10-19", "rows": prev_rows}), encoding="utf-8")
    return {"rel_dir": rel, "draws_root": draws, "seeds_path": seeds, "baseline": base, "ledger": led, "live_cal": lc,
            "audit_dir": audits, "now": NOW}


def _rows(rec):
    return {(r["model"], r["oos"]): r for r in rec["rows"]}


def test_the_registered_labels():
    assert AU.label(0.1, 0.5) == "ROBUST" and AU.label(0.1, 0.49) == "WEAKER OUT OF SAMPLE"
    assert AU.label(0.0, 2.0) == "FAILS OUT OF SAMPLE" and AU.label(-0.3, 0.9) == "FAILS OUT OF SAMPLE"
    assert AU.label(None, 0.9) is None and AU.label(0.1, None) is None


def test_each_set_comes_from_the_latest_build_holding_it(tmp_path):
    df = _trades([("vol", "A", 3, GOOD, 5.0)])
    for name in ("2026-10-12", "2026-10-13", "2026-10-14_A0", "2026-10-15_A-B-D", "2026-10-16_D"):
        _card(tmp_path, name, df)
    (tmp_path / "2026-10-17").mkdir()                     # no trades file: not a build
    (tmp_path / "notes").mkdir()
    got = {k: v.name for k, v in AU.card_dirs(tmp_path).items()}
    assert got == {"A": "2026-10-15_A-B-D", "B": "2026-10-15_A-B-D", "A0": "2026-10-14_A0", "D": "2026-10-16_D"}


def test_an_audit_labels_every_model_and_notes_a_degraded_label(tmp_path):
    w = _world(tmp_path, prev_rows=[{"model": "etf", "oos": "A0", "label": "ROBUST", "decay": 0.9},
                                    {"model": "vol", "oos": "A0", "label": "ROBUST", "decay": 1.0},
                                    {"model": "dip", "oos": "A0", "label": "WEAKER OUT OF SAMPLE", "decay": 0.3}])
    p = AU.write(date(2026, 10, 20), log=lambda *a: None, **w)
    assert p == tmp_path / "audit" / "2026-10-20.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    r = _rows(rec)
    assert r[("vol", "A0")]["label"] == "ROBUST" and r[("dip", "A0")]["label"] == "ROBUST"
    assert r[("etf", "A0")]["label"] == "FAILS OUT OF SAMPLE" and r[("fall", "B")]["label"] == "ROBUST"
    assert r[("combination", "A0")]["status"] == "computed" and r[("combination", "A0")]["label"] is not None
    assert r[("dip", "LONG")]["status"] == "carried" and r[("dip", "LONG")]["label"] == "ROBUST"
    assert r[("vol", "D")]["status"] == "no saved outputs" and rec["sets"]["missing"] == ["D"]
    assert r[("crash", "D")]["label"] is None                                    # no in-sample split: context only
    assert rec["previous"] == "audit 2026-10-19"
    moves = {(c["model"], c["oos"]): c["move"] for c in rec["changes"]}
    assert moves[("etf", "A0")] == "degraded" and moves[("dip", "A0")] == "improved" and moves[("vol", "A0")] == "same"
    assert moves[("fall", "A0")] == "new"
    assert len(rec["notes"]) == 1 and rec["notes"][0].startswith("the ETF arm on set A0: ROBUST -> FAILS OUT OF SAMPLE")
    h = AU.health(w["audit_dir"])
    assert h["problems"] == [] and h["notes"] == ["overfitting audit 2026-10-20: " + rec["notes"][0]]
    # the inputs, the ledger's counts and the live record ride along
    assert rec["trials"]["counts"]["vol"]["low"] == 10 and rec["dsr"]["vol"]["N_high"] == 20
    assert rec["dsr"]["etf"]["A0"]["dsr_low"] < rec["dsr"]["vol"]["A0"]["dsr_low"]
    assert rec["live"]["arms"]["vol"] == {"closed": 2, "open": 1, "return_per_day": 0.4, "verdict": "too few trades"}
    assert rec["sets"]["mixed_builds"] is True and rec["sets"]["outside_their_build"] == {"2026-10-13": 0, "2026-10-14_A0": 0}


def test_the_first_audit_reads_its_changes_against_prereg103(tmp_path):
    w = _world(tmp_path)                                  # no earlier audit: PREREG103 (etf A0 ROBUST there)
    rec = AU.build(date(2026, 10, 20), log=lambda *a: None, **w)
    assert rec["previous"] == "PREREG103 (global103.json)"
    assert [n for n in rec["notes"] if "ETF arm on set A0" in n]
    assert {(c["model"], c["oos"]): c["move"] for c in rec["changes"]}[("dip", "LONG")] == "same"


def test_set_d_is_labelled_only_from_ten_trades(tmp_path):
    w = _world(tmp_path / "few", d_trades=6)
    r = _rows(AU.build(date(2026, 10, 20), log=lambda *a: None, **w))[("vol", "D")]
    assert r["label"] is None and r["status"].startswith("accruing: 6 trades")
    w = _world(tmp_path / "enough", d_trades=12)
    rec = AU.build(date(2026, 10, 20), log=lambda *a: None, **w)
    r = _rows(rec)[("vol", "D")]
    assert r["status"] == "computed" and r["label"] == "ROBUST" and r["n"]["D"] == 12
    assert rec["sets"]["windows"]["D"] == ["2026-09-25", "2026-10-20"]
    assert rec["sets"]["trades"]["D"] == 12


def test_asking_for_d_alone_loads_the_in_sample_sets_it_is_read_against(tmp_path):
    w = _world(tmp_path, d_trades=12)
    rec = AU.build(date(2026, 10, 20), sets=("D",), log=lambda *a: None, **w)
    assert rec["sets"]["loaded"] == ["A", "B", "D"]
    assert {(r["model"], r["oos"]) for r in rec["rows"] if r.get("status") != "carried"} == \
        {(m, "D") for m in AU.DESIGN}


def test_no_audit_no_note(tmp_path):
    assert AU.health(tmp_path / "nothing") == {"problems": [], "notes": []}


def test_the_audit_and_the_ledger_never_import_the_research_harness():
    """Saved outputs only (the user's rule for this layer): no simulator, adapter or account-run module."""
    banned = ("src.backtest.mimic", "src.backtest.strategies", "src.backtest.a0", "src.backtest.study",
              "src.backtest.halts")
    for mod in ("audit", "trials", "add_rule", "periods"):
        src = (Path(AU.__file__).parent / f"{mod}.py").read_text(encoding="utf-8")
        names, used = set(), set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
                names |= {f"{node.module}.{a.name}" for a in node.names}
            elif isinstance(node, ast.Import):
                names |= {a.name for a in node.names}
            elif isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)
        assert not [n for n in names if n.startswith(banned)], (mod, names)
        assert not used & {"live_books", "_init", "simulate", "live_strategies"}, (mod, used & {"live_books", "_init"})


def test_the_port_reproduces_prereg103(tmp_path):
    """The audit's statistics on PREREG103's own inputs reproduce its table 3 (decay, intervals, labels) and table 1
    exactly."""
    b = AU.baseline_json()
    if not b.exists():
        pytest.skip("PREREG103's global103.json is not on this machine")
    IN = json.loads(b.read_text(encoding="utf-8"))["inputs"]
    need = [Path(IN["cards_ab"]) / "trades_base.pkl", Path(IN["cards_a0"]) / "trades_base.pkl", Path(IN["account101"]),
            Path(IN["draws"]) / "A0_LIVE_S2_s0.pkl"]
    if not all(p.exists() for p in need):
        pytest.skip("PREREG103's inputs are not on this machine")
    res = AU.parity(log=lambda *a: None)
    assert res["identical"], res
    assert res["rows_compared"] == 6 and res["max_abs_diff_table3"] == 0.0
