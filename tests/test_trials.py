"""THE TRIAL LEDGER (src/backtest/trials.py; PREREG106 item 1): every candidate a study evaluates is recorded once, in an
append-only file; `count` = PREREG103's seed + the recorded trials (the combination's high count takes every arm's);
`set_uses` = how many recorded decisions each set carried. Every test writes its own ledger in a temporary directory."""
from __future__ import annotations

import json
import threading

import pytest

from src.backtest import periods
from src.backtest import trials as TR


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    p = tmp_path / "trials" / "ledger.jsonl"
    monkeypatch.setattr(TR, "LEDGER", p)              # the default ledger too: no test reaches the real one
    return p


def _global103(tmp_path, counts=None):
    """A global103.json holding PREREG103's ``trials`` section (the real file's shape)."""
    counts = counts or {"vol": {"low": 753, "high": 80141, "how": "optvol ..."},
                        "etf": {"low": 44, "high": 57, "how": "..."},
                        "combination": {"low": 1797, "high": 84049, "how": "..."}}
    p = tmp_path / "global103.json"
    p.write_text(json.dumps({"built_at": "2026-10-14T11:37:17-04:00",
                             "trials": {"counts": counts, "source": {"caveats": ["the counts are nominal"]}}}),
                 encoding="utf-8")
    return p


def test_the_ledger_lives_under_the_repository_whatever_the_working_directory():
    assert TR.LEDGER.is_absolute() and TR.LEDGER.parts[-4:] == ("cache", "ml", "trials", "ledger.jsonl")
    assert (TR.REPO / "src" / "backtest" / "trials.py").exists()


def test_record_is_idempotent_per_study_variant_kind(ledger):
    a = TR.record("PREREG104", "fall", "F1: half slice", ["A", "B"], {"gain_A": 0.4}, "more draws", "sizing")
    b = TR.record("PREREG104", "fall", "F1: half slice", ["B", "A"], {"gain_A": 0.4}, "more draws", "sizing")
    assert a["status"] == "recorded" and b["status"] == "unchanged"
    assert len(TR.rows()) == 1 and TR.rows()[0]["sets"] == ["A", "B"]          # one row, the sets in order
    # the same variant under another kind is another trial
    assert TR.record("PREREG104", "fall", "F1: half slice", ["A", "B"], {}, None, "rule")["status"] == "recorded"
    assert len(TR.current()) == 2


def test_a_new_payload_is_a_revision_the_key_counts_once(ledger):
    TR.record("PREREG104", "fall", "F3", ["A", "B"], {"gain_B": 0.3}, "more draws", "rule")
    first = ledger.read_bytes()
    r = TR.record("PREREG104", "fall", "F3", ["A", "B", "A0"], {"gain_B": 0.7}, "add", "rule")
    assert r["status"] == "revised" and r["row"]["revision"] == 1
    assert ledger.read_bytes().startswith(first)                     # append-only: the first line is untouched
    assert len(TR.rows()) == 2 and len(TR.current()) == 1
    cur = TR.current()[("PREREG104", "F3", "rule")]
    assert cur["decision"] == "add" and cur["sets"] == ["A0", "A", "B"]
    assert TR.count("fall")["recorded"] == 1


def test_count_is_the_seed_plus_the_recorded_trials(ledger, tmp_path):
    res = TR.seed_from_prereg103(_global103(tmp_path))
    assert sorted(res["recorded"]) == ["combination", "etf", "vol"] and res["unchanged"] == []
    c = TR.count("vol")
    assert (c["low"], c["high"], c["recorded"]) == (753, 80141, 0) and c["seed"]["source"] == "seed:PREREG103"
    TR.record("PREREG107", "vol", "give back 1.5", ["A", "B"], {}, "reject", "rule")
    TR.record("PREREG107", "vol", "give back 1.75", ["A", "B"], {}, "reject", "rule")
    TR.record("PREREG108", "combination", "vol 1/16", ["A", "B"], {}, "reject", "sizing")
    v, e, k = TR.count("vol"), TR.count("etf"), TR.count("combination")
    assert (v["low"], v["high"], v["recorded"]) == (755, 80143, 2) and v["by_kind"] == {"rule": 2}
    assert (e["low"], e["high"]) == (44, 57)
    # the combination: its low takes its own trials, its high every model's (PREREG103's convention)
    assert (k["low"], k["high"], k["recorded"], k["recorded_all_models"]) == (1798, 84052, 1, 3)
    assert TR.count("squeeze_long") == {"model": "squeeze_long", "low": 0, "high": 0, "seed": None, "recorded": 0,
                                        "by_kind": {}, "studies": []}


def test_the_seed_is_marked_idempotent_and_never_a_recorded_trial(ledger, tmp_path):
    src = _global103(tmp_path)
    TR.seed_from_prereg103(src)
    again = TR.seed_from_prereg103(src)
    assert again["recorded"] == [] and sorted(again["unchanged"]) == ["combination", "etf", "vol"]
    seeds = TR.seed_rows()
    assert len(seeds) == 3 and all(r["kind"] == "seed" and r["source"] == "seed:PREREG103" and r["sets"] == []
                                   and r["at"] == "2026-10-14T11:37:17-04:00" for r in seeds)
    with pytest.raises(ValueError, match="reserved"):
        TR.record("PREREG107", "vol", "x", ["A"], {}, None, "seed")
    # a changed count in PREREG103's file is a revision of the seed, never a second seed
    TR.seed_from_prereg103(_global103(tmp_path, {"vol": {"low": 760, "high": 80141, "how": "recount"}}))
    assert TR.count("vol")["low"] == 760 and len(TR.seed_rows()) == 3


def test_the_real_prereg103_counts_seed_the_ledger(ledger):
    src = TR.prereg103_json()
    if not src.exists():
        pytest.skip("PREREG103's global103.json is not on this machine")
    TR.seed_from_prereg103(src)
    want = json.loads(src.read_text(encoding="utf-8"))["trials"]["counts"]
    for m, c in want.items():
        got = TR.count(m)
        assert (got["low"], got["high"]) == (c["low"], c["high"])


def test_set_uses_count_the_current_rows_per_set_and_skip_the_seed(ledger, tmp_path):
    TR.seed_from_prereg103(_global103(tmp_path))
    TR.record("PREREG104", "fall", "F1", ["A", "B", "A0"], {}, "reject", "sizing", at="2026-10-14T13:30:00-04:00")
    TR.record("PREREG104", "fall", "F2", ["A", "B"], {}, "more draws", "mechanism", at="2026-10-14T13:40:00-04:00")
    TR.record("PREREG104", "fall", "F2", ["A", "B"], {}, "add", "mechanism", at="2026-10-14T14:40:00-04:00")
    TR.record("PREREG109", "vol", "x", ["D"], {}, None, "rule", at="2026-11-02T10:00:00-05:00")   # after the DST change
    u = TR.set_uses()
    assert list(u) == list(TR.SETS)
    assert u["A"]["trials"] == 2 and u["A"]["decisions"] == {"reject": 1, "add": 1}     # F2's revision counts once
    assert u["A0"]["trials"] == 1 and u["A0"]["studies"] == ["PREREG104"] and u["A0"]["models"] == {"fall": 1}
    assert u["D"]["decisions"] == {"undecided": 1} and u["C"]["trials"] == 0 and u["LONG"]["trials"] == 0
    assert u["A"]["first"] == "2026-10-14T13:30:00-04:00" and u["A"]["last"] == "2026-10-14T14:40:00-04:00"


def test_a_trial_names_known_sets_and_its_fields(ledger):
    with pytest.raises(ValueError, match="unknown set"):
        TR.record("PREREG107", "vol", "x", ["A", "2025"], {}, None, "rule")
    with pytest.raises(ValueError, match="variant"):
        TR.record("PREREG107", "vol", "  ", ["A"], {}, None, "rule")
    assert TR.record("PREREG107", "vol", "y", "A, B", {}, None, "rule")["row"]["sets"] == ["A", "B"]
    # every borrow-data test set is a set a trial may name, PREREG103's LONG histories too
    for k in TR.SETS:
        if k != "LONG":
            assert periods.get(k).key == k


def test_metrics_are_stored_json_ready(ledger):
    import numpy as np
    r = TR.record("PREREG107", "vol", "z", ["A"], {"gain": np.float64(0.5), "n": np.int64(20), "se": float("nan"),
                                                   "gains": (1.0, 2.0)}, "add", "rule")
    assert r["row"]["metrics"] == {"gain": 0.5, "n": 20, "se": None, "gains": [1.0, 2.0]}
    assert json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])["metrics"]["se"] is None


def test_a_malformed_line_is_skipped_never_fatal(ledger):
    TR.record("PREREG107", "vol", "a", ["A"], {}, None, "rule")
    with open(ledger, "a", encoding="utf-8") as fh:
        fh.write("{not json\n\n")
    TR.record("PREREG107", "vol", "b", ["A"], {}, None, "rule")
    assert [r["variant"] for r in TR.rows()] == ["a", "b"]


def test_parallel_writers_lose_nothing(ledger):
    def work(i):
        for j in range(10):
            TR.record("PREREG107", "vol", f"w{i}-{j}", ["A", "B"], {"j": j}, None, "rule")
    ts = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    rows = TR.rows()
    assert len(rows) == 60 and len({r["variant"] for r in rows}) == 60
    assert TR.count("vol")["recorded"] == 60
