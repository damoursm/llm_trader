"""The pre-registration checker (`src/backtest/prereg.py`): a result file older than the study's registration is
a violation, newer ones are fine, and an amendment only lists what existed when it was written.

The clock is controlled: the PREREG text carries the instants and the files' modification times are set with
``os.utime`` (UTC epoch seconds), so the test also pins the UTC -> ET conversion — a checker that compared naive
times would be off by four or five hours, enough to turn a same-morning violation into a pass.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.backtest import prereg

ET = ZoneInfo("America/New_York")


def _et(y, mo, d, hh, mm, ss=0) -> float:
    """Epoch seconds of an ET wall-clock instant."""
    return datetime(y, mo, d, hh, mm, ss, tzinfo=ET).timestamp()


def _touch(path: Path, when: float, text: str = "{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.utime(path, (when, when))


def _study(tmp_path: Path, name: str, md: str) -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / f"{name.upper()}.md").write_text(md, encoding="utf-8")
    return d


REGISTERED = "# PREREG901 — a plan\n\nRegistered 2026-10-12 11:38 ET (Monday), before any number below was computed.\n"


def test_result_file_older_than_the_registration_is_a_violation(tmp_path):
    d = _study(tmp_path, "prereg901", REGISTERED)
    _touch(d / "early.json", _et(2026, 10, 12, 11, 37, 59))          # one second before the registration minute
    _touch(d / "later.json", _et(2026, 10, 12, 11, 38, 5))           # inside the registration's minute: after it
    _touch(d / "script.py", _et(2026, 10, 12, 11, 0))                # code, not a result: never listed
    res = prereg.check(d)
    assert res["ok"] is False
    assert [v["file"] for v in res["violations"]] == ["early.json"]
    assert res["violations"][0]["registered_at"] == "2026-10-12T11:38:00-04:00"
    assert res["violations"][0]["mtime"] == "2026-10-12T11:37:59-04:00"
    assert [f["name"] for f in res["files"]] == ["early.json", "later.json"]
    assert res["registrations"][0]["kind"] == "registration"
    assert res["registrations"][0]["files_before"] == ["early.json"]
    assert "VIOLATION" in prereg.verdict_line(res) and "early.json" in prereg.verdict_line(res)


def test_files_newer_than_the_registration_are_ok(tmp_path):
    d = _study(tmp_path, "prereg902", REGISTERED.replace("PREREG901", "PREREG902"))
    _touch(d / "run.json", _et(2026, 10, 12, 12, 15))
    _touch(d / "run.pkl", _et(2026, 10, 12, 12, 16))
    _touch(d / "pieces" / "deep.npy", _et(2026, 10, 13, 9, 0))        # a subdirectory's results count too
    _touch(d / "__pycache__" / "x.txt", _et(2020, 1, 1, 0, 0))        # __pycache__ never does
    res = prereg.check(d)
    assert res["ok"] is True
    assert res["violations"] == [] and res["problems"] == []
    assert [f["name"] for f in res["files"]] == ["run.json", "run.pkl", "pieces/deep.npy"]   # oldest first
    assert res["first_registration"] == "2026-10-12T11:38:00-04:00"
    assert prereg.verdict_line(res).startswith("prereg902: OK")


def test_amendment_lists_the_files_between_registration_and_amendment(tmp_path):
    md = ("# PREREG903 — a plan\n\nRegistered 2026-10-12 11:38 ET (Monday), before any number below was computed.\n\n"
          "## Part 1\n\n## Amendment 1 (registered 2026-10-12 11:41 ET, before F4 is computed)\n\nmore plan\n\n"
          "## Amendment 2 (registered 2026-10-12 12:53 ET, before L4 is calibrated or run)\n")
    d = _study(tmp_path, "prereg903", md)
    _touch(d / "fee.json", _et(2026, 10, 12, 11, 40))               # between the registration and amendment 1
    _touch(d / "f4.json", _et(2026, 10, 12, 11, 43))                # between amendment 1 and amendment 2
    _touch(d / "l4.json", _et(2026, 10, 12, 13, 0))                 # after both
    res = prereg.check(d)
    assert res["ok"] is True and res["violations"] == []            # amendments are informational
    kinds = [(r["kind"], r["label"], r["at"]) for r in res["registrations"]]
    assert kinds == [("registration", "Registered", "2026-10-12T11:38:00-04:00"),
                     ("amendment", "Amendment 1", "2026-10-12T11:41:00-04:00"),
                     ("amendment", "Amendment 2", "2026-10-12T12:53:00-04:00")]
    a1, a2 = res["registrations"][1], res["registrations"][2]
    assert a1["files_since_registration"] == ["fee.json"]
    assert a1["files_before"] == ["fee.json"]
    assert a2["files_since_registration"] == ["fee.json", "f4.json"]
    assert "files_since_registration" not in res["registrations"][0]
    assert "2 amendments" in prereg.verdict_line(res)


def test_the_first_registration_rules_a_directory_with_several_plans(tmp_path):
    """`studies/prereg44/` holds PREREG44 and PREREG52-56: PREREG44's results predate PREREG52's registration
    legitimately. Only the directory's earliest registration decides a violation."""
    d = _study(tmp_path, "prereg904", "# PREREG904\n\nRegistered 2026-10-01 09:00 ET, before any run.\n")
    (d / "PREREG905.md").write_text("# PREREG905\n\nRegistered 2026-10-05 09:00 ET, before any run.\n",
                                    encoding="utf-8")
    _touch(d / "first_study.json", _et(2026, 10, 2, 10, 0))
    _touch(d / "too_early.json", _et(2026, 9, 30, 10, 0))
    res = prereg.check(d)
    assert [v["file"] for v in res["violations"]] == ["too_early.json"]
    later = [r for r in res["registrations"] if r["prereg"] == "PREREG905.md"][0]
    assert later["files_before"] == ["too_early.json", "first_study.json"]        # informational, per plan


def test_older_stamped_form_and_dst_offset(tmp_path):
    """The earlier studies wrote "(stamped 2026-10-10 22:48 ET, before any run)"; and a winter instant is -05:00."""
    d = _study(tmp_path, "prereg906",
               "# PREREG906 — the sizing (stamped 2026-01-10 22:48 ET, before any run)\n\n"
               "## Amendment 1 (2026-01-11 00:47 ET, before any variant is run)\n")
    _touch(d / "grid.pkl", _et(2026, 1, 10, 22, 47))
    res = prereg.check(d)
    assert res["first_registration"] == "2026-01-10T22:48:00-05:00"
    assert [v["file"] for v in res["violations"]] == ["grid.pkl"]
    assert res["registrations"][1] == {"kind": "amendment", "label": "Amendment 1", "prereg": "PREREG906.md",
                                       "line": 3, "at": "2026-01-11T00:47:00-05:00", "approximate": False,
                                       "files_before": ["grid.pkl"], "files_since_registration": []}


def test_approximate_instant_is_accepted_and_flagged(tmp_path):
    """PREREG44 / 87 wrote "Registered 2026-10-11 ~14:50 ET": still a registration, carried as approximate."""
    d = _study(tmp_path, "prereg911",
               "# PREREG911\n\nRegistered 2026-10-11 ~14:50 ET (Sunday), before any number below was computed.\n")
    _touch(d / "late.json", _et(2026, 10, 11, 15, 0))
    res = prereg.check(d)
    assert res["ok"] is True
    assert res["first_registration"] == "2026-10-11T14:50:00-04:00"
    assert res["first_registration_approximate"] is True and res["registrations"][0]["approximate"] is True
    assert "registered ~2026-10-11T14:50:00-04:00" in prereg.verdict_line(res)


def test_unverifiable_studies_are_not_ok(tmp_path):
    no_md = tmp_path / "prereg907"
    no_md.mkdir()
    _touch(no_md / "x.json", _et(2026, 10, 12, 12, 0))
    res = prereg.check(no_md)
    assert res["ok"] is False and res["violations"] == []
    assert res["problems"] == ["no PREREG*.md in the directory"]
    assert prereg.verdict_line(res).startswith("prereg907: UNVERIFIED")

    no_stamp = _study(tmp_path, "prereg908", "# PREREG908 — a plan with no instant\n")
    res = prereg.check(no_stamp)
    assert res["ok"] is False and res["first_registration"] is None
    assert res["problems"] == ["PREREG908.md: no registration instant ('Registered YYYY-MM-DD HH:MM ET')"]

    res = prereg.check(tmp_path / "missing")
    assert res["ok"] is False and res["problems"] == [f"not a directory: {tmp_path / 'missing'}"]


def test_cli_prints_one_verdict_per_study_and_exits_nonzero_on_a_violation(tmp_path, capsys):
    good = _study(tmp_path, "prereg909", REGISTERED.replace("PREREG901", "PREREG909"))
    _touch(good / "ok.json", _et(2026, 10, 12, 12, 0))
    bad = _study(tmp_path, "prereg910", REGISTERED.replace("PREREG901", "PREREG910"))
    _touch(bad / "early.csv", _et(2026, 10, 12, 8, 0))
    assert prereg.main([str(good)]) == 0
    assert prereg.main([str(good), str(bad)]) == 1
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 3
    assert out[0].startswith("prereg909: OK") and out[1].startswith("prereg909: OK")
    assert out[2].startswith("prereg910: VIOLATION") and "early.csv" in out[2]


@pytest.mark.parametrize("line,expected", [
    ("Registered 2026-10-12 15:45 ET (Monday), before any stratified draw exists.", "2026-10-12T15:45:00-04:00"),
    ("# PREREG80 — the sizing (stamped 2026-10-10 22:48 ET, before any run)", "2026-10-10T22:48:00-04:00"),
    ("Saved 2026-10-09 13:46 ET (clock read 17:46 UTC), BEFORE any number below was computed.",
     "2026-10-09T13:46:00-04:00"),
    ("Saved 2026-10-08 ~14:20 ET (read the clock: `date`), BEFORE any combined number exists.",
     "2026-10-08T14:20:00-04:00"),
])
def test_registration_line_forms(tmp_path, line, expected):
    """The three forms the studies used: PREREG88 on, PREREG64-86, PREREG44-63 (some approximate)."""
    md = tmp_path / "PREREG1.md"
    md.write_text(line + "\n", encoding="utf-8")
    found = prereg.parse_registrations(md)
    assert len(found) == 1 and found[0]["kind"] == "registration"
    assert found[0]["at"].isoformat(timespec="seconds") == expected
    assert found[0]["approximate"] is ("~" in line)


@pytest.mark.parametrize("line,label,expected", [
    ("## Amendment 3 (registered 2026-10-12 19:10 ET, after the first 40 draws)", "Amendment 3",
     "2026-10-12T19:10:00-04:00"),
    ("## Amendment 1 — the user's correction of the fill (2026-10-09 14:49 ET, before any number with it)",
     "Amendment 1", "2026-10-09T14:49:00-04:00"),
    ("## Amendments (2026-10-09 16:47 ET, before any result of the changed simulator)", "Amendments",
     "2026-10-09T16:47:00-04:00"),
    ("## Amendment A1 (stamped 2026-10-10 10:00 ET — after the dataset build)", "Amendment A1",
     "2026-10-10T10:00:00-04:00"),
])
def test_amendment_line_forms_and_never_a_registration(tmp_path, line, label, expected):
    md = tmp_path / "PREREG2.md"
    md.write_text(line + "\n", encoding="utf-8")
    found = prereg.parse_registrations(md)
    assert [(r["kind"], r["label"], r["at"].isoformat(timespec="seconds")) for r in found] == [
        ("amendment", label, expected)]


def test_a_results_heading_is_not_an_instant(tmp_path):
    """'## Results (2026-10-09 14:23 ET)' and a body sentence with a deploy time are neither a registration nor
    an amendment."""
    md = tmp_path / "PREREG3.md"
    md.write_text("Saved 2026-10-09 11:33 ET (clock read), BEFORE any number of what follows.\n"
                  "The live account's rules (deployed 2026-10-09 11:00 ET, `src/performance/live_account.py`)\n"
                  "## Results and verdict (2026-10-09 12:13 ET)\n", encoding="utf-8")
    found = prereg.parse_registrations(md)
    assert [(r["kind"], r["line"]) for r in found] == [("registration", 1)]
