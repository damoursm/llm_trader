"""The test sets by borrow data (src/backtest/periods.py; user directive 2026-10-10): A = with IBKR's short-stock file,
B = without (lendability modeled), C = our own archive (live). What must hold: the sets tile the time line from the
backtests' first session with no gap and no overlap; a day falls in exactly one; their boundaries ARE the simulator's
lendability switch dates (`mimic`), so a set can never hold a modeled day it calls known; a calendar year that straddles
a boundary is reported in parts; growth is read off a NAV over a set or a year part. PREREG106's set D (the forward
holdout: C's days simulated, through the latest completed session) is opt-in and changes nothing above; it starts where
the simulator reads our own archive."""
from __future__ import annotations

import pandas as pd
import pytest

from src.backtest import mimic as M
from src.backtest import periods as P


def test_the_sets_tile_the_time_line_from_the_first_session():
    assert P.SETS[0].lo == P.DATA_START
    for a, b in zip(P.SETS[:-1], P.SETS[1:]):
        assert pd.Timestamp(b.lo) == pd.Timestamp(a.hi) + pd.Timedelta(days=1)
    assert P.SETS[-1].hi is None


def test_the_boundaries_are_the_simulators_lendability_switch():
    assert M.LEND_ARCHIVE_END is P.LEND_ARCHIVE_END and M.OWN_ARCHIVE_FROM is P.OWN_ARCHIVE_FROM
    assert P.get("A").hi == M.LEND_ARCHIVE_END and P.get("C").lo == M.OWN_ARCHIVE_FROM
    with pytest.raises(KeyError):                     # (PREREG106 added set D; an unknown key still has no set)
        P.get("E")


@pytest.mark.parametrize("day,key", [("2021-01-29", None), ("2021-02-01", "A"), ("2024-06-24", "A"),
                                     ("2024-06-25", "B"), ("2026-09-24", "B"), ("2026-09-25", "C"),
                                     ("2027-03-01", "C")])
def test_a_day_falls_in_one_set(day, key):
    assert P.set_of(day) == key


def test_windows_stop_at_the_data():
    assert P.windows("2026-09-24") == [("A", "2021-02-01", "2024-06-24"), ("B", "2024-06-25", "2026-09-24")]
    assert P.windows("2026-10-09")[-1] == ("C", "2026-09-25", "2026-10-09")
    assert [k for k, _, _ in P.windows("2023-05-01")] == ["A"]


def test_a_year_that_straddles_a_boundary_comes_in_parts():
    parts = P.year_parts("2021-02-01", "2026-10-09")
    assert [p[0] for p in parts] == ["2021", "2022", "2023", "2024 A", "2024 B", "2025", "2026 B", "2026 C"]
    assert parts[3][2:] == ("2024-01-01", "2024-06-24") and parts[4][2:] == ("2024-06-25", "2024-12-31")
    assert {p[1] for p in parts[:4]} == {"A"} and parts[-1][1] == "C"


def test_the_sets_a_b_c_are_unchanged_by_set_a0():
    """PREREG101 adds A0 as an EXTRA set: `SETS` and every function called without ``extra`` read A / B / C exactly
    as before (studies running on A and B are untouched)."""
    assert [(s.key, s.lo, s.hi) for s in P.SETS] == [("A", "2021-02-01", "2024-06-24"), ("B", "2024-06-25", "2026-09-24"),
                                                     ("C", "2026-09-25", None)]
    assert P.set_of("2019-06-03") is None and P.set_of("2021-01-29") is None
    assert [k for k, _, _ in P.windows("2026-10-09")] == ["A", "B", "C"]
    assert P.year_parts("2021-02-01", "2026-10-09")[0] == ("2021", "A", "2021-02-01", "2021-12-31")


def test_set_a0_is_asked_for_by_name_and_lies_inside_ibkrs_archive():
    s = P.get("A0")
    assert (s.key, s.lo, s.hi) == ("A0", "2018-01-02", "2021-01-29")
    assert pd.Timestamp(P.LEND_ARCHIVE_START) < pd.Timestamp(s.lo)            # IBKR's file in force from 2017-10-09
    assert pd.Timestamp(s.hi) < pd.Timestamp(P.DATA_START)                   # it ends before set A's first decision
    assert P.set_of("2019-06-03", extra=True) == "A0" and P.set_of("2021-02-01", extra=True) == "A"
    assert P.set_of("2017-12-29", extra=True) is None and P.set_of("2021-01-30", extra=True) is None
    assert P.windows("2026-09-24", extra=True)[:2] == [("A0", "2018-01-02", "2021-01-29"),
                                                       ("A", "2021-02-01", "2024-06-24")]
    with pytest.raises(KeyError):
        P.get("A1")


def test_a_year_that_straddles_set_a0s_end_comes_in_parts():
    parts = P.year_parts("2018-01-02", "2021-12-31", extra=True)
    assert [p[0] for p in parts] == ["2018", "2019", "2020", "2021 A0", "2021 A"]
    assert parts[3][1:] == ("A0", "2021-01-01", "2021-01-31") and parts[4][1:] == ("A", "2021-02-01", "2021-12-31")


def test_growth_is_read_off_the_nav():
    idx = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    nav = pd.Series(1.0, index=idx)
    nav[nav.index >= "2024-06-25"] = 1.1
    assert P.growth(nav, "2024-06-25", "2024-12-31", annual=False) == pytest.approx(0.1)
    assert P.growth(nav, "2024-01-01", "2024-06-24", annual=False) == pytest.approx(0.0)
    full = P.growth(nav, "2024-01-01", "2024-12-31")
    assert full == pytest.approx(1.1 ** (365.25 / 365) - 1, rel=1e-6)
    assert P.growth(nav, "2025-01-01", "2025-12-31") is None


# ── PREREG106: set D, the forward holdout (opt-in) ───────────────────────────

NOW = "2026-10-14 13:30"                                   # ET wall time: the 14th's session is not complete yet


@pytest.mark.parametrize("now,last", [
    ("2026-10-14 13:30", "2026-10-13"),                   # before today's close: yesterday
    ("2026-10-14 16:00", "2026-10-14"),                   # at the close
    ("2026-10-17 11:00", "2026-10-16"),                   # a Saturday: Friday
    ("2026-10-19 09:00", "2026-10-16"),                   # Monday before the close: Friday
    ("2026-11-26 17:00", "2026-11-25"),                   # Thanksgiving: the day before
    ("2026-10-14 21:30+00:00", "2026-10-14"),             # an aware instant: 17:30 ET
])
def test_set_d_ends_at_the_latest_completed_session(now, last):
    assert P.latest_completed_session(now) == last
    S = P.get("D", now=now)
    assert (S.key, S.lo, S.hi) == ("D", "2026-09-25", last)


def test_set_d_is_computed_at_call_time():
    hi = pd.Timestamp(P.get("D").hi)
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    assert today - pd.Timedelta(days=5) <= hi <= today          # the latest completed session, never a future day


def test_set_d_starts_where_the_simulator_reads_our_own_archive():
    """D's first decision is the day `mimic.lend_all` (and `borrow_history.snapshot_at`) switch to our own archive of
    IBKR's file: every set-D row's lendability is the real path, as set A's."""
    from src.data.deep import borrow_history as BH
    assert P.get("D", now=NOW).lo == M.OWN_ARCHIVE_FROM == P.get("C").lo
    assert pd.Timestamp(P.get("D", now=NOW).lo).tz_localize(BH.ET) == BH.OWN_START


def test_set_d_is_opt_in_and_a_b_c_a0_read_as_before():
    assert [s.key for s in P.SETS] == ["A", "B", "C"] and [s.key for s in P.EXTRA_SETS] == ["A0"]
    for extra in (False, True):
        assert P.set_of("2026-10-01", extra) == "C"                              # without forward: C, as before
        assert [k for k, _, _ in P.windows("2026-10-13", extra)][-1] == "C"
    assert P.year_parts("2026-01-01", "2026-12-31")[-1][:2] == ("2026 C", "C")


def test_forward_puts_set_d_in_cs_place():
    assert P.set_of("2026-10-01", forward=True, now=NOW) == "D"
    assert P.set_of("2026-10-13", forward=True, now=NOW) == "D"
    assert P.set_of("2026-10-14", forward=True, now=NOW) is None                  # not completed yet
    assert P.set_of("2024-06-24", forward=True, now=NOW) == "A" and P.set_of("2019-06-03", True, forward=True, now=NOW) == "A0"
    assert P.windows("2026-12-31", forward=True, now=NOW) == [("A", "2021-02-01", "2024-06-24"),
                                                              ("B", "2024-06-25", "2026-09-24"),
                                                              ("D", "2026-09-25", "2026-10-13")]
    assert [k for k, _, _ in P.windows("2026-10-13", extra=True, forward=True, now=NOW)] == ["A0", "A", "B", "D"]
    assert P.windows("2026-10-01", forward=True, now=NOW)[-1] == ("D", "2026-09-25", "2026-10-01")
    assert [p[:2] for p in P.year_parts("2026-01-01", "2026-12-31", forward=True, now=NOW)] == [("2026 B", "B"),
                                                                                                ("2026 D", "D")]
    # before 2026-09-25's close set D holds nothing
    assert P.windows("2026-09-30", forward=True, now="2026-09-25 10:00")[-1][0] == "B"


def test_the_simulator_reads_our_own_archive_on_set_d(monkeypatch):
    """`mimic.lend_all` on a stub engine: IBKR's archive to 2024-06-24 (the engine's mask), the model's draw after it
    (set B), our own archive from 2026-09-25 (set D) wherever it knows the name — the draw stays only where it does
    not."""
    from types import SimpleNamespace

    import numpy as np

    from src.data.deep import borrow_history as BH
    days = ["2024-06-24", "2025-03-03", "2026-09-24", "2026-09-25", "2026-10-13", "2026-10-13"]
    tkn = ["AAA", "BBB", "CCC", "DDD", "EEE", "UNK"]
    C = pd.DataFrame({"dn": [M.dnum(d) for d in days], "tkn": tkn, "bod": [3] * 6, "actual": [10.0] * 6})
    eng = SimpleNamespace(st=SimpleNamespace(C=C), lend_mask=lambda at="next": np.ones(len(C), bool))
    asked = []

    def own(sym, at, price):
        asked.append(sym)
        return {"DDD": (True, {}), "EEE": (False, {})}.get(sym, (None, {}))
    monkeypatch.setattr(BH, "lendable_at", own)
    out = M.lend_all(eng, M.Mimic(house=False), p_lend=np.full(len(C), 0.5), draw=lambda sym, day: 0.9)
    # A: the archive mask; B: the draw (0.9 >= 0.5: not lendable); D: our archive, the draw where it is silent
    assert out.tolist() == [True, False, False, True, False, False]
    assert asked == ["DDD", "EEE", "UNK"]
    assert [P.set_of(d, forward=True, now=NOW) for d in days] == ["A", "B", "B", "D", "D", "D"]
