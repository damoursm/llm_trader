"""Splits in the deep 30-minute store (user directive 2026-09-27: "When seeing a
stock split in the data ingestion we should automatically reset the historical
data before doing any kind of inference on this ticker"; 2026-09-28: "verify from
your split data if it's possible instead of assuming depending on the price
change").

Polygon serves bars adjusted as of the fetch day and the store only appended, so
after a split its pre-split bars sat on the old scale (MGN 1-for-30: $0.18 ->
$4.89 between two bars). What must hold: the SPLIT DATA decides — a split the
`splits` family records, effective after the name was last adjusted, resets the
whole history; an extension that finds the stored scale changed resets only
when the split data confirms it, and otherwise logs and appends; an unchanged
scale appends as before. No network, no real store.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data import intraday_store as st


def _bars(start: str, sessions: int, px: float) -> pd.DataFrame:
    days = pd.bdate_range(start, periods=sessions)
    idx = [d + pd.Timedelta(hours=13, minutes=30) + pd.Timedelta(minutes=30 * k) for d in days for k in range(13)]
    c = np.full(len(idx), px, float)
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1000.0},
                        index=pd.DatetimeIndex(idx))


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "DEEP_DIR", tmp_path / "bars30m_deep")
    monkeypatch.setattr(st, "SPLITS_PATH", tmp_path / "splits.parquet")
    return tmp_path


def test_scale_change_sees_a_split_and_ignores_noise():
    old = _bars("2026-09-14", 1, 0.18)
    assert st.scale_change(old, old * 30) == pytest.approx(30.0)
    assert st.scale_change(old, old * 0.5) == pytest.approx(0.5)
    assert st.scale_change(old, old * 1.01) is None                        # within tolerance
    assert st.scale_change(old, _bars("2026-09-16", 1, 5.0)) is None       # no common bar


def _splits(rows):
    import duckdb
    vals = ", ".join(f"('{t}', DATE '{d}', {f}, {to})" for t, d, f, to in rows) or "('NONE', DATE '2000-01-03', 2.0, 1.0)"
    duckdb.connect().execute(
        f"COPY (SELECT * FROM (VALUES {vals}) t(ticker, execution_date, split_from, split_to)) "
        f"TO '{st.SPLITS_PATH.as_posix()}' (FORMAT PARQUET)")


def test_an_extension_that_finds_the_overlap_rescaled_resets_when_the_split_data_confirms(store, monkeypatch):
    _splits([("MGN", "2026-09-17", 30.0, 1.0)])
    st.save_deep_30m("MGN", _bars("2026-09-08", 5, 0.18))                   # stored through 09-14, old scale
    full = _bars("2026-09-08", 9, 5.40)                                     # the history as Polygon serves it now
    calls = []

    def fetch(tk, frm, to):
        calls.append(frm)
        return full[full.index >= pd.Timestamp(frm)]
    monkeypatch.setattr(st, "_fetch_range", fetch)
    out = st.extend_deep_30m(["MGN"], workers=1, min_age_days=0, today=date(2026, 9, 18))
    assert out["reset"] == 1
    assert calls[0] == "2026-09-14"                                         # the overlap session re-read
    assert calls[-1] == st.DEEP_FROM                                        # then the whole history
    got = st.load_deep_30m("MGN")
    assert len(got) == len(full) and got["Close"].min() == pytest.approx(5.40)
    assert st._adjusted_asof()["MGN"] == date.today().isoformat()


def test_a_rescaled_overlap_the_split_data_does_not_confirm_is_not_reset(store, monkeypatch):
    """No split on record: the price change alone never resets a history."""
    _splits([])
    st.save_deep_30m("XYZ", _bars("2026-09-08", 5, 10.0))
    full = _bars("2026-09-08", 9, 12.0)                                     # Polygon now serves another scale
    calls = []

    def fetch(tk, frm, to):
        calls.append(frm)
        return full[full.index >= pd.Timestamp(frm)]
    monkeypatch.setattr(st, "_fetch_range", fetch)
    out = st.extend_deep_30m(["XYZ"], workers=1, min_age_days=0, today=date(2026, 9, 18))
    assert out["reset"] == 0 and out.get("extended") == 1
    assert st.DEEP_FROM not in calls                                        # no full refetch
    assert "XYZ" not in st._adjusted_asof()


def test_an_unchanged_scale_just_appends(store, monkeypatch):
    st.save_deep_30m("AAA", _bars("2026-09-08", 5, 10.0))
    full = _bars("2026-09-08", 9, 10.0)
    monkeypatch.setattr(st, "_fetch_range", lambda tk, frm, to: full[full.index >= pd.Timestamp(frm)])
    out = st.extend_deep_30m(["AAA"], workers=1, min_age_days=0, today=date(2026, 9, 18))
    assert out.get("extended") == 1 and out["reset"] == 0
    assert len(st.load_deep_30m("AAA")) == len(full)


def test_a_known_split_resets_names_adjusted_before_it(store, monkeypatch):
    _splits([("MGN", "2026-09-17", 30.0, 1.0), ("OLD", "2026-07-01", 2.0, 1.0), ("LATER", "2026-10-06", 3.0, 1.0)])
    for tk in ("MGN", "OLD", "LATER", "NOSPLIT"):
        st.save_deep_30m(tk, _bars("2026-09-08", 5, 1.0))
    fetched = []

    def fetch(tk, frm, to):
        fetched.append(tk)
        return _bars("2026-09-08", 5, 30.0)
    monkeypatch.setattr(st, "_fetch_range", fetch)
    out = st.reset_split_tickers(["MGN", "OLD", "LATER", "NOSPLIT"], today=date(2026, 9, 28))
    # MGN split after the default adjusted-as-of date; OLD before it; LATER not yet effective
    assert out == {"MGN": "reset"} and fetched == ["MGN"]
    assert st.load_deep_30m("MGN")["Close"].iloc[0] == pytest.approx(30.0)
    assert st.reset_split_tickers(["MGN"], today=date(2026, 9, 28)) == {}   # stamped: not again
    assert st.reset_split_tickers(["LATER"], today=date(2026, 10, 6)) == {"LATER": "reset"}


def test_concurrent_stamps_never_lose_one(store):
    """`extend_deep_30m` stamps NEW names from its worker threads: an unlocked read-modify-write of
    `_adjusted_asof.json` raced (2026-10-05, 103 new listings: some stamps lost, every OLDER stamp
    wiped once a thread read the file mid-replace) — each stamp lost re-resets a name's history."""
    from concurrent.futures import ThreadPoolExecutor
    from datetime import date
    names = [f"T{i:03d}" for i in range(400)]
    with ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda t: st._stamp_adjusted(t, date(2026, 10, 5)), names))
    got = st._adjusted_asof()
    assert sorted(got) == names and set(got.values()) == {"2026-10-05"}
