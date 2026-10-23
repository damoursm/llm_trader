"""The price stores' split guards and the integrity check (2026-10-16): the daily cache refetches a name WHOLE after a
split (never a merge across two scales), the deep 04:00-20:00 store resets a part on a split or a correction, and the
integrity check's banner. Synthetic data only — no network."""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data import split_check as SC


@pytest.fixture(autouse=True)
def tables(tmp_path, monkeypatch):
    monkeypatch.setattr(SC, "CHECKS_PATH", tmp_path / "split_checks.json")
    SC._CACHE.clear()
    yield tmp_path
    SC._CACHE.clear()


def daily(days, closes):
    c = np.asarray(closes, float)
    return pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": np.full(len(c), 1000.0)},
                        index=pd.DatetimeIndex(pd.to_datetime(days)))


# ── the daily cache ──────────────────────────────────────────────────────────

def test_overlap_scale_finds_cached_bars_on_another_scale():
    from src.data.market_data import overlap_scale
    cached = daily(["2026-05-28", "2026-05-29", "2026-06-01"], [160.0, 162.0, 164.0])   # pre-split scale
    fresh = daily(["2026-05-29", "2026-06-01", "2026-06-02"], [81.0, 82.0, 83.0])       # post-split
    assert overlap_scale(cached, fresh) == pytest.approx(0.5)
    assert overlap_scale(fresh, daily(["2026-06-01", "2026-06-02"], [82.05, 83.0])) is None


def _patch_daily(monkeypatch, tmp_path, cached, window, whole, splits=None):
    import src.data.cache as C
    import src.data.market_data as md
    import src.data.intraday_store as IS
    store = {}
    monkeypatch.setattr(md.settings, "enable_fetch_data", True)
    monkeypatch.setattr(md, "DAILY_ADJ_PATH", tmp_path / "ohlcv_meta" / "adjusted_asof.json")
    monkeypatch.setattr(C, "load_ohlcv", lambda t, interval="1d": cached)
    monkeypatch.setattr(C, "save_ohlcv", lambda t, df, interval="1d": store.__setitem__(t, df))
    monkeypatch.setattr(md.polygon_client, "get_bars", lambda t, period="3mo": whole if period == "20y" else window)
    monkeypatch.setattr(IS, "split_rows", lambda: splits or {})
    return store


def test_a_cache_on_another_scale_than_the_fresh_window_is_refetched_whole(monkeypatch, tmp_path):
    """APH: the merge kept the rows before the fresh window on the pre-split scale -> a fake -50% day."""
    import src.data.market_data as md
    days = list(pd.bdate_range("2026-05-01", "2026-06-12").strftime("%Y-%m-%d"))
    cached = daily(days[:20], [160.0] * 20)                                    # stored before the split
    window = daily(days[10:], [80.0] * (len(days) - 10))                       # fetched after it
    whole = daily(days, [80.0] * len(days))
    store = _patch_daily(monkeypatch, tmp_path, cached, window, whole)
    out = md.get_history("APH", period="3mo", force_refresh=True)
    assert float(out["Close"].max()) == 80.0 and float(out["Close"].min()) == 80.0     # one scale, whole
    assert "APH" in store and json.loads(md.DAILY_ADJ_PATH.read_text())["APH"] == date.today().isoformat()


def test_a_split_since_the_stamp_refetches_whole_even_on_a_cache_hit(monkeypatch, tmp_path):
    import src.data.market_data as md
    today = date.today()
    days = list(pd.bdate_range(end=pd.Timestamp(today), periods=30).strftime("%Y-%m-%d"))
    cached = daily(days, [10.0] * 30)
    whole = daily(days, [5.0] * 30)
    store = _patch_daily(monkeypatch, tmp_path, cached, daily(days[-5:], [5.0] * 5), whole,
                         splits={"ABC": [(pd.Timestamp(days[-3]).date(), 1.0, 2.0)]})
    md.DAILY_ADJ_PATH.parent.mkdir(parents=True, exist_ok=True)
    md.DAILY_ADJ_PATH.write_text(json.dumps({"ABC": "2026-01-01"}))
    assert md.daily_split_since_stamp("ABC") == pd.Timestamp(days[-3]).date()
    out = md.get_history("ABC")                                                # the last bar is fresh: was a hit
    assert float(out["Close"].iloc[0]) == 5.0 and "ABC" in store
    assert md.daily_split_since_stamp("ABC") is None                           # stamped: done


def test_the_daily_fetch_carries_the_split_tables_corrections(monkeypatch, tmp_path):
    import src.data.market_data as md
    SC.save_checks({"XRX|2026-02-09": {"status": "false", "from": 2.0, "to": 3.0}})
    days = ["2026-02-05", "2026-02-06", "2026-02-09", "2026-02-10"]
    polygon = daily(days, [1.41, 1.56, 2.11, 2.11])                            # Polygon's false 2/3 adjustment
    store = _patch_daily(monkeypatch, tmp_path, None, polygon, polygon)
    out = md.get_history("XRX", period="3mo", force_refresh=True)
    assert out["Close"].round(3).tolist() == [2.115, 2.34, 2.11, 2.11]         # undone before the date


# ── the 04:00-20:00 store ─────────────────────────────────────────────────────

def test_full_parts_reset_on_a_split_or_a_correction_they_do_not_carry(monkeypatch, tmp_path):
    from src.data.deep import refresh as R
    import src.data.intraday_store as IS
    fam = tmp_path / "bars30m_full"
    (fam / "parts").mkdir(parents=True)
    for tk in ("DHY", "XRX", "SPY"):
        (fam / "parts" / f"{tk}.parquet").write_bytes(b"x")
    monkeypatch.setattr(R.deep, "family_dir", lambda f: fam)
    monkeypatch.setattr(IS, "split_rows", lambda: {"DHY": [(date(2026, 9, 30), 10.0, 1.0)],
                                                   "SPY": [(date(2020, 1, 1), 1.0, 2.0)]})
    SC.save_checks({"XRX|2026-02-09": {"status": "false", "from": 2.0, "to": 3.0}})
    got = R.full_parts_to_reset(["DHY", "XRX", "SPY", "NOPART"], today=date(2026, 10, 17))
    assert set(got) == {"XRX"}                                     # DHY's split predates the audit's default stamp
    got = R.full_parts_to_reset(["DHY"], today=date(2026, 10, 17))
    (fam / "_adjusted_asof.json").write_text(json.dumps({"DHY": {"asof": "2026-09-01"},
                                                         "XRX": {"asof": "2026-10-16",
                                                                 "corrections": ["XRX|2026-02-09"]}}))
    got = R.full_parts_to_reset(["DHY", "XRX"], today=date(2026, 10, 17))
    assert got == {"DHY": "split effective 2026-09-30"}            # XRX carries its correction now


# ── the banner ───────────────────────────────────────────────────────────────

def test_health_flags_a_held_names_contradicted_split_and_a_stale_check(monkeypatch, tmp_path):
    from src.data import integrity as I
    monkeypatch.setattr(I, "OUT_DIR", tmp_path / "integrity")
    SC.save_checks({"XRX|2026-10-15": {"status": "false", "from": 2.0, "to": 3.0},
                    "ABC|2026-10-14": {"status": "suspect", "from": 1.0, "to": 2.0},
                    "OLD|2025-01-02": {"status": "false", "from": 1.0, "to": 2.0}})
    held = [{"status": "OPEN", "ticker": "XRX"}, {"status": "OPEN", "ticker": "ABC"}]
    h = I.health(held, today=date(2026, 10, 16))
    assert any("XRX" in p and "left out" in p for p in h["problems"])
    assert any("ABC" in p and "still applied" in p for p in h["problems"])
    assert not any("OLD" in p for p in h["problems"])
    assert "price-data integrity: no run yet" in h["notes"]
    (tmp_path / "integrity").mkdir()
    (tmp_path / "integrity" / "2026-10-10.json").write_text(json.dumps({"cross_store_unrepaired": {"APH": {"daily": 2.0}}}))
    h = I.health([], today=date(2026, 10, 16))
    assert any("has not run since 2026-10-10" in p for p in h["problems"])
    assert any("APH" in p and "disagree" in p for p in h["problems"])


def test_health_ignores_a_contradicted_record_older_than_the_position(monkeypatch, tmp_path):
    """TPST's false record of 2026-01-30 never touched a position opened after it: no red banner for it."""
    from src.data import integrity as I
    monkeypatch.setattr(I, "OUT_DIR", tmp_path / "integrity")
    SC.save_checks({"TPST|2026-01-30": {"status": "false", "from": 1.0, "to": 2.0}})
    h = I.health([{"status": "OPEN", "ticker": "TPST", "entry_date": "2026-10-12"}], today=date(2026, 10, 16))
    assert not any("TPST" in p for p in h["problems"])
    h = I.health([{"status": "OPEN", "ticker": "TPST", "entry_date": "2026-01-20"}], today=date(2026, 10, 16))
    assert any("TPST" in p for p in h["problems"])


# ── repairs made by whoever changed the table ─────────────────────────────────

def test_the_nightly_run_repairs_every_name_whose_corrections_changed(monkeypatch, tmp_path):
    """A record the scheduler's guard or a hand edit changed between runs: `run` hands EVERY name to
    `reset_corrected` (which resets those whose stored corrections differ), not only the keys it changed itself."""
    from src.data import integrity as I
    import src.data.intraday_store as IS
    monkeypatch.setattr(I, "OUT_DIR", tmp_path / "integrity")
    monkeypatch.setattr(IS, "raw_split_rows", lambda: {})
    monkeypatch.setattr(IS, "split_rows", lambda: {})
    monkeypatch.setattr(SC, "verify_records", lambda *a, **k: {})
    monkeypatch.setattr(SC, "find_missing", lambda *a, **k: {})
    seen = []
    monkeypatch.setattr(IS, "reset_corrected", lambda tickers=None, today=None: seen.append(tickers) or {"XRX": "reset"})
    rep = I.run(date(2026, 10, 17), names=[], do_repair=True)
    assert seen == [None] and rep["store_resets"] == {"XRX": "reset"}
    seen.clear()
    I.run(date(2026, 10, 17), names=[], do_repair=False)
    assert seen == []


def test_a_failed_record_check_does_not_stop_the_nightly_run(monkeypatch, tmp_path):
    from src.data import integrity as I
    import src.data.intraday_store as IS
    monkeypatch.setattr(I, "OUT_DIR", tmp_path / "integrity")
    monkeypatch.setattr(IS, "raw_split_rows", lambda: {})
    monkeypatch.setattr(IS, "split_rows", lambda: {})
    monkeypatch.setattr(SC, "find_missing", lambda *a, **k: {})

    def boom(*a, **k):
        raise PermissionError(5, "Access is denied")
    monkeypatch.setattr(SC, "verify_records", boom)
    rep = I.run(date(2026, 10, 17), names=[], do_repair=False)
    assert "Access is denied" in rep["verify_error"] and (tmp_path / "integrity" / "2026-10-17.json").exists()


def test_the_pre_open_check_repairs_every_changed_name(monkeypatch):
    from src.data import integrity as I
    import src.data.intraday_store as IS
    monkeypatch.setattr(IS, "raw_split_rows", lambda: {})
    monkeypatch.setattr(SC, "verify_records", lambda *a, **k: {})
    seen = []
    monkeypatch.setattr(IS, "reset_corrected", lambda tickers=None, today=None: seen.append(tickers) or {})
    I.verify_today(date(2026, 10, 19))
    assert seen == [None]


def test_a_contradicted_split_is_not_loaded_into_the_store(monkeypatch, tmp_path):
    """The pre-open run and the scorer reset a name whose split took effect: a record the raw prices of its date
    contradict (suspect) is NOT loaded — Polygon's adjustment by it would hand the scorer a fake jump to short."""
    from src.data import intraday_store as IS
    p = tmp_path / "ABC.pkl"
    p.write_bytes(b"x")
    monkeypatch.setattr(IS, "_path", lambda tk: p)
    monkeypatch.setattr(IS, "latest_splits", lambda today: {"ABC": date(2026, 10, 19)})
    monkeypatch.setattr(IS, "split_to_apply", lambda tk, today, splits=None: date(2026, 10, 19))
    resets = []
    monkeypatch.setattr(IS, "reset_deep_30m", lambda tk, today=None, why=None: resets.append(tk) or "reset")
    SC.save_checks({"ABC|2026-10-19": {"status": "suspect", "provisional": True, "from": 1.0, "to": 2.0}})
    assert IS.reset_split_tickers(["ABC"], date(2026, 10, 19)) == {"ABC": "held_suspect"} and resets == []
    SC.save_checks({"ABC|2026-10-19": {"status": "verified", "from": 1.0, "to": 2.0}})
    assert IS.reset_split_tickers(["ABC"], date(2026, 10, 19)) == {"ABC": "reset"} and resets == ["ABC"]
