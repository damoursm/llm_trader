"""The pivot label on 30-minute bars (2026-09-14 directive; the ONLY label since 2026-09-16).

"Pivots can theoretically happen during the same day": on daily bars the
first candidate bar after any anchor was tomorrow, so a swing later the same
session was unrepresentable. The label now runs the same H/L zigzag over
30-minute regular-hours bars from each row's own tick time and price. These
tests pin the properties that make that honest:

* nothing resolves at or before the tick — the bar containing it is excluded;
* a pivot later the SAME session is representable and flagged;
* no 30-minute history means NO label, never a silent daily fallback;
* the as-of cutoff hides a pivot whose confirming bar is not yet visible;
* a date-only row anchors at that session's close (next session's first bar);
* serving requires an artifact stamped with the label basis IN FORCE — the
  retired daily basis abstains;
* Polygon pagination follows next_url instead of truncating to the oldest page.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.analysis import pivot_rows as pr
from src.analysis import pivot_target as pt

EDT_OPEN_UTC = 13.5          # 09:30 ET during daylight time = 13:30 UTC


def _sessions(start: str, n_sessions: int):
    """RTH 30-min bar starts (naive UTC) for n consecutive weekdays from start."""
    days = pd.bdate_range(start, periods=n_sessions)
    out = []
    for d in days:
        for k in range(13):
            out.append(d + pd.Timedelta(hours=EDT_OPEN_UTC) + pd.Timedelta(minutes=30 * k))
    return pd.DatetimeIndex(out)


def _flat_series(idx, level=100.0):
    n = len(idx)
    c = np.full(n, level); h = np.full(n, level + 0.05); lo = np.full(n, level - 0.05)
    return c, h, lo


def _plant_peak(h, lo, c, at: int, peak=103.0, trough=99.0):
    """A swing HIGH on bar ``at`` confirmed by a >1% drop on the bars after it."""
    h[at] = peak; c[at] = peak - 0.5
    for k in range(1, 4):
        lo[at + k] = trough; c[at + k] = trough + 0.2; h[at + k] = trough + 0.4


@pytest.fixture
def intraday():
    pr.clear()
    yield
    pr.clear()


def test_pivot_never_resolves_at_or_before_the_tick(intraday):
    idx = _sessions("2026-08-03", 8)
    c, h, lo = _flat_series(idx)
    # planted peak on session 6 bar 4 (index 6*13+4 = 82); tick INSIDE that bar
    at = 6 * 13 + 4
    _plant_peak(h, lo, c, at)
    tick_inside = idx[at] + pd.Timedelta(minutes=10)
    r = pt.intraday_pivot_targets(idx, c, h, lo, [(tick_inside, 100.0)])[0]
    # the bar containing the tick cannot be split -> it is NOT eligible; the
    # next resolved pivot must start strictly after the tick
    assert r is not None and r["resolved"]
    assert pd.Timestamp(r["end_ts"]) > tick_inside
    assert r["end_idx"] > at


def test_same_session_pivot_is_representable(intraday):
    idx = _sessions("2026-08-03", 8)
    c, h, lo = _flat_series(idx)
    at = 6 * 13 + 6                        # session 6, 12:30 ET bar
    _plant_peak(h, lo, c, at)
    tick = idx[6 * 13 + 1] + pd.Timedelta(minutes=5)      # 10:05 ET the same session
    r = pt.intraday_pivot_targets(idx, c, h, lo, [(tick, 100.0)])[0]
    assert r["resolved"] and r["same_session"] is True
    assert r["target_pct"] == pytest.approx(3.0)           # 103 / 100 - 1
    assert r["is_peak"] is True
    assert pd.Timestamp(r["confirm_ts"]) > pd.Timestamp(r["end_ts"])   # confirmed on a LATER bar


def test_unresolved_marks_at_last_close_and_tick_after_data_is_none(intraday):
    idx = _sessions("2026-08-03", 6)
    c, h, lo = _flat_series(idx)
    c[-1] = 101.0
    tick = idx[3 * 13] + pd.Timedelta(minutes=1)
    r = pt.intraday_pivot_targets(idx, c, h, lo, [(tick, 100.0)])[0]
    assert r is not None and r["resolved"] is False
    assert r["target_pct"] == pytest.approx(1.0)           # last visible close, not an extreme
    assert pt.intraday_pivot_targets(idx, c, h, lo, [(idx[-1] + pd.Timedelta(hours=1), 100.0)])[0] is None


def test_no_intraday_history_means_no_label_never_daily(intraday, monkeypatch):
    monkeypatch.setattr(pt, "_series_30m", lambda tk: None)
    assert pt.next_pivot_targets("NOHIST", [("2026-08-10T15:00:00+00:00", 10.0)]) == [None]
    assert pr.pivot_fwd_row("NOHIST", "2026-08-10T15:00:00+00:00", 10.0) is None
    assert pr.has_intraday_history("NOHIST") is False


def test_asof_cutoff_hides_an_unconfirmed_pivot(intraday):
    idx = _sessions("2026-08-03", 8)
    c, h, lo = _flat_series(idx)
    at = 6 * 13 + 4
    _plant_peak(h, lo, c, at)              # confirms on bar at+1 (the >1% drop)
    tick = idx[5 * 13] + pd.Timedelta(minutes=1)
    full = pt.intraday_pivot_targets(idx, c, h, lo, [(tick, 100.0)])[0]
    assert full["resolved"]
    cut = pt.intraday_pivot_targets(idx, c, h, lo, [(tick, 100.0)], asof=idx[at + 1])[0]
    assert cut is not None and cut["resolved"] is False   # confirming bar not yet visible


def test_date_only_row_anchors_at_the_session_close(intraday, monkeypatch):
    idx = _sessions("2026-08-03", 8)
    c, h, lo = _flat_series(idx)
    at = 5 * 13 + 0                        # the OPEN of session 5 is the peak
    _plant_peak(h, lo, c, at)
    monkeypatch.setattr(pt, "_series_30m", lambda tk: (idx, c, h, lo))
    d4 = idx[4 * 13].date()
    r = pr.pivot_fwd_row("STK", d4, None, fallback_close=100.0)
    assert r is not None
    fwd, end_day, end_ts = r
    assert fwd == pytest.approx(3.0)
    assert end_day == idx[at].date() and end_day > d4           # resolved NEXT session, never same-day
    assert pd.Timestamp(end_ts) == idx[at]                      # the resolving bar's start, naive UTC
    # the trimmed scan (window + warm-up) must agree with the full one
    pr.clear()
    r2 = pr.pivot_fwd_row("STK", d4, None, fallback_close=100.0, since=d4)
    assert r2 is not None and r2[0] == pytest.approx(fwd) and r2[1] == end_day
    assert pt.session_close_utc("2026-08-03") == pd.Timestamp("2026-08-03 20:00:00")   # EDT
    assert pt.session_close_utc("2026-12-01") == pd.Timestamp("2026-12-01 21:00:00")   # EST


def test_basis_fingerprint():
    assert pt.pivot_basis() == "hl1@30m"


def test_serving_requires_the_label_basis_in_force(intraday, monkeypatch):
    """An artifact stamped with the basis in force (hl1@30m) scores; one
    stamped with the retired daily basis (hl1) or another marks basis abstains."""
    import src.signals.ml_model as mm

    class _Stub:
        def predict(self, X):
            return np.array([0.25])

    def _art(basis):
        return {"config": {"target": "pivot_rank", "pivot_basis": basis},
                "features": ["f1", "f2"], "model": _Stub()}

    monkeypatch.setattr("src.analysis.ml_dataset.ticker_feature_frame",
                        lambda tk: pd.DataFrame([{"f1": 1.0, "f2": 2.0}]))
    monkeypatch.setattr(pt, "latest_leg_features", lambda tk: {"f1": 1.0, "f2": 2.0})
    monkeypatch.setattr(mm, "_load_artifact", lambda: _art("hl1@30m"))
    mm.reset_caches()
    assert mm.compute_ml_score("AAA")[1] == "OK"
    for stale in ("hl1", "close1@30m", "hl2@30m"):
        monkeypatch.setattr(mm, "_load_artifact", lambda s=stale: _art(s))
        mm.reset_caches(); mm._BASIS_WARNED = False
        assert mm.compute_ml_score("AAA") == (0.0, "BASIS_STALE"), stale


def test_polygon_intraday_pagination_follows_next_url(monkeypatch):
    from src.data import polygon_client as pc
    from config.settings import settings
    monkeypatch.setattr(settings, "polygon_api_key", "k")
    page1 = {"results": [{"t": int(pd.Timestamp("2026-08-03 13:30", tz="UTC").timestamp() * 1000),
                          "o": 1, "h": 1, "l": 1, "c": 1, "v": 1}],
             "next_url": "https://api.polygon.io/v2/aggs/cursor?cursor=abc"}
    page2 = {"results": [{"t": int(pd.Timestamp("2026-08-04 13:30", tz="UTC").timestamp() * 1000),
                          "o": 2, "h": 2, "l": 2, "c": 2, "v": 2}]}
    monkeypatch.setattr(pc, "_get", lambda path, params=None, _attempt=0: page1)

    class _R:
        def raise_for_status(self): pass
        def json(self): return page2
    seen = {}
    monkeypatch.setattr(pc.httpx, "get", lambda url, timeout=None: seen.setdefault("url", url) and _R() or _R())
    df = pc.get_intraday_bars("AAA", lookback_days=10)
    assert len(df) == 2 and "apiKey=k" in seen["url"] and "cursor=abc" in seen["url"]


def test_asof_via_confirmation_equals_the_truncated_scan_and_shares_one_memo(intraday, monkeypatch):
    """`pivot_fwd_row` applies the as-of cutoff through the pivot's CONFIRMING bar
    on ONE memoised scan (the cutoff is not a memo key): identical to truncating
    the series before scanning, without rescanning per walk-forward date."""
    idx = _sessions("2026-08-03", 10)
    c, h, lo = _flat_series(idx)
    at = 6 * 13 + 4
    _plant_peak(h, lo, c, at)                 # peak on session 6, confirmed on bar at+1 (session 6)
    monkeypatch.setattr(pt, "_series_30m", lambda tk: (idx, c, h, lo))
    tick = idx[5 * 13] + pd.Timedelta(minutes=1)
    full = pr.pivot_fwd_row("STK", tick, 100.0)
    assert full is not None and full[0] == pytest.approx(3.0)
    d6 = idx[6 * 13].date()
    assert pr.pivot_fwd_row("STK", tick, 100.0, asof_day=d6) is None          # confirming bar not visible on d6
    d7 = idx[7 * 13].date()
    r7 = pr.pivot_fwd_row("STK", tick, 100.0, asof_day=d7)
    assert r7 is not None and r7[0] == pytest.approx(3.0)                     # visible from the next session
    assert pr.has_intraday_history("STK", asof_day=idx[2 * 13].date()) is False   # < MIN_BARS visible
    assert len(pr._SCANS) == 1                                                # three cutoffs, one scan
    # the truncated scan gives the same answers
    cut = pr._asof_cut_utc(d6); m = int(idx.searchsorted(cut, side="left"))
    trunc = pt.intraday_pivot_targets(idx[:m], c[:m], h[:m], lo[:m], [(tick, 100.0)])[0]
    assert trunc is not None and trunc["resolved"] is False


def test_close_basis_marks_on_the_bar_close(intraday, monkeypatch):
    from config.settings import settings
    idx = _sessions("2026-08-03", 8)
    c, h, lo = _flat_series(idx)
    at = 6 * 13 + 6
    _plant_peak(h, lo, c, at)                 # high 103.0, close 102.5 on the pivot bar
    tick = idx[6 * 13 + 1] + pd.Timedelta(minutes=5)
    hl = pt.intraday_pivot_targets(idx, c, h, lo, [(tick, 100.0)])[0]
    assert hl["target_pct"] == pytest.approx(3.0)
    cl = pt.intraday_pivot_targets(idx, c, h, lo, [(tick, 100.0)], basis="close")[0]
    assert cl["resolved"] and cl["target_pct"] == pytest.approx(2.5)         # the bar's CLOSE, a tradeable print
    monkeypatch.setattr(settings, "pivot_label_basis", "close")
    monkeypatch.setattr(pt, "_series_30m", lambda tk: (idx, c, h, lo))
    pr.clear()
    r = pr.pivot_fwd_row("STK", tick, 100.0)
    assert r is not None and r[0] == pytest.approx(2.5)
    assert pt.pivot_basis() == "close1@30m"
    monkeypatch.setattr(settings, "pivot_label_basis", "hl")
    assert pt.pivot_basis() == "hl1@30m"


def test_threshold_moves_the_fingerprint_and_the_label(intraday, monkeypatch):
    """`pivot_min_move_pct` is THE threshold: it moves the fingerprint (so a
    stale artifact abstains) and what counts as a pivot."""
    from config.settings import settings
    idx = _sessions("2026-08-03", 8)
    c, h, lo = _flat_series(idx)
    at = 6 * 13 + 6
    _plant_peak(h, lo, c, at, peak=101.5, trough=99.9)        # a 1.6% swing: confirms at 1%, not at 2%
    tick = idx[6 * 13 + 1] + pd.Timedelta(minutes=5)
    assert pt.intraday_pivot_targets(idx, c, h, lo, [(tick, 100.0)])[0]["resolved"] is True
    monkeypatch.setattr(settings, "pivot_min_move_pct", 2.0)
    assert pt.pivot_basis() == "hl2@30m"
    r = pt.intraday_pivot_targets(idx, c, h, lo, [(tick, 100.0)])[0]
    assert r is not None and r["resolved"] is False                 # a 1.6% reversal is no pivot at 2%
    monkeypatch.setattr(pt, "_series_30m", lambda tk: (idx, c, h, lo))
    pr.clear()
    assert pr.pivot_fwd_row("STK", tick, 100.0) is None
    monkeypatch.setattr(settings, "pivot_min_move_pct", 1.0)
    pr.clear()
    assert pr.pivot_fwd_row("STK", tick, 100.0) is not None and pt.pivot_basis() == "hl1@30m"
