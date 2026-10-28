"""PREREG209's KD+ in production (user 2026-10-25: "Build KD+ for production and deploy it."; from the 2026-10-26
session; `src/signals/dip_wsb.py`): THE WALLSTREETBETS PREFERENCE in the dip book's daily top pick. What must hold: the
deployed values; PREREG208's counting with PREREG209's eight cashtag-only tokens, each post once per name, a class
share's root; the window [09:00 ET - 120 h, 09:00 ET) read from the feed's store, nothing at or after the cut; the
fail-open statuses (a day missing, a stale store, a computation before the cut, an error) prefer nothing; the pick is
the best-ranked flagged signal at or above the floor, else production's; the history index the simulator reads counts
as the store does; the simulator's `wsb_preferred` keeps a day's flagged orders only and follows production's settings.
The look-ahead guard is in `tests/test_no_lookahead.py`."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from config.settings import Settings, settings
from src.data.feeds import wsb
from src.signals import dip_long as dl
from src.signals import dip_wsb as W

DAY = date(2026, 10, 26)                         # a Monday: 09:00 ET = 13:00 UTC (EDT)
PREV = date(2026, 10, 23)


def _post(pid: str, when: datetime, title: str) -> dict:
    return {"id": pid, "created_utc": int(when.timestamp()), "title": title, "score": 1, "num_comments": 0,
            "link_flair_text": None}


def _store(posts):
    by = {}
    for p in posts:
        by.setdefault(datetime.fromtimestamp(p["created_utc"], timezone.utc).date(), []).append(p)
    for d, ps in by.items():
        wsb.write_day(d, ps)


def _full_window(extra=(), last_minutes_before_cut: float = 5.0):
    """A store covering DAY's window: one filler post a day and a fresh one just before the cut, plus ``extra``."""
    t0, t1 = W.window(DAY)
    start, cut = datetime.fromtimestamp(t0, timezone.utc), datetime.fromtimestamp(t1, timezone.utc)
    posts, d, k = [], start.date(), 0
    while d <= cut.date():
        posts.append(_post(f"f{k}", datetime(d.year, d.month, d.day, 0, 30, tzinfo=timezone.utc), "nothing here"))
        d += timedelta(days=1)
        k += 1
    posts.append(_post("fresh", cut - timedelta(minutes=last_minutes_before_cut), "still nothing"))
    _store(posts + list(extra))
    return start, cut


def _on(monkeypatch, **kw):
    for k, v in dict({"enable_dip_long_wsb_prefer": True, "dip_long_wsb_min_titles": 5,
                      "dip_long_wsb_window_hours": 120.0, "dip_long_wsb_cut_et": "09:00",
                      "dip_long_wsb_max_feed_lag_minutes": 60.0, "dip_long_daily_top": 1, "dip_long_rank": "score"},
                     **kw).items():
        monkeypatch.setattr(settings, k, v)


AFTER = datetime(2026, 10, 26, 14, 0, tzinfo=timezone.utc)     # the opening tick's computation, after the cut


def test_the_deployed_values_are_prereg209s():
    f = Settings.model_fields
    assert f["enable_dip_long_wsb_prefer"].default is True
    assert f["dip_long_wsb_min_titles"].default == 5
    assert f["dip_long_wsb_window_hours"].default == 120.0
    assert f["dip_long_wsb_cut_et"].default == "09:00"
    assert f["dip_long_wsb_max_feed_lag_minutes"].default == 60.0


def test_the_counting_is_prereg208s_with_the_eight_cashtag_only_tokens():
    assert W.names_in("NVDA to the moon, $AMD too") == {"NVDA", "AMD"}
    assert W.names_in("RH earnings tonight") == set()          # Robinhood, read as RH only as a cashtag
    assert W.names_in("$RH earnings tonight") == {"RH"}
    assert W.names_in("EDIT: TSLA puts printed") == {"TSLA"}   # "EDIT:" is not Editas
    for tok in ("TD", "IMO", "PSA", "EV", "TV", "DOW"):
        assert W.names_in(f"{tok} is up") == set() and W.names_in(f"${tok} is up") == {tok}
    assert W.names_in("A big DD on GME, GME, GME") == {"GME"}  # a word and a single letter only as cashtags
    assert W.names_in("$A") == {"A"}
    assert W.token_of("BRK-B") == "BRK" and W.token_of("BF.B") == "BF" and W.token_of("XYZ@D") == "XYZ"
    assert W.cashtag_only() >= frozenset(wsb.WORDS) | W.EXTRA_CASHTAG_ONLY


def test_the_window_reads_only_the_120_hours_before_the_cut():
    t0, t1 = W.window(DAY)
    assert datetime.fromtimestamp(t1, timezone.utc) == datetime(2026, 10, 26, 13, 0, tzinfo=timezone.utc)
    assert t1 - t0 == 120 * 3600
    start, cut = datetime.fromtimestamp(t0, timezone.utc), datetime.fromtimestamp(t1, timezone.utc)
    inside = [_post(f"in{k}", start + timedelta(hours=k * 20), "$NVDA dip") for k in range(6)]     # 0 .. 100 h
    edges = [_post("at_start", start, "NVDA"), _post("before", start - timedelta(seconds=1), "NVDA"),
             _post("at_cut", cut, "NVDA"), _post("after", cut + timedelta(minutes=10), "NVDA NVDA")]
    _full_window(inside + edges)
    w = W.window_counts(DAY, ["NVDA", "AAPL"], now=AFTER)
    assert w["status"] == "ok"
    assert w["counts"] == {"NVDA": 7, "AAPL": 0}             # the 6 inside + the one at the start; never the cut's


def test_a_missing_day_a_stale_store_or_an_early_computation_prefers_nothing(tmp_path, monkeypatch):
    start, cut = _full_window([_post(f"n{k}", cut_k, "$NVDA") for k, cut_k in
                               enumerate([datetime(2026, 10, 25, 12, k, tzinfo=timezone.utc) for k in range(6)])])
    assert W.window_counts(DAY, ["NVDA"], now=AFTER)["status"] == "ok"
    assert W.window_counts(DAY, ["NVDA"], now=cut - timedelta(minutes=1))["status"] == "early"
    wsb.path_for(date(2026, 10, 23)).unlink()
    w = W.window_counts(DAY, ["NVDA"], now=AFTER)
    assert w["status"] == "missing_days" and w["counts"] == {} and "2026-10-23" in w["missing"]
    monkeypatch.setattr(settings, "feeds_wsb_dir", str(tmp_path / "other"))
    _full_window(last_minutes_before_cut=90.0)                # the feed's last run 90 minutes before the cut
    assert W.window_counts(DAY, ["NVDA"], now=AFTER)["status"] == "stale"


def _sig(tk, score, dv=3e9, rsi=5.0):
    return {"ticker": tk, "filter_score": score, "rsi2": rsi, "dv20": dv}


def test_the_pick_is_the_best_flagged_signal_at_or_above_the_floor(monkeypatch):
    _on(monkeypatch, dip_long_order_min_dollar_volume=2e9)
    t0, _ = W.window(DAY)
    start = datetime.fromtimestamp(t0, timezone.utc)
    _full_window([_post(f"a{k}", start + timedelta(hours=10 + k), "$AMD") for k in range(5)]
                 + [_post(f"t{k}", start + timedelta(hours=30 + k), "TSLA") for k in range(5)]
                 + [_post(f"s{k}", start + timedelta(hours=40 + k), "SMLL") for k in range(9)])
    monkeypatch.setattr(W, "window_counts", lambda d, tks, now=None, _f=W.window_counts: _f(d, tks, now=AFTER))
    sig = [_sig("NVDA", 0.30), _sig("AMD", 0.10), _sig("TSLA", 0.20), _sig("SMLL", 0.90, dv=1.5e9)]
    keep, top = dl._daily_top([dict(s) for s in sig], DAY, PREV, 1)
    info = top["info"]
    assert [s["ticker"] for s in keep] == ["TSLA"]            # the best-scored flagged one, not NVDA (unflagged)
    assert info["wsb"]["status"] == "ok" and info["wsb"]["preferred"] is True
    assert info["wsb"]["pick_without"] == "NVDA"
    assert sorted(info["wsb"]["flagged"]) == ["AMD", "SMLL", "TSLA"]   # SMLL flagged but under the floor: never picked
    assert info["picked"][0]["wsb_titles"] == 5 and info["picked"][0]["wsb_flagged"] is True
    ranks = {s["ticker"]: s.get("top_rank") for s in keep + top["not_top"]}
    assert ranks == {"TSLA": 1, "AMD": 2, "NVDA": 3, "SMLL": None}


def test_without_a_flagged_signal_or_with_the_rule_off_the_pick_is_productions(monkeypatch):
    _on(monkeypatch, dip_long_order_min_dollar_volume=2e9)
    _full_window()
    monkeypatch.setattr(W, "window_counts", lambda d, tks, now=None, _f=W.window_counts: _f(d, tks, now=AFTER))
    sig = [_sig("NVDA", 0.30), _sig("AMD", 0.10)]
    keep, top = dl._daily_top([dict(s) for s in sig], DAY, PREV, 1)
    assert [s["ticker"] for s in keep] == ["NVDA"] and top["info"]["wsb"]["preferred"] is False
    monkeypatch.setattr(settings, "enable_dip_long_wsb_prefer", False)
    keep, top = dl._daily_top([dict(s) for s in sig], DAY, PREV, 1)
    assert [s["ticker"] for s in keep] == ["NVDA"] and "wsb" not in top["info"]


def test_an_unreadable_store_fails_open_to_productions_pick(monkeypatch):
    _on(monkeypatch)

    def boom(*a, **k):
        raise OSError("disk gone")
    monkeypatch.setattr(W, "window_counts", boom)
    keep, top = dl._daily_top([_sig("NVDA", 0.3), _sig("AMD", 0.1)], DAY, PREV, 1)
    assert [s["ticker"] for s in keep] == ["NVDA"]
    w = top["info"]["wsb"]
    assert w["status"] == "error" and "disk gone" in w["error"] and w["preferred"] is False
    out = {"problems": [], "notes": []}
    dl._wsb_health(top["info"], out)
    assert out["problems"] == [] and any("was off today (error" in n for n in out["notes"])


def test_an_earnings_dip_among_the_flagged_still_ranks_after_the_other_flagged(monkeypatch):
    _on(monkeypatch)
    w = {"status": "ok", "rule": W.RULE}
    a = dict(_sig("AMD", 0.4), wsb_flagged=True, earnings_dip=True)
    b = dict(_sig("TSLA", 0.1), wsb_flagged=True, earnings_dip=False)
    c = dict(_sig("NVDA", 0.9), wsb_flagged=False, earnings_dip=False)
    ranked = sorted([a, b, c], key=dl.rank_key)               # production's order: earnings dips last
    got, pref = W.prefer(ranked, w)
    assert pref and [s["ticker"] for s in got] == ["TSLA", "AMD", "NVDA"]


def test_the_history_index_counts_as_the_store_does(monkeypatch):
    _on(monkeypatch)
    t0, _ = W.window(DAY)
    start = datetime.fromtimestamp(t0, timezone.utc)
    _full_window([_post(f"x{k}", start + timedelta(hours=3 * k), f"$GME and AMC {k}") for k in range(30)]
                 + [_post("late", datetime(2026, 10, 26, 13, 30, tzinfo=timezone.utc), "GME GME")])
    W._index.cache_clear()
    w = W.window_counts(DAY, ["GME", "AMC", "NVDA"], now=AFTER)
    for tk in ("GME", "AMC", "NVDA"):
        assert W.history_titles(tk, DAY) == w["counts"][tk]
    assert W.history_flag("GME", DAY) and not W.history_flag("NVDA", DAY)
    assert W.history_titles("GME", date(2026, 11, 5)) is None  # its window's days are not in the store


def test_the_simulator_keeps_a_days_flagged_orders_only_and_follows_production(monkeypatch):
    from src.backtest import mimic as M
    _on(monkeypatch)
    t0, _ = W.window(DAY)
    start = datetime.fromtimestamp(t0, timezone.utc)
    _full_window([_post(f"z{k}", start + timedelta(hours=5 + k), "$TSLA") for k in range(5)])
    W._index.cache_clear()
    pcs = [{"oid": f"{tk}|{DAY.isoformat()}", "tkn": tk, "dv20": 3e9} for tk in ("NVDA", "TSLA", "AMD")]
    pcs += [{"oid": "MSFT|2026-11-06", "tkn": "MSFT", "dv20": 3e9}]       # a day whose window is not in the store
    log = {}
    got = M.wsb_preferred(pcs, (5, 120.0, "09:00"), log)
    assert [p["oid"] for p in got] == [f"TSLA|{DAY.isoformat()}", "MSFT|2026-11-06"]
    assert log == {"wsb_days_preferred": 1, "wsb_orders_left_out": 2}
    assert M.Mimic().dip_wsb == (5, 120.0, "09:00")           # production's settings (on, the daily top pick on)
    monkeypatch.setattr(settings, "enable_dip_long_wsb_prefer", False)
    assert M.Mimic().dip_wsb is None and M.Mimic(**M.PRE209).dip_wsb is None
    with pytest.raises(ValueError):
        M.Mimic(dip_wsb=(0, 120.0, "09:00"))


def test_posts_filled_from_reddits_api_count_and_keep_the_window_fresh(monkeypatch):
    """The feed's gap fill (2026-10-27): the archive's last post 90 minutes before the cut, Reddit's API's copies of
    the posts after it (``src`` "reddit") — counted like the archive's, and the window is not stale."""
    t0, t1 = W.window(DAY)
    cut = datetime.fromtimestamp(t1, timezone.utc)
    _full_window(last_minutes_before_cut=90.0)
    assert W.window_counts(DAY, ["NVDA"], now=AFTER)["status"] == "stale"
    _store([dict(_post(f"r{k}", cut - timedelta(minutes=60 - 10 * k), "$NVDA dip"), src="reddit") for k in range(5)])
    w = W.window_counts(DAY, ["NVDA"], now=AFTER)
    assert w["status"] == "ok" and w["counts"] == {"NVDA": 5}
