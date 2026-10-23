"""THE NEWS HOLD, SHADOW (PREREG136, `src/signals/news_hold.py`): the vol shorts' exits journaled, the local LLM's
tone of the hold's headlines (faked), the decision, and the outcome of holding k more sessions measured with the
simulator's own `mimic.news_extend` on bars in the exit's terms — a split during the extension changing nothing."""
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.backtest import mimic as M
from src.signals import news_hold as NH

ET = "America/New_York"
_REAL_BORROW = NH._borrow_added                          # captured before the `on` fixture stubs it
DAYS = ["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08", "2024-03-11", "2024-03-12",
        "2024-03-13", "2024-03-14", "2024-03-15", "2024-03-18", "2024-03-19", "2024-03-20", "2024-03-21"]


def at(day: str, hhmm: str) -> datetime:
    return pd.Timestamp(f"{day} {hhmm}").tz_localize(ET).tz_convert("UTC").to_pydatetime()


def frame(close_of=None, split_day: str = None, ratio: float = 1.0) -> pd.DataFrame:
    """Polygon's 30-minute bars 04:00-20:00 ET as `get_intraday_bars_range` returns them (bar starts, UTC): close =
    10 + 0.01 x the bar's index unless ``close_of`` says otherwise; from ``split_day`` on the prices are x ``ratio``
    (as traded after a reverse split)."""
    idx, c = [], []
    k = 0
    for d in DAYS:
        for t in pd.date_range(f"{d} 04:00", f"{d} 19:30", freq="30min"):
            px = 10.0 + 0.01 * k
            if close_of is not None and close_of(d, t) is not None:
                px = close_of(d, t)
            if split_day is not None and d >= split_day:
                px *= ratio
            idx.append(t.tz_localize(ET).tz_convert("UTC"))
            c.append(px)
            k += 1
    c = np.asarray(c, float)
    hi = c * 1.002
    lo = c * 0.998
    return pd.DataFrame({"Open": c, "High": hi, "Low": lo, "Close": c, "Volume": np.full(len(c), 1000)},
                        index=pd.DatetimeIndex(idx))


def trade(exit_reason="sel_target", arm="vol", entry=("2024-03-04", "10:00"), exit_=("2024-03-05", "10:00"),
          entry_price=None, exit_price=None, f=None):
    f = frame() if f is None else f
    e_ts, x_ts = at(*entry), at(*exit_)

    def close_at(ts):
        end = pd.DatetimeIndex(f.index) + pd.Timedelta(minutes=30)
        return float(f["Close"].to_numpy()[np.flatnonzero(end == pd.Timestamp(ts))[0]])
    return {"recommendation_id": "abc123", "ticker": "TEST", "action": "SELL", "type": "STOCK",
            "entry_mechanism": "sel_short", "sel_arm": arm, "status": "CLOSED",
            "entry_datetime": e_ts.isoformat(), "entry_date": entry[0],
            "entry_price": entry_price if entry_price is not None else close_at(e_ts),
            "exit_datetime": x_ts.isoformat(), "exit_decision_datetime": x_ts.isoformat(), "exit_date": exit_[0],
            "exit_price": exit_price if exit_price is not None else close_at(x_ts), "exit_reason": exit_reason,
            "return_pct": 5.0, "sel_account_shares": 100}


@pytest.fixture
def on(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "enable_sel_short_news_hold_shadow", True)
    monkeypatch.setattr(settings, "enable_local_llm", True)
    monkeypatch.setattr(settings, "sel_short_news_hold_sessions", "1,2,3,5,8")
    monkeypatch.setattr(settings, "sel_short_news_hold_primary", 5)
    monkeypatch.setattr(settings, "sel_short_news_hold_lag_minutes", 30)
    monkeypatch.setattr(NH, "_RATINGS", {})
    monkeypatch.setattr(NH, "_RATINGS_MTIME", None)
    monkeypatch.setattr(NH, "kick", lambda now=None: False)              # no thread: the tests run the pass
    monkeypatch.setattr(NH, "_borrow_added", lambda rec, ns, b: (0.001, "test"))   # 0.1% of the entry
    from src.data import intraday_store
    monkeypatch.setattr(intraday_store, "split_rows", lambda: {})
    return tmp_path


def test_sessions_and_eligibility(on, monkeypatch):
    assert NH.ks() == [1, 2, 3, 5, 8]
    monkeypatch.setattr(settings, "sel_short_news_hold_sessions", "1,2,30")
    with pytest.raises(ValueError):
        NH.ks()
    assert NH.eligible(trade()) == "decide"
    assert NH.eligible(trade(exit_reason="sel_time")) == "decide"
    assert NH.eligible(trade(exit_reason="sel_cover")) == "cover"
    assert NH.eligible(trade(exit_reason="sel_margin_call")) is None      # an account-level close: not the rule's
    assert NH.eligible(trade(arm="etf")) is None and NH.eligible(trade(arm="fall")) is None
    assert NH.eligible({**trade(), "status": "OPEN"}) is None


def test_exits_are_journaled_once_and_only_when_on(on, monkeypatch):
    assert NH.after_exits([trade(), trade(arm="etf")]) == 1
    assert (NH.root() / "pending" / "abc123.json").exists()
    assert NH.after_exits([trade()]) == 0                                 # already pending
    monkeypatch.setattr(settings, "enable_sel_short_news_hold_shadow", False)
    assert NH.after_exits([{**trade(), "recommendation_id": "zzz"}]) == 0
    assert not (NH.root() / "pending" / "zzz.json").exists()


def _arts(*rows):
    return [{"id": title, "published_utc": ts.isoformat(), "title": title, "publisher": "X",
             "polygon_sentiment": None} for ts, title in rows]


def test_the_decision_reads_the_mean_tone_inside_the_window(on, monkeypatch):
    calls = []

    def rate(sym, title):
        calls.append(title)
        return {"score": {"good": 1.0, "bad": -1.0, "meh": 0.0}.get(title.split()[0]), "raw": title[:3]}
    monkeypatch.setattr(NH, "rate", rate)
    t = trade()
    x = NH._exit_instant(t)
    rows = [(at("2024-03-04", "11:00"), "good one"), (at("2024-03-04", "12:00"), "meh two"),
            (x - timedelta(minutes=29), "bad late"),                  # inside the lag: not counted, not rated
            (at("2024-03-04", "09:00"), "bad early")]                 # before the entry: not counted
    monkeypatch.setattr(NH, "fetch_headlines", lambda tk, a, b: _arts(*rows))
    rec = {"key": "abc123", "trade": t}
    d = NH.decide(rec, now=x + timedelta(minutes=5))
    assert d["decision"] == "hold" and d["mean_tone"] == pytest.approx(0.5) and d["n_counted"] == 2
    assert sorted(calls) == ["good one", "meh two"]
    assert (NH.root() / "open" / "abc123.json").exists()
    # the ratings are cached: a second reading asks the LLM nothing
    calls.clear()
    NH.decide({"key": "abc124", "trade": {**t, "recommendation_id": "abc124"}}, now=x + timedelta(minutes=6))
    assert calls == []
    # a mean of 0 is not positive; no rated headline is no news
    monkeypatch.setattr(NH, "fetch_headlines", lambda tk, a, b: _arts((at("2024-03-04", "11:00"), "good"),
                                                                      (at("2024-03-04", "11:30"), "bad")))
    assert NH.decide({"key": "k2", "trade": t}, now=x)["decision"] == "other_news"
    monkeypatch.setattr(NH, "fetch_headlines", lambda tk, a, b: _arts((x - timedelta(minutes=10), "good late")))
    assert NH.decide({"key": "k3", "trade": t}, now=x)["decision"] == "no_news"


def test_an_unanswered_source_leaves_the_exit_pending(on, monkeypatch):
    t = trade()
    monkeypatch.setattr(NH, "fetch_headlines", lambda tk, a, b: None)               # Polygon failed
    assert NH.decide({"key": "p1", "trade": t}) is None
    monkeypatch.setattr(NH, "fetch_headlines", lambda tk, a, b: _arts((at("2024-03-04", "11:00"), "good")))
    monkeypatch.setattr(NH, "rate", lambda sym, title: {"error": "connection refused"})   # the LLM is down
    assert NH.decide({"key": "p2", "trade": t}) is None
    NH.after_exits([t])
    s = NH.run_once(now=at("2024-03-05", "11:00"))
    assert s["waiting"] == 1 and (NH.root() / "pending" / "abc123.json").exists()


def _sim_bars(f: pd.DataFrame) -> dict:
    """The simulator's bar dict built straight from the frame (`mimic.ExtBars._load`'s construction)."""
    start = pd.DatetimeIndex(f.index).asi8.astype(np.int64)
    et = pd.DatetimeIndex(f.index).tz_convert(ET)
    mins = et.hour * 60 + et.minute
    return {"start": start, "end": start + M.BAR_NS, "o": f["Open"].to_numpy(float), "h": f["High"].to_numpy(float),
            "l": f["Low"].to_numpy(float), "c": f["Close"].to_numpy(float), "v": f["Volume"].to_numpy(float),
            "rth": np.asarray((mins >= 570) & (mins < 960)),
            "day": np.asarray((et.normalize().tz_localize(None) - M.EPOCH).days, np.int64)}


class _B:
    def __init__(self, b):
        self.b = b

    def get(self, name):
        return self.b


def _open_rec(t):
    return {"key": "abc123", "trade": t, "cover_multiple": 4.0, "exit_at": NH._exit_instant(t).isoformat(),
            "ks": {str(k): {"status": "pending"} for k in NH.ks()}, "primary_k": 5}


def test_the_outcome_is_the_simulators_news_extend(on):
    f = frame()
    t = trade(f=f)
    rec = NH.measure(_open_rec(t), now=at(DAYS[-1], "19:59"), frame=f)
    b = _sim_bars(f)
    p = {"tkn": "TEST", "ens": int(pd.Timestamp(t["entry_datetime"]).value),
         "xns": int(pd.Timestamp(t["exit_datetime"]).value), "e": t["entry_price"], "x": t["exit_price"],
         "cover_mult": 4.0, "pt": np.array([int(pd.Timestamp(t["exit_datetime"]).value)], np.int64),
         "pc": np.array([t["exit_price"]]), "why": "target"}
    for k in NH.ks():
        q, why = M.news_extend(p, k, _B(b), lambda d: d)
        got = rec["ks"][str(k)]
        assert got["status"] == "done" and got["why"] == why == "news_hold"
        assert got["exit_ns"] == q["xns"] and got["exit_price"] == pytest.approx(q["x"])
        assert got["gross_pct"] == pytest.approx((t["exit_price"] - q["x"]) / t["entry_price"] * 100.0, abs=1e-3)
        assert got["net_pct"] == pytest.approx(got["gross_pct"] - 0.1, abs=1e-3)
        assert got["net_usd"] == pytest.approx(got["net_pct"] / 100.0 * t["entry_price"] * 100, abs=0.01)
    # k sessions after the exit's day, at the same clock: the exit day is DAYS[1], k=2 -> DAYS[3] at 10:00
    assert rec["ks"]["2"]["exit_at"] == at(DAYS[3], "10:00").isoformat()


def test_a_reverse_split_in_the_extension_changes_nothing(on, monkeypatch):
    from src.data import intraday_store
    base = frame()
    t = trade(f=base)
    want = NH.measure(_open_rec(t), now=at(DAYS[-1], "19:59"), frame=base)["ks"]
    raw = frame(split_day=DAYS[4], ratio=10.0)                       # 1-for-10 reverse split executed on DAYS[4]
    monkeypatch.setattr(intraday_store, "split_rows", lambda: {"TEST": [(date.fromisoformat(DAYS[4]), 10.0, 1.0)]})
    got = NH.measure(_open_rec(t), now=at(DAYS[-1], "19:59"), frame=raw)["ks"]
    for k in want:
        assert got[k]["exit_ns"] == want[k]["exit_ns"]
        assert got[k]["exit_price"] == pytest.approx(want[k]["exit_price"])
        assert got[k]["net_pct"] == pytest.approx(want[k]["net_pct"])


def test_the_squeeze_cover_ends_every_later_k(on):
    def spike(d, tt):
        return 60.0 if (d == DAYS[3] and tt.hour == 11 and tt.minute == 0) else None   # the 11:00-11:30 bar
    f = frame(spike)
    t = trade(f=f)
    rec = NH.measure(_open_rec(t), now=at(DAYS[-1], "19:59"), frame=f)
    assert rec["ks"]["1"]["why"] == "news_hold"                    # DAYS[2]: before the spike
    for k in ("3", "5", "8"):
        assert rec["ks"][k]["why"] == "news_cap" and rec["ks"][k]["exit_at"] == at(DAYS[3], "11:30").isoformat()
        assert rec["ks"][k]["exit_price"] == pytest.approx(60.0)


def test_a_k_waits_for_its_session_and_for_bars_that_ended(on):
    f = frame()
    t = trade(f=f)
    rec = NH.measure(_open_rec(t), now=at(DAYS[3], "09:59"), frame=f)    # k 1 measured; k 2 at DAYS[3] 10:00 not yet
    assert rec["ks"]["1"]["status"] == "done"
    assert rec["ks"]["2"]["status"] == "pending" and rec["ks"]["2"]["waiting"] in ("no_target_mark", "no_session")
    assert not NH._finished(rec)


def test_the_background_pass_decides_measures_and_settles(on, monkeypatch):
    f = frame()
    t = trade(f=f)
    monkeypatch.setattr(NH, "fetch_headlines", lambda tk, a, b: _arts((at("2024-03-04", "11:00"), "good")))
    monkeypatch.setattr(NH, "rate", lambda sym, title: {"score": 1.0, "raw": "1"})
    from src.data import polygon_client as pc
    monkeypatch.setattr(pc, "get_intraday_bars_range", lambda *a, **k: f)
    NH.after_exits([t, trade(exit_reason="sel_cover", f=f) | {"recommendation_id": "cov1"}])
    s = NH.run_once(now=at(DAYS[-1], "19:59"))
    assert s["decided"] == 2 and s["settled"] == 1
    st = NH.status()
    assert st["decisions"] == {"hold": 1, "cover": 1} and st["settled"] == 1 and st["open"] == 0
    assert len(st["primary_net_pct"]) == 1
    notes = NH.health(now=at(DAYS[-1], "20:00"))["notes"]
    assert any("2 vol exit(s) read, 1 would have been held" in n for n in notes)
    assert NH.health()["problems"] == []


def test_a_stale_pending_exit_is_a_note(on, monkeypatch):
    t = trade()
    NH.after_exits([t], now=at("2024-03-05", "10:05"))
    notes = NH.health(now=at("2024-03-05", "18:00"))["notes"]
    assert any("undecided since" in n for n in notes)


def test_the_shadow_never_costs_the_exits(on, monkeypatch):
    from src.performance import tracker

    def boom(*a, **k):
        raise RuntimeError("journal disk full")
    monkeypatch.setattr(NH, "after_exits", boom)
    tracker._sel_short_news_hold([trade()], datetime.now(timezone.utc))      # logged, never raised


# ── the review's fixes (2026-10-17): one pass at a time, idempotent decisions, give-ups, no shared ledger state ──

def test_a_held_lock_skips_the_pass_and_a_stale_one_is_broken(on, monkeypatch):
    import os, time as _t
    NH.after_exits([trade()])
    lock = NH._lock_path()
    lock.write_text("1 other")
    assert NH.run_once(now=at("2024-03-05", "11:00")).get("locked") == 1
    assert (NH.root() / "pending" / "abc123.json").exists()          # nothing touched
    old = _t.time() - NH.LOCK_STALE_SECONDS - 10
    os.utime(lock, (old, old))
    monkeypatch.setattr(NH, "fetch_headlines", lambda tk, a, b: [])
    s = NH.run_once(now=at("2024-03-05", "11:00"))
    assert "locked" not in s and s["decided"] == 1 and not lock.exists()


def test_a_decided_exit_is_never_decided_again(on, monkeypatch):
    calls = []
    monkeypatch.setattr(NH, "fetch_headlines", lambda tk, a, b: calls.append(tk) or [])
    t = trade()
    NH.after_exits([t])
    NH.run_once(now=at("2024-03-05", "11:00"))
    assert len(calls) == 1
    # a pending file left behind by a crash after the decision line: unlinked, not decided twice
    NH._write_json(NH.root() / "pending" / "abc123.json", {"key": "abc123", "journaled_at": "2024-03-05T15:00:00+00:00",
                                                           "trade": {k: t.get(k) for k in NH._TRADE_FIELDS}})
    NH.run_once(now=at("2024-03-05", "12:00"))
    assert len(calls) == 1 and not (NH.root() / "pending" / "abc123.json").exists()
    assert NH.status()["decisions"] == {"no_news": 1}


def test_a_pending_exit_gives_up_after_its_window(on, monkeypatch):
    monkeypatch.setattr(NH, "fetch_headlines", lambda tk, a, b: None)                # Polygon never answers
    NH.after_exits([trade()], now=at("2024-03-05", "10:05"))
    NH.run_once(now=at("2024-03-06", "10:05"))
    assert NH.status()["pending"] == 1
    NH.run_once(now=at("2024-03-05", "10:05") + timedelta(hours=NH.PENDING_GIVE_UP_HOURS + 1))
    st = NH.status()
    assert st["pending"] == 0 and st["decisions"] == {"undecided": 1} and st["open"] == 0


def test_an_open_hold_without_bars_gives_up(on, monkeypatch):
    t = trade()
    rec = _open_rec(t)
    from src.data import polygon_client as pc
    monkeypatch.setattr(pc, "get_intraday_bars_range", lambda *a, **k: pd.DataFrame())
    r = NH.measure(dict(rec, ks={k: dict(v) for k, v in rec["ks"].items()}), now=at(DAYS[-1], "19:59"))
    assert all(v["status"] == "pending" for v in r["ks"].values()) and r["last_error"] == "no_bars"
    late = NH._exit_instant(t) + timedelta(days=NH.GIVE_UP_DAYS + 1)
    r = NH.measure(dict(rec, ks={k: dict(v) for k, v in rec["ks"].items()}), now=late)
    assert all(v["status"] == "unmeasured" for v in r["ks"].values()) and NH._finished(r)


def test_the_added_borrow_is_the_trades_own_rate_on_ibkrs_added_charged_days(on):
    """Held from Tue 2024-03-05 to Thu 03-07 (T+2 then): IBKR's charged days grow from settlement Thu 03-07 to Mon
    03-11 — 03-07 .. 03-10, each on the PRIOR session's close (03-06, 03-07, 03-08, 03-08)."""
    import math
    f = frame()
    t = dict(trade(f=f), borrow_rate_now=36.0)
    b = _sim_bars(f)
    new = int(pd.Timestamp(at(DAYS[3], "10:00")).value)
    frac, src = _REAL_BORROW({"trade": t}, new, b)
    last_close = {}
    for d, c, r in zip(b["day"], b["c"], b["rth"]):
        if r:
            last_close[(pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(d))).date()] = float(c)
    prior = [date(2024, 3, 6), date(2024, 3, 7), date(2024, 3, 8), date(2024, 3, 8)]
    want = sum(math.ceil(1.02 * last_close[d] - 1e-9) for d in prior) * 36.0 / 100.0 / 360.0
    assert src == "borrow_rate_now" and frac == pytest.approx(want / t["entry_price"])
    t2 = {k: v for k, v in t.items() if k != "borrow_rate_now"}
    assert _REAL_BORROW({"trade": dict(t2, borrow_fee_pct=12.0)}, new, b)[1] == "borrow_fee_pct"
    assert _REAL_BORROW({"trade": t2}, new, b)[1] == "default"


def test_an_unknown_borrow_is_no_number(on, monkeypatch):
    monkeypatch.setattr(NH, "_borrow_added", lambda rec, ns, b: (None, "error: x"))
    f = frame()
    rec = NH.measure(_open_rec(trade(f=f)), now=at(DAYS[-1], "19:59"), frame=f)
    k = rec["ks"]["5"]
    assert k["status"] == "done" and k["net_pct"] is None and k["borrow_pct"] is None and k["gross_pct"] is not None


def test_the_backfill_does_not_start_the_thread(on, monkeypatch):
    kicked = []
    monkeypatch.setattr(NH, "kick", lambda now=None: kicked.append(1) or False)
    from src.performance import tracker
    monkeypatch.setattr(tracker, "_load_trades", lambda: [trade()])
    assert NH.backfill(date(2024, 3, 1)) == 1 and kicked == []
    NH.after_exits([dict(trade(), recommendation_id="zz9")])
    assert kicked == [1]
