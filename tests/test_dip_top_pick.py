"""PREREG116's Q1 in production (user 2026-10-15: "Deploy Q1"; from the 2026-10-16 session): the dip book's DAILY TOP
PICK — of each day's signals only the best by the meta-model's score becomes an order (`dip_long_daily_top`,
`dip_long_rank`), at a 1/6 slice (`dip_long_account_slices`, as many open) — and AT MOST TWO OPEN SHORTS OF ONE NAME
across the arms (`sel_short_max_open_per_ticker`; its counting is pinned in `tests/test_sel_short.py`). What must
hold: the deployed values; the ranking is the simulator's (`strategies.dip.rank_pieces`: a scored signal by its score,
highest first, an unscored one after every scored one by its RSI2, the order id breaking a tie); no score cut (the
RP33 filter stays off); the other signals are journaled ``not_top`` and never placed, rested or filled; an order of a
day before the rule is never placed (``superseded``) — the bigger slice only ever buys a day's top pick (PREREG117);
the day's best ranks first when orders or fills compete; fail-open to the deepest dip with a banner problem; the
simulator's defaults follow production. The look-ahead guard is in `tests/test_no_lookahead.py`."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from config.settings import Settings, settings
from src.performance import tracker
from src.signals import dip_filter as DF
from src.signals import dip_long as dl
from src.signals import sel_short as ss
from tests.test_dip_filter import DAY, F, PREV, SPY_IN, _bars, _bars_of, _daily, _lsig, at

META = {"policy": {"cut": 7.0}, "fit": {"cut_date": "2026-10-08"}, "checksums": {"model_sha256": "f" * 64},
        "prior_signals": {}}


def _q1(monkeypatch, **kw):
    for k, v in dict({"dip_long_daily_top": 1, "dip_long_rank": "score", "dip_long_account_slices": 6}, **kw).items():
        monkeypatch.setattr(settings, k, v)


def test_the_deployed_values_are_q1s_with_s3s_slice_and_floor():
    """Q1 (the daily top pick, two shorts a name) with PREREG147's $2B floor from the 2026-10-19 session,
    PREREG159's slice from the 2026-10-20 session and the RP33 cut before the pick from the 2026-10-21 session
    (PREREG166 / 167): each dip long 1/2 of the account, at most 2 open, the pick among the kept signals trading $2B+ a
    day."""
    f = Settings.model_fields
    assert (f["dip_long_daily_top"].default, f["dip_long_rank"].default, f["dip_long_account_slices"].default,
            f["sel_short_max_open_per_ticker"].default, f["enable_dip_long_filter"].default) == (1, "score", 2, 2, True)
    assert f["dip_long_order_min_dollar_volume"].default == 2e9 and f["dip_long_min_dollar_volume"].default == 1e9


# ── PREREG147's S3: the daily top pick's $2B order floor ─────────────────────────────────────────────────────────────

def _scored(*rows):
    """Signals already scored (the RP33 filter's path: `_daily_top` reads their scores, asks no model)."""
    return [dict({"ticker": tk, "rsi2": rsi, "dv20": dv}, **({"filter_score": sc} if sc is not None else
                                                           {"filter_error": "unscored"}))
            for tk, sc, rsi, dv in rows]


def test_the_pick_is_the_best_signal_at_or_above_the_order_floor(monkeypatch):
    _q1(monkeypatch, dip_long_order_min_dollar_volume=2e9)
    sig = _scored(("A", 3.0, 4.0, 1.5e9), ("B", 2.0, 6.0, 2.5e9), ("C", 1.0, 2.0, 5e9), ("D", 2.5, 3.0, 1.9e9))
    keep, top = dl._daily_top(sig, DAY, PREV, 1)
    assert [s["ticker"] for s in keep] == ["B"] and keep[0]["top_rank"] == 1
    rest = {s["ticker"]: s for s in top["not_top"]}
    assert rest["C"]["top_rank"] == 2 and not rest["C"].get("below_order_floor")
    assert rest["A"]["top_rank"] is None and rest["A"]["below_order_floor"] is True and rest["D"]["below_order_floor"]
    assert top["info"]["order_floor"] == 2e9 and top["info"]["below_floor"] == ["A", "D"]
    o = dl.not_top_order(DAY, rest["A"], {"signal_session": str(PREV)})
    assert o["status"] == "not_top" and o["below_order_floor"] is True
    monkeypatch.setattr(settings, "dip_long_order_min_dollar_volume", 0.0)      # no floor: the best of every signal
    keep, top = dl._daily_top(_scored(("A", 3.0, 4.0, 1.5e9), ("B", 2.0, 6.0, 2.5e9)), DAY, PREV, 1)
    assert [s["ticker"] for s in keep] == ["A"] and top["info"]["below_floor"] == []


def test_a_day_without_a_signal_over_the_floor_orders_nothing(monkeypatch):
    _q1(monkeypatch, dip_long_order_min_dollar_volume=2e9)
    keep, top = dl._daily_top(_scored(("A", 3.0, 4.0, 1.5e9), ("B", None, 6.0, 1.1e9)), DAY, PREV, 1)
    assert keep == [] and [s["ticker"] for s in top["not_top"]] == ["A", "B"]
    assert all(s["below_order_floor"] for s in top["not_top"]) and top["info"]["picked"] == []


def test_the_floored_pick_is_the_simulators_choice(monkeypatch):
    """Production's floored pick = the simulator's: `Mimic.dip_min_order_dv` drops the pieces under the floor, then
    `rank_pieces(by="score", topk=1)` — random days, ties and unscored signals included."""
    from src.backtest.strategies import dip as DIP
    _q1(monkeypatch, dip_long_order_min_dollar_volume=2e9)
    rng = np.random.default_rng(11)
    names = ["T1", "T10", "T2", "AB", "ABC", "B"]
    day = "2026-10-16"
    for _ in range(400):
        tks = list(rng.choice(names, size=int(rng.integers(1, 6)), replace=False))
        rsi = {tk: float(rng.choice([0.5, 1.5, 4.0])) for tk in tks}
        sc = {tk: float(rng.choice([-1.0, 0.5, 2.0])) for tk in tks if rng.random() < 0.6}
        dv = {tk: float(rng.choice([1.2e9, 2.0e9, 3.5e9])) for tk in tks}
        pieces = [{"oid": f"{tk}|{day}", "rank": rsi[tk], "tkn": tk, "dv20": dv[tk]} for tk in tks]
        kept = [p for p in pieces if p["dv20"] >= 2e9]
        sim = [p["oid"].split("|")[0] for p in DIP.rank_pieces(kept, {f"{tk}|{day}": v for tk, v in sc.items()},
                                                                by="score", topk=1)]
        keep, _ = dl._daily_top(_scored(*[(tk, sc.get(tk), rsi[tk], dv[tk]) for tk in tks]), DAY, PREV, 1)
        assert sim == [s["ticker"] for s in keep]


def test_the_simulators_floor_is_productions(monkeypatch):
    from src.backtest import mimic as M
    assert M.Mimic(seed=0).dip_min_order_dv is None                    # the baseline: no daily top pick, no floor
    _q1(monkeypatch, dip_long_order_min_dollar_volume=2e9)
    assert M.Mimic(seed=0).dip_min_order_dv == 2e9
    monkeypatch.setattr(settings, "dip_long_order_min_dollar_volume", 0.0)
    assert M.Mimic(seed=0).dip_min_order_dv is None
    import inspect
    assert "mim.dip_min_order_dv" in inspect.getsource(M.live_books)


# ── the ranking ──────────────────────────────────────────────────────────────

def test_the_ranking_is_the_simulators(monkeypatch):
    sig = [{"ticker": "A", "rsi2": 1.0, "filter_score": 0.5}, {"ticker": "B", "rsi2": 8.0, "filter_score": 2.0},
           {"ticker": "C", "rsi2": 0.5}, {"ticker": "D", "rsi2": 9.0, "filter_score": 2.0}, {"ticker": "E", "rsi2": 0.2}]
    assert [s["ticker"] for s in sorted(sig, key=lambda s: dl.rank_key(s, "score"))] == ["B", "D", "A", "E", "C"]
    assert [s["ticker"] for s in sorted(sig, key=lambda s: dl.rank_key(s, "rsi2"))] == ["E", "C", "A", "B", "D"]
    monkeypatch.setattr(settings, "dip_long_rank", "best")
    with pytest.raises(ValueError):
        dl.rank_by()


def test_the_daily_top_pick_is_the_simulators_choice():
    """`strategies.dip.rank_pieces(by="score", topk=1)` on a day's orders and production's `rank_key` choose the same
    order — ties in the score or the RSI2 included, tickers that prefix one another included."""
    from src.backtest.strategies import dip as DIP
    rng = np.random.default_rng(7)
    names = ["T1", "T10", "T2", "AB", "ABC", "B"]
    day = "2026-10-16"
    for _ in range(400):
        tks = list(rng.choice(names, size=int(rng.integers(1, 6)), replace=False))
        rsi = {tk: float(rng.choice([0.5, 1.5, 4.0])) for tk in tks}
        sc = {tk: float(rng.choice([-1.0, 0.5, 2.0])) for tk in tks if rng.random() < 0.6}
        pieces = [{"oid": f"{tk}|{day}", "rank": rsi[tk], "tkn": tk} for tk in tks]
        sim = [p["oid"].split("|")[0] for p in DIP.rank_pieces(pieces, {f"{tk}|{day}": v for tk, v in sc.items()},
                                                                by="score", topk=1)]
        sig = [dict({"ticker": tk, "rsi2": rsi[tk]}, **({"filter_score": sc[tk]} if tk in sc else {})) for tk in tks]
        assert sim == [min(sig, key=lambda s: dl.rank_key(s, "score"))["ticker"]]


# ── the day's signals ────────────────────────────────────────────────────────

class _ByRsi:
    """A stand-in model scoring each name by its own RSI2 input (the store's, through D-1); keeps what it saw."""

    def __init__(self, by_rsi):
        self.by_rsi, self.seen = by_rsi, []

    def predict(self, X):
        X = np.asarray(X, float)
        self.seen.append(X)
        return np.array([self.by_rsi[r[F.index("rsi2")]] for r in X])


@pytest.fixture
def store(monkeypatch, tmp_path):
    from src.data import intraday_store
    monkeypatch.setattr(settings, "dip_long_dir", str(tmp_path / "dip_long"))
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    days = ss.sessions_before(DAY, 300)
    frames = {tk: _bars_of(_daily(days, seed=s)) for tk, s in (("AAA", 1), ("BBB", 2), ("CCC", 3))}
    monkeypatch.setattr(dl, "candidates", lambda d: (sorted(frames), "test"))
    monkeypatch.setattr(intraday_store, "load_deep_30m", lambda tk: frames.get(tk))
    monkeypatch.setattr(DF, "spy_inputs", lambda through: (dict(SPY_IN), "cache"))
    return {tk: dl.stats_at(dl.daily_bars(f), PREV) for tk, f in frames.items()}


def _model(monkeypatch, full, scores):
    m = _ByRsi({full[t]["rsi2"]: v for t, v in scores.items()})
    monkeypatch.setattr(DF, "load_model", lambda path=None: (m, META))
    return m


def test_the_day_keeps_only_its_best_scored_signal(monkeypatch, store):
    full = store
    _q1(monkeypatch)
    m = _model(monkeypatch, full, {"AAA": 0.4, "BBB": 3.0, "CCC": 1.2})     # every score under RP33's cut of 7
    out = dl.compute_signals(DAY)
    assert [s["ticker"] for s in out["signals"]] == ["BBB"] and out["signals"][0]["top_rank"] == 1
    assert [(s["ticker"], s["top_rank"]) for s in out["not_top"]] == [("CCC", 2), ("AAA", 3)]
    t = out["top"]
    assert (t["by"], t["top"], t["breadth"], t["scored"], t["error"], t["cut"]) == ("score", 1, 3, 3, None, 7.0)
    assert t["picked"] == [{"ticker": "BBB", "score": 3.0, "rsi2": full["BBB"]["rsi2"]}]
    assert "filter_keep" not in out["signals"][0] and "filter" not in out     # no cut: the RP33 filter is off
    assert len(m.seen) == 1 and (m.seen[0][:, F.index("breadth")] == 3).all()   # scored once, the day's count
    disk = json.loads(dl.signals_path(DAY).read_text(encoding="utf-8"))
    assert disk["top"]["picked"][0]["ticker"] == "BBB" and [s["ticker"] for s in disk["not_top"]] == ["CCC", "AAA"]


def test_with_the_rule_off_the_day_is_as_before(monkeypatch, store):
    _q1(monkeypatch, dip_long_daily_top=0)

    def never(*a, **k):
        raise AssertionError("scored with the daily top pick off")
    monkeypatch.setattr(DF, "score_signals", never)
    out = dl.compute_signals(DAY)
    assert len(out["signals"]) == 3 and "top" not in out and "not_top" not in out
    assert all("top_rank" not in s for s in out["signals"])


def test_a_model_that_cannot_be_read_takes_the_deepest_dip_and_raises_the_banner(monkeypatch, store):
    full = store
    _q1(monkeypatch)
    deepest = min(full, key=lambda t: (full[t]["rsi2"], t))

    def boom(path=None):
        raise FileNotFoundError("model.txt")
    monkeypatch.setattr(DF, "load_model", boom)
    out = dl.compute_signals(DAY)
    assert [s["ticker"] for s in out["signals"]] == [deepest] and len(out["not_top"]) == 2
    assert "model unavailable" in out["top"]["error"] and out["top"]["scored"] == 0
    monkeypatch.setattr(settings, "enable_dip_long", True)
    h = dl.health(at(DAY, 12, 0))
    assert any("could not score" in p and "deepest dip" in p for p in h["problems"])

    def crash(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(DF, "score_signals", crash)                           # an unexpected failure: the same
    out = dl.compute_signals(DAY)
    assert [s["ticker"] for s in out["signals"]] == [deepest] and "boom" in out["top"]["error"]


def test_the_banner_notes_the_pick_and_a_day_computed_before_the_rule(monkeypatch, tmp_path):
    _q1(monkeypatch)
    monkeypatch.setattr(settings, "enable_dip_long", True)
    monkeypatch.setattr(settings, "dip_long_dir", str(tmp_path / "dip_long"))
    noon = at(DAY, 12, 0)
    base = {"signals": [{"ticker": "AAA"}], "stale": [], "signal_session": PREV.isoformat()}
    dl._write_json(dl.signals_path(DAY), base)                                # computed before the rule
    h = dl.health(noon)
    assert h["problems"] == [] and any("before the daily top pick" in n for n in h["notes"])
    dl._write_json(dl.signals_path(DAY), dict(base, top={"error": None, "breadth": 3, "scored": 2,
                                                         "picked": [{"ticker": "AAA", "score": 1.25}]}))
    h = dl.health(noon)
    assert h["problems"] == [] and any("top pick AAA (score +1.25) of 3" in n and "1 unscored" in n
                                       for n in h["notes"])


def test_with_the_rp33_filter_on_the_pick_is_the_best_kept(monkeypatch, store):
    full = store
    _q1(monkeypatch)
    monkeypatch.setattr(settings, "enable_dip_long_filter", True)
    m = _model(monkeypatch, full, {"AAA": 9.0, "BBB": 3.0, "CCC": 8.0})      # cut 7: BBB filtered, AAA the best kept
    out = dl.compute_signals(DAY)
    assert [s["ticker"] for s in out["filtered"]] == ["BBB"]
    assert [s["ticker"] for s in out["signals"]] == ["AAA"] and [s["ticker"] for s in out["not_top"]] == ["CCC"]
    assert len(m.seen) == 1 and out["top"]["scored_by"] == "the RP33 filter"


# ── the orders ───────────────────────────────────────────────────────────────

@pytest.fixture
def q1_day(monkeypatch, tmp_path):
    for k, v in {"enable_dip_long": True, "dip_long_dir": str(tmp_path / "dip_long"), "dip_long_entry_mode": "limit",
                 "dip_long_limit_atr": 0.5, "dip_long_limit_sessions": 5,
                 "enable_sel_short": True, "enable_sel_short_vol": True, "enable_sel_short_account_sizing": True,
                 "enable_sel_short_ibkr_whatif": False, "enable_live_account": True,
                 "sel_short_traded_arms": "vol,etf,fall", "sel_short_account_initial": 10_000.0,
                 "sel_short_account_slices": 24, "sel_short_account_house_maint": 2.0,
                 "sel_short_account_house_init": 2.86}.items():
        monkeypatch.setattr(settings, k, v)
    _q1(monkeypatch)
    monkeypatch.setattr(dl, "_extend", lambda tks, through: None)
    monkeypatch.setattr(tracker, "_reference_close", lambda t: None)
    days: dict = {}
    monkeypatch.setattr(dl, "signals_for", lambda d: days.get(d, {"signals": []}))
    feed: dict = {}
    monkeypatch.setattr(dl, "session_bars", lambda tk, d: feed.get((tk, d), _bars(d, [])))
    return days, feed


def _tsig(tk, rsi, close, atr, score, rank):
    s = _lsig(tk, rsi, close, atr, score)
    s.pop("filter_keep")
    return dict(s, top_rank=rank)


def _day(top, rest=()):
    return {"day": DAY.isoformat(), "signal_session": PREV.isoformat(), "signals": [top], "not_top": list(rest),
            "top": {"by": "score", "top": 1, "breadth": 1 + len(rest), "scored": 1 + len(rest), "error": None}}


def test_only_the_top_pick_is_placed_the_rest_journaled_and_never_bought(q1_day, monkeypatch):
    days, feed = q1_day
    days[DAY] = _day(_tsig("AAA", 6.0, 100.0, 4.0, 2.5, 1), [_tsig("BBB", 1.0, 50.0, 2.0, 0.3, 2)])   # limits 98 / 49
    feed[("AAA", DAY)] = _bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    feed[("BBB", DAY)] = _bars(DAY, [(48.0, 48.5, 47.0, 48.0)])            # BBB opens under its limit: it would fill
    assert tracker.record_dip_long_trades(now=at(DAY, 9, 30)) == 1          # AAA placed; BBB neither placed nor bought
    o = {x["ticker"]: x for x in dl.load_orders(DAY)}
    assert o["AAA"]["status"] == "working" and o["AAA"]["dip_top"] == 1 and o["AAA"]["filter_score"] == 2.5
    assert o["BBB"]["status"] == "not_top" and o["BBB"]["top_rank"] == 2 and o["BBB"]["qty"] is None
    assert [e["what"] for e in o["BBB"]["events"]] == ["not_top"]
    assert o["AAA"]["qty"] == int((10_000.0 / 6) // 98.0)                    # the 1/6 slice at its limit
    feed[("AAA", DAY)] = _bars(DAY, [(101.0, 101.5, 97.5, 98.2)])           # a bar reaches AAA's limit (and BBB's)
    assert tracker.monitor_dip_long_orders(now=at(DAY, 10, 0)) == 1
    t = {x["ticker"]: x for x in tracker._load_trades() if x.get("entry_mechanism") == "dip_long"}
    assert set(t) == {"AAA"} and t["AAA"]["dip_top_rank"] == 1 and t["AAA"]["dip_account_slices"] == 6
    assert {x["ticker"]: x for x in dl.load_orders(DAY)}["BBB"]["status"] == "not_top"
    assert tracker.record_dip_long_trades(now=at(DAY, 10, 0)) == 0          # the next tick: nothing journaled twice
    assert [x["ticker"] for x in dl.load_orders(DAY)] == ["AAA", "BBB"]


def test_the_broker_never_rests_an_order_that_was_not_the_top_pick(q1_day, monkeypatch):
    days, feed = q1_day
    days[DAY] = _day(_tsig("AAA", 6.0, 100.0, 4.0, 2.5, 1), [_tsig("BBB", 1.0, 50.0, 2.0, 0.3, 2)])
    feed[("AAA", DAY)] = _bars(DAY, [(101.0, 101.5, 100.5, 101.0)])
    feed[("BBB", DAY)] = _bars(DAY, [(51.0, 51.2, 50.5, 51.0)])
    tracker.record_dip_long_trades(now=at(DAY, 9, 30))
    import src.broker.reconcile as rec
    from src.performance import market_calendar
    from tests.test_broker_reconcile import FakeBroker
    store = {"trades": tracker._load_trades()}
    monkeypatch.setattr(rec.repo, "load_trades", lambda: store["trades"])
    monkeypatch.setattr(rec.repo, "save_trades", lambda t: store.update(trades=t))
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    monkeypatch.setattr(settings, "broker_settle_seconds", 0)
    monkeypatch.setattr(settings, "enable_legacy_entries", False)
    monkeypatch.setattr(rec, "usd_per_unit", lambda ccy: 1.0)
    monkeypatch.setattr(market_calendar, "current_session", lambda now=None: "rth")
    monkeypatch.setattr(dl, "today_et", lambda now=None: DAY)

    class Resting(FakeBroker):
        def submit_order(self, req):
            res = super().submit_order(req)
            if getattr(req, "rest", False):
                res.filled_qty, res.status, res.avg_fill_price = 0, "Submitted", None
            return res
    b = Resting()
    rec.sync(broker=b)
    assert [(o.ticker, o.side) for o in b.orders] == [("AAA", "BUY")]


def test_an_order_of_a_day_before_the_rule_is_never_placed(q1_day):
    days, feed = q1_day
    d0 = ss.sessions_before(DAY, 1)[0]
    old = [dl.make_order(d0, _lsig("OLD", 2.0, 80.0, 2.0, 0.0), 1, PREV.isoformat()),
           dl.make_order(d0, _lsig("WRK", 3.0, 60.0, 2.0, 0.0), 2, PREV.isoformat())]
    old[1].update(status="working", qty=5, placed_at=at(d0, 9, 30).isoformat(), placed_session=d0.isoformat(),
                  ibkr_qty=5)
    dl.save_orders(d0, old)
    feed[("OLD", DAY)] = _bars(DAY, [(70.0, 70.5, 69.0, 70.0)])             # OLD opens under its limit
    assert tracker.record_dip_long_trades(now=at(DAY, 9, 30)) == 1          # WRK armed again; OLD never placed
    o = {x["ticker"]: x for x in dl.load_orders(d0)}
    assert o["OLD"]["status"] == "superseded" and o["OLD"]["events"][-1]["what"] == "superseded"
    assert o["WRK"]["status"] == "working" and DAY.isoformat() in o["WRK"]["session_armed_at"]
    assert not [x for x in tracker._load_trades() if x.get("ticker") == "OLD"]


def test_the_best_pick_takes_the_last_slot(q1_day):
    """Orders competing at one opening tick go best-ranked first (the engine's rank order): five working orders hold
    five of the six slots; of an earlier day's pending top pick (score 1.0) and today's (score 2.0), today's is placed,
    the other has no slot (tried again at the next open)."""
    days, feed = q1_day
    d0 = ss.sessions_before(DAY, 1)[0]
    held = []
    for k, tk in enumerate(("W1", "W2", "W3", "W4", "W5")):
        o = dl.make_order(d0, _lsig(tk, 5.0, 20.0, 1.0, 0.0), k + 2, PREV.isoformat())
        o.update(status="working", qty=1, placed_at=at(d0, 9, 30).isoformat(), placed_session=d0.isoformat(),
                 ibkr_qty=1)
        held.append(o)
    low = dl.make_order(d0, _lsig("LOW", 1.0, 40.0, 1.0, 0.0), 1, PREV.isoformat())
    low.update(dip_top=1, filter_score=1.0)
    dl.save_orders(d0, held + [low])
    days[DAY] = _day(_tsig("HIGH", 9.0, 30.0, 1.0, 2.0, 1))
    tracker.record_dip_long_trades(now=at(DAY, 9, 30))
    assert {x["ticker"]: x for x in dl.load_orders(DAY)}["HIGH"]["status"] == "working"
    lo = {x["ticker"]: x for x in dl.load_orders(d0)}["LOW"]
    assert lo["status"] == "pending" and lo["events"][-1]["what"] == "no_slot"


def test_fills_at_one_instant_go_to_the_best_pick_first(q1_day, monkeypatch):
    """Two working top picks touched in the same bar with room for one: the higher score buys (the engine's rank
    order); with the rule off the deeper dip would have."""
    from src.performance import live_account
    d0 = ss.sessions_before(DAY, 1)[0]
    a = dl.make_order(d0, _lsig("DEEP", 1.0, 50.0, 2.0, 0.0), 1, PREV.isoformat())
    b = dl.make_order(DAY, _lsig("BEST", 9.0, 50.0, 2.0, 0.0), 1, PREV.isoformat())
    for o, s in ((a, 0.5), (b, 2.0)):
        o.update(status="working", qty=10, placed_at=at(DAY, 9, 30).isoformat(), placed_session=DAY.isoformat(),
                 dip_top=1, filter_score=s)
    t_hit = datetime(2026, 10, 8, 14, 30, tzinfo=timezone.utc)

    def run():
        calls = {"n": 0}

        def cap(trades, price, now=None):
            calls["n"] += 1
            return (10 if calls["n"] == 1 else 0), {"equity": 10_000.0}
        monkeypatch.setattr(live_account, "fill_cap", cap)
        aa, bb = json.loads(json.dumps(a)), json.loads(json.dumps(b))
        tracker._fill_dip_orders([(t_hit, d0, aa, 49.0, "touch"), (t_hit, DAY, bb, 49.0, "touch")],
                                 [], t_hit, None, set())
        return aa["status"], bb["status"]
    assert run() == ("working", "filled")                                  # BEST (2.0) first; DEEP finds no room
    monkeypatch.setattr(settings, "dip_long_daily_top", 0)
    assert run() == ("filled", "working")                                  # the rule off: the deepest dip first


# ── the simulator follows production ─────────────────────────────────────────

def test_the_simulator_follows_production(monkeypatch):
    from src.backtest import mimic as M
    _q1(monkeypatch, sel_short_max_open_per_ticker=2)
    m = M.Mimic()
    assert (m.dip_rank, m.dip_topk, m.dip_slices, m.short_name_cap, m.short_name_scope) == ("score", 1, None, 2, "all")
    S = {s.name: s for s in M.live_strategies(m)}
    assert (S["dip"].slices, S["dip"].max_open) == (6.0, 6)
    assert all(S[a].group_cap == 2 and set(S[a].group_scope) == {"vol", "etf", "fall"} for a in ("vol", "etf", "fall"))
    o = M.Mimic.off()
    assert (o.dip_rank, o.dip_topk, o.short_name_cap, o.dip_slices) == ("rsi2", None, None, 10)
    _q1(monkeypatch, dip_long_daily_top=0, sel_short_max_open_per_ticker=0)
    m0 = M.Mimic()
    assert (m0.dip_rank, m0.dip_topk, m0.short_name_cap) == ("rsi2", None, None)
