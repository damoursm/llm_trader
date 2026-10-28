"""PREREG191's evening probe (src/signals/refusal_probe.py) and the in-session asking of the picks the entry step
settled before a what-if (tracker.ask_unasked_sel_short): the plan's order and cap, the guards (the slots, the spread
sweep, IBC's reconnects, the window), production's order and refusal parsing, and the extra accounts' arms."""
import json
import os
import time
from datetime import date, datetime, timezone

import pandas as pd

from config.settings import settings
from src.signals import refusal_probe as RP
from src.signals import sel_refusal as SR

D = date(2026, 10, 26)


def _root(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "sel_short_dir", str(tmp_path / "sel"))
    monkeypatch.setattr(settings, "account_name", "main")
    monkeypatch.setattr(settings, "account_state_root", "")
    root = tmp_path / "sel"
    for sub in ("picks", "scores_vol", "entries"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    RP.probes_dir().mkdir(parents=True, exist_ok=True)
    return root


def test_the_evening_plan_asks_the_queue_then_the_refusals_the_picks_the_top_names_and_the_uncertain(monkeypatch,
                                                                                                     tmp_path):
    root = _root(monkeypatch, tmp_path)
    monkeypatch.setattr(settings, "refusal_probe_max_names", 56)
    monkeypatch.setattr(settings, "refusal_probe_reask_days", 7)
    monkeypatch.setattr(settings, "refusal_probe_fresh_days", 30)
    monkeypatch.setattr(settings, "refusal_probe_watch_names", 1)
    RP.queue_path().write_text(json.dumps({"names": ["Q1", "Q2", "OLD"]}), encoding="utf-8")
    old = [{"ticker": "R1", "refused": "Small Cap", "at": "2026-10-10T17:00:00-04:00"},     # a refusal 16 days old
           {"ticker": "OLD", "init_rate": 2.9, "maint_rate": 2.0, "at": "2026-10-10T17:00:00-04:00"},  # answered
           {"ticker": "W1", "init_rate": 2.9, "maint_rate": 2.0, "at": "2026-10-10T17:00:00-04:00"},   # risky, accepted
           {"ticker": "W2", "init_rate": 2.9, "maint_rate": 2.0, "at": "2026-10-10T17:00:00-04:00"},   # less risky
           {"ticker": "R2", "refused": "close-only", "at": "2026-10-24T17:00:00-04:00"},   # too recent to re-ask
           {"ticker": "W3", "init_rate": 2.9, "maint_rate": 2.0, "at": "2026-10-24T17:00:00-04:00"},   # too recent
           {"ticker": "Q2", "error": "timeout", "at": "2026-10-24T17:00:00-04:00"}]        # no answer: still queued
    (RP.probes_dir() / "2026-10-10.jsonl").write_text("\n".join(json.dumps(r) for r in old[:4]), encoding="utf-8")
    (RP.probes_dir() / "2026-10-24.jsonl").write_text("\n".join(json.dumps(r) for r in old[4:]), encoding="utf-8")
    picks = [{"arm": "vol", "ticker": "P1", "fresh_pool": ["P2", "OLD"], "backups": [{"ticker": "P3"}]},
             {"arm": "model", "ticker": "M1", "fresh_pool": [], "backups": []},            # a shadow arm: not asked
             {"arm": "etf", "ticker": "SPYX", "fresh_pool": [], "backups": []}]            # a fund: never refused
    (root / "picks" / f"{D}.jsonl").write_text("\n".join(json.dumps(r) for r in picks), encoding="utf-8")
    names = [f"T{i:02d}" for i in range(52)]
    extra = ["W1", "W2", "W3"]                                 # in the universe, never in a bar's top 50
    pd.DataFrame({"bar": 0, "ticker": names + extra, "score": [100.0 - i for i in range(52)] + [1.0, 0.9, 0.8]}
                 ).to_pickle(root / "scores_vol" / f"{D}.pkl")
    monkeypatch.setattr("src.signals.sel_short.etf_names", lambda tks, model_names=None: [t for t in tks if t == "SPYX"])
    probs = {"T50": 0.5, "T51": 0.02, "W1": 0.3, "W2": 0.1, "W3": 0.4}
    monkeypatch.setattr(SR, "probabilities", lambda tks, d, prices=None: {t: probs.get(t, 0.0) for t in tks})
    plan = RP.plan(D, now=datetime(2026, 10, 26, 16, 30, tzinfo=RP.ET))
    assert [t for t, _ in plan[:7]] == ["Q1", "Q2", "R1", "W1", "P1", "P2", "P3"]
    assert [s for _, s in plan[:7]] == ["queue", "queue", "reask", "watch", "pick", "pick", "pick"]
    assert [t for t, s in plan if s == "top50"] == names[:50]
    assert plan[-1] == ("T50", "uncertain") and len(plan) == 2 + 56         # the queue is not counted in the cap
    assert not {"OLD", "R2", "M1", "SPYX", "T51", "W2", "W3"} & {t for t, _ in plan}   # W2: the watch's 1; W3: too recent


def test_the_probe_keeps_off_the_slots_the_sweep_and_stops_on_ibcs_reconnects(monkeypatch, tmp_path):
    at = lambda h, m, s=0: datetime(2026, 10, 26, h, m, s, tzinfo=RP.ET)     # noqa: E731
    assert RP.near_slot(at(16, 29)) and RP.near_slot(at(16, 30)) and RP.near_slot(at(16, 33))
    assert not RP.near_slot(at(16, 33, 30)) and not RP.near_slot(at(16, 45)) and not RP.near_slot(at(19, 45))
    assert RP.near_slot(at(19, 50, 30)) and RP.near_slot(at(22, 59, 30))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cache" / "eod" / "state").mkdir(parents=True)
    st = tmp_path / "cache" / "eod" / "state" / f"{D}.json"
    st.write_text(json.dumps({"steps": {"spread_sweep": {"status": "running"}}}), encoding="utf-8")
    assert RP.sweep_running(D)
    st.write_text(json.dumps({"steps": {"spread_sweep": {"status": "ok"}}}), encoding="utf-8")
    assert not RP.sweep_running(D) and not RP.sweep_running(date(2026, 10, 27))
    logs = tmp_path / "ibc"
    logs.mkdir()
    monkeypatch.setattr(settings, "ibc_log_dir", str(logs))
    f = logs / "IBC-3.23.0_GATEWAY-1047_.txt"
    f.write_text("2026-10-26 16:20:00:000 IBC: Login has completed\n", encoding="utf-8")
    mark = RP.ibc_mark()
    assert RP.ibc_events_since(mark) == 0
    with open(f, "a", encoding="utf-8") as fh:
        fh.write("IBC: detected frame entitled: Connecting to server...; event=Opened\n")
    assert RP.ibc_events_since(mark) == 1
    g = logs / "IBC-3.23.0_GATEWAY-1047_MONDAY.txt"                            # a newer log: the gateway restarted
    g.write_text("x", encoding="utf-8")
    later = time.time() + 60
    os.utime(g, (later, later))
    assert RP.ibc_events_since((str(f), f.stat().st_size)) == 1


class _Event:
    def __init__(self):
        self.fns = []

    def __iadd__(self, fn):
        self.fns.append(fn)
        return self

    def __isub__(self, fn):
        self.fns.remove(fn)
        return self


class _IB:
    RequestTimeout = 0.0

    def __init__(self, refuse=None, ic=None, mc=None):
        self.errorEvent, self.refuse, self.ic, self.mc, self.orders = _Event(), refuse, ic, mc, []

    def qualifyContracts(self, c):
        return [c]

    def whatIfOrder(self, c, o):
        self.orders.append((c.symbol, o.action, o.totalQuantity, o.lmtPrice, o.tif))
        if self.refuse:
            for fn in list(self.errorEvent.fns):
                fn(7, 201, self.refuse, c)
            return None
        return type("S", (), {"initMarginChange": self.ic, "maintMarginChange": self.mc, "warningText": ""})()

    def sleep(self, s):
        pass


def test_the_ask_is_productions_order_and_refusal_parsing():
    ib = _IB(refuse="Order rejected - reason:Ineligibility reasons:<br>No Opening Trades: Small Cap, Subject to "
                    "Compliance Restriction")
    a = RP.ask(ib, "NCT", 10.0, 1.0, "USD")
    assert a["refused"] == "No Opening Trades: Small Cap, Subject to Compliance Restriction"
    assert ib.orders == [("NCT", "SELL", 50, 10.0, "DAY")] and not ib.errorEvent.fns      # the handler is removed
    ib = _IB(ic=1430.0, mc=1000.0)                                         # the paper account's own currency
    a = RP.ask(ib, "BRK-B", 10.0, 0.7, "CAD")
    assert a["init_rate"] == round(1430 * 0.7 / 500, 4) and a["maint_rate"] == round(1000 * 0.7 / 500, 4)
    assert ib.orders[0][0] == "BRK B" and "refused" not in a
    assert "error" in RP.ask(_IB(ic=None, mc=None), "X", 10.0, 1.0, "USD")


def test_the_probe_runs_only_on_paper_in_its_evening(monkeypatch):
    monkeypatch.setattr(settings, "broker_mode", "dry_run")
    assert "broker_mode" in RP.run(D)["skipped"]
    monkeypatch.setattr(settings, "broker_mode", "ibkr_paper")
    assert "evening window" in RP.run(date(2020, 1, 2))["skipped"]
    from src.scheduler import eod
    monkeypatch.setattr(settings, "enable_eod_refusal_probe", True)
    assert eod._enabled("refusal_probe")
    assert eod._refusal_probe(date(2020, 1, 2)) == ({"launched": False}, "not launched: outside 2020-01-02's evening "
                                                                          "window")
    monkeypatch.setattr(settings, "broker_mode", "dry_run")
    assert not eod._enabled("refusal_probe")


def test_the_extra_accounts_short_arms(monkeypatch):
    from src.accounts import profiles as PR
    monkeypatch.setattr(settings, "enable_extra_accounts", False)
    assert PR.short_arms_traded() == []
    monkeypatch.setattr(settings, "enable_extra_accounts", True)
    monkeypatch.setattr(settings, "extra_accounts", ["second"])
    monkeypatch.setattr(settings, "second_account_books", ["dip", "fall", "crash"])
    assert PR.short_arms_traded() == ["fall"]


def test_the_main_account_asks_ibkr_about_a_pick_it_settled_before_its_whatif(monkeypatch, tmp_path):
    """PREREG191's change 3: a pick settled for a reason of the main account's own is asked after the live path when
    an account trades its arm and the name holds no answer this session — once a session — and the extra accounts read
    the answer (`sim_account.main_account_answer`)."""
    from src.performance import sim_account, tracker
    from src.signals import sel_short
    _root(monkeypatch, tmp_path)
    for k, v in {"enable_sel_short": True, "enable_sel_short_account_sizing": True,
                 "enable_sel_short_ibkr_whatif": True, "enable_sel_short_whatif_unasked": True,
                 "sel_short_whatif_unasked_max": 3, "sel_short_traded_arms": "vol,etf",
                 "enable_extra_accounts": True, "extra_accounts": ["second"],
                 "second_account_books": ["dip", "fall", "crash"]}.items():
        monkeypatch.setattr(settings, k, v)
    now = datetime.now(timezone.utc)
    day = now.astimezone(tracker.ET).date()
    t0, t1 = (now.replace(microsecond=0).isoformat(), now.replace(microsecond=0).isoformat())

    def line(tk, arm, outcome, at=t0, margin=None):
        return {"key": f"{day}|3|{tk}|{arm}", "day": str(day), "bar_of_day": 3, "arm": arm, "ticker": tk,
                "outcome": outcome, "at": at, "price": None, "target": 9.0, "pick_close": 11.0, "ssr": False,
                "ibkr_margin": margin}
    lines = [line("AAA", "vol", "already_open"),
             line("BBB", "model", "shadow_arm"),                                  # no account trades the model arm
             line("EEE", "fall", "shadow_arm", at=t1),                            # the second account trades it
             line("CCC", "fall", "traded_today"),
             line("CCC", "fall", "opened", margin={"init_rate": 2.9, "maint_rate": 2.0}),   # answered this session
             line("DDD", "vol", "no_borrow")]                                     # every account's reason
    p = sel_short.main_entries_path(day)
    p.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")       # as the journal writes
    asked = []
    monkeypatch.setattr(tracker, "_fetch_price", lambda tk: 12.5)
    monkeypatch.setattr(sim_account, "ibkr_margin",
                        lambda tk, px: asked.append(tk) or {"refused": "No Opening Trades: Small Cap"})
    assert tracker.ask_unasked_sel_short(now) == 2 and sorted(asked) == ["AAA", "EEE"]
    assert sim_account.main_account_answer("EEE", now)["refused"].startswith("No Opening Trades")
    new = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()][len(lines):]
    assert {r["outcome"] for r in new} == {"whatif_only"} and {r["price"] for r in new} == {12.5}
    assert tracker.ask_unasked_sel_short(now) == 0                               # once a session, whatever the answer
    monkeypatch.setattr(settings, "account_state_root", str(tmp_path / "acct"))  # an extra account never asks
    monkeypatch.setattr(settings, "account_name", "second")
    p.write_text(json.dumps(line("FFF", "vol", "already_open")) + "\n", encoding="utf-8")
    assert tracker.ask_unasked_sel_short(now) == 0
