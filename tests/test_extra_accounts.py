"""THE EXTRA ACCOUNTS (src/accounts/; from 2026-10-23: the second account — its own ledger, CALCULATED from the main
account: the user, "There's no second IBKR account creation. The second account will be calculated from the same
account."): the main account's paths never move, an extra account's state lives under its own root, its process
refuses the main account's ledger and any IBKR session, its profile names its books and a dry-run broker and holds no
secret, IBKR's what-if answers are the main account's, a dry run's empty positions are never read as buy-ins, the
launch is off by default and never doubles a running tick, and the summary reaches the main email as a note."""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from config.settings import settings
from src.accounts import launch as L
from src.accounts import paths as PA
from src.accounts import profiles as PR
from src.accounts import run as RUN
from src.accounts import summary as SUM


def test_the_main_account_keeps_every_path(monkeypatch):
    monkeypatch.setattr(settings, "account_name", "main")
    monkeypatch.setattr(settings, "account_state_root", "")
    assert PA.is_main() and PA.name() == "main"
    for p in ("cache/ml/dip_long/orders", Path("cache") / "live_account" / "margin_state.json", "busy.lock"):
        assert PA.state(p) == Path(p)


def test_an_extra_account_moves_its_state_under_its_root(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "account_name", "second")
    monkeypatch.setattr(settings, "account_state_root", str(tmp_path / "acct"))
    assert not PA.is_main() and PA.name() == "second"
    assert PA.state("cache/ml/dip_long/orders") == tmp_path / "acct" / "cache" / "ml" / "dip_long" / "orders"
    inside = PA.state(tmp_path / "x" / "y.json")                  # an absolute path: under the root, by its parts
    assert str(inside).startswith(str(tmp_path / "acct")) and inside.name == "y.json"


def test_the_runner_refuses_the_main_account_and_its_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "account_name", "main")
    monkeypatch.setattr(settings, "account_state_root", "")
    with pytest.raises(SystemExit):
        RUN.guard("main")
    with pytest.raises(SystemExit):                               # the environment does not name the account
        RUN.guard("second")
    monkeypatch.setattr(settings, "account_name", "second")
    with pytest.raises(SystemExit):                               # no state root of its own
        RUN.guard("second")
    monkeypatch.setattr(settings, "account_state_root", str(tmp_path / "acct"))
    monkeypatch.setattr(settings, "db_path", RUN.MAIN_DB)
    with pytest.raises(SystemExit):                               # the main account's ledger
        RUN.guard("second")
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "second.db"))
    for mode in ("ibkr_paper", "ibkr_live"):                      # never an IBKR session of its own
        monkeypatch.setattr(settings, "broker_mode", mode)
        with pytest.raises(SystemExit):
            RUN.guard("second")
    monkeypatch.setattr(settings, "broker_mode", "dry_run")
    RUN.guard("second")                                           # its own name, root, ledger, a dry run: runs


def test_the_second_profile_names_its_books_ledger_and_a_dry_run_and_no_secret(monkeypatch):
    monkeypatch.setattr(settings, "second_account_books", ["dip", "fall", "crash"])
    monkeypatch.setattr(settings, "second_account_fresh_start", "2026-10-26T09:30:00-04:00")
    env = PR.env_for("second")
    assert env["ACCOUNT_NAME"] == "second" and env["ACCOUNT_STATE_ROOT"] == settings.second_account_state_root
    assert env["DB_PATH"] == settings.second_account_db_path and env["DB_PATH"] != RUN.MAIN_DB
    assert env["SEL_SHORT_TRADED_ARMS"] == "fall" and env["ENABLE_SEL_SHORT"] == "true"
    assert env["ENABLE_DIP_LONG"] == "true" and env["ENABLE_CRASH_LONG"] == "true" and env["ENABLE_CAP_LONG"] == "false"
    assert env["ENABLE_ETF_DM"] == "false" and env["ENABLE_SEL_SHORT_NEWS_HOLD_SHADOW"] == "false"
    assert env["LIVE_ACCOUNT_FRESH_START"] == "2026-10-26T09:30:00-04:00"
    assert env["BROKER_MODE"] == "dry_run" and env["BROKER_MODE"] in RUN.SAFE_BROKER_MODES   # never a gateway
    assert env["BROKER_GATEWAY_AUTO_RESTART"] == env["BROKER_GATEWAY_RELOGIN_RESTART"] == "false"
    assert float(env["BROKER_PAPER_EQUITY"]) >= 20 * settings.second_account_initial   # the cap never binds first
    assert not [k for k in env if k.startswith(("IBKR_", "IBC_"))]       # the main account's gateway stays its own
    banned = ("KEY", "TOKEN", "PASSWORD", "SECRET", "LOGINID", "LOGIN_ID", "USERID")
    assert not [k for k in env if any(b in k for b in banned)]
    monkeypatch.setattr(settings, "second_account_books", ["dip", "fall"])
    two = PR.env_for("second")
    assert two["ENABLE_CRASH_LONG"] == "false" and two["SEL_SHORT_TRADED_ARMS"] == "fall"
    monkeypatch.setattr(settings, "second_account_books", ["dip", "crash"])
    longs = PR.env_for("second")                     # no short arm: never the empty list, which means every arm
    assert longs["SEL_SHORT_TRADED_ARMS"] == PR.NO_ARM and longs["ENABLE_SEL_SHORT"] == "false"
    monkeypatch.setattr(settings, "second_account_books", ["dip", "nope"])
    with pytest.raises(ValueError):
        PR.env_for("second")
    with pytest.raises(ValueError):
        PR.env_for("third")


def test_the_profile_applies_in_a_fresh_settings_object(monkeypatch):
    """pydantic reads the profile's environment over .env: the account's process sees its own values."""
    from config.settings import Settings
    monkeypatch.setattr(settings, "second_account_fresh_start", "2026-10-26T09:30:00-04:00")
    for k, v in PR.env_for("second").items():
        monkeypatch.setenv(k, v)
    s = Settings()
    assert s.account_name == "second" and s.account_state_root == settings.second_account_state_root
    assert s.db_path == settings.second_account_db_path and s.sel_short_traded_arms == "fall"
    assert s.enable_cap_long is False and s.enable_crash_long is True and s.broker_mode == "dry_run"
    assert s.broker_gateway_auto_restart is False and s.broker_gateway_relogin_restart is False
    assert s.live_account_fresh_start == "2026-10-26T09:30:00-04:00"


def test_a_profile_without_short_arms_trades_no_arm(monkeypatch):
    from src.signals import sel_short as SS
    monkeypatch.setattr(settings, "second_account_books", ["dip", "crash"])
    monkeypatch.setattr(settings, "sel_short_traded_arms", PR.env_for("second")["SEL_SHORT_TRADED_ARMS"])
    arms = SS.traded_arms()
    assert arms is not None and not (arms & {"vol", "etf", "fall", "model", "vol2"})


def test_the_launch_is_off_by_default_and_never_doubles_a_running_tick(monkeypatch, tmp_path):
    calls = []

    class FakeProc:
        pid = 4242

        def __init__(self, *a, **k):
            calls.append(k.get("env", {}).get("ACCOUNT_NAME"))

        def poll(self):
            return None                                           # still running

    monkeypatch.setattr(L.subprocess, "Popen", FakeProc)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "account_name", "main")
    monkeypatch.setattr(settings, "account_state_root", "")
    monkeypatch.setattr(settings, "second_account_state_root", str(tmp_path / "acct"))
    monkeypatch.setattr(settings, "second_account_fresh_start", "2026-10-26T09:30:00-04:00")
    L._ACTIVE.clear()
    monkeypatch.setattr(settings, "enable_extra_accounts", False)
    assert L.launch_all() == [] and calls == []
    monkeypatch.setattr(settings, "enable_extra_accounts", True)
    monkeypatch.setattr(settings, "extra_accounts", ["second"])
    assert L.launch_all() == ["second"] and calls == ["second"]
    assert L.launch_all() == [] and calls == ["second"]           # the previous tick still runs
    L._ACTIVE.clear()
    (tmp_path / "acct").mkdir(parents=True, exist_ok=True)
    (tmp_path / "acct" / "busy.lock").write_text("1", encoding="utf-8")
    assert L.launch_all() == []                                   # another process holds its lock
    old = (datetime.now() - timedelta(seconds=settings.extra_account_timeout_seconds + 60)).timestamp()
    os.utime(tmp_path / "acct" / "busy.lock", (old, old))
    assert L.launch_all() == ["second"]                           # a stale lock does not block
    monkeypatch.setattr(settings, "account_state_root", str(tmp_path / "x"))
    L._ACTIVE.clear()
    assert L.launch_all() == []                                   # an extra account never launches others


def test_no_fresh_start_no_launch(monkeypatch, tmp_path):
    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: pytest.fail("launched without a fresh start"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "account_name", "main")
    monkeypatch.setattr(settings, "account_state_root", "")
    monkeypatch.setattr(settings, "enable_extra_accounts", True)
    monkeypatch.setattr(settings, "extra_accounts", ["second"])
    monkeypatch.setattr(settings, "second_account_state_root", str(tmp_path / "acct"))
    monkeypatch.setattr(settings, "second_account_fresh_start", "")
    L._ACTIVE.clear()
    assert L.launch_all() == []


def test_the_summary_reaches_the_main_email_as_a_note(monkeypatch, tmp_path):
    root = tmp_path / "acct"
    root.mkdir()
    monkeypatch.setattr(settings, "second_account_state_root", str(root))
    monkeypatch.setattr(settings, "enable_extra_accounts", False)
    assert SUM.health() == {"problems": [], "notes": []}
    monkeypatch.setattr(settings, "enable_extra_accounts", True)
    monkeypatch.setattr(settings, "extra_accounts", ["second"])
    assert SUM.health()["notes"] == ["second account: no summary yet"]
    rec = {"at": datetime.now(timezone.utc).isoformat(), "line": "second account $50,250 (+0.50% since its start): "
           "1 open (META), 0 closed today; broker dry_run"}
    (root / "summary.json").write_text(json.dumps(rec), encoding="utf-8")
    h = SUM.health()
    assert h["problems"] == [] and h["notes"] == [rec["line"]]
    rec["at"] = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
    (root / "summary.json").write_text(json.dumps(rec), encoding="utf-8")
    assert SUM.health()["notes"][0].endswith("(last tick 8 h ago)")


def _journal(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_an_extra_account_takes_the_main_accounts_what_if_answer(monkeypatch, tmp_path):
    """Calculated from the same account: IBKR's answer to the MAIN account for the name this ET session (the latest;
    a turn entry journaled under its pick's earlier day too), never a what-if of its own."""
    from src.performance import sim_account as SA
    monkeypatch.setattr(settings, "sel_short_dir", str(tmp_path / "sel"))
    monkeypatch.setattr(settings, "account_name", "second")
    monkeypatch.setattr(settings, "account_state_root", str(tmp_path / "acct"))
    now = datetime(2026, 10, 27, 15, 0, tzinfo=timezone.utc)            # 11:00 ET
    refusal = {"qty": 42, "price": 11.7, "refused": "Margin concern/risk management: closing-only status"}
    priced = {"qty": 40, "price": 12.0, "init_rate": 2.9, "maint_rate": 2.0}
    _journal(tmp_path / "sel" / "entries" / "2026-10-26.jsonl",            # yesterday's answer: not today's
             [{"ticker": "PMI", "at": "2026-10-26T15:00:00+00:00", "ibkr_margin": priced}])
    _journal(tmp_path / "sel" / "entries" / "2026-10-22.jsonl",            # a turn entry: its pick's day
             [{"ticker": "PMI", "at": "2026-10-27T14:00:00+00:00", "ibkr_margin": priced},
              {"ticker": "PMI", "at": "2026-10-27T14:30:00+00:00", "ibkr_margin": refusal},
              {"ticker": "AEHL", "at": "2026-10-27T14:40:00+00:00", "ibkr_margin": None}])
    monkeypatch.setattr("src.broker.get_broker", lambda *a, **k: pytest.fail("an extra account asked IBKR"))
    got = SA.main_account_answer("PMI", now=now)
    assert got["refused"] == refusal["refused"] and got["source"] == "main_account"
    assert got["asked_at"].startswith("2026-10-27T14:30")
    assert SA.main_account_answer("AEHL", now=now) is None               # the main account got no answer
    assert SA.main_account_answer("PMI", now=now - timedelta(days=2)) is None    # none that session
    real = SA.main_account_answer
    monkeypatch.setattr(SA, "main_account_answer", lambda tk, now=None: real(tk, now=datetime(2026, 10, 27, 15, 0,
                                                                                            tzinfo=timezone.utc)))
    assert SA.ibkr_margin("PMI", 11.7)["source"] == "main_account"
    monkeypatch.setattr(settings, "account_name", "main")                # the main account still asks its broker
    monkeypatch.setattr(settings, "account_state_root", "")

    class B:
        def what_if_short(self, tk, qty, px):
            return {"asked": tk, "qty": qty}
    monkeypatch.setattr("src.broker.get_broker", lambda *a, **k: B())
    assert SA.ibkr_margin("PMI", 10.0) == {"asked": "PMI", "qty": 50}


def test_a_dry_runs_empty_positions_are_never_buy_ins():
    from src.broker import reconcile as R
    from src.broker.dryrun import DryRunBroker

    class Live:
        name = "ibkr_paper"
    assert R._holds_positions(DryRunBroker()) is False and R._holds_positions(Live()) is True
