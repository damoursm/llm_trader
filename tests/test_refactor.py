"""Automatic refactor: change detection and repair ordering.

Two properties carry the whole design.

**Discrimination.** The detector must fire on a change that can move output and
stay silent on one that cannot. Too sensitive and every reformat triggers a full
regeneration (or, worse, masks real history); too blunt and stale values survive
a genuine scorer fix — the failure that made `money_flow` judge its fixed
implementation on its broken one's record.

**Ordering.** Weights are fitted ON the data and the derived layer depends on
both, so the repair is data -> epochs -> weights -> derived. Reversing any pair
yields a database that looks refreshed and is internally inconsistent, which is
strictly worse than one that is visibly stale.
"""

from __future__ import annotations

import pytest

from src.analysis import code_version as cv
from src.analysis import refactor as rf


# ── discrimination ────────────────────────────────────────────────────────────

def test_every_stored_method_is_visible_to_the_detector():
    """An unmapped method opts out of automatic refactor SILENTLY — the exact
    'looks covered, isn't' shape this machinery exists to remove."""
    assert cv.unmapped_methods() == [], (
        "these method columns have no source mapping in METHOD_SOURCES, so a "
        "change to their scorer would never be detected")


def test_fingerprint_is_stable_across_calls():
    assert cv.fingerprint("tech") == cv.fingerprint("tech")


def test_fingerprint_ignores_comments_docstrings_and_blank_lines(tmp_path, monkeypatch):
    """A reformat must not trigger a database refactor."""
    import ast
    import hashlib

    def h(src: str) -> str:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Module, ast.FunctionDef,
                                     ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                node.body = body[1:] or [ast.Pass()]
        return hashlib.sha256(
            ast.dump(tree, annotate_fields=False,
                     include_attributes=False).encode()).hexdigest()[:16]

    plain = 'def f(x):\n    return x * 2\n'
    cosmetic = ('"""Module docstring."""\n\n'
                '# a comment\n'
                'def f(x):\n'
                '    """Function docstring."""\n\n'
                '    return x * 2\n')
    assert h(plain) == h(cosmetic), "cosmetic edit changed the fingerprint"

    semantic = 'def f(x):\n    return x * 3\n'
    assert h(plain) != h(semantic), "a changed constant must change the fingerprint"


def test_unknown_name_yields_no_verdict():
    """None means 'cannot judge', and must never be read as 'unchanged'."""
    assert cv.fingerprint("not_a_method") is None


# ── classification ────────────────────────────────────────────────────────────

def _changes(**names):
    return {n: {"old": "a", "new": "b", "is_new": False} for n in names}


def test_replayable_change_is_regenerated_not_masked():
    """Masking a method the replay can rebuild would throw away history for no
    reason — regeneration is strictly better wherever it is possible."""
    p = rf.plan(_changes(money_flow=1))
    assert p["regenerate"] == ["money_flow"]
    assert p["mask_candidates"] == []


def test_non_replayable_change_is_a_mask_CANDIDATE_only():
    """It cannot be regenerated, so the only remedy withholds real history. The
    detector cannot tell cosmetic from categorical, so this is reported rather
    than applied."""
    p = rf.plan(_changes(news=1))
    assert p["mask_candidates"] == ["news"]
    assert p["regenerate"] == []


def test_first_seen_is_not_treated_as_a_change():
    """On a fresh database every name is unseen. Treating that as '42 scorers
    changed' would trigger a pointless full refactor on first run."""
    ch = {"tech": {"old": None, "new": "x", "is_new": True}}
    p = rf.plan(ch)
    assert p["changed"] == []
    assert p["first_seen"] == ["tech"]
    assert p["rewalk_weights"] is False


def test_any_data_change_invalidates_the_weights():
    """Weights are FITTED on the data, so stale scores mean stale weights."""
    assert rf.plan(_changes(money_flow=1))["rewalk_weights"] is True
    assert rf.plan(_changes(news=1))["rewalk_weights"] is True


def test_a_derived_formula_change_reruns_the_backtest_but_not_the_weights():
    """Changing the confidence formula does not touch the method scores the
    weights were fitted on, so re-walking them would be wasted work."""
    p = rf.plan(_changes(confidence=1))
    assert p["rerun_backtest"] is True
    assert p["rewalk_weights"] is False
    assert p["derived_changed"] == ["confidence"]


# ── ordering and failure handling ─────────────────────────────────────────────

def test_repair_runs_in_dependency_order(monkeypatch):
    """data -> weights -> derived. A backtest run before the weights were
    re-walked would score history under a calibration that predates the data it
    scores, and nothing about the result would look wrong."""
    order = []
    monkeypatch.setattr("src.analysis.replay.materialize",
                        lambda **k: order.append("replay") or 1)
    monkeypatch.setattr("src.analysis.walkforward.materialize",
                        lambda **k: order.append("walkforward") or 1)
    monkeypatch.setattr("src.analysis.backtest.materialize",
                        lambda **k: order.append("backtest") or 1)
    monkeypatch.setattr(cv, "record", lambda names=None: len(names or []))
    monkeypatch.setattr(rf, "_audit", lambda *a, **k: None)

    rf.run_refactor(apply=True, changes=_changes(money_flow=1))
    assert order == ["replay", "walkforward", "backtest"]


def test_a_failed_step_does_not_advance_the_fingerprints(monkeypatch):
    """Recording fingerprints after a partial failure marks the database as
    consistent with code it was never refactored against — the next run sees no
    changes and the staleness becomes permanent AND invisible."""
    recorded = []
    monkeypatch.setattr("src.analysis.replay.materialize",
                        lambda **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    monkeypatch.setattr("src.analysis.walkforward.materialize", lambda **k: 1)
    monkeypatch.setattr("src.analysis.backtest.materialize", lambda **k: 1)
    monkeypatch.setattr(cv, "record", lambda names=None: recorded.append(names) or 0)
    monkeypatch.setattr(rf, "_audit", lambda *a, **k: None)

    res = rf.run_refactor(apply=True, changes=_changes(money_flow=1))
    assert res["ok"] is False
    assert recorded == [], "fingerprints advanced despite a failed step"
    assert any(s["step"] == "record" and s["status"] == "skipped"
               for s in res["steps"])


def test_check_mode_writes_nothing(monkeypatch):
    called = []
    for path in ("src.analysis.replay.materialize",
                 "src.analysis.walkforward.materialize",
                 "src.analysis.backtest.materialize"):
        monkeypatch.setattr(path, lambda **k: called.append(1) or 1)
    rf.run_refactor(apply=False, changes=_changes(money_flow=1))
    assert called == []


def test_auto_epoch_is_off_by_default():
    """Masking withholds real history and the detector cannot tell a cosmetic
    edit from a categorical one, so it must be opted into."""
    from config.settings import settings
    assert settings.refactor_auto_epoch is False
    assert rf.auto_epochs() == {}


def test_auto_epoch_never_masks_a_regenerated_method(monkeypatch):
    """A replayable method's history has been REWRITTEN by the current scorer,
    so masking it would discard the very work the refactor just did."""
    from config.settings import settings
    monkeypatch.setattr(settings, "refactor_auto_epoch", True)
    monkeypatch.setattr(cv, "load_stored", lambda: {
        "money_flow": {"fingerprint": "x", "first_seen_at": "2026-07-01T00:00:00"},
        "news": {"fingerprint": "y", "first_seen_at": "2026-07-01T00:00:00"}})
    got = rf.auto_epochs()
    assert "money_flow" not in got, "a regenerated method must not be masked"
    assert "news" in got


# ── nightly scheduling ────────────────────────────────────────────────────────

def test_nightly_rescore_fires_once_per_date_at_the_configured_time():
    from datetime import datetime, time
    import src.scheduler.runner as runner

    at = time(2, 0)
    assert not runner._should_run_nightly_rescore(datetime(2026, 7, 29, 1, 59), None, at)
    assert runner._should_run_nightly_rescore(datetime(2026, 7, 29, 2, 0), None, at)
    assert runner._should_run_nightly_rescore(datetime(2026, 7, 29, 5, 0), None, at)
    # Already ran for this date.
    assert not runner._should_run_nightly_rescore(
        datetime(2026, 7, 29, 3, 30), datetime(2026, 7, 29).date(), at)


def test_nightly_rescore_runs_on_non_market_days():
    """It repairs stored history, which is just as stale on a Saturday — and a
    weekend night is the quietest window it will ever get."""
    from datetime import datetime, time
    import src.scheduler.runner as runner
    saturday = datetime(2026, 8, 1, 2, 0)
    assert runner._should_run_nightly_rescore(saturday, None, time(2, 0))


def test_nightly_rescore_respects_the_master_flag(monkeypatch):
    from datetime import datetime, time
    from config.settings import settings
    import src.scheduler.runner as runner
    monkeypatch.setattr(settings, "enable_auto_refactor", False)
    assert not runner._should_run_nightly_rescore(datetime(2026, 7, 29, 2, 0),
                                                  None, time(2, 0))


def test_no_change_means_no_thread_is_started(monkeypatch):
    """Detection is sub-second; the ~40-minute path must only start when an
    implementation actually changed."""
    import src.scheduler.runner as runner
    monkeypatch.setattr("src.analysis.refactor.plan",
                        lambda: {"changed": [], "first_seen": [],
                                 "mask_candidates": [], "regenerate": [],
                                 "derived_changed": [], "unmapped": [],
                                 "rewalk_weights": False, "rerun_backtest": False})
    started = runner._maybe_start_nightly_rescore()
    assert started is False


def test_a_detected_change_starts_a_BACKGROUND_thread(monkeypatch):
    """The scheduler loop is single-threaded and 02:00 ET is a live overnight
    tick slot, so an inline rescore would block the 02:00/03:00/03:30 ticks —
    real missed trading to repair history that is in no hurry."""
    import threading
    import src.scheduler.runner as runner

    monkeypatch.setattr("src.analysis.refactor.plan",
                        lambda: {"changed": ["money_flow"], "first_seen": [],
                                 "mask_candidates": [], "regenerate": ["money_flow"],
                                 "derived_changed": [], "unmapped": [],
                                 "rewalk_weights": True, "rerun_backtest": True})
    done = threading.Event()
    monkeypatch.setattr("src.analysis.refactor.run_refactor",
                        lambda **k: (done.set(), {"ok": True, "steps": [],
                                                  "plan": {"changed": ["money_flow"]}})[1])
    caller = threading.current_thread()
    assert runner._maybe_start_nightly_rescore() is True
    assert runner._RESCORE_THREAD is not caller
    assert done.wait(timeout=10), "background rescore never ran"
    runner._RESCORE_THREAD.join(timeout=10)


def test_a_second_rescore_does_not_start_while_one_is_running(monkeypatch):
    """A ~40-minute job overlapping itself would double every DB write."""
    import threading
    import src.scheduler.runner as runner

    release = threading.Event()
    monkeypatch.setattr("src.analysis.refactor.plan",
                        lambda: {"changed": ["money_flow"], "first_seen": [],
                                 "mask_candidates": [], "regenerate": ["money_flow"],
                                 "derived_changed": [], "unmapped": [],
                                 "rewalk_weights": True, "rerun_backtest": True})
    monkeypatch.setattr("src.analysis.refactor.run_refactor",
                        lambda **k: (release.wait(timeout=10),
                                     {"ok": True, "steps": [],
                                      "plan": {"changed": []}})[1])
    try:
        assert runner._maybe_start_nightly_rescore() is True
        assert runner._maybe_start_nightly_rescore() is False, "started a second run"
    finally:
        release.set()
        if runner._RESCORE_THREAD:
            runner._RESCORE_THREAD.join(timeout=10)


def test_eod_maintenance_no_longer_hosts_the_refactor():
    """It must not move back inline — that is what would block the ticks."""
    import inspect
    import src.scheduler.runner as runner
    src = inspect.getsource(runner._run_eod_maintenance)
    assert "run_refactor" not in src


# ── weekly ML retrain scheduling (2026-08-12: once a week, Saturday morning) ──

def test_weekly_ml_train_fires_only_on_the_configured_weekday(monkeypatch):
    from datetime import datetime, time
    from config.settings import settings
    import src.scheduler.runner as runner

    # The subject here is the WEEKDAY/dedupe gate, so the enable flags are pinned
    # ON: since 2026-08-19 the live .env HOLDS all three off for the ML A/B
    # window, which short-circuits this function before the weekday is ever
    # consulted. Reading them from the environment would make this test assert
    # the deployment state instead of the scheduling logic — the sibling test
    # below owns the flags.
    monkeypatch.setattr(settings, "enable_eod_ml_train", True)
    monkeypatch.setattr(settings, "enable_eod_ml_buy_train", True)
    monkeypatch.setattr(settings, "enable_eod_ml_exit_train", True)
    at = time(8, 0)
    sat = datetime(2026, 8, 15, 8, 30)          # Saturday (weekday 5)
    assert runner._should_run_weekly_ml_train(sat, None, at)
    assert not runner._should_run_weekly_ml_train(datetime(2026, 8, 14, 8, 30), None, at)   # Friday
    assert not runner._should_run_weekly_ml_train(datetime(2026, 8, 16, 8, 30), None, at)   # Sunday
    assert not runner._should_run_weekly_ml_train(datetime(2026, 8, 15, 7, 59), None, at)   # too early
    # Already ran this Saturday.
    assert not runner._should_run_weekly_ml_train(sat, sat.date(), at)


def test_weekly_ml_train_respects_the_enable_flags(monkeypatch):
    from datetime import datetime, time
    from config.settings import settings
    import src.scheduler.runner as runner

    sat = datetime(2026, 8, 15, 9, 0)
    monkeypatch.setattr(settings, "enable_eod_ml_train", False)
    monkeypatch.setattr(settings, "enable_eod_ml_buy_train", False)
    monkeypatch.setattr(settings, "enable_eod_ml_exit_train", False)
    assert not runner._should_run_weekly_ml_train(sat, None, time(8, 0)),         "all trainers disabled -> the weekly job must not fire"
    monkeypatch.setattr(settings, "enable_eod_ml_buy_train", True)
    assert runner._should_run_weekly_ml_train(sat, None, time(8, 0))


def test_weekly_ml_train_launcher_is_single_flight(monkeypatch):
    import threading
    import src.scheduler.runner as runner

    started = threading.Event()
    release = threading.Event()

    def _slow():
        started.set()
        release.wait(5)

    monkeypatch.setattr(runner, "_weekly_ml_work", _slow)
    runner._WEEKLY_ML_THREAD = None
    runner._run_weekly_ml_train()
    assert started.wait(5), "first launch must start the worker"
    t1 = runner._WEEKLY_ML_THREAD
    runner._run_weekly_ml_train()                # second call while running
    assert runner._WEEKLY_ML_THREAD is t1, "second launch must be refused"
    release.set()
    t1.join(5)
    runner._WEEKLY_ML_THREAD = None


def test_eod_work_no_longer_trains_models():
    """The 2026-08-12 directive: retrains are WEEKLY (Saturday), so the EOD
    body must not reference the trainers anymore."""
    import inspect
    import src.scheduler.runner as runner
    src = inspect.getsource(runner._eod_work)
    for needle in ("eod_train_buy", "eod_train_exit", "ml_model import eod_train"):
        assert needle not in src, f"EOD still trains models ({needle})"
    wk = inspect.getsource(runner._weekly_ml_work)
    for needle in ("eod_train", "eod_train_buy", "eod_train_sell", "eod_train_exit"):
        assert needle in wk, f"weekly job lost a trainer ({needle})"
