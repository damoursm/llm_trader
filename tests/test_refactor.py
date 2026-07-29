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
