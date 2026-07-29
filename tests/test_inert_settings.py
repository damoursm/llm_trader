"""Guards against config that LOOKS live but is inert (audited 2026-07-25).

The failure mode: a setting is defined, documented as doing something, and read
by nothing. No error, no warning — flipping it appears to work and changes
nothing. Three instances were found and fixed on 2026-07-25 (overlay inversion,
unrecognised inversion names, and the combined-score gate below); these tests
stop the class from silently returning.
"""

import re
import pathlib

import pytest

from config.settings import Settings


_SRC = [p for p in pathlib.Path("src").rglob("*.py")]
_SRC += [p for p in pathlib.Path("dashboard").rglob("*.py")]
_SRC += [pathlib.Path("main.py")]

# Settings that are KNOWINGLY inert, each with the reason. A name may only sit
# here with a comment in settings.py explaining why — being unread is otherwise
# a bug, not a state to enshrine.
KNOWN_INERT = {
    "enable_combined_score_gate":  "never wired; ledger says the gate would block the BEST trades",
    "schedule_daily":              "legacy cron; runner uses the RTH/extended/overnight windows",
    "min_combined_score_for_entry": "offline policy_eval counterfactual only",
}

# Resolved dynamically via config.settings.directional(f'{name}_{side}'), so a
# static grep cannot see them.
_DIRECTIONAL_SUFFIXES = ("_long", "_short")


def _body() -> str:
    out = []
    for f in _SRC:
        out.append(f.read_text(encoding="utf-8", errors="replace"))
    sp = pathlib.Path("config/settings.py").read_text(encoding="utf-8", errors="replace")
    # Drop field DEFINITION lines so a definition never counts as a use. What
    # REMAINS still counts: several fields are consumed only by an @property on
    # Settings itself (`stocks_list` reads `self.stock_watchlist`), which is a
    # real use even though no other module ever names the field.
    out.append("\n".join(l for l in sp.splitlines()
                         if not re.match(r'\s*[a-z_][\w]*\s*:\s*[\w\[\]\.\'"| ]+\s*=', l)))
    return "\n".join(out)


def test_no_new_unread_settings():
    """Every setting must be read somewhere, or be listed as knowingly inert."""
    body = _body()
    fields = set(Settings.model_fields)
    unread = set()
    for fld in fields:
        if fld.endswith(_DIRECTIONAL_SUFFIXES):
            continue                     # resolved via directional()
        if re.search(rf'\bsettings\.{re.escape(fld)}\b', body) or \
           re.search(rf'\bself\.{re.escape(fld)}\b', body) or \
           re.search(rf'["\']{re.escape(fld)}["\']', body):
            continue
        unread.add(fld)
    unexpected = unread - set(KNOWN_INERT)
    assert not unexpected, (
        "settings defined but never read (flipping them does nothing, silently). "
        "Wire them up, delete them, or add to KNOWN_INERT with a reason in "
        f"settings.py: {sorted(unexpected)}")


def test_known_inert_settings_are_marked_in_the_source():
    """An inert setting must SAY so where a reader will see it."""
    sp = pathlib.Path("config/settings.py").read_text(encoding="utf-8", errors="replace")
    for name in KNOWN_INERT:
        i = sp.find(f"{name}:")
        assert i != -1, f"{name} missing from settings.py"
        window = sp[max(0, i - 1800):i + 200]
        assert re.search(r"inert|NOT IMPLEMENTED|LEGACY|offline", window, re.I), (
            f"{name} is inert but nothing near its definition says so")


def test_directional_overrides_actually_resolve():
    """The _long/_short pairs are exempted from the unread check above because
    `directional()` builds their names at runtime — so prove that path works,
    otherwise the exemption would hide a genuinely dead pair."""
    from config.settings import directional, settings
    for base in ("adverse_stop_pct", "horizon_expiry_floor_mult",
                 "actionable_threshold_adj"):
        for side in ("BUY", "SELL"):
            directional(base, side)          # must not raise
    # And it must actually PREFER a set override over the shared value.
    assert directional("adverse_stop_pct", "BUY") == settings.adverse_stop_pct_long
    assert directional("adverse_stop_pct", "SELL") == settings.adverse_stop_pct_short


def test_no_module_logs_into_a_silent_channel():
    """Same class of defect as an unread setting: a log line that never emits.

    The project configures **loguru** sinks only, so `logging.getLogger(__name__)`
    in `src/` produces a logger with no handler — every `.info()`/`.warning()` on
    it is discarded. That is worse than an unread setting, because the whole point
    of those calls is to make a degraded path AUDIBLE: this bug silently muted
    `replay`'s skip counts and the fail-soft warning that distinguishes "the
    restore is broken" from "the table isn't populated yet".

    Found after the same failure shape appeared twice in one session (an optional
    scipy import guarding a live verdict, then this), so it is mechanised rather
    than left to review.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "src"
    pat = re.compile(r"^\s*logger\s*=\s*logging\.getLogger", re.MULTILINE)
    offenders = [str(p.relative_to(root.parent))
                 for p in root.rglob("*.py") if pat.search(p.read_text(encoding="utf-8"))]
    assert not offenders, (
        "these modules log into a channel with no sink — their output is silently "
        f"discarded. Use `from loguru import logger`: {offenders}")
