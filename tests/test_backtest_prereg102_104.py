"""PREREG102 / PREREG104's switches in the simulator that mimics production (`src/backtest/mimic.py`,
`src/backtest/strategies/etf.py`; registered 2026-10-14), and PREREG96 Amendment 3's ``dip_filter``.

What must hold: (1) THE DEFAULTS ARE PRODUCTION'S — the ETF arm leaves the VIX-linked products out while production's
`sel_short_etf_exclude_vix` does (E1), every short arm is in the account, the fall arm keeps production's slice (12) and
volatility exit (on, 0.5) — and `Mimic.off()` keeps the arm as before the block; (2) each switch changes ONLY what it
should: without the ETF arm (E2) or the fall arm (F2) that arm is never built and every other book is the default's,
piece for piece; the fall arm's slice (F1) changes only its strategy's slices; its volatility exit (F3 off, F4 0.3)
reaches the fall build alone and `vol.FALL` is restored after (an exception too); (3) THE VIX EXCLUSION IS BY SYMBOL and
reads the RE-RANKED picks — the plain symbol decides (``XYZ@E``, ``@D`` and a deep-store name alike), a product taking a
VIX product's slot comes in, a VIX product never comes out (whatever the file holds), and a missing or stale re-ranked
file is an error, never a silent filter. Synthetic stubs only — no research store, no network."""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import settings
from config.settings import Settings
from src.backtest import mimic as M
from src.backtest.strategies import etf as ETF
from src.backtest.strategies import vol as VOL

ET = "America/New_York"
VIX = M.vix_etps()


def dn(s) -> int:
    return int((pd.Timestamp(s) - pd.Timestamp("1970-01-01")).days)


def _default(name):
    """A setting's value in the code (production's, `config/settings.py`), whatever a test pinned."""
    return Settings.model_fields[name].default


# ── (1) the defaults are production's ────────────────────────────────────────

def test_the_defaults_follow_productions_switches(monkeypatch):
    m = M.Mimic()
    assert m.short_arms == ("vol", "etf", "fall") and all(m.arm_on(a) for a in ("vol", "etf", "fall"))
    assert m.fall_slices is None and m.fall_volnorm_on is None and m.fall_volnorm is None
    monkeypatch.setattr(settings, "sel_short_etf_exclude_vix", True)
    assert M.Mimic().etf_vix is False and M.etf_exclusion(M.Mimic()) == VIX          # production today: left out
    monkeypatch.setattr(settings, "sel_short_etf_exclude_vix", False)
    assert M.Mimic().etf_vix is True and M.etf_exclusion(M.Mimic()) is None          # the switch off: ranked
    assert _default("sel_short_etf_exclude_vix") is True                             # the block is live (2026-10-14)
    assert M.Mimic.off().etf_vix is True                                             # the earlier simulator
    assert M.Mimic.borrow_v2().short_arms == M.SHORT_ARMS


def test_the_vix_products_are_productions_list():
    from src.signals import sel_short as ss
    assert VIX == frozenset(ss.VIX_ETPS) and {"UVXY", "UVIX", "VXX", "SVXY", "TVIX", "XIV"} <= VIX
    assert not {"SOXL", "LABU", "SPLV", "VXF"} & VIX


def test_the_fall_arms_default_exit_and_slice_are_productions():
    fall = dict(VOL.LIVE, **VOL.FALL)                          # how `VolEngine.cfg` merges the fall arm's params
    assert fall["volnorm_on"] is _default("enable_sel_short_volnorm_exit") is True
    assert fall["volnorm"] == _default("sel_short_volnorm_ratio") == 0.5
    assert fall["cap"] == _default("sel_short_cover_multiple")
    S = {s.name: s for s in M.live_strategies(M.Mimic())}
    from src.performance.sim_account import arm_slices
    assert S["fall"].slices == arm_slices("fall") and _default("sel_short_fall_account_slices") == 12.0


def test_nonsense_is_refused():
    for bad in ({"short_arms": ["vol"]}, {"short_arms": ("vol", "dip")}, {"short_arms": ("vol", "vol")},
                {"fall_slices": 0}, {"fall_slices": -3.0}, {"fall_slices": float("inf")}, {"fall_slices": True},
                {"fall_volnorm": 0.0}, {"fall_volnorm": 1.5}, {"fall_volnorm": True},
                {"fall_volnorm_on": 1}, {"fall_volnorm_on": False, "fall_volnorm": 0.3}, {"etf_vix": 1}):
        with pytest.raises(ValueError):
            M.Mimic(**bad)
    assert M.Mimic(short_arms=()).short_arms == ()                                   # no short arm at all: allowed


# ── (2) each switch changes only what it should ──────────────────────────────

def test_the_fall_slice_changes_the_fall_strategy_alone():
    from src.performance.sim_account import arm_slices
    base = {s.name: s for s in M.live_strategies(M.Mimic())}
    f1 = {s.name: s for s in M.live_strategies(M.Mimic(fall_slices=24.0))}
    assert f1["fall"].slices == 24.0 and base["fall"].slices == arm_slices("fall")
    assert dataclasses.replace(f1["fall"], slices=base["fall"].slices) == base["fall"]
    assert set(f1) == set(base) and all(f1[n] == base[n] for n in base if n != "fall")
    # an arm left out keeps its strategy: its slice stays unused (no pieces)
    assert {s.name for s in M.live_strategies(M.Mimic(short_arms=("vol", "fall")))} == set(base)


def _halts_tests():
    """The stub live combination of `test_backtest_halts` (`_stub_case`)."""
    p = Path(__file__).with_name("test_backtest_halts.py")
    spec = importlib.util.spec_from_file_location("_halts_stub_102", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(monkeypatch, mim, etf_none=False, dip_override=None):
    """`mimic.live_books` on the stub combination, recording what each adapter was asked for: the ETF arm's keyword
    arguments (values), `vol.FALL` while the fall arm is built, the calls of the engine."""
    T = _halts_tests()
    monkeypatch.setattr(settings, "enable_crash_long", False, raising=False)
    monkeypatch.setattr(M, "house_mix", lambda: (("X", 2.0, 2.86), ("Y", 41.1, 45.2), ("Z", 3.5, 4.0)))
    eng, etf, costs, p_lend, dip = T._stub_case()
    if dip_override is not None:
        dip = dip_override(dip)
    eng.asm = types.SimpleNamespace(cache={"stale": 1})
    seen = {"etf_kw": [], "fall_cfg": []}
    turn0, pieces0 = eng.turn_pieces, etf.pieces

    def turn_pieces(*a, **k):
        seen["fall_cfg"].append(dict(VOL.FALL))
        return turn0(*a, **k)

    def etf_pieces(lo, hi, **kw):
        seen["etf_kw"].append(dict(kw))
        return pieces0(lo, hi, **kw)
    eng.turn_pieces, etf.pieces = turn_pieces, etf_pieces
    bars = M.ExtBars()
    for t in ("AAA", "BBB", "CCC@D", "DDD", "EEE@X", "FFF", "QQQX@E", "MEGA", "BIGG"):
        bars._m[t] = None
    log = {}
    books = M.live_books("2023-03-01", "2023-03-31", mim, eng=eng, etf=None if etf_none else etf, panel=None,
                         bars=bars, costs=costs, p_lend=p_lend, settle=lambda d: int(d) + 2, log=log, dip=dip)
    return books, log, eng, etf, seen, T


def test_the_default_build_is_unchanged_and_asks_for_the_re_ranked_etf_picks(monkeypatch):
    monkeypatch.setattr(settings, "sel_short_etf_exclude_vix", True)
    keep = dict(VOL.FALL)
    books, log, eng, etf, seen, T = _run(monkeypatch, M.Mimic(seed=3))
    assert T._fields(books) == T.BEFORE and log == T.BEFORE_LOG            # the pieces of every book as before
    assert [kw.get("exclude") for kw in seen["etf_kw"]] == [VIX]            # production's VIX block
    assert seen["fall_cfg"] == [keep] and dict(VOL.FALL) == keep             # FALL untouched
    assert eng.asm.cache == {"stale": 1}                                     # the fall arm built: the stub's cache kept
    _, _, _, _, seen0, _ = _run(monkeypatch, M.Mimic(seed=3, etf_vix=True))
    assert ["exclude" in kw for kw in seen0["etf_kw"]] == [False]           # E0: the arm before the block


def test_without_the_etf_arm_it_is_never_built_and_every_other_book_is_unchanged(monkeypatch):
    base, _, _, _, _, T = _run(monkeypatch, M.Mimic(seed=3))
    from src.backtest.strategies import etf as ETFm

    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("the ETF arm was loaded although it is out of the account")
    monkeypatch.setattr(ETFm, "EtfArm", Boom)
    books, log, eng, etf, seen, _ = _run(monkeypatch, M.Mimic(seed=3, short_arms=("vol", "fall")), etf_none=True)
    assert books["etf"] == [] and seen["etf_kw"] == []
    fb, f0 = T._fields(books), T._fields(base)
    # every other book piece for piece — the ranks too: removing a ticker keeps the others' order (live: ticker, arm)
    for k in ("vol", "fall", "dip"):
        assert [r[:7] + r[8:] for r in fb[k]] == [r[:7] + r[8:] for r in f0[k]], k
        order = lambda rows: [r[0] for r in sorted(rows, key=lambda r: r[7] if r[7] is not None else -1)]  # noqa: E731
        assert order(fb[k]) == order(f0[k]), k
    assert "ext_etf_no_bars" not in log


def test_without_the_fall_arm_it_is_never_built_and_every_other_book_is_unchanged(monkeypatch):
    base, _, _, _, _, T = _run(monkeypatch, M.Mimic(seed=3))
    books, log, eng, etf, seen, _ = _run(monkeypatch, M.Mimic(seed=3, short_arms=("vol", "etf")))
    assert books["fall"] == [] and seen["fall_cfg"] == []
    assert [c[0] for c in eng.calls] == ["pieces"]                           # no turn build
    assert eng.asm.cache == {}                    # the vol arm built on an empty cache, as after the fall arm's build
    fb, f0 = T._fields(books), T._fields(base)
    for k in ("vol", "etf", "dip"):
        assert [r[:7] + r[8:] for r in fb[k]] == [r[:7] + r[8:] for r in f0[k]], k
        order = lambda rows: [r[0] for r in sorted(rows, key=lambda r: r[7] if r[7] is not None else -1)]  # noqa: E731
        assert order(fb[k]) == order(f0[k]), k
    assert "ext_fall_no_bars" not in log


@pytest.mark.parametrize("kw,over", [({"fall_volnorm_on": False}, {"volnorm_on": False}),
                                     ({"fall_volnorm": 0.3}, {"volnorm": 0.3})])
def test_the_fall_exit_reaches_the_fall_build_alone_and_is_restored(monkeypatch, kw, over):
    keep, live = dict(VOL.FALL), dict(VOL.LIVE)
    base, _, _, _, _, T = _run(monkeypatch, M.Mimic(seed=3))
    books, log, eng, etf, seen, _ = _run(monkeypatch, M.Mimic(seed=3, **kw))
    assert seen["fall_cfg"] == [dict(keep, **over)]                          # the fall build saw the override
    assert dict(VOL.FALL) == keep and dict(VOL.LIVE) == live                 # restored; the vol rule never touched
    assert T._fields(books) == T._fields(base)                               # the stub's pieces do not read FALL
    cfg = dict(VOL.LIVE, **dict(keep, **over))                               # what the engine merges: the fall cfg
    assert (cfg["volnorm_on"], cfg["volnorm"]) == ((False, 0.5) if "volnorm_on" in over else (True, 0.3))


def _two_orders(dip):
    """The stub's dip order + a second order of another name: two order ids (`dip.order_pieces`' ``oid``)."""
    a = dict(dip[0], oid="MEGA|2023-03-06")
    b = dict(dip[0], tkn="BIGG", oid="BIGG|2023-03-07", ens=int(dip[0]["ens"]) + 86400 * 10**9)
    return [a, b]


def test_the_dip_filter_is_productions_switch_and_changes_the_dip_book_alone(monkeypatch):
    """PREREG96 Amendment 3: ``dip_filter`` drops every piece of an order `dip.filter_keep()` maps to False (production's
    construction: a filtered signal never becomes an order, never holds a slot); every other book is the default's."""
    from src.backtest.strategies import dip as DIP
    assert M.Mimic().dip_filter is False and _default("enable_dip_long_filter") is False   # production: off
    with pytest.raises(ValueError):
        M.Mimic(dip_filter=1)

    def boom(*a, **k):
        raise AssertionError("the dip filter was read although it is off")
    monkeypatch.setattr(DIP, "filter_keep", boom)
    base, _, _, _, _, T = _run(monkeypatch, M.Mimic(seed=3), dip_override=_two_orders)
    assert [p["tkn"] for p in base["dip"]] == ["MEGA", "BIGG"]
    calls = []
    monkeypatch.setattr(DIP, "filter_keep", lambda *a, **k: calls.append((a, k)) or {"MEGA|2023-03-06": False,
                                                                                         "BIGG|2023-03-07": True})
    books, log, eng, etf, seen, _ = _run(monkeypatch, M.Mimic(seed=3, dip_filter=True), dip_override=_two_orders)
    assert calls == [((), {})]                                       # production's construction: research=False
    assert [p["tkn"] for p in books["dip"]] == ["BIGG"]              # the filtered order is gone, the kept one stays
    fb, f0 = T._fields(books), T._fields(base)
    assert all(fb[k] == f0[k] for k in fb if k != "dip") and fb["dip"] == [r for r in f0["dip"] if r[0] == "BIGG"]
    # an order the map does not hold is kept (outside the scored years)
    monkeypatch.setattr(DIP, "filter_keep", lambda *a, **k: {})
    books2, *_ = _run(monkeypatch, M.Mimic(seed=3, dip_filter=True), dip_override=_two_orders)
    assert T._fields(books2) == f0


def test_the_fall_exit_is_restored_after_an_exception():
    keep = dict(VOL.FALL)
    with pytest.raises(RuntimeError):
        with M.fall_exit(M.Mimic(fall_volnorm_on=False)):
            assert VOL.FALL["volnorm_on"] is False
            raise RuntimeError("boom")
    assert dict(VOL.FALL) == keep
    with M.fall_exit(M.Mimic()):
        assert dict(VOL.FALL) == keep                                        # no override: nothing patched
    assert M.fall_exit_over(M.Mimic(fall_volnorm=0.3)) == {"volnorm": 0.3}


# ── (3) the VIX exclusion: by symbol, on the re-ranked picks ────────────────

D0 = "2025-03-03"                                       # after IBKR's archive: no archive lookup in the adapter
PER = 13


def _bars(closes, day0):
    n = len(closes)
    sday = np.array([day0 + i // PER for i in range(n)], np.int64)
    bod = np.array([i % PER for i in range(n)], np.int64)
    ns = np.array([pd.Timestamp(f"{(pd.Timestamp('1970-01-01') + pd.Timedelta(days=int(d))).date()} 09:30",
                                tz=ET).value + (int(b) + 1) * 30 * 60 * 10**9 for d, b in zip(sday, bod)], np.int64)
    c = np.asarray(closes, float)
    return {"c": c, "h": c * 1.001, "l": c * 0.999, "atr": np.full(n, 4.0), "ns": ns, "sday": sday, "bod": bod,
            "pos": bod.copy()}


def _row(tkn, day, pos=0, px=12.0):
    return {"dn": dn(day), "pos": pos, "tkn": tkn, "status": "riser", "dtc": 1.0, "vratio": 2.0, "actual": px,
            "px": px, "dv20": 5e7, "year": 2025, "fee_file": np.nan, "borrowable": True}


def _arm(tmp_path, recorded_rows, period=None):
    """A stub `EtfArm` over synthetic picks and bars (every name rises steadily: each pick is a riser)."""
    from src.backtest import account as A

    class _CM:
        @staticmethod
        def predict(price, dv, atr, bod, year):
            return np.zeros(len(price))

    class _RF:
        @staticmethod
        def daily(tkn, days, fb):
            return np.zeros(len(days))
    arm = object.__new__(ETF.EtfArm)
    arm.period, arm._root = period, str(tmp_path)
    d0 = dn("2025-02-20")
    arm.cal = np.arange(d0, d0 + 40, dtype=np.int64)
    arm.cm, arm.rf, arm.settle = _CM(), _RF(), A.default_settle()
    arm.ve = types.SimpleNamespace(dnum=dn, plain=ETF.plain)
    E = pd.DataFrame(recorded_rows)
    E.index = pd.Index(np.arange(1000, 1000 + len(E)), name="prow")
    arm.E = E
    closes = [10.0 * 1.002 ** i for i in range(PER * 19)]                  # +1% a session or so: always a riser
    arm._bars = {t: _bars(closes, d0) for t in set(E.tkn) | {"LABU@E", "SOXL@E", "VXX@E", "UVIX", "TVIX@D"}}
    arm._ca = {}
    return arm


def _write_reranked(tmp_path, arm, rows, excluded, period=None, sig=None, tag="novix"):
    p = ETF.excluded_events_path(period, tag, root=str(tmp_path))
    p.parent.mkdir(parents=True, exist_ok=True)
    E = pd.DataFrame(rows)
    E.index = pd.Index(np.arange(5000, 5000 + len(E)), name="prow")
    E.to_pickle(p)
    meta = {"period": period, "excluded": sorted(excluded),
            "source_sig": sig if sig is not None else ETF.events_signature(arm.E)}
    p.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")
    return p


def test_the_re_ranked_files_live_where_the_build_writes_them(tmp_path):
    assert ETF.excluded_events_path(None, "novix", root=str(tmp_path)) == tmp_path / "etf" / "vol_events_de_novix_arm.pkl"
    assert ETF.excluded_events_path("A0", "novix", root=str(tmp_path)) == tmp_path / "etf" / "vol_events_a0_novix_arm.pkl"
    assert [ETF.plain(t) for t in ("UVXY@E", "TVIX@D", "UVIX", "SOXL@E")] == ["UVXY", "TVIX", "UVIX", "SOXL"]
    assert ETF.exclusion_key(["UVXY@E", "UVIX"]) == frozenset({"UVXY", "UVIX"}) and ETF.exclusion_key([]) is None


def test_the_exclusion_reads_the_re_ranked_picks_and_never_returns_a_vix_product(tmp_path):
    day = "2025-03-03"
    rec = [_row("UVXY@E", day), _row("SOXL@E", "2025-03-04"), _row("UVIX", "2025-03-05"), _row("TVIX@D", "2025-03-06")]
    arm = _arm(tmp_path, rec)
    # the re-ranked picks: LABU took UVXY's slot; a stray VXX row (a VIX product) must still never come out
    _write_reranked(tmp_path, arm, [_row("LABU@E", day), _row("SOXL@E", "2025-03-04"), _row("VXX@E", "2025-03-05")],
                    VIX)
    E1 = arm.events(VIX)
    assert set(E1.tkn) == {"LABU@E", "SOXL@E"}
    assert set(arm.events(None).tkn) == {"UVXY@E", "SOXL@E", "UVIX", "TVIX@D"}       # E0: the recorded picks
    assert arm.events_source(VIX).endswith("vol_events_de_novix_arm.pkl") and arm.events_source(None) == "recorded"
    lo, hi = "2025-03-01", "2025-03-10"
    p1 = arm.pieces(lo, hi, give=1.0, lend=None, turn=None, entry="pick", exclude=VIX)
    p0 = arm.pieces(lo, hi, give=1.0, lend=None, turn=None, entry="pick")
    assert p1 and not {ETF.plain(p["tkn"]) for p in p1} & VIX and "LABU@E" in {p["tkn"] for p in p1}
    assert {ETF.plain(p["tkn"]) for p in p0} & VIX                                    # E0 trades them
    # the SAME pick of a name both files hold is the same piece
    s1 = [p for p in p1 if p["tkn"] == "SOXL@E"]
    s0 = [p for p in p0 if p["tkn"] == "SOXL@E"]
    assert len(s1) == len(s0) == 1 and s1[0]["ens"] == s0[0]["ens"] and s1[0]["e"] == s0[0]["e"]


def test_no_re_ranked_picks_is_an_error_never_a_silent_filter(tmp_path):
    arm = _arm(tmp_path, [_row("UVXY@E", D0), _row("SOXL@E", D0, pos=3)])
    with pytest.raises(FileNotFoundError, match="re-ranked"):
        arm.events(VIX)
    assert arm.events_source(VIX) is None
    # a file for ANOTHER exclusion, or another period, is not this one
    _write_reranked(tmp_path, arm, [_row("SOXL@E", D0)], {"UVXY"}, tag="nouvxy")
    _write_reranked(tmp_path, arm, [_row("SOXL@E", D0)], VIX, period="A0", tag="novix")
    with pytest.raises(FileNotFoundError):
        arm.events(VIX)


def test_a_re_ranked_file_of_another_recorded_version_is_refused(tmp_path):
    arm = _arm(tmp_path, [_row("UVXY@E", D0), _row("SOXL@E", D0, pos=3)])
    _write_reranked(tmp_path, arm, [_row("SOXL@E", D0)], VIX, sig=[1, 2, 3])
    with pytest.raises(ValueError, match="another version"):
        arm.events(VIX)


def test_set_a0s_arm_reads_set_a0s_re_ranked_picks(tmp_path):
    arm = _arm(tmp_path, [_row("TVIX@E", "2025-03-03"), _row("SOXL@E", "2025-03-04")], period="A0")
    _write_reranked(tmp_path, arm, [_row("LABU@E", "2025-03-03"), _row("SOXL@E", "2025-03-04")], VIX, period="A0")
    assert set(arm.events(VIX).tkn) == {"LABU@E", "SOXL@E"}
    assert arm.events_source(VIX).endswith("vol_events_a0_novix_arm.pkl")


def test_activation_drops_only_the_pieces_an_arm_s_map_switches_off(monkeypatch, tmp_path):
    """PREREG108: ``Mimic(activation=<schedule>)`` drops each piece decided in a market cell its arm's month map leaves
    out (`mimic.apply_activation`); an arm or a month the schedule does not name keeps every piece."""
    from src.backtest import context as CX
    days = pd.bdate_range("2022-01-03", "2023-12-29")
    monkeypatch.setattr(CX, "cached_table", lambda: pd.DataFrame({"cell": ["above/mid"] * len(days)}, index=days))
    M._activation_schedule.cache_clear()
    base, _, _, _, _, T = _run(monkeypatch, M.Mimic(seed=3), dip_override=_two_orders)
    assert base["dip"] and base["vol"]
    off = tmp_path / "off.json"
    off.write_text(json.dumps({"arms": {"dip": {"2023-03": ["above/low"]}, "vol": {"2023-03": list(CX.CELLS)}}}))
    books, log, *_ = _run(monkeypatch, M.Mimic(seed=3, activation=str(off)), dip_override=_two_orders)
    fb, f0 = T._fields(books), T._fields(base)
    assert fb["dip"] == [] and log["activation_dip_off"] == len(base["dip"]) and log["activation_vol_off"] == 0
    assert all(fb[k] == f0[k] for k in fb if k != "dip")             # every other book exactly the default's
    other = tmp_path / "other_month.json"
    other.write_text(json.dumps({"arms": {"dip": {"2023-02": []}}}))  # March not named: active everywhere
    books2, *_ = _run(monkeypatch, M.Mimic(seed=3, activation=str(other)), dip_override=_two_orders)
    assert T._fields(books2) == f0
    with pytest.raises(ValueError):
        M.Mimic(activation=str(tmp_path / "missing.json"))
