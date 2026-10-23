"""PREREG110: a study's research books join the live combination through `Mimic(research=...)` and
`mimic.register_research` — never production's books, never the live configuration's fingerprint when unset."""
import pytest

from src.backtest import mimic as M


@pytest.fixture(autouse=True)
def _clean_registry():
    keep = dict(M.RESEARCH)
    yield
    M.RESEARCH.clear()
    M.RESEARCH.update(keep)


def test_default_has_no_research_book():
    assert M.Mimic(seed=0).research == ()
    names = [s.name for s in M.live_strategies(M.Mimic(seed=0))]
    assert not set(names) & set(M.RESEARCH)


@pytest.mark.parametrize("bad", [("dip",), ("vol",), ("crash",), ("x", "x"), ["x"], (1,)])
def test_research_refuses_production_books_and_bad_shapes(bad):
    with pytest.raises(ValueError):
        M.Mimic(seed=0, research=bad)


def test_register_refuses_production_names_and_bad_side():
    with pytest.raises(ValueError):
        M.register_research("dip", lambda lo, hi, period=None: [])
    with pytest.raises(ValueError):
        M.register_research("x", lambda lo, hi, period=None: [], side=0)


def test_registered_book_becomes_a_strategy_with_its_terms():
    M.register_research("tom", lambda lo, hi, period=None: [], side=1, slices=4.0, max_open=1, rate=0.75,
                        priority=5, quality=0.2)
    strat = {s.name: s for s in M.live_strategies(M.Mimic(seed=0, research=("tom",)))}
    s = strat["tom"]
    assert (s.side, s.slices, s.max_open, s.house_maint, s.house_init) == (1, 4.0, 1, 0.75, 0.75)
    assert s.book == "research_tom" and s.priority == 5 and s.quality == 0.2


def test_engine_fees_keep_the_pieces_own_leg_costs_and_free_room_only(monkeypatch):
    """PREREG123: a research book of auction fills keeps its piece's half-spread + the engine's exact commission (as
    the dip book's legs); quality None takes no one's room; the default keeps the ledger's legs (PREREG110)."""
    M.register_research("night", lambda lo, hi, period=None: [], quality=None, fees="engine")
    M.register_research("tom", lambda lo, hi, period=None: [])
    strat = {s.name: s for s in M.live_strategies(M.Mimic(seed=0, research=("night", "tom")))}
    assert strat["night"].fees == "engine" and strat["night"].quality is None
    assert strat["tom"].fees is None and strat["tom"].quality == 0.2
    with pytest.raises(ValueError):
        M.register_research("bad", lambda lo, hi, period=None: [], fees="ledger")
    seen = []

    class _Costs:
        def apply(self, p, side, legs=True, adv=None):
            seen.append((p["tkn"], side, legs))
            return dict(p, hs_mark=9.0, **({"hs_in": 20.0, "hs_out": 20.0} if legs else {}))
    piece = {"tkn": "TQQQ", "hs_in": 1.0, "hs_out": 1.0}
    night = M.research_costed("night", [piece], _Costs())
    tom = M.research_costed("tom", [piece], _Costs())
    assert seen == [("TQQQ", 1, False), ("TQQQ", 1, True)]
    assert (night[0]["hs_in"], night[0]["hs_out"], night[0]["hs_mark"]) == (1.0, 1.0, 9.0)
    assert (tom[0]["hs_in"], tom[0]["hs_out"]) == (20.0, 20.0)


def test_a_research_book_can_be_first_in_the_deleveraging_line():
    """PREREG124: ``delever_order`` 0 puts a research book before the dip book (1) in the account's de-leveraging; the
    default keeps it out of the line (never de-levered), as PREREG110's books."""
    M.register_research("night", lambda lo, hi, period=None: [], quality=None, fees="engine", delever_order=0)
    M.register_research("tom", lambda lo, hi, period=None: [])
    strat = {s.name: s for s in M.live_strategies(M.Mimic(seed=0, research=("night", "tom")))}
    assert strat["night"].delever_order == 0 and strat["tom"].delever_order is None
    assert strat["dip"].delever_order == 1


def test_unregistered_book_fails_loudly():
    with pytest.raises(KeyError):
        M.research_spec("never_registered")


def test_the_deleveraging_thresholds_default_to_productions_and_can_be_set():
    """PREREG125: `Mimic(delever=(trigger, target))` sets the account's de-leveraging; None keeps production's settings."""
    from config import settings
    assert M.Mimic(seed=0).delever is None
    assert M.live_rules(M.Mimic(seed=0)).delever == (float(settings.live_account_delever_trigger),
                                                    float(settings.live_account_delever_target))
    assert M.live_rules(M.Mimic(seed=0, delever=(0.7, 0.5))).delever == (0.7, 0.5)
    for bad in ((0.5, 0.7), (0.8,), [0.8, 0.6], (0.8, 0.0)):
        with pytest.raises(ValueError):
            M.Mimic(seed=0, delever=bad)
