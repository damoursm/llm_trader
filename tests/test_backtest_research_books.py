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


def test_unregistered_book_fails_loudly():
    with pytest.raises(KeyError):
        M.research_spec("never_registered")
