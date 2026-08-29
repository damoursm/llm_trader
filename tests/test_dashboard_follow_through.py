"""Follow-Through dashboard tab: renders against synthetic accessors (never the
DB), shows the two surfaces (panel accrual + real book), and degrades to a
readable empty state — the tab must never be the thing that breaks a page."""

from __future__ import annotations

import pandas as pd

import dashboard.app as dash_app
import dashboard.data as data


def _texts(node):
    out = []
    for n in _walk(node):
        t = getattr(n, "children", None)
        if isinstance(t, str):
            out.append(t)
    return " ".join(out)


def _walk(node):
    yield node
    ch = getattr(node, "children", None)
    if isinstance(ch, (list, tuple)):
        for c in ch:
            yield from _walk(c)
    elif ch is not None:
        yield from _walk(ch)


def test_tab_renders_with_data(monkeypatch):
    daily = pd.DataFrame({"signal_date": ["2026-08-25"], "n_scored": [324],
                          "n_selected": [10]})
    cands = pd.DataFrame({
        "signal_date": ["2026-08-25", "2026-08-25"],
        "ticker": ["MSFT", "WMT"], "ft_score": [-0.53, -0.53],
        "ft_dir": [-1.0, 1.0], "h1_ret": [1.2, None],
        "pivot_ret": [2.5, None], "settled": [True, False],
    })
    monkeypatch.setattr(data, "follow_through_panel",
                        lambda force=False: {"daily": daily, "cands": cands})
    monkeypatch.setattr(data, "follow_through_trades", lambda force=False: {
        "trades": [{"ticker": "MSFT", "action": "SELL", "status": "OPEN",
                    "entry_date": "2026-08-25", "entry_price": 487.4,
                    "return_pct": 0.4, "ft_score_at_entry": -0.53}],
        # gross_win is a PERCENT (tracker.gross_win_rate contract) — the tab
        # once rescaled it x100 and rendered "6670%"; the assertions below pin
        # the fix with a real value, not None.
        "all": {"n": 1, "open": 1, "closed": 0, "gross_win": 66.7, "avg_net": 0.4},
        "long": {"n": 0, "open": 0, "closed": 0, "gross_win": None, "avg_net": None},
        "short": {"n": 1, "open": 1, "closed": 0, "gross_win": 66.7, "avg_net": 0.4},
    })
    node = dash_app._follow_through_tab()
    txt = _texts(node)
    assert "candidates / day" in txt
    assert "Book by side" in txt and "Trades" in txt
    # the explainer states the mechanism's own holding rule, not the swing stack
    assert "opposite-direction entry" in txt
    assert "67%" in txt                      # rendered as-is (percent), not x100
    assert "6670" not in txt


def test_tab_renders_empty_state(monkeypatch):
    monkeypatch.setattr(data, "follow_through_panel",
                        lambda force=False: {"daily": pd.DataFrame(),
                                             "cands": pd.DataFrame()})
    monkeypatch.setattr(data, "follow_through_trades",
                        lambda force=False: {"trades": [], "all": {}, "long": {},
                                             "short": {}})
    node = dash_app._follow_through_tab()          # must not raise
    assert node is not None


def test_accessors_are_warm_targets():
    names = [n for n, _f in data._warm_targets()]
    assert "follow_through_panel" in names
    assert "follow_through_trades" in names
