"""The dashboard page must ship only the active tab (2026-08-15).

The regression this pins: ``serve_layout`` used to embed all six tabs' rendered
content as ``dcc.Tab`` children, so EVERY page load built the whole dashboard
server-side — measured 111 s against a cold cache (the five then-uncached
accessors re-parsed OHLCV on the request thread) and ~0.7 s warm. Tabs now
hydrate lazily: one empty container per tab, filled the first time the tab
becomes active, sticky afterwards.

Two contracts:
  1. building the layout calls NO tab renderer (the payload is chrome only);
  2. the fill rule renders exactly once per tab — wrong tab or already-filled
     container is a no-op (PreventUpdate), never a recompute.
"""

import dash
import pytest
from dash.exceptions import PreventUpdate

from dashboard import app as dash_app
from src.db import repo

# dashboard.data flips the repo read-only process-wide at import, which is right
# for the dashboard and wrong for the rest of the suite.
repo.set_read_only(False)


def _walk(node):
    yield node
    ch = getattr(node, "children", None)
    if isinstance(ch, (list, tuple)):
        for c in ch:
            yield from _walk(c)
    elif ch is not None:
        yield from _walk(ch)


def test_spec_covers_all_six_tabs():
    values = [v for v, _l, _r in dash_app._TAB_SPEC]
    assert values == ["rationale", "methods", "exit_perf", "returns",
                      "execution", "data_quality"]
    assert all(callable(r) for _v, _l, r in dash_app._TAB_SPEC)


def test_serve_layout_renders_no_tab_content(monkeypatch):
    """The layout is chrome only — a renderer running at layout time is the
    111-s page load coming back."""
    called = []
    spec = tuple((v, l, (lambda _v=v: called.append(_v)))
                 for v, l, _r in dash_app._TAB_SPEC)
    monkeypatch.setattr(dash_app, "_TAB_SPEC", spec)
    layout = dash_app.serve_layout()
    assert called == [], f"tab renderer(s) ran during layout build: {called}"
    # ...and every tab's empty container is present for the callbacks to fill.
    ids = {getattr(n, "id", None) for n in _walk(layout)}
    for value, _label, _render in spec:
        assert f"tab-{value}" in ids


def test_fill_renders_only_the_active_empty_tab():
    out = dash_app._fill_tab("methods", None, "methods", lambda: dash.html.Div("body"))
    assert out is not None                          # active + empty → renders

    with pytest.raises(PreventUpdate):              # someone else's tab
        dash_app._fill_tab("returns", None, "methods", lambda: dash.html.Div())

    with pytest.raises(PreventUpdate):              # already filled → sticky
        dash_app._fill_tab("methods", {"props": {}}, "methods",
                           lambda: (_ for _ in ()).throw(AssertionError("re-rendered")))


def test_fill_is_safe_wrapped():
    """A tab whose data layer raises must render an inline error, not break the
    callback (the _safe contract, preserved through the lazy path)."""
    def boom():
        raise RuntimeError("data hiccup")
    out = dash_app._fill_tab("methods", None, "methods", boom)
    assert out is not None
