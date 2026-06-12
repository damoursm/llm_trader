"""Plotly Dash monitoring dashboard — rationale · method performance · returns.

Single source of truth is DuckDB (read-only here). Launch with:

    python main.py --dashboard
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pandas as pd
from dash import Dash, Input, Output, dash_table, dcc, html
from dash.dash_table.Format import Format, Scheme
from loguru import logger

from config import settings
from dashboard import data, figures
from src.utils import ET

app = Dash(__name__, title="LLM Trader Monitor", suppress_callback_exceptions=True)
server = app.server  # for WSGI deployment if ever needed

_TABLE_KW = dict(
    page_size=25,           # rows shown per page (applies to every table)
    sort_action="native",   # click a column header to sort — toggles ascending → descending → off
    sort_mode="multi",      # shift-click additional headers to sort by several columns
    tooltip_delay=200,      # ms before a header tooltip appears
    tooltip_duration=None,  # keep the explanation visible until the mouse leaves
    style_table={"overflowX": "auto"},
    style_cell={
        "fontFamily": "Arial", "fontSize": 13, "padding": "6px",
        "textAlign": "left", "whiteSpace": "normal", "height": "auto",
        "maxWidth": 460,
    },
    style_header={"backgroundColor": "#f9fafb", "fontWeight": "bold", "cursor": "help"},
)


def _kpi(label: str, value: str, color: str = "#111827", tooltip: str = "") -> html.Div:
    """A stat tile. ``tooltip`` (if given) shows as a hover explanation; the label
    gets a dotted underline + help cursor to advertise that it's there."""
    label_style = {"color": "#6b7280", "fontSize": 12, "display": "inline-block"}
    if tooltip:
        label_style["borderBottom"] = "1px dotted #cbd5e1"
    return html.Div(
        [
            html.Div(label, style=label_style),
            html.Div(value, style={"color": color, "fontSize": 22, "fontWeight": "bold"}),
        ],
        title=tooltip,
        style={
            "padding": "10px 16px", "background": "white", "borderRadius": 8,
            "boxShadow": "0 1px 3px rgba(0,0,0,0.1)", "minWidth": 130, "margin": 6,
            "cursor": "help" if tooltip else "default",
        },
    )


def _h3(text: str, tooltip: str = "") -> html.H3:
    """Section heading with an optional hover explanation."""
    return html.H3(text, title=tooltip or None,
                   style={"cursor": "help"} if tooltip else {})


def _health_banner():
    """Prominent red banner when the latest run had any failed data source
    (API errors, dead live-price feed, …). Returns an empty Div when all good."""
    try:
        failures = data.latest_run_failures()
    except Exception as e:
        logger.debug(f"[dashboard] health banner skipped: {e}")
        failures = []
    if not failures:
        return html.Div()
    items = []
    for f in failures:
        lbl = f.get("source_label") or "?"
        err = f.get("error")
        items.append(f"{lbl} — {err}" if err else lbl)
    return html.Div(
        [
            html.B(f"⚠ {len(failures)} data source(s) failed in the latest run"),
            html.Div(" · ".join(items),
                     style={"marginTop": 4, "fontSize": 12, "whiteSpace": "normal"}),
        ],
        style={"background": "#fef2f2", "border": "1px solid #fecaca", "color": "#b91c1c",
               "borderRadius": 8, "padding": "10px 14px", "marginBottom": 12},
    )


def _pct(x, signed: bool = False) -> str:
    if x is None:
        return "–"
    try:
        return f"{x:+.2f}%" if signed else f"{x:.1f}%"
    except (TypeError, ValueError):
        return str(x)


def _fmt_et(iso_str) -> str:
    """ISO 8601 (any timezone) → ``'YYYY-MM-DD HH:MM'`` in US/Eastern.

    Timestamps are persisted in UTC (or with a raw offset); we convert to Eastern
    and drop the ``+00:00`` tail — the column header carries the ``(ET)`` label.
    Returns the input unchanged if it can't be parsed.
    """
    if not iso_str:
        return ""
    try:
        return datetime.fromisoformat(str(iso_str)).astimezone(ET).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(iso_str)


_INT = Format(precision=0, scheme=Scheme.fixed)
_NUM2 = Format(precision=2, scheme=Scheme.fixed)


def _columns(spec):
    """Build DataTable column dicts from ``(id, label, format, tooltip)`` tuples.

    A non-None format marks the column numeric, so it renders cleanly *and* sorts
    numerically rather than lexicographically ("9" before "10"). The tooltip is
    consumed by ``_header_tooltips``, not here.
    """
    cols = []
    for cid, label, fmt, _tip in spec:
        col = {"id": cid, "name": label}
        if fmt is not None:
            col["type"] = "numeric"
            col["format"] = fmt
        cols.append(col)
    return cols


def _header_tooltips(spec) -> dict:
    """column id → hover explanation, for a DataTable's ``tooltip_header``."""
    return {cid: tip for cid, _label, _fmt, tip in spec if tip}


# Friendly column specs — order here is the on-screen column order.
# Each tuple is (id, header label, numeric format or None, hover explanation).
_REC_COL_SPEC = [
    ("ticker", "Ticker", None, "The stock or ETF symbol."),
    ("action", "Action", None, "The call: BUY, SELL, HOLD or WATCH. Only BUY/SELL are actionable (paper-traded)."),
    ("direction", "Direction", None, "Directional lean behind the call — BULLISH or BEARISH."),
    ("confidence", "Confidence %", _INT, "Model confidence, 0–100%. A BUY/SELL is actionable only above the regime-adjusted threshold (≈78%) with ≥2 agreeing signal sources."),
    ("time_horizon", "Horizon", None, "Intended holding window (e.g. SWING, POSITION)."),
    ("actionable", "Actionable", None, "TRUE = passed the confidence + sources-agreeing gate and was paper-traded. FALSE = monitor only."),
    ("dominant_method", "Top Method", None, "The signal method that contributed most to this call (e.g. news, technical, momentum)."),
    ("type", "Type", None, "Asset class — STOCK, ETF or COMMODITY."),
    ("llm_provider", "LLM", None, "Which model synthesised the recommendation (e.g. Claude Haiku, DeepSeek)."),
    ("generated_at", "Generated (ET)", None, "When the recommendation was produced, in US/Eastern time."),
    ("rationale", "Rationale", None, "The model's plain-English reasoning for the call."),
]

_TRADE_COL_SPEC = [
    ("ticker", "Ticker", None, "The stock or ETF symbol."),
    ("action", "Action", None, "BUY (long) or SELL (short) — how the position was opened."),
    ("direction", "Direction", None, "BULLISH (long) or BEARISH (short)."),
    ("entry_dt", "Entry (ET)", None, "When the position was opened, in US/Eastern time."),
    ("session", "Session", None, "US-market session the position was ENTERED in: rth (09:30–16:00 ET), extended (pre-market 04:00–09:30 + after-hours 16:00–20:00), or overnight (20:00–04:00). Extended entries are sized down and bear the wider extended spread in their return."),
    ("entry_price", "Entry $", _NUM2, "Fill price at entry (the bid-ask spread is applied in the return, not here)."),
    ("filled_qty", "Shares", None, "Shares actually filled at IBKR (real-executions view only)."),
    ("exit_dt", "Exit (ET)", None, "When the position was closed, in US/Eastern time. Blank while still open."),
    ("exit_price", "Exit $", _NUM2, "Fill price at exit. Blank while the position is open."),
    ("held", "Held", None, "Wall-clock holding time: days + hours (e.g. 2d 5h), hours (6h), or minutes (45m) for the freshest entries. Open positions measure entry → now; closed ones entry → exit. Legacy date-only rows fall back to the trading-days count (Nd)."),
    ("return_pct", "Return %", _NUM2, "Spread-adjusted % return. For OPEN positions this is the live mark-to-market — 'what if you closed right now'."),
    ("position_size_multiplier", "Size ×", _NUM2, "Capital weight from the confidence tier (1.0× / 1.5× / 2.0×), after the correlation haircut."),
    ("filled_notional_usd", "Notional $", _NUM2, "Actual dollars at risk: filled shares × average fill price (real-executions view only)."),
    ("status", "Status", None, "OPEN (held, live mark) or CLOSED (realised)."),
    ("broker_entry", "IBKR entry", None, "Did the entry order really execute at the broker? ✓ filled (shares) · ⏳ working / partial · ↻ re-anchoring (tick-scoped cancel; resubmits at the current mark) · ✕ cancelled · ✗ rejected/failed · – never sent (broker off, duplicate twin, sizing skip, or pre-broker history). Simulated view only — the IBKR view contains only filled orders by construction."),
    ("broker_exit", "IBKR exit", None, "Same for the closing order. ⏳ pending = the ledger closed the trade and the exit goes out on the next sync. Blank while the position is open."),
]

# Method Performance table — header explanations (table is built inline below).
_METHOD_HEADER_TIPS = {
    "Method": "The signal method (e.g. news sentiment, technical, momentum) — or an LLM engine row: 'Synthesis LLM' made the final BUY/SELL call, 'Sentiment LLM' scored the per-ticker news (run-dominant engine).",
    "Win rate %": "Method rows — solo simulation: for each closed trade, what if ONLY this method had decided the direction? LLM rows — share of the engine's recommended trades (executed or not) currently positive.",
    "Trades": "Method rows: closed trades this method had a view on (|score| ≥ 0.10). LLM rows: every BUY/SELL the engine recommended — actionable or not, executed or simulated — deduped to its last call per ticker per day.",
    "Avg return %": "Average % return across those trades.",
}


# ── LLM model usage (Method Performance tab → "LLM models used" section) ──────
# Exact model ids per provider. Sources of truth in the code:
#   synthesis Claude   → settings.analyst_model
#   synthesis DeepSeek → claude_analyst._DEEPSEEK_ANALYST_MODEL  ("deepseek-v4-flash")
#   sentiment DeepSeek → sentiment.DEEPSEEK_MODEL                ("deepseek-v4-flash")
#   sentiment Claude   → sentiment.HAIKU_MODEL                   ("claude-haiku-4-5-20251001")
_PROVIDER_LABEL = {
    "anthropic": "Anthropic (Claude)", "deepseek": "DeepSeek",
    "rule-based": "Rule-based", "none": "—", "": "—",
}
_SENTIMENT_MODEL = {
    "deepseek": "deepseek-v4-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "none": "(none — cached / no LLM call)",
}


def _synthesis_model(provider) -> str:
    """Exact model id that produced the final synthesis for a given provider."""
    p = (provider or "").lower()
    if p == "anthropic":
        return settings.analyst_model           # the configured Claude model
    if p == "deepseek":
        return "deepseek-v4-flash"               # DeepSeek V4-Flash analyst fallback
    if p == "rule-based":
        return "rule-based (no LLM)"
    return "—"


def _parse_sentiment_summary(summary):
    """'deepseek×40, anthropic×2' → [('deepseek', 40), ('anthropic', 2)]."""
    if not summary:
        return []
    out = []
    for tok in str(summary).split(","):
        tok = tok.strip()
        if not tok:
            continue
        name, sep, cnt = tok.partition("×")
        try:
            n = int(cnt) if sep else 0
        except ValueError:
            n = 0
        out.append((name.strip().lower(), n))
    return out


def _models_used_rows(runs) -> list:
    """Aggregate which exact LLMs ran across all recorded runs (synthesis + sentiment)."""
    from collections import defaultdict
    syn = defaultdict(int)
    sen_runs, sen_calls = defaultdict(int), defaultdict(int)
    for r in runs.itertuples():
        sp = getattr(r, "llm_synthesis_provider", None)
        syn[(_synthesis_model(sp), _PROVIDER_LABEL.get((sp or "").lower(), sp or "—"))] += 1
        for prov, n in _parse_sentiment_summary(getattr(r, "llm_sentiment_provider", None)):
            key = (_SENTIMENT_MODEL.get(prov, prov), _PROVIDER_LABEL.get(prov, prov.title()))
            sen_runs[key] += 1
            sen_calls[key] += n
    rows = []
    for (model, api), n in sorted(syn.items(), key=lambda kv: -kv[1]):
        rows.append({"Role": "Synthesis", "Model": model, "API": api, "Runs": n, "Calls": "—"})
    for key, n in sorted(sen_runs.items(), key=lambda kv: -sen_calls[kv[0]]):
        model, api = key
        rows.append({"Role": "Sentiment", "Model": model, "API": api, "Runs": n, "Calls": sen_calls[key]})
    return rows


_MODELS_HEADER_TIPS = {
    "Role": "Synthesis = the model that made the final BUY/SELL/HOLD/WATCH call. Sentiment = the model that scored per-ticker news.",
    "Model": "The exact model id that ran (including DeepSeek or rule-based fallbacks).",
    "API": "The provider behind the model.",
    "Runs": "How many recorded pipeline runs used this model in this role.",
    "Calls": "Sentiment only — total per-ticker LLM calls made with this model across runs.",
}


def _safe(render):
    """Render a tab body, surfacing data errors inline instead of crashing the page."""
    try:
        return render()
    except FileNotFoundError:
        return html.Div("No database yet. Run the pipeline (or `python -m src.db.migrate`) first.",
                        style={"padding": 20, "color": "#dc2626"})
    except Exception as e:  # keep the dashboard alive on any data hiccup
        logger.warning(f"[dashboard] tab render failed: {e}")
        return html.Div(f"Could not load data: {e}", style={"padding": 20, "color": "#dc2626"})


def serve_layout() -> html.Div:
    """Build the page fresh on every load.

    Each tab's content is embedded directly as that ``dcc.Tab``'s ``children`` so the
    Tabs component swaps content on click entirely client-side — no callback on
    ``tabs.value`` is involved. (A value→callback round trip for the content proved
    unreliable in the browser even though the server returns the right payload, so
    every tab showed the first-rendered one; rendering as children is the robust,
    canonical pattern.) Being a function, the layout is rebuilt per page load, so a
    long-running dashboard always reflects the latest pipeline run without a restart.
    """
    body = {"marginTop": 16}
    return html.Div(
        style={"background": "#f3f4f6", "minHeight": "100vh", "fontFamily": "Arial", "padding": 16},
        children=[
            html.H1("LLM Trader — Monitor", style={"color": "#111827", "marginBottom": 4}),
            html.Div("DuckDB-backed · recommendation rationale, method performance, and returns",
                     style={"color": "#6b7280", "marginBottom": 2}),
            html.Div("Tip: hover a column header or metric for its definition · click a header to sort (⇅, shift-click multi-sort) · type in a filter cell to search",
                     style={"color": "#9ca3af", "fontSize": 12, "marginBottom": 12}),
            _health_banner(),
            dcc.Tabs(
                id="tabs", value="rationale",
                persistence=True, persistence_type="session",  # keep the selected tab across reloads
                children=[
                    dcc.Tab(label="Recommendations & Rationale", value="rationale",
                            children=dcc.Loading(html.Div(_safe(_rationale_tab), style=body))),
                    dcc.Tab(label="Method Performance", value="methods",
                            children=dcc.Loading(html.Div(_safe(_methods_tab), style=body))),
                    dcc.Tab(label="Returns", value="returns",
                            children=dcc.Loading(html.Div(_safe(_returns_tab), style=body))),
                ],
            ),
        ],
    )


app.layout = serve_layout


# ── Tab 1: Recommendations & Rationale ─────────────────────────────────────

def _rationale_tab():
    runs = data.runs_df()
    if runs.empty:
        return html.Div("No runs recorded yet. Run the pipeline first.", style={"padding": 20})
    options = [
        {
            "label": f"{_fmt_et(getattr(r, 'started_at', None))} ET"
                     f"   ·   {getattr(r, 'market_mode', None) or '–'} / {getattr(r, 'macro_regime', None) or '–'}"
                     f"   ·   LLM: {getattr(r, 'llm_synthesis_provider', None) or '–'}",
            "value": r.run_id,
        }
        for r in runs.itertuples()
    ]
    return html.Div([
        html.Div(
            [html.Label("Run:  ", title="Pick which pipeline run to inspect. Each entry is one analysis run, shown as its Eastern start time · market mode / macro regime · the LLM used.",
                        style={"cursor": "help", "borderBottom": "1px dotted #cbd5e1"}),
             dcc.Dropdown(id="run-select", options=options, value=options[0]["value"],
                          clearable=False, style={"width": 560, "display": "inline-block"})],
            style={"marginBottom": 12},
        ),
        dcc.Loading(html.Div(id="rationale-body")),
    ])


@app.callback(Output("rationale-body", "children"), Input("run-select", "value"))
def _rationale_body(run_id):
    src = data.run_sources_df(run_id)
    recs = data.recommendations_df(run_id)

    ok_n = int(src["ok"].sum()) if not src.empty else 0
    chips = [
        html.Span(
            f"{'✓' if bool(r['ok']) else '✗'} {r['source_label']}",
            title=(r.get("error") or ("Succeeded — data fetched this run." if bool(r["ok"]) else "Failed.")),
            style={"display": "inline-block", "margin": "2px 10px 2px 0", "cursor": "help",
                   "color": figures.POS if bool(r["ok"]) else figures.NEG, "fontSize": 13},
        )
        for r in src.to_dict("records")
    ]

    recs_disp = recs.copy()
    if not recs_disp.empty:
        recs_disp["confidence"] = (recs_disp["confidence"].astype(float) * 100).round(0).astype("Int64")
        recs_disp["generated_at"] = recs_disp["generated_at"].map(_fmt_et)

    run_meta = data.run_row(run_id)
    syn = (run_meta["llm_synthesis_provider"] if run_meta is not None else None) or "–"
    sent = (run_meta["llm_sentiment_provider"] if run_meta is not None else None) or "–"

    return html.Div([
        html.Div(f"LLM — synthesis: {syn}   ·   sentiment: {sent}",
                 style={"color": "#374151", "fontSize": 14, "marginBottom": 12}),
        _h3(f"APIs used this run  ·  {ok_n}/{len(src)} succeeded",
            "Each external data source the pipeline called this run — green ✓ succeeded, red ✗ failed. Hover a chip for the error or status."),
        html.Div(chips or "No source records.", style={"marginBottom": 18}),
        _h3(f"Recommendations  ·  {len(recs_disp)} shown",
            "Every BUY/SELL/HOLD/WATCH the model produced this run. Green-tinted rows are actionable (paper-traded). Hover a column header for its definition."),
        dash_table.DataTable(
            data=recs_disp.to_dict("records"),
            columns=_columns(_REC_COL_SPEC),
            tooltip_header=_header_tooltips(_REC_COL_SPEC),
            filter_action="native",
            style_data_conditional=[
                {"if": {"filter_query": "{actionable} = true"}, "backgroundColor": "#ecfdf5"},
                {"if": {"filter_query": "{action} = BUY", "column_id": "action"}, "color": figures.POS, "fontWeight": "bold"},
                {"if": {"filter_query": "{action} = SELL", "column_id": "action"}, "color": figures.NEG, "fontWeight": "bold"},
            ],
            style_cell_conditional=[
                {"if": {"column_id": "ticker"}, "fontWeight": "bold"},
                {"if": {"column_id": "rationale"}, "minWidth": 260, "maxWidth": 520},
            ],
            **_TABLE_KW,
        ),
    ])


# ── Time-window toggle (shared by the Method Performance & Returns tabs) ─────
_WINDOW_OPTIONS = [
    {"label": "1 Week", "value": "7"},
    {"label": "1 Month", "value": "30"},
    {"label": "Inception", "value": "all"},
]


def _window_toggle(component_id: str) -> html.Div:
    """A 1-week / 1-month / inception selector. The tab's metrics and plots
    recompute against trades ENTERED within the chosen window ('Inception' = every
    trade ever). Defaults to Inception so the initial view shows the full book."""
    return html.Div(
        [
            html.Label("Window:  ",
                       title="Filter the metrics and plots in this tab to trades entered in the last week, the last month, or since inception (every trade).",
                       style={"cursor": "help", "borderBottom": "1px dotted #cbd5e1", "marginRight": 4}),
            dcc.RadioItems(
                id=component_id, options=_WINDOW_OPTIONS, value="all", inline=True,
                persistence=True, persistence_type="session",
                inputStyle={"marginLeft": 14, "marginRight": 4},
                labelStyle={"cursor": "pointer"},
            ),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": 12},
    )


def _window_days(value):
    """RadioItems value → window_days int, or None for inception (all trades)."""
    return None if value in (None, "all") else int(value)


def _window_label(value) -> str:
    """RadioItems value → human label for tile captions."""
    return {"7": "1 week", "30": "1 month"}.get(str(value), "inception")


# ── Trading-session toggle (RTH / extended / overnight) ──────────────────────
_SESSION_OPTIONS = [
    {"label": "All sessions", "value": "all"},
    {"label": "RTH", "value": "rth"},
    {"label": "Extended", "value": "extended"},
    {"label": "Overnight", "value": "overnight"},
]


def _session_toggle(component_id: str) -> html.Div:
    """RTH / extended-hours / overnight selector. Filters the tab's metrics and
    plots to trades ENTERED in that US-market session. Extended-hours trading
    is live (Phase 1): the scheduler trades 04:00–20:00 ET, so RTH and
    Extended populate side by side for comparison."""
    return html.Div(
        [
            html.Label("Session:  ",
                       title="Filter to trades entered during Regular hours (09:30–16:00 ET), Extended hours (pre-market 04:00–09:30 + after-hours 16:00–20:00), or Overnight (20:00–04:00). The bot trades RTH and Extended; Overnight entries only occur from manual off-schedule runs (snapped to the next session).",
                       style={"cursor": "help", "borderBottom": "1px dotted #cbd5e1", "marginRight": 4}),
            dcc.RadioItems(
                id=component_id, options=_SESSION_OPTIONS, value="all", inline=True,
                persistence=True, persistence_type="session",
                inputStyle={"marginLeft": 14, "marginRight": 4},
                labelStyle={"cursor": "pointer"},
            ),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": 12},
    )


def _session_value(value):
    """RadioItems value → session string ('rth'|'extended'|'overnight'), or None for all."""
    return None if value in (None, "all") else value


# ── Trade-source toggle (simulated ledger vs actual IBKR fills) ──────────────
_SOURCE_OPTIONS = [
    {"label": "Simulated (model)", "value": "sim"},
    {"label": "IBKR (actual fills)", "value": "broker"},
]


def _source_toggle(component_id: str) -> html.Div:
    """Two books, one toggle. Simulated = the strategy ledger (every decision at
    its decision price through the modeled cost stack). IBKR = only orders that
    actually filled, at real fill prices with real commissions."""
    return html.Div(
        [
            html.Label("Trades:  ",
                       title="Simulated (model): every decision the strategy made, priced at decision time with modeled spread + commission costs — strategy quality, independent of execution. "
                             "IBKR (actual fills): only orders that really filled at the broker, at actual average fill prices with the commissions actually charged — execution reality, no modeled costs. "
                             "The gap between the two views is the execution gap: slippage, unfilled or expired orders, and sizing rounding.",
                       style={"cursor": "help", "borderBottom": "1px dotted #cbd5e1", "marginRight": 4}),
            dcc.RadioItems(
                id=component_id, options=_SOURCE_OPTIONS, value="sim", inline=True,
                persistence=True, persistence_type="session",
                inputStyle={"marginLeft": 14, "marginRight": 4},
                labelStyle={"cursor": "pointer"},
            ),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": 12},
    )


def _usd(x, signed: bool = True) -> str:
    if x is None:
        return "–"
    try:
        return f"${x:+,.2f}" if signed else f"${x:,.2f}"
    except (TypeError, ValueError):
        return str(x)


# ── Tab 2: Method Performance ──────────────────────────────────────────────

def _methods_tab():
    # The LLM-models-used table is run-based (not trade-windowed), so it lives
    # outside the windowed body. The per-method performance section (bar + table)
    # is filled by _methods_body() against the selected time window.
    runs = data.runs_df()
    model_rows = _models_used_rows(runs) if not runs.empty else []
    models_table = dash_table.DataTable(
        data=model_rows,
        columns=[{"name": c, "id": c} for c in ["Role", "Model", "API", "Runs", "Calls"]],
        tooltip_header=_MODELS_HEADER_TIPS,
        **_TABLE_KW,
    ) if model_rows else html.Div("No runs recorded yet.", style={"color": "#6b7280"})

    return html.Div([
        _window_toggle("methods-window"),
        _session_toggle("methods-session"),
        dcc.Loading(html.Div(id="methods-body")),
        _h3("LLM models used (synthesis & sentiment)",
            "Which exact LLMs actually ran across all recorded pipeline runs — the final-call 'synthesis' model and the per-ticker 'sentiment' model — including any DeepSeek or rule-based fallbacks. Not affected by the window toggle above (it's run-based, not trade-based). Hover a column header for details."),
        models_table,
    ])


@app.callback(Output("methods-body", "children"),
              Input("methods-window", "value"), Input("methods-session", "value"))
def _methods_body(window_value, session_value):
    return _safe(lambda: _methods_perf_section(_window_days(window_value), _session_value(session_value)))


def _methods_perf_section(window_days, session=None):
    perf = data.performance(window_days=window_days, session=session)
    solo = perf.get("solo_method_perf") or {}
    labels = perf.get("method_labels") or {}
    order = perf.get("method_order_by_winrate") or list(solo.keys())

    rows = []
    for m in order:
        overall = (solo.get(m) or {}).get("overall") or {}
        if not overall:
            continue
        rows.append({
            "Method": labels.get(m, m),
            "Win rate %": round(overall["win_rate"], 1) if overall.get("win_rate") is not None else None,
            "Trades": overall.get("trades", overall.get("n")),
            "Avg return %": round(overall["avg_return"], 2) if overall.get("avg_return") is not None else None,
        })

    # Per-LLM rows — EVERY trade each engine recommended (executed or not):
    # one pseudo-trade per (engine, ticker, day), entered at the recorded
    # recommendation-time price and marked at the latest cached close, so
    # LLM APIs are compared on their full call stream rather than the few
    # recommendations that survived the trading gates.
    llm = perf.get("llm_perf") or {}
    for role, label in (("synthesis", "Synthesis LLM"), ("sentiment", "Sentiment LLM")):
        by_model = llm.get(role) or {}
        for model, st in sorted(by_model.items(), key=lambda kv: -(kv[1].get("trades") or 0)):
            rows.append({
                "Method": f"{label} · {model}",
                "Win rate %": st.get("win_rate"),
                "Trades": st.get("trades"),
                "Avg return %": st.get("avg_return"),
            })

    # Held-positions prompt A/B — exit outcomes grouped by the per-run coin
    # flip (does telling the LLM what the system holds improve closes?).
    # Each trade's closing run stamped exit_hold_prompt; pre-experiment
    # closes carry no stamp and are excluded from both rows.
    hp = perf.get("hold_prompt_eval") or {}
    for key, label in (("on", "Exit eval · hold-prompt ON"),
                       ("off", "Exit eval · hold-prompt OFF")):
        st = hp.get(key)
        if st and st.get("trades"):
            rows.append({
                "Method": label,
                "Win rate %": st.get("win_rate"),
                "Trades": st.get("trades"),
                "Avg return %": st.get("avg_return"),
            })

    table = dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in ["Method", "Win rate %", "Trades", "Avg return %"]],
        tooltip_header=_METHOD_HEADER_TIPS,
        style_data_conditional=[
            {"if": {"filter_query": '{Method} contains "LLM"'}, "backgroundColor": "#eef2ff"},
            {"if": {"filter_query": '{Method} contains "hold-prompt"'}, "backgroundColor": "#fdf4ff"},
        ],
        **_TABLE_KW,
    ) if rows else html.Div("No per-method stats in this window yet.", style={"color": "#6b7280"})

    return html.Div([
        dcc.Graph(figure=figures.method_winrate_fig(perf)),
        _h3("Model evaluation — signal methods (solo simulation) & LLM engines",
            "Method rows: how each signal method would have performed deciding alone (each closed trade re-simulated as if only that method set the direction). "
            "Highlighted LLM rows: every BUY/SELL the engine recommended — executed or simulated — entered at the recommendation-time price, marked at the latest close, "
            "deduped to the engine's last call per ticker per day. The 50/50 A/B routing flip gives each engine its own runs to be judged on. Hover the column headers for details."),
        table,
    ])


# ── Tab 3: Returns ─────────────────────────────────────────────────────────

def _held_disp(t: dict) -> str:
    """Wall-clock holding time as a compact ``2d 5h`` / ``6h`` / ``45m`` string.

    Open positions measure entry → now; closed ones entry → exit. Falls back
    to the trading-days count (``Nd``) for legacy date-only rows. An entry
    timestamp in the future (overnight decisions snap execution to the next
    session open) clamps to 0.
    """
    def _parse(iso):
        dt = datetime.fromisoformat(str(iso))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    try:
        start = _parse(t.get("entry_datetime"))
        end = _parse(t.get("exit_datetime")) if t.get("status") == "CLOSED" \
            else datetime.now(timezone.utc)
    except (TypeError, ValueError):
        d = t.get("days_held")
        return f"{int(d)}d" if d is not None else ""
    minutes = max(0, int((end - start).total_seconds() // 60))
    days, rem = divmod(minutes, 1440)
    hours, mins = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h" if hours else f"{days}d"
    return f"{hours}h" if hours else f"{mins}m"


def _ibkr_leg_disp(t: dict, prefix: str) -> str:
    """Compact label for one broker order leg ('broker_' / 'broker_exit_'):
    did the order really go through? Raw statuses are mapped to a handful of
    glyph-led states so the column scans at a glance."""
    status = str(t.get(f"{prefix}status") or "").strip()
    if not status:
        if prefix == "broker_exit_" and t.get("broker_order_id") and t.get("status") == "CLOSED":
            return "⏳ pending"     # ledger closed; the exit goes out next sync
        return "" if (prefix == "broker_exit_" and t.get("status") == "OPEN") else "–"
    qty = int(t.get(f"{prefix}fill_qty") or 0)
    req = int(t.get(f"{prefix}requested_qty") or 0)
    if status == "Filled":
        return f"✓ filled {qty}" if qty else "✓ filled"
    if status in ("Submitted", "PreSubmitted", "PendingSubmit"):
        return f"⏳ partial {qty}/{req}" if qty else "⏳ working"
    if status in ("STALE_CANCELLED", "EXPIRED"):
        return "↻ re-anchoring"     # tick-scoped cancel; resubmits at the current mark
    if status == "DUPLICATE_REF_NOT_SUBMITTED":
        return "– duplicate, not sent"
    if status == "NOTHING_TO_CLOSE":
        return "– nothing held"
    if status == "DRYRUN":
        return "dry-run"
    if status.upper().startswith("SKIPPED"):
        return "– " + status.replace("_", " ").lower()
    if status in ("Cancelled", "ApiCancelled", "Inactive"):
        reason = t.get(f"{prefix}cancel_reason")
        return f"✕ cancelled ({reason})" if reason else "✕ cancelled"
    if status in ("RESTORED_NOT_SUBMITTED", "RESTORED_ADOPTED"):
        return "restored"
    return f"✗ {status}"            # rejects / connection failures / raw errors


def _trades_table(trades: list):
    if not trades:
        return html.Div("None.", style={"color": "#6b7280", "marginBottom": 12})
    df = pd.DataFrame(trades)
    # Show entry/exit as Eastern-time date + time (HH:MM); fall back to the
    # date-only field for any legacy row missing the full datetime.
    df["entry_dt"] = [_fmt_et(t.get("entry_datetime")) or (t.get("entry_date") or "") for t in trades]
    df["exit_dt"] = [_fmt_et(t.get("exit_datetime")) or (t.get("exit_date") or "") for t in trades]
    # Entry session (rth / extended / overnight). Stamped on new trades;
    # derived from entry_datetime for legacy rows (date-only → rth).
    from src.performance.tracker import _trade_session
    df["session"] = [t.get("entry_session") or _trade_session(t) for t in trades]
    df["held"] = [_held_disp(t) for t in trades]
    # IBKR order-status columns — simulated view only: the IBKR view contains
    # filled orders by construction, so the columns would be all-✓ noise there.
    if not trades[0].get("broker_view"):
        df["broker_entry"] = [_ibkr_leg_disp(t, "broker_") for t in trades]
        df["broker_exit"] = [_ibkr_leg_disp(t, "broker_exit_") for t in trades]
    spec = [t for t in _TRADE_COL_SPEC if t[0] in df.columns]
    df = df[[s[0] for s in spec]]
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=_columns(spec),
        tooltip_header=_header_tooltips(spec),
        filter_action="native",
        style_data_conditional=[
            {"if": {"filter_query": "{return_pct} > 0", "column_id": "return_pct"}, "color": figures.POS},
            {"if": {"filter_query": "{return_pct} < 0", "column_id": "return_pct"}, "color": figures.NEG},
        ],
        **_TABLE_KW,
    )


def _returns_tab():
    return html.Div([
        _source_toggle("returns-source"),
        _window_toggle("returns-window"),
        _session_toggle("returns-session"),
        dcc.Loading(html.Div(id="returns-body")),
    ])


@app.callback(Output("returns-body", "children"),
              Input("returns-window", "value"), Input("returns-session", "value"),
              Input("returns-source", "value"))
def _returns_body(window_value, session_value, source_value):
    if (source_value or "sim") == "broker":
        return _safe(lambda: _broker_returns_section(window_value, session_value))
    return _safe(lambda: _returns_section(window_value, session_value))


def _broker_returns_section(window_value, session_value=None):
    """The IBKR view: what actually executed, at actual prices and commissions.
    Dollar P&L leads — real fills have real notionals, so percentages alone
    hide sizing. No modeled costs anywhere in this view."""
    from datetime import date, timedelta
    from src.performance.broker_view import summarize_broker_trades

    trades = data.broker_trades()
    wd = _window_days(window_value)
    if wd:
        cutoff = (date.today() - timedelta(days=wd)).isoformat()
        trades = [t for t in trades if str(t.get("entry_date") or "") >= cutoff]
    sess = _session_value(session_value)
    if sess:
        trades = [t for t in trades if (t.get("entry_session") or "rth") == sess]
    if not trades:
        return html.Div(
            "No IBKR fills recorded in this window yet — either broker_mode is "
            "off/dry_run, or no submitted order has filled.",
            style={"color": "#6b7280", "padding": 20})

    s = summarize_broker_trades(trades)
    # "Size ×" is a sim concept; in this view the dedicated Shares/Notional
    # columns carry the real sizing, so drop the multiplier from the tables.
    strip = lambda ts: [{k: v for k, v in t.items() if k != "position_size_multiplier"} for t in ts]
    open_trades = strip(sorted((t for t in trades if t["status"] == "OPEN"),
                               key=lambda t: str(t.get("entry_datetime") or ""), reverse=True))
    closed_trades = strip(sorted((t for t in trades if t["status"] == "CLOSED"),
                                 key=lambda t: str(t.get("exit_datetime") or ""), reverse=True))

    cards = html.Div(
        [
            _kpi("Realized P&L", _usd(s.get("realized_pnl_usd")),
                 figures.POS if (s.get("realized_pnl_usd") or 0) >= 0 else figures.NEG,
                 tooltip="Sum over closed fills: signed price move × shares filled, minus the commissions IBKR actually charged. No modeled costs."),
            _kpi("Open P&L", _usd(s.get("unrealized_pnl_usd")),
                 figures.POS if (s.get("unrealized_pnl_usd") or 0) >= 0 else figures.NEG,
                 tooltip="Mark-to-market of positions still held at the broker, vs their actual entry fills, minus entry commissions (exit cost unknown until it happens)."),
            _kpi("Win rate", _pct(s.get("win_rate")),
                 tooltip="Share of CLOSED broker round-trips with a positive net return on actual fills."),
            _kpi("Avg return", _pct(s.get("avg_return"), signed=True),
                 tooltip="Mean per-round-trip % return on actual fill prices net of actual commissions, equal-weighted."),
            _kpi("Commissions", _usd(s.get("commissions_usd"), signed=False),
                 tooltip="Total commissions IBKR actually charged on these fills (exit legs counted once filled)."),
            _kpi("Closed / Open", f"{s.get('closed', 0)} / {s.get('open', 0)}",
                 tooltip="Broker round-trips completed vs positions genuinely still held at the broker (a ledger-closed trade whose exit hasn't filled is still OPEN here)."),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
    )

    return html.Div([
        cards,
        _h3("Open broker positions",
            "Shares genuinely held at IBKR right now (entry filled; exit not filled yet — even if the simulated ledger already closed the trade), marked at the latest price."),
        _trades_table(open_trades),
        _h3("Closed broker round-trips",
            "Entry and exit both filled at IBKR. Returns are computed on the actual average fill prices, net of the commissions actually charged."),
        _trades_table(closed_trades),
    ])


def _returns_section(window_value, session_value=None):
    perf = data.performance(window_days=_window_days(window_value), session=_session_value(session_value))
    stats = perf.get("stats") or {}
    pm = perf.get("portfolio_metrics") or {}
    wlabel = _window_label(window_value)

    compound = pm.get("compound_inception", stats.get("compound_return"))
    cards = html.Div(
        [
            _kpi(f"Compound ({wlabel})", _pct(compound, signed=True),
                 figures.POS if (compound or 0) >= 0 else figures.NEG,
                 tooltip="Path-faithful compound return over the selected window: each day's capital-weighted return across active positions, chained over real closing prices. Counts trades ENTERED within the window; open positions are included at their live mark."),
            _kpi("Win rate", _pct(stats.get("win_rate")),
                 tooltip="Share of trades with a positive spread-adjusted return. A flat round-trip is a loss (you pay the bid-ask spread). Open positions count at their live mark."),
            _kpi("Avg return", _pct(stats.get("avg_return"), signed=True),
                 tooltip="Mean per-trade % return, equal-weighted across all trades in the window (open trades at their live mark)."),
            _kpi("Weighted avg", _pct(stats.get("weighted_avg_return"), signed=True),
                 tooltip="Per-trade % return weighted by position size (the confidence-tier multiplier), so larger positions count more."),
            _kpi("Best", _pct(stats.get("best"), signed=True), figures.POS,
                 tooltip="Best single-trade % return in the window."),
            _kpi("Worst", _pct(stats.get("worst"), signed=True), figures.NEG,
                 tooltip="Worst single-trade % return in the window."),
            _kpi("Closed / Open", f"{stats.get('total_closed', 0)} / {stats.get('total_open', 0)}",
                 tooltip="Number of closed (realised) trades vs. positions currently open, within the selected window."),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
    )

    return html.Div([
        cards,
        dcc.Graph(figure=figures.equity_curve_fig(perf)),
        _h3("Open positions", "Positions currently held, marked to the latest price — the return shown is live mark-to-market ('what if you closed now'). Filtered to the selected entry window."),
        _trades_table(perf.get("open_trades") or []),
        _h3("Closed trades", "Realised round-trips, with their final spread-adjusted return. Filtered to the selected entry window."),
        _trades_table(perf.get("closed_trades") or []),
    ])


def _serve_once(host: str, port: int) -> None:
    """Serve the WSGI app once.

    Prefers ``waitress`` — a production-grade, multi-threaded, cross-platform WSGI
    server (the right choice on Windows, where gunicorn does not run). It stays
    responsive for always-on use, recycles stuck connections, and won't fall over
    the way the Werkzeug development server does. Falls back to the Dash dev server
    only when waitress isn't installed.
    """
    try:
        from waitress import serve
    except ImportError:
        logger.warning(
            "waitress not installed — using the Dash dev server, which is less "
            "robust for always-on use. Install it with `pip install waitress`."
        )
        app.run(host=host, port=port, debug=False)
        return

    # A few worker threads so a slow performance() render can't block the whole UI;
    # channel_timeout reaps connections that go quiet instead of leaking them.
    serve(app.server, host=host, port=port, threads=8, channel_timeout=120)


def run() -> None:
    """Run the dashboard, auto-restarting on an unexpected crash so it stays alive."""
    host, port = settings.dashboard_host, settings.dashboard_port
    logger.info(f"Dashboard starting at http://{host}:{port}  (Ctrl+C to stop)")

    backoff = 2
    while True:
        try:
            _serve_once(host, port)
            return  # clean shutdown
        except KeyboardInterrupt:
            logger.info("Dashboard stopped.")
            return
        except Exception as e:  # pragma: no cover — last-resort supervisor
            logger.error(f"Dashboard server crashed: {e!r} — restarting in {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)


if __name__ == "__main__":
    run()
