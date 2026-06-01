"""Plotly Dash monitoring dashboard — rationale · method performance · returns.

Single source of truth is DuckDB (read-only here). Launch with:

    python main.py --dashboard
"""

from __future__ import annotations

import pandas as pd
from dash import Dash, Input, Output, dash_table, dcc, html
from loguru import logger

from config import settings
from dashboard import data, figures

app = Dash(__name__, title="LLM Trader Monitor", suppress_callback_exceptions=True)
server = app.server  # for WSGI deployment if ever needed

_TABLE_KW = dict(
    style_table={"overflowX": "auto"},
    style_cell={
        "fontFamily": "Arial", "fontSize": 13, "padding": "6px",
        "textAlign": "left", "whiteSpace": "normal", "height": "auto",
        "maxWidth": 460,
    },
    style_header={"backgroundColor": "#f9fafb", "fontWeight": "bold"},
)


def _kpi(label: str, value: str, color: str = "#111827") -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"color": "#6b7280", "fontSize": 12}),
            html.Div(value, style={"color": color, "fontSize": 22, "fontWeight": "bold"}),
        ],
        style={
            "padding": "10px 16px", "background": "white", "borderRadius": 8,
            "boxShadow": "0 1px 3px rgba(0,0,0,0.1)", "minWidth": 130, "margin": 6,
        },
    )


def _pct(x, signed: bool = False) -> str:
    if x is None:
        return "–"
    try:
        return f"{x:+.2f}%" if signed else f"{x:.1f}%"
    except (TypeError, ValueError):
        return str(x)


app.layout = html.Div(
    style={"background": "#f3f4f6", "minHeight": "100vh", "fontFamily": "Arial", "padding": 16},
    children=[
        html.H1("LLM Trader — Monitor", style={"color": "#111827", "marginBottom": 4}),
        html.Div("DuckDB-backed · recommendation rationale, method performance, and returns",
                 style={"color": "#6b7280", "marginBottom": 12}),
        dcc.Tabs(id="tabs", value="rationale", children=[
            dcc.Tab(label="Recommendations & Rationale", value="rationale"),
            dcc.Tab(label="Method Performance", value="methods"),
            dcc.Tab(label="Returns", value="returns"),
        ]),
        dcc.Loading(html.Div(id="tab-content", style={"marginTop": 16})),
        dcc.Interval(id="refresh", interval=120_000, n_intervals=0),
    ],
)


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("refresh", "n_intervals"),
)
def render_tab(tab, _n):
    try:
        if tab == "methods":
            return _methods_tab()
        if tab == "returns":
            return _returns_tab()
        return _rationale_tab()
    except FileNotFoundError:
        return html.Div("No database yet. Run the pipeline (or `python -m src.db.migrate`) first.",
                        style={"padding": 20, "color": "#dc2626"})
    except Exception as e:  # keep the dashboard alive on any data hiccup
        logger.warning(f"[dashboard] tab render failed: {e}")
        return html.Div(f"Could not load data: {e}", style={"padding": 20, "color": "#dc2626"})


# ── Tab 1: Recommendations & Rationale ─────────────────────────────────────

def _rationale_tab():
    runs = data.runs_df()
    if runs.empty:
        return html.Div("No runs recorded yet. Run the pipeline first.", style={"padding": 20})
    options = [
        {
            "label": f"{r.run_id}   ·   {getattr(r, 'market_mode', None) or '–'}"
                     f" / {getattr(r, 'macro_regime', None) or '–'}"
                     f"   ·   LLM: {getattr(r, 'llm_synthesis_provider', None) or '–'}",
            "value": r.run_id,
        }
        for r in runs.itertuples()
    ]
    return html.Div([
        html.Div(
            [html.Label("Run:  "),
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
            title=(r.get("error") or ""),
            style={"display": "inline-block", "margin": "2px 10px 2px 0",
                   "color": figures.POS if bool(r["ok"]) else figures.NEG, "fontSize": 13},
        )
        for r in src.to_dict("records")
    ]

    recs_disp = recs.copy()
    if not recs_disp.empty:
        recs_disp["confidence"] = (recs_disp["confidence"].astype(float) * 100).round(0).astype("Int64")

    run_meta = data.run_row(run_id)
    syn = (run_meta["llm_synthesis_provider"] if run_meta is not None else None) or "–"
    sent = (run_meta["llm_sentiment_provider"] if run_meta is not None else None) or "–"

    return html.Div([
        html.Div(f"LLM — synthesis: {syn}   ·   sentiment: {sent}",
                 style={"color": "#374151", "fontSize": 14, "marginBottom": 12}),
        html.H3(f"APIs used this run  ·  {ok_n}/{len(src)} succeeded"),
        html.Div(chips or "No source records.", style={"marginBottom": 18}),
        html.H3(f"Recommendations  ·  {len(recs_disp)} shown"),
        dash_table.DataTable(
            data=recs_disp.to_dict("records"),
            columns=[{"name": c, "id": c} for c in recs_disp.columns],
            page_size=15,
            style_data_conditional=[
                {"if": {"filter_query": "{actionable} = true"}, "backgroundColor": "#ecfdf5"},
                {"if": {"filter_query": "{action} = BUY", "column_id": "action"}, "color": figures.POS, "fontWeight": "bold"},
                {"if": {"filter_query": "{action} = SELL", "column_id": "action"}, "color": figures.NEG, "fontWeight": "bold"},
            ],
            **_TABLE_KW,
        ),
    ])


# ── Tab 2: Method Performance ──────────────────────────────────────────────

def _methods_tab():
    perf = data.performance()
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

    table = dash_table.DataTable(
        data=rows,
        columns=[{"name": c, "id": c} for c in ["Method", "Win rate %", "Trades", "Avg return %"]],
        sort_action="native", page_size=25, **_TABLE_KW,
    ) if rows else html.Div("No per-method stats yet.", style={"color": "#6b7280"})

    return html.Div([
        dcc.Graph(figure=figures.method_winrate_fig(perf)),
        html.H3("Per-method stats (solo simulation)"),
        table,
    ])


# ── Tab 3: Returns ─────────────────────────────────────────────────────────

_TRADE_VIEW_COLS = [
    "ticker", "action", "direction", "entry_date", "entry_price",
    "exit_date", "exit_price", "return_pct", "position_size_multiplier", "status",
]


def _trades_table(trades: list):
    if not trades:
        return html.Div("None.", style={"color": "#6b7280", "marginBottom": 12})
    df = pd.DataFrame(trades)
    cols = [c for c in _TRADE_VIEW_COLS if c in df.columns]
    df = df[cols]
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in cols],
        page_size=15, sort_action="native",
        style_data_conditional=[
            {"if": {"filter_query": "{return_pct} > 0", "column_id": "return_pct"}, "color": figures.POS},
            {"if": {"filter_query": "{return_pct} < 0", "column_id": "return_pct"}, "color": figures.NEG},
        ],
        **_TABLE_KW,
    )


def _returns_tab():
    perf = data.performance()
    stats = perf.get("stats") or {}
    pm = perf.get("portfolio_metrics") or {}

    compound = pm.get("compound_inception", stats.get("compound_return"))
    cards = html.Div(
        [
            _kpi("Compound (inception)", _pct(compound, signed=True),
                 figures.POS if (compound or 0) >= 0 else figures.NEG),
            _kpi("Win rate", _pct(stats.get("win_rate"))),
            _kpi("Avg return", _pct(stats.get("avg_return"), signed=True)),
            _kpi("Weighted avg", _pct(stats.get("weighted_avg_return"), signed=True)),
            _kpi("Best", _pct(stats.get("best"), signed=True), figures.POS),
            _kpi("Worst", _pct(stats.get("worst"), signed=True), figures.NEG),
            _kpi("Closed / Open", f"{stats.get('total_closed', 0)} / {stats.get('total_open', 0)}"),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
    )

    return html.Div([
        cards,
        dcc.Graph(figure=figures.equity_curve_fig(perf)),
        html.H3("Open positions"),
        _trades_table(perf.get("open_trades") or []),
        html.H3("Closed trades"),
        _trades_table(perf.get("closed_trades") or []),
    ])


def run() -> None:
    logger.info(
        f"Dashboard starting at http://{settings.dashboard_host}:{settings.dashboard_port}  "
        f"(Ctrl+C to stop)"
    )
    app.run(host=settings.dashboard_host, port=settings.dashboard_port, debug=False)


if __name__ == "__main__":
    run()
