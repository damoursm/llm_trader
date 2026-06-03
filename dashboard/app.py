"""Plotly Dash monitoring dashboard — rationale · method performance · returns.

Single source of truth is DuckDB (read-only here). Launch with:

    python main.py --dashboard
"""

from __future__ import annotations

import time
from datetime import datetime

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
    ("entry_price", "Entry $", _NUM2, "Fill price at entry (the bid-ask spread is applied in the return, not here)."),
    ("exit_dt", "Exit (ET)", None, "When the position was closed, in US/Eastern time. Blank while still open."),
    ("exit_price", "Exit $", _NUM2, "Fill price at exit. Blank while the position is open."),
    ("return_pct", "Return %", _NUM2, "Spread-adjusted % return. For OPEN positions this is the live mark-to-market — 'what if you closed right now'."),
    ("position_size_multiplier", "Size ×", _NUM2, "Capital weight from the confidence tier (1.0× / 1.5× / 2.0×), after the correlation haircut."),
    ("status", "Status", None, "OPEN (held, live mark) or CLOSED (realised)."),
]

# Method Performance table — header explanations (table is built inline below).
_METHOD_HEADER_TIPS = {
    "Method": "The signal method (e.g. news sentiment, technical, momentum).",
    "Win rate %": "Solo simulation: for each closed trade, what if ONLY this method had decided the direction? This is the share it would have won.",
    "Trades": "How many closed trades this method had a view on (|score| ≥ 0.10).",
    "Avg return %": "Average % return of those solo-simulated trades.",
}


# ── LLM model usage (Method Performance tab → "LLM models used" section) ──────
# Exact model ids per provider. Sources of truth in the code:
#   synthesis Claude   → settings.analyst_model
#   synthesis DeepSeek → claude_analyst._DEEPSEEK_ANALYST_MODEL  ("deepseek-chat", V3)
#   sentiment DeepSeek → sentiment.DEEPSEEK_MODEL                ("deepseek-chat", V3)
#   sentiment Claude   → sentiment.HAIKU_MODEL                   ("claude-haiku-4-5-20251001")
_PROVIDER_LABEL = {
    "anthropic": "Anthropic (Claude)", "deepseek": "DeepSeek",
    "rule-based": "Rule-based", "none": "—", "": "—",
}
_SENTIMENT_MODEL = {
    "deepseek": "deepseek-chat",
    "anthropic": "claude-haiku-4-5-20251001",
    "none": "(none — cached / no LLM call)",
}


def _synthesis_model(provider) -> str:
    """Exact model id that produced the final synthesis for a given provider."""
    p = (provider or "").lower()
    if p == "anthropic":
        return settings.analyst_model           # the configured Claude model
    if p == "deepseek":
        return "deepseek-chat"                   # DeepSeek V3 analyst fallback
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
        tooltip_header=_METHOD_HEADER_TIPS,
        **_TABLE_KW,
    ) if rows else html.Div("No per-method stats yet.", style={"color": "#6b7280"})

    runs = data.runs_df()
    model_rows = _models_used_rows(runs) if not runs.empty else []
    models_table = dash_table.DataTable(
        data=model_rows,
        columns=[{"name": c, "id": c} for c in ["Role", "Model", "API", "Runs", "Calls"]],
        tooltip_header=_MODELS_HEADER_TIPS,
        **_TABLE_KW,
    ) if model_rows else html.Div("No runs recorded yet.", style={"color": "#6b7280"})

    return html.Div([
        dcc.Graph(figure=figures.method_winrate_fig(perf)),
        _h3("Per-method stats (solo simulation)",
            "How each signal method would have performed on its own. 'Solo simulation' = for each closed trade, ask what the result would be if only this method had decided the direction. Hover the column headers for details."),
        table,
        _h3("LLM models used (synthesis & sentiment)",
            "Which exact LLMs actually ran across all recorded pipeline runs — the final-call 'synthesis' model and the per-ticker 'sentiment' model — including any DeepSeek or rule-based fallbacks. Hover a column header for details."),
        models_table,
    ])


# ── Tab 3: Returns ─────────────────────────────────────────────────────────

def _trades_table(trades: list):
    if not trades:
        return html.Div("None.", style={"color": "#6b7280", "marginBottom": 12})
    df = pd.DataFrame(trades)
    # Show entry/exit as Eastern-time date + time (HH:MM); fall back to the
    # date-only field for any legacy row missing the full datetime.
    df["entry_dt"] = [_fmt_et(t.get("entry_datetime")) or (t.get("entry_date") or "") for t in trades]
    df["exit_dt"] = [_fmt_et(t.get("exit_datetime")) or (t.get("exit_date") or "") for t in trades]
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
    perf = data.performance()
    stats = perf.get("stats") or {}
    pm = perf.get("portfolio_metrics") or {}

    compound = pm.get("compound_inception", stats.get("compound_return"))
    cards = html.Div(
        [
            _kpi("Compound (inception)", _pct(compound, signed=True),
                 figures.POS if (compound or 0) >= 0 else figures.NEG,
                 tooltip="Path-faithful compound return over every trade since inception: each day's capital-weighted return across active positions, chained over real closing prices. Open positions are included at their live mark."),
            _kpi("Win rate", _pct(stats.get("win_rate")),
                 tooltip="Share of trades with a positive spread-adjusted return. A flat round-trip is a loss (you pay the bid-ask spread). Open positions count at their live mark."),
            _kpi("Avg return", _pct(stats.get("avg_return"), signed=True),
                 tooltip="Mean per-trade % return, equal-weighted across all trades (open trades at their live mark)."),
            _kpi("Weighted avg", _pct(stats.get("weighted_avg_return"), signed=True),
                 tooltip="Per-trade % return weighted by position size (the confidence-tier multiplier), so larger positions count more."),
            _kpi("Best", _pct(stats.get("best"), signed=True), figures.POS,
                 tooltip="Best single-trade % return in the book."),
            _kpi("Worst", _pct(stats.get("worst"), signed=True), figures.NEG,
                 tooltip="Worst single-trade % return in the book."),
            _kpi("Closed / Open", f"{stats.get('total_closed', 0)} / {stats.get('total_open', 0)}",
                 tooltip="Number of closed (realised) trades vs. positions currently open."),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
    )

    return html.Div([
        cards,
        dcc.Graph(figure=figures.equity_curve_fig(perf)),
        _h3("Open positions", "Positions currently held, marked to the latest price — the return shown is live mark-to-market ('what if you closed now')."),
        _trades_table(perf.get("open_trades") or []),
        _h3("Closed trades", "Realised round-trips, with their final spread-adjusted return."),
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
