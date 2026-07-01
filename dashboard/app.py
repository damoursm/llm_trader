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
    """Prominent banners for the latest run: failed data sources (red) and a
    price-provenance alert (amber) when a new trade entered far from the run
    snapshot. Returns an empty Div when all good."""
    blocks = []

    try:
        failures = data.latest_run_failures()
    except Exception as e:
        logger.debug(f"[dashboard] health banner skipped: {e}")
        failures = []
    if failures:
        items = []
        for f in failures:
            lbl = f.get("source_label") or "?"
            err = f.get("error")
            items.append(f"{lbl} — {err}" if err else lbl)
        blocks.append(html.Div(
            [
                html.B(f"⚠ {len(failures)} data source(s) failed in the latest run"),
                html.Div(" · ".join(items),
                         style={"marginTop": 4, "fontSize": 12, "whiteSpace": "normal"}),
            ],
            style={"background": "#fef2f2", "border": "1px solid #fecaca", "color": "#b91c1c",
                   "borderRadius": 8, "padding": "10px 14px", "marginBottom": 12},
        ))

    try:
        pp = (data.latest_gate_diag() or {}).get("price_provenance")
    except Exception as e:
        logger.debug(f"[dashboard] provenance banner skipped: {e}")
        pp = None
    if pp and pp.get("down"):
        blocks.append(html.Div(
            [
                html.B("🔔 Price provenance alert"),
                html.Div((pp.get("message") or "") + " — see the Execution tab.",
                         style={"marginTop": 4, "fontSize": 12, "whiteSpace": "normal"}),
            ],
            style={"background": "#fffbeb", "border": "1px solid #fcd34d", "color": "#92400e",
                   "borderRadius": 8, "padding": "10px 14px", "marginBottom": 12},
        ))

    return html.Div(blocks) if blocks else html.Div()


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
    ("time_horizon", "Horizon (LLM)", None, "The LLM's intended holding window (SHORT-TERM / SWING / POSITION). Capped at trade time to the mechanical edge horizon — the LLM may confirm or shorten it, never lengthen."),
    ("target_horizon", "Edge horizon", None, "Horizon synthesis: the cost-aware holding horizon (30m/3h/6h/1d/3d/1w/2w/1m) whose net-of-cost expected gross return is highest, from each method's MEASURED per-horizon IC (sign-aware). Blank when the IC panel is too thin or horizon synthesis is off."),
    ("horizon_net_edge_pct", "Net edge %", _NUM2, "Expected GROSS return at the edge horizon minus the round-trip cost hurdle. Positive = the edge clears costs at that horizon; ≤0 means no horizon is worth trading (prefer WATCH/HOLD)."),
    ("expected_move_pct", "Exp move %", _NUM2, "Expected FAVOURABLE move (magnitude, gross, pre-cost) at the target horizon — how far the name is expected to travel in the position's direction. The 'upside' the selection maximises."),
    ("upside_score", "Upside", _NUM2, "Selection rank key = conviction (probability of moving that way) × expected move (magnitude) × market-alignment factor. The biggest expected favourable mover in the regime's direction ranks highest; counter-regime names are haircut."),
    ("market_aligned", "Market", None, "Is the position aligned with the macro-regime market direction? aligned = with the regime (beta tailwind), counter = against it (haircut), neutral = no strong regime call. The regime layer owns the market direction; selection amplifies the biggest mover in it."),
    ("shadow_target_horizon", "Shadow horizon", None, "SHADOW (not yet live): horizon from the direction-aware, MARKET-NEUTRAL edge curve — each method weighted by its per-side (bull/bear) skill on returns net of SPY. Compare against 'Edge horizon' to see where direction-conditioning + drift-removal change the call."),
    ("shadow_direction", "Shadow dir", None, "SHADOW: the direction the market-neutral curve favours. When it DISAGREES with 'Direction', a method that is anti-predictive on this side has been flipped — the disagreement is the thing to watch before promoting the shadow curve."),
    ("shadow_horizon_net_edge_pct", "Shadow net %", _NUM2, "SHADOW: market-relative (alpha over SPY) net edge at the shadow horizon, after the cost hurdle. Smaller than 'Net edge %' by construction (market drift removed) — that gap is how much of the live edge was just beta/drift."),
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
    ("target_horizon", "Target horizon", None, "Horizon synthesis: the cost-aware holding horizon the position was opened for (e.g. 6h, 1w), capped to the LLM's call. Drives the matched exit time-stop — once held past this window the position must stay strongly confirmed to keep running. Blank for trades opened before horizon synthesis."),
    ("return_pct", "Return %", _NUM2, "Spread-adjusted % return. For OPEN positions this is the live mark-to-market — 'what if you closed right now'."),
    ("position_size_multiplier", "Size ×", _NUM2, "Capital weight from the confidence tier (1.0× / 1.5× / 2.0×), after the correlation haircut."),
    ("filled_notional_usd", "Notional $", _NUM2, "Actual dollars at risk: filled shares × average fill price (real-executions view only)."),
    ("status", "Status", None, "OPEN (held, live mark) or CLOSED (realised)."),
    ("exit_reason", "Exit reason", None, "Why the position closed: llm_signal_flipped / llm_confidence_loss (the opener's fresh re-judgment), horizon_expired (held past its target-horizon window without strong re-confirmation — the matched exit), macro_regime_exit, intraday_reversal, or a signal-decay backstop. Blank while open."),
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

# Macro Performance table — header explanations (aggregated decision layers).
_MACRO_HEADER_TIPS = {
    "Layer": "The aggregated decision layer being judged: 'LLM Synthesis' = the final BUY/SELL caller (all engines combined; the per-engine split is in the Model Evaluation table below), 'Aggregator' = the mechanical combined signal (the weighted blend of all method scores), or 'Bundle · X' = one method family (e.g. Technical, Options) voting by the sign of its summed scores. Each layer is scored on its OWN full stream of directional calls.",
    "Win rate %": "Share of that layer's directional calls currently positive — counting EVERY call it made (actionable or not, executed or not), not just the trades that survived the gates.",
    "Trades": "Number of directional calls the layer made, deduped to its last call per ticker per day (same rule as the LLM rows below).",
    "Avg return %": "Average forward % return across those calls: snapshot price at the call → latest cached close, net of the modeled one-way cost (so a brand-new call starts slightly negative, like a real position).",
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
                    dcc.Tab(label="Execution", value="execution",
                            children=dcc.Loading(html.Div(_safe(_execution_tab), style=body))),
                    dcc.Tab(label="Data Quality", value="data_quality",
                            children=dcc.Loading(html.Div(_safe(_data_quality_tab), style=body))),
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
        # Row id = ticker so an active-cell click resolves to the ticker robustly
        # (survives native sort / filter / pagination) for the review-timeline plot.
        recs_disp["id"] = recs_disp["ticker"]

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
            "Every BUY/SELL/HOLD/WATCH the model produced this run. Green-tinted rows are actionable (paper-traded). Hover a column header for its definition. Click any row to chart that ticker's hold-review confidence over time below."),
        dash_table.DataTable(
            id="rec-table",
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
        dcc.Loading(html.Div(id="rec-review-plot", style={"marginTop": 16})),
    ])


@app.callback(Output("rec-review-plot", "children"), Input("rec-table", "active_cell"))
def _rec_review_plot(active_cell):
    """Click a recommendation row → chart that ticker's opener-pinned hold-review
    confidence over time, with price + entry/exit decisions, so deterioration →
    direction-change is visible. ``row_id`` is the ticker (set in _rationale_body)."""
    if not active_cell:
        return html.Div(
            "↑ Click any recommendation row to chart its hold-review confidence over time "
            "(the per-tick re-judgment by the engines that opened it) against price and the "
            "entry/exit decisions.",
            style={"color": "#6b7280", "fontStyle": "italic", "padding": "8px 2px"})
    ticker = active_cell.get("row_id")
    if not ticker:
        return html.Div()
    return _safe(lambda: _review_timeline_section(ticker))


def _review_timeline_section(ticker: str):
    reviews = data.trade_reviews_df(ticker)
    trades = data.trades_for_ticker(ticker)
    if reviews is None or reviews.empty:
        return html.Div(
            f"No hold-review history recorded for {ticker} yet. Only LLM-opened positions accrue "
            "it, and only from this feature's first run onward — it fills in tick by tick while a "
            "position is held.",
            style={"color": "#6b7280", "padding": "8px 2px"})
    return html.Div([
        _h3(f"{ticker} — hold-review confidence over time",
            "Each point is one tick's re-judgment of this position by the SAME synthesis + sentiment "
            "engines that opened it, on fresh news + prices (so it's an apples-to-apples vs the entry "
            "confidence). Marker colour = the review's action (green BUY / red SELL / grey HOLD). "
            "Dashed line = entry confidence; dotted line = the close floor (same-direction conviction "
            "below it triggers an llm_confidence_loss exit). Grey line = price; triangles = entry, "
            "✕ = exit. Watch whether the confidence sliding toward the floor precedes a colour flip."),
        dcc.Graph(figure=figures.confidence_timeline_fig(reviews, trades)),
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


# ── Direction toggle (long / short / both) ───────────────────────────────────
_DIRECTION_OPTIONS = [
    {"label": "Long + Short", "value": "all"},
    {"label": "Long only", "value": "long"},
    {"label": "Short only", "value": "short"},
]


def _direction_toggle(component_id: str) -> html.Div:
    """Long (BUY) / Short (SELL) / both selector. Filters the tab's metrics and
    plots to positions ENTERED in that direction."""
    return html.Div(
        [
            html.Label("Direction:  ",
                       title="Filter to LONG positions (BUY entries), SHORT positions (SELL entries), or both.",
                       style={"cursor": "help", "borderBottom": "1px dotted #cbd5e1", "marginRight": 4}),
            dcc.RadioItems(
                id=component_id, options=_DIRECTION_OPTIONS, value="all", inline=True,
                persistence=True, persistence_type="session",
                inputStyle={"marginLeft": 14, "marginRight": 4},
                labelStyle={"cursor": "pointer"},
            ),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": 12},
    )


def _direction_value(value):
    """RadioItems value → 'long' | 'short', or None for both."""
    return None if value in (None, "all") else value


# ── Method-Performance evidence source (gated ledger vs all-scored panel) ────
_METHOD_SOURCE_OPTIONS = [
    {"label": "Ledger (gated trades)", "value": "ledger"},
    {"label": "All scored tickers (simulated)", "value": "panel"},
]


def _method_source_toggle(component_id: str) -> html.Div:
    """The solo-method table's evidence base. Ledger = only the gate-selected
    trades that actually opened (small, selection-biased). All scored tickers =
    every method's implied BUY/SELL on EVERY scored ticker each run (the
    simulated_trades panel), scored on gross forward returns — thousands of
    observations, unbiased by the trading gates."""
    return html.Div(
        [
            html.Label("Source:  ",
                       title="Ledger (gated trades): solo-method performance over only the trades the gates let through — apples-to-apples with the real book but a small, selection-biased sample. "
                             "All scored tickers (simulated): treat every method's sign on EVERY scored ticker as a hypothetical solo trade, scored on GROSS forward returns at 30m/3h/6h/1d/3d/1w/2w/1m — the unbiased directional-predictiveness view. Honors the Window toggle (by signal date); ignores Session/Direction.",
                       style={"cursor": "help", "borderBottom": "1px dotted #cbd5e1", "marginRight": 4}),
            dcc.RadioItems(
                id=component_id, options=_METHOD_SOURCE_OPTIONS, value="ledger", inline=True,
                persistence=True, persistence_type="session",
                inputStyle={"marginLeft": 14, "marginRight": 4},
                labelStyle={"cursor": "pointer"},
            ),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": 12},
    )


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

_IC_TOOLTIP = (
    "Spearman rank correlation between each method's score and the forward "
    "close-to-close return at 1/5/10 trading-day horizons. Computed over the "
    "persisted signals panel — EVERY scored ticker each run, not just the few that "
    "became trades — so it is unbiased by the trading gates. Split into four "
    "categories: the 8 OHLCV methods computed on 30-min, daily, and weekly candles "
    "(the SAME indicators on different bar sizes), and Other (news, sentiment, smart "
    "money, options, catalysts — most-recent data). 'Sim win %' = simulated solo win "
    "rate (share of non-zero scores whose sign matched the move); 'Sim ret %' = "
    "simulated solo return (mean sign(score)×forward-return — the gross P&L if that "
    "method alone decided the trade). 'IC std' = standard deviation of the PER-DAY IC "
    "and 'ICIR' = mean(daily IC)/std(daily IC) — the IC's reliability: each signal-day "
    "counts once, so they are NOT inflated by same-day cross-sectional correlation the "
    "way a standard error off the raw n would be (|ICIR| ≳ 0.5 is a stable edge, ≈ 0 is "
    "noise); both need several signal-days before they populate. A persistent NEGATIVE "
    "IC is sign-inverted (a logic bug); IC ≈ 0 at large n is dead weight. 'Views' = scored, non-zero "
    "observations. Run/forward-return based (NOT affected by the window toggle); "
    "n grows every run — judge nothing on a thin panel.")

_IC_HORIZONS = (1, 5, 10)


def _ic_category_table(subset, labels):
    """Build one category's IC DataTable from its rows of the ic DataFrame."""
    rows = []
    for _, r in subset.iterrows():
        row = {"method": labels.get(r["method"], r["method"]), "views": int(r["views"])}
        for h in _IC_HORIZONS:
            n, ic, hit, sim = (r.get(f"n_{h}d"), r.get(f"ic_{h}d"),
                               r.get(f"hit_{h}d"), r.get(f"simret_{h}d"))
            icstd, icir = r.get(f"icstd_{h}d"), r.get(f"icir_{h}d")
            row[f"n_{h}d"] = int(n) if pd.notna(n) else None
            row[f"ic_{h}d"] = round(float(ic), 3) if pd.notna(ic) else None
            row[f"icstd_{h}d"] = round(float(icstd), 3) if pd.notna(icstd) else None
            row[f"icir_{h}d"] = round(float(icir), 2) if pd.notna(icir) else None
            row[f"hit_{h}d"] = round(float(hit), 1) if pd.notna(hit) else None
            row[f"simret_{h}d"] = round(float(sim), 2) if pd.notna(sim) else None
        rows.append(row)
    cols = [{"name": "Method", "id": "method"},
            {"name": "Views", "id": "views", "type": "numeric", "format": _INT}]
    for h in _IC_HORIZONS:
        cols += [
            {"name": f"n@{h}d", "id": f"n_{h}d", "type": "numeric", "format": _INT},
            {"name": f"IC@{h}d", "id": f"ic_{h}d", "type": "numeric", "format": _NUM2},
            {"name": f"IC std@{h}d", "id": f"icstd_{h}d", "type": "numeric", "format": _NUM2},
            {"name": f"ICIR@{h}d", "id": f"icir_{h}d", "type": "numeric", "format": _NUM2},
            {"name": f"Sim win@{h}d %", "id": f"hit_{h}d", "type": "numeric", "format": _NUM2},
            {"name": f"Sim ret@{h}d %", "id": f"simret_{h}d", "type": "numeric", "format": _NUM2},
        ]
    longest = max(_IC_HORIZONS)
    cond = []
    for c in (f"ic_{longest}d", f"icir_{longest}d", f"simret_{longest}d"):
        cond += [
            {"if": {"filter_query": f"{{{c}}} > 0", "column_id": c},
             "color": figures.POS, "fontWeight": "bold"},
            {"if": {"filter_query": f"{{{c}}} < 0", "column_id": c},
             "color": figures.NEG, "fontWeight": "bold"},
        ]
    return dash_table.DataTable(data=rows, columns=cols, style_data_conditional=cond, **_TABLE_KW)


def _ic_section():
    """Per-method information coefficient over the signals panel, split into the
    30-min / daily / weekly technical categories plus Other."""
    from src.performance.tracker import METHOD_LABELS
    from src.analysis.signal_panel import IC_CATEGORY_ORDER
    res = data.signal_ic()
    icdf = res.get("ic")
    heading = _h3("Signal information coefficient (IC)", _IC_TOOLTIP)
    if icdf is None or getattr(icdf, "empty", True):
        return html.Div([
            heading,
            html.Div(
                f"Signals panel has {res.get('panel_rows', 0)} row(s) across "
                f"{res.get('tickers', 0)} ticker(s) — not enough forward-return history "
                "for IC yet. It accrues automatically every run.",
                style={"color": "#6b7280"}),
        ])
    labels = dict(METHOD_LABELS)
    labels["combined_score"] = "All methods (combined)"
    children = [heading]
    has_cat = "category" in icdf.columns
    for category in IC_CATEGORY_ORDER:
        subset = icdf[icdf["category"] == category] if has_cat else icdf
        if subset is None or subset.empty:
            continue
        children.append(html.Div(category, style={
            "fontWeight": "bold", "marginTop": 14, "marginBottom": 4, "color": "#cbd5e1"}))
        children.append(_ic_category_table(subset, labels))
        if not has_cat:
            break
    return html.Div(children)


_SIM_PERF_TOOLTIP = (
    "Every method's implied BUY/SELL (sign of its score) on EVERY scored ticker each "
    "run — the `simulated_trades` panel — scored on GROSS close-to-close forward "
    "returns (no costs; the question is directional predictiveness, not net P&L). This "
    "is the unbiased counterpart to the ledger solo table: it counts thousands of "
    "observations instead of only the gate-selected trades that opened. 'Trades' = "
    "total simulated solo trades for the method. Per horizon (30m/3h/6h/1d/3d/1w/2w/1m): "
    "'n@' = joint observations with a forward return, 'IC@' = Spearman rank correlation "
    "between the method's score and the forward return (ranking skill; a persistent "
    "positive IC is real edge, a persistent negative IC is sign-inverted), 'IC std@' / "
    "'ICIR@' = the IC's reliability — stdev and information-ratio (mean/std) of the "
    "PER-DAY IC, where each signal-day counts once so they aren't inflated by same-day "
    "cross-sectional correlation (|ICIR| ≳ 0.5 is stable, ≈ 0 is noise; need several days), "
    "'Win@ %' = "
    "share of the method's solo calls that were directionally right, 'Ret@ %' = mean "
    "signed forward return. Forward returns come from the OHLCV cache, so 1w/2w/1m fill "
    "in only after a post-close cache warm. Use the Horizons/Metrics pickers above to "
    "trim the columns. Run-based; honors the Window toggle by signal date. A method "
    "predictive of direction shows IC > 0 / Win > 50 PERSISTING across horizons; judge "
    "nothing on a thin n.")

_SIM_HORIZONS = ("30m", "3h", "6h", "1d", "3d", "1w", "2w", "1m")

# Per-horizon metric columns, in display order (matches the IC table: n, IC, win,
# ret). Each: (header template, id template, numeric format).
_SIM_METRIC_ORDER = ("n", "ic", "icstd", "icir", "win", "ret")
_SIM_METRIC_SPECS = {
    "n":     ("n@{}", "n_{}", _INT),
    "ic":    ("IC@{}", "ic_{}", _NUM2),
    "icstd": ("IC std@{}", "icstd_{}", _NUM2),
    "icir":  ("ICIR@{}", "icir_{}", _NUM2),
    "win":   ("Win@{} %", "win_{}", _NUM2),
    "ret":   ("Ret@{} %", "ret_{}", _NUM2),
}
_SIM_METRIC_LABELS = {"n": "n (obs)", "ic": "IC", "icstd": "IC std", "icir": "ICIR",
                      "win": "Win %", "ret": "Ret %"}


def _sim_column_filters() -> html.Div:
    """Horizons + metrics multi-selects that trim the columns of every simulated
    table below. Empty selection falls back to all (never an empty table)."""
    return html.Div(
        [
            html.Label("Simulated columns:  ",
                       title="Pick which horizons and which metrics (n / IC / Win % / Ret %) "
                             "appear in the 'All scored tickers' tables below. Applies to all "
                             "category tables at once; clearing a picker shows everything.",
                       style={"cursor": "help", "borderBottom": "1px dotted #cbd5e1", "marginRight": 8}),
            dcc.Dropdown(id="sim-horizons",
                         options=[{"label": h, "value": h} for h in _SIM_HORIZONS],
                         value=list(_SIM_HORIZONS), multi=True, placeholder="Horizons…",
                         persistence=True, persistence_type="session",
                         style={"flex": 2, "minWidth": 320}),
            dcc.Dropdown(id="sim-metrics",
                         options=[{"label": _SIM_METRIC_LABELS[m], "value": m} for m in _SIM_METRIC_ORDER],
                         value=list(_SIM_METRIC_ORDER), multi=True, placeholder="Metrics…",
                         persistence=True, persistence_type="session",
                         style={"flex": 1, "minWidth": 220, "marginLeft": 8}),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": 12},
    )


def _sim_perf_table(subset, labels, horizons, metrics):
    """One IC-category's simulated-performance DataTable, limited to the chosen
    horizons × metrics (IC/win/ret/n)."""
    horizons = [h for h in _SIM_HORIZONS if h in horizons] or list(_SIM_HORIZONS)
    metrics = [m for m in _SIM_METRIC_ORDER if m in metrics] or list(_SIM_METRIC_ORDER)
    rows = []
    for _, r in subset.iterrows():
        row = {"method": labels.get(r["method"], r["method"]), "views": int(r["views"])}
        for lbl in horizons:
            for m in metrics:
                cid = _SIM_METRIC_SPECS[m][1].format(lbl)
                v = r.get(cid)
                if m == "n":
                    row[cid] = int(v) if pd.notna(v) else 0
                else:
                    row[cid] = round(float(v), 3) if pd.notna(v) else None
        rows.append(row)
    cols = [{"name": "Method", "id": "method"},
            {"name": "Trades", "id": "views", "type": "numeric", "format": _INT}]
    for lbl in horizons:
        for m in metrics:
            name_t, id_t, fmt = _SIM_METRIC_SPECS[m]
            cols.append({"name": name_t.format(lbl), "id": id_t.format(lbl),
                         "type": "numeric", "format": fmt})
    cond = []
    for lbl in horizons:
        if "win" in metrics:
            wc = f"win_{lbl}"
            cond += [
                {"if": {"filter_query": f"{{{wc}}} >= 50", "column_id": wc},
                 "color": figures.POS, "fontWeight": "bold"},
                {"if": {"filter_query": f"{{{wc}}} < 50", "column_id": wc},
                 "color": figures.NEG, "fontWeight": "bold"},
            ]
        for mc in ("ic", "icir", "ret"):
            if mc in metrics:
                c = f"{mc}_{lbl}"
                cond += [
                    {"if": {"filter_query": f"{{{c}}} > 0", "column_id": c}, "color": figures.POS},
                    {"if": {"filter_query": f"{{{c}}} < 0", "column_id": c}, "color": figures.NEG},
                ]
    return dash_table.DataTable(data=rows, columns=cols, style_data_conditional=cond, **_TABLE_KW)


def _simulated_perf_section(window_days, sel_horizons=None, sel_metrics=None):
    """Per-method directional win rate + IC + gross return over ALL scored tickers
    (the simulated_trades panel), grouped by the same IC categories. The
    horizons/metrics selections trim every category table's columns."""
    from src.performance.tracker import METHOD_LABELS
    from src.analysis.signal_panel import IC_CATEGORY_ORDER
    sel_horizons = sel_horizons or list(_SIM_HORIZONS)
    sel_metrics = sel_metrics or list(_SIM_METRIC_ORDER)
    df = data.simulated_method_perf(days=window_days)
    heading = _h3("Simulated single-method performance — all scored tickers", _SIM_PERF_TOOLTIP)
    if df is None or getattr(df, "empty", True):
        return html.Div([
            heading,
            html.Div("No simulated single-method trades with forward returns yet. They "
                     "accrue every run; materialise existing history with "
                     "`python -m src.analysis.simulated_trades --backfill`.",
                     style={"color": "#6b7280"}),
        ])
    labels = dict(METHOD_LABELS)
    labels["combined_score"] = "All methods (combined)"
    children = [heading]
    has_cat = "category" in df.columns
    for category in IC_CATEGORY_ORDER:
        subset = df[df["category"] == category] if has_cat else df
        if subset is None or subset.empty:
            continue
        children.append(html.Div(category, style={
            "fontWeight": "bold", "marginTop": 14, "marginBottom": 4, "color": "#cbd5e1"}))
        children.append(_sim_perf_table(subset, labels, sel_horizons, sel_metrics))
        if not has_cat:
            break
    return html.Div(children)


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
        _direction_toggle("methods-direction"),
        _method_source_toggle("methods-source"),
        _sim_column_filters(),
        dcc.Loading(html.Div(id="methods-body")),
        _safe(_ic_section),
        _h3("LLM models used (synthesis & sentiment)",
            "Which exact LLMs actually ran across all recorded pipeline runs — the final-call 'synthesis' model and the per-ticker 'sentiment' model — including any DeepSeek or rule-based fallbacks. Not affected by the window toggle above (it's run-based, not trade-based). Hover a column header for details."),
        models_table,
    ])


@app.callback(Output("methods-body", "children"),
              Input("methods-window", "value"), Input("methods-session", "value"),
              Input("methods-direction", "value"), Input("methods-source", "value"),
              Input("sim-horizons", "value"), Input("sim-metrics", "value"))
def _methods_body(window_value, session_value, direction_value, source_value,
                  sim_horizons, sim_metrics):
    if source_value == "panel":
        # All scored tickers — run-based; honors the window (by signal date) and
        # the Horizons/Metrics column pickers, ignores session/direction (the
        # panel has no per-session split and the method's sign already IS its
        # direction).
        return _safe(lambda: _simulated_perf_section(_window_days(window_value),
                                                     sim_horizons, sim_metrics))
    return _safe(lambda: _methods_perf_section(_window_days(window_value), _session_value(session_value),
                                               _direction_value(direction_value)))


def _calibration_block(window_days, session, direction=None):
    """Confidence-calibration buckets + slope (item #2) — the formal summary of
    the return-vs-confidence scatter above it."""
    cal = data.confidence_calibration(window_days, session, direction)
    rows = [{"bucket": b["label"], "trades": b["trades"], "win": b["win_rate"],
             "avg": b["avg_return"], "median": b["median_return"], "wtd": b["wtd_avg_return"],
             "best": b["best"], "worst": b["worst"]} for b in (cal.get("buckets") or [])]
    cols = [
        {"name": "Confidence bucket", "id": "bucket"},
        {"name": "Trades", "id": "trades", "type": "numeric", "format": _INT},
        {"name": "Win rate %", "id": "win", "type": "numeric", "format": _NUM2},
        {"name": "Avg return %", "id": "avg", "type": "numeric", "format": _NUM2},
        {"name": "Median %", "id": "median", "type": "numeric", "format": _NUM2},
        {"name": "Wtd avg %", "id": "wtd", "type": "numeric", "format": _NUM2},
        {"name": "Best %", "id": "best", "type": "numeric", "format": _NUM2},
        {"name": "Worst %", "id": "worst", "type": "numeric", "format": _NUM2},
    ]
    table = (dash_table.DataTable(data=rows, columns=cols, **_TABLE_KW) if rows
             else html.Div("No trades with a stored confidence in this window yet.",
                           style={"color": "#6b7280"}))
    return html.Div([
        _h3("Confidence calibration — buckets + slope",
            "Trades bucketed by entry confidence (each bucket is also a position-size "
            "tier). If higher-confidence buckets earn more — and the slope is positive "
            "— confidence is carrying return-predictive information worth sizing on; a "
            "flat/negative slope means the size tiers are sizing on noise. Closed trades "
            "at realised return, open at live mark. Respects the window + session toggles."),
        html.Div(cal.get("verdict", ""),
                 style={"color": "#374151", "marginBottom": 8, "fontSize": 13}),
        dcc.Graph(figure=figures.calibration_bar_fig(cal)),
        table,
    ])


def _methods_perf_section(window_days, session=None, direction=None):
    perf = data.performance(window_days=window_days, session=session, direction=direction)
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

    # ── Macro evaluation — aggregated decision layers (synthesis vs aggregator
    # vs method bundles), each scored on its full directional-call stream so the
    # LLM's confidence and the aggregator's confidence are directly comparable.
    macro_rows = [
        {
            "Layer": r["label"],
            "Win rate %": round(r["win_rate"], 1) if r.get("win_rate") is not None else None,
            "Trades": r.get("trades"),
            "Avg return %": round(r["avg_return"], 2) if r.get("avg_return") is not None else None,
        }
        for r in (perf.get("macro_eval") or [])
    ]
    macro_table = dash_table.DataTable(
        data=macro_rows,
        columns=[{"name": c, "id": c} for c in ["Layer", "Win rate %", "Trades", "Avg return %"]],
        tooltip_header=_MACRO_HEADER_TIPS,
        style_data_conditional=[
            {"if": {"filter_query": '{Layer} contains "Synthesis"'}, "backgroundColor": "#eef2ff"},
            {"if": {"filter_query": '{Layer} contains "Aggregator"'}, "backgroundColor": "#ecfdf5"},
        ],
        **_TABLE_KW,
    ) if macro_rows else html.Div("No macro-layer stats in this window yet.", style={"color": "#6b7280"})

    return html.Div([
        dcc.Graph(figure=figures.method_winrate_fig(perf)),
        _h3("Return vs entry confidence",
            "Each dot is one trade: its entry confidence (x) against its return (y) — closed trades at their realised return, "
            "open trades (hollow diamonds) at their live mark-to-market; green = win, red = loss. Confidence sets the position-size "
            "tier (1.0×/1.5×/2.0×), so an upward-sloping dashed trend line confirms higher-confidence calls actually earn more and "
            "the sizing is justified; a flat or downward line means confidence isn't carrying directional information. Respects the "
            "window + session toggles above."),
        dcc.Graph(figure=figures.confidence_return_fig(perf)),
        _calibration_block(window_days, session, direction),
        _h3("Macro evaluation — decision layers (LLM synthesis vs aggregator vs bundles)",
            "Head-to-head performance of the aggregated decision layers, all scored on the SAME unbiased basis: "
            "every directional call each layer made — not just the few that became trades — entered at the call-time "
            "snapshot price and marked at the latest close through the real cost model. 'LLM Synthesis' is the final "
            "BUY/SELL call (all engines; the per-engine split is in the Model Evaluation table below); 'Aggregator' is "
            "the mechanical combined signal; each 'Bundle' is a method family voting by its summed score. This is the "
            "direct test of whether the LLM's confidence or the aggregator's confidence is the more reliable predictor. "
            "Respects the window + session toggles above."),
        macro_table,
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


def _pct4(x) -> str:
    """Percent with up to 3 decimals (one-way fees are small — 0.4% not 0%)."""
    if x is None:
        return "–"
    try:
        return f"{x:.3f}%"
    except (TypeError, ValueError):
        return str(x)


def _trades_table(trades: list, table_id: str | None = None):
    """Render a trade ledger as a DataTable.

    When ``table_id`` is given the table gets that ``id`` and each row gets an
    ``id`` equal to its ticker, so an ``active_cell`` click resolves to the
    ticker robustly (survives native sort / filter / pagination) — used by the
    Returns tab to chart the clicked ticker's confidence-over-time plot below.
    """
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
    records = df.to_dict("records")
    extra = {}
    if table_id:
        for rec in records:
            rec["id"] = rec.get("ticker")   # active_cell.row_id → ticker
        extra["id"] = table_id
    return dash_table.DataTable(
        data=records,
        columns=_columns(spec),
        tooltip_header=_header_tooltips(spec),
        filter_action="native",
        style_data_conditional=[
            {"if": {"filter_query": "{return_pct} > 0", "column_id": "return_pct"}, "color": figures.POS},
            {"if": {"filter_query": "{return_pct} < 0", "column_id": "return_pct"}, "color": figures.NEG},
        ],
        **extra,
        **_TABLE_KW,
    )


def _returns_tab():
    return html.Div([
        _source_toggle("returns-source"),
        _window_toggle("returns-window"),
        _session_toggle("returns-session"),
        _direction_toggle("returns-direction"),
        dcc.Loading(html.Div(id="returns-body")),
    ])


@app.callback(Output("returns-body", "children"),
              Input("returns-window", "value"), Input("returns-session", "value"),
              Input("returns-direction", "value"), Input("returns-source", "value"))
def _returns_body(window_value, session_value, direction_value, source_value):
    if (source_value or "sim") == "broker":
        return _safe(lambda: _broker_returns_section(window_value, session_value, direction_value))
    return _safe(lambda: _returns_section(window_value, session_value, direction_value))


@app.callback(Output("returns-review-plot", "children"),
              Input("returns-open-table", "active_cell"),
              Input("returns-closed-table", "active_cell"))
def _returns_review_plot(open_cell, closed_cell):
    """Click any open/closed trade row in the Returns tab → chart that ticker's
    confidence score over the days the position was held (the opener-pinned
    hold-review, left axis) against the stock price (right axis) — the same
    per-ticker plot as the Recommendations tab. ``ctx.triggered_id`` picks
    whichever of the two tables was clicked last; ``row_id`` is the ticker
    (set in ``_trades_table``). Shared by the Simulated and IBKR views (only one
    renders at a time, so the table ids never collide)."""
    from dash import ctx
    cell = closed_cell if ctx.triggered_id == "returns-closed-table" else open_cell
    if not cell:
        return html.Div(
            "↑ Click any open or closed trade row to chart that ticker's confidence "
            "score over the days it was held against the stock price.",
            style={"color": "#6b7280", "fontStyle": "italic", "padding": "8px 2px"})
    ticker = cell.get("row_id")
    if not ticker:
        return html.Div()
    return _safe(lambda: _review_timeline_section(ticker))


def _broker_returns_section(window_value, session_value=None, direction_value=None):
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
    dirn = _direction_value(direction_value)
    if dirn:
        want = "BUY" if dirn == "long" else "SELL"
        trades = [t for t in trades if (t.get("action") or "").upper() == want]
    if not trades:
        return html.Div(
            "No IBKR fills recorded in this window yet — either broker_mode is "
            "off/dry_run, or no submitted order has filled.",
            style={"color": "#6b7280", "padding": 20})

    equity_usd = data.broker_account_equity_usd()
    s = summarize_broker_trades(trades, account_equity_usd=equity_usd)
    from src.performance.broker_view import avg_one_way_cost_pct_from_legs
    lmt_cost = avg_one_way_cost_pct_from_legs(data.filled_lmt_legs())
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
            _kpi("Return (wtd)", _pct(s.get("weighted_return"), signed=True),
                 figures.POS if (s.get("weighted_return") or 0) >= 0 else figures.NEG,
                 tooltip="Total % return on actual fills over CLOSED round-trips, each weighted by its REAL filled notional (so sizing counts — more precise than the equal-weighted 'Avg return'). Net of actual commissions."),
            *([_kpi("P&L vs equity", _pct(s.get("account_return_pct"), signed=True),
                    figures.POS if (s.get("account_return_pct") or 0) >= 0 else figures.NEG,
                    tooltip="Cumulative P&L of these trades (realized + open, this window) as a % of your LATEST IBKR account NAV (NetLiquidation), converted to USD — the account-relative impact. Uses the real account equity, not just per-trade fills; approximate (latest NAV vs windowed P&L).")]
              if s.get("account_return_pct") is not None else []),
            _kpi("Win rate", _pct(s.get("win_rate")),
                 tooltip="Share of CLOSED broker round-trips with a positive net return on actual fills."),
            _kpi("Avg return", _pct(s.get("avg_return"), signed=True),
                 tooltip="Mean per-round-trip % return on actual fill prices net of actual commissions, EQUAL-weighted (compare with 'Return (wtd)', which weights by real dollars)."),
            _kpi("Commissions", _usd(s.get("commissions_usd"), signed=False),
                 tooltip="Total commissions IBKR actually charged on these fills (exit legs counted once filled)."),
            _kpi("Avg 1-way cost (LMT)", _pct4(lmt_cost),
                 tooltip="Average ALL-IN cost per ONE-WAY fill across all real LMT fills, as a % of that leg's notional: real IBKR commission PLUS the execution cost (how far the fill landed from the decision price — captures the bid-ask crossing and any latency drift, positive = adverse). LMT ONLY — market (MKT) fills are excluded, since LMT is what the system uses going forward; drift-flatten cleanups are excluded too. Signed, so a favorable fill can lower it. This is the figure the simulated cost is calibrated to once enough LMT fills accumulate."),
            _kpi("Closed / Open", f"{s.get('closed', 0)} / {s.get('open', 0)}",
                 tooltip="Broker round-trips completed vs positions genuinely still held at the broker (a ledger-closed trade whose exit hasn't filled is still OPEN here)."),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
    )

    ibkr_pnl = data.broker_account_pnl()
    pnl_block = []
    if ibkr_pnl:
        pnl_block = [
            _h3("IBKR account P&L (live — straight from IBKR)",
                "IBKR's own account P&L via reqPnL — ground truth including ALL fees, FX, and "
                "dividends, which the fill-derived numbers above can only approximate. "
                "ACCOUNT-LEVEL (all positions), NOT per-trade: 'Unrealized' is the current open "
                "P&L; 'Realized today' and 'Daily' are TODAY's figures (they reset each session). "
                "Converted to USD."),
            html.Div(
                [
                    _kpi("IBKR Unrealized", _usd(ibkr_pnl.get("unrealized")),
                         figures.POS if (ibkr_pnl.get("unrealized") or 0) >= 0 else figures.NEG,
                         tooltip="Current open-position P&L across the whole account, straight from IBKR (reqPnL.unrealizedPnL) — includes fees/FX/dividends."),
                    _kpi("IBKR Realized today", _usd(ibkr_pnl.get("realized")),
                         figures.POS if (ibkr_pnl.get("realized") or 0) >= 0 else figures.NEG,
                         tooltip="Today's realized P&L across the account, straight from IBKR (resets each session)."),
                    _kpi("IBKR Daily P&L", _usd(ibkr_pnl.get("daily")),
                         figures.POS if (ibkr_pnl.get("daily") or 0) >= 0 else figures.NEG,
                         tooltip="Today's total account P&L change, straight from IBKR (reqPnL.dailyPnL)."),
                ],
                style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
            ),
        ]

    return html.Div([
        cards,
        *pnl_block,
        _h3("Open broker positions",
            "Shares genuinely held at IBKR right now (entry filled; exit not filled yet — even if the simulated ledger already closed the trade), marked at the latest price. Click a row to chart that ticker's confidence-over-time below."),
        _trades_table(open_trades, table_id="returns-open-table"),
        _h3("Closed broker round-trips",
            "Entry and exit both filled at IBKR. Returns are computed on the actual average fill prices, net of the commissions actually charged. Click a row to chart that ticker's confidence-over-time below."),
        _trades_table(closed_trades, table_id="returns-closed-table"),
        dcc.Loading(html.Div(id="returns-review-plot", style={"marginTop": 16})),
    ])


def _exit_quality_block(window_value, session_value, direction_value=None):
    """MFE/MAE exit-quality (item #5). Sim-ledger only — the excursion fields are
    maintained on the simulated trade, not the broker view."""
    rep = data.exit_quality(_window_days(window_value), _session_value(session_value),
                            _direction_value(direction_value))
    heading = _h3("Exit quality — MFE / MAE",
                  "Where the exit landed inside each trade's own MFE→MAE range. Exit "
                  "placement 1.0 = sold at the peak; 0.0 = cut at the worst point. Capture "
                  "= fraction of the favorable peak kept. Low placement ⇒ exits skew late "
                  "(cutting near the bottom); low capture with healthy MFE ⇒ winners ridden "
                  "back to flat (no profit-taking). Closed trades only; trades with a "
                  "degenerate (entered≈closed) excursion band are excluded.")
    if not rep.get("n"):
        return html.Div([heading, html.Div(rep.get("verdict", ""), style={"color": "#6b7280"})])
    cap = rep.get("avg_capture")
    cards = html.Div(
        [
            _kpi("Avg exit placement", _pct((rep["avg_placement"] or 0) * 100),
                 tooltip="Mean position in the MFE→MAE range (100% = exited at the peak, 0% = at the worst point)."),
            _kpi("Avg capture", _pct(cap * 100) if cap is not None else "–",
                 tooltip="Mean fraction of the favorable peak (MFE) kept at exit."),
            _kpi("Exited near MAE", _pct(rep["pct_exited_near_mae"]),
                 tooltip="Share of trades exited in the bottom 20% of their range — cutting near the worst point."),
            _kpi("Gave back >½ peak", _pct(rep.get("pct_gave_back_most_mfe")),
                 tooltip="Share of trades that kept less than half their favorable peak."),
            _kpi("Trades", str(rep["n"]),
                 tooltip="Analysable closed trades (non-degenerate MFE/MAE band)."),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
    )
    rows = [{"ticker": r["ticker"], "ret": r["return_pct"], "mfe": r["mfe"], "mae": r["mae"],
             "place": r["exit_placement"], "capture": r["capture"], "giveback": r["give_back"],
             "reason": r["exit_reason"]} for r in rep["per_trade"]]
    cols = [
        {"name": "Ticker", "id": "ticker"},
        {"name": "Return %", "id": "ret", "type": "numeric", "format": _NUM2},
        {"name": "MFE %", "id": "mfe", "type": "numeric", "format": _NUM2},
        {"name": "MAE %", "id": "mae", "type": "numeric", "format": _NUM2},
        {"name": "Placement", "id": "place", "type": "numeric", "format": _NUM2},
        {"name": "Capture", "id": "capture", "type": "numeric", "format": _NUM2},
        {"name": "Give-back %", "id": "giveback", "type": "numeric", "format": _NUM2},
        {"name": "Exit reason", "id": "reason"},
    ]
    return html.Div([
        heading,
        html.Div(rep.get("verdict", ""), style={"color": "#374151", "marginBottom": 8, "fontSize": 13}),
        cards,
        dcc.Graph(figure=figures.mfe_capture_fig(rep)),
        dash_table.DataTable(data=rows, columns=cols, **_TABLE_KW),
    ])


def _returns_section(window_value, session_value=None, direction_value=None):
    perf = data.performance(window_days=_window_days(window_value), session=_session_value(session_value),
                            direction=_direction_value(direction_value))
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
            _kpi("Avg 1-way cost" + (" (real)" if perf.get("sim_cost_is_real") else ""),
                 _pct4(perf.get("sim_one_way_cost_pct")),
                 tooltip=("All-in one-way cost charged on each simulated leg, as a % of trade value. "
                          + ("CALIBRATED TO REAL IBKR FILLS: the measured average actual cost "
                             "(commission + execution vs decision price) is applied flat to every "
                             "entry and exit, so the simulated returns reflect what execution really costs."
                             if perf.get("sim_cost_is_real")
                             else "MODELED (half-spread + commission) — not enough real IBKR fills yet to "
                                  "calibrate (set by sim_real_fill_costs_min_legs); switches to real fills "
                                  "automatically once they accumulate.")
                          + " Compare with the IBKR view's 'Avg 1-way cost' (measured directly from fills).")),
            _kpi("Closed / Open", f"{stats.get('total_closed', 0)} / {stats.get('total_open', 0)}",
                 tooltip="Number of closed (realised) trades vs. positions currently open, within the selected window."),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
    )

    return html.Div([
        cards,
        dcc.Graph(figure=figures.equity_curve_fig(perf)),
        _h3("Open positions", "Positions currently held, marked to the latest price — the return shown is live mark-to-market ('what if you closed now'). Filtered to the selected entry window. Click a row to chart that ticker's confidence-over-time below."),
        _trades_table(perf.get("open_trades") or [], table_id="returns-open-table"),
        _h3("Closed trades", "Realised round-trips, with their final spread-adjusted return. Filtered to the selected entry window. Click a row to chart that ticker's confidence-over-time below."),
        _trades_table(perf.get("closed_trades") or [], table_id="returns-closed-table"),
        _exit_quality_block(window_value, session_value, direction_value),
        dcc.Loading(html.Div(id="returns-review-plot", style={"marginTop": 16})),
    ])


# ── Tab 4: Execution (price provenance · broker forensics · tracking error) ──

def _execution_tab():
    return html.Div([
        _safe(_provenance_section),
        _safe(_broker_forensics_section),
        _safe(_tracking_error_section),
    ])


def _provenance_section():
    """Price-provenance detail (item #8): trades in the latest run whose entry
    price diverged from the run snapshot beyond the session band."""
    pp = (data.latest_gate_diag() or {}).get("price_provenance")
    heading = _h3("Price provenance — entry vs snapshot",
                  "Standing guard against the stale-price class: each new trade's recorded "
                  "entry price is compared to the run's analysis snapshot for that ticker. A "
                  "divergence beyond the session band (RTH tight, off-hours wider) is flagged — "
                  "the automatic version of the one-off CRDO fill-vs-snapshot audit. Latest run.")
    if not pp:
        return html.Div([heading, html.Div(
            "No price-provenance record in the latest run (no trades opened, no snapshot, "
            "or the check is disabled).", style={"color": "#6b7280"})])
    flagged = pp.get("flagged") or []
    if not flagged:
        body = html.Div(
            f"✓ All {pp.get('n_checked', 0)} new trade(s) in the latest run entered within "
            "the snapshot band.", style={"color": figures.POS, "fontWeight": "bold"})
    else:
        rows = [{"ticker": f["ticker"], "entry": f["entry_price"], "snap": f["snapshot_price"],
                 "bps": f["bps"], "session": f["session"], "band": f["band"]} for f in flagged]
        cols = [
            {"name": "Ticker", "id": "ticker"},
            {"name": "Entry $", "id": "entry", "type": "numeric", "format": _NUM2},
            {"name": "Snapshot $", "id": "snap", "type": "numeric", "format": _NUM2},
            {"name": "Divergence bp", "id": "bps", "type": "numeric", "format": _NUM2},
            {"name": "Session", "id": "session"},
            {"name": "Band bp", "id": "band", "type": "numeric", "format": _NUM2},
        ]
        body = html.Div([
            html.Div("🔔 " + (pp.get("message") or ""),
                     style={"color": "#92400e", "marginBottom": 8}),
            dash_table.DataTable(data=rows, columns=cols, **_TABLE_KW),
        ])
    return html.Div([heading, body])


def _broker_forensics_section():
    """Slippage / fill-rate / drift / reject forensics (item #3)."""
    rep = data.broker_forensics()
    heading = _h3("Broker execution forensics",
                  "Over all persisted broker orders: fill rate vs kill rate (the settle-or-kill "
                  "design), fill slippage by session (is the LMT cap achievable?), how often broker "
                  "positions drift from the ledger, and what the broker rejects. broker_mode must be "
                  "on for rows to accrue.")
    if not rep.get("n_orders"):
        return html.Div([heading, html.Div(
            "No broker orders recorded yet (broker_mode off / dry-run, or nothing submitted).",
            style={"color": "#6b7280"})])
    fo, d = rep["fill_outcomes"], rep["drift"]
    cards = html.Div(
        [
            _kpi("Fill rate", _pct(fo.get("fill_rate")),
                 tooltip="Filled orders ÷ terminal orders (still-working and no-op rows excluded)."),
            _kpi("Order events", str(rep["n_orders"]),
                 tooltip="Total persisted broker order / fill-repair events."),
            _kpi("Drift runs", f"{d.get('runs_with_drift', 0)}/{d.get('n_runs', 0)}",
                 tooltip="Reconcile runs where a broker position diverged from the ledger."),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
    )
    outcome_rows = [{"outcome": k, "count": v}
                    for k, v in sorted(fo.get("counts", {}).items(), key=lambda kv: -kv[1])]
    outcomes_table = dash_table.DataTable(
        data=outcome_rows,
        columns=[{"name": "Fill outcome", "id": "outcome"},
                 {"name": "Count", "id": "count", "type": "numeric", "format": _INT}],
        **_TABLE_KW) if outcome_rows else html.Div()
    reject_df = rep["reject_reasons"]
    reject_rows = reject_df.to_dict("records") if reject_df is not None and not reject_df.empty else []
    reject_table = (dash_table.DataTable(
        data=reject_rows,
        columns=[{"name": "Reject reason", "id": "reason"},
                 {"name": "Count", "id": "n", "type": "numeric", "format": _INT}],
        **_TABLE_KW) if reject_rows
        else html.Div("No rejected / failed orders.", style={"color": "#6b7280", "marginTop": 8}))
    return html.Div([
        heading, cards,
        dcc.Graph(figure=figures.slippage_by_session_fig(rep["slippage_by_session"])),
        _h3("Fill outcomes", "Count of order events by terminal outcome (filled / killed / failed / working / skipped)."),
        outcomes_table,
        _h3("Reject reasons", "Failed / rejected orders grouped by the broker error message."),
        reject_table,
    ])


def _tracking_error_section():
    """Sim-vs-broker tracking error (item #4)."""
    rep = data.tracking_error()
    heading = _h3("Sim-vs-broker tracking error",
                  "The gap between the modeled ledger and actual IBKR fills, per matched trade. A "
                  "line hugging zero = the model tracks reality; a persistent one-sided drift = a "
                  "cost-model / pricing bug (the auto-catch for the stale-price class). Needs filled "
                  "broker orders.")
    if not rep.get("n_matched"):
        return html.Div([heading, html.Div(rep.get("verdict", ""), style={"color": "#6b7280"})])
    o = rep["overall"]
    cards = html.Div(
        [
            _kpi("Matched trades", str(rep["n_matched"]),
                 tooltip="Sim trades whose broker entry actually filled."),
            _kpi("Mean Δreturn", _pct(o["mean_d_return"], signed=True),
                 tooltip="Mean (sim − broker) return; + = the sim is optimistic vs real fills."),
            _kpi("Mean entry gap",
                 f"{o['mean_entry_bps']:+.0f} bp" if o.get("mean_entry_bps") is not None else "–",
                 tooltip="Mean signed entry-price gap (broker − sim) in basis points."),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
    )
    rows = [{"ticker": r["ticker"], "date": r["entry_date"], "session": r["session"],
             "sim": r["sim_return"], "broker": r["broker_return"], "dret": r["d_return"],
             "bps": r["entry_bps"]} for r in rep["per_trade"]]
    cols = [
        {"name": "Ticker", "id": "ticker"},
        {"name": "Entry date", "id": "date"},
        {"name": "Session", "id": "session"},
        {"name": "Sim %", "id": "sim", "type": "numeric", "format": _NUM2},
        {"name": "Broker %", "id": "broker", "type": "numeric", "format": _NUM2},
        {"name": "Δreturn %", "id": "dret", "type": "numeric", "format": _NUM2},
        {"name": "Entry gap bp", "id": "bps", "type": "numeric", "format": _NUM2},
    ]
    return html.Div([
        heading,
        html.Div(rep.get("verdict", ""), style={"color": "#374151", "marginBottom": 8, "fontSize": 13}),
        cards,
        dcc.Graph(figure=figures.tracking_error_fig(rep)),
        dash_table.DataTable(data=rows, columns=cols, **_TABLE_KW),
    ])


# ── Tab 5: Data Quality (source reliability · per-method coverage) ───────────

_DQ_LOOKBACK_DAYS = 14


def _data_quality_tab():
    return html.Div([
        html.Div(
            f"Source reliability + per-method coverage over the last {_DQ_LOOKBACK_DAYS} days — "
            "catches flaky/slow data sources and feeds that went dark BEFORE they quietly degrade "
            "signals (the failure mode behind every data warning so far). Built from the per-run "
            "run_sources + signals tables.",
            style={"color": "#6b7280", "marginBottom": 12}),
        _safe(_source_reliability_section),
        _safe(_method_coverage_section),
    ])


def _source_reliability_section():
    rows = data.source_reliability(_DQ_LOOKBACK_DAYS)
    heading = _h3("Source reliability",
                  "Per data source over the window, with FOUR outcomes — ok (returned data), empty "
                  "(ran fine but returned nothing), dead (upstream gone, no free replacement), error "
                  "(raised). 'Empty' is first-class: a source that runs but returns nothing is no longer "
                  "silently counted as ok. An always-on feed that goes empty is flagged ⚠ for "
                  "investigation; event-driven feeds (8-K, earnings…) are expected to be empty sometimes; "
                  "dead feeds (greyed) are acknowledged, not actionable. Sorted worst-first. From run_sources.")
    if not rows:
        return html.Div([heading, html.Div("No run-source records in the window yet.",
                                            style={"color": "#6b7280"})])
    below = [r for r in rows if (r.get("success_rate") or 100.0) < 100.0]
    unexpected = [r for r in rows if r.get("unexpected_empty")]
    dead = [r for r in rows if r.get("known_dead")]
    slowest = max(rows, key=lambda r: r.get("median_s") or 0.0)
    cards = html.Div(
        [
            _kpi("Sources tracked", str(len(rows)),
                 tooltip="Distinct enabled data sources that ran at least once in the window."),
            _kpi("Errored", str(len(below)), figures.NEG if below else figures.POS,
                 tooltip="Sources that raised on at least one run — silent data loss candidates."),
            _kpi("Empty (always-on)", str(len(unexpected)), figures.NEG if unexpected else figures.POS,
                 tooltip="Always-on feeds whose LATEST run returned nothing though they should always "
                         "have data — investigate. Excludes event-driven and dead sources."),
            _kpi("Dead feeds", str(len(dead)), "#9ca3af",
                 tooltip="Sources whose upstream is gone with no free replacement (^TICK delisted; "
                         "congressional Stock Watcher 403). Acknowledged, not actionable — shown so the "
                         "deadness stays VISIBLE rather than masked as ok."),
            _kpi("Slowest (median)", f"{slowest['source']} · {slowest.get('median_s') or 0:.0f}s",
                 tooltip="The source with the highest median fetch time — the biggest tick-budget cost."),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
    )

    def _status_disp(r):
        return "empty ⚠" if r.get("unexpected_empty") else r.get("last_status", "")

    def _note(r):
        if r.get("unexpected_empty"):
            return "⚠ always-on feed returned nothing — investigate"
        if r.get("known_dead"):
            return "known-dead — upstream gone, no free replacement"
        if r.get("last_error"):
            return r["last_error"]
        if r.get("expected_sparse"):
            return "event-driven (empty sometimes normal)"
        return ""

    table_rows = [{"source": r["source"], "runs": r["runs"], "success": r["success_rate"],
                   "empty_pct": r.get("empty_rate"), "status": _status_disp(r),
                   "median_s": r["median_s"], "p90_s": r["p90_s"], "note": _note(r)}
                  for r in rows]
    cols = [
        {"name": "Source", "id": "source"},
        {"name": "Runs", "id": "runs", "type": "numeric", "format": _INT},
        {"name": "Success %", "id": "success", "type": "numeric", "format": _NUM2},
        {"name": "Empty %", "id": "empty_pct", "type": "numeric", "format": _NUM2},
        {"name": "Last status", "id": "status"},
        {"name": "Median s", "id": "median_s", "type": "numeric", "format": _NUM2},
        {"name": "p90 s", "id": "p90_s", "type": "numeric", "format": _NUM2},
        {"name": "Note", "id": "note"},
    ]
    cond = [
        {"if": {"filter_query": "{success} < 100", "column_id": "success"},
         "color": figures.NEG, "fontWeight": "bold"},
        # An always-on feed that returned nothing this run — the actionable flag.
        {"if": {"filter_query": '{status} contains "⚠"'},
         "backgroundColor": "#3f1d1d", "color": "#fca5a5", "fontWeight": "bold"},
        # Dead feeds: greyed/italic — acknowledged, not an alarm.
        {"if": {"filter_query": '{status} = "dead"'},
         "color": "#9ca3af", "fontStyle": "italic"},
    ]
    return html.Div([
        heading, cards,
        dcc.Graph(figure=figures.source_latency_fig(rows)),
        dash_table.DataTable(data=table_rows, columns=cols, style_data_conditional=cond, **_TABLE_KW),
    ])


def _method_coverage_section():
    from src.performance.tracker import METHOD_LABELS
    cov = data.method_coverage(_DQ_LOOKBACK_DAYS)
    per = cov.get("per_method") or []
    heading = _h3("Per-method data coverage",
                  "For each signal method, the share of scored tickers with a REAL (non-zero) score — a "
                  "method reads 0.0 ('no view') when its data source failed for a ticker, so a feed going "
                  "dark shows as collapsing coverage before it shows as bad performance. Δ = recent minus "
                  "prior coverage; a large negative Δ is the alarm. From the signals panel.")
    if not per:
        return html.Div([heading, html.Div("No signal rows in the window yet.",
                                            style={"color": "#6b7280"})])
    drops = [r for r in per if r.get("delta") is not None and r["delta"] <= -20]
    cards = html.Div(
        [
            _kpi("Methods", str(len(per)),
                 tooltip="Signal methods tracked in the signals panel."),
            _kpi("Signal rows", f"{cov.get('n_rows', 0):,}",
                 tooltip="Total run×ticker rows in the window (the coverage denominator)."),
            _kpi("Coverage drops", str(len(drops)), figures.NEG if drops else figures.POS,
                 tooltip="Methods whose coverage fell ≥20pp recent-vs-prior — a feed that likely went dark."),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": 12},
    )
    rows = [{"method": METHOD_LABELS.get(r["method"], r["method"]), "coverage": r["coverage_pct"],
             "scored": r["n_scored"], "total": r["n_total"], "recent": r["recent_pct"],
             "prior": r["prior_pct"], "delta": r["delta"]} for r in per]
    cols = [
        {"name": "Method", "id": "method"},
        {"name": "Coverage %", "id": "coverage", "type": "numeric", "format": _NUM2},
        {"name": "Scored", "id": "scored", "type": "numeric", "format": _INT},
        {"name": "Of", "id": "total", "type": "numeric", "format": _INT},
        {"name": "Recent %", "id": "recent", "type": "numeric", "format": _NUM2},
        {"name": "Prior %", "id": "prior", "type": "numeric", "format": _NUM2},
        {"name": "Δ (pp)", "id": "delta", "type": "numeric", "format": _NUM2},
    ]
    cond = [{"if": {"filter_query": "{delta} <= -20", "column_id": "delta"},
             "color": figures.NEG, "fontWeight": "bold"}]
    return html.Div([
        heading,
        html.Div("Low coverage is NORMAL for sparse methods (PEAD, extended-gap, options-derived put_call / "
                 "max_pain / OI-skew / IV — they only fire for a subset of tickers). The actionable signal "
                 "is a negative Δ: a method whose coverage dropped means its feed went dark.",
                 style={"color": "#374151", "marginBottom": 8, "fontSize": 13}),
        cards,
        dcc.Graph(figure=figures.method_coverage_fig(cov)),
        dash_table.DataTable(data=rows, columns=cols, style_data_conditional=cond, **_TABLE_KW),
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


def _lan_ipv4() -> str | None:
    """Best-guess primary LAN IPv4 of this machine (the address a phone on the
    same Wi-Fi would use). No traffic is sent — a UDP socket 'connected' to a
    public IP just makes the OS pick the outbound interface. None on failure."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        return ip if not ip.startswith("127.") else None
    except Exception:
        return None
    finally:
        s.close()


def _tailscale_ipv4() -> str | None:
    """This machine's Tailscale IP (100.64.0.0/10) if Tailscale is up — the
    address a phone reaches from ANYWHERE (cellular included) over the private
    tailnet. Tries the CLI on PATH, then the default Windows install path.
    None when Tailscale isn't installed/running."""
    import subprocess
    for exe in ("tailscale", r"C:\Program Files\Tailscale\tailscale.exe"):
        try:
            out = subprocess.run([exe, "ip", "-4"], capture_output=True, text=True, timeout=5)
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            continue
        for line in (out.stdout or "").splitlines():
            ip = line.strip()
            if ip.startswith("100."):
                return ip
        return None
    return None


def run() -> None:
    """Run the dashboard, auto-restarting on an unexpected crash so it stays alive."""
    host, port = settings.dashboard_host, settings.dashboard_port
    logger.info(f"Dashboard starting at http://{host}:{port}  (Ctrl+C to stop)")
    if host in ("0.0.0.0", "::"):
        # Bound to all interfaces → reachable from other devices: the LAN when
        # home, and the Tailscale tailnet from anywhere (incl. cellular).
        lan = _lan_ipv4()
        if lan:
            logger.info(f"  📱 Same Wi-Fi: http://{lan}:{port}")
        ts = _tailscale_ipv4()
        if ts:
            logger.info(f"  🌍 Away from home (Tailscale): http://{ts}:{port}")
        else:
            logger.info(
                "  🌍 For away-from-home access, install Tailscale (see CLAUDE.md → "
                "'Monitoring dashboard') — do NOT port-forward this to the internet."
            )
        logger.info(
            "  Bound to ALL interfaces — the (read-only) dashboard is reachable by "
            f"any device that can route to it. Windows Firewall must allow inbound TCP {port}."
        )

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
