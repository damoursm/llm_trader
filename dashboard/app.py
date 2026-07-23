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

    # Feeds that WENT DARK: historically-populated sources whose recent runs are
    # all empty — invisible to the failed-sources banner (they return ok=true)
    # and to any single run's status (event-driven feeds are often legitimately
    # empty once). See data_quality.compute_dark_sources.
    try:
        dark = data.dark_sources()
    except Exception as e:
        logger.debug(f"[dashboard] dark-sources banner skipped: {e}")
        dark = []
    if dark:
        items = [f"{d['source']} ({d['prior_empty_pct']:.0f}% → 100% empty over the "
                 f"last {d['recent_empty']} fetches)" for d in dark]
        blocks.append(html.Div(
            [
                html.B(f"📡 {len(dark)} data feed(s) went dark"),
                html.Div(" · ".join(items) + " — see the Data Quality tab.",
                         style={"marginTop": 4, "fontSize": 12, "whiteSpace": "normal"}),
            ],
            style={"background": "#fffbeb", "border": "1px solid #fcd34d", "color": "#92400e",
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
_NUM1 = Format(precision=1, scheme=Scheme.fixed)
_NUM2 = Format(precision=2, scheme=Scheme.fixed)
_NUM3 = Format(precision=3, scheme=Scheme.fixed)
_NUM4 = Format(precision=4, scheme=Scheme.fixed)


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
    ("session", "Session", None, "US-market session the position was ENTERED in: rth (09:30–16:00 ET), premarket (04:00–09:30), afterhours (16:00–20:00), or overnight (20:00–04:00). Pre-market + after-hours make up the 'extended' session: those entries are sized down and bear the wider extended spread in their return."),
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

# Entry Performance table — header explanations (table is built inline below).
_METHOD_HEADER_TIPS = {
    "Method": "The signal method (e.g. news sentiment, technical, momentum) — or an LLM engine row: 'Synthesis LLM' made the final BUY/SELL call, 'Sentiment LLM' scored the per-ticker news (run-dominant engine).",
    "Win rate %": "Method rows — solo simulation: for each closed trade, what if ONLY this method had decided the direction? LLM rows — share of the engine's recommended trades (executed or not) currently positive.",
    "Trades": "Method rows: closed trades this method had a view on (|score| ≥ 0.10). LLM rows: every BUY/SELL the engine recommended — actionable or not, executed or simulated — deduped to its last call per ticker per day.",
    "Avg return %": "Average % return across those trades.",
}

# Decision-funnel table — header explanations (pipeline stage evaluation).
_STAGE_HEADER_TIPS = {
    "Stage": "One step of the decision pipeline, in execution order: the mechanical Aggregator, the LLM Synthesis stream it feeds, then each actionable gate (confidence threshold, agreement floor, PANIC/RISK_OFF BUY-block, earnings blackout, liquidity floor). '→ past Gate k' = the calls still alive after that gate; '✂ Gate k drops' = exactly what that gate discarded. Compare a drops row against its survivor row: drops performing WORSE = the gate is filtering losers (working); drops performing BETTER = the gate is throwing away winners. 'Gate 1b · agreement floor' is the newest gate (2026-07-20) — it mechanically enforces the ≥2-independent-sources rule that was previously only a prompt instruction.",
    "Trades": "Directional calls in that stage's stream, deduped to the last call per ticker per day. The shrink from row to row is each gate's real selectivity.",
    "Win rate %": "Share of the stage's calls currently positive, scored as pseudo-trades: snapshot price at call time → latest cached close, through the real cost model — every call counts, not just the ones that became ledger trades.",
    "Avg return %": "Average forward % return across the stage's calls on the same pseudo-trade basis. A gate earns its place when this rises from the pre-gate row to the post-gate row.",
}

# Macro Performance table — header explanations (aggregated decision layers).
_MACRO_HEADER_TIPS = {
    "Layer": "The aggregated decision layer being judged: 'LLM Synthesis' = the final BUY/SELL caller (all engines combined; the per-engine split is in the Model Evaluation table below), 'Aggregator' = the mechanical combined signal (the weighted blend of all method scores), or 'Bundle · X' = one method family (e.g. Technical, Options) voting by the sign of its summed scores. Each layer is scored on its OWN full stream of directional calls.",
    "Win rate %": "Share of that layer's directional calls currently positive — counting EVERY call it made (actionable or not, executed or not), not just the trades that survived the gates.",
    "Trades": "Number of directional calls the layer made, deduped to its last call per ticker per day (same rule as the LLM rows below).",
    "Avg return %": "Average forward % return across those calls: snapshot price at the call → latest cached close, net of the modeled one-way cost (so a brand-new call starts slightly negative, like a real position).",
}


# ── LLM model usage (Entry Performance tab → "LLM models used" section) ───────
# Exact model ids per provider. Sources of truth in the code:
#   synthesis Claude   → settings.analyst_model
#   synthesis DeepSeek → claude_analyst._DEEPSEEK_ANALYST_MODEL  ("deepseek-v4-flash")
#   sentiment DeepSeek → sentiment.DEEPSEEK_MODEL                ("deepseek-v4-flash")
#   sentiment Claude   → sentiment.HAIKU_MODEL                   ("claude-haiku-4-5-20251001")
_PROVIDER_LABEL = {
    "anthropic": "Anthropic (Claude)", "deepseek": "DeepSeek", "qwen": "Qwen",
    "rule-based": "Rule-based", "none": "—", "": "—",
}
_SENTIMENT_MODEL = {
    "deepseek": "deepseek-v4-flash",
    "anthropic": "claude-haiku-4-5-20251001",
    "qwen": settings.qwen_model,
    "none": "(none — cached / no LLM call)",
}


def _synthesis_model(provider) -> str:
    """Exact model id that produced the final synthesis for a given provider."""
    p = (provider or "").lower()
    if p == "anthropic":
        return settings.analyst_model           # the configured Claude model
    if p == "deepseek":
        return "deepseek-v4-flash"               # DeepSeek V4-Flash analyst fallback
    if p == "qwen":
        return settings.qwen_model                # Qwen3.7-Max — 2026-07-11 primary
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
                    dcc.Tab(label="Entry Performance", value="methods",
                            children=dcc.Loading(html.Div(_safe(_methods_tab), style=body))),
                    dcc.Tab(label="Exit Performance", value="exit_perf",
                            children=dcc.Loading(html.Div(_safe(_exit_perf_tab), style=body))),
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


# ── Time-window toggle (shared by the Entry Performance & Returns tabs) ──────
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


# ── Trading-session toggle (RTH / pre-market / after-hours / overnight) ───────
_SESSION_OPTIONS = [
    {"label": "All sessions", "value": "all"},
    {"label": "RTH", "value": "rth"},
    {"label": "Pre-market", "value": "premarket"},
    {"label": "After-hours", "value": "afterhours"},
    {"label": "Overnight", "value": "overnight"},
]


_SESSION_TOGGLE_TITLE = (
    "Filter to trades entered during Regular hours (09:30–16:00 ET), Pre-market "
    "(04:00–09:30), After-hours / post-market (16:00–20:00), or Overnight "
    "(20:00–04:00). The bot trades all four sessions (overnight on the Sun–Thu-night "
    "venue calendar).")


def _session_toggle(component_id: str, title: str = _SESSION_TOGGLE_TITLE) -> html.Div:
    """RTH / pre-market / after-hours / overnight selector. Filters the tab's
    metrics and plots to that US-market session (what "session" means per tab is
    stated in ``title`` — trades are filtered by ENTRY session, exit analyses by
    the review/exit moment, panels by signal-generation time). Pre-market and
    After-hours are the two halves of the coarse 'extended' session."""
    return html.Div(
        [
            html.Label("Session:  ",
                       title=title,
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
    """RadioItems value → session string ('rth'|'premarket'|'afterhours'|'overnight'),
    or None for all. 'premarket'/'afterhours' are the two halves of 'extended'."""
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


# ── Asset-type toggle (stocks / ETFs / commodities) ──────────────────────────
_ASSET_OPTIONS = [
    {"label": "All types", "value": "all"},
    {"label": "Stocks", "value": "stock"},
    {"label": "ETFs", "value": "etf"},
    {"label": "Commodities", "value": "commodity"},
]


def _asset_toggle(component_id: str) -> html.Div:
    """Instrument-type selector (Stocks / ETFs / Commodities / all). Filters the
    tab's metrics and plots to trades whose instrument ``type`` matches (the same
    STOCK / ETF / COMMODITY label stored at entry)."""
    return html.Div(
        [
            html.Label("Type:  ",
                       title="Filter to a single instrument type: individual Stocks, ETFs "
                             "(sector / factor / index funds), or Commodities (metals, energy, "
                             "agriculture ETFs). 'All types' = every instrument.",
                       style={"cursor": "help", "borderBottom": "1px dotted #cbd5e1", "marginRight": 4}),
            dcc.RadioItems(
                id=component_id, options=_ASSET_OPTIONS, value="all", inline=True,
                persistence=True, persistence_type="session",
                inputStyle={"marginLeft": 14, "marginRight": 4},
                labelStyle={"cursor": "pointer"},
            ),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": 12},
    )


def _asset_value(value):
    """RadioItems value → 'stock' | 'etf' | 'commodity', or None for all types."""
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
                             "All scored tickers (simulated): one simulated trade per NEW directional call a method makes (the run it first called the direction — not one per run/day), scored on GROSS forward returns at 30m/3h/6h/1d/3d/1w/2w/1m — the unbiased directional-predictiveness view. Honors the Window toggle (by signal date), Session (the session the ENTRY was decided in — sessions partition the trades, so All = their sum), and Direction (the side of the method's call — a positive score is its long call).",
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


# ── Tab 2: Entry Performance ────────────────────────────────────────────────

_IC_TOOLTIP = (
    "Spearman rank correlation between each method's score and the forward "
    "close-to-close return at 1/5/10 trading-day horizons. Computed over the "
    "persisted signals panel — EVERY scored ticker each run, not just the few that "
    "became trades — so it is unbiased by the trading gates. Split into five "
    "categories: the 8 OHLCV methods computed on 30-min, daily, and weekly candles "
    "(the SAME indicators on different bar sizes), Fundamentals, and Other (news, "
    "sentiment, smart money, options, catalysts — most-recent data). The signed trend "
    "signals (Kaufman efficiency / ADX·DMI) are evaluated separately in the 'Stock "
    "discovery — trend signal IC by direction' table below. 'Sim win %' = simulated solo win "
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
    "Each method's simulated solo ENTRIES — one trade per NEW directional call: the run "
    "where the method first called the direction (its first view, a sign flip, or a "
    "re-emerged call after a gap), NOT one per run/day, so a standing call isn't "
    "pseudo-replicated and the Session buckets are a true partition (All sessions = the "
    "sum of the four sessions). Scored on GROSS close-to-close forward returns from the "
    "entry tick (no costs; the question is directional predictiveness, not net P&L). This "
    "is the unbiased counterpart to the ledger solo table: every scored ticker counts, "
    "not only the gate-selected trades that opened. 'Trades' = the method's entry events. Per horizon (30m/3h/6h/1d/3d/1w/2w/1m): "
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


def _sim_column_filters(h_id: str = "sim-horizons", m_id: str = "sim-metrics") -> html.Div:
    """Horizons + metrics multi-selects that trim the columns of every simulated
    table below. Empty selection falls back to all (never an empty table). The
    ids are parameterised so the Method-Performance and Exit-Performance tabs each
    get an independent pair (duplicate component ids would break Dash)."""
    return html.Div(
        [
            html.Label("Simulated columns:  ",
                       title="Pick which horizons and which metrics (n / IC / Win % / Ret %) "
                             "appear in the category tables below. Applies to all "
                             "category tables at once; clearing a picker shows everything.",
                       style={"cursor": "help", "borderBottom": "1px dotted #cbd5e1", "marginRight": 8}),
            dcc.Dropdown(id=h_id,
                         options=[{"label": h, "value": h} for h in _SIM_HORIZONS],
                         value=list(_SIM_HORIZONS), multi=True, placeholder="Horizons…",
                         persistence=True, persistence_type="session",
                         style={"flex": 2, "minWidth": 320}),
            dcc.Dropdown(id=m_id,
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


def _simulated_perf_section(window_days, session=None, direction=None,
                            sel_horizons=None, sel_metrics=None):
    """Per-method directional win rate + IC + gross return over ALL scored tickers
    (the simulated_trades panel), grouped by the same IC categories. The
    horizons/metrics selections trim every category table's columns. ``session``
    filters by signal-generation session; ``direction`` by the method's own call
    side (a filter caption states any active filter so the basis is unambiguous)."""
    from src.performance.tracker import METHOD_LABELS
    from src.analysis.signal_panel import IC_CATEGORY_ORDER
    sel_horizons = sel_horizons or list(_SIM_HORIZONS)
    sel_metrics = sel_metrics or list(_SIM_METRIC_ORDER)
    df = data.simulated_method_perf(days=window_days, session=session, direction=direction)
    heading = _h3("Simulated single-method performance — all scored tickers", _SIM_PERF_TOOLTIP)
    filt_bits = []
    if session:
        filt_bits.append(f"session = {session} (signal-generation time)")
    if direction:
        filt_bits.append(f"direction = {direction} calls only")
    filt_note = (html.Div("Filtered: " + " · ".join(filt_bits),
                          style={"color": "#94a3b8", "fontSize": 12, "marginBottom": 8})
                 if filt_bits else None)
    if df is None or getattr(df, "empty", True):
        return html.Div([
            heading,
            *( [filt_note] if filt_note is not None else [] ),
            html.Div("No simulated single-method trades with forward returns match "
                     "this window/session/direction yet. They accrue every run; "
                     "materialise existing history with "
                     "`python -m src.analysis.simulated_trades --backfill`.",
                     style={"color": "#6b7280"}),
        ])
    labels = dict(METHOD_LABELS)
    labels["combined_score"] = "All methods (combined)"
    children = [heading] + ([filt_note] if filt_note is not None else [])
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


_POLICY_TOOLTIP = (
    "Offline policy evaluation — the COUNTERFACTUAL P&L of alternative sizing policies, "
    "replayed over the signals panel (every scored ticker, not just the trades that opened, "
    "so it's free of the ledger's selection bias). Because forward returns are observable for "
    "EVERY candidate, no importance-sampling is needed — each policy is simply replayed with "
    "known outcomes. All rows share the SAME actionable gate, so they trade the identical set "
    "(same 'Decisions' and equal-weighted 'Avg net %'); they differ ONLY in how they SIZE, "
    "which shows up in 'Cap-wtd net %' (capital-weighted return — where the sizing actually "
    "lands the money). A sizing policy EARNS ITS KEEP only if its cap-weighted return beats "
    "'flat (gate only)'. 'Info ratio' is mean/std of the per-day return (each day counted once). "
    "Net of the same calibrated round-trip cost the ledger charges. Forward-collected + thin "
    "(a 5-day horizon needs 5 sessions of cache past each signal) — directional, not yet "
    "conclusive; watch it thicken.")


def _policy_eval_section():
    """Head-to-head counterfactual of the sizing policies over the unbiased
    signals panel — the standing answer to 'does breadth/confidence sizing earn
    its keep?'. Shown at 1-day and 5-day horizons (short horizons fill in first)."""
    children = [_h3("Sizing policy comparison — counterfactual (offline eval)", _POLICY_TOOLTIP)]
    any_data = False
    for h in (1, 5):
        df = data.policy_comparison(days=90, horizon=h)
        if df is None or getattr(df, "empty", True):
            continue
        any_data = True
        rows, flat = [], None
        for _, r in df.iterrows():
            if str(r["policy"]).startswith("flat"):
                flat = r.get("cap_wtd_ret")
        for _, r in df.iterrows():
            cw = r.get("cap_wtd_ret")
            vs = (round(float(cw) - float(flat), 3)
                  if cw is not None and flat is not None and not str(r["policy"]).startswith("flat")
                  else None)
            rows.append({
                "policy": r["policy"], "decisions": r.get("n_decisions"), "days": r.get("n_days"),
                "win": r.get("win_rate"), "avg": r.get("avg_net_ret"),
                "capwtd": cw, "vs_flat": vs, "ir": r.get("info_ratio"),
            })
        cols = [
            {"name": "Policy", "id": "policy"},
            {"name": "Decisions", "id": "decisions", "type": "numeric", "format": _INT},
            {"name": "Days", "id": "days", "type": "numeric", "format": _INT},
            {"name": "Win %", "id": "win", "type": "numeric", "format": _NUM2},
            {"name": "Avg net %", "id": "avg", "type": "numeric", "format": _NUM2},
            {"name": "Cap-wtd net %", "id": "capwtd", "type": "numeric", "format": _NUM2},
            {"name": "vs flat", "id": "vs_flat", "type": "numeric", "format": _NUM2},
            {"name": "Info ratio", "id": "ir", "type": "numeric", "format": _NUM2},
        ]
        children.append(html.Div(f"{h}-day horizon", style={
            "fontWeight": "bold", "marginTop": 12, "marginBottom": 4, "color": "#cbd5e1"}))
        children.append(dash_table.DataTable(
            data=rows, columns=cols,
            style_data_conditional=[
                {"if": {"filter_query": "{vs_flat} > 0", "column_id": "vs_flat"},
                 "color": figures.POS, "fontWeight": "bold"},
                {"if": {"filter_query": "{vs_flat} < 0", "column_id": "vs_flat"},
                 "color": figures.NEG, "fontWeight": "bold"},
                {"if": {"filter_query": '{policy} contains "flat"'}, "backgroundColor": "#1f2937"},
            ],
            **_TABLE_KW))
    if not any_data:
        children.append(html.Div(
            "No decidable decisions yet — the signals panel needs forward-return history "
            "(warm it with `python -m src.analysis.signal_panel --refresh`).",
            style={"color": "#6b7280"}))
    return html.Div(children)


_PREDICT_TOOLTIP = (
    "Predictability-feature IC — the measurement behind 'find stocks whose direction is easier to "
    "forecast for swing trading'. For every scored ticker it computes cheap per-stock features from "
    "OHLCV **as of the signal date** (no look-ahead), buckets the whole signals panel into quantiles "
    "of each feature, and reports how well the aggregate combined_score predicted the forward return "
    "INSIDE each bucket. Features: Kaufman trend efficiency (20d — clean move vs chop), ADX (14d — "
    "trend strength), realized volatility (20d), and signal breadth (methods agreeing). Per horizon: "
    "'IC' = Spearman(score, forward return), 'hit %' = directional accuracy, 'sim %' = mean signed "
    "return. The signal is SEPARATION ACROSS BUCKETS, not the level: a feature is a useful "
    "predictability filter iff hit/IC/sim climb Low→High (trend features) or peak in the MID bucket "
    "(volatility). Uses the whole panel (features are OHLCV-derived, not stamp-dependent), so it has "
    "signal now — but it is ~2 weeks of one regime; treat as directional, and lean on the 5-day "
    "swing horizon (10-day n is still thin).")

_PREDICT_EDGE_TOOLTIP = (
    "The punchline: per feature, how much its buckets SEPARATE prediction quality — the "
    "best-minus-worst-bucket spread in hit % and signed-return %, and which bucket is best, per "
    "horizon. A large spread with a sensible best bucket (High for trend efficiency / ADX, Mid for "
    "volatility) means that feature sorts predictable names from unpredictable ones and is worth "
    "promoting to a discovery-prioritisation / sizing tilt (Tier 1). A spread ≈ 0, or a 'best' "
    "bucket that contradicts the hypothesis, means it doesn't.")

_PRED_HORIZONS = (1, 5, 10)

_PRICEVOL_TOOLTIP = (
    "Do penny / thin-volume names behave differently from pricier / liquid ones — the question behind "
    "widening the discovery filter to < $1 / < $5M? Two datasets on fixed, interpretable bands (aligned "
    "to the $1/$5 price and $5M/$20M dollar-volume gate thresholds): (1) realized TRADE returns from the "
    "ledger — the strategy's actual P&L, by the trade's entry price and the stock's as-of-entry 20-day "
    "dollar volume (small + selection-biased, read the n on each bar); (2) combined_score across the "
    "UNBIASED signals panel (thousands of rows) with the mean 5-day forward return alongside — the "
    "large-sample view of how conviction and the realized move vary across the price/volume grid. Bars "
    "are green ≥ 0 / red < 0.")


def _pv_row(figs):
    """A responsive flex row of dcc.Graphs (wraps on a narrow screen)."""
    return html.Div(
        [html.Div(dcc.Graph(figure=f), style={"flex": "1 1 440px", "minWidth": 0}) for f in figs],
        style={"display": "flex", "flexWrap": "wrap", "gap": "10px"})


def _price_volume_section():
    """Return & score by stock price and dollar volume — the penny-vs-pricier
    divergence behind the widened discovery filter."""
    res = data.price_volume_perf()
    tr = (res or {}).get("trades") or {}
    sc = (res or {}).get("signals") or {}
    children = [_h3("Return & score by price / dollar-volume", _PRICEVOL_TOOLTIP)]

    # (1) realized trade returns from the ledger
    children.append(html.Div(
        f"Realized trade return — {tr.get('n_trades', 0)} trades "
        f"({tr.get('n_with_dvol', 0)} with a volume read). Small, selection-biased sample — "
        "watch the n on each bar.",
        style={"fontWeight": "bold", "marginTop": 8, "marginBottom": 4, "color": "#cbd5e1"}))
    children.append(_pv_row([
        figures.bucket_bar_fig(tr.get("by_price"), "Trade return by stock price", "Avg return %", pct=True),
        figures.bucket_bar_fig(tr.get("by_dvol"), "Trade return by dollar volume", "Avg return %", pct=True),
    ]))

    # (2) combined_score across the unbiased panel + the 5-day forward return
    children.append(html.Div(
        f"Signal conviction & realized move — combined_score and mean 5-day forward return over "
        f"{sc.get('n_rows', 0):,} unbiased signals-panel rows.",
        style={"fontWeight": "bold", "marginTop": 12, "marginBottom": 4, "color": "#cbd5e1"}))
    children.append(_pv_row([
        figures.bucket_bar_fig(sc.get("by_price"), "Score by stock price", "Avg combined_score"),
        figures.bucket_bar_fig(sc.get("by_dvol"), "Score by dollar volume", "Avg combined_score"),
    ]))
    children.append(_pv_row([
        figures.bucket_bar_fig(sc.get("fwd_by_price"), "Forward 5d return by price (unbiased)", "Avg 5d fwd %", pct=True),
        figures.bucket_bar_fig(sc.get("fwd_by_dvol"), "Forward 5d return by dollar volume (unbiased)", "Avg 5d fwd %", pct=True),
    ]))
    return html.Div(children)


def _predictability_section():
    """Bucketed conditional IC of combined_score by per-stock predictability
    feature — the Tier-0 measurement of which features make our direction call
    more forecastable at a swing horizon."""
    res = data.predictability()
    buckets = res.get("buckets") if isinstance(res, dict) else None
    edges = res.get("edges") if isinstance(res, dict) else None
    heading = _h3("Predictability by stock feature (conditional IC)", _PREDICT_TOOLTIP)
    if buckets is None or getattr(buckets, "empty", True):
        return html.Div([heading, html.Div(
            "No signals-panel rows with forward returns yet — warm forward closes with "
            "`python -m src.analysis.signal_panel --refresh`. Accrues every run.",
            style={"color": "#6b7280"})])

    children = [heading]

    # ── the edge summary (headline) ──
    if edges is not None and not getattr(edges, "empty", True):
        erows = []
        for _, r in edges.iterrows():
            row = {"label": r["label"]}
            for h in _PRED_HORIZONS:
                hs, hb, ss = (r.get(f"hit_spread_{h}d"), r.get(f"hit_best_{h}d"),
                              r.get(f"simret_spread_{h}d"))
                row[f"hitsp_{h}d"] = round(float(hs), 2) if pd.notna(hs) else None
                row[f"best_{h}d"] = hb if (hb is not None and pd.notna(hb)) else "—"
                row[f"simsp_{h}d"] = round(float(ss), 3) if pd.notna(ss) else None
            erows.append(row)
        ecols = [{"name": "Feature", "id": "label"}]
        for h in _PRED_HORIZONS:
            ecols += [
                {"name": f"Hit spread@{h}d %", "id": f"hitsp_{h}d", "type": "numeric", "format": _NUM2},
                {"name": f"Best@{h}d", "id": f"best_{h}d"},
                {"name": f"Sim spread@{h}d %", "id": f"simsp_{h}d", "type": "numeric", "format": _NUM2},
            ]
        children += [
            html.Div("Feature edge — best-minus-worst bucket separation", style={
                "fontWeight": "bold", "marginTop": 8, "marginBottom": 4, "color": "#cbd5e1"}),
            html.Div("Larger spread = the feature sorts predictable from unpredictable names. "
                     "'Best' should be High for trend efficiency / ADX, Mid for volatility.",
                     title=_PREDICT_EDGE_TOOLTIP,
                     style={"color": "#94a3b8", "fontSize": 12, "marginBottom": 6, "cursor": "help"}),
            dash_table.DataTable(data=erows, columns=ecols, **_TABLE_KW),
        ]

    # ── the bucketed detail ──
    brows, cond = [], []
    for _, r in buckets.iterrows():
        is_base = r["feature"] == "(all rows)"
        rng = "all" if pd.isna(r.get("lo")) else f"[{float(r['lo']):g}, {float(r['hi']):g}]"
        row = {"label": "BASELINE (all)" if is_base else r["label"],
               "bucket": "—" if is_base else r["bucket"], "range": rng,
               "n_rows": int(r["n_rows"])}
        for h in _PRED_HORIZONS:
            n, ic, hit, sim = (r.get(f"n_{h}d"), r.get(f"ic_{h}d"),
                               r.get(f"hit_{h}d"), r.get(f"simret_{h}d"))
            row[f"n_{h}d"] = int(n) if pd.notna(n) else 0
            row[f"ic_{h}d"] = round(float(ic), 3) if pd.notna(ic) else None
            row[f"hit_{h}d"] = round(float(hit), 1) if pd.notna(hit) else None
            row[f"sim_{h}d"] = round(float(sim), 2) if pd.notna(sim) else None
        brows.append(row)
    bcols = [{"name": "Feature", "id": "label"}, {"name": "Bucket", "id": "bucket"},
             {"name": "Range", "id": "range"},
             {"name": "Rows", "id": "n_rows", "type": "numeric", "format": _INT}]
    for h in _PRED_HORIZONS:
        bcols += [
            {"name": f"n@{h}d", "id": f"n_{h}d", "type": "numeric", "format": _INT},
            {"name": f"IC@{h}d", "id": f"ic_{h}d", "type": "numeric", "format": _NUM2},
            {"name": f"Hit@{h}d %", "id": f"hit_{h}d", "type": "numeric", "format": _NUM2},
            {"name": f"Sim@{h}d %", "id": f"sim_{h}d", "type": "numeric", "format": _NUM2},
        ]
        for c in (f"ic_{h}d", f"sim_{h}d"):
            cond += [
                {"if": {"filter_query": f"{{{c}}} > 0", "column_id": c}, "color": figures.POS},
                {"if": {"filter_query": f"{{{c}}} < 0", "column_id": c}, "color": figures.NEG},
            ]
        hc = f"hit_{h}d"
        cond += [
            {"if": {"filter_query": f"{{{hc}}} >= 50", "column_id": hc}, "color": figures.POS},
            {"if": {"filter_query": f"{{{hc}}} < 50", "column_id": hc}, "color": figures.NEG},
        ]
    cond.append({"if": {"filter_query": '{bucket} = "—"'}, "backgroundColor": "#1f2937"})
    children += [
        html.Div("Bucket detail — combined_score prediction quality within each feature bucket",
                 style={"fontWeight": "bold", "marginTop": 14, "marginBottom": 4, "color": "#cbd5e1"}),
        dash_table.DataTable(data=brows, columns=bcols, style_data_conditional=cond, **_TABLE_KW),
    ]
    return html.Div(children)


_SOURCE_PERF_TOOLTIP = (
    "Discovery-source performance — which parts of the universe-construction funnel actually "
    "surface names that move. The pipeline stamps the FIRST discovery source that surfaced each "
    "ticker (watchlist / trending / screener / macro→holdings / smart_money / sector_etf / "
    "related-company / catalyst …) onto every scored row; this groups the signals panel (every "
    "scored ticker, joined with forward returns — NOT just the gate-selected trades, so it's free "
    "of selection bias) by that stamp. 'Rows' = scored ticker-rows; 'Funnel %' = the source's "
    "slice of the STAMPED funnel (the '(unstamped)' bucket is pre-stamp history — the stamp is "
    "forward-collected from 2026-07-03 — so it is excluded from that denominator and sinks to the "
    "bottom; it still carries forward returns while the freshly-stamped sources' fill in). Per "
    "horizon (1d/5d/10d): 'n@' = rows with a forward return; 'Fwd ret@ %' = mean RAW forward "
    "return of the source's names (discovery quality — do these names tend to rise?, "
    "direction-agnostic); 'Win@ %' = share of moved names that rose; 'IC@' = Spearman correlation "
    "of the aggregate combined_score against the forward return (signal skill ON this source's "
    "names — a persistent NEGATIVE IC means the source's names 'trade but predict backwards'). A "
    "source earns more discovery budget when it combines a large-enough n with a positive Fwd ret "
    "AND a non-negative IC; a big-share source that is flat-return / negative-IC is funnel noise "
    "the near-zero-IC gates then have to sift. Run/forward-return based (ignores the window "
    "toggle); forward-collected — n grows every run, judge nothing on a thin panel.")

_SOURCE_TRADE_TOOLTIP = (
    "Realized trade outcomes grouped by the discovery source that first surfaced the ticker — the "
    "small-n, gate-selected counterpart to the forward-return table above (this is what actually "
    "opened and made or lost money). Direction is baked into the return, so a win is simply "
    "return > 0. Open trades contribute their live mark-to-market. Judge alongside the unbiased "
    "panel view — a handful of trades from one source is anecdote, not evidence.")

_SRC_HORIZONS = (1, 5, 10)


def _source_perf_section():
    """Per-discovery-source forward-return performance over the signals panel
    (the unbiased accumulator behind an adaptive discovery budget) plus the
    realized per-source trade outcomes from the ledger."""
    children = [_h3("Discovery source performance (forward returns by provenance)",
                    _SOURCE_PERF_TOOLTIP)]

    perf = data.source_performance()
    if perf is None or getattr(perf, "empty", True):
        children.append(html.Div(
            "No per-source signal rows with forward returns yet — the signals panel accrues "
            "every run (warm forward closes with `python -m src.analysis.signal_panel --refresh`).",
            style={"color": "#6b7280"}))
    else:
        rows = []
        for _, r in perf.iterrows():
            fp = r.get("funnel_pct")
            row = {"source": r["source"], "rows": int(r["rows"]),
                   "funnel_pct": round(float(fp), 1) if pd.notna(fp) else None}
            for h in _SRC_HORIZONS:
                n, fwd, win, ic = (r.get(f"n_{h}d"), r.get(f"fwd_{h}d"),
                                   r.get(f"win_{h}d"), r.get(f"ic_{h}d"))
                row[f"n_{h}d"] = int(n) if pd.notna(n) else 0
                row[f"fwd_{h}d"] = round(float(fwd), 2) if pd.notna(fwd) else None
                row[f"win_{h}d"] = round(float(win), 1) if pd.notna(win) else None
                row[f"ic_{h}d"] = round(float(ic), 3) if pd.notna(ic) else None
            rows.append(row)
        cols = [{"name": "Source", "id": "source"},
                {"name": "Rows", "id": "rows", "type": "numeric", "format": _INT},
                {"name": "Funnel %", "id": "funnel_pct", "type": "numeric", "format": _NUM2}]
        cond = []
        for h in _SRC_HORIZONS:
            cols += [
                {"name": f"n@{h}d", "id": f"n_{h}d", "type": "numeric", "format": _INT},
                {"name": f"Fwd ret@{h}d %", "id": f"fwd_{h}d", "type": "numeric", "format": _NUM2},
                {"name": f"Win@{h}d %", "id": f"win_{h}d", "type": "numeric", "format": _NUM2},
                {"name": f"IC@{h}d", "id": f"ic_{h}d", "type": "numeric", "format": _NUM2},
            ]
            for c in (f"fwd_{h}d", f"ic_{h}d"):
                cond += [
                    {"if": {"filter_query": f"{{{c}}} > 0", "column_id": c}, "color": figures.POS},
                    {"if": {"filter_query": f"{{{c}}} < 0", "column_id": c}, "color": figures.NEG},
                ]
            wc = f"win_{h}d"
            cond += [
                {"if": {"filter_query": f"{{{wc}}} >= 50", "column_id": wc}, "color": figures.POS},
                {"if": {"filter_query": f"{{{wc}}} < 50", "column_id": wc}, "color": figures.NEG},
            ]
        children.append(dash_table.DataTable(data=rows, columns=cols,
                                             style_data_conditional=cond, **_TABLE_KW))

    trade_perf = data.source_trade_perf()
    children.append(_h3("Realized trades by discovery source", _SOURCE_TRADE_TOOLTIP))
    if not trade_perf:
        children.append(html.Div("No attributed trades yet.", style={"color": "#6b7280"}))
    else:
        tcols = [
            {"name": "Source", "id": "source"},
            {"name": "Trades", "id": "trades", "type": "numeric", "format": _INT},
            {"name": "Win %", "id": "win_rate", "type": "numeric", "format": _NUM2},
            {"name": "Avg return %", "id": "avg_return", "type": "numeric", "format": _NUM2},
            {"name": "Median %", "id": "median_return", "type": "numeric", "format": _NUM2},
            {"name": "Best %", "id": "best", "type": "numeric", "format": _NUM2},
            {"name": "Worst %", "id": "worst", "type": "numeric", "format": _NUM2},
        ]
        tcond = [
            {"if": {"filter_query": "{avg_return} > 0", "column_id": "avg_return"}, "color": figures.POS},
            {"if": {"filter_query": "{avg_return} < 0", "column_id": "avg_return"}, "color": figures.NEG},
            {"if": {"filter_query": "{win_rate} >= 50", "column_id": "win_rate"}, "color": figures.POS},
            {"if": {"filter_query": "{win_rate} < 50", "column_id": "win_rate"}, "color": figures.NEG},
        ]
        children.append(dash_table.DataTable(data=trade_perf, columns=tcols,
                                             style_data_conditional=tcond, **_TABLE_KW))
    return html.Div(children)


_MC_METHODS_TOOLTIP = (
    "Monte Carlo luck-vs-skill — is each method's track record statistically distinguishable "
    "from a coin flip at its sample size? Judged on the GROSS solo win rate (sign(score) × raw "
    "price move, pre-cost) — the exact number the win-rate method filter selects on, same train "
    "split. Two resampling tests per method (2000 sims, fixed seed): BOOTSTRAP resamples the "
    "method's own trades with replacement → the 5–95% CI on its win rate and mean oriented "
    "return (how wide is the evidence); PERMUTATION NULL replaces every direction call with a "
    "fair coin on the same |price moves| → p(luck) = probability a NO-SKILL method would post "
    "at least this win rate by chance (p(ret) = same test on the mean return — a method right "
    "on the BIG moves scores better here than raw hit rate shows). One-sided: p < 0.05 ⇒ "
    "evidence of real skill; p > 0.95 ⇒ reliably WORSE than chance (inversion candidate); "
    "anything between = the record is consistent with noise — a keep/drop decision based on it "
    "is provisional. 'Filter state' shows what the live win-rate filter did with the method. "
    "The selection-bias line above the table runs the WHOLE filter on synthetic coin-flip "
    "methods at the real trade counts: if chance alone would keep about as many methods as the "
    "filter kept, the current kept set is not yet evidence of skill (expect churn as trades "
    "accrue). Small samples move these p-values a lot — re-read as the ledger grows.")


def _mc_overfit_section():
    """Monte Carlo overfitting check — per-method luck-vs-skill + the win-rate
    filter's selection-bias null (src/analysis/monte_carlo.py)."""
    rep = data.monte_carlo_methods()
    rows = rep.get("rows") or []
    heading = _h3("Overfitting check — Monte Carlo luck vs skill", _MC_METHODS_TOOLTIP)
    if not rows:
        return html.Div([heading, html.Div(
            "No closed trades with method attribution yet — accrues with the ledger.",
            style={"color": "#6b7280"})])

    sel = rep.get("selection") or {}
    sel_line = None
    if sel.get("n_judgeable"):
        sel_line = html.Div(
            f"Win-rate filter selection-bias null: at the real per-method trade counts, pure "
            f"chance would keep {sel['kept_null_mean']} ± {sel['kept_null_sd']} of "
            f"{sel['n_judgeable']} judgeable methods (5–95%: {sel['kept_null_lo']}–"
            f"{sel['kept_null_hi']}); the live filter kept {sel['kept_actual']} "
            f"(p ≥ actual = {sel['p_ge_actual']}). → {sel.get('verdict', '')}",
            style={"color": "#cbd5e1", "marginBottom": 8})
    elif sel.get("verdict"):
        sel_line = html.Div(sel["verdict"], style={"color": "#6b7280", "marginBottom": 8})

    trows = [{
        "method": r["method"], "state": r.get("filter_state", "—"), "n": r["n"],
        "wr": r["win_rate"], "wr_ci": f"{r['wr_lo']:.0f} – {r['wr_hi']:.0f}",
        "p_luck": r["p_luck"],
        "ret": r["mean_ret"], "ret_ci": f"{r['ret_lo']:.2f} – {r['ret_hi']:.2f}",
        "p_ret": r["p_ret"], "verdict": r["verdict"],
    } for r in rows]
    tcols = [
        {"name": "Method", "id": "method"},
        {"name": "Filter state", "id": "state"},
        {"name": "Trades", "id": "n", "type": "numeric", "format": _INT},
        {"name": "Gross WR %", "id": "wr", "type": "numeric", "format": _NUM1},
        {"name": "WR CI 5–95%", "id": "wr_ci"},
        {"name": "p (luck)", "id": "p_luck", "type": "numeric", "format": _NUM3},
        {"name": "Mean ret %", "id": "ret", "type": "numeric", "format": _NUM2},
        {"name": "Ret CI 5–95%", "id": "ret_ci"},
        {"name": "p (ret)", "id": "p_ret", "type": "numeric", "format": _NUM3},
        {"name": "Verdict", "id": "verdict"},
    ]
    cond = [
        {"if": {"filter_query": "{p_luck} < 0.05", "column_id": "p_luck"}, "color": figures.POS},
        {"if": {"filter_query": "{p_luck} > 0.95", "column_id": "p_luck"}, "color": figures.NEG},
        {"if": {"filter_query": "{p_ret} < 0.05", "column_id": "p_ret"}, "color": figures.POS},
        {"if": {"filter_query": "{p_ret} > 0.95", "column_id": "p_ret"}, "color": figures.NEG},
        {"if": {"filter_query": '{verdict} contains "SKILL"', "column_id": "verdict"},
         "color": figures.POS},
        {"if": {"filter_query": '{verdict} contains "worse"', "column_id": "verdict"},
         "color": figures.NEG},
        {"if": {"filter_query": '{state} = "FILTERED"', "column_id": "state"},
         "color": "#6b7280"},
    ]
    return html.Div([heading] + ([sel_line] if sel_line is not None else []) + [
        dash_table.DataTable(data=trows, columns=tcols, style_data_conditional=cond,
                             **_TABLE_KW),
    ])


_CONF_COMPONENTS_TOOLTIP = (
    "Isolates each multiplier in the confidence formula (confidence = raw × coherence × "
    "movement × volume × family × tape — src/signals/aggregator.py::_score_ticker) to see "
    "which ones actually earn their keep. 'Raw score only' = min(1, |combined_score| / 0.5) "
    "with no multiplier applied; each other row applies exactly ONE factor on top of raw "
    "(capped at 1.0) — in ISOLATION, not stacked cumulatively. 'Live (all combined)' is the "
    "actual confidence the system uses today, shown as the reference row. IC = Spearman rank "
    "correlation between the variant's value and the DIRECTION-ORIENTED forward return "
    "(sign(combined_score) × forward return) — positive means higher readings of that "
    "variant genuinely predict better outcomes; ≈0 means it doesn't discriminate despite "
    "moving the number (this top table deliberately omits win%/return — ungated, those "
    "never depend on the variant's value, only IC does; they'd be redundant at best, "
    "missingness noise at worst). The conviction-band table below splits each variant's OWN "
    "value into Low (0.10–0.35) / Medium (0.35–0.65) / High "
    "(0.65+) — the same cut points tracker._eval_stats uses for per-method calibration — so "
    "you can see whether win rate / return actually RISES with that variant's own conviction "
    "(a well-behaved component shows Low < Medium < High; flat or inverted means it isn't "
    "separating good calls from bad ones). Forward-collected from 2026-07-21 when the factor "
    "columns were added to the signals panel — 0 rows at first, fills in every run.")

_CONF_COMPONENTS_EXIT_NOTE = (
    "Exit-side: the SAME isolation, but over signals-panel rows re-scored on an ALREADY-OPEN "
    "position mid-hold (oriented by the trade's own direction, not the ticker's possibly-"
    "since-drifted current call; forward return measured from the re-read tick, not the "
    "original entry) — does a component's reading, taken WHILE HOLDING, predict what happens "
    "to the position from that point on. No separate capture path: held tickers stay in the "
    "scored universe every tick, so this is a join against the trades ledger's open interval, "
    "not a new signal.")


_DEFAULT_CONF_HORIZONS = (1, 5, 10)


def _conf_component_ic_table(icdf: pd.DataFrame) -> dash_table.DataTable:
    rows = icdf.rename(columns={"label": "Variant"}).to_dict("records")
    cols = [{"name": "Variant", "id": "Variant"}]
    for h in _DEFAULT_CONF_HORIZONS:
        cols += [
            {"name": f"n@{h}d", "id": f"n_{h}d", "type": "numeric", "format": _INT},
            {"name": f"IC@{h}d", "id": f"ic_{h}d", "type": "numeric", "format": _NUM3},
            {"name": f"ICIR@{h}d", "id": f"icir_{h}d", "type": "numeric", "format": _NUM2},
        ]
    cond = [{"if": {"filter_query": '{Variant} = "Live (all combined)"'},
            "backgroundColor": "#1f2937"}]
    for h in _DEFAULT_CONF_HORIZONS:
        cond += [
            {"if": {"filter_query": f"{{ic_{h}d}} > 0.03", "column_id": f"ic_{h}d"},
             "color": figures.POS},
            {"if": {"filter_query": f"{{ic_{h}d}} < -0.03", "column_id": f"ic_{h}d"},
             "color": figures.NEG},
        ]
    return dash_table.DataTable(data=rows, columns=cols, style_data_conditional=cond, **_TABLE_KW)


def _conf_component_band_table(banddf: pd.DataFrame) -> dash_table.DataTable:
    rows = banddf.rename(columns={"label": "Variant", "band_label": "Band"}).to_dict("records")
    cols = [{"name": "Variant", "id": "Variant"}, {"name": "Band", "id": "Band"}]
    for h in _DEFAULT_CONF_HORIZONS:
        cols += [
            {"name": f"n@{h}d", "id": f"n_{h}d", "type": "numeric", "format": _INT},
            {"name": f"Win@{h}d %", "id": f"win_{h}d", "type": "numeric", "format": _NUM1},
            {"name": f"Ret@{h}d %", "id": f"ret_{h}d", "type": "numeric", "format": _NUM2},
        ]
    cond = [{"if": {"filter_query": '{Band} = "High (0.65+)"'}, "backgroundColor": "#1f2937"}]
    return dash_table.DataTable(data=rows, columns=cols, style_data_conditional=cond, **_TABLE_KW)


def _confidence_components_section():
    """Entry-side confidence-component isolation (src/analysis/confidence_components.py)."""
    rep = data.confidence_components_entry()
    heading = _h3("Confidence-formula component isolation", _CONF_COMPONENTS_TOOLTIP)
    if not rep.get("has_factors"):
        return html.Div([heading, html.Div(
            "Forward-collecting — the per-factor columns (coherence / movement / volume / "
            "family / tape) were just added to the signals panel; this fills in from the "
            "next pipeline run onward.", style={"color": "#6b7280"})])
    icdf, banddf = rep.get("ic"), rep.get("bands")
    if icdf is None or icdf.empty:
        return html.Div([heading, html.Div(
            f"{rep.get('panel_rows', 0)} signal row(s) with factor data — not enough "
            "forward-return history yet.", style={"color": "#6b7280"})])
    return html.Div([
        heading,
        html.Div(f"{rep['panel_rows']} scored ticker-tick(s) with factor data",
                 style={"color": "#cbd5e1", "marginBottom": 8}),
        _conf_component_ic_table(icdf),
        html.Div("By conviction band (does win rate / return rise with THIS variant's own "
                 "conviction level?):",
                 style={"marginTop": 14, "marginBottom": 4, "color": "#cbd5e1"}),
        _conf_component_band_table(banddf),
    ])


def _exit_confidence_components_block(session=None, direction=None):
    """Exit-side confidence-component isolation — held-position mid-hold re-reads."""
    rep = data.confidence_components_exit(session=session, direction=direction)
    note = html.Div(_CONF_COMPONENTS_EXIT_NOTE, style={"color": "#6b7280", "marginBottom": 8})
    if not rep.get("has_factors") or not rep.get("panel_rows"):
        return html.Div([note, html.Div(
            "No held-position re-reads with factor data yet — accrues once a position "
            "opened after 2026-07-21 is held past its entry day.",
            style={"color": "#6b7280"})])
    icdf, banddf = rep.get("ic"), rep.get("bands")
    if icdf is None or icdf.empty:
        return html.Div([note, html.Div(
            f"{rep.get('panel_rows', 0)} held re-read(s) — not enough forward-return "
            "history yet.", style={"color": "#6b7280"})])
    return html.Div([
        note,
        html.Div(f"{rep['panel_rows']} held-position re-read(s) with factor data",
                 style={"color": "#cbd5e1", "marginBottom": 8}),
        _conf_component_ic_table(icdf),
        html.Div("By conviction band:",
                 style={"marginTop": 14, "marginBottom": 4, "color": "#cbd5e1"}),
        _conf_component_band_table(banddf),
    ])


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
        _asset_toggle("methods-asset"),
        _method_source_toggle("methods-source"),
        _sim_column_filters(),
        dcc.Loading(html.Div(id="methods-body")),
        _safe(_ic_section),
        _safe(_mc_overfit_section),
        _safe(_confidence_components_section),
        _safe(_policy_eval_section),
        _safe(_predictability_section),
        _safe(_price_volume_section),
        _safe(_source_perf_section),
        _h3("LLM models used (synthesis & sentiment)",
            "Which exact LLMs actually ran across all recorded pipeline runs — the final-call 'synthesis' model and the per-ticker 'sentiment' model — including any DeepSeek or rule-based fallbacks. Not affected by the window toggle above (it's run-based, not trade-based). Hover a column header for details."),
        models_table,
    ])


@app.callback(Output("methods-body", "children"),
              Input("methods-window", "value"), Input("methods-session", "value"),
              Input("methods-direction", "value"), Input("methods-asset", "value"),
              Input("methods-source", "value"),
              Input("sim-horizons", "value"), Input("sim-metrics", "value"))
def _methods_body(window_value, session_value, direction_value, asset_value, source_value,
                  sim_horizons, sim_metrics):
    if source_value == "panel":
        # All scored tickers — honors the window (by signal date), the session
        # the signal was GENERATED in, the direction of the method's own call
        # (positive score = its long call), and the Horizons/Metrics pickers.
        # (The unbiased signal panel has no instrument-type column, so the Type
        # filter applies only to the ledger-based view below.)
        return _safe(lambda: _simulated_perf_section(_window_days(window_value),
                                                     _session_value(session_value),
                                                     _direction_value(direction_value),
                                                     sim_horizons, sim_metrics))
    return _safe(lambda: _methods_perf_section(_window_days(window_value), _session_value(session_value),
                                               _direction_value(direction_value),
                                               _asset_value(asset_value)))


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


def _methods_perf_section(window_days, session=None, direction=None, asset_type=None):
    perf = data.performance(window_days=window_days, session=session, direction=direction,
                            asset_type=asset_type)
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

    # Blind-synthesis A/B — entry outcomes grouped by whether the entry run hid
    # the aggregator's verdict from the synthesis prompt (ON = the LLM's own
    # independent judgment; OFF = the legacy echo-prone sighted prompt).
    bs = perf.get("blind_synthesis_eval") or {}
    for key, label in (("on", "Entry eval · blind-synthesis ON"),
                       ("off", "Entry eval · blind-synthesis OFF")):
        st = bs.get(key)
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
            {"if": {"filter_query": '{Method} contains "blind-synthesis"'}, "backgroundColor": "#fefce8"},
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

    # ── Decision funnel — per-stage evaluation (aggregator → synthesis → the
    # four actionable gates), each stage's survivors AND each gate's drops
    # scored on the same pseudo-trade basis so every step's marginal value
    # (filtered losers vs discarded winners) is directly visible.
    stage_rows = [
        {
            "Stage": r["label"],
            "Trades": r.get("trades"),
            "Win rate %": round(r["win_rate"], 1) if r.get("win_rate") is not None else None,
            "Avg return %": round(r["avg_return"], 2) if r.get("avg_return") is not None else None,
        }
        for r in (perf.get("stage_eval") or [])
    ]
    stage_table = dash_table.DataTable(
        data=stage_rows,
        columns=[{"name": c, "id": c} for c in ["Stage", "Trades", "Win rate %", "Avg return %"]],
        tooltip_header=_STAGE_HEADER_TIPS,
        style_data_conditional=[
            {"if": {"filter_query": '{Stage} contains "✂"'}, "backgroundColor": "#fef2f2"},
            {"if": {"filter_query": '{Stage} contains "ACTIONABLE"'}, "backgroundColor": "#ecfdf5"},
        ],
        **_TABLE_KW,
    ) if stage_rows else html.Div("No stage-funnel stats in this window yet.", style={"color": "#6b7280"})

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
        _h3("Decision funnel — per-stage performance (aggregator → LLM → gates 1-4)",
            "Every step of the decision pipeline evaluated on the SAME pseudo-trade basis: the mechanical "
            "aggregator's directional calls, the LLM synthesis stream, then the four actionable gates in "
            "execution order — Gate 1 regime confidence threshold, Gate 2 PANIC/RISK_OFF BUY-block, Gate 3 "
            "earnings blackout, Gate 4 tradeable-liquidity floor. Each '→ past Gate k' row is the surviving "
            "stream after that gate; the '✂' row under it is exactly what the gate discarded (gate outcomes "
            "are reconstructed exactly from each run's persisted threshold/allow_buys/actionable flags, plus "
            "the per-ticker gate_outcomes stamp on new runs). Read each gate by comparing its two rows: "
            "drops worse than survivors = the gate filters losers; drops better = it discards winners. "
            "The final green row is the actionable set the sizing layer actually received. "
            "Respects the window + session toggles above."),
        stage_table,
        _h3("Model evaluation — signal methods (solo simulation) & LLM engines",
            "Method rows: how each signal method would have performed deciding alone (each closed trade re-simulated as if only that method set the direction). "
            "Highlighted LLM rows: every BUY/SELL the engine recommended — executed or simulated — entered at the recommendation-time price, marked at the latest close, "
            "deduped to the engine's last call per ticker per day. The 50/50 A/B routing flip gives each engine its own runs to be judged on. Hover the column headers for details."),
        table,
    ])


# ── Tab: Exit Performance ──────────────────────────────────────────────────

_EXIT_PERF_TOOLTIP = (
    "Every held position is re-scored each tick with signed HOLD-CONVICTION per exit "
    "method (+ = keep running, − = reverse/exit) — the `exit_signals` panel. The table "
    "shows each method's ACTIVATION EVENTS: the tick the method first turned against the "
    "position (its conviction crossed into negative — 'the exit fired'), attributed to "
    "that tick's session, so a method repeating 'exit' for days counts once and the "
    "Session buckets are a true partition (All sessions = the sum of the four). Each "
    "activation is joined to the position's DIRECTION-ORIENTED forward return from that "
    "tick (a short's forward return is negated) and scored in the direction the method "
    "CALLED (sign(score)×forward), so the usual reading holds: 'Win@ %' > 50 = after the "
    "method said get out, the position usually DID move adversely (the exit was right; "
    "below 50 = it fires too early), 'Ret@ %' positive = the average post-activation move "
    "vindicated the exit, 'IC@' positive = deeper exit-conviction ⇒ more adverse "
    "subsequent move. Per horizon (30m…1m): 'n@' = activations with a forward return, "
    "'IC std@' / 'ICIR@' = the IC's reliability (per-day; needs several days). The "
    "synthesized `llm_review` row is history-backed from `trade_reviews`; the rest "
    "accrue as the panel fills. Forward-collected — judge nothing on a thin n.")

_EXIT_SHADOW_TOOLTIP = (
    "The SIMULATED exit book: every scored ticker (the signals panel) treated as a hypothetical "
    "position held in its own aggregate direction, with each position-independent exit method scored "
    "as a signed hold-conviction (method score × the ticker's direction; aggregator = combined_score). "
    "Shown as ACTIVATION EVENTS — the run where the method's conviction first crossed negative for "
    "that hypothetical position (a direction flip starts a new position), attributed to that run's "
    "session, so the Session buckets are a true partition (All sessions = the sum of the four). Each "
    "activation joins the direction-oriented forward return through the SAME engine as the held book, "
    "but over the WHOLE UNIVERSE instead of only the gate-selected positions we actually held — "
    "escaping the held book's tiny, selection-biased sample. Read like the held table: Win% > 50 / "
    "positive Ret / positive IC = the method's exit calls were vindicated by the subsequent move. "
    "Only the position-INDEPENDENT methods appear here (aggregator + the signal-methods-as-exits); "
    "`horizon` and the synthesized `llm_review` need a real entry, so they exist ONLY in the Held view.")


# ── Exit-perf source toggle (real held book vs simulated universe shadow) ────
_EXIT_SOURCE_OPTIONS = [
    {"label": "Held positions (ledger)", "value": "held"},
    {"label": "All scored tickers (simulated)", "value": "shadow"},
]


def _exit_source_toggle(component_id: str) -> html.Div:
    """The exit-IC evidence base. Held = the exit methods scored on the positions we
    actually held (real book; the only place horizon / llm_review exist). Simulated =
    every scored ticker as a hypothetical position held in its aggregate direction —
    the position-independent methods over the whole universe (large, unbiased)."""
    return html.Div(
        [
            html.Label("Source:  ",
                       title="Held positions (ledger): each exit method's ACTIVATIONS against the positions we ACTUALLY held — the tick it first said 'get out' (conviction crossed negative), attributed to that tick's session. The real book (small, selection-biased), and the ONLY view with horizon + the synthesized llm_review. "
                             "All scored tickers (simulated): the same activation events over EVERY scored ticker treated as a hypothetical position held in its aggregate direction (aggregator + the signal-methods-as-exits) — the large, unbiased sample backfilled from the signals panel. horizon / llm_review are held-only and don't appear here.",
                       style={"cursor": "help", "borderBottom": "1px dotted #cbd5e1", "marginRight": 4}),
            dcc.RadioItems(
                id=component_id, options=_EXIT_SOURCE_OPTIONS, value="held", inline=True,
                persistence=True, persistence_type="session",
                inputStyle={"marginLeft": 14, "marginRight": 4},
                labelStyle={"cursor": "pointer"},
            ),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": 12},
    )


def _exit_perf_section(window_days, source="held", sel_horizons=None, sel_metrics=None,
                       session=None, direction=None):
    """Per-exit-method IC / win / ret, grouped by the two exit categories. ``source``
    picks the evidence base: 'held' = the real exit_signals + trade_reviews book;
    'shadow' = the simulated universe (all scored tickers). ``session`` filters by
    the session the review happened in (the exit decision's firing moment);
    ``direction`` by the (hypothetical) position's side. A caption states which
    basis + filters are shown so the view is never ambiguous. Reuses the
    simulated-perf table renderer."""
    from src.performance.tracker import METHOD_LABELS
    from src.analysis.exit_methods import EXIT_CATEGORY_ORDER, EXIT_METHOD_LABELS
    sel_horizons = sel_horizons or list(_SIM_HORIZONS)
    sel_metrics = sel_metrics or list(_SIM_METRIC_ORDER)
    shadow = (source == "shadow")
    if shadow:
        df = data.shadow_exit_method_perf(days=window_days, session=session, direction=direction)
        heading = _h3("Exit-method performance — ALL scored tickers (simulated shadow book)",
                      _EXIT_SHADOW_TOOLTIP)
        caption = ("Source: SIMULATED — every scored ticker as a hypothetical position held in its "
                   "aggregate direction, over the whole universe (backfilled from the signals panel). "
                   "Position-independent methods only; horizon + llm_review are held-only and not shown here.")
        empty_msg = ("No simulated exit returns match this window/session/direction yet — the signals "
                     "panel needs forward-return history "
                     "(warm it with `python -m src.analysis.signal_panel --refresh`).")
    else:
        df = data.exit_method_perf(days=window_days, session=session, direction=direction)
        heading = _h3("Exit-method performance — HELD positions (ledger)", _EXIT_PERF_TOOLTIP)
        caption = ("Source: HELD (ledger) — the exit methods scored on the positions we ACTUALLY held "
                   "each tick. The only view with horizon + the synthesized llm_review; small + "
                   "selection-biased. Switch Source to 'All scored tickers' for the large simulated sample.")
        empty_msg = ("No exit-method forward returns match this window/session/direction yet. The "
                     "synthesized hold-review row populates from `trade_reviews`; the decomposed "
                     "methods accrue once positions are held.")
    if session:
        caption += f" — Session filter: {session} (review moment)."
    if direction:
        caption += f" — Direction filter: {direction} positions only."
    cap = html.Div(caption, style={"color": "#94a3b8", "fontSize": 12, "marginBottom": 8})
    if df is None or getattr(df, "empty", True):
        return html.Div([heading, cap, html.Div(empty_msg, style={"color": "#6b7280"})])
    labels = dict(METHOD_LABELS)
    labels.update(EXIT_METHOD_LABELS)
    children = [heading, cap]
    has_cat = "category" in df.columns
    for category in EXIT_CATEGORY_ORDER:
        subset = df[df["category"] == category] if has_cat else df
        if subset is None or subset.empty:
            continue
        children.append(html.Div(category, style={
            "fontWeight": "bold", "marginTop": 14, "marginBottom": 4, "color": "#cbd5e1"}))
        children.append(_sim_perf_table(subset, labels, sel_horizons, sel_metrics))
        if not has_cat:
            break
    return html.Div(children)


def _exit_reason_block(session=None, direction=None):
    """Realized outcome per exit RULE (exit_reason) over closed trades — the
    concrete, ledger-based companion to the forward-looking IC table above.
    ``session`` filters by the session the trade EXITED in; ``direction`` by
    the position's side."""
    rows = data.exit_reason_breakdown(session=session, direction=direction)
    if not rows:
        return html.Div("No closed trades match this session/direction yet.",
                        style={"color": "#6b7280"})
    data_rows = [{
        "reason": r["exit_reason"], "trades": r["trades"], "win": r["win_rate"],
        "avg": r["avg_return"], "median": r.get("median_return"),
        "compound": r["compound_return"], "best": r["best"], "worst": r["worst"],
    } for r in rows]
    cols = [
        {"name": "Exit reason", "id": "reason"},
        {"name": "Trades", "id": "trades", "type": "numeric", "format": _INT},
        {"name": "Win rate %", "id": "win", "type": "numeric", "format": _NUM2},
        {"name": "Avg return %", "id": "avg", "type": "numeric", "format": _NUM2},
        {"name": "Median %", "id": "median", "type": "numeric", "format": _NUM2},
        {"name": "Compound %", "id": "compound", "type": "numeric", "format": _NUM2},
        {"name": "Best %", "id": "best", "type": "numeric", "format": _NUM2},
        {"name": "Worst %", "id": "worst", "type": "numeric", "format": _NUM2},
    ]
    return dash_table.DataTable(data=data_rows, columns=cols, **_TABLE_KW)


def _exit_forward_block(session=None, direction=None):
    """Post-exit forward returns — what each CLOSED trade would have earned if
    held 1/3/5/10 more sessions, anchored at the actual exit fill and oriented
    by the position's side. Positive = the exit left money on the table."""
    rep = data.exit_forward(session=session, direction=direction)
    if not rep.get("n"):
        return html.Div(rep.get("verdict") or "No closed trades with post-exit bars yet.",
                        style={"color": "#6b7280"})
    hs = rep["horizons"]

    reason_rows = [{
        "reason": r["exit_reason"], "trades": r["trades"],
        **{f"mean_{h}": r.get(f"mean_{h}d") for h in hs},
        **{f"pos_{h}": r.get(f"pct_pos_{h}d") for h in hs},
    } for r in rep["by_reason"] + [{"exit_reason": "ALL exits", **rep["overall"]}]]
    reason_cols = ([{"name": "Exit reason", "id": "reason"},
                    {"name": "Trades", "id": "trades", "type": "numeric", "format": _INT}]
                   + [{"name": f"Mean +{h}d %", "id": f"mean_{h}", "type": "numeric", "format": _NUM2}
                      for h in hs]
                   + [{"name": f"%+ @{h}d", "id": f"pos_{h}", "type": "numeric", "format": _NUM1}
                      for h in hs])

    trade_rows = [{
        "ticker": r["ticker"], "exit_date": r["exit_date"], "ret": r["return_pct"],
        **{f"fwd_{h}": r.get(f"fwd_{h}d") for h in hs},
        "reason": r["exit_reason"],
    } for r in rep["per_trade"]]
    trade_cols = ([{"name": "Ticker", "id": "ticker"},
                   {"name": "Exit date", "id": "exit_date"},
                   {"name": "Realized %", "id": "ret", "type": "numeric", "format": _NUM2}]
                  + [{"name": f"Fwd +{h}d %", "id": f"fwd_{h}", "type": "numeric", "format": _NUM2}
                     for h in hs]
                  + [{"name": "Exit reason", "id": "reason"}])

    pending = (f" · {rep['n_pending']} exit(s) pending forward bars"
               if rep.get("n_pending") else "")
    return html.Div([
        html.Div(f"{rep['verdict']}  ({rep['n']} closed trade(s) with forward bars{pending})",
                 style={"color": "#cbd5e1", "marginBottom": 8}),
        dash_table.DataTable(data=reason_rows, columns=reason_cols, **_TABLE_KW),
        html.Div("Per-trade detail (most recent exits first)", style={
            "fontWeight": "bold", "marginTop": 14, "marginBottom": 4, "color": "#cbd5e1"}),
        dash_table.DataTable(data=trade_rows, columns=trade_cols, **_TABLE_KW),
    ])


_EXIT_SESSION_TITLE = (
    "Filter the exit analyses to a US-market session — Regular hours (09:30–16:00 ET), "
    "Pre-market (04:00–09:30), After-hours (16:00–20:00), or Overnight (20:00–04:00). "
    "For the exit-method tables this is the session the REVIEW happened in (the moment "
    "the exit decision would fire); for the exit-reason table below it is the session "
    "the trade actually EXITED in.")


_MC_EXITS_TOOLTIP = (
    "Monte Carlo exit-timing test — does each exit rule TIME its exits better than random, or "
    "is its realized outcome just what any exit in the same windows would have gotten? For every "
    "CLOSED trade the feasible exit window is each session close from the first session after "
    "entry through the actual hold + 10 more sessions; the RANDOM-EXIT NULL draws one uniform "
    "random exit per trade per simulation (2000 sims, fixed seed) and records the group's mean "
    "gross oriented return. Both arms anchor entry at the real entry fill and exit at SESSION "
    "CLOSES (the actual arm at the actual exit date's close) so the comparison is apples-to-"
    "apples; gross of costs (both arms pay the same one-way exit cost). 'Percentile' = where the "
    "rule's actual mean lands inside its own random-exit distribution: ≥ 95 ⇒ the rule genuinely "
    "times exits (green); ≤ 5 ⇒ RANDOM exits would have beaten the rule (red — the rule "
    "destroys timing value); anything between ≈ the rule adds no measurable timing skill — its "
    "realized P&L is explained by WHICH trades it closed, not WHEN. p(random ≥ actual) is the "
    "one-sided probability. Trades without enough cached sessions are skipped, never guessed. "
    "Small groups (n < ~10) are noise — judge only as trades accrue.")


def _exit_mc_block(session=None, direction=None):
    """Exit-timing-vs-random Monte Carlo per exit rule (src/analysis/monte_carlo.py)."""
    rep = data.monte_carlo_exits(session=session, direction=direction)
    if not rep.get("n"):
        return html.Div(rep.get("verdict") or "No closed trades with cached exit windows yet.",
                        style={"color": "#6b7280"})
    rows = [{
        "reason": r["reason"], "trades": r["trades"],
        "actual": r["actual_mean"], "null": r["null_mean"],
        "null_ci": f"{r['null_lo']:.2f} – {r['null_hi']:.2f}",
        "pctile": r["percentile"], "p_rand": r["p_random_beats"],
        "verdict": r["verdict"],
    } for r in rep["rows"]]
    cols = [
        {"name": "Exit reason", "id": "reason"},
        {"name": "Trades", "id": "trades", "type": "numeric", "format": _INT},
        {"name": "Actual mean %", "id": "actual", "type": "numeric", "format": _NUM2},
        {"name": "Random-exit mean %", "id": "null", "type": "numeric", "format": _NUM2},
        {"name": "Null CI 5–95%", "id": "null_ci"},
        {"name": "Percentile", "id": "pctile", "type": "numeric", "format": _NUM1},
        {"name": "p (random ≥ actual)", "id": "p_rand", "type": "numeric", "format": _NUM3},
        {"name": "Verdict", "id": "verdict"},
    ]
    cond = [
        {"if": {"filter_query": "{pctile} >= 95", "column_id": "pctile"}, "color": figures.POS},
        {"if": {"filter_query": "{pctile} <= 5", "column_id": "pctile"}, "color": figures.NEG},
        {"if": {"filter_query": '{verdict} contains "BETTER"', "column_id": "verdict"},
         "color": figures.POS},
        {"if": {"filter_query": '{verdict} contains "BEATEN"', "column_id": "verdict"},
         "color": figures.NEG},
        {"if": {"filter_query": '{reason} = "ALL exits"'}, "backgroundColor": "#1f2937"},
    ]
    skipped = (f" · {rep['n_skipped']} trade(s) skipped (no cached window)"
               if rep.get("n_skipped") else "")
    return html.Div([
        html.Div(f"{rep['n']} closed trade(s) in the MC{skipped}",
                 style={"color": "#cbd5e1", "marginBottom": 8}),
        dash_table.DataTable(data=rows, columns=cols, style_data_conditional=cond,
                             **_TABLE_KW),
    ])


def _exit_perf_tab():
    return html.Div([
        _window_toggle("exit-window"),
        _session_toggle("exit-session", title=_EXIT_SESSION_TITLE),
        _direction_toggle("exit-direction"),
        _exit_source_toggle("exit-source"),
        _sim_column_filters("exit-horizons", "exit-metrics"),
        dcc.Loading(html.Div(id="exit-body")),
    ])


@app.callback(Output("exit-body", "children"),
              Input("exit-window", "value"), Input("exit-session", "value"),
              Input("exit-direction", "value"), Input("exit-source", "value"),
              Input("exit-horizons", "value"), Input("exit-metrics", "value"))
def _exit_body(window_value, session_value, direction_value, source_value,
               sim_horizons, sim_metrics):
    session = _session_value(session_value)
    direction = _direction_value(direction_value)
    return html.Div([
        _safe(lambda: _exit_perf_section(_window_days(window_value), source_value,
                                         sim_horizons, sim_metrics,
                                         session=session, direction=direction)),
        _h3("Exit-reason outcomes — realized P&L by exit rule (closed ledger trades)",
            "For every CLOSED trade, the realized return grouped by the exit_reason that "
            "fired (llm_signal_flipped / llm_confidence_loss / horizon_expired / "
            "macro_regime_exit / the aggregator backstops / intraday_reversal). Always the "
            "real ledger (independent of the Source toggle above) — the concrete realized "
            "outcome of each exit rule, companion to the forward-looking IC table. Honors "
            "the Session (session the trade EXITED in) and Direction toggles; the Window "
            "toggle does not apply (closed trades are few). Open trades excluded (no exit "
            "yet)."),
        _safe(lambda: _exit_reason_block(session=session, direction=direction)),
        _h3("Post-exit forward returns — what if we had held longer? (closed ledger trades)",
            "For every CLOSED trade, the oriented return the position would have earned had it "
            "stayed on 1/3/5/10 more trading sessions — anchored at the ACTUAL exit fill "
            "(sign × (close_{exit+N} / exit_price − 1); long +, short −), forward closes from the "
            "daily OHLCV cache (recently-exited tickers stay warmed by the EOD maintenance pass). "
            "Positive = the position kept moving our way after we left it (the exit left money on "
            "the table); negative = the exit dodged a drawdown. Gross of costs (holding defers the "
            "same exit cost rather than adding one). '%+ @Nd' = share of exits still going our way "
            "at that horizon — a rule with a persistently positive mean is firing too early. "
            "Grouped by exit rule; always the real ledger. Honors the Session (exit session) and "
            "Direction toggles; the Window toggle does not apply. Exits without forward bars yet "
            "(closed today / cache gap) are counted as pending, never guessed."),
        _safe(lambda: _exit_forward_block(session=session, direction=direction)),
        _h3("Exit timing vs random exits — Monte Carlo (closed ledger trades)",
            _MC_EXITS_TOOLTIP),
        _safe(lambda: _exit_mc_block(session=session, direction=direction)),
        _h3("Confidence-formula component isolation — held positions", _CONF_COMPONENTS_TOOLTIP),
        _safe(lambda: _exit_confidence_components_block(session=session, direction=direction)),
        _safe(_horizon_edge_section),
        _safe(_exit_policy_eval_section),
    ])


_HORIZON_EDGE_TOOLTIP = (
    "The realized EDGE-DECAY of combined_score by holding horizon — measured tick-by-tick, "
    "ticker-by-ticker over the whole signals panel (every scored ticker at every tick is a "
    "hypothetical entry in its signal's direction), restricted to the ACTIONABLE subset "
    "(confidence ≥ 0.85 — the traded population). This is the ground truth the horizon time-stop "
    "rests on, at thousands of observations where the held-position `horizon` IC can't reach "
    "(only ~5 real positions have ever outlived their window). Per horizon: 'n' = observations, "
    "'IC' = Spearman(combined_score, forward return), 'win %' = directional hit, 'edge %' = mean "
    "sign(score)×forward-return (the P&L of following the signal that long). A peak-then-decay "
    "shape (edge positive early, ≤0 later) justifies a time-stop at the decay point. The measured "
    "**edge window** (last horizon with positive edge) drives the `edge_decay` exit stop, which "
    "evidence-throttled raises the close floor once a position is held past it. Run-based; "
    "forward-collected — long horizons thin until the cache warms; ~one regime so far.")


def _horizon_edge_section():
    """The realized edge-decay curve + the calibrated edge-window that feeds the
    edge-decay time-stop."""
    res = data.horizon_edge_curve()
    curve = res.get("curve") if isinstance(res, dict) else None
    cal = res.get("cal") if isinstance(res, dict) else {}
    heading = _h3("Signal edge by holding horizon (edge-decay time-stop)", _HORIZON_EDGE_TOOLTIP)
    if curve is None or getattr(curve, "empty", True):
        return html.Div([heading, html.Div(
            "No forward-return history yet — accrues every run.", style={"color": "#6b7280"})])
    ed, strength, peak = cal.get("edge_days"), cal.get("strength", 0.0), cal.get("peak_day")
    cap = (f"Measured edge window: {ed} trading day(s) (peak ~{peak}d) · "
           f"time-stop strength {strength:.2f} (evidence-throttled)"
           if ed else "No edge-decay window measured yet (edge not yet observed to turn "
                      "negative) — the stop stays inert until it does.")
    rows = []
    for _, r in curve.iterrows():
        rows.append({"horizon": f"{int(r['horizon'])}d", "n": int(r["n"]),
                     "ic": round(float(r["ic"]), 3) if pd.notna(r["ic"]) else None,
                     "win": round(float(r["win"]), 1) if pd.notna(r["win"]) else None,
                     "edge": round(float(r["edge"]), 3) if pd.notna(r["edge"]) else None})
    cols = [{"name": "Hold", "id": "horizon"},
            {"name": "n", "id": "n", "type": "numeric", "format": _INT},
            {"name": "IC", "id": "ic", "type": "numeric", "format": _NUM2},
            {"name": "Win %", "id": "win", "type": "numeric", "format": _NUM2},
            {"name": "Edge %", "id": "edge", "type": "numeric", "format": _NUM2}]
    cond = []
    for c in ("edge", "ic"):
        cond += [
            {"if": {"filter_query": f"{{{c}}} > 0", "column_id": c}, "color": figures.POS},
            {"if": {"filter_query": f"{{{c}}} < 0", "column_id": c}, "color": figures.NEG},
        ]
    cond += [
        {"if": {"filter_query": "{win} >= 50", "column_id": "win"}, "color": figures.POS},
        {"if": {"filter_query": "{win} < 50", "column_id": "win"}, "color": figures.NEG},
    ]
    return html.Div([heading,
                     html.Div(cap, style={"color": "#94a3b8", "fontSize": 12, "marginBottom": 8}),
                     dash_table.DataTable(data=rows, columns=cols,
                                          style_data_conditional=cond, **_TABLE_KW)])


_EXIT_POLICY_TOOLTIP = (
    "Offline EXIT policy evaluation — the COUNTERFACTUAL value of alternative CLOSE rules, "
    "the exit-side twin of the entry sizing comparison. Each held position-day is a "
    "close-vs-hold decision; the reward is what the position DID NEXT: holding captures its "
    "oriented forward return, closing captures 0. A good close rule therefore CLOSES the days "
    "whose forward return is about to go negative (cutting losers) and HOLDS the rest. "
    "'avg_fwd_on_close' is the mean oriented forward return of the days each rule closed — "
    "you WANT it negative (you avoided a drop). 'exit_alpha' = held_mean − allhold_mean: how "
    "much better the book you CARRY does than holding everything; > 0 means the rule earns its "
    "keep, and 'always hold' is the 0 baseline. This validates whether an exit-BREADTH or "
    "aggregator rule beats the current LLM-scalar close BEFORE any of it is wired live. "
    "Replayed over the exit_signals panel (every held tick, deduped to last-per-day) + OHLCV "
    "forward returns — but that panel is NEW, so this fills in slowly; judge nothing until it "
    "spans many days (info_ratio populates only after >1 day).")


def _exit_policy_eval_section():
    """Counterfactual close-rule comparison — does exit-breadth / aggregator beat
    the current LLM-scalar close? Shown at 1-day and 5-day horizons."""
    children = [_h3("Close-rule comparison — counterfactual (offline exit eval)",
                    _EXIT_POLICY_TOOLTIP)]
    any_data = False
    for h in (1, 5):
        df = data.exit_policy_comparison(days=90, horizon=h)
        if df is None or getattr(df, "empty", True):
            continue
        any_data = True
        rows = [{
            "policy": r["policy"], "decisions": r.get("n_decisions"), "days": r.get("n_days"),
            "close_pct": r.get("close_rate"), "fwd_on_close": r.get("avg_fwd_on_close"),
            "fwd_on_hold": r.get("avg_fwd_on_hold"), "alpha": r.get("exit_alpha"),
            "ir": r.get("info_ratio"),
        } for _, r in df.iterrows()]
        cols = [
            {"name": "Close rule", "id": "policy"},
            {"name": "Decisions", "id": "decisions", "type": "numeric", "format": _INT},
            {"name": "Days", "id": "days", "type": "numeric", "format": _INT},
            {"name": "Close %", "id": "close_pct", "type": "numeric", "format": _NUM2},
            {"name": "Fwd on close %", "id": "fwd_on_close", "type": "numeric", "format": _NUM2},
            {"name": "Fwd on hold %", "id": "fwd_on_hold", "type": "numeric", "format": _NUM2},
            {"name": "Exit alpha %", "id": "alpha", "type": "numeric", "format": _NUM2},
            {"name": "Info ratio", "id": "ir", "type": "numeric", "format": _NUM2},
        ]
        children.append(html.Div(f"{h}-day horizon", style={
            "fontWeight": "bold", "marginTop": 12, "marginBottom": 4, "color": "#cbd5e1"}))
        children.append(dash_table.DataTable(
            data=rows, columns=cols,
            style_data_conditional=[
                {"if": {"filter_query": "{alpha} > 0", "column_id": "alpha"},
                 "color": figures.POS, "fontWeight": "bold"},
                {"if": {"filter_query": "{alpha} < 0", "column_id": "alpha"},
                 "color": figures.NEG, "fontWeight": "bold"},
                # A good close: forward return on the days it cut is negative.
                {"if": {"filter_query": "{fwd_on_close} < 0", "column_id": "fwd_on_close"},
                 "color": figures.POS},
                {"if": {"filter_query": '{policy} contains "always hold"'},
                 "backgroundColor": "#1f2937"},
            ],
            **_TABLE_KW))
    if not any_data:
        children.append(html.Div(
            "No decidable exit decisions yet — the exit_signals panel needs forward-return "
            "history (it is newer than the entry panel; accrues as positions are held and the "
            "OHLCV cache warms past each review day).", style={"color": "#6b7280"}))
    return html.Div(children)


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
    # Entry session, shown at the finer premarket/afterhours grain so the column
    # agrees with the session filter (derived from entry_datetime; date-only
    # legacy rows → rth). The stored coarse 'extended' stamp is split here.
    from src.performance.tracker import _trade_session_fine
    df["session"] = [_trade_session_fine(t) for t in trades]
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
        _asset_toggle("returns-asset"),
        dcc.Loading(html.Div(id="returns-body")),
    ])


@app.callback(Output("returns-body", "children"),
              Input("returns-window", "value"), Input("returns-session", "value"),
              Input("returns-direction", "value"), Input("returns-asset", "value"),
              Input("returns-source", "value"))
def _returns_body(window_value, session_value, direction_value, asset_value, source_value):
    if (source_value or "sim") == "broker":
        return _safe(lambda: _broker_returns_section(window_value, session_value, direction_value,
                                                     asset_value))
    return _safe(lambda: _returns_section(window_value, session_value, direction_value, asset_value))


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


def _broker_returns_section(window_value, session_value=None, direction_value=None,
                            asset_value=None):
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
        from src.performance.tracker import _session_matches
        trades = [t for t in trades if _session_matches(t, sess)]
    dirn = _direction_value(direction_value)
    if dirn:
        want = "BUY" if dirn == "long" else "SELL"
        trades = [t for t in trades if (t.get("action") or "").upper() == want]
    atype = _asset_value(asset_value)
    if atype:
        trades = [t for t in trades if (t.get("type") or "STOCK").upper() == atype.upper()]
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
                 tooltip="Share of broker positions with a positive net return on actual fills — CLOSED round-trips at their realized return, still-OPEN positions at their live mark (same open-inclusive convention as the Simulated view, so the two toggle sides are comparable)."),
            _kpi("Avg return", _pct(s.get("avg_return"), signed=True),
                 tooltip="Mean % return on actual fill prices net of actual commissions, EQUAL-weighted across every position (open at its live mark; compare with 'Return (wtd)', which is CLOSED-only and weights by real dollars)."),
            _kpi("Median return", _pct(s.get("median_return"), signed=True),
                 tooltip="Median % return on actual fills, net of actual commissions, across every position (open at its live mark) — the middle one, unaffected by a single outsized win/loss."),
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


def _returns_section(window_value, session_value=None, direction_value=None, asset_value=None):
    perf = data.performance(window_days=_window_days(window_value), session=_session_value(session_value),
                            direction=_direction_value(direction_value),
                            asset_type=_asset_value(asset_value))
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
            _kpi("Median return", _pct(stats.get("median_return"), signed=True),
                 tooltip="Median per-trade % return in the window — the middle trade, unaffected by a single outsized win/loss (open trades at their live mark). Compare with 'Avg return': a median well below the average means a few big winners are lifting the mean."),
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
        _safe(_calibrations_section),
    ])


def _calibrations_section():
    """Live view of every SELF-CALIBRATED parameter: the exact value in force
    on the latest run vs its documented prior and the evidence count behind it
    — so a drifting or mis-learning parameter is as visible as a dark feed."""
    cals = (data.latest_gate_diag() or {}).get("calibrations") or []
    heading = _h3(
        "Calibrations — self-adapting parameters (latest run)",
        "Each row is a parameter the system LEARNS from its own data instead of a hardcoded "
        "constant: real-fill trading cost, per-session spread multipliers, the horizon cost "
        "hurdle, the breadth-sizing ramp, … 'Value' is what the latest run actually traded "
        "with; 'Prior' is the documented fallback it shrinks toward when evidence is thin; "
        "'Evidence n' is how many observations back the current value (0 = prior fully in "
        "force). A value drifting far from its prior on strong evidence is the system "
        "learning; on WEAK evidence it deserves a look. Snapshotted per run into gate_diag.")
    if not cals:
        return html.Div([heading, html.Div(
            "No calibration snapshot yet — appears after the next pipeline run on this code.",
            style={"color": "#6b7280"})])
    rows = [{
        "name": c.get("name"), "value": c.get("value"), "prior": c.get("prior"),
        "n": c.get("n_evidence"), "unit": c.get("unit"), "note": c.get("note"),
    } for c in cals]
    cols = [
        {"name": "Parameter", "id": "name"},
        {"name": "Value", "id": "value", "type": "numeric", "format": _NUM4},
        {"name": "Prior", "id": "prior", "type": "numeric", "format": _NUM4},
        {"name": "Evidence n", "id": "n", "type": "numeric", "format": _INT},
        {"name": "Unit", "id": "unit"},
        {"name": "Basis", "id": "note"},
    ]
    return html.Div([heading, dash_table.DataTable(data=rows, columns=cols, **_TABLE_KW)])


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
