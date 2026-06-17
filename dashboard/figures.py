"""Plotly figures for the dashboard, reusing src.charts.builder where possible."""

from __future__ import annotations

import plotly.graph_objects as go

POS = "#16a34a"
NEG = "#dc2626"
MUTED = "#6b7280"


def _empty(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=MUTED))
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=30, b=20), height=300,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def method_winrate_fig(perf: dict) -> go.Figure:
    """Horizontal bar of solo win rate per signal method (green ≥50%, red <50%)."""
    solo = perf.get("solo_method_perf") or {}
    labels_map = perf.get("method_labels") or {}
    order = perf.get("method_order_by_winrate") or list(solo.keys())

    rows = []
    for m in order:
        overall = (solo.get(m) or {}).get("overall") or {}
        wr = overall.get("win_rate")
        n = overall.get("trades", overall.get("n", 0))
        if wr is None or not n:
            continue
        rows.append((labels_map.get(m, m), float(wr), int(n)))

    if not rows:
        return _empty("No method performance yet — needs closed trades with attribution.")

    rows.reverse()  # strongest at the top of a horizontal bar
    names = [r[0] for r in rows]
    wrs = [r[1] for r in rows]
    ns = [r[2] for r in rows]
    colors = [POS if w >= 50 else NEG for w in wrs]

    fig = go.Figure(go.Bar(
        x=wrs, y=names, orientation="h", marker_color=colors,
        text=[f"{w:.0f}%  (n={n})" for w, n in zip(wrs, ns)], textposition="auto",
        hovertemplate="%{y}: %{x:.1f}% win rate<extra></extra>",
    ))
    fig.add_vline(x=50, line_dash="dot", line_color=MUTED)
    fig.update_layout(
        title="Solo win rate by method", xaxis_title="Win rate (%)",
        margin=dict(l=10, r=10, t=40, b=10), height=max(340, 24 * len(rows)),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def confidence_return_fig(perf: dict) -> go.Figure:
    """Scatter of each trade's return against the confidence that opened it.

    This is the position-sizing calibration check: confidence drives the
    1.0×/1.5×/2.0× size tier, so higher-confidence trades *should* earn more on
    average — an upward-sloping trend confirms it, a flat/negative one says the
    confidence number isn't carrying directional information worth sizing on.
    Closed trades use their realised return; open trades their live mark-to-market
    (the same convention as the rest of the dashboard). Marker colour is win
    (green) / loss (red); shape is closed (filled circle) / open (hollow diamond).
    A least-squares line summarises the relationship.
    """
    def _pts(trades):
        xs, ys, txt = [], [], []
        for t in trades or []:
            c, r = t.get("confidence"), t.get("return_pct")
            if c is None or r is None:
                continue
            try:
                xs.append(float(c) * 100.0)
                ys.append(float(r))
            except (TypeError, ValueError):
                continue
            txt.append(t.get("ticker", ""))
        return xs, ys, txt

    cx, cy, ctxt = _pts(perf.get("closed_trades"))
    ox, oy, otxt = _pts(perf.get("open_trades"))
    if not cx and not ox:
        return _empty("Return-vs-confidence needs trades with a stored confidence.")

    fig = go.Figure()
    if cx:
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="markers", name="Closed",
            marker=dict(size=10, symbol="circle",
                        color=[POS if v >= 0 else NEG for v in cy], line=dict(width=0)),
            text=ctxt,
            hovertemplate="%{text}: %{y:+.2f}% @ %{x:.0f}% conf<extra>closed</extra>",
        ))
    if ox:
        fig.add_trace(go.Scatter(
            x=ox, y=oy, mode="markers", name="Open (live M2M)",
            marker=dict(size=11, symbol="diamond-open",
                        color=[POS if v >= 0 else NEG for v in oy], line=dict(width=2)),
            text=otxt,
            hovertemplate="%{text}: %{y:+.2f}% @ %{x:.0f}% conf<extra>open</extra>",
        ))

    # Least-squares trend over all points (needs ≥2 distinct confidences).
    ax, ay = cx + ox, cy + oy
    if len(ax) >= 2 and len(set(ax)) >= 2:
        import numpy as np
        slope, intercept = np.polyfit(ax, ay, 1)
        x0, x1 = min(ax), max(ax)
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[slope * x0 + intercept, slope * x1 + intercept],
            mode="lines", name=f"Trend ({slope:+.2f}%/pt)",
            line=dict(color=MUTED, dash="dash", width=2),
            hoverinfo="skip",
        ))

    fig.add_hline(y=0, line_dash="dot", line_color=MUTED)
    fig.update_layout(
        xaxis_title="Entry confidence (%)", yaxis_title="Return (%)",
        margin=dict(l=10, r=10, t=30, b=10), height=380,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def confidence_timeline_fig(reviews, trades) -> go.Figure:
    """Per-ticker hold-review trajectory (fix #2): the opener-pinned review
    confidence over time (left axis) + price (right axis) + entry/exit decisions,
    so you can see whether conviction deterioration precedes a direction change.

    Confidence markers are coloured by the review's action (BUY green / SELL red /
    HOLD·WATCH grey); a dashed line marks the entry confidence and a dotted line
    the close floor (the level below which same-direction conviction triggers
    ``llm_confidence_loss``). Entry/exit markers come from the ledger.
    """
    import pandas as pd
    from plotly.subplots import make_subplots

    if reviews is None or getattr(reviews, "empty", True):
        return _empty("No review history yet for this ticker (LLM-opened positions accrue it each tick).")

    df = reviews.copy()
    df["t"] = pd.to_datetime(df["reviewed_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["t"]).sort_values("t")
    if df.empty:
        return _empty("No review history yet for this ticker.")

    act_color = {"BUY": POS, "SELL": NEG, "HOLD": MUTED, "WATCH": MUTED}
    mcolors = [act_color.get(str(a).upper(), MUTED) for a in df["action"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Confidence trajectory (left), markers coloured by the review's action.
    fig.add_trace(go.Scatter(
        x=df["t"], y=df["confidence"], mode="lines+markers", name="Review confidence",
        line=dict(color="#2563eb", width=2),
        marker=dict(size=9, color=mcolors, line=dict(width=1, color="#1e3a8a")),
        customdata=df[["action", "direction", "return_pct"]].values,
        hovertemplate=("%{x|%Y-%m-%d %H:%M} ET<br>conf %{y:.2f} · %{customdata[0]} "
                       "(%{customdata[1]})<br>ret %{customdata[2]:+.2f}%<extra></extra>"),
    ), secondary_y=False)

    # Entry-confidence baseline + close floor (constant per position; latest value),
    # drawn as full-width lines (robust across plotly versions vs add_hline).
    x0, x1 = df["t"].iloc[0], df["t"].iloc[-1]
    ec = df["entry_confidence"].dropna()
    if not ec.empty:
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[float(ec.iloc[-1])] * 2, mode="lines", name="entry conf",
            line=dict(color=POS, dash="dash", width=1.5), hoverinfo="skip"), secondary_y=False)
    fl = df["conf_floor"].dropna()
    if not fl.empty:
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[float(fl.iloc[-1])] * 2, mode="lines", name="close floor",
            line=dict(color=NEG, dash="dot", width=1.5), hoverinfo="skip"), secondary_y=False)

    # Price (right axis).
    if df["price"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["t"], y=df["price"], mode="lines", name="Price",
            line=dict(color="#9ca3af", width=1.5),
            hovertemplate="%{x|%Y-%m-%d %H:%M} ET<br>price %{y:.2f}<extra></extra>",
        ), secondary_y=True)

    # Entry / exit decision markers from the ledger (on the price axis).
    def _parse(iso):
        try:
            ts = pd.to_datetime(iso, utc=True)
            return None if pd.isna(ts) else ts
        except Exception:
            return None

    for tr in (trades or []):
        e_t, e_p = _parse(tr.get("entry_datetime")), tr.get("entry_price")
        if e_t is not None and e_p:
            is_buy = tr.get("action") == "BUY"
            fig.add_trace(go.Scatter(
                x=[e_t], y=[e_p], mode="markers", showlegend=False,
                marker=dict(symbol="triangle-up" if is_buy else "triangle-down",
                            size=14, color=POS if is_buy else NEG,
                            line=dict(width=1, color="#111827")),
                hovertemplate=f"ENTRY {tr.get('action')} @ %{{y:.2f}}<extra></extra>",
            ), secondary_y=True)
        if tr.get("status") == "CLOSED":
            x_t, x_p = _parse(tr.get("exit_datetime")), tr.get("exit_price")
            if x_t is not None and x_p:
                fig.add_trace(go.Scatter(
                    x=[x_t], y=[x_p], mode="markers", showlegend=False,
                    marker=dict(symbol="x", size=12, color="#111827"),
                    hovertemplate=f"EXIT @ %{{y:.2f}}<br>{tr.get('exit_reason') or 'close'}<extra></extra>",
                ), secondary_y=True)

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10), height=420,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Confidence", range=[0, 1], secondary_y=False)
    fig.update_yaxes(title_text="Price ($)", secondary_y=True, showgrid=False)
    return fig


def equity_curve_fig(perf: dict) -> go.Figure:
    """Reuse the existing equity-curve builder; fall back to a placeholder."""
    try:
        from src.charts.builder import build_equity_curve
        fig = build_equity_curve(perf.get("closed_trades") or [])
        if fig is not None:
            return fig
    except Exception:
        pass
    return _empty("Equity curve needs ≥2 closed trades.")


def calibration_bar_fig(rep: dict) -> go.Figure:
    """Bar of average return per confidence bucket (the bucketed companion to the
    return-vs-confidence scatter). Rising green bars left→right + a positive slope
    confirm higher-confidence (bigger-sized) trades actually earn more."""
    buckets = rep.get("buckets") or []
    if not buckets:
        return _empty("Confidence calibration needs trades with a stored confidence.")
    names = [b["label"] for b in buckets]
    avgs = [b["avg_return"] or 0.0 for b in buckets]
    ns = [b["trades"] for b in buckets]
    colors = [POS if a >= 0 else NEG for a in avgs]
    fig = go.Figure(go.Bar(
        x=names, y=avgs, marker_color=colors,
        text=[f"{a:+.2f}% (n={n})" for a, n in zip(avgs, ns)], textposition="auto",
        hovertemplate="%{x}<br>avg %{y:+.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=MUTED)
    slope = rep.get("slope")
    title = "Avg return by confidence bucket"
    if slope is not None:
        title += f"   ·   slope {slope:+.3f}%/pt"
    fig.update_layout(
        title=title, yaxis_title="Avg return (%)",
        margin=dict(l=10, r=10, t=40, b=10), height=340,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


def mfe_capture_fig(rep: dict) -> go.Figure:
    """Per closed trade: peak favorable excursion (MFE, x) vs realized return (y).

    The dashed y=x line is "kept the entire peak". Points far below it gave back
    most of the move (no profit-taking); points near or below zero on the y-axis
    were round-trips that rode a gain back to a loss. Colour = win/loss."""
    rows = rep.get("per_trade") or []
    if not rows:
        return _empty("Exit quality needs closed trades with an MFE/MAE band.")
    xs = [r["mfe"] for r in rows]
    ys = [r["return_pct"] for r in rows]
    txt = [r.get("ticker") or "" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers", name="trades", text=txt,
        marker=dict(size=10, color=[POS if v >= 0 else NEG for v in ys], line=dict(width=0)),
        hovertemplate="%{text}: kept %{y:+.2f}% of a %{x:+.2f}% peak<extra></extra>",
    ))
    lim = max([abs(v) for v in xs + ys] + [1.0])
    fig.add_trace(go.Scatter(
        x=[0, lim], y=[0, lim], mode="lines", name="full capture",
        line=dict(color=MUTED, dash="dash", width=1.5), hoverinfo="skip",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=MUTED)
    fig.update_layout(
        title="Exit capture — peak (MFE) vs kept (return)",
        xaxis_title="MFE — peak favorable (%)", yaxis_title="Realized return (%)",
        margin=dict(l=10, r=10, t=40, b=10), height=380,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def slippage_by_session_fig(slip_df) -> go.Figure:
    """Grouped bars of mean + p90 fill-vs-decision slippage (bp, + = adverse) by
    session — the test of whether the LMT cap (20 bp RTH / 80 bp extended) is
    actually being achieved."""
    if slip_df is None or getattr(slip_df, "empty", True):
        return _empty("No filled legs with recorded slippage yet.")
    sessions = slip_df["session"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sessions, y=slip_df["mean_bps"], name="mean",
                         marker_color="#2563eb",
                         text=[f"n={int(n)}" for n in slip_df["n"]], textposition="auto"))
    fig.add_trace(go.Bar(x=sessions, y=slip_df["p90_bps"], name="p90", marker_color="#93c5fd"))
    fig.add_hline(y=0, line_dash="dot", line_color=MUTED)
    fig.update_layout(
        barmode="group", title="Fill slippage by session (bp, + = adverse)",
        yaxis_title="Slippage (bp)", margin=dict(l=10, r=10, t=40, b=10), height=340,
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def tracking_error_fig(rep: dict) -> go.Figure:
    """Time series of the sim − broker return gap by entry date. A line hugging
    zero = the model tracks reality; a persistent one-sided drift = a cost/price
    bug (the auto-catch for the stale-price class)."""
    by_date = rep.get("by_date") or []
    if not by_date:
        return _empty("Tracking error needs trades with matching broker fills.")
    xs = [r["entry_date"] for r in by_date]
    ys = [r["mean_d_return"] for r in by_date]
    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="lines+markers", line=dict(color="#2563eb", width=2),
        hovertemplate="%{x}<br>sim − broker %{y:+.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=MUTED)
    fig.update_layout(
        title="Sim − broker return gap by entry date",
        yaxis_title="Δreturn (sim − broker, %)",
        margin=dict(l=10, r=10, t=40, b=10), height=340,
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig
