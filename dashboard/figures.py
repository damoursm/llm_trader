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
