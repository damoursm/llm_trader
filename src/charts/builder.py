"""
Chart builder — generates Plotly figures for individual stocks and portfolio overview.

Charts produced per BUY/SELL ticker:
  - Candlestick (60 d OHLCV) with SMA 20, SMA 50, EMA 9, Bollinger Bands, Volume
  - RSI 14 subplot (overbought/oversold lines at 70/30)
  - MACD + signal + histogram subplot

Summary charts (all tickers):
  - Signals confidence heatmap (horizontal bar, coloured by action)
  - Equity curve (cumulative P&L of closed trades)
"""

import base64
import io
import concurrent.futures
from datetime import date, timedelta
from typing import List, Optional

PNG_TIMEOUT = 60   # seconds per chart before giving up

import pandas as pd
import yfinance as yf
from loguru import logger
from config import settings
from src.data.cache import load_ohlcv, save_ohlcv

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly not installed — charts disabled. Run: pip install plotly kaleido")

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

from src.models import Recommendation

# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------

ACTION_COLORS = {
    "BUY":   "#16a34a",
    "SELL":  "#dc2626",
    "HOLD":  "#2563eb",
    "WATCH": "#d97706",
}

CHART_THEME = "plotly_dark"
BG_COLOR    = "#0f172a"
GRID_COLOR  = "#1e293b"
TEXT_COLOR  = "#e2e8f0"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fetch_ohlcv(ticker: str, initial_period: str = "3mo") -> Optional[pd.DataFrame]:
    """
    Return OHLCV history for a ticker.

    ENABLE_FETCH_DATA=false
        Load from cache only; skip yfinance entirely.

    ENABLE_FETCH_DATA=true
        1. Load existing cache.
        2. Determine the first missing date (day after last cached row).
        3. Fetch only that missing range from yfinance (or the full initial_period
           if there is no cache yet).
        4. Append new rows, deduplicate, sort, and save back to cache.
    """
    cached = load_ohlcv(ticker)

    if not settings.enable_fetch_data:
        if cached is not None and not cached.empty:
            return cached
        logger.debug(f"[charts] ENABLE_FETCH_DATA=false and no OHLCV cache for {ticker} — skipping chart")
        return None

    # Determine the date range to fetch
    today = date.today()
    if cached is not None and not cached.empty:
        last_cached = cached.index[-1]
        if hasattr(last_cached, "date"):
            last_cached = last_cached.date()
        if last_cached >= today:
            # Cache is already up to date
            return cached
        fetch_start = last_cached + timedelta(days=1)
        fetch_end   = today + timedelta(days=1)   # yfinance end is exclusive
    else:
        fetch_start = None   # will use initial_period instead
        fetch_end   = None

    try:
        if fetch_start is not None:
            new_df = yf.download(
                ticker,
                start=fetch_start.isoformat(),
                end=fetch_end.isoformat(),
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
        else:
            new_df = yf.download(
                ticker,
                period=initial_period,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )

        if not new_df.empty:
            new_df.columns = [c[0] if isinstance(c, tuple) else c for c in new_df.columns]
            new_df = new_df[["Open", "High", "Low", "Close", "Volume"]].dropna()

            if cached is not None and not cached.empty and fetch_start is not None:
                combined = pd.concat([cached, new_df])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                logger.debug(f"[charts] {ticker}: appended {len(new_df)} new rows to OHLCV cache")
            else:
                combined = new_df
                logger.debug(f"[charts] {ticker}: initialised OHLCV cache ({len(combined)} rows)")

            save_ohlcv(ticker, combined)
            return combined

        # No new data (e.g. weekend/holiday) — existing cache is fine
        if cached is not None and not cached.empty:
            return cached

    except Exception as e:
        logger.warning(f"[charts] OHLCV fetch failed for {ticker}: {e}")

    return cached  # best effort


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical indicators and append them as columns."""
    close = df["Close"]

    # Moving averages
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["EMA9"]  = close.ewm(span=9, adjust=False).mean()

    if TA_AVAILABLE:
        # Bollinger Bands (20, 2σ)
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        df["BB_upper"] = bb.bollinger_hband()
        df["BB_mid"]   = bb.bollinger_mavg()
        df["BB_lower"] = bb.bollinger_lband()

        # RSI 14
        df["RSI"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

        # MACD (12, 26, 9)
        macd_ind = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df["MACD"]        = macd_ind.macd()
        df["MACD_signal"] = macd_ind.macd_signal()
        df["MACD_hist"]   = macd_ind.macd_diff()
    else:
        # Minimal fallback without ta library
        df["RSI"] = None
        df["MACD"] = df["MACD_signal"] = df["MACD_hist"] = None
        df["BB_upper"] = df["BB_mid"] = df["BB_lower"] = None

    return df


# ---------------------------------------------------------------------------
# Per-ticker stock chart
# ---------------------------------------------------------------------------

def build_stock_chart(ticker: str, rec: Recommendation) -> Optional["go.Figure"]:
    """
    Build a multi-panel chart for a single ticker:
      Panel 1 (55%): Candlestick + SMA20/50 + EMA9 + Bollinger Bands
      Panel 2 (10%): Volume bars
      Panel 3 (17%): RSI 14 with 30 / 70 reference lines
      Panel 4 (18%): MACD histogram + signal line
    """
    if not PLOTLY_AVAILABLE:
        return None

    df = _fetch_ohlcv(ticker)
    if df is None or len(df) < 30:
        logger.warning(f"[charts] Insufficient data for {ticker}")
        return None

    df = _add_indicators(df)

    action_color = ACTION_COLORS.get(rec.action, "#94a3b8")
    action_label = f"{rec.action} | {rec.direction} | conf {rec.confidence:.0%} | {rec.time_horizon}"

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.10, 0.17, 0.18],
        vertical_spacing=0.03,
        subplot_titles=("", "Volume", "RSI 14", "MACD"),
    )

    # --- Panel 1: Candlestick + overlays ---
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        name="Price", showlegend=False,
    ), row=1, col=1)

    for col, color, dash, name in [
        ("SMA20", "#60a5fa", "solid",  "SMA 20"),
        ("SMA50", "#f59e0b", "solid",  "SMA 50"),
        ("EMA9",  "#a78bfa", "dot",    "EMA 9"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], name=name,
                line=dict(color=color, width=1.2, dash=dash),
                showlegend=True,
            ), row=1, col=1)

    if "BB_upper" in df.columns and df["BB_upper"].notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"], name="BB Upper",
            line=dict(color="#475569", width=1, dash="dash"), showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"], name="BB Lower",
            line=dict(color="#475569", width=1, dash="dash"),
            fill="tonexty", fillcolor="rgba(71,85,105,0.08)", showlegend=False,
        ), row=1, col=1)

    # --- Panel 2: Volume ---
    colors_vol = [
        "#22c55e" if c >= o else "#ef4444"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume",
        marker_color=colors_vol, showlegend=False,
    ), row=2, col=1)

    # --- Panel 3: RSI ---
    if "RSI" in df.columns and df["RSI"].notna().any():
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"], name="RSI 14",
            line=dict(color="#38bdf8", width=1.5), showlegend=False,
        ), row=3, col=1)
        for level, color in [(70, "rgba(239,68,68,0.35)"), (30, "rgba(34,197,94,0.35)")]:
            fig.add_hline(y=level, line_dash="dash", line_color=color, row=3, col=1)
        fig.update_yaxes(range=[0, 100], row=3, col=1)

    # --- Panel 4: MACD ---
    if "MACD" in df.columns and df["MACD"].notna().any():
        hist_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_hist"], name="MACD Hist",
            marker_color=hist_colors, showlegend=False,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color="#818cf8", width=1.5), showlegend=False,
        ), row=4, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_signal"], name="Signal",
            line=dict(color="#f472b6", width=1.5), showlegend=False,
        ), row=4, col=1)

    # --- Annotation: recommendation badge ---
    fig.add_annotation(
        text=action_label,
        xref="paper", yref="paper",
        x=0.01, y=0.99, xanchor="left", yanchor="top",
        showarrow=False,
        font=dict(size=13, color=action_color, family="monospace"),
        bgcolor=BG_COLOR, bordercolor=action_color, borderwidth=1,
        borderpad=6,
    )

    fig.update_layout(
        title=dict(text=f"<b>{ticker}</b>  —  {rec.rationale[:120]}{'…' if len(rec.rationale) > 120 else ''}",
                   font=dict(size=13, color=TEXT_COLOR)),
        template=CHART_THEME,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        legend=dict(orientation="h", y=1.04, x=0, font=dict(size=11)),
        height=780,
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=20, t=80, b=30),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)

    return fig


# ---------------------------------------------------------------------------
# Signals overview chart
# ---------------------------------------------------------------------------

def build_signals_overview(
    recommendations: List[Recommendation],
    signals_by_ticker: Optional[dict] = None,
) -> Optional["go.Figure"]:
    """
    Horizontal stacked bar chart of all recommendations sorted by |confidence|.
    When signals_by_ticker (Dict[str, TickerSignal]) is provided, each BUY/SELL bar
    is split into per-method contribution segments:
      BUY  → greens : news=#22c55e  tech=#4ade80  smart_money=#0d9488
      SELL → warm   : news=#ef4444  tech=#f97316  smart_money=#eab308
    HOLD/WATCH use single-color bars. SELL bars point left (negative x).
    """
    if not PLOTLY_AVAILABLE or not recommendations:
        return None

    sorted_recs = sorted(recommendations, key=lambda r: r.confidence)
    tickers = [r.ticker for r in sorted_recs]
    n = len(tickers)
    ticker_idx = {t: i for i, t in enumerate(tickers)}

    has_signals = bool(signals_by_ticker)

    if not has_signals:
        # ── Fallback: original single-colour bars ──────────────────────────
        values = [r.confidence if r.action in ("BUY", "HOLD", "WATCH") else -r.confidence
                  for r in sorted_recs]
        colors = [ACTION_COLORS.get(r.action, "#94a3b8") for r in sorted_recs]
        labels = [f"{r.action} {r.confidence:.0%}" for r in sorted_recs]
        fig = go.Figure(go.Bar(
            x=values, y=tickers,
            orientation="h",
            marker_color=colors,
            text=labels,
            textposition="outside",
            textfont=dict(size=11, color=TEXT_COLOR),
        ))
        fig.add_vline(x=0.75, line_dash="dash", line_color="#64748b",
                      annotation_text="75% threshold", annotation_position="top right")
        fig.add_vline(x=-0.75, line_dash="dash", line_color="#64748b")
        fig.update_layout(
            title=dict(text="<b>Signal Confidence Overview</b>  (← SELL  /  BUY →)",
                       font=dict(size=14, color=TEXT_COLOR)),
            template=CHART_THEME, paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
            xaxis=dict(range=[-1.1, 1.1], showgrid=True, gridcolor=GRID_COLOR,
                       tickformat=".0%", title="Confidence"),
            yaxis=dict(showgrid=False),
            height=max(300, n * 32 + 120),
            margin=dict(l=80, r=120, t=60, b=40),
            showlegend=False,
        )
        return fig

    # ── Stacked method-breakdown bars ──────────────────────────────────────
    # Colors: BUY (greens/teal/cyan), SELL (red/orange/amber/purple)
    BUY_COLORS  = ["#22c55e", "#4ade80", "#0d9488", "#06b6d4"]  # news, tech, insider, put/call
    SELL_COLORS = ["#ef4444", "#f97316", "#eab308", "#a855f7"]  # news, tech, insider, put/call
    METHOD_NAMES = ["News / Sentiment", "Technical", "Smart Money", "Put/Call"]

    use_news     = settings.enable_news_sentiment
    use_tech     = settings.enable_technical_analysis and settings.enable_fetch_data
    use_insider  = (settings.enable_insider_trades or
                    settings.enable_options_flow or
                    settings.enable_sec_filings)
    use_put_call = settings.enable_put_call
    method_enabled = [use_news, use_tech, use_insider, use_put_call]

    def _get_scores(sig) -> list:
        pc = getattr(sig, "put_call_score", 0.0)
        return [
            abs(sig.sentiment_score) if use_news     else 0.0,
            abs(sig.technical_score) if use_tech     else 0.0,
            abs(sig.insider_score)   if use_insider  else 0.0,
            abs(pc)                  if use_put_call else 0.0,
        ]

    traces = []

    for m_idx in range(4):
        if not method_enabled[m_idx]:
            continue

        buy_x  = [0.0] * n
        sell_x = [0.0] * n

        for rec in sorted_recs:
            if rec.action not in ("BUY", "SELL"):
                continue
            sig = signals_by_ticker.get(rec.ticker)
            if sig is None:
                # No signal data — give equal weight across enabled methods
                active_count = sum(method_enabled)
                frac = rec.confidence / active_count if active_count else 0.0
            else:
                scores = _get_scores(sig)
                total  = sum(scores)
                if total > 0:
                    frac = scores[m_idx] / total * rec.confidence
                else:
                    # All method scores are zero — fall back to equal weighting
                    active_count = sum(method_enabled)
                    frac = rec.confidence / active_count if active_count else 0.0

            i = ticker_idx[rec.ticker]
            if rec.action == "BUY":
                buy_x[i] = frac
            else:
                sell_x[i] = -frac

        label = METHOD_NAMES[m_idx]
        # BUY segment
        if any(v != 0 for v in buy_x):
            traces.append(go.Bar(
                x=buy_x, y=tickers, orientation="h",
                name=f"{label} — BUY",
                marker_color=BUY_COLORS[m_idx],
                legendgroup=f"buy_{m_idx}",
                hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.0%}}<extra>BUY</extra>",
                showlegend=True,
            ))
        # SELL segment
        if any(v != 0 for v in sell_x):
            traces.append(go.Bar(
                x=sell_x, y=tickers, orientation="h",
                name=f"{label} — SELL",
                marker_color=SELL_COLORS[m_idx],
                legendgroup=f"sell_{m_idx}",
                hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.0%}}<extra>SELL</extra>",
                showlegend=True,
            ))

    # HOLD / WATCH — single-colour bars (always point right)
    for action, color in [("HOLD", ACTION_COLORS["HOLD"]), ("WATCH", ACTION_COLORS["WATCH"])]:
        x_vals = [r.confidence if r.action == action else 0.0 for r in sorted_recs]
        if any(v != 0 for v in x_vals):
            traces.append(go.Bar(
                x=x_vals, y=tickers, orientation="h",
                name=action,
                marker_color=color,
                hovertemplate="<b>%{y}</b><br>conf: %{x:.0%}<extra>" + action + "</extra>",
                showlegend=True,
            ))

    fig = go.Figure(traces)
    fig.update_layout(barmode="relative")

    # Annotate total confidence outside each BUY/SELL bar
    for rec in sorted_recs:
        if rec.action not in ("BUY", "SELL"):
            continue
        x_pos   = rec.confidence + 0.02 if rec.action == "BUY" else -rec.confidence - 0.02
        anchor  = "left"                 if rec.action == "BUY" else "right"
        color   = ACTION_COLORS["BUY"]   if rec.action == "BUY" else ACTION_COLORS["SELL"]
        fig.add_annotation(
            x=x_pos, y=rec.ticker,
            text=f"{rec.action} {rec.confidence:.0%}",
            xanchor=anchor, yanchor="middle",
            showarrow=False,
            font=dict(size=10, color=color),
        )

    fig.add_vline(x=0.75,  line_dash="dash", line_color="#64748b",
                  annotation_text="75% threshold", annotation_position="top right")
    fig.add_vline(x=-0.75, line_dash="dash", line_color="#64748b")

    fig.update_layout(
        title=dict(text="<b>Signal Confidence Overview</b>  (← SELL  /  BUY →)",
                   font=dict(size=14, color=TEXT_COLOR)),
        template=CHART_THEME,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        xaxis=dict(range=[-1.15, 1.15], showgrid=True, gridcolor=GRID_COLOR,
                   tickformat=".0%", title="Confidence"),
        yaxis=dict(showgrid=False),
        height=max(300, n * 32 + 160),
        margin=dict(l=80, r=140, t=80, b=40),
        legend=dict(orientation="h", y=1.06, x=0, font=dict(size=11)),
        showlegend=True,
    )
    return fig


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def build_equity_curve(closed_trades: list) -> Optional["go.Figure"]:
    """Cumulative P&L line chart from all closed paper trades."""
    if not PLOTLY_AVAILABLE or not closed_trades:
        return None

    sorted_trades = sorted(closed_trades, key=lambda t: t.get("exit_date") or "")
    dates   = [t["exit_date"] for t in sorted_trades]
    returns = [t["return_pct"] for t in sorted_trades]
    cum_pnl = []
    total   = 0.0
    for r in returns:
        total += r
        cum_pnl.append(round(total, 3))

    bar_colors = ["#22c55e" if r >= 0 else "#ef4444" for r in returns]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=("Cumulative P&L (%)", "Per-trade return (%)"),
    )

    fig.add_trace(go.Scatter(
        x=dates, y=cum_pnl,
        mode="lines+markers",
        line=dict(color="#60a5fa", width=2),
        fill="tozeroy",
        fillcolor="rgba(96,165,250,0.12)",
        name="Cumulative P&L",
        showlegend=False,
    ), row=1, col=1)

    fig.add_hline(y=0, line_dash="solid", line_color="#475569", row=1, col=1)

    fig.add_trace(go.Bar(
        x=dates, y=returns,
        marker_color=bar_colors,
        name="Per-trade return",
        showlegend=False,
        text=[f"{r:+.1f}%" for r in returns],
        textposition="outside",
        textfont=dict(size=10),
    ), row=2, col=1)

    win_rate = sum(1 for r in returns if r > 0) / len(returns) * 100 if returns else 0
    avg_ret  = sum(returns) / len(returns) if returns else 0

    fig.update_layout(
        title=dict(
            text=f"<b>Equity Curve</b>  —  {len(closed_trades)} closed trades | "
                 f"Win rate {win_rate:.0f}% | Avg {avg_ret:+.2f}%",
            font=dict(size=14, color=TEXT_COLOR),
        ),
        template=CHART_THEME,
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        height=480,
        margin=dict(l=60, r=30, t=70, b=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)

    return fig


# ---------------------------------------------------------------------------
# PNG export (for email embedding)
# ---------------------------------------------------------------------------

def fig_to_png_b64(fig: "go.Figure", width: int = 1100, height: int = None) -> Optional[str]:
    """
    Render a Plotly figure to a base64-encoded PNG string.
    Requires the `kaleido` package. Times out after PNG_TIMEOUT seconds
    so a stalled kaleido process never blocks the pipeline.
    """
    if fig is None:
        return None

    def _render():
        kwargs = dict(format="png", width=width, engine="kaleido")
        if height:
            kwargs["height"] = height
        return pio.to_image(fig, **kwargs)

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(_render)
    try:
        img_bytes = future.result(timeout=PNG_TIMEOUT)
        pool.shutdown(wait=False)
        return base64.b64encode(img_bytes).decode("utf-8")
    except concurrent.futures.TimeoutError:
        logger.warning(f"[charts] PNG export timed out after {PNG_TIMEOUT}s — skipping image")
        pool.shutdown(wait=False, cancel_futures=True)
        return None
    except Exception as e:
        logger.warning(f"[charts] PNG export failed: {e}")
        pool.shutdown(wait=False)
        return None