"""Send recommendation digest emails via SMTP.

Layout:
  1. Signal overview chart
  2. Aggregated recommendations — summary table of every ticker
  3. Trade details — full per-method breakdown for each BUY / SELL
  4. Monitor list — compact HOLD / WATCH table
  5. Smart money signals
  6. Portfolio performance

Charts are embedded as inline base64 PNG images using kaleido.
If kaleido is not installed the email falls back gracefully to text-only.
"""

import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from jinja2 import Template
from loguru import logger
from typing import Dict, List, Optional

from config import settings
from src.models import NewsArticle, Recommendation, InsiderTrade, TickerSignal
from src.utils import now_et, fmt_et, ET
from src.charts.builder import (
    build_signals_overview,
    build_stock_chart,
    build_equity_curve,
    fig_to_png_b64,
)


ACTION_COLOR = {
    "BUY":   "#16a34a",
    "SELL":  "#dc2626",
    "HOLD":  "#2563eb",
    "WATCH": "#d97706",
}

TYPE_COLOR = {
    "STOCK":     "#334155",
    "ETF":       "#1e3a5f",
    "COMMODITY": "#78350f",   # amber-brown for precious metals
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body  { font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0;
          padding: 24px; max-width: 860px; margin: 0 auto; }
  h1   { color: #f8fafc; font-size: 20px; margin-bottom: 4px; }
  h2   { color: #94a3b8; font-size: 12px; font-weight: bold; text-transform: uppercase;
          letter-spacing: .07em; margin-top: 36px; border-bottom: 1px solid #1e293b;
          padding-bottom: 8px; }
  .sub { color: #64748b; font-size: 13px; margin-bottom: 28px; }

  /* Summary table */
  .tbl     { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px; }
  .tbl th  { background: #1e293b; padding: 7px 10px; text-align: left;
             color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
  .tbl td  { padding: 9px 10px; border-top: 1px solid #1e293b55; vertical-align: middle; }

  /* Recommendation card */
  .card    { background: #1e293b; border-radius: 8px; padding: 18px 20px; margin: 14px 0; }
  .badge   { display: inline-block; padding: 3px 9px; border-radius: 4px;
             color: #fff; font-weight: bold; font-size: 12px; }
  .ticker  { font-size: 21px; font-weight: bold; color: #f8fafc; }
  .meta    { font-size: 12px; color: #64748b; margin-top: 5px; line-height: 1.8; }

  /* Synthesis box */
  .synth   { background: #0f172a55; border-radius: 6px; padding: 11px 14px;
             font-size: 13px; color: #cbd5e1; line-height: 1.65; margin: 10px 0; }

  /* Method rows */
  .mrow    { padding: 10px 12px; background: #0f172a55; border-radius: 6px; margin: 6px 0; }
  .mhdr    { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
  .mlabel  { font-size: 11px; font-weight: bold; color: #94a3b8;
             text-transform: uppercase; letter-spacing: .05em; }
  .mscore  { font-size: 12px; font-weight: bold; padding: 2px 7px; border-radius: 3px; }
  .sp { color: #4ade80; background: #16a34a22; }
  .sn { color: #f87171; background: #dc262622; }
  .sz { color: #94a3b8; background: #33415522; }

  /* Score bar */
  .bar-wrap { height: 4px; background: #334155; border-radius: 2px; margin: 5px 0 8px; overflow: hidden; }
  .bar      { height: 100%; border-radius: 2px; }

  .mtext   { font-size: 12px; color: #94a3b8; line-height: 1.55; }

  /* News articles */
  .art     { font-size: 12px; color: #94a3b8; padding: 5px 0;
             border-bottom: 1px solid #1e293b55; }
  .art:last-child { border-bottom: none; }
  .art-time { color: #475569; font-size: 11px; }
  .art-src  { color: #7c3aed; font-size: 11px; }

  /* Compact monitor table */
  .ctbl    { width: 100%; border-collapse: collapse; font-size: 12px; margin: 10px 0; }
  .ctbl th { background: #0f172a; padding: 6px 10px; text-align: left; color: #64748b; }
  .ctbl td { padding: 5px 10px; border-top: 1px solid #1e293b; }

  /* Smart money */
  .sm-type { font-size: 10px; background: #0f172a; padding: 2px 5px;
             border-radius: 3px; color: #94a3b8; margin-right: 5px; }

  /* Performance */
  .pt    { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }
  .pt th { background: #0f172a; padding: 7px 10px; text-align: left; color: #94a3b8; }
  .pt td { padding: 6px 10px; border-top: 1px solid #334155; }
  .pos { color: #4ade80; }
  .neg { color: #f87171; }

  .chart-img { width: 100%; max-width: 960px; border-radius: 6px;
               margin-top: 14px; display: block; }
  .footer    { text-align: center; color: #475569; font-size: 12px;
               margin-top: 36px; border-top: 1px solid #1e293b; padding-top: 16px; }
</style>
</head>
<body>
<h1>LLM Trader &mdash; Daily Report</h1>
<div class="sub">
  {{ generated_at }}
  &bull; {{ total }} tickers analysed
  &bull; {{ actionable_recs|length }} actionable signal(s)
</div>

<!-- ══════════════════════════════════════
     SIGNAL OVERVIEW CHART
     ══════════════════════════════════════ -->
{% if overview_png %}
<h2>Signal Overview</h2>
<img class="chart-img" src="data:image/png;base64,{{ overview_png }}" alt="Signal Overview">
{% endif %}

<!-- ══════════════════════════════════════
     1 — AGGREGATED RECOMMENDATIONS
     Summary of every ticker, BUY/SELL first
     ══════════════════════════════════════ -->
<h2>Aggregated Recommendations</h2>
<table class="tbl">
  <thead>
    <tr>
      <th>Ticker</th>
      <th>Action</th>
      <th>Direction</th>
      <th>Confidence</th>
      <th>Sources</th>
      <th>Horizon</th>
      <th>At (UTC)</th>
    </tr>
  </thead>
  <tbody>
    {% for rec in all_recs_sorted %}
    {% set sig = signals_by_ticker.get(rec.ticker) %}
    <tr>
      <td>
        <strong style="color:#f8fafc;">{{ rec.ticker }}</strong>
        {% if rec.type != "STOCK" %}
        <span style="font-size:10px;padding:1px 5px;border-radius:3px;margin-left:5px;
                     background:{{ type_colors.get(rec.type,'#334155') }};color:#e2e8f0;">
          {{ rec.type }}
        </span>
        {% endif %}
      </td>
      <td>
        <span class="badge" style="background:{{ colors[rec.action] }};font-size:11px;">
          {{ rec.action }}
        </span>
      </td>
      <td style="color:{{ '#4ade80' if rec.direction=='BULLISH' else ('#f87171' if rec.direction=='BEARISH' else '#94a3b8') }};">
        {{ rec.direction }}
      </td>
      <td><strong>{{ (rec.confidence*100)|int }}%</strong></td>
      <td style="color:#94a3b8;">
        {% if sig %}{{ sig.sources_agreeing }}/{{ active_methods }}{% else %}&mdash;{% endif %}
      </td>
      <td style="color:#94a3b8;">{{ rec.time_horizon }}</td>
      <td style="color:#64748b;font-size:11px;">{{ fmt_et(rec.generated_at, include_date=False) }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>

<!-- ══════════════════════════════════════
     2 — TRADE DETAILS  (BUY / SELL only)
     Per-method breakdown with dates
     ══════════════════════════════════════ -->
{% if actionable_recs %}
<h2>Trade Details</h2>
{% for rec in actionable_recs %}
{% set sig = signals_by_ticker.get(rec.ticker) %}
<div class="card" style="border-left: 4px solid {{ colors[rec.action] }};">

  <!-- Header row -->
  <div style="display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-bottom:8px;">
    <span class="ticker">{{ rec.ticker }}</span>
    <span class="badge" style="background:{{ colors[rec.action] }};">{{ rec.action }}</span>
    <span style="font-weight:bold;font-size:14px;
                 color:{{ '#4ade80' if rec.direction=='BULLISH' else '#f87171' }};">
      {{ rec.direction }}
    </span>
    <span style="color:#94a3b8;font-size:13px;">{{ (rec.confidence*100)|int }}% confidence</span>
    {% if rec.type != "STOCK" %}
    <span style="font-size:11px;padding:2px 7px;border-radius:3px;
                 background:{{ type_colors.get(rec.type,'#334155') }};color:#e2e8f0;">
      {{ rec.type }}
    </span>
    {% endif %}
  </div>

  <!-- Meta line -->
  <div class="meta">
    Horizon: <strong style="color:#cbd5e1;">{{ rec.time_horizon }}</strong>
    &bull; Recommendation generated:
    <strong style="color:#cbd5e1;">{{ fmt_et(rec.generated_at) }}</strong>
    {% if sig %}
    &bull;
    <span style="color:{{ '#4ade80' if sig.sources_agreeing >= 2 else '#d97706' }};">
      {{ sig.sources_agreeing }}/{{ active_methods }} signal sources agree
    </span>
    {% endif %}
  </div>

  <!-- Claude's synthesis -->
  <div style="margin-top:14px;">
    <div style="font-size:11px;font-weight:bold;color:#64748b;
                text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px;">
      Claude&rsquo;s Synthesis
    </div>
    <div class="synth">{{ rec.rationale }}</div>
  </div>

  <!-- Per-method signal breakdown -->
  {% if sig %}
  <div>
    <div style="font-size:11px;font-weight:bold;color:#64748b;
                text-transform:uppercase;letter-spacing:.05em;
                margin-top:18px;margin-bottom:8px;">
      Signal Breakdown
    </div>

    <!-- News / Sentiment -->
    {% if use_news %}
    <div class="mrow">
      <div class="mhdr">
        <span class="mlabel">News &amp; Sentiment</span>
        <span class="mscore {{ 'sp' if sig.sentiment_score > 0.05 else ('sn' if sig.sentiment_score < -0.05 else 'sz') }}">
          {{ "%+.2f"|format(sig.sentiment_score) }}
        </span>
      </div>
      <div class="bar-wrap">
        <div class="bar"
             style="width:{{ (sig.sentiment_score|abs * 100)|int }}%;
                    background:{{ '#16a34a' if sig.sentiment_score >= 0 else '#dc2626' }};"></div>
      </div>
      {% if sig.rationale %}
      <div class="mtext">{{ sig.rationale }}</div>
      {% endif %}
      <!-- Individual articles with publish timestamps -->
      {% set arts = articles_by_ticker.get(rec.ticker, []) %}
      {% if arts %}
      <div style="margin-top:8px;">
        {% for art in arts %}
        <div class="art">
          <span class="art-time">{{ fmt_et(art.published_at) }}</span>
          &nbsp;<span class="art-src">[{{ art.source }}]</span>
          &nbsp;{{ art.title }}
        </div>
        {% endfor %}
      </div>
      {% endif %}
    </div>
    {% endif %}

    <!-- Technical Analysis -->
    {% if use_tech %}
    <div class="mrow">
      <div class="mhdr">
        <span class="mlabel">Technical Analysis</span>
        <span class="mscore {{ 'sp' if sig.technical_score > 0.05 else ('sn' if sig.technical_score < -0.05 else 'sz') }}">
          {{ "%+.2f"|format(sig.technical_score) }}
        </span>
      </div>
      <div class="bar-wrap">
        <div class="bar"
             style="width:{{ (sig.technical_score|abs * 100)|int }}%;
                    background:{{ '#16a34a' if sig.technical_score >= 0 else '#dc2626' }};"></div>
      </div>
    </div>
    {% endif %}

    <!-- Smart Money -->
    {% if use_insider and sig.insider_summary %}
    <div class="mrow">
      <div class="mhdr">
        <span class="mlabel">Smart Money</span>
      </div>
      <div class="mtext" style="white-space:pre-line;">{{ sig.insider_summary }}</div>
    </div>
    {% endif %}

  </div>
  {% endif %}

  <!-- Per-ticker price chart -->
  {% if charts and rec.ticker in charts %}
  <img class="chart-img"
       src="data:image/png;base64,{{ charts[rec.ticker] }}"
       alt="{{ rec.ticker }} price chart">
  {% endif %}

</div>
{% endfor %}
{% endif %}

<!-- ══════════════════════════════════════
     3 — MONITOR LIST  (HOLD / WATCH)
     ══════════════════════════════════════ -->
{% if passive_recs %}
<h2>Monitor List</h2>
<table class="ctbl">
  <thead>
    <tr>
      <th>Ticker</th>
      <th>Action</th>
      <th>Direction</th>
      <th>Confidence</th>
      {% if use_news %}<th>News</th>{% endif %}
      {% if use_tech %}<th>Technical</th>{% endif %}
      <th>Sources</th>
      <th>Horizon</th>
    </tr>
  </thead>
  <tbody>
    {% for rec in passive_recs %}
    {% set sig = signals_by_ticker.get(rec.ticker) %}
    <tr>
      <td><strong>{{ rec.ticker }}</strong></td>
      <td>
        <span class="badge"
              style="background:{{ colors[rec.action] }};font-size:11px;padding:2px 7px;">
          {{ rec.action }}
        </span>
      </td>
      <td style="color:{{ '#4ade80' if rec.direction=='BULLISH' else ('#f87171' if rec.direction=='BEARISH' else '#94a3b8') }};">
        {{ rec.direction }}
      </td>
      <td>{{ (rec.confidence*100)|int }}%</td>
      {% if use_news %}
      <td style="color:{{ '#4ade80' if sig and sig.sentiment_score > 0 else '#f87171' }};">
        {% if sig %}{{ "%+.2f"|format(sig.sentiment_score) }}{% else %}&mdash;{% endif %}
      </td>
      {% endif %}
      {% if use_tech %}
      <td style="color:{{ '#4ade80' if sig and sig.technical_score > 0 else '#f87171' }};">
        {% if sig %}{{ "%+.2f"|format(sig.technical_score) }}{% else %}&mdash;{% endif %}
      </td>
      {% endif %}
      <td style="color:#94a3b8;">
        {% if sig %}{{ sig.sources_agreeing }}/{{ active_methods }}{% else %}&mdash;{% endif %}
      </td>
      <td style="color:#64748b;">{{ rec.time_horizon }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
{% endif %}

<!-- ══════════════════════════════════════
     4 — SMART MONEY SIGNALS
     ══════════════════════════════════════ -->
{% if insider_trades is not none %}
<h2>Smart Money Signals</h2>
{% if not insider_trades %}
<div class="card" style="border-left: 4px solid #475569;">
  <span style="color:#64748b;font-size:13px;">No unusual smart money activity detected.</span>
</div>
{% endif %}
{% for ticker, trades in (insider_trades or {}).items() %}
<div class="card" style="border-left: 4px solid #7c3aed;">
  <span class="ticker">{{ ticker }}</span>
  {% if ticker in rec_actions %}
  <span class="badge"
        style="background:{{ colors[rec_actions[ticker]] }};margin-left:10px;">
    {{ rec_actions[ticker] }}
  </span>
  {% endif %}
  <br><br>
  {% for t in trades %}
  <div style="font-size:13px;color:#cbd5e1;margin-bottom:8px;">
    <span class="sm-type">{{ t.trader_type.upper().replace('_', ' ') }}</span>
    <strong>{{ t.trader_name }}</strong>
    <span style="color:#94a3b8;font-size:12px;margin-left:4px;">({{ t.role }})</span>
    &mdash;
    <span style="color:{{ '#4ade80' if t.is_bullish else '#f87171' }};">
      {{ t.action_label }}
    </span>
    &nbsp;{{ t.amount_range }}
    &bull; <span style="color:#64748b;font-size:12px;">{{ t.transaction_date }}</span>
    {% if t.notes %}
    <br><span style="color:#64748b;font-size:11px;padding-left:4px;">{{ t.notes }}</span>
    {% endif %}
  </div>
  {% endfor %}
</div>
{% endfor %}
{% endif %}

<!-- ══════════════════════════════════════
     5 — PORTFOLIO PERFORMANCE
     ══════════════════════════════════════ -->
{% if perf %}
<h2>Portfolio Performance</h2>
{% if equity_png %}
<img class="chart-img" src="data:image/png;base64,{{ equity_png }}" alt="Equity Curve">
{% endif %}
{% if perf.stats %}
<div class="card" style="border-left: 4px solid #2563eb;">
  <strong>Closed trades ({{ perf.stats.total_closed }})</strong><br><br>
  Win rate: <strong>{{ perf.stats.win_rate }}%</strong>
  &bull; Avg return: <strong>{{ "%+.2f"|format(perf.stats.avg_return) }}%</strong>
  &bull; Best: <strong class="pos">{{ "%+.2f"|format(perf.stats.best) }}%</strong>
  &bull; Worst: <strong class="neg">{{ "%+.2f"|format(perf.stats.worst) }}%</strong>
</div>
{% endif %}
{% if perf.open_trades %}
<table class="pt">
  <thead>
    <tr>
      <th>Ticker</th><th>Action</th><th>Entry</th>
      <th>Current</th><th>P&amp;L</th><th>Days</th>
    </tr>
  </thead>
  <tbody>
    {% for t in perf.open_trades %}
    <tr>
      <td><strong>{{ t.ticker }}</strong></td>
      <td>{{ t.action }}</td>
      <td>${{ "%.2f"|format(t.entry_price) }}</td>
      <td>${{ "%.2f"|format(t.current_price) }}</td>
      <td class="{{ 'pos' if t.return_pct > 0 else 'neg' }}">
        {{ "%+.2f"|format(t.return_pct) }}%
      </td>
      <td>{{ t.days_held }}d</td>
    </tr>
    {% endfor %}
  </tbody>
</table>
{% endif %}
{% endif %}

<div class="footer">
  Generated by LLM Trader &bull; {{ generated_at }}<br>
  <em>Not financial advice. Always do your own research.</em>
</div>
</body>
</html>
"""


def _build_chart_pngs(
    actionable: List[Recommendation],
    all_recommendations: List[Recommendation],
    performance: Optional[dict],
) -> tuple[dict, Optional[str], Optional[str]]:
    charts: dict[str, str] = {}
    for rec in actionable:
        fig = build_stock_chart(rec.ticker, rec)
        b64 = fig_to_png_b64(fig, width=1100, height=780)
        if b64:
            charts[rec.ticker] = b64
            logger.debug(f"[email] Chart rendered for {rec.ticker}")
    overview_fig = build_signals_overview(all_recommendations or actionable)
    overview_png = fig_to_png_b64(overview_fig, width=1100, height=None)
    closed = (performance or {}).get("closed_trades", [])
    equity_fig = build_equity_curve(closed)
    equity_png = fig_to_png_b64(equity_fig, width=1100, height=480)
    return charts, overview_png, equity_png


def send_recommendations(
    recommendations: List[Recommendation],
    total_analysed: int = 0,
    performance: Optional[dict] = None,
    all_recommendations: Optional[List[Recommendation]] = None,
    insider_trades: Optional[List[InsiderTrade]] = None,
    signals: Optional[List[TickerSignal]] = None,
    articles: Optional[List[NewsArticle]] = None,
) -> bool:
    """Render and send the recommendation email with embedded chart images."""
    all_recs_check = all_recommendations or recommendations
    if not all_recs_check:
        logger.warning("No recommendations to send.")
        return False

    now_str = fmt_et(now_et())

    # ── Signal lookup ──────────────────────────────────────────────────────
    signals_by_ticker: Dict[str, TickerSignal] = {}
    if signals:
        for s in signals:
            signals_by_ticker[s.ticker] = s

    # ── Active methods (for "N/M sources agree" display) ──────────────────
    use_news    = settings.enable_news_sentiment
    use_tech    = settings.enable_technical_analysis and settings.enable_market_data
    use_insider = (
        settings.enable_insider_trades or
        settings.enable_options_flow or
        settings.enable_sec_filings
    )
    active_methods = sum([use_news, use_tech, use_insider])

    # ── All recs sorted: BUY/SELL first, then by confidence desc ──────────
    all_recs = all_recommendations or recommendations
    # Sort order: actionable first, then by type group (COMMODITY before ETF), then confidence
    _TYPE_ORDER = {"COMMODITY": 0, "STOCK": 1, "ETF": 2}
    all_recs_sorted = sorted(
        all_recs,
        key=lambda r: (r.action not in ("BUY", "SELL"), _TYPE_ORDER.get(r.type, 3), -r.confidence),
    )
    actionable_recs = [r for r in all_recs_sorted if r.action in ("BUY", "SELL")]
    passive_recs    = [r for r in all_recs_sorted if r.action not in ("BUY", "SELL")]

    # ── Top 3 news articles per actionable ticker (with publish timestamps) ─
    articles_by_ticker: Dict[str, List[NewsArticle]] = {}
    if articles:
        try:
            from src.analysis.sentiment import filter_relevant_articles
            for rec in actionable_recs:
                relevant = filter_relevant_articles(rec.ticker, articles)
                if relevant:
                    articles_by_ticker[rec.ticker] = sorted(
                        relevant, key=lambda a: a.published_at, reverse=True
                    )[:3]
        except Exception as e:
            logger.debug(f"[email] Article lookup skipped: {e}")

    # ── Smart money grouped by ticker ─────────────────────────────────────
    any_smart_money_enabled = use_insider
    if insider_trades is None and not any_smart_money_enabled:
        insider_by_ticker = None
    else:
        insider_by_ticker: dict = {}
        if insider_trades:
            actionable_tickers = {r.ticker for r in recommendations}
            for trade in insider_trades:
                insider_by_ticker.setdefault(trade.ticker, []).append(trade)
            for ticker in insider_by_ticker:
                insider_by_ticker[ticker] = sorted(
                    insider_by_ticker[ticker],
                    key=lambda t: (not t.is_bullish, t.transaction_date),
                )[:5]
            insider_by_ticker = dict(
                sorted(
                    insider_by_ticker.items(),
                    key=lambda kv: (kv[0] not in actionable_tickers, kv[0]),
                )
            )

    # ── Charts ─────────────────────────────────────────────────────────────
    if settings.enable_charts:
        logger.info("[email] Rendering chart images...")
        charts, overview_png, equity_png = _build_chart_pngs(
            recommendations, all_recs, performance
        )
        logger.info(
            f"[email] {len(charts)} chart image(s) embedded"
            if charts else "[email] No chart images (kaleido not available)"
        )
    else:
        charts, overview_png, equity_png = {}, None, None

    rec_actions = {r.ticker: r.action for r in recommendations}

    html_body = Template(HTML_TEMPLATE).render(
        fmt_et=fmt_et,
        type_colors=TYPE_COLOR,
        generated_at=now_str,
        total=total_analysed or len(all_recs),
        colors=ACTION_COLOR,
        # recommendation lists
        all_recs_sorted=all_recs_sorted,
        actionable_recs=actionable_recs,
        passive_recs=passive_recs,
        # signal breakdown
        signals_by_ticker=signals_by_ticker,
        articles_by_ticker=articles_by_ticker,
        use_news=use_news,
        use_tech=use_tech,
        use_insider=use_insider,
        active_methods=active_methods,
        # smart money
        insider_trades=insider_by_ticker,
        rec_actions=rec_actions,
        # charts
        charts=charts,
        overview_png=overview_png,
        equity_png=equity_png,
        # performance
        perf=performance,
    )

    # ── Plain-text fallback ────────────────────────────────────────────────
    lines = [f"LLM Trader Daily Report — {now_str}", ""]
    lines.append("AGGREGATED RECOMMENDATIONS")
    for rec in all_recs_sorted:
        lines.append(
            f"  {rec.ticker:<6} {rec.action:<5} {rec.direction:<8} "
            f"conf={rec.confidence:.0%}  [{rec.time_horizon}]"
            f"  generated={fmt_et(rec.generated_at, include_date=False)}"
        )
    lines.append("")
    for rec in actionable_recs:
        lines.append(f"{'='*60}")
        lines.append(f"{rec.ticker} — {rec.action} | {rec.direction} | {rec.confidence:.0%} | {rec.time_horizon}")
        lines.append(f"Generated: {fmt_et(rec.generated_at)}")
        sig = signals_by_ticker.get(rec.ticker)
        if sig:
            lines.append(f"Sources agreeing: {sig.sources_agreeing}/{active_methods}")
            if use_news:
                lines.append(f"News sentiment:  {sig.sentiment_score:+.2f}")
                lines.append(f"  {sig.rationale}")
            if use_tech:
                lines.append(f"Technical score: {sig.technical_score:+.2f}")
            if use_insider and sig.insider_summary:
                lines.append(f"Smart money:\n{sig.insider_summary}")
        lines.append(f"Claude's synthesis: {rec.rationale}")
        lines.append("")
    if insider_by_ticker:
        lines.append("SMART MONEY SIGNALS")
        for ticker, trades in insider_by_ticker.items():
            for t in trades:
                lines.append(
                    f"  {ticker}: {t.action_label} — {t.trader_name} ({t.role})"
                    f" | {t.amount_range} | {t.transaction_date}"
                )
        lines.append("")
    text_body = "\n".join(lines)

    buys  = [r.ticker for r in actionable_recs if r.action == "BUY"]
    sells = [r.ticker for r in actionable_recs if r.action == "SELL"]
    if buys or sells:
        subject = (
            f"LLM Trader | {now_str}"
            + (f" | BUY: {', '.join(buys)}"   if buys  else "")
            + (f" | SELL: {', '.join(sells)}" if sells else "")
        )
    else:
        subject = f"LLM Trader | {now_str} | No actionable signals — daily report"

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = settings.smtp_user
        msg["To"]      = ", ".join(settings.recipients_list)
        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(settings.smtp_user, settings.recipients_list, msg.as_string())

        logger.info(f"Email sent to {settings.recipients_list}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False
