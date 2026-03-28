"""Send recommendation digest emails via SMTP.

Charts are embedded as inline base64 PNG images using kaleido.
If kaleido is not installed the email falls back gracefully to text-only cards.
"""

import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from jinja2 import Template
from loguru import logger
from typing import List, Optional

from config import settings
from src.models import Recommendation
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

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <style>
    body  { font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; }
    h1    { color: #f8fafc; font-size: 20px; margin-bottom: 4px; }
    h2    { color: #94a3b8; font-size: 15px; margin-top: 28px;
            border-bottom: 1px solid #1e293b; padding-bottom: 6px; }
    .subtitle { color: #64748b; font-size: 13px; margin-bottom: 28px; }
    .card { background: #1e293b; border-radius: 8px; padding: 16px;
            margin: 12px 0; border-left: 4px solid {{ '#16a34a' }}; }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 4px;
             color: white; font-weight: bold; font-size: 13px; }
    .ticker    { font-size: 20px; font-weight: bold; color: #f8fafc; }
    .meta      { font-size: 12px; color: #94a3b8; margin-left: 10px; }
    .rationale { color: #cbd5e1; margin-top: 10px; font-size: 13px; line-height: 1.5; }
    .chart-img { width: 100%; max-width: 960px; border-radius: 6px;
                 margin-top: 14px; display: block; }
    .perf-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 10px; }
    .perf-table th { background: #0f172a; padding: 7px 10px; text-align: left; color: #94a3b8; }
    .perf-table td { padding: 6px 10px; border-top: 1px solid #334155; }
    .pos { color: #4ade80; } .neg { color: #f87171; }
    .footer { text-align: center; color: #475569; font-size: 12px;
              margin-top: 36px; border-top: 1px solid #1e293b; padding-top: 16px; }
  </style>
</head>
<body>
  <h1>LLM Trader — Actionable Signals</h1>
  <div class="subtitle">{{ generated_at }} &nbsp;&bull;&nbsp; {{ total }} tickers analysed</div>

  {% if overview_png %}
  <h2>Signal Overview</h2>
  <img class="chart-img" src="data:image/png;base64,{{ overview_png }}" alt="Signal Overview">
  {% endif %}

  {% if stocks %}
  <h2>Stocks</h2>
  {% for rec in stocks %}
  <div class="card" style="border-left-color: {{ colors[rec.action] }};">
    <span class="ticker">{{ rec.ticker }}</span>
    <span class="meta">{{ rec.direction }} &bull; conf {{ (rec.confidence * 100)|int }}% &bull; {{ rec.time_horizon }}</span>
    <br><br>
    <span class="badge" style="background:{{ colors[rec.action] }}">{{ rec.action }}</span>
    <p class="rationale">{{ rec.rationale }}</p>
    {% if charts and rec.ticker in charts %}
    <img class="chart-img" src="data:image/png;base64,{{ charts[rec.ticker] }}" alt="{{ rec.ticker }} chart">
    {% endif %}
  </div>
  {% endfor %}
  {% endif %}

  {% if etfs %}
  <h2>ETFs / Markets</h2>
  {% for rec in etfs %}
  <div class="card" style="border-left-color: {{ colors[rec.action] }};">
    <span class="ticker">{{ rec.ticker }}</span>
    <span class="meta">{{ rec.direction }} &bull; conf {{ (rec.confidence * 100)|int }}% &bull; {{ rec.time_horizon }}</span>
    <br><br>
    <span class="badge" style="background:{{ colors[rec.action] }}">{{ rec.action }}</span>
    <p class="rationale">{{ rec.rationale }}</p>
    {% if charts and rec.ticker in charts %}
    <img class="chart-img" src="data:image/png;base64,{{ charts[rec.ticker] }}" alt="{{ rec.ticker }} chart">
    {% endif %}
  </div>
  {% endfor %}
  {% endif %}

  {% if perf %}
  <h2>Portfolio Performance</h2>
  {% if equity_png %}
  <img class="chart-img" src="data:image/png;base64,{{ equity_png }}" alt="Equity Curve">
  {% endif %}

  {% if perf.stats %}
  <div class="card" style="border-left-color: #2563eb;">
    <strong>Closed trades ({{ perf.stats.total_closed }})</strong><br><br>
    Win rate: <strong>{{ perf.stats.win_rate }}%</strong> &bull;
    Avg return: <strong>{{ "%+.2f"|format(perf.stats.avg_return) }}%</strong> &bull;
    Best: <strong class="pos">{{ "%+.2f"|format(perf.stats.best) }}%</strong> &bull;
    Worst: <strong class="neg">{{ "%+.2f"|format(perf.stats.worst) }}%</strong>
  </div>
  {% endif %}

  {% if perf.open_trades %}
  <table class="perf-table">
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
    """
    Render all charts to base64 PNG strings.

    Returns:
        charts      — {ticker: b64_png} for each BUY/SELL ticker
        overview_png — b64 PNG of the signals overview chart
        equity_png   — b64 PNG of the equity curve chart
    """
    charts: dict[str, str] = {}

    # Per-ticker stock charts
    for rec in actionable:
        fig = build_stock_chart(rec.ticker, rec)
        b64 = fig_to_png_b64(fig, width=1100, height=780)
        if b64:
            charts[rec.ticker] = b64
            logger.debug(f"[email] Chart rendered for {rec.ticker}")

    # Signals overview
    overview_fig = build_signals_overview(all_recommendations or actionable)
    overview_png = fig_to_png_b64(overview_fig, width=1100, height=None)

    # Equity curve
    closed = (performance or {}).get("closed_trades", [])
    equity_fig = build_equity_curve(closed)
    equity_png = fig_to_png_b64(equity_fig, width=1100, height=480)

    return charts, overview_png, equity_png


def send_recommendations(
    recommendations: List[Recommendation],
    total_analysed: int = 0,
    performance: Optional[dict] = None,
    all_recommendations: Optional[List[Recommendation]] = None,
) -> bool:
    """Render and send the recommendation email with embedded chart images."""
    if not recommendations:
        logger.warning("No recommendations to send.")
        return False

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    stocks = sorted([r for r in recommendations if r.type == "STOCK"],  key=lambda x: x.confidence, reverse=True)
    etfs   = sorted([r for r in recommendations if r.type == "ETF"],    key=lambda x: x.confidence, reverse=True)

    # Build charts (gracefully skips if plotly/kaleido not installed)
    logger.info("[email] Rendering chart images...")
    charts, overview_png, equity_png = _build_chart_pngs(
        recommendations, all_recommendations or recommendations, performance
    )
    if charts:
        logger.info(f"[email] {len(charts)} chart image(s) embedded")
    else:
        logger.info("[email] No chart images (kaleido not available — text-only email)")

    html_body = Template(HTML_TEMPLATE).render(
        stocks=stocks,
        etfs=etfs,
        generated_at=now_str,
        total=total_analysed or len(recommendations),
        colors=ACTION_COLOR,
        perf=performance,
        charts=charts,
        overview_png=overview_png,
        equity_png=equity_png,
    )

    # Plain-text fallback
    lines = [f"Actionable Signals — {now_str}", ""]
    for section_name, section in [("STOCKS", stocks), ("ETFs / MARKETS", etfs)]:
        if section:
            lines.append(section_name)
            for rec in section:
                lines.append(f"  {rec.ticker}: {rec.action} ({rec.direction}, {rec.confidence:.0%}, {rec.time_horizon})")
                lines.append(f"    {rec.rationale}")
                lines.append("")
    text_body = "\n".join(lines)

    buys = [r.ticker for r in recommendations if r.action == "BUY"]
    subject = f"LLM Trader | {now_str} | BUY: {', '.join(buys) if buys else 'none'}"

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