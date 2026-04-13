"""Send recommendation digest emails via SMTP.

Layout:
  1. Signal overview chart
  2. Aggregated recommendations — summary table of every ticker
  3. Trade details — full per-method breakdown for each BUY / SELL
  4. Monitor list — compact HOLD / WATCH table
  5. Smart money signals
  6. Portfolio performance

Charts are attached as inline MIME images (CID references) so Gmail renders them.
If kaleido is not installed the email falls back gracefully to text-only.
"""

import smtplib
from datetime import datetime, timezone
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import base64
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
<img class="chart-img" src="cid:overview_chart" alt="Signal Overview">
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
      {% if enable_gex %}<th>GEX</th>{% endif %}
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
      {% if enable_gex %}
      <td style="font-size:11px;">
        {% if sig and sig.gex_signal %}
        {% set gex_col = {'PINNED': '#0ea5e9', 'AMPLIFIED': '#f87171', 'NEUTRAL': '#94a3b8'}.get(sig.gex_signal, '#94a3b8') %}
        <span style="color:{{ gex_col }};">{{ sig.gex_signal }}</span>
        {% if sig.expected_move_pct %}<span style="color:#475569;"> ±{{ "%.1f"|format(sig.expected_move_pct) }}%</span>{% endif %}
        {% else %}&mdash;{% endif %}
      </td>
      {% endif %}
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

    <!-- Put/Call Ratio -->
    {% set pc = sig.put_call_score if sig.put_call_score is defined else 0 %}
    {% if use_put_call and pc != 0 %}
    <div class="mrow">
      <div class="mhdr">
        <span class="mlabel">Put/Call Ratio</span>
        <span class="mscore {{ 'sp' if pc > 0 else 'sn' }}">
          {{ "%+.2f"|format(pc) }}
        </span>
      </div>
      <div class="bar-wrap">
        <div class="bar"
             style="width:{{ (pc|abs * 100)|int }}%;
                    background:{{ '#06b6d4' if pc >= 0 else '#a855f7' }};"></div>
      </div>
      <div class="mtext">
        {{ 'Elevated put volume → contrarian bullish signal' if pc > 0 else 'Elevated call volume → contrarian bearish signal' }}
      </div>
    </div>
    {% endif %}

    <!-- Max Pain gravity -->
    {% if sig.max_pain_score is defined and sig.max_pain_score != 0 %}
    {% set mp = sig.max_pain_score %}
    <div class="mrow">
      <div class="mhdr">
        <span class="mlabel">Max Pain</span>
        <span class="mscore {{ 'sp' if mp > 0 else 'sn' }}">
          {{ "%+.2f"|format(mp) }}
        </span>
      </div>
      <div class="bar-wrap">
        <div class="bar"
             style="width:{{ (mp|abs * 100)|int }}%;
                    background:{{ '#16a34a' if mp >= 0 else '#dc2626' }};"></div>
      </div>
      <div class="mtext">
        {% if mp > 0 %}Spot trading below max pain → gravity pulls price higher into expiry.
        {% else %}Spot trading above max pain → gravity pulls price lower into expiry.{% endif %}
        {% if sig.max_pain_bias %}<span style="color:#64748b;"> ({{ sig.max_pain_bias }} bias)</span>{% endif %}
      </div>
    </div>
    {% endif %}

    <!-- OI Skew -->
    {% if sig.oi_skew_score is defined and sig.oi_skew_score != 0 %}
    {% set sk = sig.oi_skew_score %}
    <div class="mrow">
      <div class="mhdr">
        <span class="mlabel">OI Skew</span>
        <span class="mscore {{ 'sp' if sk > 0 else 'sn' }}">
          {{ "%+.3f"|format(sk) }}
        </span>
      </div>
      <div class="bar-wrap">
        <div class="bar"
             style="width:{{ (sk|abs * 100)|int }}%;
                    background:{{ '#16a34a' if sk >= 0 else '#dc2626' }};"></div>
      </div>
      <div class="mtext">
        {% if sk > 0.3 %}Strong call-side OI lean → bullish positioning bias.
        {% elif sk > 0.1 %}Mild call-side OI lean → slight bullish tilt.
        {% elif sk < -0.3 %}Strong put-side OI lean → bearish positioning bias.
        {% elif sk < -0.1 %}Mild put-side OI lean → slight bearish tilt.
        {% else %}OI roughly balanced between calls and puts.{% endif %}
        <span style="color:#64748b;font-size:10px;"> (directional, not contrarian)</span>
      </div>
    </div>
    {% endif %}

    <!-- VWAP Distance -->
    {% if sig.vwap_score is defined and sig.vwap_score != 0 %}
    {% set vw = sig.vwap_score %}
    <div class="mrow">
      <div class="mhdr">
        <span class="mlabel">VWAP Distance</span>
        <span class="mscore {{ 'sp' if vw > 0 else 'sn' }}">
          {{ "%+.2f"|format(vw) }}
        </span>
        {% if sig.vwap_distance_pct %}
        <span style="font-size:10px;color:#64748b;margin-left:6px;">
          ({{ "%+.1f"|format(sig.vwap_distance_pct) }}% from VWAP)
        </span>
        {% endif %}
      </div>
      <div class="bar-wrap">
        <div class="bar"
             style="width:{{ (vw|abs * 100)|int }}%;
                    background:{{ '#16a34a' if vw >= 0 else '#dc2626' }};"></div>
      </div>
      <div class="mtext">
        {% if vw > 0.5 %}Price stretched well below VWAP — institutions likely to step in as buyers.
        {% elif vw > 0.15 %}Price below VWAP — mild mean-reversion pull upward.
        {% elif vw < -0.5 %}Price stretched well above VWAP — institutions likely to sell into strength.
        {% elif vw < -0.15 %}Price above VWAP — mild mean-reversion pull downward.
        {% else %}Price near VWAP — no significant reversion pressure.{% endif %}
        <span style="color:#64748b;font-size:10px;"> (mean-reversion; weaker in strong trends)</span>
      </div>
    </div>
    {% endif %}

    <!-- Gamma Exposure (GEX) per-ticker -->
    {% if sig.gex_signal %}
    {% set gex_col = {'PINNED': '#0ea5e9', 'AMPLIFIED': '#f87171', 'NEUTRAL': '#94a3b8'}.get(sig.gex_signal, '#94a3b8') %}
    <div class="mrow">
      <div class="mhdr">
        <span class="mlabel">Gamma Exposure</span>
        <span style="font-size:11px;font-weight:bold;padding:2px 7px;border-radius:3px;
                     background:#1e293b;color:{{ gex_col }};border:1px solid {{ gex_col }};">
          {{ sig.gex_signal }}
        </span>
      </div>
      <div class="mtext" style="display:flex;gap:16px;flex-wrap:wrap;margin-top:4px;">
        {% if sig.gamma_flip is not none %}
        <span>Gamma flip: <strong style="color:#e2e8f0;">${{ "%.2f"|format(sig.gamma_flip) }}</strong></span>
        {% endif %}
        {% if sig.max_pain_bias %}
        <span>Max pain bias:
          <strong style="color:{{ {'BULLISH':'#4ade80','BEARISH':'#f87171'}.get(sig.max_pain_bias,'#94a3b8') }};">
            {{ sig.max_pain_bias }}
          </strong>
        </span>
        {% endif %}
        {% if sig.expected_move_pct %}
        <span>Exp. move: <strong style="color:#e2e8f0;">±{{ "%.1f"|format(sig.expected_move_pct) }}%</strong></span>
        {% endif %}
      </div>
      <div class="mtext" style="margin-top:5px;color:#64748b;font-size:11px;">
        {% if sig.gex_signal == 'PINNED' %}Positive GEX → dealers suppress volatility; price likely stays near gamma flip.
        {% elif sig.gex_signal == 'AMPLIFIED' %}Negative GEX → dealers amplify moves; expect wider swings than normal.
        {% else %}Balanced GEX → no structural dealer-flow bias.{% endif %}
      </div>
    </div>
    {% endif %}

  </div>
  {% endif %}

  <!-- Per-ticker price chart -->
  {% if charts and rec.ticker in charts %}
  <img class="chart-img"
       src="cid:chart_{{ rec.ticker }}"
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
    {% if rec.rationale %}
    <tr>
      <td colspan="{{ 6 + (1 if use_news else 0) + (1 if use_tech else 0) }}"
          style="padding:4px 12px 10px 12px;color:#94a3b8;font-size:12px;
                 line-height:1.5;border-top:none;">
        {{ rec.rationale }}
      </td>
    </tr>
    {% endif %}
    {% endfor %}
  </tbody>
</table>
{% endif %}

<!-- ══════════════════════════════════════
     3b — ANALYST RATINGS
     ══════════════════════════════════════ -->
{% if analyst_articles %}
<h2>Analyst Ratings <span style="font-size:13px;font-weight:400;color:#94a3b8;">(upgrades · downgrades · price targets)</span></h2>
{% for art in analyst_articles %}
{% set is_bull = 'upgrade' in art.title.lower() or 'initiated' in art.title.lower() %}
{% set is_bear = 'downgrade' in art.title.lower() %}
{% set accent = '#4ade80' if is_bull else ('#f87171' if is_bear else '#0ea5e9') %}
<div class="card" style="border-left: 4px solid {{ accent }}; padding: 14px 18px; margin: 10px 0;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
    <span style="font-size:14px;font-weight:700;color:#f8fafc;">{{ art.title }}</span>
  </div>
  <div style="font-size:12px;color:#94a3b8;line-height:1.6;">{{ art.summary }}</div>
  <div style="font-size:11px;color:#475569;margin-top:6px;">
    <span class="art-src">[{{ art.source }}]</span>
    &nbsp;&bull;&nbsp;{{ fmt_et(art.published_at) }}
    &nbsp;&bull;&nbsp;<a href="{{ art.url }}" style="color:#475569;">Yahoo Finance →</a>
  </div>
</div>
{% endfor %}
{% endif %}

<!-- ══════════════════════════════════════
     3c — EPS SURPRISES + ALTERNATIVE DATA
     ══════════════════════════════════════ -->
{% set all_alt = eps_articles + alt_data_articles %}
{% if all_alt %}
<h2>Alternative Signals <span style="font-size:13px;font-weight:400;color:#94a3b8;">(EPS surprises · Google Trends · Reddit · Short Interest)</span></h2>
<table class="ctbl">
  <thead>
    <tr>
      <th>Source</th>
      <th>Signal</th>
      <th>Published</th>
    </tr>
  </thead>
  <tbody>
  {% for art in all_alt %}
  {% set src_color = {
      'Earnings/EPS':   '#f59e0b',
      'Google Trends':  '#0ea5e9',
      'Reddit/WSB':     '#7c3aed',
      'Short Interest': '#ec4899'
  }.get(art.source, '#94a3b8') %}
  {% set is_bull = 'beat' in art.title.lower() or 'surge' in art.title.lower() or 'squeeze' in art.title.lower() or 'covering' in art.title.lower() or 'bullish' in art.summary.lower()[:80] %}
  {% set is_bear = 'miss' in art.title.lower() or 'drop' in art.title.lower() or 'bearish' in art.title.lower() or 'bearish' in art.summary.lower()[:80] %}
    <tr>
      <td>
        <span style="background:#0f172a;color:{{ src_color }};border:1px solid {{ src_color }};
                     border-radius:3px;padding:2px 6px;font-size:10px;font-weight:700;white-space:nowrap;">
          {{ art.source }}
        </span>
      </td>
      <td>
        <span style="color:{{ '#4ade80' if is_bull else ('#f87171' if is_bear else '#e2e8f0') }};font-size:12px;">
          {{ art.title }}
        </span>
      </td>
      <td style="color:#475569;font-size:11px;white-space:nowrap;">{{ fmt_et(art.published_at) }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
{% endif %}

<!-- ══════════════════════════════════════
     4 — MACRO CONTEXT (FRED)
     ══════════════════════════════════════ -->
{% if macro_context %}
{% set regime_color = {'EXPANSION': '#4ade80', 'SLOWDOWN': '#fb923c', 'LATE_CYCLE': '#f59e0b', 'RECESSION': '#f87171'} %}
{% set regime_bg    = {'EXPANSION': '#14532d', 'SLOWDOWN': '#431407', 'LATE_CYCLE': '#451a03', 'RECESSION': '#450a0a'} %}
<h2>Macro Context <span style="font-size:13px;font-weight:400;color:#94a3b8;">(FRED)</span></h2>
<div class="card" style="border-left: 4px solid {{ regime_color.get(macro_context.regime, '#94a3b8') }};">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
    <span class="ticker" style="font-size:15px;">{{ macro_context.regime }}</span>
    <span class="badge" style="background:{{ regime_bg.get(macro_context.regime, '#1e293b') }};color:{{ regime_color.get(macro_context.regime, '#94a3b8') }};border:1px solid {{ regime_color.get(macro_context.regime, '#94a3b8') }};">
      Macro Regime
    </span>
  </div>
  <p style="color:#cbd5e1;font-size:13px;margin:0 0 14px 0;">{{ macro_context.summary }}</p>
  <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:10px;">

    {% if macro_context.yield_spread_10y2y is not none %}
    <div style="background:#0f172a;border-radius:6px;padding:10px;">
      <div style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.5px;">Yield Curve (10Y-2Y)</div>
      <div style="font-size:20px;font-weight:700;margin:4px 0;color:{{ '#f87171' if macro_context.yield_spread_10y2y < 0 else '#4ade80' }};">
        {{ "%+.2f"|format(macro_context.yield_spread_10y2y) }}%
      </div>
      <div style="font-size:12px;color:#94a3b8;">{{ macro_context.yield_curve_signal }}</div>
    </div>
    {% endif %}

    {% if macro_context.fed_funds_rate is not none %}
    <div style="background:#0f172a;border-radius:6px;padding:10px;">
      <div style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.5px;">Fed Funds Rate</div>
      <div style="font-size:20px;font-weight:700;margin:4px 0;color:#e2e8f0;">
        {{ "%.2f"|format(macro_context.fed_funds_rate) }}%
      </div>
    </div>
    {% endif %}

    {% if macro_context.cpi_yoy is not none %}
    {% set inf_color = {'HIGH': '#f87171', 'ELEVATED': '#fb923c', 'MODERATE': '#4ade80', 'LOW': '#60a5fa'} %}
    <div style="background:#0f172a;border-radius:6px;padding:10px;">
      <div style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.5px;">CPI YoY</div>
      <div style="font-size:20px;font-weight:700;margin:4px 0;color:{{ inf_color.get(macro_context.inflation_signal, '#e2e8f0') }};">
        {{ "%+.1f"|format(macro_context.cpi_yoy) }}%
      </div>
      <div style="font-size:12px;color:#94a3b8;">{{ macro_context.inflation_signal }}</div>
    </div>
    {% endif %}

    {% if macro_context.unemployment_rate is not none %}
    {% set unemp_color = {'RISING': '#f87171', 'STABLE': '#e2e8f0', 'FALLING': '#4ade80'} %}
    <div style="background:#0f172a;border-radius:6px;padding:10px;">
      <div style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.5px;">Unemployment</div>
      <div style="font-size:20px;font-weight:700;margin:4px 0;color:{{ unemp_color.get(macro_context.unemployment_trend, '#e2e8f0') }};">
        {{ "%.1f"|format(macro_context.unemployment_rate) }}%
      </div>
      <div style="font-size:12px;color:#94a3b8;">{{ macro_context.unemployment_trend }}</div>
    </div>
    {% endif %}

    {% if macro_context.hy_spread is not none %}
    {% set credit_color = {'STRESSED': '#f87171', 'ELEVATED': '#fb923c', 'NORMAL': '#e2e8f0', 'TIGHT': '#4ade80'} %}
    <div style="background:#0f172a;border-radius:6px;padding:10px;">
      <div style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.5px;">HY Credit Spread</div>
      <div style="font-size:20px;font-weight:700;margin:4px 0;color:{{ credit_color.get(macro_context.credit_signal, '#e2e8f0') }};">
        {{ "%.2f"|format(macro_context.hy_spread) }}%
      </div>
      <div style="font-size:12px;color:#94a3b8;">{{ macro_context.credit_signal }}</div>
    </div>
    {% endif %}

    {% if macro_context.m2_growth_yoy is not none %}
    <div style="background:#0f172a;border-radius:6px;padding:10px;">
      <div style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.5px;">M2 Growth YoY</div>
      <div style="font-size:20px;font-weight:700;margin:4px 0;color:{{ '#4ade80' if macro_context.m2_growth_yoy > 2 else ('#f87171' if macro_context.m2_growth_yoy < -2 else '#e2e8f0') }};">
        {{ "%+.1f"|format(macro_context.m2_growth_yoy) }}%
      </div>
    </div>
    {% endif %}

  </div>
</div>
{% endif %}

<!-- ══════════════════════════════════════
     5 — IPO PIPELINE (SEC S-1/S-11)
     ══════════════════════════════════════ -->
{% if ipo_context and ipo_context.total_new > 0 %}
{% set activity_color = '#4ade80' if ipo_context.total_new >= 20 else ('#fb923c' if ipo_context.total_new >= 10 else '#94a3b8') %}
<h2>IPO Pipeline <span style="font-size:13px;font-weight:400;color:#94a3b8;">(SEC S-1/S-11 — last {{ ipo_context.lookback_days }} days)</span></h2>
<div class="card" style="border-left: 4px solid #0ea5e9;">
  <p style="color:#cbd5e1;font-size:13px;margin:0 0 14px 0;">{{ ipo_context.summary }}</p>

  <!-- Sector bar chart (relative widths) -->
  {% set max_count = ipo_context.sector_counts.values() | max %}
  <div style="margin-bottom:18px;">
    {% for sector, count in ipo_context.sector_counts.items() %}
    {% set bar_pct = (count / max_count * 100) | int %}
    {% set is_hot  = sector in ipo_context.hot_sectors %}
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
      <div style="width:160px;font-size:12px;color:{{ '#e2e8f0' if is_hot else '#94a3b8' }};text-align:right;flex-shrink:0;">
        {{ sector }}
      </div>
      <div style="flex:1;background:#1e293b;border-radius:3px;height:16px;">
        <div style="width:{{ bar_pct }}%;background:{{ '#0ea5e9' if is_hot else '#334155' }};border-radius:3px;height:100%;"></div>
      </div>
      <div style="width:28px;font-size:12px;color:{{ '#0ea5e9' if is_hot else '#64748b' }};font-weight:{{ '700' if is_hot else '400' }};">
        {{ count }}
      </div>
    </div>
    {% endfor %}
    <div style="font-size:11px;color:#475569;margin-top:6px;">
      + {{ ipo_context.total_amendments }} amendment(s) (S-1/A, S-11/A) — companies advancing toward listing
    </div>
  </div>

  <!-- Recent initial filings table -->
  {% if ipo_context.filings %}
  <div style="font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;">Recent Registrations</div>
  <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-size:12px;">
      <thead>
        <tr style="color:#475569;">
          <th style="text-align:left;padding:4px 8px;">Date</th>
          <th style="text-align:left;padding:4px 8px;">Form</th>
          <th style="text-align:left;padding:4px 8px;">Sector</th>
          <th style="text-align:left;padding:4px 8px;">Company</th>
        </tr>
      </thead>
      <tbody>
      {% for f in ipo_context.filings[:12] %}
        <tr style="border-top:1px solid #1e293b;">
          <td style="padding:5px 8px;color:#64748b;white-space:nowrap;">{{ f.filing_date }}</td>
          <td style="padding:5px 8px;color:#94a3b8;">{{ f.form_type }}</td>
          <td style="padding:5px 8px;color:#94a3b8;font-size:11px;">{{ f.sector }}</td>
          <td style="padding:5px 8px;color:#e2e8f0;">{{ f.company }}</td>
        </tr>
      {% endfor %}
      {% if ipo_context.total_new > 12 %}
        <tr><td colspan="4" style="padding:5px 8px;color:#475569;font-size:11px;">… and {{ ipo_context.total_new - 12 }} more</td></tr>
      {% endif %}
      </tbody>
    </table>
  </div>
  {% endif %}
</div>
{% endif %}

<!-- ══════════════════════════════════════
     5b — UPCOMING EARNINGS CALENDAR
     ══════════════════════════════════════ -->
{% if earnings_context and earnings_context.upcoming %}
<h2>Upcoming Earnings <span style="font-size:13px;font-weight:400;color:#94a3b8;">(next {{ earnings_context.upcoming[-1].days_until }} days)</span></h2>
<div class="card" style="border-left: 4px solid #f59e0b;">
  <p style="color:#cbd5e1;font-size:13px;margin:0 0 14px 0;">{{ earnings_context.summary }}</p>
  <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <thead>
        <tr style="color:#64748b;text-transform:uppercase;font-size:11px;letter-spacing:.5px;">
          <th style="text-align:left;padding:6px 8px;">Ticker</th>
          <th style="text-align:left;padding:6px 8px;">Earnings Date</th>
          <th style="text-align:right;padding:6px 8px;">Days</th>
          <th style="text-align:right;padding:6px 8px;">EPS Estimate</th>
          <th style="text-align:left;padding:6px 8px;">Alert</th>
        </tr>
      </thead>
      <tbody>
      {% for ev in earnings_context.upcoming %}
      {% set urgency_color = '#f87171' if ev.days_until <= 3 else ('#f59e0b' if ev.days_until <= 7 else '#94a3b8') %}
        <tr style="border-top:1px solid #1e293b;">
          <td style="padding:7px 8px;color:#e2e8f0;font-weight:600;">{{ ev.ticker }}</td>
          <td style="padding:7px 8px;color:#94a3b8;">{{ ev.earnings_date }}</td>
          <td style="padding:7px 8px;text-align:right;color:{{ urgency_color }};font-weight:700;">{{ ev.days_until }}d</td>
          <td style="padding:7px 8px;text-align:right;color:#cbd5e1;">
            {% if ev.estimated_eps is not none %}${{ "%.2f"|format(ev.estimated_eps) }}{% else %}&mdash;{% endif %}
          </td>
          <td style="padding:7px 8px;">
            {% if ev.days_until <= 3 %}
            <span style="background:#450a0a;color:#f87171;border:1px solid #f87171;border-radius:4px;padding:2px 7px;font-size:11px;">BINARY EVENT</span>
            {% elif ev.days_until <= 7 %}
            <span style="background:#451a03;color:#f59e0b;border:1px solid #f59e0b;border-radius:4px;padding:2px 7px;font-size:11px;">IV EXPANSION</span>
            {% endif %}
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  <p style="color:#475569;font-size:11px;margin:10px 0 0 0;">
    BINARY EVENT: avoid POSITION-length trades. IV EXPANSION: pre-earnings options plays may be favoured over stock.
  </p>
</div>
{% endif %}

<!-- ══════════════════════════════════════
     6 — COT FUTURES POSITIONING (CFTC)
     ══════════════════════════════════════ -->
{% if cot_context %}
{% set sig_color = {
    'EXTREME_LONG':  '#f87171',
    'BULLISH_TREND': '#4ade80',
    'NEUTRAL':       '#94a3b8',
    'BEARISH_TREND': '#fb923c',
    'EXTREME_SHORT': '#60a5fa'
} %}
{% set dir_color = {'BULLISH': '#4ade80', 'BEARISH': '#f87171', 'NEUTRAL': '#94a3b8'} %}
<h2>COT Futures Positioning <span style="font-size:13px;font-weight:400;color:#94a3b8;">(CFTC — as of {{ cot_context.report_date }})</span></h2>
<div class="card" style="border-left: 4px solid #7c3aed;">
  <p style="color:#cbd5e1;font-size:13px;margin:0 0 14px 0;">{{ cot_context.summary }}</p>
  <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-size:13px;">
      <thead>
        <tr style="color:#64748b;text-transform:uppercase;font-size:11px;letter-spacing:.5px;">
          <th style="text-align:left;padding:6px 8px;">Contract</th>
          <th style="text-align:left;padding:6px 8px;">Tickers</th>
          <th style="text-align:right;padding:6px 8px;">Net Spec %</th>
          <th style="text-align:right;padding:6px 8px;">WoW</th>
          <th style="text-align:right;padding:6px 8px;">52W Pct</th>
          <th style="text-align:center;padding:6px 8px;">Signal</th>
          <th style="text-align:center;padding:6px 8px;">Direction</th>
        </tr>
      </thead>
      <tbody>
      {% for s in cot_context.signals %}
        <tr style="border-top:1px solid #1e293b;">
          <td style="padding:7px 8px;color:#e2e8f0;font-weight:600;">{{ s.contract }}</td>
          <td style="padding:7px 8px;color:#94a3b8;font-size:12px;">{{ s.tickers | join(', ') }}</td>
          <td style="padding:7px 8px;text-align:right;color:{{ '#4ade80' if s.net_speculator_pct > 0 else '#f87171' }};">
            {{ "%+.1f"|format(s.net_speculator_pct) }}%
          </td>
          <td style="padding:7px 8px;text-align:right;color:{{ '#4ade80' if s.net_change_wow > 0.5 else ('#f87171' if s.net_change_wow < -0.5 else '#94a3b8') }};">
            {{ "%+.1f"|format(s.net_change_wow) }}%
          </td>
          <td style="padding:7px 8px;text-align:right;color:#e2e8f0;">{{ "%.0f"|format(s.percentile_52w) }}th</td>
          <td style="padding:7px 8px;text-align:center;">
            <span style="background:#1e293b;color:{{ sig_color.get(s.signal, '#94a3b8') }};border:1px solid {{ sig_color.get(s.signal, '#334155') }};border-radius:4px;padding:2px 7px;font-size:11px;white-space:nowrap;">
              {{ s.signal.replace('_', ' ') }}
            </span>
          </td>
          <td style="padding:7px 8px;text-align:center;color:{{ dir_color.get(s.direction, '#94a3b8') }};font-weight:700;">
            {{ '▲' if s.direction == 'BULLISH' else ('▼' if s.direction == 'BEARISH' else '→') }}
            {{ s.direction }}
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  <p style="color:#475569;font-size:11px;margin:12px 0 0 0;">
    Contrarian applied at extremes: EXTREME_LONG → bearish signal; EXTREME_SHORT → bullish signal.
    Managed Money (commodities) / Leveraged Money (index futures) as proxy for hedge fund positioning.
  </p>
</div>
{% endif %}

<!-- ══════════════════════════════════════
     6a — VIX & TERM STRUCTURE
     ══════════════════════════════════════ -->
{% if vix_context and vix_context.vix is not none %}
{% set vix_sig_color = {
    'PANIC':        '#7c3aed',
    'EXTREME_FEAR': '#f87171',
    'HIGH':         '#fb923c',
    'ELEVATED':     '#fbbf24',
    'NORMAL':       '#94a3b8',
    'LOW':          '#86efac',
    'COMPLACENCY':  '#4ade80',
    'UNKNOWN':      '#64748b'
} %}
{% set ts_color = {
    'BACKWARDATION': '#f87171',
    'FLAT':          '#94a3b8',
    'CONTANGO':      '#4ade80',
    'UNKNOWN':       '#64748b'
} %}
{% set dir_color = {'BULLISH': '#4ade80', 'BEARISH': '#f87171', 'NEUTRAL': '#94a3b8'} %}
<h2>VIX &amp; Volatility Regime <span style="font-size:13px;font-weight:400;color:#94a3b8;">(^VIX · ^VXN · ^VVIX · term structure)</span></h2>
<div class="card" style="border-left: 4px solid {{ vix_sig_color.get(vix_context.vix_signal, '#94a3b8') }};">

  <!-- Top row: VIX gauge + term structure -->
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px;">

    <!-- VIX gauge -->
    <div style="background:#0f172a;border-radius:8px;padding:12px 18px;text-align:center;min-width:110px;">
      <div style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.5px;">VIX</div>
      <div style="font-size:30px;font-weight:800;color:{{ vix_sig_color.get(vix_context.vix_signal, '#e2e8f0') }};margin:4px 0;">
        {{ "%.1f"|format(vix_context.vix) }}
      </div>
      <div style="font-size:11px;font-weight:700;color:{{ vix_sig_color.get(vix_context.vix_signal, '#94a3b8') }};">
        {{ vix_context.vix_signal.replace('_', ' ') }}
      </div>
      <div style="font-size:11px;color:{{ dir_color.get(vix_context.vix_direction, '#94a3b8') }};margin-top:3px;">
        {{ '▲ BULLISH' if vix_context.vix_direction == 'BULLISH' else ('▼ BEARISH' if vix_context.vix_direction == 'BEARISH' else '→ NEUTRAL') }} bias
      </div>
    </div>

    <!-- Term structure curve -->
    <div style="background:#0f172a;border-radius:8px;padding:12px 16px;flex:1;min-width:240px;">
      <div style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;">
        Vol Curve
        <span style="color:{{ ts_color.get(vix_context.term_structure, '#94a3b8') }};margin-left:6px;">
          {{ vix_context.term_structure }}
          {% if vix_context.slope_1m_3m is not none %}({{ "%+.1f"|format(vix_context.slope_1m_3m) }}pt){% endif %}
        </span>
      </div>
      <div style="display:flex;gap:8px;align-items:flex-end;height:50px;">
        {% set levels = [
            ('9D', vix_context.vix9d),
            ('1M', vix_context.vix),
            ('3M', vix_context.vix3m),
            ('6M', vix_context.vix6m)
        ] %}
        {% set max_val = [vix_context.vix9d or 0, vix_context.vix or 0, vix_context.vix3m or 0, vix_context.vix6m or 0] | max %}
        {% for label, val in levels %}
        {% if val is not none %}
        {% set bar_h = ((val / max_val) * 44) | int if max_val > 0 else 20 %}
        <div style="display:flex;flex-direction:column;align-items:center;flex:1;">
          <div style="font-size:10px;color:#94a3b8;margin-bottom:2px;">{{ "%.1f"|format(val) }}</div>
          <div style="height:{{ bar_h }}px;width:100%;background:{{ ts_color.get(vix_context.term_structure,'#334155') }};border-radius:3px 3px 0 0;opacity:0.8;"></div>
          <div style="font-size:10px;color:#475569;margin-top:2px;">{{ label }}</div>
        </div>
        {% endif %}
        {% endfor %}
      </div>
    </div>

    <!-- Related indices -->
    <div style="background:#0f172a;border-radius:8px;padding:12px 16px;min-width:130px;">
      <div style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;">Related</div>
      {% if vix_context.vxn is not none %}
      <div style="margin-bottom:6px;">
        <span style="color:#475569;font-size:11px;">VXN (Nasdaq)</span>
        <span style="color:#e2e8f0;font-weight:700;float:right;">{{ "%.1f"|format(vix_context.vxn) }}</span>
      </div>
      {% endif %}
      {% if vix_context.vvix is not none %}
      {% set vvix_color = '#f87171' if vix_context.vvix > 120 else ('#fb923c' if vix_context.vvix > 100 else '#94a3b8') %}
      <div>
        <span style="color:#475569;font-size:11px;">VVIX (vol-of-vol)</span>
        <span style="color:{{ vvix_color }};font-weight:700;float:right;">{{ "%.1f"|format(vix_context.vvix) }}</span>
      </div>
      {% endif %}
    </div>
  </div>

  <p style="color:#cbd5e1;font-size:13px;margin:0 0 6px 0;">{{ vix_context.summary }}</p>
  <p style="color:#475569;font-size:11px;margin:0;">
    Contrarian: EXTREME_FEAR / PANIC → fade SELLs, upgrade BUY conviction on quality names.
    BACKWARDATION = near-term panic spike, often marks short-term lows.
    COMPLACENCY → reduce aggressive long exposure.
  </p>
</div>
{% endif %}

<!-- ══════════════════════════════════════
     6b — PUT/CALL RATIO
     ══════════════════════════════════════ -->
{% if put_call_context %}
{% set mkt_color = {
    'EXTREME_GREED': '#f87171',
    'GREED':         '#fb923c',
    'NEUTRAL':       '#94a3b8',
    'FEAR':          '#4ade80',
    'EXTREME_FEAR':  '#60a5fa',
    'UNKNOWN':       '#64748b'
} %}
{% set dir_color = {'BULLISH': '#4ade80', 'BEARISH': '#f87171', 'NEUTRAL': '#94a3b8'} %}
{% set sig_color = {
    'EXTREME_PUTS':  '#f87171',
    'PUTS_HEAVY':    '#fb923c',
    'BALANCED':      '#94a3b8',
    'CALLS_HEAVY':   '#86efac',
    'EXTREME_CALLS': '#4ade80'
} %}
<h2>Put/Call Ratio <span style="font-size:13px;font-weight:400;color:#94a3b8;">(CBOE equity · yfinance per-ticker)</span></h2>
<div class="card" style="border-left: 4px solid {{ mkt_color.get(put_call_context.market_signal, '#94a3b8') }};">

  <!-- Market-wide gauge -->
  <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:14px;">
    <div style="background:#0f172a;border-radius:8px;padding:12px 18px;text-align:center;min-width:120px;">
      <div style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.5px;">CBOE Equity P/C</div>
      <div style="font-size:28px;font-weight:800;color:{{ mkt_color.get(put_call_context.market_signal, '#e2e8f0') }};margin:4px 0;">
        {% if put_call_context.market_pc_ratio is not none %}{{ "%.2f"|format(put_call_context.market_pc_ratio) }}{% else %}N/A{% endif %}
      </div>
      <div style="font-size:12px;color:{{ mkt_color.get(put_call_context.market_signal, '#94a3b8') }};font-weight:700;">
        {{ put_call_context.market_signal.replace('_', ' ') }}
      </div>
    </div>
    <div>
      <div style="font-size:13px;color:#cbd5e1;margin-bottom:4px;">{{ put_call_context.summary }}</div>
      <div style="font-size:12px;color:#475569;">
        Contrarian signal: FEAR → market bottom risk; GREED → complacency risk.
        Range: &lt;0.60 extreme calls, 0.80–1.00 neutral, &gt;1.20 extreme puts.
      </div>
    </div>
  </div>

  <!-- Per-ticker extremes -->
  {% if put_call_context.ticker_signals %}
  <div style="font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;">Per-Ticker Extremes</div>
  <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-size:12px;">
      <thead>
        <tr style="color:#475569;text-transform:uppercase;font-size:10px;letter-spacing:.5px;">
          <th style="text-align:left;padding:5px 8px;">Ticker</th>
          <th style="text-align:right;padding:5px 8px;">P/C Ratio</th>
          <th style="text-align:right;padding:5px 8px;">Put Vol</th>
          <th style="text-align:right;padding:5px 8px;">Call Vol</th>
          <th style="text-align:center;padding:5px 8px;">Signal</th>
          <th style="text-align:center;padding:5px 8px;">Direction</th>
        </tr>
      </thead>
      <tbody>
      {% for s in put_call_context.ticker_signals %}
        <tr style="border-top:1px solid #1e293b;">
          <td style="padding:6px 8px;color:#e2e8f0;font-weight:600;">{{ s.ticker }}</td>
          <td style="padding:6px 8px;text-align:right;color:{{ '#f87171' if s.put_call_ratio > 1.5 else ('#4ade80' if s.put_call_ratio < 0.5 else '#e2e8f0') }};font-weight:700;">
            {{ "%.2f"|format(s.put_call_ratio) }}
          </td>
          <td style="padding:6px 8px;text-align:right;color:#94a3b8;">{{ "{:,}".format(s.put_volume) }}</td>
          <td style="padding:6px 8px;text-align:right;color:#94a3b8;">{{ "{:,}".format(s.call_volume) }}</td>
          <td style="padding:6px 8px;text-align:center;">
            <span style="background:#1e293b;color:{{ sig_color.get(s.signal, '#94a3b8') }};border:1px solid {{ sig_color.get(s.signal, '#334155') }};border-radius:4px;padding:2px 7px;font-size:10px;white-space:nowrap;">
              {{ s.signal.replace('_', ' ') }}
            </span>
          </td>
          <td style="padding:6px 8px;text-align:center;color:{{ dir_color.get(s.direction, '#94a3b8') }};font-weight:700;">
            {{ '▲' if s.direction == 'BULLISH' else ('▼' if s.direction == 'BEARISH' else '→') }}
            {{ s.direction }}
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  {% else %}
  <div style="font-size:12px;color:#475569;">No per-ticker extremes detected — all watchlist tickers have balanced put/call activity.</div>
  {% endif %}
</div>
{% endif %}

<!-- ══════════════════════════════════════
     6c — NYSE TICK INDEX
     ══════════════════════════════════════ -->
{% if tick_context and tick_context.signal != 'UNKNOWN' %}
{% set tick_sig_color = {
    'EXTREME_BULLS': '#f87171',
    'EXTREME_BEARS': '#4ade80',
    'WHIPSAW':       '#fbbf24',
    'NEUTRAL':       '#94a3b8'
} %}
{% set tick_dir_color = {'BULLISH': '#4ade80', 'BEARISH': '#f87171', 'NEUTRAL': '#94a3b8'} %}
<h2>NYSE TICK Index <span style="font-size:13px;font-weight:400;color:#94a3b8;">(^TICK · breadth exhaustion)</span></h2>
<div class="card" style="border-left: 4px solid {{ tick_sig_color.get(tick_context.signal, '#94a3b8') }};">
  <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px;">

    <!-- TICK gauge -->
    <div style="background:#0f172a;border-radius:8px;padding:12px 18px;text-align:center;min-width:130px;">
      <div style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.5px;">{{ tick_context.session_date }}</div>
      <div style="font-size:22px;font-weight:800;color:{{ tick_sig_color.get(tick_context.signal, '#e2e8f0') }};margin:4px 0;">
        {{ tick_context.signal.replace('_', ' ') }}
      </div>
      <div style="font-size:11px;color:{{ tick_dir_color.get(tick_context.direction, '#94a3b8') }};margin-top:3px;">
        {{ '▲ BULLISH' if tick_context.direction == 'BULLISH' else ('▼ BEARISH' if tick_context.direction == 'BEARISH' else '→ NEUTRAL') }} bias (contrarian)
      </div>
    </div>

    <!-- High / Low / Close tiles -->
    <div style="display:flex;flex-direction:column;gap:6px;justify-content:center;">
      {% for label, val, color in [
          ('Session High', tick_context.tick_high, '#f87171'),
          ('Session Low',  tick_context.tick_low,  '#4ade80'),
          ('Close',        tick_context.tick_close, '#94a3b8')
      ] %}
      {% if val is not none %}
      <div style="background:#0f172a;border-radius:6px;padding:6px 14px;display:flex;gap:12px;align-items:center;">
        <span style="color:#64748b;font-size:11px;min-width:90px;">{{ label }}</span>
        <span style="font-weight:700;color:{{ color }};font-size:14px;">{{ "%+.0f"|format(val) }}</span>
        {% if label == 'Session High' and val >= 1000 %}
        <span style="font-size:10px;color:#f87171;margin-left:4px;">⚠ EXTREME</span>
        {% elif label == 'Session Low' and val <= -1000 %}
        <span style="font-size:10px;color:#4ade80;margin-left:4px;">⚠ EXTREME</span>
        {% endif %}
      </div>
      {% endif %}
      {% endfor %}
    </div>

    <!-- 5-session extreme counts -->
    <div style="background:#0f172a;border-radius:8px;padding:12px 16px;font-size:12px;">
      <div style="color:#64748b;font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;">5-Session Pattern</div>
      <div style="margin-bottom:5px;">
        <span style="color:#94a3b8;">TICK &gt; +1000:</span>
        <span style="font-weight:700;color:{{ '#f87171' if tick_context.extreme_high_count >= 3 else '#e2e8f0' }};margin-left:6px;">
          {{ tick_context.extreme_high_count }} / 5
        </span>
        {% if tick_context.extreme_high_count >= 3 %}<span style="color:#f87171;font-size:10px;"> sustained buying</span>{% endif %}
      </div>
      <div>
        <span style="color:#94a3b8;">TICK &lt; −1000:</span>
        <span style="font-weight:700;color:{{ '#4ade80' if tick_context.extreme_low_count >= 3 else '#e2e8f0' }};margin-left:6px;">
          {{ tick_context.extreme_low_count }} / 5
        </span>
        {% if tick_context.extreme_low_count >= 3 %}<span style="color:#4ade80;font-size:10px;"> repeated flush</span>{% endif %}
      </div>
    </div>
  </div>

  <p style="color:#94a3b8;font-size:12px;margin:0;">{{ tick_context.summary }}</p>
  <p style="color:#475569;font-size:11px;margin:10px 0 0 0;">
    TICK measures NYSE upticks − downticks in real time.
    Extreme readings (&gt;+1000 or &lt;−1000) are contrarian short-term reversal signals.
    EXTREME_BEARS = capitulation flush → fade the sell; EXTREME_BULLS = buying climax → fade the ramp.
    Directional edge weakens when the same extreme repeats 3+ sessions (distribution / accumulation phase).
  </p>
</div>
{% endif %}

<!-- ══════════════════════════════════════
     6d — GAMMA EXPOSURE (GEX)
     ══════════════════════════════════════ -->
{% if gex_context and gex_context.signals %}
{% set gex_sig_color = {
    'PINNED':    '#0ea5e9',
    'NEUTRAL':   '#94a3b8',
    'AMPLIFIED': '#f87171'
} %}
{% set bias_color = {'BULLISH': '#4ade80', 'BEARISH': '#f87171', 'NEUTRAL': '#94a3b8'} %}
<h2>Gamma Exposure <span style="font-size:13px;font-weight:400;color:#94a3b8;">(dealer hedging flows · gamma flip · max pain)</span></h2>
<div class="card" style="border-left: 4px solid #0ea5e9;">
  <p style="color:#cbd5e1;font-size:13px;margin:0 0 14px 0;">{{ gex_context.summary }}</p>
  <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;font-size:12px;">
      <thead>
        <tr style="color:#64748b;text-transform:uppercase;font-size:10px;letter-spacing:.5px;">
          <th style="text-align:left;padding:5px 8px;">Ticker</th>
          <th style="text-align:center;padding:5px 8px;">Signal</th>
          <th style="text-align:right;padding:5px 8px;">Net GEX ($B)</th>
          <th style="text-align:right;padding:5px 8px;">Gamma Flip</th>
          <th style="text-align:right;padding:5px 8px;">Max Pain</th>
          <th style="text-align:center;padding:5px 8px;">Pain Bias</th>
          <th style="text-align:right;padding:5px 8px;">Pain Score</th>
          <th style="text-align:right;padding:5px 8px;">Exp. Move</th>
          <th style="text-align:right;padding:5px 8px;">OI Skew</th>
        </tr>
      </thead>
      <tbody>
      {% for s in gex_context.signals %}
      {% set sig_col = gex_sig_color.get(s.gex_signal, '#94a3b8') %}
      {% set tsig = signals_by_ticker.get(s.ticker) %}
      {% set mp_score = tsig.max_pain_score if tsig else 0 %}
        <tr style="border-top:1px solid #1e293b;">
          <td style="padding:6px 8px;color:#e2e8f0;font-weight:600;">{{ s.ticker }}</td>
          <td style="padding:6px 8px;text-align:center;">
            <span style="background:#1e293b;color:{{ sig_col }};border:1px solid {{ sig_col }};
                         border-radius:4px;padding:2px 7px;font-size:10px;white-space:nowrap;">
              {{ s.gex_signal }}
            </span>
          </td>
          <td style="padding:6px 8px;text-align:right;color:{{ '#4ade80' if s.net_gex_bn >= 0 else '#f87171' }};font-weight:700;">
            {{ "%+.2f"|format(s.net_gex_bn) }}B
          </td>
          <td style="padding:6px 8px;text-align:right;color:#94a3b8;">
            {% if s.gamma_flip is not none %}${{ "%.2f"|format(s.gamma_flip) }}{% else %}&mdash;{% endif %}
          </td>
          <td style="padding:6px 8px;text-align:right;color:#94a3b8;">
            {% if s.max_pain is not none %}${{ "%.2f"|format(s.max_pain) }}{% else %}&mdash;{% endif %}
          </td>
          <td style="padding:6px 8px;text-align:center;color:{{ bias_color.get(s.max_pain_bias, '#94a3b8') }};font-weight:700;">
            {{ '▲' if s.max_pain_bias == 'BULLISH' else ('▼' if s.max_pain_bias == 'BEARISH' else '→') }}
            {{ s.max_pain_bias }}
          </td>
          <td style="padding:6px 8px;text-align:right;font-weight:700;
                     color:{{ '#4ade80' if mp_score > 0.05 else ('#f87171' if mp_score < -0.05 else '#64748b') }};">
            {% if mp_score %}{{ "%+.2f"|format(mp_score) }}{% else %}&mdash;{% endif %}
          </td>
          <td style="padding:6px 8px;text-align:right;color:#e2e8f0;">
            ±{{ "%.1f"|format(s.expected_move_pct) }}%
          </td>
          <td style="padding:6px 8px;text-align:right;font-weight:700;
                     color:{{ '#4ade80' if s.oi_skew > 0.1 else ('#f87171' if s.oi_skew < -0.1 else '#64748b') }};">
            {% if s.oi_skew %}{{ "%+.3f"|format(s.oi_skew) }}{% else %}&mdash;{% endif %}
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
  </div>
  <p style="color:#475569;font-size:11px;margin:12px 0 0 0;">
    PINNED: positive GEX → dealers sell rallies/buy dips → price stays anchored near gamma flip.
    AMPLIFIED: negative GEX → dealers amplify moves → expect wider swings.
    Gamma flip = price where dealer hedging switches direction.
    Pain Score: expiry-weighted gravity signal — fades beyond 14 days, strongest within 4 days of expiry.
    OI Skew: dollar-distance-weighted call vs put OI lean (+1 = all calls, −1 = all puts). Directional (not contrarian).
  </p>
</div>
{% endif %}

<!-- ══════════════════════════════════════
     6 — SMART MONEY SIGNALS
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
     7 — PORTFOLIO PERFORMANCE
     ══════════════════════════════════════ -->
{% if perf %}
<h2>Portfolio Performance</h2>
{% if equity_png %}
<img class="chart-img" src="cid:equity_chart" alt="Equity Curve">
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
<h3 style="color:#94a3b8;font-size:13px;margin:16px 0 6px;">Open positions</h3>
<table class="pt">
  <thead>
    <tr>
      <th>Ticker</th><th>Action</th><th>Date</th><th>Entry</th>
      <th>Current</th><th>P&amp;L</th><th>Days</th>
    </tr>
  </thead>
  <tbody>
    {% for t in perf.open_trades %}
    <tr>
      <td><strong>{{ t.ticker }}</strong></td>
      <td>{{ t.action }}</td>
      <td style="color:#94a3b8;">{{ t.entry_date }}</td>
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
{% if perf.closed_trades %}
<h3 style="color:#94a3b8;font-size:13px;margin:16px 0 6px;">Closed positions</h3>
<table class="pt">
  <thead>
    <tr>
      <th>Ticker</th><th>Action</th><th>Date</th><th>Entry</th>
      <th>Exit</th><th>P&amp;L</th><th>Days</th>
    </tr>
  </thead>
  <tbody>
    {% for t in perf.closed_trades %}
    <tr>
      <td><strong>{{ t.ticker }}</strong></td>
      <td>{{ t.action }}</td>
      <td style="color:#94a3b8;">{{ t.entry_date }}</td>
      <td>${{ "%.2f"|format(t.entry_price) }}</td>
      <td>${{ "%.2f"|format(t.exit_price) }}</td>
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
    signals_by_ticker: Optional[dict] = None,
) -> tuple[dict, Optional[str], Optional[str]]:
    charts: dict[str, str] = {}
    for rec in actionable:
        fig = build_stock_chart(rec.ticker, rec)
        b64 = fig_to_png_b64(fig, width=1100, height=780)
        if b64:
            charts[rec.ticker] = b64
            logger.debug(f"[email] Chart rendered for {rec.ticker}")
    overview_fig = build_signals_overview(all_recommendations or actionable, signals_by_ticker)
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
    macro_context=None,     # Optional[MacroContext]    — avoid circular import
    cot_context=None,       # Optional[COTContext]      — avoid circular import
    ipo_context=None,       # Optional[IPOContext]      — avoid circular import
    vix_context=None,       # Optional[VIXContext]      — avoid circular import
    put_call_context=None,  # Optional[PutCallContext]  — avoid circular import
    tick_context=None,      # Optional[TICKContext]     — avoid circular import
    earnings_context=None,  # Optional[EarningsContext] — avoid circular import
    gex_context=None,       # Optional[GEXContext]      — avoid circular import
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
    use_tech    = settings.enable_technical_analysis and settings.enable_fetch_data
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
                )[:settings.smart_money_top_tickers]
            )

    # ── Analyst ratings articles (dedicated section) ─────────────────────
    analyst_articles: List[NewsArticle] = sorted(
        [a for a in (articles or []) if a.source == "Analyst Ratings"],
        key=lambda a: a.published_at, reverse=True,
    )

    # ── EPS surprise articles ─────────────────────────────────────────────
    eps_articles: List[NewsArticle] = sorted(
        [a for a in (articles or []) if a.source == "Earnings/EPS"],
        key=lambda a: a.published_at, reverse=True,
    )

    # ── Alternative data articles (Google Trends + Reddit + Short Interest) ──
    alt_data_articles: List[NewsArticle] = sorted(
        [a for a in (articles or []) if a.source in ("Google Trends", "Reddit/WSB", "Short Interest")],
        key=lambda a: a.published_at, reverse=True,
    )

    # ── Charts ─────────────────────────────────────────────────────────────
    # The overview chart is always generated — it's a single cheap chart that
    # shows method-breakdown stacked bars for every recommendation.
    # Per-ticker stock charts and the equity curve are only built when ENABLE_CHARTS=true.
    charts: dict = {}
    equity_png: Optional[str] = None
    try:
        from src.charts.builder import PLOTLY_AVAILABLE
        if PLOTLY_AVAILABLE:
            overview_fig = build_signals_overview(all_recs, signals_by_ticker)
            overview_png = fig_to_png_b64(overview_fig, width=1100, height=None)
        else:
            overview_png = None
    except Exception as e:
        logger.debug(f"[email] Overview chart skipped: {e}")
        overview_png = None

    if settings.enable_charts:
        logger.info("[email] Rendering per-ticker chart images...")
        try:
            charts, _, equity_png = _build_chart_pngs(
                recommendations, all_recs, performance, signals_by_ticker
            )
            logger.info(
                f"[email] {len(charts)} chart image(s) embedded"
                if charts else "[email] No chart images (kaleido not available)"
            )
        except Exception as e:
            logger.warning(f"[email] Per-ticker charts failed: {e}")

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
        use_put_call=settings.enable_put_call,
        enable_gex=settings.enable_gex,
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
        # analyst ratings / EPS surprises / alternative data
        analyst_articles=analyst_articles,
        eps_articles=eps_articles,
        alt_data_articles=alt_data_articles,
        # macro context (FRED)
        macro_context=macro_context,
        # COT context (CFTC)
        cot_context=cot_context,
        # IPO pipeline (SEC S-1/S-11)
        ipo_context=ipo_context,
        # VIX volatility regime
        vix_context=vix_context,
        # Put/Call ratio (CBOE + per-ticker)
        put_call_context=put_call_context,
        # NYSE TICK breadth
        tick_context=tick_context,
        # Earnings calendar
        earnings_context=earnings_context,
        # Gamma Exposure (GEX)
        gex_context=gex_context,
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
            if settings.enable_put_call and sig.put_call_score != 0:
                lines.append(f"Put/call score:  {sig.put_call_score:+.2f}")
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
        # Outer wrapper: multipart/related so inline CID images are recognised
        msg = MIMEMultipart("related")
        msg["Subject"] = subject
        msg["From"]    = settings.smtp_user
        msg["To"]      = ", ".join(settings.recipients_list)

        # Inner alternative part (plain-text + HTML)
        alt = MIMEMultipart("alternative")
        alt.attach(MIMEText(text_body, "plain"))
        alt.attach(MIMEText(html_body, "html"))
        msg.attach(alt)

        # Attach each chart image with its Content-ID so Gmail renders it
        def _attach_png(cid: str, b64_data: str) -> None:
            img_bytes = base64.b64decode(b64_data)
            part = MIMEImage(img_bytes, "png")
            part["Content-ID"] = f"<{cid}>"
            part["Content-Disposition"] = "inline"
            msg.attach(part)

        if overview_png:
            _attach_png("overview_chart", overview_png)
        if equity_png:
            _attach_png("equity_chart", equity_png)
        for ticker, b64 in (charts or {}).items():
            _attach_png(f"chart_{ticker}", b64)

        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(settings.smtp_user, settings.recipients_list, msg.as_string())

        logger.info(f"Email sent to {settings.recipients_list}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False
