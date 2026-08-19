"""Rank-era synthesis prompt (2026-08-14, SYNTHESIS_PROMPT_VERSION
"2026-08-14-rank-process").

What the rewrite added and why, pinned here:

* a <selection_process> block (shared by all three arms, cache-prefix-safe)
  describing the MEASURED process the recommendations feed — cross-sectional
  rank selection on the tradeable pool, the next-swing pivot objective and its
  ~2-day edge peak, mechanical fast exits, asymmetric buy/sell selectivity,
  the anti-chase gate, and the earnings blackout;
* percentile STANDINGS on the camp scores. Camp scores are comparable only
  within a run — the weighted combine and the ML stackers live on ~5x
  different scales, so on an ML-arm run the day's strongest view read as
  `buy_score=0.08` and the dual-case rules told the model two weak convictions
  mean HOLD. The '(pNN today)' midrank percentile among ALL scored tickers is
  engine-invariant. Guarded to >=30 scored so the hold-review's tiny universe
  never shows degenerate anchors;
* a prompt-era stamp (`synthesis_prompt_version` in the run meta) so arm evals
  and LLM-confidence calibrations can split eras after the fact.

All fakes, no network — same harness as test_dual_case_synthesis.py.
"""

import json

import pytest

from config.settings import settings
import src.analysis.claude_analyst as ca
import src.signals.aggregator as agg
from src.models import TickerSignal


@pytest.fixture
def captured(monkeypatch):
    box = {}

    def fake(prompt, **kw):
        box["prompt"] = prompt
        return json.dumps([{"ticker": "T00", "type": "STOCK", "direction": "BULLISH",
                            "action": "WATCH", "confidence": 0.5, "rationale": "x"}])

    for name in ("_call_claude_analyst", "_call_deepseek_analyst", "_call_qwen_analyst"):
        monkeypatch.setattr(ca, name, fake)
    monkeypatch.setattr(settings, "llm_ab_synthesis_models", "deepseek-v4-flash")
    monkeypatch.setattr(agg, "side_filtered_methods", lambda side: frozenset())
    monkeypatch.setattr(agg, "winrate_filtered_methods", lambda: frozenset())
    return box


def _sig(i, buy=0.30, sell=0.05, conf=0.5):
    return TickerSignal(ticker=f"T{i:02d}", direction="BULLISH", confidence=conf,
                        combined_score=buy - sell, combined_buy_score=buy,
                        combined_sell_score=sell, sources_agreeing=2,
                        sentiment_score=0.0, technical_score=0.0, rationale="r")


def _universe(n=40):
    """n signals with distinct camp scores; T00 is the strongest buy view."""
    return [_sig(i, buy=0.40 - 0.008 * i, sell=0.02 + 0.001 * i) for i in range(n)]


# ── the shared <selection_process> block ────────────────────────────────────

@pytest.mark.parametrize("arm_kwargs", [
    {},                                              # sighted
    {"blind_synthesis": True},                       # blind
    {"dual_case": True},                             # dual-case
])
def test_selection_process_block_in_every_arm(captured, arm_kwargs):
    ca.generate_recommendations(_universe(), **arm_kwargs)
    p = captured["prompt"]
    assert "<selection_process>" in p and "</selection_process>" in p
    # The claims that reorient the model, one probe each.
    assert "WITHIN-RUN RANK" in p                       # cross-sectional selection
    assert "next swing extreme" in p                    # the pivot objective
    assert "peaks around day 2" in p                    # measured edge decay
    assert "bottom 5%" in p                             # asymmetric selectivity
    assert "12%" in p                                   # anti-chase gate
    assert "2 days of a scheduled report" in p          # earnings blackout


def test_process_block_precedes_cache_sentinel_position(captured, monkeypatch):
    """The block is static text in the CACHED prefix: within one tick the three
    arms + hold reviews must reuse the same prefix, so it may not vary by arm."""
    ca.generate_recommendations(_universe())
    sighted = captured["prompt"]
    ca.generate_recommendations(_universe(), blind_synthesis=True)
    blind = captured["prompt"]

    def prefix(p):
        return p[:p.index("Today's date:")]

    a, b = prefix(sighted), prefix(blind)
    assert "<selection_process>" in a
    assert a == b, "cached prefix differs between arms — prefix cache broken"


# ── percentile standings ────────────────────────────────────────────────────

def test_sighted_lines_carry_standing(captured):
    ca.generate_recommendations(_universe())
    p = captured["prompt"]
    # The strongest buy view stands at the top of the pool.
    assert "buy_score=0.40 (p99 today)" in p or "buy_score=0.40 (p98 today)" in p
    assert "(pNN today)" in p        # the instruction explains the annotation


def test_dual_case_lines_carry_standing_on_both_camps(captured):
    ca.generate_recommendations(_universe(), dual_case=True)
    p = captured["prompt"]
    assert "BULL CASE conviction=0.40 (p9" in p
    assert "BEAR CASE conviction=" in p and "today)" in p


def test_blind_lines_carry_no_camp_scores(captured):
    ca.generate_recommendations(_universe(), blind_synthesis=True)
    p = captured["prompt"]
    assert "buy_score=" not in p
    assert "BULL CASE" not in p


def test_small_universe_shows_no_standing(captured):
    """Hold-review calls pass a handful of held tickers — percentiles over a
    5-name pool are noise dressed as calibration, so they are suppressed."""
    ca.generate_recommendations(_universe(5))
    assert "today)" not in captured["prompt"].split("<signals>")[1].split("</signals>")[0]


def test_ml_scale_view_is_rescued_by_standing(captured):
    """The defect this exists for: on an ML-arm run the day's strongest view is
    an absolute 0.08. The standing annotation must mark it as top-of-pool."""
    sigs = [_sig(i, buy=0.005 + 0.002 * i, sell=0.001) for i in range(40)]
    ca.generate_recommendations(sigs)
    p = captured["prompt"]
    assert "buy_score=0.08 (p99 today)" in p or "buy_score=0.08 (p98 today)" in p


# ── era stamp ───────────────────────────────────────────────────────────────

def test_prompt_version_exists_and_is_stamped_into_the_run_meta():
    import inspect
    from src import pipeline

    # Pinned as a LITERAL on purpose: the failure this produces on any prompt
    # edit is the point — it forces a conscious version bump, without which the
    # eval surfaces silently pool two prompt eras. Update it WITH the edit.
    assert ca.SYNTHESIS_PROMPT_VERSION == "2026-08-19-confidence-placement"
    src = inspect.getsource(pipeline)
    assert '"synthesis_prompt_version"' in src
    assert "SYNTHESIS_PROMPT_VERSION" in src


# ── held-position review addendum ───────────────────────────────────────────

def test_hold_review_judges_the_remaining_move(captured):
    pos = [{"ticker": "T00", "action": "BUY", "entry_date": "2026-08-10",
            "days_held": 3, "entry_price": 10.0, "current_price": 11.0,
            "return_pct": 10.0}]
    ca.generate_recommendations(_universe(), open_positions=pos)
    p = captured["prompt"]
    assert "Judge the REMAINING move" in p
    assert "next swing extreme" in p
