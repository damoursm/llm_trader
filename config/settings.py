from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import List, Optional

# Absolute path to <project root>/.env so settings load no matter the current
# working directory (e.g. when launched via Windows Task Scheduler from System32,
# or `python <abs path>/main.py` from your home dir). config/settings.py →
# parent (config/) → parent (project root) → .env
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    # Claude API
    anthropic_api_key: str

    # Model selection
    # Options: "claude-haiku-4-5-20251001" (fast/cheap), "claude-opus-4-6" (highest quality)
    analyst_model: str = "claude-haiku-4-5-20251001"

    # LLM A/B split — probability that a run picks the Anthropic engine as its
    # PRIMARY (the other provider stays as fallback). Applied independently to
    # synthesis (Claude analyst vs DeepSeek) and sentiment (Haiku vs DeepSeek),
    # re-flipped once per run. 0.5 = even split for side-by-side evaluation in
    # the dashboard's per-LLM rows; 1.0 = always Anthropic-first (legacy
    # behavior); 0.0 = always DeepSeek-first.
    llm_ab_anthropic_share: float = 0.5

    # SENTIMENT engine policy. When False (default), per-ticker news sentiment is
    # scored by DeepSeek ONLY — Claude/Haiku is never called for sentiment (not even
    # as fallback or via a hold-review engine pin); Claude is reserved for SYNTHESIS.
    # Set True to restore the Haiku ⇄ DeepSeek A/B on sentiment (governed by
    # llm_ab_anthropic_share). Synthesis routing is unaffected either way.
    enable_claude_sentiment: bool = False

    # SYNTHESIS-only N-way model bake-off. When set to a comma-separated list of
    # model ids, the per-run synthesis engine is picked UNIFORMLY from this pool
    # (equal split) instead of the binary llm_ab_anthropic_share flip — so 3+
    # models (e.g. Haiku, Opus 4.8, DeepSeek) accumulate comparable samples and
    # each shows as its own row in the dashboard's per-LLM evaluation. The chosen
    # model's provider is inferred (deepseek* → DeepSeek, else Anthropic); the
    # OTHER provider is the error fallback. Empty = legacy binary behavior above.
    # DeepSeek arms may carry a "-thinking" suffix (logical id → API model + reasoning
    # mode, decoded by claude_analyst._deepseek_spec). Sentiment is unaffected (stays
    # Haiku ⇄ DeepSeek flash non-thinking via llm_ab_anthropic_share). Example (2026-07-11 —
    # DeepSeek's reasoning family is V4, not R1):
    #   LLM_AB_SYNTHESIS_MODELS=claude-haiku-4-5-20251001,claude-opus-4-8,deepseek-v4-flash-thinking
    # Current pool (2026-07-22, flat 2-way — see .env): deepseek-v4-flash,
    # qwen/qwen3.7-plus-thinking (~50% each). Consolidated onto ONE DeepSeek model
    # after the bake-off found the arms statistically indistinguishable: eleven
    # paired tests (same ticker, same day) returned no significant difference and
    # the models disagreed on direction 0–6 times out of 25–170 shared calls, so
    # the choice is cost/latency, not accuracy. Dropped deepseek-v4-pro-thinking
    # (+233s per tick, ~3x tokens, repeated tickers in 26% of responses) and
    # deepseek-v4-flash-thinking (see llm_max_thinking below). Qwen moved from
    # qwen3.7-max to qwen3.7-plus — same capabilities on this route (reasoning,
    # response_format, seed, 1M ctx, 65K out) at ~4.6x cheaper input / ~3.5x
    # cheaper output.
    llm_ab_synthesis_models: str = ""

    # Same-engine TRANSIENT retry before spending a fallback hop on a DIFFERENT
    # provider (2026-07-22). A read-timeout or connection drop mid-stream is often
    # just a blip — DeepSeek/Qwen/Anthropic all occasionally stall or reset a
    # single request while otherwise fine (observed: a DeepSeek ReadTimeout in the
    # same tick DeepSeek answered ~25 other calls) — so it's worth retrying the
    # SAME engine once before moving to a different provider, which may itself be
    # out of credits right now. A HARD failure (bad key, bad request, insufficient
    # credits/balance) is never retried — it will just fail again immediately.
    # Applies to every synthesis attempt, including the opener-pinned hold-review
    # call (still the same engine either way, so Fix #2's same-engine invariant is
    # untouched). 0 = off (fail straight to the next engine/attempt, pre-2026-07-22
    # behavior).
    llm_transient_retries: int = 1
    llm_transient_retry_wait_seconds: float = 3.0

    # Anthropic prompt caching (cache_control) on the SYNTHESIS prompt. The large
    # persona + macro-context prefix is identical across the main call and the
    # opener-pinned hold-review calls within a tick, so caching it (5-min ephemeral)
    # makes the 2nd+ same-model call within a tick read the prefix at ~10% cost
    # instead of full price. Output is unchanged (same prompt, just billed cheaper).
    # A cache WRITE costs 1.25× the prefix tokens, a READ 0.1×, so this is a net win
    # only when prefixes are reused — measure via the "[claude] prompt cache:" log
    # and flip off if your model mix yields few reads. Sentiment is NOT cached (its
    # prefix is below Haiku's 2048-token cache minimum). Default on.
    enable_prompt_caching: bool = True

    # Held-positions prompt A/B — probability that a run includes the
    # <open_positions_context> block (the system's current holdings + a
    # (held-positions prompt) — the `open_positions_prompt_share` A/B was RETIRED
    # 2026-08-14 and the prompt is now unconditional (pipeline Step 4.6). ON led
    # on both metrics over 401 closed trades but needed ~1,500 per arm to reach
    # significance (~15 more months); the leading arm was adopted rather than
    # keep spending half the runs on an experiment that could not conclude.
    # The setting is deleted rather than pinned to 1.0 so it cannot read as a
    # live knob — `tests/test_inert_settings.py` would fail it as unread anyway.

    # Massive / Polygon.io market data — primary source for equity/ETF price + all
    # OHLCV timeframes (daily bulk via grouped-daily; 30-min intraday via aggregates,
    # REAL-TIME on the Stocks Advanced plan). Key: https://polygon.io (works globally,
    # no credit card for the free tier). If absent, the pipeline falls back to yfinance.
    polygon_api_key: str = ""

    # Company fundamentals (TTM valuation / profitability / leverage ratios) from the
    # Massive/Polygon financials & ratios endpoint — requires the Stocks Advanced plan
    # (or the ratios add-on). Fed into the LLM synthesis prompt as a quality/valuation
    # overlay. Key-gated + fail-graceful: inert (no block) without the entitlement.
    enable_fundamentals: bool = True
    # Per-ticker fundamentals ENRICHMENT (Massive short-interest + short-volume +
    # income-statement margin/YoY-growth) on top of the batched ratios — ~3 extra
    # calls/ticker, so capped to the first N (watchlist + early discovery). 0 = off.
    fundamentals_enrich_max_tickers: int = 50

    # Ticker-events watch (Massive corporate ticker-events: symbol/name changes,
    # delistings) — surfaces recent events on held + watchlist names as material
    # NewsArticles so a rename/delisting can't silently strand a position.
    enable_ticker_events: bool = True
    ticker_events_lookback_days: int = 45

    # Corporate actions (dividends + splits) from Massive/Polygon — upcoming ex-dividend
    # dates + recent/upcoming splits, fed to synthesis as a WHEN/mechanics overlay (§29).
    enable_corporate_actions: bool = True
    corp_actions_div_lookahead_days: int = 14   # surface ex-dividends within this many days
    corp_actions_split_window_days: int = 30    # surface splits within ± this many days
    # f_dividend EVENT window (2026-08-16 rework): score only dividends DECLARED
    # within this many trailing days, decaying to 0 across it — the measured
    # drift is confined to the first ~5 sessions after declaration. Discovery is
    # one market-wide declaration-window query (full-universe coverage); history
    # is fetched per fresh declarer only, capped below.
    corp_actions_div_event_window_days: int = 10
    corp_actions_div_max_event_fetches: int = 40
    # Additive (NOT normalised-pool) overlay weight for the directional corporate-action
    # factors (f_split + f_dividend) on combined_score — event-driven so it never dampens
    # the ~95% of tickers with no action. Placeholder; tune once IC data accrues.
    corp_action_factor_weight: float = 0.10
    # Additive-overlay weight for the 4 Massive fundamental factors (value/quality/
    # growth/short-squeeze), folded into combined_score 2026-06-24. Applied OUTSIDE
    # the normalised pool (like corp_action_factor_weight) so the capped/sparse
    # fundamentals nudge the combine without dampening non-enriched tickers. Small +
    # tunable; they remain forward-IC-validated in the dashboard Signal-IC table.
    fundamental_factor_weight: float = 0.08

    # Related-company peer discovery (Massive related-companies graph) — widens the
    # universe with peers of the watchlist + held names (liquidity-gated in Step 0).
    enable_related_discovery: bool = True
    related_discovery_max: int = 25

    # Massive/Polygon server-side technical indicators (RSI + MACD) scored as the
    # `massive` method, run ALONGSIDE our own `tech` for head-to-head dashboard
    # comparison. 2 API calls/ticker, so capped per tick. Set the weight via
    # aggregator `_BASE_WEIGHTS["massive"]`; off → the method scores 0 (no effect).
    enable_massive_tech: bool = True
    massive_tech_max_tickers: int = 0    # 0 = every ticker. Now a WEIGHTED member of
    # combined_score (promoted 2026-06-24), so it must score every ticker — a positive
    # cap would leave capped-out tickers with massive=0 while still reserving its weight
    # in the normalised pool, dampening their combined_score. Keep 0 unless reverting.

    # DeepSeek API (used for low-reasoning tasks)
    deepseek_api_key: str = ""

    # Qwen (Alibaba Cloud Model Studio / DashScope, OpenAI-compatible). qwen3.7-max
    # (May 2026): 1M context, ~66K max output, thinking via extra_body
    # {"enable_thinking": bool}. Keys are REGION-BOUND: an international-console key
    # only works on the intl endpoint below; a mainland-console key needs
    # https://dashscope.aliyuncs.com/compatible-mode/v1 instead.
    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    # Model id for the active route: bare "qwen3.7-max" on DashScope direct,
    # "qwen/qwen3.7-max" via OpenRouter (route derived from qwen_base_url —
    # see src/analysis/qwen_api.py for the dialect differences).
    qwen_model: str = "qwen3.7-max"
    # PRIMARY OpenAI-compatible LLM provider ("deepseek" | "qwen") for every
    # non-bake-off LLM call: per-ticker sentiment, the macro-news classifier, and
    # the pinned hold-review COERCION target. "qwen" would route ALL LLM calls to
    # Qwen and coerce every opener pin to qwen; "deepseek" (the 2026-07-13 cost tune)
    # applies NO blanket coercion — hold-review pins honor the opener engine (Fix #2),
    # macro-news classifies DeepSeek-only, and sentiment follows the 90/10 DeepSeek/
    # Qwen per-run split (sentiment_qwen_share) rather than an all-qwen override.
    # Synthesis routing is independent (llm_ab_synthesis_models — currently 50/50).
    llm_primary_provider: str = "deepseek"

    # SENTIMENT engine split (2026-07-13 cost tune): probability a given run scores
    # per-ticker sentiment with Qwen (the pricier engine) vs DeepSeek-flash. A per-RUN
    # flip (not per-call) so each engine accrues whole-run samples for the dashboard's
    # per-LLM eval and the run's dominant sentiment model attributes cleanly. 0.10 ⇒
    # ~10% Qwen / ~90% DeepSeek; the non-primary engine is the per-call error fallback.
    # Ignored when enable_claude_sentiment runs its own anthropic A/B ahead of this,
    # and only fires when a Qwen key is configured. 0.0 = DeepSeek-only sentiment.
    sentiment_qwen_share: float = 0.0

    # MAXIMUM-THINKING policy (2026-07-12 directive: "maximum thinking everywhere").
    # ON → every LLM call reasons at the model's MAX budget:
    #   • Qwen — enable_thinking=True with thinking_budget OMITTED (DashScope defaults
    #     it to the model's maximum chain-of-thought; Qwen bills reasoning tokens
    #     SEPARATELY from max_tokens, so the answer cap is untouched);
    #   • DeepSeek (fallback) — reasoning mode ON, with a raised max_tokens because
    #     DeepSeek counts thinking AGAINST max_tokens;
    #   • Anthropic (if ever pooled) — adaptive thinking + output_config effort="max".
    # This flips the two paths that were deliberately non-thinking for cost —
    # per-ticker sentiment and the macro-news classifier — to thinking too, so
    # reasoning is billed on every call (notably the high-volume sentiment path).
    # OFF → the prior cost-tuned mix (decisions think; bulk sentiment/macro don't).
    #
    # TURNED OFF 2026-07-22 (.env). The flag OVERRIDES the model id — it forced
    # `thinking=True` even on an arm named `deepseek-v4-flash`, so the "cheap flash"
    # arm silently billed reasoning tokens AND recorded provenance as plain
    # `deepseek-v4-flash`, colliding with the 2,685 genuinely non-thinking rows in
    # the history. Off restores a true non-thinking flash. Justified by the bake-off:
    # flash vs flash-thinking agreed on direction 97.6% of the time with no
    # significant accuracy difference at any horizon, so the reasoning tokens were
    # buying agreement with the cheaper setting. An arm that should reason must now
    # say so in its id (the `-thinking` suffix, e.g. qwen/qwen3.7-plus-thinking) —
    # which is also what makes the dashboard's per-model rows honest.
    llm_max_thinking: bool = False

    # BLIND-SYNTHESIS A/B (2026-07-12). The agreement eval found 96% of LLM calls
    # simply ECHO the aggregator's verdict (the prompt leads every ticker line with
    # direction= / combined_confidence= / sources_agreeing=, and §3 says "Trust
    # it") — so the LLM is an expensive rubber stamp, not a second opinion, and it
    # degrades performance on the way through (funnel: −2.06% in → −3.18% out).
    # Share of runs whose MAIN synthesis goes BLIND: the aggregate verdict fields,
    # the HORIZON MODEL / EXPECTED MOVE opinion lines, and the "trust the
    # pre-computed confidence" instructions are removed, so the LLM must form its
    # own direction/confidence from the RAW per-method evidence. The flip is
    # stamped per run (gate_diag.blind_synthesis) and per trade
    # (entry_blind_synthesis) → dashboard "Entry eval · blind-synthesis ON/OFF"
    # rows accumulate the outcome comparison (same idiom as
    # open_positions_prompt_share). Hold-review calls are NEVER blinded (stable
    # exit governance; only the entry side is under test). 0.0 = off, 1.0 = always.
    # RETIRED 2026-08-16 → 0.0 (see the arm-decommission note on
    # enable_shadow_arms below). Raise it to re-open the blind/sighted axis.
    blind_synthesis_share: float = 0.0

    # ── Dual-case synthesis (2026-07-25) ─────────────────────────────────────
    # Presents the BULL CASE and the BEAR CASE side by side, each built from its
    # OWN camp, and lets the LLM weigh them. Fixes a real incoherence rather
    # than chasing the echo rate: the per-side win-rate filter means the two
    # camps hold DIFFERENT valid method sets (news and iv_rank are currently
    # excluded from the bullish camp), but the prompt filtered methods GLOBALLY
    # only — so the model saw `news` while considering a BUY, on a method the
    # system had already decided must not inform buying. Here each per-method
    # line is gated by the side its own score points to, exactly as the combine
    # and coherence now gate it.
    # Measured expectations, so this is not oversold: the blind A/B moved the
    # echo rate only 94.5% -> 91.4%, i.e. the LLM echoes because it reads the
    # same method scores, NOT because it sees the verdict — so do not expect
    # this to change direction-picking much. What it protects is the cohort
    # where the LLM demonstrably adds value: when the aggregator is NEUTRAL and
    # the model supplies the direction, its BUY calls returned +0.21% / 66% win
    # versus -3.19% / 33% when echoing a bullish aggregator (same side, so the
    # long/short asymmetry is controlled for).
    # SUPERSEDES the blind/sighted axis when active (a symmetric two-case
    # presentation is inherently verdict-free), so the arms stay at three —
    # dual / blind / sighted — instead of fragmenting into four cells.
    # RETIRED 2026-08-16 → 0.0 (see the arm-decommission note on
    # enable_shadow_arms below). Raise it to re-open the dual-case arm.
    dual_case_synthesis_share: float = 0.0

    # ── Shadow synthesis arms (per-ticker arm bake-off, 2026-07-25) ────────
    # Ask EVERY prompt arm about EVERY ticker each tick: the live arm drives the
    # run, the other two are shadow calls nobody acts on. Without this an arm is
    # only observable on the runs where its coin came up, so the arms are
    # compared over different ticker-days -- exactly the unpaired design that
    # made Qwen look best and pro-thinking look broken in the 2026-07-22
    # bake-off (both pure window artifacts; every difference vanished paired).
    # Cost: two extra SYNTHESIS calls per tick. Synthesis is one call for the
    # whole top-N ticker set (unlike per-ticker sentiment, the real cost
    # driver), and shadow calls run off the critical path, so orders never wait.
    #
    # ── ARMS DECOMMISSIONED 2026-08-16 (user directive) — SIGHTED ONLY ───────
    # The bake-off ran 2026-07-25 → 2026-08-16 and ANSWERED: on the PIVOT basis
    # (the decision basis) with day-clustered statistics over 16 settled days /
    # 22,534 labeled arm rows, NO arm is distinguishable from another — the
    # paired disagreement subset (the only place a prompt changes a decision)
    # gives |day-t| < 1 for all three pairs, the nominal winner wins just 38-44%
    # of DAYS, and dual-vs-blind FLIPS SIGN between row- and day-weighting.
    # Unpaired means (sighted +0.071%/day, t 0.74) are not evidence. So the 3x
    # synthesis cost was buying nothing and all three knobs are now off:
    # every run is SIGHTED, no shadow calls. Sighted was kept on the
    # tiebreakers, not on P&L: the shortest per-ticker block (dual renders BULL
    # AND BEAR lines for every ticker) and the least anti-informative
    # confidence (+0.015 vs dual −0.033), which matters because confidence
    # feeds sizing. The dual/blind rendering code and `arm_recommendations`
    # history are KEPT (same convention as the retired Qwen/Claude sentiment
    # split): flipping any share back above 0 re-opens the experiment, and the
    # accrued rows stay analysable via `python -m src.analysis.arm_eval`.
    # See memory/shadow-arms-2026-07.md for the full verdict.
    enable_shadow_arms: bool = False
    # Hard bound on the join at persist time. A shadow arm that overruns is
    # abandoned (live arm still persisted) rather than delaying the run.
    shadow_arms_join_timeout_seconds: float = 300.0

    # Two-tick confirmation on the NOISY LLM exits (llm_signal_flipped /
    # llm_confidence_loss): with max thinking the hold-review is non-deterministic,
    # and llm_confidence_loss was measured leaving ~+7% @5d on the table — so a
    # single-review breach no longer closes. The breach ARMS a pending marker on
    # the trade; only a SECOND consecutive breaching review closes (any LLM-exit
    # reason counts as the confirmation). A review that says hold clears the
    # marker; a tick with NO review leaves it armed (absence is not evidence).
    # Mechanical/trailing/macro/horizon exits are NOT gated (they are not
    # review-noise-driven). Off = the pre-2026-07-12 single-review behavior.
    enable_llm_exit_confirmation: bool = True

    # The IN-WINDOW confidence-degradation exit (`llm_confidence_loss`): while a
    # position is still inside its target-horizon window, close it if the opener's
    # same-direction conviction falls below the entry-relative floor.
    #
    # **OFF SINCE 2026-08-03 — it was MEASURED TO EXIT TOO EARLY.** Per-reason
    # POST-EXIT forward returns (`src/analysis/exit_forward.py`, the "persistently
    # positive mean ⇒ the rule exits too early" test) single this rule out as the
    # only one the evidence condemns:
    #     rule                  n    fwd+1d   fwd+5d
    #     trailing_stop        93    -8.13   -15.68   (excellent — dodges crashes)
    #     horizon_expired      98    -0.49    -0.93   (correct)
    #     llm_signal_flipped   66    +0.28    +0.44   (~neutral)
    #     llm_confidence_loss  45    +1.50    +2.24   ← money left on the table
    # (+1.50% mean / +0.80% median next day, 62% of exits kept going our way.)
    # It was already patched twice for the same fault — the two-tick confirmation
    # above, and `signal_decay_confidence_floor_relative` 0.65→0.55 — which is the
    # tell that the rule itself, not its tuning, is the problem: LLM confidence is
    # measured ~uninformative about forward returns, so its DRIFT is not evidence.
    #
    # Turning it off does NOT create zombie positions. This is the IN-WINDOW half
    # of a test whose PAST-WINDOW half (`horizon_expired`, same conviction test on
    # a ramped floor) is measured correct and still fires — so a decayed position
    # still closes, just at its horizon instead of before it, which is exactly what
    # the forward returns say to do. `trailing_stop`, `llm_signal_flipped`,
    # `macro_regime_exit` and `adverse_stop` are all untouched. Arm-cohort trades
    # additionally get `ml_exit` (see _ARM_ML_OWNED_EXITS).
    # Re-validate with `python -m src.analysis.exit_forward` before re-enabling.
    enable_llm_confidence_loss_exit: bool = False

    # BUY/SELL SPLIT COMBINE (2026-07-22 user directive). Each weighted method's
    # inversion-corrected view is decomposed into a BUY component max(0, eff) and a
    # SELL component max(0, −eff), each side is weight-averaged over the methods
    # holding that view ONLY (its own camp — abstainers and the opposing camp no
    # longer dilute it, unlike the old single normalised pool), and
    # combined_score = combined_buy − combined_sell (+ the additive overlays,
    # unchanged). Direction fires on the DIFFERENCE clearing this band:
    # diff >= +band → BULLISH, <= −band → BEARISH, else NEUTRAL. 0.15 was the old
    # single-pool band; recalibrated on the 11.3k-row panel it keeps a comparable —
    # and notably MORE BALANCED — mix (old: 32% bullish / 15% bearish; split:
    # 18% / 21%), directly countering the measured BUY-side output skew. The two
    # side scores are persisted per ticker (signals.combined_buy_score /
    # combined_sell_score) so each side's IC is monitored on the dashboard.
    buy_sell_diff_threshold: float = 0.15
    # ── Method score basis (2026-08-13 user directive) ─────────────────────────
    # "rank": the combine (and coherence / sources_agreeing / family votes)
    # consumes each method's CENTERED WITHIN-RUN RANK (+1 = the run's strongest
    # view, −1 = weakest, median ≈ 0 — the ML models' convention); zeros abstain,
    # and a method with < method_rank_min_views views this run gets WEIGHT 0
    # (raw scores stay visible; excluded from coherence/agreement/votes like a
    # win-rate-filtered method — no absolute fallback, 2026-08-13 directive).
    # Consumption-time only: every persisted score stays RAW (inversion
    # architecture). "absolute" reverts to the pre-directive combine.
    # CONFIDENCE_EPOCH 2026-08-14 marks the switch (|combined| scale changes).
    method_score_basis: str = "rank"
    method_rank_min_views: int = 5
    # Payoff-shaped rank mapping (2026-08-14 directive): each method's rank is
    # mapped through its own SHRUNK measured rank->pivot-payoff decile curve
    # (demeaned + normalized; src/signals/rank_shaping.py) instead of the
    # linear grid — an anti-predictive extreme (momentum's top decile) becomes
    # reversal signal. Identity fail-soft per method (thin evidence or a flat
    # curve keeps the plain centered rank).
    # Go-live thresholds for the rank basis (2026-08-14 quantile match over the
    # reconstructed ranked+shaped combine vs the absolute era, 21,411 rows /
    # 46 days, base-weight pool): the ranked |combined| runs hotter (mean 0.204
    # vs 0.151), so the direction band and the raw-confidence divisor are
    # re-anchored to PRESERVE the absolute era's pass rates (42.9% directional;
    # raw_conf scale stable 0.619-0.644 across p75/90/95, tradeable-ranked config). Consumed ONLY when
    # method_score_basis="rank"; the absolute basis keeps buy_sell_diff_threshold
    # and the hardcoded 0.5. Directional pins (_long/_short) resolve via
    # settings.directional() if the asymmetric-cutoff finding is adopted later.
    # 2026-08-14 directive: the rank POOL is the Gate-4-eligible (tradeable)
    # subset — the non-tradeable segment has negative drift + different reversal
    # dynamics, so it is INTERPOLATED against the tradeable distribution rather
    # than voting in it. False -> full-universe ranking.
    rank_tradeable_only: bool = True
    rank_diff_threshold: float = 0.182
    # ADOPTED asymmetric bands (2026-08-14 directive #4, calibrated on the
    # TRADEABLE ranked+shaped combine, 16,926 rows): bullish ≈ the day's top
    # 10% (P(c>=0.324)=10.0%), bearish ≈ the bottom 5% (P(c<=-0.346)=5.0%) —
    # the 10/5 asymmetric cutoff beat symmetric 15/15 at t +2.56 (bear-state
    # windows strongest; every sell-heavy grid lost). Resolved per side via
    # settings.directional("rank_diff_threshold", ...); set both to None to
    # fall back to the symmetric rank_diff_threshold above.
    rank_diff_threshold_long: Optional[float] = 0.324
    rank_diff_threshold_short: Optional[float] = 0.346
    rank_raw_confidence_scale: float = 0.642
    enable_rank_shaping: bool = True
    rank_shape_prior_n: int = 200        # per-decile shrink prior (observations)
    rank_shape_min_rows: int = 500       # settled rows before a method is shaped
    rank_shape_ttl_seconds: int = 21600  # calibration cache TTL (6h)

    # ── ML-combine scale (2026-08-14 A/B repair) ──────────────────────────
    # The stacker convictions (`max(0, 2*P(up)-1)`) live on a FAR smaller scale
    # than the weighted combine — mean |combined| 0.057 vs 0.297 — because a
    # calibrated probability near 0.5 is a genuine "no strong view", not a
    # z-like score. Both live gates are built on the weighted scale, so the ML
    # arm was structurally unable to act: measured over 46 arm runs it produced
    # 4.2% directional rows vs the weighted arm's 70.5%, and only 0.2% of its
    # rows cleared Gate 1 vs 41.7% — and once the rank bands went live it fell
    # to EXACTLY ZERO directional signals. The A/B was therefore comparing a
    # near-empty book against a full one.
    #
    # Repaired the way the rank basis was: quantile-match the pair (band +
    # raw_confidence divisor) so the arm reproduces the weighted arm's OWN live
    # pass rates, making the two cohorts differ in WHICH names they pick rather
    # than in how often they act. Fit on the tradeable (Gate-4) subset,
    # ~17.7k ML rows over 4 days vs 2,600 live weighted rows.
    #
    # The BANDS quantile-match the weighted arm's live 36.9%/16.1% split. The
    # DIVISOR is solved against GATE 1 rather than against the |combined|
    # quantiles: one scalar cannot align two differently-SHAPED distributions
    # everywhere, and matching the quantile ratio (0.642*0.1915 = 0.123) still
    # left the arm at 19.8% joint exposure against the weighted 36.5%, because
    # Gate 1 — not the band — is what actually binds. Solving on Gate 1 gives
    #   ml_raw_confidence_scale = 0.0658  -> ML Gate-1 36.72% vs weighted 36.73%
    #                                        joint (directional AND Gate 1)
    #                                        33.38% vs weighted 36.46%
    # Sensitivity is gentle: +/-10% on the divisor moves Gate 1 by ~2-3pp.
    #
    # NOTE the honest caveat: this equalises EXPOSURE, it does not manufacture
    # skill. If the stacker's flat convictions reflect genuine ignorance, the
    # A/B will now say so in P&L instead of hiding it behind an empty book.
    # Applied ONLY when combine_source == "ml" (both sides stacker-driven); a
    # partial ml_buy/ml_sell swap mixes two scales in one difference and keeps
    # the weighted bands, which is the conservative side (fires less).
    #
    # RETUNED 2026-08-18 (0.0658 -> 0.12) on the multi-day read the note above
    # asked for. The ML arm was over-exposed by a factor that REPLICATED on
    # every post-fix day: ML 2.28/1.67/3.43/2.40 Gate-1 survivors per run vs the
    # weighted arm's 1.06/0.00/1.89/1.00 — 2.15x / 1.81x / 2.40x, and 2.12x
    # (08-14+) vs 2.11x (08-16+, post-retrain) on the aggregates.
    #
    # THE MECHANISM, which is not what the note above assumed. The divisor does
    # not reach Gate 1 mechanically — the LLM restates its own confidence. It
    # reaches Gate 1 by SATURATION: every row with |combined| >= the divisor
    # clips to raw_confidence 1.0 and is shown to the model as
    # "combined_confidence=100%". Measured on ml-source BUY/SELL candidates,
    # the anchor bin is decisive:
    #     anchor <0.60 -> 12.4% pass | 0.60-0.85 -> 6.4% | 0.85-1.00 -> 16.7%
    #     anchor ==1.00 -> 67.2% pass
    # and the LLM tracks the ML anchor closely (spearman +0.68 stated-vs-anchor)
    # BECAUSE it varies. On weighted runs the anchor is saturated for ~all of
    # the top-40 prompt slots (mean 0.997), so it carries no information and the
    # model ignores it (spearman +0.015, its own ~0.86 prior instead). A 100%
    # anchor is only persuasive when other rows are not at 100%.
    #
    # So the divisor is a SATURATION control, and 0.0658 clipped 60-77% of
    # ml-source candidates. 0.12 cuts that to ~21% and predicts 1.45 survivors
    # per run against the weighted arm's 1.24 (1.16x). Deliberately left
    # slightly HOT: the response model holds each anchor bin's pass rate fixed
    # while the distribution moves, which is least trustworthy once saturation
    # is squeezed out, and an arm that goes SILENT teaches nothing and looks
    # identical to one that is losing (the 08-12 failure this whole line of work
    # exists to fix). Not a point estimate: the parity crossing is only
    # identified to ~0.095-0.152 across windows because the response flattens
    # above ~0.14; 0.12 is the middle of that band and inside a 0.12-0.16
    # near-parity plateau (1.16x -> 0.96x).
    #
    # KNOWN DRIFT: the stackers retrain WEEKLY and their conviction scale moves
    # with the artifact (all-row saturation went 16.3% -> ~26% across the 08-15
    # retrain), so a hardcoded divisor decays. Re-read after each retrain;
    # the durable fix is to pin the SATURATION SHARE (weighted's ~9%) as a
    # self-calibrating quantile of the live ml-source |combined| distribution.
    # Also note the divisor feeds SIZING through the confidence ramp, so this
    # brings ML position sizes toward the weighted arm's too.
    ml_diff_threshold_long: Optional[float] = 0.0463
    ml_diff_threshold_short: Optional[float] = 0.0374
    ml_raw_confidence_scale: float = 0.12    # saturation control; see above

    # SELF-CALIBRATING ML DIVISOR (2026-08-18) — src/signals/ml_scale.py.
    # The constant above cannot hold: it controls SATURATION, and the stackers
    # retrain weekly onto a new conviction scale, so a divisor tuned this week
    # drifts out of exposure parity next week and the A/B silently becomes
    # unreadable again. What is pinned instead is the saturation SHARE — the
    # weighted arm's own clip rate, measured live and shrunk toward
    # `ml_saturation_target` — and the divisor is re-solved as the matching
    # quantile of the ml-source |combined_score| distribution. That quantile is
    # NOT censored by the divisor in force (combined_score is persisted
    # unclipped) and the ML divisor never enters its own inputs, so re-deriving
    # it from the panel is sound rather than a backtest. Shrunk toward the
    # constant above by evidence, clamped to a fixed multiplicative band
    # (module constants, not settings), reported to the calibration registry,
    # and fail-soft to the constant on any error. Honours analysis_asof.
    enable_ml_scale_calibration: bool = True
    # Trailing evidence window (TRADING days; the query pads for weekends).
    # Short enough to re-centre within days of a retrain, long enough that the
    # quantile is stable — ~4k ml rows/day makes 5 days ~20k observations.
    ml_scale_window_days: int = 5
    # Evidence floor: below this many ml-source rows the static value holds.
    ml_scale_min_rows: int = 2000
    # Shrinkage strength, in observations, for BOTH the target share (toward
    # ml_saturation_target) and the divisor (toward ml_raw_confidence_scale).
    ml_scale_prior_n: int = 4000
    # Documented prior for the parity target: the weighted arm's measured
    # saturation share (9.17% of its rows clipped at raw_confidence 1.0 over
    # 2026-08-12..18). The live weighted measurement overrides this as evidence
    # accrues; the prior only holds when the weighted arm is thin.
    ml_saturation_target: float = 0.09

    # ANTI-CHASE / OVEREXTENSION GATE (Gate 5 of the actionable filter, 2026-07-22).
    # The BUY-vs-SELL forensics (signals panel, 8.3k ticker-days) found the BUY
    # side's failure is CHASING: 40% of BUY calls landed in the top quintile of
    # trailing-5-day gainers (2x the universe share) and those hit 38% at 5d with
    # a -2.4% median — recent big gainers mean-revert (universe Q5 P(up) 45%),
    # while the combined score peaks exactly then (momentum+tech+news all read
    # the same run-up; top score decile P(up) 38%, the worst cohort in the panel).
    # Gate: a BUY whose ticker's trailing close-to-close return over the last
    # `overextension_lookback_bars` COMPLETED daily bars exceeds
    # `overextension_runup_pct` is dropped from the actionable set (observe-only
    # this tick; it re-qualifies at the next tick it has cooled below the bar —
    # entering on the pullback is the measured sweet spot, universe Q2 +0.64%).
    # BUY-ONLY by design: SELLs on crashed names ride continuation (61.9% hit on
    # t5<-8% names) and SELLs on spiked names fade correctly (64% hit) — gating
    # them would destroy measured edge. Threshold from the 2026-07-22 sweep:
    # blocked cohort at 12% hit 32.5% (median -4.1%) vs kept 43.4%; the 10-15%
    # region is a plateau so 12 is not knife-edge (vol-normalized variants
    # separated WORSE than the raw %). Fail-OPEN: a ticker with no/short cached
    # history is NOT blocked (this is an opportunity filter on measured harm,
    # not a risk guard — the liquidity gate already owns the thin-data case).
    enable_overextension_gate: bool = True
    overextension_runup_pct: float = 12.0
    overextension_lookback_bars: int = 5

    # Qwen structured output: request response_format={"type":"json_object"} on the
    # synthesis call so the answer is guaranteed-parseable JSON (the prompt asks for
    # {"recommendations": [...]}; the parser accepts both the wrapped object and the
    # legacy bare array). If DashScope rejects the parameter (e.g. an
    # enable_thinking incompatibility), the call retries once WITHOUT it — Qwen
    # stays primary either way. Off = free-text JSON + fence/truncation repair only.
    qwen_json_mode: bool = True

    # Confidence RECALIBRATION for sizing (2026-07-12): a standalone multiplicative
    # tilt whose input is the LEDGER's empirical win rate for the trade's stated-
    # confidence band, not the stated number (weakly calibrated, ρ≈+0.07 — the
    # compressed ramp only LIMITS the damage; this layer follows the evidence,
    # including sizing DOWN a high-stated band that empirically loses). House
    # idiom: per-band win rates shrunk toward the pooled win rate by prior_n
    # pseudo-observations (the shrinkage IS the evidence throttle — thin band ⇒
    # neutral), tilt = 1 + span × clamp((p_band − p_pool)/half_width, ±1), inert
    # below min_trades closes, fail-soft to 1.0, registry-reported
    # (confidence_recal_spread). Bands shared with the dashboard's confidence-
    # calibration buckets. See src/performance/confidence_sizing.py.
    enable_confidence_recal_sizing: bool = True
    confidence_recal_span: float = 0.25        # max ± size tilt at a ≥half_width win-rate gap
    confidence_recal_half_width: float = 0.10  # win-rate gap (10 pp) that saturates the ramp
    confidence_recal_prior_n: int = 20         # shrinkage strength (pseudo-observations toward pool)
    confidence_recal_min_trades: int = 20      # eligible closes before the layer speaks at all
    # max_tokens (ANSWER room) for the DeepSeek fallback on the normally-small
    # sentiment / macro-news calls when llm_max_thinking is on — DeepSeek's thinking
    # shares this budget, so it needs headroom the tiny answer wouldn't. Qwen ignores
    # the slack (its reasoning has its own budget).
    llm_thinking_sentiment_max_tokens: int = 16000
    llm_thinking_macro_max_tokens: int = 8000

    # News sources
    newsapi_key: str = ""
    alpha_vantage_key: str = ""
    # Finnhub — real-time company news (free tier). Empty key → the source is
    # skipped. Free company-news has no per-article sentiment, so it adds news
    # COVERAGE; the provider-sentiment LLM-skip is driven by Polygon insights.
    finnhub_api_key: str = ""
    enable_finnhub_news: bool = False
    # Polygon/Massive news + per-article sentiment "insights" (each article carries
    # {ticker, sentiment, reasoning}). ON by default now that we're on the Advanced
    # plan — real-time Benzinga-sourced coverage; feeds the provider-sentiment hybrid
    # below so the LLM scorer can be skipped for these articles.
    enable_polygon_news: bool = True
    # Provider-sentiment hybrid: when an article carries a provider sentiment
    # (Polygon insights), derive the per-ticker news score from those instead of
    # calling the DeepSeek/Haiku scorer — a latency + cost win. Falls back to the
    # LLM when too few provider-scored articles exist. ON by default (Advanced plan).
    enable_provider_sentiment: bool = True
    provider_sentiment_min_articles: int = 2   # min provider-scored relevant articles to skip the LLM
    provider_sentiment_magnitude: float = 0.6  # |score| a positive/negative label maps to ([-1,1] scale)

    # Quiver Quantitative — alternative data (Hobbyist tier). Empty key → every
    # Quiver source is skipped. Congress trades revive the smart-money congressional
    # feed (dead since the Stock Watcher S3 went 403); gov-contracts / lobbying /
    # off-exchange (dark-pool) are rendered as synthetic NewsArticles and scored by
    # the sentiment pipeline (same pattern as Trends/Reddit/short-interest).
    quiver_api_key: str = ""
    enable_quiver_congress: bool = True        # → smart_money (List[InsiderTrade])
    enable_quiver_gov_contracts: bool = True   # → NewsArticle (federal contract awards = revenue catalyst)
    enable_quiver_lobbying: bool = True        # → NewsArticle (lobbying spend = regulatory-attention context)
    enable_quiver_offexchange: bool = True     # → NewsArticle (per-ticker dark-pool accumulation/distribution)
    quiver_lookback_days: int = 30             # window for congress / contracts / lobbying events
    quiver_offexchange_max_tickers: int = 60   # cap the per-ticker dark-pool loop (Hobbyist rate limits)

    # Financial Modeling Prep — market-wide analyst upgrades/downgrades feed (Section E catalyst
    # discovery). Free key: https://site.financialmodelingprep.com/developer/docs . Empty → the
    # market-wide analyst discovery source is skipped (yfinance analyst data is per-ticker only).
    fmp_api_key: str = ""

    # Email (all optional — only needed when running with --email)
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_recipients: str = ""  # comma-separated

    # Watchlist
    stock_watchlist: str = "AAPL,MSFT,NVDA,TSLA,AMZN,META,GOOGL"
    sector_etfs: str = "XLK,XLF,XLE,XLV,XLY,XLP,XLI,XLB,XLU,XLRE,XLC"

    # Commodities — always included in every run regardless of trending
    # GLD=Gold, SLV=Silver, IAU=Gold(alt), GDX=Gold Miners, PPLT=Platinum, PALL=Palladium, CPER=Copper
    commodity_etfs: str = "GLD,SLV,IAU,GDX,PPLT,PALL,CPER"

    # Feature flags
    enable_fetch_data: bool = True        # set to false to skip all live data fetching (Polygon + yfinance)
    enable_charts: bool = False           # set to true to build Plotly charts and HTML report

    # Analysis method flags (at least one should be true)
    enable_news_sentiment: bool = True    # method 1: LLM sentiment from news/RSS
    # Per-ticker news via yfinance Ticker.news — gives EVERY symbol real,
    # ticker-tagged articles (not just the ~30 mega-caps the keyword aliases
    # cover), which is what actually feeds the news + sentiment-velocity scores.
    # Fetched once per hour (cached with the news pool). Kill switch if Yahoo
    # rate-limits.
    enable_ticker_news: bool = True

    # Per-ticker Google News RSS — free, no key, near-real-time, and far broader
    # than the 5 fixed market feeds (surfaces Reuters/Bloomberg/Barron's/FT AND
    # Business Wire, the one wire our direct feeds miss). Fetched fresh every tick
    # (reactivity fast-lane). google_news_max_tickers caps the per-tick request
    # burst; google_news_business_wire adds the per-ticker site:businesswire.com query.
    enable_google_news: bool = True
    google_news_max_tickers: int = 50
    google_news_business_wire: bool = True

    # FDA / MedWatch regulatory catalyst RSS (free, no key) — drug approvals/CRLs +
    # device recalls, on the fresh-every-tick fast lane. High signal for drug/device
    # names; market-wide (mapped via keyword aliases like the other RSS feeds).
    enable_fda_news: bool = True

    # Alpha Vantage NEWS_SENTIMENT — pre-scored per-ticker news that feeds the
    # LLM-skip hybrid (like Polygon insights). ONE batched call, hourly-cached.
    # OFF by default: the free tier is ~25 req/DAY, shared with AV discovery +
    # earnings — only enable on a paid tier (or if you don't use AV elsewhere).
    enable_alpha_vantage_news: bool = False
    alpha_vantage_news_max_tickers: int = 50

    # StockTwits crowd sentiment — one synthetic chatter-summary article per ticker
    # (LLM-scored, like Reddit). The public endpoint now 403s without auth, so this
    # needs a (free) StockTwits API token and is OFF by default.
    enable_stocktwits: bool = False
    stocktwits_access_token: str = ""
    stocktwits_max_tickers: int = 30

    # Sentiment velocity (Δsentiment, not level) — the rate of change of news tone leads
    # short-horizon (1–5 day) moves better than the absolute level. Deterministic lexical
    # polarity per article, bucketed by published_at into a recent vs prior window;
    # velocity = recent_tone − prior_tone. No extra LLM/API cost (reuses stored timestamps).
    enable_sentiment_velocity: bool = True
    sentiment_velocity_recent_hours: int = 24   # "recent" window: articles ≤ N hours old
    sentiment_velocity_prior_hours: int = 96    # "prior" window: from recent_hours to N hours old

    # ── news_shock — abnormal news attention (2026-08-14, panel-first weight 0) ──
    # sign(news) × how far today's recency-mass sits ABOVE the ticker's own
    # trailing baseline (log2 ratio, full score at 8×). The quantitative
    # complement to the news LEVEL: a real catalyst is both directional and
    # LOUD vs the ticker's normal; a routine story is not. Baseline = per-ticker
    # median of daily `signals.news_recency_mass` (forward-collected — the
    # method self-activates as the column accrues; abstains until then).
    enable_news_shock: bool = True
    news_shock_baseline_days: int = 20   # trailing window for the baseline median
    news_shock_min_days: int = 5         # covered days required before scoring
    # news_bear_fresh (2026-08-15, panel-first weight 0 — signals/news_bear_fresh.py):
    # bear-news freshness guard — the bearish news score scaled DOWN by how much
    # of the bad news the 3-session tape has already priced (abstains entirely at
    # a 2σ aligned decline: never short into a hole) and up to 1.5x when the tape
    # ignored or rose against it. Bull/zero news abstains. Measured: bear-event
    # daily IC +0.095 (t +2.96) vs +0.029 for news alone, pivot basis, gated.
    enable_news_bear_fresh: bool = True
    # catalyst_tilt (2026-08-15, panel-first weight 0 — signals/catalyst_tilt.py):
    # the news read × a learned per-(catalyst, side) orientation from the
    # news-event dataset (self-calibrating, 6h refresh, shrunk toward ABSTAIN).
    # Held-out LOMO: IC +0.039 (t +2.22) where raw news measured −0.054 on the
    # same rows. Needs the v4 catalyst capture; abstains without one.
    enable_catalyst_tilt: bool = True

    enable_technical_analysis: bool = True  # method 2: RSI, MACD, SMA, Bollinger Bands
    enable_insider_trades: bool = True    # method 3: politician + corporate insider trades

    # Insider trades config
    insider_lookback_days: int = 90      # how far back to look for trades
    smart_money_top_tickers: int = 5     # max number of ticker groups shown in smart money section

    # Insider buying persistence — amplify insider_score when the SAME insider buys the same
    # ticker on multiple SEPARATE days within the lookback window. Repeated accumulation by one
    # name (depth of conviction) is a stronger tell than a one-off purchase; distinct from the
    # cluster amplifier (which measures breadth — many DIFFERENT insiders buying at once).
    # Amplifier scales 1.0 + 0.25×(distinct_buys − 1), capped at 1.75× (mirrors the cluster ceiling).
    enable_insider_persistence: bool = True
    insider_persistence_min_buys: int = 2   # min distinct buy days by one insider to qualify
    # Comma-separated names to prioritise (empty = include all politicians)
    tracked_politicians: str = (
        "Nancy Pelosi,Paul Pelosi,Austin Scott,Michael McCaul,"
        "Dan Crenshaw,Tommy Tuberville,Shelley Moore Capito,"
        "Josh Gottheimer,Ro Khanna,Brian Higgins"
    )

    # Unusual options flow (strategy 4)
    enable_options_flow: bool = True    # scan yfinance options chains for unusual sweeps

    # SEC EDGAR filings (strategies 5, 6, 7)
    enable_sec_filings: bool = True     # 13D/13G activist stakes, Form 144, 13F
    sec_filings_lookback_days: int = 30  # lookback window for 13D/13G and Form 144
    # Comma-separated institution names for 13F superinvestor tracking (broader = stronger
    # consensus when multiple filers buy the same name). The pipeline resolves CIKs dynamically
    # from EDGAR — no manual lookup needed. Each adds EDGAR calls, so keep the list reasonable.
    tracked_institutions: str = (
        "Berkshire Hathaway,Pershing Square Capital Management,"
        "Appaloosa Management,Baupost Group,Scion Asset Management,"
        "Greenlight Capital,Third Point,Icahn Capital,"
        "Tiger Global Management,Duquesne Family Office"
    )

    # Market-wide Form 4 open-market-buy scan — surfaces insider ACCUMULATION everywhere (not
    # just the watchlist). Parses recent Form 4 XML for transaction code "P" (open-market
    # purchase); these become corporate_insider "purchase" records that feed the insider CLUSTER
    # + PERSISTENCE detectors and discovery. Open-market buys are only ~1-2% of all Form 4s, so
    # the scan parses a bounded number of the most recent filings (cached daily). enable_sec_filings.
    enable_form4_scan: bool = True
    form4_scan_lookback_days: int = 3
    form4_scan_max_filings: int = 150     # recent Form 4 filings parsed per run (cached daily)

    # FRED (Federal Reserve of St. Louis) — macro regime context
    # Free API key: https://fred.stlouisfed.org/docs/api/api_key.html
    fred_api_key: str = ""
    enable_fred: bool = True    # fetch yield curve, CPI, unemployment, credit spreads, M2

    # Earnings whisper vs. consensus gap — infers the implied "whisper number" from:
    # historical beat rate, avg EPS surprise magnitude, and consensus revision trend.
    # Uses yfinance earnings_dates + eps_trend + eps_revisions (free, no key required).
    enable_earnings_whisper: bool = True

    # Analyst estimate revision momentum — compares PT/rating changes over two 30-day windows
    # to detect whether analyst consensus is accelerating (improving) or decelerating (deteriorating).
    # Requires enable_analyst_ratings=True for yfinance data; cached daily.
    enable_revision_momentum: bool = True

    # CESI-style Macro Surprise Index — compare recent FRED releases to trailing 3-period averages
    # Consistent beats → cyclical tailwind; consistent misses → defensive bias. Requires FRED_API_KEY.
    enable_macro_surprise: bool = True

    # Market-implied Fed rate expectations — T-bill spreads proxy for CME FedWatch.
    # Derives P(cut/hold/hike) at next FOMC meeting + 12m cumulative cuts in bp. Requires FRED_API_KEY.
    enable_fedwatch: bool = True

    # CFTC Commitment of Traders — weekly futures positioning (no API key required)
    enable_cot: bool = True     # download COT data from CFTC; cached by ISO week

    # SEC 8-K material event filings — faster than RSS feeds (no API key required)
    enable_8k_filings: bool = True
    eight_k_lookback_days: int = 5   # fetch 8-Ks filed in the last N days

    # SEC S-1/S-11 IPO pipeline — sector-level institutional demand signal (no API key required)
    enable_ipo_pipeline: bool = True
    ipo_lookback_days: int = 30      # S-1 filings accumulate over weeks; 30 days gives a full picture

    # Analyst upgrades/downgrades/price-target changes — yfinance (free, no key required)
    enable_analyst_ratings: bool = True
    analyst_ratings_lookback_days: int = 30

    # Put/Call ratio — market sentiment + per-ticker directional bias
    # Market-wide: CBOE equity P/C CSV (free, no key); per-ticker: yfinance options volume
    enable_put_call: bool = True

    # Credit market leading indicator — HYG vs SPY divergence (yfinance, no key required)
    # High-yield bonds lead equities by 1-3 days; divergence warns of coming equity moves.
    enable_credit: bool = True

    # VIX & term structure — CBOE volatility indices via yfinance (no key required)
    # ^VIX, ^VXN, ^VVIX, ^VIX9D, ^VIX3M, ^VXMT
    enable_vix: bool = True

    # NYSE TICK index — breadth exhaustion / reversal signal (^TICK via yfinance, no key required)
    # Extreme readings (>+1000 or <-1000) are contrarian reversal signals.
    enable_tick: bool = True

    # VWAP distance — rolling 20-day volume-weighted average price vs current price.
    # Mean-reversion signal: large deviations attract institutional order flow back toward VWAP.
    enable_vwap: bool = True

    # Earnings calendar + EPS surprises
    # Upcoming dates: yfinance (free) + Alpha Vantage EARNINGS_CALENDAR (free with key)
    # EPS beat/miss: yfinance earnings_dates (free)
    enable_earnings: bool = True
    earnings_lookback_days: int = 90   # how far back to look for recent EPS surprises
    earnings_upcoming_days: int = 30   # how many days ahead to include in calendar

    # Short interest — FINRA Reg SHO daily short volume + yfinance (no API key required)
    # Squeeze setups (high SI + low days-to-cover), bearish positioning, short covering signals
    enable_short_interest: bool = True

    # Gamma Exposure (GEX) — options market structure: dealer positioning, gamma flip,
    # max pain, and expected move derived from yfinance options chains (no key required).
    # Covers SPY/QQQ/IWM always + any watchlist ticker with OI ≥ 1000 contracts.
    enable_gex: bool = True

    # McClellan Oscillator & Summation Index — NYSE A/D breadth momentum (^NYAD via yfinance, no key required)
    # Oscillator = EMA19 − EMA39 of daily net advances; Summation = running total; zero crosses = swing timing
    enable_mcclellan: bool = True

    # New 52-week highs vs. lows — HL Spread = %near_highs − %near_lows over sector ETFs + watchlist (yfinance, no key)
    # Divergence: SPY near 52w high + HL spread declining → bearish; SPY near 52w low + HL spread rising → bullish
    enable_highs_lows: bool = True

    # Market breadth — % of S&P 500 sector ETFs above their 200-day SMA (yfinance, no key required)
    # < 30% = broadly oversold; rising from < 30% = confirmed breadth thrust (strong multi-month bullish signal)
    enable_breadth: bool = True

    # Google Trends — search interest spike/drop as retail attention proxy (no API key required)
    enable_google_trends: bool = False  # OFF 2026-06-25: pytrends' unofficial API is chronically 429'd by Google (low-value retail-attention proxy, no clean fix). Cached daily when on.

    # Ticker discovery — extra sources that widen the analysis universe (Step 0, fail-graceful).
    # WSB cashtag discovery: most-mentioned valid tickers across r/wallstreetbets, r/stocks,
    # r/investing hot/rising posts via Reddit's public JSON (no key) — validated against the SEC
    # ticker universe. NewsAPI/headline discovery is open-vocabulary via the same SEC map.
    # StockTwits trending is implemented but its public API is Cloudflare-gated (403 without a
    # browser/token), so it's OFF by default — enable only if you have working access.
    enable_stocktwits_discovery: bool = False
    enable_wsb_discovery: bool = True
    wsb_discovery_min_mentions: int = 3   # min distinct hot/rising posts mentioning a ticker

    # Opportunity Screener — PROACTIVE setup discovery over a broad liquid universe.
    # Shifts discovery from "what's trending" to "what's technically set up". Screens a
    # curated liquid universe (+ everything already in the OHLCV cache) cache-first for:
    # unusual volume, 52-week breakouts / new-low reversals, relative strength vs SPY, and
    # golden/death crosses — then injects qualifying names into the analysis universe. Reuses
    # the OHLCV cache (works offline once warm); a bounded warm-up fetch primes the cache.
    enable_opportunity_screener: bool = True
    screen_volume_ratio: float = 2.0           # today vol / 20d avg ≥ this → unusual volume
    screen_rs_lookback_days: int = 63          # ~3 months for relative strength vs SPY
    screen_rs_threshold_pct: float = 10.0      # excess return vs SPY (pp) → strong/weak RS
    screen_cross_lookback: int = 5             # bars within which a 50/200 SMA cross is "fresh"
    screen_near_high_pct: float = 2.0          # within this % of the 52-week high → breakout-watch
    screen_min_price: float = 5.0               # liquidity gate: minimum last close ($)
    screen_min_dollar_volume: float = 20_000_000  # liquidity gate: min 20d avg $ volume (filters thin pumps)
    screen_max_fetch_per_run: int = 30         # cap cold OHLCV fetches per run (cache warms over time)
    screen_max_results: int = 20               # max setups the screener injects into the universe

    # Macro → Discovery loop — closes the gap between the macro/regime modules and stock selection.
    # The sector-rotation (top inflows), business-cycle (phase leaders), and DIX (regime → factor
    # tilt) modules identify FAVORED sector/factor ETFs; this auto-pulls their top holdings as
    # candidates so the analysis universe is biased toward where macro money is flowing. Holdings
    # come from yfinance funds_data (cached daily) with a static SPDR fallback.
    enable_macro_discovery: bool = True
    macro_discovery_top_sectors: int = 3       # favored ETFs to pull from each source (rotation / cycle)
    macro_discovery_holdings_per_etf: int = 8  # top N holdings pulled per favored ETF
    macro_discovery_max: int = 25              # max constituent names injected into the universe

    # ── Section E: Catalyst & relationship expansion ──────────────────────────────
    # Widen the universe along three axes the static watchlist misses: (1) market-wide CATALYST
    # discovery — names with imminent earnings or fresh analyst rating changes ANYWHERE in the
    # market (not just the watchlist); (2) RELATIONSHIP discovery — pull the partner leg of a
    # cointegrated pair when one leg is already in the universe; (3) a richer factor/thematic ETF
    # universe. All fail-graceful and capped; discovered names are injected at Step 0 so the
    # existing per-ticker enrichment (analyst ratings, earnings, signal stack) picks them up.

    # Earnings-calendar discovery — inject names reporting within the window (market-wide via the
    # Alpha Vantage EARNINGS_CALENDAR feed; yfinance has no market-wide calendar). Needs
    # alpha_vantage_key; skipped without it.
    enable_earnings_discovery: bool = True
    earnings_discovery_window_days: int = 7    # report within N days to be injected
    earnings_discovery_max: int = 15           # cap names injected per run

    # Analyst-ratings discovery — inject names with fresh upgrades/downgrades anywhere in the market
    # (Financial Modeling Prep upgrades-downgrades RSS feed). Needs fmp_api_key; skipped without it.
    enable_analyst_discovery: bool = True
    analyst_discovery_lookback_days: int = 3   # rating change within N days
    analyst_discovery_min_firms: int = 1       # min distinct firms acting on a name to inject
    analyst_discovery_max: int = 15            # cap names injected per run

    # Cointegration peer-expansion — when a tradeable cointegrated pair has one leg in the universe
    # and the partner outside it, pull the partner in so the relationship is tradeable both ways.
    enable_coint_peer_discovery: bool = True
    coint_peer_max: int = 10                   # cap partner legs injected per run

    # Factor / thematic ETF universe — pinned ETFs (like commodities) broadening coverage beyond the
    # 11 GICS sectors: style factors (momentum/quality/value/size/low-vol/growth) + high-interest
    # themes (semis, software, biotech, defense, clean energy, homebuilders, airlines, regional banks).
    enable_factor_etfs: bool = True
    factor_etfs: str = "MTUM,QUAL,VLUE,SIZE,USMV,IWF,IWD"
    thematic_etfs: str = "SMH,IGV,XBI,ITA,TAN,LIT,IBB,XHB,JETS,KRE"

    # ── Section F: Discovery liquidity gate ───────────────────────────────────────
    # A uniform quality floor on EVERY discovered candidate (trending/open-vocab, screener,
    # macro→discovery, market-wide earnings/analyst catalysts, cointegration peers) so widening the
    # funnel (Sections A–E) doesn't inject untradeable microcaps — the names the tracker's bid-ask
    # model charges up to 250 bp a side. NEVER gates the pinned universe (watchlist, sector ETFs,
    # commodities, factor/thematic ETFs) or open-trade tickers. Cache-first with a bounded warm-up
    # fetch; a name whose liquidity can't be verified is dropped (fail-closed).
    enable_discovery_liquidity_gate: bool = True
    # Loosened 2026-07-05 to WIDEN the net toward penny / lower-volume names so the
    # predictability panel can measure whether they are easier or harder to
    # predict (bucket features `price` + `dollar_vol`). The $1 floor keeps out
    # sub-$1 OTC junk (awful spreads / data). NOTE: dollar-volume is the main
    # universe-SIZE lever (most tickers are gated by it, not price) — if ticks get
    # slow or costly, raise it. 2026-07-08: these are now the OBSERVATION floor —
    # kept LOW so penny / thin-volume names enter the universe and keep accruing
    # performance data (the signals panel + the price/volume + predictability
    # panels); a SEPARATE, HIGHER TRADE floor (trade_min_* below) decides what can
    # actually trade, so sub-threshold names are scored + tracked but observe-only.
    discovery_min_price: float = 1.0                 # observation floor — min last close ($)
    discovery_min_dollar_volume: float = 1_000_000   # observation floor — min 20d avg $ volume ($)
    # TRADE liquidity floor (separate from + higher than the discovery/observation
    # floor above): a BUY/SELL for a ticker below EITHER of these is OBSERVE-ONLY —
    # still scored + persisted to the signals panel (so penny-stock performance
    # keeps accruing) but NEVER actionable (no sim trade, no broker order). Applied
    # in the pipeline actionable filter (Gate 4); fail-closed via is_liquid (a name
    # whose liquidity can't be verified is not traded). Flag false = disable.
    enable_trade_liquidity_gate: bool = True
    trade_min_price: float = 5.0                      # min last close ($) to be tradeable
    trade_min_dollar_volume: float = 5_000_000        # min 20d avg $ volume to be tradeable
    discovery_gate_max_fetch: int = 25               # cap cold OHLCV fetches per run for the gate
    # Drop exotic security TYPES from discovery (preferred series, warrants, units,
    # rights, OTC foreign ordinaries) — redundant with a primary listing and/or not
    # on the US consolidated tape (can't be priced deterministically). The pinned
    # watchlist bypasses the gate, so an explicitly-chosen preferred is still honored.
    enable_security_type_filter: bool = True

    # Reddit social sentiment — r/wallstreetbets, r/stocks, r/investing
    # Free Reddit API credentials: https://www.reddit.com/prefs/apps (create "script" app)
    enable_reddit_sentiment: bool = True
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "llm_trader/1.0 (stock analysis bot)"

    # OpEx calendar — options expiration week effects (pure date math, no API calls)
    # OpEx week (3rd Friday) → max pain pinning; Triple Witching (Mar/Jun/Sep/Dec) → strongest
    enable_opex: bool = True

    # Seasonality calendar — end-of-month rebalancing, quarter-end window dressing,
    # January effect (small-cap rebound), and monthly historical biases (pure date math, no API calls)
    enable_seasonality: bool = True

    # Bond market internals — 1–8 week macro regime signals via yfinance (no API key required)
    # Treasury curve (10Y-3M), TLT/IEF/TIP/LQD price momentum, real yield, IG credit
    enable_bond_internals: bool = True

    # MOVE Index — ICE BofA Treasury implied volatility (bond market VIX) via yfinance (no key required)
    # ^MOVE primary ticker; VXTLT fallback. Spikes precede equity dislocations by 1–5 days.
    enable_move: bool = True

    # Dark Pool Index (DIX) + market-wide GEX — SqueezeMetrics free CSV (no API key required).
    # DIX = dollar/volume-weighted off-exchange short volume → proxy for hidden institutional
    # accumulation; high DIX is historically bullish for forward S&P returns (leads ~1–4 weeks).
    # Market-wide GEX gauges whole-index dealer gamma (low/negative = vol expansion). High DIX +
    # low GEX = classic "hidden buying with room to run". Macro-context overlay (not per-ticker);
    # feeds the Claude prompt and the Macro Regime Filter. Cached daily.
    enable_dix: bool = True

    # Global macro cross-asset regime — DXY strength (DX-Y.NYB) + Copper/Gold ratio (HG=F / GC=F)
    # DXY: strong dollar = headwind for EM, commodities, multinationals.
    # Copper/Gold: rising ratio = risk-on expansion; declining = risk-off contraction.
    enable_global_macro: bool = True

    # Pattern recognition — detects 8 classical chart patterns and scores them by historical win rate.
    # On first run per ticker: fetches 2y of OHLCV data and builds a pattern success library
    # (cache/patterns/<TICKER>.json, TTL 7 days). Subsequent runs are instant (warm-cache lookup).
    enable_pattern_recognition: bool = True

    # Pattern-registry feedback loop — record the outcome of every real
    # BUY/SELL the system takes when a chart pattern is active at entry, then
    # blend that live win rate into the synthetic per-ticker prior used by
    # compute_pattern_score. Bayesian shrinkage with ``pattern_registry_prior_n``
    # virtual trials on the synthetic side: live evidence only dominates once
    # ``live_n >> prior_n``. Stored at ``cache/pattern_registry.json``.
    enable_pattern_registry: bool = True
    pattern_registry_prior_n: int = 10

    # Market Mode Switching — dynamically adjusts signal weights based on TRENDING/CHOPPY regime.
    # TRENDING (low VIX, healthy breadth): up-weights tech/news, down-weights vwap/put_call.
    # CHOPPY   (high VIX, mixed breadth):  up-weights vwap/put_call, down-weights tech.
    # NEUTRAL:  uses baseline _BASE_WEIGHTS unchanged.
    enable_market_mode_switching: bool = True

    # Macro Regime Filter — top-down overlay that gates BUY entries and adjusts thresholds.
    # Reads VIX, MOVE, bond internals, global macro, FRED, breadth, and credit to produce:
    #   PANIC     → threshold 0.88, BUY entries blocked
    #   RISK_OFF  → threshold 0.82, BUY entries blocked
    #   CAUTION   → threshold 0.80
    #   NEUTRAL   → threshold 0.78 (baseline, unchanged)
    #   RISK_ON   → threshold 0.72
    enable_macro_regime_filter: bool = True
    # Minimum number of macro inputs (of ~10: VIX, MOVE, bond, global, FRED, breadth,
    # credit, DIX, intermarket, macro_news) that must be available for the composite
    # regime to be trusted. Below this, the regime is forced to at least CAUTION rather
    # than allowed to fail-OPEN to a permissive NEUTRAL when the feeds go dark.
    macro_regime_min_inputs: int = 3

    # Sector Rotation / "Ebb and Flow" — cross-sector money flow via relative momentum + volume
    # Ranks all 11 SPDR sector ETFs by excess return vs SPY (1w/1m/3m) adjusted for volume.
    # Identifies top inflow/outflow sectors, rotation regime (RISK_ON/NEUTRAL/RISK_OFF),
    # and explicit rotation pairs (e.g., "XLK → XLP"). Cached daily, yfinance/Polygon OHLCV.
    enable_sector_rotation: bool = True

    # Rotation Drivers — rate-cycle phase from actual DFF trajectory (3m/12m) + CPI trend.
    # Maps Fed hiking/pausing/cutting cycles to cross-asset rotation implications:
    # favoured/avoided asset classes per phase (EARLY_TIGHTENING → EASING_CYCLE).
    # Requires FRED_API_KEY. Cached daily.
    enable_rotation_drivers: bool = True

    # Price Momentum (Perceived Value) — multi-period price trend normalised against own history.
    # Captures the self-reinforcing dynamic: rising perceived value attracts more capital →
    # trend continues. Scores 1m and 3m returns vs trailing 252-day return distribution.
    # Uses OHLCV chart cache first (works with ENABLE_FETCH_DATA=false); falls back to yfinance.
    enable_price_momentum: bool = True

    # Market-Relative Momentum — diagnostic ticker − SPY residual.
    # Answers "is this name lagging the broad market?" independently of its
    # sector. NOT added to the weighted aggregator combo (would double-count
    # beta against the sector-relative score, since market_rel = sector_rel +
    # (sector − market)). The email/prompt surface it side-by-side with the
    # sector-relative reading so divergences are visible:
    #   sector_rel positive, market_rel negative → best-of-a-bad-sector
    #   sector_rel ≈ 0,      market_rel negative → "beta drag" (riding weak sector)
    #   sector_rel negative, market_rel positive → stock-specific weakness in a
    #                                              strong sector
    enable_market_relative_momentum: bool = True

    # Sector-Relative Momentum — beta-stripped alpha factor (ticker − sector ETF).
    # The classic absolute momentum signal mixes idiosyncratic alpha with sector
    # beta — if the whole sector is ripping, every constituent looks like it has
    # momentum even though none is genuinely outperforming. This module subtracts
    # the benchmark return to leave only the residual: did NVDA *beat* tech, or
    # just ride the wave? Resolved benchmark: sector ETF for stocks (via the
    # aggregator map + yfinance lookup, SPY fallback), SPY for ETFs, no benchmark
    # for commodities. Uses cached OHLCV; falls back to a one-shot fetch when
    # the benchmark's series is missing.
    enable_sector_relative_momentum: bool = True

    # Money Flow Indicators — accumulation/distribution composite (MFI + CMF + OBV slope).
    # MFI (14-period): volume-weighted RSI; < 20 = accumulation, > 80 = distribution.
    # CMF (20-period): Chaikin Money Flow; positive = institutional buying.
    # OBV slope z-score: sustained volume trend direction.
    # Uses OHLCV chart cache first (works with ENABLE_FETCH_DATA=false); falls back to yfinance.
    enable_money_flow: bool = True

    # Trend Strength — ADX/DMI directional movement (Welles Wilder) + Donchian channel breakout
    # (the "Turtle" system). Measures trend QUALITY/strength + breakout confirmation — a dimension
    # not captured by momentum (return size) or RSI/Bollinger (overbought/oversold). ADX<20 = chop
    # (signal dampened); strong +DI/-DI separation with high ADX or a 20-day breakout = confirmed
    # trend. Uses OHLCV chart cache first (works with ENABLE_FETCH_DATA=false); falls back to yfinance.
    enable_trend_strength: bool = True
    trend_adx_period: int = 14        # Wilder ADX/DMI smoothing period
    trend_donchian_period: int = 20   # Donchian breakout channel lookback (Turtle = 20)

    # Post-Earnings Announcement Drift (PEAD) — one of the most-replicated cross-sectional
    # anomalies in academic finance. Stocks that beat (miss) EPS estimates tend to continue
    # drifting in the surprise direction for ~60 days as the market under-reacts. Score is
    # tanh(surprise_pct / surprise_scale_pct) × max(0, 1 − days_since_report / decay_window).
    # Uses yfinance earnings_dates (same source as enable_earnings); cached daily.
    enable_pead: bool = True
    pead_decay_window_days: int = 60       # days for the linear time-decay to reach zero
    pead_surprise_scale_pct: float = 25.0  # tanh saturation point (±25% surprise -> ±0.76)

    # IV Rank + Directional — volatility-regime-aware directional bias.
    # Uses 21-day realized vol percentile (vs trailing 252-day distribution) as a proxy
    # for IV Rank, combined with 5-day return / ATR to switch between contrarian (high IR)
    # and trend-confirming (low IR) directional scoring. Robust to regime shifts because
    # both inputs are self-normalised against each ticker's own vol footprint.
    # Uses OHLCV chart cache first (works with ENABLE_FETCH_DATA=false); falls back to yfinance.
    enable_iv_rank: bool = True

    # IV Expression — stock-vs-options expression decision from the real options chain.
    # Pulls live market-implied vol (expected_move_pct from GEX context) and ranks it
    # against the ticker's own trailing IV history (reconstructed from prior gex_*.json
    # caches). Combines with options-market oi_skew to derive expression bias:
    # cheap options + strong skew → CHEAP_DIRECTIONAL (confirm);
    # expensive options + strong skew → FADE_PREMIUM (contrarian).
    # No new data fetching — reuses the already-fetched GEX context.
    enable_iv_expr: bool = True

    # Cointegration Pairs — statistical-arbitrage market-neutral alpha (beyond sector_pairs).
    # Engle-Granger two-step: OLS hedge ratio on log prices → native (numpy) ADF test on
    # the residual spread. Cointegrated pairs whose spread z-score is stretched past the
    # entry band become market-neutral LONG-cheap / SHORT-rich trades. Also derives a
    # per-ticker directional lean fed into the aggregator. Cache-first OHLCV (works with
    # ENABLE_FETCH_DATA=false). No statsmodels dependency.
    enable_cointegration: bool = True
    cointegration_entry_z: float = 2.0     # |z| at/above which a pair is an actionable ENTRY
    cointegration_exit_z: float = 0.5      # |z| below which a pair is fair-value / no edge
    cointegration_pvalue: float = 0.05     # ADF significance level (0.01 | 0.05 | 0.10)

    # Cross-sectional ranking — measures how each ticker's per-method scores deviate from
    # the universe mean on each method, then averages the (capped) z-scores into a single
    # "stand-out" score per ticker. Composes additively into combined_score so the absolute
    # aggregation stays intact while the cross-sectional view adds a relative-value dimension
    # (robust to bull/bear regimes where absolute scores all skew one way).
    enable_cross_sectional: bool = True
    cross_sectional_weight: float = 0.20   # how strongly cs_score adjusts combined_score
    cross_sectional_zcap: float = 2.5      # cap individual z-scores to this magnitude

    # ── Multi-timeframe technical signals (30-min / daily / weekly) ───────────────
    # Every OHLCV-based method (tech, vwap, momentum, money_flow, trend_strength,
    # iv_rank, pattern, sector_momentum) is also computed on a faster 30-min candle
    # and a slower weekly candle. The three timeframe scores are BLENDED (weights
    # below, renormalised over whichever timeframes are available) into the live
    # combined_score, and every per-(method, timeframe) score is persisted to the
    # signals panel for the dashboard's 4-category Information-Coefficient table.
    # Master OFF ⇒ exactly the legacy daily-only behaviour.
    enable_multi_timeframe_signals: bool = True
    enable_intraday_30m: bool = True       # fetch + score the 30-min candle (Massive/Polygon → yfinance)
    enable_weekly_signals: bool = True     # resample the daily cache → weekly candle (free, no fetch)
    intraday_30m_lookback_days: int = 120  # 30-min history depth per Massive/Polygon fetch
    # Strategy blend weights across timeframes (renormalised at runtime over the
    # timeframes that actually produced a score for a given ticker). Daily-dominant
    # by default; set tf_blend_1d=1.0 to revert to daily-only without flipping the flag.
    tf_blend_30m: float = 0.20
    tf_blend_1d: float = 0.60
    tf_blend_1w: float = 0.20

    # ── Horizon synthesis (term-structure of edge) ─────────────────────────
    # Per-ticker edge curve: each method's LIVE score weighted by its MEASURED
    # per-horizon information coefficient (from the simulated_trades panel) — pure
    # IC, no static blend, SIGN-AWARE (a negative-IC method is flipped, a no-skill
    # one drops out). Evaluated at every simulated-trade horizon
    # (30m/3h/6h/1d/3d/1w/2w/1m), then COST-AWARE selection picks the holding
    # horizon whose net-of-cost expected gross return is highest. The LLM may
    # CONFIRM or SHORTEN it (never lengthen — enforced mechanically at trade time);
    # the matched exit raises the hold-review floor once a position outlives its
    # horizon window. Judge nothing until the IC panel is thick.
    enable_horizon_synthesis: bool = True
    horizon_ic_days: int = 120            # lookback window for the per-horizon IC matrix
    horizon_ic_min_n: int = 15            # min joint obs before a (method,horizon) IC is used
    horizon_min_conviction: float = 0.05  # min |edge(h)| for a horizon to be a trade candidate
    horizon_cost_hurdle_pct: float = 0.40  # round-trip cost the net edge must clear (~27-41bp)
    horizon_ic_cache_seconds: int = 1800  # reuse the heavy IC matrix across ticks (30 min)
    # Matched exit: once a position is held ≥ its target-horizon duration, multiply
    # the opener-pinned hold-review confidence floor by this so it must still be
    # STRONGLY confirmed to survive past its edge window (short-horizon trades exit
    # sooner; a still-conviction winner is not cut). Own flag so it can be disabled
    # independently of the horizon assignment + dashboard.
    enable_horizon_matched_exit: bool = True
    horizon_expiry_floor_mult: float = 1.5
    # The matched-exit floor is CONTINUOUS, not a cliff: past the horizon the
    # required re-confirmation floor ramps from the normal base floor (at the
    # window) up to base × horizon_expiry_floor_mult, reaching full strength
    # horizon_expiry_ramp_windows windows past expiry (1.0 → full effect at 2× the
    # horizon window; short-horizon trades still tighten fast, long-horizon trades
    # stay patient). A same-direction position whose conviction drops below the
    # ramped floor closes (horizon_expired). Set ramp_windows→0 for the old cliff.
    horizon_expiry_ramp_windows: float = 1.0
    # A neutral HOLD/WATCH re-judgment past the horizon does NOT close at the
    # boundary (that was a premature-cut bug — it contradicted the same
    # never-close-on-neutral rule the within-window gate enforces). Instead a
    # persistent neutral position is flushed only once FULLY past the window
    # (ramp saturated) — and only when this is on. False = the matched exit is
    # purely a ramped conviction bar and never force-closes a neutral hold.
    horizon_expiry_flush_neutral: bool = True
    # Fallback horizon label for positions with NO target_horizon (opened before
    # horizon synthesis existed, or a run where it failed). Without it those
    # positions have no time-stop at all: a persistent-neutral loser rides
    # forever (observed 2026-07-01: the June-17 cohort at −20% with HOLD-0.09
    # reviews). Must be a HORIZON_HOURS label ("1w" → flush at 2× = 14 days with
    # the default ramp). Empty string disables the fallback (legacy behavior).
    horizon_default_window: str = "1w"
    # Re-entry cooldown: skip a new entry when a SAME ticker + SAME direction
    # trade was closed within this many hours. Stops the close→reopen churn where
    # a rule-based exit (horizon_expired / llm_confidence_loss) fires and the same
    # tick's entry pass immediately reopens the position (observed 2026-06-29: HUM
    # closed 11:38:30, reopened 11:38:31 — a pure round-trip cost). 0 disables.
    # Opposite-direction entries (a genuine flip) are never blocked.
    reentry_cooldown_hours: float = 4.0
    # When the opener-pinned hold-review engine is unavailable (e.g. Anthropic
    # credits exhausted — observed 2026-06-26→07-01: Claude-opened positions went
    # unreviewed for days, leaving them with NO exit gate), re-judge the position
    # with the OTHER provider instead of holding blind. The review row records the
    # actual reviewing engine, so provenance stays honest. False = strict pinning
    # (no review ⇒ hold, the original Fix #2 behavior).
    hold_review_engine_fallback: bool = True

    # ── Direction-aware, market-neutral edge curve (SHADOW MODE) ───────────
    # A second edge curve that weights each method by its DIRECTION-CONDITIONAL,
    # MARKET-RELATIVE skill: a method's bullish calls and bearish calls are scored
    # separately (some methods are reliable one way only), on returns net of SPY's
    # same-horizon move (so market drift can't masquerade as directional skill).
    # Per-side skill = 2·(market-relative hit rate − 0.5), shrunk toward the
    # method's both-sides skill by sample size. SHADOW ONLY — it is computed,
    # persisted, and shown on the dashboard next to the live (pooled) horizon, but
    # does NOT drive entries or exits, so it can be validated against outcomes
    # before promotion.
    enable_directional_shadow: bool = True
    horizon_dir_shrink_prior_n: int = 30   # virtual both-sides obs the per-side skill shrinks toward
    horizon_market_benchmark: str = "SPY"  # market leg subtracted to neutralise drift

    # ── Expected-move / market-aligned upside ranking ──────────────────────
    # Selection should favour the names with the biggest EXPECTED FAVOURABLE MOVE
    # (probability × magnitude) in the MARKET's direction: when the regime is
    # risk-on, the highest-upside longs; when risk-off, the biggest-downside shorts
    # (beta as a deliberate tailwind, decided by the regime layer — not the
    # drift-contaminated stock skill). expected_move = the edge curve's gross
    # expected favourable return at the target horizon; upside = conviction ×
    # expected_move × an alignment factor. SOFT: counter-market candidates are
    # haircut, not banned. Fed to the synthesis prompt + persisted; does not change
    # the actionable gate mechanics.
    enable_expected_move_ranking: bool = True
    horizon_counter_market_mult: float = 0.5   # upside haircut for a position fighting the regime
    # 0 = attempt every ticker (full coverage). A positive cap throttles the fetch;
    # only needed on the yfinance fallback path (≤60d history, per-IP 429s) — the
    # Massive/Polygon Advanced plan is unlimited-rate, so leave at 0 when configured.
    intraday_30m_max_tickers: int = 0
    # Re-fetch the 30-min OHLCV cache when its newest bar is older than this (minutes).
    intraday_30m_ttl_minutes: int = 25
    # In-process memo of PARSED OHLCV frames (cache.load_ohlcv), bounded in MB.
    # pd.read_json is the floor cost of every analytics sweep and the dashboard's
    # panels re-read the same tickers in both timeframes, so this is sized to hold
    # the whole working set — otherwise one panel re-parses exactly what the
    # previous one just evicted. Entries are keyed on (path, mtime, size) so a
    # pipeline rewrite invalidates them automatically, and each caller gets a COPY,
    # so this is safe in the writer process too. 0 disables the memo (always re-parse).
    #
    # RAISED 160 → 400 on 2026-08-04. The old value was sized when daily frames
    # averaged ~8 KB; the 5-year OHLCV backfill (median 106 → 1,257 bars/ticker)
    # made them ~37 KB, and the MEASURED working set is now:
    #     daily   37 KB x 3,647 files = 134 MB
    #     30-min  51 KB x 2,266 files = 116 MB   → 250 MB total
    # At 160 MB the memo could hold barely half of that, so any full-universe
    # sweep thrashed: evict, re-parse, evict again. That is a LIVE tick cost, not
    # just an analysis one — the scorers sweep OHLCV ~7,500 times per tick.
    # 400 MB leaves headroom for further backfill depth and is ~1.5% of this
    # machine's RAM (66 GB, 27.6 GB free), even counting the dashboard's separate
    # copy. Re-measure with cache.ohlcv_parse_cache_stats() if the cache grows again.
    # 1200 MB since 2026-08-11: the 20y backfill made frames ~4x bigger and the
    # 400 MB bound (~1,000 frames) thrashed against a ~3,400-ticker universe x
    # multiple consumers — measured as 10-24 min ticks. ~1200 MB keeps the whole
    # active universe's parsed frames resident.
    ohlcv_parse_cache_mb: int = 1200
    # Per-ticker scoring concurrency. The build_signals loop is I/O-bound (DeepSeek
    # sentiment ~7s/ticker + Massive/OHLCV reads), so a bounded thread pool collapses
    # the serial sum to ~max wall-time with IDENTICAL scores. 1 = sequential (legacy).
    # DeepSeek (sentiment = deepseek-v4-flash) caps at 2500 CONCURRENT, not RPM, and
    # throttles by latency, not 429s — so 32 still has huge headroom. Raised 16→32
    # (2026-07-08): Step 4 wall time scales ≈ tickers ÷ workers (measured 229s at
    # 433 tickers / 16 workers) — the latency-profile work. Watch for DeepSeek /
    # yfinance 429s in the logs; revert per-deploy via the env var if they appear.
    signal_scoring_max_workers: int = 32

    # ── Tick→order latency levers (2026-07-08 latency profile) ──────────────
    # Run the opener-pinned hold-review CONCURRENTLY with Steps 4–5 instead of
    # strictly after Step 5 (the review needs no main-synthesis output — it
    # refetches its own news/prices and re-judges with the OPENING engines; its
    # pinned synthesis still waits for the completed synthesis context, so the
    # review prompt is identical to the sequential path). Saved ~2.5 min/tick
    # measured. False = legacy sequential review (same code path, inline).
    # NOTE: the ledger-mark refresh (calibrate_sim_costs + update_open_trades)
    # runs before Step 4 in BOTH modes now — marks are stamped ~5 min earlier.
    enable_hold_review_overlap: bool = True
    # Cache the raw per-ticker sentiment LLM verdict keyed by (ticker, engine,
    # exact article set). A new article changes the key → fresh score, so news
    # reactivity is unchanged; only re-scoring an IDENTICAL digest is skipped
    # (the next 30-min tick, and the hold-review re-scoring held names minutes
    # after the main pass). TTL bounds staleness of the digest's age labels.
    enable_sentiment_cache: bool = True
    sentiment_cache_ttl_minutes: int = 180
    # Bounded concurrency for the per-ticker yfinance options-chain scan (the
    # fetch pool's slowest source, ~64s median). Kept LOW deliberately: the scan
    # shares yfinance's unofficial per-IP rate limit with the GEX pass (which
    # stays sequential after it) — history shows ~20+ req/s combined causes 429s
    # on every ticker. 1 = sequential (legacy).
    options_flow_max_workers: int = 2
    # Bounded concurrency for the discovery liquidity gate's COLD OHLCV warm-up
    # fetches (Polygon-first, so per-IP rate limits are not a concern; yfinance
    # is only the fallback). The sequential loop was the 7-minute stall on the
    # first tick after midnight (~200 uncached smart-money names). 1 = legacy.
    liquidity_gate_fetch_workers: int = 8

    # ── Classic cross-sectional anomalies (2026-07-08, panel-first) ─────────
    # Three literature-proven OHLCV-only methods (signals/classic_anomalies.py):
    # 52-week-high proximity (George-Hwang 2004 continuation), 12-1 skip-month
    # momentum (Jegadeesh-Titman 1993), and short-term reversal (Lehmann 1990,
    # 1-week, sign-flipped, liquid names only). Scored on every ticker and
    # IC-tracked in the signals panel at ZERO combine weight — promotion into
    # combined_score is a later, evidence-gated code change, not a flag flip.
    enable_high_52w: bool = True
    enable_momentum_12_1: bool = True
    enable_st_reversal: bool = True
    # Liquidity floor for the reversal signal (20-day average dollar volume) —
    # deliberately far ABOVE the $5M trade floor: below institutional size the
    # measured "reversal" is mostly bid-ask bounce, not a real snapback.
    st_reversal_min_dollar_volume: float = 50_000_000
    # Mean-reversion additions (2026-08-10, panel-first at weight 0 — selected
    # by the 20y full-history battery, de-correlated cluster winners on Gate-4
    # names; see memory/pivot-horizon-target-2026-08.md). Both share the
    # st_reversal liquidity floor (same bid-ask-bounce argument).
    enable_rsi2_rev: bool = True        # Connors RSI(2) snapback
    enable_dloc_rev: bool = True        # daily candle-location reversal

    # ── Tier-2 panel-first methods (2026-07-08, weight 0 — same contract) ───
    # TTM Squeeze: BB(20,2σ) coiling inside Keltner(20,1.5×ATR); the release
    # fires in the direction of Carter's momentum oscillator (ttm_squeeze.py).
    enable_ttm_squeeze: bool = True
    # IV term-structure slope: front vs back ATM IV captured for free during
    # the GEX chain fetch (needs enable_gex for coverage; sparse ⇒ no view).
    # Backwardation = near-term event/stress premium → bearish tilt.
    enable_iv_term_structure: bool = True
    # Anchored VWAP from the 52-week high/low anchor days — POSITIONING read
    # (above the anchor = support = bullish), deliberately the opposite
    # convention of the mean-reversion rolling `vwap` method.
    enable_anchored_vwap: bool = True

    # ── Tier-3 panel-first methods (2026-07-08, weight 0 — same contract) ───
    # Residual momentum (Blitz-Huij-Martens 2011): 12-1 momentum on the
    # residual of a true-beta regression vs SPY — unlike sector/market
    # momentum's implicit beta of 1, a high-beta name in an up-market gets no
    # free momentum credit (residual_momentum.py).
    enable_residual_momentum: bool = True
    # Volume profile: 60-day volume-at-price histogram → POC + 70% value area;
    # acceptance outside value = trend score, inside value = small POC gravity
    # (volume_profile.py).
    enable_volume_profile: bool = True

    # Business Cycle Rotation — Fidelity-style structural economic cycle phase → sector biases.
    # Derives EARLY_EXPANSION|MID_EXPANSION|LATE_EXPANSION|LATE_CYCLE|CONTRACTION from the
    # already-fetched FRED macro context (regime, yield curve, inflation, unemployment).
    # Pure synthesis module: no new API calls, no cache, instant computation.
    enable_business_cycle_rotation: bool = True

    # Catalyst Timing — three event-driven guards and amplifiers:
    #   1. Earnings Blackout: block BUY/SELL for tickers within 2 days of earnings (IV crush/gap risk)
    #   2. OpEx Max-Pain Amplifier: boost max_pain weight during OpEx week (0.20) / Triple Witching (0.28)
    #   3. 8-K + Insider Buy: auto-elevate to WATCH when both signals coincide for the same ticker
    enable_catalyst_timing: bool = True

    # ── Open-position monitoring: signal-decay exits + MFE/MAE tracking ──────
    # For every open trade, every pipeline tick re-evaluates the per-ticker
    # signal against today's data and exits early when the thesis has
    # materially deteriorated — even when no counter-direction recommendation
    # appears in the day's top-10 (which is the gap close_trades_on_signal_reversal
    # leaves open). Exit triggers, in order of severity:
    #
    #   1. signal_flipped     — today's combined score crossed against the trade.
    #                           For a BUY this means today's combined < flip_threshold
    #                           (e.g., -0.10). For a SELL, > -flip_threshold.
    #   2. signal_decay       — signal weakened by more than drop_threshold from entry.
    #                           Sized in oriented combined-score space so a BUY at
    #                           +0.65 dropping to +0.10 = 0.55 decay > 0.40 threshold.
    #   3. confidence_loss    — today's confidence dropped below confidence_floor.
    #   4. macro_regime_exit  — macro regime flipped to PANIC/RISK_OFF while long;
    #                           mirrors the existing entry-side block.
    #
    # MFE (max favorable excursion) and MAE (max adverse excursion) are tracked
    # passively on every tick — pure observability, doesn't trigger anything.
    # Fix #2 — symmetric entry/exit rationale. When True (default), an LLM-opened
    # position is HELD/CLOSED by its OWN opening engine's fresh re-judgment each
    # run (the same synthesis call that gates entry), and ONLY on a tick that
    # engine is running — a Claude position is never closed by a DeepSeek run, and
    # vice versa. The aggregator's combined_score/confidence (a different
    # decision-maker, near zero on LLM-conviction names) is NO LONGER consulted
    # for LLM-opened trades; it survives only as the backstop for legacy /
    # rule-based-opened trades (gated by enable_signal_decay_exits below). Set
    # False to revert entirely to the aggregator-driven monitor.
    enable_llm_hold_review: bool = True
    # Every-tick, opener-pinned hold-review. When True (default), each open
    # LLM-opened position is re-evaluated on EVERY trading tick by the SAME
    # synthesis AND sentiment engines that opened it — fresh news + prices are
    # refetched, the ticker's signal is re-aggregated with the pinned sentiment
    # engine, and the pinned synthesis engine re-judges it — so entry-vs-now is a
    # same-engine, apples-to-apples comparison (temp=0 ⇒ low volatility). When
    # False, falls back to reviewing a position only on ticks whose A/B engine
    # happens to match the opener (the cheaper, partial behaviour). Costs extra
    # LLM calls per tick (one synthesis + pinned sentiment per engine combo held).
    enable_pinned_hold_review: bool = True
    enable_signal_decay_exits: bool = True
    signal_decay_flip_threshold: float = -0.10   # oriented combined < this -> flipped
    signal_decay_drop_threshold: float = 0.40    # oriented (entry - today) > this -> decayed
    # ── Confidence-loss floor — entry-relative with absolute backstop ──
    # Effective floor = max(signal_decay_confidence_floor,
    #                       signal_decay_confidence_floor_relative × entry_confidence)
    # where entry_confidence is the aggregator-confidence captured at trade entry
    # (signal_at_entry.confidence, NOT Claude's adjusted confidence — apples-to-apples
    # with today's aggregator confidence).
    #
    # Why entry-relative: a fixed 0.60 floor was firing on routine day-to-day signal
    # variance (78% entries decaying to 55% next day is normal noise), causing every
    # trade to exit on confidence_loss before the thesis got room to develop. Tying
    # the floor to entry conviction means a high-conviction trade gets a tighter
    # floor and must really collapse to trigger exit, while a borderline entry gets
    # a looser floor that tolerates routine variance. The absolute backstop catches
    # genuine conviction collapse.
    #
    # 2026-07-11 recalibration (post-exit forward-return monitor, exit_forward.py):
    # at 0.65 the relative leg bound at ~0.52-0.65 and fired on reviews in the
    # 0.50-0.62 band — closes that kept running +2.5% @1d / +7.2% @5d / +12.6% @10d
    # in the position's direction (n=19). Sweeping candidate floors over those
    # closes: 0.55 was the clean split — what it keeps ran +9.2% @5d, what it still
    # fires (conviction ≤~0.45-0.50) genuinely bled (−6.2% @5d, ADBE −12% @10d
    # avoided). The all-reviews panel agrees: NO confidence level separates
    # hold-profitable from bleed (exit_floor_calibration boundary = None at 1d AND
    # 5d over 162/67 review-days; the 0.5-0.6 bucket runs +6.8% @5d) — review-
    # confidence LEVEL is ~uninformative, so the floor should only catch collapse,
    # not lukewarmness. Freed trades stay governed by the flip / trailing-stop /
    # mechanical / horizon-ramp / macro exits (horizon_expired measures GOOD:
    # −1.4% @5d). Re-measure via `python -m src.analysis.exit_forward` as closes
    # accrue; the absolute stays calibration-adaptive (exit_floor_calibration).
    signal_decay_confidence_floor: float = 0.45          # absolute hard backstop
    signal_decay_confidence_floor_relative: float = 0.55  # factor × entry_confidence (2026-07-11: 0.65→0.55, measured)
    signal_decay_regime_exit: bool = True        # exit longs in PANIC/RISK_OFF

    # ── Adaptive signal weighting ────────────────────────────────────────────
    # Multiply each method's static weight in the aggregator by a per-method
    # multiplier derived from its rolling solo win rate (data from
    # ``tracker.compute_solo_method_performance``). Methods that have been
    # right historically get up-weighted; methods that have underperformed
    # 50% get down-weighted. Bayesian shrinkage with ``prior_n`` virtual
    # trials at 50% smooths small-sample noise so 3-for-3 doesn't immediately
    # blow up to 2× weight.
    #
    # Formula:
    #   shrunk_wr = (wins + 0.5 × prior_n) / (n + prior_n)         in [0, 1]
    #   raw_mult  = shrunk_wr / 0.5                                 → 1.0 at 50% WR
    #   final     = clip(raw_mult, min_multiplier, max_multiplier)
    #
    # The multiplier is applied on top of whichever weight_profile is active
    # (base, market-mode override, or opex amplifier). So adaptivity composes
    # with regime-aware weighting rather than overriding it.
    enable_adaptive_weights: bool = True
    adaptive_weight_prior_n: int = 10              # virtual trades at 50% WR (Bayesian prior)
    adaptive_weight_min_multiplier: float = 0.5    # floor: a bad method keeps at least half its baseline
    adaptive_weight_max_multiplier: float = 2.0    # cap:   a great method gets at most double

    # ── Win-rate method filter (HARD exclusion; panel monitoring unaffected) ──
    # A hard version of the adaptive-weight tilt above. Any method whose GROSS solo
    # win rate (tracker.compute_solo_method_gross_winrate — "if this method alone had
    # decided the trade, did it pick the right DIRECTION on the raw price move?",
    # BEFORE fees and spread — costs are an execution concern, not a measure of signal
    # quality) sits below winrate_filter_threshold, ONCE it has at least
    # winrate_filter_min_trades attributed solo trades to judge it, is DROPPED from:
    #   • combined_score — weight → 0; the surviving methods renormalise so the
    #     score stays on the same scale (not just shrunk like the adaptive floor),
    #   • coherence / sources_agreeing (and therefore the confidence it feeds),
    #   • the synthesis prompt — the LLM never sees a sub-coin-flip method's
    #     per-ticker score line, so it can't lean on it.
    # The method is STILL scored and persisted to the `signals` panel + trade
    # attribution every run, so IC / win-rate / simulation monitoring keeps
    # accruing and a filtered method can re-earn its place (or be inverted) once
    # the evidence turns. Composes with the soft adaptive/IC tilts (a filtered
    # method is simply held out of the pool before they apply).
    # Two guards keep it safe:
    #   • min_trades — never drop a method on small-sample noise (a 1-loss method
    #     is not a bad method); below the floor the method keeps its full weight.
    #   • INVERTED methods (inverted_methods) are EXEMPT — their sign is already
    #     corrected in the combine, so a sub-50% RAW win rate is exactly WHY they
    #     are useful, not a reason to drop them.
    # And build_signals never lets the filter zero out EVERY active method (it is
    # suppressed for that run if it would, so the book can't go all-NEUTRAL).
    enable_winrate_method_filter: bool = True
    winrate_filter_threshold: float = 0.50   # drop a method whose solo win rate is below this…
    winrate_filter_min_trades: int = 10      # …once it has at least this many attributed solo trades

    # ── Per-SIDE win-rate filter + weighting (2026-07-24) ────────────────────
    # A method's bullish and bearish calls are separate skills, and the audit
    # measured them far apart: on 237 attributed trades most methods hit ~40% on
    # their BUY-side views and ~47-59% on their SELL-side views (e.g. insider
    # 37.7/51.9, iv_rank 43.9/54.5, st_reversal 42.9/64.2). One blended win rate
    # per method hides that, so the filter above either keeps a method whose long
    # calls are a coin-flip-loser, or drops one whose short calls are genuinely
    # good. With the buy/sell split combine these are separable: each camp is
    # filtered and weighted on ITS OWN record.
    #   • side filter — a method is dropped from the BUY camp when its buy-side
    #     gross win rate is below winrate_filter_threshold (with at least
    #     winrate_filter_min_trades buy-side views), and independently from the
    #     SELL camp on its sell-side record. Same exemptions as the global
    #     filter: inverted methods are never dropped, thin samples keep full
    #     weight.
    #   • side weights — within a camp, each surviving method's weight is scaled
    #     by its demonstrated skill on THAT side, Bayesian-shrunk toward 50% by
    #     side_weight_prior_n so a handful of views can't dominate.
    # NOTE this can legitimately leave a side with NO qualifying methods, in
    # which case that side's score is 0 and the system simply does not signal
    # in that direction — the intended consequence of the evidence, not a bug.
    enable_side_winrate_filter: bool = True
    enable_side_adaptive_weights: bool = True
    side_weight_prior_n: int = 10             # virtual views at 50% (Bayesian prior)
    side_weight_min_multiplier: float = 0.5   # floor for a weak-but-surviving side
    side_weight_max_multiplier: float = 2.0   # cap for a strong side

    # ── Cross-family agreement + tape confirmation (2026-07-19) ──────────────
    # Upgrades to agreement quality (src/signals/agreement.py). The flat
    # sources_agreeing count treats every method as an independent voter, but the
    # OHLCV-derived technicals all read the same tape (pseudo-replication) —
    # METHOD_FAMILIES groups the weighted pool into 7 independent INFORMATION
    # families (Sentiment / Price-Trend / Rel-Strength / Volume-Flow / Options /
    # Smart-Money / Event-Arb); per-family magnitude-weighted votes are counted at
    # the FAMILY level (a family votes only when |family score| ≥
    # family_vote_threshold — no more hairline-0.01 "agreement"), exposed to the
    # synthesis prompt per ticker (which families align/oppose, with values), and
    # folded into confidence via a modest breadth-of-independent-confirmation
    # factor ∈ [1−span, 1+span] (1 family alone → 1.0 neutral; 4+ aligned → the
    # +span cap; an opposing family drags 1.5× as hard as an aligned one helps).
    # Filtered (win-rate) methods are excluded; inverted methods vote with their
    # corrected sign (what the combine consumes).
    enable_family_agreement: bool = True
    family_vote_threshold: float = 0.05
    family_agreement_factor_span: float = 0.12

    # Tape confirmation — a SCORE-INDEPENDENT raw price/volume state check from
    # the cached daily OHLCV (20d range position + up/down-day volume share 10d +
    # last-bar RVOL signed by its direction → composite ∈ [-1,+1], + = bullish
    # structure). Not an alpha method in the combine — an agreement QUALIFIER:
    # multiplies confidence by 1 + span×(tape × direction) when it confirms /
    # diverges from the combined direction, rides the synthesis prompt as a
    # per-ticker TAPE STRUCTURE line, and is persisted to the signals panel as
    # the `tape` pseudo-method so its forward IC is monitored like any method.
    # Cache-only (never fetches; cold cache → NO_DATA → neutral 1.0).
    enable_tape_confirmation: bool = True
    tape_confirmation_factor_span: float = 0.08

    # ── Agreement floor (Gate 1b; 2026-07-20) — DECOMMISSIONED OFF 2026-08-17 ──
    # Added 2026-07-20 to mechanically enforce the then-documented "a single strong
    # source never produces a BUY/SELL" invariant (previously only a prompt
    # instruction): drop any BUY/SELL whose sig.sources_agreeing <
    # min_sources_agreeing_gate whenever the recommendation ECHOES the aggregator's
    # direction (an LLM override passes unchecked — the count isn't attributable to
    # a call the aggregator didn't make).
    # SWITCHED OFF 2026-08-17 (user directive: remove gates not helping) on the
    # pivot-basis gate funnel, measured twice: the cohort it DROPS outperforms the
    # cohort it KEEPS (gate value −0.168pp on 45d/153 drops, re-run −0.062pp on
    # 46d/158 drops — wrong-signed both times, never right-signed; not significant,
    # like everything in the cascade, but the only gate whose point estimate
    # SUBTRACTS). Under the rank-basis direction bands it is nearly vestigial
    # anyway (0.7% of candidates in the current era — a name clearing the
    # top-decile/bottom-5% band almost always has ≥2 agreeing sources). The prompt
    # instruction remains; the mechanical gate reverts to pre-07-20 behaviour.
    # Machinery kept revivable: flip this on to restore the gate; the floor below
    # is its threshold. gate_diag.dropped_low_agreement stays (reads 0), and
    # sources_agreeing is still computed/persisted everywhere (it feeds coherence,
    # family votes and the panel — only the GATE is off). See
    # memory/gate-funnel-pivot-2026-08.md.
    enable_agreement_gate: bool = False
    min_sources_agreeing_gate: int = 2

    # ── Gate 1c — per-run RANK CAP on Gate-1 passers (2026-08-21) ─────────────
    # Among the BUY/SELL calls clearing the regime confidence floor, keep only
    # the run's top-K by stated confidence; the rest are DEFERRED ("rank_capped"
    # in gate_outcomes — they re-qualify any tick they make the cut). WHY BOTH
    # HALVES: the absolute floor preserves ABSTENTION (an empty run is
    # informative — Gate 1 is the funnel's one right-signed gate) but inherits
    # the LLM's confidence SCALE, so the trade rate rides calibration drift
    # (2026-08-17: the prompt incident DOUBLED it with no signal change). A pure
    # rank gate is scale-immune but forces trades on weak runs — measured WORSE
    # (-0.36..-0.50 %/day paired, t -1.0..-1.5). The hybrid measured
    # selection-NEUTRAL (-0.027 %/day, t -0.30, 44 days) while cutting the
    # trade-rate cv 0.46 -> 0.28 — free precisely because within-run confidence
    # rank carries no information (per-run rank IC -0.023, t -1.17). K=3 is the
    # measured point; 0 disables (pure absolute gate, pre-2026-08-21 behavior).
    # See memory/ranking-stage-map-2026-08.md.
    gate1_rank_cap: int = 3

    # ── Prompt shortlist key (claude_analyst top-40 cut; 2026-08-21) ──────────
    # "confidence" (live default) | "camp_max". Confidence measured
    # ANTI-selective on the pivot basis (random-40 beats it, 30/30 seeds);
    # camp_max — max(|combined_buy|,|combined_sell|) — is the best measured
    # replacement (+0.43%/run, t +1.70) but its columns have only 22 days of
    # history, all one window, so it CANNOT yet clear the replication bar.
    # Flip to "camp_max" once an independent-half test exists (~2026-09-15).
    # Full rationale at the claude_analyst._shortlist_key definition.
    prompt_shortlist_key: str = "confidence"

    # ── IC-informed adaptive weights (panel-driven; ON, but CONFIDENCE-GATED) ──
    # A better-founded sibling of the win-rate layer above. Instead of solo win rate
    # from the gate-selected (thin, biased) trade ledger, it tilts each method's weight
    # by its RELIABILITY-ADJUSTED information coefficient — ICIR = mean(daily IC)/
    # std(daily IC) — measured over the UNBIASED `signals` panel (every scored ticker,
    # not just the trades the gates let through). By default the IC is MARKET-NEUTRAL
    # (shadow basis — ticker return minus SPY, pooled both directions), so a method is
    # weighted by its regime-robust SELECTION skill (alpha), NOT by how much market beta
    # it happened to ride in the lookback window (which may not repeat if the regime
    # turns); the market-direction call is owned separately by the regime/mode layers.
    # Set ic_weight_basis="edge" to weight by absolute-return IC instead. By design:
    #   • CONFIDENCE GATE: a method is reweighted ONLY when its mean-daily-IC t-stat
    #     clears the bar — t = |ICIR|·sqrt(n_days) ≥ ic_weight_min_t (t≥2 ≈ 95%). A
    #     method whose IC is not yet statistically distinguishable from zero keeps its
    #     base weight untouched. So on a thin panel NOTHING clears the gate and this is
    #     a pure NO-OP; methods earn a tilt only once the data proves their IC is real.
    #     (This is what makes it safe to run live now — it respects "judge nothing on a
    #     thin panel" automatically, then reweights method-by-method as each matures.)
    #   • Positive-only among the confident: a confident method's boost = clip(ICIR /
    #     median(confident positive ICIR), min, max) — the typical confident method is
    #     unchanged (1.0×), better ones boosted, weaker-but-still-positive ones trimmed.
    #     A confidently NON-positive method is floored to ic_weight_min_multiplier (it
    #     is anti-predictive — minimise it until the inversion switch flips its sign).
    # Applied as a multiplier on the active weight profile, exactly like the win-rate
    # layer (the two STACK if both on; set enable_adaptive_weights=false to use IC alone).
    enable_ic_weights: bool = True
    ic_weight_basis: str = "shadow"        # "shadow" = market-neutral IC (alpha, regime-robust) | "edge" = absolute-return IC
    ic_weight_horizon_days: int = 5        # EDGE-basis forward horizon in sessions (≈ a typical hold)
    # "pv" = the signed PIVOT target (return to the next pivot — the promotion
    # basis) since 2026-08-12 (user directive: pivot metrics for every DECISION
    # evaluation; fixed horizons stay for monitoring/holding/exits). Any fixed
    # label (30m|3h|6h|1d|3d|1w|2w|1m) restores the old basis.
    ic_weight_shadow_horizon: str = "pv"
    ic_weight_min_days: int = 5            # min distinct signal-days before an ICIR is even computed
    ic_weight_min_per_day: int = 5         # min cross-section per day for that day's IC to count toward ICIR
    ic_weight_min_t: float = 2.0           # CONFIDENCE GATE: reweight only if |ICIR|·sqrt(n_days) ≥ this
    ic_weight_min_multiplier: float = 0.25 # floor for a confidently anti-predictive method
    ic_weight_max_multiplier: float = 3.0  # cap on a strongly-predictive method's boost
    # Reuse the heavy panel calibrations (IC weights, win-rate filter, per-side
    # skill, market-relative skill) across ticks and hold-review calls.
    # RAISED 1800 → 7200 on 2026-08-04: 1800 s was EXACTLY the RTH tick interval,
    # so "reuse across ticks" landed on the expiry boundary and the heavy work was
    # effectively redone every tick. These calibrations move far more slowly than
    # that — one tick adds ~400 rows to a ~16,000-row panel (2.5%) and ~1 trade to
    # ~356 (0.3%), and every one of them is Bayesian-shrunk over weeks — so a
    # 30-minute versus 2-hour refresh is statistically indistinguishable while
    # costing 4× less. EOD maintenance clears these caches explicitly, so a fresh
    # panel/retrain is picked up at once rather than waiting out the TTL.
    ic_weight_cache_seconds: int = 7200
    # Method INVERSION (manual, evidence-driven) — comma-separated method names whose
    # RAW score is reliably anti-predictive NET OF BETA across horizons (confirm via
    # `python -m src.analysis.scorecard` or `simulated_trades --directional`: a side
    # whose market-relative ICIR is confidently negative and STAYS negative across
    # horizons). Such a method contributes with a FLIPPED sign in combined_score — a
    # reliably-backwards signal, corrected, is a reliably-right one — and is lifted
    # off the IC anti-predictive floor to base weight. The signals panel keeps the RAW
    # score, so the inversion stays re-validatable (raw IC still shows backwards ⇒
    # keep inverting; raw IC flips positive ⇒ remove it). Empty = none. Set via
    # INVERTED_METHODS in .env — a reversible ops call, not a baked-in regime bet.
    inverted_methods: str = ""

    # ── Automatic method inversion (2026-07-25) ───────────────────────────
    # Sign-flip a method whose RAW score is reliably anti-predictive, without a
    # human editing INVERTED_METHODS. Two bars must BOTH clear:
    #   1. one-sided exact binomial vs p=0.5 at `inversion_alpha`, Bonferroni-
    #      corrected across the methods judged that run (testing ~20 methods at
    #      0.05 uncorrected produces a false inversion ~2/3 of the time);
    #   2. the FLIPPED view must itself clear `winrate_filter_threshold` — ties
    #      count as losses BOTH ways, so flipped != 100-raw, and a mostly-tied
    #      method is noise rather than backwards information.
    # Manual `inverted_methods` pins are unioned in and always win.
    enable_auto_inversion: bool = True
    inversion_min_trades: int = 30      # evidence floor; higher than the filter's
                                        # 10 — inverting claims more than dropping
    inversion_alpha: float = 0.05       # plain per-method significance, no correction
    # NO multiple-comparison correction, deliberately (2026-07-26). We do run one
    # test per method per run, but a correction computed from the COUNT of tests
    # is indefensible here: the methods are heavily correlated (measured +0.72
    # mean pairwise Spearman over 246k scored ticker-days; momentum vs
    # market_momentum is 0.98 — the same signal twice). Bonferroni assumes
    # INDEPENDENCE and so over-corrects, dividing alpha by 13 when the effective
    # number of independent tests is nearer 2-3 (it selected nothing at all);
    # plain alpha has the mirror flaw, reporting 6 "findings" that are really one
    # phenomenon counted six times. The count is simply not the right
    # denominator, so no alpha setting fixes it.
    #
    # Replication replaces it: each method gets ONE plain test on its own full
    # history, and the finding must reproduce on the INDEPENDENT panel arm
    # (different data, ~1000x the observations, different statistic, its own
    # shared-decay control) before anything is inverted. That handles correlated
    # methods by construction — six correlated ledger hits that don't reproduce
    # are one unconfirmed phenomenon — and it also dampens the multiplicity that
    # alpha-correction never addressed at all: we re-evaluate every run (~28x a
    # day), so a method parked near the boundary crosses it eventually by chance.
    inversion_require_replication: bool = True

    # ── Per-method horizon skill (2026-07-26) ─────────────────────────────
    # Where each method's edge actually LIVES, measured over the solo-method
    # simulation (`simulated_trades`) gated at buy_sell_diff_threshold — the gate
    # a single method driving combined_score would really face. ~1,500-3,300
    # observations per method versus 78-174 attributed trades, which is the
    # difference between "cannot tell" and an answer.
    #
    # Judged at 1 DAY ONLY, exactly one method cleared p<0.05 and the book would
    # have collapsed to a single signal. Across 1/3/5/10 days five methods clear,
    # with coherent shapes: sent_velocity peaks at 1d and decays (it measures a
    # RATE OF CHANGE, so that is its expected profile), pattern builds
    # monotonically and clears at 5d AND 10d, oi_skew is slowest. A method is not
    # good or bad — it is good over a particular holding period.
    # Actionable threshold as a CONTINUOUS function of the macro composite
    # rather than five discrete steps (2026-07-27). The step table is a cliff:
    # composites of -0.801 and -0.799 sit in different regimes, and the live
    # composite has sd 0.146 — so that boundary is well inside one standard
    # deviation of ordinary daily variation and the same market, sampled two
    # ticks apart, could jump the bar a full step. The curve reproduces the
    # documented threshold EXACTLY at every band boundary and interpolates
    # between, so only the discontinuity goes. The regime LABEL and the BUY
    # block remain step-based — those are state decisions, and a half-blocked
    # BUY is not a meaningful thing. false → the legacy step table.
    # RISK_OFF takes a size HAIRCUT instead of the outright BUY ban it shared
    # with PANIC (2026-07-27). The 7-input reconstruction over 2000-2026 found
    # RISK_OFF to be the BEST-performing regime (+2.25% SPY at 21d, 70.0% up
    # over 337 days) while PANIC is the only bad one (-0.69%, 45.5% over 66) —
    # the system was sitting out its strongest state. A haircut rather than
    # full size because RISK_OFF is a stressed regime by definition and its
    # days are concentrated in a few episodes; the ban stays for PANIC.
    # Method WEIGHTING uses market-relative win rates (2026-07-27). An absolute
    # win rate cannot separate "this signal works" from "the market went up":
    # measured on 296k ticker-days, a low-volatility screen won 53.5% at 5 days
    # and only 46.1% net of SPY. Scoped to weighting only — the hard filter and
    # all P&L/sizing stay absolute, because the book is outright long/short and
    # alpha you cannot capture must not drive sizing.
    #
    # The neutral point is the MEASURED baseline (~48.6%), not 50%: the
    # cap-weighted index beats its typical constituent, so the median stock is
    # market-relative-negative. A half-migrated basis (relative numerator, 50%
    # bar) is worse than either pure basis.
    # Mask confidence values produced before the buy/sell split combine
    # (2026-07-22) — see method_epochs.CONFIDENCE_EPOCH. The column mixes
    # formulas across the panel's life; masking is the same remedy the scorer
    # epochs use, and was chosen over a RETROFIT because 78% of rows predate the
    # component capture, the OHLCV cache is retroactively split-adjusted, and
    # today's weights are calibrated FROM this panel (rescoring the past with
    # them would be look-ahead into the dataset that feeds live sizing).
    enable_confidence_epoch: bool = True

    # Restore-before-mask (src/analysis/replay.py). Where a method is faithfully
    # replayable from the cached OHLCV, a superseded score is REGENERATED by the
    # current scorer rather than blanked by the epoch mask — so a calibration
    # fits values today's code produced without discarding the row. Serves the
    # materialised `signals_replay` table; off => pure epoch masking (the
    # pre-2026-07-27 behaviour). The mask remains the correctness backstop for
    # everything replay cannot regenerate.
    enable_panel_replay_restore: bool = True
    # Populate `signals_replay` during EOD maintenance so a scorer change is
    # reflected without a manual `python -m src.analysis.replay --write`.
    enable_eod_replay_refresh: bool = True
    eod_replay_refresh_days: int = 45       # 0/None => all history
    # Tier-2 sibling (2026-08-20): refresh `signals_backtest` — the CURRENT
    # entry architecture recomputed over history — on the same trailing span,
    # right after the walk-forward step. Auto-refactor already rewrites it when
    # scorer CODE moves; this covers plain new data, so "what would today's
    # strategy have decided" stays queryable without a manual
    # `python -m src.analysis.backtest --write`. Firewalled analysis surface;
    # never feeds calibrations.
    enable_eod_backtest_refresh: bool = True
    # Walk-forward SHAPE history (2026-08-21): append today's as-of rank-shaping
    # curves to `shape_history` at EOD — `weight_history`'s sibling for the
    # fitted consumption layer, consumed by the tier-2 backtest's wf mode so
    # history is never scored through curves fitted on it (the 2026-08-20
    # in-sample finding: today's curves alone flip the backtest's pivot IC from
    # −0.041 to +0.049). Backfill: `python -m src.signals.rank_shaping
    # --materialize`.
    enable_eod_shape_history: bool = True

    # Walk-forward calibration (src/analysis/walkforward.py). Appends today's
    # point-in-time weight state to `weight_history`, which is what lets a
    # backtest use the weights the system WOULD have had on each date instead of
    # applying today's to all history (measured: the book genuinely ran 18
    # active methods in late June vs 10 now, so a fixed-weight backtest
    # misstates that period wholesale). Incremental — a step is a fixed fact
    # once its data is in, and each costs ~60s, so only the tail is rewalked.
    enable_eod_walkforward: bool = True
    walkforward_eod_days: int = 3           # tail rewalked each EOD

    # ML OHLCV model (signals/ml_model.py) — a trained GBM wired as a PANEL-FIRST
    # method (weight 0: scored + persisted + IC-tracked, zero combine impact) so
    # its survivorship-free forward IC accrues before it can earn a weight.
    # v2 (2026-08-08, current): signed-pivot within-day-rank LightGBM regressor,
    # 85 features, full universe — cleared its pre-registered forward-panel gate
    # (`ml_validate --target pivot`: IC +0.0639, t +2.31, edge +2.58pp). The v1
    # rel-10d conditioned classifier (its gate: rel-10d IC +0.08, hit 55%) remains
    # the fallback serving path for old artifacts. `enable_ml_ohlcv` gates the
    # per-ticker scorer; fail-soft when the artifact or lightgbm is absent
    # (method inactive). `enable_eod_ml_train` retrains the artifact at EOD.
    # (ml_ohlcv PROMOTED to a 0.12 combine weight 2026-08-11 — user-directed.)
    enable_ml_ohlcv: bool = True
    enable_eod_ml_train: bool = True
    # Which ml_ohlcv generation trains + serves (2026-08-08): "pivot_rank" = the
    # v2 signed-pivot within-day-rank GBM (85 features incl. leg state, full
    # universe, uniform day-equal weights — the measured winner; promotion gate
    # `python -m src.analysis.ml_validate --target pivot`); "classic" reverts to
    # the v1 rel-10d conditioned classifier. Serving dispatches on the ARTIFACT's
    # own config, so flipping this changes what EOD trains, not how an existing
    # pickle is read.
    ml_ohlcv_target: str = "pivot_rank"
    # v2 retrain throttle (days): the full-universe dataset rebuild is ~20-40 min
    # and measured staleness of a few days costs ~nothing, so weekly by default.
    ml_pivot_retrain_days: int = 7

    # Long-horizon buy arm (2026-08-01, src/analysis/ml_stacker.py) — an A/B path
    # that replaces the weighted combined_buy_score with the learned 5d stacker
    # (measured to beat the weighted combine at 5d) AND holds those buys longer,
    # to capture the 5d+ edge the short-hold book leaves on the table. Default
    # OFF; `aggregator.ml_combine_arm_active` reads this (or the pipeline's per-run
    # A/B override). Fail-soft: no artifact ⇒ the weighted combine is kept.
    enable_ml_combine: bool = False
    # Minimum HOLD for an arm trade (suppresses time/conviction exits until held
    # this many trading days). **DEFAULT 0 = OFF, because it was MEASURED HARMFUL**
    # (src/analysis/exit_policy_sim.py, 3,294 simulated positions): forcing the
    # hold destroyed ~37% of the ML exit model's timing edge (excess over a
    # hold-matched control +1.27 → +0.80) and cost ~1.8pp per position — it holds
    # through exactly the deterioration the model detected. The horizon is now
    # captured where it belongs (the exit model is trained at the 5d horizon the
    # entry optimises), not by a clock. Kept as a knob for a future regime; safety
    # exits always fire regardless. Re-measure with exit_policy_sim before raising.
    ml_arm_min_hold_days: int = 0
    # Per-run probability the ML combine arm is active (the A/B share). 0 =
    # never via A/B (the default; the manual `enable_ml_combine` still forces
    # it). Set e.g. 0.5 to paper-test the arm against the current short-hold book.
    # LIVE production value is 0.5 via .env ML_COMBINE_ARM_SHARE (re-enabled
    # 2026-08-11 for the stacker-vs-promoted-combine A/B); the code default
    # stays 0 so a missing .env fails to the weighted combine, not to an arm.
    ml_combine_arm_share: float = 0.0
    # Retrain the buy stacker (ml_buy) at EOD on the freshly-materialised panel.
    enable_eod_ml_buy_train: bool = True

    # PROBABILITY CALIBRATION for the ML combine (2026-08-03). The combine reads
    # the model's output AS a probability — `max(0, 2p−1)` is a conviction only if
    # `p` means what it says — but gradient-boosted probabilities are known to be
    # distorted (Niculescu-Mizil & Caruana, ICML 2005). MEASURED on ml_buy's
    # walk-forward panel (n=10,248): raw P(up) spanned 0.087→0.929 while the
    # realised up-rate stayed FLAT at ~0.47 in every bin — Brier skill −0.098,
    # worse than always predicting the base rate — so conviction up to 0.86 was
    # reaching Gate 1 and sizing on coin-flip names. An isotonic calibrator (PAVA,
    # pure numpy) fitted on WALK-FORWARD OUT-OF-FOLD predictions is stored on the
    # artifact and applied before the centering. **Expected effect while the model
    # has no ranking power: conviction collapses toward 0, i.e. the model ABSTAINS
    # rather than asserting — that is the intended safety property.** Fail-soft:
    # no calibrator on the artifact (or this off) ⇒ the raw probability is used,
    # exactly the pre-2026-08-03 behaviour.
    enable_ml_probability_calibration: bool = True

    # ML EXIT model (2026-08-02, src/analysis/ml_exit_dataset.py) — the learned
    # exit-timer over position state (MFE/MAE/days-held/combine-degradation) + the
    # oriented method scores. Its baseline (`exit_method_consensus`) is measured
    # ANTI-PREDICTIVE at 3d and 5d, so this is the exit-side counterpart to the
    # entry stacker swap. COUPLED to the entry arm: it DRIVES exits only for trades
    # stamped `ml_arm` (opened while the ml_buy/ml_sell arm was active), so
    # ML entry and ML exit ride the same A/B flip. For every other held position it
    # is still scored + persisted to the exit_signals panel (IC-tracked live) but
    # never closes it. Fail-soft: no artifact ⇒ the hand-built exit machinery is
    # kept. A confident hold-conviction ≤ −ml_exit_threshold triggers the close.
    enable_ml_exit_model: bool = True
    # held_rank exit signal (2026-08-22, PANEL-FIRST): today's aggregate score
    # ranked within the position's OWN tick history since entry (abs-basis
    # combine, self-normalizing per ticker). Scored + persisted to exit_signals
    # so its exit-side IC accrues; in _CONSENSUS_SKIP and owns NO closing rule
    # until the panel proves it -- the standard new-method probation.
    enable_held_rank_exit_signal: bool = True
    # Minimum pool size (ticks incl. today) before held_rank scores; below it
    # the method abstains (a rank among two observations is a coin).
    held_rank_min_ticks: int = 5
    ml_exit_threshold: float = 0.35            # oriented hold-conviction ≤ −this → ml_exit
    ml_exit_horizon_days: int = 5              # label look-ahead the exit model optimises
    # Retrain the exit model at EOD on the freshly-materialised panel.
    enable_eod_ml_exit_train: bool = True

    # Automatic database refactor (src/analysis/refactor.py). Detects that an
    # implementation changed (AST fingerprint per method) and repairs the stored
    # history in dependency order: data -> epochs -> weights -> derived.
    # A walk-forward step whose fail-soft layers returned EMPTY despite this
    # many visible panel rows is treated as DEGRADED, not as a market fact.
    # Observed once (2026-07-26 stored 21 active methods between neighbours at
    # 10, non-reproducible) — a transient DB read lost to the live scheduler
    # yields "nothing filtered", a materially more permissive calibration that
    # looks entirely legitimate. Degraded steps are stored but skipped when
    # resolving weights for a date, so the last GOOD calibration is used.
    walkforward_min_rows_to_filter: int = 20000

    enable_auto_refactor: bool = True
    # When the nightly rescore runs (ET). Its own slot rather than part of EOD
    # maintenance: the rescore is ~40 minutes and the scheduler loop is
    # single-threaded, so it runs in a BACKGROUND thread and must not be hosted
    # inside an inline step. 02:00 ET sits inside the overnight session
    # (ticks at 01:00/02:00/03:00/03:30), which is precisely why it cannot
    # block — change detection is sub-second, so an unchanged night is free and
    # the expensive path only starts when an implementation actually changed.
    rescore_time: str = "02:00"

    # Mask (epoch) a method whose code changed but which CANNOT be regenerated.
    # OFF by default and deliberately so: masking withholds real history, and
    # the detector cannot tell a cosmetic edit from a categorical one — a rename
    # inside a sentiment module should not silently blank 78% of the panel.
    # Replayable methods are regenerated regardless of this flag, since there a
    # false positive costs only CPU.
    refactor_auto_epoch: bool = False

    enable_market_relative_weighting: bool = True
    # "pv" = pivot basis for the per-side weighting/baseline (2026-08-12
    # directive); a fixed label ("1w") restores the calendar-horizon basis.
    market_relative_horizon: str = "pv"
    market_relative_min_obs: int = 200
    # The HARD filter judges on PROMOTION LOGIC (rebased 2026-08-12, user-
    # directed): a weighted method is dropped only when the promotion statistic
    # itself — market-neutral per-day IC over the directional panel — is
    # SIGNIFICANTLY negative (t = |ICIR|·sqrt(n_days) ≥ ic_weight_min_t) at
    # every horizon judgeable with ≥ market_relative_min_obs rows. Same
    # evidence bar to lose weight as to earn it; unproven ≠ disproven. The
    # 2026-07-27→08-12 rule (point-estimate hit rate net of benchmark < 50%,
    # no significance test) zeroed 14/27 weighted methods on the live window,
    # including the strongest 20y-validated promotions (hi52, mom_12_1).
    # Filtered methods are STILL scored, persisted to the signals panel and
    # IC-tracked, so a dropped method can re-earn its place.
    enable_market_relative_filter: bool = True

    enable_regime_size_haircut: bool = True
    risk_off_size_multiplier: float = 0.5

    enable_continuous_regime_threshold: bool = True

    enable_method_horizons: bool = True
    method_horizon_alpha: float = 0.05
    method_horizon_min_obs: int = 30
    method_horizon_cache_seconds: float = 21600.0    # 6h — heavy join, slow-moving
    # Weight multiplier for a method that is neither proven nor disproven.
    # Absence of evidence is not evidence of absence: on a five-week sample most
    # methods land here, and zeroing them would collapse the ensemble (coherence,
    # sources_agreeing, family agreement and Gate 1b all need several methods).
    unproven_weight_multiplier: float = 0.5

    # ── Inversion arm B: the signals panel (2026-07-25) ───────────────────
    # The ledger arm is starved (~150 attributed trades). This arm reads the
    # simulated_trades panel — tens of thousands of solo directional calls —
    # via compute_directional_perf (market-relative, net of beta).
    #
    # CRITICAL: each method is scored on its EXCESS ICIR over combined_score at
    # the same horizon, not against zero. Measured 2026-07-25, the share of
    # methods with negative ICIR runs 35% @30m → 61% @1d → 76% @2w → 79% @1m,
    # and combined_score itself is −0.227 @1d to −2.187 @2w: that is the
    # system's holding-period edge DECAY, shared by everything. Testing against
    # zero at a long horizon would invert most of the book while measuring only
    # how long positions are held. Horizons are therefore restricted to the real
    # holding period (median hold ~1.3 days, measured edge peak 1–2d).
    enable_inversion_panel_arm: bool = True
    # Weekly ML retrain slot (2026-08-12, user directive: retrain models ONCE
    # A WEEK, Saturday morning — nightly EOD retraining removed). weekday is
    # Python convention (Mon=0 … Sat=5, Sun=6). The enable_eod_ml_* flags gate
    # WHICH models the weekly job trains (names kept for .env compatibility).
    ml_retrain_weekday: int = 5
    ml_retrain_time: str = "08:00"
    # Minimum pivot swing (2026-08-12, user directive): a reversal only
    # CONFIRMS once price retraces this % from the running extreme, so legs
    # smaller than the round-trip cost never become pivots ("not worth buying
    # and selling weak price runs"). Changing it changes what every pivot
    # label MEANS — the ml_ohlcv artifact basis string encodes it, so a
    # threshold change makes stale artifacts abstain until retrained.
    pivot_min_move_pct: float = 1.0
    # Stacker training label (2026-08-12): "pivot_rank" = within-day rank of
    # the signed pivot target (fails soft to the fixed rank_5d label while the
    # panel's settled-pivot rows are thin); "rank_5d" pins the legacy label.
    stacker_label_basis: str = "pivot_rank"
    # ── Liquidity-class cost buckets (2026-08-25, user directive) ───────────
    # "Decommission the in-house cost model": filled legs already charge their
    # OWN realized cost; UNFILLED legs now take the average realized cost of
    # fills in the same LIQUIDITY CLASS (price band × trailing dollar-volume
    # band — edges are module constants in spread.py), same tick first, then
    # the time-of-day period, Bayesian-shrunk toward the session mean and
    # clamped. The hand-built spread formula survives ONLY as the cold-start
    # prior (fills-free DB) and the sub-$ penny-stock guard.
    sim_cost_bucket_min_legs: int = 3       # session-class bucket needs this many fills
    sim_cost_tick_bucket_min_legs: int = 2  # same-tick same-class average needs this many
    sim_cost_bucket_prior_n: int = 8        # shrink bucket mean toward the session mean

    # ── FOLLOW-THROUGH (2026-08-25, user directive; src/signals/follow_through.py).
    # At each tick, every Gate-4 scored name's hypothetical held positions
    # (cohorts entered 1..ft_max_cohort_days sessions ago in the panel direction
    # of that day) are scored with the LIVE ml_exit artifact via the production
    # serving path; the most-negative cohort score is the ticker's ft_score. The
    # within-tick extreme tail becomes OPPOSITE-direction entry candidates — the
    # market followed through against the position; the entry joins it.
    # Validated 2026-08-24/25 (walk-forward, live pivot labels): spec tail
    # +2.62% pivot potential (t +4.05), h=1-close realized +1.42% (t +2.51),
    # RT cost 0.21-0.53%; edge is a POINT EVENT (next-day entry −0.88%) and
    # capture is the overnight gap => same-tick entry + next-session exit.
    enable_follow_through: bool = True            # score + persist (panel-first accrual)
    enable_follow_through_trading: bool = True    # open the mechanical one-session trades
    ft_tail_pct: float = 0.05                     # within-tick bottom share of ft_score
    ft_score_max: float = -0.50                   # level guard (abstains on weak days)
    ft_max_cohort_days: int = 15                  # hypothetical-position entry ages scanned
    ft_size_multiplier: float = 0.5               # conservative flat sizing (new mechanism)
    ft_max_entries_per_day: int = 12              # the measured daily candidate rate
    ft_max_hold_days: int = 2                     # hard stop if the h=1 window is missed
    ft_exit_after_et: str = "15:30"               # first RTH tick at/after this on day+1 closes
    ft_score_budget_seconds: float = 150.0        # per-tick scoring time guard (fail-soft;
                                                  # warm-process sweep ~5-15s, cold ~45-90s)
    # ml_exit TRAINING POPULATION (2026-08-23): "all" = simulate a held position
    # for EVERY scored name (combine-sign orientation where no direction fired)
    # — ~2.2x the training rows from the same days (145k vs 66k; most neutral
    # names have combine exactly 0 and correctly get nothing); "directional" =
    # only names the panel fired a direction on (the historical band-conditioned
    # population). Evaluation stays on the directional population either way
    # (that is what the model serves). Walk-forward verdict (scratchpad
    # exit_upgrade_0823): wide training +0.0128 IC/day paired on the directional
    # eval set, same-sign halves, LONG +0.011 / SHORT flat -> "all".
    ml_exit_train_population: str = "all"
    # Stacker MODEL CLASS (2026-08-22): "logistic" = pure-numpy SoftmaxLogistic —
    # the measured winner of the 27-arm redesign (paired vs the GBM classifier
    # +0.0345 IC/day, t +2.17, same-sign halves; scratchpad stacker_redesign*_0822):
    # at ~50 panel days the GBM's capacity is spent memorising one regime while
    # the linear model generalises. "gbm" reverts to the pre-2026-08-22
    # LightGBMModel classifier (STACKER_GBM_PARAMS). Serving is class-agnostic —
    # both implement bull_bear and the artifact carries the fitted object.
    stacker_model_class: str = "logistic"
    # "pv" since 2026-08-12 (pivot basis; "1d,3d" restores the fixed pair — the
    # arm requires negativity at EVERY listed label, so one pivot label = one test).
    inversion_panel_horizons: str = "pv"
    inversion_panel_min_t: float = 2.0    # before the multiple-comparison bump
    inversion_panel_min_obs: int = 200    # min observations per method/horizon
    inversion_panel_cache_seconds: float = 21600.0   # 6h — the join is expensive
                                                     # and the panel moves in days

    # ── Macro News Regime (geopolitics / oil / tariffs / policy) ────────────
    # Scans the day's news flow for macro-level themes (active wars and
    # geopolitical escalation, trade / tariff actions, oil/energy shocks,
    # central-bank surprises, fiscal/policy events, black swans) and produces
    # a composite regime read with sector implications. The composite feeds
    # the Macro Regime Filter so a CRISIS-grade headline actually tightens
    # the BUY threshold (and blocks longs during PANIC), and is passed to the
    # Claude analyst so geopolitical narrative gets folded into BUY/SELL/HOLD.
    #
    # Uses the news articles already fetched in step 1 — no extra feed cost.
    # Classification: a single DeepSeek call when ``DEEPSEEK_API_KEY`` is set,
    # otherwise a deterministic keyword-density heuristic (lower fidelity but
    # still useful). Cached hourly matching the news cache.
    enable_macro_news: bool = True
    macro_news_max_articles: int = 60   # cap for LLM payload size
    macro_news_min_articles: int = 3    # below this count, skip (not enough signal)

    # ── Intermarket Divergence (broad index ETFs vs SPY) ─────────────────────
    # Cross-market regime detector — tracks whether IWM (small-caps), RSP (equal-
    # weight S&P), QQQ (NASDAQ-100), DIA (Dow), MDY (mid-caps), EFA (developed
    # ex-US), EEM (emerging markets), IWF (growth), and IWD (value) are leading
    # or lagging SPY over 1m / 3m windows.
    #
    # Named regime labels surface canonical intermarket tells:
    #   NARROW_LEADERSHIP   — IWM + RSP both lag → mega-cap dependence, classic
    #                         late-cycle / distribution warning
    #   BROAD_PARTICIPATION — IWM + RSP both lead → healthy rally
    #   GROWTH_ROTATION     — QQQ + IWF lead vs IWD
    #   VALUE_ROTATION      — IWD leads vs IWF + QQQ
    #   US_EXCEPTIONALISM   — EFA + EEM both lag → dollar strength regime
    #   INTERNATIONAL_STRENGTH — EFA or EEM leading → softer dollar / global growth
    #
    # The composite intermarket_health ∈ [-1, +1] feeds the Macro Regime Filter
    # alongside VIX/MOVE/credit/breadth, so narrow leadership tightens the BUY
    # confidence threshold and broad participation relaxes it. Also passed to
    # the Claude analyst as macro context so it can synthesise the regime read
    # into the BUY/SELL/HOLD/WATCH decision.
    enable_intermarket: bool = True

    # ── Correlation-aware position sizing ────────────────────────────────────
    # Replaces the legacy hard per-sector cap, which caught only GICS-sector
    # concentration and missed cross-sector factor exposure: three high-beta
    # semis (NVDA + AVGO + SMH) load the same factor even though SMH is an
    # ETF in a different sector bucket; three "high-beta growth" names rated
    # independently move together when the growth factor rolls.
    #
    # At trade-open time we compute realized pairwise correlations from the
    # last N trading days of OHLCV closes between the candidate and every
    # OPEN same-direction trade. Same-direction matters: a long-X / short-Y
    # pair is a hedge, not concentration — so opposite-direction positions
    # do NOT count toward the haircut.
    #
    # Multiplier scales linearly from 1.0× at mean_corr ≤ low_threshold to
    # min_multiplier at mean_corr ≥ high_threshold. A separate portfolio cap
    # caps the SUM of pairwise-correlation-weighted exposure across all
    # open same-direction positions — when adding the candidate would push
    # that sum over the cap, the trade is skipped entirely.
    enable_correlation_sizing: bool = True
    correlation_lookback_days: int = 60
    correlation_low_threshold: float = 0.30   # ρ̄ ≤ this → no haircut (1.0×)
    correlation_high_threshold: float = 0.80  # ρ̄ ≥ this → full haircut to min_multiplier
    correlation_min_multiplier: float = 0.25  # deepest soft haircut applied to the candidate
    correlation_portfolio_cap: float = 2.5    # Σ(size·ρ) hard skip threshold per direction
    correlation_min_overlap_days: int = 20    # minimum overlapping bars to trust a pair ρ
    correlation_health_max_fail_pct: float = 0.5  # >this share of intended pairs failing to compute → flag the sizing-correlation feed unhealthy (Data Quality)

    # ── Out-of-sample validation (deterministic hash split) ──────────────────
    # Every closed trade is permanently assigned to "train" or "holdout" via a
    # deterministic hash of (seed, ticker, entry_date). Adaptive weights and
    # any other fit-on-history machinery use train ONLY — the holdout slice is
    # reserved for honest evaluation so a method that looks great on the data
    # it was tuned against can be seen failing on data it never touched.
    #
    # NOTE: this is a fixed train/holdout PARTITION, not a walk-forward (no
    # rolling window, no time-ordered re-training). The split is stable across
    # runs so a given trade is always evaluated the same way.
    #
    # Increase oos_split_seed to ANY new string to reshuffle the split (e.g.
    # after a strategy revamp where you want fresh evaluation). Setting
    # oos_holdout_pct=0 disables the holdout (degenerate: all data is train).
    enable_oos_validation: bool = True
    oos_holdout_pct: int = 30        # % of trades reserved as holdout (clamped 0..50)
    oos_split_seed: str = "llm_trader_v1"

    # ── Hypothetical always-open trades ──────────────────────────────────────
    # A separate, isolated category of trades where the listed tickers are ALWAYS
    # in an open position (BUY or SELL) — used as a baseline / reference book
    # alongside the real signal-driven trades. Each entry is "TICKER:BUY" or
    # "TICKER:SELL" (default BUY if no direction given; legacy LONG/SHORT are
    # accepted and normalised to BUY/SELL). Stored in
    # cache/hypothetical_trades.json — completely separate from cache/trades.json
    # so they NEVER contaminate real-trade performance metrics.
    enable_hypothetical_trades: bool = True
    hypothetical_trades: str = "GLD:BUY,SLV:BUY,GDX:BUY,NVDA:BUY"

    # Scheduling — LEGACY, INERT (audited 2026-07-25): read nowhere. The runner
    # is driven by the RTH window + `extended_windows` + `overnight_windows`
    # below, not by a cron string. Kept only to avoid breaking a .env that
    # still sets SCHEDULE_DAILY; changing it has no effect.
    schedule_daily: str = "0 8 * * 1-5"    # inert — see above

    # Intraday scheduling — the runner ticks every 30 min and only acts inside
    # the regular session window below (ET, Mon-Fri). Combined with live prices
    # and completed-only daily bars, there is no dependency on an unclosed bar.
    intraday_session_start: str = "09:30"
    intraday_session_end: str = "16:00"

    # Extended-hours operation:
    #   "off"     — scheduler ticks RTH only (legacy behavior).
    #   "observe" — Phase 0: extended ticks run the FULL pipeline (signals,
    #               recommendations, DB persistence — the session-tagged
    #               evidence base) but NO ledger or broker mutation and no
    #               email.
    #   "trade"   — Phase 1 (default): extended ticks are FULL trading ticks —
    #               ledger entries/exits/marks and broker paper orders happen
    #               off-hours too, with session-aware costs (×4 spread),
    #               sizing (extended_size_multiplier), a stricter actionable
    #               gate (extended_confidence_bump), and LMT+outsideRth broker
    #               submissions. The daily email still fires only on the
    #               16:00 RTH closing tick.
    extended_hours_mode: str = "trade"        # "off" | "observe" | "trade"
    # Extended observation windows (ET, comma-separated HH:MM-HH:MM, optional
    # "@MM" per-window cadence override). Default covers the FULL extended day
    # 04:00–20:00: the liquid shoulders (07:00–09:30 pre-market ramp,
    # 16:00–17:30 earnings-reaction window) tick every extended_tick_minutes;
    # the thin dead zones (04:00–07:00, 18:00–19:00) tick hourly — spreads
    # there are widest and the evidence value per LLM dollar lowest, and the
    # hourly cadence matches the news cache TTL so each tick sees fresh news.
    # The LAST slot is 19:50, not 20:00: the pipeline needs ~4 min from tick to
    # order submission AND the every-tick engine-pinned hold-review adds more on
    # the critical path (fresh news/price refetch + per-engine re-synthesis), so
    # the final slot leaves a ~10-min pre-close buffer — a 20:00 (or even 19:55)
    # slot's orders would reach IBKR after the extended session closed and could
    # never fill same-day. 19:50 leaves the orders a live book before the close.
    extended_windows: str = "04:00-07:00@60,07:00-09:30,16:00-17:30,18:00-19:00@60,19:50-19:50"
    extended_tick_minutes: int = 30
    # Bid-ask half-spread multipliers outside RTH (commission is session-
    # independent — only the spread term widens). Rough placeholders to be
    # calibrated against IBKR paper fills later, same plan as commission_buffer.
    spread_extended_multiplier: float = 4.0
    spread_overnight_multiplier: float = 10.0

    # ── Overnight session (20:00 ET → 04:00 ET; IBKR overnight venue) ───────
    # Same three-state rollout as extended_hours_mode, for the OVERNIGHT
    # session. "trade": overnight slots are FULL trading ticks — the ledger
    # enters/exits/marks at the (heavily penalised: ×10 spread,
    # overnight_size_multiplier, overnight_confidence_bump) overnight terms and
    # the broker leg routes to IBKR's overnight venue (broker_overnight_routing).
    # The venue trades Sunday night → Thursday night, 20:00–03:50 ET
    # (market_calendar.is_overnight_session_open — Friday/Saturday/holiday-eve
    # nights have NO session and are never ticked or booked).
    overnight_hours_mode: str = "trade"       # "off" | "observe" | "trade"
    # Overnight tick slots (same "HH:MM-HH:MM[@MM]" grammar as extended_windows;
    # windows may NOT cross midnight — list the evening and morning halves
    # separately). Hourly: overnight books are the thinnest of the day and the
    # hourly cadence matches the news-cache TTL. The 20:30 first slot lets the
    # 20:00 close of after-hours settle; the 03:30 single slot is the last tick
    # whose orders can still work the book before the venue's 03:50 close
    # (mirroring the 19:50-not-20:00 rule) — 04:00 is already the first
    # extended slot.
    overnight_windows: str = "20:30-23:30@60,01:00-03:00@60,03:30-03:30"
    # Off-RTH entry sizing: overnight entries are sized harder down than
    # extended ones (×0.25 vs ×0.5) — the modeled spread is 10× RTH and no
    # fill evidence exists yet; accumulate evidence at low weight first.
    overnight_size_multiplier: float = 0.25
    # Actionable-threshold bump for OVERNIGHT runs (replaces, not stacks with,
    # extended_confidence_bump): the thinnest books demand the most conviction.
    overnight_confidence_bump: float = 0.10

    # ── Price-provenance health check ───────────────────────────────────────
    # Per-run guard against the stale-price class (the 2026-06-15 CRDO bug: an
    # entry booked at Friday's stale close while the live pre-market print — and
    # the analysis snapshot — was ~4.5% higher). After trades open, every leg's
    # recorded entry_price is compared to the run's snapshot price for that
    # ticker; a divergence beyond the session-appropriate band is flagged
    # (CRITICAL log + email/dashboard banner). RTH is tight (the snapshot and the
    # entry fetch are near-simultaneous); off-hours allows more drift between the
    # snapshot and the entry fetch on a thin, fast-moving tape.
    enable_price_provenance_check: bool = True
    price_provenance_band_rth_bps: float = 100.0
    price_provenance_band_extended_bps: float = 350.0
    price_provenance_band_overnight_bps: float = 600.0

    # ── Extended-session signal profile ─────────────────────────────────────
    # Outside RTH the information landscape changes: options chains (put/call,
    # max pain, OI skew, IV expression) are FROZEN at the last regular-session
    # close, while news flow and the extended-session price action are the
    # live, tradeable information. These knobs adapt the run accordingly.
    #
    # Confidence bump: added to the macro-regime actionable threshold for runs
    # executing outside RTH (extended AND overnight) — thin books + wide
    # spreads demand more conviction before a signal counts as actionable.
    # In "observe" mode it only shapes the persisted `actionable` flag; in
    # "trade" mode it directly gates which extended signals become positions.
    extended_confidence_bump: float = 0.06

    # ── Aggregator-agreement entry gate (combined_score) ───────────────────
    # A trade is accepted ONLY when BOTH decision-makers endorse it: the synthesis
    # (the LLM produced a BUY/SELL above the confidence threshold) AND the
    # aggregator (its weighted combined_score points the SAME way with at least
    # this magnitude). Prevents the LLM from opening a position the underlying
    # weighted methods don't support. Empirically accepted trades cluster at
    # |combined_score| 0.20–0.46 in their direction (none have ever opposed), so the
    # default is a floor that catches future weak/contradicted calls without
    # blocking well-formed ones — raise it to demand stronger method agreement.
    # ⚠ NOT IMPLEMENTED IN THE LIVE PIPELINE, AND DELIBERATELY SO (audited
    # 2026-07-25). `enable_combined_score_gate` is read NOWHERE, and
    # `min_combined_score_for_entry` only by the OFFLINE counterfactual harness
    # `analysis/policy_eval.py` — the live actionable filter has Gates 1, 1b,
    # 2, 3, 4, 5 and no combined-score gate. The description above therefore
    # describes behaviour that does not exist; it is kept only because
    # policy_eval still measures the counterfactual.
    #
    # It was NOT wired up when the gap was found, because the ledger says the
    # gate would BLOCK THE BEST TRADES. Over 241 closed trades joined to their
    # entry-day signals row, the cohort this gate would have removed
    # (|combined_score| < 0.15 or opposing the LLM's direction) returned
    # +1.60% / 45% win, while the cohort it would have kept returned −1.46% /
    # 34% win; the 7 trades that outright OPPOSED the aggregator returned
    # +2.77%. That corroborates the much larger recommendation-stream finding
    # behind the dual-case prompt (echo BUYs −3.19% / 33% at n=4,698 vs
    # neutral-origin BUYs +0.21% / 66% at n=150, genuine overrides 67.6% win):
    # the LLM adds value exactly where the aggregator is undecided or wrong, so
    # forcing agreement with the weighted methods removes its best cohort.
    # Two independent samples, same direction. Do not enable without new
    # evidence that reverses both.
    enable_combined_score_gate: bool = True     # inert — see above
    min_combined_score_for_entry: float = 0.15  # offline (policy_eval) only
    # Position-size multiplier applied ON TOP of the confidence tier and the
    # correlation haircut for trades ENTERED outside RTH. Extended books are
    # thin and the modeled spread 4× wider, so pre-prod sizes off-hours
    # entries at half weight until paper fills prove the edge. 1.0 = off.
    extended_size_multiplier: float = 0.5

    # ── Evidence-based conviction sizing (2026-07-02 ledger study, n=44) ────
    # Two findings from the attributed ledger drive these knobs:
    # (1) LLM entry CONFIDENCE carries almost no return information (Spearman
    #     +0.10 with outcome; calibration slope ~0; the ≥0.92 bucket actually
    #     UNDERPERFORMED 0.85–0.92). The old ramp paid up to 2.0× for it. The
    #     ramp's span above 1.0× is therefore compressed:
    #       multiplier = 1.0 + (legacy_ramp − 1.0) × confidence_size_span
    #     1.0 restores the legacy 1.0→2.0× ramp; 0.0 = confidence-blind sizing.
    confidence_size_span: float = 0.5
    # (2) Agreement BREADTH — how many methods agreed with the direction at
    #     entry — was the strongest entry-time discriminator (Spearman +0.48;
    #     realized-only win rates 46% above the median split vs 23% below,
    #     Laplace-smoothed gap d≈0.20 over 26 closed trades; survives
    #     excluding the 2026-06-25 winner cohort). The convergence multiplier
    #     saturates at 2 agreeing methods, so breadth was previously unused
    #     above that. Sizing tilt, CONTINUOUS + SELF-CALIBRATING (2026-07-03):
    #       frac  = n_agreeing / len(attribution set)   ← survives method-set
    #               growth (the set already grew 19→28 once; absolute
    #               thresholds would have silently broken). NOT normalized by
    #               "methods that voted" — that flips the signal negative
    #               (the ledger shows information RICHNESS is the edge).
    #       ramp  = clamp((frac − center) / half_width, −1, +1)
    #       mult  = 1 + breadth_size_span × edge × ramp
    #     center/half_width are re-estimated each tick from the ledger's own
    #     recent breadth distribution (median / IQR over the last
    #     breadth_adaptive_window attributed trades once ≥ min_trades exist;
    #     the measured priors below until then), so the tilt ranks entries
    #     against the CURRENT book — new methods or regime shifts recenter it
    #     automatically. `edge` throttles the whole tilt by REALIZED evidence,
    #     Bayesian-shrunk: d_post = (prior_n·d_prior + n·d_obs)/(prior_n+n),
    #     edge = clamp(d_post / edge_ref, 0, 1) — grows toward full span as
    #     closed trades keep confirming the effect, decays to NEUTRAL (never
    #     auto-inverts) if it stops holding. Priors measured 2026-07-02.
    breadth_sizing_enabled: bool = True
    breadth_size_span: float = 0.2          # max ± size tilt at full evidence + saturation
    breadth_center_prior: float = 0.46      # ledger median frac (measured)
    breadth_halfwidth_floor: float = 0.09   # min ramp half-width (measured IQR)
    breadth_adaptive_min_trades: int = 10   # attributed trades before center/width adapt
    breadth_adaptive_window: int = 200      # recent attributed trades used to calibrate
    breadth_edge_prior: float = 0.20        # prior hi−lo realized win-rate gap (measured)
    breadth_edge_prior_n: int = 30          # pseudo-trades behind the prior (shrinkage)
    breadth_edge_ref: float = 0.25          # gap that counts as FULL evidence (edge=1)

    # Master switch for the session-dependent SIGNAL profile (default OFF so
    # scores are comparable across the trading day — the requirement behind the
    # fix #2 confidence trajectory). When OFF: one fixed weight profile is used
    # in every session (no stale-options/fresh-news overlay), `ext_gap` is part
    # of the active method set in ALL sessions (so the method set + weight
    # normalisation are identical — it still reads 0 in RTH by design, since the
    # daily technical stack already captures the gap), and the synthesis prompt
    # carries NO extended-session context block. When ON: restores the original
    # session-adaptive behaviour (overlay below + ext_gap off-hours-only +
    # the prompt's SESSION CONTEXT block). NOTE: the actionable-threshold bump
    # `extended_confidence_bump` is a GATE, not a score, so it is independent —
    # set it to 0.0 if you also want session-invariant actionability.
    enable_extended_signal_profile: bool = False
    # Aggregator weight overlay for extended/overnight runs (only applied when
    # enable_extended_signal_profile=True): options-derived methods (put_call,
    # max_pain, oi_skew, iv_expr) are scaled DOWN (stale since the RTH close —
    # yesterday's positioning, not live confirmation); news + sentiment-velocity
    # + the extended gap are scaled UP (the live extended-hours edge is overnight
    # news repricing).
    extended_stale_options_weight_mult: float = 0.5
    extended_news_weight_mult: float = 1.25

    # Extended-session gap momentum (ext_gap) — per-ticker scorer active ONLY
    # outside RTH (returns 0.0 = "no view" during the regular session, where
    # the open gap is already captured by technicals). Measures the live
    # extended print vs the last COMPLETED daily close, normalised by the
    # ticker's own ATR: pre-market gap-and-go / after-hours earnings reaction.
    # Score = tanh(gap_atr / scale) with a deadband so micro-moves don't read
    # as signal: |gap| < deadband × ATR → 0.0.
    enable_extended_gap: bool = True
    extended_gap_deadband_atr: float = 0.25   # min |gap| in ATR units to register a view
    extended_gap_scale_atr: float = 1.5       # tanh saturation: 1.5 ATR gap → ~0.76 score

    # Scheduler resilience on laptops / Modern-Standby machines. This box exposes
    # only S0 "Connected Standby" (no S1/S2/S3): when the display sleeps Windows
    # *suspends* this process, so cron fires come due while frozen and APScheduler's
    # default 1-second misfire grace silently drops them (the runner logged
    # "Scheduler started" but never ticked). Two guards:
    #   • keep_awake       — issue a Windows ES_SYSTEM_REQUIRED power request so the
    #     OS will not idle into standby while the scheduler runs (no-op off-Windows).
    #   • misfire_grace_sec — if the process IS suspended across a fire time, run the
    #     tick on resume (coalesced) instead of dropping it. Kept under the 30-min
    #     cadence so a late tick never duplicates the next scheduled one.
    scheduler_keep_awake: bool = True
    # How often the poll-loop runner re-reads the wall clock to decide whether a
    # 30-min slot is due. Short so that after a Modern-Standby suspension it
    # re-evaluates within this many seconds of resuming (instead of relying on a
    # precomputed long sleep, which standby skews).
    scheduler_poll_seconds: int = 30
    # How late a 30-min slot may still run after its boundary. If the machine was
    # suspended past this, the slot is skipped (logged) rather than run stale.
    scheduler_misfire_grace_sec: int = 1500
    # Flip this on to email on EVERY in-window tick (every 30 min) — handy for
    # confirming the scheduler fires. Turn back off once verified to avoid 13/day.
    # Takes precedence over scheduler_email_times below.
    scheduler_email_every_tick: bool = False
    # Comma-separated ET slot times (HH:MM) at which the daily report is emailed.
    # Non-empty (default) → the report sends ONLY at these slots; each MUST be an
    # actual tick slot (the RTH 30-min grid or an extended_windows boundary) to fire
    # — the scheduler logs a warning at startup for any configured time that isn't a
    # slot. Empty "" → legacy behaviour (only the 16:00 close). Overridden by
    # scheduler_email_every_tick. Default = 4 AM (pre-market open), 9:30 (RTH open),
    # 4 PM (RTH close), 7:50 PM (last after-hours tick).
    scheduler_email_times: str = "04:00,09:30,16:00,19:50"
    # Always send the report email — even on a non-email slot — when this run
    # detected a PROBLEM: a broker/execution issue (disconnect, rejects, drift),
    # an LLM-layer outage (credits/keys), or a price-provenance flag. So an issue
    # that surfaces between the scheduled email slots reaches you at the next tick
    # (~30 min) instead of waiting for 09:30/16:00/19:50. The forced email carries
    # the usual 🔔 banner + subject tag. Independent of scheduler_email_every_tick.
    email_on_problem: bool = True

    # Intraday timing overlay (Hybrid: daily trend decides direction; a 30-min
    # momentum read only gates entry/exit *timing*).
    enable_intraday_timing: bool = True
    intraday_timing_defer_threshold: float = 0.50   # |opposing 30-min momentum| above this defers an entry
    enable_intraday_exit: bool = False               # close a position on a strong intraday reversal against it
    intraday_exit_threshold: float = 0.60

    # ── Database (DuckDB — single source of truth for trades / recs / run meta) ──
    # One embedded file. The daily pipeline is the sole writer; the dashboard
    # reads it read-only. Created on first run (parent dir auto-made).
    db_path: str = "data/llm_trader.db"

    # ── Dashboard (Plotly Dash monitoring app: rationale · methods · returns) ──
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8050

    # HTTP Basic-Auth gate in front of the WHOLE dashboard (every route, including
    # the Dash callback XHRs). One shared credential pair handed out with the link
    # — the "sharable password" model, not per-person identity.
    #
    # Empty password = NO gate (the default: a loopback-only dashboard needs none).
    # The gate is what makes public exposure safe, so the tunnel launcher
    # (scripts/run_public_dashboard.ps1) PROBES it — an unauthenticated request
    # must come back 401 — and refuses to open the tunnel otherwise. That check is
    # mechanical on purpose: "middleware silently not applied" and "gate working"
    # are indistinguishable from the config alone, and the cost of being wrong is
    # publishing live P&L.
    dashboard_auth_username: str = "viewer"
    dashboard_auth_password: str = ""

    # Networks whose requests SKIP the password prompt (comma-separated CIDRs).
    # Default: this machine only — browse http://127.0.0.1:8050 with no login,
    # while the public tunnel still demands the shared password.
    #
    # ⚠ REMOTE_ADDR alone cannot express this: ngrok dials the tunnel into
    # 127.0.0.1, so every PUBLIC visitor also arrives from a loopback address.
    # The bypass therefore needs all three of (loopback peer, no proxy headers,
    # loopback Host) — see dashboard/app.py::_is_local_request, which also
    # explains why _serve_once must stop waitress from eating the proxy headers.
    # Empty = no bypass (everything is gated, incl. localhost).
    dashboard_auth_bypass_networks: str = "127.0.0.0/8,::1"

    # Hostnames that count as "this machine" for the bypass above, on top of the
    # loopback names and bare IP literals it already accepts. Needed for Tailscale
    # MagicDNS: a phone browsing http://<machine>.<tailnet>.ts.net:8050 arrives
    # from a tailnet IP but sends a NAME in the Host header, and a name never
    # parses as an IP — so without an entry here it would still be asked for the
    # password. Comma-separated; an entry starting with "." matches any host
    # ending in it (".ts.net" survives a tailnet rename, an exact name is tighter).
    # Matched case-insensitively. Empty = only loopback names and IP literals.
    #
    # This does NOT weaken the tunnel gate: a tunnelled request still arrives from
    # loopback carrying proxy headers, and its Host is the public ngrok domain.
    dashboard_auth_bypass_hosts: str = ""

    # ── Broker / live execution (paper-first; OFF by default → no broker calls) ──
    # Pre-production: drive a real broker's PAPER account in parallel with the
    # internal NAV sim ("shadow & reconcile"), then flip to live with a port swap.
    # IBKR is the only API broker offering a Canadian resident both paper and live
    # for US securities (CIRO blocks API only for Canadian-listed names, which this
    # system never trades — the universe is 100% US stocks/ETFs). See src/broker/.
    #   off        — no broker calls; internal simulation only (default; unchanged behavior)
    #   dry_run    — log the orders that WOULD be placed (sizing + idempotency); submit nothing
    #   ibkr_paper — submit to IB Gateway PAPER account
    #   ibkr_live  — submit to IB Gateway LIVE account  [gated: only after paper validation]
    broker_mode: str = "off"
    # ── Broker advisor (IBKR account / short-borrow-aware method group) ──
    # First method in a broker-aware group: scores short-borrow state (hard/expensive
    # to short → bullish squeeze tilt → fades a SELL). The only method aware of
    # IBKR-unique data; decision-only (never trades). Needs a live IBKR connection,
    # so it is OFF unless broker_mode != off AND this flag is on. Each scored ticker
    # costs one market-data request, so the per-tick fetch is capped.
    enable_broker_advisor: bool = False
    broker_advisor_max_score: float = 0.6          # cap on the squeeze tilt (a single tell can't dominate)
    broker_advisor_expensive_fee_pct: float = 10.0 # borrow fee %/yr mapping to a strong tilt (tanh scale)
    broker_advisor_hard_shares: float = 200000.0   # shortable shares at/below which a name is "hard to borrow"
    broker_advisor_max_tickers: int = 60           # cap on per-tick borrow fetches (held names + universe slice)
    ibkr_host: str = "127.0.0.1"
    ibkr_port: int = 4002              # IB Gateway: 4002 paper / 4001 live  (TWS: 7497 / 7496)
    ibkr_client_id: int = 11           # any stable int unique to this API connection
    ibkr_account: str = ""             # optional: pin a specific IBKR account id (else the sole/first)
    ibkr_connect_timeout: int = 15     # seconds to wait for the Gateway socket
    # Bound every ib_async API REQUEST (reqExecutions / reqPositions / placeOrder /
    # reqPnL / reqMktData) so a stuck request can't freeze the whole scheduler —
    # the 2026-07-06 hang: the overnight reconcile blocked on reqExecutions after
    # connecting, froze for 6+ h, missed every tick + the 04:00 email. ib_async's
    # IB.RequestTimeout raises after this many seconds instead of waiting forever;
    # the reconcile's fail-soft try/excepts then continue the tick. 0 = the old
    # freeze-forever behaviour. See src/broker/ibkr.py.
    broker_request_timeout_seconds: float = 45.0
    # SHORT timeout for the real-time quote snapshot in get_market_price only. A
    # live quote arrives in ~1-3s in RTH (reqTickers returns as soon as the snapshot
    # completes, so this never delays a successful fetch); pre-market/thin data never
    # arrives, so without a tighter bound each priced ticker would burn the full
    # broker_request_timeout_seconds (45s) and the reconcile — which prices every
    # open/drift position one-by-one — could march into the sync watchdog and force a
    # pre-market respawn loop. Fail fast to None instead (caller falls back). 0 = use
    # the global request timeout.
    broker_price_timeout_seconds: float = 8.0
    # Hard wall-clock backstop on the ENTIRE broker reconcile: if it somehow still
    # exceeds this (a hang RequestTimeout doesn't cover), force-exit the process so
    # the task manager restarts it (a fresh tick > a permanent freeze). Set well
    # above a legitimate slow reconcile; 0 disables. See pipeline.py.
    broker_sync_watchdog_seconds: int = 600
    # Hard wall-clock backstop on a WHOLE scheduled tick (scheduler/runner.py):
    # the broker watchdog above only covers reconcile, so a hang anywhere else —
    # a wedged data fetch blocking the pipeline pool join (the 2026-08-17
    # yfinance/bond_internals 4.5-hour freeze), a stuck LLM stream — froze the
    # scheduler with no recovery (--supervise needs an EXIT). Fires CRITICAL +
    # os._exit(1) so the supervisor relaunches. Must sit far above the slowest
    # legitimate tick (~15 min measured; EOD/ML retrains run in background
    # threads and are NOT under this timer) and comfortably above the broker
    # watchdog so a broker hang still gets attributed by ITS watchdog first.
    # 0 disables.
    tick_watchdog_seconds: int = 2700
    # Prefer IBKR's real-time last/mark price (free Cboe One + IEX feed via the
    # broker connection) over yfinance in tracker._fetch_price — the same data
    # that fills the orders, so the mark matches the execution venue. Requires an
    # IBKR broker_mode (ibkr_paper/ibkr_live); falls back to yfinance/Polygon when
    # off, disconnected, or the quote is missing. OFF by default for A/B.
    enable_ibkr_price_feed: bool = False
    # Position sizing. Two modes (broker_sizing_mode), each × the 1.0/1.5/2.0×
    # confidence tier already on every trade:
    #   "notional"   — fixed base order size: broker_base_notional in broker_base_notional_ccy.
    #                  US securities are USD-priced, so a non-USD base is converted to a USD share
    #                  budget via live FX (src/broker/fx.py); broker_fx_fallback_cad_usd is used
    #                  only if the live quote is unavailable.
    #   "equity_pct" — broker_base_position_pct of account equity.
    # shares = floor(budget_in_USD × size_multiplier / price).
    broker_sizing_mode: str = "notional"          # "notional" | "equity_pct"
    broker_base_notional: float = 1000.0          # base order size per 1.0× position (notional mode)
    broker_base_notional_ccy: str = "CAD"         # currency of broker_base_notional (matches the account)
    broker_fx_fallback_cad_usd: float = 0.73      # CAD→USD, used only if the live FX quote is unavailable
    broker_base_position_pct: float = 0.05        # equity_pct mode: 5% of equity per 1.0× position
    # IBKR's API can't place FRACTIONAL equity orders (error 10243 — GUI-only), so a small
    # base is rounded to whole shares. "nearest" rounds half-up (a name still places as long
    # as it's ≥ half a share, i.e. priced up to ~2× the budget); "floor" sticks to the budget
    # (skips any name priced above one position's worth). "nearest" avoids needless skips.
    broker_share_rounding: str = "nearest"        # "nearest" | "floor"
    broker_max_positions: int = 20                # hard cap on concurrent broker positions
    broker_max_gross_exposure_pct: float = 1.0    # cap on Σ|notional_usd| / equity_usd (1.0 = no leverage)
    # Order type for broker submissions:
    #   "LMT" (default) — marketable limit at model price ± the session cap in the
    #           adverse direction (BUY above / SELL below). Bounds the worst
    #           acceptable fill; on liquid names it executes as fast as MKT.
    #           The settle pass removes LMT's historical downside: an order
    #           whose cap is missed is re-anchored at a fresh quote within the
    #           tick and KILLED if still unfilled — it never rests to fill
    #           late at a stale price.
    #   "MKT" — market order: fills immediately at whatever the book offers,
    #           with NO price bound (observed 2026-06-12: −107 bp ZUMZ and
    #           −91 bp ATGL fills on thin RTH books). Off-RTH ticks force LMT
    #           regardless (IBKR rejects MKT outside regular hours).
    broker_order_type: str = "LMT"                # "LMT" | "MKT"
    # LMT only: max adverse distance from the model price (RTH).
    # Raised 20 -> 45 (2026-08-17). 20 bp was measurably BINDING: of RTH fills,
    # p90 landed at 19.1 bp and p95 at 20.5 against the 20 bp cap, with 11.9%
    # inside a tenth of the cap and 8.1% only clearing it because a re-anchor
    # had moved the reference. A cap that the fills pile up against is one that
    # is turning marginal orders into no-fills.
    # This is a CEILING, not a price paid: a marketable limit fills at the best
    # available price up to the limit, so widening it cannot make an existing
    # good fill worse -- it only admits fills that previously did not happen.
    # Off-RTH is deliberately unchanged; there the cap is NOT the constraint
    # (fills consume a p50 of only ~8-20 bp of an 80/150 bp cap), liquidity is.
    broker_limit_cap_bps: float = 45.0
    # SPREAD-AWARE limits (2026-08-17). The limit is priced off a LAST/MID —
    # get_market_price never returns the ask — so an order is marketable only
    # when the cap happens to exceed the HALF-SPREAD. That holds in RTH and
    # fails badly off-hours, where a thin book quotes hundreds of bp: measured
    # 56.6% RTH vs 4.8% overnight fills. The surviving fills consumed only a
    # small slice of their cap, which looked like "the cap is fine" but was
    # survivorship — the wide names never filled, so they never entered the
    # statistic at all.
    # With this on, a real two-sided quote raises the cap to cover
    # half_spread * broker_spread_cap_mult (never LOWERS it below the
    # configured cap) and the limit is pushed to the far side of the book when
    # the cap covers it. Reaching the far side costs nothing: a marketable
    # limit executes at the touch, so this decides whether the order can trade,
    # not the price paid. No quote ⇒ unchanged mid-based behaviour.
    broker_spread_aware_limits: bool = True
    # ...but RTH ONLY, by measurement (2026-08-17). Widening only creates fills
    # whose half-spread exceeded the old cap, so the newly-fillable off-hours
    # cohort pays >=160 bp round trip (extended) / >=300 bp (overnight). Measured
    # GROSS returns for off-hours entries are nowhere near that -- real closed
    # trades +1.09% mean / +0.07% median extended and -0.46% / +0.07% overnight;
    # the simulated panel -0.09% / -0.66% and -0.14% / -0.64% over 15k rows. So
    # those fills would book a measured loss regardless of signal quality, and
    # the newly-fillable names are the widest-spread tail of that population.
    # Flip to True only with evidence that off-hours gross returns clear the
    # round trip. See src/broker/reconcile.py::_effective_cap_bps.
    broker_spread_aware_off_rth: bool = False
    broker_spread_cap_mult: float = 1.5     # x half-spread; >1 leaves room to cross
    # Absolute ceiling on the widened cap — the backstop that keeps a garbage
    # quote (a stale one-tick book at 4am) from authorising an unbounded price.
    broker_limit_cap_bps_max: float = 400.0
    # Off-RTH the book is thin and the REAL spread is ~4× wider (the sim's own
    # cost model charges 4× extended / 10× overnight half-spreads) — a 20 bp
    # cap sits INSIDE the extended spread, so every off-RTH order rests
    # unfilled, is cancelled next tick and chases the market on stale data.
    # A wider extended cap keeps off-RTH orders genuinely marketable: fill in
    # the decision tick at the current (wide) spread rather than later at a
    # drifted price.
    broker_limit_cap_bps_extended: float = 80.0   # LMT cap outside regular hours
    # Overnight LMT cap: the overnight book is thinner still (sim models it at
    # ×10 RTH spread) — a cap inside the spread can never fill. Placeholder to
    # calibrate against real overnight paper fills, like the extended cap.
    broker_limit_cap_bps_overnight: float = 150.0
    # Route overnight-session orders to IBKR's overnight venue (contract
    # exchange "OVERNIGHT" instead of SMART; LMT-only, TIF DAY, ~10k eligible
    # US stocks/ETFs). Fail-soft: an ineligible symbol / unentitled account
    # rejects, lands as SUBMIT_FAILED, and the tick-scoped lifecycle retries or
    # kills it — never breaking the run. False = legacy behavior (off-venue
    # orders rest until the 04:00 pre-market open, where the settle pass kills
    # them — i.e. no overnight broker fills).
    broker_overnight_routing: bool = True
    broker_paper_equity: float = 100000.0         # USD equity used for the exposure cap in dry_run

    # ── Order-submission reliability (acceptance check + bounded retry) ──
    # Every submission's broker answer is verified; TRANSIENT failures
    # (connection drop, timeout, pacing/rate limit) retry in-tick a few
    # times. The window is deliberately SHORT — a delayed fill must stay
    # anchored to THIS tick's model price — and every retry goes out as a
    # marketable LMT capped at broker_limit_cap_bps from that model price,
    # so the worst acceptable fill never drifts past the cap no matter when
    # the retry lands. Hard rejects (insufficient funds, permissions,
    # invalid contract) never retry — a retry can't fix them. Before each
    # retry the broker is checked for an order already carrying this
    # client_ref: an attempt that errored AFTER transmission may have
    # reached the broker, and resubmitting blind would double the position.
    broker_submit_retries: int = 2        # transient retries per order (0 = off)
    broker_retry_wait_seconds: int = 5    # pause before each retry (reconnect window)
    # Connect retries at sync start: IB Gateway has a daily re-login window —
    # without a retry, one badly-timed tick loses its whole order cycle
    # (observed 2026-06-11 15:41 ET: "not connected", 4 positions waited a tick).
    broker_connect_retries: int = 2           # extra connect attempts at sync start
    broker_connect_retry_wait_seconds: int = 10
    # Automatic in-tick reconnect (2026-07-11): when the Gateway session DROPS
    # (gateway restart, network blip, daily re-login), every IBKRBroker call
    # revives it via a throttled implicit connect (_ensure_connected) instead of
    # failing soft until the next tick's sync. After a failed dial, implicit
    # revives pause this long so a down/wedged gateway can't charge each price
    # fetch / sync step the full ibkr_connect_timeout; explicit connect() calls
    # (the sync-start retry loop, _submit_with_retry) bypass the pause, and any
    # successful dial clears it. 0 = no throttle (dial on every touchpoint).
    broker_reconnect_cooldown_seconds: float = 60.0
    # ALIVE-BUT-WEDGED gateway detection (2026-07-13). A wedged IB Gateway keeps
    # the socket open — isConnected() stays True — while EVERY API request times
    # out (blank-message TimeoutError). The plain reconnect never fires (the
    # session reads as connected), so the whole tick burns request-timeouts and
    # the failures get mislabeled "rejected". After this many CONSECUTIVE request
    # timeouts the broker treats the session as effectively dropped and forces a
    # fresh client dial: on a still-wedged gateway that dial also times out and
    # arms the cooldown above, so the rest of the tick fast-fails instead of each
    # call hanging — and it auto-recovers the instant the gateway (or IBC) is back.
    # Counts CONSECUTIVE timeouts (any success resets), so on a healthy gateway the
    # frequent successful reads keep it near 0 — a thin pre-market ticker whose
    # reqTickers times out (bounded by broker_price_timeout_seconds) can't
    # accumulate. Set >a handful so a run of thin-ticker price timeouts can't
    # false-trip it, while a real wedge (EVERY request times out) trips within the
    # first few calls. 0 = disable wedge detection (legacy: trust isConnected()).
    broker_wedge_timeout_threshold: int = 5
    # REPEATED-wedge escalation (2026-08-04). The wedge handling above force-
    # recycles the CLIENT, and gateway recovery used to fire only when that forced
    # redial FAILED. Observed live: a wedged gateway ACCEPTED every reconnect and
    # then timed out every request, so the redial succeeded ~14 times in a row and
    # recovery never fired — the loop ran 07:39→08:00 with auto-restart enabled and
    # did nothing. A gateway needing this many force-recycles inside the window is
    # therefore treated as the corpse REGARDLESS of whether it answers the dial.
    # Any genuinely successful request clears the history (only a real response
    # proves it is back). Keep the limit >2 so an ordinary blip can't escalate.
    broker_wedge_recycle_limit: int = 3
    broker_wedge_recycle_window_seconds: float = 900.0
    # ── GATEWAY auto-recovery (2026-07-13, the last manual ops step automated) ──
    # The app-side self-healing above (auto-reconnect, wedge detection + forced
    # client recycle) can only fix the APP's side of the session. When the GATEWAY
    # itself is the corpse — alive-but-wedged (process up, port open, API backend
    # dead: 2026-07-06, 2026-07-13) or fully down — every redial hits a dead
    # socket, and IBC's process watchdog can't see it (it only checks the process
    # is alive). Recovery was a documented MANUAL procedure: kill the gateway java
    # process, trigger the 'IBC Gateway' scheduled task (relaunch + auto-login).
    # broker_gateway_auto_restart automates exactly that procedure, fired from the
    # two places that KNOW the gateway is the problem: the sync-start connect loop
    # exhausting its retries, and a wedge-forced redial failing. PAPER-MODE ONLY —
    # ibkr_live downgrades to a CRITICAL log (bouncing a live-money gateway, which
    # may need 2FA to re-login, is a human decision; same posture as drift
    # auto-flatten). Cooldown-guarded so a persistently-broken gateway can't be
    # kill-looped; every step fail-soft (a recovery failure never breaks the run).
    broker_gateway_auto_restart: bool = True
    broker_gateway_task_name: str = "IBC Gateway"   # the IBC relaunch scheduled task
    broker_gateway_restart_cooldown_minutes: int = 30
    # sync-path restarts wait up to this long for the fresh gateway's port before
    # the final in-tick dial; the wedge-path fires without waiting (next touchpoint
    # redials). RAISED 90 → 300 on 2026-08-04: the "~60s" estimate was optimistic —
    # a MEASURED cold IBC relaunch + auto-login took **250s** to start listening, so
    # at 90s the recovery logged "port not listening — will reconnect on a later
    # touchpoint" and returned False while the gateway was in fact coming up fine.
    # That turned a successful restart into a reported failure (and, on the sync
    # path, skipped the final in-tick dial for no reason). The wait costs nothing
    # when the port comes up sooner — the loop returns as soon as it is listening.
    broker_gateway_restart_wait_seconds: int = 300
    # ── Settle pass: fill fast or kill ───────────────────────────────────
    # After this tick's orders are submitted, actively watch them for up to
    # this many seconds: fills are recorded the moment they land; a zero-fill
    # order is cancelled and re-anchored at a fresh capped quote EARLY and
    # REPEATEDLY (see broker_settle_reanchor_every) so a mispriced order is
    # re-priced within seconds and keeps chasing the spread in bounded steps;
    # anything still unfilled at the deadline is CANCELLED — nothing ever rests
    # across ticks, so an order either executes within this window of its
    # decision or it does not exist (the next tick re-decides from fresh data
    # and prices). Partial fills are left working. 0 = legacy (rest until next
    # tick). Lowered 60→30 (2026-07-07) to bound the added tick time while the
    # tighter poll + repeated re-anchor keep fills fast.
    broker_settle_seconds: int = 30
    # Fill-poll cadence inside the settle window (seconds). Smaller = fills are
    # detected + orders re-anchored sooner (more reliable), at more broker
    # round-trips. 3s catches a fast fill in ~one poll; well inside IBKR pacing.
    broker_settle_poll_seconds: int = 3
    # Re-anchor a still-unfilled order every N settle polls (not just once at the
    # halfway mark): with a 3s poll this re-prices at ~6s, ~12s, ~18s … so an
    # order that missed on a moving book gets several fresh-quote retries within
    # the budget instead of one. The final poll never re-anchors (it observes).
    broker_settle_reanchor_every: int = 2
    # ── Order lifetime: tick-scoped (default) or age-based ──────────────
    # Tick-scoped (True): an order lives exactly one tick. Any order still
    # unfilled at the next sync is cancelled and re-decided from THIS tick's
    # data — entries resubmit only if the trade survived this tick's signal
    # pass (monitor_open_positions runs first), re-anchored at the current
    # mark; exits always resubmit re-anchored. No order ever works the book
    # on a previous tick's price. False: legacy age rule below.
    broker_tick_scoped_orders: bool = True
    # LET UNFILLED ORDERS REST ACROSS TICKS while the price stays near the
    # decision (2026-08-17). Tick-scoping alone put an order in the book for
    # broker_settle_seconds (30 s) out of a ~30-minute tick -- about 1.7%
    # presence -- and then killed it. Measured consequence: a 19.3% fill rate
    # overall, and 4.8% overnight / 10.2% pre-market, where liquidity arrives
    # sporadically and a 30-second window almost never coincides with a
    # counterparty. RTH, whose ticks overlap continuous trading, filled 56.6%.
    #
    # With this on, an order that predates the sync is KEPT WORKING instead of
    # cancelled, so long as the mark is still within broker_rest_max_drift_bps
    # of the price the decision was made at. Drift beyond that (either way) is
    # a reason to re-decide, so it falls through to the normal stale-cancel and
    # is resubmitted re-anchored. broker_unfilled_cancel_minutes remains the
    # absolute ceiling, and a resting order can never fill worse than its own
    # capped limit -- the exposure this adds is adverse selection (we fill when
    # the market comes to us), which is exactly what the drift bound limits.
    # False restores strict one-tick lifetimes.
    broker_rest_unfilled_orders: bool = True
    # Symmetric |mark - decision| bound, in bp, for keeping an order resting.
    # Deliberately tighter than broker_resubmit_max_adverse_bps (100), because
    # that ceiling gates a one-shot resubmission decision while this governs
    # CONTINUOUS presence in the book between ticks.
    broker_rest_max_drift_bps: float = 60.0
    # Age fallback (used when tick-scoped is False; also an upper bound when
    # True): an ACCEPTED order resting unfilled this many minutes is
    # cancelled and resubmitted re-anchored at the current mark. Partial
    # fills are left working. 0 = never (no age rule).
    broker_unfilled_cancel_minutes: int = 90
    # Each sync, CANCEL any working broker order that no ledger leg points at.
    # The invariant: an order may work the book only while a trade leg owns it.
    # Orders outlive the ledger whenever the process dies between placing one
    # and persisting it (the reconcile watchdog's os._exit) — the orphans then
    # fill unattributed and read as "position drifted from the ledger" for ever
    # (2026-07-23: 13 watchdog kills left 88 working orders, 18 refs duplicated,
    # one HQY exit ref stacked 8 deep → 112 shares sold against a 14-share
    # long). Drift-flatten refs are exempt (no leg by design; _flatten_orphan
    # runs its own cycle). False = legacy behaviour, orphans rest for ever.
    broker_orphan_order_sweep: bool = True
    # Ceiling on how far an unfilled ENTRY may be chased away from the price the
    # decision was made at. A resubmit whose current price is worse than the
    # trade's entry_price by more than this many bps is HELD (never re-sent this
    # tick), whether or not the signal still fires — the decision and its sizing
    # were made at entry_price, so past this the modeled edge is gone and
    # chasing only books a bad entry. Applies to entries only; an EXIT must get
    # flat regardless of price. 0 = no ceiling (chase at any price).
    broker_resubmit_max_adverse_bps: float = 100.0
    # Sanity band (%) for the real-fill cost calibration: a filled LMT leg whose
    # measured one-way cost is beyond ±this is DISCARDED as a bad record, not
    # averaged in. A capped marketable LMT (20 bp RTH / 80 bp extended / 150 bp
    # overnight) cannot legitimately fill percent-points away from its own
    # anchor — such a leg means the recorded decision price was stale. Without
    # this the plain mean has no defence: 17 corrupt legs (all negative) once
    # pulled the measured cost to ~0, so the sim charged no transaction cost at
    # all. 0 = no filtering (the old, unguarded behaviour).
    sim_real_fill_cost_sanity_pct: float = 2.0
    # Per-trade cost attribution (2026-07-23): charge each sim leg the cost that
    # best fits it — its OWN realized cost if it filled at the broker; else the
    # average of the trades that filled in the same tick; else the average for
    # its time-of-day period (rth/premarket/afterhours/overnight); else the
    # modeled/global cost. Refines the flat override WITHOUT breaking ledger
    # purity (every sim trade still assumes it fills). False → the flat/session
    # override for every leg, i.e. the pre-2026-07-23 behaviour.
    # ── Direction-scoped overrides (2026-07-25) ──────────────────────────────
    # Resolved by config.settings.directional(name, direction). Each is None by
    # default, meaning "use the shared value" — so the split machinery is
    # present and inert until a value is deliberately set. Only parameters with
    # a STRUCTURAL reason to differ are exposed here; see `directional`.
    horizon_expiry_floor_mult_long: Optional[float] = None
    horizon_expiry_floor_mult_short: Optional[float] = None
    # Self-calibrating per-side time pressure. Rather than freezing two numbers
    # from a sample that fails multiple-comparison correction, the ramp
    # multiplier is nudged per direction by MEASURED evidence: the gap between
    # what horizon_expired trades returned and what that side's other exits
    # returned. A NEGATIVE gap means the time-stop is firing too late (the
    # position kept bleeding) → tighten; POSITIVE means patience paid → loosen.
    # Measured 2026-07-25: longs −1.88 pp (time-outs do worse → tighten),
    # shorts +2.54 pp (time-outs do better → loosen) — opposite directions,
    # which is exactly the asymmetry a shared parameter cannot express.
    # Shrunk by horizon_ramp_prior_n so it is ~inert on thin evidence, clamped
    # to ±horizon_ramp_max_adjust, and reported to the calibration registry.
    # ── Per-side actionable threshold (2026-07-25) ───────────────────────────
    # Splitting Gate 1 — the single highest-leverage parameter in the system —
    # so it is deliberately the most conservative split of the four: MEASURED,
    # shrunk, small-capped, and it can only ever TIGHTEN a side (loosening a
    # risk gate automatically is not something a calibration should do).
    #
    # The evidence is NOT "which side wins more" — it is whether confidence
    # DISCRIMINATES on that side, because raising a bar only helps if the bar
    # sorts good calls from bad. Measured 2026-07-25 over closed trades:
    #   longs  Spearman(confidence, return) = +0.008  -> no information; the
    #          >=0.90 cohort was the WORST (-2.11%). Raising the long bar would
    #          cut volume without improving quality, so it stays put.
    #   shorts Spearman = +0.233, monotonic bands, and >=0.90 is the only
    #          profitable cohort (+1.91%, 43.6% win) -> a higher short bar
    #          concentrates a real edge.
    # Set an override to pin a side manually (in confidence points, added to the
    # regime threshold); None = calibrate.
    enable_side_threshold_calibration: bool = True
    actionable_threshold_adj_long: Optional[float] = None
    actionable_threshold_adj_short: Optional[float] = None
    side_threshold_max_adjust: float = 0.04   # cap: at most +4 confidence points
    side_threshold_prior_n: int = 40          # virtual trades at "no adjustment"
    side_threshold_rho_ref: float = 0.30      # rho that maps to a full-strength raise

    enable_horizon_ramp_calibration: bool = True
    horizon_ramp_prior_n: int = 30        # virtual observations at "no adjustment"
    horizon_ramp_max_adjust: float = 0.30  # ±30% of the base multiplier
    horizon_ramp_gap_ref_pp: float = 4.0   # gap (pp) that maps to a full-strength adjustment
    # Adverse stop, split by MEASUREMENT — and the split runs OPPOSITE to the
    # structural prior. The reasoning "a short's loss is unbounded, so stop it
    # tighter" is sound about tail risk but wrong about where the tail starts: a
    # threshold sweep over the closed ledger (Δ = capped-at-stop − realised, via
    # each trade's stored MAE) found a stop HELPS longs at every level from −5%
    # to −12% (+2.13 pp/trade at −6%, +1.22 at −8%) and HURTS shorts at every
    # level from −5% to −15% (−2.80 pp/trade at −5%, worsening to −20.87 at
    # −15%). Shorts that go against this book tend to RECOVER — consistent with
    # the earlier finding that SELLs on spiked names fade correctly — so cutting
    # them at a normal stop locks in a loss that would have come back.
    # Both readings are stable across a wide band rather than a single point,
    # which is what makes them worth acting on at n=153/88.
    adverse_stop_pct_long: Optional[float] = 8.0      # inside the helping plateau
    # NOT disabled: the unbounded-loss tail is real even though a normal stop
    # destroys value here. −20% is a pure runaway guard — the sweep shows no harm
    # there (n=2) while still capping a genuine squeeze.
    adverse_stop_pct_short: Optional[float] = 20.0

    # ── Hard adverse stop (2026-07-25) ───────────────────────────────────────
    # The system's exits are otherwise ALL conviction-based, so nothing caps a
    # single position's loss: a position the engine keeps re-affirming can bleed
    # indefinitely. Fires on the cost-adjusted M2M return, and only when the LLM
    # review HELD (like trailing_stop / mechanical_exit), so it never overrides a
    # macro or flip close. Per-side thresholds above; 0 disables a side.
    enable_adverse_stop: bool = True
    adverse_stop_pct: float = 10.0            # shared fallback if a side is unset

    # ── Short borrow / carry (2026-07-25) ────────────────────────────────────
    # The one genuinely direction-ASYMMETRIC cost: a short pays a daily
    # stock-loan fee a long never does. Nothing charged it before, so every
    # short's simulated return read BETTER than reality by roughly the rate ×
    # holding period — a bias that grows the longer a short is held and is
    # largest on exactly the squeeze-prone names the (currently disabled)
    # broker_advisor method was built to flag. Charged per CALENDAR day (borrow
    # accrues over weekends), on SELL legs only, in both return engines.
    # The rate prefers the broker's REAL fee for that name when it was captured
    # at entry (trade["borrow_fee_pct"], from Broker.get_short_borrow) and falls
    # back to this blended assumption: liquid large caps typically borrow well
    # under 1%/yr, small caps and hard-to-borrow names far more, and this
    # universe mixes both. Deliberately a round, conservative placeholder —
    # revisit once real borrow rates accrue. 0 = charge nothing.
    enable_short_borrow_cost: bool = True
    short_borrow_annual_pct: float = 3.0
    sim_per_trade_cost_attribution: bool = True
    # Min filled legs in a tick (run) before that tick's average is trusted for
    # its unfilled trades; below it, fall through to the time-of-day average.
    sim_cost_tick_min_legs: int = 3
    # Price-aware next-tick resubmit for a previously-unfilled ENTRY. Instead of
    # blindly chasing the current price on the next tick, decide per the fresh
    # signal + the price vs the original decision: (a) still actionable same
    # direction → resubmit; (b) decayed to non-actionable BUT the price is
    # as-good-or-better than the decision (≤ entry for a long / ≥ entry for a
    # short) → resubmit at the better price; (c) decayed AND the price drifted
    # adverse → HOLD (don't chase — the tick after re-evaluates); (d) signal
    # flipped to the opposite side → don't resubmit. EXITS are unaffected — they
    # always resubmit to get flat. Needs the tick's actionable set threaded into
    # reconcile.sync(); with none supplied it falls back to the legacy chase.
    broker_price_aware_resubmit: bool = True

    # ── Drift auto-reconciliation ────────────────────────────────────────
    # The ledger is the source of truth; a broker position the ledger cannot
    # explain (no OPEN trade, no close in flight) is an ORPHAN. Drift is
    # prevented at the source where possible (working entry orders are
    # cancelled the moment their trade closes; exits flatten what the broker
    # ACTUALLY holds), and whatever still slips through (ledger restores,
    # manual TWS trades, fill races) is handled per this setting:
    #   "flatten" (default) — submit a price-capped marketable LMT at a live
    #                         quote to close the orphan; re-anchored each tick
    #                         while it rests, so the broker converges to the
    #                         ledger in bounded ≤cap steps.
    #   "report"            — legacy behavior: surface it, touch nothing.
    # SAFETY: in ibkr_live mode "flatten" is refused and downgraded to
    # "report" — auto-selling unexpected REAL-money positions (e.g. something
    # bought manually in TWS) is a deliberate human decision, not a default.
    broker_drift_action: str = "flatten"   # "flatten" | "report"

    # ── Simulated trading costs (commission term in the sim's return math) ──
    # The bid-ask half-spread model (src/performance/spread.py) covers spread cost;
    # this adds per-order commission so small positions don't overstate their edge
    # (at ~$730 USD/position, IBKR Fixed's $1 minimum is ~27 bp round trip — larger
    # than the modeled spread on a liquid large cap). Applied symmetrically in
    # _pct_return AND the daily-NAV walk's entry/exit anchors, so both engines agree.
    #
    # CONSERVATIVE BY DEFAULT: the model deliberately errs toward OVERSTATING fees
    # so reported performance understates the edge rather than flattering it —
    # the pricier all-in plan (ibkr_fixed) is assumed, and commission_buffer adds
    # headroom for everything the published schedule excludes. Actual commissions
    # captured from fills (broker_orders.commission in DuckDB) are the ground
    # truth for calibrating the model down later.
    #   "ibkr_fixed"  — max($1.00, $0.005/share), capped at 1% of trade value
    #                   (default — the more expensive plan, exchange fees included)
    #   "ibkr_tiered" — max($0.35, $0.0035/share), capped at 1% of trade value
    #                   (commission only — venue/clearing fees are NOT in the base
    #                   rate; rely on commission_buffer to cover them)
    #   "none"        — spread-only (legacy behavior)
    commission_model: str = "ibkr_fixed"          # "ibkr_fixed" | "ibkr_tiered" | "none"
    # Fee ceiling multiplier applied AFTER the min/cap schedule math. Covers the
    # pass-throughs the schedule omits — SEC transaction fee + FINRA TAF on sells,
    # exchange/clearing fees (tiered), odd venue surcharges — plus schedule drift.
    # 1.5 ≈ $1.50 min/side → ~41 bp round trip at the $730 base notional (actual
    # paper fills run ~27 bp — the gap is the intended safety margin). Set 1.0 to
    # model the published schedule exactly.
    commission_buffer: float = 1.5
    # Assumed USD notional per 1.0× position, used to convert the per-order minimum
    # into percentage terms. Keep roughly in sync with broker_base_notional × FX
    # (1000 CAD × 0.73 ≈ 730 USD). A deliberate constant (not live FX) so the
    # return math stays 100% deterministic — and conservative: larger (1.5–2.0×)
    # positions experience a SMALLER min-commission floor in % terms, so pricing
    # every trade at the 1.0× base notional is the worst case.
    commission_notional_usd: float = 730.0
    # ── Calibrate the sim cost to REAL IBKR fills ────────────────────────
    # When on, the simulated per-leg cost (half-spread + commission in
    # _one_side_cost) is REPLACED by the measured average all-in one-way cost
    # from actual broker fills (real commission + execution cost vs the
    # decision price, the same number the dashboard's IBKR "Avg 1-way cost"
    # shows), applied flat to every entry/exit leg. The sim then charges what
    # execution actually costs instead of the modeled estimate. Falls back to
    # the model until at least sim_real_fill_costs_min_legs filled legs exist
    # (a flat average over 2-3 fills would be noise) and is clamped ≥ 0 (a
    # net-favorable fill streak never pays the sim to trade). Still fully
    # deterministic: the calibration is a pure function of the broker fills in
    # the same DuckDB. Set False to keep the modeled cost.
    sim_use_real_fill_costs: bool = True
    sim_real_fill_costs_min_legs: int = 10
    # ── Per-session cost calibration (2026-07-03) ────────────────────────
    # The flat real-fill override blends all sessions into one number, which
    # under-charges extended/overnight legs and slightly over-charges RTH.
    # When enough fills exist the override carries a per-session split: RTH
    # measured directly (needs ≥ session_cost_min_legs RTH legs); extended /
    # overnight = RTH × a session multiplier Bayesian-shrunk from that
    # session's own fills toward the documented ×4/×10 priors
    # (spread_extended_multiplier / spread_overnight_multiplier), floored at
    # 1.0 (off-hours never cheaper than RTH) and capped at 2× the prior. So
    # the punitive ×10 overnight assumption relaxes toward MEASURED overnight
    # costs as venue fills accrue — automatically, with the conservative
    # prior in force until then. Deterministic (a pure function of the
    # broker_orders fills in the same DuckDB).
    session_spread_calibration_enabled: bool = True
    session_cost_min_legs: int = 5        # min legs before a session's own mean counts
    session_spread_prior_n: int = 20      # pseudo-legs behind the ×4/×10 priors
    # ── Derived horizon cost hurdle (2026-07-03) ─────────────────────────
    # horizon_cost_hurdle_pct froze the round-trip hurdle at 0.40% while the
    # system MEASURES its real cost (~0.16% round trip from LMT fills). When
    # a real-fill calibration is active the hurdle derives instead:
    #   hurdle% = 2 × calibrated one-way% × cost_hurdle_safety
    # (clamped to [0.05, 2.0]%), so horizon selection tracks actual execution
    # costs. The static setting remains the no-calibration fallback.
    cost_hurdle_use_calibrated: bool = True
    cost_hurdle_safety: float = 1.5
    # ── Engine-relative actionable threshold (2026-07-03) ────────────────
    # Confidence distributions are ENGINE-specific (DeepSeek hands out 1.00s,
    # Claude tops out lower), so absolute thresholds silently change meaning
    # when the A/B flip or ANALYST_MODEL changes engines. When enabled, each
    # static threshold (regime ladder included) is translated into the run
    # engine's own confidence scale by matching the gate's SELECTIVITY —
    # effective = Q_engine(F_global(static)) over the last
    # threshold_calibration_days of BUY/SELL recommendations — then shrunk
    # toward the static anchor while the engine's history is thin and clamped
    # to ±threshold_max_shift (the gate drifts with evidence, never jumps).
    threshold_engine_relative_enabled: bool = True
    threshold_calibration_days: int = 30
    threshold_min_global_recs: int = 300   # min all-engine sample before translating
    threshold_engine_prior_n: int = 150    # pseudo-recs behind the static anchor
    threshold_max_shift: float = 0.08      # max drift from the static threshold
    # ── Calibrated exit-confidence floor (2026-07-03) ────────────────────
    # signal_decay_confidence_floor froze the absolute close threshold at 0.45
    # while trade_reviews records, every tick, exactly the evidence that
    # decides it: re-affirmation confidence vs the position's oriented forward
    # return. The calibrated floor = the lowest confidence where holding is
    # still profitable (below-floor reviews lose money, above-floor make it),
    # Bayesian-shrunk toward the static prior and clamped to the band below.
    # The static setting remains the prior/fallback; the relative floor
    # (× entry confidence) stays static by design.
    exit_floor_calibration_enabled: bool = True
    exit_floor_calibration_days: int = 60
    exit_floor_prior_n: int = 150          # pseudo-reviews behind the static prior
    exit_floor_min_side: int = 25          # min reviews on each side of a candidate
    exit_floor_min: float = 0.30
    exit_floor_max: float = 0.70
    # ── Exit-conviction consensus (2026-07-03) — the entry-breadth analog ──
    # The exit decision used a single LLM-scalar hold-review and ignored the
    # 27-method exit-conviction panel. This nudges the same-direction close
    # floor by the breadth-of-raw-methods EXIT CONSENSUS (mean of the signal
    # methods' oriented exit scores; LLM review + aggregator excluded so Fix #2
    # trigger-happiness isn't reintroduced): consensus says exit → floor raised
    # (close more readily); says hold → floor lowered (hold more readily).
    # EVIDENCE-THROTTLED + BOUNDED: span_eff ramps with the closed-trade sample
    # from a small prior toward a cap, so a confident LLM hold is never
    # overridden (only borderline convictions get tipped) and with no data it's
    # a gentle nudge. UNVALIDATED by design — to be confirmed/denied over the
    # coming weeks by the offline exit_policy_eval harness; set enabled=False to
    # revert to the pure LLM-scalar close. See analysis/exit_conviction.py.
    enable_exit_conviction: bool = True
    exit_conviction_span_prior: float = 0.03   # floor nudge with ~no evidence (gentle)
    exit_conviction_span_max: float = 0.10     # bounded cap at full evidence
    exit_conviction_prior_n: int = 40          # closed trades for the ramp half-life
    # ── Edge-decay time-stop (2026-07-06) ─────────────────────────────────────
    # The realized edge of combined_score decays with holding horizon (measured
    # tick-by-tick over the signals panel — analysis/horizon_edge.py). On the
    # traded subset the edge peaks ~1-2d and turns negative by ~5d. This layer
    # measures the edge-positive WINDOW and, evidence-throttled, raises the
    # confidence-loss close floor once a position is held past it — an
    # edge-decay time-stop that fires around the realized decay point rather than
    # only the entry target_horizon. Inert on today's thin long-horizon sample;
    # firms up as the decay confirms. Also emitted as the `edge_decay` exit signal
    # so its OWN exit-IC is measured before it earns real weight.
    enable_edge_decay_exit: bool = True
    edge_decay_conf_min: float = 0.85          # confidence subset the edge curve is measured on (the traded population; tracks the 0.85 NEUTRAL gate, raised 2026-07-21)
    edge_decay_cal_days: int = 90              # panel window for the edge curve
    edge_decay_min_n: int = 20                 # min obs per horizon before it counts
    edge_decay_prior_obs: int = 250            # obs prior shrinking the evidence strength (gentle now)
    edge_decay_floor_cap: float = 0.08         # max confidence-loss-floor raise from the edge-decay stop
    edge_decay_cal_ttl_seconds: int = 21600    # calibration cache TTL (6h; changes ~daily)
    # ── Exit fixes (2026-07-08) — the scorecard flagged a −58% MFE give-back (no
    # profit-taking: winners round-trip) and that `llm_signal_flipped` closes
    # positions already ~5.6% DOWN (the LLM flip fires too LATE), while the mechanical
    # exit signals carry POSITIVE exit-IC (money_flow +0.21, max_pain +0.34). Two
    # monitor-level exits address both, leaning on the mechanical side — they only
    # fire when the LLM exit logic (_evaluate_decay) did NOT already close.
    #  (1) Trailing profit-capture: once a position's MFE (cost-adjusted peak return)
    #      arms past trailing_arm_pct, close it if it has given back ≥
    #      trailing_give_back_frac of that peak (exit_reason "trailing_stop") — locks a
    #      consistent fraction of every winner instead of round-tripping it.
    enable_trailing_exit: bool = True
    trailing_arm_pct: float = 3.0              # arm the trail once MFE ≥ this (cost-adj %)
    trailing_give_back_frac: float = 0.5       # close after giving back this fraction of the peak MFE
    #  (2) Mechanical-consensus exit: close when the raw SIGNAL methods' exit
    #      consensus (`exit_method_consensus`: money_flow / max_pain / …; − = exit,
    #      LLM + aggregator + time-overlays EXCLUDED) is confidently "exit"
    #      (≤ −mechanical_exit_threshold), independent of the LLM review (exit_reason
    #      "mechanical_exit").
    #
    #      **OFF SINCE 2026-08-02 — DELIBERATELY, and do NOT "fix" the threshold.**
    #      Two independent measurements condemn this rule:
    #        (a) it is ANTI-PREDICTIVE — over the simulated held-position panel its
    #            exit signal runs IC −0.066 / hit 46% at 3d AND 5d, i.e. it times
    #            multi-day exits BACKWARDS (it averages ~21 mostly 1-day methods, the
    #            same horizon mismatch that made the weighted entry combine fail at 5d);
    #        (b) at the threshold below it CANNOT FIRE ANYWAY. The consensus is a MEAN
    #            of ~21 near-zero scores, so its whole observed range is [−0.26, +0.32]
    #            over 23,387 held-days — 0.35 is outside it (0.000% fire rate; the live
    #            ledger shows 4 hits in 328 closes, all thin-method tickers).
    #      So the rule was already inert BY ACCIDENT, which was quietly protecting the
    #      book. Lowering the threshold to "make it work" (≤−0.10 fires ~5%) would wire
    #      in a rule measured to exit wrong ~54% of the time. Turned off explicitly so
    #      that protection is intentional and survives a future threshold tweak.
    #      The learned replacement is `ml_exit` (enable_ml_exit_model), which DOES fire
    #      and has measured skill. Re-validate with `python -m src.analysis.exit_policy_sim`
    #      before ever re-enabling. Exits do NOT depend on this rule: horizon_expired,
    #      trailing_stop, llm_signal_flipped, llm_confidence_loss and adverse_stop
    #      account for ~99% of live closes.
    enable_mechanical_exit: bool = False
    mechanical_exit_threshold: float = 0.35    # consensus ≤ −this → mechanical exit

    # ── End-of-day maintenance (2026-07-04) — scalability for the weeks ahead ──
    # Once per market day, at/after eod_maintenance_time ET (robust to missed
    # slots: fires on the first poll past the trigger), the scheduler (a) WARMS
    # the forward-return OHLCV cache for every learning-panel ticker — the fuel
    # for the IC panels / policy evals / calibrations, which otherwise starve on
    # a stale cache (observed 2026-07-03) — and (b) runs table RETENTION.
    enable_eod_maintenance: bool = True
    eod_maintenance_time: str = "16:20"        # ET; after the 16:00 close tick settles
    eod_cache_warm_days: int = 120             # panel lookback to warm
    eod_cache_warm_max_tickers: int = 0        # 0 = all panel tickers
    # Retention: simulated_trades is a derived long-format reshape of `signals`
    # (~25×/row) growing ~130k rows/day. Keep a recent RAW window (the entry-
    # event detector needs its intraday sequence), collapse older data to the
    # deduped last-per-(day,ticker,method) the analysis reads (behavior-neutral
    # over the old window), and hard-delete beyond the keep window. exit_signals
    # is age-pruned only; `signals`/`trade_reviews` (source/primary) are left to
    # a generous prune. Set enable_sim_retention=false to keep everything.
    enable_sim_retention: bool = True
    sim_retention_raw_days: int = 14           # keep full intraday resolution this recent
                                               # (tune down to ~7 for a tighter bound)
    sim_retention_keep_days: int = 150         # hard-delete simulated_trades beyond this
    exit_signals_keep_days: int = 150          # hard-delete exit_signals beyond this
    # ── Unified expected-edge sizing (2026-07-03) ────────────────────────
    # The learned successor to the hand-shaped conviction tiers: a ridge model
    # of REALIZED returns on standardized entry features (breadth · confidence
    # · combined_score · news · momentum · off-RTH, direction-oriented) whose
    # say over the final size grows with closed-trade evidence —
    #   final = tier chain × ((1−w)·prior + w·edge_mult)/prior,
    #   w = n_closed/(n_closed + edge_prior_n), 0 below edge_min_closed —
    # so today it nudges (~15% weight) and takes over only as the ledger
    # earns it. Never a gate, never flips direction; bounded by
    # edge_size_span and a hard ratio clamp. See performance/edge_sizing.py.
    edge_sizing_enabled: bool = True
    edge_min_closed: int = 20              # realized closes before the model has ANY say
    edge_prior_n: int = 150                # closes for a 50/50 split with the tier prior
    edge_size_span: float = 0.25           # max ± tilt the model alone can express
    edge_ridge_lambda: float = 1.0         # ridge strength (× n, standardized features)
    # ── Predictability sizing (Tier 1) ────────────────────────────────────────
    # A per-stock "is this name's direction forecastable at a swing horizon"
    # tilt, sized up for clean-trend names and down for chop. The score blends
    # Kaufman trend efficiency + ADX (the Tier-0 predictability panel found both
    # separate a ~60%-hit clean-trend cohort from a ~50% coin-flip at 5 days).
    # SELF-CALIBRATING on the same evidence-throttled idiom as breadth sizing:
    #   mult = 1 + span_eff × clamp((score − center)/half_width, −1, +1)
    #   span_eff = predictability_size_span × clamp(d_post/edge_ref, 0, 1)
    #   d_post = (prior_n·d_prior + n_days·d_obs)/(prior_n + n_days)
    # where d_obs is the measured 5-day directional-hit gap (high vs low
    # predictability) over the UNBIASED signals panel and n_days = signal-days of
    # evidence — so it starts ~inert (heavily shrunk toward the small prior on
    # ~2 weeks of one regime) and STRENGTHENS as the panel thickens over weeks/
    # months, or fades to neutral if the edge doesn't hold. Never a gate (every
    # name still trades — we keep accumulating outcomes), never inverts the sign.
    # See performance/predictability_sizing.py.
    enable_predictability_sizing: bool = True
    predictability_size_span: float = 0.12      # max ± size tilt at full evidence
    predictability_er_weight: float = 0.5       # Kaufman efficiency-ratio weight in the score
    predictability_adx_weight: float = 0.5      # ADX weight in the score
    predictability_adx_cap: float = 40.0        # ADX value that maps to a full 1.0 score component
    predictability_er_window: int = 20          # efficiency-ratio lookback (sessions)
    predictability_adx_period: int = 14         # Wilder ADX period
    predictability_horizon: int = 5             # swing horizon the edge is measured at (sessions; fallback + panel build)
    # 2026-08-13 standardization: the sizing edge is measured on the H/L PIVOT
    # target ("pv") like every other decision surface; "fixed" pins the
    # pre-directive fwd_ret_{predictability_horizon}d basis.
    predictability_label_basis: str = "pv"
    # Same directive, exit side: ml_exit trains on the ORIENTED remaining move
    # to the next H/L pivot ("pv" — the rank-degradation exit thesis: + = the
    # leg still runs our way, − = the next turn is against us), failing soft to
    # the fixed held-return label below 500 settled rows; "fixed" pins the old
    # fwd_ret_pos_{ml_exit_horizon_days}d label.
    ml_exit_label_basis: str = "pv"
    predictability_halfwidth_floor: float = 0.10   # min ramp half-width (guards a degenerate spread)
    predictability_edge_prior: float = 0.02     # documented prior hit-gap (heavily shrunk early)
    predictability_edge_prior_n: int = 40       # signal-DAYS for the prior to fade (≈2 months)
    predictability_edge_ref: float = 0.10       # hit-gap that counts as FULL strength (edge=1)
    predictability_cal_days: int = 60           # panel window used to calibrate (recent regime)
    predictability_cal_min_rows: int = 60       # min panel rows before the edge is measured
    predictability_cal_ttl_seconds: int = 21600  # calibration cache TTL (6h; changes ~daily)
    # ── Trend-predictability METHODS (Kaufman/ADX × trend context) ─────────────
    # The signed Kaufman efficiency ratio + ADX·DMI, expressed as four methods by
    # trend CONTEXT — kaufman_long / kaufman_short = uptrend / downtrend context,
    # same for adx_* — each active only in its context. They fold into
    # combined_score (entry AND exit scores) as an additive overlay OUTSIDE the
    # normalised weight pool (sparse/one-sided context, so pooling would dampen
    # non-trending names — same idiom as the fundamental/corp-action f_* factors);
    # tracked per-method in the signals panel + IC table + trade attribution.
    #
    # Each method's raw trend signal is multiplied by a LEARNED orientation ∈
    # [−1,+1]: +1 predicts CONTINUATION (with the trend), −1 predicts REVERSAL
    # (against it), magnitude = confidence. The orientation is measured per method
    # from the signals panel — how often that trend context has CONTINUED vs
    # reversed at the swing horizon — shrunk toward a CONTINUATION prior (+1) by
    # signal-days, so each method starts as continuation and flips toward reversal
    # only as the forward returns confirm it (e.g. it would learn "clean downtrends
    # bounce" and predict the bounce). See signals/trend_predictability.py.
    enable_trend_predictability_methods: bool = True
    trend_method_weight: float = 0.10        # additive-overlay weight on the 4 oriented trend scores
    # Signal-days of shrinkage toward 0 = ABSTAIN (2026-08-16 rebase — was a +1
    # continuation prior, which held measured-descending contexts at positive
    # multipliers; see trend_predictability.calibrate_trend_orientation).
    trend_orientation_prior_n: int = 25
    trend_orientation_cal_days: int = 60     # panel window used to calibrate (recent regime)
    trend_orientation_cal_min_rows: int = 40  # min active rows before a method's orientation is measured
    trend_orientation_cal_ttl_seconds: int = 21600  # orientation cache TTL (6h; changes ~daily)
    # The flat real-fill override is measured from LIQUID LMT fills; applying it
    # to instruments far outside that basis grossly understates their cost (a
    # $0.05 warrant with a ~35%-wide book was being charged 8 bp — observed
    # ARQQW 2026-07-01). Legs priced below this floor keep the modeled
    # price-tiered half-spread + commission instead of the flat override.
    sim_real_fill_min_price: float = 1.0
    # ── Scheduler outage alerting ────────────────────────────────────────
    # Email an alert when the poll loop discovers it slept through tick slots
    # (machine suspended — observed 2026-06-30: a full trading day dark with 24
    # open positions and no notification). Uses the normal SMTP settings; off
    # when email is unconfigured.
    scheduler_alert_email: bool = True
    # After a missed slot, run ONE catch-up tick immediately when a trading
    # session (RTH or extended) is still live — managing positions late beats
    # not at all. The catch-up never emails the report.
    scheduler_catchup_tick: bool = True

    @property
    def tracked_politicians_list(self) -> List[str]:
        return [p.strip() for p in self.tracked_politicians.split(",") if p.strip()]

    @property
    def tracked_institutions_list(self) -> List[str]:
        return [i.strip() for i in self.tracked_institutions.split(",") if i.strip()]

    @property
    def recipients_list(self) -> List[str]:
        return [r.strip() for r in self.email_recipients.split(",") if r.strip()]

    @property
    def stocks_list(self) -> List[str]:
        return [s.strip() for s in self.stock_watchlist.split(",") if s.strip()]

    @property
    def sectors_list(self) -> List[str]:
        return [s.strip() for s in self.sector_etfs.split(",") if s.strip()]

    @property
    def commodities_list(self) -> List[str]:
        return [s.strip() for s in self.commodity_etfs.split(",") if s.strip()]

    @property
    def hypothetical_trades_list(self) -> List[tuple]:
        """Parsed [(ticker, action)] list for the always-open hypothetical book.

        Action is 'BUY' or 'SELL' (default BUY when unspecified). Legacy
        'LONG'/'SHORT' values are accepted and normalised to 'BUY'/'SELL'
        for convenience. Empty list when the feature is disabled.
        """
        if not self.enable_hypothetical_trades:
            return []
        _aliases = {"LONG": "BUY", "SHORT": "SELL", "BUY": "BUY", "SELL": "SELL"}
        pairs: List[tuple] = []
        seen: set = set()
        for spec in self.hypothetical_trades.split(","):
            spec = spec.strip()
            if not spec:
                continue
            if ":" in spec:
                tk, action = spec.split(":", 1)
                tk = tk.strip().upper()
                action = _aliases.get(action.strip().upper(), "BUY")
            else:
                tk = spec.upper()
                action = "BUY"
            if tk and tk not in seen:
                seen.add(tk)
                pairs.append((tk, action))
        return pairs

    @property
    def factor_list(self) -> List[str]:
        """Merged, de-duplicated factor + thematic ETF universe (empty when disabled)."""
        if not self.enable_factor_etfs:
            return []
        out: List[str] = []
        seen: set = set()
        for s in f"{self.factor_etfs},{self.thematic_etfs}".split(","):
            sym = s.strip().upper()
            if sym and sym not in seen:
                seen.add(sym)
                out.append(sym)
        return out

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


# ── direction-scoped parameter resolution (2026-07-25) ──────────────────────

_LONG_WORDS = frozenset({"BUY", "LONG", "BULLISH"})
_SHORT_WORDS = frozenset({"SELL", "SHORT", "BEARISH"})


def directional(name: str, direction, obj=None):
    """Resolve a setting for ONE direction: ``{name}_long`` / ``{name}_short``
    when that override is set, otherwise the shared ``{name}``.

    The mechanism behind "a parameter set optimised for buying and another for
    selling", built so that adding a split is a config change rather than a
    refactor. **Every override defaults to None**, so with nothing configured
    this returns exactly the shared value and behaviour is unchanged.

    Deliberately NOT applied to every threshold. The 2026-07-25 analysis found
    the long/short gap in the ledger does not survive multiple-comparison
    correction (permutation p 0.014, Bonferroni ×8 → 0.111), so hand-setting two
    values from that sample would be fitting noise. Splits are justified where
    the asymmetry is STRUCTURAL — borrow cost, tail-loss geometry, the speed of
    downside moves — not merely observed once. Prefer a shrunk per-side
    calibration (see ``aggregator.side_weight_multipliers``) over a frozen pair
    wherever the quantity can be measured.
    """
    s = obj if obj is not None else settings
    d = str(direction or "").upper()
    suffix = "_long" if d in _LONG_WORDS else "_short" if d in _SHORT_WORDS else None
    if suffix is not None:
        override = getattr(s, f"{name}{suffix}", None)
        if override is not None:
            return override
    return getattr(s, name, None)
