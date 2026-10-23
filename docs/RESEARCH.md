# The research program — the state of the algorithm, and how to improve it

What the live algorithm is built from, how well it measures, what has been tried, and what is open. CLAUDE.md
documents production as it runs NOW; the runbooks (`docs/SEL_SHORT.md`, `docs/DIP_LONG.md`, `docs/CRASH_LONG.md`)
each book's mechanics; `docs/BACKTEST.md` the simulator; `docs/OVERFITTING.md` the overfitting report. THIS file is
the research state: start here before proposing a change, and add every new study's row (user 2026-10-17: "make sure
the documentation is updated with the best results and logic. We want to build over it, improve over time our models
and whole algorithm"). Every study's own file is `llm_trader_research/<folder>/PREREG<n>.md` (its registration, its
amendments, its result); the distilled findings are the memory notes (`memory/*.md`, index `MEMORY.md`).

Last updated 2026-10-23 afternoon (THE SECOND ACCOUNT deployed — its code live from the 14:00 ET tick — PREREG189's T3, CALCULATED from the main account, trading from the 2026-10-26 open: the user's "There's no second IBKR account creation. The second account will be calculated from the same account." — and PREREG190 registered: IBKR's refusals of opening shorts by name, the user's "make sure we can correctly predict refused short ... for the simulator to find more stocks that are realistically tradeable"). Before it: 2026-10-23 morning (PREREG189 and the add rule's margin events AS A RATE — the user on PREREG186's second account: "What about crash + fall + dip? These are the three that are over t=2.": together at their live rules all three pass the floor (the dip book t 4.36, the fall arm 3.30, the crash book 2.22; 59.8% a year, the drawdown 21.3%, no margin event in 40 draws), and "update the add rule to judge margin calls as a rate" (`add_rule.margin_events`, the one-sided exact binomial on the discordant pairs). Before it: 2026-10-23 night (PREREG186-188 and the forward check in production — the user, after PREREG184 / 185: a SECOND $50,000 ACCOUNT under the market-noise floor (PREREG186: the dip book and the fall arm only), the dip exits at the open (PREREG187: the print stays), the forward check wired into the EOD chain and the email, and set D in the simulator (PREREG188)). Before it: 2026-10-22 evening (PREREG184 / 185, the user: "Do 1 and 5" — two of the plan's ways to be more sure the backtest's growth transfers to live: THE ADD RULE BACKTESTED OUT OF TIME on its own 303 recorded verdicts, every pair re-decided at each year-end on the years before it and scored on the next — its adds gained no more the following year than its refusals and kept about a tenth of their in-sample gain, a market-noise floor (t >= 2) chose a fifth as many changes that did better for the same total, and the one-pair margin-call count refused changes that met no more calls; and THE FORWARD CHECK, every live trade rebuilt the simulator's way on the same prices — all 14 closed trades left on the simulator's own tick, the dips within -0.12% a trade, two leads (the dips' exits at the opening tick, the quote model on a thin small cap), the live account at the 4th percentile of its tracking band because of the META concentration; rows 184-185, section 4 items 5, 6 and 14, section 5). Before it: 2026-10-22 afternoon (PREREG183, the user: "Copy the formulas to improve our own skills": the Sharpe-ratio inference of López de Prado, Lipton & Zoonekynd ported into `src/analysis/stats.py` and the research tools as REPORTS — the live record needs about a year at best before it says anything, the vol arm's deflated Sharpe counted on months is 0.10, the long books' gains are mostly alpha (the crash book's is its timing of 3x exposure), the dip meta-model's folds purged change nothing that trades, and each change deployed 10-19..21 is 12-30% likely noise against the market's own variation; row 183, section 1 item 5, section 2's live record, `docs/OVERFITTING.md`). Before it: 2026-10-22 morning (PREREG179 / 180 / 181, the user: "Start with 1." and "Do task 2 also": the bear context split at the time into a GRINDING decline — SPY under its 50- and its 200-session averages — and the rest; large-cap shorts there lose on 2018-2026 (on 2007-2017 only through 2008), and a book of the beaten-down large caps bought there (+0.23-0.31 %/day a trade in both eras) loses in the account in either capital place and every size — the edge sits on the capitulation book's mass-rebound days; the dip book's meta-model given the market context it lacked decides no better; and UPRO bought at each exit from the grind (PREREG182, the user's follow-up) loses to holding it at random — most exits are false and the 3x fund takes their losses: all REJECTED; rows 179-182, section 4 items 1 and 2). Before it: 2026-10-22 early morning (PREREG177 / 178, the user: "Train ML models on different market contexts so that they are specialized ... improve overall growth by optimizing on return per day of those models": one LightGBM model per market context, per trend group and one global, walk-forward on a survivorship-free 2007-2026 panel of large caps, ranking long (177) and short (178) trades by their predicted return per day — the specialists rank WORSE than one global model given the context as an input, and every book of them makes the weak times worse: REJECTED; section 1 item 11, rows 177-178, section 3.1, section 4 item 2). Before it: 2026-10-21 night (PREREG175: production's account liquidates as IBKR does — the engine's model, a parity test, deployed 22:30 ET — and the simulator's check reads every bar between the marks, as production's: the live configuration 75.8% a year, margin events in 12 draws of 20; PREREG176: THE TRAINING PROTOCOL documented (section 1 item 10) and run under IBKR's liquidation — held out -4.9 pp a year against live, its in-sample choice carried by 2025 with more margin events: nothing deployed; rows 175-176, section 1 items 3 and 10, section 5). Before it: 2026-10-21 evening (PREREG174: the simulator's margin calls are IBKR's LIQUIDATION — researched on IBKR's own pages: the real-time check approximated along each bar, the most recent positions first in 100-share lots only back to the requirement, no Soft Edge Margin, the end-of-day Reg T floor; the default from 19:40 ET; the live configuration 76.6 -> 75.7% a year, margin events in 13 draws of 20 instead of 2, the drawdown unchanged; V15 against ALL3 now an ADD; row 174, section 1 item 3, section 2, section 5). Before it: 2026-10-21 afternoon (PREREG173: each short arm's own time limit — the three searched together do not hold out; the ETF and fall limits stay best at 10; the vol arm at 15 sessions +3.0 pp a year in sample and +2.7 held out (exploratory) but one more margin-call pair: REJECT by the add rule, the user's call; row 173, section 4 item 12). Before it: 2026-10-21 midday (PREREG172: the optimization method — the very-good books' holding parameters searched together by return per day and by return per day x growth, then the sizing by growth, cross-validated by year: R x G not shown better, the growth-chosen sizing brings margin calls, the vol arm's own time limit the lead; section 1 item 8, row 172, section 4 item 12). Before it: 2026-10-20 night (PREREG168-171: new trades and models from new data sources — the filings' events, the Fed's calendar, the companies' fundamentals, sector and attention, the IPO calendar; nothing deployable; the three changes deployed for the 2026-10-21 session — the dip book's meta-model cut, the crash book's after-hours exit, the capitulation book (PREREG167, the user: "Deploy all three.") — and the return-per-day practices, section 1 item 9).

## 1. How a change gets in (the method)

The standing rules are CLAUDE.md's "Standing rules"; in short:

1. **Pre-register** the question, the data, the tests and the decision rule in `PREREG<n>.md` BEFORE any outcome is
   read (`python -m src.backtest.prereg <study_dir>` checks every result file postdates the registration). A choice
   made after seeing numbers is an AMENDMENT, labelled exploratory, and the overfitting report shrinks it
   (`docs/OVERFITTING.md`). FROM THE REGISTRATIONS OF 2026-10-21 the checker also fails a study (the user's "Fix
   all" after the audit of 2026-10-20: three slips recurred because only memory guarded them): (a) a stamp the clock
   never gave — approximate, earlier than the section above it, later than the PREREG file's own last write or the
   clock, or a RESULTS stamp earlier than the last write of a result file its heading names (`--now` prints the
   clock's stamp to paste); (b) no `Prior studies:` line — `python -m src.backtest.prereg --prior <study_dir>` (or
   `--prior --text "..."`) lists this log's closest rows before registering (it ranks PREREG111 first for PREREG171);
   (c) a promised output ("reported beside ...") not tagged `[R1]`, `[R2]` ... or never reappearing in a RESULTS
   section or NOTE (`[R2] dropped: why` counts). Research processes run at IDLE from the interpreter's start
   (`scripts/research_idle.py`, a `.pth` in the venv and miniconda): no script needs the low-priority call.
2. **Survivorship-free and point in time**: the delisted names are in every backtest; every live input has a
   `tests/test_no_lookahead.py` guard checked to FAIL against an injected leak.
3. **The production-mimicking simulator** (`src/backtest/mimic.py`, `Mimic()` — its defaults FOLLOW production's
   settings): 20 stratified draws in each borrow-data set, decided on THE POOLED HISTORY A0 + A + B, decisions
   2018-01-02 .. 2026-09-24 (`src/backtest/periods.py`). ITS MARGIN CALLS ARE IBKR'S LIQUIDATION from 2026-10-21
   (PREREG174; the user: "I want the backtest to do exactly the same"): the real-time check along each bar, the most
   recent positions first in 100-share lots only until equity is back at the requirement, each lot 10% against the
   account from where the deficit appears, never past the bar, strict (no Soft Edge Margin), the end-of-day Reg T floor
   (`docs/BACKTEST.md`, "IBKR's liquidation"); PRODUCTION'S ACCOUNT LIQUIDATES THE SAME WAY (the user: "production and
   backtest needs to liquidate like IBKR does"; `live_account.plan_liquidation`, pinned to the engine by a parity
   test) and, from PREREG175 (22:05 ET), the simulator's check sees what production's sees — every bar between the
   marks, every book marked at the extended-hours ticks. A "margin call" in the add rule is a (set, draw) pair with any
   liquidation: the live configuration meets 14 (12 draws of 20), mostly one lot of one squeezed short. From
   2026-10-23 the add rule judges them AS A RATE (4, below).
4. **A change to the live combination** passes THE POOLED ADD RULE (`add_rule.pooled_verdict`): paired against the
   live configuration on the same seeds, a pooled gain of at least +0.5 log points a year that no single year carries,
   its margin events not significantly more frequent — AS A RATE from 2026-10-23 (the user: "update the add rule to
   judge margin calls as a rate. We shouldn't judge the performance based on number of margin calls anymore, just the
   performance in rates (return per day and growth)": the one-sided exact binomial on the (set, draw) pairs where one
   side alone meets a margin event, p < 0.05, `add_rule.margin_events`; one extra pair refused before,
   `margin="count"`) —, a drawdown at most 1 point deeper. Risk protection (significantly fewer events / a shallower
   drawdown, no growth) is the user's call.
5. **A new arm** must QUALIFY for the bank (`src/backtest/bank.py`): at least 100 trades in 30 entry months, its return
   per day's 95% interval (a bootstrap over entry months) above its bar — 0 for a short, SPY's average day for a long,
   whatever it holds —, above the bar in both halves and with any year left out, and growth alone at its live slice.
   The floor (100 trades in 30 months) is a convention set with the bank on 2026-10-14, not derived; it is the
   bank's main guard against a small lucky sample (the interval is not corrected for the rules tried; the deflated
   Sharpe ratio is reported beside, never gated — per trade, and from PREREG183 on the arm's MONTHS, its episodes,
   with the months' autocorrelation, skew and kurtosis: `bank.dsr_months`, the vol arm 0.10 on its months at its 753
   trials against 0.56 per trade). Checked on
   PREREG169's L2 (`mintrl169.py`): its edge (a per-trade Sharpe 0.26) needs 134 trades to clear 95% once
   corrected for 4 rules tried, 230 for 20; at 100 trades an arm needs a per-trade Sharpe of at least 0.30.
   Its card: `python -m src.backtest.reliability --build` (A, B) and `--build --sets A0`.
6. **Report** growth in % a year (changes in percentage points), return per day ((1 + average net trade)^(1 / average
   24-hour hold) - 1) and the gross win rate; publish the report as an Artifact.
7. **Deploy** only on the user's go-ahead: production code + its switch, the simulator following it (`Mimic()`
   reproduces the study's draws), tests (the full suite), CLAUDE.md, the runbook, this file, the memory note; then the
   scheduler child restart (`docs/DEPLOYMENT.md`, "Deploying new scheduler code"). A new arm follows CLAUDE.md's
   "Adding a live book (an ARM)".
8. **A TUNING** (user 2026-10-19: "The parameters should be optimized togheter yes", and "by cross validating and
   tuning we should always have out is sample tuning data") searches its parameters TOGETHER on the FULL ACCOUNT — one
   parameter's best value found with the others at live and combined with the others' fails for growth (PREREG152:
   35.7% a year held out, margin calls in every draw; its return-per-day protocol, each rule chosen on its own arm's
   trades, held best: 61.0%, no call) — and is judged OUT OF SAMPLE: the folds a year held out
   each (PREREG153) or a walk-forward; the held-out result is the headline, a lower number than in sample the expected
   outcome, never re-tuned to look better. Measured (PREREG153, 2026-10-20): the joint search's
   held-out track 57.1% a year (the return-per-day search: 57.8%) against 82.1% in sample and live's 67.3% (in sample).
   THE TWO-STEP METHOD (rules by each book's return per day, then the sizing by the account's growth) measured again
   (PREREG172, 2026-10-21, on ALL3; the user: "For models that have a very good return per day, optimize them for
   return per day and growth"): each very-good book's holding parameters searched TOGETHER (vol, ETF, fall, crash: 366
   configurations), by R or by R x G (G = the book's growth contribution at its slice), cross-validated by year on
   seeds 0-4 — (a) R x G is NOT better out of sample (-1.6 +- 2.7 pp a year against R's track; with the sizing fixed
   at live -2.6 +- 1.9): its freedom on the ETF and fall books picks growth-heavy settings that lose held out (2018,
   2026); (b) THE SIZING BY GROWTH OVERFITS: in every rule configuration it picks the fall slice 1/6, the crash 1/2.5
   and the de-leveraging 80 / 60% for +0.2 .. +0.9 log points each in sample, and its held-out tracks meet margin calls
   in 4-5 draws of 5 where live meets none — a sizing step must count the margin calls (or keep the live sizing);
   (c) by R on the joint grid the vol arm's time limit moves 10 -> 12 (7 of 9 folds; the one-at-a-time protocol chose
   10 with the give-back at 1.25): held out level with live (+0.3 +- 0.6 pp, the live sizing); (d) at the BOOK level
   the user's diagnosis holds out — the vol arm's 10-session limit covers squeezed shorts at their worst, a 20-session
   limit raises its held-out growth contribution in 7 of 9 years and brings its card's significance back (in sample).
   AND A JOINT SEARCH OF A FEW PARAMETERS STILL OVERFITS WHERE THEY DO NOT MATTER (PREREG173, the three short arms'
   time limits, 343 combinations on the account): in sample the best combinations move only the vol arm, yet each
   fold's search also moves the ETF or fall limit on noise and loses those held-out years (-4.3 pp a year against
   live); tuning the vol arm's limit ALONE holds out (+2.7 +- 1.2 pp, exploratory). Tune a parameter only where its
   in-sample profile shows a real gain; keep the flat ones at live.
9. **RETURN PER DAY IS THE METRIC TO OPTIMIZE** (user 2026-10-20: "Return per day is the metric to optimize on because
   it will increase total growth by having multiple models with high return per day. Beating the market is better than
   not beating it, but if the market is highly bullish and we are bullish, then it's ok because we'll still have good
   growth."; CLAUDE.md's standing rule of that name). What the review of every lead and failed attempt (R166) and
   PREREG161-167 established:
   * judge and improve each BOOK by its own return per day, net of the real costs, over the longest history its data
     allow, with the account's pooled growth as the guard — many books each with a high return per day raise the total
     growth; never the ACCOUNT's per-trade return per day (it rises by cutting a slower book: cutting the dip book +0.07
     %/day for -47 pp a year; a good slow book lowers it: the capitulation book 0.745 -> 0.704 %/day for +3.4 pp);
   * a rule change for return per day passes when the changed book's return per day rises (paired t >= 2 over the
     draws), the pooled growth does not fall, its margin events not significantly more frequent (the rate test from
     2026-10-23), the drawdown at most 1 point deeper
     (PREREG166's rule); a shorter-hold exit must keep its AVERAGE TRADE too (XC held its return per day on 2010-2017,
     not its average trade);
   * the market's average day stays the bar for a long arm's QUALIFICATION as an edge, but a bullish book that only keeps
     up with a strongly bullish market may still join on the add rule (its growth in the account);
   * a new long book is tested in its natural capital PLACE — one that should yield gives way first (a quality under the
     dip book's, preemptible whole, de-levered first, `Rules.fills_make_room`): PREREG165's crash place lost on A0, its
     give-way place added; at the dip book's own place every new long book crowded the 1/2 slices (PREREG164);
   * a market-wide condition is measured on a FIXED-SIZE universe (the share of the 100 most-traded stocks), never a count
     of signals that grows with the market (PREREG163's K5 drifted, PREREG165's breadth did not);
   * before 2018 the whole-market grouped bars give a SURVIVORSHIP-FREE panel (`studies/prereg165/panel165.py`): the place
     to confirm a rule's per-trade quality out of sample;
   * a CRISIS BUY may be carried by one year (the user: "it's a buy and we expect most years to be good for buying"): read
     in context, not refused; what lifts the WEAK months and contexts is reported beside the yearly rate;
   * the bank's counts are trades AND entry months (PREREG165's registered theta* passed on trades, failed on 25 months);
     a book whose trades do not depend on the draw is judged by its trades' bootstrap — a paired t over draws there
     measures only the other books' noise.
10. **THE TRAINING PROTOCOL** (the user, 2026-10-21: "we first optimize model by model for return per day, then we
   optimize sizing for growth ... For models that have a very good return per day, optimize them for return per day and
   growth", and "retune using the training protocol of optimizing return per day and a mix of return per day and growth
   for models that have a very good return per day"; CLAUDE.md's standing rule of that name; PREREG172 / 176,
   `studies/prereg176/h176.py`). A RETUNE takes four steps, every choice made out of sample: (1) each book's HOLDING
   parameters searched together on a full grid, the book alone (`reliability.arm_alone`), by its return per day R — a
   book whose R is at least the vol arm's on the bank of the day by R x G (G its growth contribution at its live slice);
   at least 2/3 of the live configuration's training trades; (2) then the sizing for growth, every sizing parameter at
   once on the full account (a joint local search from the live sizing), a step allowed only with its margin events not
   significantly more frequent (the add rule's rate test from 2026-10-23; PREREG172 / 176 counted every pair) and a worst within-year drawdown at most 1 point deeper than the live configuration's on the
   training years; (3) leave-one-year-out — each year at the configuration chosen on the other eight, the held-out track
   the headline (at least 2 SE better than live's on the same draws, 5 of 9 years won, margin events not significantly more frequent, the
   drawdown at most 1 point deeper); (4) the in-sample configuration by the pooled add rule. READINGS: under the margin
   calls before PREREG174 (PREREG172) R x G was not better than R alone held out (-1.6 +- 2.7 pp a year) and a sizing step
   without the guards chose margin calls; under IBKR's liquidation with production's marks (PREREG176) the protocol's
   held-out track is 71.0% a year against live's 76.0% (-4.9 pp, 2 of 9 years won, margin-event cells 7 against 3) —
   its stage-1 choices (the vol arm 20 sessions, the fall exit 0.6, the dip and capitulation exits on the 2-session
   average with 7 sessions) raise each book's own return per day in sample but do not hold out, and the guarded sizing
   step moves in only 2 of 10 folds (the new rules alone already break the guards). The protocol stays the way a retune
   is run; the live configuration stands.
11. **A MARKET CONTEXT IS AN INPUT, NOT A SPECIALIZATION** (PREREG177 / 178, 2026-10-22; the user: "Train ML models on
   different market contexts so that they are specialized"): one model per context (SPY's trend x its volatility
   tercile), trained only on that context's past trades, ranks WORSE out of sample than one model trained on every
   trade with the context among its inputs — inside every cell (2018-2026 daily rank IC in the bear cell +0.065 against
   +0.088; above / mid +0.072 against +0.096): a cell's model sees a fraction of the data and the cells' rare
   episodes differ (the bear cell holds both 2022's grind and 2020's and 2025's V-shaped crashes). A context-aware
   model is built by giving it the context, and an arm is ACTIVATED by context through its own rule or the bank's
   activation map — never by training it on one context's rows.

## 2. The live configuration and how it measures (from the 2026-10-21 session: RPD_IS + the dip book's RP33 cut + the crash book's after-hours exit + the capitulation book, PREREG167; from 2026-10-21 15:00 ET the vol arm's own 15-session limit, PREREG173's V15; the bank of 2026-10-18)

ONE simulated IBKR margin account (no deposits) funds every book (CLAUDE.md "THE LIVE ACCOUNT") — $50,000 from THE
FRESH START of 2026-10-20 (every live position closed at the 09:30 tick; the user: "I actually want the live account
to restart at $50,000"), $10,000 before:

| Book / arm | Side, horizon | The rule in one line | Slice | Since |
|---|---|---|---|---|
| vol arm | short, ~18 days | the most volatile name (30-min ATR%), fresh against its own 20 sessions, risen over 5 sessions, short interest ≤ 1 day, relative volume ≥ 1.58; cover at 50% give-back of the run-up, 4x, or 15 sessions from 2026-10-21 (PREREG173's V15, its own limit; 10 from 2026-10-20; 125% / 15 to 2026-10-19) | 1/20 | 2026-09-28 |
| ETF arm | short, ~9 days | the vol rule on exchange-traded products (no VIX-linked), entered on the turn (½ ATR off the high within 5 sessions); cover at half the give-back, the volatility exit in profit, 6x, or 10 sessions (the whole / 15 to 2026-10-19) | 1/16 | 2026-10-05 |
| fall arm | short, ~5 days | the vol arm's own pick when it did NOT rise, entered on the turn; volatility exit at 70% of the pick bar's ATR%, 6x cover, 10 sessions (50% / 15 to 2026-10-19) | 1/8 (1/12 to 2026-10-19) | 2026-10-09 |
| V6 pressure extras | short | the vol pick's passing backups under high borrow pressure (LightGBM) | vol's | 2026-10-10 |
| dip book | long, ~6 days | $1B+/day stocks above the 200-day average with 2-day RSI < 10; the meta-model's RP33 cut first (from 2026-10-21), then each day's top pick by its score among those trading $2B+ a day, earnings dips ranked last; ½-ATR limit, 3 sessions; sell after the first close above the 3-day average or 5 sessions (5 / 5 / 10 to 2026-10-19) | 1/2 (2 open; 1/3 to 2026-10-19) | 2026-10-09 |
| crash book | long, ~4 days | QQQ under its 250-day average with 2-day RSI < 5 -> TQQQ at the open; sell after QQQ's first close above its 3-day average or 5 sessions (200 / 5 / 10 to 2026-10-19) — at the 16:30 after-hours tick of the triggering session from the 2026-10-20 evening (the next open before) | 1/3 (1/4 to 2026-10-19) | 2026-10-12 |
| capitulation book | long, ~5 days | on a session whose breadth (the share of the 100 most-traded stocks under their 200-day average with 2-day RSI < 10) at the previous close is at least 20%: every $2B+/day stock under its own 200-day average with 2-day RSI < 10, ½-ATR limit, 3 sessions, the dip book's exit (3-day average or 5 sessions); its capital gives way first | 1/6 (3 open) | 2026-10-21 |
| shadows | — | the model arm (V2) and vol2 (scored, journaled, never traded); the vol shorts' news hold (PREREG136) | — | — |

De-leveraging at 70% -> 50% of the equity (80 -> 60% to 2026-10-19), the capitulation book's longs first.

THE ACCOUNT AS DEPLOYED, UNDER IBKR'S LIQUIDATION (PREREG174, the simulator's default from 2026-10-21 19:40 ET; ALL3
with the vol arm's own 15-session limit — PREREG173's V15 —, 20 draws a set, `studies/prereg174/runs`, label PATH =
`bank.LIVE_RUNS`): **+75.7% a year** (draws +65.1 .. +86.5), the worst drawdown 25.8% (34.2% in the worst draw), a margin
event in 13 draws of 20 (15 set-draw pairs: A0 2, A 11, B 2 — 11 of them one after-hours bar, ICCT's de-SPAC squeeze of
2023-08-30, one 100-share lot each), months positive 76.8%, the median month +3.81%, the worst -15.4%, the worst 12
months +9.3%, the longest time under water 5.6 months. With every position closed at the tick marks (the rule before,
V15's draws `studies/prereg173/runs`): +76.6% a year, margin calls in 2 draws (2024-05-09), the same drawdown and months.
V15 against ALL3 under IBKR's liquidation +2.87 pp a year, margin-event pairs 16 -> 15: the add rule's ADD (its REJECT
under the old rule was one call pair).

THE ACCOUNT AS DEPLOYED FROM 2026-10-21 (PREREG167's ALL3 = RPD_IS + RP33 + XC + CBg6_20; the user 2026-10-20: "Deploy
all three."; 20 draws a set, `studies/prereg167/runs`, the bank's reference `bank.LIVE_RUNS`; the default `Mimic()`
reproduces its 60 accounts exactly): **+73.7% a year** (+7.19 +- 0.13 pp over RPD_IS; A0 +2.69, A +12.95, B +4.28 pp),
margin-call draws 1 -> 1, the worst drawdown 26.0 -> 26.5% (36.5% in the worst draw), months positive 75.0 -> 75.3%, the
median month +3.18 -> +3.57%, the worst 12 months +6.7 -> +9.5%, the longest time under water 5.8 -> 5.6 months; every
market context higher (below SPY's trend in high volatility +21.1 -> +32.7% a year, below / mid +24.7 -> +30.4, above /
high +121 -> +134, above / mid +74 -> +78, above / low +67 -> +72); years 2018 +40.5%, 2019 +23.1, 2020 +110.2 (-18 pp),
2021 +82.0 (-7), 2022 +66.5 (+27.5), 2023 +33.5, 2024 +74.0, 2025 +111.2, 2026 to September +137.0; the trades' return
per day 0.745 -> 0.758 %/day; no year carries it (without 2022 +2.44 log points); shrunk for 30 trials +0.63 log
points. Caveats: XC is not confirmed per trade on 2010-2017, RP33 alone is carried by 2024, the capitulation book's
breadth plateau fails at 15% and 2022 carries much of it. THE PARAGRAPHS BELOW are the configurations before it.

THE ACCOUNT AS DEPLOYED FROM 2026-10-20 (PREREG152's return-per-day tuning on all nine years, RPD_IS — the user
2026-10-20: "Can you deploy the config of the 'Held out: return per day, one at a time' to the live prod?"; PREREG159,
20 draws a set, `studies/prereg159/`): **+66.4% a year** (draws +52.8 .. +74.0); worst drawdown 26.0% (35.3% in the
worst draw); a margin call in 1 draw of 20 (A0 1); 75% of months positive, the median month +3.2%, the worst -15.7%,
the worst 12 months +6.7%, the longest time under water 5.8 months; 2018 +24%, 2019 +20%, 2020 +128%, 2021 +89%, 2022
+39%, 2023 +33%, 2024 +52%, 2025 +110%, 2026 to September +133%; the trades' return per day 0.75 %/day (held 4.9
days). Against the configuration below on the same draws: -1.4 +- 1.0 pp a year (all from 2026 to September), margin-
call draws 7 -> 1, the worst drawdown 31.1 -> 26.0% — RISK PROTECTION, the user's call; held out (each year at the
settings tuned on the other eight, PREREG152) 61.0% a year with no margin call. EXPLORATORY (chosen after PREREG152 /
153; 14 changes at once). THE PARAGRAPHS BELOW describe the configuration deployed from 2026-10-19 (the cards,
PREREG150's shares and AUDIT151 were measured on it); the bank's cards and the live calibration's
reference were rebuilt for the deployed configuration on 2026-10-21 (THE BANK below).

THE ACCOUNT on the pooled history (PREREG147's S3 draws = the deployed configuration — the dip book's pick among the
$2B+ signals at 1/3 on the own-security dip panels —, which the default `Mimic()` reproduces exactly; 20 draws):
**+71.5% a year** (draws +58.7 .. +86.0); worst drawdown 30.3% (41.2% in the worst draw); a margin call in 7 draws of
20 (A0 1, A 1, B 5: set B's lendability is modelled and likely pessimistic); 79% of months positive, the median month
+3.7%, the worst -15.5% (-22.8% in the worst draw), the worst 12 months +6.6% on average, the longest time under water
7.0 months; every calendar year positive in every draw — 2018 +29% (SPY -6%), 2019 +29% (+29%), 2020 +130% (+16%), 2021
+50% (+27%), 2022 +51% (-19%), 2023 +32% (+24%), 2024 +54% (+23%), 2025 +122% (+16%), 2026 to September +207% (+13%).
By set: A0 +54.6%, A +52.1%, B +136.7% a year. By market context (SPY's trend x its volatility tercile at each month's
start): above the trend +86.1% (low volatility) / +75.1% (mid) / +87.1% (high), below it +31.8% (high) — the weakest
cell, where a new arm is worth the most. The configuration before it (the $1B pick at 1/6; PREREG142's R142_OLD,
`cache/ml/bank/2026-10-18.json`'s first build) read +56.7% a year (+56.1% before the reused-ticker fix).

THE BANK (the cards of 2026-10-21 under the configuration deployed for the 2026-10-21 session — RPD_IS with ALL3 —,
each arm alone at the live costs, pooled 2018-2026; `cache/ml/bank/2026-10-21.json`, seed 0):

| Arm | Verdict | Trades (months) | Return per day, 95% | Gross win | Alone at its slice | Earns above its bar in |
|---|---|---|---|---|---|---|
| vol | QUALIFIES (from 2026-10-21 15:40 ET: the 15-session limit, PREREG173's V15; FAILED significance under 10 sessions, 218 (79), +0.751%, -0.177 .. +1.649) | 217 (79) | +0.910% (+0.217 .. +1.634) | 83% | +10.4%/yr, dd 21% | SPY above its trend, high and mid volatility |
| ETF | QUALIFIES | 115 (56) | +1.210% (+0.499 .. +2.199) | 88% | +4.8%/yr, dd 4% | above the trend, low and mid volatility |
| fall | INSUFFICIENT (too few) | 59 (44) | +3.177% (+1.002 .. +6.866) | 95% | +6.2%/yr, dd 8% | — |
| dip | QUALIFIES (bar SPY +0.038) | 331 (89) | +0.537% (+0.374 .. +0.718) | 72% | +48.8%/yr at 1/2, dd 18% | above SPY's trend in every tercile |
| crash | QUALIFIES on 1988-2026 | 15 (14) pooled; 116 longest | +4.664% (+1.821 .. +10.514) | 100% | +4.3%/yr, dd 0% | — |
| cap | QUALIFIES re-judged on 2007-2026 | 135 (40) pooled; 284 (72) longest | +0.394% (+0.036 .. +0.847); longest +1.138% (+0.586 .. +1.702) | 67% | +4.0%/yr at 1/6, dd 7% | — |

From the 2026-10-21 23:35 ET rebuild (the marks production's check sees, PREREG175: every card's trades unchanged, the combination's draws NEWMARKS on 40 draws a set — +74.8% a year, the worst drawdown 25.4%, a margin event in 28 draws of 40, the worst 12 months +9.0%), the 19:55 ET one (the simulator's margin calls as IBKR liquidates, PREREG174: the combination's draws PATH — +75.7% a year on 20 draws, a margin event in 13 draws of 20, mostly one 100-share lot; with every position closed at the tick marks +76.6% and 2 draws) and the 15:40 ET one (the vol arm's own 15-session limit deployed, PREREG173's V15) the vol card QUALIFIES (above). Before it, the vol arm's card under RPD_IS's exits (the give-back 0.5, 10 sessions) FAILED significance: its average trade halved
(+11.3 -> +5.4%; the 2026-10-19 card under the 1.25 give-back qualified, +0.60 %/day, +0.23 .. +0.92) while its
squeezes stayed (RGC -318% in 2025): the shorter holds raise its return per day and lower its significance (deflated
Sharpe 0.18 over 753 trials). One seed; the arm is unchanged (RPD_IS was the user's risk protection) — section 5.
On the account's construction (V6's extras included, PREREG172's stage 1) the live vol card reads 229 trades, +0.825
%/day (95% -0.087 .. +1.707): still failing; at a 12- or 20-session time limit it qualifies (+0.989 / +0.924 %/day,
+0.247 / +0.342 .. — in sample, section 4 item 12).
The capitulation book misses significance on the pooled history by 0.002 %/day and passes every other check, so the
re-judging reads its longest history: PREREG165's untouched 2006-2017 trades (`bank.LONG_TRADES`; every signal on
daily bars at 2 bp a side, the pooled part the card's arm-alone trades) — deflated Sharpe 0.73 over 25 trials.

The cards build each arm WITHOUT V6's pressure extras (`reliability.build` uses a plain `VolEngine`; the study harness
installs V6 for the combination's draws only), so the vol card leaves out its ~7 extra shorts a year; the account
figures above include them.

EACH ARM INSIDE THE ACCOUNT (PREREG150, 2026-10-18: S3's 20 draws, and the account with each arm removed on the same
seeds; `studies/prereg150/verdicts150.json`): trades a week / average net trade / return per day / net win / share of
the P&L / what the account loses without it — dip 0.90 / +2.05% / 0.36% / 68% / 64% / -43.2 pp a year; vol 0.56 /
+12.1% / 0.64% / 74% / 20% / -22.6 pp; fall 0.16 / +8.3% / 1.46% / 71% / 8% / -7.3 pp; ETF 0.30 / +2.7% / 0.30% / 72% /
5% / -3.4 pp; crash 0.04 (2 a year) / +5.8% / 1.56% / 76% / 3% / -2.3 pp. Without the three short arms the account
grows 38.7% a year (-32.8 pp; every year lower, 2022 +8% against +51%), meets no margin call (7 draws of 20 with them)
and its worst drawdown is 25.3% (30.3%); the dip book trades as often beside the shorts (0.90 vs 0.89 a week; the
give-way trims its average trade +2.19 -> +2.05%). The dip book alone in the account: 36.8% a year, 60% of months up.

THE BACKTEST VALIDATED (AUDIT151, 2026-10-18; `studies/audit151/`, report
https://claude.ai/artifact/8YXe8DrUXiAMskzY6wCuLz): the simulator makes production's decisions — on the 15 live
sessions its construction reproduces every journaled short-arm pick (the top-1 on all 195 bars of each arm) and the dip
and crash decisions, the dip scores are walk-forward (every year reproduced), the account's rules are production's —
but its model of the market had seven mistakes (section 5): with all of them corrected the deployed account reads
**68.0% a year** (-3.6 +- 0.9 pp; calls 7 = 7, dd 31.1%, months up 77%, the worst 12 months +3.1%; 2021 +89% with the
dip book active, 2024 +44%, 2025 +96%); inside it vol 0.47 trades a week at +10.6% (0.56 / +12.2% as stored), the
dip book 1.01 at +2.04%. THE CORRECTIONS ARE `Mimic()`'s DEFAULTS FROM 2026-10-19 (the user: "Yes please do 1 and
2"; `docs/BACKTEST.md` "The market model corrected"): the default's own draws DEF151 — every traded leg at its real
quote where Polygon has one, the open-fill table keyed by limit and covering every order — read **67.9% a year**
(draws 55.9 … 82.5; -3.66 +- 0.85 pp against the stored construction, -0.07 +- 0.04 against the audit's ALLW; calls
7 = 7, dd 31.1%), the bank's reference (`bank.LIVE_RUNS`, `cache/ml/bank/2026-10-19.json`: +67.8% a year from the
mean log rate). The figures above this paragraph are the stored construction's (the studies before 2026-10-19). In
sample still: the measured retention (rule tweaks ~15%, sizing ~50%) puts a central live expectation near 55% a
year. OUT OF SAMPLE (PREREG153, 2026-10-20): every year
scored by the 32 parameters tuned jointly WITHOUT it grows 57.1% a year (the growth search; 57.8% by return per day) against 67.3% for the live settings on the same five draws — live's figure is in sample; the cautious expectation is between the high 50s and about 60% (PREREG152's
one-at-a-time return-per-day tuning held out 61.0% with no margin call).

THE LIVE RECORD (set C, the live account from 2026-10-09 — its equity also carries an ETF short of 10-08;
`cache/ml/live_calibration/<day>.json`, the ledger): to 2026-10-16, 14 trades (12 dip longs under the earlier rules, an
ETF and a faller short), the account at $8,679 (-13.2%)
at the 2026-10-16 marks — the dip longs -$1,378 (9 open, all chip-related, in a sector fall: SOXX -16% from 10-06 to
10-16, SPY -2%), the shorts +$57 — far too few to judge an arm; the daily calibration compares each assumption
(lendability, refusals, entry delay and drift, fills, borrow charges) with the simulator's. HOW LONG IT NEEDS
(PREREG183, the paper's formulas with the account's own skew, kurtosis and autocorrelation; the bank prints it as
``combination.track_record``): at the backtest's own monthly Sharpe (0.61), 10 months for an even chance to show the
account makes money and 21 months for 80% power; at half of it, 41 and 85 months; 80% power to show the live Sharpe is
half the backtest's takes 67 months (44% at two years). A few weeks of live results prove nothing either way.

HOW THE CONFIGURATION GREW (the account's growth on the pooled history where measured; each step's evidence and
verdict — several were the user's call beside the add rule — in section 3.2):
PREREG57's combination (vol + ETF + fall + dip) -> the turn, the vol 4x cover and the ETF whole give-back (PREREG76-77)
-> per-arm slices (PREREG80) -> the crash book in TQQQ (PREREG86 / 88) -> the vol give-back 1.25 and the ETF arm's VIX
block (PREREG97 / 102) — +48.5% a year -> the dip book's daily top pick at 1/6 with at most two shorts of one name
(PREREG116's Q1, +2.89 log points) — +52.8% -> the earnings-dip demotion (PREREG137, +3.25 percentage points, post hoc)
— +56.1% -> the reused-ticker fix (2026-10-18: the dip book's names read from their own security, production and the
simulator's panels; PREREG142's measured correction, +0.6 points) — +56.7% (the bank's pooled reading; 56.8% as the
studies' mean over draws) -> the dip book's pick among the $2B+ signals at 1/3 (PREREG147's S3, the user's call: +14.8
pp, the drawdown 26.4 -> 30.3%) — +71.5%.

## 3. What the studies found

### 3.1 What each area has taught (the logic the configuration rests on)

- **The vol arm's rule is hard to beat by searching.** Every parameter search over its rules — Optuna on a tune / test
  split, sliding and walk-forward windows, weekly re-tunes, a "less random" selector — lost to the fixed live rule
  out of sample (PREREG1-6, 8, 23, 26, 30, 33-35, 44); the live values sit on their pooled plateaus (PREREG109). What
  moved it forward were STRUCTURAL changes measured one at a time: the give-back (25% -> 100% -> 125%, PREREG6 / 7 / 97),
  freshness (+7.6 pp, PREREG36), the same-bar fallback for an unlendable pick (PREREG16), the 4x squeeze cover with
  no volatility exit (PREREG77), the extras under borrow pressure (PREREG71). Its edge is the single most volatile
  name: widening to more names, more bars or weaker filters fails on squeezes (PREREG72, 89, 90).
- **Borrow is a real cost and a real constraint, and production must read IBKR's own data.** IBKR's real daily fees
  cut the vol arm's return per day 2.19 -> 1.72% and its growth +57% -> +28% a year against an assumed flat fee
  (PREREG13); only ~1 pick in 3 was lendable 2021-24 (PREREG14);
  waiting or retrying later loses (PREREG17, 18, 25, 28), covering when the pool empties loses (PREREG19), forecasting
  unlendability to short loses (PREREG69); set B's missing file is modelled (PREREG67 / 94) and that modelling does not
  inflate set B's growth but deepens its drawdowns and margin calls (PREREG93).
- **The account's risk is the squeeze on a held name.** Every margin call of the live draws was a squeeze on a name the
  account held (PREREG113); the defences are the per-arm slices (PREREG80: bigger slices meet calls), IBKR's house
  margin with the what-if and the 2.86 shrink (PREREG12), the 4x / 6x covers, at most two shorts of one name
  (PREREG113 / 116), the dip book giving way and de-leveraging at 80% -> 60% (PREREG54, 125). Sizing by squeeze risk
  failed (PREREG91); every 3x add-on failed on the cushion (PREREG123-125).
- **The dip book's edge is the mega caps' reversal, and ranking matters.** RSI2 < 10 above the 200-day average in $1B+
  names beats SPY untouched 2007-2020 (PREREG43); the ½-ATR limit raises its return per day (PREREG60 / 61); the
  meta-model's daily top pick at 1/6 adds growth (PREREG107 / 116) — and the score's ranking is worth about 11
  percentage points a year over picking the deepest dip (PREREG137 Amendment 5); passing over earnings-reaction dips
  adds ~3 more in sample (PREREG137, post hoc). It earns little below $1B a day (PREREG140) and nothing on ETFs
  (PREREG45).
- **New arms are rare.** Of dozens of candidate arms (classic long patterns, insider and earnings events, index dips, gap-down
  reversal, momentum, IPO lockups, insider clusters, overnight / extended-hours drifts, 3x books, news and LLM rules,
  slow data, post-earnings drift; on 2026-10-19 the defensive Treasuries / gold book, S&P 500 index changes, merger
  arbitrage and distance-method pairs, PREREG155-158; on 2026-10-20, on the deployed configuration, the six
  bank-qualified 3x books, gold on the weak days, large-cap dips under their own trend and the pre-earnings
  run-up, PREREG161-164; on 2026-10-20 night, from NEW DATA SOURCES, the event shorts after offerings, reverse splits
  and deficiency notices (168: the offering's drop is in the price by the next open; the reverse-split short's
  2018-2026 edge fails before 2018), the pre-FOMC drift (169: real 1999-2026, gone in the account's years), the
  dip model given fundamentals, sector state and attention (170) and the IPO lock-up expiries again (171); on 2026-10-22, MACHINE-LEARNING models trained per market context ranking large-cap longs and shorts, PREREG177 / 178, and the grinding decline's own short and long books, PREREG179 / 180), only
  the crash-rebound book in TQQQ passed (PREREG86 / 88), and
  the fall arm came from studying the vol arm's own picks (PREREG38 / 57). The weakest market context — below SPY's
  trend in high volatility — is where an arm would be worth the most. A sleeve of idle capital (PREREG86's, PREREG154's
  QQQ lots; PREREG160 on the deployed configuration) does not pass: it spends the margin cushion that absorbs the
  shorts' squeezes, and its lots, sold to make room most of the time, earn about nothing. On the deployed
  configuration the dip book (1/2, 2 open) takes the long side's whole room: a new long book in its place crowds it
  out (PREREG164's pre-earnings book -16 pp a year, PREREG163's every-context dips -22), and one that gives way adds
  exposure, not edge (the gold book, PREREG163's daily picks and PREREG164's trades in a give-way place: -0.1 to -6.8
  pp a year). The one new pattern: large caps oversold
  under their own trend pay on MARKET-WIDE capitulation days only (PREREG163 Amendment 1). Measured on the 100
  most-traded stocks (PREREG165), that capitulation book is confirmed out of sample on 2006-2017 (survivorship-free,
  +1.79 %/day) and, giving way to the other books, adds +3.4 pp a year to RPD_IS (2022 the largest share) — with
  PREREG166's two rule changes +7.2 pp (PREREG167): DEPLOYED from the 2026-10-21 session (the user: "Deploy all
  three."), its threshold's plateau not met at 15%.
  Read by RETURN PER DAY (the review of 2026-10-20), no failed new book earned per day in the account what the
  deployed books earn; the rule changes the review recovered (the dip meta-model's cut, the crash book's after-hours
  exit) did (PREREG166).
- **News and the local LLM have not produced a trading edge.** After-close 8-Ks to the next open, headline sentiment,
  headline filters on every arm, news-timed exits, slow filings data and large-cap earnings drift all fail
  (PREREG129-135, 139, 141); the LLM answers FACTUAL questions usefully (was guidance raised / lowered?) but not
  judgment ones (is the business improving?: 95% "improving"). The two survivors are a shadow (the news hold,
  PREREG136) and the earnings demotion, which uses the release's timing, not its text.
- **Method lessons.** A one-year or one-set gain is usually a data artifact or a peak (PREREG116's O4, PREREG122's M3,
  the WDC spin-off); context activation by market cell did not help (PREREG108); the overfitting report keeps ~15% of
  a rule tweak's in-sample gain (PREREG81 / 106); look for data bugs first (false splits, symbol reuse, late overnight
  prints, EDGAR's drifting timestamps — section 5).

### 3.2 Every study, one line

Verdicts: DEPLOYED (live now unless noted), SHADOW, BUILT-OFF (built, switched off), REJECTED / FAILED, INSUFFICIENT,
READING (no decision asked), SUPERSEDED, WITHDRAWN. Growth in % a year; lp = log points a year (pooled add rule unless
noted); rpd = return per day. Files: `llm_trader_research/optvol/` (1-39), `long/` (40-43), `etfalgo/` (46-51),
`studies/prereg44/` (44, 52-56), `studies/prereg<n>/` (the rest).

| # | Date | Area | Question | Verdict — key result |
|---|---|---|---|---|
| 1 | 10-03 | vol | Optuna on 13 rules, tune 2021-24, judge 2025-26 | REJECTED — holdout -2.51 vs live +2.47 %/day |
| 2 | 10-03 | vol | the same, objective rpd and average trade | REJECTED — +1.28 vs +2.47 %/day |
| 3 | 10-03 | vol | the same, every tuning year equal | REJECTED — +0.22 vs +2.47 %/day |
| 4 | 10-03 | vol | sliding-window re-optimization (108 grids) | REJECTED — 0 of 108 significant |
| 5 | 10-04 | vol | walk-forward, 1-3-month tuning | REJECTED — best +0.13 %/day (-1.22..+1.23) |
| 6 | 10-04 | vol | walk-forward with rule levels | REJECTED; its post-hoc 25% give-back lead deployed, replaced the same day by 7 |
| 7 | 10-04 | vol | give-back x slice count for compounding | DEPLOYED 100% give-back (superseded by 97's 125%) — +0.38 log over 25% |
| 7b | 10-04 | vol | 7 with deposits | not deployed |
| ETF25 | 10-04 | ETF | 25% vs 50% give-back | REJECTED (superseded by 77's 100%) |
| 8 | 10-04 | vol | weekly re-tune on account growth | REJECTED — every re-tuned account margin-called |
| 9 | 10-05 | vol | add ~2,700 stocks outside the deep store | REJECTED by its bar, DEPLOYED by the user (400-bar floor) — added trades +2.15 %/day |
| 10 | 10-05 | sizing | slices x squeeze cover against calls | SUPERSEDED by 11 / 12 |
| 11 | 10-05 | sizing | cover x slices, zero calls (Reg T) | 6x cover live on ETF / fall; 10 slices overturned by 12 |
| 12 | 10-05 | sizing | the same under IBKR's house margin | DEPLOYED house rates, the 2.86 shrink, the what-if (24 slices superseded by 80) |
| 13 | 10-05 | borrow | re-measure with IBKR's real daily fees | READING — rpd 2.19 -> 1.72, growth +57 -> +28% |
| 14 | 10-06 | borrow | could IBKR lend each pick 2021-24 | READING — 94 of 260 lendable |
| 15 | 10-06 | borrow | gate on 2x the order's shares instead of $10k | REJECTED ($10k kept) |
| 16 | 10-06 | borrow | short the same bar's next lendable name | DEPLOYED (ranks 2-5) — +0.86 pp (+0.04..+1.94) |
| 17 | 10-06 | borrow | enter an unlendable pick the next session | REJECTED — delayed trades +1.6% vs +9.5% |
| 18 | 10-07 | borrow | retry later while half the room is kept | REJECTED |
| 19 | 10-07 | borrow | cover when IBKR's pool empties | REJECTED (+2.50 vs +9.21%); journal-only watch live |
| 20 | 10-07 | vol | lower the $5 / $5M floors | REJECTED |
| 21 | 10-07 | borrow | fallback depth 10 instead of 3 | REJECTED; the user deployed depth 5 |
| 22 / 22b / 22c | 10-07 | vol | skip / cover halted names | REJECTED; the entry halt guard is live |
| 23 | 10-07 | vol | Optuna on entry rules under live limits | REJECTED |
| 24 | 10-07 | sizing | fewer slices on the lendable book | REJECTED (superseded by 80) |
| 25 | 10-07 | borrow | retry keeping 10-30% of the room | REJECTED |
| 26 | 10-07 | borrow | Optuna with borrow variables | REJECTED |
| 27 | 10-07 | borrow | skip names whose pool drained | REJECTED — the removed trades averaged +15.4% |
| 28 | 10-07 | borrow | retry if the price is less than x% worse | REJECTED |
| 29 | 10-07 | sizing | size thin names by tier | REJECTED |
| 30 | 10-07 | evaluation | objective = final growth with calls inside | rule REJECTED; the objective became the standard |
| 31 | 10-07 | other | short the reopen after a halt | REJECTED (73% of pauses lower 5 sessions later, untradeable) |
| 32 | 10-07 | vol | thin stocks ($1-5M) as an arm | REJECTED; deployed 10-07, BUILT-OFF since 10-08 |
| 33 / 34 / 35 | 10-07 | evaluation | weekly Optuna walk-forward / on $10k / less random | REJECTED — none beats the fixed rule |
| 36 | 10-07 | vol | each live rule's effect | READING — freshness +7.6 pp, fallback +1.1, crowding -4.3 (still live) |
| 37 | 10-07 | vol | the vol arm with five rules only | REJECTED; its variant is vol2 (SHADOW since 10-09) |
| 38 | 10-08 | fall | long after a cover; short the fallers | long REJECTED; fallers -> 57 -> the fall arm |
| 39 | 10-08 | model | rank as if every order were accepted | REJECTED (shadow suggestion) |
| 40-42 | 10-08 | long | classic patterns; learned 5-day rise; insider / earnings events | FAILED |
| 43 | 10-08 | dip | RSI2 dips in the most liquid stocks | DEPLOYED M1_1B — 2007-20 3,063 trades, 71% win, +0.54% a trade |
| 44 | 10-08 | evaluation | weekly joint re-tune of vol + dip | REJECTED |
| 45 | 10-08 | ETF / dip | dips on ETFs; trend-filtered ETF shorts | REJECTED |
| 46-50 | 10-08 | ETF DM | overnight, risk parity, intraday 3x, GTAA, 3x Nasdaq trend | FAILED |
| 51 | 10-08 | ETF DM | 1x dual momentum on untouched 1988-2004 | BUILT-OFF (passed: +15.5 vs +12.1%) — never re-judged under the bank |
| 52-56 | 10-08 | sizing | capital priority, give-way variants, rpd-sized orders | 54 (the dip book gives way, whole) live; the rest REJECTED / tie |
| 57 | 10-08 | combination | better models and the best combination | DEPLOYED 10-09 (vol + ETF + fall + dip, give-way, de-leveraging) — fallers +6.3 lp |
| 58 | 10-09 | sizing | sell the latest or the earliest dip long first | latest-first live |
| 59 / 60 | 10-09 | dip | delay the dip buy; judged on rpd | 60's ½-ATR limit DEPLOYED — rpd 0.214 vs 0.170 (Holm p 0.004) |
| 61 | 10-09 | dip | the resting limit in the shared account | DEPLOYED (fills cut to the room) |
| 62-64 | 10-09 | simulator | make the simulator mimic production; fix its bugs and production's | DEPLOYED `Mimic()` (and PREREG64's production fixes: give-way, BRK, splits) |
| 65 | 10-09 | evaluation | the combination on the corrected simulator | READING |
| 66 | 10-09 | shorts | ½-ATR sell limits for the short entries | REJECTED — misses the immediate fallers |
| 67 | 10-10 | borrow | model set B's lendability and fees | DEPLOYED (fees superseded by 94's F4) |
| 68 | 10-10 | borrow | daily backfill models on unseen years | lend part FAILED; nothing adopted |
| 69 | 10-10 | borrow | short names forecast to turn unlendable | REJECTED — the fee prices the fall |
| 70 | 10-10 | vol / fall | pick the arms' shorts by the forecast | REJECTED; the V4 lead -> 71 |
| 71 | 10-10 | vol | V6: extra shorts under borrow pressure, set B | DEPLOYED — extras +2.42 %/day (Holm p 0.033) |
| 72-74 | 10-10 | vol / borrow / other | widen the pressure shorts; availability rules; buy predicted squeezes | REJECTED |
| 75 | 10-10 | evaluation | 31 variants of the books | READING — none better |
| 76 | 10-10 | fall | timed entries: a limit or the turn | the faller turn DEPLOYED on the user's order (fall rpd 0.90 -> 1.62 / 1.86 -> 2.10) |
| 77 | 10-10 | vol / ETF | earlier leads under the add rule | DEPLOYED vol 4x cover, no vol exit, ETF whole give-back, ETF turn |
| 78 / 80 | 10-10 | sizing | more trades or bigger; per-arm slices | 80 DEPLOYED vol 1/20, ETF 1/16, fall 1/12 — +7.1 / +15.4 pp, no call |
| 79 | 10-10 | evaluation | remove any model? | REJECTED — every book adds growth |
| 81 | 10-11 | evaluation | significance and overfitting of the combination | READING — PBO 0.23-0.32; rule tweaks keep ~10-25% out of sample |
| 82 | 10-11 | evaluation | tighten the add rule, re-check deployments | rule superseded (pooled, 10-14); the ETF reverts it suggested NOT applied |
| 83 | 10-11 | borrow | set B's fees vs set A's | READING — set B dearer (vol 315 vs 147 %/yr) |
| 84 | 10-11 | dip | is set B's dip edge the bull market? | READING — 2.5x the signals, not better trades |
| 85 / 87 | 10-11 | evaluation | each book by year vs SPY / QQQ; rpd and capital share | READING — shorts carry bear years, the dip book bull years |
| 86 | 10-11 | new arms | four models for uncovered contexts | QQQ crash-rebound DEPLOYED; mirror-dip, SPY crash, earnings drift FAILED |
| 88 | 10-11 | crash | trade the QQQ decisions in TQQQ or FNGU | DEPLOYED TQQQ at 1/4, 75% margin — +1.6 / +2.0 lp, no call |
| 89 / 90 | 10-11 | fall / vol | more faller-style trades; re-short after the target | REJECTED |
| 91 / 92 | 10-11 | sizing / exits | size by squeeze risk; close trades not in profit | REJECTED |
| 93 | 10-12 | borrow | is set B inflated by the modelling? | READING — not inflated; drawdowns pessimistic |
| 94 | 10-12 | borrow | F4 fees + linked lend draws | DEPLOYED (the user's choice; registered bar not met) |
| 95 | 10-12 | simulator | stratified draws | DEPLOYED (paired noise -22..-50%) |
| 96 | 10-12 | dip | the meta-model: skip low-scored signals | filter BUILT-OFF (RP33); the regressor's score ranks the daily top pick (116) |
| 97 | 10-13 | all arms | parameters to their plateau centres | G1 (vol give-back 1.25) DEPLOYED; the others REJECTED |
| 98 | 10-13 | all arms | horizons from half-lives; RSI2 exits | REJECTED |
| 99 | 10-13 | shorts / sim | halted picks: wait or backup | REJECTED; the simulator's halt guard BUILT-OFF |
| 100 | 10-13 | dip / shorts | cap positions per sector | REJECTED |
| 101 | 10-14 | evaluation | set A0 (2018-01 .. 2021-01) with IBKR's file | DEPLOYED (part of the pooled history) |
| 102 | 10-14 | ETF | the ETF arm without VIX-linked products | the interim block LIVE; RISK PROTECTION put to the user (no recorded decision) |
| 103 | 10-14 | evaluation | every model in vs out of sample | READING — vol and dip robust; ETF and fall fail out of sample |
| 104 / 109 | 10-14 | fall / all | fall arm variants; 21 parameters re-tuned pooled | REJECTED — every live value kept |
| 105 | 10-14 | simulator | rebuild A and B with A0's construction | BUILT-OFF, under review (never run) |
| 106 | 10-14 | evaluation | guardrails: trial ledger, set D, deflation | the measurement layer live; the gates not adopted |
| 107 / 114-117 | 10-14 | dip | best picks first; top-k; plateau; concentration | Q1 (top-1 at 1/6 + two shorts a name) DEPLOYED 10-16 — +2.89 lp, calls 10 -> 7 |
| 108 | 10-14 | evaluation | activate arms by market context | REJECTED |
| 110 / 118 | 10-14 | new arms | arms for the weak contexts (index dips...) | REJECTED in the account |
| 111 / 112 | 10-14 | new arms | IPO lockup shorts; insider clusters | FAILED |
| 113 | 10-15 | sizing | cap stacked shorts of one name | C2 (two) DEPLOYED via Q1 |
| 119 / 120 | 10-15 | evaluation | gap-down reversal on 2007-17; the bar for long arms | N4 NOT CONFIRMED; re-judging on the longest history DEPLOYED |
| 121 / 122 | 10-15 | 24h | overnight marks / exits, extended-hours orders; overnight arms | REJECTED / FAILED |
| 123 / 124 | 10-16 | evaluation / 24h | the bar = SPY's day for every long; the 3x books | the bar DEPLOYED; every 3x book REJECTED (margin calls) |
| 125 | 10-16 | sizing | de-leveraging thresholds | REJECTED — 80 / 60 kept |
| 126 / 127 | 10-16 | data | the data audits (no PREREG folder: `studies/dataaudit/`) | DEPLOYED split verification and integrity checks; symbol reuse NOT fixed |
| 128 | 10-16 | simulator | the halt guard on Q1; reproducibility | READING — -0.44 lp, not switched on |
| 129 / 130 | 10-16 | news | after-close 8-Ks to the next open; confirm on 2010-17 | FAILED; leads reversed |
| 131 / 132 | 10-16 | news | headline sentiment overnight; headline filters on every arm | FAILED |
| 133 | 10-17 | slow data | insiders, 13D, dividends, congress, analysts, late filers | FAILED |
| 134 / 135 | 10-17 | news / exits | exit or hold on news; dip after-hours earnings exit | FAILED |
| 136 | 10-17 | vol / news | hold vol shorts k sessions after LLM-positive news | SHADOW (from 10-17) — +0.85..+1.96 pp in sample, set A only |
| 137 | 10-17 | dip / news | earnings-reaction dips | DEMOTE_E DEPLOYED for 10-19 (post hoc) — +3.25 pp; Amendment 3 WITHDRAWN (A5: the score's pick is worth ~11 pp over the deepest dip) |
| 138 | 10-17 | dip / news | the effect by the release's guidance | INSUFFICIENT (5 lowered-guidance trades) |
| 139 | 10-17 | news | LLM-read guidance as a large-cap drift arm | FAILED |
| 140 | 10-17 | dip | the earnings-dip effect at $200M-$1B | NOT CONFIRMED |
| 141 | 10-17 | dip / news | the surprise: misses vs beats; drift arms | FAILED (misses did better than beats) |
| 142 | 10-17 | dip | the meta-model's table rebuilt on the repaired stores (reused tickers cut, renamed histories joined), refit | REJECTED — -1.6 pp a year (A +1.7, B -3.8, A0 -2.0 lp); the deployed model stays; the correction alone +0.6 pp |
| 143 | 10-17 | dip | the best meta-model: ten-seed ensembles, LambdaRank by day, within-day / winsorized / fill-aware labels, a smaller model, three dip-day inputs, the volatility heuristic; the seed lottery | REJECTED — no recipe beats the deployed fit (best per pick +0.045, PBO 0.47; account -1.4 .. -9.8 pp); one fit's seed alone moves growth ~3 pp; the deployed fit is the best of its five seeds |
| 144 | 10-17 | dip | the same recipe learning from more dips ($500M / $250M floors, 2.7x / 6x the rows) | REJECTED — -9.4 / -8.3 pp a year; smaller stocks' dips misrank the $1B ones |
| 145 | 10-18 | dip | one model per stock size: the deployed model on the $1B dips, a model fit only on the medium tier ($250M / $500M - $1B) serving it, one daily top pick over both (or the $1B dips first) | REJECTED — -5.0 to -8.8 pp a year; margin-call pairs 7 -> 8-26 (every A0 call on 2020-03-16); the medium model ranks medium dips no better than the deployed one; the deployed model on both tiers +2.4 pp but carried by 2025, calls 7 -> 25 |
| 146 | 10-18 | dip | the dip book on bigger stocks: the pool cut to $2B / $3B / $5B+ (the deployed model ranks), the bigger dips ranked first, a model fit only on the $2B+ / $3B+ dips | REJECTED — $2B pool +0.7 pp (+0.43 lp < +0.5, carried by 2019, calls 15 -> 16 pairs, dd +1.8) at 40 draws; $3B / $5B -6.9 / -11.5; bigger first -1.5 to -4.7; the big-stock models -4.2 / -8.6 (worse than the all-$1B ones); bigger stocks' dips ARE better trades (+1.0% a trade at $1-2B, +1.9-2.3% above), but the model's pick beats the biggest dip on the days both fire and the cut leaves slots idle |
| 147 | 10-18 | dip / sizing | the $2B+ pool with bigger positions (1/5, 1/4, 1/3; the $1B pool at 1/5, 1/4 as controls) | DEPLOYED 10-19 (the user's call: "Deploy the $2B+ stocks only, 1/3 per position"; not the rule's ADD) — +5.4 / +10.0 / +14.8 pp a year (S5 / S4 / S3; S3 71.6 vs 56.8%), margin-call pairs 8 / 8 / 7 vs 7, worst drawdown 28.4 / 29.1 / 30.3 vs 26.4 (S3 fails only the drawdown: a larger gain for a deeper one); the $1B pool at 1/5 / 1/4 +2.1 / -0.5 (the size alone does not do it) |
| 148 | 10-18 | dip / sizing | the $2B+ book at 1/3 with its remaining capital in $1-2B dips (on the days without a $2B+ dip, or the live pick sized by tier) | the question REJECTED — every form loses against the $2B+ book alone (-0.9 to -4.5 pp; the $1-2B trades earn ~0 outside set B); but T3_6 (the live pick, 1/3 at $2B+, 1/6 at $1-2B) ADDS against the live account: +9.8 pp a year (66.6%), calls 7 -> 6, drawdown +0.7 — exploratory, neighbours in PREREG149 |
| 149 | 10-18 | dip / sizing | T3_6's neighbours: the $2B+ slice 1/2 and 1/4, the $1-2B slice 1/8, the tier line at $1.5B and $3B | T3_6 ON A PLATEAU (every neighbour +5.1 to +9.8 pp, >= half its gain; none over one extra call pair) -> OFFERED to the user: +9.8 pp a year (66.6 vs 56.8%), calls 7 -> 6, drawdown +0.7; 1/2 grows as much with a drawdown 2.6 points deeper; without 2020 T3_6 still +3.96 lp |
| 150 | 10-18 | evaluation | the deployed account (S3) with each model removed: the three shorts, each short arm, the dip book, the crash book (and the dip book alone) | EVERY MODEL STAYS — without the shorts 38.7% a year (-32.8 pp; calls 7 -> 0 draws, drawdown 30.3 -> 25.3; every year lower); vol -22.6, fall -7.3 (calls 7 -> 10 without it), ETF -3.4, crash -2.3 (calls 8, drawdown +2.2 without it), dip -43.2 pp; the ETF arm the close call: -3.4 pp against 3 fewer call draws (7 -> 4, all in set B's modelled borrow; on A0 it prevents one) and +5.7 on the worst 12 months -> the user's call, recommended kept |
| AUDIT151 | 10-18 | evaluation | is the backtest representative of live? every live rule and model traced to the simulator, the live days replayed, the inputs' knowledge times, costs and cash flows, the data, execution and operations (six reviews), each mistake measured on 20 paired draws a set (`studies/audit151/`) | THE DECISIONS MATCH (the 15 live sessions: the top-1 short on all 195 bars of each arm, 371 / 371 decisions, the dip and crash decisions identical; the dip scores walk-forward, every year reproduced; the account rules equal; 5,002 tests pass); THE MARKET MODEL HAD 7 MISTAKES: optimistic — short-leg costs (real quotes) -4.3 pp, lendability on the archive's morning file -4.5 (-3.5 .. -5.3), the halt guard -0.7; pessimistic — set A's dip book without its warm-up +5.4, the dip open fills +0.6, dividends +0.2, the ETF arm's reused tickers +0.2; ALL TOGETHER 71.6 -> 68.0% a year (-3.6 +- 0.9), calls 7 = 7, dd 30.3 -> 31.1. Bounds: real-time margin 0 .. -4.6 pp with up to 15 call draws of 20; the A0 pressure model in sample <= -1.0. The corrections are run-scoped patches, not yet `Mimic` defaults (the user's call) |
| AUDIT151 impl. | 10-19 | simulator + production | the user's go-ahead ("Yes please do 1 and 2"): the seven corrections as `Mimic()` defaults (quotes, lendability at the trade, the halt guard, open fills, dividends, the ETF stock-era tickers, set A's dip warm-up) with their data made general (every dip open fill: 5,468 (name, day, limit) rows; the quote store + a fetch tool), the reliability cards and the bank rebuilt; in production the dip open fills linked to IBKR's execution and booked at its price, a leg without a usable fill costed at its own quote, dividends in the ledger, the Flex statement diagnosed | DONE: the default's own draws DEF151 67.9% a year (-3.66 +- 0.85 pp vs the stored 71.6; the bank +67.8%); parity — set B seed 0 equals ALLW exactly; A / A0 every short / ETF / crash piece equals the audit's construction re-run the same day, the differences traced to the audit patch's (name, day) open-fill key and IBKR's fee history rebuilt after its runs; the cards: vol 218 trades +0.60 %/day, ETF 125 +0.52, dip 446 +0.37 (alone +43.0%), fall 59 INSUFFICIENT; production deployed (5,026 tests pass); the Flex statement NOT fixed: a 7-day window stays "in progress" too — the user's Client Portal check |
| 152 | 10-19/20 | tuning / evaluation | leave-one-year-out cross-validation of a ONE-AT-A-TIME tuning (the user's first ask): each of 25 rule parameters chosen on its own arms' trades (seed 0, arm alone; by return per day with 2/3 of the trades kept, or by growth contribution), the 7 sizing parameters on the full account with the live rules (seeds 0-4), the choices combined per fold and run on seeds 0-4 paired with DEF151 | RETURN PER DAY HOLDS, GROWTH FAILS — held out 61.0% a year by return per day (dd 25.3, no margin call in 5 draws; -6.3 +- 1.3 pp against live's 67.3% in sample; optimism only +5.0 pp; return per day 0.785 %/day held out vs 0.780 in sample and live's 0.500, above live in 7 of 9 years); by growth 35.7% (-31.5 pp; calls in 5 of 5 draws, every configuration — the per-parameter optima combine into squeezes, hence PREREG153's joint search). The RPD sizing saw the scoring draws (only the year held out). Its in-sample configuration RPD_IS (66.0%, dd 24.0, no call in 5 draws) -> PREREG159 |
| 153 | 10-19/20 | tuning / evaluation | leave-one-year-out cross-validation of a JOINT tuning (the user: "The parameters should be optimized togheter yes"): PREREG152's 32 parameters (25 rules, 7 sizing) by coordinate ascent on the full account, draws 0-1, per fold (each year held out) and on all years, two objectives (growth; return per day under live's growth), risk limits on the training years; finals on draws 0-4 paired with DEF151 | HELD OUT 57.1% a year (growth search; dd 37.4, calls 3 draws of 5) / 57.8% (return-per-day search; dd 27.8, calls 1) against live's 67.3% in sample on the same draws (dd 31.5, calls 2); in sample 82.1% / 67.8% — the optimism +24.9 +- 5.1 / +10.0 +- 1.3 pp, live - held out +9.7 +- 4.5 / +9.6 +- 1.5 pp; growth search held out minus live by year 2018 +8.1, 2019 +3.4, 2020 +15.8, 2021 +1.6, 2022 -12.2, 2023 -6.4, 2024 -19.1, 2025 -51.7, 2026 -15.1 pp; every held-out year positive, 8 of 9 above SPY; the return-per-day search keeps 0.63 %/day held out against live's 0.50 (7 of 9 years; holds 5.9 vs 9.1 days) at the growth cost above; the growth search's risk limits, met on its two draws, fail on three new ones (GROWTH_IS calls 3 of 3 vs live 1). Consistent choices: vol cover 8x (17 of 18 folds), crash 1/3, fall 1/8 (growth), dip exit average 7 (growth) vs 3 (rpd). NOTHING DEPLOYED; no configuration a candidate (report https://claude.ai/artifact/7qdZxg75gUc6AgkK6BN5S7, with 152 and 159) |
| 154 | 10-19 | new arm / sizing | the QQQ sleeve: the capital no book uses held in QQQ above its 200-session average, as LOTS (at least 5% of equity each, at the open) that give way first — to a short, a dip fill, the crash long and de-leveraging (the user: "deploy it live if results are good"; `strategies/sleeve.py`, `Mimic(sleeve=True)`, the engine's `Strategy.min_frac`) | REJECTED, NOT DEPLOYED — 67.8 -> 56.6% a year (-11.2 +- 0.7 pp; -6.92 +- 0.44 log points), margin calls in 15 draws of 20 against 7 (19 set-draw pairs against 7), dd 31.1 -> 34.9, worst 12 months +3.1 -> -2.4%: it takes the margin room that absorbs a squeeze (the gate: without it 60 / 60 draws equal DEF151). Amendment A (EXPLORATORY): the sleeve only while the account's initial requirement after its lot stays <= 30% of equity (C30) ADDs +3.15 +- 0.38 pp (70.9% a year; +1.85 +- 0.23 log points; A0 +2.72, A +2.58, B -0.44; without 2019 +1.16; calls 7 -> 5; dd 30.1; worst 12 months +6.6%; shrunk +0.28) — C50 +0.17 pp (more draws), C70 -4.46 pp; Amendment B's plateau check: C10 +1.18 pp, C20 -0.25 pp (11 call draws, 14 pairs), C40 +2.63 pp (B -4.7) — C30 is a PEAK (no neighbour within 10%), the response path-dependent: NOT recommended (a lead) |
| 155 | 10-19 | new arm | the defensive book: TLT or GLD (the better positive 6-month return) held while SPY's previous close is under its 200-session average — the weak context's arm | FAILED — alone 2007-26 76 holds +0.019 %/day (95% -0.027 .. +0.058) against SPY's day +0.029 (TLT's holds lose, GLD's +0.63% a hold n.s.; 2020 / 2022 lose); in the account 1/4 -0.07 pp and 1/2 +0.55 pp a year (-0.04 / +0.33 log points, < +0.5) — REJECTED. Post hoc lead: GLD held on every weak-context day +0.066 %/day |
| 156 | 10-19 | new arm | S&P 500 index changes (fja05680's point-in-time membership, 388 events 2018-26): buy each deleted stock / short each added one at the effective session's open, 5 / 10 / 20 sessions | FAILED — the 95 tradable deletions (the other 100 delist with the change) -0.12 / -0.04 / +0.03 %/day against SPY's +0.038 (10: 95% -0.29 .. +0.22, halves disagree, worst -54%); the 190 additions shorted -0.08 / -0.03 / -0.06 %/day (bar 0) |
| 157 | 10-19 | new arm | merger arbitrage read off prices (Polygon's whole-market daily bars 2017-26): a +15% jump on 3x volume, then 3 sessions pinned within 1% (a cash deal's terms) -> buy at J+4's open, sell at the last close before delisting or after 250 sessions | FAILED — 221 trades in 92 months -0.007 %/day (95% -0.021 .. +0.005) against SPY's day +0.037; gross win 82% but mean -0.77% (worst -90%): the 201 completed deals +1.03% a trade, +0.010 %/day (under T-bills), the 20 that did not close -9 to -38% |
| 158 | 10-19 | new arm | pairs trading, the distance method (Gatev-Goetzmann-Rouwenhorst) in the point-in-time S&P 500: 20 closest pairs of each 252-session formation, open at 2 sigma, close at the crossing, 18 six-month periods 2018-26 | FAILED — 466 trades -0.007 %/day (95% -0.017 .. +0.002) against 0, both halves negative; +0.12% a trade gross, -0.51% net of 4 commissions on $1,000 legs, spreads and borrow; 42% converge in their period |
| 159 | 10-20 | tuning / risk | the pooled add rule for PREREG152's return-per-day configuration tuned on all years (RPD_IS: the vol give-back 0.5, a 10-session limit on the shorts, the ETF give-back 0.5, the fall volatility exit 0.7 at 1/8, the dip exit average 3 / 5 sessions / 3-session order at 1/2 with 2 open, the crash book on the 250-day average with a 3-day exit / 5 sessions at 1/3, de-leveraging 0.7 / 0.5), 20 draws a set; EXPLORATORY (chosen after PREREG152 / 153's held-out readings) | REJECTED AS AN ADD, RISK PROTECTION (the user's call) — 66.4 vs 67.8% a year (-1.4 +- 1.0 pp; all from 2026 to September: without it +2.3 pp); margin-call draws 7 -> 1 of 20, dd 31.1 -> 26.0 (worst draw 44.0 -> 35.3), worst 12 months +3.1 -> +6.7%, return per day 0.51 -> 0.75 %/day (holds 9.1 -> 4.9 days); held out (PREREG152) 61.0%, no call. Which of the 14 changes cuts the calls is not measured. DEPLOYED from the 2026-10-20 session (the user: "Can you deploy the config of the 'Held out: return per day, one at a time' to the live prod?"), with THE FRESH START: every live position closed at the 09:30 tick, the account restarting at $50,000 |
| 160 | 10-20 | sizing / risk | PREREG154's QQQ sleeve (the capital no book uses in QQQ above its 200-session average, lots that give way first) on the configuration deployed 2026-10-20 (RPD_IS), with its cushion profile C10-C70 registered in advance (a cushion recommended only on a plateau); paired with PREREG159's draws (the gate: 60 of 60 exact), 20 draws a set (C30 40) | REJECTED, THE CUSHION A PEAK — the registered sleeve 66.4 -> 58.0% a year (-8.5 pp; set B -29.7), NO added margin call (1 -> 1 draw; descriptively, the de-leveraging from 70% sells the lots first; on DEF151 7 -> 15), dd 26.0 -> 28.2, worst 12 months +6.7 -> -5.5%; cushions (pp a year) C10 +1.3, C20 +3.1, C30 +0.7, C40 -1.1, C50 -3.3, C70 -6.4: C20 ADD (69.5%, calls equal, dd +0.2, set B -1.2, shrunk +0.27 lp) but a lone peak (neighbours under half its gain) and the best cushion moved from C30 on DEF151: not recommended, the user's call. The other books trade as without it; most lots are sold to make room at about 0% each |
| 161 | 10-20 | new arms (re-test) | the six 3x books that qualify under the market's bar (PREREG123) and failed their account tests on the configurations of 10-14 .. 16 — the bull index dips in TQQQ / UPRO, the turn of the month in TQQQ, TQQQ / UPRO overnight on the uptrend nights, TQQQ every night — on the deployed configuration RPD_IS, each in its last construction (the night books de-levered first, auction costs); paired with PREREG159's draws (the gate: 60 / 60 exact), 20 draws a set (TOMQ / NBQ 40) | ALL REJECTED — N3Q -3.9 pp a year, N3S -6.6, TOMQ +0.8 (+0.48 log points < +0.5; without 2020 -1.14; margin-call pairs 3 -> 18; drawdown +1.2), NBQ +0.6 (+0.36; 2020 carries it; drawdown +1.5), NAQ -3.4, NBS -0.7 (no added call now — no growth instead); the weak context no better (+21.1% a year -> 8.3 .. 21.6) |
| 162 | 10-20 | new arm | gold on the weak market's days, PREREG155's post hoc lead as a rule: G1 = GLD held while SPY's previous close is under its 200-session average; G2 = also its high-volatility tercile; out of sample on 2000-2006 (gold futures, then GLD; ^GSPC's decisions), the bank on 2000-2026, the account on RPD_IS (1/4, 1/2; quality 0.1, gives way to every book) | FAILED — out of sample G1 +0.009 %/day (95% -0.036 .. +0.038) against the S&P's +0.002: consistent, not confirmed (+0.017% a session against 2007-26's +0.066); G2 +0.030 (+0.008 .. +0.066) on 14 holds; the longest history G1 85 holds / G2 74 holds, not significant (too few); the account G1_4 -0.33, G1_2 -0.84, G2_4 -0.83, G2_2 -1.72 pp a year (REJECT); the lead closed |
| 163 | 10-20 | new arm | large-cap dips in a weak market: the dip rule ($2B+, RSI2 < 10, the live limit order and exits) with the stock UNDER its own 200-session average — W1 in the weak cell, W2 any below-trend cell, W3 every context (the control) — the daily pick by the deepest RSI2; the account on RPD_IS at the dip book's place (1/4, 1/2, 2 open). Amendment 1 (EXPLORATORY, after both stages): W4_K, every signal's order on sessions with at least K such signals (the capitulation breadth), and the daily picks, in a give-way place | FAILED / REJECTED — the picks alone W1 +0.116 %/day (95% -0.41 .. +0.65; 22 months), W2 +0.108, W3 +0.081 (the bar +0.038); every signal's order earns more (W3 +0.270, +0.071 .. +0.507) — on market-wide capitulation days; in the account W1_4 -5.0 pp a year (drawdown 26 -> 33.5), W2_4 -5.7, W3_4 -21.6; W4_K qualifies at K 2 / 3 (+0.355 / +0.474 %/day) but in the account -3.97 / +0.68 / +1.95 / -0.02 pp at K 2 / 3 / 5 / 8 — K 5's gain is 2022's alone (without it -0.75 log points) and a lone peak: not recommended (the weak context +21 -> +29% a year at K 5) |
| 164 | 10-20 | new arm | the pre-earnings run-up: from each 8-K Item 2.02 release, next year's expected date = + 52 weeks; bought at the open five sessions before when the stock trades $1B+ a day and has not reported yet; E1 sold at the close before the expected release (or at the reaction when it comes first), E2 held through it (the reaction session's close); the bank 2008-2026; the account on RPD_IS (dip book's place; Amendment 1: give-way place) | FAILED / REJECTED — E1 $1B+ pooled 1,031 trades +0.076 %/day (95% -0.021 .. +0.171), over SPY +0.15% a trade (t 0.82; from 2021-10 +0.04%); E2 +0.057, over SPY +0.005%; $200M+ no better; the date estimate works (89% of releases within 7 days); the account E1_4 -16.2, E2_4 -11.5 pp a year (crowding the dip book), give-way E1g_4 -6.8, E2g_4 -0.1 (40 draws) |
| R166 | 10-20 | review | every lead and failed attempt re-read with RETURN PER DAY as the metric (the user: "Review all of the leads and failed attempts considering the return per day as the metric to optimize."): ~220 stored account variants (PREREG95-164, 41 run folders) by the account's and each BOOK's return per day against their own references; PREREG152's 25 rule grids by pure return per day; every new book's own return per day in the account | READING — no failed new book earns per day what the deployed books earn in the account (on RPD_IS vol 0.91, ETF 0.56, fall 2.42, crash 2.27, dip 0.44 %/day; the best new one, the capitulation breadth at 5 signals in a give-way place, 0.25; sleeves, 3x night books, gold, pre-earnings at or under 0.15); the account's per-trade rpd also rises by cutting lower-rpd trades (the dip book removed +0.07 %/day for -47 pp a year), so a book is judged by its own rpd with growth as the guard (the user's rule of 2026-10-09); four rejected rule changes raised a book's rpd — the meta-model cut, context activation, the crash book's after-hours exit, the dip limit's depth and life -> PREREG166 (context activation waits for RPD_IS's cards); pure rpd departs from RPD_IS's choices only on the dip limit's depth and life |
| 166 | 10-20 | rules (rpd) | the rule changes the review found raising a book's rpd, on RPD_IS, judged RETURN PER DAY FIRST (the changed book's rpd up, t >= 2; growth not lower; calls; drawdown): RP33 (the dip meta-model's cut before the daily pick), XC (the crash book sells at the 16:30 after-hours tick on its exit day instead of the next open), LD10 (dip limit 1.0 ATR), LL1 (dip order life 1 session); Amendment 1: their union U, XC on 2010-2017 with TQQQ's real after-hours bars | RP33 RPD-ADD (dip 0.44 -> 0.53 %/day, +1.79 pp a year; the standard rule's REJECT: 2024 carries it); XC RPD-ADD and ADD (crash 2.27 -> 4.66 %/day, +1.83 pp, every set; 2010-2017: rpd 1.31 -> 1.52 but the average trade -0.65 pp — NOT CONFIRMED, the overnight after the trigger +0.59% there vs -1.76% in 2018-26); U = both: ADD by the standard rule (+3.62 pp, 66.5 -> 70.1%, calls 1 -> 1, dd +0.6, without 2018 still +0.75 lp, worst 12 months +6.7 -> +10.0%, weak context +21.1 -> +25.6%, shrunk +0.32 lp) -> RECOMMENDED to the user; LD10 / LL1 REJECT (rpd up, growth down, calls 1 -> 2) |
| 165 | 10-20 | new arm | THE CAPITULATION BOOK (the user: the lead "is a good lead even if one year carries it ... it lifts the weak months"): on a session whose breadth B (the share of the 100 most-traded stocks under their own 200-session average with RSI2 < 10, from Polygon's whole-market daily bars) is at least theta, every $2B+ dip under its own trend becomes the dip book's limit order with RPD_IS's exit; theta 5-30%; stage 1 on the pooled history and OUT OF SAMPLE on 2006-2017 (a survivorship-free panel rebuilt from the whole-market bars: Lehman, WaMu, Wachovia, Fannie, Freddie, old GM in it); the account in the crash book's place or giving way, 1/4 x 2 or 1/6 x 3 (Amendments: theta 0.20 and its neighbours) | QUALIFIES, CONFIRMED OUT OF SAMPLE, IN THE ACCOUNT AN ADD THAT FAILS ITS PLATEAU CHECK — stage 1 at 20%: 364 trades in 42 months, +0.683 %/day (95% +0.230 .. +1.272) pooled, 149 / 32, +1.792 (+0.996 .. +2.536) on 2006-2017 against SPY's +0.023 (every theta above its bar out of sample); the account CBg6_20 (give-way, 1/6, three open) +3.38 pp a year (66.5 -> 69.9%; ADD by the standard rule: without 2022 +0.51 log points; calls 1 -> 1; drawdown +0.0; the weak context +21.1 -> +28.0% a year) but 15% keeps a third of it (+1.02 pp) -> its registered plateau check fails; the crash book's place loses on A0; its trades in the account +0.3 %/day (sold to make room) |
| 167 | 10-20 | combination | PREREG166's RP33 + XC with PREREG165's CBg6_20 together on RPD_IS, paired with RPD152 and with U | ADD — +7.19 pp a year (66.5 -> 73.7%; +4.23 log points; A0 +2.7, A +13.0, B +4.3 pp; without 2022 +2.44), calls 1 -> 1, drawdown 26.0 -> 26.5, the worst 12 months +6.7 -> +9.5%, every market context higher (below SPY's trend in high volatility +21.1 -> +32.7% a year), 2020 / 2021 lower (-18 / -7 pp), the account's return per day 0.745 -> 0.758; the capitulation book on top of U +3.57 pp (ADD); shrunk +0.63 log points (30 trials) -> RECOMMENDED to the user as a package, not deployed |
| D167 | 10-20 | deploy | PREREG167's ALL3 into production for the 2026-10-21 session (the user: "Deploy all three."): `enable_dip_long_filter` on (RP33 before the daily top pick), `crash_long_pm_exit` (the crash book sells at the first after-hours tick at/after 16:30 ET of its triggering session; `tracker.monitor_crash_long_pm_exit`), THE CAPITULATION BOOK (`src/signals/cap_long.py`: the breadth on Polygon's unadjusted whole-market bars with the effective splits, every $2B+ dip under its trend a ½-ATR limit order at 1/6, at most 3 open, its capital giving way first — the shorts', the crash book's and the dip fills' room, de-leveraging, margin calls; `docs/CAP_LONG.md`); the simulator's defaults follow (`Mimic.dip_filter` / `crash_pm_exit` / `cap`, `mimic.PRE167` for the configuration before) | PARITY: the default `Mimic()` builds PREREG167's dip (1,130), crash (15) and capitulation (887) pieces identically and reproduces its 60 accounts exactly (growth, the daily NAV, the margin calls); production's breadth equals PREREG165's on 59 of 60 sessions of 2026, every gate equal (2026-09-28: WBD, delisted since, is missing from today's type table); the suite green. The capitulation book's bank card is not built yet |
| 168 | 10-20 | new arms (new data) | event shorts from the SEC filings and the split records (the user: "can you find other trades and models? Maybe with more data sources?"): E1 an equity offering (424B1/4/5, 8-K Item 3.02; $5M-$200M a day), E2 a reverse split, E3 a listing-deficiency notice (8-K Item 3.01) — short at the next open, 5 / 10 / 5 sessions, the 3x cover; Polygon's UNADJUSTED whole-market bars (the $5 floor as traded); IBKR's file at 09:30 of the entry (A0 / A), its API fee (B) | FAILED — E1 the short LOSES (5,982 trades, -0.59% a trade, -0.097 %/day, 95% -0.150 .. -0.043: the drop is in the price by the next open); E3 loses (279, -1.17%); E2 +2.17% a trade over 13.5 days (352, +0.159 %/day, 95% -0.018 .. +0.362; A0 / A / B +1.6 / +2.8 / +2.1%; IBKR's mean fee 4 / 16 / 66% a year, 29% of A0 + A unlendable) failed only significance -> THE RE-JUDGING on 2005-2026 (Amendment 1: the 2005-2017 unadjusted bars fetched, every pre-2018 trade traded at A0 + A's mean fee 11.4%): before 2018 -0.29% a trade (205, -0.022 %/day), 2005-2026 +0.094 %/day (95% -0.043 .. +0.240, 557 trades): FAILS; its account test (Amendment 2, registered in advance) not run |
| 169 | 10-20 | new arms (new data) | calendar and market-state longs from new sources (the user: "can you find other trades and models? Maybe with more data sources?"): L1 the pre-FOMC drift (the Fed's meeting calendar 1994-2027, `studies/prereg169/data/fomc_dates.json`: bought at the close before each scheduled decision day, sold at its close; QQQ x3 = TQQQ), L2 the end of a VIX term-structure inversion, L3 a high DIX (SqueezeMetrics' dark-pool index) | L1 QUALIFIES on 1999-2026 (219 trades, +0.84 %/day, 95% +0.17 .. +1.50 against SPY's day) but on 2018-2026 only +0.52% a trade (n.s.; 2023 -1.00, 2026 -1.83) and in the account on ALL3 (Amendment 1: the crash book's place, 1/3, 75% margin) -4.32 +- 0.12 pp a year (73.7 -> 69.4%), calls 1 -> 2: REJECTED (futility); L2 INSUFFICIENT (66 trades 2006-2026, +0.375 %/day, significant); L3 FAILS (+0.091 %/day, 95% -0.08 .. +0.25 against +0.038); Amendment 2 (the user's request): L2 in the account on ALL3 — in the crash book's place -0.24 +- 0.12 pp a year, calls 1 -> 5 of 20: REJECTED; in the give-way place (quality 0.05, de-levered first) +0.75 +- 0.08 pp on 40 draws (+0.43 log points, under +0.5), without 2024 -0.39, 2018-2020 -4 to -11 points a year, the worst 12 months +8.9 -> -2.8%: REJECTED |
| 170 | 10-20 | dip meta-model (new data) | the dip book's meta-model given NEW INFORMATION (research lead 1): FUND (the companies' own SEC XBRL numbers point in time by filing date — revenue growth, net margin, leverage, dilution, earnings yield on the AS-TRADED close, cash, filing age), SECTOR (its SPDR sector fund's trend and 5-day move, the stock against it), ATTN (Wikipedia views against their 60-day median); N1-N4 ten-seed walk-forward ensembles (PREREG143's construction) against the same-construction control R1; the $2B+ daily top pick 2011-2026 | FAILED — no candidate's Bonferroni interval clears 0 (N1 +FUND +0.017% a pick, 2011-20 +0.46 / 2021-26 -0.18; N2 +SECTOR +0.039, +0.22 / -0.04; N3 +ATTN -0.141, both eras negative; N4 all +0.077, +0.42 / -0.08); the fundamentals helped every year 2014-2020 but 2016 and hurt every year 2021-2026; the deployed recipe stays ahead of every re-fit (+0.103 over R1); the control reproduces PREREG143's R1 exactly; no look-ahead in 200 rewritten rows |
| 171 | 10-20 | new arm (new data) | IPO lock-up expiry shorts on Polygon's IPO calendar — a REPLICATION of PREREG111 (not searched for before the registration) with other windows: short X-4 .. X (the 180th day), S1 X-2 .. X+2, S2 X .. X+4, S3 X-9 .. X; IBKR's file at the entry; as-traded bars | FAILED, CONFIRMS 111 — 490 trades, -0.19% a trade, -0.029 %/day (95% -0.21 .. +0.15); the pre-expiry drift is there gross (+0.65% over 5 sessions) but costs and borrow take 0.84; A0 +1.01, A -0.32, B -1.76% a trade, negative since 2023; after the expiry the stocks rise (S2's short -2.00%) but as a long only +0.36% a trade (n.s., 2022 and 2024 carry it) |
| 172 | 10-21 | tuning method | the user: "For models that have a very good return per day, optimize them for return per day and growth": the books at or above the vol arm's 0.75 %/day (vol, ETF, fall, crash) — each book's holding parameters searched TOGETHER (vol give-back x time limit x cover 150, ETF 80, fall volatility exit x time limit x cover 120, crash 16; each book alone, seed 0) by RPD (argmax R, 2/3 of live's trades) and by RG (argmax R x G, G its growth contribution at its slice); then the sizing one at a time by training growth (S1-S7) on each fold's rules; leave-one-year-out folds, seeds 0-4 (Q1: RG vs RPD); RG's in-sample choice by the pooled add rule (Q2) | Q1 NOT SHOWN, Q2 REJECT — in sample RPD takes the vol time limit 12, RG 20 (+ the fall exit 0.6), ETF and crash live; held out RPD 74.2%, RG 72.3% a year against live 73.1% (RG - RPD -1.6 +- 2.7 pp, 4 of 9 years), both with margin calls in 4-5 draws of 5 from the growth-chosen sizing (fall 1/6, crash 1/2.5, de-leveraging 80 / 60%; live none); rules only at the live sizing (exploratory) RPD +0.3 +- 0.6, RG -2.3 +- 2.0 pp (2018, 2026 lost: the ETF / fall choices); RG_IS +6.6 pp in sample (73.7 -> 80.3%) but calls 1 -> 9 pairs, RG_IS_R (its rules at the live sizing) +3.5 pp but calls 1 -> 2: REJECT. The vol arm: the 10-session limit covers ~10 squeezed shorts at their worst (SBET -104 -> -3%); at 12 / 20 sessions its card QUALIFIES (+0.99 / +0.92 %/day, 95% +0.25 / +0.34 ..; live +0.83, -0.09 ..), its held-out growth contribution up in 7 of 9 years (20) — the lead (section 4, item 12) |
| 173 | 10-21 | rules (time limits) | the user: "Optimize each short arm time limit": the vol, ETF and fall arms' OWN time limits {7, 10, 12, 15, 20, 25, 30} searched TOGETHER on the full account (343 combinations, ALL3, seeds 0-4, the short books composed per limit — checked against direct builds), by ACCT (the account's training growth under the add rule's guards: no more margin-call pairs, drawdown +1) and BOOK (each arm's R x G alone), leave-one-year-out (Q1), the in-sample choices by the pooled add rule (Q2); Amendment 1 (exploratory, after Q1): the vol arm alone at 15 / 20 | Q1 NOT SHOWN, Q2 REJECT, AMENDMENT 1 REJECT (one margin-call pair) — in sample the gain is the vol arm's (15 / 20 sessions +3.7 pp a year, others at 10, no call in 5 draws); the ETF's best is 10 (every other value -0.5 .. -4.1 pp), the fall's flat to 12 with calls beyond; held out ACCT -4.3 +- 1.3 pp (its folds also moved the ETF / fall limits: 2018 +16 against +40%, 2024 +53 against +76%), BOOK +1.0 +- 0.9; the vol arm's limit tuned alone (exploratory) +2.7 +- 1.2 pp held out (folds 15 / 20), the ETF's alone -4.7, the fall's -2.3. On 20 draws: ACCT_IS (15, 10, 12) +3.4 pp, BOOK_IS (20, 10, 7) +3.3, V15 (15, 10, 10) +3.0 pp (73.7 -> 76.7%, every set, drawdown 26.5 -> 25.8, the vol book's rpd kept 0.91 -> 0.89), V20 +3.2 (its rpd 0.91 -> 0.80) — every one meets calls in 2 (set, draw) pairs against live's 1 (the same event: AGBA held past 10 sessions + two fall squeezes on 2024-05-09, set A draws 11 and 17; live's January 2021 call gone): REJECT — V15 DEPLOYED by the user's call (2026-10-21: "Move the vol arm to 15 sessions"; the vol picks from the 15:00 ET tick; the default `Mimic()` reproduces V15's draws exactly; `bank.LIVE_RUNS` = V15) |
| 174 | 10-21 | simulator (margin calls) | the user: "Research online what IBKR does exactly during margin calls. I want the backtest to do exactly the same, and, if we can't be exaact, approximate closely and be only slightly on the safe side": IBKR's rules from its own margin page (real-time excess liquidity, liquidation "to bring the Excess Liquidity balance up to at least zero", "most recent transactions in a minimum of 100 share increments", market orders, Soft Edge Margin 10% until 15 minutes before the close, the end-of-day Reg T SMA) modelled in the engine (`Rules(liquidation="ibkr")`) and run on V15's 20 draws a set: M_IBKR (registered: every lot at the bar's extreme, strict) and its brackets; Amendments 2-3 (after the registered events): M_PATH — the check along the bar, each lot when the deficit appears, filled one LULD band (10%) against the account, never past the bar | DEFAULT FROM 19:40 ET (M_PATH, the user's instruction; decided before its numbers) — 76.6 -> 75.7% a year (-0.96 +- 0.20 pp; M_IBKR -1.12, at the tick marks +0.02, everything closed at the bars -3.88), margin events in 13 draws of 20 (15 pairs) instead of 2, the drawdown 25.8% / 34.2% and the worst 12 months +9.3% unchanged; the events are one or two 100-share lots (11 pairs: ICCT's after-hours de-SPAC squeeze of 2023-08-30, $7.30 -> $42.00); no ruin, no Reg T-floor event; SEM 10% +0.04 pp (2 pairs fewer), the largest requirement first -0.09; V15 against ALL3 under it +2.87 pp, pairs 16 -> 15: ADD (PREREG173's REJECT gone). Cards identical (they keep the old rule: `reliability.card_rules`), `bank.LIVE_RUNS` = prereg174 PATH; production's account unchanged (offered); report https://claude.ai/artifact/WcrVSfjrxnyuJk1ZhmXSry |
| 175 | 10-21 | simulator + production (margin calls) | the user: "Yes, production and backtest needs to liquidate like IBKR does": production's account liquidates with the engine's M_PATH (`live_account.plan_liquidation`, `margin_watch`: each check reads the 30-minute bars completed since the last one and the live price; lots split off as their own trades, `tracker._split_margin_lot`; the broker's lot leg `MARGIN_LOT`); the simulator's check sees what production's sees — an off-grid tick's high / low over EVERY bar since the previous mark (`Mimic.margin_bars` "all"; the hourly slots' first half-hours, 09:00-09:30, 19:30-20:00 and the 04:00 bar had fallen between marks), every long book marked at the ticks (`ext_longs`: the crash and capitulation books too), the Reg T check at the marks; V15's books rebuilt, 20 draws a set | DEFAULT FROM 22:05 ET (decided before its numbers: production's check) — 75.7 -> 75.8% a year (+0.18 +- 0.02 pp), margin events 18 in 15 pairs -> 17 in 14 (12 draws of 20), the drawdown 25.8 / 34.2% unchanged, the weakest context +36.0 -> +37.1%; the newly checked bars add NO event (the drift is the long books' extra marks moving a few give-ways and de-leveraging sales); gates exact (the builders reproduce PREREG168 / 173's books, PREREG174's 60 accounts reproduced with the Reg T change), production's plan = the engine's lots on its scenarios (`tests/test_live_liquidation.py`); `bank.LIVE_RUNS` = prereg175 NEWMARKS; DEPLOYED to production 22:30 ET; report https://claude.ai/artifact/Pmg1g1w5SXv3ZKEBSnb8Qk |
| 176 | 10-21 | tuning method (THE TRAINING PROTOCOL) | the user: "retune using the training protocol of optimizing return per day and a mix of return per day and growth for models that have a very good return per day": each book's holding parameters together, the book alone — vol / ETF / fall / crash by R x G (PREREG172's 366 units, reused: identical trades under today's construction), dip / capitulation by R (exit average {2..5} x limit {3..10}) —, then the sizing (8 parameters) by a JOINT local search on the full account with the margin events and the drawdown guarded, leave-one-year-out, then the pooled add rule | NOT SHOWN / REJECT — held out 71.0 against live's 76.0% a year (-4.9 pp, 2 of 9 years, margin-event cells 7 vs 3); in sample (vol 20 sessions, fall exit 0.6, dip and capitulation (2, 7); the sizing unmoved — the rules alone break the guards in 8 of 10 folds) +2.13 +- 0.40 pp on 40 draws but carried by 2025 (A0 -11, B +20 log points; without 2025 -2.75), event pairs 40 -> 44, the drawdown 25.4 -> 26.7, the worst 12 months +9.0 -> +3.8%; exploratory: the dip book's 2-session exit carries it (+2.25 pp in sample, loses in 6 of 9 years), the vol arm's 20 sessions +0.02, the fall exit 0.6 +0.95 (noise), the capitulation exit -0.98. Nothing deployed; report https://claude.ai/artifact/Pmg1g1w5SXv3ZKEBSnb8Qk |
| 177 | 10-22 | new arms (ML by market context, long) | the user: "Train ML models on different market contexts so that they are specialized ... improve overall growth by optimizing on return per day of those models": on PREREG165's survivorship-free daily panel (147,599 large-cap candidates, $1B+ a day, 2007-09 .. 2026-09; 57 point-in-time features), each candidate's trade = the dip book's ½-ATR limit order and exit (the label = its log return per day); three LightGBM families walk-forward (yearly refits from 2010, 5-seed ensembles): S6 one model per context cell (the user's design), S2 one per trend group, G one global (the control); the book's top-k / score floor chosen per trend group on 2010-2017 by return per day; the pooled history out of sample; the account in the give-way place (ALL / WEAK cells x 1/8, 1/6, 1/4); Amendment 1 (exploratory): S2 in the account | REJECTED — the specialists rank worse than the global model inside every cell (bear-cell IC +0.065 vs +0.088); F* = G (2010-2017 R 0.234 %/day vs S6 0.217, S2 0.201) FAILS significance out of sample (0.181, 95% -0.017 .. +0.384; S6 0.080 FAILS, S2 0.263 QUALIFIES); in the bear cell every book's picks earn what an average candidate's do (0.11-0.19 vs 0.175); in the account every variant -7.1 .. -11.4 log points a year (-12 .. -19 pp; set A -23 .. -35, set B +3 .. +15), margin-event pairs 14 -> 15-17, the weak times WORSE (the bear cell +37 -> +25-30% a year, the worst 12 months +9 -> -9 .. -33%, 2022 +65 -> -6 .. +3%); the book loses per day in the account (-0.03 .. -0.12 %/day, about a fifth of its trades sold to make room); S2 (exploratory): its card QUALIFIES (0.233 %/day) but the account ALL -2.1 .. -5.0, WEAK8 -0.5, WEAK6 +0.88 lp (+1.6 pp, 2025 carries it, one more margin-event pair) — none passes; report https://claude.ai/artifact/CbnsiPh7diL1SiKZhhKKam |
| 178 | 10-22 | new arms (ML by market context, short) | PREREG177's panel, features and three families on the MIRRORED trade — a large-cap SHORT sold into a rally (a limit at the close + ½ ATR, 3 sessions), covered at the open after the first close under the 3-session average or 5 sessions, the borrow at a flat 0.5% a year; selection on 2010-2017 (k in 1, 2, 3, 5, the day's top decile; the floor widened to 1.2); the account on the free margin room (IBKR's margin 50%), 1/8, 1/12, 1/16 | REJECTED — every large-cap short loses on average (-0.034 %/day; only 2008 and 2022 clearly positive; the bear cell +0.037); the models' positive IC (+0.06) ranks trades that lose (top decile -0.04 .. -0.05); 2010-2017: S6 / S2 no eligible rule, G's bear-cell top decile 0.008 %/day -> F* = G, out of sample -0.026 %/day (95% -0.42 .. +0.34) FAILS; the account (futility at 5 draws) -0.08 / -1.43 / -2.15 log points a year, margin-event pairs 3 -> 6-7; it lifts the GRINDING bear (2022 +62 -> +84-89%, 2018 +40 -> +44-47%) and loses the V-shaped crashes and rebounds and the bull years' pullbacks (2019 +23 -> +14%, 2025 +120 -> +106-108%, 2026 +151 -> +128-132%): the bear cell +38 -> +21-29% a year; report https://claude.ai/artifact/CbnsiPh7diL1SiKZhhKKam |
| 179 | 10-22 | new arm (the bear context split, short) | the user's "Start with 1.": a point-in-time GRIND state (SPY's close under its 50- and its 200-session averages at the close before the order session; BEAR-OTHER = under the 200 only) and PREREG178's large-cap short book activated only there — every $1B+ stock under its own 200-session average, the most overbought first; judged on 2007-2017 first, then 2018-2026 and in the account (the free margin room; 1/8 x 3, 1/6 x 3, 1/12 x 4) | REJECTED — 805 GRIND sessions (17%) in 27 episodes; every candidate's short in GRIND +0.187 %/day on 2007-2017 (95% +0.005 .. +0.353) but -0.021 without 2008, and -0.051 on 2018-2026 (2022 +0.01): no state separates in both eras (the death cross, a falling 200-session average, no volatility spike, 40 of 60 sessions under the trend — descriptive); the book's top 3 +0.102 on 2007-2026, short of significance; the account futile at every size (S8 -0.96 +- 0.17 log points a year, -1.7 pp; 2022 +62 -> +67% but 2019 23 -> 18%, 2023 27 -> 20%, below / high 36 -> 34%); THE SURPRISE: the beaten-down large caps BOUGHT in the grind earn +0.31 / +0.23 %/day in both eras (-> 180); the live configuration by state +44% a year in GRIND months, +6% in BEAR-OTHER, +86% in BULL; report https://claude.ai/artifact/FBTqPb2CUyzuPSg3hsccNU |
| 180 | 10-22 | new arm (the grind long book, EXPLORATORY) | PREREG179's long-side finding: on a GRIND session every $1B+ stock under its own 200-session average as the dip book's ½-ATR limit order, the ranking chosen on 2007-2017 (the deepest RSI2 first, over the largest and the least oversold first), tested on 2018-2026 and in the account in two capital places (OWN: never preempted, the free room only; GIVE: PREREG177's give-way place) at 1/8 x 3, 1/6 x 3, 1/12 x 4 | REJECTED — the chosen top 3 on 2018-2026 +0.029 %/day (falling knives: 2018 -0.37, 2020 -0.57; post hoc every key's top 3 +0.03 .. +0.08 against the unranked +0.227 — the edge sits on the days many large caps rebound together, the capitulation book's); the orders alone +0.214 %/day (short of significance by 0.002); the account futile in every variant (OWN8 -1.61 +- 0.10 log points, -2.8 pp, its trades -0.38% each though never preempted; GIVE8 -1.36, 28% preempted) and the weak times worse (2022 +62 -> +47%, below / high 36 -> 32%, the worst 12 months +8.7 -> +3.4%); report https://claude.ai/artifact/FBTqPb2CUyzuPSg3hsccNU |
| 181 | 10-22 | dip meta-model (market context) | the user's "Do task 2 also": the market context the 17 inputs lack — GRIND (SPY against its 50-session average, the 50 against the 200), REGIME (SPY's 20-session volatility's percentile in its own history), BREADTH (of the 100 most-traded stocks on PREREG165's panel, the share oversold under their trend and the share under it); M1-M4 ten-seed walk-forward ensembles against the same-construction control R1x, each year's RP33 cut from its own out-of-fold scores; judged on the WHOLE daily decision (the cut and the $2B+ top pick, 1,182 sessions 2011-2026) | FAILED at stage 1 — the gain a day M1 -0.032 (95% -0.100 .. +0.034; 2011-2020 +0.075, 2021-2026 -0.118), M2 +0.007 (-0.045 .. +0.058), M3 -0.042, M4 (all five) -0.082 (-0.162 .. -0.004); PREREG170's pick-only measure the same reading; the deployed recipe stays ahead (+0.119 a day over R1x); the control equals PREREG143's R1 on all 8,360 rows; no look-ahead in 200 rows and 20 sessions; report https://claude.ai/artifact/FBTqPb2CUyzuPSg3hsccNU |
| 182 | 10-22 | new arm (3x, the end of a grind) | the user: "at the end of an episode, SPY usually goes up in the next 20 sessions ... Can you backtest a model where we buy the 3x SPY after the end of an episode?": UPRO bought at the open after SPY's FIRST close out of PREREG179's GRIND state (point in time: every exit, the false ones included), held 20 sessions, one at a time; the untouched 1985-2007 first (PREREG110 / 120's stitched S&P 500, UPRO synthetic before 2009-06), then 2007-2026, the bank's card and the account in the crash book's place (quality 0.2, 75% margin) at 1/3, 1/4, 1/6 | REJECTED — 150 exits, 57% back in the grind within 10 sessions; 1985-2007 SPY +1.70% a trade against +0.81% for any 20-session hold (excess +0.89, 95% -0.23 .. +1.98: not significant), 2007-2026 +0.10% against +0.76% (excess -0.66); UPRO's worst trades are the false exits in falling markets (2020-03-03 -52%, 2009-01 -39%, 2018-12 -31%, 2022-09 -31%); the card FAILS (89 trades 1985-2026, t 1.29); the account futile at every size (1/3 -2.69 +- 0.22 log points, -4.7 pp, the drawdown 23.8 -> 29.7%): it lifts 2019 (+23 -> +34%), 2023 (+27 -> +49%) and the rebound months (+7 -> +43% a year) but costs 2018 (-23 points), 2020 (-29), 2022 (-24), 2025 (-17), below / high 36 -> 22%; the "23 of 27" of PREREG179's calendar was hindsight (episodes merged and measured from their last grind session); report https://claude.ai/artifact/FBTqPb2CUyzuPSg3hsccNU |
| 183 | 10-22 | evaluation (Sharpe inference) | the user, after the survey of public Claude trading skills (none worth installing): "Copy the formulas to improve our own skills": López de Prado, Lipton & Zoonekynd's Sharpe inference (SSRN 5520741) ported into `stats.py` (the variance with skew, kurtosis and AR(1) autocorrelation, the PSR under the null, MinTRL, power, pFDR / oFDR, FDR control, the best-of-K test; Newey-West and cluster-robust OLS), pinned to the reference code's values, wired as reports (the cards' `dsr_months`, the bank's `track_record`, `add_rule.market_noise`), and read once | READING (nothing decided) — the live record needs 10 months for an even chance to show a profit at the backtest's Sharpe (21 for 80% power; 41 / 85 at half of it) and 67 months to show it earns half; on its months the vol arm's deflated Sharpe is 0.10 at its 753 trials (0.56 per trade), the capitulation book's 0.69 (0.73), the others 0.87-1.00; the dip book's trade +2.22% is +2.00% alpha (β 1.26), the capitulation book's +1.85% alpha, the crash book none (β 3.95: 3x timing by design), the account's monthly β -0.06 (alpha +5.1% a month, t 5.4); the dip meta-model's purged folds move the RP33 cut 0.252 -> 0.246 (5 of 5,439 signals); against the market's noise the five changes deployed 10-19..21 have t 1.6-1.9 (RPD152 -0.1), each 12-30% likely noise at the 22% deploy-rate prior, and a no-effect change shows the +0.5 gain 32-47% of the time; report https://claude.ai/artifact/TRcTayDM5fpF6HpobqQ1rT |
| 184 | 10-22 | evaluation (the deploy rule out of time) | the user's "Do 1 and 5", item 1: the pooled add rule backtested on its OWN record — 16,056 saved draws, 303 verdict entries matched exactly to the (variant, baseline) draws they compared (214 pairs; the rule rebuilt from them reproduces 305 of 305 recorded decisions); the 169 deployment pairs (baseline = the live configuration of the day, 34 studies) re-decided at each year-end from 2019 on the calendar years before it ((a)-(d) on those years only) and scored on the next year, each study equal; stricter bars on the same decisions (the gain at +1 / +2 / +3.3, the market-noise t on the in-sample months at 1 / 1.65 / 2); study and year bootstraps | READING — the rule's adds gained +0.67 pp a year the next year against its refusals' +1.66 (adds minus refusals -0.98, study bootstrap -9.7 .. +5.1; in log points +0.33 against +0.46; without 2026 +0.07 against +1.39 pp); an add kept 0.11 of its in-sample gain (+3.09 -> +0.33 log points; 0 without 2026), the slope over all decisions +0.40 (+0.13 .. +0.65: large effects persist, the +0.5 margin is noise); the market-noise floor t >= 2 passed 31 decisions (all rule / sizing) against the rule's 161, 75% gained, +6.19 pp against +0.67 (p 0.06; +2.1 log points, p 0.05), the same total; the gain bars alone no better; the one-pair margin-call count refused changes that gained +12.5 pp out of time with no more calls, the drawdown test changes that went 1.3 points deeper; by year the adds lost in 2020 (-13.3, decided on 2018-19 alone) and 2022; DMTE gained in all 7 years but was added at 2; leave-one-year-out (exploratory) adds +3.8 against +0.3 pp; upper bounds (every design saw all years); a market-noise floor and the margin-call test as a rate put to the user, nothing changed; report https://claude.ai/artifact/FS1aAMgCjdStBbjasrJgBv |
| 185 | 10-22 | evaluation (the forward check, research only) | item 5: every live trade from 2026-09-25 (a ledger copy; 16 trades — 14 dips, the ETF short AAOX, the fall short AEHL) rebuilt the simulator's way on Polygon's as-traded bars (a short at its decision bar's close, its own exit on the simulator's marks; a dip order at the attainable 09:30:30 price or its limit at the touching bar's end, out at the opening print after its rule; Polygon's quotes + IBKR's fees), paired trade by trade: entry, exit at the same tick, exit timing, costs, the net trade, IBKR's fills; the live account in its tracking band (every n-session window of NEWMARKS' 120 draws, and a held-out band at 61% a year) | READING + TOOL — all 14 closed trades left on the simulator's own tick (5 by their rule, 9 at the fresh start's 09:30 tick); the dip touch fills equal to the cent since the limit orders; the 12 closed dips -0.12% a trade live minus the simulator (one episode: no standard error), their 09:30 exits -0.17% under the opening print (AUDIT151 corrected the open ENTRIES, never the exits: ~-2 pp a year at 1/2 if it persists), costs 0.08% under the simulator's; META 10-22's open fill at IBKR 663.80 against the attainable 665.00 (the print 676.35); the shorts +3.6 / +3.2% (minutes of volatile names; AEHL's quoted spreads 104 / 339 bp where IBKR's cover filled 1.5% under the bar); the live account -4.34% after 3 sessions, the 4th percentile of its band (the two META longs); the full set-D simulation's route recorded (PREREG105's AB2 extended + a runner, about half a day); `studies/prereg185/h185.py all <copy>` reruns it; report https://claude.ai/artifact/FS1aAMgCjdStBbjasrJgBv |
| 186 | 10-23 | combination (the second account) | the user: "I want to add a second account of $50,000 that will have less noisy, more significant trades. Add the market noise floor for the models in that second account." — the live configuration's 6 models and the 10 rule changes deployed since 10-09 each judged by its gain against the MARKET's noise (`add_rule.market_noise` of the account with and without it, 40 draws a set, NEWMARKS), the lowest t under 2 removed and everything re-measured (backward elimination); kept as risk protection when its removal deepens the drawdown > 1 point or adds margin-event draws beyond chance (a one-sided exact binomial, the rate test of the user's open question) | THE SECOND ACCOUNT = THE DIP BOOK + THE FALL ARM — step 0: only the dip book (t 4.25), the fall arm (2.93) and the crash book (2.39) pass; every rule change under 2; the vol arm +8.8 pp at t 1.79 (it brings 39 of the 40 margin-event draws); the elimination removed the ETF turn, V6, the ETF arm, RP33, the capitulation book, the earnings demotion, the crash book's after-hours exit, the crash book, the vol arm's volatility exit off and V15, then the vol arm (t 1.19 without its rules; 2.0-2.5 with them); kept as risk protection the fall turn, the dip top pick and the $2B pool. The second account: 51.0% a year (draws 46-55 against live's 64-87), drawdown 19.9% (25.4), NO margin event (28 of 40 draws), the worst 12 months -5.5% (+9.0), +10.5% a year below SPY's trend (+35.9); market t -2.59 against live. The add rule's count would refuse re-adding the dip book (+50 pp) on 37 -> 40 event draws. Exploratory: the vol arm with its own rules bundled passes (60.4% a year, drawdown 24.3, events 16 of 40). The production form (simulated, a ledger account, a second IBKR paper account) is the user's call; report https://claude.ai/artifact/RMHqUBLK7YFWxyYcNziZz4 |
| 187 | 10-23 | dip / capitulation exits (the user: "Run the exit study of dips exit at the open") | the price production can sell a dip or capitulation long at when its rule exits at the open — AUDIT151's arrival rule for a sale at 09:30:30 (E30) and at 09:30:20, the ledger's booking instant (E20; Polygon's minute bars and trades) — against the opening print the simulator books, on all 665 such exits 2018-2026 (390 sessions, the standard error clustered by session); the account rerun at E30 (40 draws a set); the 12 live exits | THE PRINT STAYS — E30 -2.1 +- 2.2 bp (t -0.95; dip -1.2, capitulation -3.2; 54% under the print), E20 -2.4 +- 1.9; the first 30 seconds reverse the opening move (the biggest gap-ups -12 bp, gap-downs +13 bp); the account -0.25 +- 0.06 pp a year (74.9 -> 74.6%, market t -0.23); live's -0.17% was 2026-10-20's falling morning (the attainable prices track the 12 live exits within 11-13 bp, the print 36-47 bp); a market-on-open exit would gain about +0.25 pp a year (not worth building). Pitfall: moving an exit's instant to 09:30:30 puts it after the same session's open fills in the engine (-7.5 pp, an artifact; Amendment 1) |
| 188 | 10-23 | evaluation (set D, the live period, in the simulator) | the user's "Do both for number 4", its second half: PREREG105's continuous stores extended to 2026-10-22 (10,989 Polygon requests, a 23-minute build), set D's gaps closed (the ETF arm and the fees on our own archive of IBKR's file, D's dip signals scored by production's deployed meta-model, the breadth by production's own computation, D's quotes and open fills, the evaluator's data end moved), the live configuration and PREREG186's second account run on D (40 draws), parity first: set B on the rebuilt stores and D's decisions against production's journals | SET D RUNS AND REPRODUCES PRODUCTION — the vol arm's pick per bar 167 of 169 from 10-06, production's 8 ETF picks all present, the dip daily top pick 4 of 4 days, set B's long books identical and short books 91-100%; from the fresh start the simulator takes live's trades (META at 694.22, live's fill; META at 665.00 against IBKR's 663.80) and reads -4.47% without PMI against live's -4.34% — the one divergence the PMI short IBKR REFUSED live, which the simulator's random refusal draw takes in 39 of 40 draws (+35%). Over the four weeks from 09-25: the live configuration -10.7% (the October chip dips), the second account -11.6% (noise for growth). Lead: a refusal model by name. `studies/prereg188/h188.py` (prep / books / legs / decisions / work / second / parityB) |
| 189 | 10-23 | combination (the second account) | the user on PREREG186's second account: "What about crash + fall + dip? These are the three that are over t=2." — T3 = the live configuration's dip book, fall arm and crash book with every rule they run live (the vol arm, the ETF arm, the capitulation book and their own rules out): each model's market-noise t in T3, each rule change's t (a reading), PREREG186's K2 with the crash book (with / without its after-hours exit); PREREG186's harness and cached books, 40 draws a set | THE THREE PASS TOGETHER — the dip book t 4.36, the fall arm 3.30, the crash book 2.22 (beside K2's reverted dip rules it fails: 1.85 with its after-hours exit, 1.24 without); T3 59.8% a year (52.4 .. 64.7) against K2's 51.0% and live's 74.8%, the drawdown 21.3% (19.9 / 25.4), no margin event in 40 draws (live 28), the worst 12 months +6.7% (-5.5 / +9.0), +20.2% a year below SPY's trend (+10.5 / +35.9); T3 - K2 +8.8 pp (market t 1.77; the add rule REJECT on the drawdown, 1.45 points deeper). In T3 every rule change reads under 2: the fall turn (17 event pairs without it), the daily top pick and the $2B pool protect; RP33's cut DEEPENS the drawdown (17.5% without it; t 0.34, one year carries it). NOTE (exploratory): T3 without RP33's cut 58.5% a year, the drawdown 17.5%, ADD against K2 (market t 2.67). DEPLOYED as THE SECOND ACCOUNT from the 2026-10-26 open (the user: "Yes please" to the merge, and "There's no second IBKR account creation. The second account will be calculated from the same account."): T3 on its own ledger and $50,000, CALCULATED from the main account — its decisions read, its orders a dry run, IBKR's what-if answers the main account's (`src/accounts/`, `docs/SECOND_ACCOUNT.md`). `studies/prereg189/h189.py` |
| FWD 10-23 | 10-23 | production (the forward check) | the user: "Do both for number 4" — PREREG185's forward check moved into production: `src/backtest/forward_check.py`, the EOD chain's `forward_check` step after the live calibration (`enable_eod_forward_check`, `forward_check_dir`), every live-book trade from the fresh start rebuilt by the simulator's rules on the same prices (settings-driven, its own quote and opening-trade caches), the live account in the backtest's tracking band; a line and notes in the email's scorer banner (the scheduler restarted 2026-10-23 00:23 ET for the hook); tests `tests/test_forward_check.py` | DEPLOYED (a reading, never a decision) — first record on a ledger copy: the two META dips, the account -4.3% after three sessions at the 4th percentile of its band |
| RATE 10-23 | 10-23 | evaluation (the add rule) | the user: "Yes update the add rule to judge margin calls as a rate. We shouldn't judge the performance based on number of margin calls anymore, just the performance in rates (return per day and growth)." | ADOPTED — `add_rule.pooled_verdict` judges the margin events AS A RATE by default (`margin_events`: on the (set, draw) pairs where exactly one side meets a margin event, the one-sided exact binomial — McNemar's exact test — p < 0.05 refuses: five more pairs and none fewer refuse, four do not; risk protection by significantly fewer); `margin="count"` reproduces every decision before (one more pair refused: V15's 2 against 1, PREREG186's dip book 37 -> 40); the return-per-day rule and the training protocol's sizing guard read the same test; tests `tests/test_add_rule.py` |
| AUDIT 10-20 | 10-20 | process + production | "Are there still common mistakes ... following all the changes?": live trading clean (the dip book's 0/12 is real: nine fresh-start closes of five chip names bought under the old rules, in a selloff); the live score counted forced exits and correlated trades one by one against a reference two deploys old with no capitulation card; the research slips that recur are the ones only memory guarded (stamps, a repeated study, a registered output skipped, labels, the low-priority call) | FINDINGS (`memory/audit-common-mistakes-2026-10-20.md`) |
| FIX 10-21 | 10-21 | process + production | the user: "Fix all". The live score: own exits only (forced apart), episodes, configuration epochs from the fresh start, a fingerprinted reference (stale -> rebuild note); the cards rebuilt for ALL3 with the capitulation book's (above) and its 25 trials in the ledger; the registration checker gates stamps, a `Prior studies:` line (`--prior` search) and `[R1]` promised outputs; research at IDLE from the interpreter's start (`scripts/research_idle.py`); `bank.load_runs` reads PREREG167's "capb" as "cap"; IBKR quotes skipped an hour after a subscription refusal; the CAD warning's text | DEPLOYED (the broker part live since the 01:45 relaunch; the live score from the 10-21 EOD) |

## 4. Open leads (where the next gain may be), most promising first

1. **The dip book's meta-model — searched, nothing beats the deployed fit (PREREG142-146).** Its ranking is worth about
   11 percentage points a year over the deepest-dip pick (PREREG137 Amendment 5). Rebuilt on the repaired stores and
   refit (142), refit ten recipes (143: ten-seed ensembles, a within-day ranking objective, robust and fill-aware
   labels, a smaller model, three dip-day inputs, the volatility heuristic), fit on 2.7-6x more dips from smaller
   stocks (144) and given a second, medium-tier model serving the $250M / $500M - $1B dips beside it (145: the
   medium tier's dips earn +0.36% a trade unranked and ~0 as a day's best when no $1B dip fires; adding them, however
   ranked, costs return a day and brings the March 2020 crash into the account), then pointed at the BIGGER stocks
   (146: their dips earn about twice as much a trade, but restricting the pool to $2B+ / $3B+ / $5B+, ranking them
   first, or fitting a model on them alone adds nothing — the model's pick already beats the biggest dip on the days
   both fire, and a cut pool leaves its slots idle): every alternative loses in the account (-1.4 to -11.5 pp a year,
   the $2B pool +0.7 pp short of the rule) and none ranks the daily pick better (PBO 0.47). The measured obstacle is FIT NOISE: a single refit's growth moves ~3 pp a year with its random seed
   alone (seeds 0-4 of the deployed recipe: 53.8-56.8%; the deployed fit is the best of its five), more than any
   change tried — and the corrected data, fairly compared, neither helps nor hurts (single-seed means 55.4 vs 55.0%).
   That different kind of information was tried on 2026-10-20 (PREREG170): the companies' own SEC XBRL numbers point
   in time, the sector fund's trend and move, Wikipedia attention, each as ten-seed ensembles against the same
   construction's control — none passes (the fundamentals helped the picks of 2014-2020 and hurt every year
   2021-2026; attention loses in both eras). The market context it lacked was tried on 2026-10-22 (PREREG181: the
   grind state, the volatility regime against its own history, the breadth of the 100 most-traded stocks, judged on
   the whole daily decision — the cut and the pick): none decides better, all five together worse (-0.08% of a
   position a day). What is left untried is the release's CONTENT and data the store does not hold; any yearly refit
   is judged across several seeds (`docs/DIP_LONG.md`, Refit).
2. **The weakest context — below SPY's trend in high volatility** (+38% a year against +50-69% elsewhere): an arm that
   earns there is worth the most (the bank of 2026-10-19: +27% a year below the trend in high volatility, the ETF arm
   losing there). Tried and failed: index dips (PREREG110 / 118), the gap-down reversal (119), 3x books (123 / 124),
   overnight drifts (122), the defensive Treasuries / gold book (155); on the deployed configuration (+21% a year in
   that context, 63% of its 18 months positive; +67-121% elsewhere) the 3x books (161), gold on the weak days (162:
   consistent out of sample on 2000-2006, not confirmed; -0.3 to -1.7 pp in the account), the dip rule under the
   stock's own trend (163: falling knives), the pre-earnings run-up (164) and, on 2026-10-22, machine-learning models
   trained per market context ranking large-cap longs (177) and shorts (178) — the bear cell holds two opposite
   markets: the GRINDING bear (2022, 2008), where large-cap shorts earn (a short book lifted 2022 +62 -> +84-89%), and
   the V-SHAPED crash and rebound (2020, 2025, the fresh breaks of 2019 and 2023), where they lose and the crash and
   capitulation books earn; neither the context cells nor the models tell them apart in advance, and neither does the
   point-in-time GRIND state (SPY under its 50- and 200-session averages, PREREG179; the death cross, a falling
   200-session average, no volatility spike and persistence beside it): large-cap shorts in the grind earn only in
   2008 and break even in 2022, while the beaten-down large caps BOUGHT in the grind rebound in both eras (+0.31 /
   +0.23 %/day) — yet a three-slot book of them loses in the account, preempted or not (PREREG180: the edge sits on
   the mass-rebound days the capitulation book already takes); and the 3x fund bought when the grind ends (PREREG182) loses to holding it at random, since at the time 57% of the exits are false. THE CLOSEST: the capitulation breadth
   (163 Amendment 1, exploratory) — every $2B+ dip under its own trend on a session with at least 5 such signals,
   in a give-way place — lifts the weak context to +29% a year and the account +1.95 pp, but 2022 alone carries it
   and K 5 is a lone peak: a fresh registration on unseen data (set D's accrual, a survivorship-free 2007-2017
   panel) is its next step, with breadth as a SHARE of the $2B+ names (their count grew from 10-20 a day in 2020 to
   many more since: 1 five-signal session in 2020, 31 in 2022, 21 in 2026). The end of a VIX term-structure inversion (PREREG169's L2, a crisis buy) is significant per trade (+2.26% over 6 sessions, +0.375 %/day, 95% +0.047 .. +0.703, 2006-2026) on 66 trades only (22 since 2018): INSUFFICIENT. Year over year (PREREG169's note of 10-20 21:31, `years169.py` / `overlap169.py`) it shows no drift beyond chance (the year-to-year spread p 0.78, the halves 0.367 / 0.385 %/day, 13 of the 18 years with a trade positive), but 2018-2020 lost (12 trades, win 42%, -0.455 %/day) and on 2018-2026 it earned what QQQ x3 bought at any session's open earned (+0.141 against +0.140 %/day); it rarely overlaps the crash book (5 of 66 holds) and mostly follows a capitulation-book session (41 of 66, the weaker trades: +0.243 against +0.605 %/day); at about 3 trades a year more history (VIX futures from 2004: two calm years) cannot reach the bank's 100 trades. In the account (Amendment 2) it is rejected in both places: -0.24 pp a year with 4 more margin-call draws in the crash book's place; +0.75 pp (40 draws, under the +0.5 log-point bar) in the give-way place, carried by 2024 while 2018-2020 lose 4-11 points a year and the worst 12 months turn negative. Untested: the selloff after a LOWERED outlook rebounds +4.1% in 20 sessions, mostly in
   bear markets (PREREG139, post hoc — crash-book territory). THE CAPITULATION BOOK (PREREG165 / 167, 2026-10-20): every $2B+ dip under its own trend on a session when at least 20% of the 100 most-traded stocks are oversold under theirs — confirmed out of sample on 2006-2017 (+1.79 %/day, survivorship-free) and, giving way to the other books at 1/6 a position, +3.38 pp a year on RPD_IS (+3.57 on top of RP33 + XC; the three together +7.2 pp, the weak context +21 -> +33% a year) — DEPLOYED from the 2026-10-21 session with RP33 and XC (the user: "Deploy all three."); its plateau in the breadth threshold fails at 15%: the threshold is the first parameter to revisit on set D's accrual, and its bank card is to build.
3. **The fall arm is INSUFFICIENT** — significant per trade (+1.70 %/day, 95% +0.76 .. +2.96) but 78 trades; more
   history decides (set D's accrual, a longer reconstruction). PREREG103: its slice is the first to revisit if the
   live record stays flat.
4. **V6 has never been re-tested under today's borrow construction**: its pass rests on 16 distinct set-B extras under
   PREREG67's independent lend draws (PREREG71), and the bank's cards leave it out.
5. **Decisions waiting for the user**: the 2010-2017 replay of the deployed $2B+ book at 1/3 (PREREG147's S3, the
   user's call over the rule's ADD — the tier-sized pick, PREREG148 / 149's T3_6; both exploratory, 36 trials on the
   same history); the ETF arm (PREREG150: +3.4 pp a year against 3 of the 7 margin-call draws, all in set B's
   modelled borrow — recommended kept); PREREG102's VIX block (live as the interim;
   risk protection; no decision recorded); PREREG77's ETF changes (the whole give-back and the turn failed the two-set rule, PREREG82; no pooled
   re-reading yet); the ETF dual-momentum book (built, off, never judged under the bank's standard); the pressure
   extras' retry (production retries an extra IBKR cannot lend only while its pick is pending — the docstring and the
   runbook said the whole window; a one-line change either way); from PREREG184 / 185 (2026-10-22): a MARKET-NOISE
   FLOOR for deploying (t >= 2 on the in-sample months, or 1.65 with the drawdown test — its own record chose a fifth
   as many changes that did better, the same total; it tightens the 2026-10-10 rule admitting non-significant gains),
   applied to the SECOND ACCOUNT instead (PREREG186 / 189; its configuration and form the user's call); the
   margin-call criterion as a RATE ADOPTED 2026-10-23 (RATE 10-23); the forward check in the EOD chain and the email
   the same day (FWD 10-23).
6. **Out-of-sample records accruing**: the earnings demotion (about 4-14 changed picks a year; the `not_top` journal
   keeps the passed-over dips with their numbers), the news-hold shadow (~2 events a year), set D (PREREG106), the
   forward check's paired trades (PREREG185: 14 closed on 2026-10-22, re-run on any later ledger copy). Reviewing
   a deployed change on set D (G-FORWARD) is proposed, not adopted (`docs/OVERFITTING.md`).
7. **Parameters off their plateau**: the crash book's exit average (5) is a one-point peak on 1988-2020 and its RSI cut
   (5) lies off that history's plateau ([3]) (PREREG97 Amendment 1).
8. **The margin cushion**: every 3x add-on failed on margin calls; the next growth lever is protection that keeps the
   cushion (`memory/margin-cushion-3x-books-2026-10.md`). PREREG154 measured it directly: a QQQ sleeve of all the idle
   capital costs 11.2 points of yearly growth (67.8 -> 56.6%) and more than doubles the margin-call draws (7 -> 15 of
   20); one held only while the initial requirement stays at most 30% of equity adds +3.15 pp (70.9%) — but as a lone
   peak (20% -0.25 pp with 11 call draws, 40% +2.63, 50% +0.17):
   a cushion rule that is smooth in its parameter (on the maintenance requirement, or a sleeve that leaves before the
   shorts need room) is the next form to register (`memory/qqq-sleeve-2026-10.md`). On the configuration deployed
   2026-10-20 (PREREG160) the full sleeve costs -8.5 pp a year with no added margin call, and the cushion's best moved
   to 20% (+3.1 pp), again a lone peak: the sleeve's response tracks which squeezes meet a held lot. On it the 3x night
   books no longer add margin calls but add no growth either (PREREG161: NBQ +0.6, NBS -0.7 pp; 2020 carries them).
9. **One continuous account from 2018** (PREREG105's rebuilt sets A2 / B2): built, never run; it would replace the
   chained sets.
10. **The joint tuning's consistent choices (PREREG153)**: the vol arm's squeeze cover at 8x (17 of 18 folds; live
   4x, kept as risk protection by PREREG82), the crash book at 1/3 and the fall arm at 1/8 (the growth folds), the
   dip exit average 3 for return per day: each a candidate for its own paired add-rule test (20 draws a set; the
   folds' risk limits used two). The stricter out-of-sample check of a tuning is a walk-forward (each year tuned on
   the years before it only).
11. **Context activation under return per day**: PREREG108's K1 raised the dip book's return per day (0.16 ->
   0.20 %/day, growth +0.5 pp, rejected on growth); its maps must be rebuilt from RPD_IS's arms (the reliability
   cards are DEF151's) before its account test — with the cards' rebuild for the deployed configuration.
12. **The vol arm's own time limit (PREREG172, 2026-10-21)**: the shorts' 10-session limit covers the vol arm's
   squeezed shorts at their worst, and many come back within days (SBET -104% at 10 sessions, -3% at 12; MLGO -112 ->
   +10% at 20; others squeeze on: POLA -59 -> -146%). The vol arm ALONE at 15 / 20 sessions (the other books live, the
   live sizing): +3.7 / +3.7 pp a year in sample, no margin call (5 draws); at 20 its held-out growth contribution
   rises in 7 of 9 years (book level), its card qualifies again (+0.92 %/day, 95% +0.34 .. +1.53, in sample) and its
   alone drawdown falls 26.6 -> 20.0%. MEASURED (PREREG173, 2026-10-21): tuned alone by year-held-out folds the vol
   arm's limit chooses 15 or 20 in every fold and holds out at +2.7 +- 1.2 pp a year (exploratory: Amendment 1, after
   the registered three-arm search failed to hold out); on 20 draws V15 (the vol arm at 15, ETF and fall at 10) +3.0 pp
   a year in sample (73.7 -> 76.7%, every set, the drawdown 26.5 -> 25.8, the vol book's return per day kept) but a
   margin call in 2 (set, draw) pairs against live's 1 — one event, 2024-05-09 (AGBA held past 10 sessions while two
   fall shorts squeeze), in two set-A draws: REJECT by the add rule — DEPLOYED BY THE USER'S CALL on 2026-10-21 ("Move the vol arm to 15 sessions"; `sel_short_vol_max_hold_sessions`, the vol picks from the 15:00 ET tick); its live record is the check. The ETF and fall limits stay at
   10 (other values lose). Production would need a vol-arm setting of its own (the vol / ETF / fall arms share
   `sel_short_arm_max_hold_sessions`; the simulator's fall arm inherits the vol arm's limit unless set).
13. **The dip book's shorter exit is a 2025 effect (PREREG176, 2026-10-21)**: the exit on the first close above the 2-session average (at most 7 sessions) raises the dip book's own return per day 0.54 -> 0.65 %/day and its growth alone (49.3 -> 52.1% a year), but in the account it loses in 6 of 9 years (2018 -2.3 .. 2020 -4.1 log points) and 2025 carries it (+10.9). If it is worth anything it is in a strong bull market: a context-activated exit (THE EVALUATION STANDARD's context map, point in time), tested out of sample, not the default exit.
14. **The dip book's exits at the open — SETTLED (PREREG187, 2026-10-23)**: the 09:30:30 sale averages -2 bp against the opening print on 665 exits 2018-2026 (t -0.95), -0.25 pp a year in the account: the simulator's print stays. PREREG185's -0.17% on the first live exits was 2026-10-20's falling morning. Open only if the forward check's dip exits drift beyond their noise over many episodes.
15. **IBKR's refusals by name (PREREG188, 2026-10-23)**: the simulator refuses a random 7% of the stock picks (`mimic.REFUSAL_RATE`); IBKR refuses particular names — small caps under a compliance restriction or close-only risk management, the same name at every check (PMI on 2026-10-20: the simulator's +35% fall short, refused live). If the refused names are the volatile winners, the backtest overstates the short arms. The what-if answers are journaled for every funded pick since 2026-10-05 (`entries/<day>.jsonl` ``ibkr_margin``): a refusal model by name (exchange, price, market cap, Reg SHO) fitted on them, then the shorts' pieces re-judged. THE DATA (2026-10-23): the entry journal holds 3 what-if answers since 2026-10-06 (one refusal: PMI, close-only); the 10-05 probe 13 refusals (about 7% of the names asked: 8 "Small Cap, Subject to Compliance Restriction", 5 risk management's close-only) — too few to fit. Both reasons are a NAME's state (its size, IBKR's risk desk), not a daily draw: first measure how long they last (the 13 names re-asked, one what-if a minute after hours), then build the sample (a few candidates a night at that pace, never in bulk) — touches the production gateway, the user's call.

## 5. Known gaps and defects (not fixed)

* **SET D RUNS (PREREG188, 2026-10-23)** — on PREREG105's continuous stores extended to the latest session (`fetch105.py extend` + `work`, `build105.py work`; then `studies/prereg188/h188.py prep / books / legs / work`, `etf188.py` for the ETF picks without the VIX-linked and stock-era products): it reproduces production's decisions (99% of the vol arm's bars from 10-06, every ETF pick, every dip top pick) and its account from the fresh start within 0.13 points on the same trades. Remaining: IBKR's REFUSALS are drawn at random (`mimic.REFUSAL_RATE`) where IBKR refuses particular names (PMI on 10-20, a +35% short the simulator takes); the halts after 2026-10-06 and DRAM's type in the rebuilt dip panel (an ETF whose symbol was once a stock: a signal, never a pick) are known; AB2 itself stays under review for sets A / B (91-100% of set B's short pieces identical, the long books identical). Extending D: rerun the fetch and the build (about 30 minutes), then the runner.

* **THE LEDGER NEVER CREDITS A FAVOURABLE FILL** (PREREG185): a leg's realized cost (`spread.real_leg_cost_frac`) is
  floored at 0, while the trade keeps the ledger's own price — AEHL's cover filled 1.5% under the ledger's price and
  its costs read 0. The ledger is conservative against IBKR's fills there, by design; the live account's P&L is the
  ledger's, not IBKR's (the paper account is CAD-based).

* **THE MARGIN CALLS** (PREREG174 / 175, 2026-10-21): the simulator and production's account both liquidate as IBKR
  documents it (the most recent positions first, 100-share lots, only back to the requirement, along every bar since
  the last check). What remains: not documented by IBKR — whether it liquidates US stocks after hours (most of the
  default's margin-event pairs are an 18:00 ET bar; both liquidate there, the safe side), how it orders positions
  beyond "most recent transactions" (the largest requirement first: -0.09 pp), when it withdraws Soft Edge Margin (on:
  +0.04 pp); not modelled — IBKR's intraday house-rate raises on a squeezing name, the overnight session in the
  simulator (no stored price; production checks it at its overnight ticks); production checks at its ticks, so it
  reaches a crossing up to an hour after it (the hourly slots) and books the lot at the modelled price while the broker
  covers at the price then; a bad print in Polygon's bars would liquidate both alike.

* **THE CAPITULATION BOOK (from the 2026-10-21 session)**: its card is built (2026-10-21; QUALIFIES re-judged on
  2007-2026 with PREREG165's untouched trades, which are every signal on daily bars, not the engine's arm-alone trades);
  production places its orders after the dip book's open fills of the same tick (the engine places both books' orders
  first: a capitulation order with no room left is `account_full` live, `unfunded` in the engine — the fills agree);
  the engine's give-way for a dip fill follows production's (`Rules.fill_room_exact`, on with the book: the fill's own
  size decides; identical to the slice-capped rule on PREREG167's 60 accounts); the breadth of a PAST
  session read today uses today's type table (a name delisted since is missing; live reads the table of its day).
* **THE BANK'S REFERENCE is PREREG167's ALL3 from the 2026-10-21 session** (`bank.LIVE_RUNS`); a study that must
  reproduce the configuration before it passes `Mimic(**mimic.PRE167)` (the research harness of PREREG152-167 pins
  `dip_filter=False` through study102's `BASE_KW` and builds its books with the crash and capitulation switches as they
  are: a new study on that harness passes the three explicitly).
* **THE VOL ARM FAILS THE BANK'S SIGNIFICANCE under the deployed RPD_IS exits** (the cards rebuilt 2026-10-21: 218
  trades, +0.751 %/day, 95% -0.177 .. +1.649; halves, every year left out and growth pass). One seed (set B's trades
  follow its lendability draw; RGC -318% is set B's): the next check is the vol card over several seeds, then — if it
  holds — a registered look at the vol give-back with significance beside the return per day (PREREG152 chose 0.5 on
  return per day alone). The account-level result (ALL3, +73.6% a year) is unchanged; nothing is decided on it yet.
* **THE CARDS FOLLOW THE DEPLOYED CONFIGURATION from 2026-10-21** (they were the previous configuration's until then):
  `reliability.SPEC` names each arm's settings, the live calibration's reference carries their fingerprints and the
  simulator's, and a deploy that changes either makes the reference STALE — the email's scorer line asks for
  `python -m src.backtest.reliability --build` (and `--build --sets A0`) until it is rebuilt.

In the research tooling and the data:

- **THE SIMULATOR'S MODEL OF THE MARKET (AUDIT151, 2026-10-18; the corrections are `Mimic()`'s DEFAULTS FROM
  2026-10-19 — the residuals below stay)**: the deployed account's backtest is 71.6% a year as stored and 67.9% with
  all seven corrections as defaults (DEF151; calls 7 = 7, dd 31.1). Residual: a leg Polygon has no quote for (50 of
  2,480 traded) pays its arm's median; the intraday-lendability estimate rests on 16 sessions of our archive (refresh
  it); a study's new legs must be fetched before it decides (`src/backtest/quotes.py`). Optimistic: the short legs' costs
  (the ledger's calm-name buckets, ~23 bp; Polygon's real quotes at the 2,418 traded legs: vol entries 51 bp / exits 83
  bp on average) -4.3 pp; lendability judged on the community archive's ONE morning file a day (07:15 ET) while live
  judges the file at the trade — a morning-lendable small pool (<= $250k) is still lendable at the pick 55-73% of the
  time (our per-tick archive, 16 sessions) -4.5 pp (-3.5 .. -5.3; set B's lend model was fit on the same morning
  files); the halt guard -0.7 pp (it replaces PREREG128's -0.44 lp reading). Pessimistic: set A's dip panel starts
  2021-01-04, so its dip book had no signal before 2021-11 (+5.4 pp with the A0 panel's identical 2016-2020 rows
  prepended); the dip open fills (+0.6); dividends (+0.2); the ETF arm shorted 8 tickers that were stocks then
  (CGRO, ESSC, EYES, FIXX, GRIN, LAZR, SPAQ, STRN; +0.2). Bounds: real-time margin (a real account is margined
  continuously; at each bar's worst price 15 call draws of 20, -4.6 pp); the A0 pressure flags from a model trained on
  A0 (<= -1.0 pp). Unmeasured: the 1,039 delisted ETPs and the renamed tickers absent from sets A / B (PREREG105's
  rebuild, never run to its account runs: ~1 in 4 of set A's vol trades would differ), house rates rising during a
  hold, days to cover landing 2 sessions late live. The fitted spread model the evaluator carries
  (`VolEngine(nbbo=False)`) reads ~4x the real quotes: never cost a study with it.
- **Symbol-reuse stitches** (META, SPCX, SNOW; 297 breaks, ~20 trades in reach): a reused ticker stitches two
  securities into one history. FIXED FOR THE DIP BOOK on 2026-10-18 (user: "Yes please apply the fix"): production
  reads each name from its own security (`src/signals/dip_security.py`, PREREG142's construction) and the simulator's
  default dip panels are PREREG142's corrected ones (`strategies.dip.OWN_DEFAULT`; the stored `dip_daily*.pkl` kept for
  the studies before it) — the deployed configuration +56.1 -> +56.7% a year. NOT checked: the vol engine's store and
  the other arms' series (the shorts' ATR% history of a reused ticker; `sel_short._stale_base` only journals a gap).
- **The 24-hour research options** (`overnight`, `pm_exits`, `dip_ext_orders`): five defects (the after-hours dip leg
  charged twice, ...); PREREG121 / 123's overnight verdicts are biased and were not re-run.
- **The data audit's residuals** (AUDIT127): ~1,300 split records unchecked; the GE 2023-01-04 spin-off is recorded as
  a split and stays in the effective set; the dip book's 2007-2017 history lacks the companies delisted before 2021.
- **The delisted company filings store** holds most acceptance times 4-5 h early (`memory/edgar-acceptance-drift.md`;
  no dip flag affected).
- **Costs**: from 2026-10-19 every market leg pays its real quoted half-spread (AUDIT151's C_QUOTE, the default);
  PREREG62's fitted-model ~61 bp (`VolEngine(nbbo=False)`) is ~4x too wide — never cost a study with it.
- **Set B's lendability is modelled** (its level drifts; read with `lend_shift=±0.37`): its margin calls are likely
  pessimistic.
- **The overfitting bookkeeping**: the trial ledger holds only PREREG103's seed (studies pass N by hand); the audit has
  run once, by hand (`docs/OVERFITTING.md`).
- PREREG128: two A0 seeds differ by 2e-5 between runs (untraced). Polygon's late overnight prints: re-check after
  2026-12-06. Provenance: some early PREREG plans (10-12, 35, 39-43) were saved after their results, and three header
  times (22b, 22c, 31) were estimates.

In production (each also named in CLAUDE.md or its runbook):

- The pressure extras' retry (section 4, item 5).
- Overnight ticks judge exits on the 19:59 price (PREREG121, kept); bar-12 picks enter in the post-market; a Rule 201
  trigger in the pre-market is missed.
- The daily OHLCV cache's split guard covers only the Polygon path (a failed whole refetch or the yfinance fallback
  merges across two scales), and the 30-minute TICK cache has none — both feed the NAV and the shadow pipeline, not the
  live books.
- The VIX-linked list is fixed (`sel_short.VIX_ETPS`): a product listed later is ranked until it is added.
- IBKR's Flex statement (the borrow fees IBKR charged) has never been collected: every statement IBKR issued since
  2026-10-07 stays "generation in progress" (code 1019) for the hour polled, on IBKR's documented v3 host; asked for a
  7-day window with IBKR's `fd` / `td` override it stayed so for 15 minutes too (2026-10-19, AUDIT151's implementation)
  — not the query's 365-day period. The next check is the user's: run the query by hand in Client Portal (a query
  holding only the Borrow Fees Details section, a short period, a new token if it never completes there).
- FIXED from the 2026-10-19 session (AUDIT151, the user's go-ahead): the dip book's OPEN fills are linked to the
  session's resting IBKR order and booked at IBKR's own average fill (the print kept as `dip_open_print`); a leg with
  no usable fill of its own is costed at its own quote at submit (half-spread + IBKR's fees on its shares) before the
  calm-name buckets; the live books' trades carry their dividends (a long receives, a short pays). The two 10-14 open
  fills before the fix (LRCX, AMAT) keep the print.
- `requirements.txt` pins yfinance 0.2.51; the venv runs 1.2.0.
- The production code deployed 2026-10-16 / 17 (`split_check`, `integrity`, `news_hold`, `dip_earnings`) is not
  committed to git.

## 6. Where things are

| What | Where |
|---|---|
| a study's registration, amendments and result | `llm_trader_research/optvol/` (1-39), `long/` (40-43), `etfalgo/` (46-51), `studies/prereg<n>/` |
| the deployed configuration's draws (20 a set, with daily NAV) | `llm_trader_research/studies/prereg142/runs/{A0,A,B}_R142_OLD_s<seed>.pkl` (`bank.LIVE_RUNS`: the own-security dip panels, which `Mimic()` reproduces exactly); on the stored panels `prereg116/runs/*_DMTE_*` (Q1's and L136's beside) |
| the bank (each arm's qualification, the combination's profile) | `cache/ml/bank/<day>.json` (`python -m src.backtest.bank`) |
| each arm's card | `cache/ml/reliability/<day>/` and `<day>_A0/` (`python -m src.backtest.reliability --build [--sets A0]`) |
| the live record against the simulator | `cache/ml/live_calibration/<day>.json` (the EOD chain) |
| the distilled findings, one per note | `memory/*.md`, index `MEMORY.md` (Claude's memory folder) |
| the reports | Artifacts linked from each memory note (e.g. the earnings studies: https://claude.ai/artifact/9gVkGD4XAdMnhRhAgsQNPn) |
