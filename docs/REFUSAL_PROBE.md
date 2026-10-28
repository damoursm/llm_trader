# The refusal probe — IBKR's refusals of opening shorts, asked every evening

From 2026-10-26 (PREREG191; the user: "Implement changes 1 to 4"). The code is `src/signals/refusal_probe.py`. The
settings are `enable_eod_refusal_probe` and `refusal_probe_*`. Tests: `tests/test_refusal_probe.py`.

## What it is for

IBKR refuses any opening short in some names: small caps under its compliance restriction, risk management's
close-only names, DTC chills. PREREG190's model (`src/signals/sel_refusal.py`) predicts those refusals by name. The
simulator needs that model to know which stocks it can realistically short. The model learns only from IBKR's own
answers, and IBKR keeps no history of them. The probe collects more of those answers every evening.

Nothing trades on the probe. The entry step keeps asking IBKR at the moment of each trade, and that answer decides.

## How it runs

1. At 16:20 ET on market days, the EOD chain's `refusal_probe` step launches the probe as a detached process
   (`python -m src.signals.refusal_probe --day D`, log `logs/refusal_probe_<day>.log`). The chain never waits for it.
   The probe only runs on the evening of its own session, on the paper gateway (`broker_mode` `ibkr_paper`), on its
   own API client (`ibkr_client_id` + `refusal_probe_client_offset`, 61).
2. It plans the evening's names, in this order:
   - the research queue (`cache/ml/sel_short/refusal/queue.json`), every name until it is answered, not capped;
   - every name whose last answer was a refusal, again after `refusal_probe_reask_days` (7);
   - THE WATCH (from 2026-10-24): the names IBKR last accepted that the model finds riskiest in the vol arm's
     universe that day, again after 7 days, at most `refusal_probe_watch_names` (10) a night. Asking the same
     names over time is the only way to see a name turn refused (close-only, a chill); in the arms' universe the
     riskiest accepted names carry 1-7% (2026-10-26), so a turn is expected to be rare;
   - the day's live picks and their backups that IBKR was not asked about;
   - the day's new names in the vol arm's top 50;
   - the names in the vol arm's universe that the model is least sure of (the highest p(1 - p)).

   The last five sources together take at most `refusal_probe_max_names` (60). A name answered within
   `refusal_probe_fresh_days` (30) is skipped, except for re-asks and the watch. Funds are left out: the model never refuses one.
3. It asks one name every `refusal_probe_spacing_seconds` (60), using production's what-if: a ~$500 DAY limit SELL
   that IBKR prices and never transmits, parsed by `ibkr.whatif_refusal`. Its guards:
   - it asks nothing from one minute before to three minutes after any :00 or :30 slot, or 19:50;
   - it pauses while the EOD spread sweep runs (another client on the same gateway);
   - it stops at once when IBC logs a server reconnect or a re-login demand;
   - it ends at `refusal_probe_end_et` (23:15 ET), before the nightly deep refresh and IBC's restart.

   Two bulk probes at 1-2 s spacing each coincided with the gateway losing IBKR (2026-10-05). Never speed it up.

## In session: the picks settled before a what-if

The entry step asks IBKR before it sizes a pick, so a pick the account cannot fund already has its answer. A pick the
entry step settles before its what-if, for a reason of the main account's own, has none:

- `already_open`: the name is at its stack cap or held by another book;
- `traded_today`: its arm already traded the name today;
- `shadow_arm`: an arm the main account does not trade.

After the live path's broker sync, and before the extra accounts run, `tracker.ask_unasked_sel_short` asks IBKR about
those picks. It asks only when an account trades the pick's arm and the name has no answer that session, at most
`sel_short_whatif_unasked_max` (3) a pass. The answer is journaled ``whatif_only`` in the entry journal, where the
second account reads it (`sim_account.main_account_answer`).

## Files (`cache/ml/sel_short/refusal/`) — never delete `probes/`

- `model.json`: the installed refusal model (PREREG190's M2c).
- `queue.json`: the research queue.
- `probes/<day>.jsonl`: one line per ask. Each line holds the ticker, its source, the time, the price, and either
  IBKR's refusal, its rates or an error, plus the model's probability at the ask.
- `probes/<day>.state.json`: the evening's run (planned, asked, refused, why it stopped).
- `probes/import_<source>.jsonl`: a study's answers (PREREG190's, PREREG191's), counted as asked.

## Reading it

The live calibration (the EOD chain's `live_calibration` step) scores the answers against the model in its
``refusal_probe`` section, by source and for the arms' candidates (the picks and top-50 names) apart, and lists
every name whose answer changed when it was asked again an hour or more later (``transitions``: to refused, to
accepted). A note goes to
the email's scorer banner when those refusals fall outside the model's 95% range, or when the evening's run failed.

## Commands

    python -m src.signals.refusal_probe --day D --plan                     # the evening's plan; asks nothing
    python -m src.signals.refusal_probe --queue FILE                       # set the research queue (FILE's "order")
    python -m src.signals.refusal_probe --import FILE --source NAME        # a study's answers

To stop a running probe, end its process; the lock (`probe.lock`) is ignored once its process is dead.
