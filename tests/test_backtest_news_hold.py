"""PREREG136's `Mimic(news_hold=...)`: a short's own exit postponed k sessions after the rule's news (`news_hold_piece`,
`news_extend`, `fell_on_filing`, `llm_positive`, `news_hold_spec`) — on synthetic stored bars."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import mimic as M

ET = "America/New_York"
DAYS = ["2024-03-04", "2024-03-05", "2024-03-06", "2024-03-07", "2024-03-08", "2024-03-11", "2024-03-12"]


def ns(day: str, hhmm: str) -> int:
    return int(pd.Timestamp(f"{day} {hhmm}").tz_localize(ET).tz_convert("UTC").value)


def make_bars(close_of=None):
    """Each session: pre-market bars 08:00-09:30, regular bars 09:30-16:00, post bars 16:00-17:00 (30 minutes);
    close = 10 + 0.01 x the bar's index unless ``close_of(day, start)`` says otherwise."""
    start, c, rth, day = [], [], [], []
    k = 0
    for d in DAYS:
        times = pd.date_range(f"{d} 08:00", f"{d} 16:30", freq="30min")
        for t in times:
            s = int(t.tz_localize(ET).tz_convert("UTC").value)
            start.append(s)
            reg = (t.hour * 60 + t.minute) >= 570 and (t.hour * 60 + t.minute) < 960
            rth.append(reg)
            px = 10.0 + 0.01 * k
            if close_of is not None:
                v = close_of(d, t)
                if v is not None:
                    px = v
            c.append(px)
            day.append(int((pd.Timestamp(d) - pd.Timestamp("1970-01-01")).days))
            k += 1
    start = np.asarray(start, np.int64)
    c = np.asarray(c, float)
    return {"start": start, "end": start + M.BAR_NS, "o": c.copy(), "h": c + 0.05, "l": c - 0.05, "c": c,
            "v": np.full(len(c), 1000.0), "rth": np.asarray(rth, bool), "day": np.asarray(day, np.int64)}


class FakeBars:
    def __init__(self, b):
        self.b = b

    def get(self, name):
        return self.b


def make_piece(b, exit_day="2024-03-05", exit_hhmm="10:00", why="target", cover=4.0):
    """A short entered at the first session's 10:00 bar end, exiting at ``exit_day`` ``exit_hhmm``; its marks are the
    regular-hours bar closes in between (the piece's own terms = the bars')."""
    ens, xns = ns(DAYS[0], "10:00"), ns(exit_day, exit_hhmm)
    m = b["rth"] & (b["end"] >= ens) & (b["end"] <= xns)
    d0 = int((pd.Timestamp(DAYS[0]) - pd.Timestamp("1970-01-01")).days) + 1
    cum = np.cumsum(np.full(60, 10.0))                   # 10 collateral-dollar-percent a day
    p = {"tkn": "TEST", "strategy": "vol", "ens": ens, "xns": xns, "e": float(b["c"][b["end"] == ens][0]),
         "x": float(b["c"][b["end"] == xns][0]), "pt": b["end"][m].astype(np.int64), "pc": b["c"][m].astype(float),
         "ph": b["h"][m].astype(float), "d0": d0, "cum": cum, "bfac": 1.0 / 100.0 / 360.0, "full_ps": 0.0,
         "cover_mult": cover, "why": why, "pick_day": d0 - 1}
    M._reborrow(p, lambda d: d + 1)
    return p


def test_spec_parsing():
    assert M.news_hold_spec("fb:vol:3") == ("fb", ("vol",), 3)
    assert M.news_hold_spec("fb:shorts:10") == ("fb", ("vol", "etf", "fall"), 10)
    assert M.news_hold_spec("lg:shorts:1") == ("lg", ("vol",), 1)       # the LLM rule acts on the vol arm only
    for bad in ("fb:vol", "xx:vol:3", "fb:all:3", "fb:vol:0", "fb:vol:21", "fb:vol:two", "lg:etf:3"):
        with pytest.raises(ValueError):
            M.news_hold_spec(bad)
    assert M.Mimic().news_hold is None                 # production's exits by default
    assert M.Mimic(news_hold="fb:vol:5").news_hold == "fb:vol:5"
    with pytest.raises(ValueError):
        M.Mimic(news_hold="fb:vol:0")


def test_fell_on_filing_reads_the_reaction_after_the_lag():
    b = make_bars(lambda d, t: 9.0 if (d == DAYS[0] and t.hour >= 14) else None)   # falls after 14:00 on day 1
    p = make_piece(b)
    a = ns(DAYS[0], "13:40")                           # accepted 13:40; first mark >= 14:10 is the 14:30 close
    assert M.fell_on_filing(p, np.array([a])) is True
    b2 = make_bars(lambda d, t: 11.0 if (d == DAYS[0] and t.hour >= 14) else None)  # rises instead
    assert M.fell_on_filing(make_piece(b2), np.array([a])) is False
    # no filing in the hold / one before the entry / one less than 30 minutes before the exit: no news
    assert M.fell_on_filing(p, None) is None
    assert M.fell_on_filing(p, np.array([ns(DAYS[0], "09:00")])) is None
    assert M.fell_on_filing(p, np.array([p["xns"] - 10 * 60 * 10**9])) is None
    # the LAST filing of the hold decides
    assert M.fell_on_filing(p, np.array([a, ns(DAYS[0], "15:10")])) is False   # 15:10's reaction: flat at 9.0


def test_llm_positive_mean_tone_and_lag():
    p = make_piece(make_bars())
    t = np.array([ns(DAYS[0], "11:00"), ns(DAYS[0], "12:00"), p["xns"] - 5 * 60 * 10**9], np.int64)
    assert M.llm_positive(p, (t, np.array([1.0, 0.0, -1.0]))) is True          # the last is too late to count
    assert M.llm_positive(p, (t, np.array([1.0, -1.0, 1.0]))) is False         # mean 0 is not positive
    assert M.llm_positive(p, (t[2:], np.array([1.0]))) is None                 # only a too-late headline
    assert M.llm_positive(p, None) is None


def test_news_extend_moves_the_exit_k_sessions():
    b = make_bars()
    p = make_piece(b)
    q, why = M.news_extend(p, 2, FakeBars(b), lambda d: d + 1)
    assert why == "news_hold"
    assert q["xns"] == ns(DAYS[3], "10:00")              # the exit's day is DAYS[1]: 2 sessions later is DAYS[3]
    assert q["x"] == pytest.approx(float(b["c"][b["end"] == q["xns"]][0]))
    assert q["news_held_from"] == p["xns"]
    assert np.all(np.diff(q["pt"]) > 0) and q["pt"][-1] < q["xns"] and len(q["pt"]) > len(p["pt"])
    assert q["full_ps"] > p["full_ps"]                   # two more days of borrow
    # extended-hours slots are marks in between (the 16:30 slot of the exit's day)
    assert ns(DAYS[1], "16:30") in set(q["pt"].tolist())


def test_news_extend_covers_on_the_way():
    e0 = None

    def spike(d, t):
        return 50.0 if (d == DAYS[2] and t.hour == 11 and t.minute == 0) else None
    b = make_bars(spike)
    p = make_piece(b)
    q, why = M.news_extend(p, 3, FakeBars(b), lambda d: d + 1)
    assert why == "news_cap"
    assert q["xns"] == ns(DAYS[2], "11:30") and q["x"] == pytest.approx(50.0)
    assert e0 is None


def test_cover_exits_and_other_news_are_never_extended():
    b = make_bars(lambda d, t: 9.0 if (d == DAYS[0] and t.hour >= 14) else None)
    a = ns(DAYS[0], "13:40")

    class NIn:
        filings = {"TEST": np.array([a])}
        llm = {}

        def company(self, tkn, ens, a0):
            return "TEST"
    nin, bars = NIn(), FakeBars(b)
    p = make_piece(b, why="cap")
    assert M.news_hold_piece(p, "fb", 3, nin, False, bars, lambda d: d + 1) == (p, "cover")
    p = make_piece(b)
    q, why = M.news_hold_piece(p, "fb", 1, nin, False, bars, lambda d: d + 1)
    assert why == "news_hold" and q["xns"] == ns(DAYS[2], "10:00")
    b2 = make_bars(lambda d, t: 11.0 if (d == DAYS[0] and t.hour >= 14) else None)
    p2 = make_piece(b2)
    assert M.news_hold_piece(p2, "fb", 1, nin, False, FakeBars(b2), lambda d: d + 1)[1] == "other_news"
    assert M.news_hold_piece(p2, "lg", 1, nin, False, FakeBars(b2), lambda d: d + 1)[1] == "no_news"


def test_company_map():
    nin = object.__new__(M.NewsInputs)
    nin.a0map = {"FB": "META"}
    d = int((pd.Timestamp("2022-06-01") - pd.Timestamp("1970-01-01")).days)
    nin.delisted = {"OLD": [("OLD_DELISTED", d - 100, d + 100)]}
    nin.under = {"NVDL": "NVDA"}
    t = ns("2022-06-01", "10:00")
    assert nin.company("FB", t, a0=True) == "META"
    assert nin.company("FB", t, a0=False) == "FB"
    assert nin.company("OLD@D", t, a0=False) == "OLD_DELISTED"
    assert nin.company("OLD@D", ns("2023-06-01", "10:00"), a0=False) is None    # outside the company's segment
    assert nin.company("NVDL@E", t, a0=False) == "NVDA"
    assert nin.company("SPXL@E", t, a0=False) is None                            # no single stock's fund


def test_robust_scale_ignores_a_stray_bar_but_not_a_mismatch():
    b = make_bars()
    p = make_piece(b, exit_day=DAYS[2])
    assert M.robust_scale(p, b) == pytest.approx(1.0)
    q = dict(p, pc=p["pc"].copy())
    q["pc"][3] *= 1.10                                   # one stray mark: the strict scale refuses, the robust one not
    assert M.price_scale(q, b) is None
    assert M.robust_scale(q, b) == pytest.approx(1.0)
    r = dict(p, pc=p["pc"] * np.linspace(1.0, 1.2, len(p["pc"])))      # a drifting ratio: refused
    assert M.robust_scale(r, b) is None
    s = dict(p, pc=p["pc"] * 0.1)                        # a constant factor (a split one side adjusted): its median
    assert M.robust_scale(s, b) == pytest.approx(0.1)


def test_dip_demote_ranks_the_flagged_orders_last_in_their_day(tmp_path):
    """PREREG137 Amendment 1: `rank_pieces(demote=...)` — a demoted order loses the day's top pick to any other order of
    its day, scored or not, keeps it when it is the day's only order; `Mimic(dip_demote=...)` refuses a missing file."""
    import pickle
    from src.backtest.strategies import dip as DIP
    P = [{"oid": "AAA|2024-03-05", "rank": 3.0}, {"oid": "BBB|2024-03-05", "rank": 1.0},
         {"oid": "CCC|2024-03-05", "rank": 2.0}, {"oid": "AAA|2024-03-06", "rank": 1.0}]
    scores = {"AAA|2024-03-05": 2.0, "BBB|2024-03-05": 1.0, "AAA|2024-03-06": 0.5}     # CCC unscored
    top = DIP.rank_pieces(P, scores, by="score", topk=1)
    assert {p["oid"] for p in top} == {"AAA|2024-03-05", "AAA|2024-03-06"}
    top = DIP.rank_pieces(P, scores, by="score", topk=1, demote={"AAA|2024-03-05", "AAA|2024-03-06"})
    assert {p["oid"] for p in top} == {"BBB|2024-03-05", "AAA|2024-03-06"}   # the 03-06 order has no rival
    top = DIP.rank_pieces(P, scores, by="score", topk=1, demote={"AAA|2024-03-05", "BBB|2024-03-05"})
    assert {p["oid"] for p in top} == {"CCC|2024-03-05", "AAA|2024-03-06"}   # an unscored rival beats the demoted
    assert DIP.rank_pieces(P, scores, by="score", topk=1, demote=None) == DIP.rank_pieces(P, scores, by="score", topk=1)
    with pytest.raises(ValueError):
        M.Mimic(dip_demote=str(tmp_path / "missing.pkl"))
    f = tmp_path / "d.pkl"
    pickle.dump({"AAA|2024-03-05"}, open(f, "wb"))
    assert M.Mimic(dip_demote=str(f)).dip_demote == str(f) and M.Mimic().dip_demote is None
    assert M._demoted(str(f)) == frozenset({"AAA|2024-03-05"}) and M._demoted(None) is None
