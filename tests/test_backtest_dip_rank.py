"""PREREG107: the dip book's orders ranked by the meta-model's walk-forward score, and only each day's top k."""
import pytest

from src.backtest import mimic as M
from src.backtest.strategies import dip as DIP


def _p(tk, day, rsi, k=0):
    return {"oid": f"{tk}|{day}", "tkn": tk, "rank": float(rsi), "ord_k": k}


PIECES = [_p("AAA", "2024-01-02", 3.0), _p("AAA", "2024-01-02", 3.0, k=1), _p("BBB", "2024-01-02", 1.0),
          _p("CCC", "2024-01-02", 8.0), _p("DDD", "2024-01-03", 5.0), _p("EEE", "2024-01-03", 2.0)]
SCORES = {"AAA|2024-01-02": 0.9, "BBB|2024-01-02": -0.4, "CCC|2024-01-02": 0.1, "DDD|2024-01-03": 0.3}


def test_best_first_ranks_by_score_and_puts_unscored_last_by_rsi2():
    out = DIP.rank_pieces(PIECES, SCORES, by="score")
    r = {p["oid"]: p["rank"] for p in out}
    assert r["AAA|2024-01-02"] < r["CCC|2024-01-02"] < r["BBB|2024-01-02"]
    assert r["EEE|2024-01-03"] > max(v for k, v in r.items() if k != "EEE|2024-01-03")
    assert [p["rank"] for p in PIECES] == [3.0, 3.0, 1.0, 8.0, 5.0, 2.0]          # the shared pieces untouched


def test_top_k_keeps_each_days_best_orders_with_all_their_pieces():
    top1 = DIP.rank_pieces(PIECES, SCORES, by="score", topk=1)
    assert sorted({p["oid"] for p in top1}) == ["AAA|2024-01-02", "DDD|2024-01-03"]
    assert sum(1 for p in top1 if p["oid"] == "AAA|2024-01-02") == 2                # an order's every piece
    top2 = DIP.rank_pieces(PIECES, SCORES, by="score", topk=2)
    assert sorted({p["oid"] for p in top2}) == ["AAA|2024-01-02", "CCC|2024-01-02", "DDD|2024-01-03",
                                                "EEE|2024-01-03"]


def test_production_ranks_are_kept_without_score_ordering():
    out = DIP.rank_pieces(PIECES, SCORES, by="rsi2")
    assert [p["rank"] for p in out] == [p["rank"] for p in PIECES]


@pytest.mark.parametrize("kw", [{"dip_rank": "best"}, {"dip_topk": 0}, {"dip_topk": True}, {"dip_slices": 0}])
def test_mimic_refuses_bad_dip_options(kw):
    with pytest.raises(ValueError):
        M.Mimic(seed=0, **kw)


def test_dip_slices_sets_the_slice_and_the_open_cap():
    s = {x.name: x for x in M.live_strategies(M.Mimic(seed=0, dip_slices=5))}["dip"]
    assert (s.slices, s.max_open) == (5.0, 5)
