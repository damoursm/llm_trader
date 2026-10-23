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


def test_dip_scores_defaults_to_the_production_map_and_refuses_a_missing_file(tmp_path):
    """PREREG142: ``Mimic.dip_scores`` (a research score file) is None by default — `strategies.dip.scores_path()`
    ranks the daily top pick: the own-security panels' scores from 2026-10-18 (the reused-ticker fix), PREREG96's with
    the stored panels, PREREG96's always for its research flags — and a path that does not exist is refused."""
    assert M.Mimic(seed=0).dip_scores is None
    assert DIP.OWN_DEFAULT is True and DIP.scores_path().name == DIP.OWN_SCORES.name
    assert DIP.scores_path(research=True).name == DIP.FILTER_SCORES.name
    with pytest.raises(ValueError):
        M.Mimic(seed=0, dip_scores=str(tmp_path / "missing.pkl"))


def test_dip_scores_file_is_the_map_the_ranking_reads(tmp_path):
    import inspect

    import pandas as pd
    p = tmp_path / "scores.pkl"
    pd.DataFrame({"oid": list(SCORES), "score": list(SCORES.values())}).to_pickle(p)
    assert DIP.filter_scores(p) == SCORES
    assert M.Mimic(seed=0, dip_scores=str(p)).dip_scores == str(p)
    # `live_books` (too heavy to run here) hands the label's file to the ranking: the wiring, checked on its source
    assert "DIP.filter_scores(mim.dip_scores)" in inspect.getsource(M.live_books)
