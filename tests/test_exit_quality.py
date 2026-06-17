"""Tests for src.analysis.exit_quality (MFE/MAE exit analysis)."""

from src.analysis.exit_quality import compute_exit_quality


def _closed(ticker, ret, mfe, mae, reason="signal"):
    return {
        "ticker": ticker, "status": "CLOSED", "return_pct": ret,
        "max_favorable_excursion": mfe, "max_adverse_excursion": mae,
        "exit_reason": reason, "entry_session": "rth",
    }


def test_no_trades():
    assert compute_exit_quality([])["n"] == 0


def test_exit_at_peak_is_full_placement_and_capture():
    rep = compute_exit_quality([_closed("AAA", 10.0, 10.0, -2.0)])
    r = rep["per_trade"][0]
    assert r["exit_placement"] == 1.0        # exited at MFE
    assert r["capture"] == 1.0
    assert r["give_back"] == 0.0


def test_exit_near_mae_is_low_placement():
    # Rode +8% then closed at −1.5% (near the −2% worst point).
    rep = compute_exit_quality([_closed("BBB", -1.5, 8.0, -2.0)])
    r = rep["per_trade"][0]
    assert r["exit_placement"] < 0.20        # cut near the bottom
    assert rep["pct_exited_near_mae"] == 100.0


def test_gave_back_most_of_peak():
    # Peak +10%, kept only +2% → capture 0.2 (< 0.5).
    rep = compute_exit_quality([_closed("CCC", 2.0, 10.0, -1.0)])
    assert rep["per_trade"][0]["capture"] == 0.2
    assert rep["pct_gave_back_most_mfe"] == 100.0


def test_degenerate_band_excluded():
    # MFE≈MAE≈return (entered+closed same tick / legacy) → not analysable.
    assert compute_exit_quality([_closed("DDD", 0.0, 0.0, 0.0)])["n"] == 0


def test_open_trades_excluded():
    t = {"ticker": "EEE", "status": "OPEN", "return_pct": 5.0,
         "max_favorable_excursion": 7.0, "max_adverse_excursion": -1.0}
    assert compute_exit_quality([t])["n"] == 0
