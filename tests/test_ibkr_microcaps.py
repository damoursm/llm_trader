"""IBKR's U.S. Microcap list archived daily (src/data/ibkr_microcaps.py; PREREG190, 2026-10-23): a download is checked
for IBKR's header and size before it is kept, one raw copy a day and never fetched twice, and a session reads the list
of a day BEFORE it (point in time), at most `ibkr_microcap_max_age_days` old."""
import gzip
from datetime import date

import pytest

from config.settings import settings
from src.data import ibkr_microcaps as IM


def _csv(symbols):
    rows = ["#SYMBOL,MAIN_EXCHANGE,DESCRIPTION,IB_CONTRACT_ID"] + [f"{s},NASDAQ,{s} INC,{i}" for i, s in enumerate(symbols)]
    return ("\n".join(rows) + "\n").encode("ascii")


@pytest.fixture
def micro_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ibkr_microcap_dir", str(tmp_path / "micro"))
    monkeypatch.setattr(settings, "ibkr_microcap_max_age_days", 7)
    monkeypatch.setattr(IM, "MIN_ROWS", 3)
    IM._CACHE.clear()
    return tmp_path / "micro"


def test_a_download_is_checked_then_kept_once_a_day(micro_dir):
    calls = []

    def fetch():
        calls.append(1)
        return _csv(["TJGC", "BQ", "BRK B", "NCI"])
    p, n = IM.archive(date(2026, 10, 23), fetch=fetch)
    assert n == 4 and p.exists() and p.name == "2026-10-23.csv.gz"
    assert IM.parse(gzip.open(p, "rb").read()) == {"TJGC", "BQ", "BRK B", "NCI"}     # the raw copy
    assert IM.archive(date(2026, 10, 23), fetch=fetch)[1] == 4 and len(calls) == 1      # never fetched twice


@pytest.mark.parametrize("raw", [b"<html>blocked</html>", b"#SYMBOL,MAIN_EXCHANGE\nA,PINK,X,1\n"])
def test_a_download_that_is_not_ibkrs_list_is_never_kept(micro_dir, raw):
    with pytest.raises(ValueError):
        IM.archive(date(2026, 10, 23), fetch=lambda: raw)
    assert not list(micro_dir.glob("*.csv.gz")) if micro_dir.exists() else True


def test_a_session_reads_the_list_of_a_day_before_it(micro_dir):
    IM.archive(date(2026, 10, 23), fetch=lambda: _csv(["TJGC", "BQ", "BRK B"]))
    assert IM.on_list("TJGC", date(2026, 10, 23)) is None          # the day's own file comes after its close
    assert IM.on_list("TJGC", date(2026, 10, 26)) is True
    assert IM.on_list("BRK-B", date(2026, 10, 26)) is True          # a class share in IBKR's form
    assert IM.on_list("AAPL", date(2026, 10, 26)) is False
    assert IM.on_list("TJGC", date(2026, 11, 5)) is None            # no list in force: unknown, not "not listed"
    assert IM.latest_day() == date(2026, 10, 23)
