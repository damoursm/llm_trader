"""A stock's sector for the per-sector cap (src/backtest/sectors.py, PREREG100): the Fama-French 12-industry group
of its SIC (Kenneth French's Siccodes12), the SIC from Polygon's stored details, else SEC's record for the CIK
(injected: no network here), else unknown; funds have none."""
from __future__ import annotations

import duckdb
import pytest

from src.backtest import sectors as SE


@pytest.mark.parametrize("sic, sector", [
    (3569, "Manuf"), (3570, "BusEq"), (3579, "BusEq"), (3580, "Manuf"), (3674, "BusEq"), (3559, "Manuf"),
    (3827, "BusEq"), (3830, "Manuf"), (2834, "Hlth"), (3693, "Hlth"), (3692, "BusEq"), (3694, "BusEq"),
    (6211, "Money"), (6199, "Money"), (7372, "BusEq"), (4911, "Utils"), (4950, "Other"), (1311, "Enrgy"),
    (5961, "Shops"), (3714, "Durbl"), (3715, "Manuf"), (3716, "Durbl"), (3711, "Durbl"), (3713, "Manuf"),
    (2800, "Chems"), (2830, "Hlth"), (2840, "Chems"), (100, "NoDur"), (3990, "Durbl"), (3989, "NoDur"),
    (4813, "Telcm"), (8062, "Hlth"), (9999, "Other"), (1000, "Other"), (6770, "Money"),
    ("3674", "BusEq"), ("0100", "NoDur"), (3674.0, "BusEq"),
])
def test_ff12_ranges(sic, sector):
    assert SE.ff12(sic) == sector


@pytest.mark.parametrize("sic", [0, "0", "0000", None, "", "  ", float("nan"), "n/a", -5])
def test_a_missing_or_zero_sic_is_unknown_not_other(sic):
    assert SE.ff12(sic) is None


def test_the_ranges_never_overlap():
    seen = {}
    for name, ranges in SE.FF12_RANGES.items():
        for lo, hi in ranges:
            assert lo <= hi
            for v in range(lo, hi + 1):
                assert v not in seen, f"{v} in {seen.get(v)} and {name}"
                seen[v] = name
    assert set(SE.FF12_RANGES) | {"Other"} == set(SE.FF12)


DETAILS = {
    "INTC": {"sic": "3674", "cik": "0000050863", "type": "CS"},
    "ARM": {"sic": None, "cik": "0001973239", "type": "ADRC"},           # an ADR: Polygon has no SIC
    "SPY": {"sic": None, "cik": "0000884394", "type": "ETF"},
    "ATVI@D": {"sic": "7372", "cik": "0000718877", "type": "CS"},          # the delisted company
    "ATVI": {"sic": "2834", "cik": "0000000001", "type": "CS"},            # (a symbol reused today)
    "OLD@D": {"sic": None, "cik": "0000000777", "type": "CS"},
    "XFD@D": {"sic": None, "cik": None, "type": "CS"},
}
SEC = {"0001973239": "3674", "0000000777": "6022", "0000000999": "4911", "0000000888": ""}
TICKER_CIK = {"NEWCO": "999", "XFD": "888", "LC": "999"}


def _labels(names):
    return SE.label_records(names, DETAILS, sec_sic=SEC.get, cik_of=TICKER_CIK.get)


def test_labels_read_polygon_then_sec_by_cik_and_funds_have_none():
    L = _labels(["INTC", "ARM", "SPY", "ATVI@D", "OLD@D", "XFD@D", "NEWCO@X", "LC", "NONE@X"])
    assert (L["INTC"]["sector"], L["INTC"]["source"]) == ("BusEq", "polygon")
    assert (L["ARM"]["sector"], L["ARM"]["source"], L["ARM"]["sic"]) == ("BusEq", "sec", "3674")
    assert (L["SPY"]["sector"], L["SPY"]["source"]) == (None, "fund")
    assert (L["ATVI@D"]["sector"], L["ATVI@D"]["source"]) == ("BusEq", "polygon_delisted")   # not today's ATVI
    assert (L["OLD@D"]["sector"], L["OLD@D"]["source"]) == ("Money", "sec")                 # the delisted store's CIK
    assert L["XFD@D"]["sector"] is None and L["XFD@D"]["source"] is None   # never today's symbol for a delisted name
    assert (L["NEWCO@X"]["sector"], L["NEWCO@X"]["cik"]) == ("Utils", "0000000999")      # SEC's ticker map
    assert L["LC"]["sector"] == "Utils"
    assert L["NONE@X"]["sector"] is None and L["NONE@X"]["source"] is None
    assert SE.load_labels(["INTC", "SPY"], DETAILS) == {"INTC": "BusEq", "SPY": None}


def test_without_the_sec_lookup_the_labels_are_built_offline():
    L = SE.label_records(["INTC", "ARM", "NEWCO@X"], DETAILS)
    assert L["INTC"]["sector"] == "BusEq" and L["ARM"]["sector"] is None and L["NEWCO@X"]["sector"] is None


def test_an_empty_sec_answer_is_unknown():
    L = SE.label_records(["XFD@X"], DETAILS, sec_sic=SEC.get, cik_of=TICKER_CIK.get)
    assert L["XFD@X"]["sector"] is None and L["XFD@X"]["cik"] == "0000000888"


def test_polygon_details_map_the_delisted_store_to_the_backtest_names(tmp_path):
    deep, dl = tmp_path / "deep.parquet", tmp_path / "dl.parquet"
    duckdb.execute(f"""COPY (SELECT * FROM (VALUES ('INTC', '0000050863', '3674', 'CS'),
                                                   ('SPY', '0000884394', NULL, 'ETF'))
                         t(ticker, cik, sic_code, type)) TO '{deep.as_posix()}' (FORMAT PARQUET)""")
    duckdb.execute(f"""COPY (SELECT * FROM (VALUES ('ATVI_DELISTED', '0000718877', '7372', 'CS'),
                                                   ('ODD', '1', '1311', 'CS'))
                         t(ticker, cik, sic_code, type)) TO '{dl.as_posix()}' (FORMAT PARQUET)""")
    det = SE.polygon_details(deep, dl)
    assert det["INTC"]["sic"] == "3674" and det["SPY"]["type"] == "ETF"
    assert det["ATVI@D"] == {"sic": "7372", "cik": "0000718877", "type": "CS", "store": "delisted"}
    assert "ODD" not in det and "ODD@D" not in det                         # not a delisted store row
    assert SE.load_labels(["ATVI@D", "INTC"], det) == {"ATVI@D": "BusEq", "INTC": "BusEq"}


def test_coverage_and_stamping():
    labels = {"A": "BusEq", "B": None, "C": "Money"}
    cov = SE.coverage(labels, ["A", "B", "C", "D"], weights={"A": 3, "B": 1, "C": 0, "D": 1})
    assert (cov["n"], cov["with_sector"], cov["share"]) == (4, 2, 0.5)
    assert cov["unknown"] == ["B", "D"] and cov["by_sector"] == {"BusEq": 1, "Money": 1}
    assert cov["weighted_share"] == pytest.approx(3 / 5)
    pieces = [{"tkn": "A"}, {"tkn": "B"}, {"tkn": "Z"}]
    out = SE.stamp_groups(pieces, labels)
    assert [p["group"] for p in pieces] == ["BusEq", None, None]
    assert out == {"with_sector": 1, "none": 2, "missing_names": ["Z"]}
