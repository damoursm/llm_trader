"""The add rule, mechanically (src/backtest/add_rule.py; user 2026-10-13: "Apply the best decisions you're
recommending")."""
import math

from src.backtest import add_rule as R


def _runs(n, growth, dd=20.0, calls=()):
    return {s: {"growth": growth(s) if callable(growth) else growth, "maxdd": dd, "calls": ["x"] if s in calls else []}
            for s in range(n)}


def _variant(live, gain_lp, noise=0.0, dd=None, calls=()):
    out = {}
    for s, r in live.items():
        g = math.expm1((100 * math.log1p(r["growth"]) + gain_lp + noise * ((s % 2) * 2 - 1)) / 100)
        out[s] = {"growth": g, "maxdd": r["maxdd"] if dd is None else dd, "calls": ["x"] if s in calls else r["calls"]}
    return out


def test_twenty_paired_draws_are_required():
    live = {"A": _runs(5, 0.37), "B": _runs(5, 1.0)}
    var = {k: _variant(v, 2.0) for k, v in live.items()}
    v = R.verdict(live, var)
    assert v["decision"] == "more draws" and v["undecided"] == ["A", "B"]


def test_a_clear_similar_gain_on_both_sets_is_added():
    live = {"A": _runs(20, 0.37), "B": _runs(20, 1.0)}
    var = {"A": _variant(live["A"], 1.2, noise=0.3), "B": _variant(live["B"], 1.6, noise=0.5)}
    v = R.verdict(live, var)
    assert v["decision"] == "add", v["why"]
    assert abs(v["sets"]["A"]["gain"] - 1.2) < 1e-9 and v["sets"]["B"]["n"] == 20


def test_a_gain_on_one_set_only_is_rejected_and_a_dissimilar_one_too():
    live = {"A": _runs(20, 0.37), "B": _runs(20, 1.0)}
    v = R.verdict(live, {"A": _variant(live["A"], 3.0, noise=0.2), "B": _variant(live["B"], -2.0, noise=0.2)})
    assert v["decision"] == "reject" and "set B" in v["why"][0]                          # far below: futility
    # B at 0.0 +- 0.21: not futile (3 SE reach +0.6) but decided (2 SE stop at +0.41): rejected on the criterion
    x = R.verdict(live, {"A": _variant(live["A"], 3.0, noise=0.2), "B": _variant(live["B"], 0.0, noise=0.9)})
    assert x["decision"] == "reject" and not x["criteria"]["gain_each"]["B"]
    w = R.verdict(live, {"A": _variant(live["A"], 4.0, noise=0.2), "B": _variant(live["B"], 1.0, noise=0.1)})
    assert w["decision"] == "reject" and w["criteria"]["similar"] is False             # 1.0 < half of 4.0


def test_a_gain_near_the_threshold_asks_for_forty_draws_then_decides_on_the_estimate():
    live = {"A": _runs(20, 0.37), "B": _runs(20, 1.0)}
    near = {"A": _variant(live["A"], 1.0, noise=0.2), "B": _variant(live["B"], 0.6, noise=3.0)}   # B: SE ~0.69
    v = R.verdict(live, near)
    assert v["decision"] == "more draws" and v["undecided"] == ["B"]
    live40 = {"A": _runs(20, 0.37), "B": _runs(40, 1.0)}
    near40 = {"A": near["A"], "B": _variant(live40["B"], 0.6, noise=3.0)}
    w = R.verdict(live40, near40)
    assert w["decision"] == "add" and w["sets"]["B"]["n"] == 40                        # judged on the estimate at 40


def test_margin_calls_are_judged_against_the_live_configuration_on_the_same_seeds():
    live = {"A": _runs(20, 0.37), "B": _runs(20, 1.0, calls=(3, 7))}
    same = {"A": _variant(live["A"], 1.2, noise=0.2), "B": _variant(live["B"], 1.2, noise=0.2)}
    assert R.verdict(live, same)["decision"] == "add"                                   # the live config's own calls
    more = {"A": same["A"], "B": _variant(live["B"], 1.2, noise=0.2, calls=(3, 7, 11))}
    v = R.verdict(live, more)
    assert v["decision"] == "reject" and not v["criteria"]["margin_calls"]["B"]


def test_a_drawdown_more_than_one_point_deeper_is_rejected():
    live = {"A": _runs(20, 0.37, dd=15.0), "B": _runs(20, 1.0, dd=25.0)}
    var = {"A": _variant(live["A"], 1.2, noise=0.2, dd=16.5), "B": _variant(live["B"], 1.2, noise=0.2, dd=25.5)}
    v = R.verdict(live, var)
    assert v["decision"] == "reject" and not v["criteria"]["drawdown"]["A"] and v["criteria"]["drawdown"]["B"]
    assert "drawdown" in R.summary(v)


def test_a_clear_failure_is_rejected_from_five_draws_without_more():
    """Futility: 5 paired draws with set B's gain more than 3 SE below +0.5 reject at once (PREREG96's RP20: -1.99 +-
    0.74 on set B after 5 draws)."""
    live = {"A": _runs(5, 0.37), "B": _runs(5, 1.0)}
    var = {"A": _variant(live["A"], 3.5, noise=0.25), "B": _variant(live["B"], -2.0, noise=1.6)}
    v = R.verdict(live, var)
    assert v["decision"] == "reject" and "futility" in v["why"][0]
    near = {"A": var["A"], "B": _variant(live["B"], 0.4, noise=1.3)}                   # PREREG96's CP20: undecided
    assert R.verdict(live, near)["decision"] == "more draws"


# ── PREREG106: the overfitting report (report only — the add rule's outputs stay byte for byte) ─────────────────

import hashlib  # noqa: E402
import json  # noqa: E402
from statistics import NormalDist  # noqa: E402

import pytest  # noqa: E402

from src.analysis.stats import expected_max_sharpe  # noqa: E402

# sha256 of json.dumps(verdict) + "\n" + summary(verdict) for each scenario below, computed with the add rule as it
# stood before PREREG106 (src/backtest/add_rule.py md5 fe4a21c1928a39b4bb305c8c591db49b, 2026-10-14 13:09 ET): every
# call without the new arguments must return exactly what it returned then
GOLDEN = {
    "more_draws": "1f5ad5f20c041da5ef1c1a2bfccd5104c2819979ec2a86e02a87201a12bf8337",
    "add": "0cbd5ec941560011543a44809d7d0e72a552dffb0f8855acae1ab857d010b36c",
    "futility": "2ccfc3486cf2f061c65d69df051f900d50a68baf0fc3cda8326f540fb413e2b4",
    "gain_each": "edd5d4f7e36dca727d909f54259744e037e1010c8f94aaf87b56d2528fa7e9e6",
    "similar": "0f7407daf9aeb32333fbfe045eb8292aa85de391d523515454a689de1d6439d0",
    "undecided20": "a2f8d7f845a2b4d0abbdf11247eed5af8450be53b2f8013bc88e9ccc83250bea",
    "decided40": "4133bb0ff58cf6b5621dc508fbf613e2d38f5856a1e088dd39b0581b4056461b",
    "calls_same": "28604efe1806f3bb2f17afd479f2af692825659b5367f5c2fae40007ce050830",
    "calls_more": "b76a7238780e95a267750f131b7dade550f34a234f470a8a4e03c41dde31e087",
    "drawdown": "08643bbd63b210ce9ff2166e36b91c84242bb2fdff6db8654c5636ad58d96387",
    "futility5": "46b27e771af32b2604a60b838a6b29896ea317594a73e7ab8d249745f9530fe5",
    "near5": "c29ee4f3f95745b9faa7a9d7ebdcb811928baaf83960f06245da5dedd7dec901",
    "missing_set": "80789cf03f0f42d81406f1626f2942370b1180321b5476a7750ff3bdd6550e7d",
    "one_draw": "0dfd695b07cfa0edda2b307336db621a3599e5b88b89ca6e04d425c9c47cb45f",
    "three_sets": "d1591b26bce23a4499af9d908124f12668677e75913a91500309001aa62eaf00",
}


def _golden_scenarios():
    """(name, live, variant, sets or None) for every path of `verdict`: more draws, add, futility, each criterion,
    the undecided band at 20 and its decision at 40, a missing set, a single draw, three sets."""
    L20 = {"A": _runs(20, 0.37), "B": _runs(20, 1.0)}
    L5 = {"A": _runs(5, 0.37), "B": _runs(5, 1.0)}
    L40 = {"A": _runs(20, 0.37), "B": _runs(40, 1.0)}
    Lc = {"A": _runs(20, 0.37), "B": _runs(20, 1.0, calls=(3, 7))}
    Ldd = {"A": _runs(20, 0.37, dd=15.0), "B": _runs(20, 1.0, dd=25.0)}
    L3 = {"A": _runs(20, 0.37), "B": _runs(20, 1.0), "A0": _runs(20, 0.19)}
    return [
        ("more_draws", L5, {k: _variant(v, 2.0) for k, v in L5.items()}, None),
        ("add", L20, {"A": _variant(L20["A"], 1.2, noise=0.3), "B": _variant(L20["B"], 1.6, noise=0.5)}, None),
        ("futility", L20, {"A": _variant(L20["A"], 3.0, noise=0.2), "B": _variant(L20["B"], -2.0, noise=0.2)}, None),
        ("gain_each", L20, {"A": _variant(L20["A"], 3.0, noise=0.2), "B": _variant(L20["B"], 0.0, noise=0.9)}, None),
        ("similar", L20, {"A": _variant(L20["A"], 4.0, noise=0.2), "B": _variant(L20["B"], 1.0, noise=0.1)}, None),
        ("undecided20", L20, {"A": _variant(L20["A"], 1.0, noise=0.2), "B": _variant(L20["B"], 0.6, noise=3.0)}, None),
        ("decided40", L40, {"A": _variant(L40["A"], 1.0, noise=0.2), "B": _variant(L40["B"], 0.6, noise=3.0)}, None),
        ("calls_same", Lc, {"A": _variant(Lc["A"], 1.2, noise=0.2), "B": _variant(Lc["B"], 1.2, noise=0.2)}, None),
        ("calls_more", Lc, {"A": _variant(Lc["A"], 1.2, noise=0.2),
                            "B": _variant(Lc["B"], 1.2, noise=0.2, calls=(3, 7, 11))}, None),
        ("drawdown", Ldd, {"A": _variant(Ldd["A"], 1.2, noise=0.2, dd=16.5),
                           "B": _variant(Ldd["B"], 1.2, noise=0.2, dd=25.5)}, None),
        ("futility5", L5, {"A": _variant(L5["A"], 3.5, noise=0.25), "B": _variant(L5["B"], -2.0, noise=1.6)}, None),
        ("near5", L5, {"A": _variant(L5["A"], 3.5, noise=0.25), "B": _variant(L5["B"], 0.4, noise=1.3)}, None),
        ("missing_set", L20, {"A": _variant(L20["A"], 1.2, noise=0.3)}, None),
        ("one_draw", {"A": _runs(1, 0.37), "B": _runs(1, 1.0)},
         {"A": _variant(_runs(1, 0.37), 1.0), "B": _variant(_runs(1, 1.0), 1.0)}, None),
        ("three_sets", L3, {k: _variant(v, 1.2, noise=0.3) for k, v in L3.items()}, ("A", "B", "A0")),
    ]


def _fingerprint(live, var, sets):
    v = R.verdict(live, var) if sets is None else R.verdict(live, var, sets)
    return hashlib.sha256((json.dumps(v) + "\n" + R.summary(v)).encode("utf-8")).hexdigest()


def test_every_existing_call_is_byte_identical_without_the_new_arguments():
    names = [n for n, *_ in _golden_scenarios()]
    assert sorted(names) == sorted(GOLDEN)
    for name, live, var, sets in _golden_scenarios():
        assert _fingerprint(live, var, sets) == GOLDEN[name], name
        v = R.verdict(live, var) if sets is None else R.verdict(live, var, sets=sets)
        assert list(v) == ["sets", "criteria", "undecided", "decision", "why"]           # no new key


@pytest.mark.parametrize("name", list(GOLDEN))
def test_the_report_never_changes_the_decision(name):
    live, var, sets = next((l, v, s) for n, l, v, s in _golden_scenarios() if n == name)
    kw = {} if sets is None else {"sets": sets}
    base = R.verdict(live, var, **kw)
    full = R.verdict(live, var, n_trials=(50, 5000), kind="rule", **kw)
    assert list(full) == list(base) + ["overfitting"]
    assert json.dumps({k: full[k] for k in base}) == json.dumps(base)                   # every add-rule field as before
    assert R.summary(full).startswith(R.summary(base)) and "overfitting report" in R.summary(full)


def test_the_deflated_probability_is_the_stated_formula():
    nd = NormalDist()
    d = R.deflated_gain(1.0, 0.4, 50)
    z0 = expected_max_sharpe(50, 1.0)                      # (1 - g) Phi^-1(1 - 1/50) + g Phi^-1(1 - 1/(50 e))
    g = 0.5772156649015329
    assert z0 == pytest.approx((1 - g) * nd.inv_cdf(1 - 1 / 50) + g * nd.inv_cdf(1 - 1 / (50 * math.e)))
    assert d["z"] == pytest.approx(2.5) and d["z0"] == pytest.approx(z0)
    assert d["p"] == pytest.approx(nd.cdf(2.5 - z0)) and d["p_raw"] == pytest.approx(nd.cdf(2.5))
    # in the verdict: per set z = the paired gain over its standard error
    live = {"A": _runs(20, 0.37), "B": _runs(20, 1.0)}
    var = {"A": _variant(live["A"], 1.2, noise=0.3), "B": _variant(live["B"], 0.6, noise=2.0)}
    v = R.verdict(live, var, n_trials=50)
    for s in ("A", "B"):
        p, o = v["sets"][s], v["overfitting"]["sets"][s]
        assert o["z"] == pytest.approx(p["gain"] / p["se"])
        assert o["deflated"]["low"]["p"] == pytest.approx(nd.cdf(p["gain"] / p["se"] - z0))
        assert o["deflated"]["low"]["N"] == o["deflated"]["high"]["N"] == 50


def test_one_trial_is_the_undeflated_probability_and_more_trials_deflate_more():
    one = R.deflated_gain(0.8, 0.5, 1)
    assert one["z0"] == 0.0 and one["p"] == pytest.approx(one["p_raw"])
    ps = [R.deflated_gain(0.8, 0.5, n)["p"] for n in (1, 4, 50, 753, 80141)]
    assert all(a > b for a, b in zip(ps, ps[1:]))


def test_the_retention_shrink_by_kind():
    live = {"A": _runs(20, 0.37), "B": _runs(20, 1.0)}
    var = {"A": _variant(live["A"], 2.0, noise=0.3), "B": _variant(live["B"], -0.4, noise=0.3)}
    o = R.verdict(live, var, kind="rule")["overfitting"]
    a, b = o["sets"]["A"], o["sets"]["B"]
    assert a["shrunk"]["gain"] == pytest.approx(0.15 * a["gain"]) and a["shrunk"]["clears_min_gain"] is False
    assert a["shrunk"]["band"] == pytest.approx([0.09 * a["gain"], 0.23 * a["gain"]])
    assert b["shrunk"]["band"] == pytest.approx([0.23 * b["gain"], 0.09 * b["gain"]])      # a loss: low .. high
    assert "deflated" not in a and o["n_trials"] is None                                   # kind alone: no N
    s = R.verdict(live, var, kind="sizing")["overfitting"]["sets"]["A"]["shrunk"]
    assert s["gain"] == pytest.approx(0.5 * a["gain"]) and s["band"] == pytest.approx([0.47 * a["gain"], 0.5 * a["gain"]])
    assert s["clears_min_gain"] is True
    with pytest.raises(ValueError, match="no measured retention"):
        R.verdict(live, var, kind="mechanism")


def test_model_reads_the_trial_ledger(tmp_path):
    from src.backtest import trials as TR
    led = tmp_path / "ledger.jsonl"
    src = tmp_path / "global103.json"
    src.write_text(json.dumps({"trials": {"counts": {"fall": {"low": 55, "high": 65},
                                                     "combination": {"low": 1797, "high": 84049}}}}),
                   encoding="utf-8")
    TR.seed_from_prereg103(src, led)
    TR.record("PREREG104", "fall", "F3", ["A", "B"], {}, None, "rule", path=led)
    live = {"A": _runs(20, 0.37), "B": _runs(20, 1.0)}
    var = {"A": _variant(live["A"], 1.2, noise=0.3), "B": _variant(live["B"], 1.6, noise=0.5)}
    o = R.verdict(live, var, model="fall", kind="rule", ledger=led)["overfitting"]
    assert (o["n_trials"]["low"], o["n_trials"]["high"]) == (56, 66)
    assert "seed 55 / 65 + 1 recorded" in o["n_trials"]["source"]
    assert o["sets"]["A"]["deflated"]["high"]["N"] == 66
    c = R.verdict(live, var, model="combination", ledger=led)["overfitting"]["n_trials"]
    assert (c["low"], c["high"]) == (1797, 84050)                      # the high count takes every model's trial
    assert R.verdict(live, var, model="never_seen", ledger=led)["overfitting"]["n_trials"]["low"] == 1


def test_the_new_arguments_are_checked():
    import numpy as np
    live = {"A": _runs(20, 0.37), "B": _runs(20, 1.0)}
    var = {"A": _variant(live["A"], 1.2, noise=0.3), "B": _variant(live["B"], 1.6, noise=0.5)}
    for bad in (0, (5, 3), True, (1.5, 2), "10"):
        with pytest.raises(ValueError, match="n_trials"):
            R.verdict(live, var, n_trials=bad)
    with pytest.raises(ValueError, match="not both"):
        R.verdict(live, var, n_trials=5, model="vol")
    assert R.verdict(live, var, n_trials=np.int64(7))["overfitting"]["n_trials"]["low"] == 7


def test_a_set_without_draws_or_with_one_draw_reports_no_probability():
    sc = {n: (l, v) for n, l, v, _ in _golden_scenarios()}
    o = R.verdict(*sc["missing_set"], n_trials=10, kind="rule")["overfitting"]["sets"]
    assert o["B"] == {"n": 0} and o["A"]["z"] is not None
    o1 = R.verdict(*sc["one_draw"], n_trials=10)["overfitting"]["sets"]["A"]
    assert o1["z"] is None and o1["deflated"]["low"]["p"] is None and o1["p_raw"] is None
    assert "p n/a" in R.summary(R.verdict(*sc["one_draw"], n_trials=10))


# ── the POOLED rule (user directive 2026-10-14: "check for the full dataset (A0+A+B)") ─────────────────────────

from src.backtest import periods  # noqa: E402

YEARS = {"A0": ("2018", "2019", "2020", "2021"), "A": ("2021", "2022", "2023", "2024"), "B": ("2024", "2025", "2026")}


def _pruns(n, growth, dd=20.0, calls=(), year_lp=None, st="A"):
    """Live records of set ``st`` with per-year log growth (each year of the set the same share unless given)."""
    out = _runs(n, growth, dd, calls)
    for s, r in out.items():
        yl = year_lp or {y: 100 * math.log1p(growth) / len(YEARS[st]) for y in YEARS[st]}
        r["years"] = dict(yl)
    return out


def _pvariant(live, gain_lp, noise=0.0, dd=None, calls=(), year_gain=None):
    """``_variant`` + its per-year records: the gain spread evenly over the set's years, or ``year_gain`` {year: lp}."""
    out = _variant(live, gain_lp, noise, dd, calls)
    for s, r in out.items():
        base = live[s]["years"]
        if year_gain is None:
            share = gain_lp / len(base)
            r["years"] = {y: v + share for y, v in base.items()}
        else:
            r["years"] = {y: v + year_gain.get(y, 0.0) for y, v in base.items()}
    return out


def _pool(n=20, calls=None, dd=20.0):
    calls = calls or {}
    return {st: _pruns(n, g, dd=dd, calls=calls.get(st, ()), st=st) for st, g in (("A0", 0.19), ("A", 0.38), ("B", 1.0))}


def test_the_pooled_gain_is_the_years_weighted_mean_and_no_set_must_gain_alone():
    live = _pool()
    var = {"A0": _pvariant(live["A0"], 3.0, noise=0.2), "A": _pvariant(live["A"], -0.6, noise=0.2),
           "B": _pvariant(live["B"], 1.5, noise=0.3)}
    v = R.pooled_verdict(live, var)
    w = periods.pool_weights()
    assert abs(v["pooled"]["gain"] - (3.0 * w["A0"] - 0.6 * w["A"] + 1.5 * w["B"])) < 1e-9
    assert v["decision"] == "add", v["why"]                  # set A loses alone: the pool decides (old rule: reject)
    old = R.verdict({k: live[k] for k in ("A", "B")}, {k: var[k] for k in ("A", "B")})
    assert old["decision"] == "reject"
    assert "pooled" in R.pooled_summary(v) and "A -0.60" in R.pooled_summary(v)


def test_a_pooled_gain_one_year_carries_is_rejected():
    live = _pool()
    var = {st: _pvariant(live[st], 0.0) for st in live}
    for st in var:                                            # the whole gain sits in 2020 (set A0)
        for s, r in var[st].items():
            g = 100 * math.log1p(r["growth"]) + (6.0 if st == "A0" else -0.3)
            r["growth"] = math.expm1(g / 100)
            r["years"] = {y: v + (18.0 if y == "2020" else 0.0) - (0.3 / len(r["years"]) * 3 if st != "A0" else 0)
                          for y, v in live[st][s]["years"].items()}
    v = R.pooled_verdict(live, var)
    assert v["pooled"]["gain"] >= R.MIN_GAIN and v["criteria"]["not_one_year"] is False
    assert v["decision"] == "reject" and "2020" in v["why"][0]


def test_pooled_margin_calls_are_summed_and_the_drawdown_is_the_worst_set_per_seed():
    live = _pool(calls={"B": (1, 2)})
    shift = {"A0": _pvariant(live["A0"], 1.0, noise=0.1, calls=(5,)),
             "A": _pvariant(live["A"], 1.0, noise=0.1),
             "B": _pvariant(live["B"], 1.0, noise=0.1, calls=(1,))}
    shift["B"][2]["calls"] = []                                           # a call moves from B to A0: 2 -> 2
    v = R.pooled_verdict(live, shift)
    assert v["pooled"]["calls_live"] == 2 and v["pooled"]["calls_variant"] == 2 and v["decision"] == "add"
    more = dict(shift, B=_pvariant(live["B"], 1.0, noise=0.1, calls=(1, 2)))
    assert R.pooled_verdict(live, more)["decision"] == "reject"
    deep = dict(shift, B=_pvariant(live["B"], 1.0, noise=0.1, calls=(1,), dd=21.5))
    deep["B"][2]["calls"] = []
    x = R.pooled_verdict(live, deep)
    assert x["pooled"]["dd_variant"] == 21.5 and x["decision"] == "reject" and x["criteria"]["drawdown"] is False


def test_pooled_records_without_years_cannot_be_added_and_risk_protection_is_flagged():
    live = {st: _runs(20, g) for st, g in (("A0", 0.19), ("A", 0.38), ("B", 1.0))}
    var = {st: _variant(live[st], 1.2, noise=0.2) for st in live}
    v = R.pooled_verdict(live, var)
    assert v["decision"] == "needs per-year growth" and v["criteria"]["not_one_year"] is None
    livec = _pool(calls={"A0": (0, 1, 2, 3)})
    safer = {st: _pvariant(livec[st], -0.2, noise=0.2) for st in livec}
    for s in safer["A0"].values():
        s["calls"] = []
    r = R.pooled_verdict(livec, safer)
    assert r["decision"] == "reject" and r["risk_protection"] and "RISK PROTECTION" in r["why"][-1]


def test_pooled_draws_futility_and_the_undecided_band():
    live5 = {st: _pruns(5, g, st=st) for st, g in (("A0", 0.19), ("A", 0.38), ("B", 1.0))}
    bad = {st: _pvariant(live5[st], -2.0, noise=0.3) for st in live5}
    v = R.pooled_verdict(live5, bad)
    assert v["decision"] == "reject" and "futility" in v["why"][0]
    assert R.pooled_verdict(live5, {st: _pvariant(live5[st], 2.0, noise=0.3) for st in live5})["decision"] == "more draws"
    live = _pool()
    near = {st: _pvariant(live[st], 0.6, noise=2.5) for st in live}
    u = R.pooled_verdict(live, near)
    assert u["decision"] == "more draws" and set(u["undecided"]) == {"A0", "A", "B"}


def test_year_lp_reads_each_calendar_year_off_a_daily_nav():
    days = ["2018-01-02", "2018-12-31", "2019-06-03", "2019-12-31", "2020-01-02"]
    vals = [1.0, 1.10, 1.20, 1.21, 1.331]
    y = R.year_lp(days, vals)
    assert abs(y["2018"] - 100 * math.log(1.10)) < 1e-9 and abs(y["2019"] - 100 * math.log(1.21 / 1.10)) < 1e-9
    assert abs(y["2020"] - 100 * math.log(1.331 / 1.21)) < 1e-9
