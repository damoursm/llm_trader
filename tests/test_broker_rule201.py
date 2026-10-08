"""Rule 201 in the broker (user 2026-10-05: "The broker doesn't handle Rule 201 when the short-sale
restriction is on" — fix it).

What must hold: a short ENTRY under the short-sale price test — recorded at the pick (``sel_ssr``) or
triggered since (today's low >= 10% under the previous close) — is a limit one tick above the national
best bid (the lowest price the rule allows; the model price without a bid), stamped ``broker_ssr``, and
the settle pass leaves it resting instead of killing it; a short outside the test and every buy keep the
marketable limit; a refused restricted short is resent at the bid + 1 tick again.
"""
from datetime import datetime, timedelta, timezone

import pytest

import src.broker.reconcile as rec
from config.settings import settings
from src.broker.base import Broker, OrderResult
from tests.test_broker_reconcile import FakeBroker, repo_store  # noqa: F401  (fixture)


def test_the_rule201_price_is_one_tick_above_the_bid():
    assert rec._rule201_limit(10.00, 10.2) == 10.01
    assert rec._rule201_limit(9.995, 10.2) == 10.01                    # rounded UP to the tick
    assert rec._rule201_limit(None, 10.2) == 10.2                      # no book: the model price
    assert rec._rule201_limit(0.5, 0.52) == 0.5001                     # sub-dollar tick
    assert rec._rule201_limit(10.0, 0.0) is None


def test_the_restriction_is_the_picks_or_a_trigger_since(monkeypatch):
    from src.data import polygon_client as pc
    assert rec._rule201_in_force({"ticker": "A", "sel_ssr": True}) == (True, "pick")
    monkeypatch.setattr(pc, "get_day_low_prev_close", lambda t: (8.9, 10.0))
    assert rec._rule201_in_force({"ticker": "A", "sel_ssr": False}) == (True, "live")
    monkeypatch.setattr(pc, "get_day_low_prev_close", lambda t: (9.5, 10.0))
    assert rec._rule201_in_force({"ticker": "A"}) == (False, "")
    monkeypatch.setattr(pc, "get_day_low_prev_close", lambda t: None)
    assert rec._rule201_in_force({"ticker": "A"}) == (False, "")


def _short(ticker, ssr):
    return {"ticker": ticker, "action": "SELL", "status": "OPEN", "entry_price": 10.2, "position_size_multiplier": 1.0,
            "recommendation_id": f"rec-{ticker}", "entry_mechanism": "sel_short", "sel_ssr": ssr}


def test_a_restricted_short_is_offered_one_tick_above_the_bid(repo_store, monkeypatch):  # noqa: F811
    from src.data import polygon_client as pc
    monkeypatch.setattr(settings, "enable_broker_rule201", True)
    monkeypatch.setattr(settings, "broker_order_type", "LMT")
    monkeypatch.setattr(rec, "_bid_now", lambda broker, t: 10.00)
    monkeypatch.setattr(pc, "get_day_low_prev_close", lambda t: (9.9, 10.0))          # no live trigger
    buy = dict(_short("BUYME", None), action="BUY")
    repo_store["trades"] = [_short("SSR", True), _short("FREE", False), buy]
    b = FakeBroker()
    rec.sync(broker=b)
    by = {o.ticker: o for o in b.orders}
    assert by["SSR"].order_type == "LMT" and by["SSR"].limit_price == pytest.approx(10.01)
    assert by["FREE"].limit_price < 10.2                               # the marketable cap below the model price
    t = {x["ticker"]: x for x in repo_store["trades"]}
    assert t["SSR"]["broker_ssr"] is True and t["SSR"]["broker_ssr_source"] == "pick"
    assert t["FREE"]["broker_ssr"] is False
    assert "broker_ssr" not in t["BUYME"]                               # a buy is never a short sale
    monkeypatch.setattr(settings, "enable_broker_rule201", False)
    repo_store["trades"] = [_short("OFF", True)]
    b2 = FakeBroker()
    rec.sync(broker=b2)
    assert b2.orders[0].limit_price < 10.2 and "broker_ssr" not in repo_store["trades"][0]


class _RestingBroker(Broker):
    """Orders rest unfilled; cancels are recorded."""
    name = "resting"

    def __init__(self):
        self.cancelled, self.requests = [], []

    def connect(self):
        return True

    def is_connected(self):
        return True

    def get_account(self):
        return None

    def get_positions(self):
        return []

    def get_fills(self):
        return []

    def submit_order(self, req):
        self.requests.append(req)
        return OrderResult(ok=True, ticker=req.ticker, side=req.side, requested_qty=req.quantity, filled_qty=0,
                           order_id=f"n{len(self.requests)}", client_ref=req.client_ref, status="Submitted")

    def cancel_order(self, client_ref):
        self.cancelled.append(client_ref)
        return True


def test_the_settle_pass_leaves_a_restricted_short_resting(monkeypatch):
    monkeypatch.setattr(settings, "broker_settle_seconds", 30)
    monkeypatch.setattr(settings, "broker_settle_poll_seconds", 3)
    monkeypatch.setattr(settings, "broker_settle_reanchor_every", 2)
    monkeypatch.setattr(rec, "_live_price", lambda _t: 10.0)
    clock = {"t": 0.0}
    monkeypatch.setattr(rec.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(rec.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    now = datetime.now(timezone.utc)
    legs = [dict(_short(tk, ssr), broker_order_id=f"o-{tk}", broker_client_ref=f"rec-{tk}", broker_status="Submitted",
                 broker_fill_qty=0, broker_requested_qty=10, broker_submitted_at=now.isoformat(), broker_ssr=ssr)
            for tk, ssr in (("SSR", True), ("FREE", False))]
    b = _RestingBroker()
    rec._settle_unfilled_this_tick(b, legs, rec._new_report(), False, now - timedelta(minutes=1))
    ssr, free = legs
    assert "rec-SSR" not in b.cancelled and ssr["broker_order_id"] == "o-SSR"   # resting until the next tick
    assert free["broker_status"] == "UNFILLED_KILLED"                         # the old rule for the rest
    assert all(r.ticker == "FREE" for r in b.requests)                       # only the free short re-anchored


def test_a_refused_restricted_short_is_resent_one_tick_above_the_bid(monkeypatch):
    monkeypatch.setattr(settings, "broker_refused_resends_per_tick", 1)
    monkeypatch.setattr(rec, "_live_price", lambda _t: 10.2)
    monkeypatch.setattr(rec, "_bid_now", lambda broker, t: 10.05)
    b = _RestingBroker()
    t = dict(_short("SSR", True), broker_ssr=True)
    req = rec.OrderRequest(ticker="SSR", side="SELL", quantity=10, order_type="LMT", limit_price=10.01,
                           client_ref="rec-SSR", intent="ENTRY")
    refused = OrderResult(ok=False, ticker="SSR", side="SELL", requested_qty=10, filled_qty=0, order_id="x",
                          client_ref="rec-SSR", status="Inactive", error="refused")
    monkeypatch.setattr(rec, "_is_refusal", lambda res: res.status == "Inactive")
    req2, res2, _ = rec._resend_refused(b, t, "broker_", "ENTRY", req, refused, 10.2, False, rec._new_report(), set())
    assert req2.limit_price == pytest.approx(10.06) and req2.order_type == "LMT"
