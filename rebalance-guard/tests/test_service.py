import json

import numpy as np
import pytest

from rebalance_guard import service
from rebalance_guard.core import RebalanceError

H = {"AAPL": 20000, "GE": 15000, "AMD": 5000, "WMT": 10000, "BAC": 15000, "T": 10000, "XOM": 10000,
     "BBY": 5000, "PFE": 5000, "JPM": 5000}


def test_describe_demo_data():
    d = service.describe_demo_data()
    assert len(d["tickers"]) == 10 and "survivorship" in d["note"] and d["disclaimer"]


def test_risk_report_invariants():
    r = service.risk_report(H)
    json.dumps(r)
    k = r["risk"]
    assert k["monthly_var"] <= k["monthly_cvar"] <= k["monthly_evar"] <= k["worst_month"] + 1e-9
    assert sum(h["weight"] for h in r["holdings"]) == pytest.approx(1.0, abs=1e-3)
    assert sum(h["share_of_risk"] for h in r["holdings"]) == pytest.approx(1.0, abs=1e-2)
    assert r["portfolio_value"] == 100000
    ends = [p["window_end"] for p in r["worst_periods"]]
    assert len(ends) == len(set(ends)) and [p["loss"] for p in r["worst_periods"]] == sorted(
        [p["loss"] for p in r["worst_periods"]], reverse=True)
    assert "not investment" in r["disclaimer"]


def test_propose_min_evar_structure_and_invariants():
    p = service.propose_rebalance(H, stability_samples=8)
    json.dumps(p)
    assert [x["name"] for x in p["proposals"]] == ["point_estimate", "bagged"] and p["recommended"] == "bagged"
    for prop in p["proposals"]:
        assert sum(t["trade_value"] for t in prop["trades"]) == pytest.approx(0.0, abs=1.0)   # self-financing (to the cent rounding)
        assert sum(t["proposed_weight"] for t in prop["trades"]) == pytest.approx(1.0, abs=1e-3)
        assert prop["estimated_cost"] >= 0 and 0 <= prop["turnover"] <= 1
    point = p["proposals"][0]
    assert point["risk_after"]["monthly_evar"] <= p["risk_before"]["monthly_evar"] + 1e-6   # in-sample improvement
    st = p["stability"]
    assert st["samples"] == 8 and 0 <= st["turnover_caused_by_data_noise_median"] <= 1
    assert all(0 <= a["trade_direction_agreement"] <= 1 for a in st["by_asset"])
    assert p["assumptions"] and "does not place orders" in p["disclaimer"]


def test_no_stability_samples_returns_point_only():
    p = service.propose_rebalance(H, stability_samples=0)
    assert [x["name"] for x in p["proposals"]] == ["point_estimate"] and p["stability"] is None


def test_max_weight_and_turnover_cap_flow_through():
    p = service.propose_rebalance(H, max_weight=0.2, turnover_cap=0.05, stability_samples=4, cost_bps=0)
    for prop in p["proposals"]:
        assert max(t["proposed_weight"] for t in prop["trades"]) <= 0.2 + 1e-3
        assert prop["turnover"] <= 0.05 + 1e-3


def test_target_mode():
    tw = {t: 0.1 for t in H}
    p = service.propose_rebalance(H, mode="target", target_weights=tw, trigger_band=0.05)
    assert p["recommended"] == "tolerance_band" and p["stability"] is None
    assert any("breached" in n for n in p["notes"]) and len(p["drift_vs_target"]) == 10
    inside = {t: 10000 for t in H}                                   # exactly on target: nothing to do
    q = service.propose_rebalance(inside, mode="target", target_weights=tw)
    assert q["proposals"][0]["turnover"] == 0 and any("No holding is outside" in n for n in q["notes"])
    scaled = service.propose_rebalance(H, mode="target", target_weights={t: 10 for t in H})
    assert any("rescaled" in n for n in scaled["notes"])


@pytest.mark.parametrize("kwargs,msg", [
    (dict(zeta=0.9), "zeta"), (dict(zeta=0.0), "zeta"), (dict(cost_bps=-1), "cost_bps"), (dict(max_weight=0.05), "max_weight"),
    (dict(max_weight=1.5), "max_weight"), (dict(turnover_cap=0), "turnover_cap"), (dict(mean_weight=2), "mean_weight"),
    (dict(stability_samples=1000), "stability_samples"), (dict(mode="trade_now"), "mode"), (dict(band_type="x"), "band_type"),
    (dict(lookback_days=5), "lookback_days"), (dict(prices="/nonexistent/x.csv"), "not found"),
])
def test_input_validation_messages(kwargs, msg):
    with pytest.raises(RebalanceError, match=msg):
        service.propose_rebalance(H, stability_samples=kwargs.pop("stability_samples", 2), **kwargs)


def test_target_mode_validation():
    with pytest.raises(RebalanceError, match="same tickers"):
        service.propose_rebalance(H, mode="target", target_weights={"AAPL": 1.0})
    with pytest.raises(RebalanceError, match="max_weight"):
        service.propose_rebalance(H, mode="target", target_weights={**{t: 0.05 for t in H}, "AAPL": 0.55}, max_weight=0.3)
    with pytest.raises(RebalanceError, match="landing_band"):
        service.propose_rebalance(H, mode="target", target_weights={t: 0.1 for t in H}, trigger_band=0.05, landing_band=0.1)
