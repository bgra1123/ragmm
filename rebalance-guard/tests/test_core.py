import math

import numpy as np
import pandas as pd
import pytest

from rebalance_guard import core
from rebalance_guard.core import RebalanceError


@pytest.fixture(scope="module")
def demo():
    px = core.load_prices("demo")
    tickers = list(px.columns)
    r = core.daily_returns(px, tickers, 756)
    return tickers, r.to_numpy(), core.monthly_scenarios(r.to_numpy())


# ---------------------------------------------------------------- risk measures
def test_risk_measure_ordering_and_gaussian_closed_form():
    loss = np.random.default_rng(0).normal(0.01, 0.05, 200_000)
    var, cv, ev = core.value_at_risk(loss), core.cvar(loss), core.evar(loss)
    assert var < cv < ev
    # Gaussian: EV@R_{1-zeta} = mean + sigma * sqrt(2 log(1/zeta)) exactly
    assert ev == pytest.approx(0.01 + 0.05 * math.sqrt(2 * math.log(20)), rel=0.01)


def test_evar_increases_as_zeta_decreases_and_is_at_least_cvar():
    loss = -core.monthly_scenarios(core.load_prices("demo").pct_change().dropna().iloc[-756:].to_numpy())[:, :3].mean(1)
    assert core.evar(loss, 0.01) > core.evar(loss, 0.05) > core.evar(loss, 0.25)
    for z in (0.01, 0.05, 0.1):
        assert core.evar(loss, z) >= core.cvar(loss, z) >= core.value_at_risk(loss, z)


def test_ledoit_wolf_matches_scikit_learn():
    covariance = pytest.importorskip("sklearn.covariance")
    rng = np.random.default_rng(0)
    x = rng.standard_normal((400, 8)) @ rng.standard_normal((8, 8))
    np.testing.assert_allclose(core.ledoit_wolf(x), covariance.LedoitWolf().fit(x).covariance_, atol=1e-10)


def test_monthly_scenarios_compounding_and_stride():
    r = np.full((100, 2), 0.01)
    s = core.monthly_scenarios(r, 21)
    assert s.shape == (80, 2)
    np.testing.assert_allclose(s, 1.01 ** 21 - 1)
    assert core.monthly_scenarios(r, 21, stride=2).shape == (40, 2)


def test_max_drawdown():
    assert core.max_drawdown(np.array([0.1, -0.5, 0.2])) == pytest.approx(0.5)
    assert core.max_drawdown(np.array([0.01, 0.02])) == 0.0


# ---------------------------------------------------------------- data loading and validation
def test_demo_prices_load():
    px = core.load_prices("demo")
    assert px.shape[1] == 10 and px.index.is_monotonic_increasing and (px > 0).all().all()


def test_price_file_validation(tmp_path, monkeypatch):
    good = tmp_path / "p.csv"
    pd.DataFrame({"A": [1.0, 2, 3], "B": [1.0, 1, 2]}, index=pd.date_range("2020-01-01", periods=3, name="date")).to_csv(good)
    assert core.load_prices(str(good)).shape == (3, 2)
    with pytest.raises(RebalanceError, match="csv"):
        core.load_prices(str(tmp_path / "p.txt"))
    with pytest.raises(RebalanceError, match="not found"):
        core.load_prices(str(tmp_path / "missing.csv"))
    bad = tmp_path / "neg.csv"
    pd.DataFrame({"A": [1.0, -2], "B": [1.0, 1]}, index=pd.date_range("2020-01-01", periods=2)).to_csv(bad)
    with pytest.raises(RebalanceError, match="positive"):
        core.load_prices(str(bad))
    one = tmp_path / "one.csv"
    pd.DataFrame({"A": [1.0, 2]}, index=pd.date_range("2020-01-01", periods=2)).to_csv(one)
    with pytest.raises(RebalanceError, match="two"):
        core.load_prices(str(one))


def test_data_dir_restriction(tmp_path, monkeypatch):
    inside, outside = tmp_path / "ok", tmp_path / "other"
    inside.mkdir(), outside.mkdir()
    frame = pd.DataFrame({"A": [1.0, 2, 3], "B": [1.0, 1, 2]}, index=pd.date_range("2020-01-01", periods=3))
    frame.to_csv(inside / "p.csv"), frame.to_csv(outside / "p.csv")
    monkeypatch.setenv(core.DATA_DIR_ENV, str(inside))
    assert core.load_prices(str(inside / "p.csv")).shape[1] == 2
    with pytest.raises(RebalanceError, match=core.DATA_DIR_ENV):
        core.load_prices(str(outside / "p.csv"))
    with pytest.raises(RebalanceError, match=core.DATA_DIR_ENV):
        core.load_prices(str(inside / ".." / "other" / "p.csv"))          # path traversal


@pytest.mark.parametrize("holdings", [{"A": 1}, {"A": 1, "B": -1}, {"A": 0, "B": 0}, {"A": float("nan"), "B": 1}, [1, 2]])
def test_bad_holdings_are_rejected(holdings):
    with pytest.raises(RebalanceError):
        core.holdings_to_weights(holdings)


def test_unknown_ticker_and_short_history_are_rejected():
    px = core.load_prices("demo")
    with pytest.raises(RebalanceError, match="no price history"):
        core.daily_returns(px, ["AAPL", "NOPE"], 756)
    with pytest.raises(RebalanceError, match="lookback_days"):
        core.daily_returns(px, ["AAPL", "XOM"], 10)


# ---------------------------------------------------------------- optimizers
def _evar_of(w, scen, zeta=0.05, mean_weight=0.0):
    sc = scen - (1 - mean_weight) * scen.mean(0)
    return core.evar(-(sc @ w), zeta)


def test_min_evar_is_feasible_and_beats_alternatives(demo):
    tickers, rd, scen = demo
    n = len(tickers)
    w0 = np.ones(n) / n
    w = core.min_evar_weights(w0, scen, zeta=0.05, cap=0.4, cost_bps=0.0)
    assert w.sum() == pytest.approx(1.0) and (w >= -1e-9).all() and (w <= 0.4 + 1e-6).all()
    best = _evar_of(w, scen)
    assert best <= _evar_of(w0, scen)
    rng = np.random.default_rng(0)
    for _ in range(300):                                           # no random feasible portfolio is better
        x = rng.dirichlet(np.ones(n))
        if (x <= 0.4).all():
            assert best <= _evar_of(x, scen) + 1e-4


def test_min_evar_matches_an_independent_local_optimizer(demo):
    """The exp-cone solution should agree with a direct minimization of the empirical EV@R (scipy SLSQP)."""
    from scipy.optimize import minimize
    tickers, rd, scen = demo
    n = len(tickers)
    w = core.min_evar_weights(np.ones(n) / n, scen, zeta=0.05, cap=1.0, cost_bps=0.0)
    sc = scen - scen.mean(0)
    f = lambda x: core.evar(-(sc @ x), 0.05)                      # noqa: E731
    res = minimize(f, w, method="SLSQP", bounds=[(0, 1)] * n, constraints={"type": "eq", "fun": lambda x: x.sum() - 1},
                   options={"maxiter": 300, "ftol": 1e-12})
    assert f(w) <= res.fun + 2e-4                                  # exact solver is at least as good


def test_cost_reduces_turnover_and_turnover_cap_is_respected(demo):
    tickers, rd, scen = demo
    n = len(tickers)
    w0 = np.ones(n) / n
    turnovers = [0.5 * np.abs(core.min_evar_weights(w0, scen, zeta=0.05, cap=1.0, cost_bps=c) - w0).sum()
                 for c in (0, 10, 100, 500)]
    assert turnovers == sorted(turnovers, reverse=True) and turnovers[0] > turnovers[-1]
    w = core.min_evar_weights(w0, scen, zeta=0.05, cap=1.0, cost_bps=0.0, turnover_cap=0.1)
    assert 0.5 * np.abs(w - w0).sum() <= 0.1 + 1e-6


def test_min_evar_ignores_expected_returns_unless_asked(demo):
    tickers, rd, scen = demo
    n = len(tickers)
    w0 = np.ones(n) / n
    shifted = scen + np.linspace(0, 0.05, n)                       # a large return tilt towards the last assets
    a = core.min_evar_weights(w0, scen, zeta=0.05, cap=1.0, cost_bps=0)
    b = core.min_evar_weights(w0, shifted, zeta=0.05, cap=1.0, cost_bps=0)
    np.testing.assert_allclose(a, b, atol=2e-3)                    # mean_weight = 0: shift-invariant
    c = core.min_evar_weights(w0, shifted, zeta=0.05, cap=1.0, cost_bps=0, mean_weight=1.0)
    assert c[-1] > b[-1] + 0.05                                    # mean_weight = 1: tilts towards higher returns


def test_min_vol_recovers_the_inverse_variance_solution():
    rng = np.random.default_rng(0)
    scen = rng.standard_normal((20_000, 2)) * np.array([0.02, 0.04])        # uncorrelated, vol ratio 2
    w = core.min_vol_weights(np.array([0.5, 0.5]), scen, cap=1.0, cost_bps=0)
    assert w[0] == pytest.approx(0.8, abs=0.02)                               # w1 = s2^2 / (s1^2 + s2^2)


def test_target_rebalance_band_logic():
    b = np.array([0.4, 0.3, 0.2, 0.1])
    same, traded = core.target_rebalance_weights(b + np.array([0.02, -0.02, 0, 0]), b, trigger_band=0.05,
                                                 band_type="absolute", landing_band=0.025, cap=1.0)
    assert not traded
    w0 = np.array([0.50, 0.25, 0.15, 0.10])                                  # first holding is 10 pp over, band is 5 pp
    w, traded = core.target_rebalance_weights(w0, b, trigger_band=0.05, band_type="absolute", landing_band=0.025, cap=1.0)
    assert traded and w.sum() == pytest.approx(1.0) and (w >= -1e-9).all()
    assert (np.abs(w - b) <= 0.025 + 1e-6).all()
    full, _ = core.target_rebalance_weights(w0, b, trigger_band=0.05, band_type="absolute", landing_band=0.0, cap=1.0)
    np.testing.assert_allclose(full, b, atol=1e-6)
    assert np.abs(w - w0).sum() < np.abs(full - w0).sum()                     # wider landing band => fewer trades
    wr, traded_rel = core.target_rebalance_weights(w0, b, trigger_band=0.5, band_type="relative", landing_band=0.25, cap=1.0)
    assert not traded_rel                                                     # 10pp on 40% is 25% relative < 50%


# ---------------------------------------------------------------- stability
def test_bootstrap_is_reproducible_and_stays_on_the_simplex(demo):
    tickers, rd, scen = demo
    n = len(tickers)
    w0 = np.ones(n) / n
    solve = lambda s: core.min_vol_weights(w0, s, cap=0.5, cost_bps=10)       # noqa: E731
    a = core.bootstrap_weights(rd, solve, n_boot=6, horizon=21, seed=3)
    b = core.bootstrap_weights(rd, solve, n_boot=6, horizon=21, seed=3)
    c = core.bootstrap_weights(rd, solve, n_boot=6, horizon=21, seed=4)
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)
    assert np.allclose(a.sum(1), 1.0) and (a <= 0.5 + 1e-6).all()
    st = core.stability_summary(a, solve(scen), w0, tickers)
    assert st["bagged_weights"].sum() == pytest.approx(1.0) and (st["bagged_weights"] <= 0.5 + 1e-6).all()
    assert 0 <= st["turnover_noise_median"] <= st["turnover_noise_p95"] <= 1


def test_stability_summary_hand_computed():
    """bagged = mean of the resampled solutions; direction agreement and turnover noise computed by hand."""
    W = np.array([[0.5, 0.5], [0.7, 0.3], [0.6, 0.4]])
    w0, w_point = np.array([0.55, 0.45]), np.array([0.6, 0.4])
    st = core.stability_summary(W, w_point, w0, ["A", "B"])
    np.testing.assert_allclose(st["bagged_weights"], [0.6, 0.4])
    np.testing.assert_allclose(st["weight_std_by_asset"], W.std(0))
    # bagged trade is +5pp in A (resamples: -5, +15, +5 -> 2 of 3 agree) and -5pp in B (+5, -15, -5 -> 2 of 3 agree)
    np.testing.assert_allclose(st["trade_direction_agreement"], [2 / 3, 2 / 3])
    # 0.5 * L1 distance of each resample to the point solution: 0.1, 0.1, 0
    assert st["turnover_noise_median"] == pytest.approx(0.1)
    assert st["turnover_noise_p95"] == pytest.approx(0.1, abs=1e-9)
    # a negligible bagged trade counts as full agreement
    tiny = core.stability_summary(np.array([[0.55, 0.45], [0.5501, 0.4499]]), w0, w0, ["A", "B"])
    np.testing.assert_allclose(tiny["trade_direction_agreement"], [1.0, 1.0])
