"""
Numerics for Rebalance Guard (analysis only: nothing here places or prepares orders).

Risk is measured on *monthly* portfolio losses built from overlapping historical
``horizon_days`` returns. The worst-case measure is the entropic value at risk (EV@R),

    EV@R_{1-zeta}(L) = inf_{t>0} (1/t) log( E[exp(t L)] / zeta ),

the coherent risk measure used in Can & Gürbüzbalaban (2025). It is the tightest bound on the
expected loss in the worst ``zeta`` of cases that follows from the moment generating function, so
it is at least CVaR and at least VaR at the same level.

Rebalancing problems are convex and solved exactly with cvxpy (no stochastic solver is involved).
"""
from __future__ import annotations

import math
import os
import pathlib
import warnings
from typing import Callable

import cvxpy as cp
import numpy as np
import pandas as pd

DEMO_PATH = pathlib.Path(__file__).resolve().parent / "data" / "demo_prices.csv"
DATA_DIR_ENV = "REBALANCE_GUARD_DATA_DIR"
MAX_FILE_BYTES = 100 * 1024 * 1024
TRADING_DAYS = 252


class RebalanceError(ValueError):
    """A problem with the user's inputs, with a message meant to be shown to the user."""


# ---------------------------------------------------------------------- #
# Data                                                                     #
# ---------------------------------------------------------------------- #

def load_prices(source: str = "demo") -> pd.DataFrame:
    """
    Daily adjusted close prices, one column per ticker, dates as the index.

    ``source`` is ``"demo"`` (bundled: 10 US stocks, 1989-2018, survivorship-biased) or the path of a
    CSV file whose first column is the date. If the environment variable ``REBALANCE_GUARD_DATA_DIR``
    is set, only files inside that directory can be read.
    """
    if source == "demo":
        path = DEMO_PATH
    else:
        path = pathlib.Path(source).expanduser().resolve()
        allowed = os.environ.get(DATA_DIR_ENV)
        if allowed and not path.is_relative_to(pathlib.Path(allowed).expanduser().resolve()):
            raise RebalanceError(f"prices must be inside {DATA_DIR_ENV}={allowed}")
        if path.suffix.lower() != ".csv":
            raise RebalanceError("prices must be a .csv file (first column: date, other columns: tickers)")
        if not path.is_file():
            raise RebalanceError(f"file not found: {path}")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise RebalanceError("price file is larger than 100 MB")
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception as exc:                                   # noqa: BLE001
        raise RebalanceError(f"could not read the price file: {exc}") from exc
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RebalanceError("the first column of the price file must contain dates")
    df = df.sort_index().apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if df.shape[1] < 2:
        raise RebalanceError("need at least two price columns")
    if (df <= 0).any().any():
        raise RebalanceError("prices must be positive")
    return df


def holdings_to_weights(holdings: dict[str, float]) -> tuple[list[str], np.ndarray, float]:
    if not isinstance(holdings, dict) or len(holdings) < 2:
        raise RebalanceError("holdings must map at least two tickers to market values")
    tickers = list(holdings)
    values = np.array([float(holdings[t]) for t in tickers])
    if not np.all(np.isfinite(values)) or (values < 0).any():
        raise RebalanceError("holding values must be finite and non-negative")
    total = values.sum()
    if total <= 0:
        raise RebalanceError("the total portfolio value must be positive")
    return tickers, values / total, float(total)


def daily_returns(prices: pd.DataFrame, tickers: list[str], lookback_days: int) -> pd.DataFrame:
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise RebalanceError(f"no price history for: {missing}; available: {list(prices.columns)}")
    if not 63 <= lookback_days <= 2520:
        raise RebalanceError("lookback_days must be between 63 and 2520")
    p = prices[tickers].dropna()
    r = p.pct_change().dropna().iloc[-lookback_days:]
    if len(r) < max(63, lookback_days // 2):
        raise RebalanceError(f"only {len(r)} days of common history for these tickers; need more")
    return r


def monthly_scenarios(r_daily: np.ndarray, horizon: int = 21, stride: int = 1) -> np.ndarray:
    """Overlapping compounded ``horizon``-day returns, shape ((T - horizon + 1) / stride, n)."""
    if len(r_daily) <= horizon + 20:
        raise RebalanceError("not enough history for the chosen horizon")
    cum = np.vstack([np.ones(r_daily.shape[1]), np.cumprod(1.0 + r_daily, axis=0)])
    return (cum[horizon:] / cum[:-horizon] - 1.0)[::stride]


# ---------------------------------------------------------------------- #
# Risk measures (on losses L = -portfolio return)                          #
# ---------------------------------------------------------------------- #

def value_at_risk(loss: np.ndarray, zeta: float = 0.05) -> float:
    return float(np.quantile(loss, 1.0 - zeta))


def cvar(loss: np.ndarray, zeta: float = 0.05) -> float:
    loss = np.sort(np.asarray(loss))
    k = max(1, int(math.ceil(zeta * len(loss))))
    return float(loss[-k:].mean())


def evar(loss: np.ndarray, zeta: float = 0.05) -> float:
    """Empirical entropic value at risk, ``inf_t (log mean exp(t L) - log zeta) / t`` over a fine grid of t."""
    loss = np.asarray(loss, float)
    sd = max(loss.std(), 1e-12)
    ts = np.geomspace(0.05, 60.0, 400) / sd
    m = loss.max()
    log_mean_exp = np.log(np.mean(np.exp(np.outer(ts, loss - m)), axis=1)) + ts * m
    return float(((log_mean_exp - math.log(zeta)) / ts).min())


def ledoit_wolf(x: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf (2004) linear shrinkage of the sample covariance towards ``mu I``, ``mu = tr(S)/n``.
    ``x``: observations in rows (demeaned inside).
    """
    x = x - x.mean(0)
    t, n = x.shape
    s = x.T @ x / t
    mu = np.trace(s) / n
    d2 = np.sum((s - mu * np.eye(n)) ** 2) / n
    if d2 <= 0:
        return s
    # (1/(n t^2)) sum_k || x_k x_k^T - S ||_F^2  without forming the outer products
    q = np.einsum("ti,ij,tj->t", x, s, x)
    b2_bar = (np.sum(np.sum(x ** 2, axis=1) ** 2) - 2.0 * np.sum(q) + t * np.sum(s ** 2)) / (n * t ** 2)
    b2 = min(b2_bar, d2)
    return (b2 / d2) * mu * np.eye(n) + ((d2 - b2) / d2) * s


def max_drawdown(r_port_daily: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + r_port_daily)
    peak = np.maximum.accumulate(np.maximum(wealth, 1.0))
    return float((1.0 - wealth / peak).max())


def risk_summary(w: np.ndarray, r_daily: np.ndarray, scen: np.ndarray, zeta: float = 0.05) -> dict:
    w = np.asarray(w, float)
    loss = -(scen @ w)
    r_p = r_daily @ w
    cov = ledoit_wolf(r_daily)
    var_p = float(w @ cov @ w)
    contrib = w * (cov @ w) / var_p if var_p > 0 else np.zeros_like(w)
    return {
        "annual_volatility": float(r_p.std() * math.sqrt(TRADING_DAYS)),
        "monthly_var": value_at_risk(loss, zeta),
        "monthly_cvar": cvar(loss, zeta),
        "monthly_evar": evar(loss, zeta),
        "worst_month": float(loss.max()),
        "max_drawdown": max_drawdown(r_p),
        "effective_number_of_holdings": float(1.0 / np.sum(w ** 2)),
        "risk_contribution": contrib,
    }


# ---------------------------------------------------------------------- #
# Optimization (exact convex programs)                                     #
# ---------------------------------------------------------------------- #

def _finish(w: np.ndarray, cap: float) -> np.ndarray:
    w = np.clip(w, 0.0, cap)
    return w / w.sum()


INACCURATE_SOLVES = 0       # incremented whenever the best available answer was only "optimal_inaccurate"


def _solve(problem: cp.Problem) -> None:
    """Solve with CLARABEL, then SCS; prefer a fully optimal status and count inaccurate fallbacks."""
    global INACCURATE_SOLVES
    fallback = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for solver in (cp.CLARABEL, cp.SCS):
            try:
                problem.solve(solver=solver)
            except cp.SolverError:
                continue
            if problem.status == "optimal":
                return
            if problem.status == "optimal_inaccurate" and fallback is None:
                fallback = {v: v.value.copy() for v in problem.variables()}
    if fallback is not None:
        for v, val in fallback.items():
            v.value = val
        INACCURATE_SOLVES += 1
        return
    raise RebalanceError(f"the optimizer failed (status: {problem.status}); try a larger max_weight or cost")


def _common_constraints(w, w0, cap, turnover_cap):
    cons = [cp.sum(w) == 1, w >= 0, w <= cap]
    if turnover_cap is not None:
        cons.append(0.5 * cp.sum(cp.abs(w - w0)) <= turnover_cap)
    return cons


def min_evar_weights(w0, scen, *, zeta, cap, cost_bps, turnover_cap=None, mean_weight=0.0):
    """
    Minimize ``EV@R_{1-zeta}(monthly loss) + cost`` exactly, using the exponential-cone form

        min tau  s.t.  mean_s( z exp((L_s - tau)/z) ) <= zeta z,   z > 0,

    (Ahmadi-Javid & Pichler). ``mean_weight`` in [0, 1] keeps that share of the sample mean return in the
    scenarios: 0 (default) removes expected returns, the noisiest input, and optimizes dispersion and tails only.
    """
    n = scen.shape[1]
    scen = scen - (1.0 - mean_weight) * scen.mean(0)
    scale = max(float((-(scen @ np.ones(n) / n)).std()), 1e-6)
    s_count = scen.shape[0]
    w, tau, z, u = cp.Variable(n), cp.Variable(), cp.Variable(), cp.Variable(s_count)
    loss = -(scen / scale) @ w
    cost = cost_bps * 1e-4 * cp.sum(cp.abs(w - w0)) / scale
    cons = _common_constraints(w, w0, cap, turnover_cap) + [
        z >= 1e-6,
        cp.sum(u) <= s_count * zeta * z,
        cp.constraints.ExpCone(loss - tau, z * np.ones(s_count), u),
    ]
    _solve(cp.Problem(cp.Minimize(tau + cost), cons))
    return _finish(w.value, cap)


def min_vol_weights(w0, scen, *, cap, cost_bps, turnover_cap=None, **_):
    """Minimize monthly volatility (Ledoit-Wolf covariance) plus trading cost."""
    n = scen.shape[1]
    cov = ledoit_wolf(scen)
    chol = np.linalg.cholesky(cov + 1e-10 * np.eye(n))
    scale = max(float(math.sqrt(np.ones(n) @ cov @ np.ones(n)) / n), 1e-6)
    w = cp.Variable(n)
    obj = cp.norm(chol.T @ w) / scale + cost_bps * 1e-4 * cp.sum(cp.abs(w - w0)) / scale
    _solve(cp.Problem(cp.Minimize(obj), _common_constraints(w, w0, cap, turnover_cap)))
    return _finish(w.value, cap)


def target_rebalance_weights(w0, target, *, trigger_band, band_type, landing_band, cap, turnover_cap=None):
    """
    Tolerance-band rebalancing. If no holding is outside its band, do nothing. Otherwise return the smallest
    set of trades (an LP) that brings every holding within ``landing_band`` of its target.
    """
    target = np.asarray(target, float)
    trigger = trigger_band * (target if band_type == "relative" else np.ones_like(target))
    if not (np.abs(w0 - target) > trigger + 1e-12).any():
        return w0.copy(), False
    landing = landing_band * (target if band_type == "relative" else np.ones_like(target))
    n = len(w0)
    w = cp.Variable(n)
    miss = cp.sum(cp.pos(cp.abs(w - target) - landing))
    cons = _common_constraints(w, w0, cap, turnover_cap)
    _solve(cp.Problem(cp.Minimize(cp.sum(cp.abs(w - w0)) + 1e3 * miss), cons))
    return _finish(w.value, cap), True


# ---------------------------------------------------------------------- #
# Decision stability                                                       #
# ---------------------------------------------------------------------- #

def block_bootstrap(r_daily: np.ndarray, rng: np.random.Generator, block: int = 21) -> np.ndarray:
    t = len(r_daily)
    starts = rng.integers(0, t - block + 1, size=int(math.ceil(t / block)))
    return np.concatenate([r_daily[s:s + block] for s in starts])[:t]


def bootstrap_weights(r_daily, solve: Callable[[np.ndarray], np.ndarray], *, n_boot, horizon, seed, block=21, stride=1):
    """Re-solve on block-bootstrapped histories; returns an array (n_boot, n) of weights."""
    rng = np.random.default_rng(seed)
    return np.array([solve(monthly_scenarios(block_bootstrap(r_daily, rng, block), horizon, stride))
                     for _ in range(n_boot)])


def stability_summary(W: np.ndarray, w_point: np.ndarray, w0: np.ndarray, tickers: list[str]) -> dict:
    w_bag = W.mean(0)
    instability = 0.5 * np.abs(W - w_point).sum(1)
    trade_bag = w_bag - w0
    agree = []
    for i in range(len(tickers)):
        if abs(trade_bag[i]) < 1e-3:
            agree.append(1.0)
        else:
            agree.append(float(np.mean(np.sign(W[:, i] - w0[i]) == np.sign(trade_bag[i]))))
    return {
        "bagged_weights": w_bag,
        "weight_std_by_asset": W.std(0),
        "trade_direction_agreement": np.array(agree),
        "turnover_noise_median": float(np.median(instability)),
        "turnover_noise_p95": float(np.quantile(instability, 0.95)),
    }
