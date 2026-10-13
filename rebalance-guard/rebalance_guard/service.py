"""Tool logic for Rebalance Guard: plain functions returning JSON-serializable dicts (analysis only)."""
from __future__ import annotations

import math
from functools import partial

import numpy as np

from . import core
from .core import RebalanceError

DISCLAIMER = ("Educational analysis based on historical data, not investment, tax or legal advice. Past "
              "performance does not predict future results. This tool does not place orders.")

ASSUMPTIONS = [
    "Long-only, fully invested in the listed holdings; cash, taxes, lot selection, fractional-share limits "
    "and bid-ask spreads beyond the stated cost are not modeled.",
    "Risk is measured on overlapping historical monthly returns from the chosen lookback window.",
    "Expected returns are not forecast: optimization targets risk, not return, unless mean_weight > 0.",
]


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _validate(*, zeta, cost_bps, max_weight, n, turnover_cap, mean_weight, stability_samples):
    if not 0.0 < zeta <= 0.5:
        raise RebalanceError("zeta must be in (0, 0.5], e.g. 0.05 for the worst 5% of months")
    if not 0.0 <= cost_bps <= 500.0:
        raise RebalanceError("cost_bps must be between 0 and 500")
    if not 1.0 / n - 1e-12 <= max_weight <= 1.0:
        raise RebalanceError(f"max_weight must be between 1/{n} = {1 / n:.4f} and 1")
    if turnover_cap is not None and not 0.0 < turnover_cap <= 1.0:
        raise RebalanceError("turnover_cap must be in (0, 1]")
    if not 0.0 <= mean_weight <= 1.0:
        raise RebalanceError("mean_weight must be in [0, 1]")
    if not 0 <= stability_samples <= 100:
        raise RebalanceError("stability_samples must be between 0 and 100")


def _worst_months(loss, dates, horizon, k=5):
    """The k worst monthly losses that are at least one horizon apart (overlapping windows share crashes)."""
    order, chosen = np.argsort(-loss), []
    for i in order:
        if all(abs(int(i) - j) >= horizon for j in chosen):
            chosen.append(int(i))
        if len(chosen) == k:
            break
    return [{"window_end": str(dates[i + horizon - 1].date()), "loss": _r(loss[i])} for i in chosen]


def describe_demo_data() -> dict:
    px = core.load_prices("demo")
    return {
        "tickers": list(px.columns),
        "first_date": str(px.index[0].date()),
        "last_date": str(px.index[-1].date()),
        "note": ("Adjusted daily closes of 10 large US stocks chosen in 2018 (survivorship bias: absolute "
                 "returns are flattering). For demonstrations only; pass your own CSV as `prices` for real use."),
        "disclaimer": DISCLAIMER,
    }


def risk_report(holdings: dict, prices: str = "demo", lookback_days: int = 756, horizon_days: int = 21,
                zeta: float = 0.05) -> dict:
    tickers, w, total = core.holdings_to_weights(holdings)
    _validate(zeta=zeta, cost_bps=0, max_weight=1.0, n=len(tickers), turnover_cap=None, mean_weight=0,
              stability_samples=0)
    r = core.daily_returns(core.load_prices(prices), tickers, lookback_days)
    scen = core.monthly_scenarios(r.to_numpy(), horizon_days)
    rs = core.risk_summary(w, r.to_numpy(), scen, zeta)
    contrib = rs.pop("risk_contribution")
    return {
        "as_of": str(r.index[-1].date()),
        "window": f"{r.index[0].date()} to {r.index[-1].date()} ({len(r)} trading days)",
        "portfolio_value": _r(total, 2),
        "level": f"worst {zeta:.0%} of {horizon_days}-day periods",
        "risk": {k: _r(v) for k, v in rs.items()},
        "holdings": [{"ticker": t, "weight": _r(w[i]), "share_of_risk": _r(contrib[i])}
                     for i, t in enumerate(tickers)],
        "worst_periods": _worst_months(-(scen @ w), r.index, horizon_days),
        "how_to_read": ("monthly_evar >= monthly_cvar >= monthly_var: EV@R is the most conservative of the "
                        "three. share_of_risk is each holding's share of the portfolio's variance."),
        "disclaimer": DISCLAIMER,
    }


def _trades(tickers, w0, w1, total):
    rows = []
    for i, t in enumerate(tickers):
        d = w1[i] - w0[i]
        rows.append({"ticker": t, "current_weight": _r(w0[i]), "proposed_weight": _r(w1[i]),
                     "trade_weight": _r(d), "trade_value": _r(d * total, 2),
                     "action": "BUY" if d > 5e-4 else "SELL" if d < -5e-4 else "HOLD"})
    return rows


def _proposal(name, tickers, w0, w1, total, cost_bps, r, scen, zeta, risk_before):
    trade = np.abs(w1 - w0)
    rs = core.risk_summary(w1, r.to_numpy(), scen, zeta)
    rs.pop("risk_contribution")
    return {
        "name": name,
        "trades": _trades(tickers, w0, w1, total),
        "turnover": _r(0.5 * trade.sum()),
        "estimated_cost": _r(cost_bps * 1e-4 * trade.sum() * total, 2),
        "risk_after": {k: _r(v) for k, v in rs.items()},
        "monthly_evar_change": _r(rs["monthly_evar"] - risk_before["monthly_evar"]),
    }


def propose_rebalance(holdings: dict, prices: str = "demo", mode: str = "min_evar",
                      target_weights: dict | None = None, max_weight: float = 1.0, cost_bps: float = 10.0,
                      turnover_cap: float | None = None, trigger_band: float = 0.05,
                      band_type: str = "absolute", landing_band: float | None = None, zeta: float = 0.05,
                      lookback_days: int = 756, horizon_days: int = 21, mean_weight: float = 0.0,
                      stability_samples: int = 30, seed: int = 0) -> dict:
    if mode not in ("min_evar", "min_vol", "target"):
        raise RebalanceError("mode must be 'min_evar', 'min_vol' or 'target'")
    if band_type not in ("absolute", "relative"):
        raise RebalanceError("band_type must be 'absolute' or 'relative'")
    tickers, w0, total = core.holdings_to_weights(holdings)
    n = len(tickers)
    _validate(zeta=zeta, cost_bps=cost_bps, max_weight=max_weight, n=n, turnover_cap=turnover_cap,
              mean_weight=mean_weight, stability_samples=stability_samples)
    inaccurate_before = core.INACCURATE_SOLVES
    r = core.daily_returns(core.load_prices(prices), tickers, lookback_days)
    rd = r.to_numpy()
    scen = core.monthly_scenarios(rd, horizon_days)
    before = core.risk_summary(w0, rd, scen, zeta)
    before.pop("risk_contribution")

    out = {"mode": mode, "as_of": str(r.index[-1].date()), "portfolio_value": _r(total, 2),
           "level": f"worst {zeta:.0%} of {horizon_days}-day periods",
           "risk_before": {k: _r(v) for k, v in before.items()}, "proposals": [], "stability": None, "notes": []}

    if mode == "target":
        if not isinstance(target_weights, dict) or set(target_weights) != set(tickers):
            raise RebalanceError("target mode needs target_weights for exactly the same tickers as the holdings")
        tw = np.array([float(target_weights[t]) for t in tickers])
        if (tw < 0).any() or tw.sum() <= 0:
            raise RebalanceError("target weights must be non-negative and sum to a positive number")
        if abs(tw.sum() - 1.0) > 1e-6:
            out["notes"].append(f"target weights summed to {tw.sum():.4f}; they were rescaled to sum to 1.")
            tw = tw / tw.sum()
        if (tw > max_weight + 1e-9).any():
            raise RebalanceError("a target weight exceeds max_weight")
        landing = trigger_band / 2 if landing_band is None else landing_band
        if landing > trigger_band:
            raise RebalanceError("landing_band cannot exceed trigger_band")
        w1, traded = core.target_rebalance_weights(w0, tw, trigger_band=trigger_band, band_type=band_type,
                                                   landing_band=landing, cap=max_weight, turnover_cap=turnover_cap)
        out["drift_vs_target"] = [{"ticker": t, "current": _r(w0[i]), "target": _r(tw[i]),
                                   "drift": _r(w0[i] - tw[i])} for i, t in enumerate(tickers)]
        out["proposals"].append(_proposal("tolerance_band", tickers, w0, w1, total, cost_bps, r, scen, zeta, before))
        out["recommended"] = "tolerance_band"
        out["notes"].append(
            "No holding is outside its band, so no trades are proposed." if not traded else
            f"At least one holding breached its {band_type} band of {trigger_band}; the proposal is the "
            f"smallest set of trades that brings every holding within {landing} of its target.")
    else:
        solve_raw = core.min_evar_weights if mode == "min_evar" else core.min_vol_weights
        solve = partial(solve_raw, w0, zeta=zeta, cap=max_weight, cost_bps=cost_bps, turnover_cap=turnover_cap,
                        mean_weight=mean_weight)
        w_point = solve(scen)
        out["proposals"].append(_proposal("point_estimate", tickers, w0, w_point, total, cost_bps, r, scen, zeta, before))
        out["recommended"] = "point_estimate"
        if stability_samples > 0:
            W = core.bootstrap_weights(rd, solve, n_boot=stability_samples, horizon=horizon_days, seed=seed)
            st = core.stability_summary(W, w_point, w0, tickers)
            out["proposals"].append(_proposal("bagged", tickers, w0, st["bagged_weights"], total, cost_bps, r, scen,
                                              zeta, before))
            out["recommended"] = "bagged"
            out["stability"] = {
                "samples": stability_samples,
                "turnover_caused_by_data_noise_median": _r(st["turnover_noise_median"]),
                "turnover_caused_by_data_noise_p95": _r(st["turnover_noise_p95"]),
                "by_asset": [{"ticker": t, "weight_std": _r(st["weight_std_by_asset"][i]),
                              "trade_direction_agreement": _r(st["trade_direction_agreement"][i], 2)}
                             for i, t in enumerate(tickers)],
                "how_to_read": ("Each sample re-solves the problem on a block-bootstrapped history. "
                                "turnover_caused_by_data_noise is how much of the portfolio the proposal moves "
                                "between resamples: the part of any trade that is estimation noise, not signal. "
                                "trade_direction_agreement near 1 means resamples agree on buy vs sell."),
            }
            out["notes"].append("'bagged' averages the resampled solutions (a standard way to reduce "
                                "estimation noise); 'point_estimate' is the single solution on the observed history.")
        if mean_weight == 0.0:
            out["notes"].append("Expected returns were ignored (mean_weight = 0): the proposal reduces risk, it is "
                                "not a return forecast.")
        small = out["proposals"][-1]["turnover"]
        if small is not None and small < 0.005:
            out["notes"].append("The portfolio is already very close to the proposal; trading is unlikely to be "
                                "worth the cost.")
    n_inaccurate = core.INACCURATE_SOLVES - inaccurate_before
    if n_inaccurate:
        out["notes"].append(f"{n_inaccurate} optimizer solve(s) only reached reduced accuracy; treat small weight "
                            "differences as noise.")
    out["assumptions"] = ASSUMPTIONS
    out["disclaimer"] = DISCLAIMER
    return out
