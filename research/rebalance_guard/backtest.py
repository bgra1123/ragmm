"""
Backtest of Rebalance Guard's recommendations. Design: PLAN.md in this folder.

Run from the repository root:
    python research/rebalance_guard/backtest.py                  # primary: max weight 0.30
    python research/rebalance_guard/backtest.py --cap 1.0        # sensitivity
"""
import argparse
import json
import math
import pathlib
import sys
import time
from functools import partial

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "rebalance-guard"))
from rebalance_guard import core  # noqa: E402

LOOKBACK, HORIZON, ZETA, COST_BPS, B = 756, 21, 0.05, 10.0, 30


def first_trading_days(index):
    start = np.flatnonzero(np.r_[True, index.month[1:] != index.month[:-1]])
    return [int(i) for i in start if i >= LOOKBACK + 1]


def solver_for(objective, w0, cap):
    fn = core.min_evar_weights if objective == "min_evar" else core.min_vol_weights
    return partial(fn, w0, zeta=ZETA, cap=cap, cost_bps=COST_BPS, mean_weight=0.0)


def run_strategy(name, rets, dates_idx, cap, n_boot):
    n = rets.shape[1]
    w_drift = np.ones(n) / n
    monthly, targets, turnover = [], [], []
    daily_all = []
    for j, i in enumerate(dates_idx):
        window = rets[i - LOOKBACK:i]
        if name == "EW":
            target = np.ones(n) / n
        else:
            objective = "min_evar" if "EVaR" in name else "min_vol"
            solve = solver_for(objective, w_drift, cap)
            scen = core.monthly_scenarios(window, HORIZON)
            target = solve(scen)
            if name.endswith("bagged"):
                W = core.bootstrap_weights(window, solve, n_boot=n_boot, horizon=HORIZON, seed=i)
                target = W.mean(0)
        trade = np.abs(target - w_drift).sum()
        turnover.append(0.5 * trade)
        targets.append(target)
        end = dates_idx[j + 1] if j + 1 < len(dates_idx) else len(rets)
        # daily path with drift (vectorized)
        gross = np.cumprod(1.0 + rets[i:end], axis=0) * target                 # (T, n) holdings value, start = 1
        port = gross.sum(1)
        daily = np.diff(np.r_[1.0, port]) / np.r_[1.0, port[:-1]]
        daily = daily.copy()
        daily[0] = (1.0 + daily[0]) * (1.0 - COST_BPS * 1e-4 * trade) - 1.0     # cost paid on the first day
        daily_all.append(daily)
        monthly.append(np.prod(1.0 + daily) - 1.0)
        w_drift = gross[-1] / gross[-1].sum()
    return dict(monthly=np.array(monthly), daily=np.concatenate(daily_all), turnover=np.array(turnover),
                targets=np.array(targets))


def summarize(res):
    d, m = res["daily"], res["monthly"]
    years = len(d) / 252
    wealth = np.cumprod(1.0 + d)
    k = max(1, int(round(0.05 * len(m))))
    target_instability = 0.5 * np.abs(np.diff(res["targets"], axis=0)).sum(1)
    return dict(
        ann_return=float(wealth[-1] ** (1 / years) - 1), ann_vol=float(d.std() * math.sqrt(252)),
        sharpe=float(d.mean() / d.std() * math.sqrt(252)), max_drawdown=core.max_drawdown(d),
        realized_cvar5_month=float(-np.sort(m)[:k].mean()), worst_month=float(-m.min()),
        mean_monthly_turnover=float(res["turnover"].mean()), target_instability=float(target_instability.mean()),
        terminal_wealth=float(wealth[-1]), months=int(len(m)))


def stationary_bootstrap_indices(n, rng, mean_block=6):
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    for t in range(1, n):
        idx[t] = rng.integers(n) if rng.random() < 1.0 / mean_block else (idx[t - 1] + 1) % n
    return idx


def paired_diff(a, b, stat, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a["monthly"])
    out = np.empty(n_boot)
    for k in range(n_boot):
        ix = stationary_bootstrap_indices(n, rng)
        out[k] = stat(a, ix) - stat(b, ix)
    full = np.arange(n)
    return dict(diff=float(stat(a, full) - stat(b, full)), ci95=[float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))])


def s_sharpe(res, ix):
    m = res["monthly"][ix]
    return m.mean() / m.std() * math.sqrt(12)


def s_cvar(res, ix):
    m = np.sort(res["monthly"][ix])
    return -m[:max(1, int(round(0.05 * len(m))))].mean()


def s_turnover(res, ix):
    return res["turnover"][ix].mean()


def s_return(res, ix):
    return (np.prod(1 + res["monthly"][ix])) ** (12 / len(ix)) - 1


def s_instab(res, ix):
    t = res["targets"]
    d = 0.5 * np.abs(np.diff(t, axis=0)).sum(1)
    return d[ix[ix < len(d)]].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=0.30)
    ap.add_argument("--boot", type=int, default=B)
    ap.add_argument("--max_dates", type=int, default=None)
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    px = core.load_prices("demo")
    rets = px.pct_change().dropna()
    dates_idx = first_trading_days(rets.index)
    if args.max_dates:
        dates_idx = dates_idx[: args.max_dates]
    R = rets.to_numpy()
    t0 = time.time()
    names = ["EW", "MinVol-point", "MinVol-bagged", "MinEVaR-point", "MinEVaR-bagged"]
    results = {}
    for name in names:
        results[name] = run_strategy(name, R, dates_idx, args.cap, args.boot)
        s = summarize(results[name])
        print(f"{name:15s} ret {s['ann_return']:.3f} vol {s['ann_vol']:.3f} sharpe {s['sharpe']:.2f} mdd {s['max_drawdown']:.3f} "
              f"cvar5m {s['realized_cvar5_month']:.3f} turn {s['mean_monthly_turnover']:.3f} instab {s['target_instability']:.3f}"
              f"  [{time.time() - t0:.0f}s]", flush=True)
    summary = {n: summarize(r) for n, r in results.items()}
    stats = dict(sharpe=s_sharpe, realized_cvar5_month=s_cvar, turnover=s_turnover, ann_return=s_return,
                 target_instability=s_instab)
    pairs = [("MinEVaR-point", "EW"), ("MinVol-point", "EW"), ("MinEVaR-bagged", "EW"), ("MinVol-bagged", "EW"),
             ("MinEVaR-bagged", "MinEVaR-point"), ("MinVol-bagged", "MinVol-point"), ("MinEVaR-point", "MinVol-point")]
    comparisons = {f"{a} minus {b}": {k: paired_diff(results[a], results[b], f) for k, f in stats.items()}
                   for a, b in pairs}
    tag = f"cap{args.cap:g}" + ("_smoke" if args.max_dates else "")
    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)
    (out / f"backtest_{tag}.json").write_text(json.dumps(dict(
        config=dict(cap=args.cap, lookback=LOOKBACK, horizon=HORIZON, zeta=ZETA, cost_bps=COST_BPS, boot=args.boot,
                    first=str(rets.index[dates_idx[0]].date()), last=str(rets.index[dates_idx[-1]].date()),
                    months=len(dates_idx)),
        summary=summary, comparisons=comparisons), indent=2))
    np.savez_compressed(out / f"paths_{tag}.npz", **{f"{n}|monthly": r["monthly"] for n, r in results.items()},
                        **{f"{n}|turnover": r["turnover"] for n, r in results.items()},
                        **{f"{n}|targets": r["targets"] for n, r in results.items()},
                        dates=np.array([str(rets.index[i].date()) for i in dates_idx]))
    print(f"wrote results/backtest_{tag}.json in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
