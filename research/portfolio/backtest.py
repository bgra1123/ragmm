"""
Experiment P1: monthly rebalancing backtest (problem R1) with stochastic solvers.

Pre-registered design: PLAN.md, section 10.

Run from the repository root:
    python research/portfolio/backtest.py
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "model_multiplicity"))
from paper_tuning import ra_gmm_params  # noqa: E402

STOCKS = ["AAPL", "GE", "AMD", "WMT", "BAC", "T", "XOM", "BBY", "PFE", "JPM"]
WINDOW = 252
COST_RATE = 0.001          # 10 bps per unit of one-way turnover


# ---------------------------------------------------------------------- #
# Data and problem                                                         #
# ---------------------------------------------------------------------- #

def load_returns():
    p = pd.read_csv(HERE / "data" / "stock_prices.csv", parse_dates=["date"]).set_index("date")
    r = p[STOCKS].pct_change().iloc[1:]
    assert not r.isna().any().any()
    return r.to_numpy(), r.index


def rebalance_days(dates):
    starts = np.flatnonzero(np.r_[True, dates.month[1:] != dates.month[:-1]])
    return [int(i) for i in starts if i >= WINDOW]


def null_basis(n):
    P = np.eye(n) - np.ones((n, n)) / n
    return np.linalg.svd(P)[0][:, : n - 1]          # orthonormal, columns sum to 0


class Problem:
    """R1 at one rebalance date (annualized units), in reduced coordinates z: w = w0 + N z."""

    def __init__(self, R, i, theta, c, lam, N):
        self.Rw = R[i - WINDOW:i]
        self.mbar = self.Rw.mean(0)
        self.D = self.Rw - self.mbar
        self.m = 252 * self.mbar
        self.C = 252 * (self.D.T @ self.D) / WINDOW
        self.theta, self.c, self.lam, self.N = theta, c, lam, N
        n = R.shape[1]
        self.Hz = N.T @ (theta * self.C + (c + lam) * np.eye(n)) @ N
        ev = np.linalg.eigvalsh(self.Hz)
        self.mu, self.L = float(ev[0]), float(ev[-1])

    def grad_w0(self, W0):
        """∇f at w = w0 (trading-cost term is zero there), rows = seeds."""
        return -self.m + self.theta * W0 @ self.C + self.lam * W0

    def exact(self, W0):
        gz0 = self.grad_w0(W0) @ self.N
        Z = -np.linalg.solve(self.Hz, gz0.T).T
        return Z, gz0

    def subopt(self, Z, Zstar, gz0):
        """f(w0+NZ) - f* (exact, since f is quadratic in z)."""
        dZ = Z - Zstar
        return 0.5 * np.einsum("si,ij,sj->s", dZ, self.Hz, dZ)

    def stoch_grad(self, W, W0, idx):
        """Unbiased minibatch gradient in z-space. W, W0: (S, n); idx: (S, b) day indices."""
        rb = self.Rw[idx]                                   # (S, b, n)
        db = self.D[idx]
        b = idx.shape[1]
        proj = np.einsum("sbn,sn->sb", db, W)
        g = (-252 * rb.mean(1) + 252 * self.theta * np.einsum("sbn,sb->sn", db, proj) / b
             + self.c * (W - W0) + self.lam * W)
        return g @ self.N


def solve_gmm(prob, W0, idx, alpha, beta, gamma, tail_average=False):
    """K stochastic GMM steps from z = 0 (current holdings). idx: (S, K, b)."""
    S, K, _ = idx.shape
    Z = np.zeros((S, prob.N.shape[1]))
    Dm = np.zeros_like(Z)
    avg, n_avg = np.zeros_like(Z), 0
    for k in range(K):
        Y = Z + gamma * Dm
        G = prob.stoch_grad(W0 + Y @ prob.N.T, W0, idx[:, k])
        Dm = beta * Dm - alpha * G
        Z = Z + Dm
        if tail_average and k >= K // 2:
            n_avg += 1
            avg += (Z - avg) / n_avg
    return avg if tail_average else Z


def draw_indices(seeds, date_no, K, b):
    return np.stack([np.random.default_rng((s, date_no)).integers(0, WINDOW, size=(K, b))
                     for s in seeds]).astype(np.int32)


# ---------------------------------------------------------------------- #
# Method configurations                                                    #
# ---------------------------------------------------------------------- #

def textbook(prob):
    kappa = prob.L / prob.mu
    m = (math.sqrt(kappa) - 1) / (math.sqrt(kappa) + 1)
    return {"SGD (textbook)": dict(alpha=1 / prob.L, beta=0.0, gamma=0.0),
            "Nesterov (textbook)": dict(alpha=1 / prob.L, beta=m, gamma=m),
            "Heavy-ball (textbook)": dict(alpha=1 / prob.L, beta=m, gamma=0.0)}


def tune(R, reb, N, args, ideal_W0):
    """Pick step multipliers (and Nesterov beta) by MEAN suboptimality on the first 24 dates."""
    mults = np.logspace(-4, math.log10(2.0), 18)
    grids = {
        "SGD (tuned for mean)": [dict(a=a, beta=0.0, nest=False, avg=False) for a in mults],
        "Nesterov (tuned for mean)": [dict(a=a, beta=bt, nest=True, avg=False)
                                      for a in mults for bt in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95)],
        "SGD + tail averaging (tuned)": [dict(a=a, beta=0.0, nest=False, avg=True) for a in mults],
    }
    tseeds = range(10_000, 10_020)
    scores = {name: np.zeros(len(g)) for name, g in grids.items()}
    for j in range(args.tune_dates):
        prob = Problem(R, reb[j], args.theta, args.c, args.lam, N)
        W0 = np.repeat(ideal_W0[j][None], len(tseeds), 0)
        Zs, gz0 = prob.exact(W0)
        idx = draw_indices(tseeds, 100_000 + j, args.K, args.batch)
        for name, grid in grids.items():
            for gi, cfg in enumerate(grid):
                Z = solve_gmm(prob, W0, idx, cfg["a"] / prob.L, cfg["beta"],
                              cfg["beta"] if cfg["nest"] else 0.0, cfg["avg"])
                sub = prob.subopt(Z, Zs, gz0)
                scores[name][gi] += np.mean(sub) if np.all(np.isfinite(sub)) else np.inf
    return {name: grids[name][int(np.argmin(scores[name]))] for name in grids}


def configs_for(prob, tuned, ra_cache, j, eps_list, zeta):
    cfgs = textbook(prob)
    for name, t in tuned.items():
        cfgs[name] = dict(alpha=t["a"] / prob.L, beta=t["beta"], gamma=t["beta"] if t["nest"] else 0.0,
                          tail_average=t["avg"])
    for eps in eps_list:
        key = (j, eps)
        if key not in ra_cache:
            ra_cache[key] = ra_gmm_params(prob.L, prob.mu, prob.N.shape[1], zeta, eps)
        p = ra_cache[key]
        cfgs[f"RA-GMM (eps={eps})"] = dict(alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"])
    return cfgs


# ---------------------------------------------------------------------- #
# Backtest                                                                 #
# ---------------------------------------------------------------------- #

def hold(W, R_period):
    """Hold weights W (S, n) through the period; return daily portfolio returns (S, T) and drifted weights."""
    H = W.copy()
    rets = []
    for r in R_period:
        before = H.sum(1)
        H = H * (1 + r)
        rets.append(H.sum(1) / before - 1)
    return np.array(rets).T, H / H.sum(1, keepdims=True)


def ideal_path(R, reb, N, args):
    n = R.shape[1]
    w0 = np.ones((1, n)) / n
    W0s, daily, turnover = [], [], []
    for j, i in enumerate(reb):
        prob = Problem(R, i, args.theta, args.c, args.lam, N)
        Zs, _ = prob.exact(w0)
        w = w0 + Zs @ N.T
        W0s.append(w0[0].copy())
        tv = np.abs(w - w0).sum(1)
        turnover.append(float(tv[0]))
        end = reb[j + 1] if j + 1 < len(reb) else len(R)
        d, w0 = hold(w, R[i:end])
        d[:, 0] = (1 + d[:, 0]) * (1 - COST_RATE * tv) - 1
        daily.append(d[0])
    return np.array(W0s), np.concatenate(daily), np.array(turnover)


def log_growth(daily):
    """Daily log growth; a day with return <= -100% wipes the account out (ruin) and it stays at ~0."""
    g = np.log(np.clip(1 + daily, 1e-12, None))
    ruined = np.cumsum(1 + daily <= 0, 1) > 0
    g[ruined & ~(1 + daily <= 0)] = 0.0
    return g, ruined[:, -1]


def perf(daily):
    """daily: (S, T) returns. Annualized stats per seed."""
    g, ruin = log_growth(daily)
    logw = np.cumsum(g, 1)
    wealth = np.exp(logw)
    peak = np.maximum.accumulate(np.maximum(wealth, 1.0), 1)
    mdd = (1 - wealth / peak).max(1)
    mean, vol = daily.mean(1) * 252, daily.std(1) * math.sqrt(252)
    years = daily.shape[1] / 252
    return dict(ann_return=np.exp(logw[:, -1] / years) - 1, ann_vol=vol, sharpe=mean / vol,
                max_drawdown=mdd, terminal_wealth=wealth[:, -1], ruined=ruin.astype(float))


def dist(v):
    v = np.asarray(v, float)
    k = max(1, int(round(0.05 * len(v))))
    s = np.sort(v)
    return dict(mean=float(v.mean()), p05=float(np.quantile(v, 0.05)), median=float(np.median(v)),
                p95=float(np.quantile(v, 0.95)), worst5_low=float(s[:k].mean()), worst5_high=float(s[-k:].mean()),
                min=float(v.min()), max=float(v.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=10.0)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--zeta", type=float, default=0.05)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.25, 0.05])
    ap.add_argument("--tune_dates", type=int, default=24)
    ap.add_argument("--max_dates", type=int, default=None, help="for smoke tests only")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    R, dates = load_returns()
    reb = rebalance_days(dates)
    if args.max_dates:
        reb = reb[: args.max_dates]
    n = R.shape[1]
    N = null_basis(n)
    end_all = len(R) if not args.max_dates else reb[-1] + 21
    R = R[:end_all]

    ideal_W0, ideal_daily, ideal_turn = ideal_path(R, reb, N, args)
    tuned = tune(R, reb, N, args, ideal_W0)
    print(f"tuned: {tuned}  [{time.time()-t0:.0f}s]")

    seeds = list(range(args.seeds))
    S = len(seeds)
    names = None
    state, daily, sub_all, phantom_all, turn_all, month_wealth = {}, {}, {}, {}, {}, {}
    ra_cache, kappas = {}, []
    for j, i in enumerate(reb):
        prob = Problem(R, i, args.theta, args.c, args.lam, N)
        kappas.append(prob.L / prob.mu)
        cfgs = configs_for(prob, tuned, ra_cache, j, args.eps, args.zeta)
        if names is None:
            names = list(cfgs)
            for nm in names:
                state[nm] = np.ones((S, n)) / n
                daily[nm], sub_all[nm], phantom_all[nm], turn_all[nm], month_wealth[nm] = [], [], [], [], []
        idx = draw_indices(seeds, j, args.K, args.batch)
        end = reb[j + 1] if j + 1 < len(reb) else len(R)
        for nm in names:
            W0 = state[nm]
            Zs, gz0 = prob.exact(W0)
            Z = solve_gmm(prob, W0, idx, **cfgs[nm])
            W = W0 + Z @ N.T
            sub_all[nm].append(prob.subopt(Z, Zs, gz0))
            phantom_all[nm].append(np.abs((Z - Zs) @ N.T).sum(1))
            tv = np.abs(W - W0).sum(1)
            turn_all[nm].append(tv)
            d, state[nm] = hold(W, R[i:end])
            d[:, 0] = (1 + d[:, 0]) * (1 - COST_RATE * tv) - 1
            daily[nm].append(d)
        if j % 50 == 0:
            print(f"date {j}/{len(reb)} {dates[i].date()} kappa={kappas[-1]:.1f} [{time.time()-t0:.0f}s]")

    ideal = perf(ideal_daily[None])
    summary = dict(config=vars(args) | dict(stocks=STOCKS, n_rebalances=len(reb),
                                            first=str(dates[reb[0]].date()), last=str(dates[reb[-1]].date()),
                                            kappa=dist(kappas)),
                   tuned=tuned,
                   ra_params_by_eps={str(e): dist([ra_cache[(j, e)]["alpha"] * 1 for j in range(len(reb))]) | dict(
                       rate_ratio=dist([ra_cache[(j, e)]["rate_ratio"] for j in range(len(reb))]),
                       beta=dist([ra_cache[(j, e)]["beta"] for j in range(len(reb))]),
                       gamma=dist([ra_cache[(j, e)]["gamma"] for j in range(len(reb))]))
                       for e in args.eps},
                   ideal=dict({k: float(v[0]) for k, v in ideal.items()},
                              mean_monthly_turnover=float(ideal_turn.mean())),
                   methods={})
    arrays = {}
    for nm in names:
        D = np.concatenate(daily[nm], 1)
        P = perf(D)
        sub = np.stack(sub_all[nm], 1)         # (S, dates)
        ph = np.stack(phantom_all[nm], 1)
        tv = np.stack(turn_all[nm], 1)
        te = (D - ideal_daily[None]).std(1) * math.sqrt(252)
        summary["methods"][nm] = dict(
            subopt_all=dist(sub.ravel()), phantom_turnover_all=dist(ph.ravel()),
            phantom_turnover_per_seed_avg=dist(ph.mean(1)), monthly_turnover_per_seed_avg=dist(tv.mean(1)),
            tracking_error_vs_exact=dist(te), **{k: dist(v) for k, v in P.items()})
        month_ends = np.cumsum([d.shape[1] for d in daily[nm]]) - 1
        arrays[f"{nm}|log_wealth_monthly"] = np.cumsum(log_growth(D)[0], 1)[:, month_ends]
        arrays[f"{nm}|phantom"] = ph
    arrays["ideal|log_wealth_monthly"] = np.cumsum(np.log1p(ideal_daily))[month_ends]
    arrays["rebalance_dates"] = np.array([str(dates[i].date()) for i in reb])
    tag = "smoke" if args.max_dates else "main"
    (out / f"backtest_{tag}.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(out / f"backtest_{tag}_arrays.npz", **arrays)
    print(f"wrote {out}/backtest_{tag}.json in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
