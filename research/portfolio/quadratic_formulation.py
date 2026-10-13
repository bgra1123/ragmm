"""
Experiment P5: the paper's quadratic formulation (Section 3) on the quadratic portfolio problem R1.

Pre-registered design: PLAN.md, section 13. Run from the repository root:
    python research/portfolio/quadratic_formulation.py
"""
import argparse
import json
import math
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np
from scipy.special import logsumexp

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import backtest as bt  # noqa: E402
from risk_averse_tuning import quadratic as q  # noqa: E402

G = {}          # shared with forked workers


def empirical_evar(x, zeta):
    x = np.asarray(x, float)
    scale = max(x.mean(), 1e-300)
    ts = np.geomspace(0.01, 200.0, 400) / scale
    vals = [(logsumexp(t * x) - math.log(len(x)) - math.log(zeta)) / t for t in ts]
    return float(min(vals))


def noise_cov(prob, W0row):
    """Covariance (z-space, per minibatch) of the stochastic gradient at the reference solution."""
    Zs, _ = prob.exact(W0row)
    ws = (W0row + Zs @ prob.N.T)[0]
    v = -252 * prob.Rw + 252 * prob.theta * prob.D * (prob.D @ ws)[:, None]
    v = v @ prob.N
    v = v - v.mean(0)
    return v.T @ v / len(v) / G["batch"], -Zs[0]


def params_for_date(j):
    a = G["args"]
    prob = bt.Problem(G["R"], G["reb"][j], a.theta, a.c, a.lam, G["N"])
    ev = np.linalg.eigvalsh(prob.Hz)
    out = {}
    s = q.solve_quadratic(ev, a.zeta, 1.0, 50, zoom=0) if True else None
    out["Q32-grid (eps=1)"] = dict(alpha=s.alpha, beta=s.beta, gamma=s.gamma)
    for eps in (0.25, 0.05):
        s = q.solve_quadratic(ev, a.zeta, eps, 50, alpha_max=2 / prob.L, zoom=3)
        out[f"Q32 (eps={eps})"] = dict(alpha=s.alpha, beta=s.beta, gamma=s.gamma)
    s = q.solve_quadratic(ev, a.zeta, 0.25, 50, alpha_max=2 / prob.L, use_exact=True, zoom=3)
    out["Q32-exact (eps=0.25)"] = dict(alpha=s.alpha, beta=s.beta, gamma=s.gamma)
    Sigma, e0 = noise_cov(prob, G["ideal_W0"][j][None])
    sel = q.select_parameters(prob.Hz, a.K, Sigma, e0, a.zeta)
    out["EXT (exact-K)"] = dict(alpha=sel.alpha, beta=sel.beta, gamma=sel.gamma)
    p = bt.ra_gmm_params(prob.L, prob.mu, prob.N.shape[1], a.zeta, 0.25)
    out["RA-GMM (51, eps=0.25)"] = dict(alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"])
    return out


def paired_boot(A, B, stat, n_boot=10_000, seed=0):
    """A, B: (S, dates). Resample seeds; difference stat(A) - stat(B)."""
    rng = np.random.default_rng(seed)
    S = A.shape[0]
    d = np.empty(n_boot)
    for k in range(n_boot):
        ix = rng.integers(0, S, S)
        d[k] = stat(A[ix]) - stat(B[ix])
    return dict(diff=float(stat(A) - stat(B)), ci95=[float(np.quantile(d, .025)), float(np.quantile(d, .975))])


def s_mean(x): return x.mean()
def s_p95(x): return np.quantile(x, 0.95)
def s_cvar(x):
    v = np.sort(x.ravel())
    return v[-max(1, int(round(0.05 * v.size))):].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=10.0)
    ap.add_argument("--c", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--zeta", type=float, default=0.05)
    ap.add_argument("--tune_dates", type=int, default=24)
    ap.add_argument("--max_dates", type=int, default=None, help="smoke test only")
    ap.add_argument("--calib_seeds", type=int, default=2000)
    ap.add_argument("--calib_dates", type=int, default=40)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    t0 = time.time()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    R, dates = bt.load_returns()
    reb = bt.rebalance_days(dates)
    if args.max_dates:
        reb = reb[: args.max_dates]
        R = R[: reb[-1] + 21]
    n = R.shape[1]
    N = bt.null_basis(n)
    ideal_W0, ideal_daily, _ = bt.ideal_path(R, reb, N, args)
    tuned = bt.tune(R, reb, N, args, ideal_W0)
    print(f"tuned: {tuned} [{time.time() - t0:.0f}s]", flush=True)

    G.update(args=args, R=R, reb=reb, N=N, ideal_W0=ideal_W0, batch=args.batch)
    with mp.get_context("fork").Pool(args.procs) as pool:
        params = pool.map(params_for_date, range(len(reb)), chunksize=4)
    print(f"parameters selected for {len(reb)} dates [{time.time() - t0:.0f}s]", flush=True)

    seeds = list(range(args.seeds))
    S = len(seeds)
    names = None
    state, daily, sub_all, ph_all, tv_all = {}, {}, {}, {}, {}
    for j, i in enumerate(reb):
        prob = bt.Problem(R, i, args.theta, args.c, args.lam, N)
        cfgs = bt.textbook(prob)
        for nm, t in tuned.items():
            cfgs[nm] = dict(alpha=t["a"] / prob.L, beta=t["beta"], gamma=t["beta"] if t["nest"] else 0.0,
                            tail_average=t["avg"])
        for nm, p in params[j].items():
            cfgs[nm] = dict(p)
        if names is None:
            names = list(cfgs)
            for nm in names:
                state[nm] = np.ones((S, n)) / n
                daily[nm], sub_all[nm], ph_all[nm], tv_all[nm] = [], [], [], []
        idx = bt.draw_indices(seeds, j, args.K, args.batch)
        end = reb[j + 1] if j + 1 < len(reb) else len(R)
        for nm in names:
            W0 = state[nm]
            Ws = bt.exact_W(prob, W0)
            W = bt.run_W(prob, W0, idx, cfgs[nm])
            sub_all[nm].append(bt.subopt_W(prob, W, Ws, W0))
            ph_all[nm].append(np.abs(W - Ws).sum(1))
            tv = np.abs(W - W0).sum(1)
            tv_all[nm].append(tv)
            d, state[nm] = bt.hold(W, R[i:end])
            d[:, 0] = (1 + d[:, 0]) * (1 - bt.COST_RATE * tv) - 1
            daily[nm].append(d)
        if j % 50 == 0:
            print(f"date {j}/{len(reb)} [{time.time() - t0:.0f}s]", flush=True)

    k0 = args.tune_dates
    summary = dict(config=vars(args) | dict(n_rebalances=len(reb), oos_dates=len(reb) - k0), tuned=tuned,
                   param_summary={nm: {k: bt.dist([p[nm][k] for p in params]) for k in ("alpha", "beta", "gamma")}
                                  for nm in params[0]},
                   methods={})
    subs, phs = {}, {}
    ideal = bt.perf(ideal_daily[None])
    for nm in names:
        D = np.concatenate(daily[nm], 1)
        P = bt.perf(D)
        sub = np.stack(sub_all[nm], 1)[:, k0:]
        ph = np.stack(ph_all[nm], 1)[:, k0:]
        subs[nm], phs[nm] = sub, ph
        te = (D - ideal_daily[None]).std(1) * math.sqrt(252)
        tw = P["terminal_wealth"]
        summary["methods"][nm] = dict(
            subopt_oos=bt.dist(sub.ravel()), subopt_cvar5=float(s_cvar(sub)), phantom_oos=bt.dist(ph.ravel()),
            phantom_cvar5=float(s_cvar(ph)), monthly_turnover=float(np.stack(tv_all[nm], 1).mean()),
            tracking_error=bt.dist(te), fan_width_95_over_5=float(np.quantile(tw, .95) / np.quantile(tw, .05)),
            **{k: bt.dist(v) for k, v in P.items()})
    summary["ideal"] = {k: float(v[0]) for k, v in ideal.items()}

    pairs = [("Q32 (eps=0.25)", "Nesterov (textbook)"), ("Q32 (eps=0.25)", "SGD (textbook)"),
             ("Q32 (eps=0.25)", "RA-GMM (51, eps=0.25)"), ("Q32 (eps=0.25)", "Nesterov (tuned for mean)"),
             ("Q32 (eps=0.25)", "SGD + tail averaging (tuned)"), ("EXT (exact-K)", "Q32 (eps=0.25)"),
             ("EXT (exact-K)", "Nesterov (tuned for mean)"), ("EXT (exact-K)", "SGD + tail averaging (tuned)"),
             ("Q32-exact (eps=0.25)", "Q32 (eps=0.25)"), ("Q32 (eps=0.05)", "Q32 (eps=0.25)")]
    summary["paired"] = {}
    for a, b in pairs:
        summary["paired"][f"{a} minus {b}"] = {
            "subopt_mean": paired_boot(subs[a], subs[b], s_mean, 2000),
            "subopt_p95": paired_boot(subs[a], subs[b], s_p95, 2000),
            "subopt_cvar5": paired_boot(subs[a], subs[b], s_cvar, 2000),
            "phantom_p95": paired_boot(phs[a], phs[b], s_p95, 2000)}
    print(f"backtest done [{time.time() - t0:.0f}s]", flush=True)

    # ---- calibration ----
    rng_dates = np.linspace(0, len(reb) - 1, args.calib_dates).astype(int)
    calib = []
    for j in rng_dates:
        i = reb[j]
        prob = bt.Problem(R, i, args.theta, args.c, args.lam, N)
        W0 = np.repeat(ideal_W0[j][None], args.calib_seeds, 0)
        Ws = bt.exact_W(prob, W0)
        idx = bt.draw_indices(range(50_000, 50_000 + args.calib_seeds), 200_000 + j, args.K, args.batch)
        Sigma, e0 = noise_cov(prob, ideal_W0[j][None])
        sigma2 = float(np.trace(Sigma) / Sigma.shape[0])
        ev = np.linalg.eigvalsh(prob.Hz)
        row = dict(date=int(j), kappa=float(prob.L / prob.mu))
        cfgs = bt.textbook(prob)
        cfgs = {"Nesterov (textbook)": cfgs["Nesterov (textbook)"], "Q32 (eps=0.25)": params[j]["Q32 (eps=0.25)"],
                "EXT (exact-K)": params[j]["EXT (exact-K)"]}
        for nm, c in cfgs.items():
            W = bt.run_W(prob, W0, idx, c)
            sub = bt.subopt_W(prob, W, Ws, W0)
            pred, mean_pred = q.evar_quadratic(prob.Hz, c["alpha"], c["beta"], c["gamma"], args.K, Sigma, e0,
                                               args.zeta, return_mean=True)
            row[nm] = dict(empirical_evar=empirical_evar(sub, args.zeta), empirical_mean=float(sub.mean()),
                           empirical_p95=float(np.quantile(sub, .95)), predicted_exact_K_evar=float(pred),
                           predicted_exact_K_mean=float(mean_pred),
                           stationary_exact_evar_isotropic=float(q.quad_evar_exact(c["alpha"], c["beta"], c["gamma"], ev, args.zeta, sigma2)),
                           theorem1_bound_isotropic=float(q.quad_evar_bound(c["alpha"], c["beta"], c["gamma"], ev, args.zeta, sigma2)))
        calib.append(row)
    summary["calibration"] = calib
    (out / "quadratic_formulation.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out}/quadratic_formulation.json in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
