"""
Experiment P3: tune the solver for the worst case (EV@R) instead of the average.

Pre-registered design: PLAN.md, section 12.

Run from the repository root:
    python research/portfolio/worst_case_tuning.py --long_only     # main
    python research/portfolio/worst_case_tuning.py                 # secondary (unconstrained)
"""
import argparse
import json
import math
import pathlib
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "model_multiplicity"))
import backtest as b  # noqa: E402
from paper_tuning import ra_gmm_params  # noqa: E402

THETA, C, LAM = 10.0, 1.0, 1.0
K, BATCH = 200, 16
TUNE_DATES = 24
TUNE_SEEDS = range(10_000, 10_100)
EVAL_SEEDS = range(200)
ZETA = 0.05


# ---------------------------------------------------------------------- #
# Risk measures                                                            #
# ---------------------------------------------------------------------- #

def evar(x, zeta=ZETA):
    """Empirical entropic value at risk: inf_{t>0} (1/t) log(mean(exp(t x)) / zeta)."""
    x = np.asarray(x, float)
    if not np.all(np.isfinite(x)):
        return np.inf
    scale = max(x.std(), 1e-12)
    ts = np.logspace(-3, 4, 400) / scale
    m = x.max()
    lme = np.array([t * m + np.log(np.mean(np.exp(t * (x - m)))) for t in ts])   # log mean exp, stable
    vals = (lme - math.log(zeta)) / ts
    return float(vals.min())


def cvar(x, q=0.05):
    x = np.sort(np.asarray(x, float))
    k = max(1, int(round(q * len(x))))
    return float(x[-k:].mean()) if np.all(np.isfinite(x)) else np.inf


# ---------------------------------------------------------------------- #
# Search space and tuning                                                  #
# ---------------------------------------------------------------------- #

def search_space():
    mults = np.logspace(-4, math.log10(2.0), 18)
    space = []
    for a in mults:
        for bt in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
            for nest in ((False,) if bt == 0.0 else (False, True)):
                for avg in (False, True):
                    space.append(dict(a=float(a), beta=bt, nest=nest, avg=avg))
    return space


def cfg_for(prob, t):
    return dict(alpha=t["a"] / prob.L, beta=t["beta"], gamma=t["beta"] if t["nest"] else 0.0,
                tail_average=t["avg"])


def tune(R, reb, N, ideal_W0):
    space = search_space()
    outcomes = np.zeros((len(space), TUNE_DATES, len(TUNE_SEEDS)))
    for j in range(TUNE_DATES):
        prob = b.Problem(R, reb[j], THETA, C, LAM, N)
        W0 = np.repeat(ideal_W0[j][None], len(TUNE_SEEDS), 0)
        Ws = b.exact_W(prob, W0)
        idx = b.draw_indices(TUNE_SEEDS, 100_000 + j, K, BATCH)
        for ci, t in enumerate(space):
            with np.errstate(all="ignore"):
                W = b.run_W(prob, W0, idx, cfg_for(prob, t))
                outcomes[ci, j] = b.subopt_W(prob, W, Ws, W0)
    pooled = outcomes.reshape(len(space), -1)
    crit = {
        "average": np.array([np.mean(p) if np.all(np.isfinite(p)) else np.inf for p in pooled]),
        "evar": np.array([evar(p) for p in pooled]),
        "cvar": np.array([cvar(p) for p in pooled]),
    }
    picks = {k: int(np.argmin(v)) for k, v in crit.items()}
    table = {k: dict(config=space[i], average=float(crit["average"][i]), evar=float(crit["evar"][i]),
                     cvar=float(crit["cvar"][i])) for k, i in picks.items()}
    return space, picks, table


# ---------------------------------------------------------------------- #
# Backtest                                                                 #
# ---------------------------------------------------------------------- #

def paired_bootstrap(a, c, stat, n_boot=10_000, seed=0):
    """a, c: (S, dates) arrays for two methods with common seeds; resample seeds."""
    rng = np.random.default_rng(seed)
    S = a.shape[0]
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        j = rng.integers(0, S, S)
        diffs[i] = stat(a[j].ravel()) - stat(c[j].ravel())
    return dict(diff=float(stat(a.ravel()) - stat(c.ravel())),
                ci95=[float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long_only", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results"))
    args = ap.parse_args()
    b.LONG_ONLY = args.long_only
    t0 = time.time()

    R, dates = b.load_returns()
    reb = b.rebalance_days(dates)
    N = b.null_basis(R.shape[1])
    ns = SimpleNamespace(theta=THETA, c=C, lam=LAM)
    ideal_W0, ideal_daily, ideal_turn = b.ideal_path(R, reb, N, ns)

    space, picks, table = tune(R, reb, N, ideal_W0)
    print(f"tuning done [{time.time()-t0:.0f}s]")
    for k, v in table.items():
        print(f"  {k:8s} -> {v['config']}  mean={v['average']:.4g} evar={v['evar']:.4g} cvar={v['cvar']:.4g}")

    methods = {
        "Nesterov (textbook)": None,
        "Average-tuned": space[picks["average"]],
        "Worst-case-tuned (EV@R)": space[picks["evar"]],
        "Worst-case-tuned (CVaR)": space[picks["cvar"]],
        "RA-GMM (eps=0.25)": "ra",
    }
    S, n = len(EVAL_SEEDS), R.shape[1]
    state = {m: np.ones((S, n)) / n for m in methods}
    daily = {m: [] for m in methods}
    sub = {m: [] for m in methods}
    phantom = {m: [] for m in methods}
    turn = {m: [] for m in methods}
    for j, i in enumerate(reb):
        prob = b.Problem(R, i, THETA, C, LAM, N)
        idx = b.draw_indices(EVAL_SEEDS, j, K, BATCH)
        end = reb[j + 1] if j + 1 < len(reb) else len(R)
        for m, t in methods.items():
            if t is None:
                cfg = b.textbook(prob)["Nesterov (textbook)"]
            elif t == "ra":
                p = ra_gmm_params(prob.L, prob.mu, N.shape[1], ZETA, 0.25)
                cfg = dict(alpha=p["alpha"], beta=p["beta"], gamma=p["gamma"])
            else:
                cfg = cfg_for(prob, t)
            W0 = state[m]
            Ws = b.exact_W(prob, W0)
            with np.errstate(all="ignore"):
                W = b.run_W(prob, W0, idx, cfg)
            sub[m].append(b.subopt_W(prob, W, Ws, W0))
            phantom[m].append(np.abs(W - Ws).sum(1))
            tv = np.abs(W - W0).sum(1)
            turn[m].append(tv)
            d, state[m] = b.hold(W, R[i:end])
            d[:, 0] = (1 + d[:, 0]) * (1 - b.COST_RATE * tv) - 1
            daily[m].append(d)
        if j % 50 == 0:
            print(f"date {j}/{len(reb)} [{time.time()-t0:.0f}s]")

    oos = slice(TUNE_DATES, None)
    out = dict(config=dict(long_only=args.long_only, theta=THETA, c=C, lam=LAM, K=K, batch=BATCH,
                           tune_dates=TUNE_DATES, tune_seeds=len(TUNE_SEEDS), eval_seeds=S, zeta=ZETA,
                           search_space_size=len(space)),
               tuning=table, ideal=dict({k: float(v[0]) for k, v in b.perf(ideal_daily[None]).items()},
                                        mean_monthly_turnover=float(ideal_turn.mean())),
               methods={}, comparisons={})
    arrays = {}
    month_ends = None
    for m in methods:
        D = np.concatenate(daily[m], 1)
        if month_ends is None:
            month_ends = np.cumsum([x.shape[1] for x in daily[m]]) - 1
        P = b.perf(D)
        SU = np.stack(sub[m], 1)
        PH = np.stack(phantom[m], 1)
        TV = np.stack(turn[m], 1)
        te = (D - ideal_daily[None]).std(1) * math.sqrt(252)
        su_o, ph_o = SU[:, oos].ravel(), PH[:, oos].ravel()
        tw = P["terminal_wealth"]
        out["methods"][m] = dict(
            subopt_oos=dict(mean=float(su_o.mean()), p95=float(np.quantile(su_o, 0.95)), cvar5=cvar(su_o),
                            evar=evar(su_o)),
            phantom_oos=dict(mean=float(ph_o.mean()), p95=float(np.quantile(ph_o, 0.95)), cvar5=cvar(ph_o)),
            terminal_wealth=dict(p05=float(np.quantile(tw, 0.05)), median=float(np.median(tw)),
                                 p95=float(np.quantile(tw, 0.95)),
                                 fan_ratio_p95_over_p05=float(np.quantile(tw, 0.95) / np.quantile(tw, 0.05))),
            tracking_error=b.dist(te), max_drawdown=b.dist(P["max_drawdown"]), sharpe=b.dist(P["sharpe"]),
            ann_return=b.dist(P["ann_return"]), monthly_turnover=float(TV.mean()),
            ruined=float(P["ruined"].mean()))
        arrays[f"{m}|log_wealth_monthly"] = np.cumsum(b.log_growth(D)[0], 1)[:, month_ends]
        arrays[f"{m}|phantom"] = PH
        arrays[f"{m}|subopt"] = SU
    A = "Worst-case-tuned (EV@R)"
    for ref in ("Average-tuned", "Nesterov (textbook)"):
        out["comparisons"][f"{A} minus {ref}"] = {
            "phantom_p95": paired_bootstrap(np.stack(phantom[A], 1)[:, oos], np.stack(phantom[ref], 1)[:, oos],
                                            lambda v: np.quantile(v, 0.95)),
            "phantom_cvar5": paired_bootstrap(np.stack(phantom[A], 1)[:, oos], np.stack(phantom[ref], 1)[:, oos], cvar),
            "subopt_p95": paired_bootstrap(np.stack(sub[A], 1)[:, oos], np.stack(sub[ref], 1)[:, oos],
                                           lambda v: np.quantile(v, 0.95)),
            "subopt_cvar5": paired_bootstrap(np.stack(sub[A], 1)[:, oos], np.stack(sub[ref], 1)[:, oos], cvar),
            "subopt_mean": paired_bootstrap(np.stack(sub[A], 1)[:, oos], np.stack(sub[ref], 1)[:, oos], np.mean),
        }
    arrays["ideal|log_wealth_monthly"] = np.cumsum(np.log1p(ideal_daily))[month_ends]
    arrays["rebalance_dates"] = np.array([str(dates[i].date()) for i in reb])
    tag = "worstcase_longonly" if args.long_only else "worstcase"
    o = pathlib.Path(args.out)
    o.mkdir(parents=True, exist_ok=True)
    (o / f"{tag}.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(o / f"{tag}_arrays.npz", **arrays)
    print(f"wrote {o / (tag + '.json')} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
