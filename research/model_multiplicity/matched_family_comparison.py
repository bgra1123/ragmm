"""
Experiment 1c: matched-family comparison (SGD family beta = gamma = 0, AGD family gamma = beta).

Pre-registered design: PLAN.md section 8.

Run from the repository root:
    python research/model_multiplicity/matched_family_comparison.py
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from loan_experiment import (COST_THRESHOLD, batch_indices, constants, evaluate, load_data,  # noqa: E402
                             objective, paired_bootstrap, run_gmm, solve_exact)
from risk_averse_tuning import RAGMMBounds  # noqa: E402

LAM, K, BATCH, SEEDS, ZETA = 0.01, 2000, 16, 200, 0.05


def agd_beta(alpha, mu):
    return (1 - math.sqrt(alpha * mu)) / (1 + math.sqrt(alpha * mu))


def tune(grid, X, y, fstar):
    idx = batch_indices(range(10_000, 10_020), K, X.shape[0], BATCH)
    scores = []
    for cfg in grid:
        W, _ = run_gmm(X, y, LAM, idx, **cfg)
        sub = objective(W, X, y, LAM) - fstar
        scores.append(np.mean(sub) if np.all(np.isfinite(sub)) else np.inf)
    return grid[int(np.argmin(scores))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid_lo", type=float, default=0.01, help="lowest step-size multiplier of the tuning grid (x 1/L)")
    ap.add_argument("--grid_points", type=int, default=12)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    t0 = time.time()
    Xtr, ytr, Xte, yte = load_data()
    n, d = Xtr.shape
    L, mu = constants(Xtr, LAM)
    wstar, fstar = solve_exact(Xtr, ytr, LAM)
    kappa = L / mu
    m_text = (math.sqrt(kappa) - 1) / (math.sqrt(kappa) + 1)
    mults = np.logspace(math.log10(args.grid_lo), math.log10(2.0), args.grid_points)
    bounds = RAGMMBounds(zeta=ZETA)

    sgd_tuned = tune([dict(alpha=a / L, beta=0.0, gamma=0.0) for a in mults], Xtr, ytr, fstar)
    agd_tuned = tune([dict(alpha=a / L, beta=agd_beta(a / L, mu), gamma=agd_beta(a / L, mu)) for a in mults],
                     Xtr, ytr, fstar)
    methods = {
        "SGD (textbook)": ("beta = gamma = 0", dict(alpha=1 / L, beta=0.0, gamma=0.0)),
        "SGD (tuned for mean)": ("beta = gamma = 0", sgd_tuned),
        "AGD (textbook)": ("gamma = beta", dict(alpha=1 / L, beta=m_text, gamma=m_text)),
        "AGD, beta tied to alpha (tuned for mean)": ("gamma = beta", agd_tuned),
    }
    solutions = {}
    for eps in (0.05, 0.25):
        sol = bounds.solve(L, mu, d, rate_slack=eps, method="agd")
        solutions[f"RA-AGD (eps={eps})"] = sol
        methods[f"RA-AGD (eps={eps})"] = ("gamma = beta", dict(alpha=sol.alpha, beta=sol.beta, gamma=sol.gamma))
    sol = bounds.solve(L, mu, d, rate_slack=0.05)
    solutions["RA-GMM (eps=0.05)"] = sol
    methods["RA-GMM (eps=0.05)"] = ("general", dict(alpha=sol.alpha, beta=sol.beta, gamma=sol.gamma))

    idx = batch_indices(range(SEEDS), K, n, BATCH)
    out = dict(config=dict(lam=LAM, K=K, batch=BATCH, seeds=SEEDS, zeta=ZETA, L=L, mu=mu, kappa=kappa, d=d,
                           n_train=n, n_test=int(len(yte)), tuning_grid_lo=args.grid_lo,
                           tuning_grid_points=args.grid_points), methods={})
    arrays = {}
    for name, (family, cfg) in methods.items():
        W, _ = run_gmm(Xtr, ytr, LAM, idx, **cfg)
        metrics, arr = evaluate(W, Xtr, ytr, Xte, yte, LAM, fstar, wstar, COST_THRESHOLD)
        entry = dict(family=family, params={k: float(v) for k, v in cfg.items()}, metrics=metrics)
        if name in solutions:
            s = solutions[name]
            entry["selection"] = dict(vartheta=s.vartheta, psi=s.psi, rate_ratio=s.rate_ratio, region=s.region,
                                      evar_bound_sigma1=s.evar_bound)
        if cfg["beta"] == 0.0 and cfg["gamma"] == 0.0:
            entry["gd_evar_bound_sigma1"] = bounds.gd_evar_bound(cfg["alpha"], L, mu, d=d)
        out["methods"][name] = entry
        arrays[name] = arr
        print(f"{name:42s} alpha*L={cfg['alpha'] * L:8.4f} beta={cfg['beta']:.4f} gamma={cfg['gamma']:.4f}  "
              f"flips mean {metrics['flips_per_retrain']['mean']:6.1f} p95 {metrics['flips_per_retrain']['p95']:6.1f}  "
              f"S p95 {metrics['suboptimality']['p95']:.2e}  cost {metrics['cost_per_applicant']['mean']:.3f}  "
              f"[{time.time() - t0:.0f}s]")

    comps = {}
    for fam_a, fam_b in [("RA-AGD (eps=0.05)", "AGD (textbook)"),
                         ("RA-AGD (eps=0.05)", "AGD, beta tied to alpha (tuned for mean)"),
                         ("RA-AGD (eps=0.25)", "AGD (textbook)"),
                         ("SGD (tuned for mean)", "SGD (textbook)"),
                         ("AGD, beta tied to alpha (tuned for mean)", "AGD (textbook)")]:
        comps[f"{fam_a} minus {fam_b}"] = dict(
            flips_mean=paired_bootstrap(arrays[fam_a]["F"], arrays[fam_b]["F"], np.mean),
            flips_p95=paired_bootstrap(arrays[fam_a]["F"], arrays[fam_b]["F"], lambda v: np.quantile(v, 0.95)),
            subopt_p95=paired_bootstrap(arrays[fam_a]["sub"], arrays[fam_b]["sub"], lambda v: np.quantile(v, 0.95)))
    out["comparisons"] = comps
    (HERE / "results").mkdir(exist_ok=True)
    name = f"matched_family{args.suffix}.json"
    (HERE / "results" / name).write_text(json.dumps(out, indent=2))
    print(f"wrote results/{name} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
