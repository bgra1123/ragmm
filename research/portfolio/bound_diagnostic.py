"""
Diagnostic for horizon-aware RA-GMM on the rebalancing problem (exploratory).

At each rebalance date (along the exact-solution path), compare
  - the starting error f(x0) - f*                  (what a solver can gain at most),
  - the smallest noise term sigma^2 * min A(theta) of the EV@R bound (44),
  - the empirical 95th percentile of f(w_K) - f* for tuned SGD,
and record which step size the finite-horizon bound selects.

Run from the repository root:
    python research/portfolio/bound_diagnostic.py
"""
import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "model_multiplicity"))
import backtest as b  # noqa: E402
from paper_tuning import bound_parts, candidate_grid, horizon_aware_params  # noqa: E402

K, BATCH, ZETA = 200, 16, 0.05


def main():
    R, dates = b.load_returns()
    reb = b.rebalance_days(dates)
    N = b.null_basis(R.shape[1])
    args = SimpleNamespace(theta=10.0, c=1.0, lam=1.0)
    W0s, _, _ = b.ideal_path(R, reb, N, args)
    tuned_mult = json.loads((HERE / "results" / "backtest_main.json").read_text())["tuned"]["SGD (tuned for mean)"]["a"]
    rows = []
    for j, i in enumerate(reb):
        p = b.Problem(R, i, args.theta, args.c, args.lam, N)
        w0 = W0s[j][None]
        G = p.stoch_grad(np.repeat(w0, b.WINDOW, 0), np.repeat(w0, b.WINDOW, 0), np.arange(b.WINDOW)[:, None])
        sigma2 = G.var(0).mean() / BATCH                      # per-coordinate variance of a batch-16 gradient
        Zs, gz0 = p.exact(w0)
        f0 = float(p.subopt(np.zeros_like(Zs), Zs, gz0)[0])
        V0 = float((gz0 ** 2).sum() / p.mu)
        cand = candidate_grid(p.L, p.mu)
        A, rho = bound_parts(cand, p.L, p.mu, N.shape[1], ZETA)
        q = horizon_aware_params(cand, A, rho, np.array([sigma2]), np.array([V0]), K)
        # empirical tail of tuned SGD at this date (20 seeds)
        W0 = np.repeat(w0, 20, 0)
        idx = b.draw_indices(range(20), 200_000 + j, K, BATCH)
        Z = b.solve_gmm(p, W0, idx, tuned_mult / p.L, 0.0, 0.0)
        sub = p.subopt(Z, np.repeat(Zs, 20, 0), np.repeat(gz0, 20, 0))
        rows.append(dict(date=str(dates[i].date()), sigma2=float(sigma2), V0=V0, f0=f0,
                         min_noise_term=float(sigma2 * np.min(A)), chosen_alpha_times_L=float(q["alpha"][0] * p.L),
                         chosen_bound=float(q["bound"][0]), tuned_sgd_p95=float(np.quantile(sub, 0.95))))
    f0 = np.array([r["f0"] for r in rows])
    mn = np.array([r["min_noise_term"] for r in rows])
    tp = np.array([r["tuned_sgd_p95"] for r in rows])
    aL = np.array([r["chosen_alpha_times_L"] for r in rows])
    summary = dict(
        n_dates=len(rows),
        share_dates_noise_floor_above_start_error=float((mn > f0).mean()),
        median_ratio_noise_floor_to_start_error=float(np.median(mn / f0)),
        median_ratio_noise_floor_to_tuned_sgd_p95=float(np.median(mn / tp)),
        share_dates_tuned_sgd_p95_below_start_error=float((tp < f0).mean()),
        chosen_alpha_times_L_median=float(np.median(aL)), chosen_alpha_times_L_max=float(aL.max()),
        rows=rows)
    out = HERE / "results" / "bound_diagnostic.json"
    out.write_text(json.dumps(summary, indent=2))
    print({k: v for k, v in summary.items() if k != "rows"})


if __name__ == "__main__":
    main()
