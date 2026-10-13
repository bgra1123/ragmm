"""
P5 diagnostic (PLAN.md section 13, addendum): calibration of the exact-K EV@R when the gradient noise is the paper's
additive Gaussian N(0, Sigma) instead of the real minibatch noise. Run after quadratic_formulation.py:
    python research/portfolio/calibration_additive.py
"""
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import backtest as bt  # noqa: E402
import quadratic_formulation as qf  # noqa: E402
from risk_averse_tuning import quadratic as q  # noqa: E402

THETA, C, LAM, K, BATCH, ZETA, SEEDS = 10.0, 1.0, 1.0, 200, 16, 0.05, 2000


def simulate_additive(H, alpha, beta, gamma, Sigma, e0, rng, n):
    m = H.shape[0]
    Lc = np.linalg.cholesky(Sigma + 1e-18 * np.eye(m))
    x = np.tile(e0, (n, 1))
    d = np.zeros_like(x)
    for _ in range(K):
        y = x + gamma * d
        d = beta * d - alpha * (y @ H + rng.normal(size=(n, m)) @ Lc.T)
        x = x + d
    return 0.5 * np.einsum("ni,ij,nj->n", x, H, x)


def main():
    main_json = json.loads((HERE / "results" / "quadratic_formulation.json").read_text())
    R, dates = bt.load_returns()
    reb = bt.rebalance_days(dates)
    N = bt.null_basis(R.shape[1])
    qf.G["batch"] = BATCH
    args = type("A", (), dict(theta=THETA, c=C, lam=LAM, K=K, zeta=ZETA))()
    ideal_W0, _, _ = bt.ideal_path(R, reb, N, args)
    qf.G.update(args=args, R=R, reb=reb, N=N, ideal_W0=ideal_W0, batch=BATCH)
    rng = np.random.default_rng(0)
    rows = []
    for row in main_json["calibration"]:
        j = row["date"]
        prob = bt.Problem(R, reb[j], THETA, C, LAM, N)
        Sigma, e0 = qf.noise_cov(prob, ideal_W0[j][None])
        res = dict(date=j, kappa=row["kappa"])
        pj = qf.params_for_date(j)
        for nm in ("Nesterov (textbook)", "Q32 (eps=0.25)", "EXT (exact-K)"):
            kappa = prob.L / prob.mu
            if nm == "Nesterov (textbook)":
                m = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
                c = dict(alpha=1 / prob.L, beta=m, gamma=m)
            else:
                c = pj[nm]
            sub = simulate_additive(prob.Hz, c["alpha"], c["beta"], c["gamma"], Sigma, e0, rng, SEEDS)
            res[nm] = dict(additive_empirical_evar=qf.empirical_evar(sub, ZETA),
                           predicted_exact_K_evar=row[nm]["predicted_exact_K_evar"],
                           real_minibatch_empirical_evar=row[nm]["empirical_evar"])
        rows.append(res)
    (HERE / "results" / "calibration_additive.json").write_text(json.dumps(rows, indent=2))
    print("wrote results/calibration_additive.json")


if __name__ == "__main__":
    main()
