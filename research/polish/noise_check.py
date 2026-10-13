"""Gaussian vs real minibatch noise in the linearized polish model. Plan: ADAPTER_PLAN.md section 8."""
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import polish_experiments as pe  # noqa: E402
from polish_experiments import le, g0  # noqa: E402
from risk_averse_tuning.quadratic import evar_quadratic, gmm_matrices  # noqa: E402
from risk_averse_tuning.tail_report import empirical_evar  # noqa: E402

N_PATHS, B = 10_000, pe.BATCH


def simulate(H, alpha, beta, gamma, Gc, K, rng):
    d = H.shape[0]
    A, Bm = gmm_matrices(H, alpha, beta, gamma)
    s = np.zeros((N_PATHS, 2 * d))
    At = A.T.copy()
    n = len(Gc)
    for _ in range(K):
        w = Gc[rng.integers(0, n, size=(N_PATHS, B))].mean(1)
        s = s @ At
        s[:, :d] += w @ Bm[:d, :d].T
    z = s[:, :d]
    return 0.5 * np.einsum("ni,ij,nj->n", z, H, z)


def main():
    rows = []
    for name in ("compas", "heart", "german"):
        X, y, _, _ = g0.prepare(name)
        L, mu = le.constants(X, pe.LAM)
        wstar, _ = le.solve_exact(X, y, pe.LAM)
        H = pe.hessian(X, y, wstar)
        G = -(y * le.sigmoid(-y * (X @ wstar)))[:, None] * X
        Gc = G - G.mean(0)
        Sig = Gc.T @ Gc / len(Gc) / B
        rng = np.random.default_rng(0)
        for c in (1.0, 0.3, 0.1, 0.03):
            for (b, g) in ((0.0, 0.0), (0.9, 0.9)):
                for K in (200, 1000):
                    sim = simulate(H, c / L, b, g, Gc, K, rng)
                    emp = empirical_evar(sim, 0.05)
                    gau = evar_quadratic(H, c / L, b, g, K, Sig, np.zeros(H.shape[0]), 0.05)
                    k4 = float(((sim - sim.mean()) ** 4).mean() / sim.var() ** 2)
                    rows.append(dict(dataset=name, alpha_over_invL=c, beta=b, K=K, gaussian_evar=gau, real_evar=emp,
                                     ratio=emp / gau, p95=float(np.quantile(sim, .95)), kurtosis=k4,
                                     window=1 / (c / L * mu) if b == 0 else None))
                    print(rows[-1], flush=True)
    (HERE / "results" / "noise_check.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
