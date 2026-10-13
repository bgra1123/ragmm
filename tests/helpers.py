"""Shared helpers for the tests (numpy only)."""
import math

import numpy as np


def stationary_variances(eigs, alpha, beta, gamma, sigma2=1.0):
    """
    Stationary variance of x_i for noisy GMM on f(x) = 0.5 * sum_i eigs[i] x_i^2.

    Each coordinate is the linear system z_{k+1} = A z_k + b w_k, z = (x_k, x_{k-1}),
    w ~ N(0, sigma2), so the stationary state is Gaussian with covariance P solving
    P = A P A^T + sigma2 b b^T. Returns None if the system is unstable.
    """
    out = []
    for lam in eigs:
        A = np.array([[(1 + beta) - alpha * lam * (1 + gamma), -beta + alpha * lam * gamma], [1.0, 0.0]])
        if max(abs(np.linalg.eigvals(A))) >= 1.0:
            return None
        b = np.array([[-alpha], [0.0]])
        Q = sigma2 * b @ b.T
        P = np.linalg.solve(np.eye(4) - np.kron(A, A), Q.reshape(-1)).reshape(2, 2)
        out.append(P[0, 0])
    return np.array(out)


def exact_evar(eigs, variances, zeta):
    """
    Exact EV@R_{1-zeta}[f(x_inf) - f*] for the Gaussian stationary state:
    inf_{u>0} (log E exp(u f) - log zeta) / u,  E exp(u f) = prod (1 - u lam_i s_i)^(-1/2).
    """
    ls = np.asarray(eigs) * np.asarray(variances)
    u = (1.0 / ls.max()) * np.linspace(1e-5, 1 - 1e-7, 40000)
    log_mgf = -0.5 * np.log1p(-np.outer(u, ls)).sum(1)
    return float(((log_mgf + math.log(1.0 / zeta)) / u).min())
