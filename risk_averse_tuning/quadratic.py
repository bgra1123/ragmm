"""
Quadratic objectives: the paper's exact formulation (Can & Gürbüzbalaban 2025, Section 3) and an extension.

**The paper's formulation** (stationary regime, gradient noise ``N(0, sigma^2 I)``). For
``f(x) = 1/2 (x - x*)^T Q (x - x*) + f*`` with eigenvalues ``lambda_i`` of ``Q``, noisy GMM is the linear system
``z_{k+1} - z* = A_Q (z_k - z*) + B w_{k+1}`` (eqs. 15-16). With

    c_i = (1 + beta) - alpha (1 + gamma) lambda_i,        d_i = -(beta - alpha gamma lambda_i),
    rho_i = |c_i|/2 + sqrt(c_i^2 + 4 d_i)/2 if c_i^2 + 4 d_i >= 0 else sqrt(|d_i|)     (eq. 18),
    u_i = (1 + d_i) [(1 - d_i)^2 - c_i^2] / (lambda_i (1 - d_i) alpha^2)                (eq. 21),

* the stable set is ``S_q = {rho(A_Q) = max_i rho_i < 1}`` (eq. 23),
* the infinite-horizon entropic risk is ``r(theta) = -(sigma^2/theta) sum_i log(1 - theta / (2 u_i))`` (Prop. 2),
  finite iff ``theta < 2 min_i u_i``,
* ``EV@R_{1-zeta} = inf_{0 < theta < 2 min u_i} r(theta) + (2 sigma^2/theta) log(1/zeta)`` (eq. 12), a one-dimensional
  convex minimization, so **it is exactly computable**,
* Theorem 1 (eq. 29) bounds it in closed form, ``sigma^2/(2 theta_0 u_bar) [-d log(1 - theta_0) + 2 log(1/zeta)]``,
* problem (32) chooses ``(alpha, beta, gamma)`` in ``S_q`` minimizing that bound subject to
  ``rho^2 / rho_{q,*}^2 <= 1 + eps`` with ``rho_{q,*} = 1 - 2/sqrt(3 kappa + 1)``, by grid search.

``zeta`` is a tail probability (see ``risk.py``). The argmin of (32) does not depend on ``sigma``.

**Extension beyond the paper** (``evar_quadratic``, ``select_parameters``): the exact EV@R of ``f(x_K) - f*`` at a
*finite* number of steps ``K``, from a start away from the optimum, with any noise covariance ``Sigma``. This keeps the
transient (a slow method is penalized) and the actual noise level, neither of which the stationary formulation sees.
Its derivation is in ``evar_quadratic``'s docstring.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


class QuadraticSelection(NamedTuple):
    alpha: float
    beta: float
    gamma: float
    evar: float          # exact EV@R of f(x_K) - f* at the selected parameters
    mean: float          # exact E[f(x_K) - f*]
    candidates: int      # number of stable candidates evaluated


# ---------------------------------------------------------------------- #
# The paper's quadratic formulation (Section 3)                            #
# ---------------------------------------------------------------------- #

def _as_grid(alpha, beta, gamma, eigs):
    a, b, g = (np.asarray(v, float)[..., None] for v in (alpha, beta, gamma))
    return a, b, g, np.asarray(eigs, float)


def quad_cd(alpha, beta, gamma, eigs):
    """``c_i`` and ``d_i`` of Lemma 3.1. Parameters broadcast; ``eigs`` is the last axis."""
    a, b, g, lam = _as_grid(alpha, beta, gamma, eigs)
    return (1 + b) - a * (1 + g) * lam, -(b - a * g * lam)


def quad_spectral_radius(alpha, beta, gamma, eigs):
    """``rho(A_Q) = max_i rho_i`` with ``rho_i`` of eq. (18)."""
    c, d = quad_cd(alpha, beta, gamma, eigs)
    disc = c ** 2 + 4 * d
    with np.errstate(invalid="ignore"):
        rho_i = np.where(disc >= 0, 0.5 * np.abs(c) + 0.5 * np.sqrt(np.where(disc >= 0, disc, 0.0)), np.sqrt(np.abs(d)))
    return rho_i.max(axis=-1)


def quad_u(alpha, beta, gamma, eigs):
    """``u_i`` of eq. (21); the entropic risk is finite iff ``theta < 2 min_i u_i``."""
    a, b, g, lam = _as_grid(alpha, beta, gamma, eigs)
    c, d = quad_cd(alpha, beta, gamma, eigs)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1 + d) * ((1 - d) ** 2 - c ** 2) / (lam * (1 - d) * a ** 2)


def quad_in_stable_set(alpha, beta, gamma, eigs):
    """Membership in ``S_q`` (eq. 23): ``|c_i| < |1 - d_i|`` and ``u_i > 0`` for all i."""
    c, d = quad_cd(alpha, beta, gamma, eigs)
    return np.all(np.abs(c) < np.abs(1 - d), axis=-1) & np.all(quad_u(alpha, beta, gamma, eigs) > 0, axis=-1)


def quad_entropic_risk(theta, alpha, beta, gamma, eigs, sigma2=1.0):
    """Infinite-horizon entropic risk ``r_{sigma^2}(theta)`` of Proposition 2 (``inf`` if theta >= 2 min u_i)."""
    u = quad_u(alpha, beta, gamma, eigs)
    x = theta / (2 * u)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = -(sigma2 / theta) * np.sum(np.log1p(-x), axis=-1)
    return np.where(np.all(x < 1, axis=-1) & np.all(u > 0, axis=-1), out, np.inf)


def quad_evar_bound(alpha, beta, gamma, eigs, zeta=0.05, sigma2=1.0):
    """Closed-form bound of Theorem 1, eq. (29); ``inf`` outside ``S_q``."""
    u = quad_u(alpha, beta, gamma, eigs)
    d = len(np.asarray(eigs))
    log_z = math.log(1.0 / zeta)
    theta0 = log_z / d * (math.sqrt(1 + 2 * d / log_z) - 1)
    u_bar = u.min(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        bound = sigma2 / (2 * theta0 * u_bar) * (-d * math.log1p(-theta0) + 2 * log_z)
    ok = quad_in_stable_set(alpha, beta, gamma, eigs) & (u_bar > 0)
    return np.where(ok, bound, np.inf)


def quad_evar_exact(alpha, beta, gamma, eigs, zeta=0.05, sigma2=1.0, n_grid=80, refine=40):
    """
    Exact stationary ``EV@R_{1-zeta}[f(x_inf) - f*]``: ``inf_theta r(theta) + (2 sigma^2/theta) log(1/zeta)``
    (eq. 12) over ``0 < theta < 2 min_i u_i`` (convex), by a grid followed by golden-section refinement.
    Parameters may be arrays (broadcast); ``inf`` outside ``S_q``.
    """
    u = quad_u(alpha, beta, gamma, eigs)
    shape = u.shape[:-1]
    u = u.reshape(-1, u.shape[-1])
    umin = u.min(axis=1)
    ok = np.all(u > 0, axis=1) & np.isfinite(umin)
    log_z = math.log(1.0 / zeta)
    theta_hi = np.where(ok, 2 * umin, 1.0)

    def value(frac):                                               # theta = frac * 2 umin, frac in (0, 1)
        theta = (frac * theta_hi)[:, None] if np.ndim(frac) else frac * theta_hi[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            risk = -(sigma2 / theta[:, 0]) * np.sum(np.log1p(-theta / (2 * u)), axis=1)
        return risk + 2 * sigma2 / theta[:, 0] * log_z

    fracs = np.geomspace(1e-5, 1 - 1e-7, n_grid)
    vals = np.stack([value(np.full(len(u), f)) for f in fracs], axis=1)
    i = np.argmin(vals, axis=1)
    lo = fracs[np.clip(i - 1, 0, n_grid - 1)]
    hi = fracs[np.clip(i + 1, 0, n_grid - 1)]
    for _ in range(refine):
        a_, b_ = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        better_a = value(a_) < value(b_)
        hi = np.where(better_a, b_, hi)
        lo = np.where(better_a, lo, a_)
    best = np.minimum(vals.min(axis=1), value(0.5 * (lo + hi)))
    return np.where(ok, best, np.inf).reshape(shape)


def rho_star_quadratic(L: float, mu: float) -> float:
    """``rho_{q,*} = 1 - 2/sqrt(3 kappa + 1)``: the fastest rate of AGD on quadratics, the benchmark of (32b)."""
    return 1.0 - 2.0 / math.sqrt(3.0 * L / mu + 1.0)


class QuadraticSolution(NamedTuple):
    alpha: float
    beta: float
    gamma: float
    objective: float       # Theorem 1 bound (or the exact EV@R if use_exact) for sigma^2 = 1
    rate: float            # rho(A_Q)
    rate_ratio: float      # rho^2 / rho_{q,*}^2
    candidates: int


def solve_quadratic(eigs, zeta=0.05, rate_slack=0.25, grid_size=50, alpha_max=None, use_exact=False):
    """
    The paper's problem (32) by grid search, as in the authors' reference code: ``alpha`` in ``(0, 1/L]`` and
    ``beta, gamma`` in ``[0, 1]`` on ``grid_size`` points each, keep ``rho(A_Q) < 1`` and
    ``rho^2 <= (1 + rate_slack) rho_{q,*}^2``, minimize the Theorem 1 bound. ``use_exact=True`` minimizes the exact
    stationary EV@R instead (the paper computes it for illustration only).
    """
    lam = np.asarray(eigs, float)
    L, mu = float(lam.max()), float(lam.min())
    if not L > mu > 0:
        raise ValueError(f"need L > mu > 0, got L={L}, mu={mu}")
    alphas = np.linspace(0.0, (alpha_max or 1.0 / L), grid_size)[1:]
    betas = np.linspace(0.0, 1.0, grid_size)
    A, Bm, G = np.meshgrid(alphas, betas, betas, indexing="ij")           # alpha-major, as in the reference loop
    rho = quad_spectral_radius(A, Bm, G, lam)
    rho2_star = rho_star_quadratic(L, mu) ** 2
    feasible = (rho < 1.0) & (rho ** 2 <= (1.0 + rate_slack) * rho2_star)
    feasible &= quad_in_stable_set(A, Bm, G, lam)
    if not feasible.any():
        raise ValueError("no feasible parameters; increase rate_slack or grid_size")
    idx = np.flatnonzero(feasible.ravel())
    a, b, g = A.ravel()[idx], Bm.ravel()[idx], G.ravel()[idx]
    obj = (quad_evar_exact(a, b, g, lam, zeta) if use_exact else quad_evar_bound(a, b, g, lam, zeta))
    obj = np.where(np.isfinite(obj), obj, np.inf)
    j = int(np.argmin(obj))
    if not np.isfinite(obj[j]):
        raise ValueError("no feasible parameters with a finite objective")
    return QuadraticSolution(float(a[j]), float(b[j]), float(g[j]), float(obj[j]), float(rho.ravel()[idx][j]),
                             float(rho.ravel()[idx][j] ** 2 / rho2_star), int(len(idx)))



# ---------------------------------------------------------------------- #
# Extension: finite horizon, any noise covariance, start away from the optimum #
# ---------------------------------------------------------------------- #

def gmm_matrices(H: np.ndarray, alpha: float, beta: float, gamma: float):
    m = H.shape[0]
    eye = np.eye(m)
    A = np.block([[(1 + beta) * eye - alpha * (1 + gamma) * H, -beta * eye + alpha * gamma * H],
                  [eye, np.zeros((m, m))]])
    B = np.vstack([-alpha * eye, np.zeros((m, m))])
    return A, B


def moments_after(A: np.ndarray, B: np.ndarray, Sigma: np.ndarray, s0: np.ndarray, steps: int):
    """Mean and covariance of s_K after ``steps`` steps from the deterministic start ``s0`` (binary doubling)."""
    q = B @ Sigma @ B.T
    p_acc, a_acc = np.zeros_like(q), np.eye(A.shape[0])        # moments after the steps combined so far
    p_blk, a_blk, n = q.copy(), A.copy(), steps                # block of 2^i steps
    while n:
        if n & 1:
            p_acc = p_blk + a_blk @ p_acc @ a_blk.T
            a_acc = a_blk @ a_acc
        p_blk = p_blk + a_blk @ p_blk @ a_blk.T
        a_blk = a_blk @ a_blk
        n >>= 1
    return a_acc @ s0, p_acc


def _log_mgf(u: float, M: np.ndarray, mu: np.ndarray, P: np.ndarray) -> float:
    n = len(mu)
    c = np.eye(n) - u * (M @ P)
    sign, logdet = np.linalg.slogdet(c)
    if sign <= 0:
        return math.inf
    return float(-0.5 * logdet + 0.5 * u * mu @ np.linalg.solve(c, M @ mu))


def evar_quadratic(H, alpha, beta, gamma, steps, Sigma, e0, zeta=0.05, return_mean=False):
    """
    Exact EV@R_{1-zeta} of f(x_K) - f* after ``steps`` GMM steps (see module docstring).

    H: Hessian (m x m, symmetric positive definite). Sigma: gradient-noise covariance (m x m).
    e0: starting error x_0 - x* (also x_{-1} = x_0). Returns ``inf`` if the iteration is unstable.
    """
    H = np.asarray(H, float)
    A, B = gmm_matrices(H, alpha, beta, gamma)
    if max(abs(np.linalg.eigvals(A))) >= 1.0 + 1e-9:
        return (math.inf, math.inf) if return_mean else math.inf
    e0 = np.asarray(e0, float)
    mu, P = moments_after(A, B, np.asarray(Sigma, float), np.concatenate([e0, e0]), steps)
    m = H.shape[0]
    M = np.zeros((2 * m, 2 * m))
    M[:m, :m] = H
    mean = 0.5 * float(np.trace(M @ P) + mu @ M @ mu)
    lam_max = float(np.max(np.linalg.eigvals(M @ P).real))
    if lam_max <= 1e-300:                                     # noise-free and already at the optimum
        value = 0.5 * float(mu @ M @ mu)
        return (value, mean) if return_mean else value
    log_zeta = math.log(zeta)

    def objective(frac):                                       # u = frac / lam_max, frac in (0, 1)
        u = frac / lam_max
        return (_log_mgf(u, M, mu, P) - log_zeta) / u

    grid = np.geomspace(1e-5, 1.0 - 1e-7, 120)
    vals = np.array([objective(f) for f in grid])
    i = int(np.argmin(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    for _ in range(60):                                        # golden-section refinement of a convex 1-d problem
        a, b = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if objective(a) < objective(b):
            hi = b
        else:
            lo = a
    best = min(float(vals[i]), objective(0.5 * (lo + hi)))
    return (best, mean) if return_mean else best


def select_parameters(H, steps, Sigma, e0, zeta=0.05, alpha_range=(1e-4, 2.0), n_alpha=24,
                      betas=(0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98), lipschitz=None) -> QuadraticSelection:
    """
    Choose (alpha, beta, gamma) minimizing the exact EV@R of f(x_K) - f*.

    The search covers gradient descent (beta = gamma = 0), heavy ball (gamma = 0) and Nesterov (gamma = beta) on
    a log grid of alpha = a / L, ``a`` in ``alpha_range``. Because the EV@R is evaluated at ``steps`` with the
    actual noise ``Sigma`` and start ``e0``, no rate constraint is needed: a slow method is penalized by its
    transient, a fast one by the noise it amplifies.
    """
    L = float(np.linalg.eigvalsh(np.asarray(H, float)).max()) if lipschitz is None else lipschitz
    best, count = None, 0
    for a in np.geomspace(alpha_range[0], alpha_range[1], n_alpha):
        alpha = a / L
        for beta in betas:
            for gamma in ((0.0,) if beta == 0.0 else (0.0, beta)):
                value, mean = evar_quadratic(H, alpha, beta, gamma, steps, Sigma, e0, zeta, return_mean=True)
                if not math.isfinite(value):
                    continue
                count += 1
                if best is None or value < best[3]:
                    best = (alpha, beta, gamma, value, mean)
    if best is None:
        raise ValueError("no stable candidate found; widen alpha_range")
    return QuadraticSelection(*best, candidates=count)
