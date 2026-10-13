"""Tests for risk_averse_tuning.quadratic: the paper's Section 3 formulation and the finite-horizon extension."""
import math

import numpy as np
import pytest

from risk_averse_tuning import quadratic as q
from tests.helpers import exact_evar, stationary_variances

RNG = np.random.default_rng(0)


def random_stable(eigs, n=200):
    out = []
    while len(out) < n:
        a, b, g = RNG.uniform(0, 1.5 / max(eigs)), RNG.uniform(0, 1), RNG.uniform(0, 1)
        if q.quad_in_stable_set(a, b, g, eigs):
            out.append((a, b, g))
    return out


EIGS = np.array([0.1, 0.5, 1.0, 2.0])


def test_rho_equals_spectral_radius_of_AQ():
    for _ in range(300):
        a, b, g = RNG.uniform(0, 2), RNG.uniform(0, 1.2), RNG.uniform(0, 1.2)
        lam = RNG.uniform(0.05, 3, size=4)
        rho = 0.0
        for l in lam:
            A = np.array([[(1 + b) - a * l * (1 + g), -b + a * l * g], [1.0, 0.0]])
            rho = max(rho, max(abs(np.linalg.eigvals(A))))
        assert q.quad_spectral_radius(a, b, g, lam) == pytest.approx(rho, abs=1e-9)


def test_stable_set_is_spectral_radius_below_one():
    agree = 0
    for _ in range(1000):
        a, b, g = RNG.uniform(0, 2), RNG.uniform(0, 1.2), RNG.uniform(0, 1.2)
        agree += bool(q.quad_in_stable_set(a, b, g, EIGS)) == bool(q.quad_spectral_radius(a, b, g, EIGS) < 1)
    assert agree == 1000


def test_prop2_matches_stationary_lyapunov_solution():
    for a, b, g in random_stable(EIGS, 40):
        s = stationary_variances(EIGS, a, b, g)
        assert q.quad_u(a, b, g, EIGS) == pytest.approx(1.0 / (EIGS * s), rel=1e-7)
        theta = 0.4 * 2 * q.quad_u(a, b, g, EIGS).min()
        direct = -(1.0 / theta) * np.sum(np.log1p(-theta / (2 * q.quad_u(a, b, g, EIGS))))
        assert q.quad_entropic_risk(theta, a, b, g, EIGS) == pytest.approx(direct)


def test_exact_evar_matches_independent_computation():
    for a, b, g in random_stable(EIGS, 25):
        for zeta in (0.05, 0.5):
            ref = exact_evar(EIGS, stationary_variances(EIGS, a, b, g), zeta)
            assert q.quad_evar_exact(a, b, g, EIGS, zeta) == pytest.approx(ref, rel=2e-3)


def test_theorem1_bound_dominates_exact_and_is_finite_only_in_Sq():
    for a, b, g in random_stable(EIGS, 100):
        exact = q.quad_evar_exact(a, b, g, EIGS, 0.05)
        assert q.quad_evar_bound(a, b, g, EIGS, 0.05) >= exact * (1 - 1e-9)
    assert np.isinf(q.quad_evar_bound(1.9 / EIGS.max() * 1.5, 0.0, 0.0, EIGS))


def test_sigma2_scaling():
    a, b, g = random_stable(EIGS, 1)[0]
    assert q.quad_evar_bound(a, b, g, EIGS, 0.05, sigma2=4.0) == pytest.approx(4 * q.quad_evar_bound(a, b, g, EIGS, 0.05))
    assert q.quad_evar_exact(a, b, g, EIGS, 0.05, sigma2=4.0) == pytest.approx(4 * q.quad_evar_exact(a, b, g, EIGS, 0.05))


def test_solve_quadratic_feasible_and_optimal_on_its_grid():
    sol = q.solve_quadratic(EIGS, 0.05, rate_slack=0.25, grid_size=30)
    assert q.quad_in_stable_set(sol.alpha, sol.beta, sol.gamma, EIGS)
    assert sol.rate_ratio <= 1.25 + 1e-12
    assert sol.rate == pytest.approx(q.quad_spectral_radius(sol.alpha, sol.beta, sol.gamma, EIGS))
    assert sol.objective == pytest.approx(q.quad_evar_bound(sol.alpha, sol.beta, sol.gamma, EIGS))
    grid = np.linspace(0, 1, 30)
    for a in np.linspace(0, 1 / EIGS.max(), 30)[1:][::3]:
        for b in grid[::3]:
            for g in grid[::3]:
                rho = q.quad_spectral_radius(a, b, g, EIGS)
                if rho < 1 and rho ** 2 <= 1.25 * q.rho_star_quadratic(2.0, 0.1) ** 2:
                    assert q.quad_evar_bound(a, b, g, EIGS) >= sol.objective - 1e-12


def test_solve_quadratic_rejects_bad_spectrum():
    with pytest.raises(ValueError):
        q.solve_quadratic([1.0, 1.0])
    with pytest.raises(ValueError):
        q.solve_quadratic([-1.0, 1.0])


# ---------------------------- extension ---------------------------- #

def test_finite_horizon_converges_to_stationary_for_long_horizon():
    a, b, g = 0.5, 0.5, 0.5
    H = np.diag(EIGS)
    ev = q.evar_quadratic(H, a, b, g, steps=3000, Sigma=np.eye(4), e0=np.zeros(4), zeta=0.05)
    assert ev == pytest.approx(q.quad_evar_exact(a, b, g, EIGS, 0.05), rel=2e-3)


def test_finite_horizon_matches_monte_carlo():
    H = np.diag([0.2, 1.0])
    a, b, g, K = 0.8, 0.6, 0.3, 15
    Sigma, e0 = np.diag([0.3, 0.1]), np.array([1.0, -0.5])
    ev, mean = q.evar_quadratic(H, a, b, g, K, Sigma, e0, 0.1, return_mean=True)
    rng = np.random.default_rng(1)
    n = 400_000
    x, xp = np.tile(e0, (n, 1)), np.tile(e0, (n, 1))
    d = np.zeros_like(x)
    for _ in range(K):
        y = x + g * d
        dn = b * d - a * (y @ H + rng.normal(size=(n, 2)) * np.sqrt(np.diag(Sigma)))
        x = x + dn
        d = dn
    f = 0.5 * np.einsum("ni,ij,nj->n", x, H, x)
    assert mean == pytest.approx(f.mean(), rel=0.01)
    assert ev >= np.quantile(f, 0.9) * 0.98                      # EV@R is above the VaR at the same level
    assert ev >= np.mean(np.sort(f)[-n // 10:]) * 0.98           # ... and above the CVaR (EV@R >= CVaR)


def test_select_parameters_trades_speed_for_noise():
    H = np.diag(EIGS)
    e0 = np.array([0, 0, 0, 0.0]) + 3.0
    quiet = q.select_parameters(H, 40, 1e-8 * np.eye(4), e0, 0.05)
    noisy = q.select_parameters(H, 40, 1.0 * np.eye(4), e0, 0.05)
    assert noisy.alpha < quiet.alpha                              # noise calls for smaller steps
    assert noisy.evar > 0 and quiet.candidates > 10


def test_zoom_finds_feasible_point_missed_by_coarse_grid():
    eigs = np.array([2.3, 6.5])                                   # kappa ~ 2.8: optimum needs alpha > 1/L
    with pytest.raises(ValueError):
        q.solve_quadratic(eigs, 0.05, 0.25, 50)
    sol = q.solve_quadratic(eigs, 0.05, 0.25, 50, alpha_max=2 / 6.5, zoom=3)
    assert sol.rate_ratio <= 1.25 + 1e-9 and q.quad_in_stable_set(sol.alpha, sol.beta, sol.gamma, eigs)
    plain = q.solve_quadratic(eigs, 0.05, 1.0, 50)
    zoomed = q.solve_quadratic(eigs, 0.05, 1.0, 50, zoom=3)
    assert zoomed.objective <= plain.objective + 1e-12            # zooming never makes the incumbent worse
