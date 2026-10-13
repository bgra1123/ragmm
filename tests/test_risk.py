import importlib.util
import math
import pathlib

import numpy as np
import pytest

from risk_averse_tuning import RAGMMBounds

from helpers import exact_evar, stationary_variances

L, MU = 1.0, 0.1


def bounds(zeta=0.05):
    return RAGMMBounds(zeta=zeta)


# ---------------------------------------------------------------------- #
# Parametrization (Theorem 2, Remarks 3 and 4)                             #
# ---------------------------------------------------------------------- #

def test_beta_matches_hand_computation_of_eq_35b():
    # L=1, mu=0.1, vartheta=0.9, psi=0.5: alpha=0.2; beta = (1-sqrt(.9*.2*.1))/(1-.2*.5*.1) * (1-sqrt(.2*.1/.9))
    alpha, beta, gamma = RAGMMBounds.parameters(0.9, 0.5, L, MU)
    assert alpha == pytest.approx(0.2)
    assert beta == pytest.approx(0.744199, abs=1e-5)
    assert gamma == pytest.approx(0.5 * beta)


@pytest.mark.parametrize("alpha", [0.05, 0.4, 1.0])
def test_agd_special_case(alpha):                              # Remark 3: (vartheta, psi) = (1, 1)
    a, beta, gamma = RAGMMBounds.parameters(1.0, 1.0, L, MU, alpha=alpha)
    expected = (1 - math.sqrt(alpha * MU)) / (1 + math.sqrt(alpha * MU))
    assert (a, beta, gamma) == (alpha, pytest.approx(expected), pytest.approx(expected))
    assert RAGMMBounds.rate(1.0, alpha, MU) == pytest.approx(1 - math.sqrt(alpha * MU))


def test_heavy_ball_special_case():                            # Remark 4: psi = 0 gives gamma = 0, alpha = (1-vartheta)/L
    alpha, beta, gamma = RAGMMBounds.parameters(0.7, 0.0, L, MU)
    assert alpha == pytest.approx(0.3 / L) and gamma == 0.0
    assert beta == pytest.approx((1 - math.sqrt(0.7 * alpha * MU)) * (1 - math.sqrt(alpha * MU / 0.7)))


def test_vartheta_alpha_round_trip():
    for alpha, psi in [(0.3, 0.2), (0.05, 3.0), (0.9, 0.0)]:
        v = RAGMMBounds.vartheta_of(alpha, psi, L)
        assert RAGMMBounds.alpha_of(v, psi, L) == pytest.approx(alpha)


# ---------------------------------------------------------------------- #
# Stable sets (Theorem 2)                                                  #
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize("vartheta,psi,expected", [
    (0.9, 0.5, True),        # S_-
    (0.95, 0.9, True),
    (1.05, 1.5, True),       # S_+ (psi > 1): the old feasibility test could not represent this region
    (0.8, 0.2, False),       # in S_- but violates the S_1 inequality
    (1.4, 1.5, False),       # S_+ requires vartheta <= 2 - 1/psi = 1.333
    (1.0, 0.5, False),       # S_- requires vartheta < 1
    (0.9, -0.1, False),      # psi >= 0
    (0.05, 0.5, False),      # below the S_- lower bound
])
def test_stable_set_membership(vartheta, psi, expected):
    assert bounds().in_stable_set(vartheta, psi, L, MU) is expected


def test_s0_requires_alpha_in_zero_one_over_L():
    b = bounds()
    assert b.in_stable_set(1.0, 1.0, L, MU, alpha=1.0 / L)
    assert b.in_stable_set(1.0, 1.0, L, MU, alpha=0.2)
    assert not b.in_stable_set(1.0, 1.0, L, MU, alpha=1.5 / L)
    assert not b.in_stable_set(1.0, 1.0, L, MU)                # alpha is free in S_0, must be given


def test_is_feasible_uses_the_stable_set():
    b = bounds()
    assert b.is_feasible(alpha=0.2, psi=0.5, L=L, mu=MU)       # vartheta = 0.9
    assert not b.is_feasible(alpha=0.5, psi=0.2, L=L, mu=MU)   # vartheta = 0.6 < alpha-dependent lower bound


def test_deterministic_rate_matches_theorem_on_a_quadratic():
    """On a noiseless quadratic, the spectral radius of GMM at the selected parameters is <= rho (Theorem 2)."""
    for vartheta, psi in [(0.9, 0.5), (0.95, 0.9), (1.05, 1.5)]:
        alpha, beta, gamma = RAGMMBounds.parameters(vartheta, psi, L, MU)
        rho = math.sqrt(RAGMMBounds.rate(vartheta, alpha, MU))
        for lam in np.linspace(MU, L, 7):
            A = np.array([[(1 + beta) - alpha * lam * (1 + gamma), -beta + alpha * lam * gamma], [1.0, 0.0]])
            assert max(abs(np.linalg.eigvals(A))) <= rho + 1e-9


# ---------------------------------------------------------------------- #
# EV@R bound (Proposition 3, Theorem 3)                                    #
# ---------------------------------------------------------------------- #

def test_bound_is_a_valid_upper_bound_on_the_exact_evar():
    """
    Formula-independent check of eq. (44). For a quadratic with Gaussian noise the stationary
    state is Gaussian, so the exact EV@R of f(x) - f* has a closed form. The theorem's bound
    must never fall below it, for every stable (vartheta, psi) and every zeta.
    """
    d = 10
    eigs = np.linspace(MU, L, d)
    rng = np.random.default_rng(0)
    checked = 0
    while checked < 60:
        psi = float(rng.choice([0.0, rng.uniform(0, 1), rng.uniform(1, 6)]))
        lo, hi = ((1 / (1 + (L / MU) * (1 - psi)), 1.0) if psi < 1
                  else (1.0, min(2 - 1 / psi, 0.5 * (1 + math.sqrt(1 + 4 * (L / MU) * (psi - 1))))))
        vartheta = float(rng.uniform(lo, hi))
        b = bounds()
        if not b.in_stable_set(vartheta, psi, L, MU):
            continue
        alpha, beta, gamma = b.parameters(vartheta, psi, L, MU)
        s = stationary_variances(eigs, alpha, beta, gamma)
        assert s is not None, "a stable-set point must give a stable iteration"
        for zeta in (0.05, 0.5, 0.95):
            bound = RAGMMBounds(zeta=zeta).evar_bound_vp(vartheta, psi, L, MU, d=d)
            assert bound >= exact_evar(eigs, s, zeta), (vartheta, psi, zeta)
        checked += 1


def test_bound_scales_with_noise_variance():
    b = bounds()
    base = b.evar_bound_vp(0.9, 0.5, L, MU, d=10)
    assert b.evar_bound_vp(0.9, 0.5, L, MU, d=10, sigma=3.0) == pytest.approx(9.0 * base)


def test_bound_grows_with_dimension_and_with_confidence():
    b = bounds()
    assert b.evar_bound_vp(0.9, 0.5, L, MU, d=20) > b.evar_bound_vp(0.9, 0.5, L, MU, d=10)
    assert RAGMMBounds(zeta=0.01).evar_bound_vp(0.9, 0.5, L, MU, d=10) > \
        RAGMMBounds(zeta=0.5).evar_bound_vp(0.9, 0.5, L, MU, d=10)


def test_bound_requires_dimension():
    with pytest.raises(ValueError, match="dimension"):
        bounds().evar_bound(0.2, 0.5, L, MU)


@pytest.mark.parametrize("zeta", [0.0, 1.0, -0.1, 1.5])
def test_invalid_zeta_is_rejected(zeta):
    with pytest.raises(ValueError):
        RAGMMBounds(zeta=zeta)


# ---------------------------------------------------------------------- #
# Problem (51)                                                             #
# ---------------------------------------------------------------------- #

CASES = [(1.0, 0.1, 10, 0.25), (0.9888, 0.01, 55, 0.05), (5.28, 2.3, 9, 0.25), (10.0, 0.5, 20, 0.05)]


@pytest.mark.parametrize("L_,mu,d,eps", CASES)
def test_solution_is_feasible_and_meets_the_rate_constraint(L_, mu, d, eps):
    b = bounds()
    sol = b.solve(L_, mu, d, rate_slack=eps, grid_size=40)
    rho2_star = 1 - math.sqrt(mu / L_)
    assert sol.rate_ratio <= 1 + eps + 1e-12
    assert sol.rate / rho2_star == pytest.approx(sol.rate_ratio)
    if sol.region == "S_0":
        assert b.in_stable_set(1.0, 1.0, L_, mu, alpha=sol.alpha)
    else:
        assert b.in_stable_set(sol.vartheta, sol.psi, L_, mu)
    assert (sol.alpha, sol.beta, sol.gamma) == pytest.approx(b.parameters(sol.vartheta, sol.psi, L_, mu, alpha=sol.alpha))
    assert sol.gamma == pytest.approx(sol.psi * sol.beta)


@pytest.mark.parametrize("L_,mu,d,eps", CASES[:2])
def test_solution_minimizes_the_bound_over_the_feasible_grid(L_, mu, d, eps):
    b = bounds()
    sol = b.solve(L_, mu, d, rate_slack=eps, grid_size=25)
    rho2_star = 1 - math.sqrt(mu / L_)
    for vartheta, psi, alpha, _ in b.candidates(L_, mu, 25):
        if b.rate(vartheta, alpha, mu) / rho2_star <= 1 + eps:
            assert b.evar_bound_vp(vartheta, psi, L_, mu, alpha=alpha, d=d) >= sol.evar_bound - 1e-12


def test_every_candidate_is_in_the_stable_set():
    b = bounds()
    regions = set()
    for vartheta, psi, alpha, region in b.candidates(1.0, 0.1, 30):
        regions.add(region)
        assert b.in_stable_set(vartheta, psi, 1.0, 0.1, alpha=alpha if region == "S_0" else None)
    assert regions == {"S_minus", "S_plus", "S_0"}        # every region of Theorem 2 is searched


def test_loan_regime_no_longer_selects_a_vanishing_step_size():
    """v0.1.0 picked alpha ~ 1e-6, beta ~ 0.9998 here (rate constraint measured against an alpha-dependent benchmark)."""
    sol = bounds().solve(0.9888, 0.01, 55, rate_slack=0.05)
    assert sol.alpha > 1e-2
    assert sol.rate_ratio <= 1.05
    assert (sol.alpha, sol.beta, sol.gamma) == pytest.approx((0.33284, 0.89021, 0.70327), rel=1e-3)


def test_solver_may_select_gamma_above_one():
    """S_+ (psi > 1) is part of the search space and gamma = psi * beta may exceed 1 (Theorem 2)."""
    b = bounds()
    found = [c for c in b.candidates(1.0, 0.1, 100) if c[3] == "S_plus"]      # default grid; 20 bins misses S_+
    assert found
    assert any(b.parameters(v, p, 1.0, 0.1)[2] > 1.0 for v, p, _, _ in found)


@pytest.mark.parametrize("L_,mu", [(1.0, 1.0), (0.5, 1.0), (1.0, 0.0), (1.0, -1.0)])
def test_invalid_smoothness_constants_are_rejected(L_, mu):
    with pytest.raises(ValueError):
        bounds().solve(L_, mu, 5)


def test_matches_the_independent_implementation_used_in_the_experiments():
    path = pathlib.Path(__file__).resolve().parents[1] / "research" / "model_multiplicity" / "paper_tuning.py"
    if not path.exists():
        pytest.skip("research/model_multiplicity/paper_tuning.py not present")
    spec = importlib.util.spec_from_file_location("paper_tuning", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for L_, mu, d, eps in CASES:
        for zeta in (0.05, 0.95):
            ref = module.ra_gmm_params(L_, mu, d, zeta, eps, phi=0.99, nbins=40)
            sol = RAGMMBounds(zeta=zeta).solve(L_, mu, d, rate_slack=eps, grid_size=40)
            assert (sol.alpha, sol.beta, sol.gamma, sol.evar_bound) == pytest.approx(
                (ref["alpha"], ref["beta"], ref["gamma"], ref["bound"]), rel=1e-9)


# ---------------------------------------------------------------------- #
# AGD (gamma = beta) and GD (beta = gamma = 0) special cases               #
# ---------------------------------------------------------------------- #

def test_agd_bound_is_a_valid_upper_bound_on_the_exact_evar():
    """Same formula-independent check as above, on the set S_0 (vartheta = psi = 1)."""
    d, eigs = 10, np.linspace(MU, L, 10)
    for alpha in np.linspace(0.05, 1.0, 12) / L:
        _, beta, gamma = RAGMMBounds.parameters(1.0, 1.0, L, MU, alpha=alpha)
        assert beta == gamma
        s = stationary_variances(eigs, alpha, beta, gamma)
        assert s is not None
        for zeta in (0.05, 0.5, 0.95):
            bound = RAGMMBounds(zeta=zeta).evar_bound_vp(1.0, 1.0, L, MU, alpha=alpha, d=d)
            assert bound >= exact_evar(eigs, s, zeta), (alpha, zeta)


def test_gd_bound_is_a_valid_upper_bound_on_the_exact_evar():
    """Theorem 4 (eq. 50) for beta = gamma = 0, against the exact EV@R of noisy GD on a quadratic."""
    d, eigs = 10, np.linspace(MU, L, 10)
    for alpha in np.linspace(0.05, 2 / (MU + L), 12):
        s = stationary_variances(eigs, alpha, 0.0, 0.0)
        for zeta in (0.05, 0.5, 0.95):
            bound = RAGMMBounds(zeta=zeta).gd_evar_bound(alpha, L, MU, d=d)
            assert bound >= exact_evar(eigs, s, zeta), (alpha, zeta)


def test_gd_bound_closed_form_and_domain():
    b = bounds()
    alpha = 0.5
    rho2 = max(abs(1 - alpha * MU), abs(1 - alpha * L)) ** 2
    expected = alpha ** 2 * L / (2 * (1 - rho2)) * (math.sqrt(2 * math.log(1 / 0.05)) + math.sqrt(7)) ** 2
    assert b.gd_evar_bound(alpha, L, MU, d=7) == pytest.approx(expected)
    assert b.gd_evar_bound(alpha, L, MU, d=7, sigma=2.0) == pytest.approx(4 * expected)
    assert b.gd_evar_bound(2.5 / (MU + L), L, MU, d=7) == float("inf")      # alpha > 2/(mu + L)
    assert b.gd_evar_bound(0.0, L, MU, d=7) == float("inf")


def test_ra_agd_is_the_s0_restriction_of_problem_51():
    b = bounds()
    for L_, mu, d, eps in [(1.0, 0.1, 10, 0.05), (0.9888, 0.01, 55, 0.05), (44.13, 1.0, 100, 0.05)]:
        sol = b.solve(L_, mu, d, rate_slack=eps, method="agd")
        assert sol.region == "S_0" and sol.vartheta == 1.0 and sol.psi == 1.0
        assert sol.gamma == sol.beta == pytest.approx((1 - math.sqrt(sol.alpha * mu)) / (1 + math.sqrt(sol.alpha * mu)))
        assert 0 < sol.alpha <= 1 / L_
        assert sol.rate_ratio <= 1 + eps + 1e-12
        # minimal over the S_0 grid among rate-feasible alphas
        rho2_star = 1 - math.sqrt(mu / L_)
        for _, _, alpha, _ in b.candidates(L_, mu, 100, agd_only=True):
            if b.rate(1.0, alpha, mu) / rho2_star <= 1 + eps:
                assert b.evar_bound_vp(1.0, 1.0, L_, mu, alpha=alpha, d=d) >= sol.evar_bound - 1e-12
        # the RA-AGD grid (10 x finer) contains the general solver's S_0 candidates, so it is at least as good as each
        for vartheta, psi, alpha, region in b.candidates(L_, mu, 100):
            if region == "S_0" and b.rate(1.0, alpha, mu) / rho2_star <= 1 + eps:
                assert sol.evar_bound <= b.evar_bound_vp(1.0, 1.0, L_, mu, alpha=alpha, d=d) + 1e-12


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="method"):
        bounds().solve(1.0, 0.1, 5, method="gd")


def test_solve_fixed_alpha_is_feasible_and_minimizes_the_bound_on_its_curve():
    import math
    L, mu, d = 1.0, 0.01, 55
    b = RAGMMBounds(zeta=0.05)
    for c in (1.0, 0.3, 0.1, 0.03):
        sol = b.solve_fixed_alpha(c / L, L, mu, d)
        assert sol.alpha == pytest.approx(c / L)
        assert sol.beta == pytest.approx(b.parameters(sol.vartheta, sol.psi, L, mu, alpha=sol.alpha)[1])
        assert sol.gamma == pytest.approx(sol.psi * sol.beta)
        assert math.isfinite(sol.evar_bound) and sol.rate < 1.0
        if sol.psi == 1.0:
            assert sol.region == "S_0" and sol.beta == pytest.approx((1 - math.sqrt(sol.alpha * mu)) / (1 + math.sqrt(sol.alpha * mu)))
        else:
            assert b.in_stable_set(sol.vartheta, sol.psi, L, mu)
            assert sol.vartheta == pytest.approx(1 - sol.alpha * L * (1 - sol.psi))
        for psi in (0.0, 0.25, 0.5, 0.75, 0.9):          # no feasible psi on the curve beats it
            vt = b.vartheta_of(sol.alpha, psi, L)
            if b.in_stable_set(vt, psi, L, mu):
                assert b.evar_bound_vp(vt, psi, L, mu, alpha=sol.alpha, d=d) >= sol.evar_bound - 1e-9


def test_solve_fixed_alpha_validates_inputs():
    b = RAGMMBounds()
    with pytest.raises(ValueError):
        b.solve_fixed_alpha(-0.1, 1.0, 0.1, 5)
    with pytest.raises(ValueError):
        b.solve_fixed_alpha(0.1, 0.1, 1.0, 5)
