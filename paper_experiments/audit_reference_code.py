"""
Audit of the strongly-convex RA-GMM code in ``paper_experiments/EntropicsFOMs`` against the paper
(Can & Gürbüzbalaban, 2025) and against ``risk_averse_tuning``. See AUDIT.md for the findings.

Run from the repository root (needs ``pip install pandas plotly tabulate`` to import the reference code):

    python -W ignore paper_experiments/audit_reference_code.py

Part A  four formula deviations in ``sFoms.str_cnvx_evar_bound``: exact EV@R of a noisy quadratic vs bounds
Part B  the rate column returned by ``sFoms.str_cnvx_stab_reg_params`` is in reverse order
Part C  parameters selected for the paper's logistic-regression setting (needs the reference package)
"""
import importlib.util
import itertools
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from helpers import exact_evar, stationary_variances  # noqa: E402
from risk_averse_tuning import RAGMMBounds  # noqa: E402


def load_reference():
    """Import the authors' package (its folder is ``EntropicsFOMs`` but it imports ``EntropicFOMs``)."""
    folder = ROOT / "paper_experiments" / "EntropicsFOMs"
    spec = importlib.util.spec_from_file_location("EntropicFOMs", folder / "__init__.py",
                                                  submodule_search_locations=[str(folder)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["EntropicFOMs"] = pkg
    spec.loader.exec_module(pkg)
    from EntropicFOMs.entropic_ra_sFoms import entropic_ra_sFoms
    obj = object.__new__(entropic_ra_sFoms)
    obj.noise_std = 1
    return obj


try:
    REF = load_reference()
except Exception as exc:                                     # pragma: no cover
    REF = None
    print(f"[reference package not importable ({type(exc).__name__}: {exc}); parts B and C skipped]\n")


# ---------------------------------------------------------------------- #
# Part A                                                                   #
# ---------------------------------------------------------------------- #

def reference_bound(vart, psi, L, mu, d, zeta, fix_beta, fix_theta, fix_half, fix_cond, varphi=0.99):
    """
    Port of ``sFoms.str_cnvx_evar_bound``. With all four ``fix_*`` False it reproduces the reference code
    exactly; with all True it equals the paper's eq. (44) (``RAGMMBounds.evar_bound_vp``).
    """
    alpha = (1 - vart) / (L * (1 - psi))
    if fix_beta:      # eq. (35b)
        beta = (1 - np.sqrt(vart * alpha * mu)) / (1 - alpha * psi * mu) * (1 - np.sqrt(alpha * mu / vart))
    else:             # reference code: 1 - sqrt(.)/(.)*(.)   (operator precedence)
        beta = 1 - np.sqrt(vart * alpha * mu) / (1 - alpha * psi * mu) * (1 - np.sqrt(alpha * mu / vart))
    gamma = psi * beta
    v = (2 * L ** 2 / mu * (2 * (beta - gamma) ** 2 + (1 - alpha * L) ** 2 * (1 + 2 * gamma + 2 * gamma ** 2))
         + 0.5 * vart / alpha * (1 - np.sqrt(vart * alpha * mu)))
    s = np.sqrt(vart * mu)
    if fix_theta:     # eq. (37)
        th_u = 2 * s / (alpha * (8 * v * np.sqrt(alpha) + s * (vart + alpha * L)))
    else:             # reference code: no factor 2, extra alpha
        th_u = s / (alpha * (8 * v * np.sqrt(alpha) + alpha * s * (vart + alpha * L)))
    th = varphi * th_u
    D = 2 - th * alpha * (vart + alpha * L)
    a = 1 - np.sqrt(vart * alpha * mu) + th * 4 * alpha ** 2 * v / D
    if fix_half:      # eq. (40)
        rho = 0.5 * a + 0.5 * np.sqrt(a ** 2 + 16 * th * alpha ** 2 * v / D)
    else:             # reference code: bbrho_1 = 0.5*a is used both outside and inside the square root
        rho = 0.5 * (0.5 * a) + 0.5 * np.sqrt((0.5 * a) ** 2 + 16 * th * alpha ** 2 * v / D)
    D41 = D if fix_cond else 2 - th * (vart + alpha * L)     # eq. (41) denominator
    if np.log(1 / zeta) < 0.5 * d / (1 - rho) * ((th * alpha * (vart + alpha * L)) / D41) ** 2:
        return 0.5 * alpha * (vart + alpha * L) * (np.sqrt(d / (1 - rho)) + np.sqrt(2 * np.log(1 / zeta))) ** 2
    return d * alpha * (vart + alpha * L) / ((1 - rho) * D) + 2 * np.log(1 / zeta) / th


def part_a():
    L, mu, d = 1.0, 0.1, 10
    eigs = np.linspace(mu, L, d)
    rng = np.random.default_rng(0)
    b = RAGMMBounds()
    cases = []
    for _ in range(600):
        psi = float(rng.choice([0.0, rng.uniform(0, 1), rng.uniform(1, 6)]))
        lo, hi = ((1 / (1 + (L / mu) * (1 - psi)), 1.0) if psi < 1
                  else (1.0, min(2 - 1 / psi, 0.5 * (1 + math.sqrt(1 + 4 * (L / mu) * (psi - 1))))))
        vart = float(rng.uniform(lo, hi))
        if not b.in_stable_set(vart, psi, L, mu):
            continue
        s = stationary_variances(eigs, *b.parameters(vart, psi, L, mu))
        if s is None:
            continue
        for zeta in (0.05, 0.95):
            cases.append((vart, psi, zeta, exact_evar(eigs, s, zeta)))
    print(f"PART A  {len(cases)} (stable parameter, zeta) cases; quadratic L=1, mu=0.1, d=10, sigma^2=1")
    err_all = max(abs(reference_bound(v, p, L, mu, d, z, 1, 1, 1, 1) - RAGMMBounds(zeta=z).evar_bound_vp(v, p, L, mu, d=d))
                  / RAGMMBounds(zeta=z).evar_bound_vp(v, p, L, mu, d=d) for v, p, z, _ in cases)
    print(f"  port with all four fixed vs risk_averse_tuning: max rel diff {err_all:.1e}")
    if REF is not None:
        err_none = max(abs(reference_bound(v, p, L, mu, d, z, 0, 0, 0, 0) - REF.str_cnvx_evar_bound(L, mu, v, p, d, z, 1))
                       / REF.str_cnvx_evar_bound(L, mu, v, p, d, z, 1) for v, p, z, _ in cases)
        print(f"  port with none fixed vs the reference code itself: max rel diff {err_none:.1e}")
    ratios = [RAGMMBounds(zeta=z).evar_bound_vp(v, p, L, mu, d=d) / ex for v, p, z, ex in cases]
    print(f"  paper's bound / exact EV@R: min {min(ratios):.0f}x, median {np.median(ratios):.0f}x, max {max(ratios):.0f}x"
          " (valid but conservative)")
    print("  'violation' = bound below the exact EV@R (the bound is then not an upper bound)")
    print("  fix: beta(35b) theta_u(37) rho_bar(40) cond(41)")
    for fx in [(0, 0, 0, 0), (1, 1, 0, 1), (0, 0, 1, 0), (1, 1, 1, 1)]:
        viol = [ex / reference_bound(v, p, L, mu, d, z, *fx) for v, p, z, ex in cases
                if reference_bound(v, p, L, mu, d, z, *fx) < ex * (1 - 1e-9)]
        label = {(0, 0, 0, 0): "reference code as is", (1, 1, 0, 1): "all fixed except rho_bar",
                 (0, 0, 1, 0): "only rho_bar fixed", (1, 1, 1, 1): "paper's formulas"}[fx]
        extra = f", exact EV@R up to {max(viol):.1f}x the 'bound'" if viol else ""
        print(f"  {str(fx):13s} {label:26s} violations {len(viol):3d}/{len(cases)}{extra}")
    print()


# ---------------------------------------------------------------------- #
# Parts B and C                                                            #
# ---------------------------------------------------------------------- #

def part_b(L=44.131, mu=1.0, eps=0.05):
    vart, psi, rate_ref = REF.str_cnvx_stab_reg_params(L, mu)
    alpha = (1 - vart) / (L * (1 - psi))
    rate_true = 1 - np.sqrt(vart * alpha * mu)
    rho2_star = 1 - math.sqrt(mu / L)
    keep_ref = rate_ref <= (1 + eps) * rho2_star
    keep_true = rate_true <= (1 + eps) * rho2_star
    print(f"PART B  rate column of str_cnvx_stab_reg_params (L={L}, mu={mu}, {len(vart)} stable-set rows)")
    print(f"  rate column == rho^2 of its own (vartheta, psi): {np.allclose(rate_ref, rate_true)}")
    print(f"  rate column == rho^2 in reverse row order      : {np.allclose(rate_ref, rate_true[::-1])}")
    print(f"  eps={eps}: rows kept by the reference code {keep_ref.sum()}, of which truly feasible {(keep_ref & keep_true).sum()}"
          f" ({100 * (keep_ref & ~keep_true).sum() / keep_ref.sum():.0f}% violate rho^2 <= (1+eps) rho_*^2,"
          f" worst true ratio {np.max(rate_true[keep_ref]) / rho2_star:.3f})")
    alpha_agd, rate_agd = REF.str_cnvx_agd_stab_reg_params(L, mu)           # AGD helper (line 846): rho^2 = 1 - sqrt(alpha mu)
    true_agd = 1 - np.sqrt(alpha_agd * mu)
    print(f"  AGD helper str_cnvx_agd_stab_reg_params ({len(alpha_agd)} rows): rate column aligned with alpha: "
          f"{np.allclose(rate_agd, true_agd)}; in reverse order: {np.allclose(rate_agd, true_agd[::-1])}")
    print()


def part_c():
    from EntropicFOMs.entropic_ra_sFoms import entropic_ra_sFoms
    det = entropic_ra_sFoms(optim_problem="synthetic_logit", data_std=5, data_size=1000, num_of_feat=100,
                            reg_param=1, noise_std=0, init_x0="fixed")
    L, mu = det.L_and_mu_logistic_reg()
    L, d = float(np.real(L)), det.num_of_feat
    print(f"PART C  the paper's logistic-regression setting: L={L:.3f}, mu={mu}, kappa={L / mu:.1f}, d={d}")
    rho2_star = 1 - math.sqrt(mu / L)
    for zeta, eps in [(0.95, 0.05), (0.05, 0.05), (0.95, 0.25)]:
        a, be, ga = REF.raTMM_str_cnv_params(L, mu, eps, d, confidence=zeta)
        psi = ga / be
        vart = RAGMMBounds.vartheta_of(a, psi, L)
        ratio_ref = RAGMMBounds.rate(vart, a, mu) / rho2_star
        sol = RAGMMBounds(zeta=zeta).solve(L, mu, d, rate_slack=eps, grid_size=100)
        print(f"  zeta={zeta} eps={eps}")
        print(f"    reference code : alpha={a:.4f} beta={be:.4f} gamma={ga:.4f}  rho^2/rho_*^2 = {ratio_ref:.3f}"
              f"  ({'violates' if ratio_ref > 1 + eps + 1e-9 else 'meets'} the constraint {1 + eps})")
        print(f"    risk_averse_tuning: alpha={sol.alpha:.4f} beta={sol.beta:.4f} gamma={sol.gamma:.4f}  "
              f"rho^2/rho_*^2 = {sol.rate_ratio:.3f}  region {sol.region}")
    print()


if __name__ == "__main__":
    part_a()
    if REF is not None:
        part_b()
        part_c()
