"""
Audit of the strongly-convex RA-GMM code in ``paper_experiments/EntropicsFOMs`` against the paper
(Can & Gürbüzbalaban, 2025) and against ``risk_averse_tuning``. See AUDIT.md for the findings.

Run from the repository root (needs ``pip install pandas plotly tabulate`` to import the reference code):

    python -W ignore paper_experiments/audit_reference_code.py

Part A  four formula deviations in ``sFoms.str_cnvx_evar_bound``: exact EV@R of a noisy quadratic vs bounds
Part B  the rate column returned by ``sFoms.str_cnvx_stab_reg_params`` is in reverse order
Part C  parameters selected for the paper's logistic-regression setting (needs the reference package)
Part D  library GMM vs the reference ``train()`` and vs torch SGD / momentum / Nesterov in the special cases
        GD (beta = gamma = 0), heavy ball (gamma = 0), AGD (gamma = beta)
Part E  the quadratic path (paper Section 3): reference ``quad_*`` / ``raTMM_quad_params`` vs
        ``risk_averse_tuning.quadratic``, and how the reference treats the noise scale ``std``
"""
import importlib.util
import itertools
import math
import os
import pathlib
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from helpers import exact_evar, stationary_variances  # noqa: E402
from risk_averse_tuning import RAGMMBounds  # noqa: E402
from risk_averse_tuning import quadratic as quad  # noqa: E402


def load_reference():
    """
    Import the authors' package (its folder is ``EntropicsFOMs`` but it imports ``EntropicFOMs``).
    The package creates ``./risk-averse GMM experiments/`` in the working directory at import time and
    ``train()`` writes CSV files there, so the import and all later use happen inside a temporary directory.
    """
    folder = ROOT / "paper_experiments" / "EntropicsFOMs"
    os.chdir(TMP.name)
    spec = importlib.util.spec_from_file_location("EntropicFOMs", folder / "__init__.py",
                                                  submodule_search_locations=[str(folder)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["EntropicFOMs"] = pkg
    spec.loader.exec_module(pkg)
    from EntropicFOMs.entropic_ra_sFoms import entropic_ra_sFoms
    obj = object.__new__(entropic_ra_sFoms)
    obj.noise_std = 1
    return obj


TMP = tempfile.TemporaryDirectory()
_CWD = os.getcwd()


try:
    REF = load_reference()
except Exception as exc:                                     # pragma: no cover
    REF = None
    print(f"[reference package not importable ({type(exc).__name__}: {exc}); parts B, C and D skipped]\n")
finally:
    os.chdir(TMP.name if REF is not None else _CWD)          # keep the temp dir as cwd while the reference is used


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


def part_d(K=300):
    import torch
    from risk_averse_tuning import GMM
    from EntropicFOMs.entropic_ra_sFoms import entropic_ra_sFoms
    torch.set_default_dtype(torch.float64)
    det = entropic_ra_sFoms(optim_problem="synthetic_logit", data_std=5, data_size=1000, num_of_feat=100,
                            reg_param=1, noise_std=0, init_x0="fixed")
    L, mu = det.L_and_mu_logistic_reg()
    L = float(np.real(L))
    X = torch.tensor(np.asarray(det.X, dtype=np.float64))        # exact conversion; the reference computes in float64
    Y = torch.tensor(np.asarray(det.Y, dtype=np.float64)).reshape(-1)
    reg = float(det.reg_param)
    kappa = L / mu
    mom = (math.sqrt(kappa) - 1) / (math.sqrt(kappa) + 1)

    def loss_fn(w):
        return torch.nn.functional.softplus(-Y * (X @ w)).mean() + 0.5 * reg * (w ** 2).sum()

    def run_library(a, b, g):
        w = torch.ones(100, requires_grad=True)
        opt = GMM([w], lr=a, beta=b, gamma=g)
        path = [float(loss_fn(w))]
        def closure():
            opt.zero_grad()
            loss = loss_fn(w)
            loss.backward()
            return loss
        for _ in range(K):
            opt.step(closure)
            path.append(float(loss_fn(w)))
        return w.detach().numpy(), np.array(path)

    def run_torch(a, b, nesterov):
        w = torch.ones(100, requires_grad=True)
        opt = torch.optim.SGD([w], lr=a, momentum=b, nesterov=nesterov)
        for _ in range(K):
            opt.zero_grad()
            loss_fn(w).backward()
            opt.step()
        return w.detach().numpy()

    print(f"PART D  library GMM vs reference train() and torch.optim.SGD, logistic regression of the paper "
          f"(L={L:.2f}, mu={mu}), {K} steps")
    print(f"  {'case':28s} {'max|x_K - x_K(reference)|':>26s} {'max|f_k - f_k(reference)|':>26s} {'vs torch':>22s}")
    cases = [("GD (beta=gamma=0)", 1 / L, 0.0, 0.0), ("heavy ball (gamma=0)", 1 / L, mom, 0.0),
             ("AGD / Nesterov (gamma=beta)", 1 / L, mom, mom), ("general (beta=0.6, gamma=0.35)", 0.5 / L, 0.6, 0.35)]
    for name, a, b, g in cases:
        det.train(learning_rate=a, iter_momentum=b, grad_momentum=g, Samples=1, stop_criteria=K, seed=1)
        x_ref = det.train_result[:, 0]
        f_ref = det.suboptimality_data.values[:, 0]
        x_lib, f_lib = run_library(a, b, g)
        dx, df = np.abs(x_lib - x_ref).max(), np.abs(f_lib - f_ref).max()
        if (b, g) == (0.0, 0.0):
            vs = f"SGD: {np.abs(x_lib - run_torch(a, 0.0, False)).max():.1e}"
        elif g == 0.0:
            vs = f"SGD momentum: {np.abs(x_lib - run_torch(a, b, False)).max():.1e}"
        elif g == b:
            vs = "(see tests)"      # torch's Nesterov tracks the lookahead point y_k, compared in tests/test_gmm.py
        else:
            vs = "-"
        print(f"  {name:28s} {dx:26.1e} {df:26.1e} {vs:>22s}")
    print("  (the reference rounds its stored f-path to 5 decimals, so f differences below 1e-5 are rounding)")
    print()


# ---------------------------------------------------------------------- #
# Part E                                                                   #
# ---------------------------------------------------------------------- #

def part_e():
    eigs = np.array([0.1, 0.4, 1.0, 2.0])
    rng = np.random.default_rng(0)
    pts = []
    while len(pts) < 100:
        a, b, g = rng.uniform(0, 1 / eigs.max()), rng.uniform(0, 1), rng.uniform(0, 1)
        if quad.quad_in_stable_set(a, b, g, eigs):
            pts.append((a, b, g))
    print(f"PART E  paper Section 3 (quadratic): eigenvalues {eigs.tolist()}, {len(pts)} stable random parameter points")
    d_risk = d_bound = d_evar = 0.0
    for a, b, g in pts:
        theta = 0.5 * 2 * quad.quad_u(a, b, g, eigs).min()
        ref_r = REF.quad_risk_measure(a, b, g, eigs, theta=theta, std=1)
        d_risk = max(d_risk, abs(ref_r - quad.quad_entropic_risk(theta, a, b, g, eigs)) / abs(ref_r))
        d_bound = max(d_bound, abs(REF.quad_evar_bound(a, b, g, eigs, std=1, conf_level=0.05)
                                   - quad.quad_evar_bound(a, b, g, eigs, 0.05)) / quad.quad_evar_bound(a, b, g, eigs, 0.05))
        ref_e = REF.quad_evar(a, b, g, eigs, nbins_theta=2000, std=1, conf_level=0.05)
        ours = float(quad.quad_evar_exact(a, b, g, eigs, 0.05))
        d_evar = max(d_evar, (ref_e - ours) / ours)
    print(f"  entropic risk r(theta), Prop. 2        max rel. diff {d_risk:.1e}")
    print(f"  Theorem 1 bound (std = 1)              max rel. diff {d_bound:.1e}")
    print(f"  EV@R by grid over theta in (0, 5]      worst overestimate {d_evar:.1e} x exact (grid of 0.1 steps can miss the interval (0, 2 min u))")

    # the reference's stable region S_q only tests rho < 1; the paper's S_q also requires u_i > 0
    region = REF.quad_stab_reg(eigs, nbins=30)
    n_u = int((~quad.quad_in_stable_set(region[:, 0], region[:, 1], region[:, 2], eigs)).sum())
    print(f"  reference stable region: {len(region)} points with rho < 1, of which {n_u} fail the u_i > 0 test of eq. (23)")

    # problem (32): same grid, same constraint, same bound -> same parameters
    print("  problem (32): reference raTMM_quad_params vs quad.solve_quadratic (same 50-point grids, confidence = 1 - zeta)")
    for eps in (0.05, 0.25, 1.0, 3.0):
        REF.eig_vals = eigs
        try:
            ref = tuple(round(float(v), 4) for v in REF.raTMM_quad_params(eps, confidence=0.05))
        except Exception:
            ref = "no feasible point"
        try:
            sol = quad.solve_quadratic(eigs, zeta=0.05, rate_slack=eps, grid_size=50)
            ours = (round(sol.alpha, 4), round(sol.beta, 4), round(sol.gamma, 4))
        except ValueError:
            ours = "no feasible point"
        print(f"    eps={eps:<5}  reference {ref}   ours {ours}")

    # noise scale: std is documented as a standard deviation but is used as a variance
    a, b, g = pts[0]
    print("  noise scale (std = 3, so sigma^2 = 9): value / value at std = 1  [a variance would scale by 9]")
    theta = quad.quad_u(a, b, g, eigs).min()
    print(f"    quad_risk_measure  {REF.quad_risk_measure(a, b, g, eigs, theta, 3) / REF.quad_risk_measure(a, b, g, eigs, theta, 1):.1f}  (uses std**2)")
    print(f"    quad_evar_bound    {REF.quad_evar_bound(a, b, g, eigs, 3, 0.05) / REF.quad_evar_bound(a, b, g, eigs, 1, 0.05):.1f}  (uses std)")
    e1 = REF.quad_evar(a, b, g, eigs, 2000, 1, 0.05)
    e3 = REF.quad_evar(a, b, g, eigs, 2000, 3, 0.05)
    print(f"    quad_evar          {e3 / e1:.1f}  (std**2 in the risk term, 2*std in the log term: mixed)")
    print("  (the paper's experiments use std = 1, where all three coincide)")
    print()


if __name__ == "__main__":
    part_a()
    if REF is not None:
        part_b()
        part_c()
        part_d()
        part_e()
