"""
RA-GMM parameter selection exactly as specified in Can & Gürbüzbalaban (2025),
for strongly convex smooth (non-quadratic) objectives:

- Stable set S_c = (S_- ∪ S_+) ∩ S_1 and S_0 from Theorem 2
- Parameters (35a)-(35c):
      alpha = (1 - vartheta) / (L (1 - psi))            on S_c,   alpha in (0, 1/L] on S_0
      beta  = (1 - sqrt(vartheta alpha mu)) / (1 - alpha psi mu) * (1 - sqrt(alpha mu / vartheta))
      gamma = psi * beta
- Rate rho^2 = 1 - sqrt(vartheta alpha mu); fastest certified rate rho_*^2 = 1 - sqrt(mu / L)
- Asymptotic EV@R bound E_{1-zeta}(vartheta, psi) from Theorem 3, eq. (44), with theta_phi = phi * theta_u (37)
- Problem (51): minimize E_{1-zeta} over S_c ∪ S_0 subject to rho^2 / rho_*^2 <= 1 + eps

The grid over S_c follows paper_experiments/EntropicsFOMs/sFoms.py::str_cnvx_stab_reg_params.

Note: this is written separately from risk_averse_tuning.RAGMM because that
implementation (v0.1.0) differs from the paper in two places: its rate
constraint uses an alpha-dependent benchmark (1 - sqrt(alpha mu / L)), which
makes the constraint vacuous for small alpha, and its beta uses
(1 - alpha psi L) in the denominator instead of (1 - alpha psi mu).
"""
import math

import numpy as np


def gmm_params(vartheta, psi, L, mu, alpha=None):
    if alpha is None:
        alpha = (1 - vartheta) / (L * (1 - psi))
    beta = (1 - math.sqrt(vartheta * alpha * mu)) / (1 - alpha * psi * mu) * (1 - math.sqrt(alpha * mu / vartheta))
    return alpha, beta, psi * beta


def evar_bound(vartheta, psi, alpha, beta, gamma, L, mu, d, zeta, phi=0.99, sigma2=1.0):
    """Asymptotic EV@R bound (44). Returns inf where the bound is undefined."""
    v = (2 * L ** 2 / mu * (2 * (beta - gamma) ** 2 + (1 - alpha * L) ** 2 * (1 + 2 * gamma + 2 * gamma ** 2))
         + vartheta / (2 * alpha) * (1 - math.sqrt(vartheta * alpha * mu)))
    theta_u = 2 * math.sqrt(vartheta * mu) / (alpha * (8 * v * math.sqrt(alpha) + math.sqrt(vartheta * mu) * (vartheta + alpha * L)))
    theta = phi * theta_u
    c = alpha * (vartheta + alpha * L)
    denom = 2 - theta * c
    if denom <= 0:
        return float("inf")
    a = 1 - math.sqrt(vartheta * alpha * mu) + theta * 4 * alpha ** 2 * v / denom
    rho_bb = 0.5 * a + 0.5 * math.sqrt(a ** 2 + 16 * theta * alpha ** 2 * v / denom)
    if not (0 < rho_bb < 1):
        return float("inf")
    log_z = math.log(1 / zeta)
    if log_z <= d / (2 * (1 - rho_bb)) * (theta * c / denom) ** 2:            # condition (41)
        return sigma2 * c / 2 * (math.sqrt(d / (1 - rho_bb)) + math.sqrt(2 * log_z)) ** 2
    return sigma2 * d * c / ((1 - rho_bb) * denom) + 2 * sigma2 / theta * log_z


def stable_set(L, mu, vart_nbins=100, psi_nbins=100):
    """Grid over S_c (Theorem 2). Yields (vartheta, psi)."""
    kappa = L / mu
    psi_low = np.linspace(0, 1, psi_nbins + 1)[:-1]
    psi_high = np.linspace(1, 2 * kappa * (1 + math.sqrt(1 - 1 / kappa)), psi_nbins + 1)[1:]
    for psi in np.concatenate([psi_low, psi_high]):
        if psi < 1:
            lo = 1 / (1 + kappa * (1 - psi)) if psi == 0 else max(2 - 1 / psi, 1 / (1 + kappa * (1 - psi)))
            varts = np.linspace(lo, 1, vart_nbins + 1)[:-1]
        else:
            hi = min(2 - 1 / psi, 0.5 * (1 + math.sqrt(1 + 4 * kappa * (psi - 1))))
            varts = np.linspace(1, hi, vart_nbins + 1)[1:]
        for vart in varts:
            alpha = (1 - vart) / (L * (1 - psi))
            if alpha <= 0:
                continue
            m_psi = mu * psi ** 2 - L * (1 - psi) ** 2
            rho2 = 1 - math.sqrt(vart * alpha * mu)
            if rho2 * (1 - alpha * m_psi / vart) - (1 - alpha * psi * mu) ** 2 <= 0:      # S_1
                yield float(vart), float(psi)


def ra_gmm_params(L, mu, d, zeta, eps, phi=0.99, nbins=100, include_s0=True):
    """Solve problem (51) by grid search. Returns dict with alpha, beta, gamma, vartheta, psi, bound, rate."""
    rho2_star = 1 - math.sqrt(mu / L)
    best = None
    candidates = []
    for vart, psi in stable_set(L, mu, nbins, nbins):
        candidates.append((vart, psi, None))
    if include_s0:
        for alpha in np.linspace(0, 1 / L, nbins + 1)[1:]:
            candidates.append((1.0, 1.0, float(alpha)))
    for vart, psi, a0 in candidates:
        alpha, beta, gamma = gmm_params(vart, psi, L, mu, alpha=a0)
        rho2 = 1 - math.sqrt(vart * alpha * mu)
        if rho2 / rho2_star > 1 + eps:
            continue
        b = evar_bound(vart, psi, alpha, beta, gamma, L, mu, d, zeta, phi)
        if best is None or b < best["bound"]:
            best = dict(alpha=alpha, beta=beta, gamma=gamma, vartheta=vart, psi=psi,
                        bound=b, rate=rho2, rate_ratio=rho2 / rho2_star)
    return best
