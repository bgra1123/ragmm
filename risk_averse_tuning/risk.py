"""
EV@R bounds and parameter selection for RA-GMM.

Follows B. Can and M. Gürbüzbalaban, "Entropic risk-averse generalized momentum
methods", Optimization Methods and Software 40(6), 2025:

* Theorem 2   - the parametrization (alpha, beta, gamma) of GMM by two free
                variables (vartheta, psi), the stable sets S_-, S_+, S_1, S_0
                and the rate  rho^2 = 1 - sqrt(vartheta * alpha * mu).
* Proposition 3 / Theorem 3 - bound on the entropic value at risk of
                f(x_k) - f(x*) (eqs. 37, 40, 41, 44).
* Problem (51) - choose (vartheta, psi) in S_c U S_0 minimizing the EV@R bound
                subject to  rho^2 / rho_*^2 <= 1 + eps,  rho_*^2 = 1 - sqrt(mu / L).

Notation. GMM runs ``y_k = x_k + gamma * d_k``, ``d_{k+1} = beta * d_k - alpha *
grad f(y_k)``, ``x_{k+1} = x_k + d_{k+1}`` (``d_k = x_k - x_{k-1}``), with
gradient noise ``N(0, sigma^2 I_d)``.

Conventions that are easy to get wrong
--------------------------------------
* ``zeta`` is the **tail probability**: ``P{f(x_k) - f(x*) >= EV@R_{1-zeta}} <= zeta``
  and ``zeta`` enters the bounds only through ``log(1/zeta)``. Small ``zeta``
  (e.g. 0.05) means a demanding 95% guarantee. The paper's experiments pass
  ``zeta = 0.95``, i.e. log(1/zeta) ~ 0.05, a very weak guarantee.
* All bounds are stated for noise variance ``sigma^2 = 1`` and scale linearly in
  ``sigma^2``. The minimizer of problem (51) does not depend on ``sigma``.
* ``gamma = psi * beta`` can exceed 1 (the set S_+ has psi > 1); this is
  allowed by Theorem 2 and is not an instability.
"""
import math
from typing import NamedTuple, Optional


class RAGMMSolution(NamedTuple):
    """Solution of problem (51)."""
    alpha: float
    beta: float
    gamma: float
    vartheta: float
    psi: float
    evar_bound: float      # asymptotic EV@R bound (44) for sigma^2 = 1
    rate: float            # rho^2 = 1 - sqrt(vartheta * alpha * mu)
    rate_ratio: float      # rho^2 / rho_*^2
    region: str            # "S_minus", "S_plus" or "S_0"


class RAGMMBounds:
    """
    EV@R convergence bounds for GMM with the (vartheta, psi) parametrization.

    Args:
        zeta: tail probability in ``P{S >= EV@R_{1-zeta}} <= zeta``, in (0, 1)
              (default 0.05, i.e. a 95% guarantee).
        eps:  tolerance used in set-membership tests at the boundary of the
              stable sets (default 1e-12).
        phi:  the constant ``phi`` in (0, 1) of Theorem 3, ``theta_phi = phi *
              theta_u`` (default 0.99, as in the paper's experiments).
    """

    def __init__(self, zeta=0.05, eps=1e-12, phi=0.99):
        if not 0.0 < zeta < 1.0:
            raise ValueError(f"zeta must be in (0, 1), got {zeta}")
        if not 0.0 < phi < 1.0:
            raise ValueError(f"phi must be in (0, 1), got {phi}")
        self.zeta = zeta
        self.phi = phi
        self.eps = eps
        self.d: Optional[int] = None   # problem dimension, required by evar_bound

    # ------------------------------------------------------------------ #
    # Parametrization                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def vartheta_of(alpha, psi, L):
        """vartheta = 1 - alpha L (1 - psi)   (inverse of eq. 35a)."""
        return 1.0 - alpha * L * (1.0 - psi)

    @staticmethod
    def alpha_of(vartheta, psi, L):
        """alpha = (1 - vartheta) / (L (1 - psi))   (eq. 35a, psi != 1)."""
        if psi == 1.0:
            raise ValueError("alpha is a free parameter when psi == 1 (set S_0)")
        return (1.0 - vartheta) / (L * (1.0 - psi))

    @staticmethod
    def parameters(vartheta, psi, L, mu, alpha=None):
        """(alpha, beta, gamma) from eqs. (35a)-(35c)."""
        if alpha is None:
            alpha = RAGMMBounds.alpha_of(vartheta, psi, L)
        beta = ((1.0 - math.sqrt(vartheta * alpha * mu)) / (1.0 - alpha * psi * mu)
                * (1.0 - math.sqrt(alpha * mu / vartheta)))
        return alpha, beta, psi * beta

    @staticmethod
    def rate(vartheta, alpha, mu):
        """Convergence rate rho^2 = 1 - sqrt(vartheta alpha mu)  (Theorem 2)."""
        return 1.0 - math.sqrt(vartheta * alpha * mu)

    # ------------------------------------------------------------------ #
    # Stable sets (Theorem 2)                                              #
    # ------------------------------------------------------------------ #

    def in_stable_set(self, vartheta, psi, L, mu, alpha=None):
        """
        True iff ``(vartheta, psi)`` lies in ``S_c U S_0`` of Theorem 2.

        ``S_c = (S_- U S_+) n S_1``; ``S_0 = {vartheta = psi = 1}`` with
        ``alpha in (0, 1/L]`` (pass ``alpha`` for this case; it is free there).
        """
        tol = self.eps
        if psi == 1.0:                                      # S_0 (AGD)
            return vartheta == 1.0 and alpha is not None and 0.0 < alpha <= 1.0 / L * (1 + tol)
        kappa = L / mu
        a = self.alpha_of(vartheta, psi, L)
        if not a > 0.0:
            return False
        if psi < 1.0:                                       # S_-
            if psi < 0.0:
                return False
            lo = 1.0 / (1.0 + kappa * (1.0 - psi))
            if psi > 0.0:
                lo = max(2.0 - 1.0 / psi, lo)
            if not (lo - tol <= vartheta < 1.0):
                return False
        else:                                               # S_+
            if not (1.0 < vartheta <= 2.0 - 1.0 / psi + tol):
                return False
        prod = vartheta * a * mu                            # rho^2 in (0, 1)
        if not 0.0 < prod < 1.0:
            return False
        rho2 = 1.0 - math.sqrt(prod)
        m_psi = mu * psi ** 2 - L * (1.0 - psi) ** 2
        # S_1:  rho^2 (1 - alpha m_psi / vartheta) <= (1 - alpha psi mu)^2
        return rho2 * (1.0 - a * m_psi / vartheta) - (1.0 - a * psi * mu) ** 2 <= tol

    def is_feasible(self, alpha, psi, L, mu):
        """``(alpha, psi)`` version of :meth:`in_stable_set`."""
        return self.in_stable_set(self.vartheta_of(alpha, psi, L), psi, L, mu, alpha=alpha)

    # ------------------------------------------------------------------ #
    # EV@R bound (Proposition 3, Theorem 3)                                #
    # ------------------------------------------------------------------ #

    def lyapunov_coeff(self, vartheta, psi, alpha, beta, gamma, L, mu):
        """``v_{vartheta,psi}`` of Proposition 3."""
        return (2.0 * L ** 2 / mu
                * (2.0 * (beta - gamma) ** 2 + (1.0 - alpha * L) ** 2 * (1.0 + 2.0 * gamma + 2.0 * gamma ** 2))
                + vartheta / (2.0 * alpha) * (1.0 - math.sqrt(vartheta * alpha * mu)))

    def theta_upper(self, vartheta, alpha, v, L, mu):
        """``theta_u^g`` of eq. (37): the entropic risk is finite for theta < theta_u."""
        s = math.sqrt(vartheta * mu)
        return 2.0 * s / (alpha * (8.0 * v * math.sqrt(alpha) + s * (vartheta + alpha * L)))

    def rho_bar(self, vartheta, alpha, v, theta, L, mu):
        """Rate ``rho_bar^2`` of eq. (40) at ``theta = theta_phi``."""
        c = alpha * (vartheta + alpha * L)
        denom = 2.0 - theta * c
        a = 1.0 - math.sqrt(vartheta * alpha * mu) + 4.0 * theta * alpha ** 2 * v / denom
        return 0.5 * a + 0.5 * math.sqrt(a ** 2 + 16.0 * theta * alpha ** 2 * v / denom)

    def evar_bound_vp(self, vartheta, psi, L, mu, alpha=None, sigma=1.0, zeta=None, d=None):
        """
        Asymptotic EV@R bound ``E_bar_{1-zeta}(vartheta, psi)`` of eq. (44).

        Returns ``inf`` where the bound is undefined (denominators non-positive
        or ``rho_bar >= 1``).
        """
        zeta = self.zeta if zeta is None else zeta
        d = self.d if d is None else d
        if d is None:
            raise ValueError("problem dimension d is required (set .d or pass d=)")
        alpha, beta, gamma = self.parameters(vartheta, psi, L, mu, alpha=alpha)
        v = self.lyapunov_coeff(vartheta, psi, alpha, beta, gamma, L, mu)
        theta = self.phi * self.theta_upper(vartheta, alpha, v, L, mu)
        c = alpha * (vartheta + alpha * L)
        denom = 2.0 - theta * c
        if not (theta > 0.0 and denom > 0.0):
            return float("inf")
        rho_bar = self.rho_bar(vartheta, alpha, v, theta, L, mu)
        if not 0.0 < rho_bar < 1.0:
            return float("inf")
        log_z = math.log(1.0 / zeta)
        if log_z <= d / (2.0 * (1.0 - rho_bar)) * (theta * c / denom) ** 2:      # condition (41)
            bound = c / 2.0 * (math.sqrt(d / (1.0 - rho_bar)) + math.sqrt(2.0 * log_z)) ** 2
        else:
            bound = d * c / ((1.0 - rho_bar) * denom) + 2.0 * log_z / theta
        return sigma ** 2 * bound

    def evar_bound(self, alpha, psi, L, mu, sigma=1.0, zeta=None):
        """``(alpha, psi)`` version of :meth:`evar_bound_vp` (uses ``self.d``)."""
        return self.evar_bound_vp(self.vartheta_of(alpha, psi, L), psi, L, mu,
                                  alpha=alpha, sigma=sigma, zeta=zeta)

    # ------------------------------------------------------------------ #
    # Gradient descent (beta = gamma = 0): Proposition 4, Theorem 4        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def gd_rate(alpha, L, mu):
        """Rate ``rho_GD(alpha)^2`` of noisy GD for f(x_k) - f*, ``rho_GD = max(|1 - alpha mu|, |1 - alpha L|)``."""
        return max(abs(1.0 - alpha * mu), abs(1.0 - alpha * L)) ** 2

    def gd_evar_bound(self, alpha, L, mu, d=None, sigma=1.0, zeta=None):
        """
        Asymptotic EV@R bound of noisy gradient descent, Theorem 4, eq. (50):

            sigma^2 alpha^2 L / (2 (1 - rho_GD^2)) * (sqrt(2 log(1/zeta)) + sqrt(d))^2,

        valid for ``alpha in (0, 2/(mu + L)]``; ``inf`` outside. Unlike GMM, the paper gives no
        problem of type (51) for GD, so this class only evaluates the bound.
        """
        zeta = self.zeta if zeta is None else zeta
        d = self.d if d is None else d
        if d is None:
            raise ValueError("problem dimension d is required (set .d or pass d=)")
        if not 0.0 < alpha <= 2.0 / (mu + L) * (1.0 + self.eps):
            return float("inf")
        rho2 = self.gd_rate(alpha, L, mu)
        return (sigma ** 2 * alpha ** 2 * L / (2.0 * (1.0 - rho2))
                * (math.sqrt(2.0 * math.log(1.0 / zeta)) + math.sqrt(d)) ** 2)

    # ------------------------------------------------------------------ #
    # Problem (51)                                                         #
    # ------------------------------------------------------------------ #

    def candidates(self, L, mu, grid_size=100, include_agd=True, agd_only=False):
        """
        Grid over ``S_c U S_0`` as in the authors' reference code
        (``sFoms.str_cnvx_stab_reg_params``): ``grid_size`` bins in psi on
        ``[0, 1)`` and on ``(1, 2 kappa (1 + sqrt(1 - 1/kappa))]``, and
        ``grid_size`` bins in vartheta over the admissible interval for each psi.
        Yields ``(vartheta, psi, alpha, region)``. With ``agd_only`` only the set
        ``S_0`` (AGD, ``gamma = beta``) is searched, on ``10 * grid_size`` values of alpha.
        """
        n = int(grid_size)
        if agd_only:
            for i in range(1, 10 * n + 1):
                yield 1.0, 1.0, i / (10 * n * L), "S_0"
            return
        kappa = L / mu
        psi_hi = 2.0 * kappa * (1.0 + math.sqrt(max(1.0 - 1.0 / kappa, 0.0)))
        psis = [i / n for i in range(n)] + [1.0 + (psi_hi - 1.0) * i / n for i in range(1, n + 1)]
        for psi in psis:
            if psi < 1.0:
                lo = 1.0 / (1.0 + kappa * (1.0 - psi))
                if psi > 0.0:
                    lo = max(2.0 - 1.0 / psi, lo)
                varthetas = [lo + (1.0 - lo) * i / n for i in range(n)]
                region = "S_minus"
            else:
                hi = min(2.0 - 1.0 / psi, 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * kappa * (psi - 1.0))))
                varthetas = [1.0 + (hi - 1.0) * i / n for i in range(1, n + 1)]
                varthetas[-1] = hi
                region = "S_plus"
            for vartheta in varthetas:
                if self.in_stable_set(vartheta, psi, L, mu):
                    yield vartheta, psi, self.alpha_of(vartheta, psi, L), region
        if include_agd:
            for i in range(1, n + 1):
                yield 1.0, 1.0, i / (n * L), "S_0"

    def solve_fixed_alpha(self, alpha, L, mu, d, grid_size=400):
        """
        RA-GMM at a **given step size**: minimize the EV@R bound (44) over ``psi`` on the curve
        ``vartheta = 1 - alpha L (1 - psi)``, subject to membership in the stable set ``S_c`` (Theorem 2).
        The rate constraint of (51) is not applied (``alpha`` is fixed by the caller, so the rate is whatever it is).
        Candidates ``psi`` lie on ``[0, 1)`` and ``(1, 2 kappa (1 + sqrt(1 - 1/kappa))]``; ``psi = 1`` (Nesterov,
        ``S_0``, needs ``alpha <= 1/L``) is included. Returns :class:`RAGMMSolution` or raises ``ValueError`` if no
        feasible ``psi`` has a finite bound.
        """
        if not (L > mu > 0.0):
            raise ValueError(f"need L > mu > 0, got L={L}, mu={mu}")
        if not alpha > 0.0:
            raise ValueError("alpha must be positive")
        self.d = d
        kappa = L / mu
        psi_hi = 2.0 * kappa * (1.0 + math.sqrt(max(1.0 - 1.0 / kappa, 0.0)))
        psis = [i / grid_size for i in range(grid_size)] + [1.0 + (psi_hi - 1.0) * i / grid_size
                                                           for i in range(1, grid_size + 1)]
        best = None
        for psi in psis:
            vt = self.vartheta_of(alpha, psi, L)
            if not self.in_stable_set(vt, psi, L, mu):
                continue
            bound = self.evar_bound_vp(vt, psi, L, mu, alpha=alpha)
            if math.isfinite(bound) and (best is None or bound < best[0]):
                best = (bound, vt, psi)
        if alpha <= 1.0 / L * (1 + self.eps):                       # S_0: Nesterov at this alpha
            bound = self.evar_bound_vp(1.0, 1.0, L, mu, alpha=alpha)
            if math.isfinite(bound) and (best is None or bound < best[0]):
                best = (bound, 1.0, 1.0)
        if best is None:
            raise ValueError(f"no feasible psi with a finite EV@R bound at alpha={alpha}; try a smaller alpha")
        bound, vt, psi = best
        a, beta, gamma = self.parameters(vt, psi, L, mu, alpha=alpha)
        rho2 = self.rate(vt, a, mu)
        region = "S_0" if psi == 1.0 else ("S_minus" if psi < 1.0 else "S_plus")
        return RAGMMSolution(a, beta, gamma, vt, psi, bound, rho2, rho2 / (1.0 - math.sqrt(mu / L)), region)

    def solve(self, L, mu, d, rate_slack=0.05, grid_size=100, include_agd=True,
              method="gmm") -> RAGMMSolution:
        """
        Solve problem (51) by grid search:

            minimize  E_bar_{1-zeta}(vartheta, psi)   over (vartheta, psi) in S_c U S_0
            s.t.      rho^2 / rho_*^2 <= 1 + rate_slack,   rho_*^2 = 1 - sqrt(mu / L).

        ``rate_slack`` is the trade-off parameter ``eps`` of (51b). The paper uses
        0.25 (quadratic example) and 0.05 (logistic regression). Note that
        ``rho^2 < 1`` always, so the constraint is vacuous when
        ``1 + rate_slack >= 1 / rho_*^2``; then the minimizer depends on the grid.
        Coarse grids can miss the narrow feasible band of S_+ entirely (e.g. 20 bins
        at kappa = 10); keep ``grid_size`` at 100 or more.

        ``method="agd"`` is the paper's RA-AGD: the same problem restricted to
        ``vartheta = psi = 1`` (``gamma = beta``, Nesterov's method), a one-dimensional
        search over ``alpha in (0, 1/L]`` with ``beta = (1 - sqrt(alpha mu)) / (1 + sqrt(alpha mu))``.
        """
        if method not in ("gmm", "agd"):
            raise ValueError(f"method must be 'gmm' or 'agd', got {method!r}")
        if not (L > mu > 0.0):
            raise ValueError(f"need L > mu > 0 (kappa > 1: for L == mu the reference rate "
                             f"rho_*^2 = 1 - sqrt(mu / L) is 0), got L={L}, mu={mu}")
        rho2_star = 1.0 - math.sqrt(mu / L)
        best = None
        for vartheta, psi, alpha, region in self.candidates(L, mu, grid_size, include_agd,
                                                            agd_only=(method == "agd")):
            rho2 = self.rate(vartheta, alpha, mu)
            if rho2 / rho2_star > 1.0 + rate_slack:
                continue
            bound = self.evar_bound_vp(vartheta, psi, L, mu, alpha=alpha, d=d)
            if best is None or bound < best[0]:
                best = (bound, vartheta, psi, alpha, rho2, region)
        if best is None or not math.isfinite(best[0]):
            raise RuntimeError("no feasible (vartheta, psi) with a finite EV@R bound; "
                               "increase rate_slack or grid_size")
        bound, vartheta, psi, alpha, rho2, region = best
        alpha, beta, gamma = self.parameters(vartheta, psi, L, mu, alpha=alpha)
        return RAGMMSolution(alpha=alpha, beta=beta, gamma=gamma, vartheta=vartheta, psi=psi,
                             evar_bound=bound, rate=rho2, rate_ratio=rho2 / rho2_star, region=region)
