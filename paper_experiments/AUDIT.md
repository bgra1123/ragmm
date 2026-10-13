# Audit of `EntropicsFOMs` (strongly convex RA-GMM path) against the paper

Reproduce every number below with
`python -W ignore paper_experiments/audit_reference_code.py`
(needs `pip install pandas plotly tabulate` to import the reference package).

**Scope.** Only the strongly convex, non-quadratic path was audited:
`sFoms.str_cnvx_stab_reg_params`, `sFoms.str_cnvx_agd_stab_reg_params`,
`sFoms.str_cnvx_evar_bound`, `entropic_ra_sFoms.raTMM_str_cnv_params`
and `raAGD_str_cnv_params`. The quadratic helpers (`quad_*`, `raGD/raAGD/raTMM_quad_params`)
and `str_cnvx_agd_evar_bound` were **not** audited.

**What "the paper" means here.** The printed Theorem 2 (eq. 35, sets S_-, S_+, S_1, S_0),
Proposition 3 (eqs. 37, 39), Theorem 3 (eqs. 40, 41, 44), problem (51), and the proof of
Proposition 3 (Appendix B.2), from which eq. (40) can be re-derived.

I did not have the published figures' data (the notebook outputs are cleared), so I cannot say
which of these differences were present when the paper's figures were produced.

## Findings

| # | Location | What the code does | What the paper says | Fix |
|---|---|---|---|---|
| 1 | `sFoms.py:809`, `:846` | `stable_rate = np.append(rho2, stable_rate)` **prepends**, while `stable_vart`/`stable_psi`/`stable_alpha` append. After `np.argsort(stable_psi)` the rate column belongs to different rows than its (vartheta, psi). | Constraint (51b) is `rho^2(vartheta, psi) / rho_*^2 <= 1 + eps` for the same (vartheta, psi) | `stable_rate = np.append(stable_rate, rho2)` |
| 2 | `sFoms.py:952-954` | `bbrho_1 = 0.5*(a)`; `bbrho_2 = bbrho_1**2 + 16(...)`; `bbrho = 0.5*bbrho_1 + 0.5*sqrt(bbrho_2)` | Eq. (40): `rho_bar^2 = a/2 + (1/2) sqrt(a^2 + 16 theta alpha^2 v/D)`, `a = 1 - sqrt(vartheta alpha mu) + 4 theta alpha^2 v/D`. This is the root of `x^2 = (rho^2 + c)x + c` that the recursion in the proof (eq. 105) gives | `bbrho_1 = 1 - np.sqrt(...) + (...)` (no leading `0.5*`) |
| 3 | `sFoms.py:949` | `theta_u = sqrt(vartheta mu)/(alpha(8 v sqrt(alpha) + alpha sqrt(vartheta mu)(vartheta + alpha L)))` | Eq. (37): `2 sqrt(vartheta mu)/(alpha(8 v sqrt(alpha) + sqrt(vartheta mu)(vartheta + alpha L)))`, equal to `2(1-rho^2)/(8 alpha^2 v + (1-rho^2) alpha^2 (L + 2 lambda_P))` in the proof | add the factor 2; remove the extra `alpha` |
| 4 | `sFoms.py:941` | `beta = 1 - sqrt(.)/(1 - alpha psi mu)*(1 - sqrt(.))` (operator precedence) | Eq. (35b): `beta = (1 - sqrt(vartheta alpha mu))/(1 - alpha psi mu) * (1 - sqrt(alpha mu/vartheta))`, which is what `raTMM_str_cnv_params` uses; at (vartheta, psi) = (1, 1) it must give the AGD value `(1 - sqrt(alpha mu))/(1 + sqrt(alpha mu))` | parenthesize the numerator |
| 5 | `sFoms.py:956` | `2 - theta_var*(varth + alpha*L)` | Eq. (41): `2 - theta_phi alpha (vartheta + alpha L)` | add the missing `alpha` |
| 6 | `entropic_ra_sFoms.py:1-2`, `sFoms.py:9-10` | `import EntropicFOMs` | the folder in this repository is `EntropicsFOMs` | rename the folder or the imports |

Items 2-5 are inside `str_cnvx_evar_bound`, so the bound is evaluated at a different beta than the
algorithm runs with (item 4) and with a different rate and theta (items 2-3). Item 1 is in the
stable-set builder used by both `raTMM_str_cnv_params` and `raAGD_str_cnv_params`.

## Evidence

**Part A: the bound, against the exact EV@R (formula-independent).** For a quadratic with Gaussian
noise the stationary state is Gaussian, so the exact EV@R of `f(x) - f*` has a closed form
(`tests/helpers.py`). 232 random stable (vartheta, psi) points and zeta in {0.05, 0.95};
L = 1, mu = 0.1, d = 10, sigma^2 = 1. A "violation" is a bound below the exact EV@R.

| Bound | Violations |
|---|---|
| Paper's formulas (`risk_averse_tuning`) | **0 / 232** (conservative: 165x to 817x the exact value) |
| Reference code as is | **6 / 232** (exact EV@R up to 2.9x the "bound") |
| All fixed except item 2 | 13 / 232 (up to 5.3x) |
| Only item 2 fixed | 0 / 232 |

My port of the reference function reproduces the reference code exactly (max relative difference 0)
and, with all four items fixed, equals `risk_averse_tuning` to 4e-16. So **item 2 is the deviation
that makes the code's number fall below the true EV@R**; items 3-5 differ from the printed
formulas but did not break validity on these cases.

**Part B: item 1.** For L = 44.131, mu = 1 (the paper's logistic setting) the rate column returned by
`str_cnvx_stab_reg_params` equals `rho^2` of the rows in *reverse* order (`np.allclose` is True for the
reversed array, False for the aligned one). With eps = 0.05 the reference code keeps 1,530 rows;
only 488 of them satisfy `rho^2 <= 1.05 rho_*^2` (68% violate it, worst ratio 1.159). The AGD helper
(`str_cnvx_agd_stab_reg_params`, line 846) has the same problem: its rate column matches `1 - sqrt(alpha mu)`
only in reverse order (99 rows).

**Part C: effect on the selected parameters** (the paper's logistic-regression setting: data_std 5, N = 1000,
d = 100, lambda_reg = 1, giving L = 44.131, mu = 1; the same instance the notebook builds):

| zeta, eps | Reference code | `risk_averse_tuning` |
|---|---|---|
| 0.95, 0.05 (the notebook's values) | alpha 0.0003, beta 0.966, gamma 0.251, `rho^2/rho_*^2` = **1.157 (violates 1.05)** | alpha 0.0118, beta = gamma 0.804 (set S_0, an AGD-type point), ratio 1.049 |
| 0.05, 0.05 | alpha 0.0003, beta 0.967, gamma 0.193, ratio **1.158 (violates 1.05)** | same as above |
| 0.95, 0.25 | alpha 0.0003, beta 0.966, gamma 0.251, ratio 1.157 (meets 1.25) | alpha 0.0002, beta 0.971, gamma 0, ratio 1.160 |

At eps = 0.25 the two agree in order of magnitude; at eps = 0.05, the value used in the paper's
logistic-regression experiment, they do not.

**Part D: the GMM update itself.** `risk_averse_tuning.GMM` against the reference `train()` on the paper's
logistic-regression problem (L = 44.13, mu = 1, 300 steps, noiseless so both are deterministic):

| Case | max abs difference in the last iterate | vs `torch.optim.SGD` |
|---|---|---|
| GD (beta = gamma = 0) | 1.4e-17 | SGD: 1.4e-17 |
| heavy ball (gamma = 0) | 9.7e-17 | SGD with momentum: 1.4e-17 |
| AGD (gamma = beta) | 1.1e-16 | Nesterov SGD: matches in terms of the lookahead point (`tests/test_gmm.py`) |
| general (beta = 0.6, gamma = 0.35) | 4.2e-17 | n/a |

(The reference's stored f-path is rounded to 5 decimals, so its f-differences, about 5e-6, are rounding.)
So the optimizer step is correct in the special cases; the differences above are in the *parameter-selection*
code, not in `train()`.

## What I did not check

* whether the published Figure 6 / Table values came from this version of the code;
* whether the qualitative conclusions of the logistic-regression experiment survive with parameters that
  satisfy the constraint (this needs the experiment to be re-run; I did not do that);
* the quadratic-case code and the AGD bound `str_cnvx_agd_evar_bound`.

## A convention question for the authors

In eq. (11)-(12), `P{f - f* >= EV@R_{1-zeta}} <= zeta`, and `zeta` enters only through `log(1/zeta)`.
The experiments pass `zeta = 0.95` and `0.99` and call it a "confidence level"; in the formulas that is a
95-99% *tail probability* (`log(1/0.95) = 0.05`), i.e. an almost risk-neutral setting. If a 95% guarantee
was intended, the formulas need `zeta = 0.05`. `risk_averse_tuning` follows the formulas and documents this.
