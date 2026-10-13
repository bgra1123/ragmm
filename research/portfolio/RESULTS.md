# Results — Experiment P1: monthly rebalancing backtest (R1)

**Run:** 10 US stocks with full history (AAPL, GE, AMD, WMT, BAC, T, XOM, BBY,
PFE, JPM), 328 monthly rebalances from 1991-01 to 2018-04, 200 solver seeds
per method, K = 200 stochastic steps per rebalance, minibatch of 16 days.
Design, hypotheses and two changes recorded before the main run: `PLAN.md` §10.
Raw summary: `results/backtest_main.json`. Figures: `results/figures/`.
Reproduce: `python research/portfolio/backtest.py` (~3 minutes), then `make_figures.py`.

**Not investment advice.** The stock list was chosen in 2018 (survivorship
bias), so absolute returns are inflated; compare methods with each other and
with the exact solver, not with the market.

## Reference: the exact solver

Exact solution of R1 at every rebalance (no randomness): 23.3% annualized
return, 23.1% volatility, Sharpe 1.02, max drawdown 47%, terminal wealth
303× over 27 years, 28.7% average monthly turnover.

## Headline numbers

| Method | Tuning runs | Phantom trades per rebalance (mean) | 95th pct phantom | Tracking error vs exact (ann.) | Sharpe: mean / worst 5% | Terminal wealth: 5th–95th pct | Ruined |
|---|---|---|---|---|---|---|---|
| SGD (textbook) | 0 | 192% | 314% | 1.69 | 0.42 / −0.18 | ~0 – 993 | 17% |
| Nesterov (textbook) | 0 | 211% | 345% | 3.51 | 0.34 / −0.23 | ~0 – 783 | 34% |
| Heavy-ball (textbook) | 0 | 220% | 346% | 8.49 | 0.36 / −0.19 | ~0 – 930 | 28% |
| SGD (tuned for mean) | 8,640 | **18.4%** | **29.6%** | **0.034** | **1.02 / 0.97** | **238 – 386** | **0%** |
| Nesterov (tuned for mean) | 51,840 | 19.0% | 32.2% | 0.036 | 1.02 / 0.97 | 239 – 367 | 0% |
| SGD + tail averaging (tuned) | 8,640 | 19.5% | 33.4% | 0.037 | 1.02 / 0.98 | 236 – 358 | 0% |
| **RA-GMM (ε = 0.25)** | **0** | 168% | 292% | 1.67 | 0.52 / −0.03 | ~0 – 841 | 8.5% |
| RA-GMM (ε = 0.05) | 0 | 206% | 339% | 2.66 | 0.37 / −0.15 | ~0 – 915 | 30.5% |

"Phantom trades" = `‖w_K − w*‖₁`: the part of the portfolio that differs from
the exact solution only because of the solver's random seed, as a share of
wealth. "Ruined" = the account's wealth reached zero at some point.
"Tuning runs" counts solver runs used for tuning (configs × 24 dates × 20 seeds).

## Hypotheses

| | Result |
|---|---|
| **H1** RA-GMM (ε = 0.25) has a lower 95th percentile of suboptimality and phantom turnover than textbook Nesterov / heavy-ball | **Supported, modestly.** Suboptimality p95 2.83 vs 5.12 / 5.17; phantom-turnover p95 292% vs 345% / 346%. |
| **H2** RA-GMM has lower tracking error and a narrower wealth fan than textbook Nesterov / heavy-ball | **Supported for tracking error and ruin**: tracking error 1.67 vs 3.51 / 8.49; ruin 8.5% vs 34% / 28%; worst-5% Sharpe −0.03 vs −0.23 / −0.19. The wealth fan is still extremely wide (5th percentile ≈ 0 for all three). |
| **H3** (open) RA-GMM vs tuned baselines | **RA-GMM is far worse.** Phantom trades 168% vs 18–20%; tracking error 1.67 vs 0.03; 8.5% ruin vs none. |

## Why (diagnosis)

1. **The noise is very large relative to the problem.** Each stochastic
   gradient estimates annualized expected returns from 16 days of data. Good
   performance requires tiny steps: the tuned methods use α ≈ 0.006–0.011 / L.
2. **Problem (51) forces near-maximal steps here.** At this condition number
   (κ: median 2.3, max 7.9) the fastest certified rate is `ρ²_* = 1 − √(μ/L)`
   ≈ 0.34, so "within 25% of it" requires `√(ϑαμ) ≳ 0.57`, i.e. a large step.
   RA-GMM selected α ≈ 0.99 / L at the median date (minimum 0.32 / L), about
   100× the tuned step.
3. **The selection ignores the noise level and the budget.** The EV@R bound
   scales with σ², but the constrained minimizer of (51) does not depend on σ²,
   on the distance to the optimum, or on K. When noise dominates, the right
   answer is "go slower", and (51) cannot express that.
4. **Together with the loan experiment:** at large κ (loan, κ ≈ 99) the
   ε-constraint is vacuous and the step shrinks to the grid edge; at small κ
   (portfolio, κ ≈ 2) it is too tight and forces large steps. ε is a relative
   rate on ρ², which behaves very differently across condition numbers.

## What this means

- **The problem is real and large.** With textbook solver settings, the same
  strategy on the same data ends anywhere between ruin and ~1,000× depending
  only on the seed, and the account is wiped out in 17–34% of seeds. Even
  well-tuned solvers trade ~18% of wealth per rebalance purely because of the
  seed (vs 29% real turnover), though their outcomes stay close to the exact
  solver (terminal wealth 236–386× vs 303×).
- **RA-GMM as specified by problem (51) reduces ruin from 34% to 8.5% versus
  textbook Nesterov with zero tuning runs, but is not competitive with tuning.**
- The fix needs the selection to see the noise level and the budget: a
  **horizon-aware** criterion that minimizes the finite-K EV@R bound
  (Theorem 3, eqs. 42–43) using σ² and a bound on `‖x₀ − x*‖`. For this
  problem both are available: σ² from the per-day gradient spread, and
  `‖x₀ − x*‖ ≤ ‖∇f(x₀)‖ / μ`.

## Caveats

- 10 stocks: the exact solve is trivial. The stochastic solver stands in for
  large-universe, scenario-based or many-portfolio settings.
- Minibatch gradients use single trading days; the noise is heavy-tailed, not
  the Gaussian noise of the theory.
- Short positions are allowed and leverage is unconstrained; with constraints
  (long-only, leverage caps) phantom positions would be bounded and ruin rarer.

---

# Experiment P2: long-only, no leverage (exploratory)

Same backtest with `w ≥ 0`, `1ᵀw = 1`; projected solvers (`PLAN.md` §11).
Raw summary: `results/backtest_main_longonly.json`. Figures:
`results/figures/wealth_fans_longonly.png`, `phantom_trades_longonly.png`.
Reproduce: `python research/portfolio/backtest.py --long_only` (~6 minutes).

**Exact solver (long-only):** 22.9% annualized return, 21.3% volatility,
Sharpe 1.08, max drawdown 43%, terminal wealth 279×, 18.6% monthly turnover.

| Method | Phantom trades per rebalance (mean / p95) | Monthly turnover | Tracking error vs exact | Sharpe: mean / worst 5% | Max drawdown: mean / worst 5% | Terminal wealth: 5th / median / 95th |
|---|---|---|---|---|---|---|
| SGD (textbook) | 97% / 151% | 112% | 0.198 | 0.82 / 0.58 | 64% / 82% | 58× / 207× / 1083× |
| Nesterov (textbook) | 99% / 152% | 113% | 0.200 | 0.82 / 0.58 | 64% / 82% | 56× / 207× / 1057× |
| Heavy-ball (textbook) | 100% / 153% | 114% | 0.202 | 0.82 / 0.57 | 64% / 82% | 58× / 210× / 1047× |
| SGD (tuned for mean) | 15% / 26% | 15% | 0.027 | 1.06 / 1.03 | 46% / 50% | 229× / 257× / 291× |
| Nesterov (tuned for mean) | 15% / 26% | 15% | 0.027 | 1.06 / 1.03 | 46% / 50% | 225× / 257× / 294× |
| **SGD + tail averaging (tuned)** | **14% / 24%** | 16% | **0.025** | **1.07 / 1.04** | **45% / 49%** | **236× / 270× / 320×** |
| **RA-GMM (ε = 0.25)** | 89% / 143% | 103% | 0.170 | 0.85 / 0.64 | 59% / 74% | 59× / 198× / 784× |
| RA-GMM (ε = 0.05) | 98% / 151% | 113% | 0.198 | 0.82 / 0.58 | 63% / 82% | 54× / 209× / 1022× |

## Hypotheses

| | Result |
|---|---|
| **H1** RA-GMM lower tail of suboptimality / phantom trades than textbook momentum | **Supported, small effect.** Suboptimality p95 0.90 vs 1.23 / 1.24; phantom p95 143% vs 152% / 153%. |
| **H2** lower tracking error, narrower wealth fan | **Supported, small effect.** Tracking error 0.170 vs 0.200 / 0.202; worst-5% drawdown 74% vs 82%; the fan's top shrinks (784× vs ~1,050×) but its bottom does not (59× vs 56×). |
| **H3** (open) vs tuned baselines | **RA-GMM is far worse**: ~6× the phantom trades and tracking error of tuned SGD / averaging. |
| Expected: constraints shrink all gaps vs P1 | **Yes.** No ruin is possible; textbook phantom trades fall from ~210% to ~100% per rebalance. |

## What changes with the constraints

- **No account is wiped out**, and the worst seeds still end at ~56× (vs ruin
  in P1). The constraints cap the damage a noisy step can do.
- **The seed still matters a lot with default settings:** the same strategy
  ends anywhere from ~56× to ~1,060×, with ~100% of the portfolio traded each
  month purely because of the seed (real turnover of the exact strategy:
  19%). Monthly turnover is about 6× the intended level.
- **Tuning still removes most of it:** tail averaging keeps the fan at
  236×–320× around the exact solver's 279×.
- **RA-GMM's diagnosis is unchanged:** it selects α ≈ 1/L because of the rate
  constraint at κ ≈ 2, while the tuned methods use α ≈ 0.006–0.011/L. Against
  the textbook defaults it helps only slightly here.

---

# Diagnostic: can the EV@R bound itself choose worst-case-safe settings? (exploratory)

Idea tested: replace problem (51) by minimizing the paper's **finite-horizon**
EV@R bound (Theorem 3, eqs. 42–43) at the real budget K = 200, using the
measured gradient noise σ² and `V_P(z₀) ≤ ‖∇f(x₀)‖²/μ` (with zero initial
momentum; this reading of eq. 34 still needs the authors' confirmation). Code:
`paper_tuning.py` (`candidate_grid`, `bound_parts`, `horizon_aware_params`),
`bound_diagnostic.py`. Output: `results/bound_diagnostic.json`.

| Quantity (328 dates, exact-solution path) | Value |
|---|---|
| Dates where the bound's smallest possible noise term exceeds the starting error `f(x₀) − f*` | **96%** |
| Median ratio: bound's noise floor / starting error | 3.8× |
| Median ratio: bound's noise floor / tuned SGD's actual 95th percentile | **4.6×** |
| Step size selected by the bound (α·L) | ~1e-7 at every date, i.e. "do not trade" |
| Dates where tuned SGD's 95th percentile is below the starting error | 63% |

**Correction (later).** The bound examined here is the strongly-convex Theorem 3 bound used by problem (51). The portfolio
objective is an exact quadratic, for which the paper's Section 3 (Theorem 1, problem (32)) applies; it is tested in PLAN.md section 13.

**Reading.** In this noise regime the EV@R bound is about 4–5× looser than
what a well-tuned solver actually achieves in its worst 5% of runs, so the
bound cannot certify that *any* trade beats standing still, and its minimizer
freezes the portfolio. The bound is not wrong (it is an upper bound), but it
is too loose to choose parameters here. The last row is a separate,
interesting fact: on 37% of dates even the tuned solver's worst 5% of runs
ends up farther from the optimum than not rebalancing at all.

---

# Experiment P3: tune for the worst case (EV@R) vs the average

Design: `PLAN.md` §12 (pre-registered; search-space count corrected to 468
before the run). Code: `worst_case_tuning.py`. Raw results:
`results/worstcase.json`, `results/worstcase_longonly.json`. Figures:
`results/figures/worstcase_*`. Tuning: 468 configurations × 24 months
(1991–92) × 100 seeds. Evaluation: 200 different seeds, out-of-sample dates
1993–2018 for solver metrics, the whole backtest for wealth metrics.

## Selected settings (α = a/L)

| Setting | Average-tuned | Worst-case-tuned (EV@R) | Worst-case-tuned (CVaR) |
|---|---|---|---|
| Unconstrained | a = 0.0059, Nesterov β = 0.3 | a = 0.0106, Nesterov β = 0.3, **tail averaging** | a = 0.0010, Nesterov β = 0.9 |
| Long-only | a = 0.00032, Nesterov β = 0.95 | a = 0.00032, heavy-ball β = 0.95 | same as EV@R |

## Unconstrained (out-of-sample 1993–2018)

| Metric | Average-tuned | Worst-case-tuned (EV@R) | Difference (paired bootstrap 95% CI) |
|---|---|---|---|
| Phantom trades, 95th percentile | 32.4% | **29.2%** | −3.2 pts [−3.4, −3.1] (−10%) |
| Phantom trades, worst 5% (CVaR) | 38.3% | **33.8%** | −4.4 pts [−4.5, −4.3] (−12%) |
| Solver error, 95th percentile | 0.0282 | **0.0248** | −0.0035 [−0.0037, −0.0031] (−12%) |
| Solver error, worst 5% (CVaR) | 0.0418 | **0.0346** | −0.0072 [−0.0075, −0.0068] (−17%) |
| Solver error, mean | 0.0105 | **0.0097** | −0.0008 [−0.0008, −0.0008] (−7%) |
| Tracking error vs exact: mean / worst 5% | 0.036 / 0.038 | **0.034 / 0.036** | |
| Terminal wealth 5th / median / 95th | 239× / 295× / 367× | 243× / 303× / 381× | exact solver: 303× |
| Fan width (95th / 5th) | **1.53** | 1.57 | |
| Max drawdown, worst 5% | **51.8%** | 52.3% | |
| Monthly turnover | 25.0% | 28.2% | exact solver: 28.7% |

## Long-only (out-of-sample 1993–2018)

Both criteria select the same step size and momentum; they differ only in
γ (Nesterov vs heavy-ball). All metrics agree to three decimals (phantom
trades p95 25.6% vs 25.5%; terminal wealth 225× / 257× / 294× vs 225× / 258× / 294×).
This is a **tie**, as anticipated in the pre-registration.

## Hypotheses

| | Unconstrained | Long-only |
|---|---|---|
| **H1** lower worst-case solver error and phantom trades | **Supported** (−10% to −17% in the tails, CIs exclude 0) | Tie |
| **H2** narrower wealth fan, lower worst-5% tracking error | **Partly**: tracking error lower (0.036 vs 0.038 worst 5%); fan *not* narrower (1.57 vs 1.53, both ends higher) | Tie |
| **H3** worse average (price of risk aversion) | **Not supported**: the average also improved (−7%) out of sample | Tie |

## Reading

- Choosing the solver's settings by the **worst 5% of outcomes** (EV@R)
  instead of the average gave a modest but consistent improvement in the bad
  tail when positions are unconstrained, with no cost to the average. The
  selected solver uses iterate averaging, which smooths out unlucky steps.
- With long-only constraints, the constraints already cut off the bad tail, so
  both criteria pick the same solver.
- Scale: the big effect is **deliberate tuning vs default settings** (≈90%
  fewer phantom trades, no wipe-outs vs 34%). Worst-case vs average tuning is a
  second-order refinement on top (≈10–17% in the tails).
- This is empirical tuning with the paper's risk measure (EV@R) on past data,
  not the paper's theoretical parameter formula, which does not yet work in
  this noise regime (see the bound diagnostic above).

---

# P5: the paper's quadratic formulation on the quadratic portfolio problem

Design: PLAN.md §13 (pre-registered; one addendum after a 40-date smoke test). Unconstrained R1, 328 dates (out-of-sample
from date 24), 200 seeds, K = 200, batch 16, 10 bps. Data: `results/quadratic_formulation.json`,
`results/calibration_additive.json`. Code: `quadratic_formulation.py`, `calibration_additive.py`.
This tests the formulation the earlier P1–P3 should have used (§13): Section 3 of the paper, problem (32).

## Headline (out-of-sample dates, pooled over seeds × dates)

| Method | Mean subopt. | 95th pct | CVaR₅% | Phantom trades 95th pct | Ruin | Median terminal wealth |
|---|---|---|---|---|---|---|
| Exact solver (reference) | – | – | – | – | – | 303 |
| SGD (tuned for mean) | **0.0101** | **0.026** | **0.036** | 0.30 | 0% | 306 |
| Nesterov (tuned for mean) | 0.0105 | 0.028 | 0.042 | 0.32 | 0% | 295 |
| SGD + tail averaging (tuned) | 0.0109 | 0.030 | 0.044 | 0.34 | 0% | 290 |
| EXT: exact EV@R at K (extension; oracle start) | 0.0315 | 0.084 | 0.121 | 0.57 | 0% | 208 |
| RA-GMM, problem (51), ε = 0.25 | 1.20 | 2.75 | 8.7 | 2.90 | 9% | 36 |
| Q32-grid (paper's grid, ε = 1) | 1.45 | 2.94 | 12.6 | 2.83 | 14% | 32 |
| SGD (textbook) | 1.81 | 3.96 | 15.1 | 3.11 | 17% | 16 |
| Nesterov (textbook) | 15.5 | 4.98 | 287 | 3.40 | 34% | 5.7 |
| **Q32, ε = 0.25** (paper's problem (32)) | 26.0 | 5.28 | 494 | 3.52 | 37% | 3.7 |
| Q32, ε = 0.05 | 38.3 | 6.68 | 736 | 3.85 | 45% | 0.19 |
| Q32-exact (exact EV@R as objective), ε = 0.25 | 62.5 | 5.74 | 1224 | 3.63 | 42% | 1.3 |

(Means of the untuned methods are dominated by a few runs where the account is wiped out; compare the 95th percentile
and the ruin rate.)

## Hypotheses (as pre-registered)

- **H1 – false.** Q32 (ε = 0.25) is *worse* than textbook Nesterov and SGD at the 95th percentile (+0.30, CI [0.24, 0.35];
  +1.32, CI [1.25, 1.40]) and in phantom trades; the mean differences are not distinguishable from zero against
  Nesterov (heavy tails) and worse against SGD. Only the paper's-grid variant (ε = 1) beats the textbook settings,
  and that comparison was not pre-registered.
- **H2 – false.** Q32 is worse than problem (51) (95th pct +2.52, CI [2.42, 2.63]).
- **H3 – direction as expected, magnitude not.** Q32 is about 200× worse than the tuned baselines at the 95th
  percentile, with 37% of accounts ruined against 0%.
- **H4 – half true.** EXT is far better than Q32 (95th pct 0.084 vs 5.28; CVaR 0.12 vs 494), but its mean is 3.1× the
  best tuned-for-mean baseline, not within 10%, and it is worse than the tuned baselines on every metric.
- **H5 – calibration: EXT true; the bound claim false.** EXT's exact-K prediction matched the empirical EV@R at 40/40
  dates (median ratio 1.003, within 1.5× everywhere) *with the real minibatch noise*. For textbook Nesterov and Q32 the
  prediction was 4.2–4.5× too low (empirical / predicted, median), and the Theorem 1 bound (isotropic σ² =
  tr Σ / d) was **below** the empirical EV@R (median ratio 0.14–0.16), i.e. the guarantee does not hold for these
  settings in this problem.

## Diagnosis: the noise is not the paper's noise

With the real noise replaced by additive Gaussian `N(0, Σ)` (same Σ; the paper's model), the exact-K prediction is right
for all three settings: empirical / predicted median 0.89 (Nesterov, range 0.77–1.10), 0.91 (Q32, 0.78–1.22), 0.99 (EXT,
0.91–1.07). (The first two are slightly below 1 because a 2 000-sample estimate of EV@R of a heavy-tailed variable
is biased low; my pre-registered "within 10%" was met at 47–50% of dates for them, 100% for EXT.) With the real
minibatch noise the same settings come out 4.2–4.5× higher. The minibatch gradient of this objective has a part that
grows with `w` (the `θ D_t D_tᵀ w` term), so large steps that make the weights large also make the noise larger and
can destabilize the iteration in mean square; the additive-noise theory cannot represent that. EXT avoids it
by choosing tiny steps (α·L between 2e-4 and 2e-3 on the dates checked), where the noise is effectively additive.

## What this means

1. **The paper's quadratic formulation is correct as mathematics and does what it says on its own model.** The exact
   EV@R is computable, and with additive noise it predicts the simulated tail to within the sampling error.
2. **It does not transfer to this portfolio problem.** Problem (32) asks for the fastest-converging settings (within ε of
   the optimal rate); on this objective the minibatch noise is state-dependent, and those settings are unstable in
   mean square. Zero-tuning use of (32) is worse than textbook SGD, and ruins more accounts than P1's (51).
   Problem (32) also never sees the budget K or the noise level, and its argmin does not depend on σ, so it cannot
   back off the step size.
3. **An exact finite-horizon EV@R (my extension) is a good predictor and a good selector relative to (32), but still
   does not beat a tuned-for-mean SGD here.** Its selections are well calibrated from the reference start. In the
   backtest it starts each month from its own holdings, which are farther from the optimum than the reference
   path's (mean starting gap `f(w₀) − f*` over the first 150 dates, 40 seeds: reference path 0.027, EXT 0.062, tuned SGD
   0.037; post-hoc check): a lazy solver inherits a larger error next month, which the one-month criterion with the
   reference start does not see. I did not test a self-consistent version; this explanation is supported by that
   measurement but untested as a fix.
4. **Corrected statement for the argument (ARGUMENT.md):** the earlier "bound 4–5× too loose" referred to problem (51)'s
   bound. The quadratic bound (Theorem 1) is not loose here; it is *invalid* for this noise, because its model
   (additive, stationary) does not describe minibatch gradients of this objective.

## Caveats

* One objective, one universe, one noise level; the conclusion is about this problem's noise structure, not about the paper's
  theory on additive-noise problems.
* EXT uses the reference path's holdings for e₀ (an oracle a deployed solver would not have); its backtest numbers are
  not a deployable result.
* Tuned baselines were tuned on 24 dates and tested out of sample; Q32 variants use no data.
* Textbook, Q32 and (51) means are driven by wiped-out accounts; use the 95th percentile and ruin rate.
* The diagnostic's sample EV@R from 2 000 draws is biased low for heavy tails.
