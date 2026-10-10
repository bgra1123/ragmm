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
