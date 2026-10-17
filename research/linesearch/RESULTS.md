# Risk-averse line search along the full gradient: results

Design: `PLAN.md` (pre-registered; one deviation recorded before any performance result: the step range was widened from 2/L to 1/μ, because the
line search sat on the cap). Code: `linesearch_experiment.py`; raw: `results/<dataset>.json`. Stage 1 = 2,000 textbook SGD steps; polish budgets are in
**epochs of row gradients** (a full-batch step = 1 epoch; stochastic polish = `n B / 16` minibatch steps); L2 logistic regression λ = 0.01; 200 seeds.

## 1. The proposal as stated, and what it means

`α* = argmin_α (1/θ) log mean_i exp(−θ α a_i)`, `a_i = ∇F(w)ᵀ∇f_i(w)`, `w ← w − α* ∇F(w)` (sign convention: the leading minus in the message makes the
problem unbounded below once any `a_i < 0`, so the entropic **risk** of the loss change `−α a_i` is what is minimized).

* The risk is over **data points** (how the step changes each sample's loss), not over training runs. The first-order condition is
  `Σ_i a_i exp(−θ α a_i) = 0`: the step stops where the exponentially tilted mean improvement is zero.
* Small-θ (second-cumulant) closed form `α ≈ ‖∇F‖² / (θ Var_i a_i)`. Measured against the exact minimizer on Heart and Pima it **underestimates by 25–35%**,
  because the `a_i` are strongly positively skewed (skewness 2.1–2.5): the third cumulant matters, so use the exact 1-D minimization (cheap: one pass).
* Without a curvature term the first-order objective does not bound the step by itself when θ is small: R1 at θ = 1 took steps of 2–7/L, about those of an
  exact line search, sometimes larger. Only larger θ (5) makes it conservative (0.4–1.8/L).

## 2. Equal-compute comparison: CVaR₅% of `f(w) − f*` after polishing (lower is better)

| Dataset | Budget (epochs) | none (stage 1) | adapter | SGD avg | GD exact LS | R1 θ=1 (proposal) | R1 θ=5 | R3 θ=5 | R4 θ=1 |
|---|---|---|---|---|---|---|---|---|---|
| German | 1 | 0.380 | 0.09343 | 0.11189 | 0.11284 | 0.13378 | 0.30956 | 0.31818 | 0.17875 |
| German | 5 | 0.380 | 0.03532 | 0.04519 | 0.00978 | 0.0079 | 0.13367 | 0.15596 | 0.02054 |
| German | 20 | 0.380 | 0.01314 | 0.02912 | 0.00005 | 0.00001 | 0.00108 | 0.00455 | 0.01461 |
| COMPAS | 1 | 0.228 | 0.0108 | 0.01481 | 0.0701 | 0.07584 | 0.17368 | 0.18189 | 0.49513 |
| COMPAS | 5 | 0.228 | 0.0023 | 0.00799 | 0.01081 | 0.01159 | 0.07373 | 0.0864 | 0.01204 |
| COMPAS | 20 | 0.228 | 0.00073 | 0.0068 | <1e-5 | <1e-5 | 0.00049 | 0.00218 | 0.00923 |
| Pima | 1 | 0.161 | 0.03086 | 0.02354 | 0.02229 | 0.02493 | 0.11153 | 0.11945 | 0.03052 |
| Pima | 5 | 0.161 | 0.01225 | 0.00733 | 0.0003 | 0.00025 | 0.03321 | 0.04191 | 0.01812 |
| Pima | 20 | 0.161 | 0.00259 | 0.0039 | <1e-5 | <1e-5 | 0.00029 | 0.00093 | 0.01845 |
| Heart | 1 | 0.066 | 0.03362 | 0.03675 | 0.01186 | 0.02123 | 0.04347 | 0.04729 | 0.11254 |
| Heart | 5 | 0.066 | 0.01268 | 0.01513 | 0.00065 | 0.00065 | 0.01138 | 0.0149 | 0.0881 |
| Heart | 20 | 0.066 | 0.00608 | 0.0056 | <1e-5 | <1e-5 | 0.00006 | 0.00028 | 0.08738 |

(`none` = stage 1; the adapter and SGD average are the stochastic polishers from E2 at the same number of row-gradient epochs; R3 θ = 5 is the exact-change variant.)

## 3. What the risk term protects: rows whose loss gets worse (1 epoch; share of rows made worse / 95th percentile of per-row loss change / mean step ×L)

| Dataset (1 epoch) | GD exact LS | R1 θ=1 | R1 θ=5 | R3 θ=5 |
|---|---|---|---|---|
| German | 0.41 / +0.69 / 4.3 | 0.40 / +0.50 / 3.5 | 0.34 / +0.07 / 0.7 | 0.34 / +0.06 / 0.6 |
| COMPAS | 0.50 / +0.40 / 2.2 | 0.50 / +0.38 / 2.1 | 0.49 / +0.06 / 0.4 | 0.49 / +0.05 / 0.3 |
| Pima | 0.49 / +0.38 / 2.6 | 0.49 / +0.36 / 2.6 | 0.48 / +0.06 / 0.5 | 0.48 / +0.05 / 0.4 |
| Heart | 0.43 / +0.25 / 5.6 | 0.44 / +0.34 / 7.0 | 0.39 / +0.05 / 1.4 | 0.39 / +0.04 / 1.1 |

## Hypotheses

* **H1 mostly supported.** At 20 epochs, exact line search and R1 (θ = 1) beat the stochastic adapter on all four datasets (CVaR₅% at most 5e-5 vs 0.0007–0.013).
  At 5 epochs they win on German, Pima and Heart (3–50×) but **not on COMPAS** (adapter 0.0023 vs 0.0108), where n = 4,320 makes a full pass relatively
  costlier than 16-row steps. At 1 epoch the stochastic polishers win on German and COMPAS and lose on Heart and tie on Pima.
* **H2 partly supported:** R1/R3 at θ = 5 make fewer training rows worse than the exact line search (German 0.34 vs 0.41, Heart 0.39 vs 0.43; Pima 0.48 vs 0.49,
  COMPAS 0.49 vs 0.50) and cut the 95th-percentile per-row loss increase by 80–90% (German +0.07 vs +0.69; COMPAS +0.06 vs +0.40), **but at the
  same budget (1 epoch) their objective tail is 2.5–5× worse** (they take 4–6× smaller steps). At 20 epochs all methods have converged and the differences vanish
  (about half of the rows are "worse" than at the noisy stage-1 point).
* **H3 mixed:** R3's step is smaller than the line search's at θ = 1 and 5 on every dataset; R1 at θ = 1 is not (7.0 vs 5.6 on Heart).
* **H4 false:** R4 (risk of the new loss, step along the plain gradient) is the worst variant: it stalls (Heart CVaR₅% 0.087 at 20 epochs, German 0.015, Pima 0.018)
  and does not reduce the 95th-percentile row loss (Heart 1.31 vs 1.05). Tilting the objective needs a tilted **direction**, not only a tilted step.
* **H5 mixed** (see H1).

## What this means

1. **The proposal is a sound, cheap risk-averse line search that behaves as a data-driven damping of the step.** With θ ≈ 1 it is equivalent in the objective tail to an exact
   line search; with larger θ it trades speed for fewer harmed samples. Its natural use is **updating a deployed model** (protect individuals from regressions while
   improving the average), not repairing run-to-run variance.
2. **It exposes a baseline my adapter should have been compared with: deterministic full-batch descent on the head.** For these four problems (condition number ~50–150),
   five to twenty epochs of full-gradient steps with a line search reach the optimum's neighbourhood and make the seed-to-seed tail disappear, beating the stochastic
   adapter by at least 70× at 20 epochs (and by 3–50× at 5 epochs, except COMPAS). Newton or L-BFGS would be faster still. The earlier adapter gains
   are real relative to continuing SGD, but **when full passes over the head are affordable, solve the head deterministically**; the stochastic adapter's role is limited
   to the regime where a handful of full passes are not available (very large n, streaming data, or one pass per update) and the first 1–5 epochs.
3. The tail the paper measures (across runs) and the tail this proposal measures (across data points) are different and need different tests: for the second I would measure
   the **churn** (decisions flipped between `w_in` and the updated model) against the objective improvement as θ varies; not done here.

## Caveats

* Four datasets, κ ≈ 50–150, head = the full model; 200 seeds that differ only in the stage-1 start. First-order and exact variants share the same 80-point grid plus golden section.
* "Rows worse" is relative to the noisy stage-1 weights; near the optimum roughly half of the rows are worse than at that point, so the metric is informative only at small budgets.
* The adapter's parameters were chosen once per dataset and budget from 6 pilot runs.
