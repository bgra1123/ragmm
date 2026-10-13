# Can RA-GMM help at a given learning rate, and can it polish any solver's output? (written before running)

**Motivation (from the credit report).** RA-GMM's gain in the report equals the gain from plain SGD with a smaller step
(`research/tail_analysis/RESULTS.md`): its automatic choice of a small step does the work. Two questions follow:

* **E1. Fixed learning rate.** If the user fixes the step size α, can the paper's machinery choose the *momentum*
  (β, γ) so that the tail is better than SGD at that α and better than textbook momentum at that α?
* **E2. Polish.** Train with any solver first (here: textbook SGD, and a consensus ADMM stopped early), then run a few
  more stochastic RA-GMM-style rounds. Does that lower the tail of the final objective and the number of changed
  decisions, compared with continuing, decaying the step, or averaging?

Setup as the credit report: L2 logistic regression, λ = 0.01, batch 16, six datasets (E1) / four small ones (E2),
decision threshold 0.5, held-out rows as in `research/gate0`.

## E1: fixed learning rate

α ∈ {1, 0.3, 0.1, 0.03} / L; K = 2,000 steps, 400 seeds per cell (identical minibatches across methods).

| Method (all at the same α) | β, γ |
|---|---|
| SGD | 0, 0 |
| Nesterov (textbook momentum) | β = γ = (1−√(αμ))/(1+√(αμ)) |
| **RA-GMM at α** | ψ chosen on the paper's curve ϑ = 1 − αL(1−ψ), inside the stable set S_c, to minimize the paper's EV@R bound (eq. 44); (β, γ) from (35b, 35c). The rate constraint of (51) is dropped, since α is fixed by the user. |
| **Exact-EV@R at α (extension)** | (β, γ) minimizing the exact finite-horizon EV@R of f(x_K) − f* at K under the quadratic model at the exact solution (Hessian H, gradient-noise covariance Σ/b from per-row gradients, start e₀ = −w*): β ∈ {0, .3, .5, .7, .8, .9, .95, .98}, γ ∈ {0, β/2, β}. Uses the exact solution: an oracle, labelled as such. |
| SGD at the effective step α/(1−β_RA) | 0, 0 (control: is RA-GMM at α just a bigger step?) |

Reference rows (not at α): carefully tuned SGD (step + tail averaging), as in the report.

Metrics (held-out rows): objective S = f(w_K) − f*: mean, 95th percentile, CVaR₅%; decisions changed vs exact model per
retrain (mean, worst 5%); coin-flip rows; accuracy. Paired bootstrap over seeds for CVaR₅% and mean changed decisions.

Hypotheses:
* **H1** At α ≤ 0.1/L, RA-GMM at α has lower CVaR₅% of S than SGD at α on ≥ 4 of 6 datasets (momentum is chosen to help).
* **H2** At every α, RA-GMM at α has lower CVaR₅% than textbook Nesterov at the same α on ≥ 4 of 6.
* **H3** The exact-EV@R selection has CVaR₅% ≤ min(SGD at α, RA-GMM at α) on ≥ 4 of 6 at every α.
* **H4 (open)** RA-GMM at α vs SGD at the matched effective step: a tie within the bootstrap interval is expected.
* **H5 (open)** At α = 1/L no momentum choice reaches the tail of SGD with a 30× smaller step: the step size is the
  dominant lever.

## E2: polish

Stage 1 (200 seeds; the seed controls minibatch order for SGD and the random 4-way row partition for ADMM):
* **SGD:** α = 1/L, K₁ = 2,000 steps (noisy).
* **Consensus ADMM:** 4 random blocks, ρ = 0.25, 5 iterations, 2 Newton steps per local solve (stopped early).

Stage 2: K₂ more stochastic steps (batch 16) from the stage-1 output; K₂ ∈ {200, 1000}.

| Polisher | Settings |
|---|---|
| none | – |
| continue | SGD at α = 1/L |
| decay | SGD at α = 0.05/L |
| average | SGD at α = 1/L, average of the iterates of the K₂ steps |
| RA-GMM (51) | the report's RA-GMM parameters (α ≈ 0.01/L, β ≈ 0.98) |
| **Exact-EV@R polish** | (α, β, γ) minimizing the exact EV@R of f(x_{K₂}) − f* under the quadratic model, chosen once per dataset and stage from 10 pilot seeds: Hessian at the pilot mean, Σ/b at the pilot mean, e₀ per pilot = −H⁻¹∇f(w₁) (one Newton step; no exact solve), objective averaged over the pilots; grid α ∈ {.003,.01,.03,.1,.3,1}/L × β ∈ {0,.5,.9,.98} × γ ∈ {0,β}. |

Metrics: S (mean, p95, CVaR₅%) and decisions changed vs exact model (mean, worst 5%) after polishing; share of seeds whose
S decreased; paired bootstrap for CVaR₅% against "none".

Hypotheses:
* **H6** Exact-EV@R polish lowers CVaR₅% of S relative to no polish for both stage-1 solvers on all four datasets.
* **H7** Exact-EV@R polish has CVaR₅% within 10% of, or below, the best fixed polisher (decay, average, RA-GMM (51)) on ≥ 3 of 4 datasets.
* **H8** Polishing early-stopped ADMM lowers its tail (it is not already at the noise floor).
* **H9 (open)** Polishing with large-step SGD ("continue") does not lower the tail.

If H6 and H7 hold, a `polish` option is added to the package; otherwise the result is reported and the package is unchanged.
Everything is reported whichever way it falls.
