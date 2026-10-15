# Other optimizers (Adam, RMSprop, Adagrad, momentum SGD) and the RA-GMM polish (written before running)

Question (user): does the instability story, and the adapter ("train as usual, then a short RA-GMM-style polish"), hold
for Adam and other common optimizers, not only SGD?

Setup as the credit report: L2 logistic regression (λ = 0.01), batch 16, K₁ = 2,000 steps (all optimizers, so the
budget is comparable with earlier results), threshold 0.5, held-out rows as in `research/gate0`. Datasets: German, COMPAS,
Pima, Heart. **300 seeds**, same minibatch sequence for every optimizer. The gradient includes the L2 term.

## Stage 1: optimizers at library defaults

| Optimizer | Update (PyTorch semantics) | Learning rates |
|---|---|---|
| SGD | plain | 0.001 (torch default), 0.01, 0.1 |
| SGD + momentum 0.9 | heavy ball (torch `SGD(momentum=.9)`) | 0.001, 0.01, 0.1 |
| SGD + Nesterov 0.9 | torch `nesterov=True` | 0.001, 0.01, 0.1 |
| Adam (0.9, 0.999, eps 1e-8) | torch `Adam` | 0.001 (torch default), 0.01, 0.1 |
| RMSprop (alpha 0.99, eps 1e-8) | torch `RMSprop` | 0.001, 0.01 (torch default), 0.1 |
| Adagrad (eps 1e-10) | torch `Adagrad` | 0.001, 0.01 (torch default), 0.1 |

The vectorized implementations are checked against `torch.optim` on one seed (max abs difference < 1e-8) before the run.

**Metrics** (per optimizer × lr): objective S = f(w) − f* (mean, 95th percentile, CVaR₅%); decisions that differ from the
exact model per retrain (mean, worst 5%); **retrain-vs-retrain disagreement** (probability that two retrains disagree on
a random held-out row: pure noise, no bias); **bias** ‖mean of final weights − w*‖ / ‖w*‖ (the part of the error that
is not run-to-run noise); mean AUC. This separates "unstable" from "not converged", which matters for small learning
rates (a small lr gives stable but under-trained models, and "differs from the exact model" mixes both).

## Stage 2: polish after each stage-1 configuration (K₂ = 1,000 SGD-type steps from the stage-1 weights)

| Polisher | Settings |
|---|---|
| none | – |
| average | SGD at 1/L, average of the polishing iterates |
| decay | SGD at 0.05/L |
| RA-GMM (51) | the paper's problem (51) parameters |
| **Exact-EV@R polish (adapter)** | step and momentum minimizing the exact finite-horizon EV@R of f(x_{K₂}) − f* under the quadratic model at the stage-1 mean weights (Hessian, gradient-noise covariance / batch, e₀ = Newton step from each of 6 pilot seeds, averaged), grid α ∈ {.003,.01,.03,.1,.3,1}/L × β ∈ {0,.5,.9,.98} × γ ∈ {0, β} |

Same metrics, plus the share of seeds whose S decreased. Paired bootstrap over seeds for CVaR₅% against "none".

## Hypotheses

* **H1** The ranking of optimizer configurations by tail (CVaR₅% of S) differs from the ranking by mean S (Spearman < 0.9) on ≥ 2 of 4 datasets: the tail is not the mean.
* **H2** The adapter lowers CVaR₅% relative to "none" in ≥ 80% of (optimizer, lr, dataset) cells where stage-1 CVaR₅% > 0.001.
* **H3** The adapter's CVaR₅% is below or within 20% of the better of {average, decay} in ≥ 70% of those cells.
* **H4** The adapter lowers mean changed decisions by ≥ 30% in ≥ 60% of cells where stage 1 changes ≥ 3 decisions.
* **H5 (open)** Adaptive methods (Adam, RMSprop, Adagrad) at lr ≥ 0.01 leave a larger tail than SGD at the same lr (their per-coordinate step is not controlled by 1/L); the adapter still helps them.
* **H6 (open)** Where the stage-1 retrains are already stable but biased (small lr), polishing reduces the bias too.

Reported whichever way it falls; the plan's pivot if H3 fails: ship the simplest winner (average or decay) and keep the
EV@R machinery for the report only.
