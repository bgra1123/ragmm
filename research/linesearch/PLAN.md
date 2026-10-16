# Risk-averse line search along the full gradient (user's proposal) vs polishing baselines at equal compute (written before running)

## The proposal, precisely

Given weights `w` (any trained model; head weights if the body is frozen), the ERM objective `F(w) = (1/N) Σ_i f_i(w)` and
its gradient `g = ∇F(w)`. For each sample, `a_i = gᵀ∇f_i(w)` is the first-order **directional derivative** of the sample's loss
along `-g` (`a_i > 0`: sample i improves; `a_i < 0`: it gets worse; the mean of `a_i` is `‖g‖²`). The step size is

`α* = argmin_α (1/θ) log( (1/N) Σ_i exp(−θ α a_i) )`,   update `w ← w − α* g`.

(Reading note: the problem as first typed has a leading minus sign, `−(1/θ) log(...)`; that is unbounded below as soon as some
`a_i < 0`, so I read it as the entropic risk `+(1/θ) log(...)` of the first-order loss change `−α a_i`, which is well posed.)
Stationarity: `Σ_i a_i exp(−θ α a_i) = 0`, i.e. the step stops where the exponentially tilted average improvement is zero. For small θ,
`α* ≈ ‖g‖² / (θ Var_i(a_i))`; θ → 0 removes the cap (no curvature in a first-order model), θ → ∞ gives `α* → 0` if any `a_i < 0`.
The randomness being protected against is the **spread over data points** (some samples get worse while the average improves), not the
run-to-run noise of training; for a minibatch of size b, `Var_i(a_i)/b` is exactly the variance of a minibatch step along `g`.

## Variants run (each applied T times, full-batch, from the stage-1 weights)

| Name | Step size rule |
|---|---|
| **R1 (proposal)** | first-order entropic risk of the change, as above, α ∈ (0, 2/L] |
| R3 | entropic risk of the **exact** per-sample change `f_i(w − αg) − f_i(w)` (no first-order approximation), α ∈ (0, 2/L] |
| R4 | entropic risk of the new per-sample **loss** (tilted: `(1/θ) log mean exp(θ f_i(w − αg))`), α ∈ (0, 2/L] |
| GD-LS | exact line search on the mean loss `F(w − αg)` (the θ → 0 limit of R3), α ∈ (0, 2/L] |
| GD-1/L | fixed step 1/L |

θ ∈ {1, 5} (features are standardized, L ≈ 0.5–1.5, so θ ≈ 1 gives steps of order 1/L). α is found on a 60-point geometric grid in
[1e-3, 2]/L, refined by golden section; R1 uses the same.

## Stochastic baselines (the earlier E2 polishers), same compute

Compute is counted in **epochs of row gradients**: a full-batch step costs 1 epoch; a minibatch step of 16 costs 16/n epochs. Budgets
B ∈ {1, 5, 20} epochs: stochastic polishers get `K₂ = round(B n/16)` steps, full-batch methods T = B steps.
Polishers: none; SGD decay (0.05/L); SGD average (1/L); the **adapter** (step and momentum from the exact EV@R model, 6 pilots).

## Setting

Stage 1: 2,000 textbook SGD steps (α = 1/L), batch 16; L2 logistic regression λ = 0.01; German, COMPAS, Pima, Heart; 200 seeds.
Held-out rows as in the credit report.

## Metrics

* Tail of the objective after polishing: CVaR₅% and mean of `f(w) − f*`; decisions changed vs the exact model; retrain disagreement.
* **Per-sample view (what the proposal protects)**: for the training rows, the share whose loss increased relative to the stage-1 weights,
  and the mean over seeds of the 95th percentile of the per-sample loss change.

## Hypotheses

* **H1** At equal compute, GD-LS and R1/R3 beat the stochastic adapter on CVaR₅% at B = 5 and 20 on all four datasets (full gradients
  have no noise; for these small problems a few epochs of exact gradient steps reach the optimum's neighbourhood).
* **H2** R1, R3 and R4 have a smaller share of training rows whose loss increases than GD-LS (that is what the risk term buys), at similar CVaR₅%.
* **H3** The risk-averse step is smaller than the exact line-search step (α*_R < α*_LS) and the difference grows with θ.
* **H4** R4 (tilted) reduces the loss of the worst training rows (95th percentile of per-row loss) more than R1/R3 do.
* **H5 (open)** At B = 1 epoch the adapter or decay beats full-batch steps (a single full step cannot undo the stage-1 noise).

If H1 holds, the full-batch baseline must be included in every adapter comparison and the adapter's role narrows to problems where a
full pass over the data is too expensive; reported whichever way it falls.
