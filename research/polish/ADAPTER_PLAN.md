# RA-GMM as an adapter: "train as usual, then 1-2 epochs of RA-GMM to improve the tail"

Written before the epoch-based experiment (E3) is run. Builds on the fixed-learning-rate and polish experiments in
`PLAN.md` (E1 and E2), which are still running; their results decide the details marked **[decided by E1/E2]**.

## 1. Why this framing

* **It removes the "new trainer" argument.** Nobody has to replace SGD, Adam or their ADMM pipeline. The adapter
  takes whatever weights the existing trainer produced and spends a short, fixed extra budget (1-2 epochs) on a
  step chosen to minimize the tail of the final error.
* **It matches the evidence.** (a) The credit report showed RA-GMM's gain comes from taking a small, well-chosen step.
  (b) E1's early data (heart, 40 seeds) shows that at a fixed step, adding momentum makes the tail worse. (c) E2's early
  data shows that a short low-noise phase after a noisy phase cuts the tail several-fold. So the honest product is
  a **tail-cleaning finishing phase**, not a faster trainer.
* **It is a small, checkable promise:** "after your usual training, 1-2 extra epochs reduce the number of decisions
  that change between retrains by X%, at no accuracy cost, and here is the tail report that proves it on your data".

## 2. Definition of the adapter

`polish(weights, data, epochs=1)`:
1. Take the weights `w₁` from any trainer (SGD, SGD+momentum, Adam, ADMM, L-BFGS stopped early, a warm-started model).
2. Estimate the local geometry from data and `w₁` only (no exact solve): Hessian `H` and per-row gradient covariance `Σ`
   at `w₁` (for logistic regression exactly; for a general model, the last layer on frozen features, which is
   again a strongly convex logistic problem; see section 6).
3. Choose `(α, β, γ)` for the polishing budget `K₂ = epochs · n / batch` steps by minimizing the **exact EV@R** of
   the final objective under the quadratic model (the paper's Section 3 quantity, finite-horizon extension in
   `risk_averse_tuning.quadratic`), starting error `e₀ = −H⁻¹∇f(w₁)` (one Newton step), averaged over a few
   pilot seeds if several runs exist, otherwise on the single run.
4. Run `K₂` stochastic GMM steps with those parameters; return the final iterate.

The paper's problem (51) is used as a **named alternative** ("RA-GMM (51)") in every comparison so that the claim about
the paper's own recipe stays visible.

## 3. Experiment E3 (epoch-based; pre-registered here)

Datasets: all six (German, COMPAS, Pima, Heart; Adult and Bank if runtime allows, reported separately). 300 seeds.
Split and metrics as the credit report (held-out rows, threshold 0.5).

**Stage 1 ("your usual trainer", 5 epochs each, batch 16):** SGD with α = 1/L; SGD + momentum 0.9 with α = 0.1/L;
Adam (lr 0.01); consensus ADMM (4 blocks, 5 iterations; small datasets). These are deliberately ordinary, not tuned.

**Stage 2 (polish budget: 1 epoch and 2 epochs):**

| Polisher | Meaning |
|---|---|
| none | stage 1 output |
| continue | keep training with the stage-1 trainer's settings for the same extra epochs (fair control: same compute) |
| decay | SGD at a 20× smaller step |
| average | SGD at the stage-1 step, average of the polishing iterates |
| RA-GMM (51) | the paper's problem (51) parameters |
| **Exact-EV@R polish** (the adapter) | step and momentum from section 2 |
| oracle (reference only) | the carefully tuned SGD of the report, trained from scratch with 240 tuning runs |

**Metrics:** tail of the objective (CVaR₅% of `f(w) − f*` and EV@R); decisions changed vs the exact model (mean, worst 5%);
coin-flip rows; accuracy; wrongly denied / wrongly approved rates (report section 7); extra compute in epochs.
Paired bootstrap over seeds.

**Hypotheses.**
* **H1** For every stage-1 trainer and 1 epoch of polishing, the adapter cuts CVaR₅% of the objective by ≥ 50% relative to
  "none" on ≥ 4 of 6 datasets.
* **H2** The adapter cuts mean changed decisions by ≥ 30% relative to "none" on ≥ 4 of 6, with no loss in mean accuracy
  larger than 0.5 points.
* **H3** The adapter beats "continue" (same compute) in CVaR₅% on every dataset and trainer.
* **H4 (the honest comparison)** The adapter is within 20% of the best of {decay, average} in CVaR₅%, or better, on ≥ 4 of 6.
  If not, the product claim reduces to "any careful annealing helps; the tool chooses it automatically".
* **H5 (open)** 2 epochs vs 1 epoch: further gain smaller than the gain of the first.
* **H6 (open)** The paper's (51) as a polisher is worse than the adapter and ≈ "decay".

## 4. What could fail, and the pivot

| Failure | Signal | Pivot |
|---|---|---|
| "Any annealing helps; the adapter adds nothing over averaging or decay" (H4 fails) | adapter 20%+ worse than the best simple finisher | Ship **polish** with the simplest winner (averaging or decay) and keep the EV@R machinery only for the *report* and for the choice of polishing length; say so plainly. |
| Polishing helps SGD/ADMM but not Adam (adaptive step) | no gain after Adam | State the scope (non-adaptive trainers); test Adam with its state reset and a plain-SGD finish. |
| The gain disappears when stage 1 already ran long | gain vs stage-1 epochs shrinks | Report the dose-response (stage-1 length 1, 5, 20 epochs): the adapter's value is largest for short/cheap training and large-step trainers. |
| Selection needs `H`, `Σ` that are unavailable for big models | cost / memory | Last-layer polishing on frozen features (section 6) or a diagonal approximation; test the approximation against the full one on the datasets here. |
| Accuracy cost | accuracy falls > 0.5 points | The polish length becomes a user choice with the accuracy/tail trade-off shown in the report. |

## 5. Package changes (only if H1-H3 hold)

* `risk_averse_tuning.polish(X, y, w, epochs=1, lam=..., batch_size=..., method="ev@r"|"average"|"decay"|"ragmm")`
  returning new weights and the chosen parameters; `RiskAverseClassifier.fit(..., init=w, polish_epochs=1)`.
* CLI: `ragmm data.csv --target y --init-model model.json --polish-epochs 2 --report tail.md` ("polish an existing model").
* `tail_report` gains a "before / after polish" table.
* Tests: polish never increases the objective's mean beyond a tolerance on a toy problem; the selected parameters equal
  those from `quadratic.evar_quadratic`; CLI round trip.

## 6. Beyond logistic regression (scope, not claim)

The theory needs a strongly convex smooth objective. For a neural network, freeze all but the final layer, treat the
penultimate features as data, and polish the final linear/logistic layer with the same adapter (this is the "freeze the
weights and train a strongly convex last layer" idea from the project plan). That makes the adapter applicable to any
deployed model whose last layer is linear. **Not tested yet**; the plan is to test it on one small network after E3.

## 7. How this changes the story (credit report, section 5-6)

The report's paper section becomes: "RA-GMM is not a replacement trainer. It is the finishing phase: after normal
training, a short, risk-aware phase reduces the part of the error that depends on the luck of the run." The headline
comparisons move from "RA-GMM vs textbook SGD" to "your trainer vs your trainer + 1 epoch of polish", at equal-ish compute.

## 8. Addendum: how much does the Gaussian-noise assumption matter? (written before running `noise_check.py`)

Question: in the linearized polish model `z_{k+1} = A z_k + B w_k`, replace the Gaussian noise `w_k` by the **real**
minibatch noise (mean of 16 rows drawn with replacement from the centered per-row gradients at the exact solution) and
compare the empirical EV@R (20,000 simulated paths, zeta = 0.05) of the final loss `½ zᵀ H z` with the exact Gaussian
EV@R computed from the same covariance (`evar_quadratic`). Both are linear systems, so the ratio isolates the effect of
the noise *distribution*. Start at the optimum (`e₀ = 0`) so the final loss is pure noise (the case where the
distribution matters most). Datasets: COMPAS (d = 8), Heart (d = 27), German (d = 55); α ∈ {1, 0.3, 0.1, 0.03}/L,
(β, γ) ∈ {(0, 0), (0.9, 0.9)}, K ∈ {200, 1000}.

Expectation: ratio (real / Gaussian) near 1 when the averaging window `1/(αμ)` is long compared with 1 step (small α),
and above 1 (Gaussian too optimistic about the tail) when α is large, because few independent minibatches contribute.
If the ratio stays within ±15% for the small steps that the polisher selects, the Gaussian model is adequate in
the regime where it is used and only needs a diagnostic; otherwise a cumulant correction or the bootstrap-simulation
variant is needed.
