# A function-space ("gradient boosting") version of the adapter for XGBoost (written before running)

Motivation (user): XGBoost is gradient descent in function space, so an adapter for it should work on the **boosting gradient**, not on a last-layer refit.

## Why the earlier adapter failed for trees, and what the function-space analogue is

* The earlier "head refit" re-solves only the leaf values (a convex problem). XGBoost's seed-to-seed variation comes from the **tree structures** (row/column subsampling), which a leaf refit cannot change.
* In function space, the analogue of "train noisily, then polish with exact full-batch gradient steps" (which removed the seed variation for logistic regression) is: **train as usual with subsampling, then continue boosting deterministically**
  (no row or column subsampling, full data) from the trained model's margins. Later trees fit the residuals of the noisy ensemble and can compensate its random structure.
* The user's risk-averse step from the previous discussion also has a boosting form: after each new tree `h_t` (fit to the gradient), choose its step `α_t` by minimizing the entropic risk of the per-sample loss change,
  `α_t = argmin (1/θ) log mean_i exp(θ [ℓ(y_i, F_i + α h_i) − ℓ(y_i, F_i)])` (exact change; θ → 0 is an exact line search on the mean loss).

## Variants (German Credit; 3 splits, seeds 0–2; 100 seeds each; decisions at 0.5 and the cost rule 5/6)

Baseline: XGBoost default, 300 rounds, depth 4, η = 0.1, subsample 0.8, colsample 0.8, λ = 1 (as `research/sota2`).

| Name | Stage 1 | Stage 2 |
|---|---|---|
| **Functional polish (budget matched)** | 200 noisy rounds (as baseline) | 100 deterministic rounds (subsample 1, colsample 1, η = 0.1) from the stage-1 margins |
| Functional polish (extra rounds) | 300 noisy rounds | 100 deterministic rounds, η = 0.1 |
| Polish with line-search steps | 200 noisy | 100 deterministic rounds, each tree added with the exact line-search step on the mean loss (η = 1 tree, α_t ∈ (0, 2]) |
| **Polish with risk-averse steps** | 200 noisy | 100 deterministic rounds, step by the entropic-risk rule, θ = 5 |
| Averaging 10 seeds (reference) | – | mean probability of 200 random 10-subsets of the 100 baselines |

## Metrics (held-out rows, pooled over splits)

Retrain disagreement; coin-flip share; mean AUC and mean held-out log-loss (to detect over-fitting by extra rounds); credit outcomes at the 5/6 rule (good wrongly denied, defaulters approved, default rate among approved, cost per applicant);
decisions changed vs consensus (mean, worst 5% of retrains).

## Hypotheses

* **H1** The budget-matched functional polish cuts disagreement by ≥ 30% relative to the baseline.
* **H2** The extra-rounds polish also cuts it by ≥ 30%.
* **H3** The risk-averse step gives lower disagreement than fixed η = 0.1 polishing (open).
* **H4** AUC changes by less than 0.01 and held-out log-loss does not worsen by more than 0.02 (no over-fitting damage).
* **H5** Averaging 10 seeds still beats every single-model polish in disagreement.

Reported whichever way it falls. If H1/H2 fail, the answer is that structure noise cannot be repaired by continued boosting, and averaging remains the lever.
