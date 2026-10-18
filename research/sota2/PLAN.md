# XGBoost and other state-of-the-art models as baselines, with and without post-training adapters (written before running)

Question (user): include XGBoost and other SOTA methods in the baseline analysis and test how the outputs change when an adapter is added.

What "adapter" means for models that are not logistic regression (all applied **after training, to the frozen model**):

| Adapter | Definition | Applies to |
|---|---|---|
| **Head refit** | take the model's last-layer features (tree leaf indicators, or the MLP's penultimate activations) and re-solve the logistic head **exactly** (L2, λ = 0.01, deterministic): the limit of the polish step | XGBoost, LightGBM, random forest, MLP (not HistGradientBoosting: no leaf API) |
| **Seed averaging (m = 10)** | average the predicted probabilities of 10 retrains (disjoint groups of the 100 seeds) | every randomized model |
| **Calibration** | one-dimensional logistic fit of the model's logit on out-of-fold (3-fold) predictions of the training rows, applied to the model's score | all |
| Combinations | refit + averaging; calibration + averaging | |
For logistic regression the adapter is the polish step as in the earlier studies (stochastic adapter and deterministic line search, 5 epochs), as the reference.

## Models (German Credit; 3 stratified splits, seeds 0–2; 700 train / 300 held-out each; 100 seeds per model and split)

XGBoost (300 trees, depth 4, lr 0.1, subsample 0.8, colsample 0.8) · LightGBM (300 trees, 15 leaves, lr 0.1, bagging 0.8 every iteration, feature fraction 0.8) · random forest (100 trees, max depth 8, min leaf 5, bootstrap) ·
HistGradientBoosting (300 iterations, depth 4, lr 0.1; no randomness expected) · MLP (64, 64; Adam 1e-3, 60 epochs, batch 32, L2 1e-3) · logistic regression (textbook SGD, 2,000 steps, with the polish as adapter; exact solution as reference).

## Outputs (per model × adapter; pooled over splits)

* **Stability**: retrain disagreement, coin-flip share (decision differs from the majority in ≥ 10% of retrains), decisions changed vs the consensus decision of the same configuration.
* **Worst case**: the **worst 5% of retrains** by (i) expected cost (CVaR₅% of cost per applicant across retrains), (ii) decisions changed vs consensus (mean of the worst 5%).
* **Credit outcomes** (held-out): good applicants wrongly denied, defaulters wrongly approved, default rate among approved, expected cost (5 per default approved, 1 per good denied), at the 0.5 and 5/6 rules; AUC.
* **Groups**: the ten fixed groups of `research/groups/PLAN.md`: flip rate vs consensus and wrongly-denied rate among good members, for the baseline and for each adapter.

## Hypotheses

* **H1** Baseline stability ordering: MLP and random forest/bagged models less stable than logistic regression with careful training; HistGradientBoosting has exactly zero seed disagreement (deterministic).
* **H2** Head refit cuts disagreement by ≥ 30% for the MLP and by < 30% for the tree models.
* **H3** Averaging 10 seeds cuts disagreement by ≥ 50% for every randomized model.
* **H4** For each randomized model, the flip rate of the six "unusual profile" groups (G1, G2, G3, G7, G8, G9) is ≥ 1.3× that of applicants outside all groups, in ≥ 4 of 6 groups.
* **H5** Calibration lowers the 5/6-rule cost of XGBoost, LightGBM, random forest and the MLP, and does not change their disagreement by more than 20% (relative).
* **H6** Averaging 10 seeds lowers the worst-5% cost (CVaR₅% across retrains) by ≥ 40% for every randomized model.
* **H7 (open)** After the best adapter combination, the SOTA models match or beat exactly-trained logistic regression on cost at the 5/6 rule on this dataset.

Reported whichever way it falls; the previous XGBoost/MLP results (`research/sota`) used 200 seeds and no calibration or other models, so numbers will differ in detail.
