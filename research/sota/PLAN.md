# Does the post-training step help XGBoost and neural networks on credit outcomes? (written before running)

Question (user): how does this improve XGBoost and other state-of-the-art methods used in applications?

What is already known: for XGBoost (research/trees) a refit of the last layer on leaf indicators did not reduce seed disagreement; averaging seeds did.
Not yet measured: credit **outcomes** (wrongly denied / approved, cost) for XGBoost, and anything for neural networks, the other large SOTA family.

## Models (German Credit: 700 train / 300 held-out; COMPAS 4,320 / 1,852 for disagreement only), 200 seeds each

| Name | Definition |
|---|---|
| XGBoost default | 300 trees, depth 4, lr 0.1, subsample 0.8, colsample 0.8 (seed = subsampling) |
| XGBoost seed-average m = 5, 20 | mean predicted probability of m seeds |
| XGBoost no subsampling | deterministic |
| XGBoost + exact head refit | leaf indicators → L2 logistic (λ = 0.01), solved to gradient norm < 1e-7 |
| MLP default | scikit-learn `MLPClassifier`, 2 hidden layers (64, 64), ReLU, Adam lr 1e-3, batch 32, 60 epochs, L2 alpha 1e-3; seed = init and shuffling |
| MLP seed-average m = 5, 20 | mean predicted probability |
| **MLP + exact head refit (the adapter for networks)** | freeze the hidden layers, take penultimate activations, solve the L2 logistic head (λ = 0.01) exactly (L-BFGS), deterministic |
| MLP + exact head refit + seed-average m = 5, 20 | combination |
| Logistic regression, exact | reference (deterministic) |
| Logistic careful SGD | reference from the credit report |

## Metrics (held-out rows, thresholds 0.5 and the cost-based 5/6)

Retrain disagreement (probability that two retrains differ on a random row); coin-flip rows (decision differs from the majority in ≥ 10% of retrains); mean AUC;
good applicants wrongly denied, defaulters wrongly approved, default rate among approved, expected cost per applicant (5 per default approved, 1 per good denied),
decisions changed vs the **consensus** decision of all 200 seeds (there is no exact optimum for these models). Averages over models, with the
**spread across models** of the cost (5th–95th percentile) as the risk of a single retrain.

## Hypotheses

* **H1** MLP default has retrain disagreement above careful logistic regression's on both datasets.
* **H2** The exact head refit cuts MLP disagreement by ≥ 30% (it removes head-optimization noise, not body noise).
* **H3** Seed-averaging m = 20 cuts disagreement by ≥ 60% for both XGBoost and the MLP.
* **H4** Refit + seed-average m = 20 is the lowest-disagreement MLP configuration.
* **H5** Averaging seeds narrows the 5th–95th percentile range of cost per applicant across retrains (the single-retrain risk) by ≥ 50% for both families.
* **H6 (open)** Mean credit cost (5/6 rule) of the averaged models is at or below the single-model mean; AUC of the averaged models is higher than the single-model mean.

Reported whichever way it falls.
