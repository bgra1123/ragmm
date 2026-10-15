# Does the instability story hold for XGBoost, and which lever reduces it? (written before running)

Question (user): how can risk-aversion be integrated into XGBoost and other state-of-the-art classifiers for credit
scoring? The theory behind RA-GMM needs a strongly convex smooth objective, which a tree ensemble does not have as a
whole. So before building anything, measure (a) whether XGBoost's decisions depend on its random seed and (b) which
of the available levers reduces that, with the closest tree-world analogues of what worked for logistic regression
(smaller step ≈ smaller learning rate; averaging ≈ averaging over seeds; polish ≈ refit a strongly convex last layer).

Data and split as `research/gate0` (German, COMPAS, Pima, Heart; held-out rows; threshold 0.5). **200 seeds** (0–199)
per configuration; the seed controls row and column subsampling in XGBoost (`tree_method="hist"`, one thread). All other
settings fixed. xgboost version recorded in the results.

## Configurations

| Name | Settings |
|---|---|
| default | 300 trees, max_depth 4, learning_rate 0.1, subsample 0.8, colsample_bytree 0.8, reg_lambda 1 |
| no subsampling | same with subsample = colsample_bytree = 1 (check: deterministic) |
| slow | learning_rate 0.03, 1,000 trees (3.3× more steps at a smaller step) |
| seed-average m = 5 / 20 | average predicted probability of m default models with different seeds (disjoint groups of m of the 200 seeds) |
| last-layer refit | take each default model's leaves (`pred_leaf`), one-hot encode them, fit an L2 logistic regression (λ = 0.01, exact Newton solve) on the leaf indicators of the training rows (a strongly convex problem; the polish target in the adapter idea) |
| leaf refit + seed-average 20 | the two combined |

Single-model configurations have 200 retrains; seed-average m = 5 has 40 averaged models, m = 20 has 10 (so its
disagreement estimates use fewer units; stated in the results).

## Metrics (held-out rows)

* **Retrain disagreement**: probability that two retrains give a random row different decisions (pure seed noise).
* **Coin-flip rows** (decision differs from the majority in ≥ 10% of retrains) and rows that ever change.
* **Held-out log loss**: mean, 95th percentile, CVaR₅% across retrains (the tail of the model quality); AUC mean and worst 5%.
* **Decisions changed vs the consensus**: the decision of the mean probability over all 200 seeds of the default model
  (there is no exact optimum for subsampled trees; the consensus is the closest reference).
* Compared with the logistic baselines from Gate 0: SGD at lr 0.01 and tuned SGD with tail averaging (retrain disagreement).

## Hypotheses

* **H1** Default XGBoost has retrain disagreement ≥ 2% on ≥ 3 of 4 datasets (comparable to or above logistic regression with careful SGD).
* **H2** Seed-averaging m = 20 cuts disagreement by ≥ 70% on every dataset.
* **H3** The slow configuration cuts disagreement by less than seed-averaging m = 5 does.
* **H4** The last-layer refit alone cuts disagreement by < 30% (tree structures differ by seed; the last layer cannot fix that).
* **H5 (check)** With no subsampling, disagreement is exactly 0.
* **H6 (open)** XGBoost's held-out AUC is higher than logistic regression's (it is state of the art), and the held-out
  log-loss tail is heavier relative to its mean.

Reported whichever way it falls. If H1 fails, the answer to the user is that instability is not the problem for trees at
this scale; if it holds and H2 holds, the integration is "seed-averaging plus the tail report", not a gradient-based step.

> **Deviation recorded before any result was seen:** the first launch was stopped because the last-layer refit used
> Newton's method on ~4,000 leaf indicators per model (hours at 200 seeds). The refit now uses L-BFGS (gradient norm
> < 1e-7) on the same strongly convex objective; nothing else changed.
