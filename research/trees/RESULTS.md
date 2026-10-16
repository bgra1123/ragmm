# XGBoost seed instability and which levers reduce it: results

Design: `PLAN.md` (pre-registered; one deviation recorded before any result was seen: the last-layer refit uses L-BFGS instead of
Newton's method). xgboost 3.4.1, 200 seeds per configuration (the seed controls row and column subsampling), German, COMPAS, Pima, Heart,
held-out rows, threshold 0.5. Raw: `results/<dataset>.json`.

## Retrain disagreement (probability that two retrains differ on a random held-out row)

| Configuration | German | COMPAS | Pima | Heart |
|---|---|---|---|---|
| XGBoost default (subsample 0.8, colsample 0.8) | 5.5% | 2.3% | 4.0% | 1.6% |
| XGBoost, no subsampling | **0** | **0** | **0** | **0** |
| Slower: lr 0.03, 1,000 trees | 3.0% | 1.3% | 1.8% | 0.7% |
| Average of 5 seeds | 2.2% | 1.1% | 1.4% | 0.6% |
| Average of 20 seeds | 0.8% | 0.8% | 0.9% | 0.6% |
| Last-layer refit on leaf indicators | 6.2% | 2.3% | 4.6% | 1.4% |
| Leaf refit + average of 20 | 0.6% | 0.6% | 1.7% | 0.0% |
| Logistic SGD lr 0.01 (reference) | 2.9% | 1.9% | 0.9% | 0.9% |
| Logistic careful SGD (reference) | 2.5% | 1.5% | 0.6% | 1.0% |

Mean held-out AUC (default XGBoost vs exact logistic regression): German 0.759 vs 0.762, COMPAS 0.711 vs 0.727, Pima 0.771 vs 0.795,
Heart 0.888 vs 0.878. Coin-flip rows (decision differs from the majority in ≥ 10% of retrains), default → average of 20: German 36 → 5 of 300,
COMPAS 88 → 33 of 1,852, Pima 22 → 5 of 230, Heart 3 → 1 of 91.

## Hypotheses

* **H1 supported (3 of 4):** default disagreement is ≥ 2% on German, COMPAS and Pima (Heart 1.6%), and above careful logistic SGD
  on all four (by 1.5–7× except Heart, 1.6×).
* **H2 not supported as stated:** averaging 20 seeds cuts disagreement by 86% (German), 78% (Pima), 65% (COMPAS), 62% (Heart), so
  ≥ 70% on 2 of 4. The averaged models are 10 units each, so their disagreement estimates are noisier; the floor of about 0.6–0.9%
  is partly that.
* **H3 supported (4 of 4):** the slow configuration cuts 45–57%, less than the 5-seed average (53–64%), and costs 3.3× the trees.
* **H4 supported (4 of 4):** the last-layer refit changes disagreement by −11% to +10% (it raises it on German and Pima): tree structure
  differs across seeds, and refitting the head cannot undo that.
* **H5 confirmed:** with no subsampling XGBoost is exactly deterministic on all four datasets (identical predictions).
* **H6 not supported:** on these small datasets default XGBoost's AUC is not higher than logistic regression's except on Heart; the
  held-out log-loss tail is not heavier than the mean (CVaR₅%/mean 1.0–1.05).

## What this means for the credit product

1. **XGBoost with subsampling is as seed-dependent as untuned SGD and more than careful logistic regression** (disagreement 1.6–5.5%).
   Turning subsampling off removes it entirely, at the cost of the regularization that subsampling provides (AUC −0.4 to +0.6 points here).
2. **The effective levers are seed-averaging and a smaller learning rate with more trees**, not a gradient-based head polish.
   The adapter's head polish does not apply to tree models; the report should say so.
3. A cheap rule for tree users: *average 5–20 seeds*, or turn subsampling off, and run the tail report before and after.

## Caveats

* Four small datasets and one XGBoost configuration; no tuning of depth or regularization.
* The average-of-m numbers use 40 (m = 5) or 10 (m = 20) averaged models, so their disagreement is estimated from fewer units.
* "Last-layer refit" fits an L2 logistic regression with λ = 0.01 on leaf indicators; other heads or penalties were not tried.
