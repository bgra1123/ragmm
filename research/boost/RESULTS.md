# Function-space polish for XGBoost: results

Design: `PLAN.md` (pre-registered; no deviations). Code: `boost_polish.py` (training), `analyze.py` (numbers); raw margins `results/margins_split*.npz`, summary `results/analysis.json`.
German Credit, 3 splits (900 held-out applicants pooled), 100 seeds per variant and split. Decisions at 0.5 for stability; cost at the 5/6 rule.

| Variant | Two retrains disagree | Coin-flip applicants | Decisions changed (avg / worst 5% of retrains) | AUC | Held-out log-loss | Good wrongly denied | Defaulters approved | Cost per applicant (5/6 rule) |
|---|---|---|---|---|---|---|---|---|
| Baseline: 300 noisy rounds | 5.6% | 11.8% | 35.0 / 46.0 | 0.801 | 0.531 | 32.4% | 22.3% | 0.562 |
| 200 noisy + 100 deterministic rounds (budget matched) | 5.7% | 12.9% | 36.0 / 45.4 | 0.802 | 0.525 | 33.7% | 21.5% | 0.558 |
| 300 noisy + 100 deterministic rounds | 5.3% | 12.1% | 32.9 / 46.6 | 0.799 | 0.554 | 30.8% | 23.7% | 0.571 |
| 200 noisy + 100 deterministic, line-search steps | 10.3% | 23.1% | 65.2 / 78.4 | 0.771 | 0.913 | 23.6% | 35.6% | 0.699 |
| 200 noisy + 100 deterministic, risk-averse steps (θ = 5) | 7.7% | 17.2% | 48.3 / 60.0 | 0.779 | 0.789 | 24.5% | 32.9% | 0.665 |
| Reference: average of 10 baseline seeds | **1.6%** | **3.3%** | **9.7 / 15.3** | 0.804 | 0.519 | 32.9% | 21.7% | 0.555 |

## Hypotheses

* **H1 not supported:** the budget-matched deterministic continuation leaves disagreement unchanged (5.6% → 5.7%).
* **H2 not supported:** with 100 extra deterministic rounds the cut is 5% (5.6% → 5.3%).
* **H3 not supported (opposite):** steps chosen by the entropic-risk rule (7.7%) or by line search (10.3%) are *less* stable than fixed η = 0.1 (5.7%).
* **H4 mostly supported for the fixed-step polish** (AUC +0.001 / −0.002; log-loss −0.006 / +0.023, the latter marginally above the 0.02 limit), **not** for the adaptive steps (AUC −0.03 / −0.02, log-loss +0.26 to +0.38).
* **H5 supported:** averaging 10 seeds (1.6%) beats every single-model polish (5.3–10.3%).

## Reading

1. **Continuing the boosting gradient on the full data does not remove the seed variation.** The randomness sits in the structure of the early, subsampled trees; later deterministic trees fit residuals but leave the decisions essentially where
   the noisy ensemble put them (5.6% → 5.3–5.7%). For logistic regression the analogous deterministic continuation converges to the one optimum and the starting noise disappears; boosting has no unique optimum to converge to.
2. **The risk-averse and line-search steps over-fit.** Each new tree is a full Newton step; a step chosen by the training-sample entropic risk (or the mean loss) is larger than the η = 0.1 shrinkage, lowers the training loss, and raises the held-out log-loss (0.53 → 0.79–0.91)
   and instability. A criterion measured on the training rows rewards fitting them; with a flexible base learner it needs a cap (for example α ≤ η) or an out-of-sample estimate. I did not test a capped version.
3. **What works in function space is variance reduction**, not a gradient step: averaging seeds (1.6%), a smaller learning rate with more trees (earlier result: −45% to −57%), or removing subsampling (0%).
4. This supports your point in one way and not in another: the right place for an adapter on XGBoost is function space, and a gradient-style polish there was the correct thing to try. It does not work because the instability is not an optimization error that more gradient steps repair.

## Caveats

* One dataset, one XGBoost configuration, 100 extra rounds only; logistic loss; the adaptive steps use a 40-point grid on [0.02, 2] on top of full Newton trees.
* Disagreement for the averaged reference uses overlapping random 10-subsets and is a lower bound for independent ensembles.
