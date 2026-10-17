# XGBoost and neural networks: seed instability and credit outcomes (German Credit)

Design: `PLAN.md` (pre-registered). Code: `sota_experiment.py`; raw: `results/german.json` (COMPAS: see the end). 700 training and 300 held-out applicants
(210 repaid, 90 defaulted); 200 seeds per model; scikit-learn `MLPClassifier` (64, 64), Adam, 60 epochs; XGBoost 300 trees, depth 4, subsample 0.8, colsample 0.8.
"Exact head refit" freezes the model and re-solves its last logistic layer exactly (L2, λ = 0.01); for XGBoost the "last layer" is the leaf indicators.

## Stability (German)

| Model | Retrain disagreement | Coin-flip applicants (of 300) | Mean AUC | Decisions changed vs consensus |
|---|---|---|---|---|
| Logistic regression, exact (reference) | 0 | 0 | 0.762 | – |
| Logistic careful SGD (reference) | 2.5% | 18 | 0.761 | 5.7 |
| XGBoost default | 5.5% | 36 | 0.759 | 11.3 |
| XGBoost, average of 5 seeds | 2.2% | 15 | 0.763 | 4.5 |
| XGBoost, average of 20 seeds | **0.8%** | **5** | 0.763 | 1.4 |
| XGBoost, no subsampling | 0 | 0 | 0.755 | 18.0 |
| XGBoost + exact head refit | 6.2% | 48 | 0.742 | 15.9 |
| MLP default | **15.1%** | **102** | 0.715 | 32.2 |
| MLP, average of 5 seeds | 7.4% | 49 | 0.727 | 15.8 |
| MLP, average of 20 seeds | 4.0% | 29 | 0.733 | 8.6 |
| MLP + exact head refit | 15.0% | 103 | 0.715 | 32.3 |
| MLP + head refit + average of 20 | 4.0% | 28 | 0.734 | 9.0 |

## Credit outcomes (German; averages over models; "cost" = 5 per defaulter approved + 1 per good applicant denied, per applicant; bracket = 5th–95th percentile across retrains)

**Approve at 50%**

| Model | Good wrongly denied | Defaulters approved | Default rate among approved | Cost | Spread of cost across retrains |
|---|---|---|---|---|---|
| Logistic exact | 12.9% | 58.9% | 22.5% | 0.973 | 0 |
| XGBoost default | 12.2% | 58.2% | 22.1% | 0.958 [0.900, 1.010] | 0.110 |
| XGBoost avg 20 | 11.5% | 59.3% | 22.3% | 0.971 [0.952, 0.982] | **0.029** |
| MLP default | 17.7% | 57.8% | 23.1% | 0.991 [0.903, 1.080] | 0.177 |
| MLP avg 20 | 16.1% | 56.9% | 22.5% | 0.966 [0.938, 1.004] | **0.066** |
| MLP + head refit + avg 20 | 15.6% | 56.6% | 22.3% | 0.957 [0.934, 0.997] | 0.062 |

**Approve only at 83% (cost-based rule)**

| Model | Good wrongly denied | Defaulters approved | Default rate among approved | Cost |
|---|---|---|---|---|
| Logistic exact | 48.1% | 14.4% | 10.7% | **0.553** |
| XGBoost default | 31.9% | 32.7% | 17.1% | 0.714 [0.663, 0.763] |
| XGBoost avg 20 | 31.6% | 33.7% | 17.4% | 0.726 [0.695, 0.749] |
| MLP default | 24.7% | 47.9% | 21.4% | 0.891 [0.803, 0.977] |
| MLP avg 20 | 31.5% | 37.7% | 19.1% | 0.786 [0.758, 0.819] |
| MLP + head refit + avg 20 | 32.9% | 35.1% | 18.3% | 0.757 [0.733, 0.798] |

## Hypotheses

* **H1 supported (German):** MLP default disagrees on 15.1% of applicants against 2.5% for careful logistic SGD.
* **H2 not supported:** the exact head refit leaves the MLP's disagreement at 15.0% (from 15.1%) and its coin-flip applicants at 103 (from 102). The head was already converged; the variation is in the
  hidden layers, which differ per seed. The same is true for XGBoost's leaf refit (5.5% → 6.2%).
* **H3 supported:** averaging 20 seeds cuts disagreement 85% for XGBoost (5.5% → 0.8%) and 74% for the MLP (15.1% → 4.0%).
* **H4 tie:** refit + average of 20 equals the average alone (4.0% and 4.0%; AUC 0.734 vs 0.733).
* **H5 supported:** averaging 20 seeds narrows the spread of cost across retrains by 74% (XGBoost, 0.110 → 0.029) and 63% (MLP, 0.177 → 0.066) at the 50% rule.
* **H6 mixed:** averaged models have higher AUC (XGBoost 0.759 → 0.763; MLP 0.715 → 0.733). Mean cost at the 50% rule fell for the MLP (0.991 → 0.966) but rose slightly for XGBoost
  (0.958 → 0.971, within the spread). At the cost-based rule the MLP's cost fell from 0.891 to 0.786 with averaging and 0.757 with refit + averaging.

## What this means for applying it to XGBoost and neural networks

1. **The polish idea (exact head solve) does not transfer to these models.** It leaves the variation unchanged, because their instability is in the learned structure (trees, hidden layers), not in the head.
   For logistic models the same step removed it.
2. **What does reduce it is the old, simple remedy: average several seeds.** 20 seeds take XGBoost to 0.8% disagreement (better than careful logistic regression) and cut the MLP's by 74%;
   the single-retrain risk (spread of cost) falls by 63–74%, and accuracy improves.
3. **The risk-averse selection and the tail report still apply, as measurement:** they say how many applicants are coin flips (36 for XGBoost, 102 for the MLP, 18 for careful logistic) and how many seeds
   are enough. This is a tool for choosing the ensemble size and flagging the unstable applicants, not an optimizer.
4. **On this small credit dataset, a regularized logistic regression is better than both under the cost-based rule** (cost 0.553 vs 0.71–0.89): XGBoost and the MLP are overconfident, so the 83% cutoff,
   which presumes calibrated probabilities, does not do what it should. Calibrating (e.g. a one-dimensional logistic fit of the scores) is a prerequisite for using them with a cost-based rule; I did not test it here.

## Caveats

* German Credit is small (700 training rows), where SOTA models have little advantage; results may differ at scale. One MLP architecture and one XGBoost configuration, untuned.
* Disagreement for averaged models is estimated from fewer units (40 for m = 5, 10 for m = 20).
* "Consensus" is the decision of the mean probability over all 200 seeds; there is no exact optimum for these models.
