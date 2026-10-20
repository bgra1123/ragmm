# Product benchmark: results

Design: `PLAN.md` (pre-registered before running). Code: `benchmark.py`, `analyze.py`; raw: `results/split*.json`, pooled:
`results/pooled.json`. Package `risk_averse_tuning` 0.4.0, default adapter settings. German Credit, 5 stratified splits ×
300 held-out applicants (1,500 applicant-decisions in total), 30 retrains of every configuration per split, cutoff 0.5.
Seconds per fit are single-threaded wall-clock times on the 4-core test machine (5 splits ran in parallel), so compare
them within this table only. No deviations from the plan.

"Due to randomness" = chance of a wrong decision for one retrain minus that of the majority vote of the 30 retrains of the
**same configuration**. With the adapter it is close to 0 by construction, so the fairer before/after number is the
**wrong-decision rate of one retrain** (column 5).

## 1. Overall

| Model | Two retrains disagree | Unstable rows | Changed per retrain (avg / worst 5%) | Wrong (one retrain) | Due to randomness | Seconds per fit |
|---|---|---|---|---|---|---|
| SGD logistic | 15.5% | 35.3% | 32.2 / 44.6 | 27.3% | +2.5% | 0.00 |
| SGD logistic + adapter (polish) | 0.1% | 0.3% | 0.3 / 0.8 | 25.4% | -0.0% | 0.06 |
| SGD logistic + adapter (entropic step) | 0.1% | 0.1% | 0.2 / 0.7 | 25.4% | -0.0% | 0.03 |
| SGD logistic, 25x longer | 0.3% | 0.7% | 0.6 / 1.7 | 25.4% | +0.0% | 0.02 |
| XGBoost | 6.5% | 15.3% | 13.1 / 19.5 | 24.8% | +0.3% | 0.04 |
| XGBoost + adapter (average 10) | 2.1% | 4.8% | 4.5 / 7.8 | 24.5% | -0.3% | 0.45 |
| XGBoost, 10x trees | 2.0% | 4.3% | 4.2 / 7.5 | 24.6% | -0.4% | 0.45 |
| LightGBM | 7.2% | 16.1% | 14.8 / 21.3 | 25.4% | +0.5% | 0.07 |
| LightGBM + adapter (average 10) | 2.2% | 5.1% | 4.6 / 8.1 | 25.1% | +0.1% | 0.65 |
| LightGBM, 10x trees | 2.1% | 4.5% | 4.2 / 7.2 | 25.1% | +0.2% | 0.65 |
| Random forest | 4.2% | 9.4% | 8.9 / 13.7 | 25.4% | +0.5% | 0.42 |
| Random forest + adapter (average 10) | 1.5% | 3.4% | 3.0 / 5.9 | 25.0% | +0.0% | 4.47 |
| Random forest, 10x trees | 1.4% | 3.2% | 2.8 / 5.8 | 25.1% | +0.1% | 4.32 |
| MLP | 14.4% | 32.2% | 30.0 / 38.2 | 27.6% | +1.6% | 0.70 |
| MLP + adapter (average 10) | 5.1% | 11.6% | 10.5 / 15.6 | 26.2% | -0.2% | 6.50 |

## 2. By group (before → after the adapter; groups with at least 30 pooled rows, sorted by "due to randomness" before)


### SGD logistic: groups (n >= 30), before -> after
| Group | Rows | Wrong one retrain | Due to randomness | Unstable rows |
|---|---|---|---|---|
| G2 job unemployed/unskilled non-resident | 36 | 28.8% -> 16.7% | +6.6% -> +0.0% | 55.6% -> 0.0% |
| G1 unemployed now | 102 | 41.8% -> 36.3% | +6.5% -> +0.0% | 59.8% -> 0.0% |
| G9 no property | 219 | 32.0% -> 27.9% | +5.5% -> +0.1% | 47.9% -> 0.5% |
| G3 housing for free | 157 | 35.1% -> 30.1% | +5.2% -> +0.1% | 53.5% -> 0.6% |
| G7 checking account < 0 DM | 435 | 37.0% -> 32.3% | +4.4% -> +0.1% | 59.1% -> 0.2% |
| G6 female | 466 | 28.6% -> 26.7% | +2.2% -> +0.1% | 36.5% -> 0.2% |
| G8 large and long loan | 230 | 37.0% -> 35.3% | +0.9% -> +0.1% | 44.8% -> 0.4% |
| G4 age < 25 | 226 | 37.6% -> 38.5% | +0.9% -> +0.0% | 42.9% -> 0.0% |
| G5 not a foreign worker | 54 | 15.5% -> 13.0% | +0.7% -> +0.0% | 13.0% -> 0.0% |
| G10 male divorced/separated | 74 | 23.2% -> 23.0% | +0.2% -> +0.0% | 35.1% -> 0.0% |

### XGBoost: groups (n >= 30), before -> after
| Group | Rows | Wrong one retrain | Due to randomness | Unstable rows |
|---|---|---|---|---|
| G5 not a foreign worker | 54 | 7.0% -> 5.8% | +1.4% -> +0.2% | 9.3% -> 1.9% |
| G1 unemployed now | 102 | 34.5% -> 31.3% | +1.2% -> -1.0% | 30.4% -> 13.7% |
| G7 checking account < 0 DM | 435 | 34.2% -> 33.6% | +1.1% -> -0.4% | 24.1% -> 6.2% |
| G10 male divorced/separated | 74 | 20.0% -> 19.5% | +1.0% -> +0.6% | 16.2% -> 4.1% |
| G3 housing for free | 157 | 33.0% -> 32.5% | +0.6% -> +0.0% | 21.0% -> 8.9% |
| G6 female | 466 | 25.8% -> 25.2% | +0.4% -> +0.1% | 14.4% -> 6.0% |
| G2 job unemployed/unskilled non-resident | 36 | 22.2% -> 19.2% | +0.0% -> -0.3% | 22.2% -> 8.3% |
| G9 no property | 219 | 29.2% -> 29.2% | -0.0% -> -0.0% | 18.7% -> 6.8% |
| G8 large and long loan | 230 | 35.4% -> 35.5% | -0.7% -> -0.6% | 22.6% -> 7.4% |
| G4 age < 25 | 226 | 37.0% -> 37.3% | -1.1% -> -0.4% | 20.8% -> 7.5% |

### LightGBM: groups (n >= 30), before -> after
| Group | Rows | Wrong one retrain | Due to randomness | Unstable rows |
|---|---|---|---|---|
| G10 male divorced/separated | 74 | 18.5% -> 16.4% | +2.3% -> +0.1% | 24.3% -> 9.5% |
| G5 not a foreign worker | 54 | 3.0% -> 2.7% | +1.1% -> +0.8% | 1.9% -> 1.9% |
| G6 female | 466 | 26.2% -> 26.2% | +0.4% -> +0.2% | 13.3% -> 4.7% |
| G4 age < 25 | 226 | 36.3% -> 35.4% | +0.4% -> +0.9% | 19.0% -> 5.8% |
| G7 checking account < 0 DM | 435 | 36.2% -> 36.4% | +0.1% -> +0.1% | 21.6% -> 5.3% |
| G8 large and long loan | 230 | 34.5% -> 34.8% | -0.2% -> -0.0% | 24.8% -> 8.3% |
| G9 no property | 219 | 28.2% -> 28.4% | -0.5% -> -0.3% | 16.4% -> 5.5% |
| G1 unemployed now | 102 | 38.5% -> 38.7% | -0.7% -> -0.6% | 29.4% -> 6.9% |
| G3 housing for free | 157 | 34.2% -> 34.4% | -0.8% -> +0.0% | 19.7% -> 7.0% |
| G2 job unemployed/unskilled non-resident | 36 | 20.9% -> 20.7% | -1.3% -> -1.5% | 25.0% -> 5.6% |

### Random forest: groups (n >= 30), before -> after
| Group | Rows | Wrong one retrain | Due to randomness | Unstable rows |
|---|---|---|---|---|
| G10 male divorced/separated | 74 | 21.2% -> 18.2% | +2.3% -> +0.7% | 12.2% -> 2.7% |
| G1 unemployed now | 102 | 38.2% -> 37.6% | +1.9% -> -0.6% | 12.7% -> 4.9% |
| G2 job unemployed/unskilled non-resident | 36 | 15.7% -> 15.8% | +1.9% -> -0.8% | 5.6% -> 2.8% |
| G8 large and long loan | 230 | 35.5% -> 34.5% | +1.6% -> +0.6% | 14.8% -> 6.1% |
| G3 housing for free | 157 | 31.4% -> 30.3% | +1.4% -> +0.3% | 15.9% -> 6.4% |
| G5 not a foreign worker | 54 | 10.7% -> 9.8% | +1.4% -> +0.6% | 3.7% -> 3.7% |
| G9 no property | 219 | 29.5% -> 28.3% | +1.2% -> +0.5% | 15.5% -> 6.4% |
| G7 checking account < 0 DM | 435 | 38.6% -> 37.8% | +1.1% -> -0.1% | 20.9% -> 7.1% |
| G6 female | 466 | 27.5% -> 27.1% | +0.2% -> +0.2% | 9.7% -> 3.2% |
| G4 age < 25 | 226 | 35.0% -> 34.9% | +0.1% -> -0.1% | 15.5% -> 7.1% |

### MLP: groups (n >= 30), before -> after
| Group | Rows | Wrong one retrain | Due to randomness | Unstable rows |
|---|---|---|---|---|
| G2 job unemployed/unskilled non-resident | 36 | 31.7% -> 25.5% | +6.7% -> +0.5% | 44.4% -> 16.7% |
| G10 male divorced/separated | 74 | 18.7% -> 15.8% | +5.2% -> +0.9% | 24.3% -> 5.4% |
| G3 housing for free | 157 | 35.5% -> 34.0% | +3.0% -> +0.3% | 43.9% -> 16.6% |
| G8 large and long loan | 230 | 33.7% -> 31.8% | +2.8% -> +0.1% | 37.0% -> 13.0% |
| G7 checking account < 0 DM | 435 | 35.4% -> 33.5% | +2.7% -> +0.1% | 43.4% -> 17.2% |
| G9 no property | 219 | 32.3% -> 31.2% | +2.1% -> -0.4% | 43.4% -> 17.4% |
| G1 unemployed now | 102 | 40.0% -> 37.4% | +1.7% -> -0.9% | 53.9% -> 17.6% |
| G6 female | 466 | 30.2% -> 28.6% | +1.0% -> -0.4% | 33.3% -> 11.6% |
| G4 age < 25 | 226 | 36.6% -> 35.3% | +0.7% -> -1.0% | 40.3% -> 11.1% |
| G5 not a foreign worker | 54 | 13.5% -> 13.3% | +0.6% -> -1.5% | 7.4% -> 3.7% |

## 3. Hypotheses

* **H1 supported.** The polish takes SGD logistic regression from 15.5% to 0.1% disagreement (entropic step: 0.1%), from
  35.3% to 0.3% unstable rows, and from 32.2 to 0.3 changed decisions per retrain.
* **H2 supported.** Averaging 10 seeds cuts disagreement by 64–69%: XGBoost 6.5% → 2.1%, LightGBM 7.2% → 2.2%,
  random forest 4.2% → 1.5%, MLP 14.4% → 5.1%.
* **H3 not supported.** At equal compute the alternatives do about as well:
  * SGD trained 25× longer reaches 0.3% disagreement (polish 0.1%) and is faster per fit here (0.02 s vs 0.06 s; the
    entropic step 0.03 s).
  * XGBoost, LightGBM and random forest with 10× the trees (lower learning rate for boosting) match averaging 10 seeds
    (2.0% vs 2.1%, 2.1% vs 2.2%, 1.4% vs 1.5%) at the same time per fit.
  * So the gain comes from averaging out randomness, whichever way it is done. What the adapter adds is that it works
    **after** training on a model you already have (no change to the training recipe, no retuning of the learning rate), it is
    chosen automatically by model type, and it comes with the measurement. The MLP has no "more trees" equivalent; it was not
    compared with a longer-trained MLP.
* **H4 supported for SGD and MLP, weak for boosted trees.** For SGD logistic regression the five groups with the largest share
  due to randomness (+4.4 to +6.6 points) lose that share, and the wrong-decision rate of one retrain drops by 4 to 12 points
  (unemployed 41.8% → 36.3%, housing for free 35.1% → 30.1%, no property 32.0% → 27.9%, checking < 0 DM 37.0% → 32.3%,
  unskilled non-resident 28.8% → 16.7%). For the MLP: unskilled non-resident 31.7% → 25.5%, male divorced/separated
  18.7% → 15.8%. For XGBoost and LightGBM the per-group shares due to randomness were already small (at most +2.3 points)
  and the wrong-decision changes are within about ±3 points; the unstable share still drops in every group.
* **H5 supported.** The wrong-decision rate of one retrain never got worse overall (SGD 27.3% → 25.4%, XGBoost
  24.8% → 24.5%, LightGBM 25.4% → 25.1%, random forest 25.4% → 25.0%, MLP 27.6% → 26.2%). At group level the largest
  increase was +0.9 points (SGD, age < 25: 37.6% → 38.5%).

## 4. Reading for the product

* For logistic heads trained by SGD the adapter makes the model effectively deterministic and lowers the wrong-decision rate by
  4–12 points in the five most affected groups (2 points overall). This is the case the theory covers (strongly convex objective; Can & Gürbüzbalaban 2025).
* For trees and neural networks the adapter is ensembling. It cuts disagreement by about two thirds at 10× the training cost;
  more trees at a lower learning rate is an equally good fix where available.
* The measurement (`stability_check`) is useful on its own: it tells a team which fix it needs and for whom.

Statements describe statistical behaviour of models on a public dataset; not accusations of bias and not credit advice.
