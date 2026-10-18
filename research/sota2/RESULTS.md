# XGBoost and other state-of-the-art models as baselines, with and without post-training adapters

Design: `PLAN.md` (pre-registered; one deviation recorded before any result: averaged ensembles are 200 random 10-seed subsets instead of 10 disjoint groups). Code: `fit_models.py` (training),
`analyze.py` (all numbers); raw probabilities `results/probs_split*.npz`, summary `results/analysis.json`. German Credit, **3 stratified splits** (900 held-out applicants pooled: 630 repaid, 270 defaulted),
100 retrains of every model per split. "Adapters" are applied to the frozen trained model: **head refit** (re-solve the last logistic layer exactly: leaf indicators for trees, penultimate activations for the MLP),
**averaging** (mean of 10 seeds), **calibration** (one-dimensional logistic fit of the score on out-of-fold training predictions). For logistic regression the adapter is the polish step.
Decisions at 0.5 (stability) and at the cost-based 5/6 rule (outcomes). "Tail above the mean" = (average cost of the worst 5% of retrains) − (average cost).

## 1. Stability and worst-case decision changes

| Model | Variant | Two retrains disagree | Coin-flip applicants | Decisions changed (avg / worst 5% of retrains) | AUC |
|---|---|---|---|---|---|
| Logistic regression | exact solution (reference) | 0.0% | 0.0% | 0.0 / 0.0 | 0.792 |
| Logistic regression | SGD, no adapter | 24.9% | 58.0% | 161.7 / 195.6 | 0.724 |
|  | + polish adapter (5 epochs) | 8.5% | 20.1% | 53.9 / 67.2 | 0.783 |
|  | + deterministic line search | 3.3% | 7.6% | 20.8 / 30.8 | 0.788 |
| XGBoost | no adapter | 5.6% | 11.8% | 35.0 / 46.0 | 0.801 |
|  | + head refit | 6.4% | 14.8% | 40.5 / 54.2 | 0.789 |
|  | + averaging (10 seeds) | 1.6% | 3.3% | 9.7 / 15.3 | 0.804 |
|  | + head refit + averaging | 1.8% | 4.6% | 11.1 / 17.4 | 0.793 |
|  | + calibration | 5.5% | 12.7% | 35.1 / 45.2 | 0.800 |
|  | + calibration + averaging | 1.7% | 3.9% | 11.2 / 17.5 | 0.804 |
| LightGBM | no adapter | 6.3% | 14.3% | 40.8 / 51.8 | 0.793 |
|  | + head refit | 6.2% | 13.6% | 39.3 / 51.6 | 0.791 |
|  | + averaging (10 seeds) | 2.0% | 4.8% | 12.7 / 18.8 | 0.797 |
|  | + head refit + averaging | 1.9% | 4.1% | 12.1 / 18.6 | 0.794 |
|  | + calibration | 5.2% | 12.8% | 32.8 / 43.0 | 0.793 |
|  | + calibration + averaging | 1.4% | 3.1% | 9.1 / 14.7 | 0.797 |
| Random forest | no adapter | 4.1% | 8.9% | 25.8 / 33.8 | 0.804 |
|  | + head refit | 9.1% | 20.7% | 57.9 / 69.0 | 0.794 |
|  | + averaging (10 seeds) | 1.2% | 2.9% | 7.3 / 12.3 | 0.807 |
|  | + head refit + averaging | 2.8% | 6.1% | 18.0 / 25.5 | 0.804 |
|  | + calibration | 5.2% | 11.9% | 33.0 / 46.0 | 0.803 |
|  | + calibration + averaging | 1.6% | 4.0% | 9.9 / 16.1 | 0.806 |
| HistGradientBoosting | no adapter | 0.0% | 0.0% | 0.0 / 0.0 | 0.785 |
|  | + averaging (10 seeds) | 0.0% | 0.0% | 0.0 / 0.0 | 0.785 |
|  | + calibration | 2.0% | 4.3% | 13.0 / 27.8 | 0.784 |
|  | + calibration + averaging | 0.6% | 1.2% | 4.0 / 8.1 | 0.784 |
| MLP | no adapter | 14.8% | 32.8% | 93.9 / 113.8 | 0.751 |
|  | + head refit | 14.8% | 33.4% | 93.8 / 113.8 | 0.751 |
|  | + averaging (10 seeds) | 4.7% | 10.4% | 30.2 / 40.1 | 0.773 |
|  | + head refit + averaging | 4.7% | 10.9% | 29.8 / 41.0 | 0.775 |
|  | + calibration | 10.9% | 24.7% | 70.2 / 86.0 | 0.750 |
|  | + calibration + averaging | 3.6% | 7.6% | 23.4 / 32.8 | 0.771 |

(HistGradientBoosting has no leaf interface here, so no head refit. Its calibration row differs from 0 only because the 3-fold out-of-fold fits use different training subsets per seed.)

## 2. Credit outcomes at the cost-based rule (approve only at 83% or above)

| Model | Variant | Good wrongly denied | Defaulters approved | Default rate among approved | Cost per applicant | Cost: average of worst 5% of retrains | Tail above the mean |
|---|---|---|---|---|---|---|---|
| Logistic regression | exact solution | 50.3% | 9.6% | 7.7% | 0.497 | 0.497 | 0.000 |
|  | SGD, no adapter | 39.9% | 27.2% | 16.2% | 0.688 | 0.804 | 0.116 |
|  | + polish adapter | 45.9% | 13.6% | 9.7% | 0.526 | 0.577 | 0.051 |
| XGBoost | no adapter | 32.4% | 22.3% | 12.4% | 0.562 | 0.601 | 0.039 |
|  | + averaging (10) | 32.9% | 21.7% | 12.1% | 0.555 | 0.572 | 0.017 |
|  | + calibration | 62.1% | 6.3% | 6.7% | 0.530 | 0.556 | 0.027 |
|  | + calibration + averaging | 61.8% | 6.1% | 6.4% | 0.523 | 0.536 | 0.013 |
| LightGBM | no adapter | 25.2% | 30.5% | 14.9% | 0.634 | 0.677 | 0.043 |
|  | + averaging (10) | 26.2% | 29.1% | 14.4% | 0.620 | 0.639 | 0.019 |
|  | + calibration | 70.8% | 3.8% | 5.2% | 0.552 | 0.581 | 0.029 |
|  | + calibration + averaging | 71.3% | 3.1% | 4.4% | 0.545 | 0.558 | 0.013 |
| Random forest | no adapter | 66.3% | 4.4% | 5.3% | 0.530 | 0.548 | 0.018 |
|  | + averaging (10) | 66.8% | 4.4% | 5.4% | 0.533 | 0.542 | 0.008 |
|  | + calibration | 53.7% | 8.3% | 7.1% | 0.500 | 0.528 | 0.028 |
|  | + calibration + averaging | 53.2% | 7.6% | 6.5% | 0.487 | 0.502 | 0.015 |
| HistGradientBoosting | no adapter | 34.0% | 23.0% | 13.0% | 0.582 | 0.582 | -0.000 |
|  | + averaging (10) | 34.0% | 23.0% | 13.0% | 0.582 | 0.582 | 0.000 |
|  | + calibration | 65.2% | 5.7% | 6.5% | 0.541 | 0.555 | 0.014 |
|  | + calibration + averaging | 65.0% | 5.5% | 6.3% | 0.538 | 0.547 | 0.009 |
| MLP | no adapter | 23.5% | 41.0% | 18.7% | 0.780 | 0.842 | 0.062 |
|  | + averaging (10) | 30.4% | 30.9% | 16.0% | 0.676 | 0.709 | 0.033 |
|  | + calibration | 63.5% | 8.9% | 9.4% | 0.578 | 0.635 | 0.057 |
|  | + calibration + averaging | 66.1% | 5.1% | 6.1% | 0.539 | 0.561 | 0.021 |

Exact logistic regression: AUC 0.792 and cost 0.497 on these three splits (the single-split numbers in the earlier reports differ because the test sets differ).

## 3. Where training noise lands, by group (flip rate vs consensus, %; groups as in `research/groups`)

| Group | Logistic SGD | + polish adapter | XGBoost | + averaging | + head refit | LightGBM | Random forest | + averaging | MLP | + averaging |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 unemployed | 24.6 | 8.7 | 3.9 | 1.5 | 3.8 | 4.6 | 3.4 | 0.4 | 16.3 | 6.6 |
| G3 housing for free | 24.8 | 9.6 | 3.9 | 1.2 | 5.0 | 4.7 | 5.3 | 1.1 | 14.6 | 6.7 |
| G7 checking < 0 DM | 26.8 | 10.8 | 5.7 | 1.9 | 6.6 | 7.4 | 7.3 | 2.3 | 14.2 | 5.0 |
| G8 large and long loan | 22.3 | 6.9 | 5.1 | 0.8 | 6.3 | 6.3 | 5.4 | 1.6 | 13.6 | 4.7 |
| G9 no property | 23.1 | 8.6 | 3.7 | 1.2 | 3.9 | 4.4 | 5.6 | 1.3 | 15.4 | 6.6 |
| Applicants in none of the groups | 11.1 | 2.8 | 3.1 | 0.9 | 2.9 | 3.0 | 0.4 | 0.1 | 7.5 | 1.9 |

## Hypotheses

| | Result |
|---|---|
| **H1** MLP and random forest less stable than careful logistic regression; HGB deterministic | **Partly.** The MLP (14.8% disagreement) is the least stable; random forest (4.1%), XGBoost (5.6%) and LightGBM (6.3%) are all stabler than untuned textbook SGD (24.9%) but less stable than the polished/line-search logistic models (8.5% / 3.3%). HistGradientBoosting is exactly 0. (I did not run careful tuned logistic SGD here.) |
| **H2** head refit ≥ 30% cut for the MLP, < 30% for trees | **MLP: not supported on German** (14.8% → 14.8%). **Trees: supported in the sense of no gain, but it can hurt:** XGBoost +14%, LightGBM −2%, **random forest +122%** (4.1% → 9.1%). |
| **H3** averaging 10 seeds cuts disagreement ≥ 50% for every randomized model | **Supported:** XGBoost −72%, LightGBM −68%, random forest −71%, MLP −68%. |
| **H4** flips in the six unusual-profile groups ≥ 1.3× others in ≥ 4 of 6 groups | **Logistic SGD 6/6, MLP 6/6, random forest 6/6, LightGBM 5/6; XGBoost only 2/6.** Training noise in XGBoost is spread more evenly across applicants than in the other models. |
| **H5** calibration lowers 5/6-rule cost for XGBoost, LightGBM, RF, MLP; changes disagreement by ≤ 20% | **Cost: supported** (XGBoost 0.562 → 0.530, LightGBM 0.634 → 0.552, RF 0.530 → 0.500, MLP 0.780 → 0.578, HGB 0.582 → 0.541). **Disagreement: not** for the random forest (+27%), MLP (−26%) and HGB (0 → 2.0%). |
| **H6** averaging cuts the worst-5% cost by ≥ 40% | **Not as stated:** the average cost of the worst 5% of retrains barely moves (XGBoost 0.601 → 0.572) because it is dominated by the mean. **The tail above the mean falls 47–56%** (XGBoost 0.039 → 0.017, LightGBM 0.043 → 0.019, RF 0.018 → 0.008, MLP 0.062 → 0.033); this measure was not pre-registered. |
| **H7** best adapter combination matches or beats exact logistic regression on cost at 5/6 | **Random forest + calibration + averaging: 0.487 vs 0.497** (and AUC 0.806 vs 0.792). XGBoost 0.523, MLP 0.539, HGB 0.538, LightGBM 0.545 do not. |

## What this means

1. **Adapters help most where they are a post-processing of the output.** Averaging seeds cuts disagreement by about 70% for every randomized model, the tail above the mean by about half, and raises AUC slightly.
   Calibration is what turns accurate scores into good cost-based decisions: it lowers the 5/6 cost of every SOTA model (the MLP by 26%), even though it does not make them more stable.
2. **The head refit, the closest analogue of the polish step, does not help these models and can hurt (random forest +122%).** The polish idea works for logistic regression because the head is the model.
3. **Combining calibration with averaging is the best adapter set**: XGBoost 0.562 → 0.523, LightGBM 0.634 → 0.545, RF 0.530 → 0.487, MLP 0.780 → 0.539. With it, a calibrated and averaged random forest beats exactly trained logistic regression here (0.487 vs 0.497), while XGBoost and the MLP do not.
4. **Baseline comparison:** without adapters, SOTA models are more stable than textbook SGD logistic regression (5.6–6.3% vs 24.9%) but this is a weak baseline; with the polish step logistic regression (8.5%, or 3.3% with line search) is comparable to XGBoost with averaging (1.6%).
5. **Group view:** where noise lands depends on the model: logistic SGD, the MLP and the random forest concentrate it on unusual profiles (flip rates 2–20× those of other applicants), XGBoost much less. Averaging lowers every group's flip rate; for random forest and MLP the unusual-profile groups still flip more than others after averaging.

## Caveats

* One dataset with 700 training rows per split, where gradient-boosted trees and networks are known to have little advantage; three splits that re-use the same 1,000 applicants. Models were not tuned.
* Averaged models use random 10-subsets of the 100 seeds (overlapping), so their unit-to-unit variation is somewhat understated; disagreement for averaged models is therefore a lower bound on what independent ensembles would show.
* The cost numbers depend on the 5:1 cost matrix and on this dataset's labels; the 5/6 rule assumes calibrated probabilities.
* The "exact solution" row has 2 identical units by construction.
