# Fact sheet: every number used in the content, and where it comes from

Use only these numbers in scripts, captions and posts. If a number is not here, add it with its source first.
"Wrong decision" = the model refuses someone who repaid or approves someone who defaulted (the dataset's own label [12]).

## A. The problem (SGD logistic regression, 1,000 retrains) — source: `research/report/REPORT.md`, `research/report/credit_story.json`

| ID | Fact | Used in |
|---|---|---|
| A1 | 1,000 retrains, same 700 training applicants, 300 held out; only the random seed changes | B1, Y1, T1 |
| A2 | Each retrain gives 55 of 300 applicants (18%) a different decision from the model without training randomness | B1, Y1, T1, video |
| A3 | 284 of 300 applicants got a different answer from at least one retrain; 16 never changed | B1, Y1, T1, video |

## B. Who is affected (5 splits × 100 retrains) — source: `research/groups/CONCRETE.md`

| ID | Fact | Used in |
|---|---|---|
| B1 | All applicants: 30.4% wrong with standard training vs 24.4% for the model without training randomness (+6.0 points) | B1, Y1, video |
| B2 | Housing "for free": 37.8% vs 27.4% (+10.4); after a short polish: 31.5% | B1, Y1, T2, video |
| B3 | No property: 35.4% vs 25.6% (+9.9) | B1, Y1, T2 |
| B4 | Negative checking balance: 40.0% vs 31.3% (+8.8) | B1, Y1 |
| B5 | Unemployed: 42.9% vs 37.3% (+5.6) | B1, Y1, T2 |
| B6 | Defaulters approved at the cost cutoff (83%): 26.6% standard, 9.1% without randomness, 13.1% with the polish adapter | B1, Y1 |

## C. Why the obvious fix is not enough — sources: `research/report/REPORT.md` section 6, `research/sota2/RESULTS.md`

| ID | Fact | Used in |
|---|---|---|
| C1 | RA-GMM as the trainer ties plain SGD tuned to the same average quality: it is not a better trainer | Y1, Y2 |
| C2 | Re-solving the last layer did not help the MLP on German Credit and made random forest worse (4.1% → 9.1% disagreement) | Y2, B2 |

## D. The product (package 0.4.0 defaults, 5 splits × 300 held out, 30 retrains) — source: `research/product/RESULTS.md`

| ID | Fact | Used in |
|---|---|---|
| D1 | SGD logistic: disagreement 15.5% → 0.1%; unstable rows 35.3% → 0.3%; changed decisions per retrain 32.2 → 0.3 | B2, Y2, T3 |
| D2 | SGD logistic: wrong decision (one retrain) 27.3% → 25.4% | B2, Y2 |
| D3 | SGD, groups: unemployed 41.8% → 36.3%; housing for free 35.1% → 30.1%; no property 32.0% → 27.9%; checking < 0 DM 37.0% → 32.3%; unskilled non-resident (36 cases) 28.8% → 16.7% | B2, Y2, T4 |
| D4 | Averaging 10 seeds: XGBoost 6.5% → 2.1%, LightGBM 7.2% → 2.2%, random forest 4.2% → 1.5%, MLP 14.4% → 5.1% disagreement | B2, Y2, T5 |
| D5 | Wrong-decision rate never got worse overall with the adapter; worst group-level change +0.9 points (SGD, age < 25) | B2, Y2 |
| D6 | Cost: polish adds about 0.05 s per fit on German Credit (20 full passes); averaging costs 10× | B2, Y2 |
| D7 | Equal compute: 10× trees match averaging (XGBoost 2.0% vs 2.1%; LightGBM 2.1% vs 2.2%; RF 1.4% vs 1.5%); SGD 25× longer reaches 0.3% vs 0.1% for polish | B2, Y2 |
| D8 | MLP groups: unskilled non-resident 31.7% → 25.5%; male divorced/separated 18.7% → 15.8% | Y2 |

## E. The risk-averse line search — source: `research/linesearch/RESULTS.md`

| ID | Fact | Used in |
|---|---|---|
| E1 | At 20 full passes the entropic line search (θ = 1) and the exact line search leave the worst 5% of runs at most 5·10⁻⁵ above the optimum, vs 0.0007–0.013 for the stochastic polish | B2, Y2 |
| E2 | One step at θ = 5: the 95th-percentile per-row loss increase is 84–91% smaller than with the exact line search (German +0.06 vs +0.69), with smaller steps | B2, Y2 |

| E3 | Against "one more epoch" (Adam start): the risk-averse first-order step lowers two-retrain disagreement from 8.4% (stage 1) to 3.7% after 1 epoch and 0.0% after 20 on German; best continuation (averaged iterates): 8.4% and 3.0% | Y2, B2 |
| E4 | The risk-averse first-order step matches the plain exact line search within about 1 point of disagreement; the gain over continuation is the exact full-data gradient, not the risk term | Y2, B2 |
| E5 | From an SGD start, averaged continuation beats the adapter at 1 epoch on COMPAS (3.4% vs 10.7%); the adapter wins from about 2 epochs on German, Pima, Heart and from 5 on COMPAS | Y2, B2 |

(source for E3-E5: `research/risk_effect/RESULTS.md`)

## F. Concentration across datasets — source: `research/tail_analysis/RESULTS.md`

| ID | Fact | Used in |
|---|---|---|
| F1 | Across six datasets, the 5% most affected people carry 20–58% of the bad-run losses | Y1, T6 |

## Standard caveats (say at least one in every piece)

* One public dataset from the 1990s [12]; not a claim about any lender today.
* Statistical behaviour of models; not an accusation of bias; not legal, credit or compliance advice.
* The adapter removes variation caused by the training seed; it does not fix what the model learned from the data.
