# Evidence

Every number below comes from experiments in the research repository [bgra1123/ragmm](https://github.com/bgra1123/ragmm) (folders `research/*`), each with a written plan, the code and the results.
German Credit is a public dataset from the 1990s; results describe training behaviour, not lending today.

## The problem
| Finding | Number | Source |
|---|---|---|
| Decisions that change per retrain (textbook SGD logistic, 1,000 retrains, 300 held-out applicants) | 55 of 300 (18%) | `research/report` |
| Applicants with a different answer in at least one retrain | 284 of 300 | `research/report` |
| Chance of a wrong decision, all applicants: standard / no randomness / with polish | 30.4% / 24.4% / 25.8% | `research/groups` |
| Repaying applicants refused, housing "for free": standard / no randomness / with polish | 34.6% / 24.7% / 28.4% | `research/groups` |
| Same, no property | 33.1% / 23.6% / 27.1% | `research/groups` |
| The effect is large for aggressive step sizes; with learning rate 0.01 it is 0.7-3.4% of decisions | | `research/gate0` |
| The 5% most affected people carry 20-58% of the extra loss in the worst retrains (six datasets) | | `research/tail_analysis` |

## The fixes
| Finding | Number | Source |
|---|---|---|
| Exact polish, SGD logistic: two retrains disagree | 15.5% to 0.1% | `research/product` |
| Averaging 10 seeds: XGBoost / LightGBM / random forest / MLP | 6.5→2.1 / 7.2→2.2 / 4.2→1.5 / 14.4→5.1% | `research/product` |
| 10 times more trees match averaging 10 seeds at equal compute | 2.0 vs 2.1% (XGBoost) | `research/product` |
| Head refit does not help trees; function-space polish does not help boosting | | `research/trees`, `research/boost` |
| Calibration lowers the expected cost at a cost-based cutoff for every tree model and the MLP | e.g. XGBoost 0.562 to 0.530 | `research/sota2` |
| Newton solves the logistic head exactly in 3 to 10 iterations from a trained model (Adult, 46,842 rows, needs 8 to 10) | | `research/risk_effect` |

## Against simply training longer
Averaging the iterates of one more epoch of minibatch training is a strong baseline. The exact polish beats it from the first epoch with an Adam start and from about 2 to 5 epochs with an SGD start, but not on large data with
fewer than two passes (Adult, up to 46,842 rows), where average-and-continue wins (`research/risk_effect`).

## What did not work
RA-GMM as an optimizer ties plain SGD tuned to the same average and loses to tuned SGD with averaging; the paper's quadratic formulation was worse than textbook SGD on a portfolio problem; head refit hurt random forests; steps chosen on
training rows over-fit boosted trees; a tilted objective did not help (`research/model_multiplicity`, `portfolio`, `boost`, `risk_effect`).
