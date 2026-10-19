# Where every number in the blog post comes from

| Claim in the post | Source |
|---|---|
| 1,000 retrains of textbook SGD on German Credit (700 train / 300 held-out); 55 of 300 (18%) differ from the exact model; 284 of 300 got a different answer at least once; 16 never changed | `research/report/REPORT.md` section 1; `research/report/credit_story.json` (`changed_mean` 55.16 at threshold 0.5, `ever_changed` 284) |
| Chance of a wrong decision by group (37.8 / 27.4 / +10.4; 35.4 / 25.6 / +9.9; 40.0 / 31.3 / +8.8; 38.7 / 31.7 / +7.0; 42.9 / 37.3 / +5.6; all 30.4 / 24.4 / +6.0) | `research/groups/CONCRETE.md`, `research/groups/results/concrete.json` (5 splits × 100 retrains; rule 0.5). The "+" column in the post is the table's excess (JSON values rounded). |
| Defaulters approved at the 83% cutoff: 26.6% standard, 9.1% no randomness, 13.1% with the polish adapter | `research/groups/CONCRETE.md` (second section) |
| Same pattern on six datasets; Adult: the 5% most affected carry 58% of the tail, 89% born outside the US vs 6%; 5% of people carry 20–58% across datasets | `research/tail_analysis/RESULTS.md` (concentration 20–58%; Adult profile) |
| RA-GMM vs plain SGD at the same average quality is a tie; ≈15 fewer changed decisions than textbook; carefully tuned SGD 5.7 changed decisions vs 40 for RA-GMM | `research/report/REPORT.md` section 6; `research/tail_analysis/RESULTS.md` |
| Portfolio version worse than textbook SGD (noise grows with the weights) | `research/portfolio/RESULTS.md` (P5) |
| Five passes of deterministic steps bring every trainer close to the exact model; deterministic polish beats the stochastic adapter at equal compute | `research/report/CREDIT_WITH_ADAPTER.md`; `research/linesearch/RESULTS.md` |
| Defaulter-approval fall 26.6% → 13.1% | the 13.1% figure is the stochastic adapter (`research/groups/CONCRETE.md`); the post states the deterministic version is better without quoting a separate number: for textbook SGD at 5/6 the cost fell 0.751 → 0.595 (adapter) and 0.592 (line search) in `CREDIT_WITH_ADAPTER.md` |
| XGBoost 5.6% → 1.6% and neural network 14.8% → 4.7% with 10-seed averaging; head refit does nothing for the NN on German and makes random forest worse (4.1% → 9.1%); continued boosting leaves XGBoost at 5.6% → 5.7% | `research/sota2/RESULTS.md`; `research/boost/RESULTS.md` |
| Calibration lowers cost for every model; calibrated + averaged random forest cost 0.487 vs exact logistic regression 0.497 | `research/sota2/RESULTS.md` (section 2, H5/H7) |
| Exact model still refuses 24.7% of good "housing for free" applicants vs 4.3% outside the listed groups | `research/groups/RESULTS.md` |
| Paper: Can and Gürbüzbalaban, *Entropic Risk-Averse Generalized Momentum Methods*, Optimization Methods & Software (2025), arXiv:2204.11292 | repository README / report |

## Things to check or soften before publishing

* "about 15 fewer changed decisions" is per retrain out of 300 (RA-GMM vs textbook; interval 14.4–16.0).
* The "5 passes bring every trainer within about one decision in 300" statement is for the deterministic line search: changed decisions 1.2–8.2 at 5 passes in the credit-outcome table, ≤ 0.9 at 20 passes. "About one decision" is true at 20 passes, not 5. **Edit the post to say "within a handful of decisions (1–8 of 300) after five passes and under one after twenty".**
* The 26.6% → 13.1% figure is the stochastic adapter; the deterministic version was only measured at the 0.5 cutoff in `CREDIT_WITH_ADAPTER.md` (and cost 0.592 at the cost cutoff for textbook SGD).
* German Credit is a 1990s dataset; do not present the percentages as claims about current lending.
