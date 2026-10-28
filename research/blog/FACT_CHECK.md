# Where every number in the blog post comes from

Revised with the post on 2026-10-28. Rows marked *new* were added in the revision.

| Claim in the post | Source |
|---|---|
| Opening applicant: unemployed, housing "for free", small car loan, repaid, exact model approves; approved by 56 of 100 standard-trained retrains (74 of 100 after the short polish, 90 after the deterministic polish) *new* | `research/groups/RESULTS.md` §3, applicant 2 (split 0, 100 retrains) |
| 1,000 retrains of textbook SGD on German Credit (700 train / 300 held-out); 55 of 300 (18%) differ from the exact model; 284 of 300 got a different answer at least once; 16 never changed | `research/report/REPORT.md` §1; `research/report/credit_story.json` (`changed_mean` 55.16 at threshold 0.5, `ever_changed` 284) |
| With a cautious learning rate (0.01) the effect is 1–3% of decisions *new* | `research/gate0/RESULTS.md` (flips 0.7–3.4%, two retrains disagree 0.9–2.9%) |
| Chance of a wrong decision by group (37.8 / 27.4 / +10.4; 35.4 / 25.6 / +9.9; 40.0 / 31.3 / +8.8; 38.7 / 31.7 / +7.0; 42.9 / 37.3 / +5.6; all 30.4 / 24.4 / +6.0) | `research/groups/CONCRETE.md`, `research/groups/results/concrete.json` (5 splits × 100 retrains; rule 0.5) |
| Repaying applicants in housing "for free" refused 34.6% (standard) vs 24.7% (exact); "one refusal in three" = 9.9 of 34.6 points *new* | `research/groups/RESULTS.md` §2 |
| Defaulters approved at the 83% cutoff: 26.6% standard, 9.1% no randomness | `research/groups/CONCRETE.md` (second section) |
| Six datasets; Adult: 89% born outside the US vs 6%; 5% of people carry 20–58% of the tail | `research/tail_analysis/RESULTS.md` |
| Score wobble vs rarity: Spearman 0.52–0.90 on six datasets; vs leverage 0.59–0.90 on five of six *new* | `research/rare_flip/RESULTS.md` §1 |
| Rarest 10% flip 1.4–1.9× as often at equal distance (German 1.45, COMPAS 1.39, Adult 1.79, Bank 1.86, Pima 1.63); COMPAS and Pima rare rows flip less in raw terms *new* | `research/rare_flip/RESULTS.md` §2 |
| After the polish fewer than 1% of rows flip and the rare-row excess disappears *new* | `research/rare_flip/RESULTS.md` §4 (0.0–0.9%; Adult ratio 0.88) |
| RA-GMM vs plain SGD at the same average quality is a tie; ≈15 fewer changed decisions than textbook; tuned SGD with averaging 5.7 vs 40 | `research/report/REPORT.md` §6; `research/tail_analysis/RESULTS.md` |
| Risk-averse step rule matched an exact line search on stability; a plain step of the same size did almost the same; hurts fewer rows per step *new* | `research/risk_effect/RESULTS.md` (one-more-epoch, step-matched and negative-flip addenda) |
| Newton reaches the exact model in 3–10 iterations from a trained model *new* | `research/risk_effect/RESULTS.md` (Newton addendum) |
| Five passes bring every trainer to within 1–8 of 300 decisions; twenty passes under one | `research/report/CREDIT_WITH_ADAPTER.md` (1.2–8.2 at 5 epochs; ≤ 0.9 at 20) |
| Package: disagreement 15.5% → 0.1%; wrong decision 27.3% → 25.4%; unemployed 41.8 → 36.3; housing for free 35.1 → 30.1; no property 32.0 → 27.9 *new* | `research/product/RESULTS.md` |
| Exact finish vs one more averaged epoch: wins from the first epoch with an Adam start, from 2–5 epochs with SGD; averaged epoch better on large data with under two passes *new* | `research/risk_effect/RESULTS.md` (section 1 and the Experiment C addendum) |
| Head refit: no change for the MLP, random forest 4.1% → 9.1%; continued boosting 5.6% → 5.7% | `research/sota2/RESULTS.md`; `research/boost/RESULTS.md` |
| Averaging 10 seeds: XGBoost 6.5% → 2.1%, MLP 14.4% → 5.1% (package benchmark); earlier research run 5.6% → 1.6%, 14.8% → 4.7% | `research/product/RESULTS.md`; `research/sota2/RESULTS.md` |
| Ten times more trees at a lower learning rate matches averaging 10 seeds *new* | `research/product/RESULTS.md` (XGBoost 2.0% vs 2.1%; LightGBM 2.1 vs 2.2; RF 1.4 vs 1.5) |
| Calibration lowers cost for every model; calibrated + averaged random forest 0.487 vs exact logistic regression 0.497 | `research/sota2/RESULTS.md` |
| Exact model refuses 24.7% of good "housing for free" applicants vs 4.3% outside the listed groups | `research/groups/RESULTS.md` |
| Quadratic variant worse than textbook SGD on the portfolio problem | `research/portfolio/RESULTS.md` (P5) |
| Paper: Can & Gürbüzbalaban, OMS 40(6), 2025, arXiv:2204.11292 | repository README |

## Things to keep in mind when publishing

* "About 15 fewer changed decisions" is per retrain out of 300 (RA-GMM vs textbook; interval 14.4–16.0).
* "One refusal in three" rounds 29% (9.9 / 34.6); say "roughly".
* The 1.4–1.9× range excludes Heart (too few rows) and is at the aggressive step; at learning rate 0.01 the ratio is larger but only 1–3% of rows flip at all.
* The 56 / 74 / 90 of 100 figures are for one applicant in one split; the post labels her an example, not a typical case.
* German Credit is a 1990s dataset; the post says so and makes no claim about current lending.
