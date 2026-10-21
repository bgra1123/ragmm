# What you can say about your results, and how to word it

All numbers: German Credit (public, 1990s), logistic regression, held-out applicants, decision rule "approve at 50%" unless stated. "Reliable applicant" = an applicant who repaid
(the dataset's own label). Sources: the research repository (bgra1123/ragmm), see [EVIDENCE.md](EVIDENCE.md).
Sex, age and foreign-worker status were never inputs to any model; they are used only to describe where errors land.

## 1. The claim forms, from strongest to weakest evidence

**A. "A model trained this way turns away X% of reliable applicants, and Z points of that come from the luck of the training run."**
Evidence: E10 (300 held-out applicants, 210 repaid, 300 training seeds) and E9 (five splits, 100 retrains each).

| Statement | Number |
|---|---|
| Repaying applicants refused, standard (fast) training | 19.6% |
| Same, model trained without randomness | 12.9% |
| Same, standard training + short polish | 13.9% |
| Part due to training randomness | 6.7 points, about a third (34%) of the refusals |

Wording: "In our test, a credit model trained the standard fast way refused 19.6% of applicants who went on to repay. The same model without training randomness refused 12.9%.
About a third of the refusals of reliable applicants came from the luck of the training run."

**B. "For a type of applicant, X% of reliable applicants were refused, and Z points of that are the training run."** (E9, 100 retrains x 5 splits; n = repaying cases pooled over splits)

| Group (repaying applicants who...) | n | Refused: standard | Without randomness | With polish | Points due to the run (share of refusals) |
|---|---|---|---|---|---|
| live in housing "for free" | 97 | 34.6% | 24.7% | 28.4% | 9.9 (29%) |
| have no property | 127 | 33.1% | 23.6% | 27.1% | 9.5 (29%) |
| are currently unemployed | 62 | 36.0% | 29.0% | 32.4% | 7.0 (19%) |
| have a negative checking balance | 223 | 40.1% | 34.1% | 34.7% | 6.0 (15%) |
| ask for a large and long loan | 129 | 35.6% | 26.4% | 29.5% | 9.2 (26%) |
| are divorced or separated men | 55 | 22.7% | 10.9% | 13.8% | 11.8 (52%) |
| are women | 324 | 21.8% | 12.7% | 14.7% | 9.1 (42%) |
| are under 25 | 136 | 28.8% | 25.7% | 24.6% | 3.1 (11%) |
| are not foreign workers | 48 | 21.5% | 12.5% | 13.9% | 9.0 (42%) (indicative, small) |
| have a job classed unemployed/unskilled non-resident | 27 | 31.6% | 11.1% | 21.7% | 20.5 (indicative, very small) |
| are in none of these groups | 327 | 11.7% | 4.3% | 5.0% | 7.4 |

Wording (example): "Among applicants who lived in housing 'for free' and repaid, a standard-trained model refused 34.6%. A model without training randomness would have refused 24.7%.
Nearly 10 points are the luck of training, and a short polish step removed about two thirds of them (28.4%)."

**C. "Decisions flip."** (E1, E3, E14) A standard-trained SGD model changes 55 of 300 held-out decisions per retrain (18%) against the exactly trained model; 284 of 300 applicants get a different answer
in at least one of 1,000 retrains. With the package polish, two retrains disagree on 0.1% of rows instead of 15.5%. Careful settings already cut the effect to 1-3% (E3): always say which regime.

**D. "Who is hit."** (E4) Across six datasets the 5% most affected people carry 20-58% of the extra loss in the worst retrains; they are people unlike most of the training data (rare categories, extreme values).

**E. Lender-side claim, at the cost-based cutoff (approve only at 83%).** (E9) A defaulter is approved 26.6% of the time with standard training, 9.1% without randomness, 13.1% with the polish.
At this cutoff the claim is about wrongly **approved** defaulters, not refused reliable applicants: standard training refuses *fewer* reliable applicants (39.4%) than the exact model (48.1%) because it is noisy and over-approves,
so the sentence "your model rejects more reliable applicants because of training randomness" is true at the 50% rule and **false** at the 83% rule. Always state the rule.

## 2. What "your model" requires

Every number above describes **this dataset and these models**. To say it about a customer's model you must measure it on their data: `stability_check` retrain the model many times,
give the same quantities (chance two retrains disagree, unstable rows, refusals of reliable applicants by group, part due to the run), and the sentences in section 1 become fill-in templates:

> "In {N} retrains of {model} on your data, {X}% of applicants who repaid were refused on average. A model without training randomness would refuse {Y}%. {Z} points of the refusals come from the luck of the training run."

Add the group version only for groups with enough rows (the package reports groups of at least 15 held-out rows; claims for groups under about 30 should say "indicative").

## 3. What we cannot say, and why

* **"Women / young people / non-foreign workers are Y% more likely to be rejected."** Not supported. We compared each group with the model without randomness *for the same group*, not with other groups, and the groups differ
  in income, loan size and risk. A between-group fairness claim needs a different design (matched or adjusted comparisons, larger samples, an agreed fairness metric) that we did not run. The valid statement is within-group:
  "of the refusals of reliable women applicants, 9.1 points came from training randomness."
* **"The model discriminates."** Not supported and not what the data shows. Sex, age and foreign-worker status were never inputs; the exact model's own refusal rate for these groups is a property of its features and data
  (for example 25.7% of repaying under-25 applicants even without randomness). Removing training randomness makes decisions reproducible; it does not fix that.
* **"X% of all lenders' applicants..."** One dataset from the 1990s; not a statement about any lender today or any real portfolio.
* **"The adapter makes decisions fairer / more accurate."** It moves decisions toward the exactly trained model (overall wrong decisions in the benchmark fell from 27.3% to 25.4%, a consequence of that), not beyond it (E10). For trees and networks the adapter is averaging, and the same stability comes from more trees at equal cost (E14).
* **"The risk-averse step is why."** Not supported (E15-E17): the gain comes from exact full-data steps.
* **Small groups.** Groups of 27 to 55 repaying applicants carry wide uncertainty and the same applicant appears in several splits; label them indicative.

## 4. Checklist before using any of these sentences

1. Name the dataset and its age. 2. Name the decision rule (50% or cost cutoff). 3. Say whether the number is "standard training", "without training randomness" or "with polish".
4. Give the group's size. 5. Use "of those who repaid" (not "of all applicants") for refusal rates. 6. Add the disclaimer: statistical behaviour of a model on a public dataset; not an accusation of bias; not legal or credit advice.
