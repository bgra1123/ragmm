# Where does training noise land, and does the adapter help those applicants?

Design: `PLAN.md` (pre-registered, including the rule that picked the showcase group and the example applicants). Code: `group_experiment.py`; raw: `results/groups.json`.
German Credit, five different train/test splits (300 held-out applicants each), 100 retrains per split of two baselines **without** an adapter (textbook SGD, Adam lr 0.01), each then polished with
5 epochs by the **adapter** or by a deterministic **line search**. Decisions at the 50% rule. "Flip" = a retrain's decision differs from the exactly trained model's decision for that applicant.

> **Disclaimer.** These groups are used only to **describe where training noise lands**. Sex, age and foreign-worker status were never inputs to any model. Nothing here says that a lender, a dataset or a method
> discriminates; it shows that the same applicant can get different answers depending on the training run, and that this variability is concentrated on some profiles. Group sizes are small: a member-instance is an
> applicant held out in one split (the same applicant can be held out in several splits), and uncertainty from applicant sampling is not included.

## 1. Training noise is concentrated on specific groups, and the adapter reduces it for each of them

| Group | Member-instances (good) | Flips vs exact: SGD | + adapter | + line search | Adam | Adam + adapter | Coin-flip share: SGD → + adapter |
|---|---|---|---|---|---|---|---|
| G1 unemployed now | 102 (62) | 28.6% | 12.9% | 5.0% | 11.6% | 6.9% | 0.84 → 0.40 |
| G2 job: unemployed/unskilled non-resident | 36 (27) | 25.6% | 9.8% | 4.4% | 9.6% | 6.1% | 0.83 → 0.25 |
| **G3 housing for free (showcase)** | 157 (97) | 26.2% | 11.6% | 4.9% | 10.6% | 6.1% | 0.81 → 0.39 |
| G4 age under 25 | 226 (136) | 22.8% | 8.9% | 3.4% | 8.2% | 4.9% | 0.74 → 0.31 |
| G5 not a foreign worker | 54 (48) | 18.9% | 8.1% | 2.4% | 7.6% | 4.3% | 0.52 → 0.28 |
| G6 female | 466 (324) | 19.9% | 7.7% | 2.9% | 6.9% | 3.9% | 0.62 → 0.24 |
| G7 checking account < 0 DM | 435 (223) | 28.3% | 12.5% | 5.1% | 11.5% | 6.6% | 0.89 → 0.41 |
| G8 large and long loan (>4,000 DM, >24 months) | 230 (129) | 24.0% | 8.5% | 3.3% | 8.0% | 4.1% | 0.77 → 0.29 |
| G9 no property | 219 (127) | 24.5% | 10.4% | 4.7% | 9.2% | 5.4% | 0.77 → 0.33 |
| G10 male divorced/separated | 74 (55) | 21.1% | 7.4% | 3.2% | 7.3% | 3.8% | 0.62 → 0.23 |
| Applicants in none of these groups | 403 (327) | 12.5% | 3.9% | 1.3% | 3.4% | 2.1% | 0.41 → 0.12 |

Textbook SGD flips 19–29% of decisions for these groups against 12.5% for applicants outside all of them (ratios 1.5–2.3). The adapter lowers the flip rate in **all ten groups**
(for example G3 housing for free: 26.2% → 11.6%; G1 unemployed: 28.6% → 12.9%; G7 checking account below 0: 28.3% → 12.5%); the deterministic line search lowers it more (G3: 4.9%).
Bootstrap 95% intervals for the SGD flip rates are narrow (about ±1 point, training noise only).

## 2. The wrong answers: good applicants refused

| Group | Good applicants wrongly denied: SGD | + adapter | + line search | **Exact model** | Excess over exact: SGD → + adapter |
|---|---|---|---|---|---|
| G1 unemployed now | 36.0% | 32.4% | 29.9% | 29.0% | +7.0 → +3.4 points |
| G2 job: unemployed/unskilled non-resident | 31.6% | 21.7% | 16.4% | 11.1% | +20.5 → +10.6 points |
| **G3 housing for free (showcase)** | 34.6% | 28.4% | 25.2% | 24.7% | +9.9 → +3.7 points |
| G4 age under 25 | 28.8% | 24.6% | 25.5% | 25.7% | +3.1 → -1.1 points |
| G5 not a foreign worker | 21.5% | 13.9% | 13.1% | 12.5% | +9.0 → +1.4 points |
| G6 female | 21.8% | 14.7% | 13.4% | 12.7% | +9.1 → +2.0 points |
| G7 checking account < 0 DM | 40.1% | 34.7% | 32.8% | 34.1% | +6.0 → +0.6 points |
| G8 large and long loan (>4,000 DM, >24 months) | 35.6% | 29.5% | 27.7% | 26.4% | +9.2 → +3.1 points |
| G9 no property | 33.1% | 27.1% | 24.5% | 23.6% | +9.5 → +3.4 points |
| G10 male divorced/separated | 22.7% | 13.8% | 11.7% | 10.9% | +11.8 → +2.9 points |
| Applicants in none of these groups | 11.7% | 5.0% | 4.1% | 4.3% | +7.4 → +0.8 points |

* **Group X = housing "for free" (the showcase group, chosen by the pre-registered rule: largest excess of wrongly denied among groups with ≥ 150 member-instances).** With textbook SGD, **34.6% of these applicants who
  repaid were refused**; the exactly trained model refuses 24.7%, so **9.9 percentage points are caused by the training run**. With the adapter, 28.4% are refused (excess 3.7 points); with the line search, 25.2%.
  The coin-flip share (decision changing in ≥ 10% of retrains) falls from 0.81 to 0.39.
* The same holds for most groups: excess wrongly-denied falls from +7.0 to +3.4 (unemployed), +9.2 to +3.1 (large long loans), +9.5 to +3.4 (no property), +9.1 to +2.0 (female), +6.0 to +0.6 (checking < 0);
  G4 (age under 25) goes from +3.1 to −1.1.
* **What the adapter does not remove:** the exactly trained model itself refuses 24.7% of the repaying "housing for free" applicants against 4.3% outside the listed groups, and 34.1% of those with a negative
  checking balance. The adapter removes the extra errors that come from the training run; it does not remove a model's own error pattern, which needs different features, data or modelling.
  (The comparison group is, by construction, the typical applicant, so the gap is not a like-for-like measurement.)

## 3. Examples: applicants in group X who repaid and whom the exact model approves (split 0; the three with the highest chance of being refused by textbook SGD)

| # | Applicant (all three repaid; the exact model approves) | SGD | + adapter | + line search | Adam | Adam + adapter |
|---|---|---|---|---|---|---|
| 1 | car (new), 1175 DM over 16 months; checking account: 0 <= ... < 200 DM; savings: ... < 100 DM; employment: unemployed; job: unemployed/ unskilled - non-resident; housing: for free; property: car or other, not in attribute Savings account/bonds; exact model's probability of repaying 0.50 | 47% | 52% | 59% | 41% | 49% |
| 2 | car (new), 1480 DM over 12 months; checking account: ... >= 200 DM / salary assignments for at least 1 year; savings: 500 <= ... < 1000 DM; employment: unemployed; job: unemployed/ unskilled - non-resident; housing: for free; property: unknown / no property; exact model's probability of repaying 0.57 | 56% | 74% | 90% | 63% | 74% |
| 3 | car (new), 3757 DM over 24 months; checking account: no checking account; savings: ... < 100 DM; employment: ... >= 7 years; job: skilled employee / official; housing: for free; property: unknown / no property; exact model's probability of repaying 0.62 | 57% | 81% | 100% | 85% | 95% |

Approval share = the share of the 100 retrains that approve the applicant. Applicants 2 and 3 are approved by only 56–57% of textbook SGD retrains and by 74–81% (adapter) or 90–100% (line search); the exact
model approves them. **Applicant 1 is a different case**: the exact model's probability of repaying is 0.50 (a toss-up even without training noise), and the adapter barely changes the share (47% → 52%): this is
a borderline application, not a training-noise victim.

## Hypotheses

| | Result |
|---|---|
| **H1** flips more frequent in the unusual-profile groups (ratio ≥ 1.3 for ≥ 4 of 6) | **Supported 6/6** (ratios 1.9–2.3). |
| **H2** adapter cuts flip rate and excess wrongly-denied in the showcase group by ≥ 40% | **Supported** (flips −56%: 26.2% → 11.6%; excess −63%: +9.9 → +3.7 points). |
| **H3** adapter reduces flips in ≥ 8 of 10 groups | **Supported 10/10** (reductions 55–65%); the reduction in the *excess* varies from +0.6 to +10.6 points left, largest remaining excess for the smallest group (G2, 36 member-instances, +10.6). |
| **H4** line search cuts flips at least as much as the adapter in every group | **Supported 10/10** (e.g. G3: 4.9% vs 11.6%). |
| **H5** showcase group's wrongly-denied rate approaches the exact model's, not below | **Supported** (adapter 28.4%, line search 25.2%, exact 24.7%). |

## Caveats

* One dataset; five splits that re-use the same 1,000 applicants; small groups (36–466 member-instances); intervals reflect training noise only. The chosen showcase group depends on the rule and on these splits.
* Group labels are post-hoc descriptions of the data. The showcase group was selected by the declared rule among 10 pre-specified groups; the table shows all of them.
* Polishing budget 5 epochs; the adapter parameters were chosen once per (split, trainer) from 6 pilots.
