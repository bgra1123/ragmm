# Concrete statements: the chance of a wrong decision, by type of applicant

Design: `PLAN.md` (addendum, pre-registered). Code: `concrete_statements.py`; raw: `results/concrete.json`. German Credit, five train/test splits (300 held-out applicants each, 1,500 cases pooled; the same applicant can appear in several splits), 100 retrains per split.
"Wrong decision" = the model denies an applicant who repaid or approves one who defaulted (the dataset's own label). "Standard training" = textbook SGD; "adapter" = 5 more epochs of the polish step; "model without training randomness" = the exactly trained model.
Not an allegation of bias: sex, age and foreign-worker status are never model inputs and are used only to describe where wrong decisions occur.

## At the 50% rule (the natural place to count wrong decisions)

| Applicants who... | Held-out cases | Chance of a wrong decision: standard training [95% interval] | with the adapter | model without training randomness | Part due to training randomness: standard → with adapter | Chance a defaulter is approved at the cost-based cutoff: standard → adapter (no randomness) |
|---|---|---|---|---|---|---|
| are currently unemployed | 102 | 42.9% [42.0, 43.8] | 40.1% | 37.3% | +5.6 → +2.8 points | 31.1% → 13.8% (10.0%) |
| have a job classified unemployed/unskilled non-resident | 36 (indicative) | 34.4% [32.6, 36.3] | 26.1% | 19.4% | +14.9 → +6.6 points | 30.4% → 15.1% (11.1%) |
| live in housing "for free" | 157 | 37.8% [37.1, 38.5] | 31.5% | 27.4% | +10.4 → +4.1 points | 22.6% → 9.9% (6.7%) |
| are under 25 | 226 | 40.0% [39.5, 40.5] | 38.5% | 38.5% | +1.5 → -0.0 points | 28.7% → 13.6% (7.8%) |
| are not foreign workers | 54 (indicative) | 25.2% [24.1, 26.3] | 19.1% | 14.8% | +10.4 → +4.2 points | 28.2% → 7.3% (0.0%) |
| are women | 466 | 31.2% [30.8, 31.6] | 26.9% | 26.0% | +5.2 → +1.0 points | 28.3% → 12.6% (7.7%) |
| have a negative checking balance | 435 | 40.0% [39.7, 40.5] | 34.1% | 31.3% | +8.8 → +2.8 points | 16.4% → 4.8% (2.8%) |
| ask for a large (>4,000 DM) and long (>24 months) loan | 230 | 38.7% [38.3, 39.2] | 34.1% | 31.7% | +7.0 → +2.4 points | 21.4% → 8.1% (4.0%) |
| have no property | 219 | 35.4% [34.9, 36.0] | 29.2% | 25.6% | +9.9 → +3.6 points | 19.0% → 6.9% (4.3%) |
| are divorced or separated men | 74 | 27.0% [26.1, 27.9] | 19.5% | 14.9% | +12.2 → +4.6 points | 14.4% → 0.5% (0.0%) |
| all applicants | 1500 | 30.4% [30.1, 30.6] | 25.8% | 24.4% | +6.0 → +1.4 points | 26.6% → 13.1% (9.1%) |

How to read a row: *"Among applicants who live in housing 'for free' (157 held-out cases), a model trained the standard way makes the wrong decision 37.8% of the time. A model without any training randomness would make
it 27.4% of the time, so 10.4 percentage points are caused by the randomness of training. With the polish step the figure is 31.5% (4.1 points left)."*

Examples in plain sentences (all numbers from the table):
* Applicants with **no property**: 35.4% chance of a wrong decision with standard training, 25.6% without training randomness, 29.2% with the adapter.
* Applicants with a **negative checking balance**: 40.0% (standard), 31.3% (no randomness), 34.1% (adapter).
* Applicants who are **currently unemployed**: 42.9% (standard), 37.3% (no randomness), 40.1% (adapter). Their chance of a wrong decision is high even for the model without randomness; training randomness adds 5.6 points.
* **All applicants**: 30.4% (standard), 24.4% (no randomness), 25.8% (adapter).
* The chance that training randomness **changes the decision** of a given applicant (retrained model vs the exactly trained model): 19.6% overall (28.6% for unemployed, 26.2% for housing for free, 28.3% for negative checking balance); with the adapter 7.5% overall (12.5%, 11.3%, 12.5%).

## At the cost-based cutoff (approve only at 83%): a defaulter being approved

At this cutoff a model refuses many good applicants on purpose, so the overall "wrong decision" rate is not the right yardstick (standard training 37.7% vs 40.8% for the model without randomness). The costly error is approving someone who then defaults. For defaulters:
the chance of being approved is 26.6% with standard training, **9.1% without training randomness**, 13.1% with the adapter. By type: housing for free 22.6% → 9.9% (6.7% without randomness); negative checking balance 16.4% → 4.8% (2.8%);
no property 19.0% → 6.9% (4.3%); large and long loans 21.4% → 8.1% (4.0%); unemployed 31.1% → 13.8% (10.0%) (last column of the table).

## Hypotheses

| | Result |
|---|---|
| **H1** chance of a wrong decision higher for the unusual-profile groups, and the training-randomness part larger | **Supported for the level (6/6 groups above the 30.4% average) and for the randomness part in 5 of 6** (G2 +14.9, G3 +10.4, G9 +9.9, G7 +8.8, G8 +7.0 vs +6.0 overall; G1 +5.6). |
| **H2** adapter cuts the training-randomness part by ≥ 40% in every group | **Supported 10/10** (reductions 50–100%). |
| **H3** the part left is under 5 points for every group of ≥ 60 cases | **Supported** (largest: housing for free +4.1, divorced/separated men +4.6 (74 cases)); the group with 36 cases (job unemployed/unskilled non-resident) keeps +6.6. |

## Caveats

* One dataset; five splits that re-use the same 1,000 applicants; small groups (36–466 cases); intervals reflect training randomness only, not the sampling of applicants.
* "Wrong decision" counts a refusal of a good applicant and an approval of a defaulter equally; lenders weigh them differently (the cost-based cutoff above does).
* The model without training randomness still makes many wrong decisions for these groups (for example 37.3% for unemployed applicants): the polish removes the part caused by randomness, not the model's own error pattern.
