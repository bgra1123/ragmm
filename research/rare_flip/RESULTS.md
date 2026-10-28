# Do rare applicants flip more because the boundary moves more where data is thin? Results

Design: `PLAN.md` (pre-registered before any run; no deviations). Code: `rare_flip.py`, `analyze.py`; every table: [`tables.md`](tables.md); raw: `results/<dataset>.json`.
Six datasets, L2 logistic regression (λ = 0.01), 200 retrains of 2,000 minibatch steps at the aggressive step 1/L and at learning rate 0.01, plus the aggressive models after 20 exact line-search steps. Per held-out row: the flip rate across retrains against the exactly trained model, the spread (standard deviation) of the row's score across retrains, the distance to the exact boundary |margin|, k-NN rarity (mean distance to the 10 nearest training rows), rarest-category frequency, and leverage.

## The claim, and the confound it had to beat
"The boundary changes more for rare applicants, so they are more likely to get a different decision." Rare rows could flip more simply because they sit closer to the boundary. So the test holds the distance to the boundary fixed (deciles of |margin|) and compares the rarest 10% of rows with the rest inside each decile.

## 1. The boundary does move more where data is thin (H1: supported)

Spearman correlation of a row's score spread across retrains (step 1/L) with:

| | German | COMPAS | Adult | Bank | Pima | Heart |
|---|---|---|---|---|---|---|
| leverage | +0.65 | +0.90 | +0.59 | +0.79 | +0.64 | −0.01 |
| k-NN rarity | +0.90 | +0.52 | +0.73 | +0.78 | +0.78 | +0.61 |

Rule: ≥ 0.5 with leverage on ≥ 4 of 6 (5 of 6) and ≥ 0.3 with rarity on ≥ 4 of 6 (6 of 6). The same holds at learning rate 0.01 (leverage 0.64–0.92 except Heart; rarity 0.56–0.90). Heart (212 training rows, 27 features) is the one dataset where leverage does not track the spread.

## 2. At the same distance from the boundary, rare rows flip more (H2: supported on 5 of 6)

Flip rate of the rarest 10% (k-NN) against the rest, step 1/L:

| Dataset | Raw flip rate: rarest 10% / rest | Within-decile wins (rare > rest) | Pooled ratio rare / rest at equal distance [95% CI] | Same test with leverage | Same test with rarest category |
|---|---|---|---|---|---|
| German | 31.1% / 17.0% | 9 / 10 | **1.45 [1.33, 1.66]** | 1.38, 7/10 | 1.39, 9/10 |
| COMPAS | 20.7% / 21.6% | 10 / 10 | **1.39 [1.32, 1.47]** | 1.39, 10/10 | 1.20, 9/10 |
| Adult | 18.1% / 13.1% | 9 / 10 | **1.79 [1.66, 1.91]** | 1.74, 10/10 | 1.87, 9/10 |
| Bank | 22.1% / 3.8% | 10 / 10 | **1.86 [1.70, 2.02]** | 2.29, 7/10 | 2.08, 10/10 |
| Pima | 9.0% / 9.9% | 8 / 10 | **1.63 [1.45, 3.34]** | 1.71, 7/10 | (no categoricals) |
| Heart | 2.9% / 5.5% | 3 / 10 | 4.13 [0.50, 8.87] | 2.28, 4/10 | 1.70 [0.58, 8.78] |

Rule: wins ≥ 7 of 10 and pooled ratio > 1.3 with CI above 1 on ≥ 4 of 6: met on German, COMPAS, Adult, Bank and Pima. Heart has too few rows (91 held out) to say anything.

The raw column shows why the control matters: on COMPAS, Pima and Heart the rarest rows flip **less** than the rest in raw terms, because they sit farther from the boundary on average; at equal distance they flip 1.4–1.6 times more. On Bank the raw gap (22% against 4%) is mostly distance, and the part that remains at equal distance is 1.86×. The partial Spearman of flip rate with rarity given rank distance is +0.39 to +0.67 on the five datasets.

## 3. At a careful learning rate the effect is still there, but almost nothing flips (H3: direction supported)

At learning rate 0.01, 0.7–3.4% of held-out rows flip. The pooled ratio at equal distance is 2.47 [1.56, 3.99] (German), 3.04 [1.83, 4.43] (COMPAS), 2.22 [1.32, 3.24] (Adult), 1.77 [1.09, 2.88] (Bank); Pima and Heart have too few flips for an interval. The within-decile win count (1–4 of 10) is uninformative here because six to nine deciles have no flips at all. The ratio is larger than at step 1/L, the absolute number of flips far smaller: with careful training the remaining flips concentrate on rare rows next to the boundary.

## 4. After the polish the effect is gone (H4: supported)

After 20 exact line-search steps, 0.0–0.9% of rows flip (Adult 0.9%, German 0.3%, Bank 0.2%, the others 0.0%) and on Adult the rare/rest ratio is 0.88: no excess for rare rows remains. The effect is a property of training randomness, not of the rows.

## What this licenses in content
- "The line moves most where the model has seen the fewest people like you": supported (section 1).
- "At the same distance from the line, the rarest applicants were about one and a half to two times more likely to get a different answer": supported, 1.4–1.9× on five datasets (section 2). Do not say "twice" (only Bank approaches 2) and do not say it for Heart-sized data.
- "With careful training this almost disappears; after an exact polish it does": supported (sections 3 and 4).
- Not licensed: anything about why a particular person is rare, or that the model treats a group unfairly. "Rare" here is distance to the nearest training rows in feature space, and the rarest-category and leverage versions of the test agree.

## Caveats
Logistic regression only; "rarest 10%" is a threshold on k-NN distance in the standardized one-hot feature space; deciles of |margin| hold the distance fixed only approximately within a decile; rows in a dataset share the 200 retrains, so the bootstrap over rows understates the uncertainty from the seeds; German Credit is a 1990s dataset. Statements describe statistical behaviour of models on public datasets; not accusations of bias; not credit advice.
