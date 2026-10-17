# Credit outcomes with and without a polish step

Design: `PLAN.md`, Addendum 2 (pre-registered). Code: `credit_with_adapter.py`; raw: `credit_with_adapter.json`. German Credit, 300 held-out applicants
(210 repaid, 90 defaulted), L2 logistic regression (λ = 0.01), 300 seeds. **Baselines (no polish)** are seven ways of training; each is then polished with
**5 epochs** of row-gradient compute (and 20 as a sensitivity) by: the **adapter** (stochastic, step and momentum from the exact EV@R model), **GD exact line search**
(deterministic full-batch), or the **risk-averse step R1 (θ = 5)**. "Exact model" = the exactly trained logistic regression (no training noise): 27 good applicants wrongly denied
and 53 defaulters wrongly approved at the 0.5 rule; 101 and 13 at the cost-based 5/6 rule (expected cost 0.553 per applicant).

## 1. Rule: approve at 50% (averages over 300 trained models)

| Trainer | | Good wrongly denied | Defaulters approved | Decisions changed vs exact | Coin-flip applicants | Errors caused by the training run |
|---|---|---|---|---|---|---|
| **Textbook SGD** | no polish | 19.6% (41.1) | 57.8% (52.0) | 55.8 | 177 | 34.5 |
| | + adapter | 13.9% (29.2) | 60.0% (54.0) | 23.2 | 71 | 13.2 |
| | + GD line search | 13.0% (27.4) | 60.0% (54.0) | 8.2 | 21 | 4.7 |
| **RA-GMM from scratch** | no polish | 15.9% (33.3) | 59.5% (53.6) | 39.9 | 128 | 23.4 |
| | + adapter | 13.5% (28.4) | 60.1% (54.1) | 17.7 | 60 | 10.1 |
| | + GD line search | 12.9% (27.1) | 59.4% (53.4) | 5.7 | 15 | 3.1 |
| **SGD + momentum 0.9** | no polish | 13.7% (28.8) | 59.6% (53.6) | 18.1 | 59 | 10.2 |
| | + adapter | 13.2% (27.8) | 59.3% (53.4) | 10.4 | 31 | 5.8 |
| | + GD line search | 12.6% (26.5) | 59.0% (53.1) | 3.1 | 9 | 1.4 |
| **Adam (lr 0.001, default)** | no polish | 13.8% (29.1) | 57.2% (51.5) | 7.1 | 22 | 3.8 |
| | + adapter | 13.7% (28.7) | 57.4% (51.6) | 6.2 | 19 | 3.2 |
| | + GD line search | 12.6% (26.6) | 59.0% (53.1) | 2.6 | 6 | 1.1 |
| **ADMM (5 iterations)** | no polish | 16.3% (34.2) | 53.3% (48.0) | 12.2 | **1** | 7.2 |
| | + adapter | 14.2% (29.8) | 57.8% (52.0) | 12.5 | **38** | 7.1 |
| | + GD line search | 12.4% (26.0) | 58.9% (53.0) | 3.0 | 0 | 1.0 |
| **Careful SGD (tuned + averaging)** | no polish | 12.8% (27.0) | 58.8% (52.9) | 5.7 | 18 | 2.8 |
| | + adapter | 12.8% (26.9) | 58.8% (52.9) | 5.4 | 15 | 2.6 |
| | + GD line search | 12.9% (27.0) | 59.0% (53.1) | 1.2 | 4 | 0.7 |

(Adam lr 0.01 and momentum lr 0.01 give the same numbers to the first decimal. The remaining 27 wrongly denied and 53 wrongly approved are the exact model's own.)

## 2. Rule: approve only at 83% (the cost-based rule, default = 5× a lost customer)

| Trainer | | Good wrongly denied | Defaulters approved | Default rate among approved | Expected cost per applicant | Coin-flip applicants |
|---|---|---|---|---|---|---|
| **Textbook SGD** | no polish | 39.4% | 31.7% (28.5) | 18.2% | 0.751 | 220 |
| | + adapter | 46.1% | 18.2% (16.4) | 12.6% | 0.595 | 91 |
| | + GD line search | 43.6% | 19.1% (17.2) | 12.7% | 0.592 | 28 |
| **RA-GMM from scratch** | no polish | 44.0% | 23.4% (21.0) | 15.1% | 0.659 | 159 |
| | + adapter | 47.1% | 16.6% (14.9) | 11.8% | 0.578 | 73 |
| | + GD line search | 46.2% | 15.7% (14.1) | 11.1% | 0.558 | 20 |
| **SGD + momentum 0.9** | no polish | 48.0% | 16.2% (14.6) | 11.7% | 0.579 | 73 |
| | + adapter | 47.7% | 15.3% (13.8) | 11.2% | 0.564 | 50 |
| | + GD line search | 47.6% | 14.5% (13.0) | 10.6% | 0.550 | 5 |
| **Adam (lr 0.001, default)** | no polish | 56.2% | 9.7% (8.7) | 8.6% | 0.539 | 26 |
| | + adapter | 55.7% | 10.0% (9.0) | 8.8% | 0.541 | 24 |
| | + GD line search | 48.6% | 14.4% (13.0) | 10.7% | 0.557 | 0 |
| **ADMM (5 iterations)** | no polish | 79.7% | 1.1% (1.0) | 2.3% | 0.575 | 1 |
| | + adapter | 59.0% | 8.4% (7.5) | 8.0% | 0.538 | 38 |
| | + GD line search | 48.6% | 14.4% (13.0) | 10.7% | 0.557 | 0 |
| **Careful SGD** | no polish | 48.7% | 13.9% (12.5) | 10.4% | 0.549 | 23 |
| | + adapter | 48.6% | 13.9% (12.5) | 10.4% | 0.549 | 22 |
| | + GD line search | 48.2% | 14.4% (13.0) | 10.7% | 0.554 | 2 |

Exact model: 48.1% wrongly denied, 14.4% of defaulters approved, 10.7% default rate among approved, cost 0.553.

## 3. What this says

* **For the worst baselines (textbook SGD, RA-GMM from scratch), the adapter moves outcomes most of the way to the exact model.** Textbook SGD at the 83% rule: defaulters
  approved fall from 28.5 to 16.4 per model, the default rate among approved loans from 18.2% to 12.6%, expected cost from 0.751 to 0.595 (the exact model: 0.553); at the 50% rule,
  good applicants wrongly denied fall from 19.6% to 13.9% (exact: 12.9%), coin-flip applicants from 177 to 71, errors caused by the training run from 34.5 to 13.2 (−62%).
  The paired change in wrongly denied is −11.9 people per model (95% interval −13.0 to −10.9).
* **For baselines that were already close (careful SGD; Adam at its default lr), the adapter changes little** (differences within noise at 0.5, e.g. 12.8% vs 12.8% wrongly denied).
* **The adapter can add noise to a deterministic trainer.** Early-stopped ADMM varies only with its random partition (1 coin-flip applicant); polishing it with stochastic steps raised
  that to 38. A deterministic polish (full-batch steps) left it at 0.
* **The deterministic polish does better at equal compute.** Five full-batch steps with a line search bring every baseline close to the exact model's behaviour
  (mean decisions changed 1.2–8.2 vs the adapter's 5.4–23.2; coin-flip applicants 0–21 vs 15–71). At 20 epochs, line search leaves ≤ 0.9 changed decisions and ≤ 3 coin-flip applicants for every
  baseline; the adapter 4.9–14.2 and 14–46. The risk-averse step R1 (θ = 5) falls between the adapter and the line search at 5 epochs.
* **Polishing does not make decisions better than the exact model, only reproducible.** At the 50% rule the polished models approve about 54 of the 90 defaulters (exact: 53), slightly
  *more* than the noisy ones for several baselines (textbook 52.0 → 54.0). The remaining errors (27 wrongly denied, 53 wrongly approved) belong to the model, not the training.
* **Lower expected cost is not the same as closer to the exact model.** Adam at its default lr and early-stopped ADMM had a *lower* expected cost than the exact model (0.539 and 0.575 → 0.538 after
  adapter) because they were biased toward refusing more applicants, which pays under a 5:1 default cost on this test set. Polishing to the exact solution removes that bias (0.539 → 0.557 with line search).
  Under-training acts like extra regularization here; whether that is wanted is a modelling choice, not a training-noise question.

## Hypotheses

| | Result |
|---|---|
| **H1** adapter moves wrongly denied (0.5) and cost (5/6) toward the exact model for every trainer; ≥ 50% fewer training-induced errors for the worst | **Supported** for direction (7/7); the ≥ 50% cut holds for textbook SGD (−62%) but not momentum or Adam lr 0.01 (−43%). |
| **H2** ≥ 50% fewer decisions changed for trainers changing ≥ 10 | **2 of 5** (textbook −58%, RA-GMM −56%); momentum −43%, Adam −43%, ADMM 0% at the 0.5 rule. |
| **H3** ≥ 50% fewer coin-flip applicants | **2 of 5** (textbook −60%, RA-GMM −53%); ADMM rose from 1 to 38. |
| **H4** GD line search at 5 epochs at least as close to exact as the adapter, for every trainer | **Supported 7/7** (decisions changed 1.2–8.2 vs 5.4–23.2). |
| **H5** remaining errors equal the exact model's | **Supported** (27 wrongly denied, 53 wrongly approved at 0.5). |
| **H6** defaulters approved at 0.5 may rise | **Observed**, small (textbook 52.0 → 54.0; ADMM 48.0 → 52.0). |

## Caveats

* One dataset, one 300-applicant held-out split; differences of under about one person per model in the counts are within noise. The 5/6-rule cost depends on this split.
* Polishing budgets are in epochs of row gradients (5 or 20); the adapter parameters were chosen once per trainer and budget from 6 pilot runs.
* The "errors caused by the training run" are errors the polished or noisy model makes where the exact model does not (summed over both error types).
* German Credit's labels are 1990s data; the numbers describe training behaviour, not lending policy. The report's disclaimer applies.
