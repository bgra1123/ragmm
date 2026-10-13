# Gate 0: is retrain-to-retrain decision instability common, or a German Credit quirk?

Written before running anything on these datasets. Gate rule is from `research/mcp_strategy/PLAN.md` §4:
**continue** to building Retrain Audit if, with textbook-style settings, mean flips ≥ 5% of held-out rows on at least
3 of 6 datasets (German Credit + 5 new); **otherwise** do not build it and publish the finding instead.

## Datasets (public, downloaded from GitHub raw; hashes in `data/README.md`)

| Dataset | Decision | Target (positive = 1) | Rows | Preprocessing |
|---|---|---|---|---|
| German Credit (existing) | approve loan | good credit | 1,000 | as Experiment 1 (`loan_experiment.load_data`) |
| COMPAS (ProPublica) | flag as likely to reoffend | `two_year_recid` | 6,172 after filter | ProPublica filter (`days_b_screening_arrest` in [-30, 30], `is_recid != -1`, `c_charge_degree != 'O'`, `score_text != 'N/A'`); features: age, priors_count, juv_fel/misd/other counts, c_charge_degree. COMPAS's own scores excluded. |
| Adult income | income > 50K | `>50K` | 48,842 | drop `fnlwgt`; `?` kept as its own category |
| Bank marketing (bank-additional-full) | call the client | `y == yes` | 41,188 | drop `duration` (known only after the call; UCI's leakage note) |
| Pima diabetes | flag for diabetes follow-up | `Outcome` | 768 | as provided (zeros not imputed) |
| Heart disease (Cleveland, 303 rows) | flag heart disease | `target` | 303 | `cp`, `restecg`, `slope`, `ca`, `thal` one-hot |

In every dataset `race` and `sex` are dropped (German Credit keeps its earlier preprocessing, which dropped its
protected columns). Categorical columns one-hot; numeric standardized on the training split; intercept added.

**Split.** Stratified, seed 0: held-out test = 30% of rows, at most 2,000 rows (German Credit: its existing 300).
Training = the rest.

## Model and training (identical to Experiment 1)

L2-regularized logistic regression, λ = 0.01, minibatch 16, K = 2,000 steps, 200 seeds (0–199), same minibatch
sequence for every method. Exact solution `w*` by Newton's method.

Methods:
* **SGD (textbook)**: α = 1/L. **Nesterov (textbook)**: α = 1/L, β = γ = (√κ−1)/(√κ+1). (as Experiment 1)
* **SGD (lr = 0.01)**: a common library default, constant step 0.01 on standardized features (secondary "default").
* **SGD + tail averaging (tuned for mean)**: as Experiment 1 (12 step sizes, 20 tuning seeds 10 000–10 019). Gate 1 preview only.
* **RA-GMM (ε = 0.25)**: as Experiment 1, for completeness.

## Metrics (held-out rows, decision threshold 0.5 for all datasets)

* **Gate statistic:** mean over seeds of *flips vs the exact model*: share of held-out rows whose decision differs from
  `w*`'s decision. Evaluated for SGD (textbook) and Nesterov (textbook); the gate counts a dataset as "unstable" if
  **either** textbook method has mean flips ≥ 5%. (German Credit at threshold 0.5 is re-run with this script.)
* **Co-primary (what a user sees):** *retrain-vs-retrain disagreement*: probability that two independent retrains
  give different decisions for a random held-out row, `mean_i 2 p_i (1 − p_i) S/(S−1)`, `p_i` = share of seeds that flag row i.
* Secondary: share of rows that are not unanimous across 200 retrains; rows with flip probability ≥ 10%; test AUC of
  `w*` and of the retrains (mean, worst 5%), so "stable but useless" is visible; same numbers for the other methods.
* If the gate statistic and the co-primary disagree on the outcome, both are reported; the gate decision uses the
  gate statistic as pre-registered.

## Expectations stated in advance

* Large-n datasets (Adult, Bank) may show *less* instability per step budget relative to their noise, or *more* because
  K = 2,000 × 16 is less than one pass over the data; I do not predict which.
* Tail-averaged tuned SGD is expected to cut flips by more than half wherever textbook flips are ≥ 5% (Gate 1 preview, not a gate here).

## What would change the plan

* ≥ 3 of 6 unstable → Gate 1 (build MVP).
* < 3 of 6 → do not build; write up the negative result; consider whether the instability is specific to small/high-stakes datasets.
