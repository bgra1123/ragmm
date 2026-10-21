# Does the risk adjustment do anything? Isolating it from "one more round of exact learning"

Status: PRE-REGISTRATION, written before any run. Results go to `RESULTS.md`. Branch: `claude/ragmm-product-content-s3k2po`.

## 1. What is being tested

The polish adapter has two ingredients:
- **I1, exact full-batch steps** (no minibatch noise). Already shown to remove most seed variation (`research/product`, `research/linesearch`).
- **I2, a risk-averse rule for the step length** (the entropic line search: the entropic risk measure of Can & Gürbüzbalaban 2025 applied to the per-row loss change of a step).

Claim to test: **I2 changes outcomes beyond I1.** Evidence so far suggests that with enough full-batch steps on a strongly convex problem, both reach the same optimum, so I2 can only matter when the budget is short. The experiments below test that.

## 2. Primary comparison (revised): risk-adjusted step vs one more epoch of the same training

Correction after review: the comparison that matters to a user is not "which step rule inside the adapter" but
**"for the same extra compute, is the risk-adjusted adapter better than simply training one more epoch?"** An earlier draft of
this plan compared step rules against each other; that is now a secondary attribution analysis (section 3).

All arms start from the same 200 stage-1 models per dataset (stage 1 = 2,000 minibatch steps of SGD, or of Adam; L2 logistic
regression, lambda = 0.01, batch 16, as in `research/linesearch` and `research/optimizers`) and get the **same extra budget of
B epochs of row-gradients** (B in {1, 2, 5, 20}; one epoch = n row-gradients):

| Arm | What it does with the extra budget | Role |
|---|---|---|
| C0 | nothing | reference |
| **C1 continue** | the same optimizer, same settings, B more epochs (fresh minibatch order) | the "one more epoch" baseline |
| **C2 continue, decayed** | same optimizer, step size decayed linearly to 0 over the B epochs | the best classical use of an extra epoch |
| **C3 continue, averaged** | same optimizer, average the iterates of the extra epochs (Polyak-Ruppert) | strongest standard noise reducer |
| **R risk-adjusted adapter** | B full-batch steps, each step length from the entropic rule (first-order R1 theta = 1 as the original proposal, and `entropic_exact` theta = 5) | the proposal |
| **L exact line search** | B full-batch steps with exact line search | I1 only: exact gradient without risk |

Compute accounting (fixed in advance): one full-batch step = 1 epoch of row-gradients. The step-length search needs only
matrix-vector products with the already computed gradient (`X g`), so it is counted as 0 extra epochs, but wall-clock seconds are
reported for every arm so the reader can judge.

Outcomes (held-out rows unless noted): O1 chance of a wrong decision, overall, per pre-registered group, and worst group;
O2 seed disagreement (two retrains differ); O3 decisions changed vs the exactly trained model (mean and worst 5% of retrains);
O4 gap to the optimum f(w) - f*, mean and CVaR5% (train); O5 per-row harm (share of rows worse; 95th percentile of per-row loss increase).

**Primary question and decision rule (fixed before running).** R beats the best of C1-C3 on O2 and O3 at budgets B = 1 and 2,
CI excluding 0 (paired bootstrap over stage-1 seeds), on at least 4 of 5 datasets, and not worse on O1 overall by more than
1 point. Same for R vs C1 alone as the weaker, plainly stated "one more epoch" comparison.

**What earlier results already suggest (so the plan is honest about what is at stake).** In `research/linesearch/RESULTS.md`
the stochastic polisher (more minibatch steps) beat the line searches at 1 epoch on German and COMPAS; the line searches won
at 5-20 epochs. We therefore expect "continue training" to be competitive at B = 1, and R to win from about B = 5. The test
reports the budget at which R overtakes continuation, or that it never does.

## 3. Secondary: attributing a win (only run if section 2 shows R > continuation)

If R beats continuation, which ingredient is responsible?
- L (exact full-batch line search) vs C1-C3: the effect of **exact full-data gradients**.
- R vs L: the effect of **the risk adjustment**, the question that matters for the claim.
- R vs a **step-matched** line search (L with its step shrunk so its mean step equals R's): separates "risk-weighted step" from
  "smaller step". Along one fixed direction the step length alone decides the weights, so R can only beat this control if its
  step is smaller in the right places.
- Dose-response: R at theta in {0.5, 1, 2, 5, 10}; protection (O5) should grow with theta.
- Note: only the exact-change variant tends to the ordinary line search as theta -> 0; the first-order variant is linear in
  the step length at small theta and has no minimum, so it is not used as the control.

Rule: the risk adjustment is credited only if R beats both L and the step-matched control on O5 and on the worst-group
wrong-decision rate, CI excluding 0, on at least 3 of 5 datasets. Otherwise the gain is attributed to exact gradients.

## 4. (removed)

An earlier draft of this plan included a tilted training objective (tilted ERM). It is a different method from the step-length rule under study and was removed from scope.

## 5. Experiment C: where the risk step could matter in practice (large n, few passes)

- Adult (n = 48,842, subsample sizes 5k, 20k, full), budgets of 0.25, 0.5, 1, 2 passes (partial passes use a random subset of rows for the gradient, so these arms are stochastic and must be labelled; they are the regime a practitioner with a big model is in).
- Same arms A2, A3, A4. Report O1-O4 and wall-clock.
- Purpose: find the budget below which A3 beats A4, or show there is none.

## 6. Analysis plan and guards
- Paired differences over seeds with percentile bootstrap (5,000 resamples); multiple outcomes are reported, only the primary list above decides.
- Every table lists the mean step length x L, so a reader can see when two arms differ only in step size.
- No tuning of theta on test data: theta grid fixed above; the reported theta is the pre-registered 5 (primary) and the dose-response curve.
- Negative results are published as negative.
- Statements describe statistical behaviour of models on public datasets; not accusations of bias; not credit advice.

## 7. What would change the adapter
- Section 2 supported (R beats continuation): keep `step="entropic_exact"` as the recommended short-budget rule and document the budget range where it wins.
- Neither supported (continuing training is as good): say so in the tutorial and videos: the stability gain is exact full-data steps on a convex problem; the risk-averse step is a safe way to take few steps, not an extra source of gain.

## 8. Implementation notes
- Reuse `research/linesearch/linesearch_experiment.py` (steps R1, R3, exact LS, golden-section search) and `risk_averse_tuning.adapter.polish_logistic`.
- New code: `risk_effect_experiment.py` (arms A0-A5, step matching for A4, outcomes O1-O4), `partial_pass.py` (C). Run detached with `nohup setsid`, OMP_NUM_THREADS=1, one process per dataset; logs gitignored.
- Order of work: A on German (smoke test), A on all datasets, C. Record deviations in RESULTS.md before reading any result.

## 9. Addendum before running section 3 (step-matched controls and theta sweep)

Fixed in advance, code `stepmatch_experiment.py`:
- Arms: L (exact line search); R1(theta) and R3(theta) for theta in {0.5, 1, 2, 5, 10}; **M1** = line search with every step multiplied by one constant c = mean step of the R arm / mean step of L (per dataset, budget, theta);
  **M2** = replay of the R arm's mean step length of each iteration as one fixed schedule for all seeds. Budgets 1, 2, 5 steps (20 omitted: all arms have converged and the risk term cannot matter), 200 stage-1 models from SGD and from Adam, four datasets.
- The step search uses a vectorized safeguarded Newton solver (agrees with the grid + golden-section search of `research/linesearch` to 5e-7 in the step length).
- Primary outcomes (paired bootstrap over seeds, 1,000 resamples): the 95th percentile of the per-row loss increase on held-out rows (O5), the worst-group wrong-decision rate (German only, groups with >= 30 held-out rows), disagreement, overall wrong-decision rate.
- **Credit rule (section 3):** the risk adjustment is credited if R has a lower O5 than **both** L and M1 and a lower or equal worst-group wrong-decision rate, CI excluding 0, on at least 3 of 4 datasets for some theta, with dose-response (O5 falls as theta grows). Otherwise "the risk-weighted step is equivalent to a smaller step".

## 10. Addendum before running the negative-flip test

Question: does the risk-averse step reduce **negative flips** (held-out rows the stage-1 model decided correctly that the update then decides wrongly), compared with the matched controls and with continued training?
Definitions (held-out rows, decision at 0.5): negative flip = correct before, wrong after; positive flip = wrong before, correct after; net = positive minus negative (the accuracy change); churn = decision differs before/after.
Rates are per held-out row, averaged over the 200 stage-1 models. Arms and budgets as in section 9 (L, R1 and R3 at theta in {0.5, 1, 2, 5, 10}, matched controls M1 and M2, steps 1, 2, 5) plus the continuation arms C1 (continue), C2 (decayed) and C3 (averaged) of section 2 at the same epochs.
German also reports the worst-group negative-flip rate (groups with at least 30 held-out rows).
Hypotheses: H1 R has fewer negative flips than L and than the best continuation. H2 R has fewer negative flips than M1 and M2. H3 R's net accuracy change is not worse than M1's by more than 0.5 points.
**Credit rule:** the risk adjustment is credited for negative flips if, for some theta and for both 2 and 5 steps, R has a lower negative-flip rate than both M1 and M2 (paired bootstrap CI below 0) on at least 3 of 4 datasets, with H3 satisfied. Otherwise the negative-flip reduction is attributed to step length.
Code: `negflip_experiment.py`.

## 11. Addendum before running Experiment C (large n, few passes)

Question: where a pass over the data is expensive and only a fraction of a pass to two passes is affordable, does the risk-averse step beat continued training, and does it beat the matched plain step (M1, M2)?
Setting (fixed in advance; code `large_data_experiment.py`):
- Data: Adult income (train 46,842 rows after the 2,000-row test split; test = 2,000 held-out rows). Training sizes n in {5,000, 20,000, 46,842} (random subsample of the training rows, fixed seed). 100 stage-1 models per size (fewer than before because every step touches n rows), from SGD (step 1/L) and Adam (lr 0.01), 2,000 minibatch steps of 16 (so stage 1 itself sees only 0.7 passes at the full size).
- Extra budget B in {0.25, 0.5, 1, 2} passes = B n row-gradients. Continuation arms (C1 continue, C2 decayed, C3 averaged) use round(B n / 16) minibatch steps. Full-batch arms: for B >= 1, B full-batch steps; for B < 1, **one step whose gradient and step search use a random subset of B n rows, drawn independently per model** (these arms are stochastic and labelled so; all rules share the same subset for a given model).
- Arms: L (exact line search); R1 and R3 at theta in {1, 5}; M1 (L with its step scaled to R's mean step) and M2 (R's per-iteration mean step as one schedule) for R1(5) and R3(5).
- Outcomes (held-out): two retrains disagree, decisions changed against the exactly trained model of that training size (mean over models), wrong-decision rate, 95th-percentile per-row loss increase, negative-flip and positive-flip rates relative to the stage-1 model. Paired bootstrap over models (1,000 resamples).
- Questions answered: (1) the smallest budget at which the best risk step has lower disagreement than the best continuation, per size and start (or "never"); (2) whether the risk step has lower harm and negative flips than both L and M1 at the budgets below 1 pass.
- Credit rule (for the risk term): lower harm than both L and M1 with CI below 0 at B in {0.25, 0.5, 1}, for the same theta, on at least 2 of the 3 sizes, for both starts. Otherwise the gain is attributed to the exact gradient and the shorter step.
- Deviations from section 5: 100 models, not 200; one dataset; wall-clock is taken from `cost_benchmark.py` (the step search is vectorized over models here, so per-arm seconds would not be comparable).
