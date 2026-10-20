# Does the risk adjustment do anything? Isolating it from "one more round of exact learning"

Status: PRE-REGISTRATION, written before any run. Results go to `RESULTS.md`. Branch: `claude/ragmm-product-content-s3k2po`.

## 1. What is being tested

The polish adapter has two ingredients:
- **I1, exact full-batch steps** (no minibatch noise). Already shown to remove most seed variation (`research/product`, `research/linesearch`).
- **I2, a risk-averse rule for the step length or the objective** (entropic risk, Can & Gürbüzbalaban 2025; tilted ERM, Li et al. 2021).

Claim to test: **I2 changes outcomes beyond I1.** Evidence so far suggests that with enough full-batch steps on a strongly convex problem, both reach the same optimum, so I2 can only matter when (a) the budget is short, or (b) the objective itself is tilted. Two experiments follow, one for each route.

## 2. Control ladder (same start, same direction, same number of passes)

All arms start from the same 200 stage-1 SGD models per dataset (2,000 textbook SGD steps, L2 logistic regression, lambda = 0.01, as in `research/linesearch`) and take T full-batch steps along the exact gradient g. They differ only in the step length alpha_t.

| Arm | Step length | Role |
|---|---|---|
| A0 | none (stage 1) | reference |
| A1 | fixed 1/L | no line search at all |
| A2 | exact line search | I1 only |
| A3(theta) | **entropic_exact** (risk of the exact per-row loss change), theta in {0.5, 1, 2, 5, 10} | I1 + I2 |
| A4(theta) | **step-matched control**: exact line-search direction, step = c * alpha_A2, with c chosen per (dataset, budget, theta) so that the mean step length equals that of A3(theta) | I1 + "smaller step" without risk |
| A5 | entropic first-order (R1, theta = 1, 5) | variant, reported as secondary |

Notes fixed in advance:
- **theta -> 0 limit.** Only the exact-change variant (A3) tends to the ordinary line search as theta -> 0. The first-order variant (R1) is linear in alpha as theta -> 0 and has no minimum, so it is not the control. (This corrects an earlier remark.)
- **A4 is the decisive control.** The risk rule mostly takes smaller steps. Along one fixed direction the step length alone decides the new weights, so the risk rule can only beat A4 if its step is smaller in the right places: different per seed, per budget, per state.
- Compute: A3 needs per-row loss changes (n extra row evaluations per grid point). Report seconds per fit; A2 and A4 are matched in passes, not in seconds.

## 3. Experiment A: short budget (does the risk step protect rows and groups?)

- Budgets T in {1, 2, 5, 20} passes.
- Datasets: German, COMPAS, Pima, Heart (as in `research/linesearch`) + Adult subsample (n = 20,000) where one pass is costly. 200 stage-1 seeds each. Stratified 70/30 splits; 3 splits for the groups analysis (German, COMPAS: the pre-registered groups of `research/groups`, groups with at least 30 held-out rows).
- Outcomes (all on **held-out** rows unless said):
  - O1 per-row harm: share of rows whose loss increases from A0 to the end, 95th percentile of the per-row loss increase, maximum increase (train and held-out).
  - O2 worst-5% objective gap CVaR5%[f(w) - f*] (train).
  - O3 chance of a wrong decision: overall, per group, and the worst group.
  - O4 seed disagreement (two retrains differ on a row).
- Primary comparisons (paired by stage-1 seed, bootstrap 95% CI over seeds): **A3(theta=5) vs A4(theta=5)** on O1 (p95 held-out per-row loss increase) at T = 1 and 2, and on the worst-group wrong-decision rate. Everything else is secondary.
- Dose-response: for each outcome the effect of A3 vs A2 should be monotone in theta. Report the curve; non-monotone curves are reported as such.

### Decision rules (stated before running)
- **Supported:** A3 beats A4 on the primary p95 harm outcome, CI excluding 0, on at least 4 of 5 datasets at T = 1 and 2, and the dose-response is monotone on at least 4 of 5.
- **Not supported:** otherwise. The conclusion then is: *the risk-averse step is equivalent to a smaller step*, and the adapter should be described as exact steps (I1) with the step rule as a convenience.
- **Group claim:** only if the worst-group wrong-decision rate improves with CI excluding 0 on at least 2 datasets, and overall accuracy is not worse by more than 1 point.

## 4. Experiment B: tilted objective (does risk change the final model?)

Here the objective itself changes, so the optimum moves and "one more round of learning" cannot reproduce it by running longer.

- Tilted loss: F_theta(w) = (1/theta) log mean_i exp(theta * l_i(w)) + (lambda/2)||w||^2, theta in {0, 0.5, 1, 2, 5}; theta = 0 is ordinary ERM. Convex, solved by Newton with the polish; no minibatch noise.
- Controls with the same budget of fitted models:
  - ordinary ERM (theta = 0);
  - **reweighting control:** ERM with class weights tuned to match the tilted model's overall accuracy (so any group effect is not just a shifted operating point);
  - **per-group worst-case control:** group-DRO-style reweighting over the pre-registered groups (known upper bound for what is possible with group labels).
- Outcomes: overall accuracy and AUC, worst-group wrong-decision rate and the five largest groups, calibration slope, seed disagreement (should be near 0 for all, since all are exact convex solves).
- **Supported** if at matched overall accuracy (within 0.5 points) the tilted model has a lower worst-group wrong-decision rate than the reweighting control, CI excluding 0, on at least 2 datasets. Otherwise: tilting is equivalent to reweighting and gets no separate claim.
- Caveat to report: a tilted objective fits the hardest rows harder and can trade accuracy for them; label the trade-off, do not hide it.

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
- A supported: keep `step="entropic_exact"` as the recommended short-budget rule and document the budget range where it wins.
- A not supported and B supported: add a `tilt=theta` objective mode to `StabilityAdapter`/`polish_logistic` (changes the optimum, so it must be opt-in and documented as such).
- Neither supported: say so in the tutorial and videos: the stability gain is exact full-data steps on a convex problem; the risk-averse step is a safe way to take few steps, not an extra source of gain.

## 8. Implementation notes
- Reuse `research/linesearch/linesearch_experiment.py` (steps R1, R3, exact LS, golden-section search) and `risk_averse_tuning.adapter.polish_logistic`.
- New code: `risk_effect_experiment.py` (arms A0-A5, step matching for A4, outcomes O1-O4), `tilted_experiment.py` (B), `partial_pass.py` (C). Run detached with `nohup setsid`, OMP_NUM_THREADS=1, one process per dataset; logs gitignored.
- Order of work: A on German (smoke test), A on all datasets, B, C. Record deviations in RESULTS.md before reading any result.
