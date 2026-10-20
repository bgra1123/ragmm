# Does the risk adjustment do anything? Isolating it from "one more round of exact learning"

Status: PRE-REGISTRATION, written before any run. Results go to `RESULTS.md`. Branch: `claude/ragmm-product-content-s3k2po`.

## 1. What is being tested

The polish adapter has two ingredients:
- **I1, exact full-batch steps** (no minibatch noise). Already shown to remove most seed variation (`research/product`, `research/linesearch`).
- **I2, a risk-averse rule for the step length or the objective** (entropic risk, Can & Gürbüzbalaban 2025; tilted ERM, Li et al. 2021).

Claim to test: **I2 changes outcomes beyond I1.** Evidence so far suggests that with enough full-batch steps on a strongly convex problem, both reach the same optimum, so I2 can only matter when (a) the budget is short, or (b) the objective itself is tilted. Two experiments follow, one for each route.

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
- Section 2 supported (R beats continuation): keep `step="entropic_exact"` as the recommended short-budget rule and document the budget range where it wins.
- Section 2 not supported and B supported: add a `tilt=theta` objective mode to `StabilityAdapter`/`polish_logistic` (changes the optimum, so it must be opt-in and documented as such).
- Neither supported (continuing training is as good): say so in the tutorial and videos: the stability gain is exact full-data steps on a convex problem; the risk-averse step is a safe way to take few steps, not an extra source of gain.

## 8. Implementation notes
- Reuse `research/linesearch/linesearch_experiment.py` (steps R1, R3, exact LS, golden-section search) and `risk_averse_tuning.adapter.polish_logistic`.
- New code: `risk_effect_experiment.py` (arms A0-A5, step matching for A4, outcomes O1-O4), `tilted_experiment.py` (B), `partial_pass.py` (C). Run detached with `nohup setsid`, OMP_NUM_THREADS=1, one process per dataset; logs gitignored.
- Order of work: A on German (smoke test), A on all datasets, B, C. Record deviations in RESULTS.md before reading any result.
