# Risk-adjusted adapter vs one more epoch of training: results

Design: `PLAN.md` section 2 (pre-registered before any run). Code: `risk_effect_experiment.py`, `analyze.py`; raw: `results/<dataset>.json`;
every table: [`tables.md`](tables.md). German Credit, COMPAS, Pima, Heart; 200 stage-1 models each, from SGD (step 1/L, 2,000 minibatch steps of 16)
or Adam (lr 0.01); L2 logistic regression, lambda = 0.01. Every arm gets the same extra budget: B epochs of row-gradients, B in {1, 2, 5, 20}.
Paired bootstrap over seeds (1,000 resamples). "Two retrains disagree" = chance that two of the 200 models decide differently on a held-out row.

**Deviations (recorded before reading results):** (1) 4 datasets instead of 5: the Adult subsample and Experiments B and C of the plan were not run yet; the "at least 4 of 5"
rule is therefore read as "all 4". (2) Groups (worst-group wrong-decision rate) only for German Credit (the pre-registered groups of `research/groups`, those with at least 30
held-out rows); the other datasets have no pre-registered groups. (3) The step-matched control and the theta dose-response of PLAN section 3 were not run; only theta = 1 (first-order,
the original proposal) and theta = 5 (exact change) were. R vs the exact line search (L) is reported, which is part of the attribution. (4) Wall-clock seconds come from an unoptimized NumPy
implementation of the step search (80-point grid + refinement over all seeds); compare orders of magnitude only.

Arms: **C1** continue the same optimizer unchanged; **C2** continue with the step size decayed to 0; **C3** continue and average the iterates; **R first-order** the proposal:
full-batch steps, each step length from min_a (1/theta) log mean_i exp(-theta a g^T grad f_i), theta = 1; **R exact-change** (theta = 5) the version that protects rows a step would hurt;
**L** full-batch steps with exact line search (exact gradient, no risk term).

## 1. Headline: chance that two retrains disagree on a row (lower is better)

Stage 1 = Adam:

| Dataset | Stage 1 | Extra epochs | best of C1-C3 | **R first-order** | L exact line search |
|---|---|---|---|---|---|
| German | 8.4% | 1 / 2 / 5 / 20 | 8.4 / 7.2 / 5.2 / 3.0% | **3.7 / 2.5 / 1.3 / 0.0%** | 3.7 / 2.5 / 1.3 / 0.1% |
| COMPAS | 7.2% | 1 / 2 / 5 / 20 | 3.4 / 2.4 / 1.3 / 0.6% | **1.9 / 0.8 / 0.2 / 0.0%** | 1.8 / 0.8 / 0.2 / 0.0% |
| Pima | 2.6% | 1 / 2 / 5 / 20 | 2.6 / 2.3 / 1.5 / 0.7% | **0.7 / 0.3 / 0.2 / 0.0%** | 0.7 / 0.3 / 0.2 / 0.0% |
| Heart | 2.5% | 1 / 2 / 5 / 20 | 2.4 / 2.4 / 2.3 / 1.8% | **1.5 / 0.7 / 0.0 / 0.0%** | 1.3 / 0.6 / 0.0 / 0.0% |

Stage 1 = SGD (step 1/L, the noisier start):

| Dataset | Stage 1 | Extra epochs | best of C1-C3 | **R first-order** | L exact line search |
|---|---|---|---|---|---|
| German | 25.1% | 1 / 2 / 5 / 20 | 14.1 / 10.8 / 7.1 / 3.8% | **13.8 / 8.2 / 3.1 / 0.1%** | 12.7 / 8.4 / 3.5 / 0.4% |
| COMPAS | 29.2% | 1 / 2 / 5 / 20 | 3.4 / 2.4 / 1.3 / 0.5% | 10.7 / 6.3 / 1.4 / 0.0% | 11.1 / 7.2 / 1.4 / 0.0% |
| Pima | 14.0% | 1 / 2 / 5 / 20 | 5.5 / 3.9 / 2.3 / 1.1% | **5.1 / 2.0 / 0.2 / 0.0%** | 4.8 / 1.9 / 0.2 / 0.0% |
| Heart | 7.6% | 1 / 2 / 5 / 20 | 5.3 / 4.4 / 3.0 / 2.0% | **3.3 / 1.9 / 0.6 / 0.0%** | 2.8 / 1.8 / 0.6 / 0.0% |

"Best of C1-C3" is chosen per cell by its disagreement, which favours the baseline. It is always C3 (averaged iterates), except Heart at one epoch (C2). Plain continuation (C1)
does nothing for stability: 25.3% / 29.4% / 14.2% / 7.8% at one epoch for SGD, the same as stage 1.

## 2. Pre-registered decision rule

Rule: R beats the best continuation on disagreement and on changed decisions at B = 1 and 2, CI excluding 0, on all 4 datasets, with the overall wrong-decision rate no worse by more than 1 point.

| Stage 1 | R variant | B = 1 | B = 2 | Verdict |
|---|---|---|---|---|
| Adam | first-order (theta = 1) | 4 / 4 datasets better | 4 / 4 | **supported** (wrong-decision rate within +0.3 points) |
| Adam | exact-change (theta = 5) | 3 / 4 (COMPAS worse by 2.5 points) | 3 / 4 | not supported |
| SGD | first-order (theta = 1) | **1 / 4** (German and Pima intervals include 0; COMPAS worse by 7.3 points) | 3 / 4 (COMPAS worse by 3.9 points) | **not supported** |
| SGD | exact-change (theta = 5) | 0 / 4 | 0 / 4 | not supported (worse by 0.9-22.6 points at one epoch) |

Where the adapter overtakes averaged continuation when the start is SGD: from 2 epochs on German, Pima and Heart; from 5 epochs on COMPAS (0.2 points worse, not significant);
at 20 epochs everywhere (0.0-0.1% vs 0.5-3.8%). COMPAS is the large dataset (n about 4,300): there one epoch is 270 minibatch steps, and averaging them already removes most noise.

## 3. Attribution: is it the risk adjustment?

* **No for the first-order step.** R first-order vs the plain exact line search L: differences are at most about 1 point and go both ways (L better at one epoch on German by 1.1 points, Pima 0.3, Heart 0.5; R better at 2 epochs on German by 0.2 and COMPAS by 0.9;
  identical from 5 epochs; under Adam within 0.2 points everywhere). The gain over continuation comes from **exact full-data gradients**.
* **What the risk term does change.** The exact-change variant (theta = 5) protects the rows a step would make worse: one epoch from SGD, German: 95th-percentile per-row loss increase +0.056 vs +0.695 for L and +0.80 for averaged continuation,
  share of rows made worse 0.34 vs 0.41 / 0.42; COMPAS: +0.05 vs +0.40 / +0.38. The price is smaller steps (mean 0.6 x 1/L vs 4.3 x 1/L), which leaves more seed variation at small budgets (German 23.2% vs 12.7% disagreement at one epoch),
  so it is a trade-off knob, not an extra gain. It loses to averaged continuation on disagreement for SGD stage 1 at every budget up to 5 epochs, and wins from 20 epochs on German (2.3% vs 3.8%) and Pima (0.4% vs 1.1%) but not COMPAS.
* **Groups (German, worst of the groups with at least 30 held-out rows).** All arms are within 3 points of each other (SGD stage 1, one epoch: C1 41.8%, C3 39.3%, R first-order 39.0%, R exact-change 41.0%, L 39.2%; Adam: 38.4-38.9%).
  The risk adjustment has no measurable effect on the worst group's wrong-decision rate beyond what averaging also gives.
* Not tested: the step-matched control and the theta dose-response (PLAN section 3). They would separate "risk-weighted step" from "smaller step" for the exact-change variant.

## 4. Cost (caveat: unoptimized implementation)

Continuing or averaging one epoch is nearly free (about 0.0-0.4 s for all 200 seeds on German Credit at 1-20 epochs), while the full-batch steps with a step search took 0.6-2.9 s at one epoch and 10-22 s at 20 epochs
(COMPAS: 4-10 s at one epoch, 115-214 s at 20 epochs), because each step evaluates the loss of every row at about 100 trial step lengths. The compute rule of the plan counted a full-batch step as one epoch of row-gradients; in wall-clock, averaged
continuation is much cheaper per epoch in this implementation. A fair claim is therefore about **quality at equal epochs**, not equal seconds; a production version would need a cheaper step search (for example a few Newton iterations along g, one extra `X g` product).

## 5. Reading

1. **Against "one more epoch" the adapter wins clearly when the stage-1 optimizer is Adam, and from about 2-5 epochs when it is SGD with a large step.** With plain continuation (C1) it wins everywhere by a wide margin; with the strongest baseline (continue and average) it loses on a large dataset (COMPAS) at 1-2 epochs from SGD.
2. **The win is the exact gradient, not the risk term.** The risk-averse first-order step is equivalent to an exact line search; its stability gain is the same.
3. **The risk term is a harm-control knob.** At theta = 5 it makes far fewer rows worse per step, at the price of slower convergence. A claim of "risk adjustment improves worst-case outcomes" needs experiments B (tilted objective) and the step-matched control, which are still open.

Statements describe statistical behaviour of models on public datasets; not accusations of bias; not credit advice.

---

# Experiment B: tilted objective (does risk change the final model?)

Design: `PLAN.md` section 4. Code: `tilted_experiment.py`, `analyze_tilted.py`; raw `results/tilted_<dataset>.json`; every table: [`tilted_tables.md`](tilted_tables.md).
The objective itself is tilted, F_theta(w) = (1/theta) log mean_i exp(theta l_i(w)) + (lambda/2)||w||^2, theta in {0.5, 1, 2, 5} (theta = 0 is ordinary training), lambda = 0.01.
It is convex and solved exactly by Newton (checked: gradient norm 3e-10 at the optimum; theta -> 0 reproduces ordinary training to 5e-6), so there is **no seed noise** in any arm,
and "train longer" cannot reproduce it. 20 random stratified splits per dataset (German: 700/300; others 70/30, at most 2,000 held-out rows); CIs are paired bootstraps over splits.

**Deviations (recorded before reading results):** (1) The plan's controls are tuned to match overall accuracy within 0.5 points; the reweighting control (positive-class weight on a 33-point grid between 0.25 and 4)
was matched on **training** accuracy so no test data is used, and this did not give a test-accuracy match within 0.5 points on German, Pima and Heart (see the accuracy-gap column). (2) Only German has pre-registered groups; for the other datasets the "worst group" is
the **worst class** (the larger of the false-negative and false-positive rates). (3) German worst-group error is over the nine pre-registered groups with at least 200 pooled held-out rows (G1, G3-G10).
(4) Group-DRO is approximated by 40 rounds of multiplicative weights over the groups (German only).

## Result

| Comparison (20 splits, percentage points, 95% CI) | German | COMPAS | Pima | Heart |
|---|---|---|---|---|
| Tilted theta = 5 minus ERM: accuracy | -0.3 [-0.7, +0.1] | -0.0 [-0.1, +0.0] | -0.3 [-1.0, +0.2] | **-2.0 [-2.8, -1.3]** |
| Tilted theta = 5 minus ERM: worst-class error | -0.2 [-1.1, +0.7] | +0.1 [-0.1, +0.3] | +1.0 [-0.3, +2.1] | +0.9 [-0.5, +2.3] |
| Tilted theta = 5 minus ERM: worst-group error (German) | -1.0 [-2.5, +0.9] | - | - | - |
| Tilted theta = 5 minus matched reweighting: accuracy gap | +1.1 [+0.4, +1.9] | -0.0 [-0.3, +0.1] | +0.5 [-0.3, +1.3] | +0.1 [-1.6, +1.9] |
| Tilted theta = 5 minus matched reweighting: worst-class error | -1.4 [-9.0, +6.2] | **+3.2 [+1.5, +4.9]** | **-5.0 [-9.1, -0.6]** | **-3.7 [-6.7, -0.5]** |
| Tilted theta = 5 minus matched reweighting: worst-group error (German) | -2.9 [-4.8, +0.5] | - | - | - |

Other theta (0.5, 1, 2) give the same picture (see `tilted_tables.md`). Group-DRO with known groups (German): worst-group error -1.1 [-3.9, +1.9] vs ERM at a cost of -1.8 [-2.7, -0.9] accuracy.

* **The tilt barely moves the model.** At theta = 5 the effective sample size is still 39-88% of the rows and the heaviest row gets 3-14 times the uniform weight; the optimum moves little away from ordinary training, and tilted-minus-ERM is within about 1 point on every
  error measure (except Heart accuracy, which drops 0.9-2.0 points, significant). Even oracle knowledge of the groups (group-DRO) gives no significant change in the worst-group error on German at 20 splits.
* **Against reweighting, the picture depends on what is compared.** Tilted beats the matched reweighting on worst-class error on Pima and Heart (3-6 points, CI excluding 0) and loses on COMPAS (+3 to +4). But the "matched" reweighting is not matched: on German it is 1 point less accurate, and
  reweighting trades the false-negative rate against the false-positive rate (a different operating point), which is what a worst-class error rewards or punishes depending on the dataset. These differences are about the operating point, not about hard rows.
* **Decision rule (PLAN section 4):** "at matched overall accuracy (within 0.5 points) the tilted model has a lower worst-group wrong-decision rate than the reweighting control, CI excluding 0, on at least 2 datasets" is **not supported**: the accuracy match fails on 3 of 4 datasets, the
  German worst-group difference has a CI including 0 at every theta, and the CI excludes 0 (in the right direction) only for worst-class error on Pima and Heart, where the accuracy gap is 0.1-0.6 points but the control is not a clean match. At most it is **suggestive on two of four datasets and contradicted on a third**.

## What this means

1. **No separate claim for the tilted objective** on these data: it is close to ordinary training, and what differs from reweighting is mostly which class's errors are traded.
2. A **stronger tilt** (theta well above 5, or tilting over groups instead of rows) would change the model more but is a different method with real accuracy costs (Heart: -2 points at theta = 5). If that is wanted, it needs its own pre-registered test with an operating-point-matched control (match the
   false-negative rate, not only accuracy).
3. Together with the main result above: **the measurable value of the adapter is the exact full-data steps; the risk adjustment gives a harm-control knob (fewer rows made worse per step) and, as a tilted objective, no reliable change in worst-group outcomes here.**

Experiment C (large n, few passes) is not run. Statements describe statistical behaviour of models on public datasets; not accusations of bias; not credit advice.
