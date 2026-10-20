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
3. **The risk term is a harm-control knob.** At theta = 5 it makes far fewer rows worse per step, at the price of slower convergence. A claim that the risk-averse step improves worst-case outcomes needs the step-matched control and the theta dose-response (PLAN section 3), which are still open.

Statements describe statistical behaviour of models on public datasets; not accusations of bias; not credit advice.

*Note: an earlier version of this folder also tested a tilted training objective (tilted ERM, a different method with a different optimum). It is outside the question here, which is about the risk-averse step length, and was removed; it remains in the git history.*

## Addendum: streaming step search (cost)

The first implementation of the step search (80-point grid + golden-section over all rows, for 200 models at once) made the full-batch arms look expensive in
seconds (section 4). The step rule needs only sums over rows once the gradient `g` is known, so it can be computed batch by batch. New in the package:
`polish_logistic_streaming(w0, batches, lam, step=..., search="grid" | "newton")` (also `StabilityAdapter(batch_size=...)`). Per epoch it makes two passes over the data: (1) the full gradient;
(2) the per-row quantities along `g` (margin `y x.w`, margin change `y x.g`, and the directional derivative `g.grad f_i`, which for a neural network is one forward-mode JVP per batch).
`search="grid"` accumulates log-sum-exp sums on a 120-point grid of step lengths (memory O(grid)); `search="newton"` keeps three length-n vectors and solves the convex 1-D problem by safeguarded Newton with no further data pass (memory O(n)).

**Correctness (tests in `tests/test_adapter.py`):** the streaming result equals the in-memory result to 1e-8 (Newton) / 1e-3 (grid; the parabola refinement) in the weights, for all three step rules and theta = 1 and 5,
and does not depend on the batch size (differences below 1e-9). 127 tests pass.

**Cost (`cost_benchmark.py`, `results/cost.json`; one model, 20 extra epochs, single thread, median of 3; seconds):**

| Dataset | n | SGD continue (batch 16) | line search, in memory | line search, streaming Newton | entropic, in memory | entropic, streaming grid | **entropic, streaming Newton** |
|---|---|---|---|---|---|---|---|
| German | 700 | 0.019 | 0.055 | 0.010 | 0.018 | 0.012 | **0.004** |
| COMPAS | 4,320 | 0.094 | 0.234 | 0.013 | 0.046 | 0.081 | **0.013** |
| Adult | 46,842 | 1.572 | 2.461 | 0.356 | 0.428 | 0.643 | **0.261** |
| synthetic (timing only) | 200,000 | 5.108 | 6.953 | 1.736 | 2.035 | 2.353 | **1.193** |

* The grid search is not cheaper than the in-memory search (the same number of operations per row); the saving is **memory**, so the data can be streamed.
* The Newton search on cached scalars is 2-4 times cheaper than the in-memory version and, in this implementation, **4-6 times cheaper than the same number of epochs of continued minibatch SGD**.
* Caveat for that last comparison: the SGD baseline here is an interpreted NumPy loop of 16-row steps; a framework implementation of minibatch SGD on a GPU has a different cost profile. The fair statement is that the step search is no longer the bottleneck:
  a step costs two passes plus a 1-D solve on length-n vectors.
* The quality comparison of section 1 is unchanged (it counted epochs, not seconds).

## Addendum: step-matched controls and theta sweep (PLAN section 9)

Code `stepmatch_experiment.py`, analysis `analyze_stepmatch.py`, every table in [`stepmatch_tables.md`](stepmatch_tables.md), raw `results/stepmatch_<dataset>.json`. 200 stage-1 models per dataset from SGD and from Adam,
1, 2 and 5 full-batch steps along the same full gradient; arms: L (exact line search), R1 (your first-order entropic step) and R3 (my exact-change variant) at theta in {0.5, 1, 2, 5, 10};
**M1** = the exact line search with every step multiplied by the constant c that gives the same mean step as R (so it keeps the line search's per-seed adaptivity), **M2** = R's mean step of each iteration replayed as one fixed schedule for all seeds.
Outcome of record: the 95th percentile of the per-row loss increase on held-out rows (lower = fewer rows hurt). Deviations: 20 steps omitted (all arms have converged); German groups only; paired bootstrap over seeds (1,000 resamples).

**1. Larger theta means smaller steps and less harm (dose-response holds).** Mean held-out harm falls monotonically as theta grows from 0.5 to 10 on 4/4 datasets in 10 of 12 (stage 1, steps, rule) settings
(SGD, 1 step, R1: +1.14 → +0.44 → +0.19 → +0.07 → +0.03; Adam, 1 step, R3: +0.082 → +0.011). The mean step falls with it: R1 at theta = 5 takes 0.4-1.8 times 1/L, against 2-6 times 1/L for the exact line search.

**2. Against the exact line search, the risk step hurts far fewer rows**, as before: e.g. German, SGD, 2 steps, R3(theta = 5): -0.43 [-0.44, -0.41] in p95 harm. That comparison does not isolate the risk term, because the step is also 3-7 times shorter.

**3. Against the step-matched controls the difference is small.** Over the 160 cells with 2 or 5 steps (2 stage-1 optimizers × 2 step counts × 2 rules × 5 thetas × 4 datasets):
* R has significantly lower harm than M1 (same mean step, per-seed line-search shape) in 99 cells; the median difference is -0.002, small beside the harm levels of 0.03-1.1. It is not lower at one step (0 of 4 datasets for SGD, 1 of 4 for Adam).
* R is significantly better than M2 (the same per-iteration step for every seed) in 83 cells and significantly worse in 20; the median difference is -0.002, i.e. most of the harm reduction is **the step length, not the risk weighting of each seed's step**.
* **The saving is paid for in seed variation.** In 78 of the 99 cells where R beats M1 on harm, R has significantly *more* disagreement than M1 (typically +0.3 to +4 points; German, SGD, 5 steps, R1(theta = 5): harm -0.116 vs M1 but disagreement +3.9 points). Only 9 cells have lower harm and no worse disagreement.
* **Pre-registered credit rule (harm lower than both L and M1 on at least 3 of 4 datasets for some theta): met** from two steps on, for example SGD 2 steps R1(theta = 2) and R3(theta = 1) on 4/4, SGD and Adam 5 steps for theta between 2 and 5 on 4/4, but **not at one step**. The effect sizes are small, so the rule is a weak test.
* **Worst-group wrong-decision rate (German): no effect.** R minus M1 is between -0.3 and +0.6 points across all settings (e.g. SGD, 5 steps, R3(theta = 5): +0.4 [+0.1, +0.8]). The group part of the credit rule is **not met**.

**Reading.** The risk-averse step is a way to take shorter steps that are somewhat adaptive to each seed. Most of what it does for per-row harm is what any shorter step does; the seed-adaptive part adds a small further reduction (a few thousandths of a loss unit) that costs seed stability, and it does not change who is wrong. For stability the exact full-data gradient remains what matters, and the risk term is best described as a knob that trades convergence speed for fewer rows made worse.

Experiment C (large data, few passes) remains open. Statements describe statistical behaviour of models on public datasets; not accusations of bias; not credit advice.
