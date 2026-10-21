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

## Addendum: negative-flip test (PLAN section 10)

Code `negflip_experiment.py`, analysis `analyze_negflip.py`, all tables in [`negflip_tables.md`](negflip_tables.md), raw `results/negflip_<dataset>.json`. Same 200 stage-1 models (SGD and Adam), same budgets (1, 2, 5 epochs), same arms as the step-matched test plus the continuation arms C1-C3.
Negative flip = held-out row decided correctly by the stage-1 model and wrongly after the update; positive flip = the reverse; net = positive minus negative = the accuracy change. Rates are % of held-out rows, mean over the 200 models, averaged over datasets below.

| Update (4-dataset average, % of held-out rows) | negative flips | positive flips | net accuracy change |
|---|---|---|---|
| SGD start, 1 epoch: C1 continue | 9.65 | 9.67 | +0.02 |
| SGD, 1 epoch: C3 continue and average | 5.59 | 7.94 | +2.35 |
| SGD, 1 step: exact line search L | 5.50 | 7.83 | +2.33 |
| SGD, 1 step: **R1 (theta = 5)** | **0.77** | 1.42 | +0.65 |
| SGD, 1 step: line search with R1's mean step (M1) | 0.77 | 1.42 | +0.65 |
| SGD, 5 steps: L | 5.40 | 8.27 | +2.87 |
| SGD, 5 steps: **R1 (theta = 5)** | **3.18** | 5.35 | +2.18 |
| SGD, 5 steps: M1 | 3.79 | 6.19 | +2.41 |
| Adam start, 1 step: L / R1 (theta = 5) / M1 | 1.64 / **0.33** / 0.33 | 1.80 / 0.39 / 0.39 | +0.16 / +0.07 / +0.06 |
| Adam, 5 steps: L / R1 (theta = 5) / M1 | 1.61 / **1.18** / 1.32 | 1.98 / 1.39 / 1.58 | +0.37 / +0.21 / +0.26 |

**Results over all 240 cells** (2 stage-1 optimizers × 3 step counts × 2 rules × 5 thetas × 4 datasets; paired bootstrap over seeds):
* **Against the exact line search and against the best continuation**, the risk-averse step has far fewer negative flips: significantly fewer in 179 and 190 cells (more in 34 and 34); the median difference is -0.58 and -0.64 percentage points of held-out rows.
* **Against the step-matched controls**, almost nothing: against M1 (same mean step) significantly fewer in 85, more in 46, not significant in 109 cells, median difference -0.004 points; against M2 (same schedule for all seeds) fewer in 104, more in 16, median -0.034 points.
* **The reduction is mostly because the model moves less.** The positive flips fall with the negative ones, so the net accuracy gain is smaller than with the line search (SGD, 1 step: +0.65 against +2.33 points). Negative flips per point of net gain: L 2.4, R1(theta = 5) 1.2 at one step, but M1 (the plain line search with the same mean step) is also 1.2; at five steps R1 1.5 against M1 1.6 and L 1.9.
* **Pre-registered credit rule** (lower negative-flip rate than both M1 and M2, and net no worse than M1 by more than 0.5 points, on at least 3 of 4 datasets, for some theta, at both 2 and 5 steps): **not met**. The best settings pass on 2 of 4 datasets (for example SGD, 2 steps, R1 theta = 2: COMPAS and Pima; at 5 steps, R1 theta = 5: COMPAS and Pima), and no setting passes at one step.
* **Worst group (German, groups with at least 30 held-out rows):** negative-flip rate SGD 1 step: L 10.96%, C3 12.26%, R1(theta = 5) 1.13%, M1 1.17%; 5 steps R1 5.84% against M1 8.12%. The reductions against L and continuation are large and are the step length; against M1 the 5-step difference is 2.3 points for SGD and 0.5 for Adam, with one dataset.

**Reading.** Taking short steps is a good way to avoid regressions, and a risk-averse step is one way to take them: it keeps negative flips at a fraction of what a full line-search step or one more epoch of training causes. It is **not better than a plain line search of the same size** on negative flips, apart from a small advantage at 5 steps on some datasets. The honest statement of value is "short, exact steps avoid regressions", with the risk-averse rule as one way to choose the short step that needs no hand-tuned step size (θ sets the cautiousness; the matched control needed the risk step's mean as input).

Statements describe statistical behaviour of models on public datasets; not accusations of bias; not credit advice.

## Addendum: Experiment C, large n and a fraction of a pass to two passes (PLAN section 11)

Code `large_data_experiment.py`, analysis `analyze_large.py`, all tables in [`large_tables.md`](large_tables.md), raw `results/large_<n>.json`. Adult income, training sizes 5,000, 20,000 and 46,842 rows, 2,000 held-out rows, 100 stage-1 models per size from SGD (step 1/L) and Adam (lr 0.01), extra budget 0.25, 0.5, 1 and 2 passes.
For budgets below one pass the full-batch arms take one step whose gradient and step search use a random subset of that fraction of the rows, drawn independently per model (stochastic arms). Continuation arms (C1 continue, C2 decayed, C3 averaged) use minibatch steps of 16 rows for the same row budget.
The weighted step search was checked against the earlier one (relative difference below 5e-9). Deviations: 100 models per size (not 200); one dataset; cost is not re-measured here (see the streaming-cost addendum).

**1. Two retrains disagree.** SGD start (stage 1: 17.9-19.0%): average-and-continue (C3) wins at every size and budget, by a margin that grows with n.

| Training rows | Passes 0.25 / 0.5 / 1 / 2: C3 | Best risk step (R1 or R3 at θ = 1) | Line search L |
|---|---|---|---|
| 5,000 | 8.1 / 5.9 / 4.2 / 2.9% | 11.5 / 10.3 / 9.7 / 6.5% | 11.9 / 11.2 / 10.6 / 7.4% |
| 20,000 | 4.7 / 3.2 / 2.3 / 1.8% | 11.1 / 10.8 / 10.7 / 7.2% | 12.4 / 12.0 / 11.9 / 8.1% |
| 46,842 | 3.1 / 2.3 / 1.7 / 1.2% | 11.1 / 11.0 / 11.0 / 7.2% | 12.6 / 12.3 / 12.3 / 8.2% |

The risk step is worse than the best continuation in all 12 cells (differences +3.4 to +9.2 points, all intervals above 0). Decisions changed against the exact model (46,842 rows, SGD): stage 1 273; C3 58 / 51 / 46 / 44; R1(θ = 1) 164 / 162 / 162 / 124; exact line search 192 / 188 / 187 / 124.
Wrong-decision rate: stage 1 20.9%, C3 16.3-16.6%, R1(θ = 1) 17.2-18.2%.
Reason: with n of 5,000 to 47,000, one pass of minibatch SGD is 300 to 2,900 steps, while a full-batch arm takes 1 or 2 steps and cannot converge; at 2 passes they reach about 7-8%.

**2. Adam start** (stage 1: 6.0-6.5%). The risk step beats the best continuation only on the smallest size: n = 5,000 at 0.25 passes (R3(θ = 1) 4.4% against 5.4%, −1.0 [−1.2, −0.8]), 0.5 passes (3.8% against 4.3%) and 2 passes (R1(θ = 1) 1.7% against 2.3%); at 1 pass the interval includes 0 (3.0% against 3.2%).
At 20,000 and 46,842 rows C3 wins everywhere (46,842 rows, 2 passes: 1.0% against 1.9%).

**3. Overtaking budget.** Within 2 passes the risk step never overtakes average-and-continue with an SGD start, and overtakes it with an Adam start only at 5,000 rows. Earlier experiments (E15) showed overtaking from about 5 to 20 epochs at 700 to 4,300 rows; whether the crossing for 47,000 rows lies at 5, 10 or 20 passes was not tested here.

**4. Does the risk term protect rows (theta = 5)?**
* Against the exact line search the risk step is far gentler: 95th-percentile per-row loss increase −0.7 to −0.8 (SGD, below 2 passes), negative flips −5.6 to −6.5 points of rows. But R1(θ = 5) takes steps of 1.1-1.3 times 1/L here, and the matched control M1 (same mean step) does nearly the same.
* Against M1 at less than 2 passes the harm difference is −0.001 to −0.004 (intervals touching 0) and the negative-flip difference −0.01 to −0.04 points: no practical difference. At 2 passes (two full steps) it is larger: harm −0.04 to −0.06, negative flips −0.4 to −0.55 points, but with +1.2 to +2.0 points more seed disagreement.
* Average-and-continue also hurts few rows: negative flips 4.5-5.0% of rows against 0.7% for the risk step (SGD, below 2 passes), but its positive flips are 8-9%, so its net accuracy gain is about +4 points against +1 (46,842 rows, 1 pass: C3 net +4.5, R1(θ = 5) net +1.0, exact line search net +2.3).
* **Credit rule (harm lower than both L and M1, same θ, at 0.25, 0.5 and 1 pass, at least 2 of 3 sizes, for both starts): not met.** It holds for the SGD start (R1(θ = 5) 3 of 3 sizes at each budget) but not for the Adam start (1 or 2 of 3 sizes, and 1 of 3 at 0.25 and 1 pass), and the effect against M1 is a few thousandths of a loss unit.

**Reading.** In the regime that looked most favourable to a careful step rule (large data, only a fraction of a pass to two passes), **continuing minibatch training and averaging the iterates is the better update** on stability, decisions changed and accuracy. The full-batch steps are too few to converge, and the risk term only makes them gentler,
as a plain smaller step does. This does not contradict E15: the adapter's advantage appears when several full passes are affordable, and gets there later the larger n is.

Statements describe statistical behaviour of models on a public dataset; not accusations of bias; not credit advice.

## Addendum: step multiplier on the adapter's alpha (PLAN section 12)

Code `stepscale_experiment.py`, analysis `analyze_stepscale.py`, all tables in [`stepscale_tables.md`](stepscale_tables.md), raw `results/stepscale_<name>.json`. The update is w <- w - omega alpha* g with omega in {1, 1.5, 2, 3} applied at every step (the step search is re-run along the new path), for the exact line search (L), R1 at theta = 1 and 5, and R3 at theta = 5, with 1, 2 and 5 full-batch steps, SGD and Adam starts, five datasets (German, COMPAS, Pima, Heart: 200 models; Adult with 46,842 rows: 100 models). Reference: average-and-continue (C3).

**Result.**
* **H1 supported.** For the theta = 5 rules (which take steps of 0.4-1.5 times 1/L, against 2-6 times for the line search) a multiplier above 1 lowers two-retrain disagreement on **all 5 datasets in every setting** (win rule met: 5 of 5 datasets for omega = 1.5, 2 and 3, both starts, 1, 2 and 5 steps; wrong-decision rate never worse by more than 1 point).
  Example, 2 steps, SGD start: COMPAS R1(theta = 5) 21.6% at omega = 1, 13.8% at 2, 7.3% at 3; German 20.6%, 15.8%, 11.5% (omega = 3); Adam: German 6.1%, 4.2%, 2.9%.
* **H2 supported.** The multiplier does not help the rules that already take line-search-sized steps: L and R1(theta = 1) win on 0 of 5 datasets for every omega above 1. Overshooting is costly: German, SGD, 2 steps, L: 8.4% at omega = 1, 21.8% at omega = 2; COMPAS 7.2% to 33.2%.
* **H3 supported.** A theta = 5 rule with omega chosen so that its mean step is about the line search's recovers the line search and does not beat it by more than a few tenths of a point: 5 steps, SGD, disagreement of L against R1(theta = 5) at omega = 3: German 3.5% / 3.2%, COMPAS 1.4% / 1.2%, Pima 0.2% / 0.3%, Heart 0.6% / 0.5%; Adam: German 1.3% / 1.2%, COMPAS 0.2% / 0.1%.
  At 1 to 2 steps the unscaled L or R1(theta = 1) is the best arm in most cells; a scaled theta = 5 arm is the best in 10 of 30 dataset-start-step cells (mostly at 5 steps and on Adult).
* **H4 supported at up to 2 steps and refined at 5.** On Adult (46,842 rows) no arm beats average-and-continue within 2 steps (best arm against C3, SGD start: +9.3 points at 1 step, +5.2 at 2; Adam start: +1.9, +0.8). At 5 steps with an Adam start the scaled R1(theta = 5) at omega = 2 is slightly better (0.5% against 0.6%, difference −0.10 [−0.14, −0.05]); with an SGD start it is still far behind (2.8% against 0.8%).
  This locates the crossing for large data: between 2 and 5 passes with an Adam start, and beyond 5 passes with an SGD start (not tested further).

**Reading.** Taking larger steps on the adapter's alpha helps exactly when its steps are shorter than the line search's, and the gain is to move toward the exact line-search step. It does not go beyond it: the exact line search is already the best step along the gradient, and larger steps overshoot. For the entropic rule this means a multiplier works as the way to choose how cautious to be (theta and omega trade off), not as an additional source of stability.

## Addendum: Newton's method for the exact solution (PLAN section 13)

Code `newton_experiment.py`; the streaming Newton solve is in the package (`polish_logistic_streaming(step="newton")`, tested against the in-memory solve to 1e-9, and with different starts reaching the same model to 1e-8). Tables: [`newton_tables.md`](newton_tables.md), raw `results/newton_<name>.json`. Damped Newton (Armijo backtracking) on the L2 logistic objective, started from the stage-1 weights of 200 models (100 for Adult).

| Dataset (n, d) | Start | Two retrains disagree: stage 1 → after Newton iteration 1 / 2 / 3 / 4 | Iterations to gradient norm 1e-8 |
|---|---|---|---|
| German (700, 55) | SGD | 25.1% → 9.2 / 3.4 / 1.3 / 0.3% (0.02% at 5) | 5 |
| German | Adam | 8.4% → 1.2 / 0 / 0 / 0% | 4 |
| COMPAS (4,320, 8) | SGD | 29.2% → 14.5 / 1.8 / 0.2 / 0.02% | 7 |
| COMPAS | Adam | 7.2% → 0.3 / 0 / 0 / 0% | 4 |
| Pima (538, 9) | SGD | 14.0% → 4.5 / 0.2 / 0 / 0% | 4 |
| Pima | Adam | 2.6% → 0.1 / 0 / 0 / 0% | 3 |
| Heart (212, 27) | SGD | 7.6% → 1.9 / 0.06 / 0 / 0% | 5 |
| Heart | Adam | 2.5% → 0 / 0 / 0 / 0% | 3 |
| Adult (46,842, 101) | SGD | 18.8% → 8.2 / 5.5 / 4.3 / 2.2% (0.6% at 5, 0.2% at 6) | 10 |
| Adult | Adam | 6.7% → 3.1 / 1.7 / 0.9 / 0.2% (0 at 5) | 8 |

* **H1 mostly supported.** Disagreement is below 0.1% within 5 iterations on every dataset and start except Adult with an SGD start (0.6% at 5, below 0.1% only from iteration 7 on); decisions changed against the exact model reach 0 by iteration 3 or 4 on the four smaller datasets (Adult SGD: 29 changed per 2,000 rows after 4 iterations). On Adult the early iterations take damped steps because the SGD start is far from the optimum.
* **H2 not supported.** One or two Newton iterations do not beat 20 line-search steps on the objective gap (German SGD: 6.8e-3 after two against 1.2e-5 for 20 line-search steps), except after Adam on German; the gap falls by a factor of 4 to 7 per iteration in the damped phase and then quadratically. Newton matches 20 line-search steps after about 3 to 5 iterations (on the four smaller datasets).
* **H3 partly supported (single model, one thread, streaming, seconds):** Newton to convergence took 0.005 / 0.015 / 0.007 / 0.002 s against 0.011 / 0.023 / 0.021 / 0.007 s for 20 line-search steps on German / COMPAS / Pima / Heart (2 to 7 times faster), but 0.84 s against 0.61 s on Adult (101 features, 10 iterations; the line search has not converged after 20 steps: gap 4.2e-3, disagreement 1.0%). One epoch of average-and-continue costs 0.001-0.016 s on the small data and 0.09-0.12 s on Adult.
* Newton gives the same model for every seed, so converged disagreement is 0 by construction; the question answered here is how many iterations and what it costs. The cost per iteration is a d x d Hessian (n d^2 flops), so it grows with the number of features: not tested beyond d = 101.

Statements describe statistical behaviour of models on public datasets; not accusations of bias; not credit advice.
