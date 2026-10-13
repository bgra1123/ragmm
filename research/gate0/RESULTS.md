# Gate 0 results

Design: `PLAN.md` (pre-registered, committed before any run). Script: `run_gate0.py`. Raw: `results/<dataset>.json`.
L2 logistic regression, λ = 0.01, batch 16, K = 2,000 steps, 200 seeds, threshold 0.5, held-out rows only.
Runtime: under 1 minute per dataset on CPU.

## Gate verdict: **passed as pre-registered (6 of 6), with a caveat that changes the pitch**

| Dataset | Train rows | Exact-model AUC | Flips vs exact, SGD (α = 1/L) | Flips, Nesterov (α = 1/L) | Flips, SGD lr = 0.01 | Flips, tuned SGD + tail avg. | Two retrains disagree on (SGD 1/L → lr 0.01) | Gate |
|---|---|---|---|---|---|---|---|---|
| German Credit | 700 | 0.76 | **18.4%** | 24.6% | 2.2% | 1.9% | 25.1% → 2.9% | unstable |
| COMPAS | 4,320 | 0.73 | **21.5%** | 23.9% | 1.5% | 1.1% | 29.2% → 1.9% | unstable |
| Adult income | 46,842 | 0.89 | **13.6%** | 19.7% | 2.2% | 1.7% | 18.8% → 2.0% | unstable |
| Bank marketing | 39,188 | 0.80 | **5.6%** | 13.1% | 0.7% | 0.5% | 8.1% → 0.9% | unstable |
| Pima diabetes | 538 | 0.80 | **9.8%** | 14.6% | 1.7% | 0.4% | 14.0% → 0.9% | unstable |
| Heart disease | 212 | 0.88 | **5.2%** | 11.1% | 3.4% | 0.7% | 7.6% → 0.9% | unstable |

"Flips vs exact" = share of held-out rows whose decision differs from the exactly-solved model's, averaged over 200
retrains. "Two retrains disagree" = probability that two independent retrains give a random row different decisions.

**Caveat (the important part).** The ≥ 5% instability appears with the theory's step size α = 1/L, which on
standardized features is a learning rate of 0.66–1.9 (large). With a common conservative default (lr = 0.01, i.e.
α·L ≈ 0.005–0.015) the flips are **0.7–3.4%** and two retrains disagree on **0.9–2.9%** of rows: below the 5% threshold
on all six datasets. Tuned tail averaging is lower still (0.4–1.9%). Two further facts limit the story:

* Logistic regression is usually fitted with a deterministic solver (e.g. scikit-learn's L-BFGS), which has *no* seed
  instability at all. Seed noise matters for models trained by SGD/Adam: neural networks, large-scale or online models,
  fine-tuning.
* This gate only varied the **optimizer's randomness**. Real retraining also changes the **data** (a new month, a
  resample). That source was not measured.

So: "your model's decisions depend on luck" is true and large **when the step size is aggressive**, and small but
non-zero (1–3% of people; e.g. ~40 of 2,000 Adult rows between two retrains) with a careful default. The pitch
"98% of applicants depend on luck" from German Credit came from the aggressive regime and must not be generalized.

## Secondary results

* **Accuracy cost of instability.** With α = 1/L the retrains' mean AUC is up to 0.18 below the exact model's (0.002 for SGD on Heart; worst 5%:
  up to 0.36 below, Bank). With lr = 0.01 or tuned averaging the AUC matches the exact model within 0.01. So in this
  setting stability and accuracy come together; there is no trade-off to sell.
* **Gate 1 preview (not a gate here).** Tuned tail averaging cut flips by ≥ 85% relative to SGD (α = 1/L) on every
  dataset (paired bootstrap CIs exclude zero everywhere), but only by 0.3–2.6 points relative to lr = 0.01.
* **RA-GMM (ε = 0.25)** sits between the two (3.3–13.3% flips), as in Experiment 1: better than textbook momentum, worse
  than a careful default. Consistent with all earlier results; not a selling point.

## What this means for the plan (`research/mcp_strategy/PLAN.md`)

The gate's letter says build. Its spirit (failure mode 1: "practitioners already avoid the problem") is half-triggered:
a careful default already avoids most optimizer-seed instability for logistic regression. Recommended before building
the MVP, as **Gate 0b** (pre-register first, about an hour of compute):

1. **Data-resampling instability with an exact solver:** retrain the exactly-solved model on bootstrap resamples (and on
   a time-ordered "next period" split where a date exists: Bank has months, COMPAS has screening dates). If two
   retrains still disagree on ≥ 5% of rows, the product story becomes model-agnostic and does not depend on a bad
   step size: "even a perfect optimizer gives different people different answers next month".
2. **A small neural network with Adam at library defaults** (lr 1e-3, 2–3 epochs, batch 32) on the same datasets, where
   seed noise is not removable by switching to L-BFGS.

If (1) or (2) shows ≥ 5% on at least 3 of 6 datasets, build Retrain Audit around that source (pivot 2 in the plan:
model-agnostic harness). If neither does, the honest product is a lightweight check ("your instability is ~1–3%;
here are the rows") and the content is the finding itself: instability is mostly a step-size problem, fixed by a
careful default.
