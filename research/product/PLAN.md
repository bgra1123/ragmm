# Product benchmark: the packaged StabilityAdapter on German Credit

Status: PRE-REGISTRATION, written before any run. Code: `benchmark.py`. Results will go to `RESULTS.md`.

## Question
Does the adapter as shipped in `risk_averse_tuning` 0.4.0 (default settings, no tuning) reduce how much decisions
depend on the training seed, overall and for the pre-registered groups of `research/groups/PLAN.md`, and what does it cost
in training time? Earlier research numbers (research/sota2, research/groups) used research code; this checks the
product code path a user would run.

## Design
- Data: bundled German Credit (1,000 applicants). 5 stratified splits (300 held out, seeds 0-4, same rule as the research).
- Encoding: `TabularEncoder` (one-hot + standardize), intercept column dropped (sklearn fits its own).
- 30 retrains per model per split (seeds 0-29). Decision cutoff 0.5.
- Models (the `ragmm-adapter` CLI defaults): SGD logistic (5 epochs, alpha 0.01), XGBoost, LightGBM, random forest, MLP.
- Adapter: package defaults. SGD -> polish (20 full-batch epochs, exact line search). Others -> average of 10 seeds.
  Secondary rows: SGD with the `entropic` step (theta = 1, the risk-averse line search).
- Equal-compute comparisons (efficiency): SGD trained 25x longer (125 epochs, about the cost of 5 epochs + 20 full passes
  with line search); XGBoost / LightGBM with 10x the trees and random forest with 10x the trees (the same compute as averaging
  10 copies).
- Metrics (package `stability_check`): chance two retrains disagree; unstable rows (>=10% of retrains differ from the
  majority); decisions changed per retrain (mean / worst 5%); chance of a wrong decision; part due to training randomness
  (= one retrain minus the majority vote); seconds per fit. Pooled over the 5 splits by row count.
- Groups: G1-G10 of research/groups (same definitions), reported only when pooled size >= 30.

## Hypotheses
- H1: the polish brings SGD disagreement below 0.5% (from >5%).
- H2: averaging cuts XGBoost, LightGBM, RF and MLP disagreement by at least half.
- H3: the equal-compute alternatives (longer SGD, more trees) reduce disagreement less than the adapter.
- H4: for the groups with the largest "due to randomness" share before the adapter, the adapter reduces that share.
- H5: accuracy (wrong-decision rate of one retrain) does not get worse by more than 1 point with the adapter.

Misses are reported as misses. Statements describe statistical model behaviour on a public dataset; not accusations of
bias and not credit advice.
