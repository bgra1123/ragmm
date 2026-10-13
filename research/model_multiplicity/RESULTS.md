# Results — Experiment 1: German Credit

**Run:** 200 seeds per method, batch size 16, budgets K ∈ {500, 2000},
λ ∈ {0.003, 0.01, 0.03}, 300 held-out real applicants. Design and
hypotheses: `PLAN.md` (pre-registered; one recorded deviation, see there).
Raw summaries: `results/summary_lam*.json`. Figures: `results/figures/`.

Reproduce: `python research/model_multiplicity/loan_experiment.py --lam 0.01`
(about 2 minutes on a laptop CPU), then `make_figures.py`.

## Headline numbers (λ = 0.01, κ ≈ 99, K = 2000, cost-based threshold 5/6)

| Method | Tuning runs needed | Flips per retrain: mean | Flips: worst 5% (p95) | Applicants with ≥10% flip chance | Ambiguous applicants | 95th pct of `f(w_K) − f*` | Expected cost / applicant |
|---|---|---|---|---|---|---|---|
| SGD (textbook) | 0 | 71.3 | 91 | 222 | 98% | 0.34 | 0.751 |
| Nesterov (textbook) | 0 | 99.8 | 125 | 257 | 100% | 2.1 | 0.916 |
| Heavy-ball (textbook) | 0 | 103.1 | 131 | 264 | 100% | 2.9 | 0.937 |
| SGD (tuned for mean) | 240 | 10.3 | 16 | 35 | 21% | 0.0034 | 0.552 |
| Nesterov (tuned for mean) | 1,440 | 10.9 | 16 | 39 | 23% | 0.0041 | 0.555 |
| SGD + tail averaging (tuned) | 240 | **6.6** | **10** | **22** | **16%** | **0.0014** | **0.550** |
| **RA-GMM (ε = 0.25)** | **0** | 50.8 | 65 | 158 | 91% | 0.12 | 0.659 |
| RA-GMM (ε = 0.05) | 0 | 91.2 | 112 | 251 | 99% | 1.0 | 0.867 |

"Flips" = held-out applicants whose approve/reject decision differs from the
ideal model `w*` in that retrain (out of 300). "Ambiguous" = applicants whose
decision is not the same in all 200 retrains. The ideal model approves 41% of
applicants and has test AUC 0.762.

## Hypotheses

| | Result |
|---|---|
| **H1** RA-GMM has a lower tail of `f(w_K) − f*` than textbook Nesterov / heavy-ball | **Supported** at every λ and budget. At λ = 0.01, K = 2000: p95 0.12 vs 2.1 (Nesterov); paired bootstrap difference −1.96, 95% CI [−2.03, −1.87]. |
| **H2** RA-GMM has fewer flips than textbook momentum | **Supported.** Mean flips 50.8 vs 99.8 (Nesterov), difference −49.0, CI [−51.2, −46.8]; p95 65 vs 125. Also beats textbook SGD (−20.4, CI [−22.4, −18.4]). |
| **H3** (open) RA-GMM vs baselines tuned for the mean and tail averaging | **RA-GMM is clearly worse.** Tail averaging: 6.6 vs 50.8 mean flips (difference +44.2, CI [+42.9, +45.5]). Same direction at every λ and budget. |

Sensitivity: the ranking is the same at λ = 0.003 and λ = 0.03 and with the
0.5 threshold. RA-GMM's gap to the tuned baselines is smallest at λ = 0.03
(30 vs 7–12 mean flips).

## Why RA-GMM loses to the tuned baselines (diagnosis)

1. **The rate constraint is vacuous at this condition number.** Problem (51)
   requires `ρ² ≤ (1 + ε) ρ²_*` with `ρ²_* = 1 − √(μ/L)`. Since `ρ² < 1`,
   the constraint can only bind when `ε < 1/ρ²_* − 1`. At κ ≈ 99 that is
   ε < 0.111, so **ε = 0.25 imposes no constraint at all**. (In the paper's
   experiments κ = 10, where ε = 0.25 does bind.)
2. **Without a binding constraint, the asymptotic bound keeps improving as
   α → 0.** The selected step size depends on grid resolution: α = 0.010
   (100 bins), 0.0015 (300 bins), 0.0003 (1,000 bins). On the 100-bin grid it
   lands on the smallest α available.
3. **The bound is asymptotic (k → ∞); training is not.** RA-GMM reaches its
   noise floor within ~30 iterations and then plateaus
   (`figures/tail_curves_*.png`). Along the K = 2000 runs it has the
   **lowest** 95th-percentile suboptimality of all methods up to ~40
   iterations (note: those baselines were tuned for K = 2000, not for short
   budgets, so this is not a fair short-budget comparison). The tuned
   baselines are chosen *for the actual budget K* and keep improving.
4. **The parametrization ties momentum to the step size.** Under (35b),
   β → 1 as α → 0, so RA-GMM cannot reach "small step, no momentum", the
   low-noise region that tuned SGD uses (α ≈ 0.016, β = 0 at K = 2000).
5. **Tuning budget differs.** RA-GMM uses zero training runs (only L, μ, d).
   The tuned baselines used 240–1,440 training runs on separate seeds. Against
   the zero-tuning alternatives (textbook settings), RA-GMM wins clearly.

## What this means

- **The problem is real.** On real loan applicants, a common default
  (Nesterov with textbook parameters) changes the decision for about a third of
  applicants from retrain to retrain. Even well-tuned training leaves 16–23% of
  applicants with a decision that is not unanimous across retrains.
- **RA-GMM, as specified in problem (51), halves arbitrary decisions relative to
  textbook defaults with no tuning runs**, but it is not competitive with a
  tuned step size plus iterate averaging.
- The diagnosis points to a concrete fix: choose the parameters by minimizing
  the **finite-horizon** EV@R bound at the actual budget K (Theorem 3,
  eqs. 42–43, including the `ρ̄^k · 2V_P(z₀)` term), and/or combine RA-GMM with
  tail averaging. Both are new, exploratory methods and must be reported as
  such.

## Library finding

`risk_averse_tuning.RAGMM` (v0.1.0) selects α ≈ 1e-6 on this problem (the
model barely trains). Its rate benchmark `1 − √(αμ/L)` depends on α, so the
constraint never binds, and its β formula uses `1 − αψL` where the paper's
(35b) has `1 − αψμ`. The library choice is recorded in each summary under
`ragmm_tuning`; it was not used as a method.

**Update:** both are fixed in `risk_averse_tuning` 0.2.0 (see the README). In
this setting (L = 0.9888, μ = 0.01, d = 55, ζ = 0.05, ε = 0.05) v0.1.0 chose
α = 1e-6, β = 0.9998, γ = 0.555; v0.2.0 chooses α = 0.3328, β = 0.8902,
γ = 0.7033, identical to the `paper_tuning.py` values used for the
"RA-GMM (eps=0.05)" rows above (the library is tested against `paper_tuning.py`
in `tests/test_risk.py`). The experiment results were not re-run: they used
`paper_tuning.py`, not the library.

## Limitations

- One small (1,000 applicants), old dataset; an illustration, not a statement
  about any current lender.
- Flips are measured against the ideal model `w*`, not against ground truth.
- Minibatch noise is not the additive Gaussian noise assumed by the theory.

---

# Experiment 1b (exploratory): change the objective or tune the optimizer?

Design written down before running: `PLAN.md` §7. Raw results:
`results/lambda_vs_tuning.json`. Figure: `results/figures/lambda_vs_tuning.png`.
Reproduce: `python research/model_multiplicity/lambda_vs_tuning.py` (~30 s).

Reference = the intended model `w*(λ = 0.01)`: approves 41% of held-out
applicants, expected cost 0.553 per applicant, AUC 0.762.
"Bias flips" = applicants on which the new objective's own ideal model `w*(λ)`
already disagrees with the intended model (no training noise involved).

| Arm | λ | Optimizer | Flips vs intended model (mean) | Seed-only flips (vs own ideal) | Bias flips | Expected cost | Approval rate |
|---|---|---|---|---|---|---|---|
| reference | 0.01 | SGD textbook | 71.3 | 71.3 | 0 | 0.751 | 52% |
| reference | 0.01 | Nesterov textbook | 99.8 | 99.8 | 0 | 0.916 | 62% |
| **A** | 0.01 | **SGD + tail averaging (tuned)** | **6.6** | 6.6 | **0** | **0.550** | 40% |
| A | 0.01 | SGD tuned for mean | 10.3 | 10.3 | 0 | 0.552 | 40% |
| A | 0.01 | RA-GMM (ε = 0.25) | 50.8 | 50.8 | 0 | 0.659 | 46% |
| B | 0.03 | SGD textbook | 70.8 | 75.3 | 26 | 0.718 | 46% |
| B | 0.1 | SGD textbook | 76.2 | 72.3 | 65 | 0.681 | 34% |
| B | 0.3 | SGD textbook | 94.6 | 46.9 | 113 | 0.672 | 17% |
| B | 1.0 | SGD textbook | 119.5 | 4.4 | 122 | 0.697 | 1% |
| A+B | 0.03 | SGD + tail averaging (tuned) | 26.6 | 7.2 | 26 | 0.532 | 32% |
| A+B | 0.1 | SGD + tail averaging (tuned) | 66.4 | 3.8 | 65 | 0.562 | 19% |
| A+B | 1.0 | SGD + tail averaging (tuned) | 122.0 | 0.0 | 122 | 0.700 | 0% |

(Nesterov-textbook rows for arm B follow the same pattern; see the JSON.)

## Findings

1. **Raising λ with a default optimizer does not remove seed-dependence at
   moderate λ.** At λ = 0.03–0.1, textbook SGD still flips 72–75 applicants
   per retrain relative to its own ideal model, about the same as at λ = 0.01.
   On top of that, the objective change itself moves 26–65 decisions.
2. **Raising λ only becomes "stable" by becoming degenerate.** At λ = 1 the
   retrains agree with each other (4 seed-only flips), because the model shrinks
   toward the base rate and approves almost nobody (1% approval vs the
   intended 41%). 122 of 300 decisions differ from the intended model, and
   expected cost rises from 0.553 to about 0.70.
3. **Tuning the optimizer removes seed-dependence without changing the
   model.** Tail averaging at λ = 0.01: 6.6 flips, zero bias flips, cost 0.550
   (the intended model's own cost is 0.553).
4. **The prediction in PLAN.md §7 holds**, more strongly than expected: B
   brought no stability gain at moderate λ, not just a gain paid for with bias.

## Caveats

- The approval threshold is fixed at 5/6 (cost matrix). Larger λ pulls
  predicted probabilities toward the 70% base rate, below that threshold,
  which drives approvals toward zero. A recalibrated threshold would make the
  large-λ arm less degenerate; that variant is not run here.
- λ = 0.01 is treated as the intended model. A mild increase (λ = 0.03) gives a
  slightly lower ideal cost on held-out applicants (0.540). Choosing λ is a
  legitimate modeling decision; the point here is that it is the wrong lever
  for training noise, because it changes decisions for reasons unrelated to
  the seed.
- The arm-B optimizers are textbook defaults; arm A+B shows that tuning is
  what removes the seed-only flips at every λ.

---

# Experiment 1c: matched-family comparison (SGD family and Nesterov/AGD family)

Design written before running: `PLAN.md` §8. Code: `matched_family_comparison.py`.
Raw results: `results/matched_family.json` (pre-registered tuning grid) and
`results/matched_family_extgrid.json` (wider grid, a documented sensitivity).
Same setting as Experiment 1 (λ = 0.01, K = 2000, batch 16, 200 seeds, 300 held-out
applicants, threshold 5/6, κ ≈ 99, d = 55).

Within each family only the step-size rule differs. RA-AGD is the paper's problem (51)
restricted to `γ = β` (the library's `method="agd"`, ζ = 0.05). There is no RA-GD row: the
paper gives the GD bound (Theorem 4) but no selection problem, and a (51)-type
constraint is degenerate for GD at this κ (see `PLAN.md` §8).

| Family | Method | α·L | β = γ | Flips per retrain: mean / p95 | 95th pct of `f(w_K) − f*` | Expected cost |
|---|---|---|---|---|---|---|
| β = γ = 0 | SGD (textbook) | 1 | 0 | 71.3 / 91 | 0.34 | 0.751 |
| β = γ = 0 | SGD (tuned for mean) | 0.016 | 0 | **10.3 / 16** | **0.0034** | **0.552** |
| γ = β | AGD (textbook) | 1 | 0.817 | 99.8 / 125 | 2.08 | 0.916 |
| γ = β | AGD, β tied to α, tuned (pre-registered grid [0.01, 2]/L) | 0.01 (**grid edge**) | 0.980 | 48.5 / 63 | 0.107 | 0.651 |
| γ = β | AGD, β tied to α, tuned (wider grid [1e-4, 2]/L) | 1e-4 (**grid edge**) | 0.998 | 23.1 / 30 | 0.012 | 0.544 |
| γ = β | **RA-AGD (ε = 0.05)** | 0.306 | 0.895 | 89.8 / 112 | 0.93 | 0.856 |
| γ = β | **RA-AGD (ε = 0.25)** | 0.001 (**grid edge**) | 0.994 | 27.7 / 39 | 0.031 | 0.589 |
| general | RA-GMM (ε = 0.05), library | 0.33 | 0.890 / γ 0.703 | 91.2 / 112 | 1.03 | 0.867 |

With the wider grid the SGD row becomes α·L = 0.011, 12.0 flips (interior optimum both times).

## Hypotheses

| | Result |
|---|---|
| **H1** RA-AGD (ε = 0.05) beats textbook AGD | **Supported, modestly.** Flips −10.0 [−11.6, −8.4] (89.8 vs 99.8); 95th percentile of suboptimality −1.15 [−1.24, −1.07] (0.93 vs 2.08). |
| **H2** (open) RA-AGD vs tuned tied-β AGD | RA-AGD (ε = 0.05) is **worse**: +41.3 flips [39.3, 43.4] against the pre-registered-grid tuned AGD. RA-AGD (ε = 0.25) looked **better** than that tuned AGD (27.7 vs 48.5), but this is a grid-edge artifact: with the wider grid the tuned AGD gets 23.1 flips, ahead of RA-AGD (ε = 0.25). |
| **H3** tuning for the mean beats the textbook setting in each family | **Supported.** SGD: −61.0 flips [−62.6, −59.4]; AGD: −51.4 [−53.6, −49.1] (pre-registered grid). |

## What this shows

1. **Within the AGD family, the risk-averse step size helps a little at ε = 0.05** (−10 flips), with zero tuning runs, consistent with the paper's claim of improving on standard AGD parameters.
2. **At ε = 0.25 the constraint is vacuous** (`1 + ε ≥ 1/ρ_*²` for ε ≥ 0.11 at this κ), so RA-AGD simply picks the smallest α on its grid (α·L = 0.001) with β = 0.994. That choice is edge-driven: a finer grid would pick a smaller α. It performs well here, but because the grid ends there, not because the criterion identified it.
3. **Every "tuned" AGD optimum sits at the grid edge:** with β tied to α, the mean-optimal setting at K = 2000 keeps moving toward α → 0, β → 1 (effective step α/(1 − β) ≈ 0.5/L at the pre-registered grid's edge and ≈ 0.05/L at the wider grid's edge, against 0.011–0.016/L for tuned SGD). The comparison of tuned AGD with anything is therefore limited by the grid; I report both grids.
4. **Momentum does not help in this noisy problem.** Tuned SGD (10–12 flips) beats every AGD-family variant (best 23.1) and every RA variant, in line with the paper's premise that momentum amplifies gradient noise.

## Post hoc: do the paper's bounds rank the methods correctly?

Not pre-registered; 7 methods, so treat as indicative. The asymptotic EV@R bounds for σ² = 1 (Theorem 4 for the GD family, eq. 44 for AGD/GMM), computed at each method's parameters, against the empirical results:

| Method | Bound (σ² = 1) | 95th pct of `f(w_K) − f*` | Mean flips |
|---|---|---|---|
| SGD (tuned) | 39 | 0.0034 | 10.3 |
| SGD (textbook) | 2,445 | 0.34 | 71.3 |
| AGD tied-β (tuned, pre-reg. grid) | 3,050 | 0.107 | 48.5 |
| RA-AGD (ε = 0.25) | 888 | 0.031 | 27.7 |
| RA-AGD (ε = 0.05) | 37,825 | 0.93 | 89.8 |
| RA-GMM (ε = 0.05) | 36,413 | 1.03 | 91.2 |
| AGD (textbook) | 56,182 | 2.08 | 99.8 |

Spearman rank correlation between the bound and the empirical 95th percentile: **0.93**;
with mean flips: **0.93**. The bounds are roughly 7,000–41,000 times larger than the empirical values in
absolute terms (valid but conservative, as in the quadratic check in
`paper_experiments/AUDIT.md`), yet their *ordering* of configurations is informative. This
suggests the weakness of RA-GMM in Experiments 1 and P1–P3 is in how problem (51) constrains
the search (rate constraint, asymptotic criterion, grid), not in the bound's ability to rank
parameter choices.

## Caveats

- Minibatch noise is not the additive Gaussian noise of the theory, and σ² was not estimated; the bound is used here only for ranking.
- The edge effects above mean the "tuned AGD" rows are not true optima of the AGD family.
- One dataset (1,000 applicants, 1994).
