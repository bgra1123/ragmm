# "Rejected by a Random Seed?" — Risk-Averse Training against Arbitrary Decisions

**Status:** Experiment 1 (German Credit) implemented; see `RESULTS.md` once run.
**Audience goal:** the first real-data story for a general audience. A loan
decision about a real person should not depend on a random number.

---

## 1. The problem

Training a model involves randomness: the order in which examples are seen,
which minibatch is drawn at each step, and (in private training) injected
noise. Retrain the same model on the same data with a different random seed
and some individual decisions change. This is known as **predictive
multiplicity** or **arbitrariness** (Marx, Calmon & Ustun 2020; Black,
Raghavan & Barocas 2022).

For a lender this means two equally "valid" models can disagree about the
same applicant. For the applicant it means the outcome may be decided by the
seed, not by their finances.

## 2. Why RA-GMM fits

- Credit scoring commonly uses **logistic regression**. With an L2 penalty it
  is smooth and strongly convex, so the RA-GMM theorems apply exactly:
  `L = λ_max(XᵀX)/(4n) + λ`, `μ = λ`.
- RA-GMM bounds the tail of `f(w_K) − f*` over training randomness. Strong
  convexity turns that into a bound on `‖w_K − w*‖`, and therefore on how far
  every applicant's score can drift from the ideal model's score.
- So, in principle, risk-averse training should make individual decisions
  **less dependent on the seed**, with no extra data. This plan measures
  whether it does.

## 3. Experiment 1: German Credit (real loan applications)

### Data
- Statlog German Credit, 1,000 real applications (see `data/README.md`).
- **Excluded from features:** `personal_status_and_sex`, `foreign_worker`,
  `age_in_years` (protected attributes lenders should not use).
- One-hot categorical attributes, standardize all columns with training
  statistics, add an intercept column.
- Fixed stratified split: 700 training applicants, **300 held-out applicants**
  (split seed 0). Decisions are studied on the 300 held-out applicants.

### Model and decision rule
- `f(w) = (1/n) Σ log(1 + exp(−yᵢ wᵀxᵢ)) + (λ/2)‖w‖²`, λ = 0.01 (main),
  sensitivity λ ∈ {0.003, 0.03}.
- Ideal model `w*`: Newton's method to gradient norm < 1e-10.
- **Approve** if `P(good | x) ≥ 5/6`, the Bayes threshold implied by the
  dataset's documented cost matrix (bad approved = 5, good rejected = 1).
  Sensitivity: threshold 0.5.

### Training randomness
- Minibatch SGD-style gradients, batch size 16, sampled with replacement.
- Initialization `w₀ = 0` (the usual default), so the **only** randomness is
  the minibatch sequence.
- **Common random numbers:** seed *s* gives the same minibatch sequence for
  every method, so methods are compared run by run.
- 200 evaluation seeds (0–199). Separate tuning seeds (10 000–10 019).
- Budgets K ∈ {500, 2000} iterations.

### Methods (all use the same gradient oracle and budget)
| Method | Parameters |
|---|---|
| SGD, textbook | α = 1/L |
| Nesterov, textbook | α = 1/L, β = γ = (√κ − 1)/(√κ + 1) |
| Heavy-ball, textbook | α = 1/L, β = (√κ − 1)/(√κ + 1), γ = 0 |
| SGD, tuned for mean | α from a grid, minimizing **mean** `f(w_K) − f*` on tuning seeds |
| Nesterov, tuned for mean | (α, β) from a grid, same criterion |
| SGD + tail averaging | Average of the last half of the iterates, α tuned for mean |
| **RA-GMM** | (α, β, γ) chosen by the repo's `RAGMM` (ζ = 0.05; also 0.01, 0.10), from L, μ, d only |

RA-GMM's parameters come from the library's grid search, which depends only
on (L, μ, d, ζ). They are applied through the same vectorized GMM update as the
baselines; a unit check confirms this update matches the library's `GMM`
optimizer step for step.

> **Deviation from the pre-registration (recorded before the main sweep).**
> A 20-seed smoke test showed that the library's `RAGMM` (v0.1.0) selects
> α ≈ 1e-6, β ≈ 0.9998, so the model barely trains (it rejects every
> applicant). Two differences from the paper cause this: the rate constraint
> uses an α-dependent benchmark `1 − √(αμ/L)` instead of the paper's fixed
> `ρ²_* = 1 − √(μ/L)`, which makes the constraint vacuous for small α; and
> β uses `1 − αψL` in the denominator where eq. (35b) has `1 − αψμ`.
> The experiment therefore uses `paper_tuning.py`, a direct implementation of
> Theorem 2, Theorem 3 (eq. 44) and problem (51). Settings: ζ = 0.05, φ = 0.99,
> rate trade-off ε = 0.25 (the paper's value; main) and ε = 0.05. The library's
> choice is still recorded in the results for transparency. Library
> parameters `zeta ∈ {0.01, 0.10}` are replaced by `ε ∈ {0.25, 0.05}` because ζ
> barely changes the selected parameters at this κ.

### Pre-registered metrics
Optimization (what the theory bounds), across the 200 seeds:
- `S = f(w_K) − f*`: mean, median, 95th and 99th percentile, CVaR₅% (mean of the worst 5%)
- `‖w_K − w*‖`: mean, 95th percentile

Decisions (what applicants experience), on the 300 held-out applicants:
- **Flips per retrain** `F_s`: number of applicants whose decision differs from
  the ideal model's decision in seed *s*. Report mean, 95th percentile, max.
- **Per-applicant flip probability** `p_i`: share of seeds in which applicant
  *i*'s decision differs from the ideal decision. Report mean and the number of
  applicants with `p_i ≥ 10%`.
- **Ambiguity:** share of applicants whose decision is not unanimous across the
  200 seeds.

Quality (so reliability is not bought with bad decisions):
- Test accuracy, expected cost per applicant (cost matrix), AUC: mean and worst 5%.

### Pre-registered hypotheses
- **H1:** RA-GMM has lower 95th-percentile and CVaR₅% of `S` than the textbook
  Nesterov and heavy-ball settings.
- **H2:** RA-GMM has fewer flips per retrain (mean and 95th percentile) than the
  textbook momentum settings.
- **H3 (open question):** how RA-GMM compares with the baselines **tuned for
  the mean** and with tail averaging. No direction is predicted; whatever is
  observed gets reported.

### Statistics
- Paired bootstrap (10 000 resamples over seeds) for differences in mean and
  95th percentile of `F_s` and `S` between RA-GMM and each baseline.

## 4. Later experiments (same code, more data)

| # | Dataset | Story |
|---|---|---|
| 2 | UCI Default of Credit Card Clients (30k real customers) | Same question at scale; needs UCI access |
| 3 | Same as 1–2 with DP-SGD (privacy noise) | "Privacy protection adds noise. Who controls the risk it creates?" |
| 4 | Public clinical dataset | "Would your diagnosis change if the model were retrained?" |
| 5 | Frozen-backbone features + convex head | Bridge to the prompt-injection plan |

## 5. Video plan tied to this experiment

1. **"Your loan was rejected because of a random number."** Shows the problem with
   real applicants using standard training only. Does not depend on RA-GMM winning.
2. **"I built a fix from my own research."** RA-GMM on the same applicants,
   with whatever the results show, including the comparison to tuned baselines.

## 6. Honest limitations
- 1,000 applicants is a small, old (1994) dataset. It is a clear, real
  illustration, not a claim about any current lender.
- A full-batch exact solver gives `w*` with no randomness at all. The setting
  matters where exact solves are not used: large data with SGD, streaming
  updates, or private (DP) training. Experiments 2–3 address this.
- Flips are measured relative to the ideal model `w*`, not relative to ground
  truth. Stability is not the same as correctness.

---

## 7. Experiment 1b: change the objective (λ) or tune the optimizer? (exploratory)

Added after Experiment 1; written down before running it.

**Question.** To make loan decisions less dependent on the training seed, is
it better to (A) keep the objective and tune the optimizer, or (B) change the
objective by increasing the L2 penalty λ and keep a default optimizer?

**Reference.** The model the lender set out to build: the ideal model
`w*(λ₀)` at the base penalty λ₀ = 0.01, and its decisions on the 300
held-out applicants (threshold 5/6).

**Arms** (200 seeds, batch 16, K = 2000, same minibatch sequences):
- **A (tune the optimizer, λ = λ₀):** SGD tuned for mean, SGD + tail averaging
  (tuned), RA-GMM (ε = 0.25), and the textbook SGD / Nesterov for reference.
- **B (change the objective):** λ ∈ {0.03, 0.1, 0.3, 1.0} with textbook SGD and
  textbook Nesterov (α = 1/L(λ)).
- **A + B:** SGD + tail averaging (tuned) at each λ.

**Metrics.**
- Flips per retrain against the **reference decisions** (mean, p95).
- **Bias flips:** applicants on which `w*(λ)` itself disagrees with `w*(λ₀)`,
  i.e. decisions changed by the objective, not by noise.
- Quality on held-out applicants: expected cost (cost matrix), AUC, approval rate.

**Prediction.** B reduces seed-dependence but adds bias flips and loses some
quality as λ grows; A reduces seed-dependence with no bias flips and no loss
in quality. Whatever is observed gets reported.

---

## 8. Experiment 1c: matched-family comparison (exploratory; written before running)

Suggested by the paper's own special cases: compare RA-GMM restricted to
**β = γ = 0 with SGD** and restricted to **γ = β with Nesterov's method**, so that
only the step-size rule differs within a family.

Same setting as Experiment 1: German Credit, λ = 0.01, threshold 5/6, batch 16,
K = 2000, 200 evaluation seeds, common random numbers, tuning seeds 10 000–10 019.

| Family | Method | Step size rule |
|---|---|---|
| β = γ = 0 | SGD (textbook) | α = 1/L |
| β = γ = 0 | SGD (tuned for mean) | α from a log grid, minimum mean suboptimality |
| γ = β | AGD (textbook) | α = 1/L, β = (√κ − 1)/(√κ + 1) |
| γ = β | AGD, β tied to α (tuned for mean) | α from a log grid, β = (1 − √(αμ))/(1 + √(αμ)); minimum mean suboptimality |
| γ = β | **RA-AGD** (ε = 0.05, ε = 0.25) | paper's problem (51) restricted to S_0, library `method="agd"`, ζ = 0.05 |
| general | RA-GMM (ε = 0.05) | problem (51), library (reference row) |

The paper gives the EV@R bound for GD (Theorem 4) but **no selection problem** for
GD in the strongly convex case, so there is no "RA-GD" row. A problem of type (51)
is degenerate for GD at this condition number (κ ≈ 99): with the benchmark of
(51b), ρ_*² = 1 − √(μ/L) = 0.899, GD's best rate is 0.960, so the constraint is
infeasible for ε < 0.068 and vacuous (minimizer at the smallest step size on the
grid) for ε ≥ 0.112; with GD's own best rate as the benchmark it is vacuous for
ε ≳ 4/κ ≈ 0.040. (Wording corrected after the first commit of this section; no
result had been produced.)

Metrics: as Experiment 1 (flips per retrain on the 300 held-out applicants, p95 of
`f(w_K) − f*`, expected cost, ambiguity).

Hypotheses:
- **H1:** RA-AGD (ε = 0.05) has fewer flips and a lower 95th percentile of
  suboptimality than textbook AGD (same family; only α and β differ).
- **H2 (open):** RA-AGD vs AGD with β tied to a tuned α.
- **H3:** within each family, tuning for the mean beats the textbook setting.
