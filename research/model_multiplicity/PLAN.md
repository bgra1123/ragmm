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
