# Choosing the polish parameters: Gaussian quadratic vs real-noise linearized vs direct simulation

Design: `PLAN.md` (pre-registered; no deviations except renaming the script from `selectors.py`, which shadowed a standard
library module). Code: `compare_selectors.py`; raw: `results/<dataset>.json`. Textbook SGD stage 1 (2,000 steps), polish K₂ = 500 steps,
42 candidates, 12 pilot runs per selector call, five independent pilot sets per dataset, evaluation on 300 fresh seeds.
**Regret** = evaluation EV@R of the chosen candidate ÷ that of the best of the 42 (1.0 = chose the best).

| Dataset | Selector | Regret per pilot set (5) | Mean / median / worst | Seconds per selection |
|---|---|---|---|---|
| German | B Gaussian quadratic | 1.0, 1.0, 1.0, 1.0, 1.0 | 1.00 / 1.0 / 1.0 | 44.7 |
| | C linearized, real noise | 1.5, 1.0, 1.5, 1.51, 1.5 | 1.40 / 1.5 / 1.5 | 11.9 |
| | A direct (real polish runs) | 1.0, 1.0, 1.0, 1.0, 1.0 | 1.00 / 1.0 / 1.0 | 11.5 |
| COMPAS | B | 3.18, 1.0, 1.0, 1.0, 1.0 | 1.44 / 1.0 / 3.18 | 4.8 |
| | C | 1.0, 1.0, 1.0, 1.0, 1.96 | 1.19 / 1.0 / 1.96 | 1.2 |
| | A | 3.14, 1.0, 1.0, 1.0, 1.96 | 1.62 / 1.0 / 3.14 | 5.8 |
| Pima | B | 1.06, 1.03, 1.06, 1.06, 1.03 | 1.05 / 1.06 / 1.06 | 4.9 |
| | C | 1.0, 1.06, 1.0, 1.06, 1.06 | 1.03 / 1.06 / 1.06 | 1.4 |
| | A | 1.0 (all five) | 1.00 / 1.0 / 1.0 | 4.5 |
| Heart | B | 1.0, 1.0, 1.0, 1.37, 1.0 | 1.07 / 1.0 / 1.37 | 11.9 |
| | C | 1.0 (all five) | 1.00 / 1.0 / 1.0 | 3.9 |
| | A | 1.0, 1.0, 1.0, 1.37, 1.0 | 1.07 / 1.0 / 1.37 | 6.9 |

The best candidate (oracle) is in the range α = 0.03–0.06 × 1/L with β = γ = 0.5 (German, COMPAS, Heart) or plain SGD (Pima).

## Hypotheses

* **H1 supported:** the median regret is ≤ 1.15 for B and A on all four datasets and for C on three (German, where C's median is 1.5).
  The worst single selection was 3.2× (COMPAS, one of five pilot sets, for both B and A).
* **H2 supported, as a tie:** C has a lower or equal mean regret than B on 3 of 4 datasets (Heart, COMPAS, Pima), and loses on
  German; averaged over datasets they are the same (B 1.14, C 1.16).
* **H3 false:** the direct selector A was not ≥ 20× slower; it cost 4.5–11.5 s against B's 4.8–44.7 s on these small datasets,
  because B's exact formula is evaluated in a Python loop per candidate (implementation, not theory). C was the fastest (1.2–11.9 s).
  A scales with the cost of one training run × candidates × pilot runs and would become expensive on a large model.
* **H4 false:** A was not the least stable: A and B fail on the same pilot sets (COMPAS set 1: 3.18 vs 3.14; Heart set 4: 1.37 vs 1.37),
  and C fails on different ones. Failures follow the pilot set, not the selector.
* **H5 false:** the three selectors choose the same step size in only 0–80% of pilot sets (Heart 80%, COMPAS 60%, German 20%,
  Pima 0%), though they often tie in tail quality (flat objective near the optimum).

## What this answers

1. **Dropping the Gaussian assumption did not improve the choice.** The direct simulation (A) and the real-noise linearized
   simulation (C), which are Gaussian-free, are no better than the Gaussian quadratic formula (B): all three pick a candidate within
   about 5% of the best in most cases. This agrees with the noise check (Gaussian EV@R within about ±10% for the small steps selected).
2. **What goes wrong is the pilot information, not the noise model.** When a selector fails, it fails by the same amount as
   the direct one, on the same pilot set: the estimated starting error and curvature from 12 runs misled both. The
   remedy is more or better pilots, not a different risk model.
3. **Estimating a tail from few paths adds its own error.** C (480 simulated paths) is biased low in the tail and lost on German;
   A (240 runs) has the same issue. The exact formula (B) has no sampling error, which is an advantage of the quadratic approach.
4. **Cost:** B's formula is cheap in flops but my implementation is slow; C is the fastest here; A requires real runs.

## Design recommendation for the tool

* **Default: quadratic approximation with estimated H, Σ and starting error (B), optionally the real-noise linearized check (C)
  when the Gaussian-vs-real ratio from `noise_check`-style simulation exceeds ~15%.** It is deterministic, cheap, and as accurate as the
  alternatives here.
* **Use more pilots** (or repeat the selection over several pilot sets and take the candidate most often chosen) to remove the
  occasional 1.4–3.2× misses.
* **Optional verification step (not tested here):** shortlist the top 3 candidates from B/C and decide among them with a few real polish
  runs (A on the shortlist). This is the likely best use of A; test it before claiming it.
* Direct optimization over the true risk (A) is the right answer only when linearization is doubtful (large steps, far from the
  optimum, non-convex models); not in this regime.

## Caveats

* Four datasets, logistic regression, one stage-1 solver (textbook SGD), one polish budget (500 steps), a 42-point grid.
* Five pilot sets per dataset; regrets of 1.0 in a flat objective can hide different parameter choices.
* Timings are for small datasets, one thread, with other jobs running.
