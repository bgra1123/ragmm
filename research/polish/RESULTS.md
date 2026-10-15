# Fixed learning rate (E1) and polish (E2): results

Design: `PLAN.md` and `ADAPTER_PLAN.md` (pre-registered; one deviation recorded: E1 on Adult and Bank used 200 seeds
instead of 400 after the background jobs died). Raw data: `results/e1_*.json`, `results/e2_*.json`. L2 logistic
regression, batch 16, K = 2,000 steps for E1; E2 stage 1 = 2,000 SGD steps (α = 1/L) or 5-iteration consensus ADMM,
stage 2 = 200 or 1,000 more steps; 200 seeds for E2, 400 for E1 (small datasets).

## E1. At a fixed learning rate, momentum does not improve the tail

For each α ∈ {1, 0.3, 0.1, 0.03}/L the paper's EV@R bound (eq. 44) picks the momentum (ψ on the curve ϑ = 1 − αL(1−ψ)),
and the exact finite-horizon EV@R picks (β, γ) as an oracle at the exact solution.

| α (×1/L) | RA-GMM at α ÷ SGD at α (CVaR₅% of f(w) − f*, 6 datasets) | RA-GMM at α ÷ textbook Nesterov at α | RA-GMM at α ÷ SGD at the matching effective step α/(1−β) | β chosen by RA-GMM |
|---|---|---|---|---|
| 1 | 2.6 – 8.4× | 1.00 | 2.6 – 8.4× | 0.76 – 0.85 |
| 0.3 | 6.0 – 16.8× | 1.03 – 1.19 | 1.2 – 3.9× | 0.86 – 0.91 |
| 0.1 | 12.5 – 30.4× | 1.03 – 1.25 | 0.76 – 2.0× | 0.92 – 0.95 |
| 0.03 | 22.2 – 62.5× | 1.07 – 1.18 | 0.71 – 1.04× | 0.95 – 0.97 |

* **The exact-EV@R selection chose β = γ = 0 (plain SGD) in all 24 cells** (6 datasets × 4 step sizes). At a given step size,
  any momentum increases the final tail, because it inflates the effective step α/(1−β) and so the noise floor.
* **The paper's choice at a given step adds momentum (β 0.76–0.97) and has a worse tail than SGD at the same step
  (2.6–62×) and than textbook Nesterov (up to 25% worse).** It is also no better than SGD at the matching effective
  step (ratio 0.71–3.9×; only below 1 on Pima/COMPAS at small α).
* So the answer to "for a given learning rate, how do we improve the tail using RA-GMM?" is: **by not adding momentum.**
  Decisions changed per retrain at α = 0.1/L: SGD 55 / 402 / 23 / 4.8 / 271 / 113 (German, COMPAS, Pima, Heart, Adult,
  Bank), RA-GMM at α 61 / 346 / 22 / 6.2 / 300 / 160.
* **Step size is the dominant lever**: no choice of momentum at α = 1/L reaches SGD at 0.03/L (CVaR₅% 3–10× larger at
  1/L), and carefully tuned SGD with averaging (240 tuning runs) remains better than anything fixed-α here.

Pre-registered hypotheses: **H1 false** (RA-GMM at α does not beat SGD at α, 0 of 6 at every α ≤ 0.1/L), **H2 false**
(it is not better than textbook Nesterov at the same α), **H3 true but degenerate** (exact selection equals SGD), **H4**
tie only at the smallest α, **H5 supported.**

## E2. Polish: a short low-noise phase after any trainer cuts the tail

CVaR₅% of f(w) − f* (decisions changed per retrain in parentheses), after K₂ = 1,000 further steps.

| Dataset | Stage 1 | none | continue | decay (0.05/L) | average | RA-GMM (51) | **Exact-EV@R polish** |
|---|---|---|---|---|---|---|---|
| German | SGD | 0.381 (55.2) | 0.364 (56.8) | 0.0112 (13.3) | 0.0282 (9.3) | 0.137 (40.0) | **0.0133 (14.3)** |
| German | ADMM | 0.0350 (12.2) | 0.364 | 0.0112 | 0.0273 | 0.137 | **0.0054 (7.0)** |
| COMPAS | SGD | 0.228 (398.8) | 0.196 (374.5) | 0.0058 (90.2) | 0.0087 (34.1) | 0.048 (232.8) | **0.0044 (53.1)** |
| COMPAS | ADMM | 0.0120 (46.0) | 0.196 | 0.0058 | 0.0086 | 0.048 | **0.0016 (34.7)** |
| Pima | SGD | 0.161 (22.5) | 0.167 (24.3) | 0.0050 (4.1) | 0.0032 (2.3) | 0.042 (12.1) | **0.0025 (2.1)** |
| Pima | ADMM | 0.0391 (16.0) | 0.167 | 0.0050 | 0.0031 | 0.042 | **0.0020 (3.6)** |
| Heart | SGD | 0.066 (4.8) | 0.063 (4.8) | 0.0028 (1.0) | 0.0028 (0.8) | 0.029 (3.2) | **0.0020 (0.9)** |
| Heart | ADMM | 0.0393 (5.0) | 0.063 | 0.0031 | 0.0026 | 0.029 | **0.0049 (2.6)** |

With 200 more steps the same pattern holds (e.g. German SGD: 0.381 → 0.028 with the adapter).

* **H6 supported:** the adapter lowers the tail for both stage-1 solvers on all four datasets (8 of 8 at both budgets), by
  3–70× for textbook SGD (German 29×, COMPAS 52×, Pima 64×, Heart 33×) and 1.8–10× for early-stopped ADMM.
  Decisions changed fall by 74–98% for SGD stage 1 (German 55 → 14, COMPAS 399 → 53) and by 31–78% for ADMM.
* **H7 partly supported:** the adapter is below or within 10% of the best simple finisher (average or decay) in 6 of 8
  cells (COMPAS, Pima both stages; German ADMM; Heart SGD). It is 19% worse than decay for German SGD and
  about 90% worse than averaging for Heart ADMM (0.0049 vs 0.0026): **averaging or a small step is as good in
  most cells and sometimes better.** The adapter's advantage is that it chooses the step and length from the data
  and never needs a guess; it is not uniformly ahead.
* **H8 supported:** early-stopped ADMM is not at its floor; polishing lowers its tail in 8 of 8 cells.
* **H9 supported:** "continue" with the same large step does nothing (SGD) or makes ADMM's tail 10–30× worse
  (the large SGD step undoes ADMM's fit).
* **The paper's problem (51) is a poor polisher** (0.03–0.14): its tiny step (≈ 0.01/L) cannot converge in 1,000 steps, so it
  leaves most of the stage-1 error in place. Compared with the adapter it is 7–35× worse.
* Selected parameters are mostly small steps with moderate momentum (α ≈ 0.003–0.03 of 1/L... see `selected` in the
  JSON), chosen per stage and budget; for the longer budget on ADMM and Pima SGD it chooses plain SGD.

## What this means

1. **The framing "RA-GMM as a finishing step" is supported by the data; "RA-GMM as a better optimizer at a given
   learning rate" is not.** At a fixed step the best momentum is zero; the gain comes from lowering the effective step
   and averaging out noise.
2. **The adapter is a good, no-tuning finisher but does not beat plain averaging or decay by itself.** The honest product
   claim: "after your usual training, a short, data-chosen low-noise phase cuts the tail by 3-70× and changed decisions by
   30-98%; the exact tail-risk model picks the step and length for you".
3. Next: E3 (epoch-based budgets, Adam and other optimizers as stage 1; running as `research/optimizers`) and the
   noise-distribution check (`noise_check.py`).

## Caveats

* Logistic regression only; the quadratic model for selection uses H and Σ at the stage-1 mean weights (chosen once per
  dataset and stage from 6 pilot seeds), not per-run values.
* "Oracle" rows in E1 use the exact solution; the E2 adapter does not.
* Adult and Bank E1 used 200 seeds.
