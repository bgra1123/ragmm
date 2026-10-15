# Adam and other optimizers, and the polish adapter: results

Design: `PLAN.md` (pre-registered, no deviations). Raw: `results/<dataset>.json`. L2 logistic regression, batch 16,
2,000 stage-1 steps with each optimizer, then 1,000 polishing steps; 300 seeds; German, COMPAS, Pima, Heart; 18 optimizer
configurations per dataset (72 cells). Vectorized optimizers match `torch.optim` to 3e-16 on every dataset.

## Stage 1: the optimizers on their own

* **Learning rate matters more than the optimizer.** At lr 0.1 Adam, RMSprop and momentum SGD have worst-case loss
  (CVaR₅% of f(w) − f*) 9–17× that of plain SGD at the same lr (German 0.27 / 0.29 / 0.34 vs 0.0215); at lr 0.01 they are
  1.0–6.7× larger (Adam and RMSprop 1.5–6.7×, momentum 1.0–6.7×). Adagrad is the opposite, with its shrinking steps: at lr 0.1 it is the best
  stage-1 optimizer (0.2–0.3× of SGD) and at lr 0.001 it is badly under-trained (bias 0.9, CVaR₅% 0.17–0.23).
* **Defaults are not the safe point.** PyTorch's default lr for Adam (0.001) gives stable but biased models (bias
  0.14–0.24, disagreement 0.01–0.03); the best stage-1 results come at different lrs per optimizer
  (SGD 0.01, momentum/Nesterov 0.001, Adam 0.001–0.01, RMSprop 0.001, Adagrad 0.1).
* **H1 false:** ranking the 18 configurations by tail (CVaR₅%) vs by mean gives Spearman 0.93–0.98 on all datasets, so
  at this granularity the tail ranking is the mean ranking. (This differs from the portfolio study, where noise structure
  differed.)
* **H5 partly supported:** at lr ≥ 0.01 adaptive methods have a larger tail than SGD at the same lr for Adam and RMSprop
  (1.5–17.5×), not for Adagrad at lr 0.1 (0.2–0.3×).
* Run-to-run disagreement across the 18 configurations ranges 0.018–0.246 (German), 0.013–0.214 (COMPAS), 0.003–0.096
  (Pima), 0.000–0.097 (Heart): the instability is not specific to SGD; Adam at lr 0.1 is as unstable as momentum SGD at 0.1.

## Stage 2: the polish adapter (step and momentum from the exact EV@R model) vs simple finishers

Over all **72 cells**:

| Polisher | Cells where it lowers the tail vs no polish | Cells where it makes the tail worse |
|---|---|---|
| **Exact-EV@R polish (adapter)** | **72 / 72** (median 6.7×, range 1.9–57×) | **0** |
| decay (0.05/L) | 51 / 72 | 21 |
| average (SGD 1/L, averaged) | 42 / 72 | 30 |
| RA-GMM (51) | 22 / 72 | 50 |

(Counts are over all 72 cells; the median/range line is over the 72 cells with stage-1 CVaR₅% > 0.001, which is all of them.)

* **H2 supported (72/72).** The adapter never made a cell worse. Fixed finishers do: decay or averaging **raise** the tail
  when stage 1 was already good (e.g. German SGD lr 0.01: tail 0.0032 → 0.011 with decay, 0.027 with averaging, 0.0016 with
  the adapter), because they impose a fixed noise floor. The adapter chooses the step and momentum from the data and lowers
  or keeps the floor.
* **H3 supported (69/72):** the adapter is below or within 20% of the better of {average, decay} in 69 of 72 cells; in most
  it is far below (German: 0.0016–0.0097 vs decay 0.0110 and averaging 0.027).
* **H4 supported (55/63):** where stage 1 changed ≥ 3 decisions, the adapter cut mean changed decisions by ≥ 30% in 55 of
  63 cells (e.g. German Adam lr 0.1: 51.2 → 10.9; COMPAS momentum lr 0.1: 279 → 39; Pima momentum lr 0.1: 15.3 → 1.7).
* **Retrain disagreement after the adapter** is 0.019–0.049 (German), 0.014–0.031 (COMPAS), 0.004–0.012 (Pima), 0.005–0.016
  (Heart), regardless of the stage-1 optimizer: the polish **washes out the choice of optimizer** and leaves the same
  low floor. Stage-1 disagreement ranged up to 0.25.
* **H6 supported:** where stage 1 was stable but biased (small lr or Adagrad at 0.001, bias 0.7–0.9), the adapter
  removes most of the bias and the tail (e.g. German SGD lr 0.001: 0.082 → 0.0076; Adagrad lr 0.001: 0.167 → 0.0097).
* **The paper's problem (51) as a polisher is worse than doing nothing in 50 of 72 cells** (its tiny step cannot converge in 1,000
  steps) and is never the best.

## What this means

1. **Instability is a feature of every optimizer here, not only SGD**, and Adam's defaults do not protect against it.
2. **The adapter is stable across optimizers: it never hurt, usually helped a lot (median 6.7× lower tail), and
   makes the final model nearly independent of which optimizer ran first.** It beats fixed polishing recipes because it
   is data-driven, which is the supported role for RA-GMM-style machinery here.
3. For the credit product: "train with whatever you use, then run `polish`" is justified on logistic models. Tree ensembles
   are tested separately (`research/trees`).

## Caveats

* Logistic regression only; one polish budget (1,000 steps); parameters chosen once per (dataset, stage-1 configuration)
  from 6 pilot seeds.
* Cells are not independent (same data, seeds and polish noise); the counts are descriptive.
* Stage-2 budget is in steps, not epochs (German: 1,000 steps × 16 = 23 epochs of 700 rows; Heart 75 epochs). The
  epoch-based version in `research/polish/ADAPTER_PLAN.md` (E3) is still to run.
