# How to choose the polish parameters without the Gaussian assumption (written before running)

Question (user): build the tool without the Gaussian adaptation. Is it better to (A) optimize the parameters directly on the
entropic-risk (EV@R) objective, or (B/C) build a quadratic approximation and use estimated quantities?

## Three selectors, same candidate grid, same pilot information

Setting: stage 1 = 2,000 textbook SGD steps (α = 1/L, batch 16); stage 2 = polish of K₂ = 500 further steps with one of
42 candidates (α ∈ {.003, .01, .03, .1, .3, 1}/L × (β = γ = 0; or β ∈ {.5, .9, .98} with γ ∈ {0, β})).
**Pilot set:** 12 stage-1 runs (seeds 1000–1011) from which every selector works. Datasets: German, COMPAS, Pima, Heart.

| Selector | What it does | Gaussian? | Needs |
|---|---|---|---|
| **B: Gaussian quadratic** | exact EV@R of the linearized system with Gaussian noise (`quadratic.evar_quadratic`), H and Σ estimated at the pilot-mean weights, e₀ = Newton step from each pilot (averaged over pilots) | yes | H, Σ |
| **C: linearized, real noise** | simulate the same linear system but draw the noise from the actual per-row gradients (bootstrap, batch 16) at the pilot-mean weights; EV@R of the final quadratic loss over 40 paths per pilot (480 paths), same random numbers for every candidate | no (linearization only) | H, per-row gradients |
| **A: direct** | run each candidate's real nonlinear polish from each pilot with 20 fresh minibatch sequences each (240 runs); EV@R of the final f(w) (argmin does not need f*) | no, no linearization | actual pilot runs of the polish |

EV@R at ζ = 0.05 is the empirical `inf_t (log mean exp(t X) − log ζ)/t` (as in the tail report).

## Evaluation of what each selector chose

300 evaluation seeds (stage-1 seeds 0–299, polish minibatches from fresh seeds), the chosen candidate applied to every one.
Metrics: EV@R, CVaR₅% and mean of S = f(w) − f*; decisions changed vs the exact model. **Oracle-best** = the candidate with
the lowest evaluation EV@R among the 42 (not available to a user). **Regret** = evaluation EV@R of the chosen candidate ÷
oracle-best EV@R (1 = perfect). Selection **stability**: repeat the selection with 4 more disjoint pilot sets
(seeds 2000+12k); report the regret of each repeat and how often the chosen α agrees. **Cost**: wall-clock seconds of each selector.

## Hypotheses

* **H1** All three selectors reach regret ≤ 1.15 on ≥ 3 of 4 datasets (the Gaussian shortcut costs little).
* **H2** C has lower or equal regret than B on ≥ 3 of 4 datasets, but the difference is small (< 10 percentage points on average).
* **H3** A is ≥ 20× slower than B for selection.
* **H4** A's selection is the least stable across pilot sets (largest spread of regret) because empirical EV@R from 240 runs is noisy.
* **H5 (open)** The three agree on the chosen step size (same α) in most cases; they differ in momentum.

If H1 and H2 hold, the tool keeps B for the search and adds C as an optional check (the "verify" stage); A is not worth its cost.
If C beats B clearly, C becomes the default. Reported whichever way it falls.
