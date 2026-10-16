# A risk-averse adapter for models you already trained

Status: design, not implemented. Evidence cited is from this repository; items marked **untested** are proposals.

## 1. What problem the adapter solves, and what it cannot

A trained model's output varies from run to run for two different reasons:

| Source | Example | Does a post-training adapter fix it? |
|---|---|---|
| **Optimization noise in the last stage** | SGD/Adam/ADMM stopped early; minibatch order | **Yes.** `research/polish`, `research/optimizers`: polishing a strongly convex head cut the worst-case loss 6.5–64× (median 6.7× over 72 optimizer settings) and never made one worse. |
| **Representation / structure noise** | different tree structures per seed (XGBoost subsampling); different learned features in a network | **No.** Refitting a last layer on XGBoost leaves did not reduce disagreement (German 5.5% → 6.2%, Pima 4.0% → 4.6%, Heart 1.6% → 1.4%; `research/trees`). Seed-averaging did (German 5.5% → 0.8% with 20 seeds). |
| **Data change** (new month, resample) | retraining on different rows | Not measured here. |

So the adapter is a **head polisher** plus an honest **diagnostic**; for models with structure noise it must be paired with
seed-averaging (or fixed seeds), and the report says which source dominates.

## 2. The formulation

Given a trained model with a feature map `φ(x) ∈ R^d` (what the model computes just before its final linear/logistic layer)
and training rows `(x_i, y_i)`:

1. **Head objective (strongly convex, smooth):**
   `f(w) = (1/n) Σ_i ℓ(y_i, wᵀφ(x_i)) + (λ/2)‖w‖²`, `ℓ` = logistic loss (or squared / softmax for other tasks), `λ > 0` gives
   `μ = λ`, and `L ≤ λ_max(ΦᵀΦ/n)/4 + λ` for logistic loss.
2. **Starting point:** the model's own head `w₁` (if its last layer is linear) or the output of a quick fit. The adapter never discards
   the original model; it returns `w_K` and the original.
3. **Polish with a fixed budget `K`** steps of minibatch GMM (`x ← x + d`, `d ← βd − α∇f(y)`, look-ahead `γ`) with parameters
   chosen to minimize the **worst-case tail of the final objective**:
   `(α, β, γ) = argmin EV@R_{1−ζ}[ f(w_K) − f* ]`, with `EV@R = inf_{t>0} (1/t) log(E[e^{tX}]/ζ)`.
4. **Local quadratic model for the choice** (validated in `research/selection`, `research/polish/noise_check`):
   `f(w) − f* ≈ ½ eᵀHe`, with `H = ∇²f(w₁)`, gradient-noise covariance `Σ/b` from the per-row gradients, and starting error
   `e₀ = −H⁻¹∇f(w₁)` (one Newton step, no exact solve). GMM on this model is linear with additive noise, so EV@R of the final
   loss has a closed form (paper Section 3; finite-horizon extension in `risk_averse_tuning.quadratic.evar_quadratic`).
   Gaussian noise is a convenience only: the real-noise check stayed within 0.93–1.35× (median 1.06×).
5. **Certificate and fallback:** report the predicted EV@R, the pre/post values on a held-out split, and **return the original
   head if the polish does not lower the held-out tail** (a never-worse guarantee at the level of the report).

## 3. Interfaces: what the adapter needs from "your model"

```python
class FeatureMap(Protocol):
    def features(self, X) -> np.ndarray: ...        # φ(X), shape (n, d)
    def head(self) -> np.ndarray | None: ...        # existing last-layer weights, if linear
```

| Model family | `φ(x)` | Head start | Polish fixes | Status |
|---|---|---|---|---|
| Logistic / linear | the features | the trained weights | optimization noise | **tested** (all experiments) |
| Neural net (MLP, TabNet, FT-Transformer) | penultimate activations (frozen) | existing last layer | last-layer optimization noise only; not feature learning | **untested** (next experiment) |
| Gradient-boosted trees | leaf one-hot, or margin + leaf indicators | refit from scratch | little: structure noise dominates | **tested, negative** for the refit; use seed-averaging |
| Any black-box scorer | its score `s(x)` (+ a few engineered features) | 1-D logistic calibration | calibration noise only | **untested**, expected small gain |
| LLM / embedding classifier | embedding | linear probe | probe training noise | **untested** |

## 4. Algorithm

```
adapt(model, data, budget K, ζ):
  1. φ = model.features(data); split data: fit / validation (validation never used for choosing parameters)
  2. w1 = model.head() or exact fit of the head (Newton for d ≤ ~5,000, L-BFGS above)
  3. estimate on fit rows at w1: L, μ, H (exact, or low-rank + diagonal for large d), Σ, e0
  4. candidates: α ∈ {0.003..1}/L × (β, γ) grid as in `research/polish`
     choose by exact-EV@R (quadratic model); optionally verify the top 3 with a few real polish runs
  5. run K polish steps from w1 (several seeds if a distribution is wanted)
  6. report on validation: tail of loss, changed decisions between polishes, accuracy before/after; return w_K or w1
```

Cost: one pass of feature extraction, one Hessian (d² memory), a few thousand cheap head steps. No retraining of the model body.

## 5. How this differs from the paper

* The paper trains with RA-GMM from scratch (problem 51, or 32 for quadratics). Here RA-GMM's parameters are **not** used:
  as a polisher, problem (51) was worse than doing nothing in 50 of 72 cells (`research/optimizers/RESULTS.md`).
* What is kept is the paper's **risk measure and the quadratic-model machinery**: EV@R of the final loss, closed form for linear
  systems, with the budget `K` and the noise level as inputs (the stationary formulation ignores both and failed on the portfolio problem).

## 6. Risks and pivots

| Risk | Signal | Pivot |
|---|---|---|
| Last layer of a neural net is already near-optimal; polish gains nothing | < 20% tail reduction on the planned MLP experiment | Position the adapter as the **diagnostic** (tail report with leverage) plus seed-averaging; drop the polish claim for networks. |
| Head dimension too large for `H` | `d > ~10⁴` | Diagonal + low-rank (Lanczos) curvature, or polish a random-projection subspace; test the loss in selection regret against the exact `H` on `d ≈ 1,000`. |
| Gains come from over-fitting the head on the training rows | validation loss/tail worse after polish | The fallback in §2.5 returns the original; add early stopping on validation. |
| Users expect it to stabilize trees | refit shows no gain | Ship seed-averaging as the tree-model recommendation and say why. |
| Pilot error misleads parameter choice (seen in `research/selection`: regret up to 3.2×) | repeated selections disagree | Repeat selection on several pilot subsamples and take the most common choice; verify the top 3 on real runs. |

## 7. Validation plan (each pre-registered before running)

1. **E3 (epoch-based):** logistic stage 1 with SGD/Adam/ADMM for 5 epochs, polish 1–2 epochs; compare with continue/decay/average and
   the careful baseline (`research/polish/ADAPTER_PLAN.md`).
2. **MLP last-layer:** a small MLP (2 hidden layers) trained with Adam on the credit datasets with 100 seeds; freeze the body, polish the
   head; measure disagreement of decisions and tail before/after; control: simply averaging the head over the last epoch.
3. **Trees + seeds:** XGBoost seed-average of m models, then polish a head over the concatenated leaf features of the *ensemble*.
4. **Large head:** the same on a ~1,000-dimensional embedding (e.g. random projection of the Adult features) to test the low-rank curvature.
5. **End-to-end package test:** `adapt()` on the German credit data reproduces the research numbers within tolerance.

## 8. Package surface (if validation 1 and 2 pass)

* `risk_averse_tuning.adapt(features, y, head=None, budget=..., zeta=0.05) -> AdaptResult` (new head, chosen parameters, predicted
  and measured tail, fallback flag);
* `FeatureMap` helpers for scikit-learn pipelines, XGBoost/LightGBM leaves, and PyTorch modules (hook on the penultimate layer);
* CLI: `ragmm polish model.json data.csv --epochs 2 --report tail.md`;
* the tail report gains "before / after polish" and the source-of-variance split (head noise vs structure noise vs data).
