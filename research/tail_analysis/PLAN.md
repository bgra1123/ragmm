# Tail analysis: which people (data rows) carry the loss of the worst-5% models?

Written before running. Setup as Gate 0 (`research/gate0/PLAN.md`): six datasets, L2 logistic regression,
λ = 0.01, batch 16, K = 2,000, threshold 0.5; here **1,000 seeds** per method (same minibatches for every method).

## Question

Retraining gives a distribution of models. Take the **worst 5% of models** of a method (training objective
`f(w_K) − f*` above its 95th percentile, 50 of 1,000). Their extra loss is a sum over people:
`f(w_K) − f(w*) = (1/n) Σ_i [ℓ_i(w_K) − ℓ_i(w*)] + regularization`. For each training row i define its
**excess loss in the tail**, `X_i` = mean over the tail models of `ℓ_i(w_K) − ℓ_i(w*)`. Which rows have large `X_i`,
and what do they have in common? Does RA-GMM change who carries the loss?

## Row characteristics (computed once per dataset, from the data and the exact model w*)

* label; **margin** `y_i x_iᵀw*` (negative = the exact model gets the row wrong; near 0 = borderline);
* **leverage** `h_i = s_i(1−s_i) x_iᵀ H⁻¹ x_i / n`, H = Hessian of f at w* (how much the row's prediction moves when
  the model moves along poorly-determined, low-curvature directions);
* **rarity**: frequency in the training data of the row's rarest active one-hot category (1 if no categoricals);
* standardized mean differences (SMD) of every original feature between the top rows and the rest.

## Methods

SGD (textbook, α = 1/L); RA-GMM (ε = 0.25); **SGD matched to RA-GMM's mean** (α set by bisection on tuning seeds
10 000–10 099); SGD + tail averaging (tuned for mean).

## Analyses (per dataset × method)

1. **Concentration:** share of the tail's total excess loss `Σ_i X_i⁺` carried by the top 5% of rows by `X_i`.
2. **Profile of the top-5% rows vs the rest:** median margin, share misclassified by w*, median leverage ratio, median
   rarity, label mix, the 5 features with largest |SMD|.
3. **Tail vs body:** same `X_i` computed on the body models (the other 95%); does the tail hurt *different* people or
   the same people more? (Spearman of tail `X_i` vs body `X_i`; overlap of top-5% rows.)
4. **SGD vs RA-GMM:** overlap (Jaccard) of the top-5% rows; ratio of their tail excess loss on the same rows.
5. **Predictors:** Spearman of `X_i` with leverage, with |margin|, with rarity.

## Hypotheses

* **H1** Concentrated: top 5% of rows carry ≥ 25% of the tail's excess loss on ≥ 4 of 6 datasets (SGD textbook).
* **H2** The top rows are **high-leverage** rows (median leverage ≥ 2× the rest's) rather than simply the rows the exact
  model gets wrong; leverage predicts `X_i` better than |margin| (higher Spearman) on ≥ 4 of 6.
* **H3** The same people are hurt by every method's tail (Jaccard of SGD vs RA-GMM top rows ≥ 0.5), i.e. who is hurt is a
  property of the data, not the optimizer; the optimizer changes how much.
* **H4 (open)** RA-GMM's tail excess loss on those rows is lower than textbook SGD's (expected) and lower than
  matched-mean SGD's (the real test of thinner tails).
