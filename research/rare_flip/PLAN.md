# Do rare applicants flip more because the boundary moves more where data is thin?

Status: PRE-REGISTRATION, written before any run. Code: `rare_flip.py`; results will go to `RESULTS.md`.

## Claim under test (from the content idea "life is a realization of random seeds")
"Depending only on the random seed, the model's decision boundary changes more for rare incidents, so rare applicants are more likely to get a different decision."
Two things could make rare rows flip more: (a) they sit closer to the boundary on average (nothing to do with the seed), or (b) the boundary itself moves more in their direction, because the data pins the model down poorly there (low curvature, high leverage).
Only (b) supports the claim. The test separates them.

## Design
- Six datasets as in `research/gate0` (German, COMPAS, Adult, Bank, Pima, Heart), same splits, L2 logistic regression (λ = 0.01), 200 retrains of 2,000 minibatch steps (batch 16) at (i) the aggressive step 1/L and (ii) learning rate 0.01; plus (iii) the aggressive models after 20 exact line-search steps (the polish).
- Per held-out row i: flip_i = share of retrains whose decision differs from the exactly trained model's; std_i = standard deviation across retrains of the score x_iᵀw_k (how much the boundary moves at that row); |margin_i| = |x_iᵀw*| (distance to the exact boundary).
- Rarity: (1) k-NN rarity = mean distance to the 10 nearest training rows in the standardized feature space (all datasets); (2) rarest-category frequency = the training frequency of the row's least common category (datasets with categorical columns); (3) leverage h_i = p(1−p) x_iᵀH⁻¹x_i/n (the model-based measure of "poorly pinned down").
- Analysis: Spearman correlations of std_i with leverage and with rarity; within deciles of |margin| (so distance to the boundary is held fixed), the flip rate of the rarest 10% of rows (k-NN) against the rest, and the pooled ratio; partial Spearman of flip with rarity given rank(|margin|); 1,000-row bootstrap for the pooled ratio.

## Hypotheses (decided before running)
- H1 (the boundary moves more where data is thin): Spearman(std, leverage) ≥ 0.5 on at least 4 of 6 datasets and Spearman(std, k-NN rarity) ≥ 0.3 on at least 4 of 6, at step 1/L.
- H2 (rare rows flip more at the same distance from the boundary): the rarest 10% flip more often than the rest in at least 7 of 10 margin deciles on at least 4 of 6 datasets, and the pooled ratio (rarest 10% / rest, within deciles) is above 1.3 with a bootstrap CI above 1.
- H3: the same direction at learning rate 0.01, smaller magnitude.
- H4: after the polish the ratio is no longer distinguishable from 1 or flips are below 0.5% of rows, i.e. the excess is a training-randomness effect, not a property of the rows.
- If H2 fails while H1 holds, the honest statement is "rare rows are more often near the boundary" and the content must not say the boundary moves more for them. If the ratio is between 1 and 1.3, say "slightly".

## Guard rails
Groups used in content (housing "for free", unemployed, under 25) are descriptions of where decisions change, not claims about people. Statements describe statistical behaviour of models on public datasets; not accusations of bias; not credit advice.
