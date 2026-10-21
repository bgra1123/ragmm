# How it works

## Where the variation comes from
Training randomness changes the model in two different ways, and the fix depends on which one dominates.

| Source | Example | What helps |
|---|---|---|
| **Optimization noise** in a convex model | SGD or Adam stopped early on a logistic model | Continue with exact full-batch steps, or solve the head exactly (Newton). The objective has one optimum, so every seed ends on the same model. |
| **Structure noise** | Different trees per seed (subsampling), different learned features in a network | Average several seeds (bagging). A gradient polish cannot help because there is no single optimum. More trees at a lower learning rate also works. |

## The measurement
`stability_check(make_model, X_train, y_train, X_test, y_test, seeds=..., groups=...)` retrains `make_model(seed)` for each seed on the same training data and reports on the held-out rows:
* *two retrains disagree*: the chance that two random retrains give a random row different decisions;
* *unstable rows*: rows whose decision differs from the majority in at least 10% of retrains;
* *changes per retrain*: decisions that differ from the majority, on average and in the worst 5% of retrains;
* with true labels: the chance of a wrong decision for one retrain, for the majority vote, and the difference (the part due to training randomness).
Groups give the same numbers for subsets of rows (groups of fewer than 15 rows are skipped).

## The polish
`polish_logistic` / `polish_logistic_streaming` start from the trained weights and take exact full-gradient steps on the L2 logistic objective. The step length is a one-dimensional convex problem, solved exactly:
* `line_search`: the step that lowers the mean loss most;
* `entropic`: minimizes the entropic risk of how the step changes each row's loss, `(1/θ) log mean_i exp(-θ α gᵀ∇f_i)`, a risk-averse variant that builds on the entropic risk measure of Can & Gürbüzbalaban (2025);
* `entropic_exact`: the same with the exact per-row loss change;
* `newton`: damped Newton on the full objective; converges in 3 to 7 iterations from a trained model and gives the same model for every seed.
The streaming versions read the data in batches (two passes per step, or two per Newton iteration), so the data never has to fit in memory.

## The averaging
`StabilityAdapter(..., strategy="average", n_seeds=10)` trains copies with different seeds and averages their probabilities. `calibrate=True` adds out-of-fold Platt scaling, which matters when decisions use a cost-based cutoff
rather than 0.5.

## What was found about the risk-averse step
In the research experiments the entropic step matched an exact line search on stability, and the gain over simply training longer came from using exact full-data gradients. It does make fewer rows worse per step than a long
step, but a plain smaller step does almost the same. It is therefore an option, not the default.
