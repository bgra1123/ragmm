# Make your classifier give the same answer twice: a tutorial

References in brackets point to [REFERENCES.md](REFERENCES.md). Numbers marked "benchmark" come from the research repository ([EVIDENCE.md](EVIDENCE.md)).

Train the same model twice on the same data with a different random seed, and some of its decisions change. On a public
credit dataset [12], one retrain of a standard XGBoost model [13] and the next disagreed on 6.5% of applicants, and a
logistic regression trained with five epochs of SGD disagreed on 15.5% (product benchmark, `ragmm/research/product/RESULTS.md`).
This is called predictive multiplicity [4, 5]. It is not a bug in one library: it comes from the training procedure [6, 7, 8].

This tutorial shows how to:

1. measure how much your model's decisions depend on the seed, overall and for the groups you care about;
2. add the **stability adapter** from the `seedproof` package to an existing model;
3. read the before/after numbers, and know when the adapter is the right fix and when it is not.

Everything below runs in a few minutes on a laptop. The complete script is [`examples/tutorial.py`](../examples/tutorial.py).

## 0. Install

```bash
pip install "seedproof[boosting]"        # numpy, pandas, scikit-learn, xgboost, lightgbm
```


## 1. Load data and split it

The package bundles German Credit [12]: 1,000 loan applications, 20 attributes, and whether the applicant repaid.

```python
from seedproof import TabularEncoder, load_german_credit
from seedproof.tabular import stratified_split

df = load_german_credit()
y = (df.pop("creditability") == "good").astype(int).to_numpy()   # 1 = repaid
test = stratified_split(2.0 * y - 1, 0.3, seed=0)                 # 300 held-out applicants
enc = TabularEncoder().fit(df[~test])                             # one-hot + standardize
X_train = enc.transform(df[~test])[:, :-1]                         # drop the intercept column; sklearn fits its own
X_test = enc.transform(df[test])[:, :-1]
y_train, y_test = y[~test], y[test]
```

Use your own features in place of `X_train` and `X_test`. Anything with `fit` and `predict_proba` works.

## 2. Measure first: `stability_check`

`stability_check` takes a function `seed -> unfitted model`. It retrains that model once per seed on the same data and looks at
the decisions on the held-out rows.

```python
from xgboost import XGBClassifier
from seedproof import stability_check

def make_xgb(seed):
    return XGBClassifier(n_estimators=200, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                         random_state=seed, verbosity=0, n_jobs=1)

groups = {"housing for free": (df[test].housing == "for free").to_numpy(),
          "no property": (df[test].property == "unknown / no property").to_numpy()}
before = stability_check(make_xgb, X_train, y_train, X_test, y_test, seeds=30, groups=groups, name="XGBoost")
print(before.to_markdown())
```

Output on this split:

```
- Chance that two retrains disagree on a row: 9.0%
- Unstable rows (decision differs from the majority in at least 10% of retrains): 20.7%
- Decisions that differ from the majority, per retrain: 18.8 on average, 24.0 in the worst 5% of retrains
- Chance of a wrong decision: 25.7% for one retrain, 24.0% for the majority vote (+1.7% due to training randomness)
```

What the numbers mean:

| Number | Meaning |
|---|---|
| Two retrains disagree | pick a row and two retrains at random; how often do they decide differently |
| Unstable rows | rows whose decision differs from their majority decision in at least 10% of retrains |
| Changes per retrain (worst 5%) | the bad days: how many decisions a retrain changes in its worst 5% of seeds |
| Due to randomness | wrong-decision rate of one retrain minus that of the majority vote of all retrains |

Groups give the same numbers for subsets of rows. Use groups you can justify in advance (we fixed ours before looking at
results: `ragmm/research/groups/PLAN.md`). Small groups are noisy. On this split "housing for free" has only 22 rows, so look at
several splits before you draw a conclusion about a group.

## 3. Add the adapter

```python
from seedproof import StabilityAdapter, compare_stability

model = StabilityAdapter(make_xgb(0), n_seeds=10).fit(X_train, y_train)
p_repay = model.predict_proba(X_test)[:, 1]
print(model.info_)          # {'strategy': 'average', 'fits': 10, 'fit_seconds': 0.38, 'calibrated': False}
```

The adapter picks its strategy from the model type:

| Your model | Strategy | What it does | Why |
|---|---|---|---|
| `SGDClassifier(loss="log_loss")`, `LogisticRegression`, `RiskAverseClassifier`, a network's last layer | `polish` | starts from your trained weights and takes 20 deterministic full-batch steps with a line search | the objective is strongly convex, so there is one answer; the seed only decides how close training stopped to it [1] |
| XGBoost, LightGBM, random forest, MLP, anything else | `average` | trains 10 copies with different seeds and averages their probabilities | the seed changes the model's structure, so there is no single answer to polish towards; averaging is bagging / deep ensembles [9, 10] |

To measure the adapter, wrap it in the same factory:

```python
after = stability_check(lambda s: StabilityAdapter(make_xgb(s), n_seeds=10, random_state=s),
                        X_train, y_train, X_test, y_test, seeds=30, groups=groups, name="XGBoost + adapter")
print(compare_stability([before, after]))
```

| Model | Two retrains disagree | Unstable rows | Changes per retrain (avg / worst 5%) | Wrong decision (one retrain) | Seconds per fit |
|---|---|---|---|---|---|
| XGBoost | 9.0% | 20.7% | 18.8 / 24.0 | 25.7% | 0.05 |
| XGBoost + adapter | 3.2% | 7.7% | 7.0 / 12.0 | 24.9% | 0.40 |

Disagreement fell by about two thirds, and the wrong-decision rate did not get worse. Training cost went up about 8 to 10 times.

## 4. Logistic models: polish, and the risk-averse step

For a logistic model the adapter does not need more copies. It continues from the trained weights:

```python
from sklearn.linear_model import SGDClassifier

def make_sgd(seed):
    return SGDClassifier(loss="log_loss", alpha=0.01, max_iter=5, tol=None, random_state=seed)

sgd = stability_check(make_sgd, X_train, y_train, X_test, y_test, seeds=30, name="SGD logistic")
pol = stability_check(lambda s: StabilityAdapter(make_sgd(s), random_state=s), X_train, y_train, X_test, y_test,
                      seeds=30, name="SGD + polish")
ent = stability_check(lambda s: StabilityAdapter(make_sgd(s), step="entropic", theta=1.0, random_state=s),
                      X_train, y_train, X_test, y_test, seeds=30, name="SGD + entropic polish")
```

| Model | Two retrains disagree | Unstable rows | Changes per retrain (avg / worst 5%) | Wrong decision (one retrain) | Seconds per fit |
|---|---|---|---|---|---|
| SGD logistic | 14.0% | 31.7% | 28.9 / 42.0 | 28.0% | 0.00 |
| SGD + polish | 0.1% | 0.3% | 0.3 / 1.0 | 25.9% | 0.05 |
| SGD + entropic polish | 0.0% | 0.0% | 0.0 / 0.0 | 26.0% | 0.04 |

Each polish step moves along the full gradient `g`. The step length is chosen by one of four rules (`step=`):

* `"line_search"` (default): the step that lowers the average training loss the most.
* `"entropic"`: the risk-averse line search. For each training row `i`, `a_i = gᵀ∇f_i(w)` is how fast a step of
  length α lowers that row's loss. The rule picks

  `α* = argmin_α (1/θ) log( (1/n) Σ_i exp(−θ α a_i) )`,

  the entropic risk [1, 2] of the per-row loss change. Rows the step would hurt get more weight than in a plain average.
  This rule is the entropic risk-averse line search [3], built on the entropic risk measure of the paper [1]. `θ` sets how
  cautious the step is (larger means more cautious).
* `"entropic_exact"`: a variant added in this package that uses the exact per-row loss change instead of the first-order one. It protects the rows a step makes worse: in one-step tests at θ = 5 it cut
  the 95th-percentile per-row loss increase by 84–91% compared with the plain line search, at the price of smaller steps (`ragmm/research/linesearch/RESULTS.md`).
* `"newton"`: solve the head exactly.

What the risk term does, and does not do (`ragmm/research/risk_effect/RESULTS.md`): at the same number of passes, the risk-averse step
reaches the same stability as the plain exact line search (differences of at most about 1 point). The stability gain over simply training one
more epoch comes from using the exact full-data gradient, not from the risk term. The risk term's measurable effect is on how many rows a step
makes worse, at the price of smaller steps. Against a plain line search shrunk to the same average step, the risk-averse step hurts slightly fewer rows
(a few thousandths of a loss unit at the 95th percentile, in 99 of 160 settings) and gives up some seed stability for it; it did not change the worst group's
wrong-decision rate (`ragmm/research/risk_effect/RESULTS.md`, step-matched controls).

**Large data.** The step search only needs sums over rows once the gradient is known, so it can be computed batch by batch:
`polish_logistic_streaming(w0, iter_batches(X, y, 10_000), lam, step="entropic")` reads the data twice per step and never holds `X` in memory
(`StabilityAdapter(..., batch_size=10_000)` does the same). With the default `search="newton"` the 1-D problem is solved on three length-n vectors, no further data pass;
on Adult (47,000 rows) 20 steps took 0.26 s, against 1.6 s for 20 epochs of minibatch SGD in the same NumPy setting (`ragmm/research/risk_effect/RESULTS.md`, addendum).
For a neural network, the per-row quantity `gᵀ∇f_i(w)` is a forward-mode JVP of the row's loss along `g`.

Why it works: L2-regularized logistic regression is strongly convex and smooth, the setting the paper's theory covers [1].
Every seed's model lies near the same optimum, and full-batch steps remove the noise that minibatches added.

## 5. Your own network: polish the last layer

For a neural network, take the activations before the last layer and the last-layer weights, and call the polish directly:

```python
from seedproof import polish_logistic

features = penultimate_activations_with_bias_column        # shape (n, d)
w_new, info = polish_logistic(w_last_layer, features, labels_plus_minus_one, lam=0.01, epochs=20, step="line_search")
```

Be clear about what this fixes. It removes the randomness of the last layer only. When the seed changes the learned features,
polishing the head does not help. On German Credit it did nothing for an MLP, and averaging seeds did
(`ragmm/research/sota2/RESULTS.md`).

## 6. Using a cost-based cutoff? Calibrate

German Credit's cost matrix says approving a defaulter costs 5 and refusing a good applicant costs 1. That makes the cutoff
"approve if p(repay) ≥ 5/6". This only works if the probabilities are calibrated. Tree ensembles usually are not.

```python
cal = StabilityAdapter(make_xgb(0), n_seeds=10, calibrate=True).fit(X_train, y_train)
approve = cal.predict_proba(X_test)[:, 1] >= 5 / 6
```

`calibrate=True` fits Platt scaling [11] on out-of-fold predictions. It costs 5 extra fits of the whole adapter. In the
research runs calibration lowered the expected cost of every model (`ragmm/research/sota2/RESULTS.md`).

## 7. One command for a CSV

```bash
seedproof applications.csv --target repaid --positive yes --model xgboost \
              --group-columns housing employment --seeds 20 --report stability.md
```

This writes a before/after report with a table per group value and plain sentences like:

> For rows in 'no property', a single retrain of XGBoost makes a wrong decision 20.9% of the time; the majority vote of
> 30 retrains does so 20.0% of the time, so 0.9 percentage points come from training randomness.

Models: `sgd`, `logistic`, `xgboost`, `lightgbm`, `random_forest`, `mlp`.

## 8. What to expect: the benchmark

Package defaults, German Credit, 5 splits × 300 held-out applicants, 30 retrains per configuration
(`ragmm/research/product/RESULTS.md`, pre-registered in `ragmm/research/product/PLAN.md`):

| Model | Disagreement before → after | Wrong decision (one retrain) before → after | Training cost |
|---|---|---|---|
| SGD logistic (polish) | 15.5% → 0.1% | 27.3% → 25.4% | +0.06 s per fit |
| XGBoost (average 10) | 6.5% → 2.1% | 24.8% → 24.5% | ×10 |
| LightGBM (average 10) | 7.2% → 2.2% | 25.4% → 25.1% | ×10 |
| Random forest (average 10) | 4.2% → 1.5% | 25.4% → 25.0% | ×10 |
| MLP (average 10) | 14.4% → 5.1% | 27.6% → 26.2% | ×10 |

For the SGD model, the groups most affected by the seed gained the most:

* unemployed applicants: 41.8% → 36.3% wrong decisions;
* applicants living in housing "for free": 35.1% → 30.1%;
* unskilled non-residents: 28.8% → 16.7% (36 cases, so treat this one with care).

What the benchmark also showed, and you should know before you choose:

* **Equal compute does as well for trees.** XGBoost, LightGBM and random forests with 10× more trees (and a lower learning
  rate for boosting) were as stable as averaging 10 seeds, at the same cost. Both are ways of averaging out randomness. Use
  whichever fits your pipeline. The adapter's advantage is that it wraps the model you already have and does not change your
  training recipe.
* **Training SGD 25× longer also works** (0.3% disagreement, and a little faster here). The polish wins when you cannot or
  do not want to retrain: it starts from the weights you have.

## 9. Limits

* All numbers come from one public 1990s dataset [12]. Measure your own data with `stability_check` before you assume an effect size.
* The adapter removes variation caused by the training seed. It does not change what the model learned from the data. The
  exactly trained model still makes many wrong decisions for some groups. That is a separate problem, which needs better data or features.
* The reports describe statistical behaviour of a model. They are not findings of bias and not legal, credit or compliance advice.

## Where this comes from

* The paper: Can & Gürbüzbalaban, *Entropic Risk-Averse Generalized Momentum Methods* [1]. It chooses optimizer settings to control the entropic value-at-risk [2] of the training error, not only its average. The `entropic` step option uses the same risk measure [3].
* The experiments behind each design choice and every number above: [EVIDENCE.md](EVIDENCE.md), and the research repository [bgra1123/ragmm](https://github.com/bgra1123/ragmm).
* What you can and cannot say about results: [CLAIMS.md](CLAIMS.md).
