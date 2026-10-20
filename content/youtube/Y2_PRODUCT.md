# Y2 — "A one-line fix for models that change their mind (and when it doesn't help)"

*Long-form YouTube script, about 14 minutes. Content ID **Y2**. Follows [Y1](Y1_PROBLEM.md). Written companion: the tutorial
[B2](../../docs/TUTORIAL.md). Numbers: [`../FACT_SHEET.md`](../FACT_SHEET.md) {IDs}. References: [`../REFERENCES.md`](../REFERENCES.md) [n].*

**Audience:** data scientists and ML engineers who ship classifiers (XGBoost, scikit-learn, PyTorch).
**Format:** talking head plus screen recording of `examples/tutorial.py`, plus benchmark tables as animated charts.

---

## Chapters

```
0:00 Recap: same data, different answer
1:15 Step 1: measure it (stability_check)
3:30 Step 2: the adapter, and why it has two modes
5:30 Logistic models: polish to the one right answer
7:30 The risk-averse step (from the paper)
9:00 XGBoost, LightGBM, random forest, neural nets: average
10:45 The benchmark: what it buys and what it costs
12:30 The honest part: equal compute
13:30 Try it
```

---

## 0:00 — Recap {A2, B2}

**VO:** "Last time: I retrained a credit model 1,000 times on the same data. Each retrain changed 55 of 300 decisions, and the
people least like everyone else, like applicants living in housing 'for free', paid the most: 37.8% wrong decisions against
27.4% without the randomness. Today: a tool that measures this for your model and fixes the part that can be fixed."

## 1:15 — Step 1: measure (screen recording)

```python
before = stability_check(make_xgb, X_train, y_train, X_test, y_test, seeds=30, groups=groups)
print(before.to_markdown())
```

**VO:** "You give it a function that builds your model from a seed. It retrains it, here 30 times, and tells you four things:
how often two retrains disagree, how many rows are unstable, how bad the worst retrains are, and how much of your error comes
from the seed rather than the data, overall and per group. For this XGBoost model, two retrains disagree on 9% of applicants."

**On screen:** the four definitions from the tutorial table (B2 section 2).

## 3:30 — Step 2: the adapter, and why two modes {C2}

```python
model = StabilityAdapter(XGBClassifier(...), n_seeds=10).fit(X_train, y_train)
```

**VO:** "One wrapper, two modes, chosen from the model type. Why two? Because there are two kinds of randomness. In a
logistic regression the problem is convex: one right answer exists, and the seed only decides how close training stopped. In a
boosted tree or a neural network the seed changes the structure, which trees get built and which features get learned. I
tried polishing the last layer of those models: it didn't help the network, and it made random forests worse, from 4.1% to
9.1% disagreement. So for them the adapter averages."

**On screen:** two-column table (polish vs average) from B2 section 3.

## 5:30 — Logistic models: polish {D1, D2, D6}

**VO:** "For a logistic model the adapter starts from your trained weights and takes 20 full-batch steps with a line search.
Five epochs of SGD disagree with themselves 15.5% of the time. After the polish: 0.1%. Unstable applicants go from 35% to
0.3%. And it costs about five hundredths of a second on this data."

**On screen:** bar race 15.5% → 0.1%.

## 7:30 — The risk-averse step [1, 2, 3] {E1, E2}

**On screen:**

```
α* = argmin_α (1/θ) log( (1/n) Σ_i exp(−θ α a_i) ),   a_i = gᵀ∇f_i(w)
```

**VO:** "Here's where the paper comes in. A normal line search picks the step that lowers the average loss. The entropic step
looks at every training row: how much does this step help or hurt that row? It then minimizes the entropic risk of those
changes, the same risk measure we used in the paper [1, 2]. Rows that would get hurt count more. A related idea in the
literature is tilted ERM [3]. In tests, the cautious version cut the worst per-row damage of a step by 84 to 91 percent, at the
price of smaller steps. With a few more passes it reaches the same optimum: after 20 passes the worst 5% of runs were within
5 in 100,000 of the optimum."

**On screen:** `StabilityAdapter(model, step="entropic", theta=1.0)`.

## 9:00 — Trees and networks: average [9, 10, 13, 14] {D4, D8}

**VO:** "For XGBoost [13], LightGBM [14], random forests and neural nets the adapter trains ten copies and averages them. That
is bagging [9] and deep ensembles [10]. Disagreement drops by about two thirds: XGBoost 6.5% to 2.1%, LightGBM 7.2% to 2.2%,
random forest 4.2% to 1.5%, the neural network 14.4% to 5.1%."

## 10:45 — The benchmark: gains and costs {D3, D5, D6}

**On screen:** the table from B2 section 8.

**VO:** "Who benefits most? For the logistic model, the groups the seed hurt most: unemployed applicants go from 41.8% to 36.3%
wrong decisions, 'housing for free' from 35.1% to 30.1%. Overall accuracy never got worse with the adapter, and no group got
worse by more than 0.9 points. The cost: almost nothing for the polish, ten times the training for averaging."

**Caveat on screen:** "One public dataset (German Credit, 1994). Pre-registered benchmark: research/product/PLAN.md."

## 12:30 — The honest part: equal compute {D7}

**VO:** "I pre-registered one more test, and the adapter didn't win it. If you spend the same compute differently, with ten
times more trees and a lower learning rate, XGBoost, LightGBM and random forests get just as stable as averaging ten seeds.
And training SGD 25 times longer gets close to the polish. So the gain comes from averaging out randomness, however you do it.
What the adapter gives you is that it works on the model you already have, without changing your training recipe, it picks the
right mode for you, and it measures the effect per group. If more trees fit your pipeline, use more trees, and measure with
`stability_check`."

## 13:30 — Try it

**VO:** "Everything is open source: the package, the tutorial, the benchmark, and every number in this video with its file. For
your own CSV it's one command."

```bash
ragmm-adapter applications.csv --target repaid --positive yes --model xgboost --group-columns housing --report stability.md
```

**End screen:** tutorial B2, Y1, shorts playlist.

---

## Description box

```
How to measure and reduce seed-to-seed instability in XGBoost, LightGBM, random forests, neural nets and logistic regression.

Tutorial: docs/TUTORIAL.md · Code: https://github.com/bgra1123/ragmm · Benchmark: research/product/RESULTS.md
Previous video (the problem): Y1

References
[1] Can & Gürbüzbalaban (2025). Entropic risk-averse generalized momentum methods. Optimization Methods and Software 40(6). doi:10.1080/10556788.2025.2549356 · arXiv:2204.11292
[2] Ahmadi-Javid (2012). Entropic value-at-risk. J. Optim. Theory Appl. 155(3).
[3] Li, Beirami, Sanjabi & Smith (2021). Tilted empirical risk minimization. ICLR.
[9] Breiman (1996). Bagging predictors.  [10] Lakshminarayanan, Pritzel & Blundell (2017). Deep ensembles. NeurIPS.
[11] Platt (1999). Probabilistic outputs for SVMs.  [13] Chen & Guestrin (2016). XGBoost. KDD.  [14] Ke et al. (2017). LightGBM. NeurIPS.
[12] Hofmann (1994). Statlog (German Credit Data). UCI. doi:10.24432/C5NC77

Statistical behaviour of models on a public dataset; not an accusation of bias; not legal or credit advice.
```
