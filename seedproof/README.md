# seedproof

**Ask your model twice.** Measure how much your model's decisions depend on the training seed, overall and per group, and remove the part that can be removed.

> *seedproof is a working name.* The package, the command and the docs use one name throughout, so it can be changed in one pass (see [Renaming](#renaming)).

## The problem

Train the same model on the same data twice with a different random seed and some people get a different decision. On a public credit dataset (German Credit), a credit model
trained the standard fast way changed about 1 in 6 held-out decisions between retrains, and 284 of 300 applicants got a different answer in at least one of 1,000 retrains.
The effect lands hardest on applicants who look unlike most of the data. Among applicants who repaid and lived in housing "for free", **34.6% were refused with standard training
against 24.7% for the same model without the training randomness**. Most teams never measure this, so they cannot say how much of a refusal is the model and how much is luck.

## What seedproof does

1. **Measures it.** Retrains your model many times and reports, overall and per group: how often two retrains disagree, which rows are unstable, and the chance of a wrong decision,
   with the part caused by training randomness separated out.
2. **Removes what can be removed.**
   - *Logistic models* (`SGDClassifier(loss="log_loss")`, `LogisticRegression`, a network's last layer): continue from the trained weights with exact full-batch steps,
     or solve the head exactly with Newton's method. Every seed then lands on the same model.
   - *Trees and neural networks* (XGBoost, LightGBM, random forests, MLPs): average seeds, with optional calibration for cost-based cutoffs.
3. **Reports** in plain sentences, with the limits stated.

```python
from xgboost import XGBClassifier
from seedproof import StabilityAdapter, stability_check, compare_stability

make = lambda seed: XGBClassifier(subsample=0.8, random_state=seed)
before = stability_check(make, X_train, y_train, X_test, y_test, seeds=30, groups={"housing for free": mask}, name="XGBoost")
after = stability_check(lambda s: StabilityAdapter(make(s), n_seeds=10, random_state=s), X_train, y_train, X_test, y_test,
                        seeds=30, groups={"housing for free": mask}, name="XGBoost + adapter")
print(compare_stability([before, after]))
```

Command line, on a CSV:

```bash
pip install "seedproof[boosting]"
seedproof applications.csv --target repaid --positive yes --model xgboost --group-columns housing --report stability.md
```

## What the evidence says

Measured on German Credit (five splits, 30 retrains per configuration, package defaults; details in [docs/EVIDENCE.md](docs/EVIDENCE.md)):

| Model | Two retrains disagree, before → after | Wrong decision (one retrain), before → after |
|---|---|---|
| SGD logistic (exact polish) | 15.5% → 0.1% | 27.3% → 25.4% |
| XGBoost (average 10 seeds) | 6.5% → 2.1% | 24.8% → 24.5% |
| LightGBM (average 10 seeds) | 7.2% → 2.2% | 25.4% → 25.1% |
| Random forest (average 10 seeds) | 4.2% → 1.5% | 25.4% → 25.0% |
| MLP (average 10 seeds) | 14.4% → 5.1% | 27.6% → 26.2% |

## What it does not do

* It does **not** fix what the model learned. The exactly trained model still makes many wrong decisions for some groups; seedproof makes decisions reproducible, not fair.
* It makes **no claim of fairness, bias removal or regulatory compliance**, and its reports are statistical diagnostics, not findings of discrimination.
* For tree models, **10 times more trees at a lower learning rate is an equally good fix** at the same compute (benchmark); the adapter's advantage is that it wraps the model you already have and comes with the measurement.
* With a careful learning rate the effect is smaller (1-3% of decisions); the large numbers above belong to aggressive step sizes and SGD-trained models.
* One public 1990s dataset and mostly logistic regression, tree models and a small MLP. Measure your own model before assuming an effect size.

## Install

```bash
pip install seedproof                 # numpy, pandas
pip install "seedproof[boosting]"     # + scikit-learn, XGBoost, LightGBM
```

Python 3.9 or newer.

## Documentation

* [Tutorial](docs/TUTORIAL.md): measure, fix, read the numbers
* [Evidence](docs/EVIDENCE.md): what was tested, what worked, what did not
* [What you can say about your results](docs/CLAIMS.md): wording and limits
* [How it works](docs/METHOD.md)
* [References](docs/REFERENCES.md)

## Background and credit

seedproof builds on risk-averse optimization: Can, B. & Gürbüzbalaban, M. (2025). *Entropic risk-averse generalized momentum methods.* Optimization Methods and Software, 40(6), 1535-1583.
[doi:10.1080/10556788.2025.2549356](https://doi.org/10.1080/10556788.2025.2549356), [arXiv:2204.11292](https://arxiv.org/abs/2204.11292).
The experiments behind the design, including those that showed which ideas did not work, are in the research repository [bgra1123/ragmm](https://github.com/bgra1123/ragmm).
The optional `step="entropic"` polish uses the entropic risk measure from that work; the experiments found it matches an exact line search on stability, so it is an option, not the default.
Please cite the paper if you use this work (see [CITATION.cff](CITATION.cff)).

## Renaming

The name appears in `pyproject.toml`, `src/seedproof/`, the command name, the docs and the tests. `python scripts/rename.py NEWNAME` rewrites all of them.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT. Datasets: German Credit (Statlog), Hofmann (1994), UCI Machine Learning Repository, CC BY 4.0.
