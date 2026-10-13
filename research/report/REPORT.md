# Same applicant, same data, different answer

### What retraining does to loan decisions, who it affects most, and what a risk-averse training method can do about it

*Data: German Credit (1,000 real loan applications from a German bank, a standard public dataset). All numbers are
reproducible with the commands at the end. Analysis plan written before running: `PLAN.md`.*

---

## 1. The problem: a model is not one model

When a bank trains a credit-scoring model, the training process uses randomness: it looks at the applicants in a
random order, a few at a time, and nudges the model after each small batch. Run the training again on **exactly the
same data** and you get a slightly different model.

How different? We trained the same simple, widely used credit model (logistic regression) **1,000 times** on the same
700 past applicants, changing nothing but the random order, and asked each of the 1,000 models to decide on 300 new
applicants it had never seen (approve if it thinks the applicant is at least 50% likely to repay).

With the textbook training settings:

* each retrained model decided **55 of the 300 applicants (18%) differently** from the exactly-trained model;
* **284 of the 300 applicants** got a different answer from at least one retrain; only 16 got the same answer every time;
* **175 applicants (58%) were "coin-flip" applicants**: their decision changed in at least one retrain out of ten.

![How many decisions depend on the training run](figures/decisions_by_method.png)

Nothing about these applicants changed. Only the luck of the training run did.

## 2. Why look at the tail, not the average

Averages hide bad days. A bank does not retrain once; it retrains every month or quarter. Most retrains are fine.
Some are noticeably worse. The question a risk manager asks is not "how good is a typical retrain?" but **"how bad is
a bad one, and who pays for it?"**

In this report "tail" has two concrete meanings:

* **Tail retrains (the bad months).** Score each retrain by how well it fits the training data (its training loss;
  lower is better). The **tail** is the worst 1 in 20 retrains (5%). In those, the textbook model changed **67
  decisions** instead of the usual 55, and its accuracy fell from 69.1% to 67.9%.
* **Tail applicants (who pays in the bad months).** In those worst retrains, the extra error is not spread evenly.
  For each applicant we measure how much worse the bad retrains are for them than the exactly-trained model. The
  **tail applicants** are the 5% (15 of 300) for whom it is worst.

## 3. Who are the tail applicants?

They are not random. Compared with the other applicants:

| Characteristic | Tail applicants | Everyone else |
|---|---|---|
| Currently unemployed | **60%** | 6% |
| Job: "unemployed / unskilled, non-resident" | **33%** | 2% |
| Job: "management / self-employed / highly qualified" | **47%** | 12% |
| No property (or unknown) | **40%** | 10% |
| Housed for free (not owner, not renter) | **33%** | 6% |
| Job: "skilled employee / official" (the most common job) | 13% | 65% |
| Exactly-trained model's probability of repaying (median) | **0.53** (close to the line) | 0.79 |

The common thread: **they are unusual compared with most past applicants.** Few people in the data share their
combination of job, employment and housing, so the data pins down how to score them only loosely. A statistician
would say they have **high leverage**: their score depends on parts of the model that the data barely determines
(here 2.7 times the leverage of the median applicant). Those are exactly the parts that move the most from one
training run to the next.

This is not specific to this dataset. In five other public datasets the same thing happened: the people who absorb a
bad retrain are those who look least like the majority. In US census income data they were mostly people born outside
the US (89% vs 6%), in COMPAS young people with juvenile records, in bank marketing clients contacted during the
2008–2010 low-interest-rate period, in diabetes screening people with extreme measurements
(`research/tail_analysis/RESULTS.md`).

## 4. Their loan decisions are the ones that switch

Because their scores move the most, their decisions are the ones that flip:

* **87% of tail applicants are coin-flip applicants** (decision changes in at least 10% of retrains), against 57% of everyone else;
* on average their decision differs from the exactly-trained model's in **28% of retrains** (others: 18%);
* and they can be found **before any retraining**: leverage, computed from a single fit, picks out the coin-flip
  applicants with an accuracy (ROC AUC) of **0.95**.

Three real applicants from the held-out data (all three **did repay**; the exactly-trained model approves all three):

| Applicant | How often approved by the 1,000 textbook retrains | ... by carefully tuned training |
|---|---|---|
| New-car loan, 1,549 DM over 9 months; job "unemployed / unskilled, non-resident"; employed under 1 year | **45%** | 81% |
| New-car loan, 1,175 DM over 16 months; unemployed; housed for free | **46%** | 42% |
| TV / radio, 3,758 DM over 24 months; unemployed; no property; renting | **46%** | 72% |

For these people, whether the loan is approved depends on the training run about as much as on a coin toss.

> **Disclaimer.** This report describes a statistical property of training procedures: how much a model's outputs
> vary between training runs and which applicants are most affected. It is **not an accusation** of bias,
> discrimination or wrongdoing against any lender, dataset creator or method, and not legal, credit or compliance
> advice. The concern it illustrates (that models can give different answers to the same person depending on
> arbitrary choices) is discussed in the machine-learning literature under names such as *predictive multiplicity*
> and *model arbitrariness*. Protected attributes (sex, age, foreign-worker status) were removed before training.

## 5. So the tail of training matters, and there is theory for it

If the cost of a bad retrain falls on a predictable group of people, then improving the **worst** retrains, not just
the average one, is worth doing. That is the subject of the paper this project is built on:

> Bugra Can and Mert Gürbüzbalaban, *Entropic Risk-Averse Generalized Momentum Methods*, Optimization Methods & Software (2025), arXiv:2204.11292.

In plain terms: training methods have a few dials: how big each step is (**step size**) and how much of the
previous step to carry forward (**momentum**). The paper measures the **worst case** of a training run with a tail
risk measure (EV@R, the entropic value at risk, a cautious upper bound on the bad-month loss), proves how that worst
case depends on the dials, and chooses the dials that make the worst case smallest while still training almost as
fast as possible. The method is called **RA-GMM** (risk-averse generalized momentum method). It needs only three
numbers about the problem (how curved the loss is, at most and at least, and the number of features) and **no trial
runs**.

On this dataset RA-GMM chose a small step (1/100 of the textbook step) with strong momentum (0.98).

## 6. Results: what RA-GMM changes

Same 1,000 retrains, same random orders, same 300 new applicants:

| Training method | Decisions changed per retrain (avg) | ... in the worst 5% of retrains | Coin-flip applicants | Accuracy | Training loss in the worst 5% (vs the exact model) | Trial runs needed |
|---|---|---|---|---|---|---|
| Textbook SGD | 55.2 | 78.8 | 175 | 69.0% | 0.374 | 0 |
| **RA-GMM (paper)** | **40.0** | **59.3** | **128** | **70.9%** | **0.140** | **0** |
| SGD with a smaller step (tuned to match RA-GMM's average) | 40.0 | 59.3 | 128 | 70.7% | 0.132 | 3,100 |
| Carefully tuned SGD (step tuned + averaging the last half of training) | 5.7 | 10.3 | 18 | 73.3% | 0.0015 | 240 |

**What RA-GMM achieves.** With no trial runs, RA-GMM cut decision changes by **15.2 per retrain** compared with the
textbook settings (95% confidence interval 14.4 to 16.0), turned **47 coin-flip applicants** into stable ones, cut the
worst-case training loss by **63%**, and improved accuracy slightly. The same holds on the other five datasets
(changed decisions 3.3–13.3% with RA-GMM vs 5.2–21.5% with textbook SGD, `research/gate0/RESULTS.md`).

**What it does not achieve (stated plainly).**

* Plain SGD with a step size chosen to give the same average quality does just as well (difference −0.03 decisions,
  interval −0.7 to +0.6). RA-GMM's gain comes from choosing a smaller step automatically; it does not give a better
  worst case than SGD at the same average (`research/tail_analysis/RESULTS.md`).
* Carefully tuned SGD with averaging is far better: 5.7 changed decisions and 18 coin-flip applicants. It costs 240
  trial runs to tune; RA-GMM costs none. RA-GMM's practical value here is a principled no-tuning default, not the
  best possible stability.

**Who still flips.** With RA-GMM, 80% of tail applicants are still coin-flip applicants (others 41%), and leverage still
identifies them (ROC AUC 0.92). With carefully tuned SGD only 27% are (others 5%). Better training shrinks the problem
but the people at risk are the same kind of people, and they can be identified in advance.

## 7. The takeaway

1. A model's answer for a given person can depend on the luck of its training run, by a lot with default settings.
2. The cost of a bad training run falls on **unusual applicants**, identifiable in advance from the data alone.
3. Training for the worst case helps: the paper's RA-GMM, with zero tuning, makes decisions measurably more stable
   than textbook training. Careful tuning helps more. Either way, **measure the tail** before deploying.

## 8. Try it on your own data

The repository is now a package (`risk-averse-tuning` 0.3.0, NumPy and pandas only):

```bash
pip install ./ragmm              # from a clone of the repository
ragmm applications.csv --target repaid --positive yes --drop applicant_id \
      --method ragmm --model model.json --report tail_report.md
```

* `--method ragmm` (default) trains with the paper's settings; `sgd` and `careful` are the two baselines.
* `--report` is optional: it retrains each method 200 times on a held-out split and writes a tail report: decisions
  changed per retrain (average and worst 5%), coin-flip rows, the profile of the tail rows, and the rows at risk found
  from leverage without retraining. Same disclaimer included.

In Python:

```python
import pandas as pd
from risk_averse_tuning import TabularEncoder, RiskAverseClassifier, tail_report

df = pd.read_csv("applications.csv")
rep = tail_report(df, target="repaid", positive="yes", drop=["applicant_id"])   # optional diagnostics
print(rep.to_markdown())
```

Run on German Credit, the package reproduces this report's numbers (55.2 / 39.9 / 5.7 changed decisions per retrain
with 200 retrains; tested in `tests/test_package.py`).

## Limits

* One model family (L2 logistic regression, the textbook credit model) trained with stochastic gradients. Models fitted
  by an exact solver have no seed-to-seed variation; models trained by stochastic methods (neural networks, large or
  streaming data) do.
* Only the randomness of training was varied, not the data. Real retraining also uses new data, which adds variation.
* The "tail applicant" profile rests on 15 people in one held-out split; the cross-dataset pattern
  (`research/tail_analysis`) is the stronger evidence.
* Decision threshold 0.5; with the cost-weighted threshold 5/6 commonly used for this dataset, all changes are larger
  (textbook 72.5, RA-GMM 50.7, careful 6.7 per retrain) and the ranking is the same.

## Reproduce

```bash
python research/report/credit_story.py     # all numbers in this report (about 1 minute)
python research/report/make_figures.py
python research/tail_analysis/tail_rows.py # cross-dataset tail analysis
python -m pytest tests                     # 102 tests, including the package
```
