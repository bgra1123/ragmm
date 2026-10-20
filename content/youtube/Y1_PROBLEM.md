# Y1 — "I retrained a credit model 1,000 times. It kept changing its mind."

*Long-form YouTube script, about 12 minutes. Content ID **Y1**. It covers the problem and the experiments; the product is in
[Y2](Y2_PRODUCT.md). Written companion: [B1](../../research/blog/BLOG.md). Numbers: [`../FACT_SHEET.md`](../FACT_SHEET.md) (IDs in
braces, e.g. {A2}). References: [`../REFERENCES.md`](../REFERENCES.md) (numbers in brackets, e.g. [1]).*

**Audience:** curious viewers, data scientists, people who work in lending, insurance or hiring. No maths needed until
chapter 5, which is optional.
**Visual style:** the navy and amber look of the existing explainer (`research/blog/make_video.py` renders the dot-grid and
bar-chart scenes; re-use them). Screen recordings of the code for chapter 6.

---

## Chapters (paste into the YouTube description)

```
0:00 Same applicant, different answer
0:45 The experiment
2:30 It's not one model, it's a lottery
4:00 Who loses the lottery
6:00 Why the obvious fix didn't work
7:30 The theory: risk, not averages (optional maths)
9:30 What does work
11:00 What this does and doesn't show
11:45 Next video: the tool
```

---

## 0:00 — Cold open (hook)

**On screen:** two identical loan application cards; one stamped approve, one deny (thumbnail prompt in
`research/blog/VIDEO_PROMPT.md`).

**VO:** "Same applicant. Same data. Same model. Two different answers. The only thing I changed was the random seed, the
number that decides what order the computer looks at the training examples. I did this a thousand times, and I want to show
you who ends up paying for that randomness."

## 0:45 — The experiment {A1}

**On screen:** dataset card: "German Credit, 1,000 real applications from a German bank, public dataset [12]".

**VO:** "I took a public dataset of 1,000 loan applications, trained the most common credit-scoring model, a logistic
regression, on 700 of them, and asked it to decide on the other 300. Then I did it again. And again. A thousand times. Same
data, same settings. Only the seed changed."

**B-roll:** retrain counter ticking up; dot grid flickering (scene 2 of the explainer).

## 2:30 — Not one model, a lottery {A2, A3}

**VO:** "Each retrain gave 55 of the 300 applicants, about one in six, a different decision from the model you'd get without
training randomness. And over the thousand retrains, 284 of the 300 applicants got a different answer at least once. Only 16
got the same answer every time."

**On screen:** "55 of 300 per retrain" → "284 of 300 at least once".

**VO:** "Researchers call this predictive multiplicity [4, 5]: many models that look equally good overall but disagree on
individual people. It shows up in neural networks [7], in benchmarks [8], across modern machine learning [6]. So the
question isn't 'is my model accurate'. It's 'which of my possible models did I happen to ship this month?'"

## 4:00 — Who loses the lottery {B1–B5, F1}

**On screen:** bar chart (scene 4 of the explainer), one group at a time.

**VO:** "Count a decision as wrong when the model refuses someone who repaid, or approves someone who defaulted. Across all
applicants, a model trained the standard way is wrong 30.4% of the time. Without training randomness it would be 24.4%. So 6
points come from luck alone. But look at the groups."

| Group | Standard training | Without randomness | Due to luck |
|---|---|---|---|
| Housing "for free" | 37.8% | 27.4% | +10.4 |
| No property | 35.4% | 25.6% | +9.9 |
| Negative checking balance | 40.0% | 31.3% | +8.8 |
| Unemployed | 42.9% | 37.3% | +5.6 |

**VO:** "The people who look least like everyone else in the data pay the most. That makes sense: the model has seen few
examples like them, so their part of the model is pinned down least, and the random seed moves it most. Across six datasets
I tested, the 5% most affected people carried between about a fifth and more than half of the extra error in the bad runs."

**Caveat on screen:** "Statistical behaviour of a model on a public dataset. Not an accusation of bias. Not credit advice."

## 6:00 — The fix I expected to work, and didn't {C1}

**VO:** "My research is about exactly this kind of risk. In a paper with Mert Gürbüzbalaban [1] we built RA-GMM, a way to set
the step size and momentum of training so that the bad outcomes, not just the average, stay small. So I trained with RA-GMM.
It cut changed decisions by about 15 per retrain compared with the textbook setting. But plain SGD with a smaller step, tuned
to the same average quality, did just as well. As a trainer, RA-GMM was a tie. I'm telling you this because a negative result
is a result."

## 7:30 — The theory, briefly (optional) [1, 2]

**On screen:** the GMM update and the EV@R definition.

```
y_k = x_k + γ d_k,   d_{k+1} = β d_k − α ∇f(y_k),   x_{k+1} = x_k + d_{k+1}
EV@R_{1−ζ}[X] = inf_{t>0} (1/t) log( E[e^{tX}] / ζ )
```

**VO:** "Gradient descent, heavy ball and Nesterov are all special cases of one family with three knobs. The entropic value
at risk [2] is a way to measure the bad tail of the training error. The paper bounds it and picks the knobs to minimize it.
The theory needs a strongly convex problem. A logistic regression is one. A boosted tree or a deep network is not, and that
turns out to matter."

## 9:30 — What does work (teaser for Y2)

**VO:** "Two ideas worked. First, for convex models, finish training with a few careful full passes over the data. There is
one right answer, and every seed walks to it. A short polish brought the 'housing for free' group from 37.8% to 31.5% wrong
decisions {B2}. Second, for trees and networks, where the seed changes the structure of the model, average a few copies. That
is the old idea of bagging [9] and deep ensembles [10]. In the next video I'll show the tool I built to do both, and how much
it helps for XGBoost, random forests and neural networks."

## 11:00 — Limits

**VO:** "Three limits. This is one public dataset from the 1990s, so don't read it as a claim about any lender today. These
numbers describe how a model behaves, not anyone's intent. And fixing the randomness doesn't fix what the model learned: even
the model without randomness gets 27.4% of 'housing for free' applicants wrong."

## 11:45 — End screen

**VO:** "Code, data and every number in this video are in the repository, linked below. Next: the tool."

**End screen:** link to Y2, to the tutorial B2, and to the short T1.

---

## Description box

```
Same applicant, same data, different answer. I retrained a credit model 1,000 times and measured who is affected.

Chapters: (see above)

Code and data: https://github.com/bgra1123/ragmm
Written version: research/blog/BLOG.md   ·   Tutorial: docs/TUTORIAL.md   ·   Next video: Y2

References
[1] Can & Gürbüzbalaban (2025). Entropic risk-averse generalized momentum methods. Optimization Methods and Software 40(6). doi:10.1080/10556788.2025.2549356 · arXiv:2204.11292
[2] Ahmadi-Javid (2012). Entropic value-at-risk. J. Optim. Theory Appl. 155(3).
[4] Marx, Calmon & Ustun (2020). Predictive multiplicity in classification. ICML.
[5] Black, Raghavan & Barocas (2022). Model multiplicity. FAccT.
[6] D'Amour et al. (2022). Underspecification... JMLR 23.
[7] Summers & Dinneen (2021). Nondeterminism and instability in neural network optimization. ICML.
[8] Bouthillier et al. (2021). Accounting for variance in ML benchmarks. MLSys.
[9] Breiman (1996). Bagging predictors.  [10] Lakshminarayanan et al. (2017). Deep ensembles. NeurIPS.
[12] Hofmann (1994). Statlog (German Credit Data). UCI. doi:10.24432/C5NC77

Statistical behaviour of models on a public dataset; not an accusation of bias; not legal or credit advice.
```
