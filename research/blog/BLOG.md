# Same applicant, same data, different answer

### How the luck of a training run can decide who gets a loan, why it hits rare applicants hardest, and what actually fixes it

*I retrained a standard credit model a thousand times and gave it the same data every time. The model disagreed with itself about nearly every applicant, and most of all about the applicants it had seen the fewest of. Here is what I found, what I expected to work but didn't, and what does.*

---

## One applicant, a hundred worlds

Start with one person from a public dataset of real loan applications. She is between jobs, lives with family (on the form: housing "for free"), and asks for a small car loan. She went on to repay it. The model trained without any randomness approves her.

I trained the same model on the same data a hundred times, changing nothing but the random seed, the number that decides the order in which the training procedure looks at the examples. **Fifty-six of the hundred models approved her. Forty-four rejected her.** Same file, same bank, same morning; a different world depending on a number nobody chose.

She is not a special case. That is the finding.

## The experiment

Take 1,000 real loan applications from a German bank (a standard public dataset from the 1990s). Train the most common credit-scoring model, a logistic regression, on 700 of them. Ask it to decide on 300 applicants it has never seen.

Now do it again. Same 700 applications, same model, same settings. Only the random order of the training examples changes. Do this 1,000 times.

You would expect 1,000 copies of the same answer. You get this instead:

* Each retrained model gave **55 of the 300 applicants (18%) a different decision** from the model you would get if training had no randomness at all.
* **284 of the 300 applicants** received a different answer from at least one of the 1,000 models. Only 16 got the same answer every time.

Nothing about the applicants changed. Only the luck of the training run did.

## Why it happens

Modern models are trained by a procedure that looks at a few examples at a time and nudges the model after each small batch. The order and the batches are random. With enough care the randomness washes out. With a default setting it does not: the model ends up near the right answer, but not at it, and "near" is different every time.

Two honest qualifications before going on. First, the size of the effect depends on how aggressively the model is trained: with the textbook step size it is large; with a cautious learning rate it shrinks to 1–3% of decisions. Second, models fitted by an exact solver (the default for logistic regression in most libraries) have no seed at all; the problem belongs to models trained by stochastic methods, which includes most gradient-boosted trees with subsampling, most neural networks, and anything retrained on a schedule with early stopping.

If a lender retrains monthly, which model is in production in a given month is partly a matter of luck.

## "How unstable is it?" is the wrong question. Ask: for whom, and how likely?

An average hides the story. Count a decision as wrong when the model refuses someone who went on to repay, or approves someone who defaulted. Then ask, for a type of applicant, how often a model trained the standard way gets it wrong, and compare it with the model you would get without training randomness (the floor):

| Applicants who... | Chance of a wrong decision, standard training | Without training randomness | Added by randomness |
|---|---|---|---|
| Live in housing "for free" | 37.8% | 27.4% | **+10.4 points** |
| Have no property | 35.4% | 25.6% | **+9.9 points** |
| Have a negative checking balance | 40.0% | 31.3% | **+8.8 points** |
| Ask for a large and long loan | 38.7% | 31.7% | +7.0 points |
| Are currently unemployed | 42.9% | 37.3% | +5.6 points |
| **All applicants** | 30.4% | 24.4% | +6.0 points |

Read one row aloud: *"For applicants with no property, there is a 35.4% chance that a model trained the standard way makes the wrong decision. A model without any training randomness would make it 25.6% of the time. About 10 points come from the luck of the training run."*

Seen from the applicant's side: among people who lived in housing "for free" and repaid their loan, **34.6% were refused** by the standard-trained model, against 24.7% by the model without randomness. Roughly one refusal in three of a reliable applicant in that group came from the training run, not from the file.

And the costly mistake for a lender, approving someone who then defaults: with a stricter, cost-based approval cutoff, the chance that a defaulter is approved is **26.6%** with standard training, against **9.1%** for the model without training randomness.

*(Group labels describe where wrong decisions land. Sex, age and foreign-worker status were never inputs to any model. This is not an accusation of bias against any lender or dataset: it describes how much a model's answers depend on the training run, and for whom.)*

## Who gets hit: people who look unlike everyone else

The pattern is consistent: training noise lands on applicants with an unusual profile. In this dataset those are people who are unemployed, live in housing "for free", have no property, or have a negative checking balance. There are few people like them in the training data.

I checked whether this is a quirk of one dataset. It is not. Across six public datasets the same thing appears: in US census income data, the people who absorb a bad training run are mostly those born outside the US (89% of the most-affected 5%, against 6% of everyone else). In bank marketing data they are clients contacted during an unusual low-interest-rate period. In diabetes screening they are people with extreme measurements. In every dataset, 5% of people carry between 20% and 58% of the damage from a bad retrain.

## Why them: the line moves most where the data is thin

A classifier draws a line through the population: approve on one side, refuse on the other. There are two reasons a rare applicant might get a different answer from run to run. Either rare applicants happen to sit closer to the line, which has nothing to do with the seed, or **the line itself moves more in their part of the space**, because few training examples pin it down there. Only the second explanation blames the training.

So I tested it directly, holding the distance to the line fixed. For every held-out row I measured how much its score wobbled across 200 retrains, and compared rows that sit at the same distance from the line but differ in how rare they are (how far they are from their nearest training neighbours).

* The wobble of a row's score tracks how thin the data is around it: rank correlation 0.5 to 0.9 with rarity on all six datasets, and with the model's own measure of "poorly pinned down" (leverage) on five of six.
* **At the same distance from the line, the rarest 10% of applicants got a different decision 1.4 to 1.9 times as often as the rest** (1.45 on German Credit, 1.39 on COMPAS, 1.79 on US census income, 1.86 on bank marketing, 1.63 on diabetes screening). On two datasets the rare rows flip *less* in raw terms because they sit farther from the line; at equal distance they flip more.
* The excess is a property of the training, not of the people: after an exact polish of the model (next section), fewer than 1% of rows flip and the rare-row excess disappears.

In plain words: the model is surest about the kinds of people it has seen before. For everyone else, the line is drawn a little differently every time.

## What I expected to fix it, and didn't

I started from a research paper (Can and Gürbüzbalaban, *Entropic Risk-Averse Generalized Momentum Methods*), which proposes training methods that control the worst case of a training run. My hypothesis was that using the paper's method as the trainer would make credit decisions more stable.

It did, compared with the textbook setting: about 15 fewer changed decisions out of 300. But a plain SGD run with a smaller step size gave the **same** result at the same average quality. Used as a trainer, the paper's method was a good way of picking a small step size. A carefully tuned SGD with averaging did far better still (5.7 changed decisions instead of 40).

Later I tried a risk-averse rule for choosing each step of the fix below, built on the same risk measure. It takes gentler steps and hurts fewer individual rows per step, but on stability it matched an ordinary exact line search, and a plain step of the same size did almost the same. I report these because the honest findings changed the story I had planned to tell: the value turned out to be in measuring the problem and in a plain, exact fix, not in a cleverer optimizer.

## What does work

**1. If your model is a convex problem (logistic regression, a linear head), finish with a few exact steps.**
Train as usual, then continue from the trained weights with a handful of full passes over the data, each step chosen by a line search, or solve the head exactly with Newton's method (3 to 10 iterations from a trained model). A convex problem has one answer, so every seed walks to the same model. Five passes brought every trainer I tried (SGD, momentum, Adam, ADMM) to within 1 to 8 decisions out of 300 of the model without training randomness; twenty passes left under one. With the packaged version, two retrains that used to disagree on 15.5% of applicants disagree on 0.1%, the wrong-decision rate falls from 27.3% to 25.4%, and the groups hit hardest gain the most: unemployed applicants 41.8% → 36.3% wrong decisions, housing "for free" 35.1% → 30.1%, no property 32.0% → 27.9%.

Is this better than simply training longer? Mostly. Against the strongest cheap alternative, one more epoch of training with the iterates averaged, the exact finish wins from the first epoch when the model was trained with Adam, and from about two to five epochs when it was trained with plain SGD. On a large dataset with fewer than two passes to spare, the averaged epoch is the better update. The gain comes from using exact, full-data gradients, not from any particular step rule.

**2. If your model is XGBoost, a random forest or a neural network, average a handful of seeds.**
These models are not convex; there is no single answer to converge to, and the randomness lives in the structure of the model (which trees, which features). The obvious gradient fixes failed: re-solving the last layer did nothing for the neural network and made random forests *worse* (4.1% → 9.1% disagreement), and continuing to boost on the full data left XGBoost where it was (5.6% → 5.7%). What worked was the old trick of training several models and averaging them: disagreement dropped about 70% (XGBoost 6.5% → 2.1%, neural network 14.4% → 5.1%). One thing to know before you choose: for tree models, ten times more trees at a lower learning rate gave the same stability at the same compute. Averaging seeds is not magic; it is a convenient way to spend the compute on a model you already have.

**3. Calibrate before you apply a cost-based cutoff.**
Tree models and neural networks are overconfident. A cutoff like "approve only above 83%" assumes the probabilities mean what they say. Calibrating (a one-line fit on held-out predictions) lowered the expected cost per applicant for every model I tried. A calibrated, averaged random forest ended up cheaper than the exactly trained logistic regression (0.487 vs 0.497).

## What this does not show

* **One small, old dataset for the credit numbers.** German Credit has 700 training rows per split and dates from the 1990s. Treat the percentages as illustrations of a mechanism; the mechanism held on five other datasets, but I have not repeated the credit analysis on a large lending dataset, and nothing here describes any real lender today.
* **A fixed model still makes mistakes.** Even without any training randomness, the model refuses 24.7% of the good applicants who live in housing "for free" (versus 4.3% of those outside the groups I checked). Removing training noise removes the *extra* errors, not the model's own error pattern; that needs different data, features or modelling. None of this makes a model fair.
* **Only the training randomness was varied.** Real retraining also uses new data, which adds variation of its own that I did not measure.
* **My own path included dead ends.** The paper's method as a trainer tied plain SGD at a smaller step; its quadratic variant was worse than textbook SGD on a portfolio-rebalancing version of this question; the risk-averse step rule matched a line search. The data told me to stop selling the method and start selling the measurement.

## What to do with this

1. **Measure before you trust.** Retrain your model 30–200 times with different seeds on the same data. Report, by type of applicant, the chance of a wrong decision and how much of it comes from the randomness of training. Look hardest at the rare profiles: that is where the line moves.
2. **Fix by model type.** Convex model: finish with exact steps or solve the head. Trees or networks: average seeds (or grow more trees), and calibrate before using a cost cutoff.
3. **Say what you fixed.** Removing the luck makes decisions reproducible. It does not make the model right about the people it has rarely seen; that is the next, harder problem, and now you can see exactly where it is.

I turned the measurement and the fixes into an open-source package, **edgelift**: `edgelift applications.csv --target repaid --positive yes --model xgboost --group-columns housing --report stability.md` retrains your model, writes the before/after report, and prints sentences like the ones in this post for any groups you define. Every experiment behind it, including the pre-registered designs that failed, is in the research repository.

---

*Independent work. This is a personal research project, done in my own time on public data. The experiments, code and opinions are my own and are not connected to, funded by, endorsed by or representative of any employer, university, lender or other organization. Nothing here indicates how any real institution builds or uses its models.*

*Notes. This post describes statistical behaviour of training procedures on public data from the 1990s. It is not legal, credit or compliance advice, and it makes no claim about any real lender. The opening applicant is a real held-out row of the public dataset described by her form, not an identifiable person.*

*Reproduce: `research/report/REPORT.md`, `research/groups/RESULTS.md`, `research/groups/CONCRETE.md`, `research/rare_flip/RESULTS.md`, `research/product/RESULTS.md`, `research/sota2/RESULTS.md`, `research/risk_effect/RESULTS.md` in the research repository; every number in this post is traced in `research/blog/FACT_CHECK.md`.*

---

**More:** the package: [github.com/bgra1123/edgelift](https://github.com/bgra1123/edgelift) · the research record: [github.com/bgra1123/ragmm](https://github.com/bgra1123/ragmm) · videos and shorts: [content hub](../../content/README.md) · the paper: Can & Gürbüzbalaban, *Entropic Risk-Averse Generalized Momentum Methods*, Optimization Methods and Software 40(6), 2025, doi:10.1080/10556788.2025.2549356 (arXiv:2204.11292).
