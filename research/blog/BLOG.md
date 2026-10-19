# Same applicant, same data, different answer

### How the luck of a training run can decide who gets a loan, who it hits hardest, and what actually fixes it

*I retrained a standard credit model a thousand times and gave it the same data every time. The model disagreed with itself about nearly every applicant. Here is what I found, what I expected to work but didn't, and what does.*

---

## The experiment

Take 1,000 real loan applications from a German bank (a standard public dataset). Train the most common credit-scoring model, a logistic regression, on 700 of them. Ask it to decide on 300 applicants it has never seen.

Now do it again. Same 700 applications, same model, same settings. The only thing that changes is the random order in which the training procedure looks at the examples. Do this 1,000 times.

You would expect 1,000 copies of the same answer. You get this instead:

* Each retrained model gave **55 of the 300 applicants (18%) a different decision** from the model you would get if training had no randomness at all.
* **284 of the 300 applicants** received a different answer from at least one of the 1,000 models. Only 16 got the same answer every time.

Nothing about the applicants changed. Only the luck of the training run did.

## Why it happens

Modern models are usually trained by a procedure that looks at a few examples at a time and nudges the model after each small batch. The order and the batches are random. For a well-behaved model with enough care, the randomness washes out. With a default setting it does not: the model ends up near the right answer, but not at it, and "near" is different every time.

If a lender retrains monthly, which models are in production in a given month is partly a matter of luck.

## "How unstable is it?" is the wrong question. Ask: for whom, and how likely?

An average hides the story. Here is a more useful way to say it. Count a decision as wrong when the model refuses someone who went on to repay, or approves someone who defaulted. Then ask, for a type of applicant, how often a model trained the standard way gets it wrong, and compare it with the model you would get without training randomness (the floor):

| Applicants who... | Chance of a wrong decision, standard training | Without training randomness | Added by randomness |
|---|---|---|---|
| Live in housing "for free" | 37.8% | 27.4% | **+10.4 points** |
| Have no property | 35.4% | 25.6% | **+9.9 points** |
| Have a negative checking balance | 40.0% | 31.3% | **+8.8 points** |
| Ask for a large and long loan | 38.7% | 31.7% | +7.0 points |
| Are currently unemployed | 42.9% | 37.3% | +5.6 points |
| **All applicants** | 30.4% | 24.4% | +6.0 points |

Read one row aloud: *"For applicants with no property, there is a 35.4% chance that a model trained the standard way makes the wrong decision. A model without any training randomness would make it 25.6% of the time. About 10 points come from the luck of the training run."*

And the costly mistake for a lender, approving someone who then defaults. With a stricter, cost-based approval cutoff, the chance that a defaulter is approved is **26.6%** with standard training, against **9.1%** for the model without training randomness.

*(Group labels describe where wrong decisions land. Sex, age and foreign-worker status were never inputs to any model. This is not an accusation of bias against any lender or any dataset: it describes how much a model's answers depend on the training run, and for whom.)*

## Who gets hit: people who look unlike everyone else

The pattern is consistent: training noise lands on applicants with an unusual profile. In this dataset those are people who are unemployed, live in housing "for free", have no property, or have a negative checking balance. There are few people like them in the training data, so the data pins down their score only loosely, and that is exactly the part of the model that wobbles from run to run.

I checked whether this is a quirk of one dataset. It is not. Across six public datasets the same thing appears: in US census income data the people who absorb a bad training run are mostly those born outside the US (89% of the most-affected 5%, against 6% of everyone else). In bank marketing data they are clients contacted during an unusual low-interest-rate period. In diabetes screening they are people with extreme measurements. In every dataset, 5% of people carry between 20% and 58% of the damage from a bad retrain.

## What I expected to fix it, and didn't

I started from a research paper (Can and Gürbüzbalaban, *Entropic Risk-Averse Generalized Momentum Methods*), which proposes training methods that control the worst case of a training run. My hypothesis was that using the paper's method would make credit decisions more stable.

It did, compared with the textbook setting: about 15 fewer changed decisions out of 300. But a plain SGD run with a smaller step size gave the **same** result at the same average quality. The paper's method, used as a trainer, was just a very good way of picking a small step size. A carefully tuned SGD, with averaging, did far better still (5.7 changed decisions instead of 40). I report this because the honest finding changed the story I had planned to tell.

## What does work

**1. If your model is a convex problem (logistic regression, a linear head), finish with a few deterministic steps.**
Train as usual, then run a handful of full passes over the data with a line search. Five passes brought every trainer I tried (SGD, momentum, Adam, ADMM) to within a handful of decisions (1 to 8 out of 300) of the model without training randomness, and twenty passes left under one. A shorter, stochastic version of the same idea already cut the chance that a defaulter is approved at the cost-based cutoff from 26.6% to 13.1%. The deterministic finish removes the seed-to-seed variation because a convex problem has one answer.

**2. If your model is XGBoost, a random forest or a neural network, average a handful of seeds.**
These models are not convex; there is no single answer to converge to, and the randomness lives in the structure of the model (which trees, which features). I tried the obvious fixes and they failed on German Credit: re-solving the last layer did nothing for the neural network and made random forests *worse*, and continuing to boost with full data left XGBoost's disagreement at 5.6%. What worked was the old trick of training several models and averaging them: disagreement dropped about 70% (XGBoost 5.6% → 1.6%, neural network 14.8% → 4.7%).

**3. Calibrate before you apply a cost-based cutoff.**
Tree models and neural networks are overconfident. A cutoff like "approve only above 83%" assumes the probabilities mean what they say. Calibrating (a simple one-line fit on held-out predictions) lowered the expected cost per applicant for every model I tried. A calibrated, averaged random forest ended up cheaper than the exactly trained logistic regression (0.487 vs 0.497).

## What this does not show

* **One small dataset for the credit numbers.** German Credit has only 700 training rows per split, where fancy models have little advantage. Treat percentages as illustrations; the pattern held on other datasets, but I have not repeated the credit analysis on a large lending dataset.
* **A fixed model still makes mistakes.** Even without any training randomness, the model refuses 24.7% of the good applicants who live in housing "for free" (versus 4.3% of those outside the groups I checked). Removing training noise removes the *extra* errors, not the model's own error pattern; that needs different data, features or modelling.
* **Only the training randomness was varied.** Real retraining also uses new data, which adds its own variation.
* **My own path included dead ends.** The paper's method as a trainer was a tie with plain SGD at a smaller step. An approach I built for the portfolio-rebalancing version of this question was worse than textbook SGD, because noise in that problem grows with the weights. The data told me to stop selling the method and start selling the measurement.

## What to do with this

1. **Measure before you trust.** Retrain your model 50–200 times with different seeds on the same data. Report, by type of applicant, the chance of a wrong decision and how much of it comes from the randomness of training.
2. **Fix by model type.** Convex model: finish with deterministic steps. Trees or networks: average seeds, and calibrate before using a cost cutoff.
3. **Look at who, not just how many.** The unstable decisions concentrate on unusual profiles; check those groups first.

I turned this into an open-source tool: `ragmm data.csv --target repaid --positive yes --report tail_report.md` trains the model and writes a report like the one behind this post, including the chance of a wrong decision for any groups you define. All experiments, the pre-registered designs (including the ones that failed) and the code are in the repository.

---

*Notes. This post describes statistical behaviour of training procedures on public data from the 1990s. It is not legal, credit or compliance advice, and it makes no claim about any real lender.*

*Reproduce: `research/report/REPORT.md`, `research/groups/CONCRETE.md`, `research/sota2/RESULTS.md`, `research/polish/RESULTS.md`, `research/linesearch/RESULTS.md` in the repository; every number in this post is traced in `research/blog/FACT_CHECK.md`.*
