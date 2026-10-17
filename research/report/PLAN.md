# Credit report: analysis plan (written before running)

The report (`REPORT.md`) tells one story on German Credit, so it needs held-out numbers that connect the
tail of training to individual loan decisions. All measurements below are fixed before running `credit_story.py`.

Setup: as Gate 0 / tail analysis (German Credit, 700 training / 300 held-out applicants, L2 logistic regression,
λ = 0.01, batch 16, K = 2,000). **1,000 retrains** per method on identical minibatch sequences. Decision: approve if
the model's probability of good credit is ≥ 0.5 (primary; the 5:1 cost threshold 5/6 is reported as sensitivity).

Methods: SGD (textbook, step 1/L); RA-GMM (ε = 0.25, problem 51); SGD with the same average loss as RA-GMM
(matched step); SGD careful (tuned step + tail averaging).

Measurements (held-out applicants unless stated):
1. **Tail runs**: the 5% of retrains with the highest training objective; their approval rate, accuracy and number of
   changed decisions vs the other 95%.
2. **Per applicant**: approval share across retrains `p_i`; flip probability vs the exactly-trained model; "coin-flip
   applicant" = decision differs from the majority in ≥ 10% of retrains; extra loss in the tail runs; leverage from the
   exact fit.
3. **Tail applicants** (top 5% of held-out applicants by extra loss in the tail runs, 15 people): their flip
   probability and share of coin-flip applicants vs the rest; their feature profile.
4. **Prediction without retraining**: the 10% of applicants with the highest leverage (computable from one fit):
   share of all decision changes they account for (random 10% would be ~10%), and ROC AUC of leverage for
   identifying coin-flip applicants.
5. **Methods compared**: mean and worst-5% changed decisions per retrain, number of coin-flip applicants, CVaR₅% of the
   training objective; paired bootstrap CIs (seeds) for RA-GMM vs textbook SGD and vs matched SGD.
6. Three example applicants (most-flipped under textbook SGD), described from their data, with their approval share
   under each method.

## Addendum: the two kinds of wrong loan decision (written before running `error_types.py`)

Same retrains (1,000 per method, same minibatches), same 300 held-out applicants (210 good = would repay, 90 bad =
defaulted per the dataset label), methods as above.

For each method, averaged over the 1,000 models, at threshold 0.5 (primary) and 5/6 (sensitivity: the cost-based
threshold for 5:1 default-to-lost-business cost):
* **Wrongly denied rate** = share of good applicants who are denied. **Wrongly approved rate** = share of bad
  applicants who are approved. Also: number of applicants of each kind per model, default rate among approvals,
  and expected cost per applicant (5 per wrongly approved, 1 per wrongly denied).
* Split each into errors the **exactly-trained model also makes** (unavoidable with this model family and data) and
  **training-induced** errors (the retrain errs, the exact model does not) and the reverse (retrain fixes the exact
  model's error).
* **Shared characteristics:** per error type, over all (model, applicant) error events: for each binary feature,
  the share of events with the feature (coverage) vs the share of all applicants of that class (good for denied, bad
  for approved) with it; report the 4 features with the highest lift (coverage ≥ 25%). Plus the share of events
  falling in the **top-5% most error-prone applicants** of that type.
* Same numbers for the training-induced events only.
* Paired bootstrap over models for RA-GMM minus textbook SGD and RA-GMM minus matched SGD in each rate.

Hypotheses: **H1** RA-GMM's two error rates are both lower than textbook SGD's. **H2** the training-induced errors
concentrate on a few characteristics (top-5% error-prone applicants carry ≥ 20% of events). **H3** (open) RA-GMM equals
matched SGD within the bootstrap interval.

## Addendum 2: credit outcomes with and without the polish step (written before running `credit_with_adapter.py`)

Question (user): how does the post-training adapter change **credit outcomes** relative to the same trainers without it?

Setting: German Credit as in the report (700 training, 300 held-out applicants: 210 good, 90 bad; L2 logistic regression, λ = 0.01,
batch 16). 300 seeds. **Stage-1 trainers (no adapter = the baselines):** textbook SGD (α = 1/L, 2,000 steps); SGD + momentum 0.9 (lr 0.01);
Adam (lr 0.001, the PyTorch default, and lr 0.01); consensus ADMM (4 blocks, 5 iterations); RA-GMM from scratch (paper, problem 51);
carefully tuned SGD with averaging (as in the report).

**With a polish step (same stage-1 weights, then B = 5 epochs of row-gradient compute = 219 minibatch steps or 5 full-batch steps):**
* **adapter**: step and momentum from the exact EV@R model (6 pilots), as in `research/polish`;
* **GD exact line search** (deterministic full-batch, the strong baseline found in `research/linesearch`);
* **R1, θ = 5** (the user's risk-averse step along the full gradient);
and a sensitivity at B = 20 epochs.

Outcome metrics (held-out, thresholds 0.5 and the cost-based 5/6): good applicants wrongly denied (rate and count), bad applicants wrongly approved
(rate and count), share of approved loans that default, expected cost per applicant (5 per default approved, 1 per good denied),
decisions changed vs the exactly-trained model, coin-flip applicants (decision differs from the majority in ≥ 10% of seeds), and the
errors **caused by the training run** (the retrain errs where the exact model does not). Paired bootstrap over seeds (5,000) for the
change in wrongly denied count and in cost relative to the same stage-1 trainer without a polish.

Hypotheses:
* **H1** With the adapter, wrongly denied (at 0.5) and expected cost (at 5/6) move toward the exact model's values for every stage-1 trainer, and for the
  worst trainers (textbook SGD, momentum, Adam 0.01) the training-induced errors fall by ≥ 50%.
* **H2** Mean decisions changed vs the exact model fall by ≥ 50% for trainers that change ≥ 10 decisions.
* **H3** Coin-flip applicants fall by ≥ 50% for those trainers.
* **H4** GD exact line search at B = 5 reaches outcomes at least as close to the exact model's as the adapter for every trainer (deterministic polish).
* **H5 (open)** Polishing cannot beat the exact model: the remaining errors equal the exact model's own (27 wrongly denied, 53 wrongly approved at 0.5),
  so for a trainer that is already close (careful SGD) the polish changes outcomes little.
* **H6 (open)** At the 0.5 threshold, wrongly approved defaulters may **rise** after polishing (the exact model approves 53 of the 90 defaulters).
