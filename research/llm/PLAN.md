# LLMs and agents: does the "instability tail" idea carry over?

Status: PRE-REGISTRATION. Nothing here has been run. Branch: `claude/ragmm-product-content-s3k2po`.

## 1. The question

In the credit work, the same data and the same method gave different decisions on the same
applicant depending on the random seed, and the chance of a wrong decision was higher for
unusual groups. For LLMs the same question is:

> For a fixed input, how often does the model's decision change just because of randomness
> that should not matter, and is that variation concentrated in particular categories of input?

## 2. Mapping from the credit study

| Credit study | LLM version | Agent version |
|---|---|---|
| Retrain with a different seed | Resample at temperature T; paraphrase / reorder the prompt; fine-tune with a different LoRA seed | Re-run the whole task from the same start state |
| Row decision differs from majority in >=10% of retrains | Prompt answer differs from majority answer in >=10% of runs | Final outcome differs from majority in >=10% of runs |
| Chance of wrong decision by category | Error rate by input category, with the excess over the deterministic (greedy) answer reported separately | Task success rate by task category |
| Exactly trained model (reference) | Greedy decoding / majority of many samples | Majority-vote trajectory outcome |
| Adapter: average seeds, calibrate | Self-consistency (majority over k samples), calibrated confidence, abstain when agreement is low | Vote over k runs at decision steps; verify-before-act gate |
| Leverage predicts affected rows | Agreement across samples (cheap) predicts which prompts are unstable | Per-step agreement predicts which steps compound |

The honest part of the mapping: the RA-GMM recursion itself has no direct analogue in
LLM inference. What transfers is the **measurement** (tail report, per-group statements)
and the **adapter family that worked** (averaging + calibration), not the optimizer.

## 3. Agents: why variance compounds

If each of n steps independently has probability p of an unrecoverable wrong choice, the
chance of a fully correct trajectory is (1-p)^n. With p = 2% and n = 20 this is about 67%.
The agent question is therefore not "how often is one answer different" but "how does
per-step instability turn into trajectory-level failure, and where should a vote or
verification gate be placed". Risk-averse thinking applies directly: we care about the
bad tail of trajectories (irreversible actions), not the mean success rate.

## 4. Experiments (in order, each cheap before the next)

**L1. Decision instability of a small open model on tabular-in-text credit (German Credit rendered as text).**
- Purpose: direct bridge to the existing results. Same 1,000 applicants, same labels, same groups.
- Model: smallest instruction-tuned open model that runs on 4 CPU cores / 15 GB (target <=1.5B parameters, e.g. a Qwen2.5-1.5B-Instruct class model), if weights are reachable through the proxy. Fallback in section 7.
- Randomness: 20 samples per applicant at T=0.7; 5 prompt paraphrases at T=0; 5 few-shot orderings at T=0.
- Measures: per-applicant disagreement with majority; "unstable" = >=10%; chance of wrong decision by the same pre-registered groups as `research/groups` (housing, property, checking, employment), excess over greedy.
- Adapter: majority of k in {1,3,5,9}; agreement-based abstention. Report cost per correct decision and the gain at each k.

**L2. Does agreement identify who is affected?** Using L1 data, test whether the agreement
score from k=5 samples predicts the applicant's unstable status in the held-out 15 samples (AUC), and whether
abstaining on the lowest-agreement 20% reduces the group-wise wrong-decision gap.

**L3. Agent toy: multi-step tool use with checkable outcomes.** A scripted environment
(arithmetic / lookup tools, 5-15 steps, deterministic grader) run with the same model. Measures:
per-step action disagreement, trajectory success rate vs n steps, observed vs (1-p)^n prediction,
effect of voting at each step vs verify-before-final-action. Only run if L1 shows a usable signal and
the model can follow a tool format at all.

**L4 (later, needs GPU or API budget).** LoRA fine-tune seeds on a classification task: disagreement
across seeds and whether weight averaging (model soup) reduces it. This is the closest analogue
to the tree/NN seed-averaging result.

## 5. Hypotheses (written before running)

- H1: A non-trivial share of applicants (>=5%) are unstable at T=0.7, and instability is higher for
  the same unusual groups as in the classical models.
- H2: Majority vote over k=5 reduces the unstable share by at least half. Gains saturate after k~9.
- H3: Agreement predicts instability with AUC >= 0.8.
- H4 (agents): observed trajectory success is below the independent-steps prediction when failures are
  correlated by task category, so voting helps less than (1-p)^n suggests.
- Disconfirming outcomes are reported as such: e.g. instability could be uniform across groups, or
  voting could lower accuracy because the model has a systematic (not random) bias. In the second case
  the right reading is that sampling noise is not the problem.

## 6. What could fail, and pivots

| Risk | Pivot |
|---|---|
| Weights not reachable through proxy | Use a very small model from PyPI-hosted packages or a scikit-learn text baseline as a pipeline test only; run the real L1 once the user supplies API access or a GPU. Write the harness against a model-agnostic `generate(prompt, seed, T)` interface now. |
| CPU too slow | Cut to 300 applicants, 10 samples; cache KV prefix; use logprob of "approve"/"deny" tokens (one forward pass gives a probability, no sampling needed). |
| Model is near-chance at credit | Instability is then trivially high and uninformative. Switch to a task where the model is accurate (e.g. sentiment or a rules-given eligibility check) and keep the group analysis. |
| Instability is systematic bias, not noise | Report that voting does not help; the fix is calibration or a better prompt, not adaptation. |
| Group sizes too small | Same rule as before: min group size, pre-registered groups, no cherry-picking. |
| Sensitivity of the topic | Statements describe statistical behaviour of a model on a public dataset; not accusations, not credit advice. |

## 7. Customer value and what to publish

- Product: "LLM Decision Stability Report": give it a prompt template and labelled examples, get back
  per-category chance of a wrong or flip-flopping decision, an abstain threshold, and the cost of
  k-sample voting. This is the same shape as the existing `ragmm` tail report.
- Agent version: a trajectory stability report that finds the steps where voting or a verification gate
  gives the biggest gain in tail (worst 5%) success.
- Content angle: "Ask the model twice" - what fraction of your AI decisions change when you do.

## 8. Order of work

1. Write the model-agnostic harness and tests (mock model) - no external dependency.
2. Check whether a small model can be downloaded; record the answer in RESULTS.md either way.
3. Run L1 on a small slice first as a pipeline check, then the pre-registered full run.
4. L2, then decide on L3 from L1/L2 results.

Deviations from this plan will be recorded in RESULTS.md before results are discussed.

## 9. Environment check (recorded before any run)

- Hugging Face model hosting returns 403 through the session proxy, so open-weight models cannot be downloaded here.
  PyPI is reachable (the `transformers` package installs), but weights are not.
- Consequence: the harness and tests can be built now against a mock model; the real L1 run needs
  either an allowed model host, a pip-installable small model, or API access supplied by the user.
