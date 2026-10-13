# Which MCP to build and promote: recommendation, risks, pivots

Written on branch `claude/ragmm-product-content-s3k2po`. Nothing here touches `main`; no pull request is open.
Every number below is from this repository's own experiments (paths given); nothing is projected from outside data.

## 1. What the evidence says (so the plan does not sell something we disproved)

| Finding | Source | Consequence for the product |
|---|---|---|
| With textbook training settings, the same loan model retrained on the same data flips the approve/reject decision for ~100 of 300 real applicants (Nesterov) or ~71 (SGD); 98–100% of applicants get a decision that depends on the random seed. | `research/model_multiplicity/RESULTS.md` | **The problem is real, visible, and easy to explain.** This is the hook. |
| Settings tuned for the actual training budget cut that to 6.6–10 flips (tail-averaged SGD best). | same | **The fix that works is cheap and boring** (tune for the budget, average the tail). |
| The paper's RA-GMM recipe (problem 51) reduces flips versus textbook momentum (51 vs 100) but is much worse than tuned baselines (51 vs 7–10). | same | **Do not sell RA-GMM as the fix.** |
| Same pattern on quadratic portfolio rebalancing; the paper's quadratic problem (32) is worse than textbook SGD there because minibatch noise grows with the weights. | `research/portfolio/RESULTS.md` (P1–P5) | Same: the closed-form recipe is not the product. |
| The exact finite-horizon EV@R predictor was calibrated at 40/40 dates (median ratio 1.003) with real minibatch noise, and exact under additive noise. | P5 | **The measurement side of the line of work holds up.** Worst-case (EV@R/CVaR) *reporting* is solid; automatic *tuning by the paper's recipe* is not. |
| Rebalance Guard: minimizing risk lowered realized worst-5%-month loss from 11.3% to 8.3–8.6% and drawdown 58% → 42–46% with no reliable Sharpe gain; min-vol matched min-EV@R; bagging did not reduce trading. | `research/rebalance_guard/RESULTS.md` | Useful, but a risk tool in a regulated-adjacent area, one universe, modest edge. |

So the value we can honestly prove is: **(a) noise from training/estimation changes real decisions, (b) we can measure that
precisely and in worst-case terms, (c) simple budget-aware fixes remove most of it.** The research line contributes (b) and
the framing; the tuned baselines contribute (c).

## 2. Recommendation: **Retrain Audit** (new MCP), with Rebalance Guard as the secondary demo

**One-line pitch.** "Ask Claude: *does my model give different answers to the same person every time I retrain it?* It
retrains it 50 times, lists who gets flipped, and shows the fix, with before/after numbers."

**Why this over Rebalance Guard as the lead product**
* Audience: every data scientist/analyst using Claude, not only people with a stock portfolio.
* Risk: no investment-advice exposure; no broker-adjacent anxiety.
* Evidence already exists on real data (German Credit) with a clean before/after.
* The result is checkable by the user on their own CSV in minutes, which is what makes content credible.
* It uses the research line honestly: tail-risk reporting (EV@R/CVaR over retrains) is the paper's idea applied where it is validated.

**Tools (all analysis-only, read-only on user data, run locally):**
1. `audit_training(csv, target, model="logistic", seeds=50, budget_steps, batch_size)` → per-row flip probability,
   share of rows whose decision is not the same in every retrain, mean/worst-5% flips per retrain, tail of validation loss
   across retrains (mean, 95th percentile, EV@R), and an AUC/accuracy band so "stable but useless" is visible.
2. `explain_instability(...)` → which rows are ambiguous (nearest to threshold), how many are near-ties vs genuinely noisy,
   and whether the noise comes from step size, momentum, batch size, or the data sample (bootstrap vs seed-only).
3. `stabilize(...)` → tries a *fixed menu* (budget-tuned SGD, tail averaging, seed ensembling, larger batch) on the user's
   data, reports the same metrics before/after, and states cost (extra training runs). The RA-GMM recipe is included as an
   option labelled experimental, so the tool never hides a method that loses.
4. `export_report(...)` → a Markdown/JSON report with the numbers and the exact commands, for a PR or a model card.

**Guardrails.** Local-only, no uploads; refuses datasets above a size limit; labels results as "this dataset, this model
family"; never claims fairness or legal compliance; reports the accuracy cost of every fix.

## 3. What could fail, how we will know early, and the pivot

| # | Failure | Early signal | Cheap test | Pivot |
|---|---|---|---|---|
| 1 | Practitioners already use seed ensembles / large batches and see no instability. | On 5 of the user's own models, flips < 2% of rows. | Run `audit_training` on 5 public tabular datasets and 3 friendly users' models before any marketing. | Re-aim at the regime where it is large: small data, high-stakes thresholds (credit, triage, hiring screens), low-budget training. If it is small everywhere, publish the negative result as the content and stop building. |
| 2 | Logistic regression is too narrow; people use gradient boosting / neural nets. | First 10 users ask for XGBoost/PyTorch. | Add a `bring-your-own-training-script` mode (the tool just re-runs a user command with different seeds and parses a predictions file). | Make the MCP a **generic seed-variance harness** (model-agnostic) with the stabilizer menu as optional. This is also cheaper to maintain. |
| 3 | The fix does not help on user data (tuned baselines no better than defaults). | `stabilize` shows < 20% flip reduction on most datasets. | Same 5+3 test set. | Keep the audit (diagnosis has value alone) and drop the fix claims; the pitch becomes "know your instability", like a linter. |
| 4 | Claude does not call the tool well (arguments wrong, long outputs misread). | Failed/garbled calls in a 20-prompt scripted eval. | A fixed eval of 20 natural-language prompts, scored for correct tool use and a correct summary. | Simplify to 2 tools with defaults; shorten outputs; add examples to the server instructions. |
| 5 | Runtime too slow for interactive use. | > 60 s on a 50k-row dataset. | Time it on 3 sizes. | Subsample rows for the audit with a stated margin of error; offer a background mode that writes the report to disk. |
| 6 | Content gets attention, product gets no use (distribution without value). | Installs ≫ second-day reuse. | Track one metric: users who run it on a **second** dataset. | Switch from "tool" content to "finding" content (the flip-rate study across public datasets), which stands on its own. |
| 7 | Overclaiming (calling it fairness/compliance, or implying RA-GMM fixes it). | A reader quotes a claim we cannot back. | Pre-publication claim check against the files cited in section 1. | Every public claim links to a results file; negative results stay in the main text. |
| 8 | Finance variant draws regulatory/advice objections. | Any user treats output as trade advice. | Disclaimers + analysis-only already in Rebalance Guard. | Keep it as a demo/appendix, never the headline, never connect it to a trading tool. |
| 9 | The research line looks irrelevant ("just seed averaging"). | Reviewer/audience says the theory adds nothing. | Report honestly which part comes from the paper. | Position the paper's contribution as the **worst-case metric and the calibrated predictor** (P5: exact-K EV@R matches simulation 40/40), and add a tool that predicts a run's tail risk *before* training (quadratic/convex cases) to show value beyond averaging; if it cannot beat simple heuristics, say so. |

## 4. Gates (stop/continue rules, set before building)

* **Gate 0 (2 days, before any build):** run the existing German Credit pipeline on 5 more public tabular datasets
  (e.g. via OpenML if reachable, otherwise GitHub-hosted CSVs). **Continue** if mean flips ≥ 5% of rows on at least 3 of 6
  with textbook-style settings. **Otherwise** pivot to the "finding" content (failure 6) and do not build the MCP.
* **Gate 1 (week 1, MVP: `audit_training` + `stabilize` on logistic regression):** on the 6 datasets the stabilizer cuts
  flips by ≥ 50% at ≤ 1 AUC point cost on ≥ 4 of 6, and the 20-prompt Claude eval passes ≥ 18. Else apply pivot 2 or 3.
* **Gate 2 (week 2, 5 design-partner runs on their own data):** ≥ 3 of 5 say they would act on the report (change a
  setting, add averaging, flag rows). Else apply pivot 1 or 6.
* **Gate 3 (public launch):** only after Gate 2, with the report format and all claims linked to result files.
* **Success after launch (30 days):** ≥ 25 people run it on a second dataset; ≥ 3 independent reports of a changed
  decision/setting; the content's most-shared claim is one we can reproduce. Below this, run the failure-6 pivot.

## 5. Proving the value of the research line (without overselling)

1. **Study, not product claim:** "Flip rates across N public datasets and training budgets" with the paired bootstrap
   already used here. Pre-registered, negative results included. This is publishable content and a workshop-paper seed.
2. **Worst-case reporting:** show EV@R/CVaR of validation loss over retrains next to the mean; show cases where mean looks
   fine and the tail is bad. The calibrated exact predictor (P5, quadratic/convex models) is the new, checkable piece.
3. **Honest method ablation inside the tool:** RA-GMM vs tuned vs averaging on the user's data, each with its tuning cost.
   If RA-GMM wins somewhere (small κ, tight budgets, no tuning data), that is evidence of the niche; if never, we say so.
4. **Open question to test, not assume:** does a self-consistent finite-horizon selector (P5 follow-up: select using the
   solver's own start) beat tuned-for-mean? Run it before claiming the research line improves settings.

## 6. Content plan tied to the gates

* **Video 1 (after Gate 0):** "I retrained the same loan model 200 times. 98% of applicants got a decision that depended on
  luck." Real German Credit data, flip map, then the 3-line fix. No theory.
* **Video 2 (after Gate 1):** "Ask Claude to audit your model" live on a user-supplied CSV; show before/after.
* **Video 3 (after Gate 2):** what partners found; include one case where the fix did not help.
* **Short post series:** the worst-case metric explained with the loan numbers; "why the clever method lost to boring
  tuning" (our own negative results; high-credibility content).
* **Secondary:** Rebalance Guard demo as "same idea, different domain", with the tail-loss-falls/Sharpe-does-not message.

## 7. Branch and process rules

* All work stays on `claude/ragmm-product-content-s3k2po`; no commits to `main`, no pull request unless asked.
* Pre-register each experiment in a `PLAN.md` before running; commit logs and result files with the claims they support.
