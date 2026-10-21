# Project summary (for brand generation)

## The problem
Train the same model on the same data twice with a different random seed, and some people get a different decision. On a public credit dataset, a standard
fast-trained credit model changed about 1 in 6 decisions between retrains, and 284 of 300 applicants got a different answer in at least one of 1,000 retrains.
The effect lands hardest on applicants who look unlike most of the data (for example, among repaying applicants living in housing "for free", 34.6% were refused with
standard training against 24.7% without the randomness). Teams rarely measure this, so they cannot say how much of a refusal is the model and how much is luck.

## How it works
1. **Measure.** Retrain the model many times and report, overall and per group: how often two retrains disagree, which rows are unstable, and the chance of a wrong decision, with the part caused by training randomness separated out.
2. **Fix what can be fixed.** For logistic models, a short exact polish after training moves every run to the same answer (disagreement 15.5% to 0.1%). For trees and neural networks, averaging seeds
   (with calibration for cost-based cutoffs) cuts disagreement by about two thirds. Newton's method solves the logistic head exactly and works on data streamed in batches.
3. **Report** in plain sentences a risk or compliance team can read, with the limits stated.

Built on peer-reviewed work on risk-averse optimization (Can & Gurbuzbalaban, 2025, Optimization Methods and Software). Open-source Python package and command line; scikit-learn, XGBoost, LightGBM compatible.

## The value
* **For the lender or model owner:** know how much of your decisions depend on luck, for which customers, and remove the part that is avoidable; reproducible decisions are easier to audit and explain.
* **For customers:** fewer reliable applicants turned away because of an unlucky training run.
* **What it does not do:** it does not remove the model's own errors or biases, and it does not make claims about discrimination. Honest limits are part of the brand.

## Audience and tone
Data scientists, model-risk and compliance teams, fintech and lending analytics. Credible, calm, evidence-first, plain language, no hype. Working tagline idea: "Ask your model twice."

## Constraints for any branding
No claims of fairness, bias removal or regulatory approval. No "AI" buzzwords as the main idea. Name should be short, say-able in a video, usable as a Python package and command.
Shortlist already considered: Seedproof, Keel, Retrain Audit, Steadyseed, SameTwice, Plumb.

## Ready-to-paste prompt
> Act as a brand strategist. Using the project summary above, give me: (1) 15 product names (short, pronounceable, available as a Python package name), grouped by idea (stability, trust, "ask twice", measurement), with one line on why each works; (2) three taglines; (3) a 50-word product description; (4) a 5-word logo direction and colour palette; (5) three risks of each of your top 3 names (existing brands, misleading meanings). Keep the tone calm and evidence-first; do not imply fairness or compliance guarantees.
