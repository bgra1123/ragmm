# Risk-Averse Fine-Tuning for Prompt-Injection Defenses — Research Plan

**Status:** draft plan, no experiments run yet.
**Goal:** a paper (arXiv first, then a workshop or venue) testing whether
risk-averse momentum tuning (RA-GMM, Can & Gürbüzbalaban 2025) makes the
*training* of prompt-injection defenses more reliable, with no additional data.

---

## 1. The core question

Training is random: data order, minibatch sampling, and (in private training)
injected noise. Run the same recipe twice and you can get a good defense or a
leaky one. Standard practice tunes optimizers for the **average** run. RA-GMM
tunes momentum parameters to bound the **tail** of the suboptimality
distribution (EV@R), trading a small amount of convergence rate for a large
reduction in risk.

> **Main question:** When we train a prompt-injection defense, does risk-averse
> optimization reduce the chance of shipping an unlucky, leaky model, using the
> same data and the same compute?

### What this paper does and does not claim

| In scope | Out of scope |
|---|---|
| Run-to-run reliability of training (tail of outcomes over training randomness) | Adversarial robustness to inputs as such |
| Convex detectors, where the theory applies exactly | Claims that RA-GMM "makes LLMs hack-proof" |
| Exploratory, clearly labelled tests on non-convex LLM fine-tuning | Treating heuristic LLM results as covered by the theorems |

Every result is reported as a **distribution over seeds**, not a single number.

---

## 2. Research questions and hypotheses

| ID | Question | Hypothesis | Theory covers it? |
|---|---|---|---|
| **RQ1** | Does RA-GMM reduce tail suboptimality when training a convex prompt-injection detector? | Lower CVaR₅% / 95th percentile of `f(x_K) − f*` than tuned Nesterov/HB, with a comparable mean | **Yes** |
| **RQ2** | Does lower tail suboptimality translate into fewer leaky detectors (security metrics)? | Lower 95th-percentile miss rate on attacks; lower `P(miss rate > τ)` | Partially (loss → metric link is empirical) |
| **RQ3** | Does the effect hold on attack families held out from training? | Exploratory; no directional claim in advance | No |
| **RQ4** | Is RA-GMM more reliable under label noise or outliers (heavy-tailed gradient noise)? | Exploratory stress test | No (assumptions violated) |
| **RQ5** | Does an RA-GMM heuristic help when fine-tuning neural detectors or LLMs (LoRA) for injection resistance? | Exploratory | No (non-convex, L/μ estimated) |
| **RQ6** | Is the theoretical EV@R bound predictive, i.e. does it rank configurations the way the empirical tail does? | Positive rank correlation between bound and empirical CVaR | Yes (validation) |

**Pre-registration:** commit the hypotheses, metrics, and analysis script
(`analyze.py`) to the repo *before* running the main sweep, and cite the commit
hash in the paper. This guards against the "we decided the winner first"
criticism.

---

## 3. Staged roadmap

| Stage | What | Output | Est. time | Compute |
|---|---|---|---|---|
| **0** | Engineering prerequisites (§4) | Tested, installable library | 1 week | CPU |
| **1** | **Main experiment:** convex detector, RQ1, RQ2, RQ6 (§5) | Core figures and tables | 2–3 weeks | CPU |
| **2** | Held-out attack families and label-noise stress tests, RQ3, RQ4 (§6) | Robustness section | 1–2 weeks | CPU |
| **3** | Neural and LLM extension, RQ5 (§7) | Exploratory section | 3–4 weeks | 1 GPU (24–80 GB) |
| **4** | Writing, release, content (§8–§9) | arXiv preprint, code, demo | 2 weeks | — |

**Decision gate after Stage 1:** if RA-GMM shows no tail improvement in the
convex setting, where the theory applies, do not proceed to Stage 3 as a
positive story. Write it up as a negative or nuanced result (still publishable
as a workshop paper or blog post) and investigate why: noise model mismatch,
bound looseness, or grid resolution.

---

## 4. Stage 0 — Engineering prerequisites

1. **Fix packaging.** `pyproject.toml` declares
   `build-backend = "setuptools.backends.legacy:build"`, which does not exist;
   use `setuptools.build_meta`. Remove the `.DS_Store` files.
2. **Honour `psi_range`.** `RAGMM._run_grid_search` ignores the `psi_range`
   argument (the ψ grid is hard-coded to `[psi_lo, 1.0]`).
3. **Report ψ after tuning.** `step()` returns `psi = 0.0` on every step after
   the first; store the tuned ψ.
4. **Add an L/μ helper:**
   - Closed form for L2-regularized logistic regression:
     `L = λ_max(XᵀX) / (4n) + λ`, `μ = λ`.
   - Power iteration on Hessian-vector products for general models
     (needed in Stage 3).
5. **Unit tests:** GMM reduces to GD / heavy-ball / Nesterov for the right
   (β, γ); bound functions match the paper's worked examples; RAGMM is
   deterministic given L, μ, d, ζ.
6. **Seeding utilities** for common random numbers (§5.6).

---

## 5. Stage 1 — Main experiment design (convex detector)

### 5.1 Task

Binary classification of prompts: **injection / hijacking attempt** vs
**benign**. This mirrors a real deployment pattern: a lightweight guard model
screens inputs before they reach the LLM.

### 5.2 Data

Candidate public sources (verify licenses, sizes and splits before use):

| Source | Role |
|---|---|
| `deepset/prompt-injections` (Hugging Face) | Primary labelled set (injection vs benign) |
| `Lakera/gandalf_ignore_instructions` (Hugging Face) | Additional attack prompts |
| HackAPrompt dataset (`hackaprompt/hackaprompt-dataset`) | Diverse attacks; used for held-out families in Stage 2 |
| Benign instruction data (e.g. a sample of an open instruction-tuning set) | Benign class balance and false-positive measurement |

Processing:
- Deduplicate (exact and near-duplicate via embedding cosine > 0.95) **across**
  splits to avoid leakage.
- Fixed splits: train / validation / test, with the test set frozen and hashed.
- Record dataset versions and hashes in `results/manifest.json`.

### 5.3 Model and problem

- **Features:** frozen sentence embeddings from a small open embedding model
  (e.g. a MiniLM- or BGE-small-class model, d ≈ 384–768). Compute once and cache.
- **Classifier:** L2-regularized logistic regression,
  `f(w) = (1/n) Σ log(1 + exp(−yᵢ wᵀxᵢ)) + (λ/2)‖w‖²`.
- **Constants:** `L = λ_max(XᵀX)/(4n) + λ`, `μ = λ` (computed exactly).
  Sweep λ ∈ {1e-4, 1e-3, 1e-2} to vary the condition number κ = L/μ.
- **Optimum:** compute `f*` and `w*` with a high-precision deterministic solver
  (L-BFGS to gradient norm < 1e-10). Needed to measure suboptimality exactly.

### 5.4 Noise settings

| Setting | Description | Why |
|---|---|---|
| **N1: Gaussian oracle** | Exact gradient + i.i.d. Gaussian noise σ | Matches the paper's assumptions exactly; the cleanest test of the theory |
| **N2: Minibatch SGD** | Batch sizes {8, 32, 128} | Realistic training noise |
| **N3: DP-SGD** | Per-example clipping + Gaussian noise at ε ∈ {1, 3, 8} | Noise injected by design; a strong practical motivation |

### 5.5 Methods compared

All methods get the **same data, iteration budget, and tuning budget**.

| Method | Configuration |
|---|---|
| SGD | Step size tuned on validation (mean loss) |
| Heavy-ball | Standard parameters for strongly convex problems, plus a tuned variant |
| Nesterov (AGD) | Standard `α = 1/L`, `β = (√κ − 1)/(√κ + 1)`, plus a tuned variant |
| **Polyak–Ruppert averaging** (SGD + iterate averaging) | **Key competitor:** also reduces variance with no extra data |
| AdamW | Practitioner default; tuned lr |
| **RA-GMM** | ζ ∈ {0.01, 0.05, 0.1}, `rate_slack` ∈ {0.0, 0.05, 0.1} |

Baseline tuning protocol: grid over step size (and momentum where applicable)
using the **mean** validation loss over 10 seeds, which mirrors standard
practice. Report results with this tuning applied. As a robustness check, also
report baselines tuned on validation CVaR, to show that RA-GMM's advantage is
not just "tuning for the tail".

### 5.6 Runs and randomness control

- **200 seeds** per (method × noise setting × λ × budget).
- **Common random numbers:** seed *s* uses the same initialization, data order
  and noise draws for every method, which lets us do paired comparisons and
  shrinks confidence intervals.
- Iteration budgets K ∈ {0.25, 0.5, 1, 2} × K₀, where K₀ is the budget at which
  tuned Nesterov reaches its noise floor. This shows the rate-vs-risk trade-off
  over time, not at a single point.

Cost estimate: logistic regression on ~5–20k examples × d ≤ 768 is cheap;
the full sweep (~6 methods × 3 noises × 3 λ × 4 budgets × 200 seeds ≈ 43k runs)
is feasible on CPU in hours.

### 5.7 Metrics

Per run:
- Suboptimality `S_K = f(w_K) − f*` (this is what the theory bounds)
- Test **attack miss rate** (false negative rate on injections) at a threshold
  fixed on validation for a 1% benign false-positive rate
- Test benign false-positive rate, AUROC, AUPRC

Across runs (the main object of study):
- Mean, std, median
- **95th and 99th percentiles** and **CVaR₅%** (mean of the worst 5% of runs)
- **P(miss rate > τ)** for τ ∈ {5%, 10%, 20%}: "how often you ship a leaky detector"
- Empirical EV@R estimate, compared against the theoretical bound (RQ6)

### 5.8 Statistical analysis

- Bootstrap (10k resamples) 95% CIs for quantiles and CVaR.
- Paired comparisons via common random numbers: paired bootstrap on the
  difference in CVaR₅% between RA-GMM and each baseline.
- Holm–Bonferroni correction across the main comparisons.
- Spearman rank correlation between the theoretical EV@R bound and empirical
  CVaR across the RA-GMM grid (RQ6).

### 5.9 Planned figures

1. **"100 detectors" dot plot:** each dot is one trained detector's miss rate,
   with leaky ones (above τ) in red; standard vs RA-GMM. This is the headline
   visual for both the paper and social content.
2. Distribution plots (violin/ECDF) of `S_K` per method.
3. Mean vs CVaR₅% scatter (the efficient frontier of rate vs risk), with each
   method/configuration as a point.
4. Tail metrics vs iteration budget K.
5. Theoretical bound vs empirical CVaR (RQ6).

---

## 6. Stage 2 — Robustness extensions

### 6.1 Held-out attack families (RQ3)

- Cluster or label attacks by family (e.g. "ignore previous instructions",
  role-play/persona, encoding/obfuscation, payload splitting, indirect
  injection via documents).
- Leave-one-family-out: train without family F, evaluate the miss rate on F.
- Report the tail of held-out miss rate across seeds.
- No directional hypothesis; report whatever is observed.

### 6.2 Label noise and outliers (RQ4)

- Flip 1%, 5%, 10% of training labels (uniform, and adversarially targeted at
  near-boundary points).
- Add 1–5% extreme-norm feature outliers.
- These produce heavy-tailed gradient noise, outside the paper's assumptions.
  Present this explicitly as a stress test that maps the method's limits.

---

## 7. Stage 3 — Neural and LLM extension (exploratory, RQ5)

The theorems do not cover this stage. Label all results as heuristic.

### 7.1 Required method extensions

- **L estimate:** top Hessian eigenvalue via power iteration on Hessian-vector
  products at initialization, and re-estimated periodically.
- **μ proxy:** weight-decay coefficient (an explicit strongly convex
  regularizer), or a small positive floor. Report sensitivity to this choice.
- **Periodic re-tuning:** re-run the (α, ψ) grid search every N steps with
  fresh L estimates (currently RAGMM tunes once on the first step).
- Ablation: RA-GMM with fixed vs periodically re-tuned parameters.

### 7.2 Setups

| Setup | Model | Training | Evaluation |
|---|---|---|---|
| **3a: Neural detector** | Small encoder classifier (DeBERTa-v3-small / ModernBERT-base class) | Full fine-tune or LoRA on the Stage 1 data | Same metrics as Stage 1, 20–50 seeds |
| **3b: LLM injection resistance** | Small open instruct LLM (0.5B–3B, e.g. Qwen2.5 or Llama-3.2 class) | LoRA fine-tune on instruction-hierarchy data (system instructions must beat injected instructions) | **Attack success rate (ASR)** on injection benchmarks (e.g. BIPIA for indirect injection, TensorTrust-style hijacking, AgentDojo-style agent tasks; verify availability) **plus utility** on a standard capability benchmark, 10–20 seeds |

For 3b, the key outputs are the **distribution of ASR across seeds** and the
**ASR vs utility trade-off**: a defense that refuses everything is not a win.

Baselines in Stage 3: AdamW (the de facto default), SGD+Nesterov, and
weight averaging (EMA/SWA), which is again the "no extra data" variance-reduction
competitor.

---

## 8. Paper outline

**Working title:** *Risk-Averse Optimization for Reliable Prompt-Injection
Defenses: Controlling the Tail of Training Outcomes without Additional Data*

1. Introduction: training randomness as a security risk; "the unlucky detector"
2. Background: prompt injection; GMM; entropic risk and EV@R; RA-GMM
3. Problem setup: guard classifiers; reliability as a tail metric over training randomness
4. Method: RA-GMM for convex detectors; constants for logistic regression; DP variant
5. Experiments (Stage 1): RQ1, RQ2, RQ6
6. Robustness (Stage 2): held-out families, label noise
7. Beyond convexity (Stage 3): exploratory neural/LLM results
8. Limitations: assumptions, input-space robustness out of scope, bound looseness
9. Conclusion

**Venue options:** arXiv preprint first; then a trustworthy-ML / security-ML
workshop at a major ML conference, or IEEE SaTML. Check current deadlines when
Stage 1 results are in.

---

## 9. Repository layout (proposed)

```
research/prompt_injection/
  PLAN.md                 # this document
  PREREGISTRATION.md      # hypotheses + metrics, frozen before the main sweep
  configs/                # YAML sweep configs per stage
  data.py                 # download, dedupe, split, hash
  embed.py                # compute and cache embeddings
  problem.py              # logistic objective, L, mu, f* via L-BFGS
  noise.py                # Gaussian oracle, minibatch, DP-SGD gradient oracles
  baselines.py            # SGD, HB, Nesterov, Polyak-Ruppert, AdamW
  run.py                  # seed sweep with common random numbers
  analyze.py              # tail metrics, bootstrap CIs, figures
  results/                # manifests and aggregated CSVs (raw runs git-ignored)
```

---

## 10. Content tie-ins per stage

| Stage | Content piece |
|---|---|
| 0–1 | "I'm testing my own research on a hot problem — here's the plan" (build in public) |
| 1 | The "100 AI security guards" dot plot: how many fall asleep on the job? |
| 2 | "What happens when the training data is messy?" |
| 3 | "Does it work on real LLMs?" (honest, labelled exploratory) |
| 4 | Paper launch, interactive demo, code release |

Whatever the outcome, publish it. A clear negative or nuanced result is still
credible content and still science.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| RA-GMM's tail gains disappear against tuned baselines | Decision gate after Stage 1; report honestly |
| Polyak–Ruppert averaging matches RA-GMM | Report it; test RA-GMM + averaging as a combination |
| Minibatch noise violates the theory's noise model | Setting N1 (Gaussian oracle) isolates the theory; N2/N3 test practice |
| Loss improvements don't translate into security metrics | That is RQ2 itself; report the link (or its absence) |
| Dataset leakage or near-duplicates inflate results | Cross-split dedupe, hashed frozen test set |
| LLM results are noisy and costly | Keep Stage 3 exploratory; prioritise the distribution of ASR over many seeds of small models |
