# Risk-Averse Solvers for Portfolio Optimization — Research Plan

**Status:** draft plan, no experiments run yet.
**Goal:** test RA-GMM on portfolio optimization, where risk measures such as
EV@R are already the native language of the audience.

---

## 1. Two layers of risk

Portfolio optimization already uses risk measures: a manager picks weights `w`
to control **market risk** (variance, CV@R, EV@R of portfolio losses). This plan
adds a second layer that is usually ignored:

| Layer | Random because of | Measured by | Handled by |
|---|---|---|---|
| **Market risk** | Future returns | Variance / CV@R / EV@R of portfolio loss | The portfolio objective (classic) |
| **Decision risk** | Noisy gradients in the solver (sampled returns or scenarios, streaming data, privacy noise) | EV@R of the solver's suboptimality `f(w_K) − f*` | **RA-GMM (this repo)** |

Headline: *"You manage market risk carefully. Who manages the risk of your
optimizer?"*

Scope note: using EV@R **as the portfolio risk measure** is established
(Ahmadi-Javid 2012; supported in libraries such as Riskfolio-Lib). It is not
our contribution. Our contribution is controlling the **solver's** tail risk
with RA-GMM, optionally while optimizing an entropic or EV@R-type portfolio
objective.

---

## 2. Why the theory fits

The entropic risk of the portfolio loss with risk aversion θ is

`ρ_θ(w) = (1/θ) log E[exp(−θ wᵀR)]`.

For Gaussian returns `R ~ N(m, Σ)`, this equals `−mᵀw + (θ/2) wᵀΣw`, i.e. the
**classic mean-variance objective**. With an L2 (shrinkage) term:

`f(w) = −mᵀw + (θ/2) wᵀΣw + (λ/2)‖w‖²`

- Strongly convex and smooth: `L = θ λ_max(Σ) + λ`, `μ = θ λ_min(Σ) + λ`
- Exactly the problem class the RA-GMM theorems cover
- The same entropic-risk family appears at both layers, which makes a neat
  unifying story for the paper

**Constraints.** The theory is unconstrained:
- *Budget* `1ᵀw = 1`: eliminate it with `w = w₀ + Nz` (N a basis of the null
  space of `1ᵀ`); the problem in z stays unconstrained and strongly convex.
  Theory still applies, with L and μ computed for the reduced problem.
- *Long-only, position limits:* need projection. Projected RA-GMM is outside
  the theorems, so treat it as exploratory.

---

## 3. When solver noise is real (and when it isn't)

A small mean-variance problem with known m and Σ is solved exactly by a QP
solver, so there is no decision risk. Reviewers will say this first. RA-GMM is
relevant when gradients are stochastic:

1. **Stochastic approximation of the true objective:** each step uses sampled
   returns (or a minibatch of historical / simulated scenarios) instead of
   estimated m and Σ. This targets the true expected objective directly, and
   the noise is the data sampling.
2. **Scenario-based risk objectives** (entropic risk under non-Gaussian
   returns, EV@R, CV@R) with large Monte Carlo scenario sets.
3. **Online / streaming rebalancing** with a fixed compute budget per rebalance.
4. **Large universes** (thousands of assets) or many personalized portfolios
   (robo-advisors), where only a few stochastic passes are affordable.
5. **Private or federated** settings (client data with DP noise).

Always report the exact QP solution as a reference `w*`.

---

## 4. Research questions

| ID | Question | Theory covers it? |
|---|---|---|
| **P1** | On strongly convex mean-variance / entropic problems with stochastic gradients, does RA-GMM reduce the tail of `f(w_K) − f*` and of `‖w_K − w*‖` vs SGD, Nesterov, heavy-ball, averaging, Adam? | Yes |
| **P2** | Does that translate into financial metrics: certainty-equivalent loss (bps), out-of-sample Sharpe, drawdown, and especially **turnover** (oscillating weights cost transaction fees)? | Partially |
| **P3** | Does it hold on real return data with fat tails and regime changes? | No (assumptions violated) |
| **P4** | Does projected RA-GMM help with long-only / box constraints and scenario-based EV@R objectives? | No (exploratory) |
| **P5** | Is the EV@R bound predictive across (ζ, θ, λ) configurations? | Yes (validation) |

---

## 5. Experiment design

### 5.1 Setup A: synthetic, theory-exact (P1, P5)

- Returns from a **known factor model** (e.g. 3–5 factors, 50–500 assets), so
  `m`, `Σ`, `f*` and `w*` are known exactly.
- Gradient oracle: one sampled return vector per step, or minibatches {1, 8, 32}.
- Vary condition number via θ and λ; vary noise via return volatility.
- Methods: SGD, heavy-ball, Nesterov (standard and tuned), Polyak–Ruppert
  averaging, Adam, RA-GMM (ζ ∈ {0.01, 0.05, 0.1}). Same budget and tuning protocol
  as the prompt-injection plan.
- **200 seeds** with common random numbers.
- Metrics: tail (95th/99th percentile, CVaR₅%) of suboptimality, of `‖w_K − w*‖`,
  and of **certainty-equivalent loss in basis points** (a finance-native unit).

### 5.2 Setup B: real data, rolling backtest (P2, P3)

- Data: public sources, e.g. Kenneth French data library (industry and
  size/value portfolios), or a large-cap equity universe (handle survivorship
  bias explicitly).
- Rolling rebalancing (monthly, and weekly as a stress test): at each date, run
  K solver steps on minibatches of the trailing window's returns.
- References: exact QP on the same window, 1/N (DeMiguel et al. 2009),
  Ledoit–Wolf shrinkage mean-variance.
- Metrics across seeds: out-of-sample Sharpe, volatility, max drawdown,
  realized CV@R/EV@R of returns, **turnover and cost-adjusted returns**, and
  distance to the exact-QP portfolio. The key result is the **spread across
  seeds**: how different would your fund have looked under a different random
  seed?

### 5.3 Setup C: exploratory (P4)

- Long-only and box constraints via projected GMM / RA-GMM.
- Scenario-based EV@R portfolio objective (jointly convex in (w, z); add L2 for
  strong convexity; smoothness needs care).

### 5.4 Planned figures

1. "Same strategy, 200 seeds": fan of equity curves, standard vs RA-GMM.
2. Turnover distribution per method.
3. Tail of certainty-equivalent loss (bps) vs iteration budget.
4. Bound vs empirical tail (P5).

---

## 6. Roadmap

| Step | What | Time |
|---|---|---|
| 1 | Problem module: mean-variance / entropic objective, budget reparametrization, L/μ, exact QP reference | 1 week |
| 2 | Setup A sweep and analysis | 1–2 weeks |
| 3 | Setup B backtests | 2 weeks |
| 4 | Setup C (exploratory) and write-up | 2 weeks |

**Decision gate:** if Setup A shows no tail gain, report it and stop before the
real-data backtests.

Shares infrastructure with `research/prompt_injection/` (baselines, seed
sweeps, tail-metric analysis), so build common code once.

---

## 7. Content angle

Finance is the most relatable entry point for a general audience because
everyone already understands risk vs return.

- "Your robo-advisor's math can have a bad day too."
- "Same strategy, different random seed, very different fund" (the 200-seed
  equity-curve fan is the visual).
- "Fast vs safe: the trade-off traders know, applied to the algorithm itself."

Always include a "research, not investment advice" disclaimer.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| "Just use a QP solver" | Frame around stochastic / streaming / large-scale settings; report exact QP as reference |
| Estimation error in m and Σ dominates solver error | Measure both; report solver decision risk relative to estimation risk honestly |
| Real returns violate assumptions (fat tails, non-stationarity) | Setup A isolates the theory; Setup B is labelled as empirical |
| Constraints break the theory | Budget handled exactly; long-only labelled exploratory |
