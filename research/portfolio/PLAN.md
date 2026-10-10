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

---

## 9. Rebalancing: optimization problems we can use

**Setting.** A portfolio of n stocks. At each rebalancing date the current
weights have drifted to `w₀` (yesterday's target, moved by market returns).
We choose new weights `w`; the trade is `Δ = w − w₀`. Returns `r` are random,
with mean `m` and covariance `Σ`, estimated from a trailing window.

Formulations are listed from "theory applies exactly" to "exploratory".
For each: strongly convex constant μ and smoothness L, so RA-GMM can be tuned.

### R1. Mean-variance rebalancing with quadratic trading costs (main candidate)

```
minimize   −mᵀw + (θ/2) wᵀΣw + (c/2) (w − w₀)ᵀ Λ (w − w₀) + (λ/2)‖w‖²
subject to 1ᵀw = 1
```

- `θ`: risk aversion. `Λ`: per-stock trading-cost weights (diagonal, e.g.
  inversely proportional to liquidity), `c`: cost scale. Quadratic costs are
  the standard market-impact approximation.
- Budget constraint removed exactly by `w = w₀ + N z` (N: basis of the null
  space of `1ᵀ`), so the problem is unconstrained in z.
- `μ = θ λ_min(Σ) + c λ_min(Λ) + λ`, `L = θ λ_max(Σ) + c λ_max(Λ) + λ`
  (on the reduced space, use `NᵀΣN`, `NᵀΛN`).
- **Useful property:** the trading-cost term adds strong convexity, so
  rebalancing is better conditioned than building a portfolio from scratch.
- Exact solution: one linear solve. This is the reference `w*`.

### R2. Tracking a target allocation (robo-advisor rebalancing)

```
minimize   E[(rᵀw − rᵀb)²] + (c/2)(w − w₀)ᵀΛ(w − w₀) + (λ/2)‖w‖²,   1ᵀw = 1
```

- `b`: target allocation (e.g. 60/40, a glide path, an index). Keeps the
  portfolio close to the target in *return* terms while limiting trading.
- `E[(rᵀ(w − b))²] = (w − b)ᵀ(Σ + mmᵀ)(w − b)`: strongly convex when Σ is
  positive definite. Very easy to explain to a general audience.

### R3. Entropic-risk (exponential-utility) rebalancing

```
minimize   E[exp(−θ rᵀw)] + trading cost + (λ/2)‖w‖²,   1ᵀw = 1
```

- Same family as the paper's risk measure; equals mean-variance when returns
  are Gaussian, but uses the real (fat-tailed) return distribution otherwise.
- Convex and smooth on bounded sets; per-scenario gradients are unbiased.
  (The log form `(1/θ) log E[exp(·)]` has biased minibatch gradients, so use
  the expectation form.) L needs a bound on `exp(−θ rᵀw)` over the region
  considered; label as partially covered by the theory.

### R4. Proportional transaction costs (realistic, exploratory)

- Real costs for commissions and spreads are `Σᵢ cᵢ |Δᵢ|` (non-smooth). This
  creates a "no-trade region", which is realistic and important.
- Non-smooth, so outside the theorems. Use a smooth approximation (Huber or
  `cᵢ·softplus`-type) to keep L finite; report it as exploratory.

### R5. Long-only, position limits, turnover caps (exploratory)

- Need projection (or barrier terms). Projected GMM is outside the theorems.

### R6. CV@R / EV@R of portfolio losses (exploratory)

- Tail-risk objectives over scenarios. CV@R is non-smooth; EV@R is jointly
  convex in (w, z) but needs smoothing and L2 for strong convexity.

### Where the randomness comes from (and why RA-GMM is relevant)

At each date the solver runs K stochastic steps:
- **Minibatches of scenarios:** each step samples days from the trailing
  window (or Monte Carlo scenarios) instead of using the full `m`, `Σ`. This
  is how large universes, scenario-based objectives (R3, R6), and many
  personalized portfolios are solved under a compute budget.
- Optional: privacy noise when solving on client data.

### The finance analogue of a "flip": phantom trades

A noisy solver returns `w_K ≠ w*`. The difference is trading that comes from
the solver's random seed, not from new market information:

- **Phantom turnover** per rebalance: `‖w_K − w*‖₁` (and its cost).
- Across 200 seeds: mean, 95th percentile and worst 5% of phantom turnover and
  phantom trading cost, plus out-of-sample return, volatility, drawdown, and
  **how different the whole equity curve looks under a different seed**.

Hook: *"Your robo-advisor traded because of a random number."*

### Rolling rebalancing protocol

```
for each rebalancing date t (monthly):
    w₀  ← previous weights drifted by realized returns
    estimate m, Σ from the trailing 252 trading days (shrinkage for Σ)
    w*  ← exact solution (reference)
    for each method and seed: w_K ← K stochastic steps on minibatches of days
    trade to w_K, pay trading costs, hold until t+1 (out-of-sample)
```

Methods (lessons from the loan experiment): textbook SGD / Nesterov, SGD
tuned for the mean, **SGD + tail averaging (the strongest baseline)**,
paper-faithful RA-GMM, and later horizon-aware RA-GMM. Also the analogue of
Experiment 1b: **raise the cost penalty c (change the objective) vs tune the
solver**, measured against the intended portfolio.

### Data (reachable from the current environment)

- **20 large US stocks, daily adjusted prices 1990–2018** (GOOG, AAPL, AMZN,
  JPM, XOM, WMT, PFE, …) from the PyPortfolioOpt test resources on GitHub
  (MIT license). Several stocks start late (e.g. FB 2012, BABA 2014), so either
  restrict to the stocks available over the whole window or use an expanding
  universe. Survivorship bias: the list was chosen in 2018; state this.
- Kenneth French industry portfolios, Yahoo, Stooq and FRED are blocked by the
  environment's network policy; they need the domains allowed.

---

## 10. Experiment P1: monthly rebalancing backtest with R1 (written before running)

**Question.** When the rebalancing problem is solved with a stochastic solver
under a fixed compute budget, how much trading is caused by the solver's
random seed ("phantom trades"), and does RA-GMM reduce it?

**Data.** `data/stock_prices.csv`, the 10 stocks with full history
(AAPL, GE, AMD, WMT, BAC, T, XOM, BBY, PFE, JPM), daily simple returns.
Monthly rebalancing on the first trading day of each month, starting once 252
days of history exist: about 327 rebalances, 1991–2018.

**Problem R1 at each rebalance** (annualized units, trailing 252 days):

`f(w) = −mᵀw + (θ/2) wᵀΣw + (c/2)‖w − w₀‖² + (λ/2)‖w‖²`, with `1ᵀw = 1`,

with `m = 252·mean`, `Σ = 252·cov` (sample, 1/T), θ = 10, c = 1, λ = 1
(chosen from the exact solutions only: median gross leverage 1.33, median
monthly turnover 27%, κ between about 2 and 12). Budget constraint removed by
`w = w₀ + N z` (N: orthonormal basis of the null space of 1ᵀ). L and μ are the
extreme eigenvalues of `Nᵀ(θΣ + (c + λ)I)N`; d = 9. Short positions allowed.

**Stochastic solver.** Starts at the current holdings (z = 0) and runs K = 200
steps. Each step samples a minibatch of b = 16 days from the window; the
per-day gradient `−252 r_t + 252 θ (r_t − m̄)(r_t − m̄)ᵀ w + c (w − w₀) + λ w`
is unbiased for ∇f. Common random numbers: seed s uses the same minibatch
sequence for every method. 200 seeds; each seed runs the whole backtest along
its own path.

**Methods** (same oracle, same K):
- SGD textbook (α = 1/L); Nesterov textbook (α = 1/L, β = γ = (√κ−1)/(√κ+1));
  heavy-ball textbook (same α, β, γ = 0)
- SGD tuned for mean; Nesterov tuned for mean; SGD + tail averaging (tuned):
  step multipliers on 1/L (and β for Nesterov) chosen by mean suboptimality on
  the **first 24 rebalance dates only** (tuning seeds 10 000–10 019), then
  fixed. No look-ahead.
- RA-GMM, paper-faithful (`paper_tuning.py`), ζ = 0.05, ε ∈ {0.25 (main), 0.05},
  re-solved at every date from that date's (L, μ, d). Zero tuning runs.
- Reference: the exact solution `w*` at every date (deterministic path).

**Accounting.** Trade from the drifted weights `w₀` to the solver output `w_K`,
pay 10 bps × `‖w_K − w₀‖₁` of wealth, hold with drifting weights until the next
rebalance.

**Pre-registered metrics.**
- Per rebalance: suboptimality `f(w_K) − f*` and **phantom turnover**
  `‖w_K − w*‖₁` (w* computed from that seed's own w₀). Report mean and 95th
  percentile over all (seed, date) pairs, and the per-seed average.
- Per seed over the backtest: annualized return, volatility, Sharpe ratio, max
  drawdown, total turnover and costs, **tracking error to the exact-solution
  path** (annualized std of the daily return difference), terminal wealth.
- Across seeds: mean, 5th/95th percentiles, worst 5% (e.g. lowest Sharpe,
  highest tracking error), and the width of the terminal-wealth fan.

**Hypotheses.**
- **H1:** RA-GMM (ε = 0.25) has a lower 95th percentile of suboptimality and of
  phantom turnover than textbook Nesterov and heavy-ball.
- **H2:** RA-GMM has lower tracking error to the exact path, and a narrower
  terminal-wealth fan, than textbook Nesterov and heavy-ball.
- **H3 (open):** RA-GMM vs the tuned baselines and tail averaging. Note from
  the loan experiment: there ε = 0.25 was non-binding (κ ≈ 99). Here κ is
  about 2–12, where it binds, so this is a fairer test of problem (51).

**Caveat stated in advance.** With 10 stocks the exact solve is trivial. The
stochastic solver stands in for settings where exact solves are not used
(large universes, scenario-based objectives, many personalized portfolios).

> **Recorded before the main run (after a 10-seed, 30-date smoke test):**
> (1) the tuning grid for step multipliers was widened from [0.01, 2] to
> [1e-4, 2] (18 points) because the smoke test selected the grid's lower edge;
> (2) an account whose wealth reaches zero is marked **ruined** and stays at
> zero (textbook settings produce very large phantom positions under this
> noise level); the ruin rate is reported as an additional metric.
