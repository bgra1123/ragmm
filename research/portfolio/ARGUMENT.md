# The argument: rebalance for the worst case, not the average

Audience version, built only on results in `RESULTS.md` (real daily prices of
10 large US stocks, monthly rebalancing 1991–2018, 200 random seeds per
solver). Every number below is traceable to that file. Not investment advice.

---

## 1. The problem (show it)

**Claim.** When a portfolio is rebalanced with a fast, noisy optimizer (the
kind used for big universes, many client accounts, or scenario-based risk
models), the result depends on a random number, and with default settings
that dependence is huge.

**Evidence** (same strategy, same stocks, same history; only the solver's
random seed differs):

| | Default solver settings |
|---|---|
| Final wealth across seeds, long-only (5th to 95th percentile) | **56× to 1,057×** (exact answer: 279×) |
| Accounts wiped out, when shorts and leverage are allowed | **34%** |
| Trading caused only by the seed, every month (long-only) | **~100% of the portfolio**, vs 19% real turnover |

Figures: `results/figures/wealth_fans_longonly.png` (left panel),
`results/figures/worstcase_fans.png` (left panel).

**Why it happens (one sentence for the audience).** Each solver step estimates
expected returns from a few noisy days; big steps chase that noise, and
momentum amplifies it, so the "optimal" portfolio you end up with is partly
luck.

## 2. The idea (explain it)

Most settings are chosen to make the **average** outcome good. Our paper's
idea: choose them to make the **worst outcomes** good, measured with entropic
value at risk (EV@R), the expected damage in the worst ~5% of cases.

## 3. The solution (demonstrate it)

**Step 1: stop using default settings; choose them deliberately on past data.**

| | Default settings | Settings chosen on 1991–92 data (tuned for the average) |
|---|---|---|
| Accounts wiped out (shorts allowed) | 34% | **0%** |
| Final wealth, 5th to 95th percentile, 1991–2018 (shorts allowed) | wiped out … 783× | **239× to 367×** (exact: 303×) |
| Seed-driven trading per month (long-only) | ~100% | **~15%** |

**Step 2: choose them for the worst case instead of the average.**
Same search, same data, same budget; only the criterion changes.

| Shorts allowed, tested 1993–2018 | Tuned for the average | Tuned for the worst case (EV@R) |
|---|---|---|
| Worst 5% of rebalances: seed-driven trading | 38.3% | **33.8%** (−12%) |
| Worst 5% of rebalances: distance from the optimal portfolio | 0.0418 | **0.0346** (−17%) |
| Average distance from the optimal portfolio | 0.0105 | **0.0097** (no cost; slightly better) |

All differences have 95% confidence intervals that exclude zero (paired
bootstrap over seeds). Figure: `results/figures/worstcase_tail.png`.

**One-line takeaway:** *Picking the optimizer's settings to protect the worst
5% of outcomes cut worst-case errors by 12–17% out of sample, without making
the average worse.*

## 4. What we must say honestly (limitations)

- **The biggest win is step 1** (deliberate settings vs defaults). Worst-case
  tuning is a smaller, second improvement on top.
- **Long-only portfolios: no difference** between worst-case and average
  tuning. The no-shorting, no-leverage rule already cuts off the bad tail.
- **This uses the paper's risk measure, tuned on past data.** The recipe that
  was tested here is the paper's *strongly-convex* one (problem (51)), whose
  bound is about 4–5× too loose in this setting. That was the wrong tool: this
  objective is an exact quadratic, for which the paper has a separate, tighter
  formulation (Section 3, problem (32)) that has **not** been tested here yet.
  See PLAN.md section 13 for that test.
- **Survivorship bias:** the 10 stocks were chosen in 2018, so absolute
  returns are flattering. Compare solvers with each other, not with the market.
- With 10 stocks the exact answer is cheap to compute; the noisy solver stands
  in for large or scenario-based problems.

## 5. Suggested video structure

1. **Hook (problem):** "Same strategy, same data, one random number: 56× or 1,000×."
   Show the default-settings fan.
2. **Why:** noisy estimates + big steps + momentum = trading on noise.
3. **Fix 1:** choose settings on past data → wipe-outs 34% → 0%. Show the
   tuned fan hugging the exact answer.
4. **Fix 2 (the research idea):** tune for the worst 5%, not the average →
   worst-case errors −12 to −17%, average unchanged. Show the tail chart.
5. **Honest close:** where it doesn't help (long-only), and what the paper's
   theory still needs.
