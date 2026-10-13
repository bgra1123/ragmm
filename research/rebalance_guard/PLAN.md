# Does worst-case-aware rebalancing with a stability check help out of sample? (written before running)

**Tool under test:** `rebalance-guard` (`propose_rebalance`, long-only, exact convex solves).
The tool defaults to recommending the *bagged* proposal. That default was a prior, not evidence;
this backtest decides whether it stays.

**Data.** `rebalance-guard/rebalance_guard/data/demo_prices.csv`: 10 US stocks with full daily history,
1989-12-29 to 2018-04-11. Stocks were chosen in 2018 (survivorship bias): absolute returns are flattering;
only differences between strategies are interpretable.

**Protocol.** Rebalance on the first trading day of each month from the first date with 756 days of
history (late 1992) to 2018-04. At each date: use the previous 756 daily returns, build overlapping
21-day scenarios, solve with the tool's functions, trade from the *drifted* holdings, pay 10 bps on traded
value, hold with daily drift until the next rebalance. Max weight 0.30 per stock (main specification),
zeta = 0.05, `mean_weight = 0`. Seeds: bootstrap seed = date index.

**Strategies.**
| Name | Rule |
|---|---|
| EW | equal weight, rebalanced monthly (no estimation) |
| MinVol-point | `min_vol`, point estimate |
| MinVol-bagged | `min_vol`, mean of 30 block-bootstrap solutions |
| MinEVaR-point | `min_evar`, point estimate |
| MinEVaR-bagged | `min_evar`, mean of 30 block-bootstrap solutions |

**Metrics (net of costs).** Annualized return, annualized volatility, Sharpe ratio (rf = 0), max drawdown,
**realized worst-5%-month loss** (mean of the worst 5% of monthly net returns, i.e. realized CVaR), worst month,
average monthly turnover (0.5 L1 from drifted weights), and **target instability** (mean 0.5 L1 distance between
consecutive targets). Paired stationary bootstrap over months (mean block 6, 10 000 resamples) for differences.

**Hypotheses.**
- **H1:** MinEVaR-point has a smaller realized worst-5%-month loss and a smaller max drawdown than EW.
- **H2:** bagging lowers turnover and target instability relative to the point estimate (for both objectives).
- **H3:** bagging does not lower net Sharpe relative to the point estimate (reported with its CI).
- **Prior expectation, not a hypothesis:** no optimized strategy beats EW in net Sharpe with 10 assets and a
  3-year window (DeMiguel, Garlappi & Uppal 2009). Reported either way.

**Sensitivity (labelled, not primary):** max weight 1.0 instead of 0.30.

**Decision rule for the tool's default.** Keep "bagged" as the recommended proposal only if H2 holds and H3's
confidence interval excludes a Sharpe loss larger than 0.1; otherwise recommend the point estimate and say so.
