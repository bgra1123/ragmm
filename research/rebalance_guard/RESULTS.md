# Results: does worst-case-aware rebalancing with a stability check help out of sample?

Design and hypotheses: `PLAN.md` (committed before the run). Code: `backtest.py`. Raw results:
`results/backtest_cap0.3.json` (primary), `results/backtest_cap1.json` (sensitivity), paths in `results/paths_*.npz`.
Reproduce: `python research/rebalance_guard/backtest.py [--cap 1.0]` (about 15 minutes).

**Setting.** 10 US stocks with full history (chosen in 2018: survivorship bias, so absolute returns are flattering and
only differences between strategies are interpretable). Monthly rebalancing, 304 months (1993-01 to 2018-04), 756-day
lookback, 10 bps trading cost, long-only, zeta = 0.05, expected returns ignored. Paired stationary bootstrap over months
for differences (mean block 6, 10 000 resamples); `*` marks 95% intervals that exclude zero.

## Primary specification (max weight 0.30 per stock)

| Strategy | Net return | Volatility | Sharpe | Max drawdown | Realized worst-5%-month loss | Monthly turnover | Target instability |
|---|---|---|---|---|---|---|---|
| Equal weight (monthly) | 17.5% | 22.4% | 0.83 | 58.0% | 11.3% | 3.2% | 0 |
| Min-vol, point estimate | 13.6% | 17.5% | 0.82 | 41.9% | 8.3% | 2.9% | 2.5% |
| Min-vol, bagged (30) | 13.8% | 17.4% | 0.83 | 42.5% | 8.3% | 4.7% | 4.1% |
| Min-EV@R, point estimate | 14.4% | 18.2% | 0.83 | 46.1% | 8.6% | 5.1% | 4.1% |
| Min-EV@R, bagged (30) | 13.9% | 17.5% | 0.83 | 42.5% | 8.4% | 7.3% | 6.7% |

("Target instability" = mean half-L1 distance between consecutive targets.)

## Hypotheses

| | Verdict |
|---|---|
| **H1** min-EV@R has a smaller realized worst-5%-month loss and max drawdown than equal weight | **Supported.** Realized worst-5%-month loss −2.7 pp [−4.2, −1.0] `*` (cap 1.0: −2.2 pp [−3.8, −0.6] `*`); max drawdown 46.1% vs 58.0% (point estimates; no interval computed). The same holds for min-vol (−3.0 pp [−4.4, −1.6] `*`). |
| **H2** bagging lowers turnover and target instability | **Rejected.** Bagging *raised* both. Min-EV@R: turnover +2.2 pp [+1.4, +2.9] `*`, instability +2.6 pp [+1.8, +3.3] `*`. Min-vol: turnover +1.8 pp [+1.6, +2.1] `*`, instability +1.5 pp [+1.3, +1.8] `*`. Same direction with cap 1.0. |
| **H3** bagging does not lower net Sharpe | **Holds.** Min-EV@R +0.03 [−0.07, +0.12]; min-vol +0.02 [−0.02, +0.07]. |
| Prior (not a hypothesis): optimized strategies do not beat equal weight in Sharpe | **Consistent.** Sharpe differences vs equal weight are +0.10 to +0.14 with intervals spanning about −0.16 to +0.45 (no significant difference), and net return is 3 to 4 pp lower (not significant). |

## Not pre-registered as hypotheses, but in the comparison list

* **Min-EV@R did not beat min-vol on realized worst-case outcomes.** Primary: realized worst-5%-month loss +0.3 pp
  [−0.2, +1.1] (no difference), turnover +2.2 pp [+1.4, +3.0] `*`. Sensitivity (cap 1.0): Sharpe −0.17 [−0.29, −0.04] `*`
  and realized worst-5%-month loss +1.2 pp [+0.4, +2.3] `*` (min-EV@R is worse). Min-vol is simpler, trades less and was never
  worse.
* **Bagging helps only the uncapped min-EV@R** (cap 1.0): Sharpe +0.11 [+0.01, +0.21] `*`, realized worst-5%-month loss
  −1.0 pp [−2.0, −0.2] `*`, because that point estimate concentrates in a few stocks; bagging diversifies it. It still
  increased turnover (+3.3 pp [+2.3, +4.3] `*`).

## Decision taken for the tool

By the rule written in `PLAN.md` (keep "bagged" as the recommendation only if H2 holds), the recommended proposal is the
**point estimate**. Because min-vol matched or beat min-EV@R at lower turnover, the default mode of `propose_rebalance`
is `min_vol`. EV@R remains the measure used in `risk_report`, and `min_evar` stays available on request.

## What this does and does not show

* Optimizing for risk lowers the realized tail losses and drawdowns relative to equal weight, at the price of a
  lower average return and more trading; risk-adjusted performance is statistically indistinguishable.
* The stability diagnostic (`turnover_caused_by_data_noise`) is **descriptive only**: this backtest did not test whether
  it predicts anything, and bagging, the one remedy tried, did not reduce trading.
* One universe of 10 large US stocks, one 25-year period, survivorship bias, a three-year lookback, 10 bps costs.
  Different universes (funds, bonds), lookbacks or costs may behave differently.
* Max drawdown has no confidence interval here (path-dependent); only point estimates are compared.
