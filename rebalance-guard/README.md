# Rebalance Guard

An [MCP](https://modelcontextprotocol.io) server that lets Claude analyse a portfolio's **worst-case risk** and
propose **rebalancing trades**, with an honest look at how much of a proposal is just estimation noise.

> **Analysis only.** It never connects to a broker, never places or prepares orders, and does not forecast
> returns. It is not investment, tax or legal advice. Keep it separate from any trading connector.

## What it does

| Tool | Purpose |
|---|---|
| `describe_demo_data` | Tickers and dates of a bundled demo history (10 US stocks, 1989–2018, survivorship-biased). |
| `risk_report` | Volatility, **VaR / CVaR / EV@R** of the loss over a month at the worst 5% of cases, worst historical periods (with dates), max drawdown, concentration, and each holding's share of risk. |
| `propose_rebalance` | Trades for one of three modes: `min_vol` (default), `min_evar` (minimize the worst-case EV@R loss), or `target` (return to target weights with the fewest trades once a tolerance band is breached). Long-only, per-holding weight cap, proportional trading cost, optional turnover cap, before/after risk, and a **stability check** (bootstrap re-solves: how much of the proposed trading is noise). |

All optimization problems are convex and solved exactly ([cvxpy](https://www.cvxpy.org), Clarabel); no stochastic
solver is involved. EV@R is the entropic value at risk `inf_t (1/t) log(E[exp(tL)]/ζ)`, the coherent risk measure
used in Can & Gürbüzbalaban (2025); it is never smaller than CVaR or VaR at the same level.

## Install and connect to Claude

```bash
python -m venv .venv && source .venv/bin/activate
pip install ./rebalance-guard
```

**Claude Code**

```bash
claude mcp add rebalance-guard -- /full/path/to/.venv/bin/rebalance-guard
```

**Claude Desktop** (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "rebalance-guard": {
      "command": "/full/path/to/.venv/bin/rebalance-guard",
      "env": { "REBALANCE_GUARD_DATA_DIR": "/path/to/folder/with/your/price/csvs" }
    }
  }
}
```

Then ask, for example: *"I hold $20k AAPL, $15k GE, ... (use the demo data). What is my worst-case monthly loss, and
what would you change?"*

## Your own data

`prices` is `"demo"` or the path of a CSV: first column the date, then one column of **adjusted** daily close prices
per ticker. Holdings are `{ticker: market value}`. Set `REBALANCE_GUARD_DATA_DIR` to restrict which files the server
may read; without it any `.csv` path can be read, so only run the server for people you trust with local file access.

## Reading the output

* `monthly_evar >= monthly_cvar >= monthly_var`: the loss in the worst 5% of 21-day periods in the lookback window
  (default three years). It describes *this* window, not the future.
* `min_evar` and `min_vol` **ignore expected returns** by default (`mean_weight = 0`): expected returns are the noisiest
  input to any optimizer, so the proposals reduce risk and are not a return forecast.
* `stability.turnover_caused_by_data_noise_median` is how much of the portfolio the optimum moves between
  bootstrap-resampled versions of the *same* history. If it is large, most of a proposed trade is noise.
* Proposals: `point_estimate` (solution on the observed history; **recommended**) and `bagged` (average of the resampled
  solutions; shown for comparison, see Evidence).

## Limitations

Long-only and fully invested in the listed holdings. No cash, taxes, lot selection, fractional-share limits, dividends
or bid-ask spreads beyond the stated proportional cost. Historical scenarios cannot contain events that are not in the
window. Past performance does not predict future results.

## Tests

```bash
pip install -e "./rebalance-guard[dev]" && pytest rebalance-guard
```

The suite checks the numerics against independent references (closed-form Gaussian EV@R, scikit-learn's Ledoit-Wolf,
scipy's local optimizer, hand-computed stability statistics), the optimizers' constraints and optimality, input
validation, and the MCP protocol both in-memory and over a real stdio subprocess.

## Evidence

A 25-year backtest (`research/rebalance_guard/RESULTS.md`; 10 US stocks, monthly rebalancing, 10 bps costs,
survivorship-biased) of what the tool recommends:

* Optimizing for risk lowered the **realized worst-5%-month loss** from 11.3% (equal weight) to 8.3–8.6% and max
  drawdown from 58% to 42–46%, at a lower average return and with no reliable gain in risk-adjusted return.
* **Min-volatility matched or beat min-EV@R** on realized tail losses with less trading, so it is the default mode.
* **Bagging the optimizer's solutions did not reduce trading** (it increased turnover), so the point estimate is the
  recommended proposal; the stability figures are a warning about how noisy the trades are, not a remedy.

These are results for one small universe and period; they are not a prediction for yours.
